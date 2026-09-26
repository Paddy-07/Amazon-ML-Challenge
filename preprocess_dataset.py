#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Shared Preprocessing Script

THIS IS THE ONE PIPELINE BOTH TEAMMATES RUN, UNCHANGED.
Do not fork this file per-model — LightGBM and XGBoost branches must both
read from the same cache/ output, or your candidate sets, features, and
validation scores won't be comparable, and a bug caught by one of you won't
get caught by the other.

What it does:
  1. Streams every train/test source file in chunks (memory-safe on the
     multi-million-row files).
  2. Applies normalize_name() / normalize_address() from normalize.py to
     produce business_name_norm / business_address_norm columns, without
     dropping the original raw columns (keep both — later feature steps,
     like character-level similarity, may want the raw text too).
  3. Writes each result to cache/ as Parquet (much faster to reload than
     re-parsing TSV every run, and preserves dtypes).
  4. Every cache filename is prefixed train_/test_ so it's never ambiguous
     which split a cached file came from.

Usage (run from student_resource/, or pass --data-dir / --cache-dir):

    python3 preprocess_dataset.py --data-dir dataset --cache-dir cache

Requires: pandas, pyarrow (pip install pandas pyarrow)
Optional: indic_transliteration (pip install indic_transliteration) — if
installed, MUST be installed for both teammates, same version, pinned in
requirements.txt. Otherwise one of you gets romanized Indic names as
features and the other doesn't, which silently breaks comparability.
"""

import argparse
import os
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from normalize import normalize_name, normalize_address, _HAVE_INDIC

CHUNKSIZE = 200_000


def process_source_file(in_path, out_path, chunksize=CHUNKSIZE):
    if not os.path.isfile(in_path):
        print(f"  [SKIP] not found: {in_path}")
        return

    writer = None
    total = 0
    t0 = time.time()
    for chunk in pd.read_csv(in_path, sep="\t", dtype=str, chunksize=chunksize):
        chunk["business_name_norm"] = chunk["business_name"].map(normalize_name)
        chunk["business_address_norm"] = chunk["business_address"].map(normalize_address)

        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema)
        writer.write_table(table)

        total += len(chunk)
        elapsed = time.time() - t0
        print(f"  {os.path.basename(in_path)}: {total:,} rows ({elapsed:.0f}s)", end="\r")

    if writer is not None:
        writer.close()
    print(f"  {os.path.basename(in_path)}: {total:,} rows -> {out_path}" + " " * 10)


def process_ground_truth(in_path, out_path):
    if not os.path.isfile(in_path):
        print(f"  [SKIP] not found: {in_path}")
        return
    df = pd.read_csv(in_path, sep="\t", dtype=str)
    df["matched_entity_ids"] = df["matched_entity_ids"].fillna("")
    df.to_parquet(out_path, index=False)
    print(f"  train_ground_truth.tsv: {len(df):,} rows -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Shared normalization preprocessing step.")
    parser.add_argument("--data-dir", default="dataset", help="Path to dataset/ (default: %(default)s)")
    parser.add_argument("--cache-dir", default="cache", help="Where to write cached parquet (default: %(default)s)")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    print(f"indic_transliteration installed: {_HAVE_INDIC}")
    if not _HAVE_INDIC:
        print(
            "  WARNING: without this, Devanagari/Kannada/Telugu/etc. business names "
            "stay in native script (safe, but not romanized). Make sure BOTH teammates "
            "install it, or BOTH skip it — mismatched environments = mismatched features."
        )
    print()

    jobs = [
        ("train/train_source1.tsv", "train_source1_normalized.parquet"),
        ("train/train_source2.tsv", "train_source2_normalized.parquet"),
        ("train/train_source3.tsv", "train_source3_normalized.parquet"),
        ("test/test_source1.tsv", "test_source1_normalized.parquet"),
        ("test/test_source2.tsv", "test_source2_normalized.parquet"),
        ("test/test_source3.tsv", "test_source3_normalized.parquet"),
    ]

    for rel_in, out_name in jobs:
        in_path = os.path.join(args.data_dir, rel_in)
        out_path = os.path.join(args.cache_dir, out_name)
        print(f"Processing {in_path} ...")
        process_source_file(in_path, out_path)

    gt_in = os.path.join(args.data_dir, "train", "train_ground_truth.tsv")
    gt_out = os.path.join(args.cache_dir, "train_ground_truth.parquet")
    print(f"Processing {gt_in} ...")
    process_ground_truth(gt_in, gt_out)

    print("\nDONE.")
    print(f"Shared normalized cache is in: {args.cache_dir}/")
    print("Both teammates should run this script (same normalize.py version, same")
    print("library versions) before touching blocking/features/model code, and")
    print("treat the cache/ output as the single shared starting point for both")
    print("the LightGBM and XGBoost branches.")


if __name__ == "__main__":
    sys.exit(main())