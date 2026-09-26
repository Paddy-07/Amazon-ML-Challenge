#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Dataset Exploration Script

Run this FIRST, before any modeling. It never loads a whole large file into
memory at once — it streams through with pandas chunksize for counts/stats,
and only pulls a small number of rows for samples. Safe on the ~500MB
source2/source3 files.

Usage (run from the student_resource/ directory, or pass --data-dir):

    python3 explore_dataset.py --data-dir dataset

Requires: pandas (pip install pandas)
"""

import argparse
import os
import sys
from collections import Counter

import pandas as pd

SOURCE_COLS = ["entity_id", "business_name", "business_address", "country"]
GT_COLS = ["source1_entity_id", "matched_entity_ids"]
CHUNKSIZE = 200_000


def sep(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def explore_source_file(path, label):
    """Row count, column check, country distribution, ID prefix check,
    string-length stats, and a few sample rows — all via chunked reads."""
    if not os.path.isfile(path):
        print(f"  [MISSING] {path}")
        return

    sep(f"{label}  ({path})")

    # --- header / column check (cheap) ---
    header_df = pd.read_csv(path, sep="\t", nrows=0, dtype=str)
    cols = list(header_df.columns)
    print(f"Columns found: {cols}")
    if cols != SOURCE_COLS:
        print(f"  NOTE: columns differ from expected {SOURCE_COLS}")

    # --- sample rows (cheap) ---
    sample = pd.read_csv(path, sep="\t", nrows=5, dtype=str)
    print("\nSample rows:")
    with pd.option_context("display.max_colwidth", 60, "display.width", 140):
        print(sample.to_string(index=False))

    # --- chunked full pass: row count, country dist, prefix check, lengths ---
    n_rows = 0
    country_counts = Counter()
    prefix_counts = Counter()
    name_len_sum = 0
    addr_len_sum = 0
    null_name = 0
    null_addr = 0
    null_country = 0
    example_countries_seen_order = []

    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNKSIZE):
        n_rows += len(chunk)

        if "country" in chunk.columns:
            vc = chunk["country"].value_counts(dropna=False)
            for k, v in vc.items():
                country_counts[k] += v
                if k not in example_countries_seen_order:
                    example_countries_seen_order.append(k)
            null_country += chunk["country"].isna().sum()

        if "entity_id" in chunk.columns:
            prefixes = chunk["entity_id"].astype(str).str.slice(0, 3)
            for k, v in prefixes.value_counts().items():
                prefix_counts[k] += v

        if "business_name" in chunk.columns:
            null_name += chunk["business_name"].isna().sum()
            name_len_sum += chunk["business_name"].dropna().astype(str).str.len().sum()

        if "business_address" in chunk.columns:
            null_addr += chunk["business_address"].isna().sum()
            addr_len_sum += chunk["business_address"].dropna().astype(str).str.len().sum()

    print(f"\nTotal rows: {n_rows:,}")
    print(f"entity_id prefix counts: {dict(prefix_counts)}")
    print(f"country value counts: {dict(country_counts.most_common(20))}")
    print(f"null business_name: {null_name:,} | null business_address: {null_addr:,} | null country: {null_country:,}")
    if n_rows - null_name > 0:
        print(f"avg business_name length: {name_len_sum / max(1, n_rows - null_name):.1f} chars")
    if n_rows - null_addr > 0:
        print(f"avg business_address length: {addr_len_sum / max(1, n_rows - null_addr):.1f} chars")


def explore_ground_truth(path, source1_path=None):
    if not os.path.isfile(path):
        print(f"  [MISSING] {path}")
        return

    sep(f"train_ground_truth  ({path})")

    header_df = pd.read_csv(path, sep="\t", nrows=0, dtype=str)
    cols = list(header_df.columns)
    print(f"Columns found: {cols}")
    if cols != GT_COLS:
        print(f"  NOTE: columns differ from expected {GT_COLS}")

    sample = pd.read_csv(path, sep="\t", nrows=5, dtype=str)
    print("\nSample rows:")
    with pd.option_context("display.max_colwidth", 60, "display.width", 140):
        print(sample.to_string(index=False))

    n_rows = 0
    empty_matches = 0
    single_match = 0
    multi_match = 0
    max_matches = 0
    total_match_ids = 0
    s2_only = 0
    s3_only = 0
    both_s2_s3 = 0
    dup_s1_rows = Counter()

    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNKSIZE):
        n_rows += len(chunk)
        chunk["matched_entity_ids"] = chunk["matched_entity_ids"].fillna("")

        for s1, matches_str in zip(chunk["source1_entity_id"], chunk["matched_entity_ids"]):
            dup_s1_rows[s1] += 1
            ids = [x for x in matches_str.split(",") if x.strip()]
            k = len(ids)
            total_match_ids += k
            max_matches = max(max_matches, k)
            if k == 0:
                empty_matches += 1
            elif k == 1:
                single_match += 1
            else:
                multi_match += 1

            has_s2 = any(i.startswith("S2-") for i in ids)
            has_s3 = any(i.startswith("S3-") for i in ids)
            if has_s2 and has_s3:
                both_s2_s3 += 1
            elif has_s2:
                s2_only += 1
            elif has_s3:
                s3_only += 1

    dup_count = sum(1 for v in dup_s1_rows.values() if v > 1)

    print(f"\nTotal S1 entities in ground truth: {n_rows:,}")
    print(f"  0 matches (singleton):   {empty_matches:,}  ({100*empty_matches/n_rows:.1f}%)")
    print(f"  1 match:                 {single_match:,}  ({100*single_match/n_rows:.1f}%)")
    print(f"  2+ matches:               {multi_match:,}  ({100*multi_match/n_rows:.1f}%)")
    print(f"  max matches for one S1:  {max_matches}")
    print(f"  avg matches per S1:      {total_match_ids/n_rows:.3f}")
    print(f"  matched-only-from-S2:    {s2_only:,}")
    print(f"  matched-only-from-S3:    {s3_only:,}")
    print(f"  matched-from-both:       {both_s2_s3:,}")
    if dup_count:
        print(f"  WARNING: {dup_count} duplicate source1_entity_id rows found")

    if source1_path and os.path.isfile(source1_path):
        s1_ids = set()
        for chunk in pd.read_csv(source1_path, sep="\t", dtype=str, chunksize=CHUNKSIZE, usecols=["entity_id"]):
            s1_ids.update(chunk["entity_id"].tolist())
        gt_ids = set(dup_s1_rows.keys())
        missing_from_gt = s1_ids - gt_ids
        extra_in_gt = gt_ids - s1_ids
        print(f"\n  S1 entities in train_source1.tsv: {len(s1_ids):,}")
        print(f"  S1 entities missing from ground truth: {len(missing_from_gt):,}")
        print(f"  Ground truth rows not in train_source1: {len(extra_in_gt):,}")


def main():
    parser = argparse.ArgumentParser(description="Explore the entity resolution dataset.")
    parser.add_argument("--data-dir", default="dataset", help="Path to the dataset/ folder (default: %(default)s)")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    test_dir = os.path.join(args.data_dir, "test")

    print("Amazon ML Challenge 2026 — Dataset Exploration")
    print(f"data dir: {args.data_dir}")

    explore_source_file(os.path.join(train_dir, "train_source1.tsv"), "TRAIN SOURCE 1")
    explore_source_file(os.path.join(train_dir, "train_source2.tsv"), "TRAIN SOURCE 2")
    explore_source_file(os.path.join(train_dir, "train_source3.tsv"), "TRAIN SOURCE 3")
    explore_ground_truth(
        os.path.join(train_dir, "train_ground_truth.tsv"),
        source1_path=os.path.join(train_dir, "train_source1.tsv"),
    )

    explore_source_file(os.path.join(test_dir, "test_source1.tsv"), "TEST SOURCE 1")
    explore_source_file(os.path.join(test_dir, "test_source2.tsv"), "TEST SOURCE 2")
    explore_source_file(os.path.join(test_dir, "test_source3.tsv"), "TEST SOURCE 3")

    sep("DONE")
    print("Copy/paste the full output above back into the chat.")


if __name__ == "__main__":
    sys.exit(main()) 