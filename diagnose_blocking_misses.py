#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Blocking-Miss Diagnostic

Answers the question: "for the true matches, how many share zero
normalized-name tokens (so name-token blocking can never find them), and
of those, how many DO share an address token (so address-based blocking
would rescue them)?"

This is a pure text-overlap check against a random sample of real
ground-truth pairs — it doesn't touch the (huge) already-generated
train_candidate_pairs.parquet at all, so it's cheap to run regardless of
how large that file got.

Usage (run from student_resource/, after preprocess_dataset.py):

    python3 diagnose_blocking_misses.py --cache-dir cache --n 5000

Requires: pandas, pyarrow
"""

import argparse
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

CHUNK_BATCH = 500_000


def load_needed_norm_records(path, needed_ids):
    """Stream a (potentially huge) normalized-cache parquet file in
    batches, keeping only rows whose entity_id is in `needed_ids`."""
    found = {}
    if not needed_ids:
        return found
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(
        batch_size=CHUNK_BATCH,
        columns=["entity_id", "business_name_norm", "business_address_norm"],
    ):
        chunk = batch.to_pandas()
        hit = chunk[chunk["entity_id"].isin(needed_ids)]
        if len(hit):
            for _, row in hit.iterrows():
                found[row["entity_id"]] = {
                    "name_norm": row["business_name_norm"] or "",
                    "addr_norm": row["business_address_norm"] or "",
                }
        if len(found) == len(needed_ids):
            break
    return found


def has_token_overlap(a, b):
    if not a or not b:
        return False
    return bool(set(a.split()) & set(b.split()))


def main():
    parser = argparse.ArgumentParser(description="Diagnose why blocking is missing true matches.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--n", type=int, default=5000, help="Number of S1 entities to sample (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    gt_path = os.path.join(args.cache_dir, "train_ground_truth.parquet")
    gt = pd.read_parquet(gt_path)
    gt = gt[gt["matched_entity_ids"].str.len() > 0]  # only entities that actually have matches

    sample = gt.sample(n=min(args.n, len(gt)), random_state=args.seed)
    print(f"Sampling {len(sample):,} S1 entities with at least one true match")

    needed_s1 = set(sample["source1_entity_id"])
    needed_s2, needed_s3 = set(), set()
    match_lists = {}
    for s1_id, matched in zip(sample["source1_entity_id"], sample["matched_entity_ids"]):
        ids = [x for x in matched.split(",") if x.strip()]
        match_lists[s1_id] = ids
        for i in ids:
            (needed_s2 if i.startswith("S2-") else needed_s3).add(i)

    print(f"Need {len(needed_s1):,} S1, {len(needed_s2):,} S2, {len(needed_s3):,} S3 records ...")

    s1_recs = load_needed_norm_records(os.path.join(args.cache_dir, "train_source1_normalized.parquet"), needed_s1)
    print("  loaded S1 records")
    s2_recs = load_needed_norm_records(os.path.join(args.cache_dir, "train_source2_normalized.parquet"), needed_s2)
    print("  loaded S2 records (streamed through the full file)")
    s3_recs = load_needed_norm_records(os.path.join(args.cache_dir, "train_source3_normalized.parquet"), needed_s3)
    print("  loaded S3 records (streamed through the full file)")

    total_pairs = 0
    name_overlap = 0
    no_name_but_addr_overlap = 0
    neither = 0
    missing_records = 0

    for s1_id, ids in match_lists.items():
        s1_rec = s1_recs.get(s1_id)
        if s1_rec is None:
            continue
        for mid in ids:
            rec = s2_recs.get(mid) if mid.startswith("S2-") else s3_recs.get(mid)
            if rec is None:
                missing_records += 1
                continue
            total_pairs += 1
            if has_token_overlap(s1_rec["name_norm"], rec["name_norm"]):
                name_overlap += 1
            elif has_token_overlap(s1_rec["addr_norm"], rec["addr_norm"]):
                no_name_but_addr_overlap += 1
            else:
                neither += 1

    print("\n" + "=" * 70)
    print("RESULTS (fraction of real true-match pairs)")
    print("=" * 70)
    print(f"  total true-match pairs checked:              {total_pairs:,}")
    print(f"  have >=1 shared NAME token:                   {name_overlap:,}  ({100*name_overlap/max(1,total_pairs):.1f}%)  <- current blocking can find these")
    print(f"  NO name overlap, but >=1 shared ADDRESS token: {no_name_but_addr_overlap:,}  ({100*no_name_but_addr_overlap/max(1,total_pairs):.1f}%)  <- address blocking would rescue these")
    print(f"  NO overlap in either field:                    {neither:,}  ({100*neither/max(1,total_pairs):.1f}%)  <- neither approach catches these; need char n-gram / fuzzier blocking")
    if missing_records:
        print(f"  (skipped {missing_records:,} pairs — matched ID not found in normalized cache, unexpected, worth investigating)")


if __name__ == "__main__":
    sys.exit(main())