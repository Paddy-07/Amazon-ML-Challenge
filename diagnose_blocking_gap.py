#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Blocking Generation-Loss Diagnostic

Follow-up to diagnose_blocking_misses.py. That script showed most true
matches DO share a name or address token with their S1 entity — so if
your actual measured blocking recall is much lower than that, something
in the GENERATION step (the --max-block-size bucket drop, or the
--max-candidates-per-entity truncation) is throwing real candidates away.

This script re-samples the SAME set of S1 entities (same seed), finds the
true-match pairs that DO share a name token (the ones blocking should
have caught), and checks whether they actually landed in the real
train_candidate_pairs.parquet you generated. It streams that file once
(it's large — this will take a few minutes), filtering down to just the
sampled S1 entities.

Usage (run from student_resource/, after blocking.py --split train):

    python3 diagnose_blocking_gap.py --cache-dir cache --n 5000

Requires: pandas, pyarrow
"""

import argparse
import os
import sys

import pandas as pd
import pyarrow.parquet as pq

CHUNK_BATCH = 500_000


def load_needed_norm_records(path, needed_ids):
    found = {}
    if not needed_ids:
        return found
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(
        batch_size=CHUNK_BATCH,
        columns=["entity_id", "business_name_norm"],
    ):
        chunk = batch.to_pandas()
        hit = chunk[chunk["entity_id"].isin(needed_ids)]
        if len(hit):
            for _, row in hit.iterrows():
                found[row["entity_id"]] = row["business_name_norm"] or ""
        if len(found) == len(needed_ids):
            break
    return found


def has_token_overlap(a, b):
    if not a or not b:
        return False
    return bool(set(a.split()) & set(b.split()))


def main():
    parser = argparse.ArgumentParser(description="Diagnose where blocking generation is losing candidates.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42, help="MUST match the seed used in diagnose_blocking_misses.py to reproduce the same sample")
    args = parser.parse_args()

    gt_path = os.path.join(args.cache_dir, "train_ground_truth.parquet")
    gt = pd.read_parquet(gt_path)
    gt = gt[gt["matched_entity_ids"].str.len() > 0]
    sample = gt.sample(n=min(args.n, len(gt)), random_state=args.seed)
    print(f"Sampled {len(sample):,} S1 entities (same seed as diagnose_blocking_misses.py)")

    needed_s1 = set(sample["source1_entity_id"])
    needed_s2, needed_s3 = set(), set()
    match_lists = {}
    for s1_id, matched in zip(sample["source1_entity_id"], sample["matched_entity_ids"]):
        ids = [x for x in matched.split(",") if x.strip()]
        match_lists[s1_id] = ids
        for i in ids:
            (needed_s2 if i.startswith("S2-") else needed_s3).add(i)

    print("Loading normalized names for name-overlap check ...")
    s1_names = load_needed_norm_records(os.path.join(args.cache_dir, "train_source1_normalized.parquet"), needed_s1)
    s2_names = load_needed_norm_records(os.path.join(args.cache_dir, "train_source2_normalized.parquet"), needed_s2)
    s3_names = load_needed_norm_records(os.path.join(args.cache_dir, "train_source3_normalized.parquet"), needed_s3)

    # the pairs blocking SHOULD have caught (real name-token overlap exists)
    should_catch = []  # list of (s1_id, candidate_id)
    for s1_id, ids in match_lists.items():
        s1_name = s1_names.get(s1_id, "")
        for mid in ids:
            cand_name = s2_names.get(mid) if mid.startswith("S2-") else s3_names.get(mid)
            if cand_name is None:
                continue
            if has_token_overlap(s1_name, cand_name):
                should_catch.append((s1_id, mid))

    print(f"{len(should_catch):,} true-match pairs have a shared name token (blocking should find these)")

    print(f"\nStreaming {args.cache_dir}/train_candidate_pairs.parquet to check which actually made it in ...")
    print("(this is a big file — full scan will take a few minutes)")

    actual_candidates = {}  # s1_id -> set of candidate ids, ONLY for our sampled entities
    pf = pq.ParquetFile(os.path.join(args.cache_dir, "train_candidate_pairs.parquet"))
    rows_scanned = 0
    for batch in pf.iter_batches(batch_size=CHUNK_BATCH):
        chunk = batch.to_pandas()
        rows_scanned += len(chunk)
        hit = chunk[chunk["source1_entity_id"].isin(needed_s1)]
        for s1_id, cid in zip(hit["source1_entity_id"], hit["candidate_entity_id"]):
            actual_candidates.setdefault(s1_id, set()).add(cid)
        print(f"  scanned {rows_scanned:,} rows ...", end="\r")
    print(f"  scanned {rows_scanned:,} rows total" + " " * 20)

    found = sum(1 for s1_id, cid in should_catch if cid in actual_candidates.get(s1_id, set()))
    lost = len(should_catch) - found

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  true-match pairs with shared name token:  {len(should_catch):,}")
    print(f"  actually present in candidate_pairs.parquet: {found:,}  ({100*found/max(1,len(should_catch)):.1f}%)")
    print(f"  LOST despite shared token:                    {lost:,}  ({100*lost/max(1,len(should_catch)):.1f}%)")
    print("\n  If 'LOST' is large, the --max-block-size bucket-drop and/or")
    print("  --max-candidates-per-entity truncation are throwing away real")
    print("  matches that token overlap alone would have caught.")


if __name__ == "__main__":
    sys.exit(main())