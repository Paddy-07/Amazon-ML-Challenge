#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Matched-Pair Inspector

Pulls a random sample of S1 entities from train_ground_truth.tsv and prints
each one next to its actual matched S2/S3 records, so we can SEE how noisy a
true match really looks (name/address variation, transliteration, country
agreement) instead of guessing from aggregate stats.

Memory-conscious: streams through the (large) source2/source3 files in
chunks and only keeps rows whose entity_id is in our small sampled "needed"
set — it never loads the full files into memory.

Usage (run from student_resource/, or pass --data-dir):

    python3 inspect_matches.py --data-dir dataset --n 25 --seed 42

Requires: pandas
"""

import argparse
import os
import random
import sys

import pandas as pd

CHUNKSIZE = 200_000


def load_needed_records(path, needed_ids, id_col="entity_id"):
    """Stream a large source file in chunks, keeping only rows whose ID is
    in `needed_ids`. Returns {entity_id: {name, address, country}}."""
    found = {}
    if not needed_ids:
        return found
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNKSIZE):
        hit = chunk[chunk[id_col].isin(needed_ids)]
        if len(hit):
            for _, row in hit.iterrows():
                found[row[id_col]] = {
                    "name": row.get("business_name", ""),
                    "address": row.get("business_address", ""),
                    "country": row.get("country", ""),
                }
        if len(found) == len(needed_ids):
            break  # found everything, stop scanning early
    return found


def main():
    parser = argparse.ArgumentParser(description="Inspect real matched pairs from ground truth.")
    parser.add_argument("--data-dir", default="dataset", help="Path to dataset/ (default: %(default)s)")
    parser.add_argument("--n", type=int, default=25, help="Number of S1 entities to sample (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: %(default)s)")
    parser.add_argument("--only-multi", action="store_true",
                         help="Only sample S1 entities with 2+ matches (skip singletons/1-match)")
    args = parser.parse_args()

    train_dir = os.path.join(args.data_dir, "train")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")

    print("Loading ground truth...")
    gt = pd.read_csv(gt_path, sep="\t", dtype=str)
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    gt["match_list"] = gt["matched_entity_ids"].apply(lambda s: [x for x in s.split(",") if x.strip()])

    if args.only_multi:
        gt = gt[gt["match_list"].apply(lambda ids: len(ids) >= 2)]

    random.seed(args.seed)
    sample_idx = random.sample(range(len(gt)), min(args.n, len(gt)))
    sample = gt.iloc[sample_idx]

    needed_s1 = set(sample["source1_entity_id"])
    needed_s2 = set()
    needed_s3 = set()
    for ids in sample["match_list"]:
        for i in ids:
            if i.startswith("S2-"):
                needed_s2.add(i)
            elif i.startswith("S3-"):
                needed_s3.add(i)

    print(f"Sampling {len(needed_s1)} S1 entities -> {len(needed_s2)} S2 matches, {len(needed_s3)} S3 matches")

    print("Scanning train_source1.tsv for needed S1 records...")
    s1_records = load_needed_records(s1_path, needed_s1)
    print("Scanning train_source2.tsv for needed S2 records (this streams the whole file)...")
    s2_records = load_needed_records(s2_path, needed_s2)
    print("Scanning train_source3.tsv for needed S3 records (this streams the whole file)...")
    s3_records = load_needed_records(s3_path, needed_s3)

    country_mismatches = 0
    total_pairs = 0

    print("\n" + "=" * 100)
    print("SAMPLED MATCHED PAIRS")
    print("=" * 100)

    for _, row in sample.iterrows():
        s1_id = row["source1_entity_id"]
        s1_rec = s1_records.get(s1_id)
        if s1_rec is None:
            print(f"\n[!] {s1_id} not found in train_source1.tsv (unexpected)")
            continue

        print(f"\n--- {s1_id}  [{len(row['match_list'])} match(es)] ---")
        print(f"  S1  | name: {s1_rec['name']!r}")
        print(f"      | addr: {s1_rec['address']!r}")
        print(f"      | country: {s1_rec['country']}")

        for mid in row["match_list"]:
            rec = s2_records.get(mid) if mid.startswith("S2-") else s3_records.get(mid)
            total_pairs += 1
            if rec is None:
                print(f"  {mid} | [NOT FOUND — unexpected]")
                continue
            mismatch = "  <-- COUNTRY MISMATCH" if rec["country"] != s1_rec["country"] else ""
            if mismatch:
                country_mismatches += 1
            print(f"  {mid} | name: {rec['name']!r}")
            print(f"      | addr: {rec['address']!r}")
            print(f"      | country: {rec['country']}{mismatch}")

    print("\n" + "=" * 100)
    print(f"Country mismatches on true matches: {country_mismatches} / {total_pairs} pairs")
    print("=" * 100)


if __name__ == "__main__":
    sys.exit(main())