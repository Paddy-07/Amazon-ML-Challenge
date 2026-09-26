#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Blocking / Candidate Generation

SHARED by both the LightGBM and XGBoost branches — run once, use the same
cache/*_candidate_pairs.parquet for both.

Strategy: token-based inverted-index blocking.
  - Hard filter: country must match exactly (confirmed 0 mismatches across
    82 real sampled true-match pairs during data exploration).
  - For each (country, normalized-name-token) key, build a list of S2/S3
    entity_ids that contain that token.
  - A Source-1 entity's candidates = union of every S2/S3 entity sharing
    at least one normalized-name token, within the same country.
  - The most frequent tokens across the corpus ("llc", "private",
    "limited", "corporation", ...) are excluded from being *blocking
    keys* (they'd create huge, useless blocks) — the stoplist is FIT ON
    TRAIN ONLY and then reused as-is for test, so blocking is defined
    identically on both splits (no mixing).
  - Buckets larger than --max-block-size are dropped entirely (a token
    that slipped past the stoplist but still explodes) rather than
    included, to keep runtime/memory bounded.

This only uses normalized *name* tokens for blocking (not address) to
keep the inverted index a manageable size on the ~10M-row test S2+S3
files. Address is instead used downstream as a *feature*, not a blocking
key. If measured recall (see --evaluate) turns out too low, address-token
blocking can be added as a second pass unioned into the same candidate
sets — check the recall report this script prints before deciding.

Usage (run from student_resource/, after preprocess_dataset.py):

    python3 blocking.py --cache-dir cache --split train --evaluate
    python3 blocking.py --cache-dir cache --split test

Requires: pandas, pyarrow
"""

import argparse
import os
import pickle
import sys
import time
from collections import Counter, defaultdict

import pandas as pd


def compute_stop_tokens(name_norm_series, top_k=40, min_token_len=2):
    """Most frequent normalized-name tokens across the corpus — excluded
    from blocking keys since they're too generic to discriminate (e.g.
    "llc", "private", "limited", "inc", "group", "services")."""
    counter = Counter()
    for text in name_norm_series:
        if isinstance(text, str) and text:
            counter.update(set(text.split()))
    stop = {tok for tok, _ in counter.most_common(top_k) if len(tok) >= min_token_len}
    return stop


def build_token_index(df, stop_tokens, id_col="entity_id",
                       name_col="business_name_norm", country_col="country"):
    """Return dict: (country, token) -> list[entity_id]."""
    index = defaultdict(list)
    for eid, country, text in zip(df[id_col], df[country_col], df[name_col]):
        if not isinstance(text, str) or not text:
            continue
        for tok in set(text.split()):
            if tok in stop_tokens or len(tok) < 2:
                continue
            index[(country, tok)].append(eid)
    return index


def generate_candidates(s1_df, index, stop_tokens, max_block_size,
                         id_col="entity_id", name_col="business_name_norm",
                         country_col="country"):
    """For each S1 row, union candidate ids from all its normalized-name
    tokens' blocks (same country). Returns list of (s1_id, candidate_id)
    pairs (long format)."""
    pairs = []
    n = len(s1_df)
    t0 = time.time()
    for i, (eid, country, text) in enumerate(zip(s1_df[id_col], s1_df[country_col], s1_df[name_col])):
        cands = set()
        if isinstance(text, str) and text:
            for tok in set(text.split()):
                if tok in stop_tokens or len(tok) < 2:
                    continue
                bucket = index.get((country, tok))
                if bucket and len(bucket) <= max_block_size:
                    cands.update(bucket)
        for cid in cands:
            pairs.append((eid, cid))
        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1:,}/{n:,} S1 entities blocked ({elapsed:.0f}s)", end="\r")
    print(f"    {n:,}/{n:,} S1 entities blocked" + " " * 20)
    return pairs


def run_blocking(cache_dir, split, stop_tokens=None, max_block_size=5000, fit_stopwords=False, top_k=40):
    s1_path = os.path.join(cache_dir, f"{split}_source1_normalized.parquet")
    s2_path = os.path.join(cache_dir, f"{split}_source2_normalized.parquet")
    s3_path = os.path.join(cache_dir, f"{split}_source3_normalized.parquet")

    s1_df = pd.read_parquet(s1_path, columns=["entity_id", "business_name_norm", "country"])
    print(f"  loaded {split}_source1: {len(s1_df):,} rows")

    if fit_stopwords:
        # Fit ONLY on train S1's own tokens plus a peek at S2 — kept simple
        # and train-only so the stoplist definition never depends on test.
        stop_tokens = compute_stop_tokens(s1_df["business_name_norm"], top_k=top_k)
        s2_peek = pd.read_parquet(s2_path, columns=["business_name_norm"])
        stop_tokens |= compute_stop_tokens(s2_peek["business_name_norm"], top_k=top_k)
        del s2_peek
        print(f"  fit stoplist ({len(stop_tokens)} tokens) on TRAIN only: {sorted(stop_tokens)[:15]}...")
        with open(os.path.join(cache_dir, "blocking_stopwords.pkl"), "wb") as f:
            pickle.dump(stop_tokens, f)

    if stop_tokens is None:
        stop_path = os.path.join(cache_dir, "blocking_stopwords.pkl")
        if not os.path.isfile(stop_path):
            raise SystemExit(
                "No stoplist found. Run with --split train --evaluate first "
                "(it fits and saves the shared stoplist), then run --split test."
            )
        with open(stop_path, "rb") as f:
            stop_tokens = pickle.load(f)
        print(f"  loaded shared stoplist ({len(stop_tokens)} tokens) — fit on train, reused as-is for {split}")

    index = {}
    for src_path, label in [(s2_path, "source2"), (s3_path, "source3")]:
        print(f"  indexing {split}_{label} ...")
        df = pd.read_parquet(src_path, columns=["entity_id", "business_name_norm", "country"])
        idx = build_token_index(df, stop_tokens)
        for k, v in idx.items():
            index.setdefault(k, []).extend(v)
        del df, idx
        print(f"    index now has {len(index):,} (country, token) keys")

    print(f"  generating candidates for {split}_source1 ({len(s1_df):,} entities) ...")
    pairs = generate_candidates(s1_df, index, stop_tokens, max_block_size)
    pairs_df = pd.DataFrame(pairs, columns=["source1_entity_id", "candidate_entity_id"])

    out_path = os.path.join(cache_dir, f"{split}_candidate_pairs.parquet")
    pairs_df.to_parquet(out_path, index=False)

    n_s1_with_candidates = pairs_df["source1_entity_id"].nunique()
    avg_candidates = len(pairs_df) / max(1, len(s1_df))
    print(f"  -> {out_path}")
    print(f"     {len(pairs_df):,} candidate pairs, avg {avg_candidates:.1f} candidates/S1 entity")
    print(f"     {n_s1_with_candidates:,}/{len(s1_df):,} S1 entities have >=1 candidate "
          f"({100*n_s1_with_candidates/len(s1_df):.1f}%)")

    return pairs_df, stop_tokens, s1_df


def evaluate_recall(cache_dir, pairs_df, s1_df):
    gt_path = os.path.join(cache_dir, "train_ground_truth.parquet")
    if not os.path.isfile(gt_path):
        print("  [skip recall eval] train_ground_truth.parquet not found")
        return

    gt = pd.read_parquet(gt_path)
    gt_dict = {}
    for s1_id, matched in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        ids = set(x for x in matched.split(",") if x.strip()) if isinstance(matched, str) else set()
        gt_dict[s1_id] = ids

    cand_dict = defaultdict(set)
    for s1_id, cid in zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"]):
        cand_dict[s1_id].add(cid)

    total_true = 0
    total_found = 0
    entities_full_recall = 0
    entities_with_true_matches = 0
    for s1_id, true_ids in gt_dict.items():
        if not true_ids:
            continue
        entities_with_true_matches += 1
        found = true_ids & cand_dict.get(s1_id, set())
        total_true += len(true_ids)
        total_found += len(found)
        if len(found) == len(true_ids):
            entities_full_recall += 1

    print("\n  BLOCKING RECALL (train):")
    print(f"    micro recall (true matches captured): {total_found:,}/{total_true:,} = {100*total_found/max(1,total_true):.2f}%")
    print(f"    entities with ALL true matches captured: {entities_full_recall:,}/{entities_with_true_matches:,} "
          f"= {100*entities_full_recall/max(1,entities_with_true_matches):.2f}%")
    print("    (this recall is the CEILING on your final F_0.5 recall — missed here can never be recovered by the model)")


def main():
    parser = argparse.ArgumentParser(description="Blocking / candidate generation.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--max-block-size", type=int, default=5000,
                         help="Drop a token's block entirely if it has more than this many entities (default: %(default)s)")
    parser.add_argument("--top-k-stopwords", type=int, default=40,
                         help="How many most-frequent name tokens to exclude as blocking keys, fit on train only (default: %(default)s)")
    parser.add_argument("--evaluate", action="store_true",
                         help="After blocking, measure recall against train_ground_truth.parquet (train split only)")
    args = parser.parse_args()

    print(f"Blocking for split={args.split}")
    fit_stopwords = (args.split == "train")
    pairs_df, stop_tokens, s1_df = run_blocking(
        args.cache_dir, args.split,
        max_block_size=args.max_block_size,
        fit_stopwords=fit_stopwords,
        top_k=args.top_k_stopwords,
    )

    if args.evaluate:
        if args.split != "train":
            print("  --evaluate only works on --split train (no ground truth for test)")
        else:
            evaluate_recall(args.cache_dir, pairs_df, s1_df)


if __name__ == "__main__":
    sys.exit(main())