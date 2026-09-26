#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Blocking / Candidate Generation (v3)

Changes from v2, based on diagnostics run against the real dataset:
  - v2's --max-candidates-per-entity truncated an unordered Python set
    ARBITRARILY, discarding real matches at the same rate as noise
    (confirmed: 55.3% of true matches that DID share a token were lost
    this way on the real data). v3 tracks a token-hit COUNT per candidate
    (how many name/address tokens it shares with the S1 entity) and,
    when truncating, keeps the highest-hit-count candidates first — a
    true match usually shares several tokens; a coincidental block-mate
    usually shares exactly one.
  - Added ADDRESS-token blocking as a second signal, unioned with name-
    token blocking (diagnostic showed ~14.5% of true matches have zero
    shared name token but DO share an address token).
  - Separate stoplists for name tokens vs address tokens (address text
    has its own generic filler words — "road", "street", "near", "floor",
    "no", "house" — different from name filler words like "llc"/
    "private"/"limited"). Both fit on TRAIN ONLY, reused as-is for test.
  - Raised default --max-block-size and --max-candidates-per-entity now
    that stopword filtering is doing more of the real work of excluding
    generic tokens (the block-size cap should only be catching genuine
    outliers now, not doing the bulk of the filtering).

Strategy recap:
  - Hard filter: country must match exactly.
  - Build inverted indices (country, token) -> [entity_ids] separately
    for name tokens and address tokens, over S2+S3.
  - For each S1 entity, union candidates from every name/address token it
    shares with an S2/S3 record (same country), tracking a hit-count per
    candidate.
  - Buckets larger than --max-block-size are dropped (still needed as a
    backstop against pathological tokens that slip past the stoplist).
  - If an entity's total candidate count exceeds
    --max-candidates-per-entity, keep the highest-hit-count candidates.
  - Candidate pairs stream to disk in batches (memory-safe on the full
    dataset — this was already fixed in v2).

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
import pyarrow as pa
import pyarrow.parquet as pq

PAIR_SCHEMA = pa.schema([
    ("source1_entity_id", pa.string()),
    ("candidate_entity_id", pa.string()),
])


def compute_stop_tokens(text_series, top_k, min_token_len=2):
    counter = Counter()
    for text in text_series:
        if isinstance(text, str) and text:
            counter.update(set(text.split()))
    return {tok for tok, _ in counter.most_common(top_k) if len(tok) >= min_token_len}


def build_token_index(df, stop_tokens, text_col, id_col="entity_id", country_col="country"):
    """Return dict: (country, token) -> list[entity_id]."""
    index = defaultdict(list)
    for eid, country, text in zip(df[id_col], df[country_col], df[text_col]):
        if not isinstance(text, str) or not text:
            continue
        for tok in set(text.split()):
            if tok in stop_tokens or len(tok) < 2:
                continue
            index[(country, tok)].append(eid)
    return index


def generate_candidates_streaming(s1_df, name_index, addr_index, name_stop, addr_stop,
                                   max_block_size, out_path, max_candidates_per_entity,
                                   batch_size=500_000,
                                   id_col="entity_id", name_col="business_name_norm",
                                   addr_col="business_address_norm", country_col="country"):
    """Stream candidate pairs to Parquet in batches. For each S1 entity,
    tracks a Counter of candidate_id -> hit_count (how many name/address
    tokens matched), and keeps the highest-hit-count candidates when the
    total exceeds max_candidates_per_entity — instead of an arbitrary cut."""
    writer = None
    buf_s1, buf_cand = [], []
    total_pairs = 0
    n_entities_capped = 0
    n = len(s1_df)
    t0 = time.time()

    def flush():
        nonlocal writer, buf_s1, buf_cand, total_pairs
        if not buf_s1:
            return
        table = pa.table({"source1_entity_id": buf_s1, "candidate_entity_id": buf_cand}, schema=PAIR_SCHEMA)
        if writer is None:
            writer = pq.ParquetWriter(out_path, PAIR_SCHEMA)
        writer.write_table(table)
        total_pairs += len(buf_s1)
        buf_s1, buf_cand = [], []

    cols = zip(s1_df[id_col], s1_df[country_col], s1_df[name_col], s1_df[addr_col])
    for i, (eid, country, name_text, addr_text) in enumerate(cols):
        hits = Counter()

        if isinstance(name_text, str) and name_text:
            for tok in set(name_text.split()):
                if tok in name_stop or len(tok) < 2:
                    continue
                bucket = name_index.get((country, tok))
                if bucket and len(bucket) <= max_block_size:
                    for cid in bucket:
                        hits[cid] += 1

        if isinstance(addr_text, str) and addr_text:
            for tok in set(addr_text.split()):
                if tok in addr_stop or len(tok) < 2:
                    continue
                bucket = addr_index.get((country, tok))
                if bucket and len(bucket) <= max_block_size:
                    for cid in bucket:
                        hits[cid] += 1

        if max_candidates_per_entity and len(hits) > max_candidates_per_entity:
            keep = [cid for cid, _ in hits.most_common(max_candidates_per_entity)]
            n_entities_capped += 1
        else:
            keep = list(hits.keys())

        for cid in keep:
            buf_s1.append(eid)
            buf_cand.append(cid)

        if len(buf_s1) >= batch_size:
            flush()

        if (i + 1) % 100000 == 0:
            elapsed = time.time() - t0
            print(f"    {i+1:,}/{n:,} S1 entities blocked ({elapsed:.0f}s, {total_pairs + len(buf_s1):,} pairs so far)", end="\r")

    flush()
    if writer is not None:
        writer.close()
    else:
        pq.write_table(pa.table({"source1_entity_id": pa.array([], type=pa.string()),
                                  "candidate_entity_id": pa.array([], type=pa.string())}, schema=PAIR_SCHEMA), out_path)

    print(f"    {n:,}/{n:,} S1 entities blocked" + " " * 30)
    return total_pairs, n_entities_capped


def run_blocking(cache_dir, split, max_block_size=5000, max_candidates_per_entity=800,
                  fit_stopwords=False, top_k_name=150, top_k_addr=80):
    s1_path = os.path.join(cache_dir, f"{split}_source1_normalized.parquet")
    s2_path = os.path.join(cache_dir, f"{split}_source2_normalized.parquet")
    s3_path = os.path.join(cache_dir, f"{split}_source3_normalized.parquet")

    s1_df = pd.read_parquet(
        s1_path, columns=["entity_id", "business_name_norm", "business_address_norm", "country"]
    )
    print(f"  loaded {split}_source1: {len(s1_df):,} rows")

    stop_path = os.path.join(cache_dir, "blocking_stopwords.pkl")
    if fit_stopwords:
        name_stop = compute_stop_tokens(s1_df["business_name_norm"], top_k=top_k_name)
        addr_stop = compute_stop_tokens(s1_df["business_address_norm"], top_k=top_k_addr)
        s2_peek = pd.read_parquet(s2_path, columns=["business_name_norm", "business_address_norm"])
        name_stop |= compute_stop_tokens(s2_peek["business_name_norm"], top_k=top_k_name)
        addr_stop |= compute_stop_tokens(s2_peek["business_address_norm"], top_k=top_k_addr)
        del s2_peek
        print(f"  fit name stoplist ({len(name_stop)} tokens) on TRAIN only: {sorted(name_stop)[:12]}...")
        print(f"  fit addr stoplist ({len(addr_stop)} tokens) on TRAIN only: {sorted(addr_stop)[:12]}...")
        with open(stop_path, "wb") as f:
            pickle.dump({"name": name_stop, "addr": addr_stop}, f)
    else:
        if not os.path.isfile(stop_path):
            raise SystemExit(
                "No stoplist found. Run with --split train --evaluate first "
                "(it fits and saves the shared stoplists), then run --split test."
            )
        with open(stop_path, "rb") as f:
            stops = pickle.load(f)
        name_stop, addr_stop = stops["name"], stops["addr"]
        print(f"  loaded shared stoplists (name={len(name_stop)}, addr={len(addr_stop)}) — "
              f"fit on train, reused as-is for {split}")

    name_index, addr_index = {}, {}
    for src_path, label in [(s2_path, "source2"), (s3_path, "source3")]:
        print(f"  indexing {split}_{label} ...")
        df = pd.read_parquet(
            src_path, columns=["entity_id", "business_name_norm", "business_address_norm", "country"]
        )
        n_idx = build_token_index(df, name_stop, "business_name_norm")
        a_idx = build_token_index(df, addr_stop, "business_address_norm")
        for k, v in n_idx.items():
            name_index.setdefault(k, []).extend(v)
        for k, v in a_idx.items():
            addr_index.setdefault(k, []).extend(v)
        del df, n_idx, a_idx
        print(f"    name index: {len(name_index):,} keys | addr index: {len(addr_index):,} keys")

    print(f"  generating candidates for {split}_source1 ({len(s1_df):,} entities) ...")
    out_path = os.path.join(cache_dir, f"{split}_candidate_pairs.parquet")
    total_pairs, n_capped = generate_candidates_streaming(
        s1_df, name_index, addr_index, name_stop, addr_stop,
        max_block_size, out_path, max_candidates_per_entity,
    )
    del name_index, addr_index

    avg_candidates = total_pairs / max(1, len(s1_df))
    print(f"  -> {out_path}")
    print(f"     {total_pairs:,} candidate pairs, avg {avg_candidates:.1f} candidates/S1 entity")
    if n_capped:
        print(f"     {n_capped:,} S1 entities hit the {max_candidates_per_entity}-candidate cap "
              f"(kept their highest-token-overlap candidates, not arbitrary ones)")

    return out_path, s1_df


def evaluate_recall(cache_dir, candidate_pairs_path, batch_size=1_000_000):
    gt_path = os.path.join(cache_dir, "train_ground_truth.parquet")
    if not os.path.isfile(gt_path):
        print("  [skip recall eval] train_ground_truth.parquet not found")
        return

    gt = pd.read_parquet(gt_path)
    gt_dict = {}
    for s1_id, matched in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        ids = set(x for x in matched.split(",") if x.strip()) if isinstance(matched, str) else set()
        gt_dict[s1_id] = ids
    del gt

    cand_dict = defaultdict(set)
    pf = pq.ParquetFile(candidate_pairs_path)
    for batch in pf.iter_batches(batch_size=batch_size):
        chunk = batch.to_pandas()
        for s1_id, cid in zip(chunk["source1_entity_id"], chunk["candidate_entity_id"]):
            cand_dict[s1_id].add(cid)

    total_true = total_found = entities_full_recall = entities_with_true_matches = 0
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
    parser.add_argument("--max-candidates-per-entity", type=int, default=800,
                         help="Cap on candidates kept per S1 entity; keeps the HIGHEST token-overlap ones when exceeded (default: %(default)s)")
    parser.add_argument("--top-k-name-stopwords", type=int, default=150,
                         help="Most-frequent NAME tokens to exclude as blocking keys, fit on train only (default: %(default)s)")
    parser.add_argument("--top-k-addr-stopwords", type=int, default=80,
                         help="Most-frequent ADDRESS tokens to exclude as blocking keys, fit on train only (default: %(default)s)")
    parser.add_argument("--evaluate", action="store_true",
                         help="After blocking, measure recall against train_ground_truth.parquet (train split only)")
    args = parser.parse_args()

    print(f"Blocking for split={args.split}")
    fit_stopwords = (args.split == "train")
    out_path, s1_df = run_blocking(
        args.cache_dir, args.split,
        max_block_size=args.max_block_size,
        max_candidates_per_entity=args.max_candidates_per_entity,
        fit_stopwords=fit_stopwords,
        top_k_name=args.top_k_name_stopwords,
        top_k_addr=args.top_k_addr_stopwords,
    )

    if args.evaluate:
        if args.split != "train":
            print("  --evaluate only works on --split train (no ground truth for test)")
        else:
            evaluate_recall(args.cache_dir, out_path)


if __name__ == "__main__":
    sys.exit(main())