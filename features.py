#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Feature Engineering

SHARED by both the LightGBM and XGBoost branches. Run once per split, use
the same cache/{split}_features.parquet for both model scripts.

Features computed per (S1, candidate) pair, on the NORMALIZED name/address
text produced by normalize.py:
  - name_token_jaccard, addr_token_jaccard      (order-invariant overlap)
  - name_fuzz_ratio, name_fuzz_token_sort_ratio  (rapidfuzz, char-level +
    word-order-invariant — token_sort_ratio specifically targets the
    "United Bny Clinic" vs "United Clinic Bny" word-reordering pattern)
  - addr_fuzz_ratio, addr_fuzz_token_sort_ratio
  - name_chargram_tfidf_cosine, addr_chargram_tfidf_cosine
    (character n-gram TF-IDF cosine — the backup signal for domain-blob
    names like "teamsterslocal425.com" that word-level tokenizing can't
    match, and partial credit for romanized-but-imperfect Indic names)
  - name_len_diff, addr_len_diff                 (normalized text length
    difference, a weak but free signal)
  - addr_missing_s1, addr_missing_cand           (explicit flags — do NOT
    let a missing address silently look like "0% similar"; several real
    true-match pairs in the sample data had a missing address on one side)

The TF-IDF vectorizer is FIT ON TRAIN ONLY (names+addresses from
train_source1/2/3) and saved to cache/tfidf_name.pkl / tfidf_addr.pkl —
the test-split run loads and reuses those exact same fitted vectorizers
rather than re-fitting on test text. This keeps the two splits' feature
definitions identical (the same "avoid mixing" principle as the blocking
stoplist).

Usage (run from student_resource/, after blocking.py):

    python3 features.py --cache-dir cache --split train
    python3 features.py --cache-dir cache --split test

Requires: pandas, pyarrow, scikit-learn, rapidfuzz, scipy, joblib
"""

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    sa, sb = set(a.split()), set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def load_source_lookup(cache_dir, split, source_num):
    path = os.path.join(cache_dir, f"{split}_source{source_num}_normalized.parquet")
    df = pd.read_parquet(
        path,
        columns=["entity_id", "business_name", "business_address",
                 "business_name_norm", "business_address_norm"],
    )
    df["business_name_norm"] = df["business_name_norm"].fillna("")
    df["business_address_norm"] = df["business_address_norm"].fillna("")
    return df


def attach_candidate_fields(pairs_df, s2_df, s3_df):
    """Attach candidate-side (business_name_norm, business_address_norm,
    raw fields) by splitting on ID prefix and merging with the right
    source table — vectorized pandas merges, no per-row Python dict
    lookups (important at multi-million-row scale)."""
    is_s2 = pairs_df["candidate_entity_id"].str.startswith("S2-")

    s2_part = pairs_df[is_s2].merge(
        s2_df, left_on="candidate_entity_id", right_on="entity_id", how="left"
    )
    s3_part = pairs_df[~is_s2].merge(
        s3_df, left_on="candidate_entity_id", right_on="entity_id", how="left"
    )
    out = pd.concat([s2_part, s3_part], ignore_index=True)
    out = out.rename(columns={
        "business_name": "cand_name_raw",
        "business_address": "cand_addr_raw",
        "business_name_norm": "cand_name_norm",
        "business_address_norm": "cand_addr_norm",
    })
    out = out.drop(columns=["entity_id"])
    return out


def attach_s1_fields(pairs_df, s1_df):
    out = pairs_df.merge(s1_df, left_on="source1_entity_id", right_on="entity_id", how="left")
    out = out.rename(columns={
        "business_name": "s1_name_raw",
        "business_address": "s1_addr_raw",
        "business_name_norm": "s1_name_norm",
        "business_address_norm": "s1_addr_norm",
    })
    out = out.drop(columns=["entity_id"])
    return out


def fit_or_load_tfidf(cache_dir, split, s1_df, s2_df, s3_df):
    """Fit char-n-gram TF-IDF vectorizers on TRAIN text only; on test,
    load and reuse the exact same fitted vectorizers (never refit on test
    text — keeps feature definitions identical across splits)."""
    name_path = os.path.join(cache_dir, "tfidf_name.pkl")
    addr_path = os.path.join(cache_dir, "tfidf_addr.pkl")

    if split == "train":
        all_names = pd.concat([s1_df["business_name_norm"], s2_df["business_name_norm"], s3_df["business_name_norm"]])
        all_addrs = pd.concat([s1_df["business_address_norm"], s2_df["business_address_norm"], s3_df["business_address_norm"]])
        name_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=200000)
        addr_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, max_features=200000)
        name_vec.fit(all_names[all_names.str.len() > 0])
        addr_vec.fit(all_addrs[all_addrs.str.len() > 0])
        joblib.dump(name_vec, name_path)
        joblib.dump(addr_vec, addr_path)
        print(f"  fit TF-IDF vectorizers on TRAIN text, saved to {name_path} / {addr_path}")
    else:
        if not (os.path.isfile(name_path) and os.path.isfile(addr_path)):
            raise SystemExit(
                "No fitted TF-IDF vectorizers found. Run `features.py --split train` "
                "first — it fits and saves them; test reuses the same ones."
            )
        name_vec = joblib.load(name_path)
        addr_vec = joblib.load(addr_path)
        print(f"  loaded TF-IDF vectorizers fit on TRAIN, reused as-is for {split}")

    return name_vec, addr_vec


def batched_pairwise_cosine(df, name_col_a, name_col_b, vectorizer, batch_size=200000):
    """Cosine similarity between two columns of text, row-by-row, computed
    in batches via sparse TF-IDF transform + row-wise dot product (avoids
    materializing a full pairwise similarity matrix — only the matching
    row pairs are needed, not all-vs-all)."""
    n = len(df)
    out = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        a_text = df[name_col_a].iloc[start:end].tolist()
        b_text = df[name_col_b].iloc[start:end].tolist()
        a_mat = vectorizer.transform(a_text)
        b_mat = vectorizer.transform(b_text)
        # row-wise dot product of two equally-shaped sparse matrices
        dots = np.asarray(a_mat.multiply(b_mat).sum(axis=1)).ravel()
        a_norm = np.sqrt(np.asarray(a_mat.multiply(a_mat).sum(axis=1)).ravel())
        b_norm = np.sqrt(np.asarray(b_mat.multiply(b_mat).sum(axis=1)).ravel())
        denom = a_norm * b_norm
        denom[denom == 0] = 1.0
        out[start:end] = dots / denom
    return out


def build_features(cache_dir, split):
    print(f"Building features for split={split}")

    pairs_path = os.path.join(cache_dir, f"{split}_candidate_pairs.parquet")
    pairs_df = pd.read_parquet(pairs_path)
    print(f"  loaded {len(pairs_df):,} candidate pairs")

    s1_df = load_source_lookup(cache_dir, split, 1)
    s2_df = load_source_lookup(cache_dir, split, 2)
    s3_df = load_source_lookup(cache_dir, split, 3)

    df = attach_s1_fields(pairs_df, s1_df)
    df = attach_candidate_fields(df, s2_df, s3_df)
    df["s1_name_norm"] = df["s1_name_norm"].fillna("")
    df["s1_addr_norm"] = df["s1_addr_norm"].fillna("")
    df["cand_name_norm"] = df["cand_name_norm"].fillna("")
    df["cand_addr_norm"] = df["cand_addr_norm"].fillna("")

    print("  computing token-Jaccard features ...")
    df["name_token_jaccard"] = [
        _jaccard(a, b) for a, b in zip(df["s1_name_norm"], df["cand_name_norm"])
    ]
    df["addr_token_jaccard"] = [
        _jaccard(a, b) for a, b in zip(df["s1_addr_norm"], df["cand_addr_norm"])
    ]

    print("  computing rapidfuzz features ...")
    df["name_fuzz_ratio"] = [
        fuzz.ratio(a, b) / 100.0 for a, b in zip(df["s1_name_norm"], df["cand_name_norm"])
    ]
    df["name_fuzz_token_sort_ratio"] = [
        fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(df["s1_name_norm"], df["cand_name_norm"])
    ]
    df["addr_fuzz_ratio"] = [
        fuzz.ratio(a, b) / 100.0 for a, b in zip(df["s1_addr_norm"], df["cand_addr_norm"])
    ]
    df["addr_fuzz_token_sort_ratio"] = [
        fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(df["s1_addr_norm"], df["cand_addr_norm"])
    ]

    print("  fitting/loading TF-IDF vectorizers ...")
    name_vec, addr_vec = fit_or_load_tfidf(cache_dir, split, s1_df, s2_df, s3_df)

    print("  computing char n-gram TF-IDF cosine features ...")
    df["name_chargram_tfidf_cosine"] = batched_pairwise_cosine(df, "s1_name_norm", "cand_name_norm", name_vec)
    df["addr_chargram_tfidf_cosine"] = batched_pairwise_cosine(df, "s1_addr_norm", "cand_addr_norm", addr_vec)

    print("  computing length / missingness features ...")
    df["name_len_diff"] = (df["s1_name_norm"].str.len() - df["cand_name_norm"].str.len()).abs()
    df["addr_len_diff"] = (df["s1_addr_norm"].str.len() - df["cand_addr_norm"].str.len()).abs()
    df["addr_missing_s1"] = (df["s1_addr_norm"].str.len() == 0).astype(int)
    df["addr_missing_cand"] = (df["cand_addr_norm"].str.len() == 0).astype(int)

    if split == "train":
        print("  attaching labels from ground truth ...")
        gt_path = os.path.join(cache_dir, "train_ground_truth.parquet")
        gt = pd.read_parquet(gt_path)
        gt_dict = {}
        for s1_id, matched in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
            ids = set(x for x in matched.split(",") if x.strip()) if isinstance(matched, str) else set()
            gt_dict[s1_id] = ids
        df["label"] = [
            1 if cid in gt_dict.get(s1_id, set()) else 0
            for s1_id, cid in zip(df["source1_entity_id"], df["candidate_entity_id"])
        ]
        print(f"  label balance: {df['label'].sum():,} positive / {len(df) - df['label'].sum():,} negative")

    feature_cols = [
        "name_token_jaccard", "addr_token_jaccard",
        "name_fuzz_ratio", "name_fuzz_token_sort_ratio",
        "addr_fuzz_ratio", "addr_fuzz_token_sort_ratio",
        "name_chargram_tfidf_cosine", "addr_chargram_tfidf_cosine",
        "name_len_diff", "addr_len_diff",
        "addr_missing_s1", "addr_missing_cand",
    ]
    keep_cols = ["source1_entity_id", "candidate_entity_id"] + feature_cols
    if split == "train":
        keep_cols.append("label")

    out_df = df[keep_cols].copy()
    out_path = os.path.join(cache_dir, f"{split}_features.parquet")
    out_df.to_parquet(out_path, index=False)
    print(f"  -> {out_path}  ({len(out_df):,} rows, {len(feature_cols)} features)")
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Feature engineering for candidate pairs.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--split", choices=["train", "test"], required=True)
    args = parser.parse_args()
    build_features(args.cache_dir, args.split)


if __name__ == "__main__":
    sys.exit(main())