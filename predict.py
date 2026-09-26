#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Shared Prediction / Output Script

SHARED script — pass --model lightgbm or --model xgboost. Loads the
corresponding trained model + tuned threshold, scores every test candidate
pair, and writes BOTH required output files in the exact format
validate_submission.py checks:

  output_<model>/matching_results.tsv   — final matches (thresholded)
  output_<model>/candidate_pairs.tsv    — full blocking candidate set

Guarantees enforced here (matching the official rules):
  - Every test_source1.tsv entity gets exactly one row in both files,
    even if it had zero blocking candidates (empty matched_entity_ids).
  - No duplicate IDs within a list.
  - Only S2-/S3- IDs, never S1- (self-match), in either file.
  - matching_results.tsv is a subset of candidate_pairs.tsv by construction
    (matches are only ever chosen FROM the candidate set).

Usage (run from student_resource/, after features.py --split test and
training the corresponding model):

    python3 predict.py --cache-dir cache --model lightgbm --output-dir output_lightgbm
    python3 predict.py --cache-dir cache --model xgboost --output-dir output_xgboost

Then validate each with utils/validate_submission.py before submitting.

Requires: pandas, pyarrow, joblib, and lightgbm or xgboost matching
whichever --model you pick.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import joblib
import pandas as pd


def load_model_and_threshold(cache_dir, model_name):
    model_path = os.path.join(cache_dir, f"model_{model_name}.pkl")
    thresh_path = os.path.join(cache_dir, f"threshold_{model_name}.json")
    if not os.path.isfile(model_path):
        raise SystemExit(f"No trained model found at {model_path}. Run train_{model_name}.py first.")
    bundle = joblib.load(model_path)
    with open(thresh_path) as f:
        thresh_info = json.load(f)
    return bundle["model"], bundle["feature_cols"], thresh_info["threshold"]


def predict_probabilities(model, model_name, X):
    if model_name == "lightgbm":
        return model.predict(X, num_iteration=model.best_iteration)
    elif model_name == "xgboost":
        import xgboost as xgb
        dmat = xgb.DMatrix(X)
        return model.predict(dmat, iteration_range=(0, model.best_iteration + 1))
    else:
        raise ValueError(model_name)


def write_id_list_tsv(path, header_col2, s1_ids, id_lists):
    """id_lists: {s1_id: iterable of ids}. Every s1_id in s1_ids gets
    exactly one row, in the given order, even if id_lists has nothing for
    it (writes an empty list)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"source1_entity_id\t{header_col2}\n")
        for s1_id in s1_ids:
            ids = sorted(set(id_lists.get(s1_id, [])))  # dedupe + stable order
            f.write(f"{s1_id}\t{','.join(ids)}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate submission outputs from a trained model.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--model", choices=["lightgbm", "xgboost"], required=True)
    parser.add_argument("--output-dir", default=None,
                         help="Default: output_<model>/ (kept separate per model so they never overwrite each other)")
    parser.add_argument("--threshold-override", type=float, default=None,
                         help="Use this threshold instead of the one tuned during training (for manual precision/recall trade-off experiments)")
    args = parser.parse_args()

    output_dir = args.output_dir or f"output_{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"loading model={args.model} ...")
    model, feature_cols, tuned_threshold = load_model_and_threshold(args.cache_dir, args.model)
    threshold = args.threshold_override if args.threshold_override is not None else tuned_threshold
    print(f"  using threshold: {threshold:.3f}" + (" (override)" if args.threshold_override is not None else " (tuned)"))

    print("loading test features + test S1 entity list ...")
    feat_df = pd.read_parquet(os.path.join(args.cache_dir, "test_features.parquet"))
    s1_df = pd.read_parquet(
        os.path.join(args.cache_dir, "test_source1_normalized.parquet"), columns=["entity_id"]
    )
    all_s1_ids = s1_df["entity_id"].tolist()
    print(f"  {len(all_s1_ids):,} required S1 entities, {len(feat_df):,} candidate pairs")

    if len(feat_df) > 0:
        print("scoring candidate pairs ...")
        X = feat_df[feature_cols]
        feat_df = feat_df.copy()
        feat_df["prob"] = predict_probabilities(model, args.model, X)
    else:
        feat_df["prob"] = []

    # candidate_pairs.tsv: every candidate from blocking, regardless of score
    candidate_lists = defaultdict(list)
    for s1_id, cid in zip(feat_df["source1_entity_id"], feat_df["candidate_entity_id"]):
        candidate_lists[s1_id].append(cid)

    # matching_results.tsv: only candidates scoring >= threshold
    matched = feat_df[feat_df["prob"] >= threshold]
    match_lists = defaultdict(list)
    for s1_id, cid in zip(matched["source1_entity_id"], matched["candidate_entity_id"]):
        match_lists[s1_id].append(cid)

    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    matching_path = os.path.join(output_dir, "matching_results.tsv")
    write_id_list_tsv(candidate_path, "candidate_entity_ids", all_s1_ids, candidate_lists)
    write_id_list_tsv(matching_path, "matched_entity_ids", all_s1_ids, match_lists)

    n_matched_entities = sum(1 for s1_id in all_s1_ids if match_lists.get(s1_id))
    n_singleton_predictions = len(all_s1_ids) - n_matched_entities
    print(f"\n-> {candidate_path}")
    print(f"-> {matching_path}")
    print(f"   {n_matched_entities:,}/{len(all_s1_ids):,} S1 entities predicted with >=1 match")
    print(f"   {n_singleton_predictions:,}/{len(all_s1_ids):,} S1 entities predicted as singletons (no match)")
    print(f"\nNext: run utils/validate_submission.py against {output_dir}/ before submitting.")


if __name__ == "__main__":
    sys.exit(main())