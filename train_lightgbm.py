#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — LightGBM Training Branch

Reads cache/train_features.parquet (shared output of features.py), splits
by S1 ENTITY (never by row — see common.py), trains a LightGBM binary
classifier on MATCH/NOT-MATCH, then sweeps the decision threshold on the
held-out validation set to maximize macro F_0.5 (NOT accuracy/F1 — the
challenge is scored on F_0.5, which is precision-heavy).

Saves:
  cache/model_lightgbm.pkl        — the trained model
  cache/threshold_lightgbm.json   — the tuned decision threshold + val score

Usage (run from student_resource/, after features.py --split train):

    python3 train_lightgbm.py --cache-dir cache

Requires: pandas, pyarrow, lightgbm, scikit-learn, joblib
"""

import argparse
import json
import os
import sys

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import split_s1_entities, sweep_threshold

FEATURE_COLS = [
    "name_token_jaccard", "addr_token_jaccard",
    "name_fuzz_ratio", "name_fuzz_token_sort_ratio",
    "addr_fuzz_ratio", "addr_fuzz_token_sort_ratio",
    "name_chargram_tfidf_cosine", "addr_chargram_tfidf_cosine",
    "name_len_diff", "addr_len_diff",
    "addr_missing_s1", "addr_missing_cand",
]


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM matcher.")
    parser.add_argument("--cache-dir", default="cache")
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42,
                         help="MUST match the XGBoost branch's seed for comparable validation scores.")
    parser.add_argument("--num-boost-round", type=int, default=300)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu",
                         help="'gpu' uses LightGBM's OpenCL backend, 'cuda' uses its newer CUDA backend "
                              "(LightGBM >=4.0). BOTH require a LightGBM build with that support compiled "
                              "in — the plain `pip install lightgbm` wheel is CPU-only on Windows. See the "
                              "printed message if this fails. Falls back to CPU automatically if the GPU "
                              "device can't be initialized.")
    args = parser.parse_args()

    df = pd.read_parquet(os.path.join(args.cache_dir, "train_features.parquet"))
    print(f"loaded {len(df):,} training candidate pairs")

    s1_ids = df["source1_entity_id"].unique()
    train_ids, val_ids = split_s1_entities(s1_ids, val_frac=args.val_frac, seed=args.seed)
    print(f"entity split: {len(train_ids):,} train S1 entities, {len(val_ids):,} val S1 entities")

    train_mask = df["source1_entity_id"].isin(train_ids)
    val_mask = df["source1_entity_id"].isin(val_ids)

    X_train, y_train = df.loc[train_mask, FEATURE_COLS], df.loc[train_mask, "label"]
    X_val, y_val = df.loc[val_mask, FEATURE_COLS], df.loc[val_mask, "label"]
    print(f"train pairs: {len(X_train):,} ({y_train.sum():,} positive) | "
          f"val pairs: {len(X_val):,} ({y_val.sum():,} positive)")

    train_set = lgb.Dataset(X_train, label=y_train)
    val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": args.seed,
    }
    if args.device != "cpu":
        params["device_type"] = args.device

    print(f"training LightGBM (device={args.device}) ...")
    try:
        model = lgb.train(
            params,
            train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False), lgb.log_evaluation(50)],
        )
    except lgb.basic.LightGBMError as e:
        if args.device == "cpu":
            raise
        print(f"\n  GPU training failed ({e}).")
        print("  Your installed lightgbm package is very likely the standard CPU-only pip wheel —")
        print("  GPU support needs a build compiled with it. Options:")
        print(f"    device_type=gpu (OpenCL):  pip uninstall lightgbm && "
              f"pip install lightgbm --config-settings=cmake.define.USE_GPU=ON")
        print(f"    device_type=cuda:          pip uninstall lightgbm && "
              f"pip install lightgbm --config-settings=cmake.define.USE_CUDA=ON")
        print("  (both need the matching OpenCL/CUDA toolkit installed and a working C++ build chain —")
        print("   non-trivial on Windows. Falling back to CPU for this run so you're not blocked.)")
        params["device_type"] = "cpu"
        model = lgb.train(
            params,
            train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[val_set],
            callbacks=[lgb.early_stopping(args.early_stopping_rounds, verbose=False), lgb.log_evaluation(50)],
        )

    val_prob = model.predict(X_val, num_iteration=model.best_iteration)
    if len(np.unique(y_val)) > 1:
        auc = roc_auc_score(y_val, val_prob)
        print(f"validation AUC: {auc:.4f}")

    print("sweeping decision threshold for macro F_0.5 ...")
    best_thresh, best_score, _ = sweep_threshold(
        y_val.to_numpy(), val_prob, df.loc[val_mask, "source1_entity_id"].to_numpy(), beta=0.5
    )
    print(f"best threshold: {best_thresh:.3f}  ->  validation macro F_0.5: {best_score:.4f}")
    print("NOTE: this F_0.5 is computed only over S1 entities that HAD at least one")
    print("candidate pair in the validation set. It does not include the recall lost")
    print("to blocking misses (S1 entities with zero candidates) — check blocking.py's")
    print("--evaluate recall report for that separately, and factor it in when judging")
    print("your real expected leaderboard score.")

    model_path = os.path.join(args.cache_dir, "model_lightgbm.pkl")
    thresh_path = os.path.join(args.cache_dir, "threshold_lightgbm.json")
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS}, model_path)
    with open(thresh_path, "w") as f:
        json.dump({"threshold": best_thresh, "val_macro_f0.5": best_score}, f, indent=2)

    print(f"saved model -> {model_path}")
    print(f"saved threshold -> {thresh_path}")

    print("\ntop feature importances:")
    importance = sorted(zip(FEATURE_COLS, model.feature_importance(importance_type="gain")),
                         key=lambda x: -x[1])
    for name, score in importance:
        print(f"  {name:32s} {score:.1f}")


if __name__ == "__main__":
    sys.exit(main())