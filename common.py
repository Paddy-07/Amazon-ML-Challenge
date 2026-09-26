#!/usr/bin/env python3
"""
Amazon ML Challenge 2026 — Shared Common Utilities

SHARED by both the LightGBM and XGBoost branches. Do not fork this file —
the F_0.5 scoring and the train/validation split must be byte-identical
across both branches, or your two models' validation scores aren't
comparable and you can't fairly pick a winner or ensemble them.
"""

import random


def f_beta_score(true_ids, pred_ids, beta=0.5):
    """F_beta for one S1 entity, matching the official challenge formula
    exactly, including the singleton special case:
      - true empty, pred empty  -> 1.0 (correctly predicted no match)
      - true empty, pred non-empty -> 0.0 (false merge on a singleton)
      - true non-empty, pred empty -> 0.0
    """
    true_ids = set(true_ids)
    pred_ids = set(pred_ids)

    if not true_ids:
        return 1.0 if not pred_ids else 0.0
    if not pred_ids:
        return 0.0

    tp = len(true_ids & pred_ids)
    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    if precision == 0 and recall == 0:
        return 0.0

    beta2 = beta ** 2
    denom = beta2 * precision + recall
    if denom == 0:
        return 0.0
    return (1 + beta2) * precision * recall / denom


def macro_f_beta(gt_dict, pred_dict, beta=0.5):
    """gt_dict / pred_dict: {source1_entity_id: set_or_list_of_matched_ids}.
    Every key in gt_dict must be scored, even if missing from pred_dict
    (treated as an empty prediction)."""
    if not gt_dict:
        return 0.0
    scores = [
        f_beta_score(true_ids, pred_dict.get(s1_id, set()), beta=beta)
        for s1_id, true_ids in gt_dict.items()
    ]
    return sum(scores) / len(scores)


def split_s1_entities(s1_ids, val_frac=0.15, seed=42):
    """Deterministic entity-level split. MUST be called with the same
    s1_ids, val_frac, and seed on both branches, or your validation sets
    differ and your F_0.5 numbers aren't comparable. Splitting at the
    entity level (not row/pair level) is required — a candidate pair from
    the same S1 entity must never span train and validation, or you leak
    label information across the split.
    """
    ids = sorted(s1_ids)  # sort first so shuffle order is deterministic
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids


def sweep_threshold(y_true, y_prob, groups, beta=0.5, n_steps=99):
    """Find the probability threshold that maximizes macro F_0.5 on a
    validation set, by sweeping thresholds and recomputing per-entity
    scores at each one.

    y_true: 0/1 array-like, one per candidate pair (1 = true match)
    y_prob: model probability per candidate pair
    groups: source1_entity_id per candidate pair (same length as y_true)

    Returns (best_threshold, best_score, all_gt_dict) where all_gt_dict
    is the {s1_id: set(true matched candidate row-indices)} used, mainly
    for debugging.
    """
    import numpy as np

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    groups = np.asarray(groups)

    # Build ground-truth dict from the validation candidate pairs
    # themselves: an S1 entity's "true matches" here are the candidate
    # rows where y_true == 1. Uses row position as the "id" since we only
    # need internal consistency for the sweep, not real entity IDs.
    gt_dict = {}
    for g in np.unique(groups):
        mask = groups == g
        true_positions = frozenset(np.where(mask)[0][y_true[mask] == 1])
        gt_dict[g] = true_positions

    best_thresh, best_score = 0.5, -1.0
    for t in np.linspace(0.01, 0.99, n_steps):
        pred_dict = {}
        above = y_prob >= t
        for g in np.unique(groups):
            mask = groups == g
            pred_positions = frozenset(np.where(mask)[0][above[mask]])
            pred_dict[g] = pred_positions
        score = macro_f_beta(gt_dict, pred_dict, beta=beta)
        if score > best_score:
            best_score, best_thresh = score, float(t)

    return best_thresh, best_score, gt_dict