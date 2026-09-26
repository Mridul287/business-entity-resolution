"""
Owner: Person C

Exact macro-averaged F0.5 scorer per the problem statement, train/val split
utility (by S1 entity ID, no leakage), and threshold tuning.

F0.5 = (1.25 * P * R) / (0.25 * P + R), computed per S1 entity then averaged.
A singleton (no true matches) scores 1.0 if predicted empty, 0.0 if any
match is predicted for it.

TODO: implement
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def split_train_val(s1_df: pd.DataFrame, gt_df: pd.DataFrame, val_size: float = 0.2, random_state: int = 42):
    """Split by source1_entity_id to avoid leakage. Returns s1_train, s1_val, gt_train, gt_val."""
    train_ids, val_ids = train_test_split(s1_df["entity_id"], test_size=val_size, random_state=random_state)
    s1_train = s1_df[s1_df["entity_id"].isin(train_ids)].reset_index(drop=True)
    s1_val = s1_df[s1_df["entity_id"].isin(val_ids)].reset_index(drop=True)
    gt_train = gt_df[gt_df["source1_entity_id"].isin(train_ids)].reset_index(drop=True)
    gt_val = gt_df[gt_df["source1_entity_id"].isin(val_ids)].reset_index(drop=True)
    return s1_train, s1_val, gt_train, gt_val


def score_f_half(scored_df: pd.DataFrame, gt_df: pd.DataFrame, threshold: float) -> float:
    """
    scored_df: (source1_entity_id, candidate_entity_id, match_probability) rows.
    gt_df: (source1_entity_id, matched_entity_ids) ground truth, comma-joined.
    Returns macro-averaged F0.5 across all S1 entities in gt_df.
    """
    # TODO: implement exact formula from the PS
    return 0.0


def tune_threshold(scored_df: pd.DataFrame, gt_df: pd.DataFrame, steps: int = 50) -> float:
    """Sweep thresholds, return the one maximizing score_f_half."""
    # TODO: implement sweep
    return 0.5
