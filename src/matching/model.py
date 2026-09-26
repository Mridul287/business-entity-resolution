"""
Owner: Person B

Train/infer the pairwise matching classifier.

Recommended: LightGBM/XGBoost binary classifier on the engineered features.
Positives = pairs present in train_ground_truth.tsv; negatives = blocked-but-
not-a-true-match pairs (hard negatives -- important for precision, which is
weighted 2x by F0.5).

Constraint: final model must be MIT/Apache 2.0 licensed, <= 8B parameters
(a GBDT trivially satisfies this; if using an embedding model anywhere in the
pipeline, confirm its license too).

TODO: implement
"""
import pandas as pd


def train_model(features_df: pd.DataFrame, ground_truth_df: pd.DataFrame):
    """
    Labels features_df rows using ground_truth_df (source1_entity_id ->
    matched_entity_ids), trains a classifier, returns the fitted model object.
    """
    # TODO: implement
    return None


def score_pairs(model, features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns features_df with an added `match_probability` column.
    """
    out = features_df.copy()
    # TODO: implement -- placeholder score of 0 so pipeline runs end-to-end
    out["match_probability"] = 0.0
    return out
