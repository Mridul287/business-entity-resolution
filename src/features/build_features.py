"""
Owner: Person B

Turn each (source1_entity_id, candidate_entity_id) pair into a feature vector.

Suggested features (see PS "Tips for Success"):
- name: Levenshtein ratio, Jaro-Winkler, token Jaccard, TF-IDF cosine,
  abbreviation-normalized exact match, embedding cosine sim
- address: token overlap, street-number match, city/pin/zip match,
  length diff, embedding cosine sim
- meta: country exact match (works fine on unseen labels like France --
  it's just equality), name length diff, source pair indicator (S1-S2 vs S1-S3)

TODO: implement
"""
import pandas as pd


def build_features(candidates_df: pd.DataFrame, s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: candidates_df from src.blocking.generate_candidates, plus the
    normalized source DataFrames (for looking up name/address by entity_id).
    Output: candidates_df with additional numeric feature columns.
    """
    out = candidates_df.copy()
    # TODO: implement feature computation
    return out
