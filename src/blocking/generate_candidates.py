"""
Owner: Person A

Candidate generation / blocking. This sets the recall ceiling for the whole
pipeline -- track recall against train_ground_truth on your validation split
obsessively while building this.

Suggested strategies to union:
- TF-IDF cosine similarity (name + address) with approximate nearest neighbors
- character n-gram MinHash/LSH (datasketch) for typo/transliteration tolerance
- sorted-neighborhood blocking on normalized name prefix
- optional: small MIT/Apache embedding model (e.g. all-MiniLM-L6-v2) + ANN

TODO: implement and union candidate sets, dedupe per (source1_entity_id, candidate_entity_id)
"""
import pandas as pd


def generate_candidates(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a long-format DataFrame: one row per (source1_entity_id, candidate_entity_id).
    candidate_entity_id values are S2-/S3- entity_ids only.
    """
    # TODO: implement real blocking. Placeholder returns no candidates.
    return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
