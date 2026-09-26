"""
Fixtures for the blocking (candidate generation) tests.

Synthetic and tiny, for the same reason as tests/conftest.py: the real sources
are hundreds of megabytes and are not committed.

The three-row S1/S2/S3 setup below is a miniature of the real task. The labels
are hand-written and are what a recall test should be measured against:

    S1-00001 "Prime Money Lenders"      <-> S2-00001 "Prime Money Ltd"   (match)
                                         <-> S3-00001 "PRIME MONEY"      (match)
                                         <-> S3-00003 "Prime Hardware"   (no match)
    S1-00002 "श्री बालाजी ट्रेडर्स"        <-> S3-00002 "Shree Balaji"    (match,
                                               transliteration only)
    S1-00003 "NULL"                      <-> nothing (singleton, and the
                                               name carries no signal at all)

Once `generate_candidates` is implemented, a blocking recall test is a two-line
assertion against these labels: every match listed above must appear as a
candidate. Recall of 2/2 here and high recall on the real validation split is
the bar.
"""
from __future__ import annotations

import pandas as pd
import pytest

S1_ROWS = [
    {"entity_id": "S1-00001", "business_name": "Prime Money Lenders",
     "business_address": "12 MG Road, Bengaluru", "country": "India"},
    {"entity_id": "S1-00002", "business_name": "श्री बालाजी ट्रेडर्स",
     "business_address": "Station Road, Ahmedabad", "country": "India"},
    {"entity_id": "S1-00003", "business_name": "NULL",
     "business_address": "45 Somewhere", "country": "US"},
]
S2_ROWS = [
    {"entity_id": "S2-00001", "business_name": "Prime Money Ltd",
     "business_address": "12 M.G. Road, Bangalore", "country": "India"},
    {"entity_id": "S2-00002", "business_name": "Prime Hardware",
     "business_address": "9 Ring Road, Bengaluru", "country": "India"},
]
S3_ROWS = [
    {"entity_id": "S3-00001", "business_name": "PRIME MONEY",
     "business_address": "12 MG RD BANGALORE", "country": "India"},
    {"entity_id": "S3-00002", "business_name": "Shree Balaji Traders",
     "business_address": "Station Rd, Ahmedabad", "country": "India"},
    {"entity_id": "S3-00003", "business_name": "Prime Hardware Co",
     "business_address": "9 Ring Road, Bengaluru", "country": "India"},
]

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# Hand-written labels, as (source1_entity_id, candidate_entity_id) pairs.
KNOWN_MATCH_PAIRS = [
    ("S1-00001", "S2-00001"),
    ("S1-00001", "S3-00001"),
    ("S1-00002", "S3-00002"),
]
KNOWN_NON_MATCH_PAIRS = [
    ("S1-00001", "S2-00002"),
    ("S1-00001", "S3-00003"),
]


def _frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).astype(str)


@pytest.fixture
def tiny_s1_df() -> pd.DataFrame:
    return _frame(S1_ROWS)


@pytest.fixture
def tiny_s2_df() -> pd.DataFrame:
    return _frame(S2_ROWS)


@pytest.fixture
def tiny_s3_df() -> pd.DataFrame:
    return _frame(S3_ROWS)


@pytest.fixture
def tiny_source_frames(tiny_s1_df: pd.DataFrame, tiny_s2_df: pd.DataFrame, tiny_s3_df: pd.DataFrame):
    """(s1, s2, s3) in the argument order `generate_candidates` expects."""
    return tiny_s1_df, tiny_s2_df, tiny_s3_df


@pytest.fixture
def tiny_labeled_pairs() -> pd.DataFrame:
    """The hand-written match labels in the long candidate format."""
    return pd.DataFrame(KNOWN_MATCH_PAIRS, columns=["source1_entity_id", "candidate_entity_id"])
