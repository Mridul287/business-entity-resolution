"""
Interface tests for src/blocking/generate_candidates.py.

`generate_candidates` is still a stub (see the TODO in that module), so these
tests deliberately assert only the *contract* Person B and Person C code depends
on, not any particular blocking strategy. They pass against the stub today and
keep passing once real blocking lands, and they fail loudly if the return format
drifts -- which is the thing that actually blocks the rest of the team.

The recall test that needs real blocking is the next step, not this one: the
hand-written labels in conftest.py (KNOWN_MATCH_PAIRS) give it a target of 3/3.
"""
from __future__ import annotations

import pandas as pd

from src.blocking.generate_candidates import generate_candidates

REQUIRED_COLUMNS = ["source1_entity_id", "candidate_entity_id"]


def test_returns_the_agreed_long_format(tiny_source_frames):
    s1_df, s2_df, s3_df = tiny_source_frames
    candidates = generate_candidates(s1_df, s2_df, s3_df)

    assert isinstance(candidates, pd.DataFrame)
    assert list(candidates.columns) == REQUIRED_COLUMNS
    assert candidates.shape[1] == 2, "one row per (source1_entity_id, candidate_entity_id), no joined lists"


def test_every_s1_entity_id_is_a_real_s1_id(tiny_source_frames):
    s1_df, s2_df, s3_df = tiny_source_frames
    candidates = generate_candidates(s1_df, s2_df, s3_df)

    assert set(candidates["source1_entity_id"]) <= set(s1_df["entity_id"])


def test_candidate_ids_come_from_s2_or_s3_only(tiny_source_frames):
    s1_df, s2_df, s3_df = tiny_source_frames
    candidates = generate_candidates(s1_df, s2_df, s3_df)

    known_ids = set(s2_df["entity_id"]) | set(s3_df["entity_id"])
    assert set(candidates["candidate_entity_id"]) <= known_ids
    assert not any(cid.startswith("S1-") for cid in candidates["candidate_entity_id"])


def test_no_duplicate_pairs(tiny_source_frames):
    s1_df, s2_df, s3_df = tiny_source_frames
    candidates = generate_candidates(s1_df, s2_df, s3_df)

    assert not candidates.duplicated(subset=REQUIRED_COLUMNS).any()


def test_hand_written_labels_are_usable_as_a_recall_target(tiny_source_frames, tiny_labeled_pairs):
    """
    Sanity check on the fixture itself: the labels are well-formed pairs over
    the tiny sources, so a recall test can be written against them directly.
    """
    s1_df, s2_df, s3_df = tiny_source_frames
    assert set(tiny_labeled_pairs["source1_entity_id"]) <= set(s1_df["entity_id"])
    known_ids = set(s2_df["entity_id"]) | set(s3_df["entity_id"])
    assert set(tiny_labeled_pairs["candidate_entity_id"]) <= known_ids
    assert len(tiny_labeled_pairs) == 3
