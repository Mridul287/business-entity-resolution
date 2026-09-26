"""
Tests for `src.blocking.ann_search`.

The point of this module is a recall number nobody has measured yet, built on
vectors stored in a dtype whose recall cost `embed_runtime` explicitly flags as
unverified. So these tests do not try to simulate a model. They pin the three
things that are checkable without one, and each of them is a place where a
silent error would produce a plausible wrong answer rather than a crash:

  * that `build_index` + `search` reproduce an exactly known ranking, including
    the score, so a later change to the index type or the metric is caught here
    rather than in a recall figure;
  * that the loader reads the real shard layout -- a list column of float16
    vectors under `embedding`, ids under `entity_id` -- because those are the two
    facts this module is built on and neither is visible in the type signature;
  * that `entity_filter` excludes what it says and that `None` excludes nothing,
    since the filter is the only thing standing between a mixed-source shard
    directory and a self-match at similarity 1.0.

The ranking tests use hand-placed unit vectors on a circle, where every pairwise
score is known in advance and can be written down independently of FAISS. If
these are wrong the test is wrong, so the expected scores are computed by hand
from the definition of the inner product rather than by calling FAISS.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest

from src.blocking import embed_runtime as rt
from src.blocking.ann_search import (
    ID_COLUMN,
    TEXT_COLUMN,
    VECTOR_COLUMN,
    build_index,
    load_embedding_shards,
    read_manifest,
    search,
    shard_paths,
)

DIM = 8  # small enough to write the expected scores out by hand


# --------------------------------------------------------------------------
# helpers: vectors and shards, built the way embed_runtime writes them
# --------------------------------------------------------------------------


def _unit(vec: list[float]) -> np.ndarray:
    """A unit vector, because the real ones are unit (normalize_embeddings=True)."""
    arr = np.asarray(vec, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).astype(np.float32)


def _write_shard(path: Path, ids: list[str], texts: list[str], vectors: list[np.ndarray]) -> Path:
    """
    One shard in `embed_runtime.embed_shard`'s exact layout.

    The two details that matter and that a convenient test fixture would skip:
    `embedding` is a *list column of per-row arrays* (so a reader that forgets
    `np.stack` gets an object array), and it is float16 (so a reader that forgets
    the cast hands FAISS a dtype it will not take).
    """
    frame = pd.DataFrame(
        {
            "shard_id": int(path.stem.split("_")[1]),
            ID_COLUMN: ids,
            TEXT_COLUMN: texts,
            VECTOR_COLUMN: [v.astype(rt.STORAGE_DTYPE) for v in vectors],
        }
    )
    frame.to_parquet(path, index=False)
    return path


def _write_manifest(directory: Path, **payload) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / rt.MANIFEST_NAME
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


# A 2-D set of unit vectors with known geometry:
#   A and B nearly identical (high score)      -> A's nearest is B
#   C orthogonal-ish to A                      -> A's farthest
#   D identical to E                           -> a tie, so ordering must be stable
VECTORS = {
    "A": _unit([1, 0, 0, 0, 0, 0, 0, 0]),
    "B": _unit([0.99, 0.1, 0, 0, 0, 0, 0, 0]),
    "C": _unit([0, 1, 0, 0, 0, 0, 0, 0]),
    "D": _unit([0.6, 0.8, 0, 0, 0, 0, 0, 0]),
    "E": _unit([0.6, 0.8, 0, 0, 0, 0, 0, 0]),
    "F": _unit([0, 0, 1, 0, 0, 0, 0, 0]),
}


def _score(left: np.ndarray, right: np.ndarray) -> float:
    """Inner product of two unit vectors, i.e. cosine similarity."""
    return float(np.dot(left, right))


# --------------------------------------------------------------------------
# 1. build_index + search reproduce a known ranking
# --------------------------------------------------------------------------


def test_search_returns_the_exact_expected_ranking() -> None:
    """
    Every expected score below is the inner product of two hand-written unit
    vectors, worked out independently of FAISS. A wrong index type, a wrong
    metric, or a missing normalisation all move these numbers.

    Only the *order of the score groups* is asserted, never the order inside a
    group: D/E are the same vector and C/F are both orthogonal to A, and which
    of two identical scores FAISS emits first is its business, not this
    function's contract.
    """
    candidate_ids = ["CAND-B", "CAND-C", "CAND-D", "CAND-E", "CAND-F"]
    candidate_vecs = np.stack([VECTORS[k] for k in ("B", "C", "D", "E", "F")])
    index = build_index(candidate_vecs)

    out = search(["Q-A"], np.stack([VECTORS["A"]]), candidate_ids, index, top_k=4)

    assert out["candidate_entity_id"].iloc[0] == "CAND-B", "the near-identical vector wins"
    # B alone at ~0.995, then the 0.6 pair, then the orthogonal pair at 0.0.
    assert list(out["candidate_entity_id"])[1:3] and set(
        out["candidate_entity_id"].iloc[1:3]
    ) == {"CAND-D", "CAND-E"}
    assert out["candidate_entity_id"].iloc[3] in {"CAND-C", "CAND-F"}

    expected = {
        "CAND-B": _score(VECTORS["A"], VECTORS["B"]),
        "CAND-D": _score(VECTORS["A"], VECTORS["D"]),
        "CAND-E": _score(VECTORS["A"], VECTORS["E"]),
        "CAND-C": _score(VECTORS["A"], VECTORS["C"]),
        "CAND-F": _score(VECTORS["A"], VECTORS["F"]),
    }
    for row in out.itertuples():
        assert row.embedding_similarity_score == pytest.approx(
            expected[row.candidate_entity_id], abs=1e-6
        ), f"{row.candidate_entity_id} scored wrong"

    # And the scores really are descending, which is the part that is a contract.
    scores = out["embedding_similarity_score"].to_numpy()
    assert scores.tolist() == sorted(scores.tolist(), reverse=True)


def test_scores_are_descending_and_bounded_by_one() -> None:
    """Cosine on unit vectors lives in [-1, 1], and the output must be sorted."""
    ids = [f"CAND-{k}" for k in "BCDEF"]
    index = build_index(np.stack([VECTORS[k] for k in "BCDEF"]))
    out = search(["Q-A"], np.stack([VECTORS["A"]]), ids, index, top_k=5)
    scores = out["embedding_similarity_score"].to_numpy()
    assert scores.tolist() == sorted(scores.tolist(), reverse=True)
    assert scores.max() <= 1.0 + 1e-6
    assert scores.min() >= -1.0 - 1e-6


def test_a_vector_is_its_own_nearest_neighbour_at_one() -> None:
    """
    The property that makes inner product the right metric here, and the reason
    `IndexFlatIP` is not a placeholder: on unit vectors the self-similarity is
    exactly 1.0. If normalisation were ever dropped upstream this fails.
    """
    index = build_index(np.stack([VECTORS["A"], VECTORS["C"]]))
    out = search(["Q-A"], np.stack([VECTORS["A"]]), ["CAND-A", "CAND-C"], index, top_k=2)
    top = out.iloc[0]
    assert top["candidate_entity_id"] == "CAND-A"
    assert top["embedding_similarity_score"] == pytest.approx(1.0, abs=1e-6)


def test_opposite_vectors_score_negative() -> None:
    """
    A floor on the metric, and a check that the ranking is by *descending*
    similarity rather than by absolute value. -A scores -1.0 against A, which is
    the farthest candidate, not the nearest -- so a version that sorted on
    magnitude, or clipped negatives to zero, would return it first and pass a
    test that only looked for the presence of a negative number.
    """
    minus = (-VECTORS["A"]).astype(np.float32)
    index = build_index(np.stack([minus, VECTORS["C"]]))
    out = search(["Q-A"], np.stack([VECTORS["A"]]), ["CAND-MINUS", "CAND-C"], index, top_k=2)

    assert list(out["candidate_entity_id"]) == ["CAND-C", "CAND-MINUS"]
    assert out["embedding_similarity_score"].iloc[1] == pytest.approx(-1.0, abs=1e-6)
    assert out["embedding_similarity_score"].min() < 0.0


def test_ties_keep_a_stable_order() -> None:
    """
    D and E are the same vector. Their relative order is arbitrary to FAISS, so
    this does not assert *which* comes first -- it asserts the result is the same
    on two runs, because a search whose output order varies between identical
    inputs cannot be diffed against a previous run.
    """
    ids = ["CAND-D", "CAND-E", "CAND-C"]
    vecs = np.stack([VECTORS["D"], VECTORS["E"], VECTORS["C"]])
    first = search(["Q-A"], np.stack([VECTORS["A"]]), ids, build_index(vecs), top_k=3)
    second = search(["Q-A"], np.stack([VECTORS["A"]]), ids, build_index(vecs), top_k=3)
    assert list(first["candidate_entity_id"]) == list(second["candidate_entity_id"])


def test_top_k_is_respected_per_query() -> None:
    ids = [f"CAND-{k}" for k in "BCDEF"]
    index = build_index(np.stack([VECTORS[k] for k in "BCDEF"]))
    queries = np.stack([VECTORS["A"], VECTORS["C"]])
    for k in (1, 2, 3):
        out = search(["Q-A", "Q-C"], queries, ids, index, top_k=k)
        assert (out.groupby("source1_entity_id").size() == k).all()


def test_query_batching_does_not_change_the_answer() -> None:
    """
    `query_batch` exists to cap peak memory, so it must be invisible in the
    result. One batch against many, compared row for row.
    """
    ids = [f"CAND-{k}" for k in "BCDEF"]
    index = build_index(np.stack([VECTORS[k] for k in "BCDEF"]))
    queries = np.stack([VECTORS["A"], VECTORS["C"], VECTORS["F"]])
    qids = ["Q-A", "Q-C", "Q-F"]
    single = search(qids, queries, ids, index, top_k=3, query_batch=4096)
    split = search(qids, queries, ids, index, top_k=3, query_batch=1)
    pd.testing.assert_frame_equal(single, split)


def test_more_queries_than_one_batch() -> None:
    """The batching loop's boundary, where an off-by-one drops the last query."""
    ids = [f"CAND-{k}" for k in "BCDEF"]
    index = build_index(np.stack([VECTORS[k] for k in "BCDEF"]))
    queries = np.stack([VECTORS[k] for k in "ABCDEF"])
    qids = [f"Q-{k}" for k in "ABCDEF"]
    out = search(qids, queries, ids, index, top_k=2, query_batch=2)
    assert set(out["source1_entity_id"]) == set(qids)


# --------------------------------------------------------------------------
# 2. the loader reads the real shard layout
# --------------------------------------------------------------------------


@pytest.fixture
def two_shard_dir(tmp_path: Path) -> Path:
    """
    Two shards in the exact layout `embed_runtime.embed_shard` writes, plus the
    manifest `write_manifest` writes. Built here rather than read from Drive, so
    the test needs no GPU, no network and no real vectors.
    """
    directory = tmp_path / "out_broad"
    directory.mkdir()
    _write_shard(
        directory / "shard_00000.parquet",
        ["S2-1", "S2-2", "S3-1"],
        ["Prime Money", "Prime Money Lenders", "Golden Gate"],
        [VECTORS["A"], VECTORS["B"], VECTORS["C"]],
    )
    _write_shard(
        directory / "shard_00001.parquet",
        ["S3-2", "S2-3"],
        ["Nandlal Kisan", "Apex Summit"],
        [VECTORS["D"], VECTORS["E"]],
    )
    _write_manifest(
        directory,
        identity={"model": "LaBSE", "strict": False, "storage_dtype": str(rt.STORAGE_DTYPE)},
    )
    return directory


def test_loader_reads_ids_and_vectors_across_shards(two_shard_dir: Path) -> None:
    ids, vectors = load_embedding_shards(two_shard_dir / rt.MANIFEST_NAME, verify=False)
    assert list(ids) == ["S2-1", "S2-2", "S3-1", "S3-2", "S2-3"]
    assert vectors.shape == (5, DIM)
    np.testing.assert_allclose(vectors[0], VECTORS["A"], atol=1e-3)
    np.testing.assert_allclose(vectors[4], VECTORS["E"], atol=1e-3)


def test_loader_casts_float16_storage_to_float32(two_shard_dir: Path) -> None:
    """
    Not a style preference: `IndexFlatIP` will not take float16, so a loader
    that passed the storage dtype through would fail at index time with an error
    that says nothing about the real cause.
    """
    _, vectors = load_embedding_shards(two_shard_dir / rt.MANIFEST_NAME, verify=False)
    assert vectors.dtype == np.float32
    # and the values survived the widening
    assert np.isfinite(vectors).all()


def test_loader_stacks_the_list_column(two_shard_dir: Path) -> None:
    """
    `embedding` is a list column of per-row arrays. A reader that skipped
    `np.stack` would produce a 1-D object array, and `build_index` would then
    reject it -- so this asserts the shape directly.
    """
    raw = pd.read_parquet(two_shard_dir / "shard_00000.parquet")
    assert isinstance(raw[VECTOR_COLUMN].iloc[0], (np.ndarray, list, np.floating))
    _, vectors = load_embedding_shards(two_shard_dir / rt.MANIFEST_NAME, verify=False)
    assert vectors.ndim == 2


def test_manifest_is_read_for_the_storage_dtype(two_shard_dir: Path) -> None:
    """
    The manifest records the run's storage dtype, and `verify_shards` needs the
    rest of `identity`, so the loader cannot ignore the manifest entirely.

    The asserted value is `str(np.float16)`, which is the numpy repr
    `"<class 'numpy.float16'>"` and not the string `"float16"`. That is what the
    notebook writes, and it is asserted here rather than smoothed over so that a
    future reader does not mistake the manifest for something it is not.
    """
    manifest = read_manifest(two_shard_dir / rt.MANIFEST_NAME)
    assert manifest["identity"]["storage_dtype"] == str(rt.STORAGE_DTYPE)
    assert manifest["identity"]["storage_dtype"] == "<class 'numpy.float16'>"
    assert rt.STORAGE_DTYPE == np.float16, "the shards really are float16 on disk"


def test_shard_paths_are_ordered_and_complete(two_shard_dir: Path) -> None:
    paths = shard_paths(two_shard_dir / rt.MANIFEST_NAME)
    assert [p.name for p in paths] == ["shard_00000.parquet", "shard_00001.parquet"]


def test_shard_globbing_ignores_sidecar_metadata(two_shard_dir: Path) -> None:
    """
    `run_shards` writes a `<shard>.meta.json` beside every shard. A `*.parquet`
    glob is right and `*` would be wrong, so this pins the difference.
    """
    (two_shard_dir / "shard_00000.meta.json").write_text("{}", encoding="utf-8")
    paths = shard_paths(two_shard_dir / rt.MANIFEST_NAME)
    assert all(p.suffix == ".parquet" for p in paths)
    assert len(paths) == 2


def test_loader_raises_when_there_are_no_shards(tmp_path: Path) -> None:
    directory = tmp_path / "empty"
    _write_manifest(directory)
    with pytest.raises(FileNotFoundError, match="no shard_"):
        load_embedding_shards(directory / rt.MANIFEST_NAME, verify=False)


def test_loader_rejects_a_shard_missing_the_vector_column(tmp_path: Path) -> None:
    """
    The local `embed_names.py` writes a *single* parquet with different columns.
    Pointing this loader at one should say so in the message rather than fail
    later with a shape error.
    """
    directory = tmp_path / "wrong"
    directory.mkdir()
    pd.DataFrame({ID_COLUMN: ["S2-1"], "text": ["Prime Money"]}).to_parquet(
        directory / "shard_00000.parquet", index=False
    )
    _write_manifest(directory)
    with pytest.raises(KeyError, match="embedding"):
        load_embedding_shards(directory / rt.MANIFEST_NAME, verify=False)


def test_loader_rejects_shards_of_different_widths(tmp_path: Path) -> None:
    """Two models' vectors in one directory, which no index can hold."""
    directory = tmp_path / "mixed"
    directory.mkdir()
    _write_shard(directory / "shard_00000.parquet", ["S2-1"], ["A"], [VECTORS["A"]])
    _write_shard(
        directory / "shard_00001.parquet", ["S2-2"], ["B"], [np.ones(4, dtype=np.float32)]
    )
    _write_manifest(directory)
    with pytest.raises(ValueError, match="dimensional"):
        load_embedding_shards(directory / rt.MANIFEST_NAME, verify=False)


# --------------------------------------------------------------------------
# 3. entity_filter
# --------------------------------------------------------------------------


def test_filter_none_keeps_every_row(two_shard_dir: Path) -> None:
    ids, _ = load_embedding_shards(two_shard_dir / rt.MANIFEST_NAME, None, verify=False)
    assert len(ids) == 5


def test_filter_excludes_rows_it_rejects(two_shard_dir: Path) -> None:
    """The default, `None`, keeps all 5; a filter that rejects S2 keeps 2."""
    ids, vectors = load_embedding_shards(
        two_shard_dir / rt.MANIFEST_NAME,
        lambda row: str(getattr(row, ID_COLUMN)).startswith("S3"),
        verify=False,
    )
    assert list(ids) == ["S3-1", "S3-2"]
    assert vectors.shape[0] == 2


def test_filter_can_select_on_the_text_column(two_shard_dir: Path) -> None:
    """The filter sees `(entity_id, business_name_clean)`, so it can read either."""
    ids, _ = load_embedding_shards(
        two_shard_dir / rt.MANIFEST_NAME,
        lambda row: "prime" in str(getattr(row, TEXT_COLUMN)).lower(),
        verify=False,
    )
    assert list(ids) == ["S2-1", "S2-2"]


def test_filter_that_rejects_everything_returns_empty_not_an_error(
    two_shard_dir: Path,
) -> None:
    """
    The case a real script filter hits: a shard with no matching rows. `np.stack`
    on an empty list raises, so this is the boundary that decides whether a
    legitimate filter is usable.
    """
    ids, vectors = load_embedding_shards(
        two_shard_dir / rt.MANIFEST_NAME, lambda row: False, verify=False
    )
    assert len(ids) == 0
    assert vectors.shape[0] == 0
    assert vectors.dtype == np.float32


def test_filter_is_applied_before_stacking(two_shard_dir: Path) -> None:
    """
    Filtering after stacking would allocate the full matrix and throw it away.
    The observable proof is that a filter rejecting a shard still yields the
    right width for the rows that survive -- a post-filter slice of a wrongly
    shaped stack would not.
    """
    _, vectors = load_embedding_shards(
        two_shard_dir / rt.MANIFEST_NAME,
        lambda row: str(getattr(row, ID_COLUMN)) == "S3-2",
        verify=False,
    )
    assert vectors.shape == (1, DIM)


def test_filter_receives_a_row_without_the_vector(two_shard_dir: Path) -> None:
    """
    The contract that keeps the filter linear-time. A row carrying a 1024-float
    array through `DataFrame.apply(axis=1)` is the difference between a scan and
    a stall over millions of rows.
    """
    seen: list[object] = []

    def spy(row: object) -> bool:
        seen.append(row)
        return True

    load_embedding_shards(two_shard_dir / rt.MANIFEST_NAME, spy, verify=False)
    assert len(seen) == 5
    assert set(dir(seen[0])) >= {ID_COLUMN, TEXT_COLUMN}
    assert VECTOR_COLUMN not in dir(seen[0])


# --------------------------------------------------------------------------
# 4. build_index guards
# --------------------------------------------------------------------------


def test_build_index_rejects_an_empty_matrix() -> None:
    with pytest.raises(ValueError, match="zero vectors"):
        build_index(np.empty((0, DIM), dtype=np.float32))


def test_build_index_rejects_a_one_dimensional_array() -> None:
    with pytest.raises(ValueError, match="2-D"):
        build_index(np.ones(DIM, dtype=np.float32))


def test_build_index_accepts_float64_and_casts() -> None:
    """Forgiving on input dtype, strict about the index dtype."""
    index = build_index(np.stack([VECTORS["A"], VECTORS["B"]]).astype(np.float64))
    assert isinstance(index, faiss.IndexFlatIP)
    assert index.d == DIM


def test_build_index_reports_the_vector_width(two_shard_dir: Path) -> None:
    _, vectors = load_embedding_shards(two_shard_dir / rt.MANIFEST_NAME, verify=False)
    assert build_index(vectors).d == vectors.shape[1]


# --------------------------------------------------------------------------
# 5. search guards
# --------------------------------------------------------------------------


def test_search_rejects_mismatched_ids_and_vectors() -> None:
    index = build_index(np.stack([VECTORS["A"]]))
    with pytest.raises(ValueError, match="ids but"):
        search(["Q-A", "Q-B"], np.stack([VECTORS["A"]]), ["CAND-A"], index, top_k=1)


def test_search_rejects_an_index_that_does_not_match_the_ids() -> None:
    """
    The failure this catches is silent and total: if the index and the id array
    are offset from each other, every score is real but attached to the wrong
    business, and no recall number would look wrong.
    """
    index = build_index(np.stack([VECTORS["A"], VECTORS["B"]]))
    with pytest.raises(ValueError, match="candidate ids"):
        search(["Q-A"], np.stack([VECTORS["A"]]), ["CAND-A"], index, top_k=1)


def test_search_rejects_a_non_positive_top_k() -> None:
    index = build_index(np.stack([VECTORS["A"]]))
    with pytest.raises(ValueError, match="top_k"):
        search(["Q-A"], np.stack([VECTORS["A"]]), ["CAND-A"], index, top_k=0)


def test_search_with_no_candidates_returns_the_empty_frame() -> None:
    index = build_index(np.stack([VECTORS["A"]]))
    out = search(["Q-A"], np.stack([VECTORS["A"]]), [], index, top_k=5)
    assert out.empty
    assert list(out.columns) == [
        "source1_entity_id",
        "candidate_entity_id",
        "embedding_similarity_score",
    ]


def test_search_drops_a_self_match(two_shard_dir: Path) -> None:
    """
    The one behaviour here that is not in the original sketch and is not
    optional. The embed job writes all three sources into one `OUT_DIR` under a
    single increasing shard_id, so pointing both manifests at it puts every
    source1 vector in its own index -- and each row's nearest neighbour is
    itself, at 1.0, spending the best of its `top_k` slots on a guaranteed
    useless answer.

    `top_k=1` is what makes this a real test. The self-match scores 1.0 and the
    best genuine neighbour scores less, so if the drop did not happen the single
    returned row would be the query pointing at itself.
    """
    ids, vectors = load_embedding_shards(two_shard_dir / rt.MANIFEST_NAME, verify=False)
    index = build_index(vectors)
    out = search(list(ids), vectors, list(ids), index, top_k=1)

    assert len(out) == len(ids), "every query still gets its one real neighbour"
    assert (out["source1_entity_id"] != out["candidate_entity_id"]).all(), (
        "a row was returned as its own candidate"
    )
    assert (out["embedding_similarity_score"] < 1.0).all(), (
        "a self-match at similarity 1.0 survived"
    )


def test_a_self_match_does_not_consume_a_top_k_slot(two_shard_dir: Path) -> None:
    """
    `top_k` is a budget, so dropping the self-match must leave the slot filled by
    a real neighbour rather than a hole. This is why the search asks FAISS for
    `top_k + 1`; without the extra ask this returns 4 rows per query instead of 5.
    """
    ids, vectors = load_embedding_shards(two_shard_dir / rt.MANIFEST_NAME, verify=False)
    out = search(list(ids), vectors, list(ids), index=build_index(vectors), top_k=4)

    # 5 ids, 4 others each, self dropped: exactly 4 per query.
    assert len(out) == 20
    assert (out.groupby("source1_entity_id").size() == 4).all()
    assert (out["source1_entity_id"] != out["candidate_entity_id"]).all()


def test_search_asks_for_no_more_than_the_index_holds() -> None:
    """
    A 2-vector index with top_k=20: FAISS fills the surplus with -1, and a
    reader that trusted the width would emit rows for candidates that do not
    exist.
    """
    index = build_index(np.stack([VECTORS["A"], VECTORS["B"]]))
    out = search(["Q-C"], np.stack([VECTORS["C"]]), ["CAND-A", "CAND-B"], index, top_k=20)
    assert len(out) == 2
    assert out["candidate_entity_id"].notna().all()
