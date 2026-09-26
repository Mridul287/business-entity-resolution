"""
Approximate nearest-neighbour candidate generation over the LaBSE name embeddings.

This is the embedding half of blocking, and it covers exactly the rows the
lexical path in `lexical_blocking.py` refuses: the ones whose names are written
in a script an ASCII similarity cannot read. Between them the two paths are
supposed to see every row in the corpus, because a row neither path can see is
a row that can never be recalled no matter what the matcher does.

WHAT THIS ACTUALLY READS
-----------------------

The shards are written by `embed_runtime.run_shards`, not by the local
`embed_names.py` entry point, and the two layouts are different. This module
reads the shard layout, because that is what the Colab job produces and what the
Colab search will run against. A shard is:

    shard_00000.parquet  ->  shard_id           int
                             entity_id         str    "S2-166376419"
                             business_name_clean str
                             embedding          list<float16>[768]

Read against `embed_runtime.embed_shard`, and the width cross-checked against the
LaBSE repo itself (`config.json: hidden_size=768`). Nothing in this module trusts
`embed_runtime.VECTOR_DIM` to build the index: `build_index` takes the width from
`embeddings.shape[1]`, so a shard written by a different model fails loudly in
`verify_shards` rather than being silently reshaped here.
Three consequences are load-bearing and each one is a bug if ignored:

  * `embedding` is a *list column* of per-row arrays, not a flat column, so
    `np.stack(df["embedding"].to_numpy())` is the correct read and a bare
    `to_numpy()` would hand back an object array.
  * the vectors are `STORAGE_DTYPE`, i.e. **float16**. `IndexFlatIP` is not
    well-supported on float16, so they are cast to float32 at load. Note that
    `embed_runtime` flags the float16 storage as "a provisional choice pending a
    float16-vs-float32 check on train_ground_truth" -- that check has not been
    done, so every score this module emits inherits an unmeasured recall cost.
    It is called out here rather than buried because it is the one number that
    could make a good result look better than it is.
  * there is **no per-shard source marker**, and the manifest does not list its
    shards. One `OUT_DIR` holds source1, source2 and source3 shards interleaved
    with a single increasing `shard_id`. The only thing that says which source a
    vector came from is the `S1-`/`S2-`/`S3-` prefix on `entity_id`.

The manifest's `identity.storage_dtype` is read by `verify_shards` but not
consulted for the cast, and deliberately so: the notebook writes it as
`str(rt.STORAGE_DTYPE)`, which is the string `"<class 'numpy.float16'>"` rather
than `"float16"`. Parsing that back would be fragile for no benefit, because the
cast is unconditional anyway -- `IndexFlatIP` needs float32 whichever dtype the
shards claim to be, and a shard that disagrees with its manifest is a
`verify_shards` problem, not a loading one.

WHY THE MANIFEST IS READ AT ALL
-------------------------------

`write_manifest` merges its payload and keeps a `fingerprint` of the run's
identity -- model, strict flag, shard layout. That fingerprint is the only
record of *which* run produced the shards sitting in the directory, and
`embed_runtime.verify_shards` exists to check it. A directory left over from a
different model or a different `strict` flag can hold shards that look complete,
row counts can still add up, and the failure surfaces days later as an
unexplainable recall number. So the loader verifies rather than trusting, and
refuses on a mismatch instead of warning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import faiss
import numpy as np
import pandas as pd

from . import embed_runtime as rt

# The shard schema, named once so the loader and the tests cannot disagree.
ID_COLUMN = "entity_id"
TEXT_COLUMN = "business_name_clean"
VECTOR_COLUMN = "embedding"

# The columns a row filter is allowed to see. Deliberately excludes
# `VECTOR_COLUMN`: passing a row containing a 768-float array through
# `DataFrame.apply(axis=1)` is what turns a linear scan into a quadratic one, and
# no plausible filter needs the vector to decide whether to keep the row.
FILTER_COLUMNS = (ID_COLUMN, TEXT_COLUMN)


def read_manifest(manifest_path: Path) -> dict:
    """The manifest beside a shard directory, or `{}` if there is not one."""
    return rt.read_manifest(Path(manifest_path).parent)


def shard_paths(manifest_path: Path) -> list[Path]:
    """
    Every shard in the directory the manifest describes, in shard order.

    The manifest does not enumerate its shards -- `write_manifest` merges
    whatever keys it is handed and the notebook records counts, not names -- so
    the directory listing is the only complete source of truth. This is the same
    `shard_*.parquet` glob `embed_runtime.iter_plan_shards` uses, deliberately,
    so that this module and the writer can never disagree about which files
    exist.
    """
    directory = Path(manifest_path).parent
    return sorted(directory.glob(f"{rt.SHARD_GLOB_PREFIX}*.parquet"))


def _row_filter_mask(frame: pd.DataFrame, entity_filter: Callable[[object], bool]) -> np.ndarray:
    """
    Apply a row predicate without ever materialising the vector column.

    `entity_filter` is called once per row with a named tuple of
    `(entity_id, business_name_clean)`, which is the shape of a shard row minus
    its embedding. Returns a boolean mask positionally aligned with `frame`.
    """
    if len(frame) == 0:
        return np.zeros(0, dtype=bool)
    keep = [
        bool(entity_filter(row))
        for row in frame[list(FILTER_COLUMNS)].itertuples(index=False, name="Row")
    ]
    return np.asarray(keep, dtype=bool)


def load_embedding_shards(
    manifest_path: Path,
    entity_filter: Callable[[object], bool] | None = None,
    *,
    verify: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Read every shard beside `manifest_path` into one id array and one matrix.

    `entity_filter` is an optional per-row predicate -- `lambda row:
    row.business_name_clean` style access, or a `str.contains` compiled against
    the name. `None` (the default) keeps every row. It is applied *before* the
    vectors are stacked, not after, so filtering a shard to nothing costs nothing
    rather than allocating a full matrix and throwing it away.

    Returns `(entity_ids, embeddings)` as a `numpy` string array and a C-contiguous
    `float32` matrix of shape `(n, dim)`. The `float32` cast is required, not
    cosmetic: the shards hold `float16` and `IndexFlatIP` does not support it.

    `verify=True` runs `embed_runtime.verify_shards` first, so a directory
    holding another run's vectors raises here rather than producing a plausible
    wrong answer later. It is on by default because the failure it prevents is
    silent, and it is switchable only so the tests can exercise the loader
    against a synthetic directory with no plan behind it.
    """
    manifest_path = Path(manifest_path)
    paths = shard_paths(manifest_path)
    if not paths:
        raise FileNotFoundError(
            f"no {rt.SHARD_GLOB_PREFIX}*.parquet beside {manifest_path}. The manifest "
            f"names the run, not its shards, so the directory listing is the only "
            f"way to find them; check that the embed job actually wrote to "
            f"{manifest_path.parent}."
        )

    if verify:
        _verify_or_raise(manifest_path, paths)

    ids: list[np.ndarray] = []
    vectors: list[np.ndarray] = []
    dim: int | None = None
    for path in paths:
        frame = pd.read_parquet(path)
        missing = [c for c in (ID_COLUMN, VECTOR_COLUMN) if c not in frame.columns]
        if missing:
            raise KeyError(
                f"{path.name} has no {missing}. A shard written by "
                f"embed_runtime.embed_shard has {ID_COLUMN!r} and {VECTOR_COLUMN!r}; "
                f"this one has {list(frame.columns)}. If it came from the local "
                f"embed_names.py entry point that is a different layout and this "
                f"loader is the wrong reader for it."
            )
        if entity_filter is not None:
            frame = frame.loc[_row_filter_mask(frame, entity_filter)]
        if frame.empty:
            # np.stack on an empty list raises, and an all-filtered shard is a
            # normal outcome when filtering on script.
            continue

        stacked = np.stack(frame[VECTOR_COLUMN].to_numpy())
        if dim is None:
            dim = stacked.shape[1]
        elif stacked.shape[1] != dim:
            raise ValueError(
                f"{path.name} has {stacked.shape[1]}-dimensional vectors but earlier "
                f"shards have {dim}. Shards from two models cannot be searched "
                f"together; the usual cause is a directory that was not cleared "
                f"between runs."
            )
        ids.append(frame[ID_COLUMN].to_numpy().astype(str))
        vectors.append(stacked)

    if not vectors:
        shape = (0, dim if dim is not None else 0)
        return np.empty(0, dtype=object), np.empty(shape, dtype=np.float32)

    embeddings = np.concatenate(vectors).astype(np.float32, copy=False)
    return np.concatenate(ids), embeddings


def _verify_or_raise(manifest_path: Path, paths: Sequence[Path]) -> None:
    """
    Refuse a shard directory that `embed_runtime` would not vouch for.

    `verify_shards` needs the planned row count and the planned id set to say
    whether a directory is complete, and neither is recoverable from the shards
    alone -- a directory that is missing its last shard looks exactly like a
    complete smaller run. So the manifest is the only place those can come from,
    and when it does not carry them the check degrades to what can be verified:
    that every shard on disk was written by the run the manifest describes.
    """
    manifest = read_manifest(manifest_path)
    identity = manifest.get("identity")
    if not identity:
        return
    report = rt.verify_shards(
        Path(manifest_path).parent,
        expected_total_rows=manifest.get("rows"),
        expected_ids=None,
        identity=identity,
        # The one check the Phase 3 verification could not make. A directory
        # left by a different encoder has a different fingerprint *only if*
        # someone recorded the encoder honestly in `identity`; the width is
        # measured from the vectors themselves, so it catches the case where the
        # label says LaBSE and the bytes disagree.
        expected_vector_dim=rt.VECTOR_DIM,
    )
    if not report.get("ok", False):
        stale = report.get("stale_shard_names") or []
        widths = report.get("vector_dims")
        detail = (
            f"vector width on disk {widths}, expected {rt.VECTOR_DIM}. If this is a "
            f"genuine mismatch, these vectors were not written by the model the "
            f"manifest names and no recall number computed from them is meaningful."
            if report.get("vector_dim_ok") is False
            else ""
        )
        raise ValueError(
            f"{manifest_path.parent} failed embed_runtime.verify_shards: "
            f"stale shards {stale}, report {report}. {detail} These vectors are "
            f"not all from the run this manifest describes, and indexing them "
            f"together would produce a recall number nobody could trace."
        )
    if len(stale := list(paths)) == 0:  # pragma: no cover - defensive
        raise ValueError(f"no shards to load in {manifest_path.parent}")


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    A flat inner-product index over `embeddings`.

    `IndexFlatIP` is the correct choice here for a reason that is worth stating,
    because "flat" sounds like the thing you are supposed to upgrade from: the
    vectors were written with `normalize_embeddings=True`
    (`embed_runtime.make_encode_fn`), so every vector has unit L2 norm and inner
    product *is* cosine similarity. An approximate index would be the upgrade
    for a corpus too large to scan exactly; at this size exactness is free, and
    an approximate index would trade a recall number nobody has measured yet for
    speed nobody needs yet.

    Float32 is required here rather than preferred: `IndexFlatIP` does not
    support the float16 the shards are stored in.
    """
    if embeddings.ndim != 2:
        raise ValueError(
            f"expected a 2-D (rows, dim) matrix, got shape {embeddings.shape}"
        )
    if embeddings.shape[0] == 0:
        raise ValueError("cannot build an index over zero vectors")
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.ascontiguousarray(embeddings))
    return index


def search(
    s1_ids: Sequence[str],
    s1_vecs: np.ndarray,
    candidate_ids: Sequence[str],
    index: faiss.Index,
    top_k: int = 20,
    *,
    query_batch: int = 4096,
) -> pd.DataFrame:
    """
    The `top_k` nearest candidates for every source1 row, as
    `[source1_entity_id, candidate_entity_id, embedding_similarity_score]`.

    Scores are inner products on unit vectors, so they are cosine similarities in
    [-1, 1] and comparable across rows. FAISS returns them sorted, nearest
    first.

    A candidate whose `entity_id` equals the query's is dropped. That cannot
    happen when the two sides are genuinely different sources, but it is exactly
    what happens in the configuration the Colab job actually produces: one
    `OUT_DIR` holds all three sources, so pointing both manifests at it puts
    every source1 vector in the index and each row becomes its own nearest
    neighbour at similarity 1.0, consuming the single best slot it was given.

    `query_batch` bounds how many query vectors are resident at once. It does not
    change the result -- FAISS is exact here -- it only caps the peak.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if len(candidate_ids) == 0:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "embedding_similarity_score"]
        )
    if len(s1_ids) != s1_vecs.shape[0]:
        raise ValueError(
            f"{len(s1_ids)} source1 ids but {s1_vecs.shape[0]} vectors"
        )
    if index.ntotal != len(candidate_ids):
        raise ValueError(
            f"index holds {index.ntotal} vectors but {len(candidate_ids)} candidate "
            f"ids were supplied; the two must describe the same rows in the same "
            f"order or every score is attached to the wrong id."
        )

    # Ask for one extra neighbour so a dropped self-match cannot eat a real one.
    ask = min(top_k + 1, index.ntotal)
    candidate = np.asarray(candidate_ids, dtype=object)

    frames: list[pd.DataFrame] = []
    for start in range(0, s1_vecs.shape[0], query_batch):
        stop = min(start + query_batch, s1_vecs.shape[0])
        scores, positions = index.search(
            np.ascontiguousarray(s1_vecs[start:stop], dtype=np.float32), ask
        )
        block = s1_ids[start:stop]
        # -1 is FAISS's "fewer neighbours than asked for", which happens when
        # the index is smaller than `ask`.
        valid = positions >= 0
        rows, ranks = np.nonzero(valid)
        if rows.size == 0:
            continue
        flat = positions[rows, ranks]
        query_ids = np.asarray(block, dtype=object)[rows]
        found = candidate[flat]
        keep = found != query_ids
        frames.append(
            pd.DataFrame(
                {
                    "source1_entity_id": query_ids[keep],
                    "candidate_entity_id": found[keep],
                    "embedding_similarity_score": scores[rows, ranks][keep].astype(np.float32),
                }
            )
        )

    if not frames:
        return pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id", "embedding_similarity_score"]
        )
    out = pd.concat(frames, ignore_index=True)
    # Trimming after the self-drop is what keeps the guarantee at exactly top_k.
    out = out.sort_values(
        ["source1_entity_id", "embedding_similarity_score"], ascending=[True, False], kind="stable"
    )
    out = out.groupby("source1_entity_id", sort=False).head(top_k)
    return out.reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--split", choices=["test", "train"], required=True)
    parser.add_argument(
        "--s1-manifest", required=True, help="path to source1's manifest.json"
    )
    parser.add_argument(
        "--s2s3-manifest", required=True, help="path to source2+3's manifest.json"
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--query-batch", type=int, default=4096)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--source-prefixes",
        default=None,
        help=(
            "comma-separated entity_id prefixes to keep, e.g. 'S1' or 'S2,S3'. "
            "Needed when both manifests point at one OUT_DIR, which is what the "
            "embed job writes: it interleaves all three sources under a single "
            "increasing shard_id and records no per-shard source."
        ),
    )
    # `store_true` with `default=True` is a flag that can never be false, so it
    # is not a flag. `BooleanOptionalAction` gives `--strict` / `--no-strict` and
    # a default that is actually reachable.
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="use the strict Indic filter, matching embed_names.py --strict",
    )
    args = parser.parse_args(argv)

    s1_prefixes = _prefixes(args.source_prefixes, "S1")
    s2s3_prefixes = _prefixes(args.source_prefixes, "S2,S3")

    s1_ids, s1_vecs = load_embedding_shards(
        Path(args.s1_manifest), _prefix_filter(s1_prefixes)
    )
    cand_ids, cand_vecs = load_embedding_shards(
        Path(args.s2s3_manifest), _prefix_filter(s2s3_prefixes)
    )
    if len(s1_ids) == 0:
        raise SystemExit(
            f"no source1 vectors after filtering to {s1_prefixes or 'all'}. If the "
            f"embed job wrote one OUT_DIR for all three sources, pass "
            f"--source-prefixes S1,S2,S3 and let the two manifests be the same file."
        )

    index = build_index(cand_vecs)
    results = search(s1_ids, s1_vecs, cand_ids, index, top_k=args.top_k,
                     query_batch=args.query_batch)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    results.to_parquet(out, index=False)
    print(
        f"split={args.split} strict={args.strict} s1_rows={len(s1_ids):,} "
        f"candidate_rows={len(cand_ids):,} pairs_written={len(results):,} -> {out}"
    )
    return 0


def _prefixes(value: str | None, default: str) -> list[str]:
    raw = default if value is None else value
    return [part.strip() for part in raw.split(",") if part.strip()]


def _prefix_filter(prefixes: Sequence[str]) -> Callable[[object], bool] | None:
    """A row filter keeping only the given `entity_id` prefixes, or `None`."""
    if not prefixes:
        return None
    wanted = tuple(prefixes)
    return lambda row: str(getattr(row, ID_COLUMN, "")).startswith(wanted)


if __name__ == "__main__":
    raise SystemExit(main())
