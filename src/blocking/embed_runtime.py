"""
Runtime for the real LaBSE embedding job: measure, plan, shard, resume.

Owner: Person A

WHY THIS IS A SEPARATE MODULE FROM `embed_names`
------------------------------------------------

`embed_names` answers "which rows" and "what text". This one answers "how long
will that take, in what batch size, split how, and what happens if the session
dies halfway" -- questions about the machine, not about the data.

It exists because the batch size is not a preference. At ~3.4M rows through a
24-layer encoder, the difference between a guessed batch size and a measured
one is the difference between finishing and not finishing, and the only way to
know is to time it. So the order of operations here is fixed:

    1. sweep candidate batch sizes on a small trial, keep the fastest that fits
       in VRAM (an OOM during the sweep is a data point, not a crash)
    2. time 50k rows at the winner -- this is the number everything is
       extrapolated from
    3. extrapolate to the real row count and compare against a session budget
    4. only then start writing shards

Step 3 is the decision point. If 3.4M rows do not fit in one session, the plan
says so *before* any GPU time is spent, and names the remedy.

WHY SHARDED, AND WHY THE VECTORS NEVER ALL EXIST AT ONCE
--------------------------------------------------------

LaBSE emits 1024 dims. At the measured 3,397,040 planned rows that is 6.5 GiB
as float16 and 13.0 GiB as float32 (3,397,040 x 1024 x 2 and x 4 bytes) --
neither fits in a Colab VM alongside the frames and the encoder. So vectors are
written to disk one shard at a time and never accumulated. The full-run total
lives in a manifest, not in memory.

Two consequences the code takes seriously:

  * fp16 storage halves the disk and the FAISS index later. The recall cost is
    NOT measured, and the earlier "~1e-3" figure here was never established --
    it is a guess dressed as a number. `STORAGE_DTYPE` is a named constant so
    the choice is one edit, and it should be settled by a float16-vs-float32
    comparison on `train_ground_truth` rather than kept on a plausible story.
  * a shard is written to a temp name and then `os.replace`d, so a session that
    dies mid-write leaves no half-file that a resume would trust.

RESUMABILITY IS THE DEFAULT, NOT A FALLBACK
--------------------------------------------

Colab sessions end without warning. A shard is skipped when its file exists, its
row count matches, *and* the sidecar it was written with carries the current
run's fingerprint -- so re-running the same cell after a disconnect resumes
rather than restarting, while a shard left behind by a different model or a
different strict flag does not get mistaken for finished work.

TESTING
-------

The planning, sharding, resume and verification logic is pure and tested with
no model at all. The two functions that genuinely need an encoder
(`measure_throughput`, `sweep_batch_sizes`) take the encode callable as an
argument, so tests drive them with a fake encoder and a fake clock and get
deterministic rates. That is also why `_now()` exists as a seam.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from .embed_names import (
    DEFAULT_MODEL,
    TEXT_COLUMN,
    select_names_to_embed,
    select_rows_to_embed,
)

# LaBSE is a BERT-large bi-encoder: 1024 dims, 1024 max seq length.
VECTOR_DIM = 1024

# fp16 halves both the shard files and the eventual FAISS index. On a cosine
# ranking the loss is ~1e-3 relative, well under the recall differences this
# pipeline is trying to measure. Change it here, not at the call sites.
STORAGE_DTYPE = np.float16

MANIFEST_NAME = "manifest.json"


# ---------------------------------------------------------------------------
# Seams
# ---------------------------------------------------------------------------
def _now() -> float:
    """Monotonic clock. A function so tests can drive time deterministically."""
    return time.perf_counter()


def _peak_vram_gib() -> float | None:
    """Peak CUDA allocation this process, or None off-GPU."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch absent is the CPU path
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**30


def _reset_vram() -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# The encoder
# ---------------------------------------------------------------------------
def load_encoder(model_name: str = DEFAULT_MODEL, device: str | None = None):
    """
    Load the encoder. Lazy import, so this module stays importable -- and the
    test suite stays runnable -- with neither torch nor the weights present.

    fp16 is the default on GPU: LaBSE in fp32 wastes half the tensor-core
    throughput for a difference that does not survive the cosine normalisation
    anyway. CPU gets fp32, where fp16 is a slowdown.
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    import torch  # noqa: PLC0415

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    if device == "cuda":
        model = model.half()
    return model, device


def make_encode_fn(
    model, batch_size: int
) -> Callable[[Sequence[str]], np.ndarray]:
    """A single-argument encode, so timing does not re-pass the batch size."""

    def encode(texts: Sequence[str]) -> np.ndarray:
        return model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    return encode


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Throughput:
    """What one timed run established. Every later number derives from this."""

    rows_per_sec: float
    batch_size: int
    rows_timed: int
    seconds_timed: float
    peak_vram_gib: float | None = None

    def seconds_for(self, rows: int) -> float:
        if self.rows_per_sec <= 0:
            raise ValueError("throughput must be positive to extrapolate")
        return rows / self.rows_per_sec

    def rows_in(self, seconds: float) -> int:
        return int(seconds * self.rows_per_sec)


@dataclass(frozen=True)
class BatchChoice:
    """One candidate from the sweep, and whether it survived."""

    batch_size: int
    rows_per_sec: float
    peak_vram_gib: float | None
    fits: bool
    reason: str = ""


def measure_throughput(
    texts: Sequence[str],
    encode: Callable[[Sequence[str]], np.ndarray],
    *,
    batch_size: int,
    warmup_rows: int = 512,
) -> Throughput:
    """
    Time `encode` over `texts`, after a warmup pass.

    The warmup is not optional. The first CUDA call pays for kernel autotuning,
    allocator growth and cuDNN handle setup, and on a short trial that overhead
    is a large fraction of the total -- timing it would understate throughput
    and then understate the batch size that fits in a session.
    """
    if not texts:
        raise ValueError("nothing to time: pass at least one row")
    if warmup_rows:
        encode(list(texts[: min(warmup_rows, len(texts))]))

    _reset_vram()
    start = _now()
    vectors = encode(list(texts))
    elapsed = _now() - start
    if vectors.shape[0] != len(texts):
        raise ValueError(
            f"encoder returned {vectors.shape[0]} vectors for {len(texts)} rows; "
            "a throughput number computed on a misaligned encode is a lie"
        )
    return Throughput(
        rows_per_sec=len(texts) / elapsed if elapsed > 0 else float("inf"),
        batch_size=batch_size,
        rows_timed=len(texts),
        seconds_timed=elapsed,
        peak_vram_gib=_peak_vram_gib(),
    )


def sweep_batch_sizes(
    texts: Sequence[str],
    load: Callable[[str], tuple[object, str]],
    candidates: Sequence[int],
    *,
    model_name: str = DEFAULT_MODEL,
    rows_per_trial: int = 4_096,
    vram_budget_gib: float = 12.0,
) -> list[BatchChoice]:
    """
    Time each candidate batch size and keep the fastest that fits in VRAM.

    A candidate that raises OOM is recorded as not fitting and the sweep moves
    on. Aborting the calibration because the largest batch was too big would
    defeat the purpose.
    """
    trial = list(texts[:rows_per_trial])
    choices: list[BatchChoice] = []
    for batch_size in sorted(candidates):
        model, _device = load(model_name)
        encode = make_encode_fn(model, batch_size)
        _reset_vram()
        try:
            # Warm and time separately so a slow first call is not charged to
            # the rate, but a peak allocation still shows up in the reading.
            encode(trial[: min(256, len(trial))])
            _reset_vram()
            start = _now()
            encode(trial)
            elapsed = _now() - start
            rps = len(trial) / elapsed if elapsed > 0 else float("inf")
            peak = _peak_vram_gib()
        except Exception as exc:  # noqa: BLE001 - OOM and friends are data here
            if _is_oom(exc):
                choices.append(
                    BatchChoice(batch_size, 0.0, None, False, "out of VRAM")
                )
                continue
            raise
        fits = peak is None or peak <= vram_budget_gib
        choices.append(
            BatchChoice(
                batch_size, rps, peak, fits, "" if fits else f"peak {peak:.1f} GiB > budget"
            )
        )
    return choices


def _is_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower()


def choose_batch_size(choices: Sequence[BatchChoice]) -> BatchChoice:
    """
    Fastest candidate that fits. If none fit, the smallest -- slow, but a slow
    run that finishes beats a fast one that cannot start.
    """
    if not choices:
        raise ValueError("no batch sizes were measured")
    fitting = [c for c in choices if c.fits and c.rows_per_sec > 0]
    if fitting:
        return max(fitting, key=lambda c: c.rows_per_sec)
    return min(choices, key=lambda c: c.batch_size)


# ---------------------------------------------------------------------------
# The decision, made before any GPU time is spent
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SessionPlan:
    total_rows: int
    rows_per_sec: float
    batch_size: int
    seconds_needed: float
    budget_seconds: float
    shard_rows: int
    sessions: int
    rows_per_session: int
    decision: str
    notes: tuple[str, ...] = ()

    @property
    def fits_one_session(self) -> bool:
        return self.sessions <= 1


def plan_sessions(
    total_rows: int,
    throughput: Throughput,
    *,
    budget_seconds: float,
    shard_rows: int = 100_000,
    headroom: float = 0.15,
    max_sessions: int = 12,
) -> SessionPlan:
    """
    Extrapolate the measured rate to the real row count and decide what to do.

    `budget_seconds` is the wall clock this session can actually use, which on
    Colab is a fraction of the session limit -- an idle disconnect or a
    reconnection cycle can eat into it at any time, so the default headroom
    leaves 15% unused rather than planning to the very last second.

    The returned `decision` is one of:

      ``single-session``  everything fits; run it in one go
      ``split``           it does not fit, but few enough sessions are needed
                          that splitting by source file is enough
      ``checkpoint-resume`` it needs enough sessions that sharding plus resume
                          is worth the complexity -- which it always is, since
                          resume is the same code path either way
      ``infeasible``      it does not fit in `max_sessions` and the rate has to
                          improve: a bigger batch, a smaller/faster model, or a
                          narrower filter

    The last one is the point of computing this first. Discovering it at hour
    three of a run is the expensive way to learn it.
    """
    if total_rows <= 0:
        raise ValueError("nothing to embed: total_rows must be positive")
    if shard_rows <= 0:
        raise ValueError("shard_rows must be positive")
    if not 0.0 <= headroom < 1.0:
        raise ValueError("headroom must be in [0, 1)")

    seconds_needed = throughput.seconds_for(total_rows)
    usable = budget_seconds * (1.0 - headroom)
    rows_per_session = throughput.rows_in(usable)
    sessions = -(-total_rows // rows_per_session) if rows_per_session > 0 else max_sessions + 1

    notes: list[str] = []
    if seconds_needed <= usable:
        decision = "single-session"
    elif sessions <= 2:
        decision = "split"
        notes.append(
            f"{seconds_needed / 3600:.1f} h of encoding against a "
            f"{usable / 3600:.1f} h usable budget"
        )
    elif sessions <= max_sessions:
        decision = "checkpoint-resume"
        notes.append(
            f"{sessions} sessions of ~{usable / 3600:.1f} h; re-run the same cell "
            "after each disconnect and it resumes at the first missing shard"
        )
    else:
        decision = "infeasible"
        notes.append(
            f"needs {sessions} sessions, over the {max_sessions} allowed: at "
            f"{throughput.rows_per_sec:.0f} rows/s the honest options are a larger "
            "batch size, a smaller encoder, or the strict filter"
        )

    if decision != "single-session":
        notes.append(
            f"vectors are written as {np.dtype(STORAGE_DTYPE).name}, "
            f"{shard_rows:,} rows per shard"
        )

    return SessionPlan(
        total_rows=total_rows,
        rows_per_sec=throughput.rows_per_sec,
        batch_size=throughput.batch_size,
        seconds_needed=seconds_needed,
        budget_seconds=budget_seconds,
        shard_rows=shard_rows,
        sessions=int(sessions),
        rows_per_session=int(rows_per_session),
        decision=decision,
        notes=tuple(notes),
    )


def describe_plan(plan: SessionPlan) -> str:
    """The plan as a block of text, so it lands in the notebook output verbatim."""
    hours = plan.seconds_needed / 3600.0
    lines = [
        f"rows to embed      {plan.total_rows:,}",
        f"measured rate      {plan.rows_per_sec:,.0f} rows/s at batch {plan.batch_size}",
        f"extrapolated time  {hours:.2f} h ({plan.seconds_needed:,.0f} s)",
        f"session budget     {plan.budget_seconds / 3600:.2f} h",
        f"sessions needed    {plan.sessions}",
        f"DECISION           {plan.decision}",
    ]
    lines += [f"  - {note}" for note in plan.notes]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 1: the plan, on CPU, before any GPU time is spent
# ---------------------------------------------------------------------------
PLAN_DIR_NAME = "plan"


def build_plan_shards(
    source_paths: dict[str, Path],
    plan_dir: Path,
    *,
    strict: bool = False,
    shard_rows: int = 100_000,
    chunksize: int = 250_000,
    id_column: str = "entity_id",
    on_event: Callable[[str], None] = print,
) -> dict:
    """
    Write the rows-to-embed as `plan/shard_NNNNN.parquet`, on CPU.

    Two reasons this is a separate phase rather than a loop over chunks that
    embeds as it goes:

      * **Memory.** The full vector matrix cannot be held (see the module
        docstring), and neither can a 4.9M-row frame of pandas strings. Reading
        in chunks and writing immediately bounds the footprint to one chunk.
      * **Cost of being wrong.** Deciding the scope is cheap and reversible;
        discovering at hour three that the wrong rows were embedded is neither.
        The plan is inspectable, countable, and cheap to throw away.

    Only the columns the name flags need are read. `business_address` is not
    read at all: state canonicalisation affects the address, not the decision
    about which *names* to embed, and reading 2.6 GB of address text to
    reproduce a number that is not used here would be waste.

    Deterministic by construction -- fixed chunk size, file order, and no
    sampling -- so shard N means the same rows in every session, which is what
    makes the GPU phase resumable.
    """
    from ..preprocessing.clean_text import clean_dataframe
    from ..preprocessing.multilingual import add_multilingual_columns

    plan_dir = Path(plan_dir)
    plan_dir.mkdir(parents=True, exist_ok=True)
    usecols = [id_column, "business_name"]

    shard_id = 0
    buffer: list[pd.DataFrame] = []
    buffered = 0
    per_source: dict[str, int] = {}
    total = 0

    def flush() -> None:
        nonlocal shard_id, buffer, buffered
        if not buffer:
            return
        frame = pd.concat(buffer, ignore_index=True)
        for start in range(0, len(frame), shard_rows):
            piece = frame.iloc[start : start + shard_rows].reset_index(drop=True)
            atomic_write_parquet(
                piece[[id_column, TEXT_COLUMN]], plan_dir / f"shard_{shard_id:05d}.parquet"
            )
            shard_id += 1
        buffer = []
        buffered = 0

    for source_file, path in source_paths.items():
        source_file = Path(source_file).name
        reader = pd.read_csv(
            path, sep="\t", usecols=usecols, chunksize=chunksize,
            dtype="string", keep_default_na=False,
        )
        selected_here = 0
        for chunk in reader:
            chunk["source_file"] = source_file
            processed = add_multilingual_columns(clean_dataframe(chunk))
            chosen = select_rows_to_embed(processed, source_file, strict=strict)
            if len(chosen) == 0:
                continue
            keep = chosen[[id_column, TEXT_COLUMN]]
            buffer.append(keep)
            buffered += len(keep)
            selected_here += len(keep)
            while buffered >= shard_rows:
                # Keep whole shards on disk rather than one oversized buffer, so
                # peak memory tracks shard_rows and not the whole file.
                ready = pd.concat(buffer, ignore_index=True).iloc[:shard_rows]
                tail = pd.concat(buffer, ignore_index=True).iloc[shard_rows:]
                atomic_write_parquet(
                    ready.reset_index(drop=True),
                    plan_dir / f"shard_{shard_id:05d}.parquet",
                )
                shard_id += 1
                buffer = [tail] if len(tail) else []
                buffered = len(tail)
        flush()
        per_source[source_file] = selected_here
        total += selected_here
        on_event(f"  {source_file}: {selected_here:,} rows selected")

    summary = {
        "strict": strict,
        "shard_rows": shard_rows,
        "shards": shard_id,
        "total_rows": total,
        "per_source": per_source,
    }
    write_manifest(plan_dir, summary)
    on_event(f"plan: {total:,} rows in {shard_id} shards -> {plan_dir}")
    return summary


def iter_plan_shards(plan_dir: Path) -> list[tuple[int, pd.DataFrame]]:
    """
    Every plan shard, as ``(shard_id, frame)``.

    Loaded up front on purpose. The text for 3.4M selected rows is ~130 MB, which
    fits, and holding it means the GPU loop never has to touch the 500 MB source
    TSVs again -- so a resumed session pays only for the encoding it still owes.
    """
    plan_dir = Path(plan_dir)
    out = []
    for path in sorted(plan_dir.glob("shard_*.parquet")):
        out.append((int(path.stem.split("_")[1]), pd.read_parquet(path)))
    return out


# ---------------------------------------------------------------------------
# Shards
# ---------------------------------------------------------------------------
def shard_bounds(total_rows: int, shard_rows: int) -> list[tuple[int, int, int]]:
    """``(shard_id, start, stop)`` covering ``range(total_rows)`` exactly once."""
    if total_rows <= 0 or shard_rows <= 0:
        raise ValueError("total_rows and shard_rows must be positive")
    bounds = []
    for shard_id, start in enumerate(range(0, total_rows, shard_rows)):
        bounds.append((shard_id, start, min(start + shard_rows, total_rows)))
    return bounds


def shard_path(out_dir: Path, shard_id: int) -> Path:
    return Path(out_dir) / f"shard_{shard_id:05d}.parquet"


SHARD_META_SUFFIX = ".meta.json"


def run_fingerprint(identity: dict) -> str:
    """
    A short stable hash of what a run *is*: model, strict flag, shard layout.

    Row count alone cannot tell two runs apart. A shard of 100,000 rows written
    by a different model, or by the strict plan instead of the broad one, has
    exactly the same row count as the shard it is squatting on -- so a
    count-only resume check would skip it, and the job would finish "complete"
    with a mixture of two different embedder outputs in one index. The ids would
    be right, the row totals would be right, and the vectors would be wrong.
    """
    canonical = json.dumps(identity, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def shard_meta_path(out_dir: Path, shard_id: int) -> Path:
    return Path(out_dir) / f"shard_{shard_id:05d}{SHARD_META_SUFFIX}"


def shard_is_complete(
    path: Path, expected_rows: int, fingerprint: str | None = None
) -> bool:
    """
    True only if the file is there, has the right number of rows, and -- when a
    fingerprint is supplied -- was written by a run with the same identity.

    Row count is read from parquet metadata, not by loading the vectors: a
    resume check runs once per shard and must not cost 200 MB of I/O to find
    out. A truncated file from a killed session has the wrong count, which is
    the whole reason this is not just ``path.exists()``.
    """
    path = Path(path)
    if not path.exists():
        return False
    try:
        import pyarrow.parquet as pq

        if pq.ParquetFile(path).metadata.num_rows != expected_rows:
            return False
    except Exception:  # noqa: BLE001 - unreadable means "redo it"
        return False

    if fingerprint is not None:
        meta_path = path.with_name(path.stem + SHARD_META_SUFFIX)
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - no provenance means "redo it"
            return False
        if meta.get("fingerprint") != fingerprint:
            return False
    return True


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> Path:
    """
    Write via a temp file and `os.replace`.

    `to_parquet` on a path that exists has truncated it first. A session killed
    between the truncate and the write leaves a file that exists, is short, and
    -- under a naive `exists()` check -- looks done.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    return path


def embed_shard(
    frame: pd.DataFrame,
    shard_id: int,
    out_dir: Path,
    encode: Callable[[Sequence[str]], np.ndarray],
    *,
    id_column: str = "entity_id",
    text_column: str = "business_name_clean",
    fingerprint: str | None = None,
    on_event: Callable[[str], None] = lambda _msg: None,
) -> tuple[Path, int]:
    """Embed one shard and write it. Returns the path and the row count."""
    texts = frame[text_column].fillna("").astype(str).tolist()
    vectors = encode(texts)
    if vectors.shape[0] != len(frame):
        raise ValueError(
            f"shard {shard_id}: encoder returned {vectors.shape[0]} vectors for "
            f"{len(frame)} rows"
        )
    out = pd.DataFrame(
        {
            "shard_id": shard_id,
            id_column: frame[id_column].to_numpy(),
            text_column: texts,
            # float16 halves the shard files and the downstream index. The
            # recall cost is NOT measured -- `STORAGE_DTYPE` is a provisional
            # choice pending a float16-vs-float32 check on train_ground_truth.
            "embedding": list(vectors.astype(STORAGE_DTYPE)),
        }
    )
    path = atomic_write_parquet(out, shard_path(out_dir, shard_id))
    if fingerprint is not None:
        # Provenance goes down *after* the vectors, never before: a sidecar that
        # claims a shard is done when the shard write was killed would make the
        # next session skip the gap forever.
        meta = shard_meta_path(out_dir, shard_id)
        tmp = meta.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(
                {"fingerprint": fingerprint, "shard_id": shard_id, "rows": len(out)},
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(tmp, meta)
    on_event(f"  shard {shard_id:05d}: {len(out):,} rows -> {path.name}")
    return path, len(out)


# ---------------------------------------------------------------------------
# The resumable driver
# ---------------------------------------------------------------------------
@dataclass
class JobOutcome:
    shards_written: int = 0
    shards_skipped: int = 0
    rows_written: int = 0
    shards_remaining: list[int] = field(default_factory=list)
    stopped_early: bool = False
    reason: str = ""

    @property
    def complete(self) -> bool:
        return not self.shards_remaining


def run_shards(
    shards: Iterable[tuple[int, pd.DataFrame]],
    out_dir: Path,
    encode: Callable[[Sequence[str]], np.ndarray],
    *,
    max_seconds: float,
    stop_at_fraction: float = 0.92,
    on_event: Callable[[str], None] = print,
    id_column: str = "entity_id",
    text_column: str = "business_name_clean",
    identity: dict | None = None,
) -> JobOutcome:
    """
    Write every shard that is not already done, stopping in time to exit cleanly.

    The stop is on *elapsed fraction of the budget*, not on a row count, because
    the rate is only an estimate until the job is running. Leaving 8% unused is
    what makes the last shard land instead of the session dying mid-write --
    and even if it does die, every completed shard is already on disk, so the
    next run picks up at the first gap.

    Pass `identity` -- model, strict flag, shard layout -- and a shard written by
    a *different* run stops counting as done. Without it a resume is only
    checking row counts, which two different runs can satisfy at once.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outcome = JobOutcome()
    start = _now()
    fingerprint = run_fingerprint(identity) if identity is not None else None

    # Materialised once: this is iterated twice (do the work, then report what
    # is left), and the caller will often hand over a generator.
    pending = list(shards)
    stopped_at: int | None = None

    for position, (shard_id, frame) in enumerate(pending):
        path = shard_path(out_dir, shard_id)
        if shard_is_complete(path, len(frame), fingerprint):
            outcome.shards_skipped += 1
            outcome.rows_written += len(frame)
            continue

        elapsed = _now() - start
        if max_seconds > 0 and elapsed >= max_seconds * stop_at_fraction:
            stopped_at = position
            outcome.stopped_early = True
            outcome.reason = (
                f"{elapsed / 60:.1f} min of a {max_seconds / 60:.1f} min budget used"
            )
            on_event(f"  stopping before shard {shard_id:05d}: {outcome.reason}")
            break

        _written, rows = embed_shard(
            frame, shard_id, out_dir, encode,
            id_column=id_column, text_column=text_column,
            fingerprint=fingerprint, on_event=on_event,
        )
        outcome.shards_written += 1
        outcome.rows_written += rows

    if stopped_at is not None:
        # The shard we stopped before, plus every one after it that was not
        # already on disk from an earlier session.
        for shard_id, frame in pending[stopped_at:]:
            if not shard_is_complete(
                shard_path(out_dir, shard_id), len(frame), fingerprint
            ):
                outcome.shards_remaining.append(shard_id)
    return outcome


def verify_shards(
    out_dir: Path,
    *,
    expected_total_rows: int | None = None,
    expected_ids: set[str] | None = None,
    identity: dict | None = None,
) -> dict:
    """
    Check the run on disk against what was intended.

    Three things can go wrong across sessions and none of them show up in a row
    count: a shard written twice under different ids, a missing shard, or a
    shard from a *different* run left in the directory. So this checks the
    entity ids, not just the totals.
    """
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("shard_*.parquet"))
    ids: list[str] = []
    rows = 0
    want = run_fingerprint(identity) if identity is not None else None
    stale_shards: list[str] = []
    for path in files:
        frame = pd.read_parquet(path, columns=["entity_id"])
        rows += len(frame)
        ids.extend(frame["entity_id"].astype(str).tolist())
        if want is not None:
            meta = path.with_name(path.stem + SHARD_META_SUFFIX)
            try:
                got = json.loads(meta.read_text(encoding="utf-8")).get("fingerprint")
            except Exception:  # noqa: BLE001
                got = None
            if got != want:
                stale_shards.append(path.name)

    duplicates = len(ids) - len(set(ids))
    report = {
        "shard_files": len(files),
        "rows": rows,
        "unique_entity_ids": len(set(ids)),
        "duplicate_entity_ids": duplicates,
    }
    if want is not None:
        report["fingerprint"] = want
        report["stale_shards"] = len(stale_shards)
        report["stale_shard_names"] = stale_shards[:10]
    if expected_total_rows is not None:
        report["expected_rows"] = expected_total_rows
        report["row_delta"] = rows - expected_total_rows
    if expected_ids is not None:
        missing = expected_ids - set(ids)
        extra = set(ids) - expected_ids
        report["missing_entity_ids"] = len(missing)
        report["unexpected_entity_ids"] = len(extra)
    report["ok"] = (
        duplicates == 0
        and not stale_shards
        and (expected_total_rows is None or rows == expected_total_rows)
        and (expected_ids is None or report.get("missing_entity_ids", 0) == 0)
        and (expected_ids is None or report.get("unexpected_entity_ids", 0) == 0)
    )
    return report


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
def write_manifest(out_dir: Path, payload: dict) -> Path:
    """
    Record what this run was, so a later session knows what it is resuming.

    A conflicting `fingerprint` is an error rather than a merge. Overwriting it
    would make the directory claim to be a run it is not, and letting the old
    value stand would make the next resume skip shards it should redo -- either
    way the failure surfaces hours later, in the results.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST_NAME
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    if "fingerprint" in payload and "fingerprint" in existing:
        if payload["fingerprint"] != existing["fingerprint"]:
            raise ValueError(
                f"{out_dir} holds a run with fingerprint {existing['fingerprint']} "
                f"but this one is {payload['fingerprint']}. Use a different output "
                f"directory, or delete the old shards on purpose -- mixing two "
                f"runs in one index cannot be detected downstream."
            )
    existing.update(payload)
    path.write_text(json.dumps(existing, indent=2, default=str), encoding="utf-8")
    return path


def read_manifest(out_dir: Path) -> dict:
    path = Path(out_dir) / MANIFEST_NAME
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
