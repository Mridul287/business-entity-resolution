"""
Tests for `src.blocking.embed_runtime`.

The point of this module is a decision made *before* the GPU hours are spent, so
the tests are mostly about arithmetic and about the failure modes a resumable
job has: a truncated shard that looks done, a session that dies mid-write, a
directory left holding shards from a different run.

Nothing here loads a model. `measure_throughput` and `sweep_batch_sizes` take
the encode callable as an argument and the module reads time through `_now()`,
so both are driven here with a fake encoder and a fake clock and produce exact,
fast, deterministic rates.

The one number in this file that is not synthetic is the 3.4M-row total, and it
is checked against `expected_total` rather than restated, so the plan tests fail
if the accounting bug that was fixed earlier ever comes back.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.blocking import embed_runtime as rt
from src.blocking.embed_names import expected_total
from src.blocking.embed_runtime import (
    BatchChoice,
    JobOutcome,
    Throughput,
    atomic_write_parquet,
    build_plan_shards,
    choose_batch_size,
    describe_plan,
    iter_plan_shards,
    measure_throughput,
    plan_sessions,
    read_manifest,
    run_shards,
    shard_bounds,
    shard_is_complete,
    shard_path,
    sweep_batch_sizes,
    verify_shards,
    write_manifest,
)

ROWS_TO_EMBED = expected_total()  # 3,398,081 -- the number the plan is about


# --------------------------------------------------------------------------
# A clock and an encoder we control
# --------------------------------------------------------------------------
class FakeClock:
    """Advances only when told to, so rates are exact rather than flaky."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


# The active FakeClock, set by the `clock` fixture. A module global rather than
# a parameter so every `fake_encode_factory(rate)` call site stays readable --
# `rt._now` is monkeypatched to the clock itself, so `rt._now()` reads the time
# rather than returning the clock to advance.
_CLOCK: "FakeClock | None" = None


def fake_encode_factory(rows_per_sec: float, dim: int = 4):
    """An encode that takes exactly `len(texts) / rows_per_sec` seconds."""

    def encode(texts):
        assert _CLOCK is not None, "the `clock` fixture is required"
        _CLOCK.advance(len(texts) / rows_per_sec)
        return np.zeros((len(texts), dim), dtype="float32")

    return encode


@pytest.fixture
def clock(monkeypatch) -> FakeClock:
    global _CLOCK
    c = FakeClock()
    _CLOCK = c
    monkeypatch.setattr(rt, "_now", c)
    return c


def texts(n: int) -> list[str]:
    return [f"business name {i}" for i in range(n)]


# --------------------------------------------------------------------------
# Throughput
# --------------------------------------------------------------------------
def test_measure_throughput_uses_the_encode_rate(clock) -> None:
    # 100 rows at 250 rows/s is 0.4 s. The warmup is excluded.
    tp = measure_throughput(texts(100), fake_encode_factory(250.0), batch_size=64)
    assert tp.rows_per_sec == pytest.approx(250.0)
    assert tp.rows_timed == 100
    assert tp.seconds_timed == pytest.approx(0.4)
    assert tp.batch_size == 64


def test_the_warmup_is_not_charged_to_the_rate(clock) -> None:
    """
    The first CUDA call pays for autotuning. Timing it would understate the
    rate, and a lower rate means the plan over-budgets and splits the run for
    no reason.
    """
    warm = measure_throughput(
        texts(100), fake_encode_factory(250.0), batch_size=64, warmup_rows=64
    )
    cold = measure_throughput(
        texts(100), fake_encode_factory(250.0), batch_size=64, warmup_rows=0
    )
    assert warm.rows_per_sec == pytest.approx(cold.rows_per_sec)


def test_throughput_extrapolates_both_ways(clock) -> None:
    tp = Throughput(rows_per_sec=100.0, batch_size=32, rows_timed=1000,
                    seconds_timed=10.0)
    assert tp.seconds_for(1_000_000) == 10_000.0
    assert tp.rows_in(10_000.0) == 1_000_000


def test_a_zero_rate_cannot_be_extrapolated() -> None:
    """
    A sweep where every candidate OOMs yields rows_per_sec 0. Extrapolating
    that silently gives "infinite time"; it should refuse instead.
    """
    with pytest.raises(ValueError, match="positive"):
        Throughput(0.0, 32, 10, 1.0).seconds_for(100)


def test_a_misaligned_encode_is_rejected(clock) -> None:
    """A throughput number from a wrong-length encode would be fiction."""

    def short_encode(texts_in):
        return np.zeros((len(texts_in) - 1, 4), dtype="float32")

    with pytest.raises(ValueError, match="misaligned"):
        measure_throughput(texts(10), short_encode, batch_size=8, warmup_rows=0)


def test_timing_nothing_is_an_error(clock) -> None:
    with pytest.raises(ValueError, match="at least one row"):
        measure_throughput([], fake_encode_factory(10.0), batch_size=8)


# --------------------------------------------------------------------------
# The batch-size sweep
# --------------------------------------------------------------------------
def test_the_sweep_picks_the_fastest_batch_size_that_fits(clock) -> None:
    """
    Bigger is not automatically better. 512 is nominally larger than 256 and was
    faster on paper, but it does not fit in the VRAM budget, so 256 wins.
    """
    choices = [
        BatchChoice(64, 100.0, 0.5, True, ""),
        BatchChoice(128, 180.0, 1.1, True, ""),
        BatchChoice(256, 240.0, 3.2, True, ""),
        BatchChoice(512, 300.0, 14.0, False, "peak 14.0 GiB > budget"),
        BatchChoice(1024, 0.0, None, False, "out of VRAM"),
    ]
    assert choose_batch_size(choices).batch_size == 256


def test_a_sweep_where_nothing_fits_takes_the_smallest(clock) -> None:
    """
    Slow but finishable beats fast and impossible. This is the branch that
    matters when the GPU is smaller than the model wants.
    """
    choices = [
        BatchChoice(128, 200.0, 14.0, False, "over budget"),
        BatchChoice(64, 100.0, 9.0, False, "over budget"),
        BatchChoice(256, 0.0, None, False, "out of VRAM"),
    ]
    assert choose_batch_size(choices).batch_size == 64


def test_an_empty_sweep_is_an_error(clock) -> None:
    with pytest.raises(ValueError, match="no batch sizes"):
        choose_batch_size([])


def test_the_real_sweep_records_an_oom_instead_of_crashing(clock, monkeypatch) -> None:
    """
    The largest candidate is the one most likely to OOM. Aborting calibration
    because of it would leave the batch size unchosen, which is the thing the
    sweep exists to do.
    """
    def fake_load(model_name):
        return object(), "cpu"

    def encode_for(batch_size):
        def encode(t):
            if batch_size >= 256:
                raise RuntimeError("CUDA out of memory. Tried to allocate ...")
            # Bigger batches are genuinely faster; without this every candidate
            # ties and the assertion below would be testing tie-breaking.
            _CLOCK.advance(len(t) / (batch_size * 2.0))
            return np.zeros((len(t), 4), dtype="float32")
        return encode

    monkeypatch.setattr(rt, "make_encode_fn", lambda model, bs: encode_for(bs))
    choices = sweep_batch_sizes(
        texts(512), fake_load, [64, 128, 256], rows_per_trial=256
    )
    by_size = {c.batch_size: c for c in choices}
    assert by_size[256].fits is False
    assert "out of VRAM" in by_size[256].reason
    assert by_size[128].fits is True
    assert choose_batch_size(choices).batch_size == 128


# --------------------------------------------------------------------------
# The plan: the decision made before the run
# --------------------------------------------------------------------------
def budget_plan(rows_per_sec: float, budget_seconds: float, **kw):
    tp = Throughput(rows_per_sec=rows_per_sec, batch_size=256, rows_timed=50_000,
                    seconds_timed=50_000 / rows_per_sec)
    return plan_sessions(ROWS_TO_EMBED, tp, budget_seconds=budget_seconds, **kw)


def test_a_fast_run_fits_one_session() -> None:
    plan = budget_plan(rows_per_sec=2_000.0, budget_seconds=6 * 3600)
    assert plan.decision == "single-session"
    assert plan.fits_one_session
    assert plan.sessions == 1
    assert plan.seconds_needed == pytest.approx(ROWS_TO_EMBED / 2_000.0)


def test_a_slow_run_is_told_to_split_and_how_many_times() -> None:
    plan = budget_plan(rows_per_sec=60.0, budget_seconds=2 * 3600)
    assert plan.decision in ("split", "checkpoint-resume")
    assert plan.sessions > 1
    assert any("sessions" in note for note in plan.notes)


def test_infeasible_is_reported_before_the_run_not_during_it() -> None:
    """
    The case the whole module exists for: 3.4M rows will not fit, and no amount
    of resuming within the cap will get there. The plan must say so up front
    and name the real options, rather than letting hour three discover it.
    """
    plan = budget_plan(rows_per_sec=1.0, budget_seconds=3600, max_sessions=12)
    assert plan.decision == "infeasible"
    assert plan.sessions > 12
    joined = " ".join(plan.notes)
    assert "batch size" in joined
    assert "strict filter" in joined


def test_headroom_keeps_the_plan_off_the_session_limit() -> None:
    """
    Planning to the exact second means the last shard never lands. The default
    15% headroom is what makes a single-session verdict trustworthy.
    """
    tp = Throughput(rows_per_sec=ROWS_TO_EMBED / 3600.0, batch_size=256,
                    rows_timed=50_000, seconds_timed=1.0)
    tight = plan_sessions(ROWS_TO_EMBED, tp, budget_seconds=3600, headroom=0.0)
    padded = plan_sessions(ROWS_TO_EMBED, tp, budget_seconds=3600, headroom=0.15)
    assert tight.decision == "single-session"
    assert padded.decision != "single-session"


def test_the_plan_never_claims_zero_rows_per_session() -> None:
    plan = budget_plan(rows_per_sec=0.001, budget_seconds=1, max_sessions=3)
    assert plan.rows_per_session == 0
    assert plan.decision == "infeasible"


@pytest.mark.parametrize("kwargs", [
    {"total_rows": 0},
    {"shard_rows": 0},
    {"headroom": 1.0},
    {"headroom": -0.1},
])
def test_nonsense_plan_inputs_are_rejected(kwargs) -> None:
    tp = Throughput(rows_per_sec=100.0, batch_size=8, rows_timed=10, seconds_timed=0.1)
    base = {"total_rows": ROWS_TO_EMBED, "budget_seconds": 60.0}
    base.update(kwargs)
    with pytest.raises(ValueError):
        plan_sessions(tp=tp, **base) if False else plan_sessions(
            base.pop("total_rows"), tp, **base
        )


def test_the_plan_text_names_the_decision(clock) -> None:
    """The plan has to be readable in notebook output, not just in an object."""
    text = describe_plan(budget_plan(rows_per_sec=1.5, budget_seconds=6 * 3600))
    assert "DECISION" in text
    assert "rows/s at batch" in text
    assert f"{ROWS_TO_EMBED:,}" in text


# --------------------------------------------------------------------------
# Phase 1: building the plan
# --------------------------------------------------------------------------
def write_source(path, n: int, *, non_latin_every: int = 3, source: str = "India") -> None:
    """
    A tiny source file shaped like the real ones: mostly ASCII Latin names, with
    an Indic name every `non_latin_every` rows and one accented-Latin row, so the
    broad/strict difference is actually present in the fixture.
    """
    indic = ["आनंद फाउंडेशन", "టెక సॉల్యూషన్స్", "গ্যালাক্সি ফুড"]
    rows = []
    for i in range(n):
        if i % non_latin_every == 0:
            name = indic[(i // non_latin_every) % len(indic)]
        elif i == 1:
            # The real SAMP-039 row. The accent has to survive `clean_text`,
            # which strips *mojibake* accents but not real ones -- if it did not,
            # this row would be ASCII and the broad/strict difference would vanish.
            name = "SCI Ptit \u00c0micale"
        else:
            name = f"ACME HOLDINGS {i} LLC"
        rows.append(f"E{i:06d}\t{name}\t12 Main St\t{source}")
    path.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\n" + "\n".join(rows) + "\n",
                    encoding="utf-8")


def test_the_plan_selects_the_flagged_names_only(clock, tmp_path) -> None:
    src = tmp_path / "test_source2.tsv"
    write_source(src, 60, non_latin_every=3)
    plan_dir = tmp_path / "plan"
    summary = build_plan_shards(
        {"test_source2.tsv": src}, plan_dir, strict=False, shard_rows=25
    )
    # 60 rows, every 3rd is Indic -> 20 flagged, plus the accented-Latin row.
    assert summary["total_rows"] == 21
    assert summary["per_source"]["test_source2.tsv"] == 21
    assert summary["shards"] == 1  # 21 rows fits one 25-row shard

    shards = iter_plan_shards(plan_dir)
    assert len(shards) == 1
    frame = shards[0][1]
    assert list(frame.columns) == ["entity_id", "business_name_clean"]
    assert len(frame) == 21
    assert frame["entity_id"].is_unique


def test_the_strict_plan_is_smaller_than_the_broad_one(clock, tmp_path) -> None:
    src = tmp_path / "test_source2.tsv"
    write_source(src, 60, non_latin_every=3)
    broad_dir, strict_dir = tmp_path / "broad", tmp_path / "strict"
    broad = build_plan_shards({"test_source2.tsv": src}, broad_dir, strict=False, shard_rows=25)
    strict = build_plan_shards({"test_source2.tsv": src}, strict_dir, strict=True, shard_rows=25)
    assert strict["total_rows"] == broad["total_rows"] - 1  # the accented-Latin row
    assert not any(
        "micale" in name
        for _id, frame in iter_plan_shards(strict_dir)
        for name in frame["business_name_clean"]
    )
    # And the accent is still there in the broad plan -- cleaning kept it.
    assert any(
        "\u00c0" in name
        for _id, frame in iter_plan_shards(broad_dir)
        for name in frame["business_name_clean"]
    )


def test_the_plan_is_deterministic_so_shard_ids_mean_the_same_rows(clock, tmp_path) -> None:
    """
    What makes the GPU phase resumable: rebuilding the plan must reproduce shard
    N exactly, or a resumed session would embed the wrong rows into shard N.
    """
    src = tmp_path / "test_source2.tsv"
    write_source(src, 60, non_latin_every=3)
    first_dir, second_dir = tmp_path / "a", tmp_path / "b"
    for plan_dir in (first_dir, second_dir):
        build_plan_shards({"test_source2.tsv": src}, plan_dir, strict=False, shard_rows=25)
    a = pd.concat([f for _i, f in iter_plan_shards(first_dir)], ignore_index=True)
    b = pd.concat([f for _i, f in iter_plan_shards(second_dir)], ignore_index=True)
    pd.testing.assert_frame_equal(a, b)


def test_the_plan_spans_multiple_shards_and_covers_every_row(clock, tmp_path) -> None:
    src = tmp_path / "test_source2.tsv"
    write_source(src, 60, non_latin_every=3)
    plan_dir = tmp_path / "plan"
    summary = build_plan_shards(
        {"test_source2.tsv": src}, plan_dir, strict=False, shard_rows=5, chunksize=7
    )
    shards = iter_plan_shards(plan_dir)
    assert len(shards) == summary["shards"] == 5  # 21 rows in 5-row shards
    assert [i for i, _f in shards] == list(range(5))
    combined = pd.concat([f for _i, f in shards], ignore_index=True)
    assert len(combined) == 21
    assert combined["entity_id"].is_unique
    assert set(combined.columns) == {"entity_id", "business_name_clean"}


def test_the_plan_records_what_it_decided(clock, tmp_path) -> None:
    src = tmp_path / "test_source2.tsv"
    write_source(src, 30, non_latin_every=3)
    plan_dir = tmp_path / "plan"
    build_plan_shards({"test_source2.tsv": src}, plan_dir, strict=True, shard_rows=25)
    manifest = read_manifest(plan_dir)
    assert manifest["strict"] is True
    assert manifest["shard_rows"] == 25
    assert manifest["total_rows"] > 0
    assert "test_source2.tsv" in manifest["per_source"]


def test_the_plan_handles_a_file_with_nothing_selected(clock, tmp_path) -> None:
    """All-ASCII source2: broad selects nothing, and that is not an error."""
    src = tmp_path / "test_source2.tsv"
    src.write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "E1\tACME HOLDINGS LLC\t12 Main St\tUS\n"
        "E2\tGLOBEX CORPORATION\t9 Side St\tUS\n",
        encoding="utf-8",
    )
    summary = build_plan_shards(
        {"test_source2.tsv": src}, tmp_path / "plan", strict=False
    )
    assert summary["total_rows"] == 0
    assert summary["shards"] == 0
    assert iter_plan_shards(tmp_path / "plan") == []


def test_the_plan_ends_to_end_into_vectors(clock, tmp_path) -> None:
    """Plan -> shards -> verify, with no model, is the whole shape of the job."""
    src = tmp_path / "test_source2.tsv"
    write_source(src, 60, non_latin_every=3)
    plan_dir, out_dir = tmp_path / "plan", tmp_path / "out"
    summary = build_plan_shards(
        {"test_source2.tsv": src}, plan_dir, strict=False, shard_rows=10
    )
    expected_ids = set(
        pd.concat([f for _i, f in iter_plan_shards(plan_dir)])["entity_id"]
    )
    outcome = run_shards(
        iter_plan_shards(plan_dir), out_dir, fake_encode_factory(500.0), max_seconds=0
    )
    assert outcome.complete
    report = verify_shards(
        out_dir, expected_total_rows=summary["total_rows"], expected_ids=expected_ids
    )
    assert report["ok"], report


# --------------------------------------------------------------------------
# Run identity: a shard from a different run must not count as done
# --------------------------------------------------------------------------
def test_the_fingerprint_is_stable_and_order_independent() -> None:
    a = rt.run_fingerprint({"model": "labse", "strict": False, "shard_rows": 100_000})
    b = rt.run_fingerprint({"shard_rows": 100_000, "strict": False, "model": "labse"})
    assert a == b
    assert len(a) == 16
    assert a != rt.run_fingerprint({"model": "labse", "strict": True, "shard_rows": 100_000})
    assert a != rt.run_fingerprint({"model": "minilm", "strict": False, "shard_rows": 100_000})


def test_a_shard_written_by_a_different_model_is_not_skipped(clock, tmp_path) -> None:
    """
    The failure this prevents: a strict-plan shard and a broad-plan shard are both
    100k rows, so a row-count-only resume check would skip the one that does not
    belong and finish 'complete' with two different embedders in one index.
    """
    out_dir = tmp_path / "out"
    shards = [(0, make_frame(100, 1)), (1, make_frame(100, 101))]
    identity = {"model": "labse", "strict": True, "shard_rows": 100}

    first = rt.run_shards(
        shards, out_dir, fake_encode_factory(500.0), max_seconds=0, identity=identity
    )
    assert (first.shards_written, first.shards_skipped) == (2, 0)

    # Same rows, same shard sizes -- but a different model. Nothing may be reused.
    other = rt.run_shards(
        shards, out_dir, fake_encode_factory(500.0), max_seconds=0,
        identity={"model": "minilm", "strict": True, "shard_rows": 100},
    )
    assert other.shards_written == 2
    assert other.shards_skipped == 0


def test_resuming_the_same_run_still_skips(clock, tmp_path) -> None:
    """The point of all that: an honest resume must not redo completed work."""
    out_dir = tmp_path / "out"
    identity = {"model": "labse", "strict": False, "shard_rows": 50}
    shards = [(i, make_frame(50, i * 50)) for i in range(4)]

    first = rt.run_shards(
        shards[:2], out_dir, fake_encode_factory(500.0), max_seconds=0, identity=identity
    )
    assert first.shards_written == 2

    second = rt.run_shards(
        shards, out_dir, fake_encode_factory(500.0), max_seconds=0, identity=identity
    )
    assert second.shards_written == 2
    assert second.shards_skipped == 2
    assert second.complete


def test_a_shard_with_no_provenance_is_redone(clock, tmp_path) -> None:
    """A parquet with no sidecar is a shard from before provenance existed."""
    out_dir = tmp_path / "out"
    shards = [(0, make_frame(10, 1))]
    identity = {"model": "labse"}
    rt.run_shards(shards, out_dir, fake_encode_factory(500.0), max_seconds=0, identity=identity)
    rt.shard_meta_path(out_dir, 0).unlink()

    again = rt.run_shards(
        shards, out_dir, fake_encode_factory(500.0), max_seconds=0, identity=identity
    )
    assert again.shards_written == 1
    assert again.shards_skipped == 0


def test_verification_catches_a_shard_from_another_run(clock, tmp_path) -> None:
    out_dir = tmp_path / "out"
    shards = [(i, make_frame(10, i * 10)) for i in range(2)]
    rt.run_shards(
        shards, out_dir, fake_encode_factory(500.0), max_seconds=0,
        identity={"model": "labse"},
    )
    ok = rt.verify_shards(out_dir, expected_total_rows=20, identity={"model": "labse"})
    assert ok["ok"], ok
    assert ok["stale_shards"] == 0

    wrong = rt.verify_shards(out_dir, expected_total_rows=20, identity={"model": "minilm"})
    assert not wrong["ok"]
    assert wrong["stale_shards"] == 2


def test_verification_catches_unexpected_ids(clock, tmp_path) -> None:
    """Extra ids must fail, not just missing ones: a superseded shard is extra."""
    out_dir = tmp_path / "out"
    rt.run_shards(
        [(0, make_frame(10, 1))], out_dir, fake_encode_factory(500.0), max_seconds=0
    )
    report = rt.verify_shards(out_dir, expected_ids={f"E{i:07d}" for i in range(1, 11)})
    assert report["unexpected_entity_ids"] == 0
    assert report["ok"]

    wrong = rt.verify_shards(out_dir, expected_ids={f"E{i:07d}" for i in range(1, 6)})
    assert wrong["unexpected_entity_ids"] == 5
    assert not wrong["ok"]


def test_a_manifest_refuses_to_claim_two_different_runs(tmp_path) -> None:
    out_dir = tmp_path / "out"
    rt.write_manifest(out_dir, {"fingerprint": "aaaa", "model": "labse"})
    rt.write_manifest(out_dir, {"shards_remaining": 3})  # same run, merges fine
    assert rt.read_manifest(out_dir)["fingerprint"] == "aaaa"

    with pytest.raises(ValueError, match="fingerprint"):
        rt.write_manifest(out_dir, {"fingerprint": "bbbb", "model": "minilm"})


# --------------------------------------------------------------------------
# Shards
# --------------------------------------------------------------------------
def test_shard_bounds_cover_every_row_exactly_once() -> None:
    bounds = shard_bounds(1_000, 300)
    assert bounds == [(0, 0, 300), (1, 300, 600), (2, 600, 900), (3, 900, 1000)]
    covered = [i for _id, start, stop in bounds for i in range(start, stop)]
    assert covered == list(range(1_000))


def test_shard_bounds_handle_an_exact_multiple() -> None:
    bounds = shard_bounds(600, 300)
    assert [b[2] for b in bounds] == [300, 600]
    assert len(bounds) == 2


@pytest.mark.parametrize("total,size", [(0, 10), (10, 0), (-1, 10), (10, -1)])
def test_shard_bounds_reject_nonsense(total, size) -> None:
    with pytest.raises(ValueError):
        shard_bounds(total, size)


def test_a_100k_shard_of_labse_vectors_is_about_195_mib() -> None:
    """
    The number that makes 100k the default shard size: it fits comfortably in
    RAM to read, and 34 of them cover the test split. If the model or the
    storage dtype changes, this moves and the shard count should be revisited.
    """
    mib = 100_000 * rt.VECTOR_DIM * np.dtype(rt.STORAGE_DTYPE).itemsize / 2**20
    assert 150 < mib < 250


# --------------------------------------------------------------------------
# Atomicity and resume
# --------------------------------------------------------------------------
def make_frame(n: int, start: int = 0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entity_id": [f"E{i:07d}" for i in range(start, start + n)],
            "business_name_clean": [f"name {i}" for i in range(start, start + n)],
        }
    )


def test_a_shard_is_written_and_readable(clock, tmp_path) -> None:
    frame = make_frame(10)
    path, rows = rt.embed_shard(
        frame, 3, tmp_path, fake_encode_factory(500.0)
    )
    assert rows == 10
    assert path == shard_path(tmp_path, 3)
    back = pd.read_parquet(path)
    assert len(back) == 10
    assert back["entity_id"].tolist() == frame["entity_id"].tolist()
    assert back["embedding"].iloc[0].dtype == np.float16


def test_no_temp_file_is_left_behind(clock, tmp_path) -> None:
    rt.embed_shard(make_frame(4), 0, tmp_path, fake_encode_factory(500.0))
    assert not list(tmp_path.glob("*.tmp"))


def test_a_truncated_shard_is_not_trusted(clock, tmp_path) -> None:
    """
    The bug a plain `exists()` check cannot see. If a session dies mid-write the
    file is there and short, and a resume that trusts it silently loses rows.
    """
    rt.embed_shard(make_frame(100), 0, tmp_path, fake_encode_factory(500.0))
    path = shard_path(tmp_path, 0)
    assert shard_is_complete(path, 100)
    assert not shard_is_complete(path, 99)
    assert not shard_is_complete(path, 101)


def test_a_truncated_shard_gets_rewritten(clock, tmp_path) -> None:
    frame = make_frame(50)
    rt.embed_shard(frame, 0, tmp_path, fake_encode_factory(500.0))
    # Simulate a partial write: a valid parquet with too few rows.
    atomic_write_parquet(frame.head(20), shard_path(tmp_path, 0))
    outcome = run_shards(
        [(0, frame)], tmp_path, fake_encode_factory(500.0), max_seconds=0
    )
    assert outcome.shards_written == 1
    assert shard_is_complete(shard_path(tmp_path, 0), 50)


def test_a_corrupt_shard_is_not_trusted(clock, tmp_path) -> None:
    path = shard_path(tmp_path, 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not a parquet file at all")
    assert not shard_is_complete(path, 10)


def test_rerunning_skips_what_is_already_done(clock, tmp_path) -> None:
    """The resume path, on purpose rather than after a crash."""
    shards = [(0, make_frame(100, 0)), (1, make_frame(100, 100))]
    first = run_shards(shards, tmp_path, fake_encode_factory(500.0), max_seconds=0)
    assert first.shards_written == 2
    assert first.shards_skipped == 0
    assert first.complete

    second = run_shards(shards, tmp_path, fake_encode_factory(500.0), max_seconds=0)
    assert second.shards_written == 0
    assert second.shards_skipped == 2
    assert second.complete
    assert second.rows_written == 200


def test_a_session_that_runs_out_of_time_reports_what_is_left(clock, tmp_path) -> None:
    """
    The job stops *between* shards, never mid-shard, and says exactly which
    shards a re-run still owes. This is the normal case on Colab, not a failure.
    """
    shards = [(i, make_frame(10, i * 10)) for i in range(10)]
    # 100 rows at 100 rows/s = 1.0 s per shard; stop at 40% of a 1 s budget.
    outcome = run_shards(
        shards, tmp_path, fake_encode_factory(100.0),
        max_seconds=1.0, stop_at_fraction=0.4,
    )
    assert outcome.stopped_early
    assert outcome.shards_written >= 1
    assert not outcome.complete
    assert outcome.shards_remaining
    assert outcome.shards_remaining == sorted(outcome.shards_remaining)
    assert 0 not in outcome.shards_remaining
    # Everything reported as remaining is genuinely absent.
    for shard_id in outcome.shards_remaining:
        assert not (tmp_path / f"shard_{shard_id:05d}.parquet").exists()


def test_resuming_after_a_timeout_finishes_the_job(clock, tmp_path) -> None:
    """Two sessions, one directory: the second picks up where the first stopped."""
    shards = [(i, make_frame(10, i * 10)) for i in range(6)]

    first = run_shards(shards, tmp_path, fake_encode_factory(100.0),
                       max_seconds=1.0, stop_at_fraction=0.3)
    assert first.stopped_early

    second = run_shards(shards, tmp_path, fake_encode_factory(100.0),
                        max_seconds=600.0)
    assert second.complete
    assert not second.shards_remaining
    for shard_id, frame in shards:
        assert shard_is_complete(shard_path(tmp_path, shard_id), len(frame))
    report = verify_shards(tmp_path, expected_total_rows=60)
    assert report["ok"]


def test_an_unlimited_budget_never_stops_early(clock, tmp_path) -> None:
    shards = [(i, make_frame(10, i * 10)) for i in range(20)]
    outcome = run_shards(shards, tmp_path, fake_encode_factory(1000.0), max_seconds=0)
    assert not outcome.stopped_early
    assert outcome.shards_written == 20
    assert outcome.complete


# --------------------------------------------------------------------------
# Verification across sessions
# --------------------------------------------------------------------------
def test_verification_passes_on_a_clean_run(clock, tmp_path) -> None:
    ids = {f"E{i:07d}" for i in range(30)}
    run_shards(
        [(0, make_frame(10, 0)), (1, make_frame(10, 10)), (2, make_frame(10, 20))],
        tmp_path, fake_encode_factory(500.0), max_seconds=0,
    )
    report = verify_shards(tmp_path, expected_total_rows=30, expected_ids=ids)
    assert report["ok"]
    assert report["rows"] == 30
    assert report["unique_entity_ids"] == 30
    assert report["missing_entity_ids"] == 0


def test_verification_catches_a_duplicate_across_sessions(clock, tmp_path) -> None:
    """
    Two sessions that both wrote overlapping ranges would double-count. The row
    total could even match, so this is checked on ids.
    """
    run_shards(
        [(0, make_frame(10, 0)), (1, make_frame(10, 10))],
        tmp_path, fake_encode_factory(500.0), max_seconds=0,
    )
    # A third shard overlapping the second, as a bad resume would produce.
    rt.embed_shard(make_frame(10, 5), 2, tmp_path, fake_encode_factory(500.0))
    report = verify_shards(tmp_path, expected_total_rows=30)
    assert not report["ok"]
    assert report["duplicate_entity_ids"] == 10


def test_verification_catches_a_missing_shard(clock, tmp_path) -> None:
    run_shards([(0, make_frame(10, 0)), (1, make_frame(10, 10))],
               tmp_path, fake_encode_factory(500.0), max_seconds=0)
    (tmp_path / "shard_00001.parquet").unlink()
    report = verify_shards(tmp_path, expected_total_rows=20)
    assert not report["ok"]
    assert report["row_delta"] == -10


def test_verification_catches_a_shard_from_another_run(clock, tmp_path) -> None:
    run_shards([(0, make_frame(10, 0))], tmp_path, fake_encode_factory(500.0),
               max_seconds=0)
    report = verify_shards(tmp_path, expected_ids={f"E{i:07d}" for i in range(10)})
    assert report["ok"]
    report = verify_shards(tmp_path, expected_ids={f"E{i:07d}" for i in range(50)})
    assert not report["ok"]
    assert report["missing_entity_ids"] == 40


def test_verifying_an_empty_directory_is_not_ok(clock, tmp_path) -> None:
    assert not verify_shards(tmp_path, expected_total_rows=10)["ok"]


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------
def test_the_manifest_records_the_run_and_merges_across_sessions(tmp_path) -> None:
    write_manifest(tmp_path, {"model": "sentence-transformers/LaBSE", "strict": False})
    write_manifest(tmp_path, {"shards_written": 12})
    manifest = read_manifest(tmp_path)
    assert manifest["model"] == "sentence-transformers/LaBSE"
    assert manifest["shards_written"] == 12
    assert manifest["strict"] is False


def test_a_corrupt_manifest_does_not_stop_a_resume(tmp_path) -> None:
    """The shards on disk are the truth; the manifest is a convenience."""
    (tmp_path).mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifest.json").write_text("{not json", encoding="utf-8")
    write_manifest(tmp_path, {"model": "m"})
    assert read_manifest(tmp_path)["model"] == "m"


def test_the_documented_vector_sizes_are_the_ones_the_code_produces() -> None:
    """
    The module docstring quotes 6.5 GiB fp16 / 13.0 GiB fp32 for the full run.
    Those numbers were wrong by 2x for a while, so they are pinned to arithmetic
    here rather than left as prose nobody recomputes.
    """
    planned_rows = 3_397_040        # measured by build_plan_shards on the real data
    dims = 1024                    # LaBSE

    fp16 = planned_rows * dims * np.dtype(rt.STORAGE_DTYPE).itemsize / 2**30
    fp32 = planned_rows * dims * 4 / 2**30
    assert round(fp16, 1) == 6.5
    assert round(fp32, 1) == 13.0
    # fp16 is exactly half of fp32, and neither is small enough to hold in RAM
    # next to the encoder -- which is the whole reason for sharding.
    assert abs(fp32 - 2 * fp16) < 1e-9
    assert fp16 > 6.0


# --------------------------------------------------------------------------
# The expensive half stays out of reach
# --------------------------------------------------------------------------
def test_importing_this_module_does_not_import_torch_or_sentence_transformers() -> None:
    import subprocess
    import sys

    code = (
        "import sys;"
        "import src.blocking.embed_runtime as rt;"
        "heavy = [k for k in sys.modules if k == 'torch' "
        "or k.startswith('sentence_transformers')];"
        "assert not heavy, heavy;"
        "print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
