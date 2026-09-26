"""
Tests for `src.blocking.embed_names`.

The scope policy is three lines of code and one boolean, and it is the decision
that decides whether Phase 3b costs an afternoon or a weekend, so it is pinned
from both ends: source1 is kept whole, sources 2 and 3 are kept only where the
flag is true, and no row of the sample is silently lost or invented.

Nothing here loads a model. The filtering functions are pure pandas, and that is
tested for real rather than asserted: if the import were at module scope, this
suite would need torch and a network to check a row count, so
`test_importing_the_module_does_not_pull_in_sentence_transformers` runs the
import in a subprocess and asserts the cheap half stays cheap.

Expected counts are written as literals rather than recomputed with the
functions under test. On a sample this small the balance between scripts is a
property of which rows were copied, not of the population, so these numbers
mean "this is what the policy does to these rows". The population rates are the
job of `python -m src.blocking.embed_names` against real data.
"""
from __future__ import annotations

import inspect
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

import src.blocking.embed_names as embed_names_module
from src.blocking.embed_names import (
    ALL_SOURCES,
    DEFAULT_MODEL,
    EMBED_WHOLE,
    EXPECTED_TOTAL_TEST_RANGE,
    FILTER_BY_FLAG,
    MEASURED_STRICT_PCT,
    REJECTED_MODEL,
    REPORTED_NON_ASCII_PCT,
    REPORTED_ROWS,
    TEST_SOURCES,
    build_parser,
    embed_names,
    expectation_basis,
    expected_embed_pct,
    expected_embed_rows,
    expected_non_latin_rows,
    expected_total,
    plan_by_source,
    report_expected_vs_actual,
    select_names_to_embed,
    select_rows_to_embed,
)
from src.preprocessing.clean_text import clean_dataframe
from src.preprocessing.multilingual import (
    INDIA_SCRIPT_FLAG,
    NON_LATIN_FLAG,
    add_multilingual_columns,
)
from tests.multilingual_sample import SAMPLE_ORIGINS, SAMPLE_ROWS, sample_source_df


def _processed() -> pd.DataFrame:
    """The 39 sample rows, cleaned and flagged, with no source_file assigned yet."""
    return add_multilingual_columns(clean_dataframe(sample_source_df.copy()))


# --------------------------------------------------------------------------
# Two views of the same 39 rows
# --------------------------------------------------------------------------
@pytest.fixture
def sample_by_source() -> dict[str, pd.DataFrame]:
    """
    origin file -> its rows, as the pipeline would produce them.

    These are the real origins from `SAMPLE_ORIGINS`, and they are lopsided:
    35 rows from test_source2, 3 from test_source1, 1 from test_source3. The
    Indic rows all came out of source2, and source3 contributes a single
    unflagged row. That is fine for the preprocessing tests, which care about
    the names, and useless for the policy tests -- so those use `policy_frames`.
    """
    frame = _processed()
    frame["source_file"] = [SAMPLE_ORIGINS[row["entity_id"]] + ".tsv" for row in SAMPLE_ROWS]
    return {
        name: group.reset_index(drop=True)
        for name, group in frame.groupby("source_file", sort=True)
    }


@pytest.fixture
def policy_frames() -> dict[str, pd.DataFrame]:
    """
    The same 39 real names, with the origin file *assigned* to balance the split.

    The policy is a function of (file, flag), so testing it needs every file to
    have both flagged and unflagged rows. The real origins cannot supply that --
    every Indic row in the sample happens to come from test_source2 -- so the
    row contents stay real and only the file label is chosen. Nothing here
    claims these are the files these names came from.

    Slices are taken in flag order, so the 30 Indic rows come first and
    `SAMP-039` ("SCI Ptit Amicale", accented Latin) lands last: the one row the
    broad flag keeps and the strict flag drops.
    """
    processed = _processed()
    flagged = processed[processed[NON_LATIN_FLAG].astype(bool)].reset_index(drop=True)
    plain = processed[~processed[NON_LATIN_FLAG].astype(bool)].reset_index(drop=True)
    assert len(flagged) == 31 and len(plain) == 8

    frames = {}
    for name, flag_slice, plain_slice in (
        ("test_source1.tsv", slice(0, 11), slice(0, 3)),
        ("test_source2.tsv", slice(11, 21), slice(3, 6)),
        ("test_source3.tsv", slice(21, 31), slice(6, 8)),
    ):
        parts = [flagged[flag_slice], plain[plain_slice]]
        frames[name] = (
            pd.concat(parts, ignore_index=True)
            .assign(source_file=name)
            .reset_index(drop=True)
        )
    return frames


# What the policy does to `policy_frames`, by hand:
#   source1: 11 flagged + 3 plain -> all 14 kept, whole-file policy
#   source2: 10 flagged + 3 plain -> 10 kept
#   source3: 10 flagged + 2 plain ->  9 kept, SAMP-039 being the accented-Latin one
EXPECTED_ROWS = {"test_source1.tsv": 14, "test_source2.tsv": 13, "test_source3.tsv": 12}
EXPECTED_FLAGGED = {"test_source1.tsv": 11, "test_source2.tsv": 10, "test_source3.tsv": 10}
EXPECTED_SELECTED = {"test_source1.tsv": 14, "test_source2.tsv": 10, "test_source3.tsv": 10}
EXPECTED_SELECTED_STRICT = {"test_source1.tsv": 14, "test_source2.tsv": 10, "test_source3.tsv": 9}


# --------------------------------------------------------------------------
# The policy
# --------------------------------------------------------------------------
def test_every_source_has_both_flagged_and_unflagged_rows(policy_frames) -> None:
    """A guard on the fixture: a vacuous test above would look identical."""
    for name, frame in policy_frames.items():
        flagged = int(frame[NON_LATIN_FLAG].sum())
        assert 0 < flagged < len(frame), name
        assert len(frame) == EXPECTED_ROWS[name], name
        assert flagged == EXPECTED_FLAGGED[name], name


def test_source1_is_embedded_whole_whatever_the_flag_says(policy_frames) -> None:
    """
    source1 is the anchor side: recall is measured source1 -> candidate, so a
    source1 row that is never embedded can never be recalled. Its real flag rate
    is 2.35% on the test file, so "filter it like the others" would quietly
    discard the anchor side of the whole evaluation.
    """
    s1 = policy_frames["test_source1.tsv"]
    assert int(s1[NON_LATIN_FLAG].sum()) == 11
    assert len(select_rows_to_embed(s1)) == 14


def test_source1_rows_are_kept_even_when_the_flag_is_false(policy_frames) -> None:
    s1 = policy_frames["test_source1.tsv"]
    unflagged = set(s1[~s1[NON_LATIN_FLAG].astype(bool)]["entity_id"])
    assert len(unflagged) == 3
    assert unflagged <= set(select_rows_to_embed(s1)["entity_id"])


def test_sources_two_and_three_are_filtered_by_the_flag(policy_frames) -> None:
    for name in ("test_source2.tsv", "test_source3.tsv"):
        frame = policy_frames[name]
        selected = select_rows_to_embed(frame, name)
        assert len(selected) == EXPECTED_SELECTED[name], name
        # Every selected row is a flagged row: no unflagged row sneaks through.
        assert selected[NON_LATIN_FLAG].all(), name
        # And no flagged row is dropped.
        flagged = set(frame[frame[NON_LATIN_FLAG].astype(bool)]["entity_id"])
        assert flagged == set(selected["entity_id"]), name


def test_the_unflagged_rows_dropped_are_exactly_the_ascii_ones(policy_frames) -> None:
    dropped = []
    for name in ("test_source2.tsv", "test_source3.tsv"):
        frame = policy_frames[name]
        kept = set(select_rows_to_embed(frame, name)["entity_id"])
        dropped += frame[~frame["entity_id"].isin(kept)]["business_name"].tolist()
    assert len(dropped) == 5
    assert all(name.isascii() for name in dropped)


def test_strict_mode_drops_only_the_accented_latin_row(policy_frames) -> None:
    """
    The strict reading is the cheaper run: same policy, narrower filter. The one
    row it drops here is "SCI Ptit Amicale", which is exactly the class the broad
    flag and the strict flag are documented to disagree about.
    """
    for name, expected in EXPECTED_SELECTED_STRICT.items():
        assert len(select_rows_to_embed(policy_frames[name], name, strict=True)) == expected, name

    s3 = policy_frames["test_source3.tsv"]
    broad = set(select_rows_to_embed(s3, "test_source3.tsv")["entity_id"])
    strict = set(select_rows_to_embed(s3, "test_source3.tsv", strict=True)["entity_id"])
    # The only row the two readings disagree about, in either direction.
    disputed = s3[s3["entity_id"].isin(broad - strict)]
    assert len(broad - strict) == 1
    assert disputed["entity_id"].tolist() == ["SAMP-039"]
    assert int(disputed[INDIA_SCRIPT_FLAG].sum()) == 0
    assert int(disputed[NON_LATIN_FLAG].sum()) == 1


def test_strict_is_a_subset_of_broad_for_every_source(policy_frames) -> None:
    for name, frame in policy_frames.items():
        broad = set(select_rows_to_embed(frame, name)["entity_id"])
        strict = set(select_rows_to_embed(frame, name, strict=True)["entity_id"])
        assert strict <= broad, name


def test_the_real_origins_behave_the_same_way(sample_by_source) -> None:
    """
    The balanced fixture assigns origins; this one uses the real ones, where the
    3 unflagged test_source1 rows and the 1 unflagged test_source3 row are all
    the sample has to say about those files.
    """
    # 3 rows, none flagged, all kept because the file is embedded whole.
    s1 = sample_by_source["test_source1.tsv"]
    assert len(s1) == 3 and int(s1[NON_LATIN_FLAG].sum()) == 0
    assert len(select_rows_to_embed(s1)) == 3

    # 35 rows, 31 flagged, 30 of them Indic.
    s2 = sample_by_source["test_source2.tsv"]
    assert len(s2) == 35
    assert len(select_rows_to_embed(s2, "test_source2.tsv")) == 31
    assert len(select_rows_to_embed(s2, "test_source2.tsv", strict=True)) == 30

    # 1 row, not flagged, so source3 contributes nothing to this sample.
    s3 = sample_by_source["test_source3.tsv"]
    assert len(s3) == 1 and int(s3[NON_LATIN_FLAG].sum()) == 0
    assert len(select_rows_to_embed(s3, "test_source3.tsv")) == 0


def test_the_scope_covers_every_source_file_exactly_once() -> None:
    for name in ALL_SOURCES:
        assert (name in EMBED_WHOLE) ^ (name in FILTER_BY_FLAG), name
    assert not set(EMBED_WHOLE) & set(FILTER_BY_FLAG)
    assert set(EMBED_WHOLE) | set(FILTER_BY_FLAG) == set(ALL_SOURCES)


def test_selection_keeps_rows_and_order_without_touching_the_input(policy_frames) -> None:
    """
    A filter that duplicated or reordered rows would still look right on a rate
    check, and a caller writing back in place would corrupt the frame.
    """
    for name, frame in policy_frames.items():
        before = frame.copy(deep=True)
        selected = select_rows_to_embed(frame, name)
        pd.testing.assert_frame_equal(frame, before)

        assert selected.index.isin(frame.index).all(), name
        assert list(selected.index) == sorted(selected.index), name
        assert not selected.index.duplicated().any(), name
        assert list(selected.columns) == list(frame.columns), name


def test_selection_is_a_partition(policy_frames) -> None:
    """selected + dropped == input, with no row in both and none in neither."""
    for name, frame in policy_frames.items():
        kept = set(select_rows_to_embed(frame, name)["entity_id"])
        dropped = set(frame["entity_id"]) - kept
        assert kept & dropped == set()
        assert kept | dropped == set(frame["entity_id"])
        assert len(kept) + len(dropped) == len(frame)


def test_an_empty_frame_selects_nothing_and_does_not_raise() -> None:
    empty = pd.DataFrame(
        columns=["entity_id", "business_name_clean", NON_LATIN_FLAG, INDIA_SCRIPT_FLAG,
                 "source_file"]
    )
    assert len(select_rows_to_embed(empty, "test_source2.tsv")) == 0
    assert len(select_names_to_embed(empty, "test_source2.tsv")) == 0


# --------------------------------------------------------------------------
# Failure modes that would silently embed the wrong rows
# --------------------------------------------------------------------------
def test_a_missing_flag_column_raises_rather_than_embedding_everything() -> None:
    """The dangerous failure is a KeyError swallowed into "no filter applied"."""
    frame = pd.DataFrame({"business_name_clean": ["श्री बालाजी"], "source_file": ["x.tsv"]})
    with pytest.raises(KeyError, match=NON_LATIN_FLAG):
        select_rows_to_embed(frame, "x.tsv")


def test_a_missing_strict_flag_column_names_the_strict_column() -> None:
    frame = pd.DataFrame(
        {"business_name_clean": ["x"], "source_file": ["test_source2.tsv"],
         NON_LATIN_FLAG: [True]}
    )
    with pytest.raises(KeyError, match=INDIA_SCRIPT_FLAG):
        select_rows_to_embed(frame, "test_source2.tsv", strict=True)


def test_a_frame_with_no_way_to_identify_its_source_raises(policy_frames) -> None:
    """
    With no source_file column and nothing passed, the policy cannot be applied --
    and defaulting to "embed it" would pick the expensive answer silently.
    """
    frame = policy_frames["test_source2.tsv"].drop(columns=["source_file"])
    with pytest.raises(KeyError, match="source_file"):
        select_rows_to_embed(frame)


def test_a_source_file_argument_covers_a_frame_with_no_column(policy_frames) -> None:
    frame = policy_frames["test_source2.tsv"].drop(columns=["source_file"])
    assert len(select_rows_to_embed(frame, "test_source2.tsv")) == 10
    # Told it is source1 instead, the same rows are all kept.
    assert len(select_rows_to_embed(frame, "test_source1.tsv")) == len(frame)


def test_a_frame_claiming_several_source_files_is_all_kept(policy_frames) -> None:
    """
    A frame with a per-row source_file column can be a mixed read. source1's
    whole-file rule then applies per row, which is the only coherent reading --
    but the mixed read must not be mistaken for a clean per-file split.
    """
    mixed = pd.concat(list(policy_frames.values()), ignore_index=True)
    assert len(mixed) == 39
    selected = select_rows_to_embed(mixed)
    expected = sum(
        EXPECTED_SELECTED[name] for name in policy_frames
    )
    assert len(selected) == expected


# --------------------------------------------------------------------------
# The text that gets embedded
# --------------------------------------------------------------------------
def test_blank_names_are_not_sent_to_the_encoder(policy_frames) -> None:
    """
    An empty string embeds to a vector at a meaningless fixed distance from
    everything else, at the cost of a forward pass.
    """
    frame = policy_frames["test_source2.tsv"].copy()
    frame.loc[frame.index[0], "business_name_clean"] = "   "
    text = select_names_to_embed(frame, "test_source2.tsv")
    assert (text.str.strip() != "").all()
    # Still selected by the policy, just not sent to the encoder.
    assert len(select_rows_to_embed(frame, "test_source2.tsv")) == 10
    assert len(text) == 9


def test_a_missing_name_column_raises() -> None:
    frame = pd.DataFrame({NON_LATIN_FLAG: [True], "source_file": ["test_source2.tsv"]})
    with pytest.raises(KeyError):
        select_names_to_embed(frame)


def test_the_text_column_is_the_cleaned_name(policy_frames) -> None:
    text = select_names_to_embed(policy_frames["test_source2.tsv"], "test_source2.tsv")
    frame = policy_frames["test_source2.tsv"]
    expected = frame[frame[NON_LATIN_FLAG].astype(bool)]["business_name_clean"]
    assert text.tolist() == expected.tolist()


# --------------------------------------------------------------------------
# Expected row counts
# --------------------------------------------------------------------------
def test_the_expected_test_total_is_inside_the_briefed_range() -> None:
    """
    The brief puts the full-run cost at 3.3-3.7M rows for the test sources. If
    that drifts out, the number quoted in the writeup is wrong.

    Broad only: the range in the brief is the broad figure, and the strict filter
    lands below it by construction. Asserting the strict total fits in here would
    be asserting the bug this file used to have.
    """
    total = expected_total(TEST_SOURCES, strict=False)
    assert EXPECTED_TOTAL_TEST_RANGE[0] <= total <= EXPECTED_TOTAL_TEST_RANGE[1]
    assert total == 3_398_081
    assert expected_total(TEST_SOURCES, strict=True) < EXPECTED_TOTAL_TEST_RANGE[0]


def test_expected_rows_combine_the_two_policies() -> None:
    assert expected_embed_rows("test_source1.tsv") == 1_732_544
    assert expected_embed_pct("test_source1.tsv") == 100.0
    assert expected_embed_rows("test_source2.tsv") == round(4_887_273 * 0.1899)
    assert expected_embed_pct("test_source2.tsv") == 18.99
    assert expected_embed_rows("train_source3.tsv") == round(5_285_603 * 0.1148)


def test_an_unknown_file_has_no_expected_count() -> None:
    """A wrong filename must report "no expectation", not invent one."""
    assert expected_embed_rows("nope_source9.tsv") is None
    assert expected_embed_pct("nope_source9.tsv") is None
    # Still None in strict mode: an unknown file is unknown either way.
    assert expected_embed_rows("nope_source9.tsv", strict=True) is None


# --------------------------------------------------------------------------
# The model: the default has to be the real default, not just an accepted flag
# --------------------------------------------------------------------------
def test_the_default_model_is_labse() -> None:
    """
    "Accepts --model" and "defaults to LaBSE" are different states. The brief
    for the real Colab run is LaBSE, so the constant is asserted by name --
    a rename that quietly swapped in a different encoder has to break a test.
    """
    assert DEFAULT_MODEL == "sentence-transformers/LaBSE"
    assert "LaBSE" in DEFAULT_MODEL


def test_embed_names_defaults_to_the_constant_not_an_inline_string() -> None:
    """
    The signature must read the constant. A literal pasted into the signature
    would drift from the CLI default and nobody would notice until the GPU
    bill arrived.
    """
    default = inspect.signature(embed_names).parameters["model_name"].default
    assert default == DEFAULT_MODEL
    assert "LaBSE" in default


def test_the_cli_model_flag_defaults_to_labse() -> None:
    parser = build_parser()
    assert parser.get_default("model") == DEFAULT_MODEL
    # Not a store_true in disguise: it takes a value.
    assert parser.parse_args([]).model == DEFAULT_MODEL
    assert parser.parse_args(["--model", "some/other-model"]).model == "some/other-model"


def test_the_reported_encoder_is_stated_and_the_flag_reaches_embed_names(
    policy_frames, capsys, tmp_path, monkeypatch
) -> None:
    """
    `--model` has to be load-bearing. The report prints the encoder it describes
    even when it does not embed, and `--embed-out` passes the value to
    `embed_names`, asserted here with a fake encoder rather than a 471 MB
    download.
    """
    seen: dict[str, object] = {}

    class FakeEncoder:
        def __init__(self, name, device=None):
            seen["model_name"] = name

        def encode(self, texts, **kwargs):
            seen["texts"] = list(texts)
            seen["kwargs"] = kwargs
            return np.zeros((len(texts), 3), dtype="float32")

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = FakeEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)

    out_path = tmp_path / "nested" / "emb.parquet"
    code = embed_names_module.main([
        "--rows", "50", "--strata", "1", "--model", "fake/encoder", "--embed-out", str(out_path),
    ])
    assert code in (0, 1)  # a 50-row sample is not within tolerance of the real rate
    assert seen["model_name"] == "fake/encoder"
    assert "fake/encoder" in capsys.readouterr().out
    assert out_path.exists()


def test_the_default_model_is_never_the_english_only_one() -> None:
    """
    `generate_candidates` names all-MiniLM-L6-v2. It is English-only and would
    embed exactly the rows this module selects into noise, so the default must
    never become it.
    """
    assert DEFAULT_MODEL != REJECTED_MODEL
    assert REJECTED_MODEL not in DEFAULT_MODEL


def test_strict_expectations_are_measured_not_published() -> None:
    """
    The strict union is not in the report. The report gives a per-script
    breakdown, but a row can appear under several scripts, so the nine Indic
    percentages cannot be summed into the union -- there is no published figure
    for it. The strict expectation is therefore this module's own measurement,
    kept in its own table, and the tests here check that table is the one used.
    """
    assert set(MEASURED_STRICT_PCT) == set(REPORTED_NON_ASCII_PCT)
    for name in FILTER_BY_FLAG:
        # Strict is necessarily below broad: it drops the accented-Latin rows.
        assert MEASURED_STRICT_PCT[name] < REPORTED_NON_ASCII_PCT[name], name
        assert expected_embed_pct(name, strict=True) == MEASURED_STRICT_PCT[name], name
        assert expected_embed_pct(name) == REPORTED_NON_ASCII_PCT[name], name


# --------------------------------------------------------------------------
# Regression: `strict` reached the percentage but not the count
# --------------------------------------------------------------------------
# `expected_embed_rows` and `expected_total` used to take no `strict`, so a
# strict run printed the broad total: the "~2.6M rows a strict run embeds" line
# showed 3,398,081, overstating strict by ~800k rows. That number was then read
# as a cost estimate, so the bug was not cosmetic. The tests below exist to make
# "the flag is accepted but does not change the output" a failure.
def test_strict_changes_the_expected_row_count_not_just_the_percentage() -> None:
    """The bug itself: the count has to move when the rate moves."""
    for name in FILTER_BY_FLAG:
        broad = expected_embed_rows(name, strict=False)
        strict = expected_embed_rows(name, strict=True)
        assert broad is not None and strict is not None, name
        assert strict < broad, name
        # And it is the rate that explains the difference, not a fudge factor.
        assert broad == round(REPORTED_ROWS[name] * REPORTED_NON_ASCII_PCT[name] / 100)
        assert strict == round(REPORTED_ROWS[name] * MEASURED_STRICT_PCT[name] / 100)


def test_strict_changes_the_expected_total() -> None:
    """
    The exact assertion asked for: the total must differ between modes whenever
    the underlying rates differ, so the flag cannot be inert.
    """
    assert expected_total(strict=True) != expected_total(strict=False)
    assert expected_total(ALL_SOURCES, strict=True) != expected_total(
        ALL_SOURCES, strict=False
    )
    # source1 is embedded whole under both, so the difference is only the
    # candidate side, and it has to be a real number of rows, not a rounding wobble.
    difference = expected_total(strict=False) - expected_total(strict=True)
    assert difference == sum(
        expected_embed_rows(name, strict=False) - expected_embed_rows(name, strict=True)
        for name in FILTER_BY_FLAG
        if name in TEST_SOURCES
    )
    assert difference > 500_000


def test_strict_never_inflates_the_expected_total() -> None:
    """
    Direction matters as much as magnitude: strict filters strictly more rows,
    so its total can only be lower. A strict total above the broad one would mean
    the two modes have crossed over somewhere.
    """
    for sources in (TEST_SOURCES, ALL_SOURCES):
        assert expected_total(sources, strict=True) < expected_total(
            sources, strict=False
        )


def test_every_strict_aware_function_actually_responds_to_the_flag() -> None:
    """
    A guard for the whole file, not just this bug.

    The set of public functions taking `strict` is pinned here. If someone adds
    a new one and forgets to wire the flag through, the equality below fails
    until they add a case -- so "accepts the argument, ignores it" cannot
    reappear in a function nobody thought to check.
    """
    declared = {
        "embed_names",
        "expected_embed_pct",
        "expected_embed_rows",
        "expected_non_latin_rows",
        "expected_total",
        "expectation_basis",
        "plan_by_source",
        "report_expected_vs_actual",
        "select_names_to_embed",
        "select_rows_to_embed",
    }
    actual = {
        name
        for name, obj in vars(embed_names_module).items()
        if inspect.isfunction(obj)
        and not name.startswith("_")
        and "strict" in inspect.signature(obj).parameters
    }
    assert actual == declared, f"strict-aware functions changed: {actual ^ declared}"


@pytest.mark.parametrize("name", sorted(FILTER_BY_FLAG))
def test_each_function_that_takes_strict_responds_to_it(name: str, policy_frames) -> None:
    """
    Behavioural half of the guard: for a file whose two rates differ, every
    strict-aware function must return something different per mode.
    """
    flips = {
        "expected_embed_pct": (expected_embed_pct(name), expected_embed_pct(name, strict=True)),
        "expected_embed_rows": (
            expected_embed_rows(name), expected_embed_rows(name, strict=True)
        ),
        "expected_non_latin_rows": (
            expected_non_latin_rows(name), expected_non_latin_rows(name, strict=True)
        ),
        "expected_total": (expected_total((name,)), expected_total((name,), strict=True)),
        "expectation_basis": (
            expectation_basis(name), expectation_basis(name, strict=True)
        ),
    }
    for func_name, (broad, strict) in flips.items():
        assert broad != strict, f"{func_name}({name}) ignored strict"

    # The row-selection half needs a frame holding the disputed row, which the
    # fixture puts in test_source3: that is the only place SAMP-039 landed.
    disputed = policy_frames["test_source3.tsv"]
    assert len(select_rows_to_embed(disputed, "test_source3.tsv")) != len(
        select_rows_to_embed(disputed, "test_source3.tsv", strict=True)
    )
    assert len(select_names_to_embed(disputed)) != len(
        select_names_to_embed(disputed, strict=True)
    )
    assert list(plan_by_source({"test_source3.tsv": disputed})["to_embed"]) != list(
        plan_by_source({"test_source3.tsv": disputed}, strict=True)["to_embed"]
    )
    # A frame whose two modes agree is not a bug, so assert that explicitly
    # rather than pretending the flag always changes the row count.
    clean = policy_frames["test_source2.tsv"]
    assert int(clean[INDIA_SCRIPT_FLAG].sum()) == int(clean[NON_LATIN_FLAG].sum())


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------
def test_plan_by_source_reports_both_policies(policy_frames) -> None:
    plan = plan_by_source(policy_frames)
    assert list(plan["source_file"]) == sorted(policy_frames)
    by_name = plan.set_index("source_file")
    for name, expected in EXPECTED_SELECTED.items():
        assert by_name.loc[name, "to_embed"] == expected, name
        assert by_name.loc[name, "rows"] == EXPECTED_ROWS[name], name
    assert by_name.loc["test_source1.tsv", "basis"] == "policy"
    assert by_name.loc["test_source2.tsv", "basis"] == "published"
    # The absolute column has to be the mode's too, not the broad one bolted on.
    assert by_name.loc["test_source2.tsv", "expected_full_rows"] == expected_embed_rows(
        "test_source2.tsv"
    )
    assert by_name.loc["test_source2.tsv", "expected_full_rows"] > expected_embed_rows(
        "test_source2.tsv", strict=True
    )


def test_plan_by_source_marks_strict_as_a_measured_baseline(policy_frames) -> None:
    plan = plan_by_source(policy_frames, strict=True).set_index("source_file")
    assert plan.loc["test_source2.tsv", "to_embed"] == 10
    assert plan.loc["test_source3.tsv", "to_embed"] == 9
    assert plan.loc["test_source2.tsv", "basis"] == "measured baseline"
    # Same bug, same place: the strict plan must not quote the broad row count.
    assert plan.loc["test_source2.tsv", "expected_full_rows"] == expected_embed_rows(
        "test_source2.tsv", strict=True
    )


def test_report_passes_when_the_tolerance_is_wide(policy_frames, capsys) -> None:
    """
    The 39 rows are not the population, so their rates are nowhere near the
    published ones. The report is only expected to be *runnable* here; its
    verdict on real data comes from the CLI run.
    """
    code = report_expected_vs_actual(
        policy_frames, sources=TEST_SOURCES, tolerance_pct=100.0
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "embedding scope" in out
    assert "test_source2.tsv" in out
    assert "basis" in out


def test_report_fails_loudly_when_the_rates_are_wrong(policy_frames, capsys) -> None:
    """A zero tolerance on a sample must fail, or the check means nothing."""
    code = report_expected_vs_actual(
        policy_frames, sources=TEST_SOURCES, tolerance_pct=0.0
    )
    assert code == 1
    out = capsys.readouterr().out
    assert "FAIL:" in out
    assert "test_source1.tsv" in out


def test_report_rejects_a_source_with_no_published_figure(policy_frames, capsys) -> None:
    """
    A typo in a filename must not pass unnoticed. `train_ground_truth.tsv` is in
    the dataset, has no business_name at all, and is not an embedding scope.
    """
    frames = dict(policy_frames, **{"train_ground_truth.tsv": policy_frames["test_source1.tsv"]})
    code = report_expected_vs_actual(
        frames, sources=("train_ground_truth.tsv",), tolerance_pct=1.0
    )
    assert code == 1
    assert "no published figure" in capsys.readouterr().out


def test_report_rejects_a_source_that_was_never_loaded(policy_frames, capsys) -> None:
    """
    Asking about a source that has no rows must not read as "nothing to check"
    and pass -- that is how a split silently drops out of a full run.
    """
    code = report_expected_vs_actual(
        policy_frames, sources=("train_source2.tsv",), tolerance_pct=1.0
    )
    assert code == 1
    assert "no rows were given" in capsys.readouterr().out


# --------------------------------------------------------------------------
# The expensive half stays out of reach
# --------------------------------------------------------------------------
def test_importing_the_module_does_not_pull_in_sentence_transformers() -> None:
    """
    Checked in a subprocess: by the time this test runs, other tests in the
    session may already have imported something heavy.
    """
    code = (
        "import sys;"
        "import src.blocking.embed_names as m;"
        "heavy = [k for k in sys.modules if k == 'torch' "
        "or k.startswith('sentence_transformers')];"
        "assert not heavy, heavy;"
        "print('clean')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "clean" in result.stdout
