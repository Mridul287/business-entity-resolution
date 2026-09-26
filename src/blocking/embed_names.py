"""
Phase 3b embedding step, scope and row accounting.

Owner: Person A

WHAT THIS MODULE IS
-------------------

A sentence-transformer over 3.4M names is a GPU-hours question, so the first
decision is not "which model" but "which rows". This module owns that decision
and nothing else: it decides which rows are worth embedding, counts them, and
reports the count against a published expectation. The embedding itself is a
deliberate, explicit call, so that nothing in the test suite or in CI can load a
model by accident.

WHY THE SCOPE IS ASYMMETRIC
---------------------------

Embedding every name in every source would be ~26M rows. The asymmetry that
makes it affordable:

  * **source1 is embedded whole.** It is the anchor side of the task: recall
    against `train_ground_truth` is measured as source1 -> candidate, so a
    source1 row that is never embedded can never be recalled, no matter how good
    the candidate side is. It is also the smallest file (1.73M test rows) and,
    measured, 0.00% Indic / 2.35% non-ASCII -- it is Latin transliteration and
    English, which is exactly what an off-the-shelf multilingual encoder
    handles well already.

  * **source2 and source3 are embedded only where the name is non-Latin.**
    These are the candidate side, and they are where the recall risk actually
    lives: 15-19% of their names are written in a script an ASCII-based
    similarity will not match against a Latin transliteration of the same
    business. Those rows are exactly the ones `has_non_latin_name` selects, and
    skipping the other ~85% is the entire saving.

The asymmetry is a *cost* decision, not a correctness one, and it is reversible:
`has_indian_script_name` is the strict reading of the same filter, and
`select_rows_to_embed(strict=True)` switches to it in one argument. That
narrows the candidate side to 11.2% / 6.2% of test_source2 / test_source3,
dropping the accented-Latin France rows, which carry no transliteration risk.

THE EXPECTED NUMBERS
--------------------

`REPORTED_ROWS` and `REPORTED_NON_ASCII_PCT` are literals from
`docs/data_profile_report.md`, not recomputed by this module. The point of
`report_expected_vs_actual` is to compare this step's output against the
*published* figure, so deriving the expectation from the same code that
produces the actual number would make the check vacuous.

The one exception is `strict=True`. The report publishes a per-script
breakdown, but a row can appear under several scripts, so the nine Indic
percentages cannot be summed into the strict union -- there is no published
figure for it. `MEASURED_STRICT_PCT` holds this module's own measurement of
that union, kept in a separate table so the two kinds of expectation are never
confused, and the report prints which basis it used for each row.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd

from ..preprocessing.multilingual import (
    FLAG_NAME_COLUMN,
    INDIA_SCRIPT_FLAG,
    NON_LATIN_FLAG,
    REPORTED_NON_ASCII_PCT,
    REPORTED_ROWS,
    read_stratified_sample,
)
from ..preprocessing.profile_data import NAME_COL

# The column that says which file a row came from. `profile_data`'s row counts
# and this module's accounting are both keyed on the file name, so the two
# cannot disagree about what "test_source2" means.
SOURCE_FILE_COLUMN = "source_file"

# The three splits the pipeline embeds. Order is the argument order of
# `generate_candidates`: (source1, source2, source3).
TEST_SOURCES = ("test_source1.tsv", "test_source2.tsv", "test_source3.tsv")
TRAIN_SOURCES = ("train_source1.tsv", "train_source2.tsv", "train_source3.tsv")
ALL_SOURCES = TRAIN_SOURCES + TEST_SOURCES

# The full-run cost for the test sources under the *broad* filter, as a range the
# brief puts at 3.3-3.7M. Asserted in the tests, because it is the number that
# goes in a writeup and a change to the ranges or the policy should move it out
# loud, not quietly. The strict filter lands well below this range (~2.6M) by
# construction, so a test that puts `expected_total` inside it must say
# `strict=False`.
EXPECTED_TOTAL_TEST_RANGE = (3_300_000, 3_700_000)

# source1 is embedded whole; sources 2 and 3 are filtered. Named rather than
# inlined so the policy is one readable line and the tests can assert on it.
EMBED_WHOLE = ("test_source1.tsv", "train_source1.tsv")
FILTER_BY_FLAG = ("test_source2.tsv", "test_source3.tsv", "train_source2.tsv", "train_source3.tsv")

# The encoder. LaBSE is the default, not a suggestion: it is a 15-language
# sentence encoder aligned across 109 translation pairs, so the same vector
# space holds a Devanagari name and its Latin transliteration close together --
# which is the whole recall problem here, since the anchor side is Latin
# transliteration and the candidate side is often not.
#
# It is also a 24-layer, 1024-dim BERT-large, several times the cost per name
# than the 12-layer/384-dim multilingual MiniLM. That is a deliberate trade:
# the scope decision in this module is about *which* rows, and spending the
# compute on the right rows with the better model beats the reverse. Budget the
# GPU time accordingly -- at ~3.4M rows this is the dominant cost in Phase 3b,
# an order of magnitude above the ~791k rows the strict filter would save.
DEFAULT_MODEL = "sentence-transformers/LaBSE"

# Named in `generate_candidates`, and wrong for this data: it is English-only,
# with no Devanagari or Telugu in its vocabulary, and would embed exactly the
# rows this module selects into noise. Recorded here so nobody reaches for it
# as a "smaller, faster" swap without seeing why it is excluded.
REJECTED_MODEL = "all-MiniLM-L6-v2"

# The column the text is taken from, and the one it is written to.
TEXT_COLUMN = "business_name_clean"
OUTPUT_COLUMN = "text"


def expected_rows_at_rate(source_file: str, pct: float | None) -> int | None:
    """
    `pct` percent of `source_file`'s published row count, rounded.

    Returns None for a file the report does not cover, so a caller gets an
    honest "no expectation" instead of a number invented from nothing.
    """
    rows = REPORTED_ROWS.get(source_file)
    if rows is None or pct is None:
        return None
    return round(rows * pct / 100.0)


def expected_non_latin_rows(source_file: str, *, strict: bool = False) -> int | None:
    """
    Rows in `source_file` expected to carry the flag, for the given mode.

    A thin wrapper over `expected_rows_at_rate` so that the rate and the count
    can never disagree about which mode they are describing: both go through
    `expected_embed_pct`.
    """
    return expected_rows_at_rate(source_file, expected_embed_pct(source_file, strict=strict))


# Regression baseline for the strict filter, as a percentage of each file.
#
# NOT a published figure, and deliberately kept separate from
# `REPORTED_NON_ASCII_PCT`. The report gives a per-script breakdown, but a row
# can appear under several scripts, so the nine Indic percentages cannot be
# summed into the strict union -- there is no published number for it.
#
# So these are this module's own values, measured on a 300k-row stratified
# sample (50k per file) of the real data. They are a tripwire for drift, not a
# certificate of correctness: if a change to the ranges or to the flag moves
# these by more than a point, something changed that a human should look at.
MEASURED_STRICT_PCT: dict[str, float] = {
    "train_source1.tsv": 0.00,
    "train_source2.tsv": 9.28,
    "train_source3.tsv": 5.41,
    "test_source1.tsv": 0.00,
    "test_source2.tsv": 11.22,
    "test_source3.tsv": 6.21,
}


def expected_embed_pct(source_file: str, *, strict: bool = False) -> float | None:
    """
    Percentage of `source_file` expected to be embedded, or None if unknown.

    100.0 for a whole-embedded file. For a filtered one, the published
    non-ASCII rate under the broad flag and `MEASURED_STRICT_PCT` under the
    strict one, since the strict union is not a published figure.

    This is the number a *sample* can be held to: `expected_embed_rows` is an
    absolute count over the full 500 MB file, and dividing that by a 20k sample
    gives a meaningless percentage.
    """
    if source_file in EMBED_WHOLE:
        return 100.0
    if source_file in FILTER_BY_FLAG:
        table = MEASURED_STRICT_PCT if strict else REPORTED_NON_ASCII_PCT
        return table.get(source_file)
    return None


def expectation_basis(source_file: str, *, strict: bool = False) -> str:
    """Where the expectation for `source_file` came from, for the report header."""
    if source_file in EMBED_WHOLE:
        return "policy"
    return "measured baseline" if strict else "published"


def expected_embed_rows(source_file: str, *, strict: bool = False) -> int | None:
    """
    Rows in the *full* `source_file` this module expects to embed, for the mode.

    Whole for source1, flag-filtered for source2/source3, so the two policies
    are summed in one place rather than at each call site. Absolute, and only
    comparable against a full-file row count.

    `strict` matters here and used to be ignored, which made a strict run print
    the broad total: the share of the file is the mode's, so the count has to be
    too. Both come from `expected_embed_pct`, which is the single place that
    decides which rate applies.
    """
    if source_file in EMBED_WHOLE:
        return REPORTED_ROWS.get(source_file)
    if source_file in FILTER_BY_FLAG:
        return expected_non_latin_rows(source_file, strict=strict)
    return None


def expected_total(sources: Sequence[str] = TEST_SOURCES, *, strict: bool = False) -> int:
    """Sum of `expected_embed_rows`, skipping files with no published figure."""
    return sum(
        n for n in (expected_embed_rows(s, strict=strict) for s in sources) if n is not None
    )


def _source_key(df: pd.DataFrame, source_file: str | None) -> pd.Series:
    """
    A per-row label identifying the file, as a Series aligned to `df`.

    A frame read straight from a TSV has no `source_file` column -- the column
    only exists because the stratified sampler adds it -- so a caller that
    already knows which file it is holding passes `source_file` and gets a
    constant column back. Without one of the two this cannot decide, and
    guessing would mean embedding the wrong rows.
    """
    if SOURCE_FILE_COLUMN in df.columns:
        return df[SOURCE_FILE_COLUMN].fillna("").astype(str)
    if source_file is not None:
        return pd.Series([source_file] * len(df), index=df.index, dtype=object)
    raise KeyError(
        f"cannot tell which source file these {len(df)} rows came from: the frame "
        f"has no {SOURCE_FILE_COLUMN!r} column and no source_file was passed"
    )


def select_rows_to_embed(
    df: pd.DataFrame,
    source_file: str | None = None,
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """
    The rows of `df` worth embedding, in the order they arrived.

    source1 rows are kept regardless of the flag; source2/source3 rows are kept
    only where the flag is True. With `strict=True` the Indic-only flag is used
    instead of the broad one, which drops the accented-Latin rows.

    Non-destructive: the frame is filtered, never modified, and every original
    column comes along so the caller can see *why* a row was kept.
    """
    if len(df) == 0:
        return df.copy()
    keys = _source_key(df, source_file)
    flag_column = INDIA_SCRIPT_FLAG if strict else NON_LATIN_FLAG
    if flag_column not in df.columns:
        raise KeyError(
            f"{flag_column!r} is not in the frame. Run "
            f"src.preprocessing.multilingual.add_multilingual_columns first -- this "
            f"step filters on that column and will not guess."
        )
    flags = df[flag_column].fillna(False).astype(bool)

    # The policy, as one expression: whole for source1, filtered for the rest.
    keep = (flags | keys.isin(EMBED_WHOLE)).to_numpy()
    return df.loc[keep].copy()


def select_names_to_embed(
    df: pd.DataFrame,
    source_file: str | None = None,
    *,
    strict: bool = False,
) -> pd.Series:
    """
    Just the text to embed, as a Series.

    Blank names are dropped here rather than sent to the encoder: an empty
    string embeds to a vector that is at a meaningless fixed distance from
    everything, and it costs a forward pass to learn nothing.
    """
    column = TEXT_COLUMN if TEXT_COLUMN in df.columns else NAME_COL
    selected = select_rows_to_embed(df, source_file, strict=strict)
    text = selected[column].fillna("").astype(str).str.strip()
    return text[text != ""]


def plan_by_source(
    frames: dict[str, pd.DataFrame],
    *,
    strict: bool = False,
) -> pd.DataFrame:
    """
    One row per source file: how many rows it has, how many this step embeds,
    and the rate to check that share against.

    This is the table to read before spending GPU-hours: a source whose actual
    share is far from the expected one means the flag is wrong, and that is
    worth finding out on a sample rather than on a full run.
    """
    rows = []
    for name, frame in frames.items():
        selected = select_rows_to_embed(frame, name, strict=strict)
        rows.append(
            {
                "source_file": name,
                "rows": len(frame),
                "to_embed": len(selected),
                "expected_pct": expected_embed_pct(name, strict=strict),
                "basis": expectation_basis(name, strict=strict),
                "actual_pct": 100.0 * len(selected) / len(frame) if len(frame) else 0.0,
                # Absolute count over the *full* file, for the same mode as
                # `expected_pct`. Not comparable to a sample's `to_embed`.
                "expected_full_rows": expected_embed_rows(name, strict=strict),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["source_file", "rows", "to_embed", "expected_pct", "basis",
                 "actual_pct", "expected_full_rows"],
    )


def report_expected_vs_actual(
    frames: dict[str, pd.DataFrame],
    *,
    sources: Sequence[str] = TEST_SOURCES,
    strict: bool = False,
    tolerance_pct: float = 1.0,
) -> int:
    """
    Print actual against expected per file, and return a process exit code.

    The comparison is between *shares of the file*, because that is the only
    thing a sample can be held to. `expected_full_rows` is printed alongside as
    the number a full run would embed, and the total at the end extrapolates
    the sample onto it, which is the figure worth arguing about in a writeup.

    `tolerance_pct` is in percentage points and is loose on purpose: this is
    normally run against a stratified sample, and on a 19% rate a 50k sample has
    a standard error of well under a point. A real detection bug moves the
    number by whole points, so 1.0 still catches it.

    Exit code 0 if every covered file is within tolerance, 1 otherwise. Files
    with no published figure are reported and failed rather than skipped, so a
    typo in a filename cannot pass unnoticed.
    """
    plan = plan_by_source(frames, strict=strict)
    wanted = list(sources)
    shown = plan[plan["source_file"].isin(wanted)]

    header = (f"{'file':<22}{'rows':>10}{'embed':>9}{'actual %':>10}"
              f"{'expect %':>10}{'delta':>8}{'full-run rows':>15}   basis")
    print(f"embedding scope ({'strict Indic' if strict else 'broad non-Latin'} flag)")
    print(header)
    print("-" * len(header))

    # A requested source that is not in the frames at all is a mistake -- a typo,
    # or a split that was not loaded. It must not pass as "nothing to check".
    failures: list[str] = []
    missing = [name for name in wanted if name not in set(plan["source_file"])]
    for name in missing:
        failures.append(f"{name}: requested as an embedding scope but no rows were given")
    for _, row in shown.iterrows():
        expected_pct = row["expected_pct"]
        actual_pct = row["actual_pct"]
        if expected_pct is None or pd.isna(expected_pct):
            print(f"{row['source_file']:<22}{row['rows']:>10,}{row['to_embed']:>9,}"
                  f"{actual_pct:>9.2f}%{'n/a':>10}{'n/a':>8}{'n/a':>15}")
            failures.append(f"{row['source_file']}: no published figure to check against")
            continue
        delta = actual_pct - expected_pct
        full_rows = row["expected_full_rows"]
        print(f"{row['source_file']:<22}{row['rows']:>10,}{row['to_embed']:>9,}"
              f"{actual_pct:>9.2f}%{expected_pct:>9.2f}%{delta:>+8.2f}"
              f"{('n/a' if full_rows is None or pd.isna(full_rows) else f'{int(full_rows):,}') :>15}"
              f"   {row['basis']}")
        if abs(delta) > tolerance_pct:
            failures.append(
                f"{row['source_file']}: {actual_pct:.2f}% of rows selected is "
                f"{delta:+.2f} points from the expected {expected_pct:.2f}% "
                f"(tolerance {tolerance_pct})"
            )

    print("-" * len(header))
    total_actual = int(shown["to_embed"].sum()) if len(shown) else 0
    total_expected = expected_total(wanted, strict=strict)
    print(f"{'TOTAL':<22}{int(shown['rows'].sum()):>10,}{total_actual:>9,}"
          f"{'':>10}{'':>10}{'':>8}{total_expected:>15,}")
    if total_expected and len(shown) == len(wanted):
        share = 100.0 * total_actual / total_expected
        print(f"  this {total_actual:,}-row sample is {share:.1f}% of the "
              f"{total_expected:,} rows a full run embeds")

    if failures:
        print("FAIL:")
        for line in failures:
            print(f"  !! {line}")
        return 1
    basis = "measured baseline" if strict else "published rates"
    print(f"PASS: every file within {tolerance_pct} points of its {basis}")
    return 0


def embed_names(
    df: pd.DataFrame,
    model_name: str = DEFAULT_MODEL,
    *,
    source_file: str | None = None,
    strict: bool = False,
    batch_size: int = 256,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Embed the selected names. This is the expensive call, and it is explicit.

    The model is imported inside the function on purpose. Importing
    sentence-transformers at module scope would make `import
    src.blocking.embed_names` -- which the test suite and the row-count report
    both do -- able to pull in torch and, if anything ever calls
    `SentenceTransformer(...)` at import time, download a model. Keeping the
    import local means the cheap half of this module stays cheap and offline.

    `model_name` defaults to `DEFAULT_MODEL` (LaBSE) rather than naming a model
    here, so the CLI flag, this signature, and the tests all read the same
    constant. A default written inline is a default that drifts.

    Returns a frame with the input's columns plus `text` and `embedding`.
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415 - see docstring

    text = select_names_to_embed(df, source_file, strict=strict)
    if len(text) == 0:
        return df.assign(**{OUTPUT_COLUMN: [], "embedding": []})

    model = SentenceTransformer(model_name, device=device)
    vectors = model.encode(
        text.tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    out = select_rows_to_embed(df, source_file, strict=strict)
    out = out.loc[text.index].copy()
    out[OUTPUT_COLUMN] = text
    out["embedding"] = list(vectors)
    return out


def _load_frames(dataset_dir, splits: Sequence[str], rows: int, strata: int) -> dict[str, pd.DataFrame]:
    from ..preprocessing.clean_text import clean_dataframe
    from ..preprocessing.multilingual import add_multilingual_columns

    sample = read_stratified_sample(dataset_dir, rows=rows, strata=strata)
    sample = sample[sample[SOURCE_FILE_COLUMN].isin(splits)]
    processed = add_multilingual_columns(clean_dataframe(sample))
    return {
        name: group.reset_index(drop=True)
        for name, group in processed.groupby(SOURCE_FILE_COLUMN, sort=True)
    }


def build_parser() -> argparse.ArgumentParser:
    """
    The CLI, as a separate function so the defaults are assertable.

    A default that is only reachable by running the program is a default nothing
    tests. The tests read `--model`'s default and `--strict`'s action straight
    off this parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="defaults to <repo>/dataset")
    parser.add_argument("--rows", type=int, default=50_000,
                        help="rows per source file; the sample is bounded on purpose")
    parser.add_argument("--strata", type=int, default=5)
    parser.add_argument("--strict", action="store_true",
                        help="use has_indian_script_name instead of has_non_latin_name")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"sentence-transformers id to embed with (default: {DEFAULT_MODEL})")
    parser.add_argument("--embed-out", type=Path, default=None,
                        help="embed the selected sample rows with --model and write them here "
                             "(.parquet). Omit to only report the row counts.")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="allowed drift in percentage points of the file")
    parser.add_argument("--all-splits", action="store_true",
                        help="report the train sources too, not just test")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from ..preprocessing.multilingual import DEFAULT_DATASET_DIR

    dataset_dir = args.dataset_dir or DEFAULT_DATASET_DIR
    splits = ALL_SOURCES if args.all_splits else TEST_SOURCES
    frames = _load_frames(dataset_dir, splits, args.rows, args.strata)

    # Stated up front so a report can never be read as describing an encoder it
    # did not use. The row counts below are model-independent; this line is not.
    print(f"encoder: {args.model}")
    code = report_expected_vs_actual(
        frames, sources=splits, strict=args.strict, tolerance_pct=args.tolerance
    )

    if args.embed_out is not None:
        out = pd.concat(
            [
                embed_names(frame, args.model, source_file=name, strict=args.strict)
                for name, frame in frames.items()
            ],
            ignore_index=True,
        )
        args.embed_out.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.embed_out, index=False)
        print(f"embedded {len(out):,} rows with {args.model} -> {args.embed_out}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
