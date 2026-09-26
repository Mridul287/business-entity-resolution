"""
Owner: Person A

Normalize `business_name` and `business_address` into the form every later
stage agrees on: the two cleaned columns, the abbreviation-expanded columns,
the canonical blocking keys (`name_norm`, `address_norm`), the structured
address sub-tokens, and the script flags.

Nothing here branches on a hardcoded {US, India} set. France appears only in
test, and the README is explicit that `country` is an open set of string labels,
so the only country-conditional behaviour in the whole chain is which
postal-code rule `address_parse` picks, and an unknown country gets the generic
rule rather than a wrong one.

--------------------------------------------------------------------------
The order of the chain, and why each step has to come after the one above
--------------------------------------------------------------------------
    business_name     -> clean_text        -> business_name_clean
    business_address  -> clean_text        -> business_address_clean
                       -> multilingual     -> business_address_canonical
                       -> address_parse    -> business_*_expanded
                       -> (this module)    -> name_norm, address_norm,
                                             address_pin, postal_kind,
                                             address_street_number,
                                             address_token_set,
                                             name_token_set
                       -> multilingual     -> has_non_latin_name,
                                             has_indian_script_name

clean_text first, always. It is the only pass allowed to delete characters, and
everything downstream is built on its output: it repairs the mojibake, so the
abbreviation table sees "MUSIQUE JACQUES" rather than "MUSIQUE JÂCQUES"; it
drops the bracket decoration, so "[Pvt] Ltd" reaches the abbreviation pass as a
plain "Pvt Ltd" token; and it makes `has_non_latin_name` mean what its name
says (see multilingual.py's module docstring for the 15-row case that depends
on it).

canonicalize_state before the abbreviation pass. "महाराष्ट्र" and "Maharashtra"
are the same administrative unit and have to produce the same address token, or
no address comparison downstream can work at all. The gazetteer is gated on
`country` inside multilingual.py, which is why the raw `country` column is
still present and unmodified at this point.

abbreviations before the key fold. "12 MG RD" and "12 MG ROAD" are one address
written two ways; folding the case of both without expanding the abbreviation
leaves them two different keys. `expand_abbreviations` is scoped per column, so
`co` is expanded to "company" in a name and left alone in an address, where it
is the postal abbreviation for Colorado on this corpus.

The key fold is last, and it is deliberately weak -- accent folding,
lowercasing, whitespace. It does not strip legal suffixes, does not remove
punctuation and does not transliterate. Phase 5 blocking measures its recall
against real ground truth, and a key that rewrites too much produces a
confidently wrong match rather than a missing one, which is the cheaper error
to make here. Suffix stripping belongs in a similarity *feature*, not in the
key, and `business_name_clean` is still there for whoever wants it.

--------------------------------------------------------------------------
What this module is, and is not, destructive about
--------------------------------------------------------------------------
Non-destructive, like every other pass in this package: the raw
`business_name` / `business_address` and every intermediate column stay in the
frame, and new columns are added. A similarity feature may legitimately want
"12 MG RD" as written, or the raw text with its accents, or the unclean value
with its brackets, and none of that is recoverable once it has been folded
away. The one exception is deliberate: `normalize` is a *pipeline step*, so it
is safe to call on a frame in place.

Memory note for the full-scale run. `address_token_set` and `name_token_set`
are `frozenset`, roughly 0.5-0.7 KB per row for a typical address on this
corpus, so materializing both for 10.3M S2+S3 rows is about 7 GB and will not
fit next to a TF-IDF matrix in 15 GB. On a full run, chunk the source files and
keep only the key columns per chunk; see `src/blocking/lexical_blocking.py`,
which does exactly that.

Usage:
    python -m src.preprocessing.normalize --rows 50000
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import pandas as pd

from .address_parse import (
    ADDRESS_ABBREVIATIONS,
    EXPANDED_SUFFIX,
    NAME_ABBREVIATIONS,
    PIN_OR_ZIP,
    POSTAL_KIND,
    SCOPE_ADDRESS,
    SCOPE_NAME,
    STREET_NUMBER,
    TOKEN_SET,
    add_address_tokens,
    expand_abbreviations,
    expand_abbreviations_frame,
    fold_for_key,
)
from .clean_text import clean_dataframe
from .multilingual import (
    CANONICAL_ADDRESS_OUTPUT,
    INDIA_SCRIPT_FLAG,
    NON_LATIN_FLAG,
    add_canonical_state,
    add_indian_script_name_flag,
    add_non_latin_name_flag,
    read_stratified_sample,
)
from .profile_data import ADDRESS_COL, COUNTRY_COL, NAME_COL

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"

# The columns normalize() is contracted to add. `postal_kind` is not in the
# original scaffold's TODO list and is here for a measured reason: a postal code
# appears in 0.00-0.03% of rows of this corpus in the position where one belongs
# (see address_parse's module docstring), so `address_pin` cannot be compared
# without also knowing how much it is worth. `postal_kind` is that.
NAME_NORM = "name_norm"
ADDRESS_NORM = "address_norm"
# `address_pin`, not `address_pin_or_zip`: the name is the one the task split
# and the rest of the pipeline agreed on, and inside the frame the
# `postal_kind` column already says whether the value is a PIN, a ZIP or a
# fallback digit run. The address_parse-level name stays `pin_or_zip`.
ADDRESS_PIN = "address_pin"
POSTAL_KIND_COLUMN = "address_postal_kind"
ADDRESS_STREET_NUMBER = f"address_{STREET_NUMBER}"
ADDRESS_TOKEN_SET = f"address_{TOKEN_SET}"
NAME_TOKEN_SET = f"name_{TOKEN_SET}"

# The contracted output, in the order the tests check for it.
NORMALIZED_COLUMNS: tuple[str, ...] = (
    NAME_NORM,
    ADDRESS_NORM,
    ADDRESS_PIN,
    POSTAL_KIND_COLUMN,
    ADDRESS_STREET_NUMBER,
    ADDRESS_TOKEN_SET,
    NAME_TOKEN_SET,
    NON_LATIN_FLAG,
    INDIA_SCRIPT_FLAG,
)

# The column the address sub-tokens are parsed out of. It is built here rather
# than reusing `business_address_canonical`, because the abbreviations have to
# be expanded first ("STATION RD" and "STATION ROAD" are one address) and
# because the tokens should be reproducible from this one column.
ADDRESS_WORKING = f"{ADDRESS_COL}_working"
NAME_WORKING = f"{NAME_COL}_working"


def _fold_column(series: pd.Series) -> pd.Series:
    return series.map(fold_for_key)


def add_normalized_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add `name_working`, `address_working`, `name_norm`, `address_norm` and the
    two token-set columns, in that order.

    `*_working` is the abbreviation-expanded text and `*_norm` is that folded
    for key use. Both are kept: the working column is what a human reads when a
    candidate looks wrong, and the norm column is what an index is built on.
    """
    out = expand_abbreviations_frame(df)
    if f"{NAME_COL}{EXPANDED_SUFFIX}" in out.columns:
        out[NAME_WORKING] = out[f"{NAME_COL}{EXPANDED_SUFFIX}"]
    if f"{ADDRESS_COL}{EXPANDED_SUFFIX}" in out.columns:
        out[ADDRESS_WORKING] = out[f"{ADDRESS_COL}{EXPANDED_SUFFIX}"]
    else:
        # No address column at all, or none to expand: the canonical one is the
        # best available address text and the norm column is still wanted.
        source = CANONICAL_ADDRESS_OUTPUT if CANONICAL_ADDRESS_OUTPUT in out.columns else None
        if source is not None:
            out[ADDRESS_WORKING] = out[source]

    if NAME_WORKING in out.columns:
        out[NAME_NORM] = _fold_column(out[NAME_WORKING])
    elif NAME_COL in out.columns:
        out[NAME_NORM] = _fold_column(out[NAME_COL])
    if ADDRESS_WORKING in out.columns:
        out[ADDRESS_NORM] = _fold_column(out[ADDRESS_WORKING])
    elif ADDRESS_COL in out.columns:
        out[ADDRESS_NORM] = _fold_column(out[ADDRESS_COL])
    return out


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: a source DataFrame with at least [entity_id, business_name,
    business_address, country]. A frame missing any of them is still walked --
    every pass here is additive and none of them raises on a missing column --
    so the ground-truth file, which has no name or address at all, comes back
    unchanged rather than blowing up the pipeline.

    Output: the same DataFrame with these added:

        name_norm               str        folded, lowercase, accent-free key
        address_norm            str        ditto
        address_pin             str|None   the postal code, if any
        address_postal_kind     str|None   how much to trust address_pin
        address_street_number   str|None   the leading unit before the 1st comma
        address_token_set       frozenset  normalized, order-independent
        name_token_set          frozenset  normalized, order-independent
        has_non_latin_name      bool       Phase 3's filter flag
        has_indian_script_name  bool       the strict reading of that flag

    plus the intermediates the individual passes add: `*_clean`,
        `business_address_canonical`, `business_name_expanded`,
        `business_address_expanded`, `business_name_working` and
        `business_address_working`.

    Order matters and is fixed in one place -- `_STAGES` -- because the passes
    are not independent and a future edit that reorders them would be silent.
    The chain and the reasoning behind it is in the module docstring.
    """
    return _run_stages(df)


def _run_stages(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every stage in `_STAGES`, in order. The pipeline, in one line."""
    out = df
    for stage in _STAGES:
        out = stage(out)
    return out


def _stage_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 1: mojibake repair, noise removal, NFKC, whitespace."""
    return clean_dataframe(df)


def _stage_canonical_state(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 3a: Indian state/UT names folded to their canonical English form."""
    return add_canonical_state(df)


def _stage_expand_and_fold(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 4 + key fold: abbreviation expansion, then the `*_norm` keys."""
    return add_normalized_keys(df)


def _stage_address_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 4b: postal code, street number, order-independent token set.

    `add_address_tokens` names its four columns for what they are
    (`pin_or_zip`, `street_number`, ...), which is the right name inside that
    module and the wrong name here: once they are columns of a source frame,
    `street_number` is ambiguous with a future name-side one. So they are
    renamed to the `address_`-prefixed contract.

    Both spellings are dropped and the prefixed ones are then assigned, rather
    than `rename`d in place. `DataFrame.rename` cannot overwrite a column that
    already exists -- it appends a second column with the same name -- so
    renaming in place would make a second pass over an already-normalized frame
    grow `address_pin` twice, and the frame would stop being a frame you can
    index by name. Dropping the bare names as well is deliberate: they are an
    address_parse implementation detail, and the contract has one spelling.
    """
    out = add_address_tokens(df, address_column=ADDRESS_WORKING)
    mapping = {
        PIN_OR_ZIP: ADDRESS_PIN,
        POSTAL_KIND: POSTAL_KIND_COLUMN,
        STREET_NUMBER: ADDRESS_STREET_NUMBER,
        TOKEN_SET: ADDRESS_TOKEN_SET,
    }
    carried = {
        target: out[source] for source, target in mapping.items() if source in out.columns
    }
    stale = [c for c in (*mapping, *mapping.values()) if c in out.columns]
    if stale:
        out = out.drop(columns=stale)
    for target, values in carried.items():
        out[target] = values
    if NAME_NORM in out.columns:
        out[NAME_TOKEN_SET] = [
            frozenset(name.split()) for name in out[NAME_NORM]
        ]
    return out


def _stage_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Phase 3b: the script flags, computed on the cleaned name column."""
    out = add_non_latin_name_flag(df)
    return add_indian_script_name_flag(out)


# The pipeline, in the one place it is written down.
_STAGES = (
    _stage_clean,
    _stage_canonical_state,
    _stage_expand_and_fold,
    _stage_address_tokens,
    _stage_flags,
)

__all__ = [
    "normalize",
    "add_normalized_keys",
    "NORMALIZED_COLUMNS",
    "ADDRESS_PIN",
    "POSTAL_KIND_COLUMN",
    "ADDRESS_STREET_NUMBER",
    "ADDRESS_TOKEN_SET",
    "NAME_TOKEN_SET",
    "NAME_NORM",
    "ADDRESS_NORM",
    "ADDRESS_ABBREVIATIONS",
    "NAME_ABBREVIATIONS",
    "SCOPE_ADDRESS",
    "SCOPE_NAME",
    "expand_abbreviations",
]


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

SELF_CHECK_ROWS = 50_000
SELF_CHECK_STRATA = 5


def _self_check(dataset_dir: Path, rows: int, strata: int) -> int:
    print(
        f"stratified sample: {rows:,} rows/file, {strata} strata, from {dataset_dir}",
        flush=True,
    )
    started = time.perf_counter()
    sample = read_stratified_sample(dataset_dir, rows=rows, strata=strata)
    print(f"  read {len(sample):,} rows in {time.perf_counter() - started:.1f}s", flush=True)

    out = normalize(sample)
    elapsed = time.perf_counter() - started
    print(f"  normalized {len(out):,} rows in {elapsed:.1f}s", flush=True)

    failures: list[str] = []
    missing = [c for c in NORMALIZED_COLUMNS if c not in out.columns]
    if missing:
        failures.append(f"contracted columns missing: {missing}")
    else:
        print(f"  all {len(NORMALIZED_COLUMNS)} contracted columns present")

    for column in (NAME_NORM, ADDRESS_NORM):
        if not out[column].map(lambda v: isinstance(v, str)).all():
            failures.append(f"{column} has a non-string value")

    # The keys have to be folded, or two sources spell one name two ways.
    for column in (NAME_NORM, ADDRESS_NORM):
        unfolded = int(out[column].map(lambda v: v != fold_for_key(v)).sum())
        if unfolded:
            failures.append(f"{column}: {unfolded} rows are not in folded form")

    # Idempotence: normalizing an already-normalized frame must be a no-op on
    # the key columns, or the keys cannot be an index.
    twice = normalize(out)
    for column in (NAME_NORM, ADDRESS_NORM, ADDRESS_TOKEN_SET, NAME_TOKEN_SET, ADDRESS_PIN):
        left, right = out[column], twice[column]
        if column in (ADDRESS_TOKEN_SET, NAME_TOKEN_SET):
            same = all(a == b for a, b in zip(left, right))
        else:
            same = bool((left == right).all() or left.isna().equals(right.isna()))
        if not same:
            failures.append(f"normalize is not idempotent on {column}")

    header = f"{'file':<22}{'rows':>9}{'name key':>11}{'addr key':>11}{'tok/row':>9}{'non-latin':>11}"
    print("\n" + header)
    print("-" * len(header))
    for name, group in out.groupby("source_file", sort=True):
        print(
            f"{name:<22}{len(group):>9,}"
            f"{int(group[NAME_NORM].ne('').sum()):>11,}"
            f"{int(group[ADDRESS_NORM].ne('').sum()):>11,}"
            f"{group[ADDRESS_TOKEN_SET].map(len).mean():>9.2f}"
            f"{int(group[NON_LATIN_FLAG].sum()):>11,}"
        )
    print("-" * len(header))
    print(f"\n  {len(out) / max(elapsed, 1e-9):,.0f} rows/s end-to-end, single-threaded")

    if failures:
        print("\nFAIL:")
        for line in failures:
            print(f"  !! {line}")
        return 1
    print("\nPASS: all contracted columns present, keys folded, normalize idempotent")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--rows", type=int, default=SELF_CHECK_ROWS)
    parser.add_argument("--strata", type=int, default=SELF_CHECK_STRATA)
    args = parser.parse_args(argv)
    return _self_check(args.dataset_dir, args.rows, args.strata)


if __name__ == "__main__":
    raise SystemExit(main())
