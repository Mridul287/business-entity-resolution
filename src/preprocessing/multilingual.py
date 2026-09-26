"""
Owner: Person A

Multilingual handling for the entity-resolution sources, on top of
clean_text.py's output (`business_name_clean` / `business_address_clean`).

What the profile found and what this module does about it
-------------------------------------------------------

`docs/data_profile_report.md` measures, per file, the share of
`business_name` values containing a character outside ASCII, broken down by
Unicode script. The two numbers that decide the whole design:

    file              business_name non-ASCII
    train_source1            0.00%
    train_source2           15.19%
    train_source3           11.48%
    test_source1             2.35%     <- entirely "Latin (accented)": France
    test_source2            18.99%
    test_source3            14.51%

S1 is Latin-only, so a character n-gram key built over S1 has nothing to match
in the 11-19% of S2/S3 names that are not in ASCII at all -- transliteration
handling in blocking is not optional, it is the whole reason Phase 3b exists.
So this module does exactly two things, and deliberately not a third:

1. `detect_scripts(text)` -- which scripts a value is written in, by Unicode
   codepoint range. The ranges are imported from `profile_data.SCRIPT_RANGES`
   rather than restated, so detection here and profiling there cannot drift
   apart. That import is the point: a test asserts that the flag this module
   produces equals the profiler's own "Any non-ASCII" row count on the same
   values, so "the flag is 15.19% of train_source2" and "the profiler measured
   15.19%" are the same statement and not two claims that happen to agree.

2. `add_non_latin_name_flag(df)` -- the `has_non_latin_name` boolean column.
   This is the column Phase 3b's embedding step filters on. It is *not* a
   translation and this module does not pretend to be one: a rule-based
   transliterator over ten Indic scripts would be wrong often enough to be
   dangerous, and a wrong transliteration produces a *confident* wrong key.
   Routing the real translation problem to a multilingual sentence embedding
   is the cheaper, honest answer. What the flag buys is scope: it says which
   rows need the expensive treatment, so the expensive treatment is only paid
   for on the 11-19% that need it.

   Naming caveat, deliberately kept rather than papered over: the column is
   called `has_non_latin_name` but it is true for "Cafe Zurich" as well as for
   "आनंद फाउंडेशन", because the profiler quantity it is defined against is
   "contains a non-ASCII character", and that is the number the acceptance
   thresholds (11-19% on S2/S3, ~3.4M rows across the test files) are stated
   in. Accented Latin is Latin script, so anyone who needs the strict reading
   should call `is_strictly_non_latin()`, which is `detect_scripts()` minus
   "Latin (accented)" -- `has_indian_script_name` below. On this data the
   difference is 0.00 points on all six source files except test_source1,
   where the strict reading gives 0.00% against the flag's 2.35%: the entire
   test_source1 non-ASCII share is accented Latin, i.e. the France rows.

3. `canonicalize_state(text)` -- rewrite an Indian state/UT name to its
   canonical English form, via the static gazetteer in
   `data/indian_state_names.json`, and add `business_address_canonical` in
   `add_canonical_state`. This is the one place where folding two scripts onto
   one string is safe, because the vocabulary is finite, published and
   closed: 28 states and 8 Union Territories, with a fixed English spelling
   each. "महाराष्ट्र" and "Maharashtra" are the same administrative unit and
   must produce the same token or no address comparison downstream can work.
   Anything not in the table passes through byte-for-byte.

   The table is PUBLIC REFERENCE DATA -- gazetteer knowledge, equivalent to a
   country list or a currency table. It is not a business lookup: no entry was
   fitted to, keyed on, or derived from any record in `dataset/`, and nothing
   in it encodes a match between two entities. The JSON says so in its own
   `methodology_note` as well, so the claim travels with the data if the file
   is copied on its own.

What this module deliberately does NOT do
-----------------------------------------

It does not transliterate names, and it does not translate districts, cities
or streets. `business_address_canonical` normalises the state/UT token only;
the street, city and PIN parts of the address are left exactly as they came
in. Extending the gazetteer to districts is a reasonable next step, but it is
a different piece of reference data with a different maintenance story and
belongs in its own file with its own tests.

Usage:
    python -m src.preprocessing.multilingual                          # self-check report
    python -m src.preprocessing.multilingual --rows 50000
    python -m src.preprocessing.multilingual --rows 50000 --strata 5
"""
from __future__ import annotations

import argparse
import io
import json
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .clean_text import CLEAN_SUFFIX
from .profile_data import (
    ADDRESS_COL,
    COUNTRY_COL,
    NAME_COL,
    NON_ASCII_SCRIPT,
    SCRIPT_NAMES,
    SCRIPT_RANGES,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_STATE_TABLE = REPO_ROOT / "data" / "indian_state_names.json"

# The address column this module canonicalises, and the column it adds.
CANONICAL_ADDRESS_COLUMN = f"{ADDRESS_COL}{CLEAN_SUFFIX}"
CANONICAL_ADDRESS_OUTPUT = f"{ADDRESS_COL}_canonical"
# The name column the flag is computed from, and the column it adds.
FLAG_NAME_COLUMN = f"{NAME_COL}{CLEAN_SUFFIX}"
NON_LATIN_FLAG = "has_non_latin_name"
# Same reading, as a string, for when a frame has to go through a non-pandas
# boundary (a TSV, a dict, a Colab cell).
INDIA_SCRIPT_FLAG = "has_indian_script_name"

# The scripts that mean "written in a script that is not Latin". Everything
# else the profiler tracks -- Latin (accented), General punctuation, symbol
# and emoji blocks -- is Latin-ish text for the purpose of this flag, and the
# distinction is what `is_strictly_non_latin` is for.
LATIN_SCRIPT = "Latin (accented)"
INDIC_SCRIPTS: frozenset[str] = frozenset(
    {
        "Devanagari", "Bengali", "Gurmukhi", "Gujarati", "Oriya",
        "Tamil", "Telugu", "Kannada", "Malayalam",
    }
)


# --------------------------------------------------------------------------
# 1. script detection
# --------------------------------------------------------------------------
#
# Codepoint ranges come from profile_data.SCRIPT_RANGES, imported above, and
# the per-character tag table is built from those same ranges here. Two
# consequences worth stating, because they are what keeps the profiler and
# this module honest:
#
#   - the ranges are defined once, in one file, so a script cannot be added to
#     the profiler's report and quietly be missing from the flag;
#   - "first match wins per character" is the profiler's rule (its ranges are
#     ordered most-specific-first) and is reproduced exactly, so a character
#     that two blocks claim is classified identically on both sides.
#
# Difference from the profiler, and it is deliberate: the profiler's tag table
# stops at the BMP and counts non-BMP characters separately, because a report
# is a table of blocks and did not want astral rows in it. `detect_scripts`
# classifies the whole of Unicode that the ranges mention, astral planes
# included, because a caller asking "what script is this" should get an answer.
# The only measurable consequence is that a value whose *only* non-ASCII
# characters are outside the BMP would be flagged here and not counted as
# non-ASCII by the profiler -- and the profile measured zero such rows in
# `business_name` across all 26.4M rows, so the two agree on this data. The
# `test_flag_agrees_with_the_profiler` test pins that agreement down.
_DETECTABLE_RANGES: tuple[tuple[int, int], ...] = tuple(
    lo_hi for _name, ranges in SCRIPT_RANGES for lo_hi in ranges
)
# Same ordering the profiler uses, so "first match wins" picks the same block.
_SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = SCRIPT_RANGES


def _build_char_table() -> dict[int, str]:
    """codepoint -> script name, over every range the profiler tracks."""
    table: dict[int, str] = {}
    for script, ranges in _SCRIPT_RANGES:
        for lo, hi in ranges:
            for cp in range(lo, hi + 1):
                # setdefault, not assignment: earlier (more specific) blocks win.
                table.setdefault(cp, script)
    return table


# The exhaustive table is ~150k entries, which is fine to build once but not
# to do per call, so it is built at import time and shared. Lookup is a dict
# hit; a codepoint outside every range falls through to the unclassified tag.
_CHAR_SCRIPT: dict[int, str] = _build_char_table()


def _build_pattern(scripts: Iterable[str]) -> re.Pattern[str]:
    """
    One compiled character class covering exactly the given scripts' ranges.

    This is the vectorised twin of `detect_scripts`: the flag column is computed
    on a whole pandas Series at a time, and a per-row Python loop over six
    columns of 26M rows is not affordable. The ranges are identical to the
    ones the dict above is built from, so the two agree by construction --
    `test_series_flag_matches_per_row_detection` asserts it.
    """
    wanted = set(scripts)
    spans: list[str] = []
    for script, ranges in _SCRIPT_RANGES:
        if script not in wanted:
            continue
        for lo, hi in ranges:
            spans.append(_codepoint_span(lo, hi))
    if not spans:
        return re.compile(r"(?!)")  # matches nothing, never raises
    return re.compile(f"[{''.join(spans)}]")


def _codepoint_span(lo: int, hi: int) -> str:
    """A [lo-hi] char-class fragment, using \\u/\\U escapes so it stays ASCII."""
    if hi - lo < 4:  # a literal run is shorter and clearer for tiny ranges
        return "".join(f"\\u{cp:04x}" for cp in range(lo, hi + 1))
    return f"\\U{lo:08x}-\\U{hi:08x}" if hi > 0xFFFF else f"\\u{lo:04x}-\\u{hi:04x}"


# Every script the profiler tracks, i.e. everything a "is this ASCII?" test
# needs to be able to notice. Drives `has_non_latin_name`.
_NON_ASCII_PATTERN = _build_pattern(SCRIPT_NAMES)
# The nine Indic scripts alone. Drives the strict reading of the flag.
_INDIC_PATTERN = _build_pattern(INDIC_SCRIPTS)


def detect_scripts(text: str) -> set[str]:
    """
    Names of the Unicode scripts `text` is written in.

    A value can legitimately be in several at once -- "Café बालाजी Private Ltd"
    is Latin (accented) and Devanagari -- so the result is a set. Pure ASCII
    text returns the empty set, because ASCII is not a tracked script and a
    business called "Prime Money Ltd" carries no script signal at all.

    A character that falls in no tracked range is reported as the profiler's
    unclassified bucket ("Other (non-ASCII, unclassified)") rather than
    dropped: a name in a script nobody thought to list is still a name that
    needs the embedding step, and silently reporting nothing would make this
    look like the plain-ASCII case.

    Non-string input (NaN, None) is treated as blank, because the caller is a
    pandas column.
    """
    if text is None:
        return set()
    if not isinstance(text, str):
        text = str(text)
    if text.isascii():
        return set()
    found: set[str] = set()
    lookup = _CHAR_SCRIPT.get
    for char in text:
        if char.isascii():
            continue
        found.add(lookup(ord(char), NON_ASCII_SCRIPT))
    return found


def is_strictly_non_latin(text: str) -> bool:
    """
    True only if `text` contains an Indian script, ignoring accented Latin.

    This is the reading `has_non_latin_name` does *not* take (see the module
    docstring): "Café Zürich SARL" is non-ASCII but it is not transliteration
    risk, and the embedding step does not need to see it. `detect_scripts`
    distinguishes the two cases for anyone who cares.
    """
    return bool(detect_scripts(text) & INDIC_SCRIPTS)


def flag_non_latin(series: pd.Series) -> pd.Series:
    """
    Vectorised `bool(detect_scripts(value))` over a whole column.

    Same semantics as mapping detect_scripts over the column and asking for
    non-emptiness, at C speed; `test_series_flag_matches_per_row_detection`
    holds the two to each other.
    """
    return series.str.contains(_NON_ASCII_PATTERN, na=False, regex=True)


def flag_indian_script(series: pd.Series) -> pd.Series:
    """Vectorised `is_strictly_non_latin` over a whole column."""
    return series.str.contains(_INDIC_PATTERN, na=False, regex=True)


def add_non_latin_name_flag(
    df: pd.DataFrame,
    name_column: str = FLAG_NAME_COLUMN,
) -> pd.DataFrame:
    """
    Add the `has_non_latin_name` boolean column, computed on `name_column`.

    `name_column` defaults to `business_name_clean`, i.e. this is designed to
    run *after* clean_text.py. It falls back to the raw `business_name` when
    the clean column is absent, so the function is usable on a raw frame.

    Clean-then-flag is not a stylistic preference. `clean_text` strips stray
    accents that are mojibake rather than writing -- "MUSIQUE JÂCQUES" cleans to
    "MUSIQUE J CQUES" -- and that is the difference between a French row being
    sent to the embedding step and not. On a 50k stratified sample of
    test_source2 that is 15 rows, and it is the whole reason the flag has to be
    defined on the cleaned column rather than the raw one.

    Note that the cleaner does *not* remove U+200C ZWNJ, which is legitimate
    Telugu orthography: a real row keeps it. ZWNJ is in no tracked block, so
    `detect_scripts` reports it as unclassified alongside Telugu and the flag
    is True either way.

    A frame with neither column is returned unchanged rather than raising: the
    ground-truth file has no business_name at all, and the pipeline should be
    able to walk every file it is pointed at.
    """
    if name_column in df.columns:
        source = df[name_column]
    elif NAME_COL in df.columns:
        source = df[NAME_COL]
    else:
        return df
    out = df.copy()
    out[NON_LATIN_FLAG] = flag_non_latin(source.fillna("").astype(str))
    return out


def add_indian_script_name_flag(
    df: pd.DataFrame,
    name_column: str = FLAG_NAME_COLUMN,
) -> pd.DataFrame:
    """
    Add the strict-reading companion column `has_indian_script_name`.

    Not required by anything downstream yet. It is here so that "the flag is
    true for the France rows" is a measurement rather than an assumption, and
    so that switching Phase 3b's filter from the broad flag to the strict one
    is a one-word change if the broad flag proves too expensive.
    """
    if name_column in df.columns:
        source = df[name_column]
    elif NAME_COL in df.columns:
        source = df[NAME_COL]
    else:
        return df
    out = df.copy()
    out[INDIA_SCRIPT_FLAG] = flag_indian_script(source.fillna("").astype(str))
    return out


# --------------------------------------------------------------------------
# 2. the state/UT gazetteer
# --------------------------------------------------------------------------
#
# PUBLIC REFERENCE DATA, not a business lookup. `data/indian_state_names.json`
# is the official English names of India's 28 states and 8 Union Territories
# plus their usual spellings in the nine Indic scripts the profiler found in
# this data. It is the same category of artefact as a country list or a
# currency table: published, closed-vocabulary, not derived from the rows
# being processed. The JSON repeats this in its own `methodology_note` so the
# claim travels with the file if it is copied on its own, and the "not a
# business lookup" part is the one that matters for the methodology writeup:
# nothing in it was fitted to, keyed on, or learned from `dataset/`, and no
# entry encodes a match between two entities.
#
# Which entries are attested by the data and which are authored for coverage
# is recorded in the JSON (`attested_in_sample`, `provenance`) rather than
# asserted here, because that is a fact about the data, not about the code.

# Token edges for the address scan. A state name has to stand alone: it may be
# surrounded by commas, semicolons, pipes, slashes, dashes, newlines, tabs or
# spaces, but it may not be glued to a letter or a digit, or "H.NO 4" would
# rewrite the "NO" and "MUMBAI,MAHARASHTRA" would never match. \w is
# Unicode-aware, so the lookbehind already covers all nine Indic scripts'
# letters; the explicit Devanagari span in front of it is belt and braces for
# any Indic codepoint Python's \w does not consider a word character.
_TOKEN_EDGE = r"(?<![\wऀ-ॿ])"
# A cheap pre-test: an ASCII value can only ever match an ASCII gazetteer key,
# and every ASCII key is at least two characters, so a single-letter or
# single-digit ASCII value cannot match anything. It lets the common US row
# skip the 500-way alternation entirely.
_ASCII_STATE_RE = re.compile(r"[A-Za-z]{2}")


@lru_cache(maxsize=None)
def load_state_table(path: str | Path = DEFAULT_STATE_TABLE) -> dict[str, str]:
    """
    Read the gazetteer into a flat `{name_as_it_appears: canonical_english}`.

    The table is read once and cached; the lookup below is a single compiled
    alternation over every key, so canonicalizing a column of addresses is one
    regex pass per value, not a scan of 500 keys.

    Lookup keys are case-folded, with one exception: the ISO 3166-2:IN suffix
    codes ("MH", "KA", "TN" ...) are matched case-sensitively in upper case
    only. Folding them would put "in", "as", "is", "it", "no", "or", "so", "up"
    and "on" in the gazetteer, and those are ordinary words. A code therefore
    has to arrive as the upper-case token it is.
    """
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    lookup: dict[str, str] = {}
    for unit in data["states"]:
        canonical = unit["canonical"]
        for script, names in unit["names"].items():
            for name in names:
                if script == "en" and name.isascii() and len(name) <= 2:
                    lookup.setdefault(name, canonical)  # case-sensitive code
                else:
                    lookup.setdefault(name.casefold(), canonical)
    return lookup


def _ascii_name_fragment(name: str) -> str:
    """
    A regex fragment matching `name` in any ASCII casing, and only in ASCII casing.

    Addresses write the state as "Maharashtra", "maharashtra" or "MAHARASHTRA",
    and sometimes as something in between, so a single-cased alternation misses
    real rows. Two obvious fixes are both wrong:

      * `re.IGNORECASE` is Unicode-aware, so it would let U+212A KELVIN SIGN
        match "k" and U+017F LATIN SMALL LETTER LONG S match "s" -- a name
        would be rewritten on the strength of a codepoint nobody typed;
      * enumerating `.lower()`/`.upper()` variants covers the three common
        casings and silently misses "MaHaRaShTrA".

    Spelling each ASCII letter out as a two-character class is case-insensitive
    by construction, cannot match a lookalike from another block, and costs
    about the same as the three-variant alternation it replaces. Non-ASCII
    letters are left alone: the nine Indic scripts in the gazetteer are
    caseless, and the one that is not (Latin) is handled by this branch.
    """
    out = []
    for char in name:
        if char.isascii() and char.isalpha():
            out.append(f"[{char.lower()}{char.upper()}]")
        else:
            out.append(re.escape(char))
    return "".join(out)


def _build_state_pattern(
    keys: Iterable[str],
    *,
    skip: frozenset[str] = frozenset(),
    case_sensitive: frozenset[str] = frozenset(),
) -> re.Pattern[str]:
    """
    One alternation over the given gazetteer keys, longest first.

    Longest-first is not an optimisation, it is a correctness requirement:
    Python's alternation is first-match-wins, so "Uttar Pradesh" has to be
    tried before "UP" can matter and "Dadra and Nagar Haveli and Daman and
    Diu" before "Dadra and Nagar Haveli". Sorting by descending length gives
    the longest key that matches at a position, which is what a reader expects
    from a gazetteer.

    Keys in `skip` are left out, for the caller that knows the row is not
    Indian. Keys in `case_sensitive` -- the two-letter codes -- are matched
    exactly, because case-folding them would put ordinary English words in the
    gazetteer (see `load_state_table`).
    """
    fragments: set[str] = set()
    for key in keys:
        if key.casefold() in skip:
            continue
        if key in case_sensitive:
            fragments.add(re.escape(key))
        else:
            fragments.add(_ascii_name_fragment(key))
    ordered = sorted(fragments, key=lambda n: (-len(n), n))
    return re.compile(
        _TOKEN_EDGE + "(?:" + "|".join(ordered) + ")" + r"(?!\w)"
    )


_STATE_TABLE_PATH = DEFAULT_STATE_TABLE

# Gazetteer keys that are only expanded on rows whose country is India.
#
# Two classes, and the first one is a class rather than a list because every
# member of it is unsafe:
#
# 1. Every bare two-letter code. "AR"/"LA"/"MN"/"TN" are also US postal
#    abbreviations (Arkansas, Louisiana, Minnesota, Tennessee) against
#    Arunachal Pradesh, Ladakh, Manipur and Tamil Nadu. But the problem is
#    wider than the four that collide with a state name: Ohio and Pennsylvania
#    addresses are full of "TR 253" township routes, "UP RIVER RD" and
#    "Bldg CH". Any two upper-case letters can be a US abbreviation or a road
#    name, so the whole class is gated rather than an allow-list of the four
#    that happen to collide with a state.
#
# 2. "Delhi", which is a city in New York and a street name in half the
#    country ("15165-A Delhi Avenue, Parker, CO").
#
# This is not hypothetical. Expanding codes with no country guard, on a 300k-row
# stratified sample of this dataset, rewrote 9,386 US rows and 1,082 France
# rows: "1111 Church Street, Unit 2007, Nashville, TN" became
# "... Nashville, Tamil Nadu" and "2103 Milwaukee Avenue, Minneapolis, MN"
# became "... Minneapolis, Manipur". Gating the code class takes that to 61
# rows; gating Delhi as well takes it to 0.
#
# The keys stay in the table. They are right for Indian addresses, and an
# Indian row with a blank country simply keeps "TN" rather than gaining a
# wrong state name.
AMBIGUOUS_NAMES: frozenset[str] = frozenset({"Delhi"})
INDIA = "India"


@lru_cache(maxsize=None)
def _state_matcher(
    path: str,
) -> tuple[re.Pattern[str], re.Pattern[str], dict[str, str], dict[str, str]]:
    """
    Compile the gazetteer once per path.

    Returns two patterns and the two lookups behind them:

      * `with_ambiguous` -- every key, for rows known to be Indian;
      * `unambiguous`    -- the same minus every two-letter code and
                            `AMBIGUOUS_NAMES`, for every other row.

    `exact` holds the case-sensitive two-letter codes; everything else is
    matched case-insensitively through a case-folded `lookup`.
    """
    lookup = load_state_table(path)
    exact = {
        name: canonical
        for name, canonical in lookup.items()
        if name.isupper() and name.isascii() and len(name) <= 2
    }
    keys = list(lookup)
    skip = frozenset(k.casefold() for k in exact) | frozenset(
        n.casefold() for n in AMBIGUOUS_NAMES
    )
    return (
        _build_state_pattern(keys, case_sensitive=frozenset(exact)),
        _build_state_pattern(keys, skip=skip, case_sensitive=frozenset(exact)),
        lookup,
        exact,
    )


def canonicalize_state(
    text: str,
    table_path: str | Path = _STATE_TABLE_PATH,
    country: str | None = None,
) -> str:
    """
    Replace every Indian state/UT name in `text` with its canonical English
    name. Anything the gazetteer does not know passes through untouched.

    Only whole tokens are replaced, and only the matched span: the separators
    around it, the street, the city, the PIN and any non-ASCII text that is
    not a state name all come back byte-identical. That is the contract -- an
    address column that lost its street half would be worse than one that
    still says "महाराष्ट्र".

        canonicalize_state("NAGPUR, महाराष्ट्र")      -> "NAGPUR, Maharashtra"
        canonicalize_state("MOHALI, ਪੰਜਾਬ")          -> "MOHALI, Punjab"
        canonicalize_state("VADODARA, ગુજરાત")        -> "VADODARA, Gujarat"
        canonicalize_state("2621 Cotten Road, TX")    -> unchanged

    English names match case-insensitively ("PUNE, MAHARASHTRA" ->
    "PUNE, Maharashtra") but bare two-letter codes match in upper case only,
    as described on `load_state_table`.

    `country` gates a small set of keys that mean something else outside India,
    and callers that have the column should always pass it: "AR", "LA", "MN" and
    "TN" are Indian state codes *and* US postal abbreviations, and "Delhi" is a
    city in New York. On a non-Indian row those are left alone and everything
    else -- including every unambiguous full name -- is still expanded. Without
    `country` they are expanded, which is right for an Indian address and wrong
    for a US one; see `AMBIGUOUS_KEYS`.

    The function is idempotent: the canonical English form is itself a gazetteer
    key, so running it twice gives the same string as running it once.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    # Latin-only, non-tokenizable text is the overwhelming majority of US rows
    # and must not pay for a 500-way alternation.
    if text.isascii() and not _ASCII_STATE_RE.search(text):
        return text

    with_ambiguous, unambiguous, lookup, exact = _state_matcher(str(table_path))
    pattern = with_ambiguous if country is None or country == INDIA else unambiguous

    def replace(match: re.Match[str]) -> str:
        found = match.group()
        canonical = exact.get(found)
        if canonical is None:
            canonical = lookup.get(found.casefold())
        # A key with nothing to rewrite (the canonical form itself) is
        # returned unchanged, which is what makes the pass idempotent.
        return canonical if canonical is not None else found

    return pattern.sub(replace, text)


def add_canonical_state(
    df: pd.DataFrame,
    address_column: str = CANONICAL_ADDRESS_COLUMN,
) -> pd.DataFrame:
    """
    Add `business_address_canonical`: the cleaned address with every state/UT
    name rewritten to its canonical English name.

    This is the address-side output and it is a *normalization*, not a
    translation: the street, city, district and PIN are untouched, and no
    name anywhere in the frame is altered. Frames without the address column
    are returned unchanged.
    """
    if address_column in df.columns:
        source = df[address_column]
    elif ADDRESS_COL in df.columns:
        source = df[ADDRESS_COL]
    else:
        return df
    # Passed per row so that the keys in AMBIGUOUS_KEYS are only expanded on
    # Indian rows. Without this the US and France rows get "TN" rewritten to
    # "Tamil Nadu", "DELHI, NY" rewritten to "Delhi, NY", and so on.
    if COUNTRY_COL in df.columns:
        countries = df[COUNTRY_COL].fillna("").astype(str)
        canonical = [canonicalize_state(text, country=country) for text, country in zip(source, countries)]
    else:
        canonical = source.fillna("").astype(str).map(canonicalize_state)
    out = df.copy()
    out[CANONICAL_ADDRESS_OUTPUT] = canonical
    return out


def add_multilingual_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    The whole pass in one call, in the order the pipeline wants:

    1. `business_address_canonical` -- state/UT names folded to English.
    2. `has_non_latin_name` -- the Phase 3b filter flag.
    3. `has_indian_script_name` -- the strict reading, for comparison.

    Non-destructive, like clean_dataframe: the raw and cleaned columns are all
    still there, and a similarity feature is free to want the native script
    back.
    """
    out = add_canonical_state(df)
    out = add_non_latin_name_flag(out)
    return add_indian_script_name_flag(out)


# --------------------------------------------------------------------------
# 3. self-check
# --------------------------------------------------------------------------
#
# The flag is a number someone will quote in a writeup ("11-19% of S2/S3 names
# are non-Latin", "~3.4M rows to embed"), so it gets measured against real
# data rather than asserted. Three things are checked, per file:
#
#   a. the flag rate agrees with the profiler's own "Any non-ASCII" row count
#      on the *same* sample -- the two must be equal, because they are defined
#      over the same ranges;
#   b. it agrees with the checked-in report's per-file percentage to within
#      TOLERANCE_POINTS, which is the claim that would actually be embarrassing
#      to get wrong;
#   c. S1 stays near zero, which is what justifies embedding all of it anyway.
#
# The sample is stratified by byte offset, not taken from the head of each
# file. The source files happen to be pre-shuffled -- the country mix of the
# first 100k rows of test_source1 matches its full-file mix to 0.3 points -- so
# a head sample would have been fine, but a check that is only valid because
# of an unstated property of the input is not a check.

# Per-file "business_name non-ASCII" from docs/data_profile_report.md, and the
# row count it was measured on. Kept here as literals on purpose: the point is
# to compare this module's output against the *published* number, not against
# a recomputation of it by the same code.
REPORTED_NON_ASCII_PCT: dict[str, float] = {
    "train_source1.tsv": 0.00,
    "train_source2.tsv": 15.19,
    "train_source3.tsv": 11.48,
    "test_source1.tsv": 2.35,
    "test_source2.tsv": 18.99,
    "test_source3.tsv": 14.51,
}
REPORTED_ROWS: dict[str, int] = {
    "train_source1.tsv": 2_206_821,
    "train_source2.tsv": 5_034_616,
    "train_source3.tsv": 5_285_603,
    "test_source1.tsv": 1_732_544,
    "test_source2.tsv": 4_887_273,
    "test_source3.tsv": 5_082_316,
}
# A stratified sample of ~50k rows is not a census: on a 16% rate the binomial
# standard error is ~0.17 points, but the files are templated rather than
# i.i.d., so real sampling error is larger than that. 2.0 points is a tight
# enough band to catch a detection bug (which moves the number by whole points,
# not fractions) and loose enough not to cry wolf.
TOLERANCE_POINTS = 2.0
# S1 is Latin-only apart from the accented-Latin France rows, so the flag on
# S1 has to stay small. 2.35% is test_source1's published figure.
S1_CEILING_PCT = 3.0
CHUNKSIZE = 50_000


@lru_cache(maxsize=None)
def _header_of(path: str) -> list[str]:
    """The file's own column names, read from byte 0."""
    with Path(path).open("rb") as handle:
        return handle.readline().decode("utf-8", "replace").rstrip("\r\n").split("\t")


def _read_stratum(path: Path, start_fraction: float, rows: int) -> pd.DataFrame:
    """
    `rows` data lines from `start_fraction` of the way into `path`.

    Seeks in bytes because that is the only way to sample the middle of a
    500 MB file without reading the first half of it, and then reads exactly
    `rows` lines forward from there -- not the rest of the file, which is the
    obvious way to get this wrong and silently turns a stratified sample into
    `n_strata` copies of the same block.

    The seek lands mid-line more often than not, so the first line is always
    discarded: at fraction 0 that is the header, and at every other fraction it
    is a fragment. The names come from the file's own header line (read
    separately, at byte 0) and are passed as `names=` with `header=None`, so
    the first sampled data row is data rather than a header row -- which is
    both correct and one row of every stratum.

    Sampling by line count is safe here only because the profile confirmed
    every file parses as a strict 4-column TSV with no malformed lines, i.e.
    no field contains a raw newline.
    """
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(int(size * start_fraction))
        handle.readline()  # partial line, or the header at fraction 0
        lines = [handle.readline() for _ in range(rows)]
    lines = [line for line in lines if line.strip()]
    if not lines:
        return pd.DataFrame(columns=_header_of(str(path)))
    frame = pd.read_csv(
        io.BytesIO(b"".join(lines)),
        sep="\t",
        header=None,
        names=_header_of(str(path)),
        dtype=str,
        keep_default_na=False,  # "NULL"/"nan" are literal text, not NA
        na_values=[],
        encoding="utf-8",
        encoding_errors="replace",
    )
    frame["source_file"] = path.name
    return frame


def read_stratified_sample(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    rows: int = 50_000,
    strata: int = 5,
    splits: Sequence[str] = ("train", "test"),
) -> pd.DataFrame:
    """
    ~`rows` rows per source file, spread evenly from the start of the file to
    the end. `strata` equally spaced byte offsets, `rows // strata` lines from
    each.
    """
    if strata < 1:
        raise ValueError("strata must be >= 1")
    per_stratum = max(1, rows // strata)
    frames: list[pd.DataFrame] = []
    for split in splits:
        for path in sorted((Path(dataset_dir) / split).glob("*_source*.tsv")):
            for i in range(strata):
                fraction = i / strata
                frames.append(_read_stratum(path, fraction, per_stratum))
    return pd.concat(frames, ignore_index=True)


def _measure(df: pd.DataFrame) -> dict[str, Any]:
    """
    Flag counts for one file, next to the profiler's own count.

    The profiler is pointed at the *cleaned* name column on purpose. The flag is
    defined on `business_name_clean`, so comparing it against a profile of the
    raw name is not an apples-to-apples test: `clean_text` deletes stray accents
    that are mojibake rather than writing -- "MUSIQUE JÂCQUES" cleans to
    "MUSIQUE J CQUES" -- so a handful of raw rows are non-ASCII and their
    cleaned value is not. That gap is the flag working correctly, and it is
    checked separately against the published raw rates.
    """
    from .profile_data import profile_dataframe

    names = df[FLAG_NAME_COLUMN]
    profile = profile_dataframe(df, text_columns=(FLAG_NAME_COLUMN,)).col(FLAG_NAME_COLUMN)
    if profile is None:  # the column is missing entirely, so nothing to compare
        raise KeyError(f"{FLAG_NAME_COLUMN} is not in the frame handed to _measure")
    total = len(df)
    return {
        "rows": total,
        "flag": int(df[NON_LATIN_FLAG].sum()),
        "indian": int(df[INDIA_SCRIPT_FLAG].sum()),
        "profiler_non_ascii": profile.non_ascii_rows,
        "profiler_non_bmp": profile.non_bmp_rows,
        "profiler_scripts": profile.script_rows,
    }


def _fmt_pct(count: int, rows: int) -> str:
    return f"{100.0 * count / rows:6.2f}%" if rows else "  n/a"


def _self_check(dataset_dir: Path, rows: int, strata: int) -> int:
    from .clean_text import clean_dataframe

    print(
        f"stratified sample: {rows:,} rows/file, {strata} strata, from {dataset_dir}",
        flush=True,
    )
    started = time.perf_counter()
    sample = read_stratified_sample(dataset_dir, rows=rows, strata=strata)
    print(f"  read {len(sample):,} rows in {time.perf_counter() - started:.1f}s", flush=True)

    processed = add_multilingual_columns(clean_dataframe(sample))
    print("  cleaned + flagged in {:.1f}s\n".format(time.perf_counter() - started))

    header = (
        f"{'file':<22}{'rows':>9}{'flag':>9}{'flag %':>9}"
        f"{'report %':>10}{'delta':>8}{'strict %':>9}"
    )
    print(header)
    print("-" * len(header))
    failures: list[str] = []
    for name, group in processed.groupby("source_file", sort=True):
        m = _measure(group)
        measured_pct = 100.0 * m["flag"] / m["rows"] if m["rows"] else 0.0
        reported = REPORTED_NON_ASCII_PCT.get(name)
        delta = measured_pct - reported if reported is not None else float("nan")
        strict_pct = 100.0 * m["indian"] / m["rows"] if m["rows"] else 0.0
        print(
            f"{name:<22}{m['rows']:>9,}{m['flag']:>9,}{measured_pct:>8.2f}%"
            f"{reported if reported is not None else float('nan'):>9.2f}%"
            f"{delta:>+8.2f}{strict_pct:>8.2f}%"
        )
        # (a) detection and profiling are the same measurement, so they must be
        #     equal -- not close, equal.
        if m["flag"] != m["profiler_non_ascii"]:
            failures.append(
                f"{name}: flag says {m['flag']:,} non-ASCII rows, "
                f"profiler says {m['profiler_non_ascii']:,} on the cleaned column"
            )
        if m["profiler_non_bmp"]:
            # The profiler's tag table stops at the BMP, so a non-BMP character
            # survives translate() unchanged and the profiler cannot see it. The
            # flag pattern does cover it, so equality is expected to break by
            # exactly these rows; say so rather than reporting a mystery.
            failures.append(
                f"{name}: {m['profiler_non_bmp']:,} rows contain non-BMP "
                f"characters, which the profiler cannot tag -- the flag/profiler "
                f"equality check is not meaningful here"
            )
        if reported is None:
            failures.append(f"{name}: no published percentage to check against")
        elif abs(delta) > TOLERANCE_POINTS:
            failures.append(
                f"{name}: {measured_pct:.2f}% is {delta:+.2f} points from the "
                f"published {reported:.2f}% (tolerance {TOLERANCE_POINTS})"
            )
        if name.endswith("source1.tsv") and measured_pct > S1_CEILING_PCT:
            failures.append(
                f"{name}: S1 flag rate {measured_pct:.2f}% exceeds "
                f"{S1_CEILING_PCT}% -- S1 is supposed to be Latin-only"
            )

    print("-" * len(header))
    if failures:
        print("FAIL:")
        for line in failures:
            print(f"  !! {line}")
        return 1
    print(
        f"PASS: flag == profiler on every file, and within {TOLERANCE_POINTS} "
        f"points of docs/data_profile_report.md"
    )
    print(
        "  'strict %' is the Indic-scripts-only reading: the gap between the "
        "two columns is accented Latin, which is all of test_source1's share."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--rows", type=int, default=50_000,
                        help="rows per source file; the sample is bounded on purpose")
    parser.add_argument("--strata", type=int, default=5,
                        help="equally spaced byte offsets to sample from")
    args = parser.parse_args(argv)
    return _self_check(args.dataset_dir, args.rows, args.strata)


if __name__ == "__main__":
    raise SystemExit(main())
