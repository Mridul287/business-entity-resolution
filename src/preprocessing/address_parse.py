# Raw string, not a plain one: the docstring below quotes a regex with
# backslashes in it, and a plain docstring warns on every import.
r"""
Owner: Person A

Phase 4: address parsing. Turns `business_address_clean` (Phase 1) into the
structured sub-tokens that Phase 5 blocking and the Phase 6 feature builder
need, and expands the abbreviations that stop two sources writing the same
place two ways.

    expand_abbreviations(text, scope=...) -> str
    extract_address_tokens(address, country=None) -> dict

Everything here is designed to run *after* clean_text.py, and the order is not
arbitrary. clean_text has already repaired mojibake, removed the placeholder
tokens and the bracket decoration, and NFKC-normalized. So "Clm Agro [Pvt] Ltd"
arrives here as "Clm Agro Pvt Ltd", and the abbreviation pass can treat "Pvt" as
a plain whole token instead of a decorated one.

`country` is used only to pick a postal-code rule, never to hard-code a closed
set of countries. France appears in test and not at train, and the README is
explicit that country is an open set of string labels. Unknown country => the
generic rule, which is also what France gets.

--------------------------------------------------------------------------
Every abbreviation in here was measured before it was written down
--------------------------------------------------------------------------
The rule that decided the table: **an abbreviation only needs expanding if some
other row spells the same thing out in full.** If only one spelling exists in
the corpus, it is already a consistent key, and expanding it buys nothing while
risking a wrong rewrite. That is why `sarl` (1,949 name occurrences), `sas`
(1,475), `eurl` (525) and `gmbh` are left alone: there is no long form of them
in the data to fold onto, and rewriting a French legal form into English would
be a transliteration claim this repo has already decided not to make
(see multilingual.py). The same reasoning drops `assn` and `mgmt`, which do
not occur in the sample at all.

Counts are standalone-token occurrences over a stratified sample of all six
source files (20,000 rows per file per stratum, 4 strata), over `business_address`.

  token  addr occurrences  share    verdict
  ----  ----------------  -------  ----------------------------------------
  no     23,159            2.39%    -> number    (address)
  st     10,406            1.07%    -> street,   CONTEXTUAL -- see below
  rd      9,194            0.95%    -> road      (address)
  ave     2,982            0.31%    -> avenue    (address)
  c/o     2,858            0.30%    -> care of   (address)
  nr        990            0.10%    -> near      (address)
  fl      1,011            0.10%    -> floor     (address)
  co      1,148            0.12%    -> NOT expanded in an address, see below
  pvt        174            0.02%    -> private   (both scopes)
  corp        43            0.004%   -> corporation (both scopes)

`no` is the highest-value entry in the table at 2.39% of every token in every
address in the corpus, and it is unambiguous: its contexts are "flat no 101 1st
floor", "plot no 7 & 8", "sr no 222/1", "ward no 1". It is safe *because*
matching is whole-token -- "Noida" is one token and is never touched.

--------------------------------------------------------------------------
Deliberately NOT expanded, with the measurement that rules each one out
--------------------------------------------------------------------------
This is the part of the table that matters most, because each of these looks
like an obvious win and is not.

  dr     3,773 occurrences. Measured: "264 265 DR ANNIE BESANT RD", "lane off
          DR E MOSES ROAD". It is **Doctor**, not Drive. Expanding it would
          rewrite two of this corpus's most common person-references.
  co     1,148 occurrences. Measured: "drive pueblo west CO", "park city summit
          CO UT", "four square mile CO". It is the **US state abbreviation for
          Colorado**. Safe in a name (5,208 occurrences, no such meaning),
          unsafe in an address -- hence the `scope` argument.
  ct     1,504 occurrences. Measured: "street new haven CT", "avenue new britain
          CT". It is **Connecticut**, not Court.
  ter      124 occurrences. Measured: "3 ter pl de la victoire" (French
          *terre*) against "1119 klickitat ter oak harbor wa" (US *Terrace*).
          50/50, and the French reading is not even the same word.
  av         232 occurrences. Measured: "cantonment bank kesh av nagar",
          "a 3/50 av ph aya nagar" -- Indian locality shorthand, not "avenue".
  ab / sa / lp / ms    36 / 36 / 58 / 15 occurrences. Measured as something
          else: "AB ROAD" is a real road in Indore, "SA" an Indian locality.
  n/s/e/w/m/ne/nw/se/sw    1,459 / 2,548 / 1,744 / 678 / 1,230 / 212 / 10 / 29.
          Single letters are French elisions ("rue du G N RAL DE GAULLE",
          "rue AM D E SAINT GERMAIN") and US state abbreviations ("douglas
          county NE") and Indian locality shorthand. Never expandable.

`st` needs its own rule, because it is genuinely both "Street" and "Saint" and
the split is measurable. Classifying all 10,406 occurrences by what sits
immediately after the token in the raw string:

    followed by "," or end of string   6,072   58.3%   -> Street
    followed by a space and a word     3,887   37.4%   -> mostly Street too
    followed by an attached character    425    4.1%   -> ordinal ("1st"), left

The 3,887 "space + word" cases are dominated by street and floor vocabulary --
FLOOR (1,663, from "1ST FLOOR"), STREET (297), MAIN (230), CROSS (179), ST
(156), AVENUE (146), FLR (94), AVE (85), BLOCK (62), LANE, PLACE, RTE -- with a
much smaller genuine Saint class: GEORGE (49), CHARLES (38), LOUIS (36),
PAUL (22). So "Saint" is the exception here, not the rule, which is the
opposite of the usual assumption and is why it is written down.

The rule that follows is deliberately narrow:

  * `st` immediately before ",", ";" or end of string  -> Street  (unambiguous)
  * `st` immediately before a street/floor-type word  -> Street  (measured)
  * `st` immediately before anything else, i.e. a name -> left exactly as it was

A "Saint-Nazaire" row keeps its "st" token, and its twin written "Saint
Nazaire" also keeps "saint" -- the two still agree with each other, because
the rule never *guesses*. What the rule buys is the case blocking actually
needs: "KENTUCKY ST" and "KENTUCKY STREET" folding onto one token.

--------------------------------------------------------------------------
The postal code finding, which reshapes Phase 5
--------------------------------------------------------------------------
`pin_or_zip` is requested here, so it is here -- but the measurement says it is
close to absent from this corpus, and a pre-filter that trusts it would delete
most of the true matches. Classifying every digit run in a stratified sample of
all six files (30,000 rows per file per stratum, 4 strata):

    country   rows     has ANY 5-digit run   5-digit in last 2 tokens
    France    13,216            0.49%                       0.00%
    India     78,242            0.87%                       0.01%
    US        88,542           10.67%                       0.03%

    country   rows     has ANY 6-digit run   6-digit in last 2 tokens
    France    13,216            0.01%                       0.00%
    India     78,242            0.01%                       0.00%
    US        88,542            0.96%                       0.00%

Zero. Not "rare": in the last position where a US ZIP or an Indian PIN
actually sits, a postal code appears in 0.00-0.03% of rows. The real address
tails show why -- this corpus writes US addresses as

    "1795 Westchester Drive, High Point, NC"
    "1712 Montebello Avenue, Phoenix, AZ"
    "337 Oakland Avenue, Michigan City, IN"

and Indian ones as

    "..., Bhandup West, Mumbai, Maharashtra"
    "..., Sarita Vihar, Delhi, South Delhi, Delhi"

and French ones as

    "..., Dunkerque, Hauts-de-France"
    "..., La Teste-de-Buch, 5 bis Rue Pierre Dignac"

-- state or region last, postal code omitted, in every case. The US 10.67% of
"any 5-digit run" is *house and street numbers* ("17560 Ellis Road"), so a
naive `\b\d{5}\b` does not extract a ZIP, it extracts the street number, and
every "the PINs differ, reject this pair" rule built on it fires constantly and
wrongly.

So `pin_or_zip` is extracted, and it is returned **with a `postal_kind` that
says how much it is worth**:

    "us_zip5"  a 5-digit run in the trailing "..., CITY, ST 12345" position
    "in_pin6"  a 6-digit standalone run, country India
    "generic"  the fallback: any standalone digit run of 1-10 digits
    None       nothing found

`generic` is a last resort, not a postal code. On the French sample it fires
almost exclusively on a **house number**, because France has no postal code in
this corpus and the most common digit-bearing token in a French address is a
1-3 digit street number (5,271 two-digit runs, the largest digit bucket of any
country). That is exactly what the fallback is for -- "do not silently return
None for an open country label" -- and exactly why Phase 5 must gate on
`postal_kind`, never on `pin_or_zip` alone. The test suite asserts the France
row is non-None *and* that it is marked `generic`.

Usage:
    python -m src.preprocessing.address_parse                        # self-check
    python -m src.preprocessing.address_parse --rows 200000
"""
from __future__ import annotations

import argparse
import re
import time
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .multilingual import read_stratified_sample
from .profile_data import ADDRESS_COL, COUNTRY_COL, NAME_COL

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"

# The column this module is designed to read, and the columns it adds.
ADDRESS_INPUT_COLUMN = f"{ADDRESS_COL}_clean"
NAME_INPUT_COLUMN = f"{NAME_COL}_clean"
EXPANDED_SUFFIX = "_expanded"

# Keys of the dict `extract_address_tokens` returns.
PIN_OR_ZIP = "pin_or_zip"
POSTAL_KIND = "postal_kind"
STREET_NUMBER = "street_number"
TOKEN_SET = "token_set"
TOKEN_COLUMNS: tuple[str, ...] = (PIN_OR_ZIP, POSTAL_KIND, STREET_NUMBER, TOKEN_SET)

# The three scopes `expand_abbreviations` understands. The two that differ are
# `co` and nothing else, and the reason is in the module docstring.
SCOPE_BOTH = "both"
SCOPE_ADDRESS = "address"
SCOPE_NAME = "name"
SCOPES: tuple[str, ...] = (SCOPE_BOTH, SCOPE_ADDRESS, SCOPE_NAME)

US = "US"
INDIA = "India"

_WHITESPACE = re.compile(r"\s+")


# --------------------------------------------------------------------------
# 1. abbreviations
# --------------------------------------------------------------------------

# Whole-token, case-insensitive. Built as one compiled alternation per scope for
# the same reason multilingual.py's gazetteer is: a per-token Python loop over
# 26M rows is not affordable, and one regex pass per value is.
#
# The guard on both sides is (?<!\w) / (?!\w). \w is Unicode-aware, so this also
# stops an expansion inside a Devanagari or Kannada word, and it is what keeps
# "Noida" from being read as "no" + "ida" and "Stark" as "st" + "ark".
ADDRESS_ABBREVIATIONS: dict[str, str] = {
    # street types
    "rd": "road",
    "ave": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "hwy": "highway",
    "pkwy": "parkway",
    "pl": "place",
    "sq": "square",
    "expy": "expressway",
    # unit designators
    "apt": "apartment",
    "ste": "suite",
    "fl": "floor",
    "flr": "floor",
    "gr": "ground",
    "grd": "ground",
    "bldg": "building",
    "bld": "building",
    # designators and locality words
    "no": "number",
    "nr": "near",
    "c/o": "care of",
    "sec": "sector",
    "sect": "sector",
    "dist": "district",
    # legal forms that do leak into addresses ("... CHS LTD", "AG BROS CGH")
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
}

# Name-only additions. `co` is the whole difference, and it is a measured one:
# in a name "co" is Company (5,208 occurrences) and in an address it is the US
# postal abbreviation for Colorado (1,148 occurrences).
NAME_ABBREVIATIONS: dict[str, str] = {**ADDRESS_ABBREVIATIONS, "co": "company"}

_BOTH_ABBREVIATIONS: dict[str, str] = {**ADDRESS_ABBREVIATIONS, **NAME_ABBREVIATIONS}
_SCOPE_TABLES: dict[str, Mapping[str, str]] = {
    SCOPE_ADDRESS: ADDRESS_ABBREVIATIONS,
    SCOPE_NAME: NAME_ABBREVIATIONS,
    SCOPE_BOTH: _BOTH_ABBREVIATIONS,
}

# `&` is not an abbreviation but it is the same class of problem: the problem
# statement lists "punctuation differences (& vs. "and")" as an expected noise
# pattern, and "ARCE & BELTRAN PARTNERS" is a real name in the sample. The
# replacement is a word, so it goes in with its own spaces, which is why this
# runs before tokenization rather than inside it.
_AMPERSAND = re.compile(r"\s*&\s*")

# Street/floor vocabulary that, when it directly follows a standalone `st`,
# makes that `st` the word "Street" rather than the name "Saint". Closed set,
# taken from the measurement in the module docstring. Anything not in here
# leaves `st` alone.
_ST_FOLLOWERS = frozenset(
    {
        "street", "st", "road", "rd", "main", "cross", "avenue", "ave", "lane",
        "ln", "drive", "dr", "place", "pl", "block", "floor", "fl", "flr",
        "grd", "ground", "stage", "parkway", "pkwy", "square", "sq", "circle",
        "terrace", "ter", "boulevard", "blvd", "way", "court", "ct", "hwy",
        "expy", "expressway", "rte", "route", "sector", "plot", "shop",
        "building", "bldg", "bld", "nagar", "marg",
    }
)
# The look-behind excludes digits as well as letters, so the ordinal in "1ST
# FLOOR" is not mistaken for a standalone "st". \b would not do this.
_ST_TOKEN = re.compile(r"(?<![A-Za-z0-9])[Ss][Tt](?![A-Za-z])")
_FOLLOWER_WORD = re.compile(r"[A-Za-z]+")
_UPPER = re.compile(r"[A-Z]")


def _compile_abbreviations(mapping: Mapping[str, str]) -> re.Pattern[str]:
    """
    One alternation over a whole-token abbreviation table, longest key first.

    Longest-first is a correctness requirement, not an optimisation: Python's
    alternation is first-match-wins, so `flr` has to be tried before `fl` or
    "OFFICE 105 FLR 1" becomes "OFFICE 105 FLOOR 1".
    """
    keys = sorted(mapping, key=lambda key: (-len(key), key))
    if not keys:
        return re.compile(r"(?!)")
    # The look-behind excludes a dot as well as a word character, and that is
    # not a detail: without it "H.NO 389" rewrites to "H.NUMBER 389", because
    # "." is not a \w and so satisfies a bare (?<!\w). "H.NO", "Gali No." and
    # "Plot No.99" are the most common Indian house-number forms in this
    # corpus, and "No" inside them is part of the label, not the word Number.
    # A trailing dot is still allowed, so "Plot No.99" does expand.
    return re.compile(
        r"(?<![\w.])(?:" + "|".join(re.escape(key) for key in keys) + r")(?!\w)",
        re.IGNORECASE,
    )


_SCOPE_PATTERNS: dict[str, re.Pattern[str]] = {
    scope: _compile_abbreviations(table) for scope, table in _SCOPE_TABLES.items()
}


def _match_case(original: str, replacement: str) -> str:
    """
    `replacement` in the casing of `original`, so "KENTUCKY ST," becomes
    "KENTUCKY STREET," and "Kentucky St," becomes "Kentucky Street,".

    Expansion does not have to be pretty, but an all-caps column that comes back
    half title-case is the kind of thing that makes a reviewer stop trusting the
    output, and a single `.isupper()` test is not worth the inconsistency.

    `replacement` is expected in lower case. Passing "Street" here would make
    this return "Street" for a lower-case token, because there is no branch that
    lower-cases.
    """
    if not original:
        return replacement
    if all(_UPPER.match(char) for char in original if char.isalpha()):
        return replacement.upper()
    if original[0].isupper():
        return replacement.capitalize()
    return replacement


def _is_shouted(text: str) -> bool:
    """
    True for a value written entirely in upper case, ignoring anything that
    cannot be either case.

    Digits and punctuation are skipped deliberately: "12 MG RD" is all-caps in
    the way this corpus writes addresses, while "12 mg rd" is a sentence-case
    value that happens to be short. A cased character has to exist, or a value
    like "42 & 7" would be shouted and come back as "42 AND 7".
    """
    cased = [char for char in text if char.isalpha()]
    return bool(cased) and all(_UPPER.match(char) for char in cased)


def _expand_st(text: str) -> str:
    """
    `st` -> `street` only where the measurement says it is unambiguous.

    Two admitted shapes, both from the numbers in the module docstring: `st`
    before a comma/semicolon/end-of-string (58.3% of all occurrences), and `st`
    before a street or floor word (the bulk of the 37.4% "space + word" class).
    Everything else -- "ST GEORGE", "ST NAZAIRE", "ST LOUIS" -- is left exactly
    as it was, so no row is rewritten on a guess about a saint.

    The replacement is handed to `_match_case` in lower case on purpose. That
    function decides the output case from the token it found, so passing an
    already-capitalized "Street" makes it return "Street" verbatim for a
    lower-case input, and "kentucky st" comes back as "kentucky Street".
    """

    def replace(match: re.Match[str]) -> str:
        tail = text[match.end():]
        stripped = tail.lstrip()
        # A comma, a semicolon or the end of the value: a street type never
        # takes a name after it, so this is the 58.3% class and it is
        # unambiguous. A dot is deliberately NOT in this set -- "ST. GEORGE" is
        # a saint, and putting "." here would rewrite it.
        if stripped == "" or stripped[0] in ",;":
            return _match_case(match.group(), "street")
        # Otherwise only a known street/floor word licenses the rewrite. Dots
        # are stepped over first so "ST, " and "ST. " behave alike.
        follower = _FOLLOWER_WORD.match(stripped.lstrip(". "))
        if follower is not None and follower.group().lower() in _ST_FOLLOWERS:
            return _match_case(match.group(), "street")
        return match.group()

    if not any(form in text for form in ("st", "ST", "St")):
        return text
    return _ST_TOKEN.sub(replace, text)


def expand_abbreviations(text: str, scope: str = SCOPE_BOTH) -> str:
    """
    Rewrite the abbreviations in `text` to their long form. Returns a new
    string; the caller keeps the input, because a similarity feature may
    legitimately want "12 MG RD" as it was written.

    `scope` is `SCOPE_ADDRESS` for an address, `SCOPE_NAME` for a business name
    and `SCOPE_BOTH` (the default) for anything else. It exists because `co` is
    "Company" in a name and the US postal abbreviation for Colorado in an
    address -- a difference measured on this corpus, not assumed. In `SCOPE_BOTH`
    it expands to "company", so pass the explicit scope for addresses.

    Three rules, in this order:

      1. `&` -> "and", because "ARCE & BELTRAN PARTNERS" and "ARCE AND BELTRAN
         PARTNERS" are the same business and only the spelling differs;
      2. whole-token abbreviations, via one alternation per scope;
      3. the contextual `st` rule, which needs the surrounding text and so
         cannot be part of the alternation.

    Non-string input is treated as blank, as everywhere else in this package.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return text
    if scope not in _SCOPE_PATTERNS:
        raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")

    table = _SCOPE_TABLES[scope]
    pattern = _SCOPE_PATTERNS[scope]

    # `&` carries no letters, so `_match_case` has nothing to take the case
    # from and would leave a hardcoded "and" as it is -- which turns an all-caps
    # name into "ARCE and BELTRAN PARTNERS". The case is taken from the value
    # instead: a value with no lower-case letter anywhere is all-caps, and gets
    # "AND". Same reason as every other replacement, and it is one flag for the
    # whole string rather than a guess per match.
    conjunction = " AND " if _is_shouted(text) else " and "
    text = _AMPERSAND.sub(conjunction, text)
    text = pattern.sub(
        lambda match: _match_case(match.group(), table[match.group().lower()]),
        text,
    )
    text = _expand_st(text)
    return _WHITESPACE.sub(" ", text).strip()


# --------------------------------------------------------------------------
# 2. postal code
# --------------------------------------------------------------------------

# A standalone digit run: not glued to a letter, a digit, a slash or a hyphen on
# either side. "6-137A" and "53/1" are house numbers in this corpus and are the
# single most common false positive a bare \d+ produces, so the guard is not
# optional. \b alone would not do it -- a boundary is satisfied between "6" and
# "-".
_DIGIT_RUN = re.compile(r"(?<![0-9A-Za-z/-])(\d{1,10})(?![0-9A-Za-z/-])")

# The US form: a 5-digit run in the trailing "..., CITY, ST 12345" position.
# Anchored to the end of the value with the state code in front of it, which is
# what a USPS address looks like, rather than "anywhere in the string", which
# on this corpus returns a street number 10.67% of the time.
_US_ZIP_TAIL = re.compile(r"(?:^|[\s,])([A-Z]{2})\s+(\d{5})\s*$")

POSTAL_US_ZIP5 = "us_zip5"
POSTAL_IN_PIN6 = "in_pin6"
POSTAL_GENERIC = "generic"
POSTAL_KINDS: tuple[str, ...] = (POSTAL_US_ZIP5, POSTAL_IN_PIN6, POSTAL_GENERIC)


def _extract_postal_code(address: str, country: str | None) -> tuple[str | None, str | None]:
    """
    `(value, kind)` for the postal code in `address`, or `(None, None)`.

    Order of attempts, and why:

      1. US, and the value ends in "ST 12345"  -> `us_zip5`. This is the only
         position where a real US ZIP sits, and per the measurement it is
         non-empty for 0.03% of rows, which is honest: this corpus omits ZIPs.
      2. India, and there is a standalone 6-digit run -> `in_pin6`. Also close
         to empty (0.01%), for the same reason.
      3. anything else -> the longest standalone digit run, 1-10 digits, as
         `generic`.

    `generic` is not a postal code and must not be compared as one. On the
    French rows in the sample it is a street number.
    """
    if not address:
        return (None, None)

    if country == US:
        match = _US_ZIP_TAIL.search(address.upper())
        if match is not None:
            return (match.group(2), POSTAL_US_ZIP5)

    if country == INDIA:
        for run in _DIGIT_RUN.finditer(address):
            if len(run.group(1)) == 6:
                return (run.group(1), POSTAL_IN_PIN6)

    # Generic fallback. Longest run wins, so a 6-digit "520001" is preferred
    # over a 1-digit "5" that happens to sit earlier in the same address.
    best: str | None = None
    for run in _DIGIT_RUN.finditer(address):
        value = run.group(1)
        if best is None or len(value) > len(best):
            best = value
    if best is not None:
        return (best, POSTAL_GENERIC)
    return (None, None)


# --------------------------------------------------------------------------
# 3. tokens
# --------------------------------------------------------------------------

# A hyphen or slash *between two digits* separates, so "6-137A" becomes
# "6 137a" and the two parts stay comparable with a source that wrote them with
# a space. Everywhere else the hyphen is a word separator, so "Hauts-de-France"
# becomes "hauts de france" and matches a source that wrote it with spaces --
# which is the same order-independence property the token set is for.
_DIGIT_COMPOUND = re.compile(r"(?<=\d)[-/]+(?=\d)")
_TOKEN_SPLIT = re.compile(r"[^\w]+", re.UNICODE)
# A run of nothing but separators, e.g. a value that was "---" before cleaning.
_SEPARATOR_ONLY = re.compile(r"^[-/]+$")


def fold_accents(text: str) -> str:
    """
    Strip combining marks, keeping the base letter.

    Accent folding is a normalization step and not a train-fitted assumption:
    accented Latin is 7.8-8.2% of S2/S3 test names (docs/notes.md) and it is
    the whole of test_source1's non-ASCII share, i.e. the France rows, which are
    unseen at train time. "Café" and "Cafe" have to produce the same token or the
    French pairs are unreachable.

    This is *not* transliteration. Only combining marks (U+0300-U+036F) are
    removed, and the Indic scripts in this corpus are precomposed letters that
    do not decompose into marks, so no non-Latin text is affected -- a property
    `test_fold_accents_leaves_indic_alone` holds.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def fold_for_key(text: str) -> str:
    """
    The canonical single-string form a blocking key is built from: accent-folded,
    lowercased, whitespace collapsed, trimmed.

    This is the one place that decides what two spellings of the same name are
    *the same string*, so it is shared rather than reimplemented per consumer.
    It is deliberately not aggressive -- it does not strip legal suffixes, does
    not remove punctuation and does not transliterate. A key that rewrites too
    much produces a confidently wrong match, and Phase 5's recall can be
    measured against real ground truth, so the conservative version is the one
    worth having.

    Non-string input is treated as blank.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if not text:
        return ""
    return _WHITESPACE.sub(" ", fold_accents(text).lower()).strip()


def _tokenize(text: str) -> frozenset[str]:
    """
    The order-independent address token set: lowercased, accent-folded,
    punctuation stripped, deduplicated.

    A set, so "12 MG ROAD, BENGALURU" and "BENGALURU, 12 MG ROAD" are the same
    key. That is the property Phase 5 blocking is built on: the profile and the
    problem statement both confirm that address component order is inconsistent
    across sources, so anything order-sensitive is wrong here.
    """
    if not text:
        return frozenset()
    spaced = _DIGIT_COMPOUND.sub(" ", text)
    folded = fold_accents(spaced).lower()
    tokens = []
    for raw in _TOKEN_SPLIT.split(folded):
        if raw and not _SEPARATOR_ONLY.match(raw):
            tokens.append(raw)
    return frozenset(tokens)


# The pre-comma segment is where a house or street number lives. A designator
# in front of it is fine ("H.NO 389", "Bldg 20", "DOOR NUMBER 9/1"); a locality
# or a name word in front of it is not, because then the first number in the
# segment is not the house number at all. So a prefix token is accepted if it
# is short (a single letter like "H" in "H.NO 389") or if it is a known
# designator.
#
# The allow-list has to contain the *expanded* spellings, because the pipeline
# expands abbreviations before it parses, so "DOOR NO 9/1" arrives here as
# "DOOR NUMBER 9/1" and a list holding only "no" would reject it.
#
# Deliberately absent: sector, sec, district, city, town, village, road, rd,
# street, st, main, cross, layout, colony, society. Those introduce a number
# that belongs to the locality, so "SECTOR 18, GURUGRAM" and "STREET NO. 4 NEW
# CANTT ROAD" both correctly yield None instead of 18 or 4.
_LABEL_TOKEN = re.compile(r"[A-Za-z]+")
_LEADING_NUMBER = re.compile(r"\d[\dA-Za-z/-]*")
_MAX_LABEL_CHARS = 4
_STREET_NUMBER_DESIGNATORS = frozenset(
    {
        "no", "num", "number", "nr", "flat", "plot", "bldg", "bld", "room",
        "shop", "door", "unit", "house", "hsg", "tower", "floor", "fl", "flr",
        "kh", "gh", "sr", "ste", "suite", "apt", "opp", "near", "behind",
        "beside", "c/o", "care", "of",
    }
)


def _extract_street_number(address: str) -> str | None:
    """
    The leading number-like unit of the first comma-separated segment.

    Examples that must work, all real: "12 MG Road" -> "12",
    "Bldg 20 Station Rd" -> "20", "H.NO 389 247" -> "389",
    "3/5 A BLOCK" -> "3/5", "L-1318/38 GROUND FLOOR" -> "1318/38",
    "DOOR NUMBER 9/1 SOVARAM BYSACK STREET" -> "9/1".

    Known limitation: a letter-prefixed unit such as "G-320" yields "320", not
    "G-320". Keeping the alphabetic prefix would collide with words that merely
    end in a number, like "27TH FLOOR", which is a floor and not a house
    number at all.
    """
    if not address:
        return None
    head = address.split(",", 1)[0]
    match = _LEADING_NUMBER.search(head)
    if match is None:
        return None
    for label in _LABEL_TOKEN.finditer(head[: match.start()]):
        text = label.group()
        if len(text) > _MAX_LABEL_CHARS and text.lower() not in _STREET_NUMBER_DESIGNATORS:
            return None
    return match.group()


def _as_text(value: object) -> str:
    """Blank for None and for float NaN, which is what a raw column carries."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return value if isinstance(value, str) else str(value)


def extract_address_tokens(
    address: str,
    country: str | None = None,
) -> dict[str, object]:
    """
    The structured sub-tokens of one address.

    Returns a dict with exactly these keys:

        pin_or_zip     str | None    the postal code, or None
        postal_kind    str | None    "us_zip5" | "in_pin6" | "generic" | None
        street_number  str | None    the leading unit before the first comma
        token_set      frozenset     normalized, deduplicated, order-free

    `country` selects the postal-code rule and is treated as an open set of
    labels: an unknown or missing country gets the generic rule, which is also
    what France gets. Nothing here branches on a closed country list.

    `postal_kind` is the field to branch on, not `pin_or_zip`. See the module
    docstring: on this corpus a postal code appears in 0.00-0.03% of rows in the
    position where one belongs, so a rule that requires two `pin_or_zip` values
    to be equal is a rule that fires on noise -- and where it does fire, the
    `generic` kind means the value is a house number.
    """
    address = _as_text(address)
    country = None if country is None else _as_text(country)
    pin_or_zip, postal_kind = _extract_postal_code(address, country)
    return {
        PIN_OR_ZIP: pin_or_zip,
        POSTAL_KIND: postal_kind,
        STREET_NUMBER: _extract_street_number(address),
        TOKEN_SET: _tokenize(address),
    }


# --------------------------------------------------------------------------
# DataFrame entry point
# --------------------------------------------------------------------------


def add_address_tokens(
    df: pd.DataFrame,
    address_column: str = ADDRESS_INPUT_COLUMN,
    country_column: str = COUNTRY_COL,
) -> pd.DataFrame:
    """
    Add the four `extract_address_tokens` columns to a source frame.

    Reads `business_address_clean` by default, i.e. this is designed to run
    after clean_text.py, and falls back to the raw `business_address` so it is
    usable on a frame that has not been cleaned. A frame with neither column is
    returned unchanged, for the same reason multilingual.py does that: the
    pipeline should be able to walk every file it is pointed at.

    A per-row Python loop, which is normally the wrong trade. It is right here
    because the four outputs are three scalars and a *set*, and because two of
    the four rules are context-dependent -- the `st` rule looks at what follows
    the token, the street-number rule at what precedes the digits. A vectorized
    version would need four passes to match this one's. Measured single-threaded
    cost on the 50k-rows-per-file self-check is reported by `main()`.
    """
    if address_column in df.columns:
        source = df[address_column]
    elif ADDRESS_COL in df.columns:
        source = df[ADDRESS_COL]
    else:
        return df
    if country_column in df.columns:
        pairs: Iterable[tuple[object, object]] = zip(source, df[country_column])
    else:
        pairs = ((text, "") for text in source)

    rows = [extract_address_tokens(text, country) for text, country in pairs]
    out = df.copy()
    for key in TOKEN_COLUMNS:
        out[key] = [row[key] for row in rows]
    return out


def expand_abbreviations_frame(
    df: pd.DataFrame,
    name_column: str = NAME_INPUT_COLUMN,
    address_column: str = ADDRESS_INPUT_COLUMN,
) -> pd.DataFrame:
    """
    Add `business_name_expanded` and `business_address_expanded`.

    Kept separate from `*_norm` deliberately. normalize() folds expansion,
    case-folding and accent-folding together into `*_norm` for the blocking
    keys, but a similarity *feature* may legitimately want the abbreviation
    still spelled short, and destroying it would make that impossible. The
    column this function adds is additive; the input is untouched.
    """
    out = df.copy()
    if name_column in out.columns:
        out[f"{NAME_COL}{EXPANDED_SUFFIX}"] = [
            expand_abbreviations(text, SCOPE_NAME) for text in out[name_column]
        ]
    if address_column in out.columns:
        out[f"{ADDRESS_COL}{EXPANDED_SUFFIX}"] = [
            expand_abbreviations(text, SCOPE_ADDRESS) for text in out[address_column]
        ]
    return out


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------

SELF_CHECK_ROWS = 50_000
SELF_CHECK_STRATA = 5


def _self_check(dataset_dir: Path, rows: int, strata: int) -> int:
    from .clean_text import clean_dataframe

    print(
        f"stratified sample: {rows:,} rows/file, {strata} strata, from {dataset_dir}",
        flush=True,
    )
    started = time.perf_counter()
    sample = read_stratified_sample(dataset_dir, rows=rows, strata=strata)
    print(f"  read {len(sample):,} rows in {time.perf_counter() - started:.1f}s", flush=True)

    parsed_started = time.perf_counter()
    parsed = add_address_tokens(clean_dataframe(sample))
    parse_seconds = time.perf_counter() - parsed_started
    print(f"  parsed {len(parsed):,} addresses in {parse_seconds:.1f}s", flush=True)

    failures: list[str] = []
    header = (
        f"{'file':<22}{'rows':>9}{'pin':>9}{'us_zip5':>9}{'in_pin6':>9}"
        f"{'generic':>9}{'street#':>9}{'tok/row':>9}"
    )
    print("\n" + header)
    print("-" * len(header))
    for name, group in parsed.groupby("source_file", sort=True):
        kinds = group[POSTAL_KIND]
        print(
            f"{name:<22}{len(group):>9,}{int(group[PIN_OR_ZIP].notna().sum()):>9,}"
            f"{int((kinds == POSTAL_US_ZIP5).sum()):>9,}"
            f"{int((kinds == POSTAL_IN_PIN6).sum()):>9,}"
            f"{int((kinds == POSTAL_GENERIC).sum()):>9,}"
            f"{int(group[STREET_NUMBER].notna().sum()):>9,}"
            f"{group[TOKEN_SET].map(len).mean():>9.2f}"
        )
        if not group[TOKEN_SET].map(bool).all():
            failures.append(
                f"{name}: {int((~group[TOKEN_SET].map(bool)).sum())} rows with an empty token set"
            )
        # The invariant this module exists to protect: a non-null pin always
        # carries a kind and a null pin never does, so `postal_kind` can be
        # branched on without a second null check.
        disagree = group[PIN_OR_ZIP].isna() != group[POSTAL_KIND].isna()
        if disagree.any():
            failures.append(
                f"{name}: {int(disagree.sum())} rows where pin_or_zip and postal_kind "
                f"disagree about emptiness"
            )

    print("-" * len(header))
    print(
        f"\n  {len(parsed) / max(parse_seconds, 1e-9):,.0f} addresses/s single-threaded"
        f"  ({parse_seconds / max(1, len(parsed)) * 1e6:.0f} us/row)"
    )
    print(
        "  'generic' is a fallback digit run, not a postal code -- on the France\n"
        "  rows it is a street number. Branch on postal_kind, never on pin_or_zip."
    )
    if failures:
        print("\nFAIL:")
        for line in failures:
            print(f"  !! {line}")
        return 1
    print("\nPASS: token set non-empty everywhere, pin/kind always agree")
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
