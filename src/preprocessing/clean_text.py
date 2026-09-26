"""
Owner: Person A

Noise removal for the text columns, run *before* normalize() and before any
blocking key is built. This is the first pass that is allowed to delete
characters, so the rules here are deliberately conservative: every step must be
justified by a pattern that was actually measured in the profile report
(docs/data_profile_report.md), and anything that could be a real character in a
real business name has to survive.

What the profile found, and the step that answers it:

1. Mojibake. Sampled 4.8M values across the six source files: U+00C2 in 669
   values, U+00E2 in 1,230, and 8,155 values containing some C1 control
   (U+0080-U+009F). The dominant real-world signature is not the textbook
   "â€™" but "Â" + C1 bytes, e.g.

       "Gali No. \xc2\x80\x93 01"          -> "Gali No. \u2013 01"
       "Vsp\xe2\x80\x99S Bhavana"          -> "Vsp's Bhavana"

   Both are UTF-8 bytes that were decoded one layer too few times, so they are
   repaired by re-encoding the candidate sequence to bytes and decoding it as
   UTF-8 -- the inverse of the mistake. A candidate whose bytes are not valid
   UTF-8 is left exactly as it was, which is what protects real accented text:
   "ALLÉE DES HÊTRES", "Café Müller" and "SÃO PAULO" are not valid UTF-8 when
   re-encoded, so they come back untouched. No script is transliterated, so
   Devanagari/Gujarati/Telugu rows are never candidates in the first place:
   their codepoints are all above U+00FF and the repair only ever considers
   U+00C0-U+00FF lead characters.

   One dataset-specific wrinkle: the lead byte of an E2 80 9X sequence (a smart
   quote or dash) sometimes arrives as C2 instead of E2, so the pair C2 80 9X is
   rewritten to E2 80 9X before the round-trip. That single rule recovers the
   en dash, both quote directions, and it is verified against real values in
   the tests.

2. U+FFFD. The replacement character appears where a byte was already lost
   before this code ever saw the row, so the original character is unknowable.
   It becomes a space: a wrongly split token can still share n-grams with its
   twin, whereas a wrongly fused token never matches anything.

3. Placeholder tokens. NULL / N/A / NONE / nan, case-insensitively, as whole
   words only. They are literal junk in the source, not pandas NA values, and
   "NANDAN"/"NULLABLE" style words must not be touched.

4. Junk character runs: "##", "###", "***", "???", "--" and anything else that
   is two or more identical non-alphanumeric characters. Replaced by a space
   rather than deleted, so "OFFICE NO. ##70" cannot turn into "OFFICE NO.70".

5. Square brackets around legal suffixes: "Clm Agro [Limited]" -> "Clm Agro
   Limited". Contents are kept, brackets dropped. Parentheses are left alone on
   purpose -- in addresses they carry landmark meaning ("(Near SBI ATM)"), and
   in names they decorate the legal form, which normalize() is the right layer
   to deal with.

6. NFKC, then whitespace collapse and trim. NFKC runs after the mojibake repair
   so it cannot mangle bytes that are still about to be decoded, and it folds
   the non-breaking spaces that survive as U+00A0. It is verified to leave the
   French and Telugu samples in the tests byte-identical.

clean_dataframe is the DataFrame entry point. Like normalize(), it adds new
"<column>_clean" columns and leaves the raw columns in place: some similarity
features want the raw text, and a cleaning pass should never be destructive.

Usage:
    python -m src.preprocessing.clean_text                  # self-check report
    python -m src.preprocessing.clean_text --dataset-dir dataset --rows 200000
"""
from __future__ import annotations

import argparse
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .profile_data import (
    ADDRESS_COL,
    NAME_COL,
    NOISE_TOKEN_PATTERNS,
    OTHER_TAG,
    profile_dataframe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_COLUMNS: tuple[str, ...] = (NAME_COL, ADDRESS_COL)
CLEAN_SUFFIX = "_clean"
# The columns clean_dataframe() adds, i.e. what the self-check has to profile.
CLEAN_COLUMNS: tuple[str, ...] = tuple(f"{c}{CLEAN_SUFFIX}" for c in DEFAULT_COLUMNS)


# --------------------------------------------------------------------------
# 1. mojibake repair
# --------------------------------------------------------------------------

# Every character in the 0x80-0x9F range, as it can reach us two different ways:
# as the raw C1 control (0x80-0x9F) or as the printable cp1252 character that
# stands for it (0x80 -> EUR, 0x99 -> TM, 0x9C -> oe, ...). Both spellings show
# up in the sources, so a candidate has to accept either.
_CP1252_HIGH = (
    "\u0080\u0081\u0082\u0083\u0084\u0085\u0086\u0087\u0088\u0089"
    "\u008a\u008b\u008c\u008d\u008e\u008f\u0090\u0091\u0092\u0093\u0094"
    "\u0095\u0096\u0097\u0098\u0099\u009a\u009b\u009c\u009d\u009e\u009f"
    "\u0152\u0153\u0160\u0161\u0178\u017d\u017e\u0192\u02c6\u20ac"
    "\u201a\u201e\u2020\u2021\u2026\u2030\u2039\u203a\u2122"
)
# A candidate is a Latin-1 high letter (U+00C0-U+00FF, the byte that starts a
# UTF-8 multi-byte sequence) followed by 1-3 continuation bytes. "é" in a real
# word is a candidate only if the character after it also looks like a
# continuation byte, and then the round-trip rejects it anyway.
_MOJIBAKE_SEQUENCE = re.compile(f"[\u00c0-\u00ff][{_CP1252_HIGH}]{{1,3}}")

# C2 80 9X is E2 80 9X with the lead byte replaced -- 0x93/0x98/0x99 in this data
# are the third bytes of the en dash and of both quote directions. Restoring the
# lead byte is what lets the round-trip below decode them; without it C2 80 is a
# valid (if useless) 2-byte sequence and the trailing byte is orphaned.
_MANGLED_LEAD = re.compile("\u00c2\u0080(?=[\u0090-\u00bf])")

# cp1252 first, then latin-1: the classic "â€™" spelling stores its continuation
# bytes as printable cp1252 characters (U+20AC, U+2122) which latin-1 cannot
# even encode, while this dataset's raw-C1 spelling needs latin-1 because cp1252
# leaves U+0080 undefined. Trying both and keeping the first that decodes means
# neither spelling is special-cased.
_MOJIBAKE_ENCODINGS = ("cp1252", "latin-1")

# Smart punctuation to plain ASCII, so "Vsp’s" and "Vsp's" are the same token.
# Folding these is also what turns a repaired "â€™" into the "'" the profiler's
# examples show. Folds real smart quotes too, which is the point.
#
# Keys are codepoints, not strings: str.translate() looks up ord(char), so a
# str-keyed table would silently never match. None deletes the character.
_SMART_PUNCTUATION: dict[int, str | None] = {
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'", 0x2032: "'",
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"', 0x2033: '"',
    0x2010: "-", 0x2011: "-", 0x2012: "-", 0x2013: "-", 0x2014: "-", 0x2015: "-",
    # Invisible characters that carry no text: deleted rather than spaced out.
    0x00AD: None, 0x200B: None, 0xFEFF: None, 0x200E: None, 0x200F: None,
}

# C1 controls and U+0000-U+001F are never legitimate in this data; U+FFFD is a
# byte that was already lost upstream. All of them become a space so the two
# surviving words do not fuse into one token nobody can match.
_CONTROL_AND_REPLACEMENT = re.compile("[\u0000-\u001f\u007f-\u009f\ufffd]")

# A lead character left over after the round-trip means the continuation bytes
# were destroyed upstream, so the sequence can no longer be decoded. What is
# left is decided by what follows it, which is the only evidence available:
#   - "Â" before anything that is not a lowercase letter is a lost non-breaking
#     space ("Gali No. Â 01") -> deleted. The lowercase-letter guard keeps the
#     real French "Âge" and "Âme" intact.
#   - "Ã" before anything that is not an uppercase letter -> deleted, which
#     keeps the real Portuguese "SÃO PAULO" intact.
#   - "â" before an uppercase letter is a lost apostrophe in a proper noun
#     ("VspâS" -> "Vsp'S") -> becomes "'". Before anything else -> deleted,
#     which keeps real words like "câble" intact.
_STRAY_LEAD = re.compile(
    "\u00c2(?![a-z])"      # Â not starting a lowercase word
    "|\u00c3(?![A-Z])"     # Ã not starting an uppercase word
    "|\u00e2(?![a-z])"     # â not continuing a lowercase word
    "|\u00e2(?=[A-Z])"     # â before a capital: the lost-apostrophe case
)
_STRAY_LEAD_REPLACEMENT = {"\u00e2": "'"}


# --------------------------------------------------------------------------
# 2. placeholder tokens
# --------------------------------------------------------------------------

# Same seven tokens the profiler counts, so "the cleaner removed it" and "the
# profiler measured it" are about the same set. Whole-word guards on both sides
# (with re.IGNORECASE) keep "NANDAN" and "N/A/c" untouched.
_PLACEHOLDER = re.compile(
    r"(?<!\w)(?:NULL|N/A|NONE|NAN)(?!\w)", re.IGNORECASE
)


# --------------------------------------------------------------------------
# 3. junk runs, brackets, separators
# --------------------------------------------------------------------------

# Two or more identical characters that cannot be part of a word: "##", "***",
# "???", "--", "..". Alphanumerics (and any non-Latin letter, since \w is
# Unicode-aware) are never touched, so "111" and Devanagari "क्क" survive.
_JUNK_RUN = re.compile(r"([^\w\s]|_)\1+")

# Square brackets only; the content is kept, the decoration is not.
_SQUARE_BRACKETS = re.compile(r"\[([^\[\]]*)\]")

# An unbalanced "[" ("Acme [Pvt") is left over from the same decoration, and
# this dataset treats square brackets as pure ornament, so the leftover
# bracket goes too.
_DANGLING_BRACKET = re.compile(r"[\[\]]")

# Removing a placeholder out of a comma-separated field can leave ", ," behind;
# this puts the field separator back to one.
_REPEATED_SEPARATORS = re.compile(r"(?:\s*[,;]\s*){2,}")

# Bracket/paren pairs left empty by a removal, e.g. "( )" -> "".
_EMPTY_BRACKETS = re.compile(r"[\(\[]\s*[,;]?\s*[\)\]]")

# Separators with nothing in front of or behind them. Anchored, so a bracket or
# paren that still has a matching partner is never clipped.
_EDGE_SEPARATORS = re.compile(r"^[\s,;:|/\\\-]+|[\s,;:|/\\\-]+$")

_WHITESPACE = re.compile(r"\s+")


def _decode_mojibake_sequence(sequence: str) -> str:
    """Re-decode one candidate sequence, or return it unchanged."""
    for encoding in _MOJIBAKE_ENCODINGS:
        try:
            return sequence.encode(encoding).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return sequence


def repair_mojibake(text: str) -> str:
    """
    Undo one layer of "UTF-8 decoded as Latin-1/cp1252".

    A candidate sequence is only accepted if re-encoding it to bytes and reading
    those bytes as UTF-8 succeeds, which means real accented characters are left
    alone: their bytes are not valid UTF-8, so the round-trip fails and the
    original text is returned.
    """
    if text.isascii():
        return text
    return _MOJIBAKE_SEQUENCE.sub(
        lambda match: _decode_mojibake_sequence(match.group()), text
    )


def _tidy_separators(text: str) -> str:
    """Clean up the punctuation left behind by the deletions above."""
    text = _REPEATED_SEPARATORS.sub(", ", text)
    text = _EMPTY_BRACKETS.sub("", text)
    # A value that was nothing but junk legitimately becomes empty: "NULL" and
    # "###" are noise, and an empty string is a better key input than ",".
    return _EDGE_SEPARATORS.sub("", text).strip()


def clean_text(s: str) -> str:
    """
    Return `s` with mojibake, placeholder tokens, junk runs and bracket
    decoration removed, and whitespace normalized.

    Non-string input (NaN, None) is treated as a blank value, because these
    columns come out of pandas and clean_dataframe applies this to whole
    columns.
    """
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)

    # 1. mojibake: restore the mangled lead byte first so the round-trip can
    #    decode the sequence, then decode, then fold smart punctuation to ASCII.
    text = repair_mojibake(_MANGLED_LEAD.sub("\u00e2\u0080", s))
    text = text.translate(_SMART_PUNCTUATION)

    # 2. anything still undecodable: controls, U+FFFD, and a bare lead character
    #    whose continuation bytes are gone.
    text = _CONTROL_AND_REPLACEMENT.sub(" ", text)
    text = _STRAY_LEAD.sub(lambda m: _STRAY_LEAD_REPLACEMENT.get(m.group(), " "), text)

    # 3. placeholders, junk runs, bracket decoration.
    text = _PLACEHOLDER.sub(" ", text)
    text = _JUNK_RUN.sub(" ", text)
    text = _SQUARE_BRACKETS.sub(r"\1", text)
    text = _DANGLING_BRACKET.sub(" ", text)

    # 4. NFKC last of the character-level passes (it also folds U+00A0), then
    #    collapse and trim.
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE.sub(" ", text).strip()

    # 5. separators, then trim again in case a step deleted an edge character.
    return _tidy_separators(text).strip()


def clean_dataframe(
    df: pd.DataFrame,
    columns: Sequence[str] = DEFAULT_COLUMNS,
) -> pd.DataFrame:
    """
    Input: a source DataFrame with at least [entity_id, business_name,
    business_address, country].
    Output: same DataFrame with one added column per cleaned column, named
    "<column>_clean" (business_name_clean, business_address_clean). The raw
    columns are left untouched: some similarity features want the raw text, and
    this pass must stay non-destructive.

    Columns named in `columns` that are not in the frame are skipped, so the
    same call works on a partial frame.
    """
    out = df.copy()
    for column in columns:
        if column not in out.columns:
            continue
        out[f"{column}{CLEAN_SUFFIX}"] = out[column].map(clean_text)
    return out


# --------------------------------------------------------------------------
# self-check
# --------------------------------------------------------------------------


# Scripts whose row count may legitimately *drop*, and only drop. The repair
# turns C1 garbage into ASCII and smart punctuation into ASCII, and NFKC maps
# the compatibility forms it is asked to fold ("1º" -> "1o", "½" -> "1/2"), so
# those three buckets lose rows on purpose. Every other script has to come
# through untouched, and that is the check that matters: it is the one that
# catches cleaning eating Devanagari or Telugu.
#
# The unclassified bucket is in this list because that is where the C1 controls
# and U+FFFD land -- they are below the Latin-accented block and belong to no
# tracked script. It was checked by hand rather than assumed: the only
# codepoints that disappear from it are U+0080-U+009F, U+FFFD, U+00BA and
# U+00BD. (The key is the tag character "X", not the report's label for it.)
SCRIPTS_MAY_DROP = ("Latin (accented)", "General punctuation", OTHER_TAG)

MAX_REPORTED_EXAMPLES = 5


def _read_sample(dataset_dir: Path, splits: Sequence[str], rows: int) -> pd.DataFrame:
    """First `rows` rows of every source file, columns preserved."""
    frames: list[pd.DataFrame] = []
    for split in splits:
        for path in sorted((dataset_dir / split).glob("*.tsv")):
            frame = pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
                na_values=[],
                nrows=rows,
            )
            frame["source_file"] = path.name
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _classify_residuals(token: str, values: pd.Series) -> tuple[int, int, list[str], list[str]]:
    """
    Split the values that still match the profiler's pattern for `token` into
    real survivors and profiler false positives.

    The profiler guards its word tokens with [0-9A-Za-z], so in an accented name
    it reads the "nan" of "Cronan-Eckhardt" as the placeholder "nan" -- the
    o-acute is not in its guard, but it is a word character, so _PLACEHOLDER
    (which guards on \\w) correctly leaves the name alone. Those rows are the
    profiler over-matching, not junk that survived, and they are reported
    separately so they cannot be mistaken for either.

    Returns (survived, over_matched, survived_examples, over_matched_examples).
    """
    pattern = NOISE_TOKEN_PATTERNS[token]
    matching = values[values.str.contains(pattern, regex=True)]
    if not len(matching):
        return 0, 0, [], []
    if token.upper() in ("NULL", "N/A", "NONE", "NAN"):
        really_junk = matching[matching.str.contains(_PLACEHOLDER, regex=True)]
    else:
        really_junk = matching
    over_matched = matching[~matching.index.isin(really_junk.index)]
    return (
        len(really_junk),
        len(over_matched),
        really_junk.head(MAX_REPORTED_EXAMPLES).tolist(),
        over_matched.head(MAX_REPORTED_EXAMPLES).tolist(),
    )


def _self_check(dataset_dir: Path, splits: Sequence[str], rows: int) -> int:
    """
    Profile a bounded real sample before and after cleaning and print the noise
    counts. Two conditions have to hold:

    - every token the profiler tracks is at zero afterwards, except occurrences
      that are the profiler's own over-matching on an accented name;
    - no script other than Latin (accented) and General punctuation changes its
      row count, because cleaning must not achieve zero noise by emptying the
      accented and Indic rows. Those two are allowed to *drop* and nothing else:
      the repair turns C1 garbage and smart quotes into ASCII on purpose.
    """
    print(f"reading {rows:,} rows per file from {dataset_dir} ...", flush=True)
    started = time.perf_counter()
    raw = _read_sample(dataset_dir, splits, rows)
    print(f"  {len(raw):,} rows in {time.perf_counter() - started:.1f}s", flush=True)

    before = profile_dataframe(raw)
    cleaned = clean_dataframe(raw)
    # profile_dataframe looks at business_name / business_address by default, so
    # the clean columns have to be named explicitly -- otherwise this would
    # profile the raw text twice and report a pass.
    after = profile_dataframe(cleaned, text_columns=CLEAN_COLUMNS)

    failed = 0
    false_positives = 0
    print(f"\n{'column':<18}{'token':<8}{'before':>10}{'after':>10}")
    print("-" * 46)
    for column, clean_column in zip(DEFAULT_COLUMNS, CLEAN_COLUMNS):
        pre = before.columns[column]
        post = after.columns[clean_column]
        for token in sorted(pre.noise_token_occurrences):
            pre_n = pre.noise_token_occurrences[token]
            post_n = post.noise_token_occurrences.get(token, 0)
            print(f"{column:<18}{token:<8}{pre_n:>10,}{post_n:>10,}")
            survived, over, examples, over_examples = _classify_residuals(
                token, cleaned[clean_column]
            )
            failed += survived
            false_positives += over
            if survived:
                print(f"  !! {survived} occurrence(s) survived, e.g. {examples[0]!r}")
            if over:
                print(
                    f"  ~~ {over} occurrence(s) left in place on purpose: the "
                    "profiler's [0-9A-Za-z] guard reads a word inside an accented "
                    f"name, e.g. {over_examples[0]!r}"
                )
        for script, count_before in sorted(pre.script_rows.items()):
            count_after = post.script_rows.get(script, 0)
            if count_after == count_before:
                continue
            if script in SCRIPTS_MAY_DROP and count_after < count_before:
                print(
                    f"{column:<18}{script:<8}{count_before:>10,}{count_after:>10,}"
                    f"   (mojibake/smart punctuation repaired)"
                )
                continue
            print(
                f"{column:<18}{script:<8}{count_before:>10,}{count_after:>10,}"
                f"   !! {script} must not change"
            )
            failed += abs(count_after - count_before)

    print("-" * 46)
    if failed:
        print(f"FAIL: {failed} problem(s)")
    elif false_positives:
        print(
            f"PASS: no tracked noise token survives cleaning "
            f"({false_positives} profiler over-match(es) on accented names left in place)"
        )
    else:
        print("PASS: no tracked noise token survives cleaning")
    return 0 if not failed else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--rows", type=int, default=200_000,
                        help="rows per file; the sample is bounded on purpose")
    args = parser.parse_args(argv)
    return _self_check(args.dataset_dir, args.splits, args.rows)


if __name__ == "__main__":
    raise SystemExit(main())
