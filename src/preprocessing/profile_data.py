"""
Owner: Person A

Data profiling for the entity-resolution source files. Answers the questions we
need settled before writing any normalization rules or blocking keys:

- how big is each file, and what is the country mix (train = US/India, test also
  has France, which is unseen at train time)
- how much of the text is non-ASCII, and which Unicode scripts it is written in
  (Devanagari / Gujarati / Gurmukhi / Tamil / Telugu / ... plus Latin-accented).
  Transliteration noise across sources is a blocking problem, so we need the
  actual script mix per file, not a guess.
- how much literal junk sits in the text ("NULL", "N/A", "NONE", "nan", "##",
  "###", "***"). These are placeholders, not missing values: pandas' default NA
  handling would silently turn some of them into NaN, so the reader below
  disables NA coercion and counts them as strings.
- how many business_name values are too short to carry signal (<3 chars)
- how many bracket-wrapped tokens exist, e.g. "[Limited]", "(Pvt)" style
  decorations in business_name

Two entry points:

    profile_dataframe(df, ...) -> FileProfile     # in-memory, used by tests
    profile_file(path, ...)    -> FileProfile     # chunked streaming

profile_file never holds a whole file in memory: the source TSVs are ~0.5 GB each
(~2M rows), so it reads with pandas' chunked reader and merges a profile per
chunk. Both entry points run the exact same per-chunk logic, so profiling a
DataFrame in a test and profiling the real file on disk agree by construction.

Usage:
    python -m src.preprocessing.profile_data                 # prints the report
    python -m src.preprocessing.profile_data \
        --dataset-dir dataset --out docs/data_profile_report.md

The markdown it writes is what the rest of the team reads; keep it checked in
and regenerate it whenever the dataset changes.
"""
from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_REPORT_PATH = REPO_ROOT / "docs" / "data_profile_report.md"
DEFAULT_SPLITS = ("train", "test")

NAME_COL = "business_name"
ADDRESS_COL = "business_address"
COUNTRY_COL = "country"
# Column order used in the report; columns missing from a file are skipped.
TEXT_COLUMNS = (NAME_COL, ADDRESS_COL)

# "under 3 characters" -> stripped length 0, 1 or 2.
SHORT_VALUE_MAX_LEN = 3
CHUNKSIZE = 100_000
# Example values (short names, bracket contents) are only collected from the
# first N rows of a file, so we never hold a big sample in memory.
EXAMPLE_SAMPLE_ROWS = 100_000
MAX_EXAMPLES = 8

# Bracket-wrapped token: no nesting, so "[Limited]" and "[Pvt] Ltd" each yield
# one token, and "[[x]]" is not mis-parsed.
BRACKET_PATTERN = re.compile(r"\[[^\[\]]*\]")

# Literal junk placeholders. Patterns are guarded so that (a) they match the
# token as a standalone word, not as a substring of a real word, and (b) "##"
# and "###" never double-count each other -- an exact run of two hashes is
# "##", an exact run of three is "###".
NOISE_TOKEN_PATTERNS: dict[str, str] = {
    "NULL": r"(?<![0-9A-Za-z])NULL(?![0-9A-Za-z])",
    "N/A": r"(?<![0-9A-Za-z])N/A(?![0-9A-Za-z])",
    "NONE": r"(?<![0-9A-Za-z])NONE(?![0-9A-Za-z])",
    "nan": r"(?<![0-9A-Za-z])nan(?![0-9A-Za-z])",
    "##": r"(?<!#)##(?!#)",
    "###": r"(?<!#)###(?!#)",
    "***": r"(?<!\*)\*\*\*(?!\*)",
}
COMPILED_NOISE_PATTERNS: dict[str, re.Pattern[str]] = {
    token: re.compile(pattern) for token, pattern in NOISE_TOKEN_PATTERNS.items()
}

# Unicode script blocks, as (start, end) inclusive codepoint ranges. First match
# wins per character, so more specific blocks come first (fullwidth katakana
# before generic fullwidth, emoji before generic symbols).
#
# Devanagari covers Hindi and Marathi; Gurmukhi is Punjabi; Bengali/Oriya/
# Kannada/Malayalam/Tamil/Telugu round out the Indic scripts present in Indian
# business names. "Latin (accented)" is the non-ASCII part of the Latin script:
# Western Europe needs its own normalization treatment, and the test set has
# France rows.
SCRIPT_RANGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("Latin (accented)", ((0x00C0, 0x024F), (0x1E00, 0x1EFF), (0x2C60, 0x2C7F), (0xA720, 0xA7FF))),
    ("Devanagari", ((0x0900, 0x097F), (0xA8E0, 0xA8FF))),
    ("Bengali", ((0x0980, 0x09FF),)),
    ("Gurmukhi", ((0x0A00, 0x0A7F),)),
    ("Gujarati", ((0x0A80, 0x0AFF),)),
    ("Oriya", ((0x0B00, 0x0B7F),)),
    ("Tamil", ((0x0B80, 0x0BFF),)),
    ("Telugu", ((0x0C00, 0x0C7F),)),
    ("Kannada", ((0x0C80, 0x0CFF),)),
    ("Malayalam", ((0x0D00, 0x0D7F),)),
    ("Sinhala", ((0x0D80, 0x0DFF),)),
    ("Thai", ((0x0E00, 0x0E7F),)),
    ("Khmer", ((0x1780, 0x17FF),)),
    ("Tibetan", ((0x0F00, 0x0FFF),)),
    ("Myanmar", ((0x1000, 0x109F),)),
    ("Ethiopic", ((0x1200, 0x137F),)),
    ("Cherokee", ((0x13A0, 0x13FF),)),
    ("Greek", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
    ("Cyrillic", ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))),
    ("Armenian", ((0x0530, 0x058F),)),
    ("Hebrew", ((0x0590, 0x05FF), (0xFB1D, 0xFB4F))),
    ("Arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("Hangul", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xA960, 0xA97F), (0xAC00, 0xD7AF))),
    ("Hiragana", ((0x3040, 0x309F),)),
    ("Katakana", ((0x30A0, 0x30FF), (0x31F0, 0x31FF), (0xFF66, 0xFF9D))),
    ("Han (CJK)", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF))),
    ("Currency symbols", ((0x20A0, 0x20CF),)),
    ("General punctuation", ((0x2010, 0x206F),)),
    ("Math / technical symbols", ((0x2100, 0x2BFF),)),
    ("Box drawing / shapes", ((0x2500, 0x25FF),)),
    ("Emoji", ((0x2600, 0x27BF), (0x1F000, 0x1FAFF))),
    ("Fullwidth forms", ((0xFF01, 0xFF60), (0xFFE0, 0xFFE6))),
)
SCRIPT_NAMES: tuple[str, ...] = tuple(name for name, _ in SCRIPT_RANGES)
NON_ASCII_SCRIPT = "Other (non-ASCII, unclassified)"

# Per-character tagging table: every codepoint in a tracked block maps to a
# single tag character, so one str.translate() over a whole chunk of values
# turns arbitrary text into a string of script tags. Rows are then split on a
# separator and the set of tags in each row gives that row's scripts.
#
# Tag characters are ASCII letters, and ASCII input characters are *deleted*
# before translation runs, so a literal "a" in a business name can never be
# mistaken for a tag.
_TAG_BY_SCRIPT = {
    script: chr(ord("a") + i) for i, script in enumerate(SCRIPT_NAMES)
}
OTHER_TAG = "X"
# ASCII input maps to a sentinel that is stripped afterwards. The lookup is on
# the *source* character, so an ASCII letter in "Balaji" can never be mistaken
# for a tag however much the tag characters look like letters.
_ASCII_SENTINEL = "\x01"
# Values are joined with NUL to tag a whole chunk in one pass. A TSV field
# cannot contain a raw NUL and the parser never emits one, so it is a safe row
# delimiter -- and it is mapped to itself so the sentinel sweep keeps it.
VALUE_SEPARATOR = "\x00"
CHAR_TAGS: dict[int, str] = {cp: _ASCII_SENTINEL for cp in range(0x80)}
CHAR_TAGS[ord(VALUE_SEPARATOR)] = VALUE_SEPARATOR
for _script, _ranges in SCRIPT_RANGES:
    for _lo, _hi in _ranges:
        for _cp in range(_lo, min(_hi, 0xFFFF) + 1):
            CHAR_TAGS.setdefault(_cp, _TAG_BY_SCRIPT[_script])
for _cp in range(0x80, 0x10000):
    CHAR_TAGS.setdefault(_cp, OTHER_TAG)
KNOWN_TAGS = frozenset((*_TAG_BY_SCRIPT.values(), OTHER_TAG))
# Characters outside the BMP are outside the tag table; they are counted
# separately so nothing non-ASCII goes unreported.
NON_BMP_PATTERN = re.compile(r"[^\u0000-\uffff]")

BLANK_LABEL = "(blank)"


def _series_of_strings(values: Any) -> pd.Series:
    """Coerce a column to a Series of str with no NaN (empty string for nulls)."""
    series = values if isinstance(values, pd.Series) else pd.Series(values, dtype=object)
    return series.fillna("").astype(str)


def _profile_scripts(values: pd.Series) -> tuple[dict[str, int], int]:
    """
    Count rows containing at least one character of each script.

    Returns (rows_per_script, non_ascii_rows). One join + translate + split per
    column keeps this a handful of C-level passes instead of a Python loop per
    character.
    """
    tagged = VALUE_SEPARATOR.join(values.tolist()).translate(CHAR_TAGS).replace(_ASCII_SENTINEL, "")
    script_rows: dict[str, int] = {}
    non_ascii_rows = 0
    per_row_tags = Counter(frozenset(chunk) & KNOWN_TAGS for chunk in tagged.split(VALUE_SEPARATOR))
    for tags, count in per_row_tags.items():
        if not tags:
            continue
        non_ascii_rows += count
        for tag in tags:
            name = OTHER_TAG if tag == OTHER_TAG else SCRIPT_NAMES[ord(tag) - ord("a")]
            script_rows[name] = script_rows.get(name, 0) + count
    return script_rows, non_ascii_rows


@dataclass
class ColumnProfile:
    """Per-text-column statistics. All counts are over rows of the file."""

    column: str
    rows: int = 0
    non_ascii_rows: int = 0
    non_bmp_rows: int = 0
    script_rows: dict[str, int] = field(default_factory=dict)
    noise_token_occurrences: dict[str, int] = field(default_factory=dict)
    noise_token_rows_by_token: dict[str, int] = field(default_factory=dict)
    noise_token_rows: int = 0
    blank_values: int = 0
    short_values: int = 0
    short_examples: list[str] = field(default_factory=list)
    bracket_token_occurrences: int = 0
    bracket_rows: int = 0
    bracket_examples: list[str] = field(default_factory=list)
    sampled_rows: int = 0

    def pct(self, count: int) -> float:
        return (100.0 * count / self.rows) if self.rows else 0.0

    @property
    def noise_total(self) -> int:
        return sum(self.noise_token_occurrences.values())

    def merge(self, other: "ColumnProfile") -> None:
        self.rows += other.rows
        self.non_ascii_rows += other.non_ascii_rows
        self.non_bmp_rows += other.non_bmp_rows
        self.noise_token_rows += other.noise_token_rows
        self.blank_values += other.blank_values
        self.short_values += other.short_values
        self.bracket_token_occurrences += other.bracket_token_occurrences
        self.bracket_rows += other.bracket_rows
        for script, count in other.script_rows.items():
            self.script_rows[script] = self.script_rows.get(script, 0) + count
        for counts in (other.noise_token_occurrences, other.noise_token_rows_by_token):
            mine = self.noise_token_occurrences if counts is other.noise_token_occurrences else self.noise_token_rows_by_token
            for token, count in counts.items():
                mine[token] = mine.get(token, 0) + count
        for bucket, examples in (("short", other.short_examples), ("bracket", other.bracket_examples)):
            target = self.short_examples if bucket == "short" else self.bracket_examples
            for example in examples:
                if len(target) < MAX_EXAMPLES and example not in target:
                    target.append(example)
        self.sampled_rows += other.sampled_rows


@dataclass
class FileProfile:
    """Everything the report prints for one TSV file."""

    path: str
    schema: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    missing_columns: list[str] = field(default_factory=list)
    rows: int = 0
    country_counts: Counter = field(default_factory=Counter)
    country_rows: int = 0
    skipped_lines: int = 0
    elapsed_s: float = 0.0
    columns: dict[str, ColumnProfile] = field(default_factory=dict)

    def countries_ordered(self) -> list[tuple[str, int]]:
        """Non-blank countries by descending row count, blank (if any) first."""
        blank = self.country_counts.get(BLANK_LABEL, 0)
        rows = [(BLANK_LABEL, blank)] if blank else []
        rows += sorted(
            ((k, c) for k, c in self.country_counts.items() if k != BLANK_LABEL),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return rows or [(BLANK_LABEL, 0)]

    def merge(self, other: "FileProfile") -> None:
        self.rows += other.rows
        self.country_counts.update(other.country_counts)
        self.country_rows += other.country_rows
        self.skipped_lines += other.skipped_lines
        self.elapsed_s += other.elapsed_s
        self.missing_columns = sorted(set(self.missing_columns) | set(other.missing_columns))
        for column in other.text_columns:
            if column not in self.text_columns:
                self.text_columns.append(column)
        for name, col_profile in other.columns.items():
            if name not in self.columns:
                self.columns[name] = col_profile
            else:
                self.columns[name].merge(col_profile)

    def country_pct(self, count: int) -> float:
        return (100.0 * count / self.rows) if self.rows else 0.0

    def col(self, name: str) -> ColumnProfile | None:
        return self.columns.get(name)


def _profile_country(values: pd.Series, profile: FileProfile) -> None:
    series = _series_of_strings(values)
    profile.country_rows += len(series)
    # An empty string is a missing label, not a country: bucket it separately so
    # "how many country labels does this file have" stays honest.
    profile.country_counts.update(
        Counter(BLANK_LABEL if v == "" else v for v in series.tolist())
    )


def _profile_noise_tokens(values: pd.Series, col: ColumnProfile) -> None:
    # One count pass per token gives both the number of occurrences and the
    # number of rows the token appears in, so the report never has to conflate
    # the two.
    for token, pattern in COMPILED_NOISE_PATTERNS.items():
        per_row = values.str.count(pattern)
        occurrences = int(per_row.sum())
        if occurrences:
            col.noise_token_occurrences[token] = occurrences
            col.noise_token_rows_by_token[token] = int((per_row > 0).sum())
    col.noise_token_rows = _rows_with_any_noise_token(values)


def _rows_with_any_noise_token(values: pd.Series) -> int:
    combined = "|".join(f"(?:{pattern})" for pattern in NOISE_TOKEN_PATTERNS.values())
    return int(values.str.contains(combined, regex=True).sum())


def _profile_length(values: pd.Series, col: ColumnProfile) -> None:
    stripped = values.str.strip()
    lengths = stripped.str.len()
    col.blank_values = int((lengths == 0).sum())
    is_short = lengths < SHORT_VALUE_MAX_LEN
    col.short_values = int(is_short.sum())
    if col.sampled_rows:
        col.short_examples.extend(stripped[is_short].head(MAX_EXAMPLES).tolist())


def _profile_brackets(values: pd.Series, col: ColumnProfile) -> None:
    counts = values.str.count(BRACKET_PATTERN.pattern)
    col.bracket_token_occurrences += int(counts.sum())
    col.bracket_rows += int((counts > 0).sum())
    if col.sampled_rows:
        for match in BRACKET_PATTERN.findall(VALUE_SEPARATOR.join(values.tolist())):
            if len(col.bracket_examples) >= MAX_EXAMPLES:
                break
            if match not in col.bracket_examples:
                col.bracket_examples.append(match)


def profile_dataframe(
    df: pd.DataFrame,
    path: str = "<dataframe>",
    text_columns: Sequence[str] = TEXT_COLUMNS,
    country_column: str = COUNTRY_COL,
    collect_examples: bool = True,
) -> FileProfile:
    """
    Profile an in-memory source DataFrame.

    Input: a DataFrame shaped like a source file -- [entity_id, business_name,
    business_address, country]. Columns that are absent (e.g. the ground-truth
    file has no text columns) are reported in `missing_columns` and skipped.

    `collect_examples` controls whether short-name and bracket examples are
    harvested; `profile_file` turns it off once a file's first
    EXAMPLE_SAMPLE_ROWS rows have been seen, so the examples in the report come
    from a bounded prefix of the file.

    Output: FileProfile. Chunked or not, this is the only place the per-column
    statistics are computed, so `profile_file` and the tests share one code path.
    """
    profile = FileProfile(path=path, schema=[str(c) for c in df.columns])
    n_rows = len(df)
    profile.rows = n_rows
    if country_column in df.columns:
        _profile_country(df[country_column], profile)
    else:
        profile.missing_columns.append(country_column)

    for column in text_columns:
        if column not in df.columns:
            profile.missing_columns.append(column)
            continue
        values = _series_of_strings(df[column])
        col = ColumnProfile(
            column=column,
            rows=len(values),
            sampled_rows=len(values) if collect_examples else 0,
        )
        script_rows, non_ascii_rows = _profile_scripts(values)
        col.script_rows = script_rows
        col.non_ascii_rows = non_ascii_rows
        col.non_bmp_rows = int(values.str.contains(NON_BMP_PATTERN).sum())
        _profile_noise_tokens(values, col)
        _profile_length(values, col)
        _profile_brackets(values, col)
        profile.text_columns.append(column)
        profile.columns[column] = col
    return profile


def _count_data_lines(path: Path) -> int:
    """Newline count minus the header, read in binary blocks (C speed, no RAM)."""
    newlines = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            newlines += block.count(b"\n")
    return max(newlines - 1, 0)


def _open_tsv(path: Path, chunksize: int, on_bad_lines: str):
    """A chunked, all-text, NA-preserving reader over a source TSV."""
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        chunksize=chunksize,
        keep_default_na=False,  # "NULL"/"nan" are literal text here, not NA
        na_values=[],
        encoding="utf-8",
        encoding_errors="replace",
        on_bad_lines=on_bad_lines,
    )


def profile_file(
    path: str | Path,
    chunksize: int = CHUNKSIZE,
    text_columns: Sequence[str] = TEXT_COLUMNS,
    country_column: str = COUNTRY_COL,
) -> FileProfile:
    """
    Profile a TSV on disk without loading it into memory.

    Reads in `chunksize`-row chunks and merges a profile per chunk, so peak
    memory is one chunk of text plus the accumulated counters, not the file.

    Parsing is strict first: if any line has the wrong number of fields the C
    parser raises rather than silently dropping data. Only then does it re-read
    with skipping enabled, so a report never quietly under-counts rows -- the
    skipped-line count is in the output either way.
    """
    path = Path(path)
    started = time.perf_counter()
    profile = FileProfile(path=path.as_posix())

    def consume(reader) -> int:
        chunks = 0
        for chunk in reader:
            if chunks == 0:
                profile.schema = [str(c) for c in chunk.columns]
            profile.merge(
                profile_dataframe(
                    chunk,
                    path=path.as_posix(),
                    text_columns=text_columns,
                    country_column=country_column,
                    # Examples come from the first EXAMPLE_SAMPLE_ROWS rows only.
                    collect_examples=profile.rows < EXAMPLE_SAMPLE_ROWS,
                )
            )
            chunks += 1
        return chunks

    try:
        try:
            consume(_open_tsv(path, chunksize, on_bad_lines="error"))
        except pd.errors.ParserError:
            profile.rows = 0
            profile.columns.clear()
            profile.text_columns.clear()
            profile.country_counts.clear()
            profile.country_rows = 0
            consume(_open_tsv(path, chunksize, on_bad_lines="skip"))
            profile.skipped_lines = max(_count_data_lines(path) - profile.rows, 0)
    except pd.errors.EmptyDataError:
        profile.missing_columns.append("<empty file>")
    profile.elapsed_s = time.perf_counter() - started
    return profile


def discover_data_files(
    dataset_dir: str | Path = DEFAULT_DATASET_DIR,
    splits: Sequence[str] = DEFAULT_SPLITS,
    pattern: str = "*.tsv",
) -> list[Path]:
    """TSVs under dataset/<split>/, splits in a stable order, files sorted."""
    dataset_dir = Path(dataset_dir)
    found: list[Path] = []
    for split in splits:
        found.extend(sorted((dataset_dir / split).glob(pattern)))
    return found


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_pct(count: int, rows: int) -> str:
    return f"{(100.0 * count / rows):.2f}%" if rows else "n/a"


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return lines


def _display_path(path: str | Path) -> str:
    """Repo-relative where possible, so the report is portable between machines."""
    path = Path(path)
    try:
        return f"{path.resolve().relative_to(REPO_ROOT).as_posix()}/"
    except ValueError:
        return f"{path.as_posix()}/"


def format_file_report(profile: FileProfile) -> list[str]:
    """Markdown section for one file."""
    lines: list[str] = []
    title = profile.path.replace("\\", "/")
    lines.append(f"## `{title}`")
    lines.append("")
    if profile.rows == 0:
        lines.append("_Empty or unparseable file._")
        lines.append("")
        return lines
    lines.append(f"- **Rows:** {_fmt_int(profile.rows)}")
    lines.append(f"- **Columns:** {', '.join(f'`{c}`' for c in profile.schema) or '(none)'}")
    if profile.missing_columns:
        lines.append(f"- **Expected columns not present (skipped):** "
                     f"{', '.join(f'`{c}`' for c in profile.missing_columns)}")
    if profile.skipped_lines:
        lines.append(f"- **Malformed lines skipped by the TSV parser:** {_fmt_int(profile.skipped_lines)}")
    lines.append(f"- **Profiled in:** {profile.elapsed_s:.1f}s")
    lines.append("")

    if COUNTRY_COL in profile.schema or profile.country_rows:
        lines.append("### Country mix")
        lines.append("")
        rows = [[f"`{label}`", _fmt_int(count), _fmt_pct(count, profile.rows)]
                for label, count in profile.countries_ordered()]
        lines.extend(_table(["country", "rows", "% of file"], rows))
        distinct = len([1 for label, _ in profile.countries_ordered() if label != BLANK_LABEL])
        lines.append("")
        lines.append(f"_{distinct} distinct non-blank country label(s)._")
        lines.append("")

    if not profile.text_columns:
        lines.append("_No business_name / business_address columns to profile._")
        lines.append("")
        return lines

    lines.append("### Non-ASCII content by Unicode script")
    lines.append("")
    lines.append("Rows containing at least one character from each script. A row can appear under "
                 "more than one script, so these do not sum to the row total.")
    lines.append("")
    headers = ["script"]
    for column in profile.text_columns:
        headers += [f"{column} rows", f"{column} %"]
    rows = []
    for script in (*SCRIPT_NAMES, NON_ASCII_SCRIPT):
        if all(profile.columns[c].script_rows.get(script, 0) == 0 for c in profile.text_columns):
            continue  # keep the table short: drop scripts absent from every column
        cells: list[str] = [script]
        for column in profile.text_columns:
            col = profile.columns[column]
            count = col.script_rows.get(script, 0)
            cells += [_fmt_int(count), _fmt_pct(count, col.rows)]
        rows.append(cells)
    any_cells = ["**Any non-ASCII**"]
    for column in profile.text_columns:
        col = profile.columns[column]
        any_cells += [_fmt_int(col.non_ascii_rows), _fmt_pct(col.non_ascii_rows, col.rows)]
    rows.append(any_cells)
    non_bmp = [c for c in profile.text_columns if profile.columns[c].non_bmp_rows]
    if non_bmp:
        any_cells = ["(of which outside the BMP)"]
        for column in profile.text_columns:
            col = profile.columns[column]
            any_cells += [_fmt_int(col.non_bmp_rows), _fmt_pct(col.non_bmp_rows, col.rows)]
        rows.append(any_cells)
    lines.extend(_table(headers, rows))
    lines.append("")

    lines.append("### Literal noise tokens")
    lines.append("")
    lines.append("Placeholder junk counted as literal text (NA coercion is off, so these are never "
                 "silently read as missing values). `occurrences` counts every appearance; `rows` "
                 "counts rows containing the token at least once.")
    lines.append("")
    headers = ["token"]
    for column in profile.text_columns:
        headers += [f"{column} occurrences", f"{column} rows"]
    rows = []
    for token in NOISE_TOKEN_PATTERNS:
        if all(profile.columns[c].noise_token_occurrences.get(token, 0) == 0 for c in profile.text_columns):
            continue
        cells = [f"`{token}`"]
        for column in profile.text_columns:
            col = profile.columns[column]
            cells += [
                _fmt_int(col.noise_token_occurrences.get(token, 0)),
                _fmt_int(col.noise_token_rows_by_token.get(token, 0)),
            ]
        rows.append(cells)
    total_cells = ["**any noise token**"]
    for column in profile.text_columns:
        col = profile.columns[column]
        total_cells += [_fmt_int(col.noise_total), _fmt_int(col.noise_token_rows)]
    rows.append(total_cells)
    if len(rows) == 1:
        lines.append("_None found in any column._")
    else:
        lines.extend(_table(headers, rows))
    lines.append("")

    lines.append("### Low-information business_name values")
    lines.append("")
    name_col = profile.columns.get(NAME_COL)
    if name_col is None:
        lines.append("_No business_name column._")
        lines.append("")
        return lines
    short_1_2 = name_col.short_values - name_col.blank_values
    rows = [
        [f"stripped length < {SHORT_VALUE_MAX_LEN} (low-information / garbage)", _fmt_int(name_col.short_values),
         _fmt_pct(name_col.short_values, name_col.rows)],
        ["&nbsp;&nbsp;of which empty / whitespace-only", _fmt_int(name_col.blank_values),
         _fmt_pct(name_col.blank_values, name_col.rows)],
        ["&nbsp;&nbsp;of which 1-2 characters", _fmt_int(short_1_2), _fmt_pct(short_1_2, name_col.rows)],
    ]
    lines.extend(_table(["metric", "rows", "% of file"], rows))
    if name_col.short_examples:
        lines.append("")
        lines.append(f"Examples: {', '.join(f'`{e}`' for e in name_col.short_examples)}")
    lines.append("")

    lines.append("### Bracket-wrapped tokens in business_name")
    lines.append("")
    lines.append("Decoration tokens like `[Limited]` or `(Pvt)` that normalization has to strip "
                 "or expand rather than match literally.")
    lines.append("")
    rows = [
        ["token occurrences", _fmt_int(name_col.bracket_token_occurrences)],
        ["rows containing at least one", _fmt_int(name_col.bracket_rows)],
        ["% of rows containing at least one", _fmt_pct(name_col.bracket_rows, name_col.rows)],
    ]
    lines.extend(_table(["metric", "value"], rows))
    if name_col.bracket_examples:
        lines.append("")
        lines.append(f"Example contents (first ones seen, sampled from the first "
                     f"{_fmt_int(name_col.sampled_rows)} rows): "
                     f"{', '.join(f'`{e}`' for e in name_col.bracket_examples)}")
    lines.append("")
    return lines


def format_report(profiles: Sequence[FileProfile], dataset_dir: str | Path = DEFAULT_DATASET_DIR,
                  command: str | None = None, generated_at: str | None = None) -> str:
    """Full markdown report: summary table first, then one section per file."""
    import datetime as _dt

    generated_at = generated_at or _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    dataset_dir = Path(dataset_dir)
    lines: list[str] = [
        "# Data profile report",
        "",
        f"_Generated by `src/preprocessing/profile_data.py` at {generated_at} "
        f"from `{_display_path(dataset_dir)}`._",
        "",
        "One-off profile of the provided source files, written before any normalization or "
        "blocking rules were chosen. It is a snapshot, not something the pipeline depends on: "
        "regenerate with",
        "",
        "```bash",
        command or "python -m src.preprocessing.profile_data --out docs/data_profile_report.md",
        "```",
        "",
        "**What it measures per file:** row count, country mix, share of rows containing "
        "non-ASCII characters broken down by Unicode script, counts of literal noise tokens "
        "(`NULL`, `N/A`, `NONE`, `nan`, `##`, `###`, `***`), `business_name` values shorter "
        "than 3 characters, and bracket-wrapped tokens such as `[Limited]`.",
        "",
        "**Notes on the counts:**",
        "",
        "- Rows can appear under several scripts (a name mixing Latin and Devanagari counts "
        "twice), so script rows do not sum to the row total.",
        "- Files are read in chunks, so nothing here depends on the whole file fitting in memory.",
        "- NA coercion is disabled: `NULL`, `nan` and friends are counted as the literal text "
        "they are, not silently converted to missing values.",
        "- Example values are sampled from the first 100,000 rows of a file.",
        "",
    ]

    total_rows = sum(p.rows for p in profiles)
    lines.append("## Summary")
    lines.append("")
    rows = []
    for profile in profiles:
        name_col = profile.columns.get(NAME_COL)
        country_mix = (
            ", ".join(f"{label} ({_fmt_pct(count, profile.rows)})" for label, count in profile.countries_ordered())
            if profile.country_rows
            else "n/a (no country column)"
        )
        rows.append([
            f"`{Path(profile.path).name}`",
            _fmt_int(profile.rows),
            country_mix,
            f"{name_col.pct(name_col.non_ascii_rows):.2f}%" if name_col else "n/a",
            _fmt_int(name_col.noise_token_rows) if name_col else "n/a",
            _fmt_int(name_col.short_values) if name_col else "n/a",
            _fmt_int(name_col.bracket_rows) if name_col else "n/a",
        ])
    lines.extend(_table(
        ["file", "rows", "country mix", f"{NAME_COL} non-ASCII", f"{NAME_COL} rows w/ noise",
         f"{NAME_COL} len < 3", f"{NAME_COL} rows w/ brackets"],
        rows,
    ))
    lines.append("")
    lines.append(f"**Total rows profiled: {_fmt_int(total_rows)}** across {len(profiles)} file(s).")
    lines.append("")

    for profile in profiles:
        lines.extend(format_file_report(profile))

    return "\n".join(lines).rstrip() + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile the entity-resolution TSV source files.")
    parser.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR), help="directory holding train/ and test/")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="subdirectories to scan")
    parser.add_argument("--pattern", default="*.tsv", help="glob for files inside each split directory")
    parser.add_argument(
        "--out",
        default=None,
        help=f"write the markdown report here instead of stdout (the checked-in report is {DEFAULT_REPORT_PATH})",
    )
    parser.add_argument("--chunksize", type=int, default=CHUNKSIZE, help="rows per streaming chunk")
    args = parser.parse_args(argv)

    files = discover_data_files(args.dataset_dir, args.splits, args.pattern)
    if not files:
        print(f"No {args.pattern} files under {args.dataset_dir} for splits {args.splits}.")
        return 1

    profiles = []
    for path in files:
        print(f"profiling {path} ...", flush=True)
        profile = profile_file(path, chunksize=args.chunksize)
        print(f"  {_fmt_int(profile.rows)} rows in {profile.elapsed_s:.1f}s", flush=True)
        profiles.append(profile)

    out_path = Path(args.out) if args.out else None
    command = (
        f"python -m src.preprocessing.profile_data --dataset-dir {args.dataset_dir} "
        f"--out {out_path or DEFAULT_REPORT_PATH}"
    )
    report = format_report(profiles, dataset_dir=args.dataset_dir, command=command)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"wrote {out_path} ({len(report.splitlines())} lines)")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
