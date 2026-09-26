"""
Tests for src/preprocessing/profile_data.py.

Every test here runs against the synthetic fixture from conftest.py. The real
dataset/ files are hundreds of megabytes and are not committed, so profiling
them from a test would be slow at best and a broken clone at worst. The real
data is profiled by running the module (see docs/data_profile_report.md); these
tests pin the counting rules that report depends on.

Expected counts are written out literally rather than recomputed, so a change in
the profiler's semantics has to be acknowledged here instead of quietly
agreeing with itself.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing.profile_data import (
    BLANK_LABEL,
    NOISE_TOKEN_PATTERNS,
    SCRIPT_NAMES,
    discover_data_files,
    format_report,
    profile_dataframe,
    profile_file,
)

from .conftest import (
    EXPECTED_ADDRESS_BLANK,
    EXPECTED_ADDRESS_BRACKET_TOKENS,
    EXPECTED_ADDRESS_NON_ASCII_ROWS,
    EXPECTED_ADDRESS_NOISE,
    EXPECTED_ADDRESS_NOISE_ROWS,
    EXPECTED_ADDRESS_SCRIPTS,
    EXPECTED_ADDRESS_SHORT,
    EXPECTED_COUNTRIES,
    EXPECTED_NAME_BLANK,
    EXPECTED_NAME_BRACKET_EXAMPLES,
    EXPECTED_NAME_BRACKET_ROWS,
    EXPECTED_NAME_BRACKET_TOKENS,
    EXPECTED_NAME_NON_ASCII_ROWS,
    EXPECTED_NAME_NOISE,
    EXPECTED_NAME_NOISE_ROWS,
    EXPECTED_NAME_SCRIPTS,
    EXPECTED_NAME_SHORT,
    EXPECTED_NAME_SHORT_EXAMPLES,
    EXPECTED_ROWS,
    SOURCE_COLUMNS,
)

ALL_NOISE_TOKENS = ["NULL", "N/A", "NONE", "nan", "##", "###", "***"]


@pytest.fixture
def profile(synthetic_source_df: pd.DataFrame):
    """The profiler's output for the reference synthetic frame."""
    return profile_dataframe(synthetic_source_df, path="synthetic_source.tsv")


@pytest.fixture
def name_profile(profile):
    return profile.columns["business_name"]


@pytest.fixture
def address_profile(profile):
    return profile.columns["business_address"]


# --------------------------------------------------------------------------
# frame-level counts
# --------------------------------------------------------------------------

def test_row_count_and_schema(profile, synthetic_source_df):
    assert profile.rows == EXPECTED_ROWS
    assert len(synthetic_source_df) == EXPECTED_ROWS
    assert profile.schema == SOURCE_COLUMNS
    assert profile.missing_columns == []
    assert profile.text_columns == ["business_name", "business_address"]
    assert profile.col("business_name").rows == EXPECTED_ROWS


def test_country_value_counts(profile):
    assert profile.country_rows == EXPECTED_ROWS
    assert dict(profile.country_counts) == EXPECTED_COUNTRIES
    # Blank first, then by descending count; equal counts break alphabetically
    # so the report is stable across runs.
    assert profile.countries_ordered() == [
        (BLANK_LABEL, 1), ("India", 6), ("US", 6), ("France", 1),
    ]


def test_country_pct_is_relative_to_file_rows(profile):
    assert profile.country_pct(profile.country_counts["India"]) == pytest.approx(100 * 6 / 14)
    assert profile.country_pct(profile.country_counts[BLANK_LABEL]) == pytest.approx(100 / 14)


# --------------------------------------------------------------------------
# Unicode scripts
# --------------------------------------------------------------------------

def test_non_ascii_row_counts(name_profile, address_profile):
    # business_name: the Devanagari row, the Gujarati row, the accented Latin
    # row, and the row mixing accented Latin with Devanagari.
    assert name_profile.non_ascii_rows == EXPECTED_NAME_NON_ASCII_ROWS
    assert name_profile.pct(name_profile.non_ascii_rows) == pytest.approx(100 * 4 / 14)
    # business_address: only the last row is non-ASCII.
    assert address_profile.non_ascii_rows == EXPECTED_ADDRESS_NON_ASCII_ROWS


def test_script_breakdown_counts(name_profile, address_profile):
    assert name_profile.script_rows == EXPECTED_NAME_SCRIPTS
    assert address_profile.script_rows == EXPECTED_ADDRESS_SCRIPTS


def test_expected_scripts_are_the_ones_under_test():
    # Guards the fixture, not the profiler: if these scripts ever stop being
    # interesting, the assertions above need revisiting on purpose.
    assert {"Devanagari", "Gujarati", "Latin (accented)"} <= set(SCRIPT_NAMES)


def test_ascii_only_row_is_not_counted_as_non_ascii(synthetic_source_df: pd.DataFrame):
    """The clean ASCII row must not show up anywhere in the script counts."""
    clean_only = synthetic_source_df.iloc[[0]]
    clean_profile = profile_dataframe(clean_only)
    assert clean_profile.columns["business_name"].non_ascii_rows == 0
    assert clean_profile.columns["business_name"].script_rows == {}
    assert clean_profile.columns["business_address"].non_ascii_rows == 0


def test_non_bmp_rows_are_zero_for_this_frame(name_profile):
    assert name_profile.non_bmp_rows == 0


# --------------------------------------------------------------------------
# literal noise tokens
# --------------------------------------------------------------------------

def test_noise_token_definition_is_exactly_the_requested_seven():
    assert list(NOISE_TOKEN_PATTERNS) == ALL_NOISE_TOKENS


def test_noise_token_counts(name_profile, address_profile):
    assert name_profile.noise_token_occurrences == EXPECTED_NAME_NOISE
    assert address_profile.noise_token_occurrences == EXPECTED_ADDRESS_NOISE


def test_noise_token_rows_are_counted_once_per_row(name_profile, address_profile):
    assert name_profile.noise_token_rows == EXPECTED_NAME_NOISE_ROWS
    assert name_profile.noise_total == sum(EXPECTED_NAME_NOISE.values())
    assert address_profile.noise_token_rows == EXPECTED_ADDRESS_NOISE_ROWS
    # Per token, occurrences and rows are reported separately: the summary table
    # needs both and must not conflate them.
    assert name_profile.noise_token_rows_by_token == EXPECTED_NAME_NOISE
    assert address_profile.noise_token_rows_by_token == EXPECTED_ADDRESS_NOISE


def test_repeated_token_in_one_row_is_one_row_two_occurrences():
    frame = pd.DataFrame(
        {"business_name": ["NULL NULL", "Acme"], "business_address": ["", ""], "country": ["US"] * 2}
    )
    col = profile_dataframe(frame).columns["business_name"]
    assert col.noise_token_occurrences["NULL"] == 2
    assert col.noise_token_rows_by_token["NULL"] == 1
    assert col.noise_token_rows == 1


def test_hash_tokens_do_not_double_count():
    """
    Only an exact run of two or three hashes is a noise token: "###" is not
    also counted as "##", and a longer run of hashes ("####") is neither.
    """
    frame = pd.DataFrame(
        {"business_name": ["##", "###", "####"], "business_address": ["", "", ""], "country": ["US"] * 3}
    )
    col = profile_dataframe(frame).columns["business_name"]
    assert col.noise_token_occurrences["##"] == 1
    assert col.noise_token_occurrences["###"] == 1
    assert col.noise_token_rows == 2


def test_noise_tokens_match_as_whole_tokens_not_substrings():
    """"NANO" and "NULLABLE" are real-ish words, not the placeholders."""
    frame = pd.DataFrame(
        {
            "business_name": ["NANO Devices", "NULLABLE Corp", "Mansion", "Nanoscope"],
            "business_address": ["", "", "", ""],
            "country": ["US"] * 4,
        }
    )
    col = profile_dataframe(frame).columns["business_name"]
    assert col.noise_token_occurrences == {}
    assert col.noise_token_rows == 0


def test_noise_tokens_survive_as_text_not_as_missing_values(synthetic_source_df: pd.DataFrame):
    """
    The whole point of reading with NA coercion off: "NULL" and "nan" are
    counted as the literal text they are instead of becoming NaN.
    """
    assert synthetic_source_df["business_name"].isna().sum() == 0
    assert profile_dataframe(synthetic_source_df).columns["business_name"].noise_token_occurrences["NULL"] == 1


# --------------------------------------------------------------------------
# low-information business_name values
# --------------------------------------------------------------------------

def test_short_business_name_counts(name_profile):
    # "##", "AB" and the blank; the 3-character junk tokens are not short.
    assert name_profile.short_values == EXPECTED_NAME_SHORT
    assert name_profile.blank_values == EXPECTED_NAME_BLANK
    assert name_profile.short_examples == EXPECTED_NAME_SHORT_EXAMPLES
    assert name_profile.pct(name_profile.short_values) == pytest.approx(100 * 3 / 14)


def test_short_value_threshold_is_under_three_characters(synthetic_source_df: pd.DataFrame):
    """Boundary check: 3 characters is kept, 2 characters is counted."""
    frame = pd.DataFrame(
        {"business_name": ["AB", "ABC", "  A  ", " ABC "], "business_address": [""] * 4, "country": ["US"] * 4}
    )
    col = profile_dataframe(frame).columns["business_name"]
    assert col.short_values == 2  # "AB" and "  A  " after stripping
    assert col.short_examples == ["AB", "A"]
    assert col.blank_values == 0


def test_blank_values_are_counted_separately(address_profile):
    assert address_profile.blank_values == EXPECTED_ADDRESS_BLANK
    # "##" (2 chars) and "" (0 chars) are the short ones here.
    assert address_profile.short_values == EXPECTED_ADDRESS_SHORT


# --------------------------------------------------------------------------
# bracket-wrapped tokens
# --------------------------------------------------------------------------

def test_bracket_token_counts(name_profile, address_profile):
    assert name_profile.bracket_token_occurrences == EXPECTED_NAME_BRACKET_TOKENS
    assert name_profile.bracket_rows == EXPECTED_NAME_BRACKET_ROWS
    assert name_profile.bracket_examples == EXPECTED_NAME_BRACKET_EXAMPLES
    assert address_profile.bracket_token_occurrences == EXPECTED_ADDRESS_BRACKET_TOKENS


def test_multiple_brackets_in_one_row_are_all_counted():
    frame = pd.DataFrame(
        {
            "business_name": ["Acme [Pvt] [Limited]", "Beta [Pvt]", "Gamma Ltd"],
            "business_address": ["", "", ""],
            "country": ["US"] * 3,
        }
    )
    col = profile_dataframe(frame).columns["business_name"]
    assert col.bracket_token_occurrences == 3
    assert col.bracket_rows == 2
    assert col.bracket_examples == ["[Pvt]", "[Limited]"]


# --------------------------------------------------------------------------
# chunked profiling must agree with one-pass profiling
# --------------------------------------------------------------------------

def test_profiling_in_chunks_equals_profiling_in_one_pass(
    synthetic_source_df: pd.DataFrame,
    synthetic_source_df_first_half: pd.DataFrame,
    synthetic_source_df_second_half: pd.DataFrame,
):
    """profile_file streams chunks; the merged result must match the single pass."""
    one_pass = profile_dataframe(synthetic_source_df)
    chunked = profile_dataframe(synthetic_source_df_first_half)
    chunked.merge(profile_dataframe(synthetic_source_df_second_half))

    assert chunked.rows == one_pass.rows == EXPECTED_ROWS
    assert dict(chunked.country_counts) == dict(one_pass.country_counts)
    for column in one_pass.text_columns:
        merged_col, single_col = chunked.columns[column], one_pass.columns[column]
        assert merged_col.non_ascii_rows == single_col.non_ascii_rows
        assert merged_col.script_rows == single_col.script_rows
        assert merged_col.noise_token_occurrences == single_col.noise_token_occurrences
        assert merged_col.noise_token_rows_by_token == single_col.noise_token_rows_by_token
        assert merged_col.noise_token_rows == single_col.noise_token_rows
        assert merged_col.blank_values == single_col.blank_values
        assert merged_col.short_values == single_col.short_values
        assert merged_col.bracket_token_occurrences == single_col.bracket_token_occurrences
        assert merged_col.bracket_rows == single_col.bracket_rows


def test_profile_file_reads_a_synthetic_tsv_with_a_tiny_chunk_size(tmp_path, synthetic_source_df):
    """
    Exercises the streaming reader end to end. The TSV is the synthetic frame
    written to tmp_path, one row per chunk, so the chunk boundaries land in the
    middle of every feature being counted.
    """
    path = tmp_path / "synthetic_source1.tsv"
    synthetic_source_df.to_csv(path, sep="\t", index=False)

    from_disk = profile_file(path, chunksize=1)
    in_memory = profile_dataframe(synthetic_source_df)

    assert from_disk.rows == EXPECTED_ROWS
    assert from_disk.schema == SOURCE_COLUMNS
    assert from_disk.skipped_lines == 0
    assert dict(from_disk.country_counts) == dict(in_memory.country_counts)
    assert from_disk.columns["business_name"].script_rows == EXPECTED_NAME_SCRIPTS
    assert from_disk.columns["business_name"].noise_token_occurrences == EXPECTED_NAME_NOISE
    assert from_disk.columns["business_name"].short_values == EXPECTED_NAME_SHORT
    assert from_disk.columns["business_name"].bracket_token_occurrences == EXPECTED_NAME_BRACKET_TOKENS
    assert from_disk.elapsed_s >= 0


# --------------------------------------------------------------------------
# files the profiler must not crash on
# --------------------------------------------------------------------------

def test_ground_truth_file_shape_is_handled(synthetic_ground_truth_df, tmp_path):
    """The ground-truth file has no text columns; profile it, do not crash."""
    path = tmp_path / "train_ground_truth.tsv"
    synthetic_ground_truth_df.to_csv(path, sep="\t", index=False)

    profile = profile_file(path)
    assert profile.rows == 3
    assert profile.text_columns == []
    assert set(profile.missing_columns) >= {"business_name", "business_address", "country"}
    assert "_No business_name / business_address columns to profile._" in format_report(
        [profile], generated_at="test"
    )


def test_empty_frame_does_not_divide_by_zero():
    empty = pd.DataFrame({c: pd.Series(dtype=str) for c in SOURCE_COLUMNS})
    profile = profile_dataframe(empty)
    assert profile.rows == 0
    assert profile.country_pct(0) == 0.0
    assert profile.columns["business_name"].pct(0) == 0.0
    assert format_report([profile], generated_at="test")  # renders without dividing by zero


def test_all_nan_column_is_treated_as_blank_text():
    frame = pd.DataFrame(
        {"business_name": [None, "Acme"], "business_address": [None, None], "country": [None, "US"]}
    )
    profile = profile_dataframe(frame)
    assert profile.columns["business_name"].blank_values == 1
    assert profile.country_counts[BLANK_LABEL] == 1


# --------------------------------------------------------------------------
# report rendering
# --------------------------------------------------------------------------

def test_report_renders_a_table_per_file(profile):
    report = format_report([profile], dataset_dir="dataset", generated_at="test")
    assert "## `synthetic_source.tsv`" in report
    assert "**Rows:** 14" in report
    assert "### Country mix" in report
    assert "### Non-ASCII content by Unicode script" in report
    assert "### Literal noise tokens" in report
    assert "### Low-information business_name values" in report
    assert "### Bracket-wrapped tokens in business_name" in report
    assert "Devanagari" in report and "Gujarati" in report and "Latin (accented)" in report
    for token in ALL_NOISE_TOKENS:
        assert f"`{token}`" in report
    # every table row is pipe-delimited and none of the cells are empty
    for line in report.splitlines():
        if line.startswith("| ") and not line.startswith("| ---"):
            assert all(cell.strip() for cell in line.strip("|").split("|"))


def test_report_lists_one_section_per_file(profile, synthetic_ground_truth_df, tmp_path):
    gt_path = tmp_path / "train_ground_truth.tsv"
    synthetic_ground_truth_df.to_csv(gt_path, sep="\t", index=False)
    report = format_report([profile, profile_file(gt_path)], dataset_dir="dataset", generated_at="test")
    assert report.count("\n## `") == 2
    assert "Total rows profiled: 17" in report


def test_discover_data_files_uses_a_stable_split_order(tmp_path):
    for split in ("train", "test"):
        (tmp_path / split).mkdir()
        (tmp_path / split / f"{split}_source1.tsv").write_text("entity_id\n", encoding="utf-8")
        (tmp_path / split / "notes.md").write_text("ignored", encoding="utf-8")
    found = [p.parent.name for p in discover_data_files(tmp_path, splits=("train", "test"))]
    assert found == ["train", "test"]
    assert all(p.name.endswith(".tsv") for p in discover_data_files(tmp_path))


def test_discover_data_files_tolerates_a_missing_split(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "train_source1.tsv").write_text("entity_id\n", encoding="utf-8")
    assert len(discover_data_files(tmp_path, splits=("train", "test"))) == 1
    assert discover_data_files(tmp_path, splits=("validation",)) == []
