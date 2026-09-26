"""
Fixtures for the preprocessing tests.

All inputs are synthetic and hand-written. Nothing here reads `dataset/`, so
the suite runs in milliseconds and passes on a clone with no data in it.

SYNTHETIC_SOURCE_ROWS is the reference frame for `profile_data`: one row per
behaviour we care about, each annotated with what it is there to prove. The
profiler tests assert exact counts against this frame, so when a row is added
here the expected counts have to be updated deliberately rather than drifting.
"""
from __future__ import annotations

import pandas as pd
import pytest

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# 14 rows. business_name / business_address / country.
SYNTHETIC_SOURCE_ROWS: list[dict[str, str]] = [
    {
        "entity_id": "S1-00001",
        "business_name": "Shree Balaji Traders",          # clean ASCII, the baseline row
        "business_address": "12 MG Road, Bengaluru",
        "country": "India",
    },
    {
        "entity_id": "S1-00002",
        "business_name": "श्री बालाजी ट्रेडर्स",                # Devanagari
        "business_address": "Sector 18, Gurugram",
        "country": "India",
    },
    {
        "entity_id": "S1-00003",
        "business_name": "શ્રી બાલાજી ટ્રેડર્સ",                  # Gujarati
        "business_address": "Station Road, Ahmedabad",
        "country": "India",
    },
    {
        "entity_id": "S1-00004",
        "business_name": "NULL",                          # literal noise token
        "business_address": "NULL",                       # ditto, in the address column
        "country": "US",
    },
    {
        "entity_id": "S1-00005",
        "business_name": "###",                           # literal noise token
        "business_address": "###",
        "country": "US",
    },
    {
        "entity_id": "S1-00006",
        "business_name": "##",                            # must not also count as "###"
        "business_address": "##",                         # 2 chars -> a low-information value
        "country": "US",
    },
    {
        "entity_id": "S1-00007",
        "business_name": "Café Zürich [Limited]",          # accented Latin + bracket-wrapped token
        "business_address": "63 Rue de Rivoli, Lille",
        "country": "France",
    },
    {
        "entity_id": "S1-00008",
        "business_name": "AB",                            # 2 chars -> low-information
        "business_address": "45 Somewhere",
        "country": "US",
    },
    {
        "entity_id": "S1-00009",
        "business_name": "",                              # blank -> low-information
        "business_address": "",
        "country": "India",
    },
    {
        "entity_id": "S1-00010",
        "business_name": "N/A",
        "business_address": "N/A",
        "country": "US",
    },
    {
        "entity_id": "S1-00011",
        "business_name": "NONE",
        "business_address": "NONE",
        "country": "India",
    },
    {
        "entity_id": "S1-00012",
        "business_name": "nan",
        "business_address": "nan",
        "country": "US",
    },
    {
        "entity_id": "S1-00013",
        "business_name": "***",
        "business_address": "***",
        "country": "",                                    # missing country label
    },
    {
        "entity_id": "S1-00014",
        "business_name": "Café बालाजी Private Ltd",        # two scripts in one row
        "business_address": "एम ब्लॉक, दिल्ली",               # Devanagari address
        "country": "India",
    },
]

EXPECTED_ROWS = len(SYNTHETIC_SOURCE_ROWS)
EXPECTED_COUNTRIES = {"India": 6, "US": 6, "France": 1, "(blank)": 1}

# business_name expectations over SYNTHETIC_SOURCE_ROWS
EXPECTED_NAME_SCRIPTS = {
    "Devanagari": 2,          # rows 2 and 14
    "Gujarati": 1,            # row 3
    "Latin (accented)": 2,    # rows 7 and 14
}
EXPECTED_NAME_NON_ASCII_ROWS = 4
EXPECTED_NAME_NOISE = {
    "NULL": 1, "N/A": 1, "NONE": 1, "nan": 1, "##": 1, "###": 1, "***": 1,
}
EXPECTED_NAME_NOISE_ROWS = 7
EXPECTED_NAME_BLANK = 1
# "##" (row 6), "AB" (row 8) and "" (row 9). The 3-char junk tokens ("###",
# "N/A", "nan", "***") are exactly 3 characters and so are not short.
EXPECTED_NAME_SHORT = 3
EXPECTED_NAME_SHORT_EXAMPLES = ["##", "AB", ""]
EXPECTED_NAME_BRACKET_TOKENS = 1
EXPECTED_NAME_BRACKET_ROWS = 1
EXPECTED_NAME_BRACKET_EXAMPLES = ["[Limited]"]

# business_address expectations over the same rows
EXPECTED_ADDRESS_SCRIPTS = {"Devanagari": 1}  # row 14 only
EXPECTED_ADDRESS_NON_ASCII_ROWS = 1
EXPECTED_ADDRESS_NOISE = EXPECTED_NAME_NOISE   # mirrored column
EXPECTED_ADDRESS_NOISE_ROWS = 7
EXPECTED_ADDRESS_BLANK = 1
EXPECTED_ADDRESS_SHORT = 2                    # "##" (2 chars) and "" (blank)
EXPECTED_ADDRESS_BRACKET_TOKENS = 0


def _as_source_frame(rows: list[dict[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS).astype(str)


@pytest.fixture
def synthetic_source_df() -> pd.DataFrame:
    """The reference 14-row source frame, shaped like dataset/*/[split]_source*.tsv."""
    return _as_source_frame(SYNTHETIC_SOURCE_ROWS)


@pytest.fixture
def synthetic_source_df_first_half(synthetic_source_df: pd.DataFrame) -> pd.DataFrame:
    """First 7 rows, for testing that chunked profiling matches one-pass profiling."""
    return synthetic_source_df.iloc[:7].reset_index(drop=True)


@pytest.fixture
def synthetic_source_df_second_half(synthetic_source_df: pd.DataFrame) -> pd.DataFrame:
    """Last 7 rows, the second chunk of the split above."""
    return synthetic_source_df.iloc[7:].reset_index(drop=True)


@pytest.fixture
def synthetic_ground_truth_df() -> pd.DataFrame:
    """A three-row ground-truth file in the dataset's comma-joined ID-list format."""
    return pd.DataFrame(
        {
            "source1_entity_id": ["S1-00001", "S1-00002", "S1-00009"],
            "matched_entity_ids": ["S2-00001,S2-00002", "S3-00001", ""],
        }
    )
