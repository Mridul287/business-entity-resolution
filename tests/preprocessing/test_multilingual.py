"""
Tests for `src.preprocessing.multilingual`.

The interesting property here is calibration, not just "does it run". A wrong
codepoint range for, say, Gujarati still returns *a* script for a Gujarati
name, so a test that only checks "something was detected" passes while the flag
is useless. Every script assertion below is therefore a literal: the exact
script set for each real row, written out by hand, never derived from
`detect_scripts`.

Inputs are the 39 real rows frozen in `tests/multilingual_sample.py`, which
never touch `dataset/` at test time, plus the synthetic 14-row frame for the
cases that are about pipeline wiring rather than about real scripts.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.preprocessing.clean_text import clean_dataframe
from src.preprocessing.multilingual import (
    CANONICAL_ADDRESS_OUTPUT,
    DEFAULT_STATE_TABLE,
    INDIA_SCRIPT_FLAG,
    INDIC_SCRIPTS,
    LATIN_SCRIPT,
    NON_ASCII_SCRIPT,
    NON_LATIN_FLAG,
    add_canonical_state,
    add_indian_script_name_flag,
    add_multilingual_columns,
    add_non_latin_name_flag,
    canonicalize_state,
    detect_scripts,
    flag_indian_script,
    flag_non_latin,
    is_strictly_non_latin,
    load_state_table,
)
from src.preprocessing.profile_data import SCRIPT_NAMES, profile_dataframe
from tests.multilingual_sample import (
    SAMPLE_ORIGINS,
    SAMPLE_ROWS,
    SAMPLE_ROW_COUNT,
    sample_source_df,
)

# --------------------------------------------------------------------------
# Script detection, as literals. entity_id -> the exact set detect_scripts must
# return for that row's raw business_name.
#
# Three rows per Indic script; three mixed Latin+Indic rows; eight plain ASCII
# rows; one accented-Latin row. The ZWNJ row is listed under its *raw* reading
# on purpose -- see test_zwnj_is_not_a_script below.
# --------------------------------------------------------------------------
EXPECTED_NAME_SCRIPTS: dict[str, set[str]] = {
    # --- Devanagari ---
    "SAMP-001": {"Devanagari"},
    "SAMP-002": {"Devanagari"},
    "SAMP-003": {"Devanagari"},
    # --- Gujarati ---
    "SAMP-004": {"Gujarati"},
    "SAMP-005": {"Gujarati"},
    "SAMP-006": {"Gujarati"},
    # --- Gurmukhi ---
    "SAMP-007": {"Gurmukhi"},
    "SAMP-008": {"Gurmukhi"},
    "SAMP-009": {"Gurmukhi"},
    # --- Telugu ---
    "SAMP-010": {"Telugu"},
    # Raw form carries a U+200C ZERO WIDTH NON-JOINER, which is real Telugu
    # orthography and is not a letter in any tracked block.
    "SAMP-011": {"Telugu", NON_ASCII_SCRIPT},
    "SAMP-012": {"Telugu"},
    # --- Tamil ---
    "SAMP-013": {"Tamil"},
    "SAMP-014": {"Tamil"},
    "SAMP-015": {"Tamil"},
    # --- Kannada ---
    "SAMP-016": {"Kannada"},
    "SAMP-017": {"Kannada"},
    "SAMP-018": {"Kannada"},
    # --- Malayalam ---
    "SAMP-019": {"Malayalam"},
    "SAMP-020": {"Malayalam"},
    # A genuinely mixed-script name, straight out of the file: a Malayalam
    # head ("ഗ്ലോബൽ") on a Bengali tail ("ভാરત"). Not a fixture mistake.
    "SAMP-021": {"Malayalam", "Bengali"},
    # --- Bengali ---
    "SAMP-022": {"Bengali"},
    "SAMP-023": {"Bengali"},
    "SAMP-024": {"Bengali"},
    # --- Oriya ---
    "SAMP-025": {"Oriya"},
    "SAMP-026": {"Oriya"},
    "SAMP-027": {"Oriya"},
    # --- transliterated name + native script, in ASCII Latin + one Indic ---
    # The Latin half is plain ASCII, so it contributes no script at all: only
    # the Indic half is reported. This is why the flag is a poor proxy for
    # "is this name written in Latin".
    "SAMP-028": {"Devanagari"},
    "SAMP-029": {"Bengali"},
    "SAMP-030": {"Kannada"},
    # --- plain ASCII: nothing ---
    "SAMP-031": set(),
    "SAMP-032": set(),
    "SAMP-033": set(),
    "SAMP-034": set(),
    "SAMP-035": set(),
    "SAMP-036": set(),
    "SAMP-037": set(),
    "SAMP-038": set(),
    # --- accented Latin: non-ASCII, but no transliteration risk ---
    "SAMP-039": {LATIN_SCRIPT},
}

# How many rows set each flag, derived from the table above by hand so the two
# numbers below are not just a recount of the implementation.
EXPECTED_FLAGGED = 31   # 39 - the 8 plain-ASCII rows
EXPECTED_STRICT = 30    # 31 - SAMP-039, which is accented Latin only


def test_the_sample_is_the_size_the_annotations_assume() -> None:
    """A drifting row count would silently weaken every expectation below."""
    assert SAMPLE_ROW_COUNT == 39
    assert set(EXPECTED_NAME_SCRIPTS) == {row["entity_id"] for row in SAMPLE_ROWS}


@pytest.mark.parametrize("entity_id", sorted(EXPECTED_NAME_SCRIPTS))
def test_detect_scripts_on_real_rows(entity_id: str) -> None:
    row = next(r for r in SAMPLE_ROWS if r["entity_id"] == entity_id)
    assert detect_scripts(row["business_name"]) == EXPECTED_NAME_SCRIPTS[entity_id]


def test_every_required_script_is_exercised() -> None:
    """
    Guards the reason this file exists: a range that drifted by a block would
    still return *some* script, so the sample must actually contain all nine.
    """
    seen = set().union(*EXPECTED_NAME_SCRIPTS.values())
    assert INDIC_SCRIPTS <= seen
    assert len(INDIC_SCRIPTS) == 9
    # And they are the profiler's names, so the two tools cannot drift apart.
    assert INDIC_SCRIPTS <= set(SCRIPT_NAMES)


def test_plain_english_is_detected_as_nothing() -> None:
    """The negative case, called out separately because it is what S1 relies on."""
    ascii_ids = [i for i, s in EXPECTED_NAME_SCRIPTS.items() if not s]
    assert len(ascii_ids) == 8
    for entity_id in ascii_ids:
        row = next(r for r in SAMPLE_ROWS if r["entity_id"] == entity_id)
        assert row["business_name"].isascii()
        assert detect_scripts(row["business_name"]) == set()
        assert is_strictly_non_latin(row["business_name"]) is False


def test_accents_are_latin_not_indic() -> None:
    """
    The one row that separates the two flags, and all of S1's 2.35%.

    `is_strictly_non_latin` means "contains one of the nine Indic scripts", so
    it is False here even though the broad flag is True. The docstring is
    explicit about this, and the distinction is the whole point of having two
    columns: the France rows are non-ASCII but carry no transliteration risk.
    """
    assert detect_scripts("SCI Ptit Àmicale") == {LATIN_SCRIPT}
    assert is_strictly_non_latin("SCI Ptit Àmicale") is False
    assert is_strictly_non_latin("Café बालाजी Private Ltd") is True


def test_strict_reading_excludes_unclassified_and_latin() -> None:
    """
    The strict reading keys off the nine Indic blocks and nothing else, so a
    ZWNJ-only or accented-Latin row is out. On this sample that is exactly the
    one accented row plus the ZWNJ row, which still has Telugu in it.
    """
    assert is_strictly_non_latin("SCI Ptit Àmicale") is False
    assert is_strictly_non_latin("ఇన్వెస్ట్‌మెంట్") is True  # ZWNJ is not what carries it


def test_a_mixed_name_reports_both_scripts() -> None:
    assert detect_scripts("Café बालाजी Private Ltd") == {"Devanagari", LATIN_SCRIPT}


def test_zwnj_is_unclassified_but_the_name_is_still_flagged() -> None:
    """
    U+200C is legitimate Telugu orthography, is in no tracked block, and is
    *not* removed by clean_text -- a real row keeps it. So it shows up as
    unclassified next to Telugu, and the row is flagged on the strength of the
    Telugu around it. This corrects an earlier claim in this module's docs that
    the cleaner deleted ZWNJ; it does not.
    """
    raw = next(r for r in SAMPLE_ROWS if r["entity_id"] == "SAMP-011")["business_name"]
    assert "\u200c" in raw
    assert detect_scripts(raw) == {"Telugu", NON_ASCII_SCRIPT}
    cleaned = clean_dataframe(pd.DataFrame([{"business_name": raw}], columns=["business_name"]))
    assert "\u200c" in cleaned["business_name_clean"][0]
    assert detect_scripts(cleaned["business_name_clean"][0]) == {"Telugu", NON_ASCII_SCRIPT}


@pytest.mark.parametrize("text", ["", "   ", "Balaji Traders", "12345", "NULL"])
def test_non_script_input_returns_empty(text: str) -> None:
    assert detect_scripts(text) == set()
    assert is_strictly_non_latin(text) is False


# --------------------------------------------------------------------------
# State canonicalisation
# --------------------------------------------------------------------------
# `country` is passed explicitly throughout, because that argument is load
# bearing: two-letter codes are only expanded on Indian rows. The consequences
# are asserted separately below rather than hidden in these cases.
REQUIRED_STATE_MAPPINGS: list[tuple[str, str]] = [
    ("महाराष्ट्र", "Maharashtra"),
    ("राजस्थान", "Rajasthan"),
    ("ਪੰਜਾਬ", "Punjab"),
    ("ગુજરાત", "Gujarat"),
    ("NAGPUR, महाराष्ट्र", "NAGPUR, Maharashtra"),
    ("MOHALI, ਪੰਜਾਬ", "MOHALI, Punjab"),
    ("VADODARA, ગુજરાત", "VADODARA, Gujarat"),
    ("Sector 18, Gurugram, Haryana", "Sector 18, Gurugram, Haryana"),
    ("Salt Lake, Kolkata, West Bengal", "Salt Lake, Kolkata, West Bengal"),
    # Only the state half is rewritten: the city is not translated.
    ("पुणे, महाराष्ट्र", "पुणे, Maharashtra"),
    (" enquiries: Karnataka ", " enquiries: Karnataka "),
]


@pytest.mark.parametrize("text,expected", REQUIRED_STATE_MAPPINGS)
def test_required_canonical_mappings(text: str, expected: str) -> None:
    assert canonicalize_state(text, country="India") == expected


def test_english_and_unmatched_values_pass_through_unchanged() -> None:
    for text in [
        "12 Somewhere Road, Nowhere",
        "TX, PFLUGERVILLE, 13903- CYPRESS DRIVE",
        "MA, 25 NIAGARA STREET, NULL, SPRINGFIELD",
        "",
    ]:
        assert canonicalize_state(text, country="India") == text
        assert canonicalize_state(text, country="US") == text


def test_canonicalisation_is_case_insensitive_for_english() -> None:
    """
    Addresses write the state in any casing, so all of these have to match.

    Matched by spelling each ASCII letter as a two-character class rather than
    with re.IGNORECASE, which would let U+212A KELVIN SIGN match "k" and
    U+017F LATIN SMALL LETTER LONG S match "s".
    """
    assert canonicalize_state("pune, maharashtra", country="India") == "pune, Maharashtra"
    assert canonicalize_state("PUNE, MAHARASHTRA", country="India") == "PUNE, Maharashtra"
    assert canonicalize_state("Pune, MaHaRaShTrA", country="India") == "Pune, Maharashtra"


def test_case_insensitivity_cannot_be_faked_by_a_unicode_lookalike() -> None:
    """The Kelvin sign and long s are letters to Unicode, not to an address."""
    kelvin_maharashtra = "Pune, Ma" + chr(0x212A) + "arashtra"
    long_s = "Pune, Maha" + chr(0x17F) + "arashtra"
    assert canonicalize_state(kelvin_maharashtra, country="India") == kelvin_maharashtra
    assert canonicalize_state(long_s, country="India") == long_s


def test_codes_expand_on_indian_rows_only() -> None:
    """
    The bug this guards is measured, not hypothetical: with no country gate,
    "1111 Church Street, Unit 2007, Nashville, TN" became
    "... Nashville, Tamil Nadu" on 9,386 US rows of a 300k sample.
    """
    assert canonicalize_state("Chennai, TN", country="India") == "Chennai, Tamil Nadu"
    assert canonicalize_state("Chennai, TN", country="US") == "Chennai, TN"
    assert canonicalize_state("Nashville, TN", country="US") == "Nashville, TN"
    assert canonicalize_state("Bengaluru, KA", country="India") == "Bengaluru, Karnataka"
    assert canonicalize_state("Bengaluru, KA", country="US") == "Bengaluru, KA"


def test_us_abbreviations_that_are_also_indian_codes_survive() -> None:
    """AR, LA, MN and TN are Arkansas/Louisiana/Minnesota/Tennessee over here."""
    for text, expected in [
        ("174 Stover Road, De Queen, AR", "174 Stover Road, De Queen, AR"),
        ("2103 Milwaukee Avenue, Minneapolis, MN", "2103 Milwaukee Avenue, Minneapolis, MN"),
        ("Baton Rouge, LA", "Baton Rouge, LA"),
    ]:
        assert canonicalize_state(text, country="US") == expected
        # ...and the same token still means the Indian state on an Indian row.
    assert canonicalize_state("De Queen, AR", country="India") == "De Queen, Arunachal Pradesh"
    assert canonicalize_state("Nashville, TN", country="India") == "Nashville, Tamil Nadu"


def test_us_township_and_road_abbreviations_are_not_states() -> None:
    """
    "TR 253" is an Ohio township route and "UP RIVER RD" a road name. Every
    two-letter code is gated for exactly this reason, not only the four that
    collide with a US state name.
    """
    for text in [
        "6024 TR 253, PO BOX 8978, OXFORD TWP, OH",
        "229 UP RIVER RD, WINTHROP, AR",
        "1950 Sandy Lake Drive, Bldg CH, Grove City, OH",
    ]:
        assert canonicalize_state(text, country="US") == text


def test_delhi_is_a_city_in_new_york() -> None:
    assert canonicalize_state("17 SHERWOOD RD, DELHI, NY", country="US") == "17 SHERWOOD RD, DELHI, NY"
    assert canonicalize_state("15165-A DELHI AVENUE, PARKER, CO", country="US") == "15165-A DELHI AVENUE, PARKER, CO"
    # Still canonicalised where it is the state/UT.
    assert canonicalize_state("DELHI", country="India") == "Delhi"


def test_a_state_name_inside_a_longer_word_is_not_rewritten() -> None:
    """
    Token boundaries matter: "Maharashtrian" contains "Maharashtra" as a prefix
    but is not the state, and rewriting it would corrupt a real name.
    """
    assert canonicalize_state("Maharashtrian Foods", country="India") == "Maharashtrian Foods"
    assert canonicalize_state("Punjabibagh", country="India") == "Punjabibagh"


def test_canonicalisation_is_idempotent() -> None:
    for text, _ in REQUIRED_STATE_MAPPINGS:
        for country in ("India", "US"):
            once = canonicalize_state(text, country=country)
            assert canonicalize_state(once, country=country) == once


def test_state_table_covers_every_state_and_ut() -> None:
    table = load_state_table()
    canonical = set(table.values())
    assert len(canonical) == 36
    # 28 states + 8 union territories.
    assert len([c for c in canonical if c.endswith(" Pradesh")]) >= 3
    for name in ["Maharashtra", "Gujarat", "Punjab", "Kerala", "Delhi", "Goa"]:
        assert name in canonical


def test_state_table_is_clearly_public_reference_data() -> None:
    """
    The brief is explicit that this file is a public gazetteer, not a lookup
    table harvested from the businesses. If someone swaps in scraped data the
    methodology note has to change with it, so the note is asserted, not just
    written in a comment somewhere.
    """
    payload = json.loads(DEFAULT_STATE_TABLE.read_text(encoding="utf-8"))
    methodology = " ".join(payload["methodology_note"].lower().split())
    assert "public" in methodology
    assert "gazetteer" in methodology or "reference" in methodology
    # Structurally there is nowhere for a business record to hide: every unit is
    # a canonical name, a kind, and name lists, and every leaf is a string.
    for unit in payload["states"]:
        assert set(unit) == {"canonical", "kind", "names"}
        assert all(isinstance(v, list) for v in unit["names"].values())
        assert all(isinstance(n, str) for v in unit["names"].values() for n in v)
    # And no entity id from this dataset is quoted anywhere in the file.
    serialised = json.dumps(payload).lower()
    for marker in ["entity_id", "s1-", "s2-", "s3-", "matched_entity_ids"]:
        assert marker not in serialised


def test_the_gazetteer_covers_all_36_units_in_the_required_scripts() -> None:
    """28 states + 8 UTs, each with the four scripts the brief names."""
    payload = json.loads(DEFAULT_STATE_TABLE.read_text(encoding="utf-8"))
    units = payload["states"]
    assert len(units) == 36
    assert {u["kind"] for u in units} == {"state", "union_territory"}
    assert sum(1 for u in units if u["kind"] == "state") == 28
    for unit in units:
        for script in ("Devanagari", "Gujarati", "Gurmukhi", "Telugu"):
            assert unit["names"].get(script), f"{unit['canonical']} has no {script}"


# --------------------------------------------------------------------------
# Dataframe wiring
# --------------------------------------------------------------------------
def test_flags_and_canonical_address_on_the_real_sample() -> None:
    out = add_multilingual_columns(clean_dataframe(sample_source_df))
    assert len(out) == SAMPLE_ROW_COUNT
    assert out[NON_LATIN_FLAG].sum() == EXPECTED_FLAGGED
    assert out[INDIA_SCRIPT_FLAG].sum() == EXPECTED_STRICT
    # The flags are real booleans, not truthy objects.
    assert set(out[NON_LATIN_FLAG].unique()) <= {True, False}
    assert out[CANONICAL_ADDRESS_OUTPUT].notna().all()


def test_canonicalisation_lands_in_its_own_column_and_leaves_the_input_alone() -> None:
    frame = pd.DataFrame(
        [{"business_address_clean": "पुणे, महाराष्ट्र"}], columns=["business_address_clean"]
    )
    out = add_canonical_state(frame)
    assert out[CANONICAL_ADDRESS_OUTPUT].tolist() == ["पुणे, Maharashtra"]
    assert frame["business_address_clean"].tolist() == ["पुणे, महाराष्ट्र"]


def test_a_us_row_is_never_given_an_indian_state() -> None:
    """
    The end-to-end version of the bug this gate exists for, at frame level:
    `add_canonical_state` has to read the country column, not just the address.
    """
    frame = pd.DataFrame(
        [
            {"business_address_clean": "1111 Church Street, Nashville, TN", "country": "US"},
            {"business_address_clean": "2103 Milwaukee Avenue, Minneapolis, MN", "country": "US"},
            {"business_address_clean": "17 Sherwood Rd, DELHI, NY", "country": "US"},
            {"business_address_clean": "12/4, Gandhi Road, Chennai, TN", "country": "India"},
        ]
    )
    out = add_canonical_state(frame)
    assert out[CANONICAL_ADDRESS_OUTPUT].tolist() == [
        "1111 Church Street, Nashville, TN",
        "2103 Milwaukee Avenue, Minneapolis, MN",
        "17 Sherwood Rd, DELHI, NY",
        "12/4, Gandhi Road, Chennai, Tamil Nadu",
    ]


def test_canonicalisation_without_a_country_column_still_works() -> None:
    """
    A frame with no country column keeps the old whole-key behaviour rather than
    silently doing nothing, so a caller who has no country cannot end up with a
    column of unchanged values and no signal that the gate was not applied.
    """
    frame = pd.DataFrame(
        [{"business_address_clean": "Chennai, TN"}], columns=["business_address_clean"]
    )
    assert add_canonical_state(frame)[CANONICAL_ADDRESS_OUTPUT].tolist() == [
        "Chennai, Tamil Nadu"
    ]


def test_flags_are_defined_on_the_clean_column_not_the_raw_one() -> None:
    """
    A stray accent that `clean_text` strips is non-ASCII raw and ASCII clean, so
    this is the case that separates the two. It is exactly the 15 rows of
    test_source2 that made the flag disagree with the published rate before the
    column being compared was fixed.
    """
    raw = "MUSIQUE JÂCQUES"
    frame = clean_dataframe(pd.DataFrame([{"business_name": raw}], columns=["business_name"]))
    out = add_multilingual_columns(frame)
    assert detect_scripts(raw) == {LATIN_SCRIPT}                   # raw is non-ASCII
    assert detect_scripts(out["business_name_clean"][0]) == set()  # clean is not
    assert bool(out[NON_LATIN_FLAG][0]) is False


def test_multilingual_columns_do_not_mutate_their_input() -> None:
    frame = clean_dataframe(sample_source_df)
    before = frame.copy(deep=True)
    add_multilingual_columns(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_flagging_tolerates_missing_and_null_names() -> None:
    frame = pd.DataFrame(
        {"business_name_clean": ["श्री बालाजी", None, float("nan"), "", 42]},
        columns=["business_name_clean"],
    )
    out = add_non_latin_name_flag(frame)
    assert out[NON_LATIN_FLAG].tolist() == [True, False, False, False, False]


def test_missing_columns_are_skipped_rather_than_raising() -> None:
    """Mirrors `clean_dataframe`: a frame without the columns is left alone."""
    out = add_multilingual_columns(pd.DataFrame({"entity_id": ["x"]}, columns=["entity_id"]))
    assert list(out.columns) == ["entity_id"]


def test_vectorised_flag_agrees_with_per_row_detection_on_every_sample_row() -> None:
    """
    The vectorised regex and the per-character table are two implementations of
    one definition; on 39 real rows they must agree exactly, including the
    accented-Latin row and the unclassified ZWNJ row.
    """
    cleaned = clean_dataframe(sample_source_df)["business_name_clean"]
    vectorised = flag_non_latin(cleaned)
    per_row = cleaned.map(detect_scripts).map(bool)
    assert vectorised.tolist() == per_row.tolist()
    assert vectorised.sum() == EXPECTED_FLAGGED


def test_the_flag_is_exactly_the_profilers_non_ascii_measurement() -> None:
    """
    Same guarantee the 300k-row self-check makes against real data, asserted
    here on 39 real rows so a regression is caught by the unit suite too.
    """
    out = add_multilingual_columns(clean_dataframe(sample_source_df))
    profile = profile_dataframe(out, text_columns=("business_name_clean",))
    profiler_rows = profile.col("business_name_clean").non_ascii_rows
    assert int(out[NON_LATIN_FLAG].sum()) == profiler_rows


def test_strict_flag_is_a_strict_subset_of_the_broad_flag() -> None:
    out = add_multilingual_columns(clean_dataframe(sample_source_df))
    assert (out[INDIA_SCRIPT_FLAG] & ~out[NON_LATIN_FLAG]).sum() == 0
    # The difference is exactly the accented-Latin row.
    differ = out[out[NON_LATIN_FLAG] & ~out[INDIA_SCRIPT_FLAG]]
    assert differ["entity_id"].tolist() == ["SAMP-039"]


def test_strict_flag_matches_per_row_detection() -> None:
    cleaned = clean_dataframe(sample_source_df)["business_name_clean"]
    expected = cleaned.map(lambda t: bool(detect_scripts(t) & INDIC_SCRIPTS))
    assert flag_indian_script(cleaned).tolist() == expected.tolist()


def test_the_two_flags_can_be_added_independently() -> None:
    frame = clean_dataframe(sample_source_df)
    only_broad = add_non_latin_name_flag(frame)
    assert INDIA_SCRIPT_FLAG not in only_broad.columns
    only_strict = add_indian_script_name_flag(frame)
    assert NON_LATIN_FLAG not in only_strict.columns


def test_sample_origins_cover_the_three_embedding_scopes() -> None:
    """The blocking tests scope on source file, so the sample has to span them."""
    assert set(SAMPLE_ORIGINS.values()) >= {"test_source1", "test_source2", "test_source3"}
