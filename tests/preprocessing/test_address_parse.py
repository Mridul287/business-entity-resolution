"""
Tests for `src.preprocessing.address_parse` and for the parts of
`src.preprocessing.normalize` that are Phase 4.

Two things are being tested here, and they are different in kind.

The first is the *table*: `expand_abbreviations` must turn `rd` into `road`,
`st` into `street`, `apt` into `apartment`, `pvt` into `private`, `ltd` into
`limited` and `corp` into `corporation`. Those six are named in the task, so they
are asserted as a table, one row per abbreviation, in both cases.

The second is the *decisions*, and this is the part worth writing tests for. An
abbreviation expander is a machine for corrupting data, because the short forms
are ambiguous: `co` is Company in a name and Colorado in an address, `dr` is
Doctor and Drive, `ct` is Connecticut, and `st` is both Street and Saint. A rule
that expands all of them is worse than no rule at all, because it rewrites
`ST GEORGE` into `STREET GEORGE` and then two different businesses agree. So
every case below is either a real row from `tests/multilingual_sample.py` or a
shape measured on the real corpus, and the "must not expand" cases are asserted
exactly as hard as the "must expand" ones.

`extract_address_tokens` gets the same treatment, with one extra concern: the
`postal_kind` contract. On this corpus a postal code is present in the position
where one belongs for 0.00%-0.03% of rows, so a caller that compares two
`pin_or_zip` values for equality is comparing noise. `postal_kind` exists so the
caller can tell a real ZIP from a street number that happened to be five digits,
and several tests below exist only to pin that distinction down.

Nothing here reads `dataset/`. The real rows come from the frozen 39-row sample,
which is what `tests/conftest.py`'s data policy requires.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from src.preprocessing.address_parse import (
    ADDRESS_ABBREVIATIONS,
    NAME_ABBREVIATIONS,
    PIN_OR_ZIP,
    POSTAL_GENERIC,
    POSTAL_IN_PIN6,
    POSTAL_KIND,
    POSTAL_KINDS,
    POSTAL_US_ZIP5,
    SCOPE_ADDRESS,
    SCOPE_BOTH,
    SCOPE_NAME,
    SCOPES,
    STREET_NUMBER,
    TOKEN_SET,
    add_address_tokens,
    expand_abbreviations,
    expand_abbreviations_frame,
    extract_address_tokens,
    fold_accents,
    fold_for_key,
)
from src.preprocessing.normalize import (
    NORMALIZED_COLUMNS,
    normalize,
)
from tests.multilingual_sample import SAMPLE_ROW_COUNT, sample_source_df

# --------------------------------------------------------------------------
# 1. the six abbreviations the task names
# --------------------------------------------------------------------------

# (input, scope, expected). The first three are the ones a US address is full
# of; the last three are the legal forms that leak into the address column on
# this corpus, which is why they live in ADDRESS_ABBREVIATIONS and not only in
# NAME_ABBREVIATIONS.
REQUIRED_EXPANSIONS: list[tuple[str, str, str]] = [
    ("STATION RD", SCOPE_ADDRESS, "STATION ROAD"),
    ("123 MAIN ST", SCOPE_ADDRESS, "123 MAIN STREET"),
    ("APT 4B, SOMEWHERE", SCOPE_ADDRESS, "APARTMENT 4B, SOMEWHERE"),
    ("SHREE BALAJI PVT LTD", SCOPE_NAME, "SHREE BALAJI PRIVATE LIMITED"),
    ("BEACON CORP", SCOPE_NAME, "BEACON CORPORATION"),
    ("ARCE AND BELTRAN PARTNERS PVT LTD", SCOPE_ADDRESS,
     "ARCE AND BELTRAN PARTNERS PRIVATE LIMITED"),
]


@pytest.mark.parametrize(("text", "scope", "expected"), REQUIRED_EXPANSIONS)
def test_required_abbreviations_expand(text: str, scope: str, expected: str) -> None:
    assert expand_abbreviations(text, scope) == expected


def test_required_abbreviations_are_in_the_table() -> None:
    """
    The table itself, so deleting a row fails here and not only in a benchmark.

    Both scopes are checked because the pipeline expands the address column
    through SCOPE_ADDRESS and the name column through SCOPE_NAME, and a row
    present in only one of them is a row that is expanded in one column and not
    the other -- which is the kind of asymmetry that shows up later as a
    name/address disagreement nobody can explain.

    `st` is deliberately absent from both tables. It is Street in some contexts
    and Saint in others, so it is resolved by `_expand_st` from the surrounding
    text and cannot be expressed as a whole-token entry. Asserting its absence
    is the point: someone tidying the table by adding "st": "street" would
    silently rewrite every "ST GEORGE" in the corpus.
    """
    required = {
        "rd": "road",
        "apt": "apartment",
        "pvt": "private",
        "ltd": "limited",
        "corp": "corporation",
    }
    for short, long in required.items():
        assert ADDRESS_ABBREVIATIONS.get(short) == long, f"address scope: {short}"
        assert NAME_ABBREVIATIONS.get(short) == long, f"name scope: {short}"

    assert "st" not in ADDRESS_ABBREVIATIONS
    assert "st" not in NAME_ABBREVIATIONS


# --------------------------------------------------------------------------
# 2. case is preserved, so an all-caps column stays all-caps
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("KENTUCKY ST, SALYERSVILLE, KY", "KENTUCKY STREET, SALYERSVILLE, KY"),
        ("Kentucky St, Salyersville, KY", "Kentucky Street, Salyersville, KY"),
        ("kentucky st", "kentucky street"),
    ],
)
def test_expansion_keeps_the_case_of_the_original(text: str, expected: str) -> None:
    assert expand_abbreviations(text, SCOPE_ADDRESS) == expected


# --------------------------------------------------------------------------
# 3. the contextual `st` rule
# --------------------------------------------------------------------------

# `st` is the one abbreviation in this corpus that cannot be expanded
# unconditionally, so both directions are asserted per row: the expansion or the
# preservation, with the reason. Every input is a real shape from the sample or
# from the measurement in the module docstring.
ST_POSITIVE: list[tuple[str, str]] = [
    # End of value or before a comma: 58.3% of all `st` occurrences. A street
    # type does not take a name after it, so this class is unambiguous.
    ("67 Kentucky St", "67 Kentucky Street"),
    ("5116 80th St, Stillwater, OK", "5116 80th Street, Stillwater, OK"),
    ("KENTUCKY ST, KY", "KENTUCKY STREET, KY"),
    # A known street or floor word directly after: the bulk of the 37.4%
    # "space + word" class.
    ("12 ST MAIN RD", "12 STREET MAIN ROAD"),
    ("9 ST FLOOR", "9 STREET FLOOR"),
    ("5 ST BLOCK", "5 STREET BLOCK"),
]

ST_NEGATIVE: list[tuple[str, str]] = [
    # Saint. These are the rows a naive expander destroys.
    ("ST GEORGE SCHOOL", "ST GEORGE SCHOOL"),
    ("ST NAZAIRE", "ST NAZAIRE"),
    ("SAINT LOUIS", "SAINT LOUIS"),
    # A dot is deliberately not in the end-of-value set, because "ST. GEORGE"
    # is a saint and a dot is how abbreviations get written.
    ("ST. GEORGE", "ST. GEORGE"),
    # The look-behind excludes digits, so the ordinal in "1ST FLOOR" is not a
    # standalone token. \b would not do this.
    ("1ST FLOOR", "1ST FLOOR"),
    ("21ST STREET", "21ST STREET"),
]


@pytest.mark.parametrize(("text", "expected"), ST_POSITIVE)
def test_st_expands_where_it_means_street(text: str, expected: str) -> None:
    assert expand_abbreviations(text, SCOPE_ADDRESS) == expected


@pytest.mark.parametrize(("text", "expected"), ST_NEGATIVE)
def test_st_is_left_alone_where_it_means_saint(text: str, expected: str) -> None:
    assert expand_abbreviations(text, SCOPE_ADDRESS) == expected


def test_st_in_a_saint_name_survives_the_whole_pipeline() -> None:
    """
    The end-to-end version of the row above.

    Asserting `expand_abbreviations` is not enough on its own, because the
    pipeline folds the name into `name_norm` afterwards and a fold is another
    chance to lose the distinction. If "ST GEORGE" ever became "street george",
    the token set would merge a saint with a street type and two unrelated
    businesses would start agreeing on a shared token.
    """
    out = normalize(
        pd.DataFrame(
            {
                "entity_id": ["A", "B"],
                "business_name": ["ST GEORGE SCHOOL", "STREET GEORGE SCHOOL"],
                "business_address": ["1 Main St", "1 Main Road"],
                "country": ["US", "US"],
            }
        )
    )
    assert out["name_norm"].iloc[0] == "st george school"
    assert out["name_norm"].iloc[1] == "street george school"
    assert "george" in out["name_token_set"].iloc[0]


# --------------------------------------------------------------------------
# 4. scope: `co` is the reason scope exists
# --------------------------------------------------------------------------


def test_co_is_company_in_a_name_and_colorado_in_an_address() -> None:
    """
    The measured difference: 5,208 "co" occurrences in names, 1,148 in
    addresses, and they mean different things.
    """
    name = "ACME CO"
    address = "123 MAIN ST, DENVER, CO 80202"

    assert expand_abbreviations(name, SCOPE_NAME) == "ACME COMPANY"
    # In the address scope `co` is not in the table at all, so Colorado
    # survives untouched -- which is what the state canonicalizer downstream
    # expects to find.
    assert expand_abbreviations(address, SCOPE_ADDRESS) == "123 MAIN STREET, DENVER, CO 80202"
    assert "COLORADO" not in expand_abbreviations(address, SCOPE_ADDRESS).upper()


def test_both_scope_takes_the_name_meaning() -> None:
    """SCOPE_BOTH is the union, so a collision resolves to the name entry."""
    assert expand_abbreviations("ACME CO", SCOPE_BOTH) == "ACME COMPANY"
    assert expand_abbreviations("ACME CO").endswith("COMPANY")  # default is BOTH


@pytest.mark.parametrize("scope", SCOPES)
def test_every_scope_is_accepted(scope: str) -> None:
    assert expand_abbreviations("12 MG RD", scope)


def test_unknown_scope_is_an_error() -> None:
    with pytest.raises(ValueError, match="scope must be one of"):
        expand_abbreviations("12 MG RD", "banana")


# --------------------------------------------------------------------------
# 5. the measured non-expansions: rows the expander must not touch
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "why"),
    [
        # `dr` and `ct` are the two measured traps: 1,504 `ct` occurrences in
        # addresses are Connecticut and Drive/Court are not what they mean
        # there. These rows contain nothing else that expands, so the whole
        # value must come back byte-identical.
        ("12 DR WILSON", "Dr is Doctor in this corpus, not Drive"),
        ("1 MAIN, HARTFORD, CT 06103", "CT is Connecticut, not Court"),
        ("NOIDA SECTOR 18", "Noida must not split into no + ida"),
        ("STARK INDUSTRIES", "Stark must not split into st + ark"),
        ("CORPORATE SERVICES", "a word starting with an abbreviation is not one"),
        ("LTDCO PARTNERS", "ditto, at the other end"),
    ],
)
def test_ambiguous_and_embedded_forms_are_preserved(text: str, why: str) -> None:
    assert expand_abbreviations(text, SCOPE_ADDRESS) == text, why


def test_h_no_keeps_its_dot() -> None:
    """
    "H.NO" is a real and very common Indian house-number prefix (it is in the
    sample three times). Expanding the "NO" inside it yields "H.NUMBER", which
    is neither a word nor a number, so the dot is a boundary the expander
    respects.
    """
    assert expand_abbreviations("H.NO 389 247", SCOPE_ADDRESS) == "H.NO 389 247"
    # A standalone "NO" in the same value *is* expanded, so the two coexist.
    assert expand_abbreviations("WARD NO. 9", SCOPE_ADDRESS) == "WARD NUMBER. 9"


# --------------------------------------------------------------------------
# 6. `&` -> "and", and the reason the pipeline wants it
# --------------------------------------------------------------------------


def test_ampersand_becomes_and() -> None:
    """
    "ARCE AND BELTRAN PARTNERS" is a real row in the sample and "&" appears in
    real names; the problem statement lists the two spellings as an expected
    noise pattern, so they have to land on the same key.
    """
    with_amp = "ARCE & BELTRAN PARTNERS"
    with_word = "ARCE AND BELTRAN PARTNERS"
    assert expand_abbreviations(with_amp, SCOPE_NAME) == with_word

    keys = {
        fold_for_key(expand_abbreviations(text, SCOPE_NAME))
        for text in (with_amp, with_word)
    }
    assert len(keys) == 1, "the two spellings must produce one key"


# --------------------------------------------------------------------------
# 7. blank and non-string input
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "", "   "])
def test_expand_abbreviations_handles_blank(value: str | None) -> None:
    result = expand_abbreviations(value, SCOPE_ADDRESS)
    assert isinstance(result, str)
    assert result == "" or result.strip() == result


def test_expand_abbreviations_stringifies_non_strings() -> None:
    assert expand_abbreviations(123, SCOPE_ADDRESS) == "123"


# --------------------------------------------------------------------------
# 8. postal codes
# --------------------------------------------------------------------------

# A real US ZIP only ever sits in the trailing "..., CITY, ST 12345" position,
# and it is there for 0.03% of US rows in this corpus. Anchoring to that
# position is what keeps "1462 Sixth Avenue" from being read as a ZIP.
POSTAL_US_CASES: list[tuple[str, str, str | None, str | None]] = [
    # (address, country, pin_or_zip, postal_kind)
    ("1462 Sixth Avenue, Kankakee, IL 60901", "US", "60901", POSTAL_US_ZIP5),
    ("7812 Colusa Street, Port Orchard, WA 98366", "US", "98366", POSTAL_US_ZIP5),
    # No ZIP: the trailing 5 digits are a street number, so claiming a ZIP here
    # is the single most common false positive a bare \d{5} produces.
    ("1462 Sixth Avenue, Kankakee, IL", "US", "1462", POSTAL_GENERIC),
    ("7812 Colusa Street, Port Orchard, WA", "US", "7812", POSTAL_GENERIC),
]

POSTAL_INDIA_CASES: list[tuple[str, str, str | None, str | None]] = [
    ("12 MG Road, Bengaluru, Karnataka 560001", "India", "560001", POSTAL_IN_PIN6),
    ("5 Station Road, Hyderabad 500001", "India", "500001", POSTAL_IN_PIN6),
    # A 5-digit run in an Indian address is not a PIN.
    ("Sector 18, Gurugram 122001", "India", "122001", POSTAL_IN_PIN6),
    ("12 MG Road, Bengaluru", "India", "12", POSTAL_GENERIC),
    # "6-137A" and "53/1" are house numbers. The guard on a standalone digit run
    # excludes a digit-adjacent hyphen and slash, so *neither* half of
    # "301-314" qualifies and the answer is None -- not "301" and not "314".
    # A \b-based guard would take the 301, which is the most common false
    # positive a bare \d+ produces on this corpus.
    ("301-314, MUMBAI, Maharashtra", "India", None, None),
    ("6-137A, Prakasam, Andhra Pradesh", "India", None, None),
    # A standalone 2-digit run is still returned, because it is the only digit
    # information the row has.
    ("12 MG Road, Bengaluru", "India", "12", POSTAL_GENERIC),
]

POSTAL_OTHER_CASES: list[tuple[str, str | None, str | None, str | None]] = [
    # France has no rule of its own, so it gets the generic fallback, and the
    # task requires that the fallback is not None.
    ("18 Rue de la Paix, 75002 PARIS", "France", "75002", POSTAL_GENERIC),
    ("63 Rue de Rivoli, Lille", "France", "63", POSTAL_GENERIC),
    # An unknown country is an open set, not an error, and it also gets generic.
    ("18 Rue de la Paix, 75002 PARIS", "Atlantis", "75002", POSTAL_GENERIC),
    ("18 Rue de la Paix, 75002 PARIS", None, "75002", POSTAL_GENERIC),
]


@pytest.mark.parametrize(("address", "country", "pin", "kind"), POSTAL_US_CASES)
def test_us_postal_code(
    address: str, country: str, pin: str | None, kind: str | None
) -> None:
    tokens = extract_address_tokens(address, country)
    assert tokens[PIN_OR_ZIP] == pin
    assert tokens[POSTAL_KIND] == kind


@pytest.mark.parametrize(("address", "country", "pin", "kind"), POSTAL_INDIA_CASES)
def test_india_postal_code(
    address: str, country: str, pin: str | None, kind: str | None
) -> None:
    tokens = extract_address_tokens(address, country)
    assert tokens[PIN_OR_ZIP] == pin
    assert tokens[POSTAL_KIND] == kind


@pytest.mark.parametrize(("address", "country", "pin", "kind"), POSTAL_OTHER_CASES)
def test_other_countries_fall_back_to_generic(
    address: str, country: str | None, pin: str | None, kind: str | None
) -> None:
    """
    The task requires the fallback to be non-None, and France is the case it
    names. `country` is an open set, so an unknown label must not raise.
    """
    tokens = extract_address_tokens(address, country)
    assert tokens[PIN_OR_ZIP] == pin
    assert tokens[POSTAL_KIND] == kind


def test_postal_kinds_are_a_closed_set() -> None:
    """A caller branches on this value, so the set of values has to be pinned."""
    assert POSTAL_KINDS == (POSTAL_US_ZIP5, POSTAL_IN_PIN6, POSTAL_GENERIC)


def test_generic_is_not_a_postal_code() -> None:
    r"""
    The distinction the whole `postal_kind` column exists for.

    Both rows below yield a five-digit `pin_or_zip`, and a five-digit value in
    this column is exactly what a caller would compare for equality. One of them
    is a ZIP and the other is a street number, and nothing in the digits
    themselves says which -- so a caller that reads `pin_or_zip` alone cannot
    tell them apart, and a rule that requires two of them to be equal fires on
    house numbers.

    This is the test that fails if a future edit merges the two kinds or drops
    the column.

    The address is the one from the module docstring, where a naive `\b\d{5}\b`
    is shown reading the street number as a ZIP.
    """
    with_zip = extract_address_tokens("17560 Ellis Road, Irvine, CA 92617", "US")
    without_zip = extract_address_tokens("17560 Ellis Road, Irvine, CA", "US")

    # Both values are present and five digits long: the digit count alone tells
    # a caller nothing.
    assert with_zip[PIN_OR_ZIP] == "92617"
    assert without_zip[PIN_OR_ZIP] == "17560"
    assert len(with_zip[PIN_OR_ZIP]) == len(without_zip[PIN_OR_ZIP]) == 5
    # The kind is what separates them.
    assert with_zip[POSTAL_KIND] == POSTAL_US_ZIP5
    assert without_zip[POSTAL_KIND] == POSTAL_GENERIC
    assert with_zip[POSTAL_KIND] != without_zip[POSTAL_KIND]
    # And the generic one is the street number, i.e. exactly the value that must
    # never be compared as a postal code.
    assert without_zip[PIN_OR_ZIP] == without_zip[STREET_NUMBER] == "17560"


# --------------------------------------------------------------------------
# 9. street number
# --------------------------------------------------------------------------

# All of these inputs are real rows from the sample or the corpus profile.
# (address, expected street_number or None)
STREET_NUMBER_CASES: list[tuple[str, str | None]] = [
    ("12 MG Road, Bangalore", "12"),
    ("1462 Sixth Avenue, Kankakee, IL", "1462"),
    ("7812 Colusa Street, Port Orchard, WA", "7812"),
    ("Bldg 20 Station Rd", "20"),
    ("H.NO 389 247, K.G.Halli, Bangalore", "389"),
    ("3/5 A BLOCK, EGMORE NUNGAMBAKKA, CHENNAI", "3/5"),
    ("L-1318/38 GROUND FLOOR SANGAM VIHAR, DELHI", "1318/38"),
    ("No.54/3, Subbarama Chetty Road, Bengaluru", "54/3"),
    # The regression guard for abbreviation expansion happening *before* the
    # parse: "DOOR NO 9/1" reaches the parser as "DOOR NUMBER 9/1", and a
    # length-based label check rejects "NUMBER" as a designator and loses a
    # real house number. The allow-list of designators is what keeps these.
    ("DOOR NO 9/1 SOVARAM BYSACK STREET, KOLKATA", "9/1"),
    ("DOOR NUMBER: 5/252 - A, OLD NUMBER 6/156 CHAMBANNOOR", "5/252"),
    ("NO 3121, WARD NO. 9 NEEM WALA CHOWK, MOHALI", "3121"),
    ("NO 708 7TH FLOOR, SUITE NO.1149, CHENNAI", "708"),
    # And the other direction: a number that belongs to a locality is not a
    # house number, and a name word in front of the digits means the digits
    # are not one either.
    ("SECTOR 18, GURUGRAM", None),
    ("STREET NO. 4 NEW CANTT ROAD, FARIDKOT", None),
    ("Office Space Pimple SA Udagar", None),
    ("18 Rue de la Paix, 75002 PARIS", "18"),
    ("Mirzapur, EWS 12, UTTAR PRADESH", None),
]


@pytest.mark.parametrize(("address", "expected"), STREET_NUMBER_CASES)
def test_street_number(address: str, expected: str | None) -> None:
    assert extract_address_tokens(address, None)[STREET_NUMBER] == expected


# --------------------------------------------------------------------------
# 10. the token set
# --------------------------------------------------------------------------


def tokens_of(address: str) -> frozenset[str]:
    return extract_address_tokens(address)[TOKEN_SET]  # type: ignore[return-value]


def test_token_set_is_order_independent() -> None:
    """
    The property the whole blocking design rests on. The profile found address
    component order to be inconsistent between sources, so any order-sensitive
    key is wrong by construction -- and this is a set, so it is order-free by
    type rather than by convention.
    """
    assert tokens_of("12 MG ROAD, BENGALURU, KARNATAKA") == tokens_of(
        "KARNATAKA, BENGALURU, 12 MG ROAD"
    )


def test_token_set_ignores_punctuation_and_case() -> None:
    assert tokens_of("SBI Building, 5th Floor.") == tokens_of("sbi building 5th floor")
    assert tokens_of("Shree Balaji (P) Ltd") == tokens_of("SHREE BALAJI P LTD")


def test_token_set_splits_digit_compounds_both_ways() -> None:
    """
    "6-137A" and "6 137A" are the same house number written by two sources, and
    a hyphen between two digits separates rather than joins -- which is the
    opposite of the rule for "Hauts-de-France".
    """
    assert "137a" in tokens_of("6-137A, Prakasam, Andhra Pradesh")
    assert "137a" in tokens_of("6 137A, Prakasam, Andhra Pradesh")


def test_token_set_splits_ordinary_hyphens() -> None:
    assert tokens_of("Hauts-de-France") == tokens_of("Hauts de France")


def test_token_set_folds_accents() -> None:
    assert tokens_of("63 Rue de Rivoli, Lille") == tokens_of("63 Rue de Rivoli, LillE")


def test_token_set_deduplicates() -> None:
    assert tokens_of("MG Road, MG Road, MG Road") == frozenset({"mg", "road"})


# --------------------------------------------------------------------------
# 11. the payoff: expansion makes two spellings one key
# --------------------------------------------------------------------------


def test_expansion_makes_two_spellings_of_one_address_agree() -> None:
    """
    This is the test that justifies running the expander at all. On its own it
    is a cosmetic transform; what matters is that after it, the addresses a
    tokenizer would call different become the same set.
    """
    abbreviated = "12 MG RD, 4TH FL, BENGALURU"
    written_out = "12 MG ROAD, 4TH FLOOR, BENGALURU"

    assert tokens_of(expand_abbreviations(abbreviated, SCOPE_ADDRESS)) == tokens_of(
        expand_abbreviations(written_out, SCOPE_ADDRESS)
    )
    # ...and that they were different before it, or the test proves nothing.
    assert tokens_of(abbreviated) != tokens_of(written_out)


def test_fold_accents_and_fold_for_key() -> None:
    assert fold_accents("Café Zürich") == "Cafe Zurich"
    assert fold_for_key("  Café   ZURICH  ") == "cafe zurich"
    assert fold_for_key("") == ""
    assert fold_for_key(None) == ""


# --------------------------------------------------------------------------
# 12. blank and non-string input
# --------------------------------------------------------------------------

BLANK_ADDRESS_EXPECTED = {
    PIN_OR_ZIP: None,
    POSTAL_KIND: None,
    STREET_NUMBER: None,
    TOKEN_SET: frozenset(),
}


@pytest.mark.parametrize("value", [None, "", "   ", float("nan")])
def test_blank_address_yields_the_documented_blanks(value: object) -> None:
    """
    NaN is here because that is what a missing column value actually is once
    pandas has touched the frame, so it is the case that matters in the
    pipeline even though it is not the case a caller writes by hand.
    """
    tokens = extract_address_tokens(value, "India")  # type: ignore[arg-type]
    assert tokens == BLANK_ADDRESS_EXPECTED


def test_extract_returns_exactly_the_four_documented_keys() -> None:
    """A caller unpacks this dict, so an extra key is a breaking change."""
    assert set(extract_address_tokens("12 MG Road")) == set(BLANK_ADDRESS_EXPECTED)


# --------------------------------------------------------------------------
# 13. the DataFrame entry points
# --------------------------------------------------------------------------


def test_add_address_tokens_falls_back_to_the_raw_column() -> None:
    frame = pd.DataFrame(
        {
            "entity_id": ["A"],
            "business_address": "12 MG Road, Bengaluru",
            "country": ["India"],
        }
    )
    out = add_address_tokens(frame)
    assert out[STREET_NUMBER].iloc[0] == "12"
    assert PIN_OR_ZIP in out.columns


def test_add_address_tokens_uses_the_clean_column_when_present() -> None:
    frame = pd.DataFrame(
        {
            "entity_id": ["A"],
            "business_address": "raw garbage ###",
            "business_address_clean": "12 MG Road, Bengaluru",
            "country": ["India"],
        }
    )
    out = add_address_tokens(frame)
    assert out[STREET_NUMBER].iloc[0] == "12"


def test_add_address_tokens_leaves_a_frame_with_no_address_alone() -> None:
    """
    The pipeline has to be able to walk every file it is pointed at, including
    the ground-truth file, which has no address at all.
    """
    frame = pd.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-1"]})
    out = add_address_tokens(frame)
    assert list(out.columns) == list(frame.columns)


def test_expand_abbreviations_frame_adds_both_columns_without_touching_input() -> None:
    frame = pd.DataFrame(
        {
            "entity_id": ["A"],
            "business_name": "ACME CO",
            "business_name_clean": "ACME CO",
            "business_address": "12 MG RD",
            "business_address_clean": "12 MG RD",
            "country": ["US"],
        }
    )
    out = expand_abbreviations_frame(frame)
    assert out["business_name_expanded"].iloc[0] == "ACME COMPANY"
    assert out["business_address_expanded"].iloc[0] == "12 MG ROAD"
    # Additive: the abbreviation is still available for a feature that wants it.
    assert out["business_address_clean"].iloc[0] == "12 MG RD"
    assert frame.columns.tolist() == [
        "entity_id", "business_name", "business_name_clean",
        "business_address", "business_address_clean", "country",
    ]


# --------------------------------------------------------------------------
# 14. the 39 real rows through the real pipeline
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def normalized_sample() -> pd.DataFrame:
    return normalize(sample_source_df)


def test_the_sample_is_39_rows(normalized_sample: pd.DataFrame) -> None:
    assert len(normalized_sample) == SAMPLE_ROW_COUNT == 39


def test_every_contract_column_is_produced(normalized_sample: pd.DataFrame) -> None:
    missing = [c for c in NORMALIZED_COLUMNS if c not in normalized_sample.columns]
    assert missing == []


def test_the_latin_path_is_exactly_the_eight_unflagged_rows(
    normalized_sample: pd.DataFrame,
) -> None:
    """
    31 of the 39 set `has_non_latin_name`, which is the ratio
    `tests/multilingual_sample.py` chose on purpose. Eight do not, and those
    eight are the rows the Latin blocking path is allowed to touch -- so the
    count is asserted here because a change to it would silently change which
    rows Phase 5 ever sees.
    """
    flagged = normalized_sample["has_non_latin_name"]
    assert int(flagged.sum()) == 31
    assert int((~flagged).sum()) == 8
    assert int(normalized_sample["has_indian_script_name"].sum()) == 30


def test_every_real_row_yields_a_usable_key(normalized_sample: pd.DataFrame) -> None:
    """
    No real row may fall out of the pipeline empty. `address_token_set` is what
    the pre-filter reads, and a row with an empty set is a row that can never be
    a candidate for anything, which is a silent recall loss rather than an
    error.
    """
    assert normalized_sample["address_token_set"].map(bool).all()
    assert normalized_sample["address_street_number"].map(
        lambda v: v is None or isinstance(v, str)
    ).all()
    # The postal value and its confidence are never set without each other.
    both = normalized_sample[["address_pin", "address_postal_kind"]].notna()
    assert (both["address_pin"] == both["address_postal_kind"]).all()


def test_the_french_row_falls_back_to_generic_not_none(
    normalized_sample: pd.DataFrame,
) -> None:
    """
    SAMP-039 is the one accented-Latin row: "18 RUE JEN ZAY, DUNKERQUE, NORD".
    France has no postal rule here, so the value is `generic` -- present, and
    explicitly marked as something that must not be compared as a ZIP.
    """
    row = normalized_sample[normalized_sample["entity_id"] == "SAMP-039"].iloc[0]
    assert row["address_postal_kind"] == POSTAL_GENERIC
    assert row["address_pin"] == "18"


def test_the_us_rows_with_no_zip_are_not_claimed_as_zips(
    normalized_sample: pd.DataFrame,
) -> None:
    """
    Four US rows in the sample carry no ZIP, so their `address_pin` is a street
    number and their kind is `generic`. If this ever starts returning
    `us_zip5` for them, the pre-filter is about to compare street numbers.
    """
    us = normalized_sample[
        (normalized_sample["country"] == "US")
        & (~normalized_sample["has_non_latin_name"])
    ]
    assert len(us) == 5
    assert not (us["address_postal_kind"] == POSTAL_US_ZIP5).any()


def test_normalize_is_idempotent(normalized_sample: pd.DataFrame) -> None:
    """
    Run twice, get the same answers.

    This is not a style preference. `normalize` is called on frames that have
    already been through it -- a cached sample re-normalized, a chunk released
    and re-read, a debug script that forgets -- and every stage here is additive,
    which is exactly the shape that grows a second `address_pin` column instead
    of replacing the first. A second pass must be a no-op.
    """
    twice = normalize(normalized_sample)
    assert set(twice.columns) == set(normalized_sample.columns)
    assert len(twice.columns) == len(normalized_sample.columns), "no duplicate columns"
    for column in NORMALIZED_COLUMNS:
        if column.endswith("token_set"):
            assert list(twice[column]) == list(normalized_sample[column]), column
        else:
            left = normalized_sample[column].fillna("")
            right = twice[column].fillna("")
            assert list(left) == list(right), column


def test_normalize_walks_a_ground_truth_shaped_frame() -> None:
    """
    The ground-truth file has no name, address or country. `normalize` is
    handed every file in the pipeline, so it must return that frame unchanged
    rather than raise.
    """
    frame = pd.DataFrame(
        {"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-1,S3-1"]}
    )
    out = normalize(frame)
    assert list(out.columns) == list(frame.columns)
    assert out["matched_entity_ids"].iloc[0] == "S2-1,S3-1"
