"""
Tests for clean_text / clean_dataframe.

Everything here is an exact string comparison against a hand-written reference
frame, because cleaning is only useful if the output is *byte*-exact: a cleaner
that is "nearly right" still splits business names differently across sources.
Nothing in this file reads `dataset/`; the mojibake inputs are the real
signatures copied out of the profile scan, written as explicit \\u escapes so the
invisible C1 control characters (U+0080-U+009F) survive a copy-paste.

Three groups of cases:

  PRESERVED_CASES  values that must come back byte-identical. These are the
                   regression guard for the whole point of the pass: real
                   accented Latin, Portuguese A-tilde, French, and the Indic
                   scripts must not be touched. If one of these breaks, the
                   cleaner is damaging data.
  CLEANED_CASES    (input, expected) pairs, one behaviour each, each one taken
                   from or modelled on a real value in the profile report.
  IDEMPOTENCE      cleaning twice must equal cleaning once, otherwise the
                   output cannot be used as a blocking key.

The last test is the contract with the profiler: it cleans the synthetic source
frame and asserts the noise counts the profiler tracks are all zero, and that
the non-ASCII rows are still there. Zero noise achieved by emptying the accented
and Indic rows would be a failure, not a pass.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.clean_text import (
    DEFAULT_COLUMNS,
    clean_dataframe,
    clean_text,
    repair_mojibake,
)
from src.preprocessing.profile_data import (
    NOISE_TOKEN_PATTERNS,
    profile_dataframe,
)

# --------------------------------------------------------------------------
# reference frame: values that must survive untouched
# --------------------------------------------------------------------------

PRESERVED_CASES: list[tuple[str, str]] = [
    # Clean ASCII baseline.
    ("Shree Balaji Traders", "clean ASCII name"),
    ("12 MG Road, Bengaluru", "clean ASCII address"),
    # French. The test set has France rows and nothing at train time, so the
    # accented characters are exactly what must not be normalized away.
    ("18 RUE JEN ZAY", "French address, no accents"),
    ("ALLÉE DES HÊTRES", "French with precomposed accents"),
    ("20 Rue de l'Église, Pornic, Pays de la Loire", "French elision"),
    ("Café Zurich", "accented name"),
    # Portuguese / Spanish A-tilde: a real letter, not a mojibake lead.
    ("SÃO PAULO LTDA", "real A-tilde before an uppercase word"),
    ("Café Müller SARL", "real umlaut"),
    ("L'Âge Corp", "real A-circumflex before a lowercase word"),
    ("Café câble Shop", "real circumflex inside a word"),
    # Indic scripts: outside every range the mojibake repair looks at.
    (
        "क\u0943\u0937\u094d\u0923\u093e \u0907\u0902\u092a\u0947\u0915\u094d\u0938\u094d "
        "\u0932\u093f\u092e\u093f\u091f\u0947\u0921\u094d",
        "Devanagari",
    ),
    (
        "શ\u094d\u0930\u0940 \u092c\u093e\u0932\u093e\u091c\u0940 "
        "\u091f\u094d\u0930\u0947\u0921\u0930\u094d\u0938",
        "Gujarati",
    ),
    (
        "క\u0c43\u0cb7\u094d\u0ca3\u0cbe "
        "\u0c07\u0c02\u0c2a\u0c47\u0c15\u0c4d\u0cb8\u0c4d "
        "\u0c32\u0c3f\u0c2e\u0c3f\u0c1f\u0c47\u0c21\u0c4d",
        "Telugu",
    ),
    # A placeholder token that is a substring of a real word is not a
    # placeholder.
    ("NANDAN TRADERS", "NULL is a substring, not a token"),
    ("NANAKI Paints", "NAN is a substring, not a token"),
    ("NULLABLE Systems", "NULL is a prefix of a real word"),
    # Parentheses carry landmark meaning in an address and are left alone.
    ("12 (Near SBI ATM), MG Road", "landmark parentheses"),
    ("Clm Agro (Pvt) Ltd", "parenthesised legal suffix"),
    # Digits and runs of word characters are never junk.
    ("111 Shahdara Lane", "repeated digits are not a junk run"),
]


# --------------------------------------------------------------------------
# reference frame: one behaviour per case
# --------------------------------------------------------------------------

CLEANED_CASES: list[tuple[str, str, str]] = [
    # --- 1. mojibake -----------------------------------------------------
    # "Â" + a regular space: the non-breaking space was lost and the lead byte
    # is all that survived. No "Â" left behind.
    ("Gali No. Â 01, Vikas Marg", "Gali No. 01, Vikas Marg", "stray A-circumflex"),
    # The real dataset signature: "Â" + C1 0x80 + C1 0x93, i.e. the UTF-8 bytes
    # of an en dash with the lead byte mangled to C2. Rendered: "Gali No. - 01".
    (
        "A-212, Gali No. \u00c2\u0080\u0093 01, Delhi",
        "A-212, Gali No. - 01, Delhi",
        "mangled en dash",
    ),
    # "Vsp" + UTF-8 right single quote (E2 80 99) + "S" -> "Vsp'S". The capital
    # S is what the source row actually contains, so it is preserved.
    (
        "Hyderabad, Vsp\u00e2\u0080\u0099S Bhavana",
        "Hyderabad, Vsp'S Bhavana",
        "mangled apostrophe, real value",
    ),
    # Lowercase after the quote, which is what the profile's "Lion's School"
    # example is: the apostrophe survives as ASCII, so the token matches the
    # clean row exactly.
    (
        "Lion\u00e2\u0080\u0099s School",
        "Lion's School",
        "mangled apostrophe -> ASCII",
    ),
    # Both quote directions in one value: C2 80 98 then E2 80 99.
    (
        "\u00c2\u0080\u0098Shubh-Aishwary\u00e2\u0080\u0099, Pune",
        "'Shubh-Aishwary', Pune",
        "mangled quote pair",
    ),
    # Three mangled dashes in one address.
    (
        "Sub Pl\u00e2\u0080\u00933 Tps\u00e2\u0080\u00933 Fp\u00e2\u0080\u0093300 Nandi Wadi",
        "Sub Pl-3 Tps-3 Fp-300 Nandi Wadi",
        "mangled en dash x3",
    ),
    # C3 89 is UTF-8 for a capital E-acute: the classic French mojibake.
    (
        "20 Rue de l\u00c3\u0089glise",
        "20 Rue de l\u00c9glise",
        "mangled capital accented",
    ),
    # A byte lost upstream: U+FFFD becomes a space, so the words do not fuse
    # into a token that could never match its twin.
    (
        "Prestige Lakeside\ufffdHabitat",
        "Prestige Lakeside Habitat",
        "U+FFFD becomes a space",
    ),
    # Real smart punctuation folds to ASCII so "Vsp's" and "Vsp’s" agree.
    ("Ma’s Cafe – NYC", "Ma's Cafe - NYC", "real smart punctuation"),
    ("12–14 Ring Road", "12-14 Ring Road", "real en dash"),
    # The lead character alone, continuation bytes destroyed.
    ("LionâS School", "Lion'S School", "stray a-circumflex before a capital"),
    # --- 2. placeholder tokens -------------------------------------------
    (
        "MA, 25 NIAGARA STREET, NULL, SPRINGFIELD",
        "MA, 25 NIAGARA STREET, SPRINGFIELD",
        "NULL mid-field leaves no dangling comma",
    ),
    ("NULL", "", "value is only a placeholder"),
    ("null", "", "placeholder is case-insensitive"),
    ("N/A", "", "N/A"),
    ("n/a", "", "N/A is case-insensitive"),
    ("NONE", "", "NONE"),
    ("None", "", "NONE is case-insensitive"),
    ("nan", "", "nan"),
    ("NaN", "", "nan is case-insensitive"),
    ("(NULL)", "", "placeholder inside brackets leaves none"),
    ("NULL, NULL", "", "a value made only of placeholders becomes empty"),
    # --- 3. junk runs -----------------------------------------------------
    ("OFFICE NO. ##70", "OFFICE NO. 70", "## does not fuse into the number"),
    ("SHOP ### 12", "SHOP 12", "###"),
    ("*** Yamuica Venus Pvt  Ltd", "Yamuica Venus Pvt Ltd", "*** and double space"),
    ("Shop ??? Ltd", "Shop Ltd", "??? is a junk run"),
    ("A--B Traders", "A B Traders", "-- is a junk run"),
    ("###", "", "value is only a junk run"),
    # --- 4. brackets -----------------------------------------------------
    ("Clm Agro [Limited]", "Clm Agro Limited", "bracket-wrapped legal suffix"),
    ("Acme [Pvt] Ltd", "Acme Pvt Ltd", "brackets dropped, word kept"),
    ("Acme [Pvt", "Acme Pvt", "unbalanced bracket"),
    ("Café Zürich [Limited]", "Café Zürich Limited", "brackets with accents"),
    # --- 5. whitespace ----------------------------------------------------
    ("  Shree   Balaji  ", "Shree Balaji", "collapse and trim"),
    ("Shop\tNo\u00a09", "Shop No 9", "tab and non-breaking space"),
    ("-- Holloway Peak Inc", "Holloway Peak Inc", "leading junk run"),
    ("Kotewala Rd ,", "Kotewala Rd", "trailing separator"),
]


@pytest.mark.parametrize("value,label", PRESERVED_CASES, ids=[c[1] for c in PRESERVED_CASES])
def test_preserved_values_are_unchanged(value: str, label: str) -> None:
    """Real text must survive cleaning byte-identically (guards the whole pass)."""
    assert clean_text(value) == value, label


@pytest.mark.parametrize(
    "value,expected,label", CLEANED_CASES, ids=[c[2] for c in CLEANED_CASES]
)
def test_cleaned_values_match_exactly(value: str, expected: str, label: str) -> None:
    """One behaviour per case, asserted on the exact output string."""
    assert clean_text(value) == expected, label


@pytest.mark.parametrize("value,label", PRESERVED_CASES + [(c[0], c[2]) for c in CLEANED_CASES])
def test_cleaning_is_idempotent(value: str, label: str) -> None:
    """clean(clean(x)) == clean(x), so the output is safe to use as a key."""
    once = clean_text(value)
    assert clean_text(once) == once, label


def test_mojibake_repair_does_not_touch_clean_ascii() -> None:
    """The fast path: pure ASCII never needs the regex machinery."""
    assert repair_mojibake("Shop No 9") == "Shop No 9"


def test_no_undecodable_character_survives_any_case() -> None:
    """
    Sweep the whole reference frame: no C1 control (U+0080-U+009F) and no
    U+FFFD may remain in any output, whichever case produced it. Neither can
    legitimately appear in business text.
    """
    for value, _ in PRESERVED_CASES + [(c[0], c[2]) for c in CLEANED_CASES]:
        cleaned = clean_text(value)
        assert "\ufffd" not in cleaned, f"U+FFFD survived in {cleaned!a}"
        assert not any("\u0080" <= char <= "\u009f" for char in cleaned), cleaned


def test_no_mojibake_lead_survives_in_a_cleaned_case() -> None:
    """
    A lead character is only ever allowed to stay when it is a real letter
    ("SÃO", "câble" above), so the check belongs on the cases whose whole point
    is that the lead was junk.
    """
    for value, expected, label in CLEANED_CASES:
        cleaned = clean_text(value)
        for char in ("\u00c2", "\u00c3", "\u00e2"):
            assert char not in cleaned, f"{char!r} survived in {cleaned!a} ({label})"


@pytest.mark.parametrize("value", [None, np.nan, float("nan"), 42, 3.5])
def test_non_string_input_becomes_text(value: object) -> None:
    """These columns come out of pandas, so NaN and numbers must not raise."""
    assert isinstance(clean_text(value), str)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# clean_dataframe
# --------------------------------------------------------------------------


def test_clean_dataframe_adds_clean_columns_and_keeps_raw(synthetic_source_df: pd.DataFrame) -> None:
    out = clean_dataframe(synthetic_source_df)

    for column in DEFAULT_COLUMNS:
        assert f"{column}_clean" in out.columns
        # Raw columns are untouched: some similarity features want raw text.
        pd.testing.assert_series_equal(out[column], synthetic_source_df[column])
    assert len(out) == len(synthetic_source_df)
    assert list(out["entity_id"]) == list(synthetic_source_df["entity_id"])


def test_clean_dataframe_does_not_mutate_its_input(synthetic_source_df: pd.DataFrame) -> None:
    before = synthetic_source_df.copy()
    clean_dataframe(synthetic_source_df)
    pd.testing.assert_frame_equal(synthetic_source_df, before)


def test_clean_dataframe_cleans_the_values(synthetic_source_df: pd.DataFrame) -> None:
    out = clean_dataframe(synthetic_source_df)
    names = dict(zip(out["entity_id"], out["business_name_clean"]))

    assert names["S1-00001"] == "Shree Balaji Traders"       # clean row untouched
    assert names["S1-00002"] == "श्री बालाजी ट्रेडर्स"       # Devanagari untouched
    assert names["S1-00003"] == "શ્રી બાલાજી ટ્રેડર્સ"      # Gujarati untouched
    assert names["S1-00004"] == ""                          # NULL
    assert names["S1-00005"] == ""                          # ###
    assert names["S1-00006"] == ""                          # ##
    assert names["S1-00007"] == "Café Zürich Limited"        # brackets only
    assert names["S1-00010"] == ""                          # N/A
    assert names["S1-00011"] == ""                          # NONE
    assert names["S1-00012"] == ""                          # nan
    assert names["S1-00013"] == ""                          # ***
    assert names["S1-00014"] == "Café बालाजी Private Ltd"    # mixed scripts


def test_clean_dataframe_skips_columns_that_are_absent(synthetic_source_df: pd.DataFrame) -> None:
    partial = synthetic_source_df[["entity_id", "business_name"]]
    out = clean_dataframe(partial)

    assert "business_name_clean" in out.columns
    assert "business_address_clean" not in out.columns


def test_clean_dataframe_honours_an_explicit_column_list(synthetic_source_df: pd.DataFrame) -> None:
    out = clean_dataframe(synthetic_source_df, columns=["business_address"])
    assert "business_address_clean" in out.columns
    assert "business_name_clean" not in out.columns


def test_clean_dataframe_handles_missing_values(synthetic_source_df: pd.DataFrame) -> None:
    """NaN cells must clean to "" rather than blow up mid-column."""
    frame = synthetic_source_df.copy()
    frame.loc[0, "business_name"] = np.nan
    frame.loc[1, "business_address"] = None

    out = clean_dataframe(frame)
    assert out.loc[0, "business_name_clean"] == ""
    assert out.loc[1, "business_address_clean"] == ""
    assert out["business_name_clean"].map(type).eq(str).all()


# --------------------------------------------------------------------------
# contract with the profiler
# --------------------------------------------------------------------------


CLEAN_COLUMNS = [f"{column}_clean" for column in DEFAULT_COLUMNS]


def test_profiler_sees_no_noise_after_cleaning(synthetic_source_df: pd.DataFrame) -> None:
    """
    The end-to-end contract: after cleaning, none of the seven noise tokens the
    profiler tracks may still be counted. The clean columns are profiled by
    name, since profile_dataframe only looks at business_name / business_address
    by default.
    """
    before = profile_dataframe(synthetic_source_df)
    cleaned = clean_dataframe(synthetic_source_df)
    after = profile_dataframe(cleaned, text_columns=CLEAN_COLUMNS)

    # The frame we started from really did contain the noise, otherwise this
    # test could pass on a fixture that was never dirty.
    for column in DEFAULT_COLUMNS:
        assert before.col(column).noise_token_occurrences, column

    for column, clean_column in zip(DEFAULT_COLUMNS, CLEAN_COLUMNS):
        post = after.col(clean_column)
        assert post.noise_token_occurrences == {}, (
            f"{clean_column} still contains {post.noise_token_occurrences}"
        )
        assert post.noise_token_rows == 0
        # Bracket decoration is gone too, since clean_text unwraps it.
        assert post.bracket_token_occurrences == 0
        # And the text is still there: cleaning is not deletion.
        assert post.rows == before.col(column).rows == len(cleaned)


def test_cleaning_does_not_empty_the_non_ascii_rows(synthetic_source_df: pd.DataFrame) -> None:
    """
    Zero noise must not be achieved by wiping the accented and Indic rows: the
    non-ASCII row count and the per-script breakdown have to be identical before
    and after.
    """
    before = profile_dataframe(synthetic_source_df)
    after = profile_dataframe(clean_dataframe(synthetic_source_df), text_columns=CLEAN_COLUMNS)

    for column, clean_column in zip(DEFAULT_COLUMNS, CLEAN_COLUMNS):
        pre = before.col(column)
        post = after.col(clean_column)
        assert post.non_ascii_rows == pre.non_ascii_rows, column
        assert post.script_rows == pre.script_rows, column


def test_every_profiler_noise_token_is_handled() -> None:
    """
    The profiler and the cleaner must be talking about the same token set. If
    someone adds a token to one and not the other, this is what catches it: each
    token is planted on its own in a value that is otherwise clean, and the
    result must no longer match the profiler's pattern.
    """
    for token, pattern in NOISE_TOKEN_PATTERNS.items():
        value = f"ACME TRADING {token} 12 MG ROAD"
        cleaned = clean_text(value)
        assert re.search(pattern, cleaned) is None, f"{token} survived as {cleaned!a}"
