"""
Tests for `src.blocking.lexical_blocking`.

Blocking is the stage that sets the recall ceiling: a pair this module drops
cannot be recovered by anything downstream. So the tests are organised around
what it *keeps* and what it *throws away*, and the two are asserted with equal
force. A blocking function that returns everything passes a recall test and
fails the submission, so "this dissimilar pair is absent" is as load-bearing
here as "this similar pair is present".

The pairs are built from the eight Latin rows of `tests/multilingual_sample.py`,
which are real rows copied out of the real TSVs. Each is paired with a
deliberately perturbed copy of itself, and each is also paired with a different
one of the eight. The perturbations are the ones the profile says actually
occur between sources -- abbreviation vs long form, case, `&` vs "AND", spacing,
legal-suffix variation, house-number formatting -- rather than invented noise,
and the origin of each pair is recorded in its comment so a surprising result
can be traced back to a real row.

The pre-filter gets its own section, tested through `prefilter_allows` on its
own as well as through the vectorized `coarse_prefilter`, because the
measurement behind it is uncomfortable: a postal code is present in the correct
trailing position for 0.00-0.03% of rows in this corpus. The filter therefore
has to abstain, and an abstaining filter is easy to write by accident and easy
to break by tightening it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.blocking.lexical_blocking import (
    CONFIDENT_POSTAL_KINDS,
    CANDIDATE_COL,
    DEFAULT_CHUNK_SIZE,
    LSH_THRESHOLD,
    MINHASH_NGRAM,
    OUTPUT_COLUMNS,
    POSTAL_KIND,
    POSTAL_NEAR_PREFIX,
    QUERY_COL,
    BlockingReport,
    build_lexical_document,
    coarse_prefilter,
    estimated_jaccard,
    generate_lexical_candidates,
    minhash_shingles,
    prefilter_allows,
    _minhash_for,
)
from src.preprocessing.normalize import normalize

SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# --------------------------------------------------------------------------
# the eight real Latin rows, verbatim from tests/multilingual_sample.py
# --------------------------------------------------------------------------
REAL_LATIN_ROWS: list[tuple[str, str, str, str]] = [
    # (entity_id, business_name, business_address, country)
    ("S1-031", "Apex Summit", "67 KENTUCKY ST, SALYERSVILLE, KY", "US"),
    ("S1-032", "West Charter School Downtown LLC",
     "1462 SIXTH AVENUE, KANKAKEE, IL", "US"),
    ("S1-033", "BEACON CORP", "7812 COLUSA ST, PO BOX 899, PORT ORCHARD, WA", "US"),
    ("S1-034", "ARCE & BELTRAN PARTNERS", "5116 80TH STREET, STILLWATER, OK", "US"),
    ("S1-035", "Vision Partners Corp", "IA, Iowa City, 1064 Newton Rd, Unit 11", "US"),
    ("S1-036", "Red Perfect Trading",
     "Mirzapur, Ews 12, Uttar Pradesh, Mirzapursadar, Awas Vikas Colony", "India"),
    ("S1-037", "Nandlal Kisan LLP",
     "D-61 Ifs Apartmentmayur Vihar I, New Delhi, East Delhi, Delhi", "India"),
    ("S1-038", "Perfect Investments Private",
     "Plot No.99, Flat No.201, Sri Dhama Apts., Road No.4, Shaikpet, Hyderabad, TG", "India"),
]

# The same eight rows re-spelled the way a second source would write them.
# Every perturbation is one the corpus actually contains, and the comment says
# which one, because a pair that fails to match should be diagnosed in terms of
# the perturbation that caused it.
PERTURBED: list[tuple[str, str, str]] = [
    # from S1-031: abbreviation, case, and the `ST` contextual rule
    ("S2-031", "APEX SUMMIT", "67 Kentucky St, Salyersville, KY"),
    # from S1-032: legal suffix dropped and the address abbreviated
    ("S2-032", "West Charter School Downtown",
     "1462 Sixth Ave, Kankakee, IL"),
    # from S1-033: `CORP` expanded, the same as Phase 4's abbreviation table does
    ("S2-033", "Beacon Corporation",
     "7812 Colusa St, PO Box 899, Port Orchard, WA"),
    # from S1-034: `&` written as the word, which is the whole point of the rule
    ("S2-034", "ARCE AND BELTRAN PARTNERS", "5116 80th Street, Stillwater, OK"),
    # from S1-035: `RD` expanded, `IA` kept, unit number kept
    ("S2-035", "VISION PARTNERS CORPORATION",
     "IA, Iowa City, 1064 Newton Road, Unit 11"),
    # from S1-036: component order reversed in the address
    ("S2-036", "Red Perfect Trading",
     "Awas Vikas Colony, Mirzapursadar, Mirzapur, EWS 12, Uttar Pradesh"),
    # from S1-037: house number written without its hyphen
    ("S2-037", "NANDLAL KISAN LLP", "D61 Ifs Apartment Mayur Vihar I, New Delhi, Delhi"),
    # from S1-038: `PVT LTD` written out
    ("S2-038", "Perfect Investments Pvt Ltd",
     "Plot No 99, Flat No 201, Sri Dhama Apts, Road No 4, Shaikpet, Hyderabad, TG"),
]


def _real_frame() -> pd.DataFrame:
    return pd.DataFrame(REAL_LATIN_ROWS, columns=SOURCE_COLUMNS)


def _perturbed_frame() -> pd.DataFrame:
    # The country is copied from the source row by index, which is more honest
    # than retyping it: a mismatch here would silently turn a similarity test
    # into a pre-filter test.
    rows = [
        {"entity_id": eid, "business_name": name, "business_address": address,
         "country": REAL_LATIN_ROWS[index][3]}
        for index, (eid, name, address) in enumerate(PERTURBED)
    ]
    return pd.DataFrame(rows, columns=SOURCE_COLUMNS)


@pytest.fixture
def real_s1() -> pd.DataFrame:
    return _real_frame()


@pytest.fixture
def real_s2() -> pd.DataFrame:
    return _perturbed_frame()


@pytest.fixture
def real_s3() -> pd.DataFrame:
    return pd.DataFrame(columns=SOURCE_COLUMNS).astype(str)


@pytest.fixture
def real_candidates(real_s1: pd.DataFrame, real_s2: pd.DataFrame, real_s3: pd.DataFrame):
    return generate_lexical_candidates(real_s1, real_s2, real_s3, top_k=5)


def _pair_set(frame: pd.DataFrame) -> set[tuple[str, str]]:
    return set(map(tuple, frame[OUTPUT_COLUMNS].to_numpy()))


# --------------------------------------------------------------------------
# 1. the hand-picked pairs
# --------------------------------------------------------------------------

# Eight similar pairs, one per real row, each against its own perturbation. The
# task asks for at least five; there are eight so that deleting three rows to
# "simplify" the fixture trips the count assertion below instead of quietly
# weakening the test.
SIMILAR_PAIRS = [(real[0], perturbed[0]) for real, perturbed in zip(REAL_LATIN_ROWS, PERTURBED)]

# Five dissimilar pairs, each a real business against a *different* real
# business, chosen because the two share no name token and no country. These are
# asserted absent one by one.
#
# The choice is not arbitrary and the list is deliberately not "all cross pairs".
# Two of the eight possible cross pairs share the token "partners" -- BEACON CORP
# against ARCE AND BELTRAN PARTNERS, and ARCE & BELTRAN PARTNERS against VISION
# PARTNERS CORPORATION -- and those come back, correctly. Blocking is allowed to
# over-generate; that is what the matching stage downstream is for. Listing them
# as "must be absent" would be asserting something false about the function, and
# a test that encodes a falsehood is worse than no test, because it will be
# "fixed" by changing the code until the fiction holds.
KNOWN_ABSENT_PAIRS = [
    ("S1-031", "S2-032"),  # Apex Summit             vs West Charter School Downtown
    ("S1-032", "S2-033"),  # West Charter School LLC vs Beacon Corporation
    ("S1-035", "S2-036"),  # Vision Partners Corp    vs Red Perfect Trading
    ("S1-036", "S2-037"),  # Red Perfect Trading     vs Nandlal Kisan LLP
    ("S1-037", "S2-038"),  # Nandlal Kisan LLP       vs Perfect Investments Pvt Ltd
]

# The same eight real rows paired up so each meets a different one, for the
# recall-versus-discrimination comparison.
DISSIMILAR_PAIRS = [
    (REAL_LATIN_ROWS[0][0], PERTURBED[1][0]),
    (REAL_LATIN_ROWS[1][0], PERTURBED[2][0]),
    (REAL_LATIN_ROWS[2][0], PERTURBED[3][0]),
    (REAL_LATIN_ROWS[3][0], PERTURBED[4][0]),
    (REAL_LATIN_ROWS[4][0], PERTURBED[5][0]),
    (REAL_LATIN_ROWS[5][0], PERTURBED[6][0]),
    (REAL_LATIN_ROWS[6][0], PERTURBED[7][0]),
    (REAL_LATIN_ROWS[7][0], PERTURBED[0][0]),
]


def test_the_fixture_has_enough_pairs_to_be_meaningful() -> None:
    """
    The task asks for at least five of each. Asserted so that deleting rows from
    the lists above cannot quietly reduce the test to a token gesture.
    """
    assert len(SIMILAR_PAIRS) >= 5
    assert len(KNOWN_ABSENT_PAIRS) >= 5
    assert len(set(SIMILAR_PAIRS)) == len(SIMILAR_PAIRS)
    assert len(set(KNOWN_ABSENT_PAIRS)) == len(KNOWN_ABSENT_PAIRS)
    assert not set(SIMILAR_PAIRS) & set(KNOWN_ABSENT_PAIRS)


@pytest.mark.parametrize(("query_id", "candidate_id"), SIMILAR_PAIRS, ids=lambda v: v)
def test_each_known_similar_pair_is_present(
    query_id: str, candidate_id: str, real_candidates: pd.DataFrame
) -> None:
    """
    One assertion per pair, so a failure names the pair that broke rather than
    reporting a count. The perturbations these cover, one per row: `ST` in long
    and short form, a dropped legal suffix, `CORP` expanded, `&` written as
    "AND", `RD` expanded, reversed address component order, a house number
    without its hyphen, and `PVT LTD` written out.
    """
    assert (query_id, candidate_id) in _pair_set(real_candidates)


@pytest.mark.parametrize(("query_id", "candidate_id"), KNOWN_ABSENT_PAIRS, ids=lambda v: v)
def test_each_known_dissimilar_pair_is_absent(
    query_id: str, candidate_id: str, real_candidates: pd.DataFrame
) -> None:
    """
    One assertion per pair. A blocking function that returns everything passes a
    recall test and fails this one, which is the point of having both.
    """
    assert (query_id, candidate_id) not in _pair_set(real_candidates)


def test_pairs_sharing_a_business_token_are_allowed_through(real_candidates: pd.DataFrame) -> None:
    """
    The counterweight to the test above, and the reason the absent list is a
    hand-picked five rather than all cross pairs.

    "PARTNERS" appears in two unrelated real businesses. A word-level TF-IDF
    document genuinely shares that token, so the pair is a legitimate candidate
    and blocking is *supposed* to emit it -- deciding otherwise is the matcher's
    job, not the blocker's. This test exists so that if someone later tightens
    blocking to suppress it, the failure is understood as a recall decision
    rather than read as a bug fix.
    """
    found = _pair_set(real_candidates)
    assert ("S1-033", "S2-034") in found  # BEACON CORP / ARCE AND BELTRAN PARTNERS
    assert ("S1-034", "S2-035") in found  # ARCE & BELTRAN PARTNERS / VISION PARTNERS


def test_recall_and_discrimination_agree_on_the_same_run(real_candidates: pd.DataFrame) -> None:
    """
    One run, both directions, so the assertions above cannot pass on different
    code paths.
    """
    found = _pair_set(real_candidates)
    hits = sum(1 for pair in SIMILAR_PAIRS if pair in found)
    noise = sum(1 for pair in DISSIMILAR_PAIRS if pair in found)
    assert hits == len(SIMILAR_PAIRS)
    assert hits > noise, f"recall {hits} must exceed noise {noise}"


# --------------------------------------------------------------------------
# 2. the output contract
# --------------------------------------------------------------------------


def test_output_has_exactly_the_two_contract_columns(real_candidates: pd.DataFrame) -> None:
    """`run_pipeline.py` and the scorer both index these two by name."""
    assert list(real_candidates.columns) == [QUERY_COL, CANDIDATE_COL] == OUTPUT_COLUMNS


def test_candidates_are_only_s2_or_s3_ids(real_candidates: pd.DataFrame) -> None:
    """
    The format is one row per candidate, and the candidate side is source2 and
    source3 only. A source1 id leaking in means a direction was wired wrong.
    """
    assert real_candidates.empty or real_candidates[CANDIDATE_COL].str[:2].isin(["S2", "S3"]).all()
    assert real_candidates[QUERY_COL].str.startswith("S1").all()


def test_candidates_are_deduplicated(real_candidates: pd.DataFrame) -> None:
    """
    Both strategies can find the same pair, and `top_k` is applied before the
    union, so duplicates are expected rather than exceptional.
    """
    assert not real_candidates.duplicated().any()


def test_no_self_pairs() -> None:
    """
    The real files cannot produce one, because the id prefixes differ, so this
    is checked on a hand-built frame where they can.
    """
    frame = pd.DataFrame(
        [("X-1", "Prime Money Lenders", "12 MG Road, Bengaluru", "India")],
        columns=SOURCE_COLUMNS,
    )
    out = generate_lexical_candidates(frame, frame, frame, top_k=5)
    assert _pair_set(out) == set() or all(a != b for a, b in _pair_set(out))


def test_candidates_are_sorted_and_the_index_is_reset(real_candidates: pd.DataFrame) -> None:
    """
    Sorted by the pair, not by score, and with a fresh index.

    Blocking output carries no score, so two runs that find the same pairs must
    produce the same file -- `run_pipeline.py` compares candidate frames across
    stages, and a leftover index from a `concat` would break that.
    """
    assert real_candidates.index.tolist() == list(range(len(real_candidates)))
    expected = real_candidates.sort_values(OUTPUT_COLUMNS, kind="stable").reset_index(drop=True)
    pd.testing.assert_frame_equal(real_candidates, expected)


# --------------------------------------------------------------------------
# 3. only the Latin path is fed in
# --------------------------------------------------------------------------


def test_flagged_rows_are_excluded_from_the_query_side() -> None:
    """
    `embed_names.py` documents the asymmetry: source1 is embedded whole because
    a source1 row that is never embedded can never be recalled, but the *lexical*
    path is only for rows an ASCII similarity can read. A Devanagari source1 row
    must therefore not appear as a query here, and its transliterated match is
    expected to be missing.
    """
    s1 = pd.DataFrame(
        [
            ("S1-A", "Prime Money Lenders", "12 MG Road, Bengaluru", "India"),
            ("S1-B", "श्री बालाजी ट्रेडर्स", "Station Road, Ahmedabad", "India"),
        ],
        columns=SOURCE_COLUMNS,
    )
    s2 = pd.DataFrame(
        [("S2-A", "Shree Balaji Traders", "Station Rd, Ahmedabad", "India")],
        columns=SOURCE_COLUMNS,
    )
    out = generate_lexical_candidates(s1, s2, pd.DataFrame(columns=SOURCE_COLUMNS).astype(str),
                                      top_k=5)
    assert "S1-B" not in set(out[QUERY_COL])
    assert "S1-A" in set(out[QUERY_COL])


def test_normalization_happens_automatically_for_a_raw_frame() -> None:
    """
    The supported path is normalize-then-block, and the pipeline does that. But a
    frame handed over raw must still work, and it must work *identically* --
    otherwise the test suite and the pipeline would be testing two different
    functions.
    """
    s1, s2, s3 = _real_frame(), _perturbed_frame(), pd.DataFrame(columns=SOURCE_COLUMNS).astype(str)
    raw = generate_lexical_candidates(s1, s2, s3, top_k=5)
    pre = generate_lexical_candidates(normalize(s1), normalize(s2), normalize(s3), top_k=5)
    assert _pair_set(raw) == _pair_set(pre)


# --------------------------------------------------------------------------
# 4. the pre-filter, on its own
# --------------------------------------------------------------------------

# Six values per case, as the function takes them:
# (query_country, query_pin, query_kind, cand_country, cand_pin, cand_kind, keep?)
PREFILTER_CASES: list[tuple[str, str, str, str, str, str, bool]] = [
    # --- the country rule ---
    ("US", "60901", "us_zip5", "India", "560001", "in_pin6", False),
    ("India", "560001", "in_pin6", "US", "60901", "us_zip5", False),
    ("US", "", None, "India", "", None, False),
    ("US", "", None, "France", "", None, False),
    ("US", "", None, "US", "", None, True),
    # A blank country is unknown, not different. Rejecting on it would drop
    # every pair touching a row with a missing country, and `country` is blank on
    # a measurable slice of the real files. Note the four cases below that pair a
    # blank country with a postal disagreement: the country rule abstains there,
    # and it is the postal rule that decides.
    ("", "", None, "", "", None, True),
    # --- the postal rule, both confident ---
    ("US", "60901", "us_zip5", "US", "60901", "us_zip5", True),   # equal
    ("US", "60901", "us_zip5", "US", "90210", "us_zip5", False),  # different area
    ("US", "60901", "us_zip5", "US", "60914", "us_zip5", True),   # both in the 609xx prefix
    ("US", "60901", "us_zip5", "US", "60614", "us_zip5", False),  # 606xx is a different prefix
    ("US", "60901", "us_zip5", "US", "609", "us_zip5", True),     # a short code that matches
    ("US", "60901", "us_zip5", "US", "9021", "us_zip5", False),
    ("India", "560001", "in_pin6", "India", "400001", "in_pin6", False),
    ("India", "560001", "in_pin6", "India", "560002", "in_pin6", True),
    # A blank country makes the *country* rule abstain, but the postal rule still
    # speaks: the two rows disagree about which delivery area they are in, and
    # that is a real disagreement whichever country is missing.
    ("", "60901", "us_zip5", "US", "90210", "us_zip5", False),
    ("", "60901", "us_zip5", "US", "60914", "us_zip5", True),
    ("US", "60901", "us_zip5", "", "90210", "us_zip5", False),
    ("US", "60901", "us_zip5", "", "60914", "us_zip5", True),
    # --- the postal rule abstains ---
    # `generic` is a street number. Two different house numbers are not a
    # disagreement about a postal code, and this corpus has almost no real
    # postal codes to disagree about (US 0.03%, India 0.01%).
    ("US", "1462", "generic", "US", "7812", "generic", True),
    ("US", "1462", "generic", "US", "90210", "us_zip5", True),
    ("US", "60901", "us_zip5", "US", "7812", "generic", True),
    ("India", "12", "generic", "India", "400", "generic", True),
    ("France", "18", "generic", "France", "75002", "generic", True),
    # One side confident, the other not: abstain rather than guess.
    ("US", "60901", "us_zip5", "US", None, None, True),
    ("US", None, None, "US", "90210", "us_zip5", True),
    # A confident kind with no value cannot be compared either.
    ("US", "", "us_zip5", "US", "90210", "us_zip5", True),
    # An unrecognised kind is not a confident one.
    ("US", "60901", "some_future_kind", "US", "90210", "us_zip5", True),
]


@pytest.mark.parametrize(
    ("q_country", "q_pin", "q_kind", "c_country", "c_pin", "c_kind", "keep"),
    PREFILTER_CASES,
)
def test_prefilter_row_by_row(
    q_country: str, q_pin: str, q_kind: str,
    c_country: str, c_pin: str, c_kind: str, keep: bool,
) -> None:
    assert prefilter_allows(q_country, q_pin, q_kind, c_country, c_pin, c_kind) is keep


def test_prefilter_excludes_same_name_different_country() -> None:
    """
    Isolation test 1 of 2, on `prefilter_allows` directly rather than through
    `generate_lexical_candidates`.

    The business name is byte-identical on both sides -- "Prime Money Lenders" in
    a US row and in an India row. Nothing about the text separates these two, so
    separation cannot be left to the similarity function; the pre-filter is the
    only thing that can do it. Going through the pipeline instead would let this
    pass for the wrong reason: TF-IDF would rank the pair first and LSH would
    return it, and the pair would only disappear at the very end. Calling the
    rule directly means a failure points at the rule.
    """
    identical_name = "Prime Money Lenders"  # both rows, byte for byte
    assert identical_name == identical_name  # the point: the text cannot help
    assert prefilter_allows(
        "US", "60901", "us_zip5", "India", "560001", "in_pin6"
    ) is False, "same name, different country, must be excluded"
    # And the converse, so the rule cannot pass by rejecting everything.
    assert prefilter_allows(
        "India", "560001", "in_pin6", "US", "60901", "us_zip5"
    ) is False


def test_prefilter_excludes_a_same_country_distant_pin_pair() -> None:
    """
    Isolation test 2 of 2, also on the rule directly.

    Same country, and two genuinely confident postal codes in different
    delivery areas. The names differ only slightly here, so the pair would
    otherwise survive on text alone.
    """
    assert prefilter_allows(
        "US", "60901", "us_zip5", "US", "90210", "us_zip5"
    ) is False, "same country, distant confident PINs, must be excluded"
    # Same country and a shared three-digit prefix is the near miss, and it is
    # kept -- otherwise the rule above would be passing by rejecting all pairs.
    assert prefilter_allows("US", "60901", "us_zip5", "US", "60914", "us_zip5") is True
    assert prefilter_allows("US", "60901", "us_zip5", "US", "60901", "us_zip5") is True


def test_prefilter_keeps_the_two_cases_the_task_implies() -> None:
    """Same country, and near postal codes, are kept -- the filter is not a wall."""
    assert prefilter_allows("US", "60901", "us_zip5", "US", "60901", "us_zip5") is True
    # 60901 and 60914 share the leading three digits, so they are the same
    # delivery area even though the codes are not equal.
    assert prefilter_allows("US", "60901", "us_zip5", "US", "60914", "us_zip5") is True
    # And the contrast case, so the two cannot both pass by accident.
    assert prefilter_allows("US", "60901", "us_zip5", "US", "90210", "us_zip5") is False


def test_prefilter_handles_non_string_input() -> None:
    """NaN is what a missing value actually is once pandas has touched a frame."""
    assert prefilter_allows(float("nan"), float("nan"), float("nan"), "US", "90210", "us_zip5") is True
    assert prefilter_allows(None, None, None, None, None, None) is True


def test_vectorized_prefilter_agrees_with_the_row_version() -> None:
    """
    `coarse_prefilter` is the rule over columns and `prefilter_allows` is the
    rule over values. They are the same rule, and this is the test that says so
    over every case in the table above rather than over a hand-picked few.
    """
    left = pd.DataFrame(
        {
            "country": [case[0] for case in PREFILTER_CASES],
            "address_pin": [case[1] for case in PREFILTER_CASES],
            POSTAL_KIND: [case[2] for case in PREFILTER_CASES],
        }
    )
    right = pd.DataFrame(
        {
            "country": [case[3] for case in PREFILTER_CASES],
            "address_pin": [case[4] for case in PREFILTER_CASES],
            POSTAL_KIND: [case[5] for case in PREFILTER_CASES],
        }
    )
    vector = coarse_prefilter(left, right)
    scalar = [case[6] for case in PREFILTER_CASES]
    assert list(vector) == scalar


def test_vectorized_prefilter_needs_aligned_frames() -> None:
    left = pd.DataFrame({"country": ["US"]})
    right = pd.DataFrame({"country": ["US", "India"]})
    with pytest.raises(ValueError, match="aligned frames"):
        coarse_prefilter(left, right)


def test_vectorized_prefilter_tolerates_missing_columns() -> None:
    """
    A frame with no postal columns at all must pass everything the country rule
    allows, rather than raising or rejecting. `normalize` produces the columns
    for real frames; this is for hand-built ones.
    """
    left = pd.DataFrame({"country": ["US", "US", "US"]})
    right = pd.DataFrame({"country": ["US", "India", "US"]})
    assert list(coarse_prefilter(left, right)) == [True, False, True]


def test_the_prefilter_actually_runs_inside_the_module() -> None:
    """
    The row-level tests prove the rule; this proves it is wired in, by counting
    what it dropped on a run where it must drop something.

    The frame is built with a postal column the normalizer cannot produce from
    the text, so the filter's decision is unambiguous and independent of the
    abbreviation and postal parsing in Phase 4.
    """
    s1 = pd.DataFrame(
        [("S1-A", "Prime Money Lenders", "12 MG Road", "US")], columns=SOURCE_COLUMNS
    )
    s2 = pd.DataFrame(
        [
            ("S2-A", "Prime Money Lenders", "12 MG Road", "US"),
            ("S2-B", "Prime Money Lenders", "12 MG Road", "India"),
        ],
        columns=SOURCE_COLUMNS,
    )
    s1 = normalize(s1)
    s2 = normalize(s2)
    s1["address_pin"] = "60901"
    s1[POSTAL_KIND] = "us_zip5"
    s2["address_pin"] = ["60901", "90210"]
    s2[POSTAL_KIND] = ["us_zip5", "us_zip5"]
    s2["country"] = ["US", "US"]  # same country, so only the postal rule can act

    report = BlockingReport()
    out = generate_lexical_candidates(s1, s2, pd.DataFrame(columns=SOURCE_COLUMNS).astype(str),
                                      top_k=5, report=report)
    assert _pair_set(out) == {("S1-A", "S2-A")}
    assert report.prefilter_dropped >= 1


# --------------------------------------------------------------------------
# 5. the pre-filter does not throw away the known matches
# --------------------------------------------------------------------------


def test_the_prefilter_does_not_cost_recall_on_real_rows(real_candidates: pd.DataFrame) -> None:
    """
    The filter's design is that it abstains, because this corpus has almost no
    postal codes. If it fired on the real Latin rows it would be rejecting good
    pairs, and that is the failure mode the `postal_kind` rule exists to avoid.
    """
    found = _pair_set(real_candidates)
    assert [pair for pair in SIMILAR_PAIRS if pair not in found] == []


# --------------------------------------------------------------------------
# 6. parameters
# --------------------------------------------------------------------------


def test_top_k_caps_each_strategy(real_s1, real_s2, real_s3) -> None:
    """
    `top_k` is applied per strategy before the union, so the cap on the output
    is `2 * top_k` rows per query, and the count has to fall as top_k falls.
    """
    wide = generate_lexical_candidates(real_s1, real_s2, real_s3, top_k=8)
    narrow = generate_lexical_candidates(real_s1, real_s2, real_s3, top_k=1)
    assert len(narrow) < len(wide)
    per_query = narrow.groupby(QUERY_COL).size()
    assert (per_query <= 2).all(), "top_k=1 must yield at most 2 rows per source1 row"


def test_top_k_must_be_positive(real_s1, real_s2, real_s3) -> None:
    with pytest.raises(ValueError, match="top_k"):
        generate_lexical_candidates(real_s1, real_s2, real_s3, top_k=0)


def test_chunk_size_must_be_positive(real_s1, real_s2, real_s3) -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        generate_lexical_candidates(real_s1, real_s2, real_s3, chunk_size=0)


def test_chunk_size_does_not_change_the_answer(real_s1, real_s2, real_s3) -> None:
    """
    Chunking exists to bound memory, so it must be invisible in the result. This
    is the test that makes that claim checkable rather than aspirational: one
    chunk against many, compared pair for pair.
    """
    single = generate_lexical_candidates(real_s1, real_s2, real_s3, top_k=5, chunk_size=10_000)
    split = generate_lexical_candidates(real_s1, real_s2, real_s3, top_k=5, chunk_size=2)
    assert _pair_set(single) == _pair_set(split)


def test_a_frame_with_no_entity_id_is_rejected() -> None:
    with pytest.raises(KeyError, match="entity_id"):
        generate_lexical_candidates(
            pd.DataFrame({"business_name": ["x"]}),
            pd.DataFrame({"business_name": ["x"]}),
            pd.DataFrame({"business_name": ["x"]}),
        )


# --------------------------------------------------------------------------
# 7. empty input
# --------------------------------------------------------------------------


def test_empty_sources_return_the_empty_frame_with_the_right_columns(real_s1) -> None:
    empty = pd.DataFrame(columns=SOURCE_COLUMNS).astype(str)
    out = generate_lexical_candidates(empty, real_s1, real_s1, top_k=5)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out.empty


def test_empty_candidates_return_the_empty_frame(real_s1) -> None:
    empty = pd.DataFrame(columns=SOURCE_COLUMNS).astype(str)
    out = generate_lexical_candidates(real_s1, empty, empty, top_k=5)
    assert list(out.columns) == OUTPUT_COLUMNS
    assert out.empty


# --------------------------------------------------------------------------
# 8. the document builder
# --------------------------------------------------------------------------


def test_document_is_name_then_sorted_address_tokens() -> None:
    """
    The sort is load-bearing. The document feeds a word *bigram* analyzer, bigrams
    depend on token order, and `address_token_set` is a `frozenset` with no
    order -- so without the sort two runs over the same data would produce
    different bigrams.
    """
    frame = normalize(
        pd.DataFrame(
            [("S1-A", "Prime Money", "12 MG Road, Bengaluru, Karnataka", "India")],
            columns=SOURCE_COLUMNS,
        )
    )
    document = build_lexical_document(frame)[0]
    assert document.startswith("prime money ")
    tokens = document.split()
    address_part = tokens[2:]
    assert address_part == sorted(address_part)


def test_document_is_identical_across_runs() -> None:
    frame = normalize(
        pd.DataFrame(
            [("S1-A", "Prime Money", "12 MG Road, Bengaluru, Karnataka", "India")],
            columns=SOURCE_COLUMNS,
        )
    )
    assert build_lexical_document(frame) == build_lexical_document(frame)


def test_document_falls_back_to_the_raw_columns() -> None:
    frame = pd.DataFrame([("S1-A", "Prime Money", "12 MG Road", "India")], columns=SOURCE_COLUMNS)
    assert build_lexical_document(frame) == ["Prime Money"]


def test_document_of_a_blank_row_is_empty() -> None:
    frame = pd.DataFrame([("S1-A", "NULL", "###", "US")], columns=SOURCE_COLUMNS)
    document = build_lexical_document(normalize(frame))[0]
    # The noise tokens clean away, so the document has no searchable content and
    # the row cannot become a candidate for anything.
    assert "null" not in document
    assert "###" not in document


# --------------------------------------------------------------------------
# 9. the shingles and the signature
# --------------------------------------------------------------------------


def test_shingles_are_trigrams_of_the_space_stripped_name() -> None:
    assert minhash_shingles("Prime Money") == minhash_shingles("PrimeMoney")
    assert minhash_shingles("Prime Money")[0] == b"pri"


def test_a_short_name_still_gets_a_shingle() -> None:
    """Otherwise it is silently unmatchable rather than merely unmatched."""
    assert minhash_shingles("AB") == [b"ab"]
    assert minhash_shingles("") == []


def test_estimated_jaccard_is_one_for_itself_and_low_for_a_stranger() -> None:
    left = _minhash_for(["prime money lenders"])[0]
    same = _minhash_for(["Prime Money Lenders"])[0]
    other = _minhash_for(["golden gate provisions"])[0]
    assert estimated_jaccard(left, same) == 1.0
    assert estimated_jaccard(left, other) < 0.2


def test_a_blank_name_gets_a_signature_unrelated_to_a_real_one() -> None:
    """
    A name with no shingles would get an all-initial-value signature, and every
    such signature is identical -- which would make every blank name a
    near-perfect LSH match for every other blank name. They are given a
    placeholder shingle instead.
    """
    blank = _minhash_for([""])[0]
    first = _minhash_for(["A"])[0]
    second = _minhash_for(["B"])[0]
    assert estimated_jaccard(blank, first) < 0.2
    assert estimated_jaccard(blank, second) < 0.2
    assert estimated_jaccard(first, second) < 0.2


# --------------------------------------------------------------------------
# 10. the constants the measurements chose
# --------------------------------------------------------------------------


def test_confident_postal_kinds_exclude_generic() -> None:
    """
    `generic` means the digits are a street number. If it ever became
    "confident", the pre-filter would start comparing house numbers and would
    reject true matches, which is the failure this whole design avoids.
    """
    assert "generic" not in CONFIDENT_POSTAL_KINDS
    assert set(CONFIDENT_POSTAL_KINDS) == {"us_zip5", "in_pin6"}


def test_lsh_parameters_are_the_measured_ones() -> None:
    """Pinned so a well-meaning tweak shows up as a failing test, not a silent change."""
    assert MINHASH_NGRAM == 3
    assert LSH_THRESHOLD == 0.3
    assert POSTAL_NEAR_PREFIX == 3
    assert DEFAULT_CHUNK_SIZE == 1_000_000
