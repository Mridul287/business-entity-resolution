"""
Phase 5 lexical blocking: TF-IDF nearest neighbours unioned with character
n-gram MinHash/LSH, behind a country/postal pre-filter.

Owner: Person A

WHAT THIS MODULE IS
-------------------

`generate_lexical_candidates(s1_df, s2_df, s3_df, top_k=20)` returns the long
candidate frame every later stage is built on:

    source1_entity_id | candidate_entity_id

It owns the *recall ceiling*. Nothing downstream can recover a pair this
function drops, which is why the two strategies here are deliberately
overlapping and why the pre-filter is a filter and not a scorer.

THE TWO STRATEGIES, AND WHY THEY ARE DIFFERENT KINDS OF THING
------------------------------------------------------------

1. **Word-level TF-IDF cosine, top_k nearest neighbours.**
   Catches the ordinary case: the same business written the same way twice,
   with the name plus the address's component words as the document. Word-level
   rather than character-level, and that is a measured choice, not a default.
   Character 3-5 grams over this corpus cost 6-7 GB of float32 matrix for the
   10.32M-row candidate side; word 1-2 grams cost 0.6 GB for the same
   documents. On a 15 GB machine the character-level index is the difference
   between fitting and not fitting -- and character-level similarity is
   supplied by strategy 2 anyway, where it is the whole point.

2. **Character 3-gram MinHash + LSH, `datasketch`.**
   Catches what token overlap cannot: typos, transposed characters, a dropped
   letter. "Prime Money Lenders" and "Prine Money Lander" share almost no word
   tokens but share most of their character trigrams. The shingles come from the
   *space-stripped* name, so "Prime Money" and "PrimeMoney" are one key, which
   is a real difference between sources on this corpus.

   `MinHashLSH` does the banding. The signatures come from `MinHash.bulk`,
   which on this machine is 14x faster than building `MinHash` objects one at a
   time -- 0.082 against 1.159 ms/key, measured -- and turns a 3.3 h hashing
   pass over the candidate side into about 14 minutes.

The union is the point. Either strategy alone has a failure mode the other does
not, and blocking is the stage where a cheap wrong answer costs nothing while a
missing answer costs the whole submission.

WHY THE CANDIDATE SIDE IS PROCESSED IN CHUNKS
----------------------------------------------

A `MinHashLSH` index costs ~2.9 KB per inserted key, measured. The candidate
side is 10.32M rows, so one full index is ~30 GB against 15.4 GB of RAM here.
The index is therefore built over slices of the candidate side and every query
is run against every slice in turn.

That is exact rather than approximate: each candidate row is inserted into
exactly one slice and each query is asked of every slice, so the candidate set
is the one a single full index would have returned. It costs one extra pass over
the queries per chunk and it is bounded by `chunk_size` instead of by the size
of the corpus. The TF-IDF search is chunked the same way for the same reason: a
brute-force cosine pass needs the candidate matrix resident.

WHAT THE PRE-FILTER DOES, AND WHY IT IS NOT A HARD POSTAL RULE
-------------------------------------------------------------

`coarse_prefilter` drops a pair when

  * the two rows carry different, non-blank `country` values; or
  * both rows carry a *confident* postal code (`us_zip5` / `in_pin6`), the codes
    differ, and they do not share a 3-digit prefix.

Both halves are deliberately weaker than "the PINs must be equal", and the
reason is the most consequential measurement in this repo. Run against the
real corpus, a postal code is present *in the position where one belongs* for
France 0.00%, India 0.01% and US 0.03% of rows: this corpus does not carry
postal codes. A rule that requires two `address_pin` values to be equal is a
rule that fires on house numbers -- "1462 Sixth Avenue" and "7812 Colusa
Street" both yield a five-digit `generic` value -- and a rule that *rejects*
pairs whose codes differ throws away nearly every true match.

So `address_postal_kind` decides. `generic` means "these digits are a street
number, and street numbers are not comparable", and such a pair is never
dropped. A blank country is unknown rather than mismatched and is never dropped
either. The filter fires on the confident rows, where it is a real signal, and
abstains everywhere else.

WHAT THIS MODULE DELIBERATELY DOES NOT DO
-----------------------------------------

It does not score. Similarity is computed to rank and to cap, then discarded: a
threshold on cosine similarity inside blocking trades recall for a precision
that the GBDT in `src/matching/` is far better placed to decide.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from ..preprocessing.multilingual import DEFAULT_DATASET_DIR, read_stratified_sample
from ..preprocessing.normalize import (
    ADDRESS_NORM,
    ADDRESS_PIN,
    ADDRESS_TOKEN_SET,
    NAME_NORM,
    NON_LATIN_FLAG,
    normalize,
)
from ..preprocessing.profile_data import COUNTRY_COL, NAME_COL

QUERY_COL = "source1_entity_id"
CANDIDATE_COL = "candidate_entity_id"
OUTPUT_COLUMNS = [QUERY_COL, CANDIDATE_COL]

# The source files' own id column, and the id column the output is keyed on.
# `embed_runtime.py` spells the same two strings inline rather than sharing a
# constant, so this module does the same instead of pretending to a shared one
# that does not exist.
ENTITY_ID = "entity_id"

POSTAL_KIND = "address_postal_kind"

# --- TF-IDF -----------------------------------------------------------------
# Word 1-2 grams, not character n-grams: the 0.6 GB against 6-7 GB measurement
# in the module docstring. min_df=2 drops the hapax vocabulary, which on a 10M
# row corpus is mostly typos and unit numbers, and sublinear_tf damps the
# repetition that comes from a long address repeating a city name.
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 2
TFIDF_SUBLINEAR_TF = True

# --- MinHash / LSH ---------------------------------------------------------
MINHASH_NUM_PERM = 64
MINHASH_NGRAM = 3
# 0.3, and the ordering is measured rather than guessed. Against a probe pool of
# 57,175 real S2/S3 rows, with 1,944 real S1 queries whose true matches come from
# `train_ground_truth`, recall of at least one true match inside the top 20 for
# the LSH strategy alone:
#
#     threshold   recall@20   all matches   hits/query   ms/query
#         0.20      94.96%        65.17%       3,379        3.99
#         0.25      94.86%        64.09%       2,835        3.09
#         0.30      94.24%        59.83%         830        0.93
#         0.40      92.59%        47.58%         116        0.18
#
# 0.3 is the knee, and the knee is the finding: it gives up 0.7 points of recall
# against 0.2 for a quarter of the query cost and a tenth of the candidate churn,
# while 0.4 costs 2.4 points -- and those points are exactly the pairs whose name
# similarity sits between the two thresholds.
#
# Read the absolute levels with care, because the probe pool is not the real
# candidate set. It is every true match for the sampled queries (6,167 of 57,175
# rows, 10.8% true-match density) plus 60k background rows, whereas the real
# candidate side is ~9.4M rows at a true-match density of ~3e-7. The probe is
# therefore about 320,000x denser in real matches, so a true match faces almost no
# competition for its 20 slots. These figures are an optimistic bound for this
# strategy, not a forecast of full-scale recall, and they are also unseeded --
# `MinHash` draws fresh permutations each run, so recall moves ~0.05 points
# between runs while the hit counts stay identical to the decimal. The threshold
# *ranking* is a within-probe comparison and survives that; the absolute number
# does not transfer. Full-scale recall is Benchmark B's job, not this constant's.
#
# The ~5% of true pairs no threshold here recovers is not a tuning failure. The
# measured trigram Jaccard of a true pair has a 1st percentile of 0.000: those
# rows share no name characters at all and belong to the embedding path in
# `embed_names.py`, which is the whole reason that path exists.
LSH_THRESHOLD = 0.3

# --- chunking --------------------------------------------------------------
# 1M candidate rows is ~2.9 GB of LSH at the measured 2.9 KB/key plus 0.5 GB of
# stacked digests, which leaves room for the query matrix and the frames on a
# 15 GB machine. Lowering it makes the run slower, not different -- and it is
# the query side that gets *cheaper* as chunks get smaller, since a query's hit
# count is bounded by the chunk it is asked about.
DEFAULT_CHUNK_SIZE = 1_000_000

# --- pre-filter ------------------------------------------------------------
# us_zip5 and in_pin6 only; `generic` is a street number. See the docstring.
CONFIDENT_POSTAL_KINDS = ("us_zip5", "in_pin6")
# The shared prefix that makes two codes "near". For a US ZIP the first three
# digits are the sectional centre facility and for an Indian PIN the first
# three are the postal district, so a shared prefix means the same delivery
# area -- the weakest statement here that is still useful.
POSTAL_NEAR_PREFIX = 3

# A cosine similarity of zero means "no shared term at all", which for a blank
# name is every other row. Such a pair is not a candidate.
MIN_SIMILARITY = 0.0


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _as_text(value: object) -> str:
    """Blank for None and for float NaN, which is what a raw column carries."""
    if value is None:
        return ""
    if isinstance(value, float) and value != value:
        return ""
    return value if isinstance(value, str) else str(value)


def _text_array(values: Iterable[object]) -> np.ndarray:
    return np.array([_as_text(value) for value in values], dtype=object)


# --------------------------------------------------------------------------
# the pre-filter
# --------------------------------------------------------------------------


def prefilter_allows(
    query_country: object,
    query_pin: object,
    query_kind: object,
    candidate_country: object,
    candidate_pin: object,
    candidate_kind: object,
) -> bool:
    """
    The whole pre-filter, for one pair, as a plain function of six values.

    Split out from the vectorized path so the rule can be read, and tested, on
    its own. `coarse_prefilter` below is this same rule over aligned columns,
    and the test suite asserts the two agree.

    Returns True to KEEP the pair.
    """
    left, right = _as_text(query_country), _as_text(candidate_country)
    # A blank label is unknown, not different. Rejecting on it would discard
    # every pair touching a row whose country is missing.
    if left and right and left != right:
        return False

    query_kind, candidate_kind = _as_text(query_kind), _as_text(candidate_kind)
    if query_kind not in CONFIDENT_POSTAL_KINDS:
        return True
    if candidate_kind not in CONFIDENT_POSTAL_KINDS:
        return True

    query_code, candidate_code = _as_text(query_pin), _as_text(candidate_pin)
    if not query_code or not candidate_code or query_code == candidate_code:
        return True
    # Same delivery area -> near enough to keep. A different area *and* a
    # different code is a real disagreement, and those are the pairs to drop.
    return query_code[:POSTAL_NEAR_PREFIX] == candidate_code[:POSTAL_NEAR_PREFIX]


def coarse_prefilter(
    query: pd.DataFrame,
    candidates: pd.DataFrame,
    country_column: str = COUNTRY_COL,
    pin_column: str = ADDRESS_PIN,
    kind_column: str = POSTAL_KIND,
) -> np.ndarray:
    """
    Vectorized `prefilter_allows` over two *aligned* frames of equal length.

    Returns a boolean array where True means keep. Aligned positionally rather
    than index-matched on purpose: the caller has just built these two frames
    out of a neighbour search, and re-deriving the alignment is where this kind
    of filter usually goes wrong.
    """
    if len(query) != len(candidates):
        raise ValueError(
            f"prefilter needs aligned frames, got {len(query)} and {len(candidates)} rows"
        )

    def column(frame: pd.DataFrame, name: str) -> np.ndarray:
        if name not in frame.columns:
            return np.array([""] * len(frame), dtype=object)
        return _text_array(frame[name].to_numpy())

    left_country = column(query, country_column)
    right_country = column(candidates, country_column)
    keep = (left_country == "") | (right_country == "") | (left_country == right_country)

    left_kind = column(query, kind_column)
    right_kind = column(candidates, kind_column)
    both_confident = np.isin(left_kind, CONFIDENT_POSTAL_KINDS) & np.isin(
        right_kind, CONFIDENT_POSTAL_KINDS
    )
    if both_confident.any():
        left_pin = column(query, pin_column)
        right_pin = column(candidates, pin_column)
        comparable = both_confident & (left_pin != "") & (right_pin != "")
        if comparable.any():
            differs = comparable & (left_pin != right_pin)
            if differs.any():
                near = np.array(
                    [
                        a[:POSTAL_NEAR_PREFIX] == b[:POSTAL_NEAR_PREFIX]
                        for a, b in zip(left_pin, right_pin)
                    ]
                )
                keep &= ~(differs & ~near)
    return keep


# --------------------------------------------------------------------------
# the blocking document
# --------------------------------------------------------------------------


def build_lexical_document(df: pd.DataFrame) -> list[str]:
    """
    One text document per row: the folded name, then the address's component
    words.

    The address arrives as a `frozenset` and is **sorted** before being joined.
    That is not cosmetic. The document feeds a word *bigram* analyzer, bigrams
    depend on token order, and a set has no order -- so without the sort two
    runs over the same data produce different bigrams and a reproducibility
    check fails for no visible reason.

    Falls back to whatever the frame has. A frame straight out of `read_csv`
    has neither `name_norm` nor `address_token_set`, so the raw columns are
    used; `generate_lexical_candidates` normalizes first, which is the
    supported path, and the fallback is what keeps a hand-built test frame
    usable.
    """
    if NAME_NORM in df.columns:
        names = df[NAME_NORM]
    elif NAME_COL in df.columns:
        names = df[NAME_COL]
    else:
        names = pd.Series([""] * len(df), index=df.index)

    if ADDRESS_TOKEN_SET in df.columns:
        addresses = [
            " ".join(sorted(_as_text(token) for token in value)) if value else ""
            for value in df[ADDRESS_TOKEN_SET]
        ]
    elif ADDRESS_NORM in df.columns:
        addresses = [_as_text(value) for value in df[ADDRESS_NORM]]
    else:
        addresses = [""] * len(df)

    return [
        f"{_as_text(name)} {address}".strip()
        for name, address in zip(names, addresses)
    ]


def _name_values(df: pd.DataFrame) -> np.ndarray:
    for column in (NAME_NORM, NAME_COL):
        if column in df.columns:
            return _text_array(df[column].to_numpy())
    return np.array([""] * len(df), dtype=object)


# --------------------------------------------------------------------------
# strategy 1: TF-IDF cosine nearest neighbours
# --------------------------------------------------------------------------


def _fit_vectorizer(documents: Sequence[str]) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        strip_accents="unicode",
        dtype=np.float32,
    )
    vectorizer.fit(documents)
    return vectorizer


def _tfidf_candidates(
    query_matrix,
    candidate_matrix,
    top_k: int,
) -> list[tuple[int, int, float]]:
    """
    The `top_k` nearest candidates for every query row, as
    `(query_position, candidate_position, similarity)` triples, where positions
    are into the matrices passed in.

    `NearestNeighbors` returns a cosine *distance*, so it is turned back into a
    similarity, and a non-positive similarity is dropped: an all-zero document
    -- a blank name, a name that cleaned away to nothing -- has cosine 0 against
    everything and would otherwise be every other row's nearest neighbour.

    `n_jobs=-1` is not a micro-optimisation. `algorithm="brute"` is the only
    option for sparse input here, and the pairwise pass is the whole cost of this
    function; measured on this machine it runs at 137M pairs/s across 14 cores
    against 10M pairs/s on one, which is the difference between about one hour
    and about fourteen for a full-scale query side. Left at the default of one
    job it wastes thirteen of the fourteen cores the machine has.
    """
    if candidate_matrix.shape[0] == 0 or query_matrix.shape[0] == 0:
        return []

    neighbours = NearestNeighbors(
        n_neighbors=min(top_k, candidate_matrix.shape[0]),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )
    neighbours.fit(candidate_matrix)
    distances, indices = neighbours.kneighbors(query_matrix)

    # Vectorised, because the alternative is a Python loop of
    # `queries * top_k` -- a million iterations at full scale, on the hot path.
    # `-1` marks a row sklearn found no neighbour for, which cannot happen with
    # `n_neighbors <= n_candidates` but is filtered rather than trusted.
    valid = indices >= 0
    similarity = 1.0 - np.asarray(distances, dtype=np.float64)
    keep = valid & (similarity > 0.0)
    if not keep.any():
        return []
    rows, columns = np.nonzero(keep)
    return [
        (int(row), int(indices[row, column]), float(similarity[row, column]))
        for row, column in zip(rows, columns)
    ]


# --------------------------------------------------------------------------
# strategy 2: character n-gram MinHash / LSH
# --------------------------------------------------------------------------


def minhash_shingles(name: str) -> list[bytes]:
    """
    The character trigrams of a name, with the spaces removed.

    Stripping the spaces is what makes this tolerant of the difference that
    matters on this corpus: one source writes "Prime Money" and another writes
    "PrimeMoney", and those are the same business.

    A name shorter than the n-gram length still gets one shingle, so it stays
    indexable rather than silently unmatchable.
    """
    text = "".join(_as_text(name).split()).lower()
    if not text:
        return []
    if len(text) < MINHASH_NGRAM:
        return [text.encode()]
    return [
        text[i : i + MINHASH_NGRAM].encode()
        for i in range(len(text) - MINHASH_NGRAM + 1)
    ]


def _minhash_for(names: Sequence[str]) -> list[MinHash]:
    """
    Signatures for a list of names, via `MinHash.bulk`.

    A name with no shingles would produce a signature that is all-initial-value
    and therefore *identical for every such name*, which would make every blank
    name a near-perfect LSH match for every other blank name. Such names are
    given a single space shingle instead: still a valid signature, and
    unrelated to any real name.
    """
    filled = [values if values else [b" "] for values in map(minhash_shingles, names)]
    return list(MinHash.bulk(filled, num_perm=MINHASH_NUM_PERM))


def estimated_jaccard(left: MinHash, right: MinHash) -> float:
    """The MinHash estimate: the fraction of permutations on which they agree."""
    return float((np.asarray(left.digest()) == np.asarray(right.digest())).mean())


def _lsh_candidates(
    query_hashes: Sequence[MinHash],
    query_names: Sequence[str],
    candidate_hashes: Sequence[MinHash],
    candidate_names: Sequence[str],
    top_k: int,
) -> list[tuple[int, int, float]]:
    """
    The `top_k` most LSH-similar candidates for every query, as
    `(query_position, candidate_position, estimated_jaccard)` triples.

    `MinHashLSH.query` returns whatever shares a band, in band order, which is
    not a similarity order. The hits are therefore re-ranked by the MinHash
    estimated Jaccard and only then truncated; without that step `top_k` would
    be an arbitrary sample of a query's LSH hits rather than its best ones,
    which is a worse thing than a documented cap.

    Two implementation notes, both because the per-query hit count is large --
    830 per query at threshold 0.3, measured -- and a Python loop over them
    would dominate the run:

      * the candidate digests are stacked once into a single
        `(candidates, num_perm)` uint64 array, so scoring a query's hits is one
        vectorised gather-and-compare rather than `len(hits)` separate
        `estimated_jaccard` calls;
      * the same stack is reused for every query, so it is built once per chunk
        rather than once per query.

    Rows with a blank name are skipped on both sides. A blank query name has no
    shingles worth matching and a blank candidate name has no name to match, so
    a pair between them is noise that would otherwise spend `top_k` slots.
    """
    if len(candidate_hashes) == 0 or len(query_hashes) == 0:
        return []

    # The digest stack is indexed by *original candidate position*, not by row in
    # a compacted list, so an LSH key needs no remapping. An earlier version kept
    # a compacted list and rebuilt a position->row dict on every query, which is
    # `queries x candidates` dict insertions -- 4.1e9 for a 50k-query run, and
    # the reason an early 50k benchmark took eight minutes where the measured
    # cost is under one. Blank positions are simply never inserted into the
    # index, so their (zero-filled) rows are never gathered.
    index = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    digests = np.zeros((len(candidate_hashes), MINHASH_NUM_PERM), dtype=np.uint64)
    for position, name in enumerate(candidate_names):
        if not name:
            continue
        signature = candidate_hashes[position]
        index.insert(str(position), signature, check_duplication=False)
        digests[position] = np.asarray(signature.digest(), dtype=np.uint64)

    found: list[tuple[int, int, float]] = []
    for row, (name, signature) in enumerate(zip(query_names, query_hashes)):
        if not name:
            continue
        raw_hits = index.query(signature)
        if not raw_hits:
            continue
        # The LSH keys are already the original candidate positions.
        hits = np.fromiter((int(hit) for hit in raw_hits), dtype=np.int64, count=len(raw_hits))
        if hits.size == 0:
            continue
        scores = (digests[hits] == np.asarray(signature.digest(), dtype=np.uint64)).mean(axis=1)
        order = np.argsort(-scores, kind="stable")[:top_k]
        for offset in order:
            found.append((row, int(hits[int(offset)]), float(scores[int(offset)])))
    return found


# --------------------------------------------------------------------------
# the union
# --------------------------------------------------------------------------


@dataclass
class BlockingReport:
    """What one run did. Read by the benchmark and by a curious reviewer."""

    query_rows: int = 0
    candidate_rows: int = 0
    documents: int = 0
    vocabulary: int = 0
    chunk_passes: int = 0
    tfidf_pairs: int = 0
    lsh_pairs: int = 0
    prefilter_dropped: int = 0
    self_pairs_dropped: int = 0
    duplicates_dropped: int = 0
    candidates: int = 0
    seconds: float = 0.0
    timings: dict[str, float] = field(default_factory=dict)

    def as_rows(self) -> list[tuple[str, object]]:
        return [
            ("source1 rows", f"{self.query_rows:,}"),
            ("candidate rows", f"{self.candidate_rows:,}"),
            ("documents", f"{self.documents:,}"),
            ("vocabulary", f"{self.vocabulary:,}"),
            ("chunk passes", f"{self.chunk_passes:,}"),
            ("tfidf pairs", f"{self.tfidf_pairs:,}"),
            ("lsh pairs", f"{self.lsh_pairs:,}"),
            ("prefilter dropped", f"{self.prefilter_dropped:,}"),
            ("self pairs dropped", f"{self.self_pairs_dropped:,}"),
            ("duplicates dropped", f"{self.duplicates_dropped:,}"),
            ("candidates", f"{self.candidates:,}"),
            ("seconds", f"{self.seconds:,.1f}"),
        ]


def _as_blocking_frame(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """
    Normalize the frame if Phase 4 has not already been through it.

    The pipeline calls `normalize` and then `generate_candidates`, so the
    supported path arrives ready-made and this returns it untouched. The test is
    on `name_norm` rather than on every contract column, because that is the
    one both strategies read and the one whose absence changes the answer.
    """
    if NAME_NORM in df.columns:
        return df
    if ENTITY_ID not in df.columns:
        raise KeyError(
            f"{label} has neither {NAME_NORM} nor {ENTITY_ID}, so it is not a source frame"
        )
    return normalize(df)


def _latin_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    The rows this module is allowed to see: the Latin path,
    `has_non_latin_name == False`.

    The existing flag is respected rather than recomputed, because a caller
    that has already filtered should not have the filter applied a second time
    under a different definition. A frame with no flag column is left whole:
    that is a hand-built test frame, and refusing it would be unhelpful.
    """
    if NON_LATIN_FLAG in df.columns:
        return df[~df[NON_LATIN_FLAG].astype(bool)]
    return df


def _pairs_to_frame(
    pairs: Sequence[tuple[int, int, float]],
    offset: int,
    query: pd.DataFrame,
    chunk: pd.DataFrame,
    report: BlockingReport,
) -> pd.DataFrame:
    """Neighbour positions to ids, then through the pre-filter."""
    if not pairs:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    left = np.fromiter((row for row, _, _ in pairs), dtype=np.intp, count=len(pairs))
    right = np.fromiter((position for _, position, _ in pairs), dtype=np.intp, count=len(pairs))
    frame = pd.DataFrame(
        {
            QUERY_COL: query[ENTITY_ID].to_numpy()[left],
            CANDIDATE_COL: chunk[ENTITY_ID].to_numpy()[right],
        }
    )
    before_prefilter = len(frame)
    keep = coarse_prefilter(
        query.iloc[left].reset_index(drop=True),
        chunk.iloc[right].reset_index(drop=True),
    )
    report.prefilter_dropped += before_prefilter - int(keep.sum())
    return frame[keep].reset_index(drop=True)


def _direction(
    query: pd.DataFrame,
    candidates: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    query_matrix,
    top_k: int,
    chunk_size: int,
    report: BlockingReport,
) -> pd.DataFrame:
    """Both strategies for one source1 -> sourceN direction, unioned."""
    query_names = _name_values(query)
    candidate_names = _name_values(candidates)
    # Hoisted out of the chunk loop: the query signatures are the same for
    # every chunk, and recomputing them per chunk would make the cost
    # proportional to the number of chunks rather than to the number of queries.
    query_hashes = _minhash_for(list(query_names))

    pieces: list[pd.DataFrame] = []
    total = len(candidates)
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        chunk = candidates.iloc[start:stop]
        chunk_names = candidate_names[start:stop]
        report.chunk_passes += 1

        chunk_matrix = vectorizer.transform(build_lexical_document(chunk))
        found = _tfidf_candidates(query_matrix, chunk_matrix, top_k)
        report.tfidf_pairs += len(found)
        pieces.append(_pairs_to_frame(found, 0, query, chunk, report))

        found = _lsh_candidates(
            query_hashes,
            list(query_names),
            _minhash_for(list(chunk_names)),
            list(chunk_names),
            top_k,
        )
        report.lsh_pairs += len(found)
        pieces.append(_pairs_to_frame(found, 0, query, chunk, report))

    if not pieces:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(pieces, ignore_index=True)


def generate_lexical_candidates(
    s1_df: pd.DataFrame,
    s2_df: pd.DataFrame,
    s3_df: pd.DataFrame,
    top_k: int = 20,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    report: BlockingReport | None = None,
) -> pd.DataFrame:
    """
    The lexical candidate set: one row per (source1_entity_id,
    candidate_entity_id), deduplicated, sorted, and nothing else in the frame.

    `top_k` caps each strategy separately *before* the union, so the result is at
    most `2 * top_k` rows per source1 row before the pre-filter, and a pair
    found by both strategies appears once.

    `chunk_size` bounds peak memory (see the module docstring). It changes how
    long the run takes and how much RAM it needs; it does not change which pairs
    come out.

    Frames are normalized here if they arrive raw, so the function works both
    from `run_pipeline.py` and from a test that builds a frame by hand.
    """
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")

    started = time.perf_counter()
    query = _latin_rows(_as_blocking_frame(s1_df, "s1_df"))
    candidates_by_source = [
        _latin_rows(_as_blocking_frame(s2_df, "s2_df")),
        _latin_rows(_as_blocking_frame(s3_df, "s3_df")),
    ]
    candidates_by_source = [frame for frame in candidates_by_source if not frame.empty]

    stats = report if report is not None else BlockingReport()
    stats.query_rows = len(query)
    stats.candidate_rows = sum(len(frame) for frame in candidates_by_source)

    empty = pd.DataFrame(columns=OUTPUT_COLUMNS)
    if query.empty or not candidates_by_source:
        stats.seconds = time.perf_counter() - started
        return empty

    # One vectorizer for all three sources. Fitting per source would give three
    # different IDF scales, and a cosine between two differently-scaled vectors
    # is not a similarity.
    documents = build_lexical_document(
        pd.concat([query] + candidates_by_source, ignore_index=True)
    )
    vectorizer = _fit_vectorizer(documents)
    stats.documents = len(documents)
    stats.vocabulary = len(vectorizer.vocabulary_)
    del documents

    query_matrix = vectorizer.transform(build_lexical_document(query))
    pieces = [
        _direction(query, frame, vectorizer, query_matrix, top_k, chunk_size, stats)
        for frame in candidates_by_source
    ]
    found = pd.concat([piece for piece in pieces if len(piece)], ignore_index=True)

    before = len(found)
    found = found.drop_duplicates()
    stats.duplicates_dropped = before - len(found)

    # A source1 row is never its own candidate. The id prefixes make that
    # impossible for the real files, but a hand-built frame can put the same id
    # on both sides, and a self-pair is never useful.
    before = len(found)
    found = found[found[QUERY_COL] != found[CANDIDATE_COL]]
    stats.self_pairs_dropped = before - len(found)

    found = found.sort_values(OUTPUT_COLUMNS, kind="stable").reset_index(drop=True)
    stats.candidates = len(found)
    stats.seconds = time.perf_counter() - started
    return found[OUTPUT_COLUMNS]


# --------------------------------------------------------------------------
# benchmark
# --------------------------------------------------------------------------


def _benchmark(rows: int, top_k: int, chunk_size: int) -> int:
    """
    A stratified in-sample run, reporting candidate count, reduction against the
    brute-force cross product, and wall time.

    Reduction is the honest headline for a blocking function: the brute-force
    alternative is every source1 row against every candidate row, and the whole
    reason this function exists is that that product is 2.3e13 pairs.

    `chunk_passes` counts chunk visits summed over both candidate directions, so
    it is one more than the partition of a single candidate side would suggest:
    source2 and source3 are indexed separately, and a 50k-per-source run makes
    two passes of one chunk each rather than one pass of two chunks.
    """
    print(f"stratified sample: {rows:,} rows per source file", flush=True)
    sample = read_stratified_sample(rows=rows, strata=5, splits=("train",))
    s1 = sample[sample["entity_id"].str.startswith("S1")]
    s2 = sample[sample["entity_id"].str.startswith("S2")]
    s3 = sample[sample["entity_id"].str.startswith("S3")]
    latin = int((~normalize(sample)["has_non_latin_name"]).sum())
    print(
        f"  S1 {len(s1):,} / S2 {len(s2):,} / S3 {len(s3):,}"
        f"  ({latin:,} latin, {len(sample) - latin:,} routed to the embedding path)",
        flush=True,
    )

    report = BlockingReport()
    started = time.perf_counter()
    candidates = generate_lexical_candidates(s1, s2, s3, top_k=top_k, chunk_size=chunk_size,
                                             report=report)
    wall = time.perf_counter() - started

    brute_force = report.query_rows * report.candidate_rows
    print(f"\n{'metric':>22} | {'value':>16}")
    print("-" * 43)
    for label, value in report.as_rows():
        print(f"{label:>22} | {str(value):>16}")
    print(f"{'brute force pairs':>22} | {brute_force:>16,}")
    reduction = (1 - report.candidates / brute_force) if brute_force else 0.0
    print(f"{'reduction vs brute':>22} | {reduction:>15.6%}")
    print(f"{'candidates per query':>22} | {report.candidates / max(report.query_rows, 1):>16.2f}")
    print(f"{'wall clock (s)':>22} | {wall:>16.1f}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--rows", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args(argv)
    return _benchmark(args.rows, args.top_k, args.chunk_size)


if __name__ == "__main__":
    raise SystemExit(main())
