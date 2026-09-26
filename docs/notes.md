# Working notes

Use this for running design decisions, blockers, and things to revisit —
keep meeting notes / brainstorming out of code comments.

## Data profile (Person A, 2026-09-26)

Full numbers: [`data_profile_report.md`](data_profile_report.md), regenerated with
`python -m src.preprocessing.profile_data --out docs/data_profile_report.md`
(~11 min for the 2.4 GB of TSVs; it streams in chunks, so it needs no extra RAM).
Tests for it: `pytest tests/preprocessing/ -v` (synthetic frames only, no dataset
access, so they run in about a second).

What the numbers imply for `src/preprocessing/normalize.py` and
`src/blocking/generate_candidates.py`:

- **S1 is clean, S2/S3 are not.** Source 1 has ~0 noise-token rows and 0 names
  under 3 characters; S2/S3 run 0.2–0.4% noise-token rows and 0.2–0.4% very
  short names. Normalize toward the S1 shape, and expect the raw S2/S3 text to
  disagree with itself more than with S1.
- **`##` is an address-level noise pattern, not a name one.** ~2.3–2.7% of
  S2/S3 address rows contain it, plus ~1.4–1.75% `NULL` and ~0.7–0.9% `N/A`.
  Every file parses cleanly as 4-column TSV (no malformed lines skipped), so
  these are values, not parse artefacts.
- **`***` is a name-level pattern** (~0.24–0.31% of S2/S3 names) and `[Limited]`
  -style brackets hit ~2.6–3.1% of S2/S3 names. Both need stripping before any
  character n-gram key is built or they will act as blocking keys themselves.
- **Scripts:** S1 names are 0% non-ASCII, but S2/S3 names are 11–19% non-ASCII,
  spread across ten Indic scripts (Devanagari largest at 3–6%, then Telugu,
  Kannada, Tamil, Bengali, Gujarati, Malayalam, Gurmukhi, Oriya). A Latin-only
  n-gram key will miss those pairs outright — transliteration handling in
  blocking is not optional here.
- **Latin-accented rises in test** (7.8–8.2% of S2/S3 test names vs 5.8–6.2% at
  train) because of the ~14–15% France rows, which is unseen at train time.
  Accent folding has to be a normalization step, not a train-fitted assumption.
- **Country mix:** train is 60/40 US/India; test is ~47/38/15 India/US/France.
  Test Source 1 has 1,732,544 entities, so that is the row count the submission
  must cover.
