# Task Split — 3 people

The pipeline has 5 stages but they chain tightly (blocking output feeds features,
features feed the model, model output feeds thresholding). Splitting by stage lets
each person own an interface, not a person waiting on another all week.

## Person A — Data & Blocking
`src/preprocessing/`, `src/blocking/`
- Normalize `business_name` / `business_address` (suffix expansion, abbreviation
  expansion, tokenization). Keep it country-agnostic — France hits this in test.
- Build candidate generation: TF-IDF/cosine ANN + token n-gram LSH + (optional)
  embedding-based blocking. Union candidate sets.
- Owns `candidate_pairs.tsv` generation and its recall metric (measured against
  train ground truth on the held-out split).
- **Interface out:** a function `generate_candidates(s1_df, s2_df, s3_df) -> DataFrame[source1_entity_id, candidate_entity_id]`

## Person B — Features & Model
`src/features/`, `src/matching/`
- Turn each (S1, candidate) pair into a feature vector (name/address similarity,
  country match, source-pair indicator, etc. — see PS "Tips for Success").
- Train the GBDT classifier on features + ground-truth labels (positives from
  `train_ground_truth.tsv`, negatives = blocked-but-not-matched pairs).
- **Interface in:** candidate pairs DataFrame from Person A.
- **Interface out:** a function `score_pairs(pairs_df) -> DataFrame[..., match_probability]`

## Person C — Evaluation, Thresholding & Submission
`src/evaluation/`, `run_pipeline.py`, `utils/validate_submission.py` integration, `Documentation_template.md`
- Build the exact macro F0.5 scorer from the PS formula.
- Build the train/val split (by S1 entity ID, no leakage).
- Threshold sweep to maximize F0.5 on validation.
- Wire the full pipeline end-to-end (`run_pipeline.py --mode train / --mode infer`).
- Own `output/matching_results.tsv` generation, validator runs, and the final
  methodology writeup + submission zip assembly.
- **Interface in:** scored pairs DataFrame from Person B.
- **Interface out:** `output/matching_results.tsv`, `output/candidate_pairs.tsv`.

## Suggested week-1 sequence
1. Everyone reads the PS + inspects the actual data together (1 session).
2. A starts blocking on a **dummy/small sample** while B stubs feature functions
   against A's interface (agree on column names first — that's the real
   dependency, not finished code).
3. C builds the F0.5 scorer and train/val split immediately — it's needed to
   evaluate everything else and has no dependency on A or B.
4. Integrate in `run_pipeline.py` once each piece runs standalone.
5. Iterate: A pushes candidate recall up, B/C tune precision via features/threshold.

## Shared conventions (agree before splitting up)
- Column names: `source1_entity_id`, `candidate_entity_id` (singular row per
  candidate, not comma-joined) as the internal working format; only join into
  comma-separated lists at final TSV-writing time.
- All intermediate DataFrames keep `entity_id` prefixes (`S1-`, `S2-`, `S3-`) as-is.
- Random seed fixed in one shared `config.py` / `constants.py` so validation
  splits are reproducible across everyone's machines.
