"""
End-to-end entry point.

Usage:
    python3 run_pipeline.py --mode train   # train model, report F0.5 on held-out val split
    python3 run_pipeline.py --mode infer   # run full pipeline on test set, write output/*.tsv

Each stage is intentionally a thin import from src/ — fill in the TODOs in each
module; this file should not need heavy logic itself.
"""
import argparse
import pandas as pd

from src.preprocessing import normalize  # TODO: implement
from src.blocking import generate_candidates  # TODO: implement
from src.features import build_features  # TODO: implement
from src.matching import train_model, score_pairs  # TODO: implement
from src.evaluation import split_train_val, score_f_half, tune_threshold  # TODO: implement

DATASET_DIR = "dataset"
OUTPUT_DIR = "output"


def load_sources(split: str):
    """split: 'train' or 'test'"""
    s1 = pd.read_csv(f"{DATASET_DIR}/{split}/{split}_source1.tsv", sep="\t")
    s2 = pd.read_csv(f"{DATASET_DIR}/{split}/{split}_source2.tsv", sep="\t")
    s3 = pd.read_csv(f"{DATASET_DIR}/{split}/{split}_source3.tsv", sep="\t")
    return s1, s2, s3


def write_id_list_tsv(pairs_df: pd.DataFrame, id_col: str, out_path: str, all_s1_ids):
    """
    pairs_df: one row per (source1_entity_id, <id_col>) match/candidate.
    Collapses to one row per S1 entity with a comma-joined ID list;
    ensures every S1 entity appears exactly once, even with no matches.
    """
    grouped = (
        pairs_df.groupby("source1_entity_id")[id_col]
        .apply(lambda ids: ",".join(dict.fromkeys(ids)))  # dedupe, preserve order
        .to_dict()
    )
    rows = [{"source1_entity_id": sid, id_col + "s": grouped.get(sid, "")} for sid in all_s1_ids]
    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def run_train():
    s1, s2, s3 = load_sources("train")
    gt = pd.read_csv(f"{DATASET_DIR}/train/train_ground_truth.tsv", sep="\t")

    s1_train, s1_val, gt_train, gt_val = split_train_val(s1, gt)

    s1_train_n, s2_n, s3_n = normalize(s1_train), normalize(s2), normalize(s3)
    candidates = generate_candidates(s1_train_n, s2_n, s3_n)
    features = build_features(candidates, s1_train_n, s2_n, s3_n)
    model = train_model(features, gt_train)

    # validate
    s1_val_n = normalize(s1_val)
    val_candidates = generate_candidates(s1_val_n, s2_n, s3_n)
    val_features = build_features(val_candidates, s1_val_n, s2_n, s3_n)
    scored = score_pairs(model, val_features)
    threshold = tune_threshold(scored, gt_val)
    f_half = score_f_half(scored, gt_val, threshold)
    print(f"Best threshold: {threshold:.3f}  |  Validation F0.5: {f_half:.4f}")

    # TODO: persist trained model + chosen threshold for inference


def run_infer():
    s1, s2, s3 = load_sources("test")
    s1_n, s2_n, s3_n = normalize(s1), normalize(s2), normalize(s3)

    candidates = generate_candidates(s1_n, s2_n, s3_n)
    write_id_list_tsv(candidates, "candidate_entity_id", f"{OUTPUT_DIR}/candidate_pairs.tsv", s1["entity_id"])

    features = build_features(candidates, s1_n, s2_n, s3_n)
    # TODO: load trained model + threshold from run_train()
    scored = score_pairs(model=None, features=features)  # placeholder
    matches = scored[scored["match_probability"] >= 0.5]  # placeholder threshold
    write_id_list_tsv(
        matches.rename(columns={"candidate_entity_id": "matched_entity_id"}),
        "matched_entity_id",
        f"{OUTPUT_DIR}/matching_results.tsv",
        s1["entity_id"],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    else:
        run_infer()
