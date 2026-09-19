"""Paired case-level comparison using saved KG-LCD and CNN test outputs.

The original official test splits and seed-0 predictions are unchanged. This
is an artifact-consistency check: when saved predictions differ from the
manuscript table, its case-level inference does not test the manuscript value.
"""
from __future__ import annotations

import argparse
import json
import math
import hashlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "paired_case_analysis_seed0.json"
BOOTSTRAP_REPLICATES = 20000
RNG_SEED = 20260919
MANUSCRIPT_ACCURACY = {
    "dermamnist": {"KG-LCD": 0.7741, "CNN": 0.7711},
    "bloodmnist": {"KG-LCD": 0.9746, "CNN": 0.9740},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_name(path: Path) -> str:
    """Store portable paths in released analysis metadata."""
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def analyse(dataset: str, rng: np.random.Generator,
            prediction: Path | None = None) -> dict:
    source = DATA / f"{dataset}_seed0_emb.npz"
    if prediction is None:
        prediction = DATA / f"{dataset}_seed0_ours.npz"
    with np.load(source) as original, np.load(prediction) as ours:
        truth = original["test_y"].astype(int)
        cnn_pred = original["test_prob"].argmax(axis=1)
        ours_pred = ours["pred"].astype(int)
        if "y_test" in ours and not np.array_equal(ours["y_test"], truth):
            raise ValueError(f"Test labels disagree for {dataset}")
    reported = json.loads((DATA / f"{dataset}_seed0_results.json").read_text(
        encoding="utf-8"))
    if not (truth.shape == cnn_pred.shape == ours_pred.shape):
        raise ValueError(f"Mismatched case counts for {dataset}")

    ours_correct = ours_pred == truth
    cnn_correct = cnn_pred == truth
    paired_gain = ours_correct.astype(np.int8) - cnn_correct.astype(np.int8)
    improved = int(np.count_nonzero(paired_gain == 1))
    worsened = int(np.count_nonzero(paired_gain == -1))
    discordant = improved + worsened
    if discordant:
        tail_count = sum(math.comb(discordant, k)
                         for k in range(min(improved, worsened) + 1))
        p_value = min(1.0, 2.0 * tail_count / (2 ** discordant))
    else:
        p_value = 1.0

    # Paired resampling keeps each case's two predictions together. Samples
    # are generated in batches so memory usage remains bounded.
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    batch_size = 256
    for start in range(0, BOOTSTRAP_REPLICATES, batch_size):
        stop = min(start + batch_size, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, len(truth), size=(stop - start, len(truth)))
        differences[start:stop] = paired_gain[indices].mean(axis=1)

    labels = sorted(set(truth.tolist()))
    per_class = []
    for label in labels:
        mask = truth == label
        per_class.append({
            "class_index": int(label),
            "n": int(mask.sum()),
            "cnn_recall": float(cnn_correct[mask].mean()),
            "kg_lcd_recall": float(ours_correct[mask].mean()),
        })
    matches_manuscript = round(float(ours_correct.mean()) * 100, 2) == round(
        MANUSCRIPT_ACCURACY[dataset]["KG-LCD"] * 100, 2)
    return {
        "dataset": dataset,
        "manuscript_reference_accuracy_rounded": MANUSCRIPT_ACCURACY[dataset],
        "scope": ("reproduced predictions match the manuscript's rounded accuracy"
                  if matches_manuscript else
                  "stored-prediction diagnostic; not a test of unmatched manuscript values"),
        "source_files": [source_name(source), source_name(prediction)],
        "source_sha256": {source_name(source): sha256(source),
                          source_name(prediction): sha256(prediction)},
        "test_cases": int(len(truth)),
        "cnn_accuracy": float(cnn_correct.mean()),
        "kg_lcd_accuracy": float(ours_correct.mean()),
        "results_json_kg_lcd_accuracy": float(reported["KG-LCD"]["acc"]),
        "prediction_json_accuracy_gap_pp": float(100 * (
            ours_correct.mean() - reported["KG-LCD"]["acc"])),
        "accuracy_difference_pp": float(100 * paired_gain.mean()),
        "paired_bootstrap_95pct_ci_pp": [
            float(x) for x in 100 * np.quantile(differences, [0.025, 0.975])
        ],
        "kg_lcd_correct_cnn_wrong": improved,
        "cnn_correct_kg_lcd_wrong": worsened,
        "exact_mcnemar_two_sided_p": p_value,
        "per_class_recall": per_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--derma-prediction", type=Path)
    parser.add_argument("--blood-prediction", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rng = np.random.default_rng(RNG_SEED)
    predictions = {"dermamnist": args.derma_prediction,
                   "bloodmnist": args.blood_prediction}
    output = {
        "design": "paired analysis of specified seed-0 predictions",
        "reporting_basis": "manuscript main table and specified case-level predictions",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": RNG_SEED,
        "datasets": [analyse(dataset, rng, predictions[dataset])
                     for dataset in ("dermamnist", "bloodmnist")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
