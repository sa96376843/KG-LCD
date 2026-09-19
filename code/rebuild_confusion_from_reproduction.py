"""Regenerate manuscript Figure 6 from verified seed-0 test predictions.

The original saved predictions are retained. This script verifies that both
reproduced test accuracies match the corresponding manuscript result JSON
before writing a new figure into the reproduction folder.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPRO = ROOT / "reproduction_2026-09-19"
SOURCES = {
    "dermamnist": REPRO / "dermamnist" / "attempt_4_portable" / "predictions.npz",
    "bloodmnist": REPRO / "bloodmnist" / "attempt_2_portable" / "predictions.npz",
}


def main() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 300,
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    audit = {}
    for k, dataset in enumerate(SOURCES):
        reported = json.loads((DATA / f"{dataset}_seed0_results.json").read_text(
            encoding="utf-8"))
        with np.load(DATA / f"{dataset}_seed0_emb.npz") as source, \
                np.load(SOURCES[dataset]) as prediction:
            truth = source["test_y"]
            pred = prediction["pred"]
            if not np.array_equal(truth, prediction["y_test"]):
                raise ValueError(f"Test labels differ for {dataset}")
            if not np.array_equal(pred, prediction["prob"].argmax(axis=1)):
                raise ValueError(f"Prediction/probability mismatch for {dataset}")
            accuracy = float((truth == pred).mean())
            if not np.isclose(accuracy, reported["KG-LCD"]["acc"], atol=1e-12):
                raise ValueError(f"Reproduction differs from manuscript JSON: {dataset}")
            names = reported["class_names"]
            cm = confusion_matrix(truth, pred, normalize="true")
            audit[dataset] = {"n": int(len(truth)), "accuracy": accuracy,
                              "matrix_row_sums": cm.sum(axis=1).tolist(),
                              "prediction_file": SOURCES[dataset].relative_to(ROOT).as_posix()}
        ax = axes[k]
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                if cm[i, j] > 0.35:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center",
                            va="center", fontsize=6.2,
                            color="white" if cm[i, j] > 0.6 else "#1a1a1a")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"{'ab'[k]}) {dataset.replace('mnist', 'MNIST')} "
                     f"(ACC={accuracy:.3f})")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout()
    for suffix, options in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(REPRO / f"fig6_confusion_verified.{suffix}",
                    bbox_inches="tight", **options)
    plt.close(fig)
    (REPRO / "fig6_confusion_verified_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
