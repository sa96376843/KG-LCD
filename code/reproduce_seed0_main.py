"""Reproduce the seed-0 KG-LCD main result without overwriting old outputs.

Uses the saved backbone embeddings and posteriors with the existing
run_experiments.run_kglcd implementation and its default protocol.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from sklearn.neighbors import KNeighborsClassifier

from kglcd import bloodmnist_kg, dermamnist_kg
import run_experiments
from run_experiments import metrics, onehot, run_kglcd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_ROOT = ROOT / "reproduction_2026-09-19"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main(dataset: str, attempt: str) -> None:
    out = OUT_ROOT / dataset / attempt
    out.mkdir(parents=True, exist_ok=True)
    input_path = DATA / f"{dataset}_seed0_emb.npz"
    with np.load(input_path) as source:
        y_train = source["train_y"].astype(int)
        y_val = source["val_y"].astype(int)
        y_test = source["test_y"].astype(int)
        train_emb = source["train_emb"]
        val_emb = source["val_emb"]
        test_emb = source["test_emb"]
        test_prob = source["test_prob"]
    emb = np.concatenate((train_emb, val_emb, test_emb))
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    knn = KNeighborsClassifier(n_neighbors=15).fit(train_emb, y_train)
    val_prob = knn.predict_proba(val_emb)
    prob_all = np.vstack((onehot(y_train, n_classes), val_prob, test_prob))
    kg, _, _ = (dermamnist_kg() if dataset == "dermamnist" else bloodmnist_kg())

    source_files = [input_path, Path(__file__), ROOT / "code" / "kglcd.py",
                    ROOT / "code" / "run_experiments.py"]
    manifest = {
        "status": "running",
        "dataset": dataset,
        "attempt": attempt,
        "seed": 0,
        "protocol": "run_kglcd defaults: validation-tuned alpha and eta, fold_val=True, two rounds",
        "split_sizes": [len(y_train), len(y_val), len(y_test)],
        "n_classes": n_classes,
        "input_sha256": {p.name: sha256(p) for p in source_files},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "started_unix": time.time(),
    }
    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Started {dataset}; outputs: {out}", flush=True)

    progress_path = out / "progress.log"
    crash_path = out / "faulthandler.log"
    evidence_original = run_experiments._evidence
    evidence_calls = 0

    def evidence_with_progress(*args, **kwargs):
        nonlocal evidence_calls
        result = evidence_original(*args, **kwargs)
        evidence_calls += 1
        if evidence_calls % 100 == 0:
            line = f"{time.time():.3f} evidence_calls={evidence_calls}\n"
            with progress_path.open("a", encoding="utf-8") as stream:
                stream.write(line)
            print(line.strip(), flush=True)
        return result

    run_experiments._evidence = evidence_with_progress
    crash_log = crash_path.open("w", encoding="utf-8")
    faulthandler.enable(file=crash_log, all_threads=True)

    try:
        pred, prob, _, runtime, _, _, eta, alpha = run_kglcd(
            emb, y_train, y_val, y_test, prob_all, kg, n_classes,
            verbose=True)
        if not np.array_equal(pred, prob.argmax(axis=1)):
            raise ValueError("Predicted labels disagree with saved probability argmax")
        result = metrics(y_test, pred, prob, n_classes)
        result.update({"eta_tuned": float(eta), "alpha_tuned": float(alpha),
                       "runtime_s": float(runtime)})
        pred_path = out / "predictions.npz"
        np.savez_compressed(pred_path, pred=pred, prob=prob, y_test=y_test)
        manifest.update({"status": "complete", "result": result,
                         "predictions_sha256": sha256(pred_path),
                         "finished_unix": time.time()})
        print(json.dumps(result, indent=2), flush=True)
    except Exception as exc:
        manifest.update({"status": "failed", "error": repr(exc),
                         "finished_unix": time.time()})
        raise
    finally:
        faulthandler.disable()
        crash_log.close()
        run_experiments._evidence = evidence_original
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n",
                                 encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True,
                        choices=("dermamnist", "bloodmnist"))
    parser.add_argument("--attempt", default="attempt_1")
    args = parser.parse_args()
    main(args.dataset, args.attempt)
