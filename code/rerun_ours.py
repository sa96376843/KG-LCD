"""
rerun_ours.py
-------------
Re-runs ONLY the KG-LCD method and its ablations (with validation
folding) on top of an existing results JSON, leaving all baseline
numbers untouched. Used to apply the fold_val protocol without
recomputing the expensive baselines.
"""
import argparse
import json
import os

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import KNeighborsClassifier

from kglcd import bloodmnist_kg, dermamnist_kg
from run_experiments import DATA, metrics, onehot, run_kglcd


def main(ds, seed):
    f = np.load(os.path.join(DATA, f"{ds}_seed{seed}_emb.npz"))
    emb = np.concatenate([f["train_emb"], f["val_emb"], f["test_emb"]])
    y_train, y_val, y_test = f["train_y"], f["val_y"], f["test_y"]
    n_train = len(y_train)
    n_classes = int(y_train.max() + 1)
    knn = KNeighborsClassifier(n_neighbors=15).fit(f["train_emb"],
                                                   y_train)
    val_prob = knn.predict_proba(f["val_emb"])
    prob_all = np.vstack([onehot(y_train, n_classes), val_prob,
                          f["test_prob"]])
    A_kg, cls, edges = dermamnist_kg() if ds == "dermamnist" \
        else bloodmnist_kg()

    rpath = os.path.join(DATA, f"{ds}_seed{seed}_results.json")
    results = json.load(open(rpath))

    pred, prob, sweep, rt, A1, W, eta_b, alpha_b = run_kglcd(
        emb, y_train, y_val, y_test, prob_all, A_kg, n_classes,
        verbose=True)
    results["KG-LCD"] = metrics(y_test, pred, prob, n_classes)
    results["KG-LCD"]["runtime_s"] = rt
    results["KG-LCD"]["eta_tuned"] = eta_b
    results["KG-LCD"]["alpha_tuned"] = alpha_b
    print("KG-LCD:", results["KG-LCD"], flush=True)

    for name, kw in [
            ("w/o KG-smooth", dict(beta=0.0)),
            ("w/o refinement", dict(n_rounds=1)),
            ("w/o fusion", dict(eta=0.0, tune_eta=False,
                                fold_val=False)),
            ("w/o coupling", dict(omega=1e-6, fold_val=False))]:
        p, pr, _, _, _, _, _, _ = run_kglcd(
            emb, y_train, y_val, y_test, prob_all, A_kg, n_classes,
            eta=kw.pop("eta", eta_b), alpha=alpha_b,
            tune_eta=kw.pop("tune_eta", False),
            fold_val=kw.pop("fold_val", True), **kw)
        results[name] = metrics(y_test, p, pr, n_classes)
        print(name, results[name], flush=True)

    np.save(os.path.join(DATA, f"{ds}_seed{seed}_sweep.npy"),
            sweep, allow_pickle=True)
    np.savez_compressed(
        os.path.join(DATA, f"{ds}_seed{seed}_ours.npz"),
        pred=pred, prob=prob)
    sp.save_npz(os.path.join(DATA, f"{ds}_seed{seed}_supra.npz"), A1)
    with open(rpath, "w") as fh:
        json.dump(results, fh, indent=2)
    print("updated", rpath)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dermamnist")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.dataset, a.seed)
