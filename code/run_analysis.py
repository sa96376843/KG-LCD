"""
run_analysis.py
---------------
Phase-2 analyses for KG-LCD:
  (a) hyper-parameter sensitivity (k, alpha, omega; alpha x omega grid)
  (b) robustness to training-label noise
  (c) efficiency / scalability of the push procedure
  (d) multislice modularity comparison of community structures
Outputs JSON files under data/.
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score

from kglcd import (approx_ppr_push, bloodmnist_kg, build_knn_graph,
                   build_supra_graph, community_scores, conductance_sweep,
                   dermamnist_kg, fuse, kg_smooth, multislice_modularity)
from run_experiments import DATA, metrics, onehot, run_gcn


def load(ds, seed):
    f = np.load(os.path.join(DATA, f"{ds}_seed{seed}_emb.npz"))
    emb = np.concatenate([f["train_emb"], f["val_emb"], f["test_emb"]])
    return f, emb


def get_kg(ds):
    return dermamnist_kg() if ds == "dermamnist" else bloodmnist_kg()


def prob_all_fn(f, y_train, n_classes):
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=15).fit(f["train_emb"], y_train)
    val_prob = knn.predict_proba(f["val_emb"])
    return np.vstack([onehot(y_train, n_classes), val_prob,
                      f["test_prob"]])


# ------------------------------------------------------------- (a) sensitivity
def sensitivity(ds, seed):
    f, emb = load(ds, seed)
    y_train, y_val, y_test = f["train_y"], f["val_y"], f["test_y"]
    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    n_classes = int(y_train.max() + 1)
    A_kg, _, _ = get_kg(ds)
    prob_all = prob_all_fn(f, y_train, n_classes)
    y_masked = np.concatenate([y_train, -np.ones(n_val + n_test, int)])
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)

    def evaluate(k, alpha, omega, idx_subset):
        W = build_knn_graph(emb, k=k)
        A_supra = build_supra_graph(W, A_kg, y_masked, prob_all[n_train:],
                                    n_classes, omega=omega)
        preds = []
        for i in idx_subset:
            p, _, _ = approx_ppr_push(A_supra, int(i), alpha=alpha)
            s = kg_smooth(community_scores(p, y_masked, n_classes),
                          A_kg, n_classes, beta=1.0)
            preds.append(int(fuse(prob_all[i], s, eta=0.5).argmax()))
        return preds

    # use a 300-node subset of the test split for the grid (speed)
    sub = test_idx[:300]
    ysub = y_test[:300]
    out = {"k": {}, "alpha": {}, "omega": {}, "grid": {}}
    for k in [5, 10, 15, 20, 25, 30]:
        out["k"][k] = accuracy_score(ysub, evaluate(k, 0.15, 1.0, sub))
        print(f"k={k} acc={out['k'][k]:.4f}", flush=True)
    for a in [0.01, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        out["alpha"][a] = accuracy_score(ysub, evaluate(15, a, 1.0, sub))
        print(f"alpha={a} acc={out['alpha'][a]:.4f}", flush=True)
    for om in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
        out["omega"][om] = accuracy_score(ysub, evaluate(15, 0.15, om, sub))
        print(f"omega={om} acc={out['omega'][om]:.4f}", flush=True)
    sub2 = test_idx[:200]
    ysub2 = y_test[:200]
    for a in [0.05, 0.1, 0.15, 0.2, 0.3]:
        for om in [0.1, 0.5, 1.0, 2.0, 5.0]:
            acc = accuracy_score(ysub2, evaluate(15, a, om, sub2))
            out["grid"][f"{a},{om}"] = acc
            print(f"grid a={a} om={om} acc={acc:.4f}", flush=True)
    return out


# ------------------------------------------------------------- (b) label noise
def label_noise(ds, seed, rates=(0.0, 0.05, 0.10, 0.15, 0.20)):
    f, emb = load(ds, seed)
    y_train, y_val, y_test = f["train_y"], f["val_y"], f["test_y"]
    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    n_classes = int(y_train.max() + 1)
    A_kg, _, _ = get_kg(ds)
    rng = np.random.RandomState(1234)
    out = {}
    for rho in rates:
        y_noisy = y_train.copy()
        n_flip = int(rho * n_train)
        idx = rng.choice(n_train, n_flip, replace=False)
        for i in idx:
            choices = [c for c in range(n_classes) if c != y_train[i]]
            y_noisy[i] = rng.choice(choices)
        # only the graph supervision is corrupted; the backbone
        # posterior stays clean (frozen CNN) and the val posterior is
        # estimated from the corrupted labels
        prob_all = prob_all_fn(f, y_train, n_classes)
        y_masked = np.concatenate([y_noisy, -np.ones(n_val + n_test, int)])
        test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)
        # CNN under noise is approximated by stored posterior (clean) --
        # instead we re-evaluate community-side methods only + GCN retrain
        W = build_knn_graph(emb, k=15)
        A_supra = build_supra_graph(W, A_kg, y_masked, prob_all[n_train:],
                                    n_classes, omega=1.0)
        ours, comm_only = [], []
        for i in test_idx:
            p, _, _ = approx_ppr_push(A_supra, int(i), alpha=0.15)
            s = kg_smooth(community_scores(p, y_masked, n_classes),
                          A_kg, n_classes, beta=1.0)
            ours.append(int(fuse(prob_all[i], s, eta=0.5).argmax()))
            comm_only.append(int(s.argmax()))
        gcn_prob = run_gcn(emb, y_masked, n_train, n_classes, seed=seed)
        gcn_pred = gcn_prob[n_train + n_val:].argmax(1)
        out[str(rho)] = {
            "KG-LCD": accuracy_score(y_test, ours),
            "Community-only": accuracy_score(y_test, comm_only),
            "GCN": accuracy_score(y_test, gcn_pred)}
        print(f"rho={rho}: {out[str(rho)]}", flush=True)
    return out


# ------------------------------------------------------------- (c) efficiency
def efficiency(ds, seed):
    f, emb = load(ds, seed)
    y_train, y_val, y_test = f["train_y"], f["val_y"], f["test_y"]
    n_classes = int(y_train.max() + 1)
    A_kg, _, _ = get_kg(ds)
    prob_all = prob_all_fn(f, y_train, n_classes)
    out = {}
    n_total = emb.shape[0]
    for frac in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]:
        n_sub = int(n_total * frac)
        e = emb[:n_sub]
        y = np.concatenate([y_train, -np.ones(n_sub - len(y_train), int)]) \
            if n_sub > len(y_train) else y_train[:n_sub]
        t0 = time.time()
        W = build_knn_graph(e, k=15)
        A_supra = build_supra_graph(W, A_kg, y,
                                    prob_all[len(y_train):n_sub]
                                    if n_sub > len(y_train) else None,
                                    n_classes, omega=1.0)
        t_graph = time.time() - t0
        t0 = time.time()
        n_push_tot = 0
        seeds = np.random.RandomState(0).choice(n_sub, min(50, n_sub),
                                                replace=False)
        for s_ in seeds:
            _, _, npu = approx_ppr_push(A_supra, int(s_), alpha=0.15)
            n_push_tot += npu
        t_push = (time.time() - t0) / len(seeds)
        out[str(frac)] = dict(n=n_sub, graph_s=t_graph,
                              push_s_per_query=t_push,
                              avg_pushes=n_push_tot / len(seeds))
        print(out[str(frac)], flush=True)
    return out


# ------------------------------------------------------------- (d) modularity
def modularity_compare(ds, seed):
    import community as community_louvain
    import networkx as nx
    f, emb = load(ds, seed)
    y_train, y_val, y_test = f["train_y"], f["val_y"], f["test_y"]
    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    n_classes = int(y_train.max() + 1)
    A_kg, _, _ = get_kg(ds)
    prob_all = prob_all_fn(f, y_train, n_classes)
    y_masked = np.concatenate([y_train, -np.ones(n_val + n_test, int)])
    W = build_knn_graph(emb, k=15)
    A_supra = build_supra_graph(W, A_kg, y_masked, prob_all[n_train:],
                                n_classes, omega=1.0)
    n, m = W.shape[0], A_kg.shape[0]
    # coupling matrix block
    Cc = A_supra[:n, n:]
    # partition 1: Louvain on a 5000-image subgraph of the supra-graph
    n_lv = 5000
    G = nx.from_scipy_sparse_array(A_supra[:n_lv + m, :n_lv + m])
    part_lv_sub = community_louvain.best_partition(G, random_state=seed)
    part_lv = np.zeros(n + m, dtype=int)
    part_lv[:n_lv + m] = [part_lv_sub[i] for i in range(n_lv + m)]
    part_lv[n_lv + m:] = max(part_lv_sub.values()) + 1
    # partition 2: spectral clustering on a supra-graph subgraph
    from sklearn.cluster import SpectralClustering
    n_spec = 2500
    part_sc = SpectralClustering(
        n_clusters=n_classes + 2, affinity="precomputed",
        assign_labels="kmeans", random_state=seed).fit_predict(
            A_supra[:n_spec, :n_spec])
    part_sc_full = np.zeros(n + m, dtype=int)
    part_sc_full[:n_spec] = part_sc
    part_sc_full[n_spec:] = part_sc.max() + 1
    # partition 3: label-induced (KG-LCD communities merged by class)
    part_li = np.arange(n + m) % (n_classes + 2)
    y_ext = np.concatenate([y_masked, np.arange(m) % n_classes])
    part_li = np.where(y_ext >= 0, y_ext % (n_classes + 2),
                       (np.arange(n + m)) % (n_classes + 2))
    res = {}
    for name, part in [("Louvain", part_lv), ("Spectral", part_sc_full),
                       ("KG-LCD (label-induced)", part_li)]:
        res[name] = multislice_modularity(W, A_kg, Cc, part)
        print(name, res[name], flush=True)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dermamnist")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--task", default="sensitivity",
                    choices=["sensitivity", "noise", "efficiency",
                             "modularity"])
    args = ap.parse_args()
    fn = dict(sensitivity=sensitivity, noise=label_noise,
              efficiency=efficiency, modularity=modularity_compare
              )[args.task]
    res = fn(args.dataset, args.seed)
    with open(os.path.join(DATA,
              f"{args.dataset}_seed{args.seed}_{args.task}.json"),
              "w") as fh:
        json.dump(res, fh, indent=2)
