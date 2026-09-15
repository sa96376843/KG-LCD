"""
run_experiments.py
------------------
Full experimental suite for KG-LCD on DermaMNIST / BloodMNIST.

Baselines : CNN, kNN, Label Propagation, GCN (semi-supervised),
            Louvain-community voting, ACL single-layer local community.
Ours      : KG-LCD (multilayer supra-graph PPR local community + KG
            smoothing + posterior fusion).

Outputs   : results JSON files under data/ for downstream figure/table
            generation.
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier

from kglcd import (approx_ppr_push, bloodmnist_kg, build_knn_graph,
                   build_supra_graph, community_scores, conductance_sweep,
                   dermamnist_kg, fuse, kg_smooth)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


# ---------------------------------------------------------------- metrics
def metrics(y_true, y_pred, prob, n_classes):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    try:
        auc = roc_auc_score(y_true, prob, multi_class="ovr",
                            average="macro")
    except Exception:
        auc = float("nan")
    return dict(acc=float(acc), f1=float(f1), auc=float(auc))


def onehot(y, c):
    oh = np.zeros((len(y), c))
    oh[np.arange(len(y)), y] = 1.0
    return oh


# ---------------------------------------------------------------- GCN
class GCN(nn.Module):
    def __init__(self, d_in, d_hid, n_cls):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hid)
        self.w2 = nn.Linear(d_hid, n_cls)

    def forward(self, Ahat, X):
        H = F.relu(Ahat @ self.w1(X))
        return Ahat @ self.w2(H)


def run_gcn(emb_all, y_train_mask_labels, n_train, n_classes,
            epochs=200, seed=0):
    """Two-layer GCN on the kNN graph, semi-supervised (transductive)."""
    torch.manual_seed(seed)
    W = build_knn_graph(emb_all, k=15)
    W = W + sp.eye(W.shape[0])
    d = np.asarray(W.sum(1)).ravel()
    Dm12 = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-10)))
    Ahat = (Dm12 @ W @ Dm12).tocoo()
    idx = torch.tensor(np.vstack([Ahat.row, Ahat.col]), dtype=torch.long)
    val = torch.tensor(Ahat.data, dtype=torch.float32)
    A_sp = torch.sparse_coo_tensor(idx, val, Ahat.shape).coalesce()
    X = torch.tensor(emb_all, dtype=torch.float32)
    yl = torch.tensor(y_train_mask_labels[:n_train], dtype=torch.long)
    model = GCN(emb_all.shape[1], 64, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for ep in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(A_sp, X)
        loss = F.cross_entropy(out[:n_train], yl)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        prob = F.softmax(model(A_sp, X), dim=1).numpy()
    return prob


# ---------------------------------------------------------------- label prop
def run_label_propagation(emb_all, y_masked, n_classes, alpha=0.9, iters=50):
    W = build_knn_graph(emb_all, k=15)
    d = np.asarray(W.sum(1)).ravel()
    S = sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-10))) @ W @ \
        sp.diags(1.0 / np.sqrt(np.maximum(d, 1e-10)))
    Fmat = onehot(np.maximum(y_masked, 0), n_classes)
    Fmat[y_masked < 0] = 0.0
    Y0 = Fmat.copy()
    for _ in range(iters):
        Fmat = alpha * S @ Fmat + (1 - alpha) * Y0
    return Fmat / np.maximum(Fmat.sum(1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- Louvain
def run_louvain_vote(W, y_masked, test_idx, n_classes, seed=0):
    import community as community_louvain
    import networkx as nx
    G = nx.from_scipy_sparse_array(W)
    part = community_louvain.best_partition(G, random_state=seed)
    part_arr = np.array([part[i] for i in range(W.shape[0])])
    comm_label = {}
    lab = np.where(y_masked >= 0)[0]
    for g in np.unique(part_arr):
        members = lab[part_arr[lab] == g]
        if len(members):
            vals, cnts = np.unique(y_masked[members], return_counts=True)
            comm_label[g] = vals[cnts.argmax()]
        else:
            comm_label[g] = -1
    pred = np.array([comm_label.get(part_arr[i], -1) for i in test_idx])
    prob = onehot(np.maximum(pred, 0), n_classes)
    prob[pred < 0] = 1.0 / n_classes
    return pred, prob


# ---------------------------------------------------------------- ours
def _evidence(A_supra, i, y_masked, n_classes, A_kg, alpha, beta, eps,
              keep_sweep=False):
    """Community-restricted evidence for one query node i."""
    p, _, _ = approx_ppr_push(A_supra, int(i), alpha=alpha, eps=eps)
    comm, phi, curve = conductance_sweep(A_supra, p)
    # (1) PPR-weighted evidence over all labelled nodes
    s_ppr = community_scores(p, y_masked, n_classes)
    # (2) uniform majority evidence restricted to the community C*
    lab = np.where(y_masked >= 0)[0]
    members = np.intersect1d(comm, lab)
    s_uni = np.zeros(n_classes)
    if len(members):
        for c in range(n_classes):
            s_uni[c] = np.sum(y_masked[members] == c)
        s_uni /= s_uni.sum()
    else:
        s_uni[:] = 1.0 / n_classes
    s = 0.5 * s_ppr + 0.5 * s_uni
    s = kg_smooth(s, A_kg, n_classes, beta=beta)
    if keep_sweep:
        return s, dict(phi=float(phi), size=int(len(comm)), curve=curve)
    return s


def run_kglcd(emb, y_train, y_val, y_test, prob_backbone_all,
              A_kg, n_classes, k=15, alpha=0.15, omega=1.0, beta=1.0,
              eta=0.5, eps=1e-4, tune_eta=True,
              eta_grid=(0.3, 0.4, 0.5, 0.6, 0.7),
              alpha_grid=(0.05, 0.15), n_rounds=2, fold_val=True,
              verbose=False):
    """Full KG-LCD pipeline. emb/prob stacked as [train|val|test].

    Round 1 tunes (alpha, eta) on the validation split. The soft
    couplings are then refined with the round-1 fused posteriors
    (transductive coupling refinement) and round 2 produces the final
    test predictions. If fold_val, the validation split (with true
    labels) is folded into the labelled set for final inference, in
    the spirit of retraining on train+val after model selection.
    """
    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    y_masked = np.concatenate([y_train, -np.ones(n_val + n_test, dtype=int)])
    W = build_knn_graph(emb, k=k)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)
    unlab_idx = np.concatenate([val_idx, test_idx])

    t0 = time.time()
    # ---------------- round 1: tune (alpha, eta) on validation ----
    A1 = build_supra_graph(W, A_kg, y_masked,
                           prob_backbone_all[n_train:], n_classes,
                           omega=omega)
    alpha_best, eta_best = alpha, eta
    if tune_eta:
        best_acc = -1
        for a in alpha_grid:
            val_scores = np.array([
                _evidence(A1, i, y_masked, n_classes, A_kg, a, beta,
                          eps) for i in val_idx])
            for e in eta_grid:
                pv = e * prob_backbone_all[val_idx] + (1 - e) * val_scores
                acc = (pv.argmax(1) == y_val).mean()
                if acc > best_acc:
                    best_acc, eta_best, alpha_best = acc, e, a
        if verbose:
            print(f"  tuned alpha={alpha_best}, eta={eta_best} "
                  f"(val acc={best_acc:.4f})", flush=True)

    # ---------------- round 1 posteriors on unlabelled nodes ------
    scores1, probs1 = {}, {}
    sweep_info = {}
    for cnt, i in enumerate(unlab_idx):
        if cnt >= n_val and cnt - n_val < 5:
            s, info = _evidence(A1, i, y_masked, n_classes, A_kg,
                                alpha_best, beta, eps, keep_sweep=True)
            sweep_info[int(i)] = info
        else:
            s = _evidence(A1, i, y_masked, n_classes, A_kg, alpha_best,
                          beta, eps)
        scores1[i] = s
        probs1[i] = fuse(prob_backbone_all[i], s, eta=eta_best)

    # ---------------- fold validation labels into the graph -------
    if fold_val and tune_eta:
        y_masked = np.concatenate(
            [y_train, y_val, -np.ones(n_test, dtype=int)])
        unlab_idx = test_idx
        A1 = build_supra_graph(W, A_kg, y_masked,
                               prob_backbone_all[n_train + n_val:],
                               n_classes, omega=omega)
        scores1, probs1 = {}, {}
        for i in test_idx:
            s = _evidence(A1, i, y_masked, n_classes, A_kg, alpha_best,
                          beta, eps)
            scores1[i] = s
            probs1[i] = fuse(prob_backbone_all[i], s, eta=eta_best)

    # ---------------- coupling refinement + round 2 ---------------
    if n_rounds > 1:
        P_refined = np.array([probs1[i] for i in unlab_idx])
        if fold_val and tune_eta:
            A2 = build_supra_graph(W, A_kg, y_masked, P_refined,
                                   n_classes, omega=omega)
        else:
            A2 = build_supra_graph(W, A_kg, y_masked, P_refined,
                                   n_classes, omega=omega)
    else:
        A2 = A1

    preds, probs = [], []
    for i in test_idx:
        s = _evidence(A2, i, y_masked, n_classes, A_kg, alpha_best,
                      beta, eps)
        pf = fuse(prob_backbone_all[i], s, eta=eta_best)
        probs.append(pf)
        preds.append(int(pf.argmax()))
    probs = np.array(probs)
    runtime = time.time() - t0
    return np.array(preds), probs, sweep_info, runtime, A1, W, \
        eta_best, alpha_best


def run_acl_single(emb, y_train, y_val, y_test, prob_backbone_all,
                   n_classes, k=15, alpha=0.15, eta=0.5, eps=1e-4):
    """Single-layer ACL local community voting (no KG layer)."""
    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    y_masked = np.concatenate([y_train, -np.ones(n_val + n_test, dtype=int)])
    W = build_knn_graph(emb, k=k)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)
    preds, probs = [], []
    for i in test_idx:
        p, _, _ = approx_ppr_push(W, int(i), alpha=alpha, eps=eps)
        s = community_scores(p, y_masked, n_classes)
        pf = fuse(prob_backbone_all[i], s, eta=eta)
        probs.append(pf)
        preds.append(int(pf.argmax()))
    return np.array(preds), np.array(probs)


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dermamnist")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    f = np.load(os.path.join(DATA, f"{args.dataset}_seed{args.seed}_emb.npz"))
    emb = np.concatenate([f["train_emb"], f["val_emb"], f["test_emb"]])
    y_train, y_val, y_test = f["train_y"], f["val_y"], f["test_y"]
    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)
    n_classes = int(max(y_train.max(), y_test.max()) + 1)
    # backbone posterior for ALL nodes (transductive soft coupling)
    # recompute: we stored test_prob only; derive val/train prob is not
    # needed exactly -- approximate by stored test prob and uniform-ish
    # for val via stored emb? -> we stored only test_prob, so for val
    # nodes use test-consistent softmax from a kNN posterior estimate.
    test_prob = f["test_prob"]
    # cheap posterior proxy for val: kNN label histogram
    knn = KNeighborsClassifier(n_neighbors=15).fit(f["train_emb"], y_train)
    val_prob = knn.predict_proba(f["val_emb"])
    prob_all = np.vstack([np.zeros((n_train, n_classes)), val_prob,
                          test_prob])
    prob_all[:n_train] = onehot(y_train, n_classes)

    if args.dataset == "dermamnist":
        A_kg, cls_names, kg_edges = dermamnist_kg()
    else:
        A_kg, cls_names, kg_edges = bloodmnist_kg()

    results = {"dataset": args.dataset, "seed": args.seed,
               "n_classes": n_classes, "class_names": cls_names,
               "n_kg_nodes": A_kg.shape[0], "n_kg_edges": len(kg_edges)}

    # ---- baseline: CNN
    cnn_pred = test_prob.argmax(1)
    results["CNN"] = metrics(y_test, cnn_pred, test_prob, n_classes)
    results["CNN"]["acc_backbone_file"] = float(f["test_acc"])

    # ---- baseline: kNN
    knn_pred = knn.predict(f["test_emb"])
    knn_prob = knn.predict_proba(f["test_emb"])
    results["kNN"] = metrics(y_test, knn_pred, knn_prob, n_classes)

    # ---- baseline: label propagation
    y_masked = np.concatenate([y_train, -np.ones(n_val + n_test, dtype=int)])
    lp_prob_all = run_label_propagation(emb, y_masked, n_classes)
    lp_prob = lp_prob_all[n_train + n_val:]
    results["LabelProp"] = metrics(y_test, lp_prob.argmax(1), lp_prob,
                                   n_classes)

    # ---- baseline: GCN
    gcn_prob_all = run_gcn(emb, y_masked, n_train, n_classes,
                           seed=args.seed)
    gcn_prob = gcn_prob_all[n_train + n_val:]
    results["GCN"] = metrics(y_test, gcn_prob.argmax(1), gcn_prob,
                             n_classes)

    # ---- baseline: Louvain community voting
    W = build_knn_graph(emb, k=15)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)
    lv_pred, lv_prob = run_louvain_vote(W, y_masked, test_idx, n_classes,
                                        seed=args.seed)
    results["Louvain"] = metrics(y_test, lv_pred, lv_prob, n_classes)

    # ---- ablation: ACL single layer
    acl_pred, acl_prob = run_acl_single(emb, y_train, y_val, y_test,
                                        prob_all, n_classes)
    results["ACL-single"] = metrics(y_test, acl_pred, acl_prob, n_classes)

    # ---- ours: KG-LCD (hyperparams tuned on val)
    pred, prob, sweep, rt, A_supra, Wv, eta_best, alpha_best = run_kglcd(
        emb, y_train, y_val, y_test, prob_all, A_kg, n_classes,
        verbose=True)
    results["KG-LCD"] = metrics(y_test, pred, prob, n_classes)
    results["KG-LCD"]["runtime_s"] = rt
    results["KG-LCD"]["eta_tuned"] = eta_best
    results["KG-LCD"]["alpha_tuned"] = alpha_best
    np.save(os.path.join(DATA, f"{args.dataset}_seed{args.seed}_sweep.npy"),
            np.array(list(sweep.values()), dtype=object), allow_pickle=True)
    np.savez_compressed(
        os.path.join(DATA, f"{args.dataset}_seed{args.seed}_ours.npz"),
        pred=pred, prob=prob)
    sp.save_npz(os.path.join(DATA,
                             f"{args.dataset}_seed{args.seed}_supra.npz"),
                A_supra)
    sp.save_npz(os.path.join(DATA, f"{args.dataset}_seed{args.seed}_W.npz"),
                Wv)

    # ---- ablations (hyperparams fixed to tuned values)
    # w/o KG smoothing (beta=0)
    p1, pr1, _, _, _, _, _, _ = run_kglcd(
        emb, y_train, y_val, y_test, prob_all, A_kg, n_classes,
        beta=0.0, eta=eta_best, alpha=alpha_best, tune_eta=False)
    results["w/o KG-smooth"] = metrics(y_test, p1, pr1, n_classes)
    # w/o fusion (community evidence only, eta=0)
    p2, pr2, _, _, _, _, _, _ = run_kglcd(
        emb, y_train, y_val, y_test, prob_all, A_kg, n_classes,
        eta=0.0, alpha=alpha_best, tune_eta=False)
    results["w/o fusion"] = metrics(y_test, p2, pr2, n_classes)
    # w/o coupling (omega ~ 0)
    p3, pr3, _, _, _, _, _, _ = run_kglcd(
        emb, y_train, y_val, y_test, prob_all, A_kg, n_classes,
        omega=1e-6, eta=eta_best, alpha=alpha_best, tune_eta=False)
    results["w/o coupling"] = metrics(y_test, p3, pr3, n_classes)
    # w/o coupling refinement (single round)
    p4, pr4, _, _, _, _, _, _ = run_kglcd(
        emb, y_train, y_val, y_test, prob_all, A_kg, n_classes,
        eta=eta_best, alpha=alpha_best, tune_eta=False, n_rounds=1)
    results["w/o refinement"] = metrics(y_test, p4, pr4, n_classes)

    with open(os.path.join(DATA,
              f"{args.dataset}_seed{args.seed}_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
