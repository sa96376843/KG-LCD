"""
make_figures.py
---------------
Generates all figures (PDF, 300 dpi PNG) for the KG-LCD paper.
Run after run_experiments.py / run_analysis.py.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = os.environ.get("KGLCD_DATA", str(ROOT / "data"))
FIG = os.environ.get("KGLCD_FIGURES", str(ROOT / "figures"))
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

C10 = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
       "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


def save(fig, name):
    fig.savefig(os.path.join(FIG, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, f"{name}.png"), bbox_inches="tight",
                dpi=300)
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------- Fig 1
def fig_framework():
    """Install the reviewed GPT Image 2 framework and real vector insets.

    Regenerate via KG-LCD/framework_revision_2026-09-15/README.md.
    The legacy box diagram is retained in that revision's backup folder.
    """
    from pathlib import Path
    from shutil import copy2
    paper = Path(__file__).resolve().parents[1]
    revision = paper.parent / "framework_revision_2026-09-15"
    for source, target in [
        ("KG-LCD_Fig1_framework.pdf", "fig1_framework.pdf"),
        ("KG-LCD_Fig1_framework_600dpi.png", "fig1_framework.png"),
    ]:
        asset = revision / source
        if not asset.is_file():
            raise FileNotFoundError(f"Regenerate reviewed Figure 1: {asset}")
        copy2(asset, paper / "figures" / target)
        print("installed", target)


# ---------------------------------------------------------------- Fig 2
def fig_multilayer_network(ds="dermamnist", seed=0):
    """3D illustration of the supra-network with a detected community."""
    f = np.load(os.path.join(DATA, f"{ds}_seed{seed}_emb.npz"))
    emb = np.concatenate([f["train_emb"], f["val_emb"], f["test_emb"]])
    y = np.concatenate([f["train_y"], f["val_y"], f["test_y"]])
    rng = np.random.RandomState(7)
    n_sub = 220
    idx = rng.choice(len(y), n_sub, replace=False)
    emb2d = TSNE(n_components=2, random_state=1,
                 init="pca").fit_transform(emb[idx])
    ys = y[idx]
    W = sp.load_npz(os.path.join(DATA, f"{ds}_seed{seed}_W.npz"))
    Wsub = W[idx][:, idx]

    A = sp.load_npz(os.path.join(DATA, f"{ds}_seed{seed}_supra.npz"))
    n_img = W.shape[0]
    from kglcd import approx_ppr_push, conductance_sweep
    # recompute community for a fixed test seed for reproducibility
    test_seed = n_img - 100
    p, _, _ = approx_ppr_push(A, test_seed, alpha=0.15)
    comm, phi, _ = conductance_sweep(A, p)
    comm_set = set(comm.tolist())
    # which sub-sampled nodes are in community
    in_comm = np.array([g in comm_set for g in idx])

    fig = plt.figure(figsize=(7.6, 4.4))
    ax = fig.add_subplot(111, projection="3d")
    # visual layer (z=0)
    for i in range(n_sub):
        for j in range(i + 1, n_sub):
            if Wsub[i, j] > 0:
                ax.plot([emb2d[i, 0], emb2d[j, 0]],
                        [emb2d[i, 1], emb2d[j, 1]], [0, 0],
                        color="#BBBBBB", lw=0.25, alpha=0.35, zorder=1)
    for c in np.unique(ys):
        m = ys == c
        ax.scatter(emb2d[m, 0], emb2d[m, 1], np.zeros(m.sum()),
                   s=13 if not np.any(in_comm & m) else 13,
                   c=[C10[c % 10]], alpha=0.55, depthshade=False)
    # community highlight
    ax.scatter(emb2d[in_comm, 0], emb2d[in_comm, 1], np.zeros(in_comm.sum()),
               s=58, facecolors="none", edgecolors="#C0392B", linewidths=1.2,
               depthshade=False, label=f"local community "
               f"$C^{{\\star}}(q)$, $\\Phi$={phi:.3f}")
    # seed
    seed_pos = np.where(idx == test_seed)[0]
    if len(seed_pos):
        ax.scatter(*emb2d[seed_pos[0]], 0, marker="*", s=190,
                   c="#F1C40F", edgecolors="#7D6608", depthshade=False,
                   label="seed query $q$", zorder=5)
    # semantic layer (z = offset) : KG schematic on a circle
    m_nodes = A.shape[0] - n_img
    th = np.linspace(0, 2 * np.pi, m_nodes, endpoint=False)
    kgx = np.cos(th) * 8
    kgy = np.sin(th) * 8
    zoff = np.ptp(emb2d[:, 1]) * 0.5
    Akg = A[n_img:, n_img:].tocoo()
    for i, j, w in zip(Akg.row, Akg.col, Akg.data):
        if i < j:
            ax.plot([kgx[i], kgx[j]], [kgy[i], kgy[j]], [zoff, zoff],
                    color="#8E44AD", lw=0.7, alpha=0.6)
    ax.scatter(kgx, kgy, np.full(m_nodes, zoff), s=42, c="#9B59B6",
               edgecolors="white", depthshade=False,
               label="semantic layer (KG)")
    # coupling edges: community members to their class concept
    ys_all = np.concatenate([f["train_y"], f["val_y"], f["test_y"]])
    shown = 0
    for gpos, gidx in enumerate(idx):
        if in_comm[gpos] and ys_all[gidx] >= 0 and shown < 25:
            c = ys_all[gidx] % m_nodes
            ax.plot([emb2d[gpos, 0], kgx[c]],
                    [emb2d[gpos, 1], kgy[c]], [0, zoff],
                    color="#E67E22", lw=0.5, alpha=0.55)
            shown += 1
    ax.plot([], [], [], color="#E67E22", lw=1.0,
            label="inter-layer coupling ($\\omega$)")
    ax.set_axis_off()
    ax.view_init(elev=18, azim=-60)
    ax.legend(loc="upper left", fontsize=7.2, frameon=False)
    ax.set_title(f"Multilayer supra-network on {ds.replace('mnist','MNIST')}"
                 " with a PPR local community", fontsize=9.5)
    save(fig, "fig2_multilayer")


# ---------------------------------------------------------------- Fig 3
def fig_tsne(ds_list=("dermamnist", "bloodmnist"), seed=0):
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.6))
    for r, ds in enumerate(ds_list):
        f = np.load(os.path.join(DATA, f"{ds}_seed{seed}_emb.npz"))
        names = json.load(open(os.path.join(
            DATA, f"{ds}_seed{seed}_results.json")))["class_names"]
        ours = np.load(os.path.join(DATA, f"{ds}_seed{seed}_ours.npz"))
        X, y = f["test_emb"], f["test_y"]
        pred = ours["pred"]
        pts = TSNE(n_components=2, random_state=42,
                   init="pca").fit_transform(X)
        for c, (ax, yy, ttl) in enumerate(zip(
                axes[r], [y, pred],
                ["Ground-truth labels", "KG-LCD predictions"])):
            for cl in np.unique(y):
                m = yy == cl
                ax.scatter(pts[m, 0], pts[m, 1], s=4, c=[C10[cl % 10]],
                           alpha=0.65, linewidths=0,
                           label=names[cl] if r == 0 else None)
            ax.set_title(f"{'abcdefgh'[r*2+c]}) {ds.replace('mnist','MNIST')}"
                         f" — {ttl}", fontsize=8.6)
            ax.set_xticks([]); ax.set_yticks([])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=7, frameon=False, fontsize=7.0,
               markerscale=2.2, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=[0, 0.035, 1, 1])
    save(fig, "fig3_tsne")


# ---------------------------------------------------------------- Fig 4
def _load_sweep(ds, seed):
    sw = np.load(os.path.join(DATA, f"{ds}_seed{seed}_sweep.npy"),
                 allow_pickle=True)
    if sw.shape == ():
        return sw.item()
    return {i: d for i, d in enumerate(sw.tolist())}


def fig_sweep(ds="dermamnist", seed=0):
    sweep = _load_sweep(ds, seed)
    A = sp.load_npz(os.path.join(DATA, f"{ds}_seed{seed}_supra.npz"))
    from kglcd import approx_ppr_push, conductance_sweep
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3))
    ax = axes[0]
    for k, (q, info) in enumerate(sweep.items()):
        curve = np.array(info["curve"])
        ax.plot(curve[:, 0], curve[:, 1], lw=1.2, c=C10[k % 10],
                label=f"query {q} ($\\Phi^*$={info['phi']:.3f}, "
                      f"$|C^*$={info['size']}$)")
        ax.axvline(info["size"], ls=":", lw=0.8, c=C10[k % 10])
    ax.set_xlabel("Sweep set size $|S_j|$ (PPR/$d$ order)")
    ax.set_ylabel("Conductance $\\Phi(S_j)$")
    ax.set_xscale("log")
    ax.set_title(f"a) Community profile plot — {ds.replace('mnist','MNIST')}")
    ax.legend(fontsize=6.8, frameon=False)

    ax = axes[1]
    q = list(sweep.keys())[0]
    p, _, _ = approx_ppr_push(A, q, alpha=0.15)
    n_img = A.shape[0] - (21 if ds == "dermamnist" else 16)
    dvec = np.asarray(A.sum(0)).ravel()
    order = np.argsort(-p / np.maximum(dvec, 1e-12))[:60]
    vals = p[order] / np.maximum(dvec[order], 1e-12)
    colors = ["#C0392B" if o >= n_img else "#4C72B0" for o in order]
    ax.bar(range(len(vals)), vals, color=colors, width=0.85)
    ax.set_xlabel("Supra-nodes ranked by $\\mathbf{p}_u/d_u$")
    ax.set_ylabel("Degree-normalised PPR")
    ax.set_title("b) PPR mass distribution (red = KG concept nodes)")
    save(fig, "fig4_sweep")


# ---------------------------------------------------------------- Fig 5
def fig_training(ds_list=("dermamnist", "bloodmnist"), seed=0):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1))
    for k, ds in enumerate(ds_list):
        h = np.load(os.path.join(DATA, f"{ds}_seed{seed}_history.npy"))
        ax = axes[k]
        ax.plot(h[:, 0], h[:, 2], "-o", ms=2.6, lw=1.2, c=C10[0],
                label="Train ACC")
        ax.plot(h[:, 0], h[:, 3], "-s", ms=2.6, lw=1.2, c=C10[1],
                label="Val ACC")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{'ab'[k]}) Backbone training — "
                     f"{ds.replace('mnist','MNIST')}")
        ax.legend(frameon=False)
        ax.grid(alpha=0.25)
    # push convergence
    ax = axes[2]
    A = sp.load_npz(os.path.join(DATA, f"dermamnist_seed{seed}_supra.npz"))
    d = np.asarray(A.sum(0)).ravel(); d[d == 0] = 1
    indptr, indices, data = A.indptr, A.indices, A.data
    for k, alpha in enumerate([0.05, 0.15, 0.3]):
        rng = np.random.RandomState(k)
        res_hist = []
        for trial in range(3):
            seed_node = int(rng.randint(A.shape[0]))
            p = np.zeros(A.shape[0]); r = np.zeros(A.shape[0])
            r[seed_node] = 1.0
            stack = [seed_node]
            hist = [1.0]
            while stack and len(hist) < 400:
                u = stack.pop()
                if r[u] < 1e-4 * d[u]:
                    continue
                ru = r[u]; p[u] += alpha * ru
                share = (1 - alpha) * ru / d[u]; r[u] = 0.0
                for jj in range(indptr[u], indptr[u + 1]):
                    v = indices[jj]; r[v] += share * data[jj]
                    if r[v] >= 1e-4 * d[v]:
                        stack.append(v)
                hist.append(r.sum())
            res_hist.append(hist)
        med = [np.median([h[i] for h in res_hist if i < len(h)])
               for i in range(max(map(len, res_hist)))]
        ax.semilogy(med, lw=1.3, c=C10[k], label=f"$\\alpha$={alpha}")
    ax.set_xlabel("Number of push operations")
    ax.set_ylabel("Residual mass $\\|\\mathbf{r}\\|_1$")
    ax.set_title("c) Push-procedure convergence (DermaMNIST)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig5_training")


# ---------------------------------------------------------------- Fig 6
def fig_confusion(ds_list=("dermamnist", "bloodmnist"), seed=0):
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.0))
    for k, ds in enumerate(ds_list):
        res = json.load(open(os.path.join(
            DATA, f"{ds}_seed{seed}_results.json")))
        names = res["class_names"]
        f = np.load(os.path.join(DATA, f"{ds}_seed{seed}_emb.npz"))
        verified_attempt = ("attempt_4_portable" if ds == "dermamnist"
                            else "attempt_2_portable")
        verified = (ROOT / "reproduction_2026-09-19" / ds /
                    verified_attempt / "predictions.npz")
        prediction_path = (verified if seed == 0 and verified.exists() else
                           Path(DATA) / f"{ds}_seed{seed}_ours.npz")
        ours = np.load(prediction_path)
        actual_acc = float(np.mean(f["test_y"] == ours["pred"]))
        if not np.isclose(actual_acc, res["KG-LCD"]["acc"], atol=1e-12):
            raise ValueError(f"{ds} predictions do not match result JSON: "
                             f"{actual_acc} vs {res['KG-LCD']['acc']}")
        cm = confusion_matrix(f["test_y"], ours["pred"], normalize="true")
        ax = axes[k]
        im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(names, fontsize=7)
        for i in range(len(names)):
            for j in range(len(names)):
                if cm[i, j] > 0.35:
                    ax.text(j, i, f"{cm[i,j]:.2f}", ha="center",
                            va="center", fontsize=6.2,
                            color="white" if cm[i, j] > 0.6 else "#1a1a1a")
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
        ax.set_title(f"{'ab'[k]}) {ds.replace('mnist','MNIST')} "
                     f"(ACC={actual_acc:.3f})")
        fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout()
    save(fig, "fig6_confusion")


# ---------------------------------------------------------------- Fig 7
def fig_sensitivity(ds="dermamnist", seed=0):
    s = json.load(open(os.path.join(DATA, f"{ds}_seed{seed}_sensitivity.json")))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.2))
    ax = axes[0]
    ks = sorted(map(float, s["k"].keys()))
    ax.plot(ks, [s["k"][str(int(k))] for k in ks], "-o", c=C10[0], lw=1.4)
    ax.set_xlabel("$k$ (visual-layer neighbours)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("a) Effect of $k$")
    ax.grid(alpha=0.25)
    ax = axes[1]
    als = sorted(map(float, s["alpha"].keys()))
    ax.plot(als, [s["alpha"][str(a)] for a in als], "-s", c=C10[2], lw=1.4,
            label="$\\alpha$")
    oms = sorted(map(float, s["omega"].keys()))
    ax2 = ax.twinx()
    ax2.plot(oms, [s["omega"][str(o)] for o in oms], "-^", c=C10[3], lw=1.4,
             label="$\\omega$")
    ax.set_xlabel("Teleport $\\alpha$  /  coupling $\\omega$")
    ax.set_ylabel("Test accuracy ($\\alpha$ sweep)")
    ax2.set_ylabel("Test accuracy ($\\omega$ sweep)")
    ax.set_title("b) Effect of $\\alpha$ and $\\omega$")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="lower right", fontsize=7)
    ax2.spines["right"].set_visible(True)
    # heatmap
    ax = axes[2]
    a_vals = [0.05, 0.1, 0.15, 0.2, 0.3]
    o_vals = [0.1, 0.5, 1.0, 2.0, 5.0]
    G = np.array([[s["grid"][f"{a},{o}"] for o in o_vals] for a in a_vals])
    im = ax.imshow(G, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(o_vals))); ax.set_xticklabels(o_vals)
    ax.set_yticks(range(len(a_vals))); ax.set_yticklabels(a_vals)
    ax.set_xlabel("Coupling strength $\\omega$")
    ax.set_ylabel("Teleport probability $\\alpha$")
    ax.set_title("c) $(\\alpha,\\omega)$ accuracy surface")
    for i in range(len(a_vals)):
        for j in range(len(o_vals)):
            ax.text(j, i, f"{G[i,j]:.3f}", ha="center", va="center",
                    fontsize=6.4, color="white" if G[i, j] < G.mean()
                    else "black")
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    fig.tight_layout()
    save(fig, "fig7_sensitivity")


# ---------------------------------------------------------------- Fig 8
def fig_noise(ds_list=("dermamnist", "bloodmnist"), seed=0):
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
    style = {"KG-LCD": ("-o", C10[0]), "Community-only": ("-s", C10[2]),
             "GCN": ("-^", C10[3])}
    for k, ds in enumerate(ds_list):
        r = json.load(open(os.path.join(DATA, f"{ds}_seed{seed}_noise.json")))
        ax = axes[k]
        rates = sorted(map(float, r.keys()))
        for meth, (mk, c) in style.items():
            ax.plot(np.array(rates) * 100,
                    [r[str(rt)][meth] for rt in rates], mk, c=c, lw=1.5,
                    ms=4, label=meth)
        ax.set_xlabel("Training-label noise rate (%)")
        ax.set_ylabel("Test accuracy")
        ax.set_title(f"{'ab'[k]}) {ds.replace('mnist','MNIST')}")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=7.4)
    fig.tight_layout()
    save(fig, "fig8_noise")


# ---------------------------------------------------------------- Fig 9
def fig_radar(seed=0):
    methods = ["CNN", "kNN", "LabelProp", "GCN", "Louvain", "ACL-single",
               "KG-LCD"]
    labels = ["DermaMNIST\nACC", "DermaMNIST\nF1", "DermaMNIST\nAUC",
              "BloodMNIST\nACC", "BloodMNIST\nF1", "BloodMNIST\nAUC"]
    vals = {}
    for m in methods:
        row = []
        for ds in ["dermamnist", "bloodmnist"]:
            r = json.load(open(os.path.join(DATA,
                                            f"{ds}_seed{seed}_results.json")))
            row += [r[m]["acc"], r[m]["f1"], r[m]["auc"]]
        vals[m] = row
    ang = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    ang += ang[:1]
    fig = plt.figure(figsize=(5.2, 4.6))
    ax = fig.add_subplot(111, polar=True)
    for k, m in enumerate(methods):
        v = vals[m] + vals[m][:1]
        lw = 2.0 if m == "KG-LCD" else 1.0
        ls = "-" if m == "KG-LCD" else "--"
        ax.plot(ang, v, ls, lw=lw, c=C10[k % 10], label=m,
                marker="o" if m == "KG-LCD" else None, ms=3)
    ax.set_xticks(ang[:-1])
    ax.set_xticklabels(labels, fontsize=7.2)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Multi-metric profile across datasets", y=1.10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.12), fontsize=7,
              frameon=False)
    save(fig, "fig9_radar")


# ---------------------------------------------------------------- Fig 10
def fig_scalability_modularity(ds="dermamnist", seed=0):
    eff = json.load(open(os.path.join(DATA,
                                      f"{ds}_seed{seed}_efficiency.json")))
    mod = json.load(open(os.path.join(DATA,
                                      f"{ds}_seed{seed}_modularity.json")))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    ax = axes[0]
    fr = sorted(map(float, eff.keys()))
    ns = [eff[str(fr_)]["n"] for fr_ in fr]
    tg = [eff[str(fr_)]["graph_s"] for fr_ in fr]
    tp = [eff[str(fr_)]["push_s_per_query"] * 1000 for fr_ in fr]
    ax.loglog(ns, tg, "-o", c=C10[0], lw=1.4, label="supra-graph construction")
    ax.loglog(ns, tp, "-s", c=C10[1], lw=1.4,
              label="PPR push per query (ms)")
    ref = np.array(ns, dtype=float)
    ax.loglog(ns, tg[-1] * (ref / ns[-1]) ** 1.0, ":", c="#888888",
              label="$\\mathcal{O}(n)$ reference")
    ax.set_xlabel("Number of image nodes $n$")
    ax.set_ylabel("Wall-clock time (s / ms)")
    ax.set_title("a) Scalability of KG-LCD components")
    ax.legend(frameon=False, fontsize=7.2)
    ax.grid(alpha=0.25, which="both")
    ax = axes[1]
    names = list(mod.keys())
    v = [mod[k] for k in names]
    bars = ax.bar(range(len(names)), v, color=[C10[0], C10[2], C10[3]],
                  width=0.55, edgecolor="#333")
    for b, vv in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, vv + 0.005, f"{vv:.3f}",
                ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(["Louvain", "Spectral", "KG-LCD\n(label-induced)"],
                       fontsize=8)
    ax.set_ylabel("Multislice modularity $Q_{\\mathrm{multi}}$")
    ax.set_title("b) Community quality on the supra-network")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    save(fig, "fig10_scalability")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all")
    args = ap.parse_args()
    fns = dict(framework=fig_framework, network=fig_multilayer_network,
               tsne=fig_tsne, sweep=fig_sweep, training=fig_training,
               confusion=fig_confusion, sensitivity=fig_sensitivity,
               noise=fig_noise, radar=fig_radar, scalability=fig_scalability_modularity)
    if args.only == "all":
        for name, fn in fns.items():
            try:
                fn()
            except Exception as e:
                print(f"[warn] {name} failed: {e}")
    else:
        fns[args.only]()
