"""
kglcd.py
--------
KG-LCD: Knowledge-Graph-guided Multilayer Local Community Detection
for medical image classification.

Pipeline
--------
1. Visual layer  : self-tuning Gaussian kNN graph over CNN embeddings.
2. Semantic layer: disease/cell-lineage knowledge graph (KG).
3. Supra-graph   : intra-layer blocks + inter-layer couplings
                   (hard coupling for labelled images, soft coupling
                   via backbone posterior for unlabelled images).
4. Local community detection: approximate personalized-PageRank push
                   (Andersen-Chung-Lang) + conductance sweep on the
                   supra-graph.
5. Prediction    : community label evidence, KG-smoothed and fused
                   with the backbone posterior.

All functions are numpy/scipy based and CPU-friendly.
"""
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


# ============================================================ visual layer
def build_knn_graph(emb, k=15):
    """Self-tuning Gaussian kNN affinity graph (Zelnik-Manor & Perona).

    W_ij = exp(-||z_i - z_j||^2 / (sigma_i * sigma_j)) for j in N_k(i),
    symmetrised by W = max(W, W^T). Diagonal is zero.
    """
    n = emb.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1).fit(emb)
    dist, idx = nn.kneighbors(emb)
    dist, idx = dist[:, 1:], idx[:, 1:]          # drop self
    sigma = dist[:, -1] + 1e-10                 # distance to k-th nn
    rows = np.repeat(np.arange(n), k)
    cols = idx.ravel()
    dij2 = dist.ravel() ** 2
    sig = sigma[rows] * sigma[cols]
    w = np.exp(-dij2 / np.maximum(sig, 1e-12))
    W = sp.csr_matrix((w, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)
    return W


# ============================================================ KG layer
def dermamnist_kg():
    """Dermatology KG for the 7 HAM10000/DermaMNIST classes.

    Nodes: 7 disease classes + 2 lineage groups + 2 malignancy groups
           + 8 dermatoscopic attribute concepts.
    Edges: (src, dst, relation, weight). Relations:
           is_a(1.0), has_attribute(0.6), differential_diagnosis(0.4),
           shares_lineage via group nodes.
    Class index: 0 akiec, 1 bcc, 2 bkl, 3 df, 4 mel, 5 nv, 6 vasc
    """
    cls = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    groups = {"keratinocytic": 7, "melanocytic": 8, "vascular_origin": 9,
              "malignant": 10, "premalignant": 11, "benign": 12}
    attr = {"pigment_network": 13, "streaks": 14, "blue_white_veil": 15,
            "ulceration": 16, "arborizing_vessels": 17,
            "comedo_like_openings": 18, "milium_like_cysts": 19,
            "red_lacunae": 20}
    edges = []
    # taxonomy (is_a)
    for c in ["akiec", "bcc", "bkl"]:
        edges.append((cls.index(c), groups["keratinocytic"], "is_a", 1.0))
    for c in ["mel", "nv"]:
        edges.append((cls.index(c), groups["melanocytic"], "is_a", 1.0))
    edges.append((cls.index("vasc"), groups["vascular_origin"], "is_a", 1.0))
    edges.append((cls.index("df"), groups["benign"], "is_a", 1.0))
    # malignant potential
    for c in ["mel", "bcc"]:
        edges.append((cls.index(c), groups["malignant"], "is_a", 1.0))
    edges.append((cls.index("akiec"), groups["premalignant"], "is_a", 1.0))
    for c in ["bkl", "nv", "vasc"]:
        edges.append((cls.index(c), groups["benign"], "is_a", 1.0))
    # attributes (has_attribute)
    def A(c, a):
        edges.append((cls.index(c), attr[a], "has_attribute", 0.6))
    A("mel", "pigment_network"); A("mel", "streaks"); A("mel", "blue_white_veil")
    A("nv", "pigment_network"); A("bkl", "comedo_like_openings")
    A("bkl", "milium_like_cysts"); A("bcc", "arborizing_vessels")
    A("bcc", "ulceration"); A("vasc", "red_lacunae")
    A("akiec", "streaks")
    # clinically confusable pairs (differential_diagnosis)
    for a, b in [("bkl", "mel"), ("bkl", "nv"), ("nv", "mel"),
                 ("akiec", "bcc"), ("akiec", "bkl"), ("bcc", "df"),
                 ("vasc", "mel")]:
        edges.append((cls.index(a), cls.index(b),
                      "differential_diagnosis", 0.4))
    return _kg_to_matrix(edges, n_nodes=21), cls, edges


def bloodmnist_kg():
    """Haematology lineage KG for the 8 BloodMNIST cell types.

    Class index: 0 basophil, 1 eosinophil, 2 erythroblast, 3 IG
                 (immature granulocytes), 4 lymphocyte, 5 monocyte,
                 6 neutrophil, 7 platelet
    """
    cls = ["basophil", "eosinophil", "erythroblast", "ig",
           "lymphocyte", "monocyte", "neutrophil", "platelet"]
    grp = {"myeloid": 8, "lymphoid": 9, "erythroid": 10,
           "megakaryocytic": 11, "granulocytic": 12,
           "agranulocytic": 13, "mature": 14, "immature": 15}
    edges = []
    for c in ["basophil", "eosinophil", "neutrophil", "monocyte"]:
        edges.append((cls.index(c), grp["myeloid"], "is_a", 1.0))
    edges.append((cls.index("lymphocyte"), grp["lymphoid"], "is_a", 1.0))
    edges.append((cls.index("erythroblast"), grp["erythroid"], "is_a", 1.0))
    edges.append((cls.index("platelet"), grp["megakaryocytic"], "is_a", 1.0))
    for c in ["basophil", "eosinophil", "neutrophil", "ig"]:
        edges.append((cls.index(c), grp["granulocytic"], "is_a", 1.0))
    for c in ["lymphocyte", "monocyte"]:
        edges.append((cls.index(c), grp["agranulocytic"], "is_a", 1.0))
    for c in ["basophil", "eosinophil", "neutrophil", "lymphocyte",
              "monocyte", "platelet"]:
        edges.append((cls.index(c), grp["mature"], "is_a", 1.0))
    for c in ["ig", "erythroblast"]:
        edges.append((cls.index(c), grp["immature"], "is_a", 1.0))
    # maturation adjacency (develops_into)
    edges.append((cls.index("ig"), cls.index("neutrophil"),
                  "develops_into", 0.5))
    # morphologically similar (confusable) pairs
    for a, b in [("basophil", "neutrophil"), ("eosinophil", "neutrophil"),
                 ("lymphocyte", "monocyte"), ("ig", "neutrophil"),
                 ("erythroblast", "lymphocyte")]:
        edges.append((cls.index(a), cls.index(b),
                      "similar_morphology", 0.4))
    return _kg_to_matrix(edges, n_nodes=16), cls, edges


def _kg_to_matrix(edges, n_nodes):
    rows, cols, w = [], [], []
    for s, d, _, wt in edges:
        rows += [s, d]; cols += [d, s]; w += [wt, wt]
    return sp.csr_matrix((w, (rows, cols)), shape=(n_nodes, n_nodes))


# ============================================================ supra-graph
def build_supra_graph(W_vis, A_kg, y_train, prob_unlab, n_classes,
                      omega=1.0, soft=True):
    """Supra-adjacency of the two-layer network.

    Image nodes 0..n-1 (layer V), concept nodes n..n+m-1 (layer K).
    Labelled image i  --(omega)--> its class concept (hard coupling).
    Unlabelled image i --(omega * p_ic)--> every class concept (soft
    coupling through the backbone posterior).
    """
    n = W_vis.shape[0]
    m = A_kg.shape[0]
    r, c, w = [], [], []
    y_arr = np.asarray(y_train)
    for i in np.where(y_arr >= 0)[0]:           # hard coupling (labelled)
        r += [i, n + int(y_arr[i])]
        c += [n + int(y_arr[i]), i]
        w += [omega, omega]
    if soft and prob_unlab is not None:         # soft coupling (unlabelled)
        unlab_idx = np.where(y_arr < 0)[0]
        P = np.asarray(prob_unlab)              # len(unlab) x C
        for row, i in enumerate(unlab_idx):
            for cc in range(P.shape[1]):
                wij = omega * float(P[row, cc])
                if wij > 1e-6:
                    r += [i, n + cc]; c += [n + cc, i]; w += [wij, wij]
    C = sp.csr_matrix((w, (r, c)), shape=(n + m, n + m))
    A_supra = sp.bmat([[W_vis, None], [None, A_kg]], format="csr") + C
    return A_supra


# ============================================================ PPR push
def approx_ppr_push(A, seed, alpha=0.15, eps=1e-4):
    """Approximate personalised PageRank via the ACL push procedure.

    Solves  p = alpha * s + (1 - alpha) * P^T p  with column-stochastic
    P = A D^{-1}; returns (p, residual_bound, n_push).
    """
    n = A.shape[0]
    d = np.asarray(A.sum(axis=0)).ravel()
    d[d == 0] = 1.0
    p = np.zeros(n)
    r = np.zeros(n)
    r[seed] = 1.0
    stack = [seed]
    n_push = 0
    indptr, indices, data = A.indptr, A.indices, A.data
    thr = eps * d
    while stack:
        u = stack.pop()
        if r[u] < thr[u]:
            continue
        ru = r[u]
        p[u] += alpha * ru
        # distribute (1-alpha)*r_u/d_u to out-neighbours (column u)
        share = (1 - alpha) * ru / d[u]
        r[u] = 0.0
        st, en = indptr[u], indptr[u + 1]
        if en > st:
            nbr = indices[st:en]
            np.add.at(r, nbr, share * data[st:en])
            stack.extend(nbr[r[nbr] >= thr[nbr]].tolist())
        n_push += 1
    return p, float(r.sum()), n_push


def conductance_sweep(A, p):
    """Sweep over p/d-ordered prefixes; return (best_set, best_phi, curve).

    Vectorised implementation: for the PPR/d ordering, the internal
    weight accumulated at position j equals the row-sum of the strict
    lower triangle of A[order][:, order], so the whole sweep costs
    O(nnz) instead of O(vol) Python iterations.
    """
    d = np.asarray(A.sum(axis=0)).ravel()
    order = np.argsort(-p / np.maximum(d, 1e-12))
    order = order[p[order] > 0]
    if len(order) == 0:
        return order, 1.0, []
    B = A[order][:, order].tocsr()
    lower = sp.tril(B, k=-1)
    w_in = np.asarray(lower.sum(axis=1)).ravel()
    vol_prefix = np.cumsum(d[order])
    cut = vol_prefix - 2.0 * np.cumsum(w_in)
    vol_total = d.sum()
    denom = np.minimum(vol_prefix, vol_total - vol_prefix)
    denom[denom <= 0] = np.inf
    phi = cut / denom
    phi[denom == np.inf] = 1.0
    best_j = int(np.argmin(phi)) + 1
    best_phi = float(phi[best_j - 1])
    curve = [(j + 1, float(phi[j])) for j in range(len(order))]
    return order[:best_j], best_phi, curve


# ============================================================ inference
def community_scores(p, y_train, n_classes):
    """PPR-weighted label evidence over labelled image nodes."""
    lab = np.where(y_train >= 0)[0]
    w = p[lab]
    s = np.zeros(n_classes)
    for c in range(n_classes):
        s[c] = w[y_train[lab] == c].sum()
    if s.sum() > 0:
        s /= s.sum()
    return s


def kg_smooth(scores, A_kg, n_classes, beta=1.0):
    """Knowledge propagation through the class-concept subgraph."""
    K = A_kg[:n_classes, :n_classes].toarray()
    boost = np.exp(beta * K @ scores)
    out = scores * boost
    return out / out.sum() if out.sum() > 0 else out


def fuse(p_backbone, s_comm, eta=0.5):
    out = eta * p_backbone + (1 - eta) * s_comm
    return out / out.sum()


# ============================================================ modularity
def multislice_modularity(A_vis, A_kg, C_couple, partition, gamma=1.0):
    """Mucha et al. (2010) multislice modularity of a partition given as
    (layer, community) for every supra-node. Blocks are supplied as CSR."""
    Av = A_vis.tocsr(); Ak = A_kg.tocsr()
    kv = np.asarray(Av.sum(1)).ravel(); kk = np.asarray(Ak.sum(1)).ravel()
    mv = kv.sum() / 2.0; mk = kk.sum() / 2.0
    omega_sum = C_couple.sum()
    mu = mv + mk + omega_sum
    if mu <= 0:
        return 0.0
    nv = Av.shape[0]
    part = np.asarray(partition)
    Q = 0.0
    # intra-layer visual
    for g in np.unique(part):
        nodes = np.where(part == g)[0]
        v_nodes = nodes[nodes < nv]
        k_nodes = nodes[nodes >= nv] - nv
        if len(v_nodes):
            sub = Av[v_nodes][:, v_nodes]
            Q += sub.sum() - gamma * kv[v_nodes].sum() ** 2 / (2 * mv)
        if len(k_nodes):
            sub = Ak[k_nodes][:, k_nodes]
            Q += sub.sum() - gamma * kk[k_nodes].sum() ** 2 / (2 * mk)
    # inter-layer coupling
    Cc = C_couple.tocoo()
    for i, j, w in zip(Cc.row, Cc.col, Cc.data):
        if i < nv <= j or j < nv <= i:
            if part[i] == part[j]:
                Q += w
    return Q / (2 * mu)
