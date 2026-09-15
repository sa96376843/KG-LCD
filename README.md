# KG-LCD

Official code and released experimental artifacts for:

> **KG-LCD: Knowledge-Graph-guided Multilayer Local Community Detection for Interpretable Medical Image Classification**

KG-LCD constructs a visual k-nearest-neighbor graph over CNN embeddings, couples it to a curated medical knowledge graph, detects a query-centered community using approximate personalized PageRank and a conductance sweep, then fuses community evidence with the CNN posterior.

## Repository contents

```text
code/
  backbone.py          CNN training and embedding export
  kglcd.py             KG-LCD graph construction and inference
  run_experiments.py   baselines, KG-LCD, and ablations
  rerun_ours.py        KG-LCD-only reproduction entry point
  run_analysis.py      sensitivity, noise, efficiency, and modularity
  make_figures.py      figure generation
  make_tables.py       LaTeX table generation
data/
  *_emb.npz            released embeddings, labels, and CNN posteriors
  *_W.npz              visual-layer sparse adjacency matrices
  *_supra.npz          recorded supra-network sparse adjacency matrices
  *_ours.npz           KG-LCD predictions and probabilities
  *_results.json       measured comparison and ablation results
  *_history.npy        backbone training histories
  *_sweep.npy          saved conductance sweeps
  *_noise.json         label-noise results
  *_efficiency.json    efficiency measurements
```

The repository includes the complete released **derived experimental dataset** (about 19 MB). Raw MedMNIST images are distributed by the dataset authors and are not duplicated in this Git repository.

## Raw datasets

Download the official MedMNIST v2 archives from Zenodo record 10519652:

| Dataset | File | MD5 |
|---|---|---|
| DermaMNIST | `dermamnist.npz` | `0744692d530f8e62ec473284d019b0c7` |
| BloodMNIST | `bloodmnist.npz` | `7053d0359d879ad8a5505303e11de1dc` |

```bash
mkdir -p raw_data
curl -L -o raw_data/dermamnist.npz \
  "https://zenodo.org/records/10519652/files/dermamnist.npz?download=1"
curl -L -o raw_data/bloodmnist.npz \
  "https://zenodo.org/records/10519652/files/bloodmnist.npz?download=1"
```

Verify the files against the MD5 values above before running the backbone training step.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce the experiments

All paths are resolved relative to the repository root.

```bash
# Train backbones and export embeddings.
python code/backbone.py --dataset dermamnist --seed 0 --epochs 25 \
  --data_root raw_data
python code/backbone.py --dataset bloodmnist --seed 0 --epochs 25 \
  --data_root raw_data

# Main comparisons and ablations.
python code/run_experiments.py --dataset dermamnist --seed 0
python code/run_experiments.py --dataset bloodmnist --seed 0

# KG-LCD-only rerun using released embeddings.
python code/rerun_ours.py --dataset dermamnist --seed 0
python code/rerun_ours.py --dataset bloodmnist --seed 0

# Analyses.
python code/run_analysis.py --dataset dermamnist --seed 0 --task sensitivity
python code/run_analysis.py --dataset dermamnist --seed 0 --task noise
python code/run_analysis.py --dataset dermamnist --seed 0 --task efficiency
python code/run_analysis.py --dataset dermamnist --seed 0 --task modularity
python code/run_analysis.py --dataset bloodmnist --seed 0 --task noise

# Figures and tables are written to figures/ and tables/.
python code/make_figures.py --only all
python code/make_tables.py
```

CPU-only execution is supported. Backbone training takes substantially longer than rerunning KG-LCD from the released embeddings.

## Knowledge graphs

The dermatology KG contains 21 nodes and 30 typed edges. The hematology KG contains 16 nodes and 29 typed edges. Both are constructed directly in `code/kglcd.py`; relation weights and node definitions are kept with the implementation.

## Data and split integrity

- DermaMNIST and BloodMNIST use the official MedMNIST v2 train, validation, and test splits.
- Released embeddings store train, validation, and test arrays separately.
- Test labels are never used to construct query-node couplings. Unlabelled validation/test nodes use CNN posterior probabilities for soft image-to-class-concept coupling.

## Citation

Please cite the KG-LCD paper when using this code or the released derived artifacts. Bibliographic metadata will be added after publication.
