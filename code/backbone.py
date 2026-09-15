"""
backbone.py
-----------
Lightweight CNN backbone (MedTinyNet) for feature extraction on
MedMNIST datasets (DermaMNIST / BloodMNIST), 28x28 RGB images.

Trains the backbone on the official train split, evaluates on the
official test split (baseline 'CNN' row in Table 3), and exports
128-d penultimate embeddings for train/val/test splits.

Usage:
    python backbone.py --dataset dermamnist --seed 0 --epochs 25
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------- model
class MedTinyNet(nn.Module):
    """Compact CNN for 28x28 medical images (~0.35M parameters)."""

    def __init__(self, num_classes, emb_dim=128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 14x14
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 7x7
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embed = nn.Linear(128, emb_dim)
        self.classifier = nn.Linear(emb_dim, num_classes)

    def forward(self, x, return_emb=False):
        h = self.features(x).flatten(1)
        z = F.relu(self.embed(h))
        out = self.classifier(z)
        if return_emb:
            return out, z
        return out


# ---------------------------------------------------------------- data
def load_npz(path):
    d = np.load(path)
    splits = {}
    for sp in ["train", "val", "test"]:
        x = d[f"{sp}_images"].astype(np.float32) / 255.0
        x = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous()
        y = torch.from_numpy(d[f"{sp}_labels"].ravel().astype(np.int64))
        splits[sp] = (x, y)
    return splits


def augment(x):
    """Random flips and 90-degree rotations (medical-image safe)."""
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[3])
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[2])
    k = int(torch.randint(0, 4, (1,)))
    if k:
        x = torch.rot90(x, k, dims=[2, 3])
    return x


# ---------------------------------------------------------------- train
def train_one(dataset, seed, epochs, data_root, out_root, lr=1e-3, bs=128):
    torch.manual_seed(seed)
    np.random.seed(seed)
    splits = load_npz(os.path.join(data_root, f"{dataset}.npz"))
    xtr, ytr = splits["train"]
    xva, yva = splits["val"]
    xte, yte = splits["test"]
    num_classes = int(ytr.max().item() + 1)

    model = MedTinyNet(num_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    n = len(ytr)
    history = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        ep_loss, ep_correct = 0.0, 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = augment(xtr[idx]), ytr[idx]
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
            ep_correct += (out.argmax(1) == yb).sum().item()
        sched.step()
        # validation
        model.eval()
        with torch.no_grad():
            va_acc = (model(xva).argmax(1) == yva).float().mean().item()
        history.append(dict(epoch=ep + 1, train_loss=ep_loss / n,
                            train_acc=ep_correct / n, val_acc=va_acc))
        print(f"[{dataset} s{seed}] epoch {ep+1:02d}/{epochs} "
              f"loss={ep_loss/n:.4f} train_acc={ep_correct/n:.4f} "
              f"val_acc={va_acc:.4f} ({time.time()-t0:.0f}s)", flush=True)

    # test accuracy (baseline CNN row)
    model.eval()
    with torch.no_grad():
        logits = model(xte)
        te_acc = (logits.argmax(1) == yte).float().mean().item()
        prob = F.softmax(logits, dim=1).numpy()

    # export embeddings
    embs = {}
    for sp, (x, y) in splits.items():
        with torch.no_grad():
            z_all = []
            for i in range(0, len(y), 512):
                z_all.append(model(x[i:i+512], return_emb=True)[1])
        embs[sp] = torch.cat(z_all).numpy()
    os.makedirs(out_root, exist_ok=True)
    np.savez_compressed(
        os.path.join(out_root, f"{dataset}_seed{seed}_emb.npz"),
        train_emb=embs["train"], val_emb=embs["val"], test_emb=embs["test"],
        train_y=ytr.numpy(), val_y=yva.numpy(), test_y=yte.numpy(),
        test_prob=prob, test_acc=te_acc)
    np.save(os.path.join(out_root, f"{dataset}_seed{seed}_history.npy"),
            np.array([(h["epoch"], h["train_loss"], h["train_acc"],
                       h["val_acc"]) for h in history]))
    print(f"[{dataset} s{seed}] TEST ACC (CNN baseline) = {te_acc:.4f}",
          flush=True)
    return te_acc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dermamnist")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--data_root", default="/tmp/data")
    default_out = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data")
    ap.add_argument("--out_root", default=default_out)
    args = ap.parse_args()
    train_one(args.dataset, args.seed, args.epochs,
              args.data_root, args.out_root)
