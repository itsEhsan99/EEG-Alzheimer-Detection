"""Build GNN training records and train a final GCN on all subjects."""

import re
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

from src.gnn import GCN, node_band_powers, sparsify, CHANNELS
from src.data_loader import load_recording

# Alpha is the 3rd band (index 2) in the 114x114 band-major arrays
BAND_IDX_ALPHA = 2
N_CH = 19
N_EPOCHS = 149
N_MINUTES = 5
EPOCH_LEN_S = 4.0
OVERLAP = 0.5

LABELS = {"AD": 1, "HC": 0}


def load_alpha_block(npy_path):
    """Load a (149,114,114) PLV file and return the alpha 19x19 block per epoch."""
    conn = np.load(npy_path)  # (149, 114, 114)
    s = N_CH * BAND_IDX_ALPHA
    A = conn[:, s:s + N_CH, s:s + N_CH].astype(np.float32).copy()
    for e in range(A.shape[0]):
        np.fill_diagonal(A[e], 0.0)
    return A  # (149, 19, 19)


def epoch_raw(raw):
    """Slice a recording into 149 overlapping 4s epochs (first 5 min)."""
    sfreq = raw.info["sfreq"]
    raw.reorder_channels(CHANNELS)
    sig = raw.get_data()[:, :int(N_MINUTES * 60 * sfreq)]
    spe = int(EPOCH_LEN_S * sfreq)                 # samples per epoch
    sst = int(EPOCH_LEN_S * (1 - OVERLAP) * sfreq)  # step
    eps = np.stack([sig[:, i * sst:i * sst + spe] for i in range(N_EPOCHS)])
    return eps.astype(np.float32), sfreq


def build_subject(npy_path, set_path, label):
    """Build one subject record: node features X, adjacency A, label y."""
    raw = load_recording(set_path)
    eps, sfreq = epoch_raw(raw)  # (149, 19, spe)

    X = np.stack([node_band_powers(eps[e], sfreq) for e in range(len(eps))])
    A = load_alpha_block(npy_path)
    A = np.stack([sparsify(A[e]) for e in range(len(A))])

    return {"X": X, "A": A, "y": label}


def train_final_gnn(subjects, n_classes=2, max_epochs=80, lr=0.01,
                    weight_decay=5e-4, batch=256, seed=0):
    """Train one GCN on all subjects' epoch-graphs (for deployment)."""
    fd = subjects[0]["X"].shape[-1]

    X = np.concatenate([s["X"] for s in subjects])
    A = np.concatenate([s["A"] for s in subjects])
    y = np.concatenate([[s["y"]] * len(s["X"]) for s in subjects])

    # Standardize node features using all training data
    flat = X.reshape(-1, fd)
    mu, sd = flat.mean(0), flat.std(0) + 1e-8
    X = (X - mu) / sd

    X = torch.tensor(X, dtype=torch.float32)
    A = torch.tensor(A, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    torch.manual_seed(seed)
    model = GCN(fd, n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Inverse-frequency class weights
    cw = torch.tensor([1 / ((y == c).float().mean() + 1e-8)
                       for c in range(n_classes)])
    cw /= cw.sum()
    criterion = nn.CrossEntropyLoss(weight=cw.float())

    n = len(y)
    model.train()
    for ep in range(max_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = criterion(model(X[b], A[b]), y[b])
            loss.backward()
            opt.step()

    # Return model plus the normalization stats (needed at inference)
    return model, mu, sd