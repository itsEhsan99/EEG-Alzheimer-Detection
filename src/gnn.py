"""Graph Neural Network for EEG connectivity classification.

Ports the dense-GCN model from the thesis pipeline: each epoch is a
19-node graph (nodes = electrodes, edges = connectivity), classified by
a 2-layer GCN with an MLP head. Kept dense since the graphs are small.
"""
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CH = 19
HIDDEN = 32
DROPOUT = 0.5


def norm_adj(A):
    """Symmetric-normalize an adjacency matrix with self-loops."""
    n = A.shape[-1]
    I = torch.eye(n, device=A.device).expand_as(A)
    At = A + I
    dinv = torch.pow(At.sum(-1).clamp(min=1e-8), -0.5)
    return torch.diag_embed(dinv) @ At @ torch.diag_embed(dinv)


class GCNLayer(nn.Module):
    """One graph-convolution layer: linear transform then neighbor mixing."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, X, A_norm):
        return A_norm @ self.lin(X)


class GCN(nn.Module):
    """Two GCN layers + flatten + MLP head."""

    def __init__(self, n_features, n_classes=2, hidden=HIDDEN,
                 dropout=DROPOUT, n_nodes=N_CH):
        super().__init__()
        self.g1 = GCNLayer(n_features, hidden)
        self.g2 = GCNLayer(hidden, hidden)
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.Linear(n_nodes * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, X, A):
        A_norm = norm_adj(A)
        h = F.relu(self.g1(X, A_norm))
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.g2(h, A_norm))
        return self.head(h.reshape(h.size(0), -1))
    
    import numpy as np
from scipy.signal import welch, butter, filtfilt, hilbert

CHANNELS = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
            "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz"]
FREQ_BANDS = {"Delta": (1.0, 4), "Theta": (4, 8), "Alpha": (8, 13),
              "Sigma": (12, 15), "Beta": (13, 30), "Gamma": (30, 45)}
BAND_ORDER = ["Delta", "Theta", "Alpha", "Sigma", "Beta", "Gamma"]
K_FRAC = 0.40  # keep top 40% of edges


def node_band_powers(epoch_sig, sfreq):
    """Per-node relative band powers -> (19, 6) feature matrix.

    epoch_sig : (19, n_samples) array for one epoch.
    """
    f, Pw = welch(epoch_sig, fs=sfreq, nperseg=epoch_sig.shape[1], axis=1)
    bp = np.stack([
        Pw[:, (f >= FREQ_BANDS[b][0]) & (f < FREQ_BANDS[b][1])].sum(1)
        for b in BAND_ORDER
    ], axis=1)
    bp = bp / (bp.sum(1, keepdims=True) + 1e-12)
    return bp.astype(np.float32)


def _bandpass(x, lo, hi, sfreq):
    b, a = butter(4, [lo / (sfreq / 2), hi / (sfreq / 2)], btype="band")
    return filtfilt(b, a, x, axis=-1)


def plv_matrix(epoch_sig, band, sfreq):
    """PLV connectivity (19, 19) for one epoch in a given band."""
    lo, hi = FREQ_BANDS[band]
    xb = _bandpass(epoch_sig, lo, hi, sfreq)
    phase = np.angle(hilbert(xb, axis=-1))
    C = phase.shape[0]
    P = np.zeros((C, C), dtype=np.float32)
    for i in range(C):
        d = phase[i][None, :] - phase
        P[i] = np.abs(np.exp(1j * d).mean(1))
    np.fill_diagonal(P, 0.0)
    return P


def sparsify(A):
    """Keep the strongest K_FRAC of edges (symmetric), zero the rest."""
    iu = np.triu_indices(A.shape[-1], k=1)
    w = A[iu]
    k = max(1, int(round(len(w) * K_FRAC)))
    keep = np.argsort(w)[-k:]
    r, c = iu[0][keep], iu[1][keep]
    M = np.zeros_like(A)
    M[r, c] = A[r, c]
    M[c, r] = A[c, r]
    return M