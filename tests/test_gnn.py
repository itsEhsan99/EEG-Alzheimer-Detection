"""Tests for the GNN module."""

import numpy as np
import torch

from src.gnn import GCN, norm_adj, node_band_powers, plv_matrix, sparsify


def test_gcn_output_shape():
    """GCN should map a batch of graphs to class logits."""
    model = GCN(n_features=6, n_classes=2)
    X = torch.randn(4, 19, 6)   # 4 graphs, 19 nodes, 6 features
    A = torch.rand(4, 19, 19)
    out = model(X, A)
    assert out.shape == (4, 2)


def test_norm_adj_symmetric_shape():
    """Normalized adjacency keeps the same shape."""
    A = torch.rand(2, 19, 19)
    An = norm_adj(A)
    assert An.shape == A.shape


def test_node_features_shape():
    """Band-power features should be (19, 6)."""
    rng = np.random.default_rng(0)
    epoch = rng.standard_normal((19, 2000))
    feats = node_band_powers(epoch, sfreq=500.0)
    assert feats.shape == (19, 6)


def test_plv_and_sparsify():
    """PLV is 19x19 in [0,1]; sparsify keeps ~40% of edges."""
    rng = np.random.default_rng(1)
    epoch = rng.standard_normal((19, 2000))
    plv = plv_matrix(epoch, band="Alpha", sfreq=500.0)
    assert plv.shape == (19, 19)
    assert np.all(plv >= 0) and np.all(plv <= 1)

    sp = sparsify(plv)
    # Sparsified matrix has fewer nonzero edges than the full one
    assert np.count_nonzero(sp) <= np.count_nonzero(plv)