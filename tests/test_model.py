"""Tests for the PLVNet model."""

import torch

from src.model import PLVNet


def test_model_output_shape():
    """Model should map a batch of PLV images to class logits."""
    model = PLVNet(n_classes=2)
    # batch of 4 single-channel 19x19 images
    x = torch.randn(4, 1, 19, 19)
    out = model(x)

    assert out.shape == (4, 2)