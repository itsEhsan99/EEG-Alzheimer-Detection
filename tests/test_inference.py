"""Tests for the inference layer."""

import pytest

from src.inference import load_model


def test_load_model_missing_file_raises():
    """Loading from a non-existent path should give a clear error."""
    with pytest.raises(FileNotFoundError):
        load_model("models/does_not_exist.pt")