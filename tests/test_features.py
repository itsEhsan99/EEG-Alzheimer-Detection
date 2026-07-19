"""Tests for feature extraction."""

import numpy as np
import pytest
import mne


def make_dummy_raw(sfreq=500.0, n_channels=19, duration=20.0):
    n_samples = int(sfreq * duration)
    rng = np.random.default_rng(seed=7)
    data = rng.standard_normal((n_channels, n_samples)) * 1e-6
    ch_names = [f"EEG{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def test_recording_to_plv_shape():
    """The pipeline should return one PLV matrix per epoch."""
    raw = make_dummy_raw(sfreq=500.0, n_channels=19, duration=20.0)
    # 20 s with 5 s epochs, no overlap -> 4 epochs
    plv_stack = recording_to_plv(raw, epoch_seconds=5.0, overlap=0.0)

    assert plv_stack.shape == (4, 19, 19)

from src.features import compute_plv, recording_to_plv

def test_plv_shape_and_range():
    """PLV should be square, match channel count, and lie in [0, 1]."""
    rng = np.random.default_rng(0)
    epoch = rng.standard_normal((19, 2500))
    plv = compute_plv(epoch)

    assert plv.shape == (19, 19)
    assert np.all(plv >= 0.0) and np.all(plv <= 1.0)


def test_plv_is_symmetric_with_unit_diagonal():
    """PLV matrix must be symmetric with ones on the diagonal."""
    rng = np.random.default_rng(1)
    epoch = rng.standard_normal((19, 2500))
    plv = compute_plv(epoch)

    assert np.allclose(plv, plv.T)
    assert np.allclose(np.diag(plv), 1.0)


def test_plv_identical_channels_gives_one():
    """Two identical signals are perfectly phase-locked (PLV = 1)."""
    rng = np.random.default_rng(2)
    signal = rng.standard_normal(2500)
    epoch = np.stack([signal, signal])  # two identical channels
    plv = compute_plv(epoch)

    assert np.isclose(plv[0, 1], 1.0)


def test_plv_rejects_wrong_dimensions():
    """A non-2D input should raise a clear error."""
    flat = np.random.default_rng(3).standard_normal(2500)
    with pytest.raises(ValueError):
        compute_plv(flat)


def make_dummy_raw(sfreq=500.0, n_channels=19, duration=20.0):
    n_samples = int(sfreq * duration)
    rng = np.random.default_rng(seed=7)
    data = rng.standard_normal((n_channels, n_samples)) * 1e-6
    ch_names = [f"EEG{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def test_recording_to_plv_shape():
    """The pipeline should return one PLV matrix per epoch."""
    raw = make_dummy_raw(sfreq=500.0, n_channels=19, duration=20.0)
    # 20 s with 5 s epochs, no overlap -> 4 epochs
    plv_stack = recording_to_plv(raw, epoch_seconds=5.0, overlap=0.0)

    assert plv_stack.shape == (4, 19, 19)

