"""Tests for the preprocessing functions."""

import numpy as np
import mne
import pytest
from src.preprocessing import (bandpass_filter, epoch_signal, normalize_epochs, apply_preprocessing)

def make_dummy_raw(sfreq=500.0, n_channels=4, duration=10.0):
    """Create a small synthetic Raw object for testing.

    We don't load real files in tests: tests should be fast and not
    depend on data being present. A synthetic signal is enough to check
    that our function behaves correctly.
    """
    n_samples = int(sfreq * duration)
    rng = np.random.default_rng(seed=42)
    data = rng.standard_normal((n_channels, n_samples)) * 1e-6  # ~microvolt scale
    ch_names = [f"EEG{i}" for i in range(n_channels)]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose=False)


def test_bandpass_returns_new_object():
    """The filter must not modify its input (no side effects)."""
    raw = make_dummy_raw()
    original = raw.get_data().copy()

    filtered = bandpass_filter(raw, 0.5, 45.0)

    # The original data must be unchanged
    assert np.array_equal(raw.get_data(), original)
    # The returned object must be a different one
    assert filtered is not raw


def test_bandpass_changes_the_signal():
    """Filtering should actually alter the signal values."""
    raw = make_dummy_raw()
    filtered = bandpass_filter(raw, 0.5, 45.0)

    # Filtered data should differ from the raw data
    assert not np.array_equal(filtered.get_data(), raw.get_data())


def test_bandpass_rejects_invalid_frequencies():
    """low_freq >= high_freq should raise a clear error."""
    raw = make_dummy_raw()
    with pytest.raises(ValueError):
        bandpass_filter(raw, 45.0, 0.5)


def test_epoching_shape():
    """Epoching should return the expected number and shape of windows."""
    # 10 seconds at 500 Hz, 4 channels -> with 5s epochs, expect 2 windows
    raw = make_dummy_raw(sfreq=500.0, n_channels=4, duration=10.0)
    epochs = epoch_signal(raw, epoch_seconds=5.0, overlap=0.0)

    assert epochs.shape == (2, 4, 2500)


def test_epoching_with_overlap_gives_more_windows():
    """More overlap should produce more windows from the same signal."""
    raw = make_dummy_raw(sfreq=500.0, n_channels=4, duration=10.0)
    no_overlap = epoch_signal(raw, epoch_seconds=5.0, overlap=0.0)
    with_overlap = epoch_signal(raw, epoch_seconds=5.0, overlap=0.5)

    assert with_overlap.shape[0] > no_overlap.shape[0]


def test_epoching_rejects_invalid_overlap():

    """Overlap outside [0, 1) should raise a clear error."""
    raw = make_dummy_raw()
    with pytest.raises(ValueError):
        epoch_signal(raw, epoch_seconds=5.0, overlap=1.0)



def test_normalization_produces_zero_mean_unit_std():
    """After normalization each epoch/channel should have ~0 mean, ~1 std."""
    raw = make_dummy_raw(sfreq=500.0, n_channels=4, duration=10.0)
    epochs = epoch_signal(raw, epoch_seconds=5.0)
    normalized = normalize_epochs(epochs)

    # Mean along the time axis should be ~0, std ~1
    assert np.allclose(normalized.mean(axis=2), 0.0, atol=1e-6)
    assert np.allclose(normalized.std(axis=2), 1.0, atol=1e-6)


def test_normalization_preserves_shape():
    """Normalization must not change the array shape."""
    raw = make_dummy_raw(sfreq=500.0, n_channels=4, duration=10.0)
    epochs = epoch_signal(raw, epoch_seconds=5.0)
    normalized = normalize_epochs(epochs)

    assert normalized.shape == epochs.shape


def test_normalization_rejects_wrong_dimensions():
    """A non-3D input should raise a clear error."""
    flat = np.random.default_rng(0).standard_normal((4, 2500))
    with pytest.raises(ValueError):
        normalize_epochs(flat)

def test_apply_preprocessing_resample_and_drop():
    """Preprocessing should drop channels and change the sampling rate."""
    raw = make_dummy_raw(sfreq=500.0, n_channels=4, duration=10.0)
    original_names = list(raw.ch_names)

    processed = apply_preprocessing(
        raw, resample_freq=250.0, drop_channels=[original_names[0]])

    # One channel dropped
    assert len(processed.ch_names) == 3
    assert original_names[0] not in processed.ch_names
    # Sampling rate changed
    assert processed.info["sfreq"] == 250.0
    # Original untouched
    assert len(raw.ch_names) == 4


