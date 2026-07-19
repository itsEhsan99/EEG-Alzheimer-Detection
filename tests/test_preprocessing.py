"""Tests for the preprocessing functions."""

import numpy as np
import mne
import pytest

from src.preprocessing import bandpass_filter


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