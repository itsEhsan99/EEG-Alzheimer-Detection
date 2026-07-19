"""Preprocessing operations for EEG recordings.

Each function does one thing and returns a new object, so operations
can be composed and tested independently.
"""

import numpy as np 

def bandpass_filter(raw, low_freq=0.5, high_freq=45.0):
    """Apply a band-pass filter to an EEG recording.

    Parameters
    ----------
    raw : mne.io.Raw
        The recording to filter.
    low_freq : float
        Lower cutoff frequency in Hz.
    high_freq : float
        Upper cutoff frequency in Hz.

    Returns
    -------
    mne.io.Raw
        A new, filtered copy of the recording. The input is not modified.
    """
    if low_freq >= high_freq:
        raise ValueError(
            f"low_freq ({low_freq}) must be smaller than high_freq ({high_freq})"
        )

    filtered = raw.copy()
    filtered.filter(l_freq=low_freq, h_freq=high_freq, verbose=False)
    return filtered

import numpy as np


def epoch_signal(raw, epoch_seconds=5.0, overlap=0.0):
    """Split a continuous recording into fixed-length epochs (windows).

    Parameters
    ----------
    raw : mne.io.Raw
        The recording to split.
    epoch_seconds : float
        Length of each epoch in seconds.
    overlap : float
        Fraction of overlap between consecutive epochs, from 0.0 (no
        overlap) up to but not including 1.0.

    Returns
    -------
    np.ndarray
        Array of shape (n_epochs, n_channels, n_samples_per_epoch).
    """
    if epoch_seconds <= 0:
        raise ValueError(f"epoch_seconds must be positive, got {epoch_seconds}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0.0, 1.0), got {overlap}")

    data = raw.get_data()  # shape: (n_channels, n_samples)
    sfreq = raw.info["sfreq"]

    samples_per_epoch = int(epoch_seconds * sfreq)
    step = int(samples_per_epoch * (1.0 - overlap))

    n_channels, n_samples = data.shape

    epochs = []
    start = 0
    while start + samples_per_epoch <= n_samples:
        window = data[:, start:start + samples_per_epoch]
        epochs.append(window)
        start += step

    if not epochs:
        raise ValueError(
            f"Recording too short: {n_samples} samples, "
            f"need at least {samples_per_epoch} for one epoch"
        )

    return np.stack(epochs)


def normalize_epochs(epochs):
    """Normalize each epoch per channel to zero mean and unit variance.

    Normalization is done independently for every epoch and every
    channel (per-window normalization), which helps the model generalize
    across recordings with different amplitude scales.

    Parameters
    ----------
    epochs : np.ndarray
        Array of shape (n_epochs, n_channels, n_samples).

    Returns
    -------
    np.ndarray
        Normalized array of the same shape.
    """
    if epochs.ndim != 3:
        raise ValueError(
            f"Expected 3D array (n_epochs, n_channels, n_samples), "
            f"got shape {epochs.shape}"
        )

    # Compute mean and std along the time axis (last axis), keeping dims
    # so they broadcast back over the samples.
    mean = epochs.mean(axis=2, keepdims=True)
    std = epochs.std(axis=2, keepdims=True)

    # Avoid division by zero for flat channels
    std = np.where(std == 0, 1.0, std)

    return (epochs - mean) / std