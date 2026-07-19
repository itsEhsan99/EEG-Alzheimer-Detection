"""Feature extraction from EEG epochs.

Currently provides Phase Locking Value (PLV) connectivity, which turns
each epoch into a channels x channels matrix suitable for feeding to a
CNN (as a single-channel image) or, later, a graph model.
"""

import numpy as np
from scipy.signal import hilbert


def compute_plv(epoch):
    """Compute the Phase Locking Value matrix for one epoch.

    Assumes the epoch has already been band-pass filtered to the band
    of interest (e.g. alpha 8-13 Hz), since PLV is defined per band.

    Parameters
    ----------
    epoch : np.ndarray
        Array of shape (n_channels, n_samples) for a single window.

    Returns
    -------
    np.ndarray
        Symmetric PLV matrix of shape (n_channels, n_channels), with
        values in [0, 1] and ones on the diagonal.
    """
    if epoch.ndim != 2:
        raise ValueError(
            f"Expected 2D array (n_channels, n_samples), got shape {epoch.shape}"
        )

    # Instantaneous phase of each channel via the analytic signal
    analytic = hilbert(epoch, axis=1)
    phase = np.angle(analytic)  # shape: (n_channels, n_samples)

    n_channels = phase.shape[0]
    plv = np.ones((n_channels, n_channels))

    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            phase_diff = phase[i] - phase[j]
            value = np.abs(np.mean(np.exp(1j * phase_diff)))
            plv[i, j] = value
            plv[j, i] = value  # symmetric

    return plv

from src.preprocessing import bandpass_filter, epoch_signal

def recording_to_plv(raw, low_freq=8.0, high_freq=13.0,
                     epoch_seconds=5.0, overlap=0.0):
    """Turn one raw recording into an array of PLV matrices.

    Pipeline: band-pass to the band of interest (default alpha 8-13 Hz),
    split into epochs, then compute a PLV matrix for each epoch.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw recording.
    low_freq, high_freq : float
        Band-pass edges for the band PLV is computed on.
    epoch_seconds : float
        Epoch length in seconds.
    overlap : float
        Overlap fraction between epochs.

    Returns
    -------
    np.ndarray
        Array of shape (n_epochs, n_channels, n_channels).
    """
    filtered = bandpass_filter(raw, low_freq, high_freq)
    epochs = epoch_signal(filtered, epoch_seconds, overlap)

    plv_matrices = [compute_plv(epoch) for epoch in epochs]
    return np.stack(plv_matrices)

