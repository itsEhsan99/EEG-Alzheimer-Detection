"""Preprocessing operations for EEG recordings.

Each function does one thing and returns a new object, so operations
can be composed and tested independently.
"""


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