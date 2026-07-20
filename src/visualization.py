"""Rendering EEG analyses as PNG images for the web frontend.

Each function returns raw PNG bytes, so the API can send them straight
to the browser without writing temporary files to disk.
"""

import io
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, required on a server
import matplotlib.pyplot as plt
import numpy as np
from src.features import compute_band_powers, recording_to_plv

def _figure_to_png_bytes(fig, dpi=160):
    """Convert a matplotlib figure to PNG bytes and close it."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def plot_raw_signal(raw, start_time=0.0, n_seconds=10, channels=None):
    """Plot a chosen time window of chosen EEG channels.

    Parameters
    ----------
    raw : mne.io.Raw
        The recording to plot.
    start_time : float
        Start of the window, in seconds from the beginning.
    n_seconds : float
        Length of the window, in seconds.
    channels : list of str, optional
        Which channels to show. If None or empty, all channels are shown.

    Returns
    -------
    bytes
        PNG image bytes.
    """
    sfreq = raw.info["sfreq"]
    all_names = raw.ch_names

    # Which channels: keep requested ones in file order, or all if none given
    if channels:
        selected = [ch for ch in all_names if ch in channels]
    else:
        selected = all_names
    if not selected:
        selected = all_names

    idx = [all_names.index(ch) for ch in selected]

    # Time window in samples, clamped to the recording length
    total_samples = raw.n_times
    start_sample = max(0, int(start_time * sfreq))
    n_samples = int(n_seconds * sfreq)
    end_sample = min(total_samples, start_sample + n_samples)

    data = raw.get_data()[idx, start_sample:end_sample]
    times = np.arange(start_sample, end_sample) / sfreq

    # Height scales with the number of channels so they don't overlap
    height = max(3, len(selected) * 0.32)
    fig, ax = plt.subplots(figsize=(8, height), facecolor="#0d1424")
    ax.set_facecolor("#0d1424")

    spacing = np.std(data) * 6 if np.std(data) > 0 else 1
    offset = 0
    for i, ch in enumerate(selected):
        ax.plot(times, data[i] + offset, color="#f59e0b", linewidth=0.6)
        ax.text(times[0] - 0.3, offset, ch, color="#94a3b8",
                va="center", ha="right", fontsize=8)
        offset -= spacing

    ax.set_xlabel("Time (s)", color="#94a3b8")
    ax.set_yticks([])
    ax.set_title(f"EEG signal ({start_time:.0f}-{start_time + n_seconds:.0f}s)",
                 color="#e2e8f0", fontsize=13)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values():
        spine.set_color("#2a3a52")

    return _figure_to_png_bytes(fig)

def plot_band_powers(raw):
    """Plot relative power in each frequency band as a clean bar chart."""
    band_powers = compute_band_powers(raw, relative=True)
    bands = list(band_powers.keys())
    values = [band_powers[b] * 100 for b in bands]  # to percent

    fig, ax = plt.subplots(figsize=(5.5, 3.2), facecolor="#0d1424")
    ax.set_facecolor("#0d1424")

    colors = ["#3b82f6", "#8b5cf6", "#f59e0b", "#ef4444", "#10b981"]
    bars = ax.bar(bands, values, color=colors[:len(bands)], width=0.6)

    # Label each bar with its percentage
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                f"{val:.1f}%", ha="center", color="#cbd5e1", fontsize=9)

    ax.set_ylabel("Relative power (%)", color="#94a3b8", fontsize=10)
    ax.set_ylim(0, max(values) * 1.2)
    ax.tick_params(colors="#94a3b8")
    # Minimal look: remove top/right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2a3a52")
    ax.spines["bottom"].set_color("#2a3a52")

    return _figure_to_png_bytes(fig)

# Frequency bands available for connectivity
CONN_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def _mean_plv_matrix(raw, band="alpha"):
    """Recording-level PLV matrix (mean over epochs) for a given band."""
    import numpy as np
    low, high = CONN_BANDS.get(band, CONN_BANDS["alpha"])
    plv_stack = recording_to_plv(raw, low_freq=low, high_freq=high)
    plv = plv_stack.mean(axis=0)
    np.fill_diagonal(plv, 0.0)  # a channel with itself is not meaningful
    return plv


def plot_plv_heatmap(raw, band="alpha"):
    """Plot the mean PLV matrix for a band as a labeled heatmap."""
    plv = _mean_plv_matrix(raw, band=band)
    ch_names = raw.ch_names

    fig, ax = plt.subplots(figsize=(4.6, 4.2), facecolor="#0d1424")
    ax.set_facecolor("#0d1424")

    im = ax.imshow(plv, cmap="magma", vmin=0, vmax=1, aspect="equal")

    ax.set_xticks(range(len(ch_names)))
    ax.set_yticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=5.5, color="#cbd5e1")
    ax.set_yticklabels(ch_names, fontsize=5.5, color="#cbd5e1")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("PLV", color="#94a3b8", fontsize=8)
    cbar.ax.tick_params(colors="#94a3b8", labelsize=7)
    cbar.outline.set_edgecolor("#2a3a52")

    for spine in ax.spines.values():
        spine.set_color("#2a3a52")

    return _figure_to_png_bytes(fig)


def plot_plv_topomap(raw, band="alpha", top_fraction=0.15):
    """Draw the strongest PLV connections on a head layout for a band."""
    import mne
    import numpy as np

    plv = _mean_plv_matrix(raw, band=band)
    ch_names = raw.ch_names
    n = len(ch_names)

    montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(ch_names, sfreq=raw.info["sfreq"], ch_types="eeg")
    info.set_montage(montage, on_missing="ignore")
    layout = mne.channels.find_layout(info)
    pos = {name: layout.pos[layout.names.index(name)][:2]
           for name in ch_names if name in layout.names}

    fig, ax = plt.subplots(figsize=(4.6, 4.6), facecolor="#0d1424")
    ax.set_facecolor("#0d1424")

    upper_vals = [plv[i, j] for i in range(n) for j in range(i + 1, n)]
    threshold = np.quantile(upper_vals, 1 - top_fraction)

    for i in range(n):
        for j in range(i + 1, n):
            if ch_names[i] in pos and ch_names[j] in pos:
                strength = plv[i, j]
                if strength >= threshold:
                    x = [pos[ch_names[i]][0], pos[ch_names[j]][0]]
                    y = [pos[ch_names[i]][1], pos[ch_names[j]][1]]
                    ax.plot(x, y, color="#f59e0b",
                            alpha=float(min(strength, 1.0)),
                            linewidth=1.5 * strength + 0.5)

    for name in ch_names:
        if name in pos:
            ax.plot(pos[name][0], pos[name][1], "o",
                    color="#3b82f6", markersize=7)
            ax.text(pos[name][0], pos[name][1] + 0.02, name,
                    ha="center", va="bottom", fontsize=6, color="#cbd5e1")

    ax.set_aspect("equal")
    ax.axis("off")

    return _figure_to_png_bytes(fig)

