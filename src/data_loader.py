"""Loading EEG recordings from the ds004504 dataset (.set files)."""

from pathlib import Path
import mne


def load_recording(file_path):
    """Load one EEG recording from an EEGLAB .set file.

    Parameters
    ----------
    file_path : str or Path
        Path to a sub-XX_task-eyesclosed_eeg.set file.

    Returns
    -------
    mne.io.Raw
        The loaded recording, with data available for analysis.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Recording not found: {file_path}")

    raw = mne.io.read_raw_eeglab(file_path, preload=True)
    return raw