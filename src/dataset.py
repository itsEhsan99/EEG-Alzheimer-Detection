"""Build a small PLV dataset from the ds004504 .set files.

For each subject we load the recording, run the PLV pipeline, and keep
three aligned arrays: the PLV matrices, the class label, and the subject
id. The subject id is what makes an honest leave-one-subject-out split
possible.
"""

from pathlib import Path
import re
import numpy as np

from src.data_loader import load_recording
from src.features import recording_to_plv


LABELS = {"AD": 1, "HC": 0}


def subject_id(set_path):
    """Extract a subject id like 'sub-001' from a .set filename."""
    set_path = Path(set_path)
    match = re.search(r"(sub-\d+)", set_path.name)
    if match is None:
        raise ValueError(f"Could not find subject id in {set_path.name}")
    return match.group(1)


def build_dataset(data_dir, groups=("AD", "HC"), n_per_group=10,
                  low_freq=8.0, high_freq=13.0,
                  epoch_seconds=5.0, overlap=0.0):
    """Build PLV feature arrays from real recordings.

    Returns
    -------
    X : np.ndarray
        PLV matrices, shape (total_epochs, n_channels, n_channels).
    y : np.ndarray
        Labels per epoch, shape (total_epochs,).
    subjects : np.ndarray
        Subject id per epoch, shape (total_epochs,).
    """
    data_dir = Path(data_dir)
    X_list, y_list, subj_list = [], [], []

    for group in groups:
        label = LABELS[group]
        group_dir = data_dir / group
        set_files = sorted(group_dir.glob("*.set"))[:n_per_group]
        if not set_files:
            raise FileNotFoundError(f"No .set files found in {group_dir}")

        for set_path in set_files:
            subj = subject_id(set_path)
            raw = load_recording(set_path)
            plv = recording_to_plv(
                raw, low_freq, high_freq, epoch_seconds, overlap
            )
            X_list.append(plv)
            y_list.append(np.full(len(plv), label))
            subj_list.append(np.array([subj] * len(plv)))
            print(f"{group} {subj}: {plv.shape[0]} epochs")

    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    subjects = np.concatenate(subj_list)
    return X, y, subjects