"""Build the small PLV dataset and cache it to disk."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.dataset import build_dataset


def main():
    X, y, subjects = build_dataset("DATA", n_per_group=10)

    print(f"\nDataset: X={X.shape}, y={y.shape}, subjects={subjects.shape}")
    print(f"Unique subjects: {len(np.unique(subjects))}")
    print(f"Class balance -> AD: {int((y == 1).sum())}, HC: {int((y == 0).sum())}")

    out_path = Path("DATA") / "plv_dataset.npz"
    np.savez_compressed(out_path, X=X, y=y, subjects=subjects)
    print(f"Saved cached dataset to {out_path}")


if __name__ == "__main__":
    main()