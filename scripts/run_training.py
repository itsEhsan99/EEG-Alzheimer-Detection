"""Load the cached PLV dataset and run LOSO cross-validation."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.train import run_loso


def main():
    data = np.load(Path("DATA") / "plv_dataset.npz", allow_pickle=True)
    X, y, subjects = data["X"], data["y"], data["subjects"]

    print(f"Loaded: X={X.shape}, {len(np.unique(subjects))} subjects\n")
    print("Running Leave-One-Subject-Out...\n")

    overall, per_subject = run_loso(X, y, subjects, n_epochs=15)

    print("\n=== Per-subject accuracy ===")
    accs = [a for _, a, _ in per_subject]
    print(f"Mean subject accuracy: {np.mean(accs):.3f} (std {np.std(accs):.3f})")

    print("\n=== Overall pooled metrics ===")
    print(f"Accuracy: {overall['accuracy']:.3f}")
    print(f"F1:       {overall['f1']:.3f}")
    print(f"AUC:      {overall['auc']:.3f}")


if __name__ == "__main__":
    main()