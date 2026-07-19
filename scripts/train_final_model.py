"""Train a final model on all subjects and save it to disk."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from src.train import train_full


def main():
    data = np.load(Path("DATA") / "plv_dataset.npz", allow_pickle=True)
    X, y = data["X"], data["y"]

    print(f"Training final model on all data: X={X.shape}")
    model = train_full(X, y, n_epochs=15)

    # Save the trained weights
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "plvnet.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved trained model to {out_path}")


if __name__ == "__main__":
    main()