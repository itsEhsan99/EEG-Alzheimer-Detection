"""Inference with the trained GNN on a single recording."""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.gnn import GCN, node_band_powers, plv_matrix, sparsify
from src.gnn_train import epoch_raw
from src.data_loader import load_recording

MODEL_PATH = Path("models") / "gnn.pt"
CLASS_NAMES = {0: "HC", 1: "AD"}


def load_gnn(model_path=MODEL_PATH):
    """Load the trained GNN plus its normalization stats."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained GNN not found at {model_path}. "
            f"Run scripts/train_gnn.py first."
        )
    ckpt = torch.load(model_path, weights_only=False)
    model = GCN(ckpt["n_features"], n_classes=2)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["mu"], ckpt["sd"]


def predict_recording_gnn(file_path, model=None, mu=None, sd=None):
    """Predict AD vs HC for one recording using the GNN.

    Builds a graph per epoch (alpha PLV + band-power node features),
    scores each, and aggregates by soft voting.
    """
    if model is None:
        model, mu, sd = load_gnn()

    raw = load_recording(file_path)
    eps, sfreq = epoch_raw(raw)  # (149, 19, spe)

    # Build node features and graphs for every epoch
    X = np.stack([node_band_powers(eps[e], sfreq) for e in range(len(eps))])
    A = np.stack([sparsify(plv_matrix(eps[e], "Alpha", sfreq))
                  for e in range(len(eps))])

    # Standardize node features with the training stats
    X = (X - mu) / sd

    X = torch.tensor(X, dtype=torch.float32)
    A = torch.tensor(A, dtype=torch.float32)

    with torch.no_grad():
        probs = F.softmax(model(X, A), dim=1).numpy()  # (149, 2)

    mean_prob = probs.mean(0)          # soft voting
    predicted_class = int(mean_prob.argmax())
    ad_prob = float(mean_prob[1])
    fraction_ad = float((probs.argmax(1) == 1).mean())

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "ad_probability": round(ad_prob, 3),
        "fraction_windows_ad": round(fraction_ad, 3),
        "n_windows": int(len(probs)),
    }