"""Inference: turn a raw recording into a record-level prediction.

Loads the trained model once, runs it on every epoch's PLV matrix, and
aggregates the per-epoch outputs into a single prediction for the whole
recording.
"""

from pathlib import Path
import numpy as np
import torch

from src.model import PLVNet
from src.data_loader import load_recording
from src.features import recording_to_plv


MODEL_PATH = Path("models") / "plvnet.pt"
CLASS_NAMES = {0: "HC", 1: "AD"}


def load_model(model_path=MODEL_PATH):
    """Load the trained model weights into a fresh PLVNet."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            f"Run scripts/train_final_model.py first."
        )
    model = PLVNet(n_classes=2)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict_recording(file_path, model=None):
    """Predict AD vs HC for a single recording.

    Parameters
    ----------
    file_path : str or Path
        Path to a .set recording.
    model : PLVNet, optional
        A preloaded model. If None, the model is loaded from disk.

    Returns
    -------
    dict
        Prediction summary with the label, confidence, and details.
    """
    if model is None:
        model = load_model()

    raw = load_recording(file_path)
    plv = recording_to_plv(raw)  # (n_epochs, 19, 19)

    X = torch.tensor(plv, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()  # P(AD) per epoch

    mean_ad_prob = float(np.mean(probs))
    predicted_class = 1 if mean_ad_prob >= 0.5 else 0
    fraction_ad = float(np.mean(probs >= 0.5))

    return {
        "prediction": CLASS_NAMES[predicted_class],
        "ad_probability": round(mean_ad_prob, 3),
        "fraction_windows_ad": round(fraction_ad, 3),
        "n_windows": len(probs),
    }