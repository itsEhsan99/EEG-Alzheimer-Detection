"""Leave-One-Subject-Out training and evaluation for PLVNet."""

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.model import PLVNet


def train_one_fold(X_train, y_train, n_epochs=15, lr=1e-3, batch_size=64):
    """Train a fresh PLVNet on one fold's training data."""
    model = PLVNet(n_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Add the channel dimension: (N, 19, 19) -> (N, 1, 19, 19)
    X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_train, dtype=torch.long)

    n = len(X_t)
    model.train()
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            optimizer.zero_grad()
            out = model(X_t[idx])
            loss = criterion(out, y_t[idx])
            loss.backward()
            optimizer.step()
    return model


def predict(model, X):
    """Return predicted labels and probabilities for the positive class."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    with torch.no_grad():
        logits = model(X_t)
        probs = torch.softmax(logits, dim=1)[:, 1].numpy()
        preds = logits.argmax(dim=1).numpy()
    return preds, probs


def run_loso(X, y, subjects, n_epochs=15):
    """Run Leave-One-Subject-Out cross-validation.

    Returns per-subject results and overall pooled metrics.
    """
    logo = LeaveOneGroupOut()
    all_true, all_pred, all_prob = [], [], []
    per_subject = []

    for train_idx, test_idx in logo.split(X, y, groups=subjects):
        test_subject = subjects[test_idx][0]

        model = train_one_fold(X[train_idx], y[train_idx], n_epochs=n_epochs)
        preds, probs = predict(model, X[test_idx])

        y_true = y[test_idx]
        subj_acc = accuracy_score(y_true, preds)
        per_subject.append((test_subject, subj_acc, len(y_true)))
        print(f"  {test_subject}: acc={subj_acc:.3f} ({len(y_true)} epochs)")

        all_true.extend(y_true)
        all_pred.extend(preds)
        all_prob.extend(probs)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_prob = np.array(all_prob)

    overall = {
        "accuracy": accuracy_score(all_true, all_pred),
        "f1": f1_score(all_true, all_pred),
        "auc": roc_auc_score(all_true, all_prob),
    }
    return overall, per_subject


def train_full(X, y, n_epochs=15):
    """Train a single model on all available data (for deployment)."""
    return train_one_fold(X, y, n_epochs=n_epochs)

