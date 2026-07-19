"""A small CNN that classifies PLV connectivity matrices."""

import torch
import torch.nn as nn


class PLVNet(nn.Module):
    """Lightweight 2-layer CNN for 19x19 PLV matrices.

    Input is treated as a single-channel image of shape (1, 19, 19).
    """

    def __init__(self, n_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # global average pooling -> (16, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(16, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x