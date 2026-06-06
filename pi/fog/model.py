"""The freeze-of-gait classifier — a small 1D CNN over accel windows.

This is the **only** module in the package that imports torch, so every other
layer (DSP, serial, metrics, normalisation) stays importable on a machine with
no deep-learning stack installed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LABELS, NUM_AXES

__all__ = ["FoGNet"]


class FoGNet(nn.Module):
    """~18-30k-parameter 1D CNN on ``(NUM_AXES, WINDOW_SIZE)`` accel windows.

    Three temporal conv blocks (Conv → BatchNorm → ReLU → max-pool) with widening
    receptive fields, global average pooling, then a small two-layer classifier
    head. It runs in well under 1 ms on a laptop CPU.

    ``c1/c2/c3`` are the conv channel widths, ``fc`` the dense width and
    ``dropout`` the head dropout. The defaults reproduce the baseline ~18k-param
    net exactly; the Optuna search in fog_analysis.py varies them. The hyper-
    parameters are recorded on :attr:`arch` so a checkpoint can be reloaded with
    the *same* shape — save ``{'arch': model.arch}`` and rebuild with
    ``FoGNet(num_classes=..., **ckpt['arch'])``.
    """

    def __init__(
        self,
        num_classes: int = len(LABELS),
        num_axes: int = NUM_AXES,
        c1: int = 16,
        c2: int = 32,
        c3: int = 64,
        fc: int = 32,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Everything except num_classes — i.e. enough to rebuild the same shape
        # from a checkpoint (num_classes is recovered from the saved label map).
        self.arch: dict[str, int | float] = dict(
            num_axes=num_axes, c1=c1, c2=c2, c3=c3, fc=fc, dropout=dropout
        )
        self.conv1 = nn.Conv1d(num_axes, c1, kernel_size=15, padding=7)
        self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=9, padding=4)
        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(c3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(c3, fc)
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)
