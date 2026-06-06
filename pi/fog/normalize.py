"""Per-channel z-score normalisation, fit on training data only.

Torch-free (NumPy only): the same statistics are applied in training and at
inference, so the normaliser is persisted next to the model checkpoint and
reloaded by every deployment path.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Normaliser"]


class Normaliser:
    """Z-score each channel using statistics fit on the training set.

    Operates on ``(N, C, T)`` batches: the mean and std are taken per channel
    across both the window and time axes, so every accelerometer axis is scaled
    independently. Fit on training data only — never peek at validation/test.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> Normaliser:
        """Estimate per-channel mean/std from ``(N, C, T)`` training windows."""
        self.mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        # +1e-6 guards against a dead (constant) channel producing a divide-by-0.
        self.std = (X.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the fitted z-score to ``(N, C, T)`` windows."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normaliser.transform called before fit/load")
        return ((X - self.mean) / self.std).astype(np.float32)

    def save(self, path: str) -> None:
        """Persist the fitted statistics to a ``.npz`` file."""
        if self.mean is None or self.std is None:
            raise RuntimeError("Normaliser.save called before fit")
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path: str) -> Normaliser:
        """Reconstruct a normaliser from a ``.npz`` saved by :meth:`save`."""
        d = np.load(path)
        n = cls()
        n.mean, n.std = d["mean"].astype(np.float32), d["std"].astype(np.float32)
        return n
