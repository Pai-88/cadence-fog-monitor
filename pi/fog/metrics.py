"""Evaluation metrics and the Freeze-Index baseline classifier.

Torch-free (NumPy only). Freezes are rare, so accuracy is meaningless here — a
model that always predicts "no freeze" scores ~90 %. We report **sensitivity**
(freeze recall: a missed freeze can mean a fall) and **specificity** (a cue that
does not cry wolf) instead. ``sens_spec`` is implemented directly rather than via
scikit-learn so the core metric carries no heavy dependency and is trivially
unit-testable.
"""
from __future__ import annotations

import numpy as np

from .config import FI_THRESHOLD
from .dsp import freeze_index

__all__ = ["sens_spec", "freeze_index_predict", "best_fi_threshold"]


def sens_spec(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Sensitivity, specificity and the 2x2 confusion matrix for {0, 1} labels.

    The confusion matrix matches ``sklearn.metrics.confusion_matrix(labels=[0, 1])``
    exactly: rows are the true class, columns the predicted class, so
    ``cm.ravel()`` is ``(tn, fp, fn, tp)``. Returns ``nan`` for a rate whose
    denominator is empty (e.g. specificity when there are no true negatives).
    """
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")   # freeze recall
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    return sens, spec, cm


def freeze_index_predict(X: np.ndarray, threshold: float) -> np.ndarray:
    """Engineered baseline: per-window Freeze Index vs. ``threshold`` → 0/1.

    ``X`` is ``(N, C, T)`` (channels-first, as the CNN consumes it); the Freeze
    Index is computed on the ``(T, C)`` transpose of each window.
    """
    return np.array(
        [1 if freeze_index(X[i].T) > threshold else 0 for i in range(len(X))],
        dtype=np.int64,
    )


def best_fi_threshold(X: np.ndarray, y: np.ndarray) -> float:
    """Pick the Freeze-Index threshold maximising Youden's J on ``(X, y)``.

    Sweeps the literature range and returns the threshold with the best
    ``sensitivity + specificity - 1``. Intended to be tuned on the *training*
    fold and then applied unchanged to the held-out subject.
    """
    fis = np.array([freeze_index(X[i].T) for i in range(len(X))])
    best_t, best_j = FI_THRESHOLD, -1.0
    for t in np.linspace(0.5, 6.0, 45):
        sens, spec, _ = sens_spec(y, (fis > t).astype(np.int64))
        j = (0 if np.isnan(sens) else sens) + (0 if np.isnan(spec) else spec) - 1
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t
