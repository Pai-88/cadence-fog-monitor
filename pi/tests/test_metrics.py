"""Tests for fog.metrics — sensitivity/specificity + the Freeze-Index baseline."""
from __future__ import annotations

import numpy as np

from fog.metrics import best_fi_threshold, freeze_index_predict, sens_spec


# ── sens_spec ────────────────────────────────────────────────────────────────
def test_sens_spec_hand_counted() -> None:
    # y_true: 3 freeze (1), 3 no-freeze (0). Predictions chosen so the 2x2 is
    # known by hand: tp=2, fn=1, tn=2, fp=1.
    y_true = np.array([1, 1, 1, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 0, 0, 1])
    sens, spec, cm = sens_spec(y_true, y_pred)
    assert sens == 2 / 3          # tp / (tp + fn)
    assert spec == 2 / 3          # tn / (tn + fp)
    # cm.ravel() is (tn, fp, fn, tp) — matches sklearn confusion_matrix(labels=[0,1]).
    np.testing.assert_array_equal(cm.ravel(), [2, 1, 1, 2])


def test_sens_spec_matrix_layout() -> None:
    # Rows = true, cols = predicted → [[tn, fp], [fn, tp]].
    _, _, cm = sens_spec(np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]))
    np.testing.assert_array_equal(cm, [[1, 1], [1, 1]])


def test_sens_spec_perfect() -> None:
    sens, spec, _ = sens_spec(np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]))
    assert sens == 1.0 and spec == 1.0


def test_sens_spec_nan_without_positives() -> None:
    # No true freezes → sensitivity undefined (nan), not a divide-by-zero crash.
    sens, spec, _ = sens_spec(np.array([0, 0, 0]), np.array([0, 0, 1]))
    assert np.isnan(sens)
    assert spec == 2 / 3


def test_sens_spec_nan_without_negatives() -> None:
    sens, spec, _ = sens_spec(np.array([1, 1, 1]), np.array([1, 1, 0]))
    assert np.isnan(spec)
    assert sens == 2 / 3


# ── Freeze-Index baseline ────────────────────────────────────────────────────
def _stack_channels_first(windows: list[np.ndarray]) -> np.ndarray:
    """List of (T, C) windows → (N, C, T), the layout the CNN/baseline consume."""
    return np.stack([w.T for w in windows]).astype(np.float32)


def test_freeze_index_predict_separates_classes(
    freeze_window: np.ndarray, walk_window: np.ndarray
) -> None:
    X = _stack_channels_first([freeze_window, walk_window])
    pred = freeze_index_predict(X, threshold=2.0)
    assert pred.tolist() == [1, 0]
    assert pred.dtype == np.int64


def test_best_fi_threshold_in_literature_range(
    make_window,
) -> None:
    freezes = [make_window([(5.0, 0.6)], noise=0.02, seed=s) for s in range(6)]
    walks = [make_window([(1.2, 0.6)], noise=0.02, seed=s + 100) for s in range(6)]
    X = _stack_channels_first(freezes + walks)
    y = np.array([1] * 6 + [0] * 6)
    thr = best_fi_threshold(X, y)
    assert 0.5 <= thr <= 6.0
    # The chosen threshold should actually separate this clean set well.
    sens, spec, _ = sens_spec(y, freeze_index_predict(X, thr))
    assert sens >= 0.8 and spec >= 0.8
