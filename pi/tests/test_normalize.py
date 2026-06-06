"""Tests for fog.normalize.Normaliser (per-channel z-score)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fog.config import NUM_AXES, WINDOW_SIZE
from fog.normalize import Normaliser


def _batch(n: int = 20, seed: int = 0) -> np.ndarray:
    """(N, C, T) batch with a different per-channel scale/offset each axis."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, NUM_AXES, WINDOW_SIZE)).astype(np.float32)
    x[:, 0] = x[:, 0] * 5.0 + 3.0
    x[:, 1] = x[:, 1] * 0.2 - 1.0
    return x


def test_fit_returns_self() -> None:
    norm = Normaliser()
    assert norm.fit(_batch()) is norm


def test_transform_zero_mean_unit_std_per_channel() -> None:
    norm = Normaliser().fit(_batch())
    out = norm.transform(_batch())
    means = out.mean(axis=(0, 2))
    stds = out.std(axis=(0, 2))
    np.testing.assert_allclose(means, 0.0, atol=1e-4)
    np.testing.assert_allclose(stds, 1.0, atol=1e-3)


def test_transform_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        Normaliser().transform(_batch())


def test_save_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        Normaliser().save("unused.npz")


def test_save_load_roundtrip(tmp_path: Path) -> None:
    norm = Normaliser().fit(_batch())
    path = str(tmp_path / "fog_norm.npz")
    norm.save(path)
    reloaded = Normaliser.load(path)
    X = _batch(seed=7)
    np.testing.assert_array_equal(norm.transform(X), reloaded.transform(X))


def test_dead_channel_produces_no_nan() -> None:
    # A constant (dead) channel has std 0; the +1e-6 guard must keep transform finite.
    x = _batch()
    x[:, 2] = 4.2
    out = Normaliser().fit(x).transform(x)
    assert np.all(np.isfinite(out))
