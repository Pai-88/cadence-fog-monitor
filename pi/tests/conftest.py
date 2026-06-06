"""Shared pytest fixtures + the torch-skip mechanism.

The whole point of the package split is that the DSP / serial / metrics core is
**torch-free**, so its tests must run on a machine with no deep-learning stack.
Tests that genuinely need torch (the CNN forward pass, the checkpoint arch
round-trip) are marked ``@pytest.mark.needs_torch`` and auto-skipped when torch
is absent — they still run in Colab / on the laptop where torch is installed.

The synthetic-accelerometer fixtures put the oscillation on the gravity (z) axis
so the orientation-invariant magnitude is a clean single tone, which lets the
spectral features land predictably in one band.
"""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from fog.config import NUM_AXES, SAMPLE_RATE, WINDOW_SIZE

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "needs_torch: requires PyTorch (skipped in a torch-free env)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if TORCH_AVAILABLE:
        return
    skip = pytest.mark.skip(reason="torch not installed (torch-free environment)")
    for item in items:
        if "needs_torch" in item.keywords:
            item.add_marker(skip)


# ── synthetic accelerometer windows ─────────────────────────────────────────
def accel_window(
    freqs_amps: list[tuple[float, float]],
    n: int = WINDOW_SIZE,
    fs: int = SAMPLE_RATE,
    base_z: float = 9.81,
    noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Build a ``(n, 3)`` accel window in m/s².

    ``base_z`` is the gravity component on the z-axis; each ``(freq, amp)`` adds a
    sine on top of it, so the mean-removed magnitude is that sum of tones. A small
    ``noise`` keeps both spectral bands non-empty (a pure tone leaves the other
    band at exactly zero, which makes the Freeze-Index ratio blow up).
    """
    t = np.arange(n) / fs
    az = np.full(n, base_z, dtype=np.float64)
    for f, a in freqs_amps:
        az += a * np.sin(2 * np.pi * f * t)
    ax = np.zeros(n)
    ay = np.zeros(n)
    if noise:
        rng = np.random.default_rng(seed)
        ax += rng.normal(0, noise, n)
        ay += rng.normal(0, noise, n)
        az += rng.normal(0, noise, n)
    return np.stack([ax, ay, az], axis=1).astype(np.float32)


@pytest.fixture
def freeze_window() -> np.ndarray:
    """5 Hz dominant → energy in the 3-8 Hz freeze band → high Freeze Index."""
    return accel_window([(5.0, 0.6)], noise=0.02, seed=1)


@pytest.fixture
def walk_window() -> np.ndarray:
    """1.2 Hz dominant → energy in the 0.5-3 Hz loco band → low Freeze Index."""
    return accel_window([(1.2, 0.6)], noise=0.02, seed=2)


@pytest.fixture
def still_window() -> np.ndarray:
    """Near-flat: tiny broadband noise only → movement energy ~ 0.

    Its Freeze Index can still spike (noise / noise), which is exactly why the
    standing-still energy gate exists.
    """
    return accel_window([], noise=0.02, seed=3)


@pytest.fixture
def make_window():
    """Expose :func:`accel_window` to tests that need a custom window."""
    return accel_window


@pytest.fixture
def n_axes() -> int:
    return NUM_AXES
