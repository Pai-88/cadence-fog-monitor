"""Tests for the torch-free signal-processing core (fog.dsp)."""
from __future__ import annotations

import numpy as np
import pytest

from fog.config import FI_THRESHOLD, NUM_AXES, WINDOW_HOP, WINDOW_SIZE
from fog.dsp import (
    AccelFilter,
    band_power,
    filter_offline,
    freeze_index,
    magnitude,
    movement_energy,
    parse_line,
    tremor_power,
    window_signal,
)


# ── parse_line ───────────────────────────────────────────────────────────────
def test_parse_line_bytes() -> None:
    out = parse_line(b"100,-200,300")
    assert out is not None
    np.testing.assert_array_equal(out, np.array([100, -200, 300], dtype=np.float32))
    assert out.dtype == np.float32


def test_parse_line_str() -> None:
    out = parse_line("1,2,3")
    assert out is not None
    np.testing.assert_array_equal(out, np.array([1, 2, 3], dtype=np.float32))


def test_parse_line_strips_whitespace_and_cr() -> None:
    out = parse_line(b"  4,5,6 \r\n")
    assert out is not None
    np.testing.assert_array_equal(out, np.array([4, 5, 6], dtype=np.float32))


@pytest.mark.parametrize(
    "line",
    [b"1,2", b"1,2,3,4", b"", b"boot banner", b"a,b,c", b"C", "1,2"],
)
def test_parse_line_malformed_returns_none(line: bytes | str) -> None:
    # Wrong arity, the boot banner, a stray cue-ack — all rejected as None.
    assert parse_line(line) is None


# ── magnitude ────────────────────────────────────────────────────────────────
def test_magnitude_is_mean_removed(walk_window: np.ndarray) -> None:
    mag = magnitude(walk_window)
    assert mag.shape == (WINDOW_SIZE,)
    assert abs(float(mag.mean())) < 1e-5


def test_magnitude_orientation_invariant(make_window) -> None:
    # Swapping which axis carries the signal must not change the magnitude:
    # that orientation-invariance is why the features use |a|, not a single axis.
    w = make_window([(5.0, 0.6)], noise=0.0)
    rotated = w[:, [2, 0, 1]]  # cyclic axis permutation
    np.testing.assert_allclose(magnitude(w), magnitude(rotated), atol=1e-5)


# ── band_power ───────────────────────────────────────────────────────────────
def test_band_power_concentrated_in_tone_band(make_window) -> None:
    mag = magnitude(make_window([(5.0, 0.6)], noise=0.0))
    in_band = band_power(mag, 64, (3.0, 8.0))
    out_band = band_power(mag, 64, (0.5, 3.0))
    assert in_band > 10 * out_band


def test_band_power_empty_band_is_zero(walk_window: np.ndarray) -> None:
    # A band entirely above Nyquist (32 Hz) captures no bins → 0.0, not a crash.
    assert band_power(magnitude(walk_window), 64, (40.0, 60.0)) == 0.0


# ── freeze_index / tremor / energy ───────────────────────────────────────────
def test_freeze_index_high_for_freeze(freeze_window: np.ndarray) -> None:
    assert freeze_index(freeze_window) > FI_THRESHOLD


def test_freeze_index_low_for_walk(walk_window: np.ndarray) -> None:
    assert freeze_index(walk_window) < FI_THRESHOLD


def test_movement_energy_orders_moving_above_still(
    freeze_window: np.ndarray, walk_window: np.ndarray, still_window: np.ndarray
) -> None:
    e_still = movement_energy(still_window)
    assert movement_energy(freeze_window) > e_still
    assert movement_energy(walk_window) > e_still


def test_tremor_power_nonnegative(freeze_window: np.ndarray) -> None:
    assert tremor_power(freeze_window) >= 0.0


# ── window_signal ────────────────────────────────────────────────────────────
def test_window_signal_shape_and_count() -> None:
    T = WINDOW_SIZE + 3 * WINDOW_HOP        # exactly 4 windows
    x = np.random.default_rng(0).standard_normal((T, NUM_AXES)).astype(np.float32)
    win = window_signal(x)
    assert win.shape == (4, NUM_AXES, WINDOW_SIZE)


def test_window_signal_too_short_is_empty() -> None:
    x = np.zeros((WINDOW_SIZE - 1, NUM_AXES), dtype=np.float32)
    win = window_signal(x)
    assert win.shape == (0, NUM_AXES, WINDOW_SIZE)


# ── filtering ────────────────────────────────────────────────────────────────
def test_filter_offline_removes_dc_offset() -> None:
    # A constant 9.81 g offset is gravity; the 0.5 Hz high-pass edge must kill it.
    x = np.full((WINDOW_SIZE, NUM_AXES), 9.81, dtype=np.float32)
    y = filter_offline(x)
    assert y.shape == x.shape
    assert np.max(np.abs(y)) < 1e-3


def test_accelfilter_streaming_matches_single_chunk(make_window) -> None:
    # Feeding the streaming filter sample-by-sample must equal filtering the whole
    # chunk in one call — i.e. the carried state (zi) removes block-edge artefacts.
    x = make_window([(2.0, 0.5), (5.0, 0.3)], noise=0.01)
    one_shot = AccelFilter().apply(x)
    f = AccelFilter()
    streamed = np.vstack([f.apply(row) for row in x])
    np.testing.assert_allclose(one_shot, streamed, atol=1e-5)
