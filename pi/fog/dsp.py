"""Signal-processing core: serial parsing, filtering, spectral features, windowing.

Everything here is **torch-free** — it depends only on NumPy and SciPy, so the
firmware-equivalent DSP can be imported, unit-tested and reasoned about without
dragging in the deep-learning stack. The engineered features (Freeze Index,
tremor power, movement energy) are the explainable clinical baseline and the
live display/gate metrics; they operate on the orientation-invariant accel
*magnitude*, so they are robust to how the garment rotates on the limb.
"""
from __future__ import annotations

import numpy as np
import scipy.signal as signal

from .config import (
    FREEZE_BAND,
    LOCO_BAND,
    NUM_AXES,
    SAMPLE_RATE,
    TREMOR_BAND,
    WINDOW_HOP,
    WINDOW_SIZE,
)

__all__ = [
    "parse_line",
    "filter_offline",
    "AccelFilter",
    "magnitude",
    "band_power",
    "freeze_index",
    "tremor_power",
    "tremor_power_axes",
    "movement_energy",
    "window_signal",
]


# ── Serial line parsing ─────────────────────────────────────────────────────
def parse_line(line: bytes | bytearray | str) -> np.ndarray | None:
    """Decode one ``ax,ay,az`` serial line of int16 milli-g.

    Returns a ``(3,)`` float32 array in milli-g, or ``None`` if the line is
    malformed (a partial line, the boot banner, a stray cue-ack, etc.).
    """
    try:
        text = line.decode() if isinstance(line, (bytes, bytearray)) else line
        parts = text.strip().split(",")
        if len(parts) != NUM_AXES:
            return None
        return np.array([float(p) for p in parts], dtype=np.float32)
    except (ValueError, AttributeError, UnicodeDecodeError):
        return None


# ── Filtering ───────────────────────────────────────────────────────────────
def filter_offline(x: np.ndarray, fs: int = SAMPLE_RATE) -> np.ndarray:
    """Zero-phase band-pass (0.5-15 Hz) for offline analysis / training.

    The low edge kills the gravity / orientation DC component; the high edge sits
    well above the 3-8 Hz freeze band and 4-6 Hz tremor, so nothing of interest
    is lost while sensor noise above it is suppressed. ``filtfilt`` makes it
    zero-phase, which only the offline path can afford.
    """
    nyq = fs / 2
    b, a = signal.butter(4, [0.5 / nyq, 15.0 / nyq], btype="band")
    return signal.filtfilt(b, a, x.astype(np.float64), axis=0).astype(np.float32)


class AccelFilter:
    """Streaming counterpart of :func:`filter_offline` for the live stream.

    Maintains per-axis filter state (``zi``) so the band-pass can be applied
    chunk-by-chunk without restarting — no edge artefacts at chunk boundaries.
    Causal (``lfilter``), unlike the zero-phase offline filter, because a live
    cue cannot look into the future.
    """

    def __init__(self, fs: int = SAMPLE_RATE, num_axes: int = NUM_AXES) -> None:
        nyq = fs / 2
        self.b, self.a = signal.butter(4, [0.5 / nyq, 15.0 / nyq], btype="band")
        n = max(len(self.a), len(self.b)) - 1
        self.zi = np.zeros((n, num_axes), dtype=np.float64)

    def apply(self, x: np.ndarray) -> np.ndarray:
        """Filter a ``(C,)`` sample or ``(T, C)`` chunk, advancing the state."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        x, self.zi = signal.lfilter(self.b, self.a, x, axis=0, zi=self.zi)
        return x.astype(np.float32)


# ── Engineered features — the explainable clinical baseline ─────────────────
def magnitude(window: np.ndarray) -> np.ndarray:
    """``(T, C)`` accel window → ``(T,)`` magnitude with its mean removed.

    Magnitude is orientation-invariant: it does not matter how the garment
    rotates on the body, which a single per-axis signal cannot promise. Removing
    the mean strips the residual gravity / DC offset before spectral analysis.
    """
    mag = np.linalg.norm(np.asarray(window, dtype=np.float64), axis=1)
    return mag - mag.mean()


def band_power(sig1d: np.ndarray, fs: int, band: tuple[float, float]) -> float:
    """Power within ``band`` Hz of a 1-D signal, via the Welch PSD."""
    sig1d = np.asarray(sig1d, dtype=np.float64)
    nperseg = min(len(sig1d), 256)
    f, pxx = signal.welch(sig1d, fs=fs, nperseg=nperseg)
    lo, hi = band
    mask = (f >= lo) & (f < hi)
    if not mask.any():
        return 0.0
    df = float(f[1] - f[0]) if len(f) > 1 else 1.0
    return float(np.sum(pxx[mask]) * df)


def freeze_index(window: np.ndarray, fs: int = SAMPLE_RATE) -> float:
    """Moore et al. (2008) Freeze Index on the accel magnitude.

        FI = power(3-8 Hz) / power(0.5-3 Hz)

    High when fast oscillation dominates slow locomotion → a freeze episode.
    Used as the engineered baseline AND as a live display metric.
    """
    mag = magnitude(window)
    loco = band_power(mag, fs, LOCO_BAND)
    return band_power(mag, fs, FREEZE_BAND) / (loco + 1e-9)


def tremor_power(window: np.ndarray, fs: int = SAMPLE_RATE) -> float:
    """4-6 Hz band power on the accel magnitude — a rest-tremor severity proxy.

    Note: the magnitude is orientation-invariant but *squares* the signal, so a
    tremor oscillating perpendicular to gravity partly cancels (its linear 4-6 Hz
    term collapses, leaving a 2x harmonic outside the band). For the ANKLE FoG
    surrogate this is fine; for an orientation-unknown WRIST, prefer
    :func:`tremor_power_axes`, which does not have this blind spot.
    """
    return band_power(magnitude(window), fs, TREMOR_BAND)


def tremor_power_axes(window: np.ndarray, fs: int = SAMPLE_RATE) -> float:
    """4-6 Hz band power summed over the three raw axes — orientation-robust.

    The wrist-monitor tremor feature. Unlike :func:`tremor_power`, which uses the
    accel *magnitude*, this band-passes each mean-removed axis independently and
    sums the 4-6 Hz power. Because it never squares the signal before the spectral
    estimate, a resting tremor is captured whatever its direction relative to
    gravity, and a low-frequency voluntary movement (whose pure tone has no 4-6 Hz
    content) does not leak in through a magnitude harmonic. This is the rest-tremor
    severity proxy the wrist detector and worksheet use.
    """
    w = np.asarray(window, dtype=np.float64)
    w = w - w.mean(axis=0, keepdims=True)
    return float(sum(band_power(w[:, c], fs, TREMOR_BAND) for c in range(w.shape[1])))


def movement_energy(window: np.ndarray, fs: int = SAMPLE_RATE) -> float:
    """Total accel-magnitude power in the locomotor + freeze bands (0.5-8 Hz).

    This is the "is the wearer actually moving?" signal for the standing-still
    gate. The Freeze Index is a RATIO, so when someone stands quietly both of its
    bands collapse to sensor noise and FI (or an out-of-distribution CNN) can
    spike into a false "freeze". Requiring ``movement_energy`` above a calibrated
    floor before accepting a freeze rejects that case. Mirrors ``(Pf + Pl)`` in
    the on-device firmware (cpx_fog_standalone.ino).
    """
    mag = magnitude(window)
    return band_power(mag, fs, LOCO_BAND) + band_power(mag, fs, FREEZE_BAND)


# ── Offline windowing ───────────────────────────────────────────────────────
def window_signal(
    x: np.ndarray, window_size: int = WINDOW_SIZE, hop: int = WINDOW_HOP
) -> np.ndarray:
    """Slide a window across a ``(T, C)`` signal → ``(N, C, window_size)``.

    Returns an empty ``(0, C, window_size)`` array when the signal is shorter
    than one window, so callers can concatenate results unconditionally.
    """
    n_samples = x.shape[0]
    if n_samples < window_size:
        return np.empty((0, x.shape[1], window_size), dtype=np.float32)
    n_windows = (n_samples - window_size) // hop + 1
    return np.stack(
        [x[i * hop : i * hop + window_size].T for i in range(n_windows)]
    ).astype(np.float32)
