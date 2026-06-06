"""
Engineered rest-tremor detector — the WRIST-monitor Accuracy-Worksheet baseline.

This is the wrist pivot of ``fi_detector.py``. Where the freezing-of-gait baseline
thresholds the Moore et al. (2008) Freeze Index of an ANKLE window, this thresholds
the **4-6 Hz tremor-band power** (``fog.dsp.tremor_power_axes``) of a 4 s WRIST window —
the classic Parkinsonian resting-tremor frequency — with the same two
false-positive mitigations, adapted to the wrist:

  1. a *rest gate* — resting tremor is, by definition, tremor of a limb **at rest**,
     so a window dominated by gross voluntary movement (large 0.5-3 Hz locomotor-band
     power) is rejected: ordinary arm motion must not read as tremor; and
  2. an *onset/offset debounce with hysteresis* — a one-off knock or a single noisy
     window is ignored; "tremor present" only latches on a sustained run and only
     releases after it has clearly stopped.

This device is a *monitor*, not a closed-loop cue: it asserts a "tremor present"
flag and logs the tremor-power trend for the clinician. There is no motor to drive.

  ⚠  ``TREMOR_THRESHOLD`` and ``REST_CEILING`` below are PLACEHOLDERS, not results.
     Calibrate them from YOUR wrist capture: run ``analyze_tremor.py``, read the
     per-phase table, and set the threshold just above the highest tremor-band power
     seen at rest / during normal movement — exactly how the FoG baseline fixed
     FI_THRESHOLD just above walking. Do not quote a number until you have done this.

Torch-free (only ``fog.dsp`` / NumPy / SciPy) so it runs on microcontroller-class
compute, the same as the deployed monitor firmware.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from fog.config import LOCO_BAND, SAMPLE_RATE, WINDOW_HOP, WINDOW_SIZE
from fog.dsp import band_power, magnitude, tremor_power_axes

# ── Parameters — CALIBRATE from analyze_tremor.py on your own wrist capture ──
TREMOR_THRESHOLD = 0.0            # PLACEHOLDER: 4-6 Hz band power above this => tremor.
                                  #   Set just above the max tremor power at rest/move.
REST_CEILING     = float("inf")  # PLACEHOLDER: reject windows whose 0.5-3 Hz locomotor
                                  #   power exceeds this (gross voluntary movement).
ONSET_WINDOWS    = 2             # need N consecutive tremor windows to assert "tremor"
OFFSET_WINDOWS   = 2             # hysteresis: N consecutive clear windows to release


def loco_power(win: np.ndarray, fs: int = SAMPLE_RATE) -> float:
    """0.5-3 Hz locomotor-band power — the 'is the limb being moved?' signal."""
    return band_power(magnitude(win), fs, LOCO_BAND)


@dataclass
class TremorDetection:
    t_centre_s: float
    tremor_power: float
    loco_power: float
    raw_tremor: bool   # this window alone is over threshold (and at rest)
    asserting: bool    # debounced, latched "tremor present" decision


def detect_window(win: np.ndarray,
                  threshold: float = TREMOR_THRESHOLD,
                  rest_ceiling: float = REST_CEILING) -> tuple[bool, float, float]:
    """One ``(T, 3)`` accel window -> ``(raw_tremor, tremor_power, loco_power)``.

    ``raw_tremor`` is True when the limb is at rest (locomotor-band power below
    ``rest_ceiling``) AND the 4-6 Hz tremor-band power is over ``threshold``. The
    rest gate is false-positive mitigation #1: high 4-6 Hz energy produced by gross
    voluntary movement is rejected before it can reach the debounce.
    """
    tp = tremor_power_axes(win)
    lp = loco_power(win)
    at_rest = lp < rest_ceiling
    return bool(at_rest and tp > threshold), float(tp), float(lp)


def detect_stream(windows: Iterable[np.ndarray],
                  fs: int = SAMPLE_RATE,
                  hop: int = WINDOW_HOP,
                  threshold: float = TREMOR_THRESHOLD,
                  rest_ceiling: float = REST_CEILING) -> Iterator[TremorDetection]:
    """Run the debounced tremor detector over an iterable of ``(T, 3)`` windows.

    Yields one :class:`TremorDetection` per window. ``asserting`` is the *debounced*
    decision: a sustained episode (>= ``ONSET_WINDOWS``) latches it on; it releases
    only after ``OFFSET_WINDOWS`` clear windows. Mirrors the debounce in
    ``fi_detector.py``; here it stabilises the logged "tremor present" flag.
    """
    asserting = False
    tremor_run = clear_run = 0
    for i, win in enumerate(windows):
        raw, tp, lp = detect_window(win, threshold, rest_ceiling)
        if raw:
            tremor_run += 1
            clear_run = 0
        else:
            clear_run += 1
            tremor_run = 0

        if not asserting and tremor_run >= ONSET_WINDOWS:
            asserting = True
        elif asserting and clear_run >= OFFSET_WINDOWS:
            asserting = False

        t_centre = (i * hop + WINDOW_SIZE / 2) / fs
        yield TremorDetection(t_centre, tp, lp, raw, asserting)
