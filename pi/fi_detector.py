"""
Engineered Freeze-of-Gait detector — the Accuracy-Worksheet baseline (Task 5).

This is the threshold detector evaluated in Tasks 3-5: it flags a freeze when
the Moore et al. (2008) Freeze Index of a 4 s window crosses ``FI_THRESHOLD``,
with two false-positive mitigations baked in:

  1. a *movement-energy gate* — a freeze is trembling **while trying to move**,
     so a still limb (both spectral bands collapse to sensor noise) can never
     trigger a cue, however the ratio behaves; and
  2. an *onset/offset debounce with hysteresis* — a one-off bump or a single
     noisy window is ignored; the cue only latches on a sustained episode and
     only releases after it has clearly cleared.

``FI_THRESHOLD = 1.815`` is the deployed operating point. The clean
sensitivity + specificity optimum on the worksheet capture (``capture1.csv``) is
FI = 2.10 — just above the highest Freeze Index seen during normal walking (2.08),
giving 100 % specificity — but the board ships the lower 1.815 (92 % specificity
there) to stay biased toward sensitivity on harder, real-world gait; the energy
gate and debounce below absorb the few extra false positives that costs.

Deliberately torch-free — only ``fog.dsp`` (NumPy/SciPy) — so it runs on
microcontroller-class compute. The shipped garment runs this alongside the
FoGNet 1-D CNN (see ``stream_demo.py``); the CNN is the "given more time"
upgrade discussed in Task 4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np

from fog.config import SAMPLE_RATE, WINDOW_HOP, WINDOW_SIZE
from fog.dsp import freeze_index, movement_energy

# ── Parameters fixed in Tasks 2 & 3 ─────────────────────────────────────────
FI_THRESHOLD   = 1.815  # deployed operating point (sensitivity-biased; clean optimum 2.10)
ENERGY_FLOOR   = 300.0  # FP mitigation 1: gate out standing-still (rest E ~130-270)
ONSET_WINDOWS  = 2      # FP mitigation 2: need N consecutive freeze windows to fire
OFFSET_WINDOWS = 2      # hysteresis: need N consecutive clear windows to release


@dataclass
class Detection:
    t_centre_s: float
    freeze_index: float
    energy: float
    raw_freeze: bool   # this window alone is over threshold (and moving)
    cueing: bool       # debounced, latched decision that drives the motor


def detect_window(win: np.ndarray,
                  energy_floor: float = ENERGY_FLOOR) -> tuple[bool, float, float]:
    """One ``(T, 3)`` accel window -> ``(raw_freeze, FI, energy)``.

    ``raw_freeze`` is True when the wearer is moving AND the Freeze Index is over
    threshold. The energy gate is false-positive mitigation #1: a high FI on an
    essentially still limb is rejected before it can ever reach the debounce.
    """
    fi = freeze_index(win)
    energy = movement_energy(win)
    moving = energy > energy_floor
    return bool(moving and fi > FI_THRESHOLD), float(fi), float(energy)


def detect_stream(windows: Iterable[np.ndarray],
                  fs: int = SAMPLE_RATE,
                  hop: int = WINDOW_HOP) -> Iterator[Detection]:
    """Run the debounced detector over an iterable of ``(T, 3)`` windows.

    Yields one :class:`Detection` per window. ``cueing`` is the *debounced*
    decision: a sustained episode (>= ``ONSET_WINDOWS``) latches it on; it only
    releases after ``OFFSET_WINDOWS`` clear windows. This is false-positive
    mitigation #2 and mirrors ``CueController`` in ``stream_demo.py``.
    """
    cueing = False
    freeze_run = clear_run = 0
    for i, win in enumerate(windows):
        raw, fi, energy = detect_window(win)
        if raw:
            freeze_run += 1
            clear_run = 0
        else:
            clear_run += 1
            freeze_run = 0

        if not cueing and freeze_run >= ONSET_WINDOWS:
            cueing = True
        elif cueing and clear_run >= OFFSET_WINDOWS:
            cueing = False

        t_centre = (i * hop + WINDOW_SIZE / 2) / fs
        yield Detection(t_centre, fi, energy, raw, cueing)
