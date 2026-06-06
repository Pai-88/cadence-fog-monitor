"""Shared still / walk / freeze inference — the single source of truth.

Every offline analysis tool (infer_capture, predict_vs_truth, capture_report)
used to carry its own copy of the windowing + Otsu floor + debounce logic, and
they had drifted apart (different still-floors, raw-vs-committed mismatches). This
module collapses all of that into one place so the tools agree with each other —
and with the firmware, which stores the *debounced* state.

The decision, per 4 s / 256-sample window (2 s / 128-sample hop):

    moving = movement_energy > still_floor          (still_floor found unsupervised)
    state  = still   if not moving
             freeze  if moving and FreezeIndex > FI_THRESHOLD   (3-8 Hz trembling)
             walk    otherwise
    then a 2-window debounce stabilises it into the *committed* episodes the board
    stores.

Nothing here changes the DSP math (fog.dsp) or the project constants (fog.config);
it only orchestrates them.
"""
from __future__ import annotations

import numpy as np

from .config import FI_THRESHOLD, SAMPLE_RATE, WINDOW_HOP, WINDOW_SIZE
from .dsp import freeze_index, movement_energy

FS = SAMPLE_RATE
DEBOUNCE = 2
PRETTY = {"still": "STILL", "walk": "WALKING", "freeze": "FREEZE"}

# ── Empirical still-floor (movement_energy units, mg^2) ──────────────────────
# movement_energy = locomotor + freeze band power on the mean-removed accel
# magnitude (mg^2). Measured per-window on the two host-path captures the audit
# named (recordings/smoketest.csv and recordings/capture_20260605_103351.csv):
#
#   smoketest    still:  med   847, p90  2638, max 7521  | walk: p10 14368, med 39719
#   capture_…    still:  med   655                       | walk:           med 91993
#
# So the still-noise ceiling on a clean scripted capture (smoketest) tops out
# near ~2.6k mg^2, while walking sits ~14k-90k mg^2. FLOOR_MIN = 3000 sits just
# above the still p90 and an order of magnitude below typical walking, so a
# uniform-still recording never crosses it (no spurious "walk") while real
# walking clears it comfortably. It is the absolute floor the unsupervised Otsu
# split is clamped to, so an all-still capture cannot be split into fake bands.
FLOOR_MIN: float = 3000.0

__all__ = [
    "FS",
    "DEBOUNCE",
    "PRETTY",
    "FLOOR_MIN",
    "still_floor",
    "windows",
    "infer",
    "spans",
    "freeze_episodes",
]


def still_floor(energy: np.ndarray) -> float:
    """Unsupervised still/moving energy split, stabilised against false splits.

    Otsu's threshold on ``log10(energy)`` finds the natural valley between the
    still cluster and the moving cluster. But on a *uniform* recording (all
    still, or all walking) there is no valley — Otsu will still report some cut,
    inventing a fake still/walk boundary. We guard that two ways:

      * if the energy is effectively unimodal — less than ~1.5 decades of spread,
        or Otsu lands at an extreme bin — there is no real split, so fall back to
        the absolute :data:`FLOOR_MIN`;
      * otherwise return ``max(otsu, FLOOR_MIN)`` so the threshold can rise above
        the floor for a noisy still baseline but never sink below it.
    """
    energy = np.asarray(energy, dtype=float)
    if energy.size == 0:
        return FLOOR_MIN
    le = np.log10(np.clip(energy, 1.0, None))
    spread = float(le.max() - le.min())
    # < ~1.5 decades of dynamic range ⇒ effectively unimodal ⇒ no real split.
    if spread < 1.5:
        return FLOOR_MIN

    hist, edges = np.histogram(le, bins=64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    total = hist.sum()
    if total == 0:
        return FLOOR_MIN
    cw = np.cumsum(hist)
    cs = np.cumsum(hist * centers)
    best_i, best_t, best_var = 0, centers[0], -1.0
    for i in range(1, len(hist)):
        w0, w1 = cw[i - 1], total - cw[i - 1]
        if w0 == 0 or w1 == 0:
            continue
        m0 = cs[i - 1] / w0
        m1 = (cs[-1] - cs[i - 1]) / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t, best_i = var, centers[i], i
    # Otsu landing in the first/last few bins means the "split" is just clipping
    # one tail off — not a genuine bimodal valley. Treat as unimodal.
    if best_i <= 1 or best_i >= len(hist) - 1:
        return FLOOR_MIN
    return max(float(10 ** best_t), FLOOR_MIN)


def windows(accel: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per 256/128 window → (t_centre, freeze_index, movement_energy) arrays.

    Guards short signals: if ``accel`` has fewer than ``WINDOW_SIZE`` samples (so
    no full window fits) it returns three empty arrays instead of crashing on
    ``zip(*[])``.
    """
    accel = np.asarray(accel, dtype=float)
    n = accel.shape[0] if accel.ndim >= 1 else 0
    rows = []
    for s in range(0, n - WINDOW_SIZE + 1, WINDOW_HOP):
        w = accel[s:s + WINDOW_SIZE]
        rows.append((float((s + WINDOW_SIZE / 2) / FS),
                     float(freeze_index(w)), float(movement_energy(w))))
    if not rows:
        return np.empty(0), np.empty(0), np.empty(0)
    tc, fi, en = (np.array(c) for c in zip(*rows, strict=True))
    return tc, fi, en


def infer(fi: np.ndarray, en: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    """(fi, en, floor) → (raw, committed) per-window state arrays.

    ``raw`` is the gated instantaneous state (``still`` if energy ≤ floor, else
    ``freeze`` if FI > FI_THRESHOLD, else ``walk``). ``committed`` is the same
    state after a 2-window (DEBOUNCE) hysteresis: a new state must persist for
    DEBOUNCE windows before it latches, which is exactly what the firmware
    stores. Returns string arrays of lowercase labels.
    """
    fi = np.asarray(fi, dtype=float)
    en = np.asarray(en, dtype=float)
    if fi.size == 0:
        empty = np.empty(0, dtype="<U6")
        return empty, empty.copy()
    raw = np.where(en <= floor, "still",
                   np.where(fi > FI_THRESHOLD, "freeze", "walk"))
    committed, pend, run, out = "walk", None, 0, []
    for r in raw:
        run = run + 1 if r == pend else 1
        pend = r
        if run >= DEBOUNCE:
            committed = r
        out.append(committed)
    return raw, np.array(out)


def spans(tc: np.ndarray, state: np.ndarray) -> list[tuple[float, float, str]]:
    """Collapse a per-window state array into contiguous (t0, t1, label) spans."""
    out: list[tuple[float, float, str]] = []
    if len(state) == 0:
        return out
    start = 0
    half = WINDOW_HOP / FS / 2
    for i in range(1, len(state) + 1):
        if i == len(state) or state[i] != state[start]:
            out.append((tc[start] - half, tc[i - 1] + half, state[start]))
            start = i
    return out


def freeze_episodes(tc: np.ndarray, state: np.ndarray,
                    fi: np.ndarray) -> list[tuple[float, float, float]]:
    """Contiguous freeze runs → (t_start, t_end, peak_FI) episodes."""
    eps: list[tuple[float, float, float]] = []
    start = None
    for i, s in enumerate(state):
        if s == "freeze" and start is None:
            start = i
        elif s != "freeze" and start is not None:
            eps.append((tc[start], tc[i - 1], float(fi[start:i].max())))
            start = None
    if start is not None:
        eps.append((tc[start], tc[-1], float(fi[start:].max())))
    return eps


if __name__ == "__main__":  # pragma: no cover — tiny self-check
    rng = np.random.default_rng(0)
    fake_en = np.concatenate([rng.uniform(100, 1000, 20),
                              rng.uniform(2e4, 1e5, 20)])
    fake_fi = rng.uniform(0.1, 1.0, 40)
    fl = still_floor(fake_en)
    raw, com = infer(fake_fi, fake_en, fl)
    print(f"floor={fl:.0f}  raw still={int((raw == 'still').sum())}  "
          f"committed still={int((com == 'still').sum())}")
    # unimodal all-still ⇒ must fall back to FLOOR_MIN, no split
    flat = still_floor(rng.uniform(100, 300, 40))
    assert flat == FLOOR_MIN, flat
    # empty / short signal guard
    tc, fi, en = windows(np.zeros((10, 3)))
    assert tc.size == 0 and fi.size == 0 and en.size == 0
    print("self-check OK")
