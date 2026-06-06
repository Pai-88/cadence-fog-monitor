#!/usr/bin/env python3
"""Decide whether a capture actually contains a freeze worth demoing.

The one-click launchers use this to avoid playing a recording whose "freezes"
were really done standing still (no 3-8 Hz signature) — that just shows
walking/still and the live demo falls flat. When this script says a capture has
no clear freeze, the launcher falls back to a known-good capture instead.

Criterion (raw scale, same features the live --detector fi path uses):
  a "clear freeze" = at least MIN_RUN consecutive analysis windows in which the
  raw Freeze Index clears MIN_FI *and* movement energy clears the still-floor.

MIN_FI is set well above the walking band on purpose. A genuine freeze drives
the raw FI into the tens-to-hundreds; ordinary walking transients top out around
~7-10. So a sustained run above ~10 is an unambiguous freeze, whereas a
standing-still "freeze" (FI ~1) or a lone walking spike never qualifies.

Exit 0 = clear freeze present (play this capture).
Exit 1 = no clear freeze (caller should fall back).
Exit 2 = could not read the capture.

Usage:
  python has_freeze.py <name|path> [--min-fi 10] [--still-floor 8000]
                                   [--min-run 2] [--quiet]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from fog.config import WINDOW_HOP, WINDOW_SIZE
from fog.dsp import freeze_index, movement_energy

HERE = Path(__file__).resolve().parent
CAPTURES = HERE.parent / "recordings"


def load_accel_mg(name_or_path: str) -> np.ndarray:
    """Return an (N,3) array of ax,ay,az in milli-g from a capture CSV.

    Accepts a bare capture name ('walk1') or a path. Skips the '#' comment
    header, then reads the ax_mg/ay_mg/az_mg columns by name (falling back to
    fixed columns 2,3,4 if the header is missing).
    """
    p = Path(name_or_path)
    if not p.exists():
        p = CAPTURES / f"{name_or_path}.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    rows = []
    header = None
    with open(p) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue  # truncated row — skip
            if header is None and any(c.isalpha() for c in parts[2]):
                header = parts
                continue
            rows.append(parts)
    if not rows:
        raise ValueError(f"no data rows in {p}")
    if header and {"ax_mg", "ay_mg", "az_mg"} <= set(header):
        ix = [header.index(c) for c in ("ax_mg", "ay_mg", "az_mg")]
    else:
        ix = [2, 3, 4]
    # Validate each row individually (correct width + numeric, finite) so a file
    # with mixed truncated/long rows can't raise the opaque "inhomogeneous shape"
    # np.array error — bad rows are skipped instead.
    accel = []
    need = max(ix) + 1
    for parts in rows:
        if len(parts) < need:
            continue
        try:
            vals = [float(parts[i]) for i in ix]
        except ValueError:
            continue
        if all(np.isfinite(v) for v in vals):
            accel.append(vals)
    if not accel:
        raise ValueError(f"no valid numeric data rows in {p}")
    return np.array(accel, dtype=np.float64)


def clear_freeze(accel_mg: np.ndarray, min_fi: float, still_floor: float,
                 min_run: int) -> tuple[bool, float, int]:
    """Return (present, peak_fi, longest_qualifying_run)."""
    n = accel_mg.shape[0]
    if n < WINDOW_SIZE:
        return False, 0.0, 0
    peak_fi = 0.0
    run = best_run = 0
    n_windows = (n - WINDOW_SIZE) // WINDOW_HOP + 1
    for i in range(n_windows):
        w = accel_mg[i * WINDOW_HOP : i * WINDOW_HOP + WINDOW_SIZE]
        fi = freeze_index(w)
        en = movement_energy(w)
        peak_fi = max(peak_fi, fi)
        if fi > min_fi and en > still_floor:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0
    return best_run >= min_run, peak_fi, best_run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", help="capture name (no extension) or CSV path")
    ap.add_argument("--min-fi", type=float, default=10.0)
    ap.add_argument("--still-floor", type=float, default=8000.0)
    ap.add_argument("--min-run", type=int, default=2)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    try:
        accel = load_accel_mg(a.capture)
    except (FileNotFoundError, ValueError) as e:
        if not a.quiet:
            print(f"has_freeze: cannot read {a.capture}: {e}", file=sys.stderr)
        return 2
    present, peak_fi, run = clear_freeze(accel, a.min_fi, a.still_floor, a.min_run)
    name = Path(a.capture).stem
    if not a.quiet:
        if present:
            print(f"{name}: freeze present (peak FI {peak_fi:.0f}, "
                  f"{run} windows above {a.min_fi:g})")
        elif peak_fi >= a.min_fi:
            print(f"{name}: no SUSTAINED freeze (peak FI {peak_fi:.0f} was a "
                  f"one-off spike, not held for {a.min_run} windows)")
        else:
            print(f"{name}: NO clear freeze (peak FI {peak_fi:.1f}) — looks like "
                  f"walking/standing, not a 3-8 Hz freeze")
    return 0 if present else 1


if __name__ == "__main__":
    raise SystemExit(main())
