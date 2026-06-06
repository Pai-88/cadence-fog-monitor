"""The standing-still movement-energy gate — the shared correctness story.

All three deployment paths (firmware, stream_demo.py, dashboard_server.py) make
the SAME gated decision, so it is tested once here against the torch-free
features:

    moving   = movement_energy(window) > still_floor
    detector = freeze_index(window) > FI_THRESHOLD       # FI fallback path
    freeze   = moving AND detector
    state    = STILL if not moving else FREEZE if detector else WALKING

The case that matters: a wearer standing still produces near-zero movement
energy but a Freeze Index that can spike (noise / noise). Without the gate that
is a false "freeze"; with a calibrated floor it is correctly STILL.
"""
from __future__ import annotations

import numpy as np

from fog.config import FI_THRESHOLD
from fog.dsp import freeze_index, movement_energy

# A floor between quiet-standing energy (~1e-4) and walking/freeze energy (~0.18).
STILL_FLOOR = 0.05


def gate_decision(window: np.ndarray, still_floor: float) -> tuple[str, bool]:
    """Reproduce the deployed gate; return (state, is_freeze)."""
    moving = movement_energy(window) > still_floor
    detector_freeze = freeze_index(window) > FI_THRESHOLD
    is_freeze = moving and detector_freeze
    state = "STILL" if not moving else ("FREEZE" if detector_freeze else "WALKING")
    return state, is_freeze


def test_freeze_while_moving_fires(freeze_window: np.ndarray) -> None:
    state, is_freeze = gate_decision(freeze_window, STILL_FLOOR)
    assert state == "FREEZE"
    assert is_freeze is True


def test_walking_is_not_freeze(walk_window: np.ndarray) -> None:
    state, is_freeze = gate_decision(walk_window, STILL_FLOOR)
    assert state == "WALKING"
    assert is_freeze is False


def test_standing_still_is_gated_despite_high_fi(still_window: np.ndarray) -> None:
    # Premise: the detector WOULD fire on quiet standing (FI spikes on noise)...
    assert freeze_index(still_window) > FI_THRESHOLD
    assert movement_energy(still_window) < STILL_FLOOR
    # ...but the energy gate overrides it to STILL — no false cue.
    state, is_freeze = gate_decision(still_window, STILL_FLOOR)
    assert state == "STILL"
    assert is_freeze is False


def test_gate_off_lets_standing_still_false_positive_through(
    still_window: np.ndarray,
) -> None:
    # Documents WHY the floor must be calibrated: floor=0 disables the gate, and
    # the same quiet-standing window is then misclassified as a freeze.
    _, is_freeze = gate_decision(still_window, still_floor=0.0)
    assert is_freeze is True


def test_floor_sits_between_still_and_moving(
    freeze_window: np.ndarray, walk_window: np.ndarray, still_window: np.ndarray
) -> None:
    e_still = movement_energy(still_window)
    e_freeze = movement_energy(freeze_window)
    e_walk = movement_energy(walk_window)
    assert e_still < STILL_FLOOR < min(e_freeze, e_walk)
