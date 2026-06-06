"""The on-device band-pass cascade must equal SciPy's — and the right filter.

cpx_fog_standalone.ino runs a hand-written Direct-Form-II-Transposed biquad
cascade (``runBand``) over two hard-coded second-order-section tables exported
from Colab. Two things have to hold for the firmware Freeze Index to match the
offline analysis:

  1. ``runBand`` must be numerically identical to ``scipy.signal.sosfilt`` (same
     DF-II-T recurrence) — otherwise the board filters differently from training.
  2. The exported SOS constants must still equal a fresh ``butter`` design at
     64 Hz — a guard so an accidental edit to the .ino table is caught here.

If either fails, regenerate the table with colab/gen_device_coeffs.py and paste.
"""
from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as signal

# Verbatim from cpx_fog_standalone.ino — {b0, b1, b2, a1, a2}, a0 == 1.
FREEZE_SOS = np.array([
    [0.04427971,  0.08855942, 0.04427971, -1.24067070, 0.62897179],
    [1.00000000, -2.00000000, 1.00000000, -1.69751871, 0.79495806],
])
LOCO_SOS = np.array([
    [0.01278734,  0.02557469, 0.01278734, -1.69059402, 0.75072617],
    [1.00000000, -2.00000000, 1.00000000, -1.93848688, 0.94143123],
])
FS = 64
NYQ = FS / 2


def to_scipy_sos(fw: np.ndarray) -> np.ndarray:
    """Firmware {b0,b1,b2,a1,a2} → SciPy [b0,b1,b2,a0=1,a1,a2]."""
    out = np.zeros((fw.shape[0], 6))
    out[:, :3] = fw[:, :3]
    out[:, 3] = 1.0
    out[:, 4:] = fw[:, 3:]
    return out


def runband_py(fw: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Pure-Python replica of the firmware ``runBand`` DF-II-T cascade."""
    n_sos = fw.shape[0]
    z1 = np.zeros(n_sos)
    z2 = np.zeros(n_sos)
    y = np.empty_like(x, dtype=np.float64)
    for i in range(len(x)):
        inp = float(x[i])
        for s in range(n_sos):
            b0, b1, b2, a1, a2 = fw[s]
            out = b0 * inp + z1[s]
            z1[s] = b1 * inp - a1 * out + z2[s]
            z2[s] = b2 * inp - a2 * out
            inp = out
        y[i] = inp
    return y


@pytest.fixture
def signal_in() -> np.ndarray:
    rng = np.random.default_rng(0)
    chirp = signal.chirp(np.arange(256) / FS, f0=0.5, f1=12, t1=4.0)
    return chirp + 0.1 * rng.standard_normal(256)


@pytest.mark.parametrize("name", ["freeze", "loco"])
def test_firmware_cascade_matches_sosfilt(name: str, signal_in: np.ndarray) -> None:
    fw = FREEZE_SOS if name == "freeze" else LOCO_SOS
    y_fw = runband_py(fw, signal_in)
    y_sp = signal.sosfilt(to_scipy_sos(fw), signal_in)
    np.testing.assert_allclose(y_fw, y_sp, atol=1e-9)


@pytest.mark.parametrize("name", ["freeze", "loco"])
def test_band_power_sum_of_squares_matches(name: str, signal_in: np.ndarray) -> None:
    # runBand returns Σy² (the band power, up to a constant); it must agree.
    fw = FREEZE_SOS if name == "freeze" else LOCO_SOS
    y_sp = signal.sosfilt(to_scipy_sos(fw), signal_in)
    assert np.isclose(float((runband_py(fw, signal_in) ** 2).sum()),
                      float((y_sp ** 2).sum()), rtol=1e-9)


def test_exported_constants_match_butter_design() -> None:
    # Guard the .ino tables: a 2nd-order Butterworth band-pass is two SOS sections.
    np.testing.assert_allclose(
        to_scipy_sos(FREEZE_SOS),
        signal.butter(2, [3.0 / NYQ, 8.0 / NYQ], btype="band", output="sos"),
        atol=1e-6,
    )
    np.testing.assert_allclose(
        to_scipy_sos(LOCO_SOS),
        signal.butter(2, [0.5 / NYQ, 3.0 / NYQ], btype="band", output="sos"),
        atol=1e-6,
    )
