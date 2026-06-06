"""Design the on-device band-pass biquads, tune the Freeze-Index threshold, and
emit them as a C++ block for ``cpx_fog_standalone.ino``.

The Circuit Playground Express cannot host the CNN, so it runs the classical
Freeze Index instead: ``FI = power(3-8 Hz) / power(0.5-3 Hz)`` on the accel
magnitude. The trick that makes the threshold transfer from Colab to the board
*exactly* is tuning it on the **same** filter the firmware runs — the
Direct-Form-II-Transposed biquad cascade in ``runBand`` is numerically identical
to ``scipy.signal.sosfilt`` over these second-order sections (the
``pi/tests/test_biquad_equiv.py`` suite guards both that equivalence and that
these constants still match a fresh ``butter`` design).

It prints a drop-in replacement for the ``EXPORTED FROM COLAB`` block in
``firmware/variants/cpx_fog_standalone/cpx_fog_standalone.ino``: the two band-pass SOS
tables (the firmware ``Sos`` struct stores ``{b0, b1, b2, a1, a2}`` with a0
normalised to 1) and the tuned ``FI_THRESHOLD``. Copy everything between the
box markers and paste over the matching region of the sketch.

Run as a script::

    python gen_device_coeffs.py /path/to/daphnet      # defaults to the synth set
"""
from __future__ import annotations

import sys

import numpy as np
import scipy.signal as sg

import fog_allinone as F

FS = 64                      # device sample rate (must match the Daphnet export)
ORDER = 2                    # Butterworth order per band → 2 second-order sections
LOCO = (0.5, 3.0)            # locomotor band (Hz)
FREEZE = (3.0, 8.0)          # freeze band (Hz)

# Designed once at import. SciPy SOS rows are [b0, b1, b2, a0, a1, a2] (a0 == 1);
# the firmware drops a0. Same call the board's coefficients were generated from.
SOS_LOCO = sg.butter(ORDER, LOCO, btype="band", fs=FS, output="sos")
SOS_FREEZE = sg.butter(ORDER, FREEZE, btype="band", fs=FS, output="sos")


# ── tuning ───────────────────────────────────────────────────────────────────
def biquad_fi(window_3xw: np.ndarray) -> float:
    """Freeze Index via the *firmware* band-pass cascade (matches ``runBand``).

    Filters the mean-removed accel magnitude through the very SOS tables the
    board runs, so a threshold tuned on this matches the device's behaviour.
    """
    mag = np.linalg.norm(window_3xw.T.astype(np.float64), axis=1)
    mag = mag - mag.mean()                       # DC / gravity removal
    fr = sg.sosfilt(SOS_FREEZE, mag)
    lo = sg.sosfilt(SOS_LOCO, mag)
    return float(np.sum(fr ** 2) / (np.sum(lo ** 2) + 1e-9))


def tune_threshold(
    X: np.ndarray, y: np.ndarray
) -> tuple[float, float, tuple[float, float]]:
    """Sweep the FI threshold → Youden-J-optimal ``(threshold, J, (sens, spec))``."""
    fis = np.array([biquad_fi(X[i]) for i in range(len(X))])
    best_t, best_j, best_ss = float("nan"), -1.0, (float("nan"), float("nan"))
    for t in np.linspace(0.2, 6.0, 80):
        s, sp, _ = F.sens_spec(y, (fis > t).astype(np.int64))
        j = (0 if np.isnan(s) else s) + (0 if np.isnan(sp) else sp) - 1
        if j > best_j:
            best_j, best_t, best_ss = j, float(t), (s, sp)
    return best_t, best_j, best_ss


# ── C++ emitter ──────────────────────────────────────────────────────────────
_BOX_W = 71  # inner width of the comment box, matching the .ino header


def _comment_box(text: str) -> list[str]:
    """Three ``//``-prefixed lines forming an aligned Unicode box around ``text``."""
    return [
        "// ╔" + "═" * _BOX_W + "╗",
        "// ║" + f"  {text}".ljust(_BOX_W) + "║",
        "// ╚" + "═" * _BOX_W + "╝",
    ]


def _section(row: np.ndarray) -> str:
    """SciPy SOS row [b0,b1,b2,a0,a1,a2] → a firmware ``{b0,b1,b2,a1,a2}`` line."""
    b0, b1, b2, _a0, a1, a2 = row
    vals = ", ".join(f"{c: .8f}f" for c in (b0, b1, b2, a1, a2))
    return f"  {{ {vals} }},"


def _sos_table(name: str, sos: np.ndarray, comment: str) -> str:
    """A full ``const Sos NAME[N_SOS] = { ... };`` declaration."""
    decl = f"const Sos {name}[N_SOS] = {{"
    body = "\n".join(_section(r) for r in sos)
    return f"{decl.ljust(42)}// {comment}\n{body}\n}};"


def format_device_block(threshold: float) -> str:
    """The full C++ ``EXPORTED FROM COLAB`` block, ready to paste into the .ino."""
    lines = [
        *_comment_box(f"EXPORTED FROM COLAB  (gen_device_coeffs.py, tuned at {FS} Hz)"),
        f"#define FS        {FS}           // sample rate — must match the export",
        f"#define WINDOW    {F.WINDOW_SIZE}          "
        f"// analysis window, samples ({F.WINDOW_SIZE / FS:.1f} s)",
        f"#define HOP       {F.WINDOW_HOP}          "
        f"// decision cadence, samples ({F.WINDOW_HOP / FS:.1f} s)",
        f"const float FI_THRESHOLD = {threshold:.3f}f;",
        "",
        "// Butterworth band-pass, order 2 → two second-order sections each.",
        "// Section coefficients are {b0, b1, b2, a1, a2}  (a0 normalised to 1).",
        "struct Sos { float b0, b1, b2, a1, a2; };",
        "",
        f"const uint8_t N_SOS = {ORDER};",
        _sos_table("FREEZE_SOS", SOS_FREEZE, "3–8 Hz"),
        _sos_table("LOCO_SOS", SOS_LOCO, "0.5–3 Hz"),
        "// ── end exported block ─────────────────────────────────────────────────────",
    ]
    return "\n".join(lines)


# ── entry point ──────────────────────────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    data_dir = argv[0] if argv else "/tmp/synth_daphnet10"

    subjects = F.load_daphnet(data_dir, "ankle")
    windows = {s: F.build_windows(r) for s, r in subjects.items()}
    windows = {s: (X, y) for s, (X, y) in windows.items() if len(y) > 0}
    X = np.concatenate([windows[s][0] for s in windows])
    y = np.concatenate([windows[s][1] for s in windows])

    best_t, best_j, (sens, spec) = tune_threshold(X, y)
    fis = np.array([biquad_fi(X[i]) for i in range(len(X))])

    print(f"# data: {data_dir}   FS={FS}  order={ORDER}   {len(y)} windows")
    print(f"# best threshold = {best_t:.3f}  (Youden J={best_j:.3f}; "
          f"sens={sens:.3f} spec={spec:.3f})")
    print(f"# FI median: freeze={np.median(fis[y == 1]):.2f}  "
          f"no-freeze={np.median(fis[y == 0]):.2f}")
    print("# ---- paste the block below over the matching region of "
          "cpx_fog_standalone.ino ----\n")
    print(format_device_block(best_t))


if __name__ == "__main__":
    main()
