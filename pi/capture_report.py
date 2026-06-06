#!/usr/bin/env python3
"""Three extra views of a Cadence capture, on top of plot_capture.py:

  1. interactive Plotly HTML   — ax/ay/az vs t, phase bands, hover + zoom
  2. freeze close-up PNG       — a few seconds of one freeze + its power spectrum
  3. detector-overlay PNG      — the device's Freeze Index + STILL/WALK/FREEZE
                                 decision vs the ground-truth phases (real numbers)

    python capture_report.py capture_20260605_103351

Everything is computed with the project's OWN dsp (fog.dsp), so the Freeze Index
and the gated, debounced state are exactly what the device would output.
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import welch

from fog.config import FI_THRESHOLD, WINDOW_HOP, WINDOW_SIZE
from fog.detect import FS, infer, still_floor, windows
from plot_capture import BAND, EDGE, LINE, RECORDINGS, load_capture


def truth_per_window(phase, labels, n):
    """Dominant ground-truth category index per analysis window."""
    out = []
    for k in range(n):
        s = k * WINDOW_HOP
        seg = np.clip(phase[s:s + WINDOW_SIZE].astype(int), 0, None)
        out.append(int(np.bincount(seg).argmax()))
    return np.array(out, dtype=int)


def interactive_html(t, ax, ay, az, cat, title, out):
    import plotly.graph_objects as go
    fig = go.Figure()
    # phase bands
    seen, start = set(), 0
    for i in range(1, len(cat) + 1):
        if i == len(cat) or cat[i] != cat[start]:
            c = cat[start]
            fig.add_vrect(x0=t[start], x1=(t[i] if i < len(cat) else t[-1]),
                          fillcolor=BAND.get(c, "#fff"), opacity=0.5,
                          layer="below", line_width=0)
            seen.add(c)
            start = i
    for nm, y, col in [("ax", ax, LINE["ax_mg"]), ("ay", ay, LINE["ay_mg"]),
                       ("az", az, LINE["az_mg"])]:
        fig.add_trace(go.Scattergl(x=t, y=y, name=nm, mode="lines",
                                   line=dict(width=1, color=col)))
    # legend swatches for the phase bands
    for c in ["still", "walk", "freeze"]:
        if c in seen:
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                          marker=dict(size=12, color=BAND[c], line=dict(color=EDGE[c], width=1)),
                          name=c.upper() + " (phase)"))
    fig.update_layout(title=title, template="plotly_white",
                      xaxis_title="time (s)", yaxis_title="acceleration (mg)",
                      legend=dict(orientation="h", y=-0.18), hovermode="x unified",
                      margin=dict(l=60, r=30, t=60, b=80))
    fig.write_html(out, include_plotlyjs="cdn")


def freeze_zoom(accel, phase, fi_tc, fi_vals, labels, out):
    """Close-up of the single highest-FI freeze window + its power spectrum.

    Returns ``None`` if the capture has no freeze-phase window — the caller then
    skips this panel rather than mislabelling window 0 as a fake freeze.
    """
    # pick the freeze-phase window with the highest FI
    best_i, best_fi = -1, -1.0
    for k in range(len(fi_tc)):
        s = k * WINDOW_HOP
        seg = np.clip(phase[s:s + WINDOW_SIZE].astype(int), 0, None)
        dom = int(np.bincount(seg).argmax())
        if 0 <= dom < len(labels) and labels[dom] == "freeze" and fi_vals[k] > best_fi:
            best_fi, best_i = fi_vals[k], k
    if best_i < 0:
        return None
    s = best_i * WINDOW_HOP
    w = accel[s:s + WINDOW_SIZE]
    wmag = np.linalg.norm(w, axis=1)
    wmag = wmag - wmag.mean()
    tw = np.arange(WINDOW_SIZE) / FS
    f, pxx = welch(wmag, fs=FS, nperseg=WINDOW_SIZE)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 4.6))
    a1.plot(tw, wmag, color="#C0392B", lw=1.3)
    a1.set_title(f"Freeze close-up: 4 s window @ t≈{fi_tc[best_i]:.0f}s  (FI = {best_fi:.1f})",
                 color="#1E2761", fontweight="bold")
    a1.set_xlabel("time within window (s)")
    a1.set_ylabel("mean-removed |accel| (mg)")
    a1.grid(alpha=0.3)
    a2.semilogy(f, pxx + 1e-9, color="#1E2761", lw=1.6)
    a2.axvspan(0.5, 3, color="#1C7293", alpha=0.15, label="locomotor 0.5–3 Hz")
    a2.axvspan(3, 8, color="#C0392B", alpha=0.18, label="freeze 3–8 Hz")
    a2.set_xlim(0, 15)
    a2.set_title("Power spectrum of that window", color="#1E2761", fontweight="bold")
    a2.set_xlabel("frequency (Hz)")
    a2.set_ylabel("power (log)")
    a2.legend(frameon=False, fontsize=10)
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fi_tc[best_i], best_fi


def detector_overlay(accel, phase, tc, ph, fi, en, labels, floor, out):
    mag = np.linalg.norm(accel, axis=1)
    tall_t = np.arange(len(mag)) / FS
    magc = mag - mag.mean()
    truth_cat = np.array([labels[p] if 0 <= p < len(labels) else "?" for p in ph])
    _raw, committed = infer(fi, en, floor)
    # The DEBOUNCED committed state is what the board stores and what is scored.
    det_cat = committed

    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(16, 8.4), sharex=True,
                                     gridspec_kw={"height_ratios": [2.4, 2.0, 1.1]})
    # panel 1: magnitude shaded by ground truth
    start = 0
    cats = np.array([labels[p] if 0 <= p < len(labels) else "?" for p in phase.astype(int)])
    for i in range(1, len(cats) + 1):
        if i == len(cats) or cats[i] != cats[start]:
            c = cats[start]
            a1.axvspan(tall_t[start], tall_t[i - 1], color=BAND.get(c, "#fff"), zorder=0)
            start = i
    a1.plot(tall_t, magc, color="#333", lw=0.5)
    a1.set_ylabel("|accel|−mean (mg)")
    a1.set_title("Ground-truth phases (shaded) with accelerometer magnitude",
                 color="#1E2761", fontweight="bold")
    a1.margins(x=0.004)
    # panel 2: FI per window + threshold; mark gated-still windows
    a2.step(tc, fi, where="mid", color="#C0392B", lw=1.2, label="Freeze Index (per 4 s window)")
    a2.axhline(FI_THRESHOLD, ls="--", lw=1.0, color="#1E2761",
               label=f"deployed threshold = {FI_THRESHOLD}")
    gated = en <= floor
    a2.scatter(tc[gated], fi[gated], s=14, color="#9AA0A6", zorder=5,
               label="gated STILL (energy below floor)")
    a2.set_ylabel("Freeze Index")
    a2.set_yscale("log")
    a2.margins(x=0.004)
    a2.legend(frameon=False, fontsize=9, ncol=3, loc="upper left")
    a2.grid(alpha=0.25)
    # panel 3: truth vs detected strips
    def strip(axrow, cats_w, y, lab):
        for k, c in enumerate(cats_w):
            x0 = tc[k] - WINDOW_HOP / FS / 2
            axrow.add_patch(mpatches.Rectangle((x0, y), WINDOW_HOP / FS, 0.8,
                            color=BAND.get(c, "#fff"), ec=EDGE.get(c, "#bbb"), lw=0.2))
        axrow.text(-2, y + 0.4, lab, ha="right", va="center", fontsize=11, fontweight="bold")
    strip(a3, truth_cat, 1.1, "truth")
    strip(a3, det_cat, 0.1, "detected")
    a3.set_xlim(tc[0] - 2, tc[-1])
    a3.set_ylim(0, 2.1)
    a3.set_yticks([])
    a3.set_xlabel("time (s)")
    for sp in ["top", "right", "left"]:
        a3.spines[sp].set_visible(False)
    handles = [mpatches.Patch(color=BAND[c], ec=EDGE[c], label=c.upper())
               for c in ["still", "walk", "freeze"]]
    a3.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -1.1),
              frameon=False, fontsize=10)
    fig.suptitle("What the device detects vs the ground truth", color="#1E2761",
                 fontweight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # window-level freeze metrics (real numbers)
    tf = truth_cat == "freeze"
    df = det_cat == "freeze"
    tp = int((tf & df).sum())
    fn = int((tf & ~df).sum())
    tn = int((~tf & ~df).sum())
    fp = int((~tf & df).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return dict(still_floor=floor, tp=tp, fp=fp, tn=tn, fn=fn, sens=sens, spec=spec,
                n_windows=len(tc))


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python capture_report.py <capture name | .csv>")
    a = sys.argv[1]
    path = a if os.path.exists(a) else os.path.join(RECORDINGS, a + ".csv")
    if not os.path.exists(path):
        sys.exit(f"not found: {a}")
    stem = os.path.splitext(os.path.basename(path))[0]
    arr, labels = load_capture(path)
    if arr.shape[0] == 0:
        sys.exit(f"no usable data rows in {path}")
    accel = arr[:, 2:5]
    phase = arr[:, 5].astype(int)
    t = arr[:, 1]

    # 1. interactive
    cats = np.array([labels[p] if 0 <= p < len(labels) else "?" for p in phase])
    html = os.path.join(RECORDINGS, stem + "_interactive.html")
    interactive_html(t, accel[:, 0], accel[:, 1], accel[:, 2], cats,
                     f"Cadence — {stem}: 3-axis acceleration by gait phase (interactive)", html)
    print(f"  interactive -> {html}")

    # windows + the SHARED unsupervised still-floor (single source of truth)
    tc, fi, en = windows(accel)
    if tc.size == 0:
        sys.exit(f"capture too short: need >= 256 samples ({accel.shape[0]} found)")
    ph = truth_per_window(phase, labels, len(tc))
    floor = still_floor(en)
    print(f"  still_floor (shared unsupervised Otsu split) = {floor:.0f}")

    # 2. freeze zoom — skip if there is no freeze-phase window
    zoom = os.path.join(RECORDINGS, stem + "_freeze_zoom.png")
    zr = freeze_zoom(accel, phase, tc, fi, labels, zoom)
    if zr is None:
        print("  freeze close-up -> skipped (no freeze-phase window in this capture)")
    else:
        zt, zfi = zr
        print(f"  freeze close-up -> {zoom}  (peak-FI freeze window @ {zt:.0f}s, FI {zfi:.1f})")

    # 3. detector overlay
    ov = os.path.join(RECORDINGS, stem + "_detector.png")
    m = detector_overlay(accel, phase, tc, ph, fi, en, labels, floor, ov)
    print(f"  detector overlay -> {ov}")
    print(f"  window-level freeze detection on this capture: "
          f"TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}  "
          f"sensitivity={m['sens']:.0%}  specificity={m['spec']:.0%}  (n={m['n_windows']} windows)")


if __name__ == "__main__":
    main()
