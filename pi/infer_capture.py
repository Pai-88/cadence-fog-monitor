#!/usr/bin/env python3
"""Free-form inference of a Cadence capture.

No scripted protocol, no phase labels, no telling the wearer what to do: the
person moves however and whenever they like, and the device's OWN detector infers
STILL / WALKING / FREEZE from the accelerometer alone — exactly what runs live on
the board:

    moving = movement_energy > still_floor          (still_floor found unsupervised)
    state  = STILL  if not moving
             FREEZE if moving and FreezeIndex > FI_THRESHOLD  (3-8 Hz trembling)
             WALK   otherwise
    then a 2-window debounce stabilises it into episodes.

The `phase` column in the CSV (if any) is IGNORED — this is pure inference.

    python infer_capture.py capture_20260605_103351
    python infer_capture.py /path/to/anything.csv
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from fog.config import FI_THRESHOLD, WINDOW_HOP
from fog.detect import (
    FS,
    PRETTY,
    freeze_episodes,
    infer,
    spans,
    still_floor,
    windows,
)
from plot_capture import BAND, EDGE, LINE, RECORDINGS, load_capture


def plot(t, accel, tc, fi, state, floor, stem, out_png):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(16, 7.4), sharex=True,
                                 gridspec_kw={"height_ratios": [2.6, 1.4]})
    for t0, t1, c in spans(tc, state):
        a1.axvspan(t0, t1, color=BAND.get(c, "#fff"), zorder=0)
        a2.axvspan(t0, t1, color=BAND.get(c, "#fff"), zorder=0)
    for nm, k, key in [("ax", 0, "ax_mg"), ("ay", 1, "ay_mg"), ("az", 2, "az_mg")]:
        a1.plot(t, accel[:, k], lw=0.5, color=LINE[key], label=nm)
    a1.set_ylabel("acceleration (mg)")
    a1.set_title(f"Cadence — {stem}: movement INFERRED from the signal "
                 f"(no script, no labels)", color="#1E2761", fontweight="bold")
    a1.margins(x=0.004)

    a2.step(tc, fi, where="mid", color="#C0392B", lw=1.2)
    a2.axhline(FI_THRESHOLD, ls="--", lw=1.0, color="#1E2761")
    a2.text(tc[-1], FI_THRESHOLD, f" freeze line {FI_THRESHOLD}", va="bottom", ha="right",
            fontsize=9, color="#1E2761")
    a2.set_yscale("log")
    a2.set_ylabel("Freeze Index")
    a2.set_xlabel("time (s)")
    a2.margins(x=0.004)
    a2.grid(alpha=0.25)

    axis_h, axis_l = a1.get_legend_handles_labels()
    band_h = [mpatches.Patch(facecolor=BAND[c], edgecolor=EDGE[c], label=PRETTY[c])
              for c in ["still", "walk", "freeze"]]
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.subplots_adjust(hspace=0.12)
    fig.legend(axis_h + band_h, axis_l + [h.get_label() for h in band_h],
               ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=False, fontsize=11)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def interactive(t, accel, tc, state, stem, out_html):
    import plotly.graph_objects as go
    fig = go.Figure()
    for t0, t1, c in spans(tc, state):
        fig.add_vrect(x0=t0, x1=t1, fillcolor=BAND.get(c, "#fff"), opacity=0.5,
                      layer="below", line_width=0)
    for nm, k, key in [("ax", 0, "ax_mg"), ("ay", 1, "ay_mg"), ("az", 2, "az_mg")]:
        fig.add_trace(go.Scattergl(x=t, y=accel[:, k], name=nm, mode="lines",
                                   line=dict(width=1, color=LINE[key])))
    for c in ["still", "walk", "freeze"]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=PRETTY[c] + " (inferred)",
                      marker=dict(size=12, color=BAND[c], line=dict(color=EDGE[c], width=1))))
    fig.update_layout(title=f"Cadence — {stem}: movement inferred from the signal (interactive)",
                      template="plotly_white", xaxis_title="time (s)",
                      yaxis_title="acceleration (mg)",
                      legend=dict(orientation="h", y=-0.18), hovermode="x unified",
                      margin=dict(l=60, r=30, t=60, b=80))
    fig.write_html(out_html, include_plotlyjs="cdn")


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python infer_capture.py <capture name | .csv>")
    a = sys.argv[1]
    path = a if os.path.exists(a) else os.path.join(RECORDINGS, a + ".csv")
    if not os.path.exists(path):
        sys.exit(f"not found: {a}")
    stem = os.path.splitext(os.path.basename(path))[0]
    arr, _ = load_capture(path)          # phase column deliberately ignored
    if arr.shape[0] == 0:
        sys.exit(f"no usable data rows in {path}")
    t = arr[:, 1]
    accel = arr[:, 2:5]

    tc, fi, en = windows(accel)
    if tc.size == 0:
        sys.exit(f"capture too short: need >= 256 samples ({accel.shape[0]} found)")
    floor = still_floor(en)
    raw, committed = infer(fi, en, floor)

    # The DEBOUNCED `committed` state is what the board stores and what is scored
    # everywhere — shade the figures with it. `raw` is kept only as the
    # supplementary "freeze-band windows flagged" count below.
    png = os.path.join(RECORDINGS, stem + "_inferred.png")
    html = os.path.join(RECORDINGS, stem + "_inferred.html")
    plot(t, accel, tc, fi, committed, floor, stem, png)
    interactive(t, accel, tc, committed, stem, html)

    secs = {PRETTY[c]: round(float((committed == c).sum()) * WINDOW_HOP / FS, 1)
            for c in ["still", "walk", "freeze"] if (committed == c).any()}
    eps = freeze_episodes(tc, committed, fi)
    raw_fz = int((raw == "freeze").sum())
    print(f"  still_floor (unsupervised Otsu split) = {floor:.0f}")
    print(f"  inferred time (per 4 s window):  {secs}")
    print(f"  freeze-band windows flagged: {raw_fz}  ->  "
          f"sustained FREEZE episodes (after debounce): {len(eps)}")
    for s0, s1, pk in eps:
        print(f"     {s0:6.1f}s -> {s1:6.1f}s  ({s1 - s0:4.1f}s,  peak FI {pk:.1f})")
    print(f"  plot        -> {png}")
    print(f"  interactive -> {html}")


if __name__ == "__main__":
    main()
