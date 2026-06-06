#!/usr/bin/env python3
"""Cross-examine a Cadence capture: what you DID (ground-truth phase) vs what the
MODEL PREDICTED (the device's Freeze-Index detector, inferred from the signal).

Outputs (into recordings/):
  <stem>_predicted.png      — ax/ay/az shaded by the MODEL'S predicted state
                              (the partner graph to plot_capture.py's truth graph)
  <stem>_truth_vs_pred.png  — the two timelines aligned, with disagreements marked
                              + window agreement and freeze recall/precision

    python predict_vs_truth.py capture_20260605_103351
    python predict_vs_truth.py /path/to/x.numbers
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from fog.config import FI_THRESHOLD, SAMPLE_RATE, WINDOW_HOP, WINDOW_SIZE
from fog.detect import PRETTY, infer, spans, still_floor, windows
from plot_capture import BAND, EDGE, LINE, RECORDINGS, load_capture, write_csv


def truth_per_window(phase: np.ndarray, labels, n: int) -> np.ndarray:
    out = []
    for k in range(n):
        s = k * WINDOW_HOP
        seg = np.clip(phase[s:s + WINDOW_SIZE].astype(int), 0, None)
        dom = int(np.bincount(seg).argmax())
        out.append(labels[dom] if 0 <= dom < len(labels) else "?")
    return np.array(out)


def is_freelog(path: str, labels) -> bool:
    """True if the capture's phase column is the device's OWN inference.

    Two signatures: the label set is exactly still/walk/freeze (the free-form
    map), or the CSV header carried a '# phase = device-inferred' comment. In
    that case the phase is not an independent ground truth — comparing against it
    is circular, so the figures are relabelled accordingly.
    """
    if list(labels) == ["still", "walk", "freeze"]:
        return True
    try:
        with open(path) as fh:
            for ln in fh:
                s = ln.strip()
                if not s.startswith("#"):
                    break
                if "phase = device-inferred" in s.lower():
                    return True
    except OSError:
        pass
    return False


def axes_legend(ax):
    ah, al = ax.get_legend_handles_labels()
    bh = [mpatches.Patch(facecolor=BAND[c], edgecolor=EDGE[c], label=PRETTY[c])
          for c in ["still", "walk", "freeze"]]
    return ah + bh, al + [h.get_label() for h in bh]


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python predict_vs_truth.py <name | .csv | .numbers>")
    a = sys.argv[1]
    path = a if os.path.exists(a) else os.path.join(RECORDINGS, a + ".csv")
    if not os.path.exists(path):
        sys.exit(f"not found: {a}")
    stem = os.path.splitext(os.path.basename(path))[0]
    arr, labels = load_capture(path)
    if arr.shape[0] == 0:
        sys.exit(f"no usable data rows in {path}")
    if path.endswith(".numbers"):
        write_csv(arr, os.path.join(RECORDINGS, stem + ".csv"), labels)
    t, accel, phase = arr[:, 1], arr[:, 2:5], arr[:, 5].astype(int)

    tc, fi, en = windows(accel)
    if tc.size == 0:
        sys.exit(f"capture too short: need >= 256 samples ({accel.shape[0]} found)")
    floor = still_floor(en)
    raw, committed = infer(fi, en, floor)
    pred = committed                             # DEBOUNCED state (matches the board)
    truth = truth_per_window(phase, labels, len(tc))

    # FREELOG GUARD: when the phase column is the device's OWN inference, calling
    # it "ground truth" is circular — relabel everything as a host re-inference
    # consistency check, not an accuracy score.
    freelog = is_freelog(path, labels)
    ref_lab, pred_lab = ("device-stored", "host re-inference") if freelog \
        else ("you (truth)", "the model")

    # ---- metrics (exclude out-of-range "?" truth windows from the denominator) ----
    valid = truth != "?"
    n_unknown = int((~valid).sum())
    agree = float((truth[valid] == pred[valid]).mean()) if valid.any() else float("nan")
    tf, pf = (truth == "freeze") & valid, pred == "freeze"
    tp = int((tf & pf).sum())
    fn = int((tf & ~pf).sum())
    fp = int((~tf & pf & valid).sum())
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")

    # ================= GRAPH 2: axes shaded by the MODEL'S prediction =========
    fig, ax = plt.subplots(figsize=(16, 5.2))
    for t0, t1, c in spans(tc, pred):
        ax.axvspan(t0, t1, color=BAND.get(c, "#fff"), zorder=0)
    for nm, k, key in [("ax", 0, "ax_mg"), ("ay", 1, "ay_mg"), ("az", 2, "az_mg")]:
        ax.plot(t, accel[:, k], lw=0.5, color=LINE[key], label=nm)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("acceleration (mg)")
    ax.set_title(f"{stem}: what the MODEL PREDICTED at each moment "
                 f"(still/walk/freeze inferred from the signal)",
                 color="#1E2761", fontweight="bold")
    ax.margins(x=0.004)
    h, lbls = axes_legend(ax)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(h, lbls, ncol=6, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=False, fontsize=11)
    pred_png = os.path.join(RECORDINGS, stem + "_predicted.png")
    fig.savefig(pred_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ================= CROSS-EXAM: truth vs prediction ========================
    if freelog:
        title2 = (f"{stem}: device-stored vs host re-inference  —  "
                  f"window match {agree:.0%} (NOT an accuracy score: the phase "
                  f"column is the device's own prediction)")
    else:
        title2 = (f"{stem}: you (ground truth) vs the model  —  "
                  f"window agreement {agree:.0%} · freeze caught {tp}/{tp + fn} "
                  f"(recall {recall:.0%}), {fp} false freeze")
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(16, 8.6), sharex=True,
                                     gridspec_kw={"height_ratios": [2.2, 1.5, 1.5]})
    for _nm, k, key in [("ax", 0, "ax_mg"), ("ay", 1, "ay_mg"), ("az", 2, "az_mg")]:
        a1.plot(t, accel[:, k], lw=0.5, color=LINE[key])
    a1.set_ylabel("accel (mg)")
    a1.set_title(title2, color="#1E2761", fontweight="bold")
    a1.margins(x=0.004)

    def strip(axr, st, y, lab):
        for t0, t1, c in spans(tc, st):
            axr.add_patch(mpatches.Rectangle((t0, y), t1 - t0, 0.8,
                          color=BAND.get(c, "#fff"), ec=EDGE.get(c, "#bbb"), lw=0.2))
        axr.text(-3, y + 0.4, lab, ha="right", va="center", fontsize=11, fontweight="bold")
    ref_strip = "device\n(stored)" if freelog else "you\n(truth)"
    pred_strip = "host\n(re-infer)" if freelog else "model\n(pred)"
    strip(a2, truth, 1.15, ref_strip)
    strip(a2, pred, 0.15, pred_strip)
    # mark disagreements with a red tick between the strips
    mism = truth != pred
    tick = WINDOW_HOP / SAMPLE_RATE
    for k in np.where(mism)[0]:
        a2.add_patch(mpatches.Rectangle((tc[k] - tick / 2, 1.02), tick, 0.10,
                     color="#C0392B", lw=0))
    a2.set_xlim(tc[0] - 4, tc[-1])
    a2.set_ylim(0, 2.05)
    a2.set_yticks([])
    for sp in ["top", "right", "left"]:
        a2.spines[sp].set_visible(False)
    a2.text(tc[-1], 1.0, "  red = disagree", ha="right", va="center", fontsize=8, color="#C0392B")

    a3.step(tc, fi, where="mid", color="#C0392B", lw=1.1)
    a3.axhline(FI_THRESHOLD, ls="--", lw=1.0, color="#1E2761")
    a3.text(tc[-1], FI_THRESHOLD, f" freeze line {FI_THRESHOLD}", va="bottom", ha="right",
            fontsize=9, color="#1E2761")
    a3.set_yscale("log")
    a3.set_ylabel("Freeze Index")
    a3.set_xlabel("time (s)")
    a3.grid(alpha=0.25)
    a3.margins(x=0.004)

    bh = [mpatches.Patch(facecolor=BAND[c], edgecolor=EDGE[c], label=PRETTY[c])
          for c in ["still", "walk", "freeze"]]
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.legend(bh, [h.get_label() for h in bh], ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), frameon=False, fontsize=11)
    cmp_png = os.path.join(RECORDINGS, stem + "_truth_vs_pred.png")
    fig.savefig(cmp_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- console summary ----
    print(f"  windows: {len(tc)}   still_floor={floor:.0f}")
    if freelog:
        print("  NOTE: this capture is FREE-FORM — its 'phase' column is the "
              "DEVICE'S OWN inference, not an independent ground truth.")
        print("        Reporting device-stored vs host re-inference CONSISTENCY, "
              "NOT accuracy.")
        print(f"  device-vs-host window match: {agree:.1%}")
    else:
        print(f"  WINDOW AGREEMENT ({pred_lab} vs {ref_lab}): {agree:.1%}")
    if n_unknown:
        print(f"  WARNING: {n_unknown} window(s) had an out-of-range phase index "
              f"('?') and were EXCLUDED from the agreement denominator.")
    print(f"  FREEZE: ref windows={tp + fn}, {pred_lab} caught={tp} (recall {recall:.0%}), "
          f"false freezes={fp} (precision {prec:.0%})")
    for cls in ["still", "walk", "freeze"]:
        m = (truth == cls) & valid
        if m.any():
            acc = float((pred[m] == cls).mean())
            verb = "matched" if freelog else "agreed"
            print(f"    when {ref_lab} was {cls:6s} ({int(m.sum()):3d} win) "
                  f"{pred_lab} {verb} {acc:.0%}")
    print(f"  predicted graph -> {pred_png}")
    print(f"  cross-exam      -> {cmp_png}")


if __name__ == "__main__":
    main()
