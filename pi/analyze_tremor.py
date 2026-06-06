"""
Wrist rest-tremor analyser — the wrist-monitor counterpart of analyze_worksheet.py.

Turns a wrist capture (captureN.csv from the live Field Recorder in
dashboard_server.py, or from capture_worksheet.py) into the WRIST Accuracy-Worksheet
Task 3-5 outputs, using the SAME 4-6 Hz tremor-band power the live monitor uses
(``fog.dsp.tremor_power``):

  Task 3  a phase-shaded time-series of the capture (accel magnitude on top,
          per-window tremor power below)            → <name>_tremor_trace.{pdf,png}
  Task 4  the tremor power per phase (mean / median / max / n windows) on
          4 s / 2 s windows = exactly what the monitor sees → printed table
  Task 5  with --tremor-phases, the window-level sensitivity / specificity a chosen
          tremor-power threshold gives, plus a suggested separating threshold and
          the full TP/FP/FN/TN confusion matrix.

These numbers are REAL — they come from the wrist trace you captured. This script
only does the maths and the plot; it invents nothing.

  ⚠  There is NO bundled wrist data. Record ~5 min on the WRIST first (rest /
     ~5 Hz tremor surrogate / normal movement), per worksheet/wrist_capture_protocol.txt,
     then point this script at the CSV. Until then the worksheet numbers stay blank.

    python analyze_tremor.py wrist1.csv \
        --labels rest,tremor,move,tremor --tremor-phases 1,3 --threshold 800
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from fog.config import SAMPLE_RATE, WINDOW_HOP, WINDOW_SIZE
from fog.dsp import freeze_index, movement_energy, tremor_power_axes

# Reuse the *exact* capture loader and plot styling from the FoG analyser so the
# wrist path shares the same CSV parsing and look; only the discriminating
# feature changes (Freeze Index → orientation-robust 4-6 Hz tremor power).
from analyze_worksheet import (
    ACCENT,
    NAVY,
    PHASE_TINTS,
    load_capture,
    phase_label,
    plt,
)


def windowed_tremor(cap: dict) -> dict:
    """Slide 4 s / 2 s windows; per-axis tremor power (+ energy, FI) per window."""
    accel, phase, t = cap["accel"], cap["phase"], cap["t"]
    T = accel.shape[0]
    rows = []
    for start in range(0, T - WINDOW_SIZE + 1, WINDOW_HOP):
        win = accel[start:start + WINDOW_SIZE]
        seg = phase[start:start + WINDOW_SIZE]
        dom = int(np.bincount(seg).argmax())
        rows.append((
            float(t[start + WINDOW_SIZE // 2]),
            dom,
            float(tremor_power_axes(win)),   # discriminator (orientation-robust)
            float(movement_energy(win)),     # 0.5-8 Hz energy (movement context)
            float(freeze_index(win)),        # kept for reference in the table
        ))
    if not rows:
        return {"tc": np.empty(0), "phase": np.empty(0, int),
                "tremor": np.empty(0), "energy": np.empty(0), "fi": np.empty(0)}
    tc, ph, tr, en, fi = (np.array(c) for c in zip(*rows))
    return {"tc": tc, "phase": ph.astype(int), "tremor": tr, "energy": en, "fi": fi}


def report_per_phase(feat: dict, labels: list[str] | None,
                     tremor_phases: set[int]) -> None:
    """Task 4: print the tremor-band power (and friends) per phase."""
    print("\n  Task 4 — 4-6 Hz tremor power per phase "
          f"(4 s / {WINDOW_HOP / SAMPLE_RATE:.0f} s windows):")
    print("  " + "-" * 72)
    print(f"  {'phase':<14}{'n':>4}{'tremor mean':>13}{'tremor med':>12}"
          f"{'tremor max':>12}{'energy':>9}{'FI':>7}")
    print("  " + "-" * 72)
    for p in sorted(set(feat["phase"].tolist())):
        m = feat["phase"] == p
        tag = phase_label(p, labels) + (" *" if p in tremor_phases else "")
        print(f"  {tag:<14}{m.sum():>4}{feat['tremor'][m].mean():>13.1f}"
              f"{np.median(feat['tremor'][m]):>12.1f}{feat['tremor'][m].max():>12.1f}"
              f"{feat['energy'][m].mean():>9.0f}{feat['fi'][m].mean():>7.2f}")
    print("  " + "-" * 72)
    if tremor_phases:
        print("  (* = marked as a tremor phase via --tremor-phases)")


def _confusion(x: np.ndarray, y: np.ndarray, thr: float) -> tuple[int, int, int, int]:
    pred = (x > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    return tp, fp, fn, tn


def report_threshold(feat: dict, tremor_phases: set[int],
                     threshold: float | None) -> None:
    """Task 5: window-level sens/spec at a tremor-power threshold + a suggestion."""
    if not tremor_phases:
        print("\n  Task 5 — pass --tremor-phases <idx...> to score a threshold "
              "(which phase numbers are the tremor episodes).")
        return
    y = np.isin(feat["phase"], list(tremor_phases)).astype(int)
    x = feat["tremor"]
    if y.sum() == 0 or y.sum() == len(y):
        print("\n  Task 5 — need both tremor and non-tremor windows to score.")
        return

    # Suggested threshold: the value that maximises (sens + spec) over a fine sweep.
    grid = np.linspace(float(x.min()), float(x.max()), 512)
    best_t, best_j = grid[0], -1.0
    for thr in grid:
        tp, fp, fn, tn = _confusion(x, y, thr)
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens + spec > best_j:
            best_j, best_t = sens + spec, thr

    def show(thr: float, tag: str) -> None:
        tp, fp, fn, tn = _confusion(x, y, thr)
        sens = tp / (tp + fn) if tp + fn else float("nan")
        spec = tn / (tn + fp) if tn + fp else float("nan")
        print(f"    {tag}: tremor power > {thr:.1f}")
        print(f"      TP {tp}  FP {fp}  FN {fn}  TN {tn}"
              f"   → sensitivity {sens:.0%}, specificity {spec:.0%}")

    print("\n  Task 5 — tremor-power threshold (window-level):")
    show(best_t, "suggested (max sens+spec)")
    if threshold is not None:
        show(threshold, "your --threshold")


def plot_trace(cap: dict, feat: dict, name: str, outdir: str,
               labels: list[str] | None, threshold: float | None) -> str:
    """Task 3: phase-shaded accel magnitude (top) + per-window tremor power (bottom)."""
    t, accel, phase = cap["t"], cap["accel"], cap["phase"]
    mag = np.linalg.norm(accel, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.4]})

    bounds = np.flatnonzero(np.diff(phase)) + 1
    seg_starts = np.concatenate([[0], bounds])
    seg_ends = np.concatenate([bounds, [len(phase)]])
    seen: set[int] = set()
    for s, e in zip(seg_starts, seg_ends):
        p = int(phase[s])
        for ax in (ax1, ax2):
            ax.axvspan(t[s], t[min(e, len(t) - 1)],
                       color=PHASE_TINTS[p % len(PHASE_TINTS)], zorder=0)
        if p not in seen:
            ax1.text(t[s], mag.max(), " " + phase_label(p, labels), va="top",
                     ha="left", fontsize=8, color="#555")
            seen.add(p)

    ax1.plot(t, mag, lw=0.7, color=NAVY)
    ax1.set_ylabel("‖accel‖  (mg)")
    ax1.set_title(f"{name} — wrist accel magnitude and 4-6 Hz tremor power")

    ax2.step(feat["tc"], feat["tremor"], where="mid", lw=1.1, color=ACCENT)
    ax2.set_ylabel("tremor power (4-6 Hz)")
    ax2.set_xlabel("time (s)")
    if threshold is not None:
        ax2.axhline(threshold, ls="--", lw=0.9, color="#333")
        ax2.text(t[-1], threshold, f" thr={threshold:g}", va="bottom", ha="right",
                 fontsize=8, color="#333")
    for ax in (ax1, ax2):
        ax.margins(x=0.005)
        ax.grid(True, alpha=0.25, lw=0.5)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f"{name}_tremor_trace.pdf")
    fig.savefig(pdf)
    fig.savefig(os.path.join(outdir, f"{name}_tremor_trace.png"))
    plt.close(fig)
    return pdf


def dump_windows_csv(feat: dict, name: str, outdir: str,
                     labels: list[str] | None) -> str:
    path = os.path.join(outdir, f"{name}_tremor_windows.csv")
    with open(path, "w") as f:
        f.write("t_centre_s,phase,phase_label,tremor_power,movement_energy,freeze_index\n")
        for tc, p, tr, en, fi in zip(feat["tc"], feat["phase"], feat["tremor"],
                                     feat["energy"], feat["fi"]):
            f.write(f"{tc:.3f},{int(p)},{phase_label(int(p), labels).split(':')[-1]},"
                    f"{tr:.4f},{en:.4f},{fi:.4f}\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*", help="wrist capture CSV(s); default: wrist*.csv")
    ap.add_argument("--outdir", default="accuracy_figs", help="where figures/CSVs go")
    ap.add_argument("--labels", default=None,
                    help="comma list mapping phase idx → activity, e.g. rest,tremor,move")
    ap.add_argument("--tremor-phases", default=None,
                    help="comma list of phase indices that are tremor, e.g. 1,3")
    ap.add_argument("--threshold", type=float, default=None,
                    help="a tremor-power threshold to score / draw (Task 5)")
    args = ap.parse_args()

    paths = args.csv or sorted(glob.glob("wrist*.csv"))
    if not paths:
        raise SystemExit("  no wrist*.csv found — record a wrist capture first "
                         "(see worksheet/wrist_capture_protocol.txt).")
    os.makedirs(args.outdir, exist_ok=True)
    labels = args.labels.split(",") if args.labels else None
    tremor_phases = ({int(x) for x in args.tremor_phases.split(",")}
                     if args.tremor_phases else set())

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print("\n" + "=" * 72)
        print(f"  {path}")
        print("=" * 72)
        cap = load_capture(path)
        dur = len(cap["t"]) / SAMPLE_RATE
        print(f"  {len(cap['t'])} samples, {dur:.1f}s, "
              f"phases present: {sorted(set(cap['phase'].tolist()))}")
        feat = windowed_tremor(cap)
        if feat["tremor"].size == 0:
            print(f"  capture shorter than one {WINDOW_SIZE / SAMPLE_RATE:.0f}s window "
                  "— need a longer run to compute tremor power.")
            continue
        report_per_phase(feat, labels, tremor_phases)
        report_threshold(feat, tremor_phases, args.threshold)
        wcsv = dump_windows_csv(feat, name, args.outdir, labels)
        pdf = plot_trace(cap, feat, name, args.outdir, labels, args.threshold)
        print(f"\n  wrote  {pdf}")
        print(f"  wrote  {wcsv}")


if __name__ == "__main__":
    main()
