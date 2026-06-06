"""
Turn a worksheet capture (captureN.csv) into the ENGF0031 Accuracy-Worksheet
Task 3-5 outputs, using the SAME Freeze-Index the live detector uses (fog.dsp).

  Task 3  a phase-shaded time-series plot of the capture (accel magnitude on top,
          per-window Freeze Index below)                  → <name>_trace.{pdf,png}
  Task 4  the Freeze Index per phase (mean / median / max / n windows), computed
          on 4 s / 2 s windows = exactly what the detector sees → printed table
          + a tidy per-window CSV for the appendix          → <name>_windows.csv
  Task 5  if you tell it which phases are freezes (--freeze-phases), it reports
          the window-level sensitivity / specificity the chosen FI threshold
          would give, and suggests a separating threshold.

These numbers are REAL — they come from the accelerometer trace you captured.
This script only does the maths and the plot; it invents nothing.

    # analyse every capture*.csv in the current dir
    python analyze_worksheet.py
    # one file, label the phases, mark which are freezes, set a threshold
    python analyze_worksheet.py capture1.csv \
        --labels still,walk,freeze,walk --freeze-phases 2 --threshold 2.0

Phases are the 0,1,2,… markers you set with the RIGHT button; map them to the
activity in your Task-1 protocol with --labels (purely for nicer annotation).
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from fog.config import SAMPLE_RATE, WINDOW_HOP, WINDOW_SIZE
from fog.dsp import freeze_index, movement_energy, tremor_power

# LaTeX-ish typography to match make_accuracy_figs.py (no TeX install needed).
mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "cm",
    "axes.titlesize": 12,
    "axes.labelsize": 10.5,
    "font.size": 10,
    "axes.edgecolor": "#2b2b2b",
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "figure.dpi": 140,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})
PHASE_TINTS = ["#eef3fb", "#fdeeec", "#eef7ee", "#fbf4e6",
               "#f0eef8", "#eafafa", "#fbeef6", "#f3f3f3"]
NAVY = "#1b2a4a"
ACCENT = "#c0392b"


def load_capture(path: str) -> dict:
    """Read a captureN.csv → dict of t, accel (T,3) mg, phase (T,).

    Hand-parsed (not genfromtxt) so it is robust to a varying number of ``#``
    provenance lines and to the occasional truncated serial row: comment/blank
    lines are skipped, the header maps columns by name, and any row without the
    expected field count is dropped rather than aborting the whole load.
    """
    header: list[str] | None = None
    data: list[list[float]] = []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if header is None and not (s[0].isdigit() or s[0] in "+-."):
                header = [c.strip() for c in s.split(",")]
                continue
            parts = s.split(",")
            if header and len(parts) != len(header):
                continue
            try:
                data.append([float(p) for p in parts])
            except ValueError:
                continue
    if not data:
        raise SystemExit(f"  {path}: no data rows")
    arr = np.array(data, dtype=float)
    cols = {name: i for i, name in enumerate(header)} if header else {}
    ti = cols.get("t_s", 1)
    xi, yi, zi = cols.get("ax_mg", 2), cols.get("ay_mg", 3), cols.get("az_mg", 4)
    pi = cols.get("phase", 5)
    return {"t": arr[:, ti], "accel": arr[:, [xi, yi, zi]],
            "phase": arr[:, pi].astype(int)}


def windowed_features(cap: dict) -> dict:
    """Slide 4 s / 2 s windows; FI / tremor / energy + dominant phase per window."""
    accel, phase, t = cap["accel"], cap["phase"], cap["t"]
    T = accel.shape[0]
    rows = []
    for start in range(0, T - WINDOW_SIZE + 1, WINDOW_HOP):
        win = accel[start:start + WINDOW_SIZE]                 # (W, 3)
        seg = phase[start:start + WINDOW_SIZE]
        dom = int(np.bincount(seg).argmax())                  # majority phase
        rows.append((
            float(t[start + WINDOW_SIZE // 2]),               # window-centre time
            dom,
            float(freeze_index(win)),
            float(tremor_power(win)),
            float(movement_energy(win)),
        ))
    if not rows:
        return {"tc": np.empty(0), "phase": np.empty(0, int),
                "fi": np.empty(0), "tremor": np.empty(0), "energy": np.empty(0)}
    tc, ph, fi, tr, en = (np.array(c) for c in zip(*rows))
    return {"tc": tc, "phase": ph.astype(int), "fi": fi, "tremor": tr, "energy": en}


def phase_label(idx: int, labels: list[str] | None) -> str:
    if labels and idx < len(labels):
        return f"{idx}:{labels[idx]}"
    return f"phase {idx}"


def report_per_phase(feat: dict, labels: list[str] | None,
                     freeze_phases: set[int]) -> None:
    """Task 4: print the Freeze Index (and friends) per phase."""
    print("\n  Task 4 — Freeze Index per phase "
          f"(4 s / {WINDOW_HOP / SAMPLE_RATE:.0f} s windows):")
    print("  " + "-" * 70)
    print(f"  {'phase':<14}{'n':>4}{'FI mean':>10}{'FI med':>9}"
          f"{'FI max':>9}{'tremor':>10}{'energy':>10}")
    print("  " + "-" * 70)
    for p in sorted(set(feat["phase"].tolist())):
        m = feat["phase"] == p
        tag = phase_label(p, labels) + (" *" if p in freeze_phases else "")
        print(f"  {tag:<14}{m.sum():>4}{feat['fi'][m].mean():>10.2f}"
              f"{np.median(feat['fi'][m]):>9.2f}{feat['fi'][m].max():>9.2f}"
              f"{feat['tremor'][m].mean():>10.1f}{feat['energy'][m].mean():>10.1f}")
    print("  " + "-" * 70)
    if freeze_phases:
        print("  (* = marked as a freeze phase via --freeze-phases)")


def report_threshold(feat: dict, freeze_phases: set[int],
                     threshold: float | None) -> None:
    """Task 5: window-level sens/spec at a threshold + a suggested separator."""
    if not freeze_phases:
        print("\n  Task 5 — pass --freeze-phases <idx...> to score a threshold "
              "(which phase numbers are the freezes).")
        return
    y = np.isin(feat["phase"], list(freeze_phases)).astype(int)
    fi = feat["fi"]
    if y.sum() == 0 or y.sum() == len(y):
        print("\n  Task 5 — need both freeze and non-freeze windows to score.")
        return

    # Suggested threshold: midpoint of the gap between the classes' medians,
    # then snapped to the value that maximises (sens+spec) over a fine sweep.
    grid = np.linspace(fi.min(), fi.max(), 512)
    best_t, best_j = grid[0], -1.0
    for thr in grid:
        pred = (fi > thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        sens = tp / (tp + fn) if tp + fn else 0.0
        spec = tn / (tn + fp) if tn + fp else 0.0
        if sens + spec > best_j:
            best_j, best_t = sens + spec, thr

    def score(thr: float) -> tuple[float, float]:
        pred = (fi > thr).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
        return (tp / (tp + fn) if tp + fn else float("nan"),
                tn / (tn + fp) if tn + fp else float("nan"))

    print("\n  Task 5 — Freeze-Index threshold (window-level):")
    s, sp = score(best_t)
    print(f"    suggested threshold (max sens+spec): FI > {best_t:.2f}"
          f"  → sensitivity {s:.0%}, specificity {sp:.0%}")
    if threshold is not None:
        s, sp = score(threshold)
        print(f"    your --threshold {threshold:.2f}"
              f"               → sensitivity {s:.0%}, specificity {sp:.0%}")


def plot_trace(cap: dict, feat: dict, name: str, outdir: str,
               labels: list[str] | None, threshold: float | None,
               fi_ymax: float | None = None) -> str:
    """Task 3: phase-shaded accel magnitude (top) + per-window FI (bottom)."""
    t, accel, phase = cap["t"], cap["accel"], cap["phase"]
    mag = np.linalg.norm(accel, axis=1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1.4]})

    # Shade contiguous phase regions and label each once.
    bounds = np.flatnonzero(np.diff(phase)) + 1
    seg_starts = np.concatenate([[0], bounds])
    seg_ends = np.concatenate([bounds, [len(phase)]])
    seen: set[int] = set()
    for s, e in zip(seg_starts, seg_ends):
        p = int(phase[s])
        ax1.axvspan(t[s], t[min(e, len(t) - 1)], color=PHASE_TINTS[p % len(PHASE_TINTS)],
                    zorder=0)
        ax2.axvspan(t[s], t[min(e, len(t) - 1)], color=PHASE_TINTS[p % len(PHASE_TINTS)],
                    zorder=0)
        if p not in seen:
            ax1.text(t[s], mag.max(), " " + phase_label(p, labels), va="top",
                     ha="left", fontsize=8, color="#555")
            seen.add(p)

    ax1.plot(t, mag, lw=0.7, color=NAVY)
    ax1.set_ylabel("‖accel‖  (mg)")
    ax1.set_title(f"{name} — accelerometer magnitude and Freeze Index")

    ax2.step(feat["tc"], feat["fi"], where="mid", lw=1.1, color=ACCENT)
    ax2.set_ylabel("Freeze Index")
    ax2.set_xlabel("time (s)")
    if threshold is not None:
        ax2.axhline(threshold, ls="--", lw=0.9, color="#333")
        ax2.text(t[-1], threshold, f" FI={threshold:g}", va="bottom", ha="right",
                 fontsize=8, color="#333")
    if fi_ymax is not None:                        # clip a few extreme freeze spikes so
        fimax = float(np.max(feat["fi"]))           # the confound-scale detail stays visible
        ax2.set_ylim(0, fi_ymax)
        if fimax > fi_ymax:
            ax2.text(0.99, 0.93, f"freeze peak ≈ {fimax:.0f} (off-scale)",
                     transform=ax2.transAxes, ha="right", va="top", fontsize=8,
                     color=ACCENT, bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                             ec=ACCENT, alpha=0.9))
    for ax in (ax1, ax2):
        ax.margins(x=0.005)
        ax.grid(True, alpha=0.25, lw=0.5)

    fig.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, f"{name}_trace.pdf")
    fig.savefig(pdf)
    fig.savefig(os.path.join(outdir, f"{name}_trace.png"))
    plt.close(fig)
    return pdf


def dump_windows_csv(feat: dict, name: str, outdir: str,
                     labels: list[str] | None) -> str:
    path = os.path.join(outdir, f"{name}_windows.csv")
    with open(path, "w") as f:
        f.write("t_centre_s,phase,phase_label,freeze_index,tremor_power,movement_energy\n")
        for tc, p, fi, tr, en in zip(feat["tc"], feat["phase"], feat["fi"],
                                     feat["tremor"], feat["energy"]):
            f.write(f"{tc:.3f},{int(p)},{phase_label(int(p), labels).split(':')[-1]},"
                    f"{fi:.4f},{tr:.4f},{en:.4f}\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="*", help="capture CSV(s); default: capture*.csv")
    ap.add_argument("--outdir", default="accuracy_figs", help="where figures/CSVs go")
    ap.add_argument("--labels", default=None,
                    help="comma list mapping phase idx → activity, e.g. still,walk,freeze")
    ap.add_argument("--freeze-phases", default=None,
                    help="comma list of phase indices that are freezes, e.g. 2")
    ap.add_argument("--threshold", type=float, default=None,
                    help="an FI threshold to score / draw (Task 5)")
    ap.add_argument("--fi-ymax", type=float, default=None,
                    help="cap the Freeze-Index panel y-axis; annotates the off-scale peak")
    args = ap.parse_args()

    paths = args.csv or sorted(glob.glob("capture*.csv"))
    if not paths:
        raise SystemExit("  no capture*.csv found — run capture_worksheet.py first.")
    os.makedirs(args.outdir, exist_ok=True)
    labels = args.labels.split(",") if args.labels else None
    freeze_phases = ({int(x) for x in args.freeze_phases.split(",")}
                     if args.freeze_phases else set())

    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        print("\n" + "=" * 72)
        print(f"  {path}")
        print("=" * 72)
        cap = load_capture(path)
        dur = len(cap["t"]) / SAMPLE_RATE
        print(f"  {len(cap['t'])} samples, {dur:.1f}s, "
              f"phases present: {sorted(set(cap['phase'].tolist()))}")
        feat = windowed_features(cap)
        if feat["fi"].size == 0:
            print(f"  capture shorter than one {WINDOW_SIZE / SAMPLE_RATE:.0f}s window "
                  "— need a longer run to compute a Freeze Index.")
            continue
        report_per_phase(feat, labels, freeze_phases)
        report_threshold(feat, freeze_phases, args.threshold)
        wcsv = dump_windows_csv(feat, name, args.outdir, labels)
        pdf = plot_trace(cap, feat, name, args.outdir, labels, args.threshold,
                         fi_ymax=args.fi_ymax)
        print(f"\n  wrote  {pdf}")
        print(f"  wrote  {wcsv}")


if __name__ == "__main__":
    main()
