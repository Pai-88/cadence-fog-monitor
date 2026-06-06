#!/usr/bin/env python3
"""Plot a Cadence capture: ax / ay / az vs time, shaded by gait phase.

Reads a CADENCE capture (the CSV the board logger / cpx_dump.py produces, OR an
Apple .numbers export of one) and draws the three accelerometer axes against time
with the STILL / WALKING / FREEZE phases shaded behind them, using seaborn.

    python plot_capture.py capture_20260605_103351            # by name (looks in recordings/)
    python plot_capture.py /path/to/capture.csv               # or a path
    python plot_capture.py /path/to/capture.numbers           # Apple Numbers export

Writes  <name>_axes.png  into recordings/ (and, for a .numbers input, also the
recovered <name>.csv so the rest of the toolchain can use it).
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDINGS = os.path.join(os.path.dirname(HERE), "recordings")

# The protocol the logger ships with: phase index -> activity label.
DEFAULT_LABELS = ["still", "walk", "freeze", "walk", "freeze",
                  "walk", "freeze", "walk", "still"]

# colours
LINE = {"ax_mg": "#1E2761", "ay_mg": "#E67E22", "az_mg": "#2E7D32"}
BAND = {"still": "#ECECEC", "walk": "#E1F0EF", "freeze": "#FAE0DC"}
EDGE = {"still": "#9AA0A6", "walk": "#1C7293", "freeze": "#C0392B"}


def _labels_from_comment(line: str) -> list[str] | None:
    """Pull the label list out of a '# analyze: --labels a,b,c ...' comment."""
    if "--labels" not in line:
        return None
    tail = line.split("--labels", 1)[1].strip()
    tail = tail.split("--freeze-phases")[0]
    labs = [x.strip() for x in tail.replace(" ", ",").split(",") if x.strip()]
    return labs or None


def load_capture(path: str):
    """Return (arr, labels). ``arr`` is (N, 6): idx, t, ax, ay, az, phase.

    Empty / fully unparseable input yields a 2-D ``(0, 6)`` array (never a 1-D
    array, so callers can index ``arr[:, 1]`` unconditionally). Non-finite rows
    (NaN/inf) are dropped.
    """
    if path.endswith(".numbers"):
        from numbers_parser import Document
        rows = Document(path).sheets[0].tables[0].rows(values_only=True)
        hi = next((i for i, r in enumerate(rows) if r and r[0] == "idx"), None)
        if hi is None:
            raise ValueError(f"{path}: no 'idx' header row found in the .numbers sheet")
        # Scan the rows above the data for an '# analyze:' label spec so a
        # freelog / custom-label export maps correctly (don't hardcode defaults).
        labels = None
        for r in rows[:hi]:
            for cell in (r or ()):
                if isinstance(cell, str) and "--labels" in cell:
                    labels = _labels_from_comment(cell) or labels
        labels = labels or DEFAULT_LABELS
        data = []
        for r in rows[hi + 1:]:
            try:
                rec = (int(r[0]), float(r[1]), float(r[2]),
                       float(r[3]), float(r[4]), int(r[5]))
            except (TypeError, ValueError, IndexError):
                continue
            if all(np.isfinite(v) for v in rec[1:5]):
                data.append(rec)
    else:
        labels = None
        data, header_seen = [], False
        with open(path) as fh:
            for ln in fh:
                s = ln.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    labels = _labels_from_comment(s) or labels
                    continue
                if not header_seen and s.lower().startswith("idx"):
                    header_seen = True
                    continue
                p = s.split(",")
                try:
                    rec = (int(float(p[0])), float(p[1]), float(p[2]),
                           float(p[3]), float(p[4]), int(float(p[5])))
                except (IndexError, ValueError):
                    continue
                if not all(np.isfinite(v) for v in rec[1:5]):
                    continue  # reject NaN / inf rows
                data.append(rec)
        labels = labels or DEFAULT_LABELS
    if not data:
        return np.empty((0, 6), dtype=float), (labels or DEFAULT_LABELS)
    arr = np.array(data, dtype=float)
    return arr, (labels or DEFAULT_LABELS)


def _analyze_comment(labels) -> str:
    """Reconstruct the '# analyze: --labels ... --freeze-phases ...' line.

    The freeze-phase indices are derived from ``labels`` itself so a freelog or
    custom-label capture round-trips faithfully instead of being clobbered with
    the 9-state scripted default.
    """
    labs = ",".join(labels)
    freeze_idx = ",".join(str(i) for i, lab in enumerate(labels) if lab == "freeze")
    return f"# analyze: --labels {labs} --freeze-phases {freeze_idx}\n"


def write_csv(arr, csv_path: str, labels=None) -> None:
    labels = labels or DEFAULT_LABELS
    n = len(arr)
    dur = arr[-1, 1] if n else 0.0
    with open(csv_path, "w") as f:
        f.write("# CADENCE on-board capture (CPX SPI-flash logger)\n")
        f.write(f"# samples={n}  rate_hz=64  duration_s={dur:.2f}\n")
        f.write(f"# protocol: {'>'.join(lab.upper() for lab in labels)}\n")
        f.write(_analyze_comment(labels))
        f.write("idx,t_s,ax_mg,ay_mg,az_mg,phase\n")
        for r in arr:
            f.write(f"{int(r[0])},{r[1]:.4f},{int(r[2])},{int(r[3])},"
                    f"{int(r[4])},{int(r[5])}\n")


def plot(arr, labels, title: str, out_png: str, shade: bool = True) -> dict:
    t, ax_, ay_, az_, ph = (arr[:, i] for i in (1, 2, 3, 4, 5))
    cat = np.array([labels[int(p)] if 0 <= int(p) < len(labels) else "?" for p in ph])

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(16, 6.5))

    # shade contiguous phase regions (optional — disabled with --no-shade)
    seen, start = set(), 0
    if shade:
        for i in range(1, len(cat) + 1):
            if i == len(cat) or cat[i] != cat[start]:
                c = cat[start]
                end = t[i] if i < len(cat) else t[-1]
                ax.axvspan(t[start], end, color=BAND.get(c, "#fff"), zorder=0)
                seen.add(c)
                start = i

    ax.plot(t, ax_, lw=0.6, color=LINE["ax_mg"], label="ax")
    ax.plot(t, ay_, lw=0.6, color=LINE["ay_mg"], label="ay")
    ax.plot(t, az_, lw=0.6, color=LINE["az_mg"], label="az")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("acceleration (mg)")
    ax.set_title(title, color="#1E2761", fontweight="bold")
    ax.margins(x=0.004)

    axis_handles, axis_labels = ax.get_legend_handles_labels()
    band_handles = [mpatches.Patch(facecolor=BAND[c], edgecolor=EDGE[c], label=c.upper())
                    for c in ["still", "walk", "freeze"] if c in seen]
    leg = ax.legend(axis_handles + band_handles,
                    axis_labels + [h.get_label() for h in band_handles],
                    ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.13),
                    frameon=False, fontsize=12)
    leg.set_title(None)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    secs = {str(c): round(float((cat == c).sum()) / 64.0, 1) for c in sorted(set(cat))}
    return {"n": len(arr), "dur": float(t[-1]), "phases": sorted(set(ph.astype(int).tolist())),
            "seconds_per_category": secs}


def main() -> None:
    args = [x for x in sys.argv[1:] if not x.startswith("-")]
    no_shade = ("--no-shade" in sys.argv) or ("--plain" in sys.argv)
    if not args:
        sys.exit("usage: python plot_capture.py <capture name | .csv | .numbers> [--no-shade]")
    a = args[0]
    path = a
    if not os.path.exists(path):
        for ext in (".csv", ".numbers"):
            cand = os.path.join(RECORDINGS, a + ext)
            if os.path.exists(cand):
                path = cand
                break
    if not os.path.exists(path):
        sys.exit(f"capture not found: {a}")

    stem = os.path.splitext(os.path.basename(path))[0]
    os.makedirs(RECORDINGS, exist_ok=True)
    arr, labels = load_capture(path)
    if arr.shape[0] == 0:
        sys.exit(f"no usable data rows in {path} (empty or unparseable capture)")

    if path.endswith(".numbers"):
        csv_path = os.path.join(RECORDINGS, stem + ".csv")
        write_csv(arr, csv_path, labels)
        print(f"  recovered CSV -> {csv_path}")

    suffix = "_plain" if no_shade else "_axes"
    title = (f"Cadence — {stem}: 3-axis acceleration" if no_shade
             else f"Cadence — {stem}: 3-axis acceleration by gait phase")
    out_png = os.path.join(RECORDINGS, stem + suffix + ".png")
    info = plot(arr, labels, title, out_png, shade=not no_shade)
    print(f"  samples={info['n']}  duration={info['dur']:.1f}s  phases={info['phases']}")
    print(f"  seconds per category: {info['seconds_per_category']}")
    print(f"  plot -> {out_png}")


if __name__ == "__main__":
    main()
