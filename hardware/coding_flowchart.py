"""
Coding flowchart for the Parkinson's closed-loop gait garment (ENGF0031
Scenario 2) — the design-proposal deliverable described in the brief
(Programming / Coding-Flowchart slides).

It charts the STANDALONE on-board firmware ``firmware/variants/cpx_fog_standalone/
cpx_fog_standalone.ino``: the whole loop runs on the Circuit Playground Express,
no host. Each 1/64 s it samples the accelerometer; every 2 s it band-passes the
last 4 s window into a Freeze Index, applies the movement-energy gate that
rejects quiet standing, debounces the decision, and drives the rhythmic
vibrotactile cue. The boxes below map one-to-one onto that code.

Symbols follow the standard set on the brief's flowchart slide:
    Terminal (start/end) · Process · Input/Output · Decision (Yes/No branches).

Wireless (streaming) build: the board instead streams the 64 Hz accel over a
short UART link to an ESP32, which relays it to a laptop over Wi-Fi/TCP;
the laptop's CNN returns the freeze decision and the cue command comes back the same
way. The cue/debounce logic is identical — only the detector moves off-board
(see the "Wireless build" strip at the foot of the chart).

Run:  python coding_flowchart.py                  → coding_flowchart.png, .pdf
      python coding_flowchart.py --out build/flow --formats pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

matplotlib.use("Agg")  # headless backend: render straight to file, no display

# ── palette: a restrained "engineering report" scheme — neutral slate line-work
#    on white, with ONE warm accent reserved for the freeze-detect + cue path, so
#    colour carries meaning (the clinically important thread) rather than just
#    decorating every box a different hue. ──
C_INK    = '#1d2832'                          # body text
C_ARROW  = '#3d4a57'                          # every border + arrow (one line colour)
C_TERM   = '#22303c'                          # terminal: dark slate, white text
C_PROC,   C_PROC_E   = '#ffffff', '#3d4a57'   # process: white box, slate edge
C_IO,     C_IO_E     = '#eef1f4', '#3d4a57'   # input/output: faint cool grey
C_DEC,    C_DEC_E    = '#ffffff', '#3d4a57'   # decision: white (the diamond marks it)
C_FREEZE, C_FREEZE_E = '#e3bdb4', '#8f3322'   # freeze + cue: muted terracotta accent
C_CONN,   C_CONN_E   = '#dfe4e9', '#3d4a57'   # loop connector

plt.rcParams.update({'font.size': 9.0, 'font.family': 'DejaVu Sans'})

# node registry: key -> (x, y, w, h, kind), filled as nodes are drawn
_N: dict[str, tuple[float, float, float, float, str]] = {}


# ── shape primitives ─────────────────────────────────────────────────────────
def _text(ax: Axes, x: float, y: float, s: str, color: str, weight: str = 'normal',
          size: float = 9.0) -> None:
    ax.text(x, y, s, ha='center', va='center', color=color, fontsize=size,
            fontweight=weight, linespacing=1.35, zorder=6)


def terminal(ax: Axes, key: str, x: float, y: float, w: float, h: float, s: str) -> None:
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle=f'round,pad=0,rounding_size={h / 2}',
                 fc=C_TERM, ec='#0f1318', lw=1.4, zorder=4))
    _text(ax, x, y, s, 'white', 'bold')
    _N[key] = (x, y, w, h, 'term')


def process(ax: Axes, key: str, x: float, y: float, w: float, h: float, s: str,
            *, hot: bool = False, size: float = 9.0) -> None:
    fc, ec = (C_FREEZE, C_FREEZE_E) if hot else (C_PROC, C_PROC_E)
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, fc=fc, ec=ec, lw=1.4, zorder=4))
    _text(ax, x, y, s, C_INK, size=size)
    _N[key] = (x, y, w, h, 'proc')


def io(ax: Axes, key: str, x: float, y: float, w: float, h: float, s: str) -> None:
    k = 0.5  # horizontal skew of the parallelogram
    pts = [(x - w / 2 + k, y - h / 2), (x + w / 2 + k, y - h / 2),
           (x + w / 2 - k, y + h / 2), (x - w / 2 - k, y + h / 2)]
    ax.add_patch(Polygon(pts, closed=True, fc=C_IO, ec=C_IO_E, lw=1.4, zorder=4))
    _text(ax, x, y, s, C_INK)
    _N[key] = (x, y, w, h, 'io')


def decision(ax: Axes, key: str, x: float, y: float, w: float, h: float, s: str,
             *, accent: bool = False) -> None:
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(Polygon(pts, closed=True, fc=C_DEC, ec=C_DEC_E,
                         lw=2.2 if accent else 1.4, zorder=4))
    _text(ax, x, y, s, C_INK)
    _N[key] = (x, y, w, h, 'dec')


def connector(ax: Axes, key: str, x: float, y: float, r: float, s: str) -> None:
    ax.add_patch(Circle((x, y), r, fc=C_CONN, ec=C_CONN_E, lw=1.4, zorder=4))
    _text(ax, x, y, s, C_INK, 'bold')
    _N[key] = (x, y, 2 * r, 2 * r, 'conn')


# ── anchor points on a node's edge ───────────────────────────────────────────
def _a(key: str, side: str) -> tuple[float, float]:
    x, y, w, h, _ = _N[key]
    return {'top': (x, y + h / 2), 'bottom': (x, y - h / 2),
            'left': (x - w / 2, y), 'right': (x + w / 2, y)}[side]


# ── connectors ───────────────────────────────────────────────────────────────
def arrow(ax: Axes, p0: tuple[float, float], p1: tuple[float, float]) -> None:
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>', mutation_scale=14,
                 color=C_ARROW, lw=1.5, shrinkA=0, shrinkB=0, zorder=3))


def elbow(ax: Axes, pts: list[tuple[float, float]]) -> None:
    """Poly-line through ``pts`` with a single arrowhead at the final point."""
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=C_ARROW, lw=1.5, solid_capstyle='round', zorder=3)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=14,
                 color=C_ARROW, lw=1.5, shrinkA=0, shrinkB=0, zorder=3))


def feed(ax: Axes, pts: list[tuple[float, float]]) -> None:
    """Poly-line that merges INTO a rail — no arrowhead (the rail carries it)."""
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=C_ARROW, lw=1.5, solid_capstyle='round', zorder=3)


def dot(ax: Axes, x: float, y: float) -> None:
    """A small junction dot where a feeder meets a rail."""
    ax.add_patch(Circle((x, y), 0.08, fc=C_ARROW, ec='none', zorder=4))


def blabel(ax: Axes, x: float, y: float, s: str) -> None:
    ax.text(x, y, s, ha='center', va='center', fontsize=8.2, color=C_INK,
            fontweight='bold', zorder=7,
            bbox=dict(boxstyle='round,pad=0.16', fc='white', ec='none', alpha=0.92))


# ════════════════════════════════════════════════════════════════════════════
#  FLOWCHART
# ════════════════════════════════════════════════════════════════════════════
def build_figure() -> Figure:
    fig = plt.figure(figsize=(8.4, 19.9))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(-0.5, 14.8)
    ax.set_ylim(-4.6, 31.6)
    ax.set_aspect('equal')
    ax.axis('off')

    cx = 6.2                                   # main column centre
    rail_l = 0.3                               # left return rail (loops back to A)
    rail_r = 13.3                              # right rail (STILL/WALKING merge)

    # ── main column, top → bottom ──
    terminal(ax, 'start', cx, 30.4, 3.4, 1.0, "START  (power on)")
    process(ax, 'setup', cx, 28.5, 5.4, 1.7,
            "setup():\nstart accelerometer · init LEDs\nmotor pin = OUTPUT (off)")
    connector(ax, 'A', cx, 26.8, 0.36, "A")
    io(ax, 'read', cx, 24.9, 5.6, 1.5,
       "Read ax, ay, az   @ 64 Hz\nmagnitude = √(ax² + ay² + az²)")
    process(ax, 'buffer', cx, 22.8, 5.2, 1.4,
            "Append magnitude to ring buffer\n(keeps the last 256 samples = 4 s)")
    decision(ax, 'btna', cx, 20.3, 4.2, 2.2, "Button A\npressed?")
    decision(ax, 'hop', cx, 17.0, 4.6, 2.4,
             "Buffer full AND\n2 s since the\nlast decision?")
    process(ax, 'fi', cx, 13.9, 6.2, 1.9,
            "Band-pass the window:\nFI = power(3–8 Hz) / power(0.5–3 Hz)\n"
            "energy = P_freeze + P_loco")
    decision(ax, 'gate', cx, 10.8, 5.0, 2.5,
             "energy > still_floor?\n(movement gate)", accent=True)
    decision(ax, 'thr', cx, 7.7, 4.4, 2.3, "FI > 1.815 ?")
    process(ax, 'freeze', cx, 5.3, 4.0, 1.4, "state = FREEZE\nfreeze-count++", hot=True)
    process(ax, 'cue', cx, 2.9, 6.8, 1.9,
            "Cue control (debounced):\n2 freezes in a row → motor cue ON  (2 Hz pulses)\n"
            "2 clears in a row → motor cue OFF", hot=True)
    io(ax, 'log', cx, 0.7, 5.6, 1.4, "Update NeoPixels + serial log\n(FI · energy · state)")

    # ── side nodes ──
    process(ax, 'calib', 2.1, 20.3, 3.2, 2.0,
            "Calibrate still-floor:\nstill_floor = 4 × resting\n(stand still ~4 s)", size=8.0)
    process(ax, 'still', 11.2, 10.8, 3.3, 1.4, "state = STILL\nclear-count++")
    process(ax, 'walk', 11.2, 7.7, 3.3, 1.4, "state = WALKING\nclear-count++")

    # ── straight main-line arrows ──
    arrow(ax, _a('start', 'bottom'), _a('setup', 'top'))
    arrow(ax, _a('setup', 'bottom'), _a('A', 'top'))
    arrow(ax, _a('A', 'bottom'), _a('read', 'top'))
    arrow(ax, _a('read', 'bottom'), _a('buffer', 'top'))
    arrow(ax, _a('buffer', 'bottom'), _a('btna', 'top'))
    arrow(ax, _a('btna', 'bottom'), _a('hop', 'top'))
    arrow(ax, _a('hop', 'bottom'), _a('fi', 'top'))
    arrow(ax, _a('fi', 'bottom'), _a('gate', 'top'))
    arrow(ax, _a('gate', 'bottom'), _a('thr', 'top'))
    arrow(ax, _a('thr', 'bottom'), _a('freeze', 'top'))
    arrow(ax, _a('freeze', 'bottom'), _a('cue', 'top'))
    arrow(ax, _a('cue', 'bottom'), _a('log', 'top'))

    # ── branch labels on the main spine (Yes continues straight down) ──
    blabel(ax, cx + 0.32, 18.7, "No")
    blabel(ax, cx + 0.32, 15.3, "Yes")
    blabel(ax, cx + 0.34, 9.2, "Yes")
    blabel(ax, cx + 0.32, 6.27, "Yes")

    # ── Button A = Yes → calibrate, which feeds the left return rail ──
    arrow(ax, _a('btna', 'left'), _a('calib', 'right'))
    blabel(ax, 3.9, 20.55, "Yes")
    feed(ax, [_a('calib', 'left'), (rail_l, 20.3)])

    # ── single left return rail: LOG → up → A ; HOP-No and calib feed in ──
    elbow(ax, [_a('log', 'left'), (rail_l, 0.7), (rail_l, 26.8), _a('A', 'left')])
    feed(ax, [_a('hop', 'left'), (rail_l, 17.0)])
    dot(ax, rail_l, 20.3)
    dot(ax, rail_l, 17.0)

    # ── gate = No → STILL ; thr = No → WALKING ; both merge on the freeze→cue spine ──
    arrow(ax, _a('gate', 'right'), _a('still', 'left'))
    blabel(ax, 9.13, 11.1, "No")
    arrow(ax, _a('thr', 'right'), _a('walk', 'left'))
    blabel(ax, 8.98, 8.0, "No")
    merge_y = 4.3                                   # junction on the freeze → cue arrow
    feed(ax, [_a('still', 'right'), (rail_r, 10.8), (rail_r, merge_y), (cx, merge_y)])
    feed(ax, [_a('walk', 'right'), (rail_r, 7.7)])
    dot(ax, rail_r, 7.7)
    dot(ax, cx, merge_y)

    _legend(ax)
    _streaming_band(ax)
    _titles(fig, ax)
    return fig


def _legend(ax: Axes) -> None:
    """Symbol key (top-right), matching the brief's flowchart slide."""
    lx, lh = 11.0, 0.92
    rows = [(29.9, 'term', "Terminal (start / end)"),
            (28.6, 'proc', "Process"),
            (27.3, 'io',   "Input / Output"),
            (26.0, 'dec',  "Decision (Yes / No)")]
    ax.add_patch(Rectangle((lx - 1.7, 25.2), 5.3, 5.6, fc='white', ec='#c2ccd4',
                 lw=1.0, zorder=2))
    ax.text(lx + 0.05, 30.55, "Key", ha='center', fontsize=9.5, fontweight='bold',
            color=C_INK, zorder=6)
    for y, kind, label in rows:
        if kind == 'term':
            ax.add_patch(FancyBboxPatch((lx - 0.85, y - lh / 2), 1.7, lh,
                         boxstyle=f'round,pad=0,rounding_size={lh / 2}',
                         fc=C_TERM, ec='#0f1318', lw=1.2, zorder=3))
        elif kind == 'proc':
            ax.add_patch(Rectangle((lx - 0.85, y - lh / 2), 1.7, lh,
                         fc=C_PROC, ec=C_PROC_E, lw=1.2, zorder=3))
        elif kind == 'io':
            k = 0.28
            ax.add_patch(Polygon([(lx - 0.85 + k, y - lh / 2), (lx + 0.85 + k, y - lh / 2),
                                  (lx + 0.85 - k, y + lh / 2), (lx - 0.85 - k, y + lh / 2)],
                         closed=True, fc=C_IO, ec=C_IO_E, lw=1.2, zorder=3))
        else:
            ax.add_patch(Polygon([(lx, y + lh / 2), (lx + 0.95, y), (lx, y - lh / 2),
                                  (lx - 0.95, y)], closed=True, fc=C_DEC, ec=C_DEC_E,
                         lw=1.2, zorder=3))
        ax.text(lx + 1.15, y, label, ha='left', va='center', fontsize=8.4,
                color=C_INK, zorder=6)


def _titles(fig: Figure, ax: Axes) -> None:
    ax.text(6.7, 31.2, "Coding flowchart — standalone on-board firmware",
            ha='center', fontsize=14, fontweight='bold', color=C_INK)
    ax.text(6.85, 26.8, "loop()\nrepeats", ha='left', va='center',
            fontsize=8.0, color='#6b7884', style='italic')
    ax.text(6.7, -0.6,
            "cpx_fog_standalone.ino  ·  Freeze Index (Moore 2008) + movement-energy "
            "gate (Bachlin 2010).\nWhile cueing, the motor runs a non-blocking 2 Hz "
            "square wave — the rhythmic step cue.",
            ha='center', va='center', fontsize=7.8, color='#5b6770')


def _streaming_band(ax: Axes) -> None:
    """Foot strip: the optional *wireless* build. The same sense + cue loop still
    runs on the CPX, but the freeze **detector** is offloaded — the CPX streams the
    64 Hz accel over a short UART hop to an ESP32, which relays it to a
    laptop over Wi-Fi/TCP; the laptop's CNN decides, and the C/S cue command returns
    along the same path. Only the detector moves; the cue/debounce logic above is
    unchanged."""
    # bordered panel set apart from the main chart
    ax.add_patch(Rectangle((0.4, -4.3), 13.3, 2.75, fc='#f7f9fb', ec='#c2ccd4',
                 lw=1.0, zorder=2))
    ax.text(7.05, -1.85,
            "Wireless (streaming) build — the freeze detector moves off-board to the laptop",
            ha='center', va='center', fontsize=9.5, fontweight='bold', color=C_INK,
            zorder=6)

    # three boards, left → right (centres spread for wide, label-friendly gaps)
    xc_cpx, xc_esp, xc_pi = 2.5, 7.05, 11.6
    bw, bh, by = 2.8, 1.0, -2.95
    for xc, name, role in [(xc_cpx, "CPX",   "sense + cue"),
                           (xc_esp, "ESP32", "Wi-Fi relay"),
                           (xc_pi,  "laptop",  "CNN decision")]:
        ax.add_patch(FancyBboxPatch((xc - bw / 2, by - bh / 2), bw, bh,
                     boxstyle='round,pad=0,rounding_size=0.12',
                     fc=C_IO, ec=C_PROC_E, lw=1.4, zorder=4))
        ax.text(xc, by + 0.18, name, ha='center', va='center', fontsize=9.0,
                fontweight='bold', color=C_INK, zorder=6)
        ax.text(xc, by - 0.22, role, ha='center', va='center', fontsize=8.0,
                color=C_INK, zorder=6)

    # forward data path (slate): CPX → ESP32 → laptop, labelled with the transport
    def _tlabel(x: float, s: str) -> None:
        ax.text(x, by + 0.36, s, ha='center', va='center', fontsize=7.8,
                color=C_INK, fontweight='bold', zorder=7,
                bbox=dict(boxstyle='round,pad=0.14', fc='white', ec='none', alpha=0.95))
    arrow(ax, (xc_cpx + bw / 2, by), (xc_esp - bw / 2, by))
    arrow(ax, (xc_esp + bw / 2, by), (xc_pi - bw / 2, by))
    _tlabel((xc_cpx + xc_esp) / 2, "UART")
    _tlabel((xc_esp + xc_pi) / 2, "Wi-Fi / TCP")

    # return cue path (terracotta accent): laptop → … → CPX along the foot of the panel
    cue_y = -3.78
    ax.plot([xc_pi, xc_pi, xc_cpx, xc_cpx], [by - bh / 2, cue_y, cue_y, by - bh / 2],
            color=C_FREEZE_E, lw=1.5, solid_capstyle='round', zorder=3)
    ax.add_patch(FancyArrowPatch((xc_cpx, cue_y), (xc_cpx, by - bh / 2), arrowstyle='-|>',
                 mutation_scale=14, color=C_FREEZE_E, lw=1.5, shrinkA=0, shrinkB=0,
                 zorder=3))
    ax.text(xc_esp, cue_y, "C / S cue", ha='center', va='center', fontsize=8.2,
            color=C_FREEZE_E, fontweight='bold', zorder=7,
            bbox=dict(boxstyle='round,pad=0.16', fc='white', ec='none', alpha=0.95))


# ── CLI (mirrors sleeve_layout.py) ───────────────────────────────────────────
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='coding_flowchart',
                        help='output path stem, without extension')
    parser.add_argument('--formats', nargs='+', default=['png', 'pdf'],
                        help='one or more output formats (default: png pdf)')
    parser.add_argument('--dpi', type=int, default=200, help='raster DPI (default: 200)')
    args = parser.parse_args(argv)

    fig = build_figure()
    out = Path(args.out)
    if out.parent != Path('.'):
        out.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in args.formats:
        path = out.with_suffix(f".{ext}")
        fig.savefig(path, dpi=args.dpi, facecolor='white', bbox_inches='tight')
        written.append(str(path))
    plt.close(fig)
    print("wrote " + ", ".join(written))


if __name__ == '__main__':
    main()
