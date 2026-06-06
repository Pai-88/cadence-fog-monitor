"""
Ankle-sleeve component layout for the Parkinson's closed-loop gait garment
(ENGF0031 Scenario 2). Renders a labelled engineering figure — lateral view,
transverse cross-section, and the flat-pocket layer stack — to PNG + PDF.

Design rationale baked into the figure:
  * snug compression sleeve, flat internal pockets, NOTHING hangs — a swinging
    mass would inject ~1-2 Hz pendulum motion into the locomotor band and corrupt
    the accelerometer features the freeze detector relies on;
  * CPX (~10 g) and LiPo (~15 g) on OPPOSITE faces so the combined centre of mass
    stays near the limb axis (the sleeve will not rotate or sag);
  * coin motor in a thin pocket pressed against the skin → strong tactile cue at
    low power;
  * wires run inside sewn fabric channels (no free leads to snag);
  * slide switch stays reachable through the fabric so the wearer can silence it;
  * USB is only needed for the tethered (laptop) build; the standalone
    on-board build needs no cable at all.

Run:  python sleeve_layout.py                 → sleeve_layout.png, sleeve_layout.pdf
      python sleeve_layout.py --out build/sleeve --formats png
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)

matplotlib.use("Agg")  # headless backend: render straight to file, no display

# ── palette ──
C_SKIN, C_SKIN_E     = '#f4d9c4', '#cda484'
C_FABRIC, C_FABRIC_E = '#9fb8d4', '#587ba6'
C_BOARD, C_BOARD_E   = '#2f3a45', '#161b21'
C_LIPO, C_LIPO_E     = '#9aa7b1', '#5b6770'
C_MOTOR, C_MOTOR_E   = '#d9a225', '#a87a12'
C_WIRE, C_INK        = '#555555', '#1f2933'

plt.rcParams.update({'font.size': 9.5, 'font.family': 'DejaVu Sans'})

LBOX = dict(boxstyle='round,pad=0.32', fc='white', ec='#90a0ad', lw=0.8, alpha=0.96)


# ── small annotation helpers ────────────────────────────────────────────────
def lead(
    ax: Axes,
    xy: tuple[float, float],
    xytext: tuple[float, float],
    text: str,
    ha: str = 'left',
    va: str = 'center',
    fs: float = 9.0,
    rad: float = 0.0,
) -> None:
    """A boxed label with a thin leader line from ``xytext`` to a point ``xy``."""
    ax.annotate(text, xy=xy, xytext=xytext, ha=ha, va=va, fontsize=fs, color=C_INK,
                bbox=LBOX, annotation_clip=False, zorder=10,
                arrowprops=dict(arrowstyle='-', color='#5b6770', lw=1.0,
                                connectionstyle=f'arc3,rad={rad}',
                                shrinkA=4, shrinkB=3))


def dirn(
    ax: Axes, x: float, y: float, text: str, ha: str = 'center', va: str = 'center'
) -> None:
    """An italic anatomical-direction label (proximal/distal/anterior/...)."""
    ax.text(x, y, text, ha=ha, va=va, fontsize=8.5, style='italic',
            color='#6b7884', clip_on=False)


# ════════════════════════════════════════════════════════════════════════════
#  A · LATERAL (SIDE) VIEW
# ════════════════════════════════════════════════════════════════════════════
def panel_a_lateral(ax: Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.2)
    ax.set_title("A · lateral (side) view — right leg", loc='left',
                 fontsize=11, fontweight='bold', color=C_INK)

    leg_pts = [(3.2, 12.6), (3.05, 8.0), (3.18, 5.0), (3.10, 4.0), (2.55, 3.55),
               (1.2, 3.15), (0.55, 2.75), (0.5, 2.2), (0.95, 2.0), (3.45, 1.92),
               (4.4, 2.12), (4.6, 2.95), (4.8, 4.2), (5.05, 6.0), (4.98, 9.0),
               (4.82, 12.6)]
    ax.add_patch(Polygon(leg_pts, closed=True, fc=C_SKIN, ec=C_SKIN_E, lw=1.6, zorder=1))

    # compression sleeve band over ankle + lower shin
    ax.add_patch(FancyBboxPatch((2.5, 3.85), 2.95, 4.25,
                 boxstyle='round,pad=0,rounding_size=0.45',
                 fc=C_FABRIC, ec=C_FABRIC_E, lw=1.4, alpha=0.45, zorder=2))

    # CPX board (lateral face, visible)
    cx, cy = 4.0, 6.15
    ax.add_patch(FancyBboxPatch((cx - 0.75, cy - 0.75), 1.5, 1.5,
                 boxstyle='round,pad=0,rounding_size=0.16',
                 fc=C_BOARD, ec=C_BOARD_E, lw=1.4, zorder=4))
    for k in range(10):                       # NeoPixel ring hint
        a = math.radians(36 * k)
        ax.add_patch(Circle((cx + 0.52 * math.cos(a), cy + 0.52 * math.sin(a)),
                            0.055, fc='#7fd1c7', ec='none', zorder=5))
    ax.add_patch(Circle((cx, cy), 0.17, fc='#cdd6dd', ec='none', zorder=5))  # centre IC
    ax.add_patch(Rectangle((cx - 0.16, cy - 0.95), 0.32, 0.2, fc='#cdd6dd',
                 ec=C_BOARD_E, lw=0.8, zorder=5))                            # USB notch

    # coin vibration motor (against skin, just distal of the board)
    ax.add_patch(Circle((4.0, 4.62), 0.34, fc=C_MOTOR, ec=C_MOTOR_E, lw=1.3, zorder=4))
    ax.add_patch(Circle((4.0, 4.62), 0.12, fc=C_MOTOR_E, ec='none', zorder=5))

    # wire channel CPX → motor
    ax.plot([4.0, 4.0], [5.32, 4.98], ls=(0, (3.5, 2.5)), color=C_WIRE, lw=1.5, zorder=3)

    # leaders (all to the right margin, top→bottom)
    lead(ax, (5.35, 7.6), (6.25, 9.6),
         "compression sleeve\nsnug, even pressure — nothing hangs")
    lead(ax, (4.5, 7.1), (6.25, 8.0),
         "LiPo battery on the medial\n(far) face — see panel B", rad=-0.12)
    lead(ax, (4.72, 6.2), (6.25, 6.0),
         "Circuit Playground Express\nflat pocket · accelerometer coupled\n"
         "to the limb · slide switch reachable\nthrough the fabric")
    lead(ax, (4.32, 4.62), (6.25, 3.9),
         "coin vibration motor\nthin pocket, against the skin")
    lead(ax, (4.0, 5.1), (1.55, 5.0),
         "wires in a sewn\nfabric channel", ha='center', rad=0.15)

    # directions + build note
    dirn(ax, 4.0, 12.95, "proximal (knee)")
    dirn(ax, 1.15, 1.35, "distal (foot)")
    dirn(ax, 1.85, 10.6, "anterior\n(shin)")
    dirn(ax, 6.0, 11.6, "posterior\n(calf)")
    ax.text(0.15, 0.55,
            "standalone build: no cable.\ntethered build: USB → laptop, with a\n"
            "strain-relief service loop.",
            fontsize=8.3, color='#3b4754', va='bottom', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', fc='#eef3f8', ec='#90a0ad', lw=0.8))


# ════════════════════════════════════════════════════════════════════════════
#  B · TRANSVERSE CROSS-SECTION
# ════════════════════════════════════════════════════════════════════════════
def panel_b_cross_section(ax: Axes) -> None:
    ax.set_xlim(-6.6, 7.2)
    ax.set_ylim(-6.0, 6.0)
    ax.set_title("B · cross-section (looking toward the foot)", loc='left',
                 fontsize=11, fontweight='bold', color=C_INK)

    ax.add_patch(Ellipse((0, 0), 9.3, 8.1, fc=C_FABRIC, ec=C_FABRIC_E,
                         lw=1.4, alpha=0.5, zorder=1))         # sleeve fabric
    ax.add_patch(Ellipse((0, 0), 8.0, 6.9, fc=C_SKIN, ec=C_SKIN_E, lw=1.4, zorder=2))

    # components hugging the limb inside their pockets
    ax.add_patch(FancyBboxPatch((3.55, -1.35), 0.95, 2.7,
                 boxstyle='round,pad=0,rounding_size=0.12',
                 fc=C_BOARD, ec=C_BOARD_E, lw=1.3, zorder=4))     # CPX lateral
    ax.add_patch(FancyBboxPatch((-4.6, -1.7), 1.05, 3.4,
                 boxstyle='round,pad=0,rounding_size=0.12',
                 fc=C_LIPO, ec=C_LIPO_E, lw=1.3, zorder=4))       # LiPo medial
    ax.add_patch(FancyBboxPatch((-0.5, 2.95), 1.0, 0.78,
                 boxstyle='round,pad=0,rounding_size=0.1',
                 fc=C_MOTOR, ec=C_MOTOR_E, lw=1.3, zorder=4))     # motor anterior

    # balance arc CPX <-> LiPo
    ax.add_patch(FancyArrowPatch((3.6, -1.7), (-3.7, -2.0),
                 connectionstyle='arc3,rad=-0.32', arrowstyle='<->',
                 mutation_scale=12, color='#39506b', lw=1.4, zorder=3))
    ax.text(0, -4.0, "opposite faces → combined CG near the limb axis",
            ha='center', fontsize=8.2, color='#39506b', style='italic')

    # mass tags
    ax.text(4.02, 1.75, "CPX ≈10 g", ha='center', fontsize=8.0, color=C_BOARD)
    ax.text(-4.07, 2.05, "LiPo ≈15 g", ha='center', fontsize=8.0, color=C_LIPO_E)

    # leaders
    lead(ax, (4.5, 0.4), (5.55, 2.7), "CPX\n(flat pocket)", rad=-0.12)
    lead(ax, (-4.6, 0.4), (-6.4, 2.6), "LiPo\n(flat pocket)", ha='right', rad=0.12)
    lead(ax, (0.45, 3.5), (2.7, 4.6), "vibration motor\n(against skin)", rad=-0.12)
    lead(ax, (-2.6, 2.9), (-5.8, 4.4), "compression\nfabric", ha='right', rad=0.1)

    # direction labels
    dirn(ax, 0, 5.35, "anterior (shin)")
    dirn(ax, 0, -5.4, "posterior (calf)")
    dirn(ax, 6.4, -3.6, "lateral (outer)")
    dirn(ax, -5.7, -3.8, "medial (inner)")


# ════════════════════════════════════════════════════════════════════════════
#  C · FLAT-POCKET LAYER STACK + key specs
# ════════════════════════════════════════════════════════════════════════════
def panel_c_layer_stack(ax: Axes) -> None:
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 6)
    ax.set_title("C · flat-pocket layer stack (radial section)", loc='left',
                 fontsize=11, fontweight='bold', color=C_INK)

    y0, h, x = 2.5, 2.0, 0.7
    #         label                       w     fill      edge       text    inside  callout-y
    layers = [("limb /\nskin",            1.5,  C_SKIN,   C_SKIN_E,  C_INK,  True,  None),
              ("inner fabric",            0.45, C_FABRIC, C_FABRIC_E, C_INK, False, 4.85),
              ("component\n(CPX / LiPo)", 1.25, C_BOARD,  C_BOARD_E, 'white', True, None),
              ("outer fabric\n(closes pocket)", 0.55, C_FABRIC, C_FABRIC_E, C_INK, False, 5.5)]
    for name, w, fc, ec, tc, inside, cyoff in layers:
        ax.add_patch(Rectangle((x, y0), w, h, fc=fc, ec=ec, lw=1.3))
        xc = x + w / 2
        if inside:
            ax.text(xc, y0 + h / 2, name, ha='center', va='center',
                    fontsize=8.0, color=tc)
        else:
            assert cyoff is not None  # non-inside layers always carry a callout-y
            ax.annotate(name, xy=(xc, y0 + h), xytext=(xc, cyoff),
                        ha='center', va='bottom', fontsize=7.8, color=C_INK,
                        arrowprops=dict(arrowstyle='-', color='#5b6770', lw=0.8))
        x += w
    xend = x

    ax.text(0.7, y0 - 0.5, "← toward limb", ha='left', fontsize=8,
            color='#6b7884', style='italic')
    ax.text(xend, y0 - 0.5, "outside →", ha='right', fontsize=8,
            color='#6b7884', style='italic')
    # thickness bracket BELOW the protrusion (everything past the skin)
    bx0, bx1, by = 2.2, xend, y0 - 1.2
    ax.annotate('', xy=(bx0, by), xytext=(bx1, by),
                arrowprops=dict(arrowstyle='<->', color=C_INK, lw=1.1))
    ax.text((bx0 + bx1) / 2, by - 0.3, "≈3–4 mm — sits flush, nothing to snag",
            ha='center', va='top', fontsize=8.2, color=C_INK)

    specs = ("Why it is laid out this way\n"
             "• snug compression → the sensor reads limb motion,\n"
             "   not pendulum swing (clean freeze-index features)\n"
             "• CPX & LiPo on opposite faces → balanced, no sag/rotation\n"
             "• motor against skin → strong cue at low motor power\n"
             "• ≈35 g total — lighter than a wristwatch\n"
             "• slide switch accessible; USB only for the tethered build")
    ax.text(6.6, 3.0, specs, ha='left', va='center', fontsize=8.6,
            color=C_INK, linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.5', fc='#eef3f8', ec='#90a0ad', lw=0.8))


# ════════════════════════════════════════════════════════════════════════════
#  FIGURE ASSEMBLY + CLI
# ════════════════════════════════════════════════════════════════════════════
def build_figure() -> Figure:
    """Assemble the three-panel ankle-sleeve layout figure."""
    fig = plt.figure(figsize=(13.6, 8.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.28], height_ratios=[1.28, 1.0],
                          hspace=0.16, wspace=0.10,
                          left=0.035, right=0.99, top=0.885, bottom=0.04)
    ax_lateral = fig.add_subplot(gs[:, 0])
    ax_cross = fig.add_subplot(gs[0, 1])
    ax_stack = fig.add_subplot(gs[1, 1])
    for ax in (ax_lateral, ax_cross, ax_stack):
        ax.set_aspect('equal')
        ax.axis('off')

    fig.suptitle("Parkinson's gait garment — ankle-sleeve component layout",
                 fontsize=15, fontweight='bold', y=0.975)
    fig.text(0.5, 0.915,
             "snug compression sleeve  ·  flat internal pockets  ·  nothing hangs  ·  "
             "≈35 g total  (CPX 10 g + LiPo 15 g + motor 2 g)",
             ha='center', fontsize=10.5, color='#3b4754')

    panel_a_lateral(ax_lateral)
    panel_b_cross_section(ax_cross)
    panel_c_layer_stack(ax_stack)
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='sleeve_layout',
                        help='output path stem, without extension (default: sleeve_layout)')
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
