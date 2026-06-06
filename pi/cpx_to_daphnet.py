#!/usr/bin/env python3
"""Convert a CPX on-board capture CSV into a Daphnet-format .txt so the dashboard
server can REPLAY your recording as a live 64 Hz stream on the animated UI.

    python cpx_to_daphnet.py accuracy_figs/captures/ankle_demo.csv --freeze-phases 2
    python dashboard_server.py --replay accuracy_figs/captures/ankle_demo.daphnet.txt --replay-sensor ankle
    # then open http://127.0.0.1:8000/

CPX CSV columns : idx,t_s,ax_mg,ay_mg,az_mg,phase    (accel already in milli-g)
Daphnet columns : t_ms  ankle(x y z)  thigh(x y z)  trunk(x y z)  annot   (whitespace)
  annot 1 = no-freeze, 2 = freeze.  (0 would be dropped by the replay reader, so we
  never emit it.)  The CPX ankle accel goes in the ankle columns; thigh/trunk = 0,
  which is exactly what --replay-sensor ankle reads back.
"""
import argparse
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CPX capture CSV (from cpx_dump.py)")
    ap.add_argument("-o", "--out", default=None,
                    help="output Daphnet .txt (default: <csv-without-ext>.daphnet.txt)")
    ap.add_argument("--freeze-phases", default="2",
                    help="comma-separated phase indices that are FREEZE (default: 2)")
    args = ap.parse_args()

    freeze = {int(p) for p in args.freeze_phases.split(",") if p.strip()}
    out = args.out or os.path.splitext(args.csv)[0] + ".daphnet.txt"

    n = nf = 0
    with open(args.csv) as fh, open(out, "w") as w:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#") or s.startswith("idx"):
                continue
            f = s.split(",")
            if len(f) < 6:
                continue
            t_ms = int(round(float(f[1]) * 1000))
            ax, ay, az = (int(round(float(f[2]))),
                          int(round(float(f[3]))),
                          int(round(float(f[4]))))
            phase = int(f[5])
            annot = 2 if phase in freeze else 1
            nf += annot == 2
            w.write(f"{t_ms} {ax} {ay} {az} 0 0 0 0 0 0 {annot}\n")
            n += 1

    if n == 0:
        sys.exit(f"no data rows parsed from {args.csv}")
    print(f"wrote {out}")
    print(f"  {n} samples  ({nf} freeze, {n - nf} non-freeze)  freeze-phases={sorted(freeze)}")
    print(f"  replay:  python dashboard_server.py --replay {out} --replay-sensor ankle")


if __name__ == "__main__":
    main()
