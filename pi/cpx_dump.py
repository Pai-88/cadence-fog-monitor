#!/usr/bin/env python3
"""Pull a stored on-board capture off the CPX SPI-flash logger over USB.

The cpx_fog_logger firmware records accel + auto-labelled phase to the CPX's
2 MB SPI flash while the wearer walks/stands/freezes UNTETHERED.  Plug the board
into the laptop afterwards and run this to stream the stored CSV back.

  python cpx_dump.py walk1            # saves accuracy_figs/captures/walk1.csv
  python cpx_dump.py walk1 --status   # just print sample count, don't dump
  python cpx_dump.py walk1 --port /dev/cu.usbmodem101

It auto-detects /dev/cu.usbmodem*, asserts DTR (so the CPX's USB CDC starts
emitting), sends DUMP, and saves every line through the `#END` marker.  The raw
`#` provenance lines are kept in the file — analyze_worksheet.py skips them and
they record the exact --labels / --freeze-phases to use.
"""
import argparse
import glob
import os
import re
import sys
import time

try:
    import serial  # pyserial
except ImportError:
    sys.exit("pyserial not installed — run:  pip install pyserial\n"
             "(or use the project venv: "
             "/Users/paing/Documents/scenario2_pd/pi/.venv/bin/python cpx_dump.py ...)")

CAPTURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "recordings")
BAUD = 115200


def find_port(explicit=None):
    if explicit:
        return explicit
    hits = sorted(glob.glob("/dev/cu.usbmodem*"))
    if not hits:
        sys.exit("[FAIL] no /dev/cu.usbmodem* found — is the CPX plugged into "
                 "the laptop and showing on USB? (check `ls /dev/cu.usbmodem*`)")
    if len(hits) > 1:
        print(f"[note] multiple usbmodem ports {hits}; using {hits[0]} "
              f"(override with --port)")
    return hits[0]


def open_port(port):
    s = serial.Serial()
    s.port = port
    s.baudrate = BAUD
    s.timeout = 1.0
    s.dtr = True   # assert DTR -> CDC lineState>0 -> CPX starts emitting on USB
    s.rts = True
    s.open()
    # settle: covers any USB re-enumeration / firmware boot after the host opens
    time.sleep(2.5)
    s.reset_input_buffer()
    return s


def send(s, cmd):
    s.reset_input_buffer()
    s.write((cmd + "\n").encode())
    s.flush()


def read_status(s):
    send(s, "STATUS")
    t0 = time.time()
    while time.time() - t0 < 4:
        line = s.readline().decode(errors="replace").strip()
        if line.startswith("#STATUS"):
            return line
    return None


def status_samples(line):
    m = re.search(r"samples=(\d+)", line or "")
    return int(m.group(1)) if m else None


def wait_for_recording(s, timeout=600):
    """Block until a recording completes on the board, then return its count.

    Detection is robust to any baseline: a recording cycle erases the board,
    so STATUS goes unanswered while it ERASES/COUNTS-DOWN/RUNS (board ignores
    serial), then answers again with samples>0 once it reaches DONE.  We watch
    for that busy→idle-with-data transition rather than comparing counts.
    """
    print("[wait] watching the board — flip the slide switch, follow the OLED, "
          "flip OFF when done. (Ctrl-C to give up.)", flush=True)
    t0 = time.time()
    seen_busy = False
    misses = 0
    last_beat = 0.0
    while time.time() - t0 < timeout:
        st = read_status(s)        # ~instant if answered, ~4 s if board is busy
        now = time.time()
        if st is None:
            misses += 1
            if not seen_busy and misses >= 2:
                seen_busy = True
                print("[wait] board went busy → recording in progress…", flush=True)
        else:
            n = status_samples(st)
            if seen_busy and n and n > 0:
                print(f"[wait] recording finished — {n} samples stored", flush=True)
                return n
            misses = 0
            if now - last_beat > 8:
                last_beat = now
                state = "ready to start" if not seen_busy else "finishing…"
                print(f"[wait] {int(now - t0):>3}s  armed ({state})  {st}", flush=True)
    return None


def dump(s):
    """Send DUMP, collect lines through `#END`. Returns (lines, end_count)."""
    send(s, "DUMP")
    lines = []
    end_count = None
    t0 = time.time()
    last_rx = time.time()
    while True:
        line = s.readline()
        if line:
            last_rx = time.time()
            text = line.decode(errors="replace").rstrip("\r\n")
            if text.startswith("#ERR"):
                sys.exit(f"[device] {text}")
            lines.append(text)
            if text.startswith("#END"):
                parts = text.split(",")
                if len(parts) > 1 and parts[1].strip().isdigit():
                    end_count = int(parts[1])
                break
        else:
            # idle: bail out if the device has gone quiet for too long
            if time.time() - last_rx > 8:
                sys.exit("[FAIL] device stopped sending before `#END` "
                         f"({len(lines)} lines so far) — try re-running, or "
                         "power-cycle the board and DUMP again")
        if time.time() - t0 > 180:
            sys.exit("[FAIL] dump exceeded 180 s — aborting")
    return lines, end_count


def unique_path(name):
    if not name.endswith(".csv"):
        name += ".csv"
    path = os.path.join(CAPTURES_DIR, name)
    if not os.path.exists(path):
        return path
    base = name[:-4]
    i = 2
    while True:
        cand = os.path.join(CAPTURES_DIR, f"{base}_{i}.csv")
        if not os.path.exists(cand):
            print(f"[note] {name} exists — saving as {os.path.basename(cand)} "
                  f"(not overwriting your data)")
            return cand
        i += 1


def main():
    ap = argparse.ArgumentParser(description="Dump CPX on-board capture to CSV.")
    ap.add_argument("name", help="output name, saved under captures/ as <name>.csv")
    ap.add_argument("--port", help="serial port (default: auto-detect usbmodem)")
    ap.add_argument("--status", action="store_true",
                    help="only print stored sample count, don't dump")
    ap.add_argument("--wait", action="store_true",
                    help="watch the board, auto-dump once a recording completes")
    ap.add_argument("--timeout", type=int, default=600,
                    help="--wait: seconds to wait for a recording (default 600)")
    args = ap.parse_args()

    port = find_port(args.port)
    print(f"[opening] {port} @ {BAUD} (asserting DTR, settling 2.5 s)…")
    try:
        s = open_port(port)
    except Exception as e:
        sys.exit(f"[OPEN FAIL] {e}")

    try:
        st = read_status(s)
        if st:
            print(f"[device] {st}")
        else:
            print("[warn] no #STATUS reply (older firmware?) — trying DUMP anyway")

        if args.status:
            return

        if args.wait:
            got = wait_for_recording(s, timeout=args.timeout)
            if got is None:
                sys.exit(f"[FAIL] no recording completed within {args.timeout}s")
        elif st and "samples=0" in st:
            sys.exit("[FAIL] device reports 0 stored samples — record a run "
                     "first (flip the slide switch and follow the OLED).")

        print("[dumping] sending DUMP, collecting until #END…")
        lines, end_count = dump(s)
    finally:
        s.close()

    # row lines = non-comment, non-header, non-blank
    data_rows = [ln for ln in lines
                 if ln and not ln.startswith("#") and not ln.startswith("idx,")]
    path = unique_path(args.name)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[saved] {path}")
    print(f"        {len(data_rows)} data rows"
          + (f"  (device #END count={end_count})" if end_count is not None else ""))
    if end_count is not None and end_count != len(data_rows):
        print(f"[warn] row count {len(data_rows)} != #END {end_count} — "
              f"possible truncation; re-run the dump to be safe")

    # surface the analyze command the firmware embedded
    analyze = next((ln for ln in lines if ln.startswith("# analyze:")), None)
    venv = "/Users/paing/Documents/scenario2_pd/pi/.venv/bin/python"
    print("\nNext — score sensitivity & specificity:")
    if analyze:
        flags = analyze[len("# analyze:"):].strip()
        print(f"  {venv} analyze_worksheet.py {path} {flags} --threshold 1.815")
    else:
        print(f"  {venv} analyze_worksheet.py {path} "
              f"--labels still,walk,freeze,walk,freeze,walk,freeze,walk,still "
              f"--freeze-phases 2,4,6 --threshold 1.815")


if __name__ == "__main__":
    main()
