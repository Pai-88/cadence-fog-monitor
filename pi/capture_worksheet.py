"""
Serial → CSV capture harness for the ENGF0031 Accuracy Worksheet.

Pairs with firmware/cpx_accel_logger/cpx_accel_logger.ino. That sketch logs the
CPX accelerometer to RAM while you run a test protocol (RIGHT button = start /
mark a new phase, LEFT button = stop & dump), then prints the whole capture as
CSV bracketed by ``=== DATA START ===`` / ``=== DATA END ===``.

This script is just a smart serial monitor: it echoes everything the board says
(so you can watch the phase markers live) and, every time it sees a DATA block,
saves it to the next free ``captureN.csv`` — no copy-pasting out of the Serial
Monitor. Do Capture 1, then press RIGHT again for Capture 2; each lands in its
own file. Ctrl-C to finish, then run ``analyze_worksheet.py``.

    # auto-detect the CPX (ignores the ESP32 JTAG port) and capture to ./
    python capture_worksheet.py
    # be explicit about the port / output dir
    python capture_worksheet.py --port /dev/cu.usbmodem101 --outdir captures

Needs only pyserial (``pip install pyserial``) — no torch / numpy. The board
needs no ESP32 and no Wi-Fi for this: just USB power.
"""
from __future__ import annotations

import argparse
import os
import time

import serial               # pyserial
from serial.tools import list_ports

BAUD = 115200
DATA_START = "=== DATA START ==="
DATA_END = "=== DATA END ==="
# Lines emitted just before DATA START that are worth keeping as provenance.
PREAMBLE_PREFIXES = ("# samples=", "# phase boundaries", "--- LOGGING FINISHED")


def find_cpx_port(explicit: str | None) -> str:
    """Locate the Circuit Playground's USB serial port.

    If ``explicit`` is given, trust it. Otherwise scan the USB serial ports and
    pick the Adafruit/SAMD board, explicitly skipping the ESP32 (which shows up
    as a JTAG/serial-debug unit) so the two boards don't get confused.
    """
    if explicit:
        return explicit

    ports = list(list_ports.comports())
    def describe(p: object) -> str:
        return " ".join(str(getattr(p, a, "") or "")
                         for a in ("description", "manufacturer", "product")).lower()

    # Only USB-CDC modems can be the CPX (skips debug-console / Bluetooth ports),
    # and hard-exclude the ESP32 native-USB JTAG port; it is NOT the logger.
    esp_markers = ("jtag", "esp32", "espressif", "serial debug")
    candidates = [p for p in ports
                  if "usbmodem" in str(p.device).lower()
                  and not any(m in describe(p) for m in esp_markers)]

    # Prefer a clear Circuit Playground / Adafruit / SAMD match.
    cpx_markers = ("circuit playground", "circuitplayground", "playground",
                   "adafruit", "samd")
    preferred = [p for p in candidates if any(m in describe(p) for m in cpx_markers)]

    pick = preferred or candidates
    if len(pick) == 1:
        p = pick[0]
        print(f"  auto-detected CPX → {p.device}  ({describe(p).strip() or 'unknown'})")
        return p.device

    # Ambiguous or nothing found — show the user what's on the bus and bail.
    print("  Could not unambiguously pick the CPX port. Ports seen:")
    for p in ports:
        print(f"    {p.device:24s} {describe(p).strip()}")
    if not ports:
        print("    (none) — is the board plugged in and not still re-enumerating?")
    raise SystemExit("  → re-run with  --port /dev/cu.usbmodemXXXX")


def next_capture_path(outdir: str, prefix: str) -> str:
    """Return the next free <outdir>/<prefix>N.csv (1-based, never overwrites)."""
    i = 1
    while True:
        path = os.path.join(outdir, f"{prefix}{i}.csv")
        if not os.path.exists(path):
            return path
        i += 1


def summarise(rows: list[str]) -> str:
    """One-line summary of a captured CSV block (data rows only)."""
    data = [r for r in rows if r and r[0].isdigit()]
    n = len(data)
    dur = n / 64.0
    phases = sorted({r.split(",")[-1] for r in data}) if data else []
    return f"{n} samples, ~{dur:.1f}s, phases {{{','.join(phases)}}}"


def open_serial(port: str) -> serial.Serial:
    ser = serial.Serial(port, BAUD, timeout=0.2)
    time.sleep(0.2)
    return ser


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None, help="serial device (default: auto-detect)")
    ap.add_argument("--outdir", default=".", help="where to write captureN.csv")
    ap.add_argument("--prefix", default="capture", help="output filename prefix")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    port = find_cpx_port(args.port)

    print("=" * 64)
    print("  Accuracy-Worksheet capture harness")
    print("=" * 64)
    print("  CONTROLS (on the board):")
    print("    RIGHT button  → start logging / mark a new phase boundary")
    print("    LEFT  button  → stop & dump (this script saves it to a file)")
    print("  Strap the board at the capture site (ANKLE for FoG, WRIST for tremor).")
    print("  After a dump, press RIGHT again for the next capture. Ctrl-C when done.\n")

    try:
        ser = open_serial(port)
    except serial.SerialException as e:
        raise SystemExit(f"  could not open {port}: {e}")
    print(f"  listening on {port} @ {BAUD} … waiting for you to press RIGHT.\n")

    preamble: list[str] = []     # recent '#'/'---' lines seen before a DATA block
    capturing = False
    block: list[str] = []
    saved = 0

    try:
        while True:
            try:
                raw = ser.readline()
            except serial.SerialException:
                # CPX fell off the bus (e.g. a reset / replug). Try to recover.
                print("\n  [serial dropped — trying to reconnect …]")
                ser.close()
                ser = _reconnect(port)
                continue
            if not raw:
                continue
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            print(f"  | {line}")          # live monitor echo

            if line == DATA_START:
                capturing = True
                block = list(preamble)    # carry the provenance comments in
                continue
            if line == DATA_END:
                capturing = False
                path = next_capture_path(args.outdir, args.prefix)
                with open(path, "w") as f:
                    f.write("\n".join(block) + "\n")
                saved += 1
                print(f"\n  ✓ saved → {os.path.abspath(path)}")
                print(f"    {summarise(block)}")
                print("    press RIGHT for the next capture, or Ctrl-C to finish.\n")
                preamble.clear()
                continue

            if capturing:
                block.append(line)
            elif line.startswith(PREAMBLE_PREFIXES):
                # Normalise the human-readable banner line to a CSV comment.
                preamble.append(line if line.startswith("#") else f"# {line.strip('- ')}")
    except KeyboardInterrupt:
        print(f"\n  done — {saved} capture(s) saved to {os.path.abspath(args.outdir)}/")
        print("  next:  python analyze_worksheet.py")
    finally:
        try:
            ser.close()
        except Exception:
            pass


def _quiet_cpx_port() -> str | None:
    """Silent CPX-port detect (no stdout) for use inside the reconnect loop."""
    for p in list_ports.comports():
        dev = str(p.device)
        if "usbmodem" not in dev.lower():
            continue
        desc = " ".join(str(getattr(p, a, "") or "")
                        for a in ("description", "manufacturer", "product")).lower()
        if not any(m in desc for m in ("jtag", "esp32", "espressif", "serial debug")):
            return dev
    return None


def _reconnect(port: str, max_seconds: float = 300.0) -> serial.Serial:
    """Wait (patiently, quietly) for the CPX to come back, then re-open it.

    With a marginal cable the board can drop to a 'powered but no serial node'
    state that only a RESET tap clears — so we don't give up after a few tries;
    we wait up to ``max_seconds`` and repeatedly nudge the user to tap RESET.
    """
    deadline = time.monotonic() + max_seconds
    last_hint = 0.0
    while time.monotonic() < deadline:
        cand = _quiet_cpx_port() or port
        try:
            return open_serial(cand)
        except serial.SerialException:
            pass
        if time.monotonic() - last_hint > 8:
            print("  [link down — tap the center RESET button on the CPX to bring it back]")
            last_hint = time.monotonic()
        time.sleep(0.5)
    raise SystemExit("  gave up waiting for the CPX (5 min). Re-run when ready.")


if __name__ == "__main__":
    main()
