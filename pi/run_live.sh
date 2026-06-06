#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_live.sh — launch the live freeze-monitor dashboard off the CPX over USB.
#
# The CPX's USB tty number CHANGES every time the board resets or the cable
# reconnects (e.g. /dev/cu.usbmodem101 -> usbmodem1101). Pinning a fixed port
# breaks the demo after any reset, so this script probes every usbmodem port,
# picks the one actually streaming "ax,ay,az", and launches the dashboard on it.
#
# Usage:  ./run_live.sh
# Then open http://localhost:8000
# ─────────────────────────────────────────────────────────────────────────────
cd "$(dirname "$0")" || exit 1

pkill -f "serial-monitor" 2>/dev/null   # Arduino IDE's Serial Monitor steals the port
pkill -f "dashboard_server.py" 2>/dev/null
sleep 1

echo "Looking for the live CPX stream ..."
PORT=""
for attempt in $(seq 1 20); do
  PORT=$(.venv/bin/python - <<'PY'
import glob, time, serial
best=None
for p in sorted(glob.glob("/dev/cu.usbmodem*")):
    try:
        s=serial.Serial(); s.port=p; s.baudrate=115200; s.timeout=0.3; s.dtr=True
        s.open(); time.sleep(0.3); s.reset_input_buffer()
        raw=b""; t=time.time()
        while time.time()-t<1.2: raw+=s.read(512)
        s.close()
        acc=sum(1 for l in raw.decode(errors="replace").splitlines()
                if len(l.split(","))==3 and all(q.strip().lstrip("-").isdigit() for q in l.split(",")))
        if acc>15:
            best=p; break
    except Exception:
        pass
print(best or "")
PY
)
  [ -n "$PORT" ] && break
  echo "  no stream yet (attempt $attempt) — is the board plugged in and flashed with the detector/streamer?"
  sleep 1
done

if [ -z "$PORT" ]; then
  echo "No streaming CPX port found. Check the USB cable (data, not charge-only) and that the board is running cpx_fog_standalone or cpx_fog_sensor."
  exit 1
fi

echo "Live stream on $PORT  ->  dashboard at http://localhost:8000"

# Standing-still gate: a freeze/walk is only real when band-power energy clears
# this floor. Measured resting energy on the bench is ~25-56 (p90 ~53), so 150
# (~4x the resting median, matching the board's button-A calibration) keeps quiet
# rest reading STILL while leaving walking (hundreds-to-thousands) well above it.
# Without it the gate defaults to 0 and EVERYTHING reads WALKING. Re-measure with
# pi/.venv/bin/python and the /ws feed if the mount or board changes.
exec .venv/bin/python dashboard_server.py --transport serial --port "$PORT" --still-floor 150
