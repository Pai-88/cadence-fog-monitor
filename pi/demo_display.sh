#!/usr/bin/env bash
# Cadence demo launcher — play a recorded CPX capture on the live dashboard.
#
#   ./demo_display.sh                 # display the known-good 'smoketest' capture
#   ./demo_display.sh mycapture       # display ../recordings/mycapture.csv
#
# Then open http://127.0.0.1:8000/ in your browser.
# Press Ctrl-C in this terminal to stop the dashboard.
set -euo pipefail

cd "$(dirname "$0")"                       # run from the pi/ folder no matter what
PY=.venv/bin/python                        # the venv python (has the full server stack)

NAME="${1:-smoketest}"                     # capture name (no extension); default smoketest (known-good freeze)
CSV="../recordings/${NAME}.csv"
DAPHNET="../recordings/${NAME}.daphnet.txt"
FREEZE_PHASES="${2:-2,4}"                  # which phase indices are FREEZE (CSV labels only)
# Live detector config. We drive the dashboard off the glass-box Freeze Index
# (not the CNN) so the display matches the worksheet + presentation script:
#   * --detector fi      -> reads the RAW stream; band-passing flattens the FI
#                           ratio so walking and freezing look alike, so the FI
#                           path is deliberately unfiltered (verified on smoketest).
#   * --fi-threshold     -> the worksheet operating point.
#   * --still-floor      -> movement-energy gate; below it = STILL, so quiet
#                           standing no longer reads as a freeze.
FI_THRESHOLD="${3:-1.815}"                 # Freeze-Index trip level (worksheet optimum)
STILL_FLOOR="${4:-8000}"                   # standing-still energy gate (same CPX scale)

if [ ! -f "$CSV" ]; then
  echo "ERROR: $CSV not found."
  echo "Captures available:"
  ls -1 ../recordings/*.csv 2>/dev/null | sed 's#.*/#  #'
  exit 1
fi

echo "[1/3] converting $CSV  (freeze phases = $FREEZE_PHASES)"
"$PY" cpx_to_daphnet.py "$CSV" --freeze-phases "$FREEZE_PHASES"

echo "[2/3] stopping any old dashboard server"
pkill -f "dashboard_server.py" 2>/dev/null || true
# wait for port 8000 to actually free up, force-kill if it lingers
for _ in 1 2 3 4 5; do
  lsof -iTCP:8000 -sTCP:LISTEN -n -P >/dev/null 2>&1 || break
  sleep 1
done
if lsof -iTCP:8000 -sTCP:LISTEN -n -P >/dev/null 2>&1; then
  pkill -9 -f "dashboard_server.py" 2>/dev/null || true
  sleep 1
fi

echo "[3/3] starting dashboard  ->  open  http://127.0.0.1:8000/"
echo "      detector=Freeze-Index  threshold=$FI_THRESHOLD  still-floor=$STILL_FLOOR"
echo "      (Ctrl-C here to stop)"
exec "$PY" dashboard_server.py --replay "$DAPHNET" --replay-sensor ankle \
     --detector fi --fi-threshold "$FI_THRESHOLD" --still-floor "$STILL_FLOOR"
