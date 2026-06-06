#!/bin/bash
# ============================================================
#  CADENCE — ONE-CLICK LIVE DISPLAY
#  Just double-click this file in Finder. No typing.
#
#  • If the CPX is plugged in with a recording, it pulls it
#    off the board and shows it live.
#  • If no board is connected, it shows your most recent
#    capture instead.
#  A browser opens automatically at http://127.0.0.1:8000/
#  Press Ctrl-C (or close this window) to stop.
# ============================================================

cd "$(dirname "$0")/pi" || { echo "Can't find the pi/ folder next to this file."; sleep 8; exit 1; }
PY=.venv/bin/python

echo "=================================================="
echo "   CADENCE  —  live display"
echo "=================================================="
echo

NAME="capture_$(date +%Y%m%d_%H%M%S)"

echo "[1/2]  Checking for the CPX board over USB..."
echo
if "$PY" cpx_dump.py "$NAME" ; then
    SHOW="$NAME"
    echo
    echo "  Pulled your recording off the board  ->  ${NAME}.csv"
else
    echo
    echo "  No fresh board recording — showing your most recent capture instead."
    LATEST="$(ls -t ../recordings/*.csv 2>/dev/null | head -1)"
    if [ -z "$LATEST" ]; then
        echo "  ...but there are no captures saved yet. Record a run first."
        echo
        echo "  You can close this window."
        sleep 20
        exit 1
    fi
    SHOW="$(basename "$LATEST" .csv)"
    echo "  -> ${SHOW}.csv"
fi

# Whether it came off the board or from disk, only play it if it actually has a
# freeze in it — otherwise fall back to the known-good 'smoketest' so the demo
# lands (a real freeze is fast trembling in place, not standing still).
if ! "$PY" has_freeze.py "$SHOW" && [ -f ../recordings/smoketest.csv ]; then
    echo "  -> no clear freeze in '${SHOW}'; showing 'smoketest' instead."
    SHOW="smoketest"
fi

echo
echo "[2/2]  Starting the live dashboard..."
echo "       A browser will open at  http://127.0.0.1:8000/"
echo "       Leave this window open.  Press Ctrl-C (or close it) to stop."
echo

# open the browser only once the server is actually answering (poll up to ~25s)
( for _ in $(seq 1 50); do
    curl -s -o /dev/null "http://127.0.0.1:8000/" && { open "http://127.0.0.1:8000/"; break; }
    sleep 0.5
  done ) &

exec ./demo_display.sh "$SHOW"
