#!/bin/bash
# ============================================================
#  CADENCE — ONE-CLICK: SHOW LAST CAPTURE
#  Double-click to replay your most recent saved capture live.
#  Skips the board entirely — just plays the newest CSV.
#  A browser opens automatically at http://127.0.0.1:8000/
#  Press Ctrl-C (or close this window) to stop.
# ============================================================

cd "$(dirname "$0")/pi" || { echo "Can't find the pi/ folder next to this file."; sleep 8; exit 1; }

echo "=================================================="
echo "   CADENCE  —  show last capture"
echo "=================================================="
echo

LATEST="$(ls -t ../recordings/*.csv 2>/dev/null | head -1)"
if [ -z "$LATEST" ]; then
    echo "  No captures saved yet. Record a run first."
    echo
    echo "  You can close this window."
    sleep 20
    exit 1
fi

SHOW="$(basename "$LATEST" .csv)"
echo "  Most recent capture:  ${SHOW}.csv"
echo

# Only play it if it actually contains a freeze — otherwise the dashboard just
# shows walking/still and the demo falls flat. If not, fall back to the
# known-good 'smoketest' capture so a freeze always shows.
PY=.venv/bin/python
if ! "$PY" has_freeze.py "$SHOW" && [ -f ../recordings/smoketest.csv ]; then
    echo "  -> no clear freeze in that one; showing 'smoketest' instead."
    SHOW="smoketest"
    echo
fi

echo "  Starting the live dashboard..."
echo "  A browser will open at  http://127.0.0.1:8000/"
echo "  Leave this window open.  Press Ctrl-C (or close it) to stop."
echo

# open the browser only once the server is actually answering (poll up to ~25s)
( for _ in $(seq 1 50); do
    curl -s -o /dev/null "http://127.0.0.1:8000/" && { open "http://127.0.0.1:8000/"; break; }
    sleep 0.5
  done ) &

exec ./demo_display.sh "$SHOW"
