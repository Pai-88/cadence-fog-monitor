"""
s3_dashboard_bridge.py — live laptop dashboard fed by the ESP32-S3's OWN output.

The S3 runs the freeze-of-gait CNN on-body and streams its telemetry up the USB
cable it's already plugged into:

    per sample     ax,ay,az                              raw int16 milli-g, 64 Hz
    per decision   #D,STATE,prob,fi,tremor,energy,floor,cue        once per window

This bridge does NO inference. It relays exactly what the board decided into the
existing dashboard.html (the same WebSocket JSON dashboard_server.py speaks), so
the screen shows the on-body brain's real verdict — not a laptop re-run of the
model. The ONLY signal processing here is a cosmetic band-pass on the waveform so
the trace matches the dashboard's "band-passed" label; the decision is untouched.

    pip install fastapi 'uvicorn[standard]' pyserial scipy numpy
    python s3_dashboard_bridge.py --port /dev/tty.usbmodemXXXX
    # then open http://localhost:8000/  (find the port with: ls /dev/tty.usb*)

If the serial port can't be opened it keeps serving the page (shown as "no
device") and retries, so the closed loop on the garment is never affected.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
from contextlib import asynccontextmanager, suppress
from pathlib import Path

import numpy as np
import scipy.signal as sig
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from fog.config import SAMPLE_RATE

# ── config ───────────────────────────────────────────────────
HTTP_PORT   = 8000
FS          = SAMPLE_RATE  # sample rate — single source of truth (fog.config)
WAVEFORM_HZ = 20        # waveform broadcasts per second
CONTROL_HZ  = 5         # decision broadcasts per second (S3 updates every ~2 s)
STATS_HZ    = 1
HERE        = Path(__file__).parent
NYQ         = FS / 2.0


class Hub:
    """Shared state between the serial reader thread and the asyncio broadcasters."""

    def __init__(self) -> None:
        self.disp: list[list[int]] = []     # raw [ax,ay,az] awaiting broadcast
        self.lock = threading.Lock()
        self.samples = 0
        self.running = False
        self.connected = False

        # latest decision parsed from the S3's "#D," line
        self.ctrl = {
            "state": "WALKING", "prob_freeze": 0.0, "freeze_index": 0.0,
            "tremor": 0.0, "energy": 0.0, "still_floor": 0.0, "cueing": False,
        }

        # cosmetic display band-pass (0.5–15 Hz), streaming state per axis
        self.sos = sig.butter(4, [0.5 / NYQ, 15.0 / NYQ], btype="band", output="sos")
        self.zi: list[np.ndarray | None] = [None, None, None]

    def filt_chunk(self, chunk: list[list[int]]) -> list[list[float]]:
        """Band-pass a chunk of raw samples for display, carrying filter state."""
        arr = np.asarray(chunk, dtype=float)            # (n, 3)
        out = np.empty_like(arr)
        for c in range(3):
            x = arr[:, c]
            if self.zi[c] is None:                      # seed state from first sample
                self.zi[c] = sig.sosfilt_zi(self.sos) * x[0]
            out[:, c], self.zi[c] = sig.sosfilt(self.sos, x, zi=self.zi[c])
        return out.tolist()


hub = Hub()
clients: set = set()


# ── serial reader thread ─────────────────────────────────────
def reader(port: str, baud: int) -> None:
    """Drain the S3 serial link: parse waveform + decision lines into the hub.

    Lives off the event loop because the serial read blocks. Reconnects forever
    so unplugging/replugging the board just resumes the display.
    """
    try:
        import serial  # pyserial
    except ImportError:
        print("  pyserial not installed (pip install pyserial) — no live data.")
        return

    while hub.running:
        try:
            with serial.Serial(port, baud, timeout=1) as ser:
                hub.connected = True
                print(f"  connected to {port} @ {baud}")
                while hub.running:
                    raw = ser.readline()
                    if not raw:
                        continue
                    line = raw.decode("ascii", "ignore").strip()
                    if not line:
                        continue

                    if line.startswith("#D,"):           # one decision per window
                        p = line.split(",")
                        if len(p) == 8:
                            with suppress(ValueError):       # ignore a malformed line
                                hub.ctrl = {
                                    "state": p[1],
                                    "prob_freeze": float(p[2]),
                                    "freeze_index": float(p[3]),
                                    "tremor": float(p[4]),
                                    "energy": float(p[5]),
                                    "still_floor": float(p[6]),
                                    "cueing": bool(int(p[7])),
                                }
                    else:                                # maybe a waveform sample
                        p = line.split(",")
                        if len(p) == 3:
                            try:
                                s = [int(p[0]), int(p[1]), int(p[2])]
                            except ValueError:
                                continue                 # boot diagnostics etc.
                            with hub.lock:
                                hub.disp.append(s)
                                hub.samples += 1
        except Exception as e:                            # noqa: BLE001 (report + retry)
            hub.connected = False
            print(f"  serial unavailable ({e}) — retrying in 2 s")
            time.sleep(2)


def drain_disp() -> list[list[int]]:
    with hub.lock:
        out, hub.disp = hub.disp, []
    return out


# ── broadcast loops ──────────────────────────────────────────
async def _broadcast(payload: str) -> None:
    dead = []
    for ws in list(clients):
        try:
            await ws.send_text(payload)
        except Exception:                                # noqa: BLE001
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def waveform_loop() -> None:
    period = 1.0 / WAVEFORM_HZ
    while True:
        await asyncio.sleep(period)
        chunk = drain_disp()
        if not chunk:
            continue
        await _broadcast(json.dumps({"type": "waveform", "accel": hub.filt_chunk(chunk)}))


async def control_loop() -> None:
    period = 1.0 / CONTROL_HZ
    while True:
        await asyncio.sleep(period)
        await _broadcast(json.dumps({"type": "control", "has_model": True, **hub.ctrl}))


async def stats_loop() -> None:
    period = 1.0 / STATS_HZ
    while True:
        await asyncio.sleep(period)
        await _broadcast(json.dumps({
            "type": "stats", "samples": hub.samples, "fs": FS,
            "active": hub.connected,
        }))


# ── FastAPI app ──────────────────────────────────────────────
def make_app(port: str, baud: int) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.running = True
        threading.Thread(target=reader, args=(port, baud), daemon=True).start()
        tasks = [
            asyncio.create_task(waveform_loop()),
            asyncio.create_task(control_loop()),
            asyncio.create_task(stats_loop()),
        ]
        print(f"  dashboard at http://localhost:{HTTP_PORT}/  (reading the S3 on {port})")
        yield
        hub.running = False
        for t in tasks:
            t.cancel()

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(HERE / "dashboard.html")

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            clients.discard(websocket)

    return app


if __name__ == "__main__":
    import uvicorn

    ap = argparse.ArgumentParser(description="Relay the ESP32-S3's USB telemetry to dashboard.html")
    ap.add_argument("--port", required=True,
                    help="S3 USB serial device, e.g. /dev/tty.usbmodem1101 or COM5 "
                         "(list with: ls /dev/tty.usb*  on macOS/Linux)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--http-port", type=int, default=HTTP_PORT)
    args = ap.parse_args()
    HTTP_PORT = args.http_port
    uvicorn.run(make_app(args.port, args.baud), host="0.0.0.0", port=HTTP_PORT, log_level="warning")
