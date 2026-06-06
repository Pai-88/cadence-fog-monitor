"""
Live freeze-of-gait dashboard server.

Reads the 64 Hz accel stream from the wearable — by default over the ESP32
Wi-Fi bridge (TCP), or over a direct USB tether with --transport serial — runs
the trained CNN, drives the closed-loop vibration cue, and serves a real-time
HTML/JS dashboard via FastAPI + WebSocket. One process:

  * a reader thread drains the link into a ring buffer (the socket/serial read
    can block, so it lives off the event loop);
  * asyncio broadcast loops push waveform / freeze-state / stats to the browser;
  * the same CueController as stream_demo.py closes the loop (sends 'C'/'S').

    pip install fastapi 'uvicorn[standard]' scipy numpy torch   # +pyserial for --transport serial
    python dashboard_server.py                       # default: ESP32 Wi-Fi bridge
    python dashboard_server.py --transport serial --port /dev/ttyACM0   # legacy USB
    python dashboard_server.py --replay ~/daphnet/dataset/S01R01.txt    # no hardware
    # open http://<this-host>:8000/ on any browser on the same network

If fog_model.pth + fog_norm.npz aren't here, the freeze panel stays empty —
the raw waveform + freeze-index + tremor still work (they need no model).

Note: this binds the link (the TCP port, or the serial device) — stream_demo.py
wants it too, so run only one at a time per board.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from fog.config import (
    BRIDGE_HOST,
    BRIDGE_PORT,
    FI_THRESHOLD,
    LABELS,
    SAMPLE_RATE,
    SERIAL_BAUD,
    SERIAL_PORT,
    WINDOW_SIZE,
)
from fog.dsp import freeze_index, movement_energy, tremor_power_axes
from fog.model import FoGNet
from fog.normalize import Normaliser
from fog.streaming import make_receiver
from stream_demo import CueController
from tremor_detector import (
    loco_power,
    OFFSET_WINDOWS,
    ONSET_WINDOWS,
    REST_CEILING,
    TREMOR_THRESHOLD,
)

# ── Config ───────────────────────────────────────────────────
HTTP_PORT      = 8000
WAVEFORM_HZ    = 20      # waveform broadcasts per second
CONTROL_HZ     = 2       # inference + freeze/tremor broadcasts per second
STATS_HZ       = 1
STATE_DEBOUNCE = 2       # consecutive control windows a new STILL/WALKING/FREEZE
                         # must hold before the dashboard commits to it (kills
                         # one-off false flips; mirrors the firmware debounce)
HERE           = Path(__file__).parent


# ── Hub state ────────────────────────────────────────────────
class Hub:
    def __init__(self):
        self.rx = None
        self.cue = None
        self.reader_thread = None
        self.running = False

        # Decimated accel awaiting broadcast (filled by the reader thread).
        self.disp = []           # list of [ax, ay, az]
        self._disp_lock = threading.Lock()

        # Model
        self.model = None
        self.norm = None
        self.labels = list(LABELS)

        # Latest derived values (broadcast by control_loop)
        self.prob_freeze = 0.0
        self.freeze_index = 0.0
        self.tremor = 0.0
        self.loco = 0.0             # 0.5-3 Hz locomotor power (wrist rest gate)
        self.tremor_present = False # debounced rest-tremor flag (tremor_detector)
        self._tremor_run = 0        # consecutive raw-tremor windows (onset)
        self._clear_run = 0         # consecutive clear windows (offset/hysteresis)
        self.energy = 0.0           # movement energy (0.5-8 Hz band power)
        self.still_floor = 0.0      # standing-still gate floor (0 = gate off)
        self.fi_threshold = FI_THRESHOLD  # Freeze-Index trip level (glass-box detector)
        self.state = 'WALKING'      # authoritative (debounced) STILL / WALKING / FREEZE
        self._state_pending = 'WALKING'  # latest pre-debounce decision
        self._state_run = 0         # consecutive windows _state_pending has held
        self.cueing = False

    def reader(self):
        """Thread: drain serial, push every sample to the display queue."""
        last_total = 0
        while self.running:
            self.rx.poll()
            new = self.rx.total_samples - last_total
            if new > 0:
                win = self.rx.get_latest_window(min(new, len(self.rx.buf)))
                if win is not None:
                    with self._disp_lock:
                        self.disp.extend(win.tolist())
                last_total = self.rx.total_samples
            time.sleep(0.003)

    def drain_disp(self):
        with self._disp_lock:
            out, self.disp = self.disp, []
        return out


hub = Hub()


# ── Phase-labelled CSV recorder ──────────────────────────────
class Recorder:
    """Live, phase-labelled CSV recorder for the streaming path.

    Taps the receiver's RAW milli-g samples (via StreamReceiver.on_raw_sample,
    *before* the online band-pass) and writes a capture-style CSV that is a
    drop-in for analyze_worksheet.py: the identical
    ``idx,t_s,ax_mg,ay_mg,az_mg,phase`` columns and the same
    ``# LOGGING FINISHED`` / ``# samples=`` / ``# phase boundaries`` provenance
    header the worksheet capture writes. ``phase`` is an integer segment marker
    (0, 1, 2, …) you bump live from the dashboard — mirroring the CPX RIGHT-
    button protocol; map the numbers to activities later with
    ``analyze_worksheet.py --labels still,walk,freeze,…``.

    Thread-safe: :meth:`add` runs on the reader thread (once per sample) while
    start/stop/next/set_phase run on the asyncio event loop. One lock guards the
    open handle and the counters; rows are line-buffered to disk so a crash mid-
    capture still leaves a complete, readable file (only the cosmetic ``#``
    header — written on stop — would be missing).
    """

    def __init__(self, outdir: Path) -> None:
        self.outdir = Path(outdir)
        self._lock = threading.Lock()
        self.recording = False
        self.path: Path | None = None
        self._f = None
        self.n = 0                    # samples written so far
        self.phase = 0                # current segment marker
        self.bounds: list[int] = []   # sample idx at each phase change (0 first)
        self.started_wall = 0.0

    def _next_name(self) -> str:
        """Next free streamN stem in outdir (so files never clobber captures)."""
        nums = [int(m.group(1)) for p in self.outdir.glob("stream*.csv")
                if (m := re.fullmatch(r"stream(\d+)", p.stem))]
        return f"stream{max(nums) + 1 if nums else 1}"

    def _status(self) -> dict:        # call holding the lock
        return {
            "recording": self.recording,
            "samples": self.n,
            "seconds": round(self.n / SAMPLE_RATE, 2),
            "phase": self.phase,
            "segments": len(self.bounds),
            "file": self.path.name if self.path else None,
        }

    def status(self) -> dict:
        with self._lock:
            return self._status()

    # — reader thread: one raw (pre-filter) mg sample —
    def add(self, sample) -> None:
        with self._lock:
            if not self.recording or self._f is None:
                return
            ax, ay, az = (int(round(float(v))) for v in sample[:3])
            # Match capture1.csv byte-for-byte: t_s 4 dp, mg as integers.
            self._f.write(f"{self.n},{self.n / SAMPLE_RATE:.4f},"
                          f"{ax},{ay},{az},{self.phase}\n")
            self.n += 1

    # — control (event loop) —
    def start(self, name: str | None = None) -> dict:
        with self._lock:
            if self.recording:
                return {"ok": False, "error": "already recording", **self._status()}
            self.outdir.mkdir(parents=True, exist_ok=True)
            stem = name.strip() if name and name.strip() else self._next_name()
            stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem)
            self.path = self.outdir / f"{stem}.csv"
            self._f = open(self.path, "w", buffering=1)   # line-buffered
            self._f.write("idx,t_s,ax_mg,ay_mg,az_mg,phase\n")
            self.n = 0
            self.phase = 0
            self.bounds = [0]
            self.started_wall = time.time()
            self.recording = True
            print(f"  ● recording → {self.path}")
            return {"ok": True, **self._status()}

    def set_phase(self, phase: int) -> dict:
        with self._lock:
            if not self.recording:
                return {"ok": False, "error": "not recording", **self._status()}
            phase = int(phase)
            if phase != self.phase:
                self.phase = phase
                self.bounds.append(self.n)
            return {"ok": True, **self._status()}

    def next_phase(self) -> dict:
        with self._lock:
            if not self.recording:
                return {"ok": False, "error": "not recording", **self._status()}
            self.phase += 1
            self.bounds.append(self.n)
            return {"ok": True, **self._status()}

    def stop(self) -> dict:
        with self._lock:
            if not self.recording:
                return {"ok": False, "error": "not recording", **self._status()}
            self._f.close()
            self._f = None
            self.recording = False
            path, n, bounds = self.path, self.n, list(self.bounds)
        # Finalize outside the lock: add() now no-ops and the handle is closed.
        self._finalize(path, n, bounds)
        print(f"  ■ saved {n} samples ({n / SAMPLE_RATE:.2f}s, "
              f"{len(bounds)} segment(s)) → {path}")
        return {"ok": True, "samples": n, "seconds": round(n / SAMPLE_RATE, 2),
                "segments": len(bounds), "file": path.name, "path": str(path)}

    @staticmethod
    def _finalize(path: Path, n: int, bounds: list[int]) -> None:
        """Prepend the worksheet-style provenance header (mirrors the logger)."""
        dur = n / SAMPLE_RATE
        with open(path) as fh:
            body = fh.read()
        hdr = ("# LOGGING FINISHED\n"
               f"# samples={n}  rate_hz={SAMPLE_RATE}  duration_s={dur:.2f}\n"
               "# phase boundaries (sample idx): "
               f"{' '.join(str(b) for b in bounds)}\n")
        with open(path, "w") as fh:
            fh.write(hdr + body)


# All board dumps and live stream recordings live in the project-root recordings/ folder.
recorder = Recorder(HERE.parent / "recordings")


def _load_model() -> None:
    mp, npth = HERE / "fog_model.pth", HERE / "fog_norm.npz"
    if not (mp.exists() and npth.exists()):
        print(f"  Model not found in {HERE} — freeze panel disabled "
              f"(run train_fog.py). Waveform/FI/tremor still work.")
        return
    try:
        ckpt = torch.load(str(mp), map_location='cpu', weights_only=False)
        hub.labels = list(ckpt['labels'])
        # 'arch' (conv/fc widths) lets a tuned, non-default model reload at the
        # right shape; .get keeps older checkpoints (no 'arch' key) working.
        hub.model = FoGNet(num_classes=len(hub.labels), **ckpt.get('arch', {}))
        hub.model.load_state_dict(ckpt['model_state'])
        hub.model.eval()
        hub.norm = Normaliser.load(str(npth))
        print(f"  Model loaded — classes {hub.labels}")
    except Exception as e:
        print(f"  Failed to load model: {e}")


# ── Broadcast helpers / loops ────────────────────────────────
clients: set = set()


async def _broadcast(payload: str) -> None:
    dead = []
    for ws in list(clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def waveform_loop() -> None:
    period = 1.0 / WAVEFORM_HZ
    while True:
        await asyncio.sleep(period)
        chunk = hub.drain_disp()
        if not chunk:
            continue
        await _broadcast(json.dumps({"type": "waveform", "accel": chunk}))


async def control_loop() -> None:
    period = 1.0 / CONTROL_HZ
    freeze_idx = hub.labels.index('freeze') if 'freeze' in hub.labels else 1
    while True:
        await asyncio.sleep(period)
        if hub.rx is None or hub.rx.total_samples < WINDOW_SIZE:
            continue
        window = hub.rx.get_latest_window(WINDOW_SIZE)
        if window is None:
            continue

        hub.freeze_index = freeze_index(window)
        hub.tremor = tremor_power_axes(window)   # orientation-robust 4-6 Hz — matches analyze_tremor.py / tremor_detector.py
        hub.loco = loco_power(window)
        hub.energy = movement_energy(window)

        # Debounced rest-tremor flag — identical logic + thresholds to the deployed
        # wrist baseline (tremor_detector.py), so the live monitor and the worksheet
        # agree. Calibration-gated: TREMOR_THRESHOLD <= 0 means "uncalibrated
        # placeholder", so we assert nothing — mirroring how still_floor == 0 disables
        # the standing-still gate below. Set TREMOR_THRESHOLD in tremor_detector.py
        # from analyze_tremor.py's suggestion and the flag goes live. (The live loop
        # runs every 1/CONTROL_HZ s, faster than the 2 s offline hop, so onset latency
        # is ONSET_WINDOWS/CONTROL_HZ s rather than the worksheet's 2 s-per-window.)
        if TREMOR_THRESHOLD > 0:
            raw_tremor = (hub.loco < REST_CEILING) and (hub.tremor > TREMOR_THRESHOLD)
            if raw_tremor:
                hub._tremor_run += 1; hub._clear_run = 0
            else:
                hub._clear_run += 1; hub._tremor_run = 0
            if not hub.tremor_present and hub._tremor_run >= ONSET_WINDOWS:
                hub.tremor_present = True
            elif hub.tremor_present and hub._clear_run >= OFFSET_WINDOWS:
                hub.tremor_present = False
        else:
            hub.tremor_present = False

        # Does the freeze detector fire? CNN argmax when a model is loaded,
        # else fall back to the Freeze-Index threshold so the dashboard is useful
        # even before any model has been trained.
        if hub.model is not None:
            X = hub.norm.transform(window.T[None, :, :].astype(np.float32))
            with torch.no_grad():
                p = torch.softmax(hub.model(torch.from_numpy(X)), dim=1).numpy()[0]
            hub.prob_freeze = float(p[freeze_idx])
            detector_freeze = int(p.argmax()) == freeze_idx
        else:
            detector_freeze = hub.freeze_index > hub.fi_threshold

        # Standing-still gate (matches stream_demo.py + the firmware): a freeze
        # is only real when movement energy clears the calibrated floor, so quiet
        # standing/sitting cannot trigger a cue. still_floor 0 ⇒ gate disabled.
        moving = hub.energy > hub.still_floor
        is_freeze = moving and detector_freeze
        raw_state = ('STILL' if not moving
                     else 'FREEZE' if detector_freeze else 'WALKING')
        # Two-window debounce on the displayed state (matches the firmware state
        # machine and the presentation script): a new state must persist for
        # STATE_DEBOUNCE readings in a row before the dashboard commits to it,
        # which kills the one-off false flips on walking transients. (The cue
        # path keeps its own debounce inside CueController, below.)
        if raw_state == hub._state_pending:
            hub._state_run += 1
        else:
            hub._state_pending = raw_state
            hub._state_run = 1
        if hub._state_run >= STATE_DEBOUNCE:
            hub.state = raw_state

        if hub.cue is not None:
            hub.cue.update(is_freeze, hub.freeze_index, time.monotonic())
            hub.cueing = hub.cue.cueing

        await _broadcast(json.dumps({
            "type": "control",
            "state":        hub.state,
            "prob_freeze":  hub.prob_freeze,
            "freeze_index": hub.freeze_index,
            "tremor":         hub.tremor,
            "tremor_present": hub.tremor_present,
            "tremor_threshold": TREMOR_THRESHOLD,
            "loco":           hub.loco,
            "energy":       hub.energy,
            "still_floor":  hub.still_floor,
            "fi_threshold": hub.fi_threshold,
            "cueing":       hub.cueing,
            "has_model":    hub.model is not None,
        }))


async def stats_loop() -> None:
    period = 1.0 / STATS_HZ
    while True:
        await asyncio.sleep(period)
        await _broadcast(json.dumps({
            "type": "stats",
            "samples": hub.rx.total_samples if hub.rx else 0,
            "fs": SAMPLE_RATE,
            "active": hub.running,
            "rec": recorder.status(),
        }))


# ── FastAPI app ──────────────────────────────────────────────
def make_app(transport: str = "socket", host: str = BRIDGE_HOST,
             tcp_port: int = BRIDGE_PORT, port: str = SERIAL_PORT,
             baud: int = SERIAL_BAUD, still_floor: float = 0.0,
             detector: str = "cnn", fi_threshold: float = FI_THRESHOLD,
             replay_path: str | None = None, replay_sensor: str = "ankle",
             replay_loop: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        hub.fi_threshold = fi_threshold
        if detector == "fi":
            # Glass-box demo: drive the live state straight off the Freeze Index
            # (matches the presentation script + worksheet) instead of the CNN.
            print(f"  detector: Freeze-Index threshold={fi_threshold:.3f} "
                  f"(CNN bypassed)")
        else:
            _load_model()
        hub.still_floor = still_floor
        if still_floor > 0:
            print(f"  standing-still gate: floor={still_floor:.3f}")
        try:
            if transport == "socket":
                print(f"  Waiting for the ESP32 bridge on {host}:{tcp_port} ...")
            elif transport == "replay":
                print(f"  Replaying {replay_path} ({replay_sensor}) at "
                      f"{SAMPLE_RATE} Hz" + (" (looping)" if replay_loop else ""))
            # The CNN was trained on the 0.5-15 Hz band-passed signal, so it needs
            # filtered=True. The Freeze Index is the OPPOSITE: band-passing per axis
            # and then taking the magnitude folds the 3-8 Hz freeze power out of the
            # ratio, flattening FI so walking and freezing look alike. The glass-box
            # FI detector therefore reads the RAW stream, where a genuine freeze
            # stands out (FI in the hundreds vs ~1 for walking).
            use_filter = (detector != "fi")
            hub.rx = make_receiver(transport, host=host, tcp_port=tcp_port,
                                   port=port, baud=baud, filtered=use_filter,
                                   replay_path=replay_path,
                                   replay_sensor=replay_sensor,
                                   replay_loop=replay_loop).__enter__()
            hub.rx.on_raw_sample = recorder.add   # tap raw mg for the CSV recorder
            hub.cue = CueController(rx=hub.rx)
            hub.running = True
            hub.reader_thread = threading.Thread(target=hub.reader, daemon=True)
            hub.reader_thread.start()
            if transport == "socket":
                print(f"  Bridge reader on {host}:{tcp_port}")
            elif transport == "replay":
                print(f"  Replay reader: {replay_path}")
            else:
                print(f"  Serial reader on {port} @ {baud}")
        except Exception as e:
            print(f"  Link unavailable ({e}) — dashboard serves, but no data.")
        tasks = [
            asyncio.create_task(waveform_loop()),
            asyncio.create_task(control_loop()),
            asyncio.create_task(stats_loop()),
        ]
        print(f"  Dashboard at http://<host>:{HTTP_PORT}/")
        yield
        hub.running = False
        for t in tasks:
            t.cancel()
        if hub.rx is not None:
            hub.rx.__exit__(None, None, None)

    app = FastAPI(lifespan=lifespan)

    @app.get("/")
    async def index():
        return FileResponse(HERE / "dashboard.html")

    # ── Field Recorder: save a phase-labelled CSV of the live stream ─────────
    # A drop-in for analyze_worksheet.py (same columns as capture1.csv). The
    # phase marker is bumped live from the dashboard, mirroring the CPX RIGHT
    # button used by the offline worksheet capture.
    @app.get("/rec/status")
    async def rec_status():
        return recorder.status()

    @app.post("/rec/start")
    async def rec_start(name: str | None = None):
        return recorder.start(name)

    @app.post("/rec/next")
    async def rec_next():
        return recorder.next_phase()

    @app.post("/rec/phase")
    async def rec_phase(value: int):
        return recorder.set_phase(value)

    @app.post("/rec/stop")
    async def rec_stop():
        return recorder.stop()

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--transport', choices=['socket', 'serial'], default='socket',
                        help="'socket' = ESP32 Wi-Fi bridge (default); "
                             "'serial' = legacy direct-USB tether")
    parser.add_argument('--host', default=BRIDGE_HOST,
                        help='[socket] interface to bind for the ESP32 to dial into')
    parser.add_argument('--tcp-port', type=int, default=BRIDGE_PORT,
                        help='[socket] TCP port the ESP32 connects to')
    parser.add_argument('--port', default=SERIAL_PORT,
                        help='[serial] USB serial device of the CPX')
    parser.add_argument('--baud', type=int, default=SERIAL_BAUD,
                        help='[serial] USB serial baud')
    parser.add_argument('--still-floor', type=float, default=0.0,
                        help='movement-energy floor for the standing-still gate '
                             '(0 = gate off; see stream_demo.py --calibrate to '
                             'find a value for this board/wearer)')
    parser.add_argument('--detector', choices=['cnn', 'fi'], default='cnn',
                        help="freeze detector: 'cnn' = trained model if present "
                             "(default); 'fi' = glass-box Freeze-Index threshold "
                             '(matches the presentation script & worksheet)')
    parser.add_argument('--fi-threshold', type=float, default=FI_THRESHOLD,
                        help='Freeze-Index trip level for --detector fi '
                             f'(default {FI_THRESHOLD})')
    parser.add_argument('--replay', metavar='DAPHNET_FILE', default=None,
                        help='replay a recorded Daphnet S*R*.txt as a live 64 Hz '
                             'stream (no hardware needed) — forces transport '
                             "'replay'; great for a dry-run demo")
    parser.add_argument('--replay-sensor', choices=['ankle', 'thigh', 'trunk'],
                        default='ankle',
                        help='[replay] which Daphnet sensor columns to stream')
    parser.add_argument('--replay-once', action='store_true',
                        help='[replay] play the file once and stop (default: loop)')
    args = parser.parse_args()
    transport = 'replay' if args.replay else args.transport
    uvicorn.run(make_app(transport, args.host, args.tcp_port,
                         args.port, args.baud, args.still_floor,
                         detector=args.detector, fi_threshold=args.fi_threshold,
                         replay_path=args.replay,
                         replay_sensor=args.replay_sensor,
                         replay_loop=not args.replay_once),
                host="0.0.0.0", port=HTTP_PORT, log_level="info")
