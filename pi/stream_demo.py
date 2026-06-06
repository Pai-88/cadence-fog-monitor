"""
Real-time freeze-of-gait detection + closed-loop cueing from the live CPX stream.

  * Reads the 64 Hz accel serial stream, filters online, slides a 4 s window.
  * Runs the trained CNN every WINDOW_HOP samples (~2 s).
  * Debounces: only after N consecutive 'freeze' windows does it fire the cue —
    one twitchy window shouldn't buzz the wearer.
  * Standing-still gate: a freeze is only accepted when there's real movement
    energy (0.5-8 Hz band power) above a calibrated floor — so quietly standing
    or sitting never triggers a cue, even if the CNN/FI twitches. Calibrate it at
    startup by standing still for a few seconds.
  * On a confirmed freeze it sends 'C' to the board (start the rhythmic vibration
    cue) and 'S' when the freeze clears — this is the closed loop.
  * Logs every episode (start, end, duration, peak freeze-index) to a CSV — that
    file IS the "clinician report" deliverable.
  * Also prints the live tremor band-power so you can see the monitoring channel.

    # default: accept the ESP32 Wi-Fi bridge (the untethered garment)
    python stream_demo.py --model fog_model.pth --norm fog_norm.npz
    # legacy: CPX tethered straight over USB
    python stream_demo.py --transport serial --port /dev/ttyACM0

Mirrors emg_hand/predict_realtime.py: if no link comes up (no bridge connects, or
pyserial/board missing) it drops to a print-only dry-run so you can demo the
logic with no hardware attached.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import deque

import numpy as np
import torch

from fog.config import (
    BRIDGE_HOST,
    BRIDGE_PORT,
    CUE_OFF,
    CUE_ON,
    SAMPLE_RATE,
    SERIAL_BAUD,
    SERIAL_PORT,
    WINDOW_HOP,
    WINDOW_SIZE,
)
from fog.dsp import freeze_index, movement_energy, tremor_power
from fog.model import FoGNet
from fog.normalize import Normaliser
from fog.streaming import StreamReceiver, make_receiver


class CueController:
    """Owns the closed-loop cue state and the episode log.

    Two commands go to the board: b'C' (start cueing) and b'S' (stop). We only
    transition on a debounced, stable freeze decision so the wearer isn't buzzed
    by a single noisy window. Every confirmed episode is appended to a CSV that
    doubles as the clinician report.

    rx may be None (dry-run) — then cue commands are just printed.
    """
    # Fire the cue after this many consecutive freeze windows (~N × 2 s).
    ONSET_WINDOWS  = 2
    # Drop the cue after this many consecutive no-freeze windows (hysteresis).
    OFFSET_WINDOWS = 2

    def __init__(self, rx: StreamReceiver | None = None,
                 log_path: str = 'fog_episodes.csv') -> None:
        self.rx = rx
        self.cueing = False
        self._freeze_run = 0
        self._clear_run = 0
        self._episode_start: float | None = None
        self._episode_peak_fi = 0.0
        self.log_path = log_path
        with open(self.log_path, 'w', newline='') as f:
            csv.writer(f).writerow(
                ['start_unix', 'start_iso', 'duration_s', 'peak_freeze_index'])
        print(f"  [CueController] episode log → {self.log_path}")
        if self.rx is None:
            print("  [CueController] no serial — dry-run, cue commands printed only")

    def _send(self, command: bytes) -> None:
        if self.rx is not None:
            self.rx.send(command)

    def update(self, is_freeze: bool, fi: float, now: float) -> None:
        """Feed one window's decision. Manages debounce, cue I/O, episode log."""
        if is_freeze:
            self._freeze_run += 1
            self._clear_run = 0
            self._episode_peak_fi = max(self._episode_peak_fi, fi)
        else:
            self._clear_run += 1
            self._freeze_run = 0

        if not self.cueing and self._freeze_run >= self.ONSET_WINDOWS:
            self.cueing = True
            self._episode_start = now
            self._send(CUE_ON)
            print(f"  ▶ FREEZE detected — cue ON   (FI={fi:.2f})")
        elif self.cueing and self._clear_run >= self.OFFSET_WINDOWS:
            self.cueing = False
            self._send(CUE_OFF)
            dur = now - (self._episode_start or now)
            self._log_episode(dur)
            print(f"  ■ freeze cleared — cue OFF   (lasted {dur:.1f}s, "
                  f"peak FI={self._episode_peak_fi:.2f})")
            self._episode_peak_fi = 0.0

    def _log_episode(self, duration: float) -> None:
        with open(self.log_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                f"{self._episode_start:.0f}",
                time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(self._episode_start)),
                f"{duration:.1f}",
                f"{self._episode_peak_fi:.2f}",
            ])


def calibrate_still_floor(rx: StreamReceiver, seconds: float, margin: float) -> float:
    """Collect a few seconds of standing-still stream and return the movement-
    energy floor below which a freeze decision is refused. Kills the standing-
    still false positive. Needs >= ~4 s (one full window) to produce any reading.
    """
    print(f"  CALIBRATION — stand still for {seconds:.0f}s ... ", end='', flush=True)
    energies, t0 = [], time.monotonic()
    while time.monotonic() - t0 < seconds:
        rx.poll()
        w = rx.get_latest_window(WINDOW_SIZE)
        if w is not None:
            energies.append(movement_energy(w))
        time.sleep(0.05)
    if not energies:
        print("no window captured — gate OFF (floor=0). Try a longer --calibrate.")
        return 0.0
    rest = float(np.median(energies))
    floor = margin * rest
    print(f"done. resting~{rest:.2f} -> floor={floor:.2f}")
    return floor


def main() -> None:
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
    parser.add_argument('--model', default='fog_model.pth')
    parser.add_argument('--norm',  default='fog_norm.npz')
    parser.add_argument('--smooth', type=int, default=3,
                        help='majority-vote length over recent windows')
    parser.add_argument('--still-floor', type=float, default=None,
                        help='movement-energy floor for the standing-still gate; '
                             'set this to skip startup calibration')
    parser.add_argument('--calibrate', type=float, default=6.0,
                        help='seconds of standing-still calibration at startup '
                             '(0 disables; ignored if --still-floor is given)')
    parser.add_argument('--still-margin', type=float, default=4.0,
                        help='floor = margin x resting energy from calibration')
    args = parser.parse_args()

    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    labels = list(ckpt['labels'])
    model = FoGNet(num_classes=len(labels), **ckpt.get('arch', {}))
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    norm = Normaliser.load(args.norm)
    freeze_idx = labels.index('freeze')

    smoothing: deque[int] = deque(maxlen=args.smooth)
    print("=" * 60)
    print("  Freeze-of-Gait live demo")
    print("=" * 60)
    print(f"  Classes: {labels}   window={WINDOW_SIZE} ({WINDOW_SIZE / SAMPLE_RATE:.1f}s)")

    try:
        rx_ctx = make_receiver(args.transport, host=args.host, tcp_port=args.tcp_port,
                               port=args.port, baud=args.baud, filtered=True)
        if args.transport == 'socket':
            print(f"  Waiting for the ESP32 bridge on {args.host}:{args.tcp_port} ...")
        rx = rx_ctx.__enter__()
        if args.transport == 'socket':
            print(f"  Bridge connected ({args.host}:{args.tcp_port})\n")
        else:
            print(f"  Listening on {args.port} @ {args.baud} baud\n")
    except Exception as e:   # no bridge connected in time, or pyserial/board missing
        rx, rx_ctx = None, None
        print(f"  Link unavailable ({e}) — dry-run, no inference loop.\n")

    cue = CueController(rx=rx)

    if rx is None:
        # Nothing to stream from; the controller logic is unit-demonstrable but
        # there is no signal, so just exit cleanly.
        return

    # ── standing-still gate: a freeze is only accepted above this energy floor ──
    if args.still_floor is not None:
        still_floor = args.still_floor
        print(f"  movement-energy gate: floor={still_floor:.2f} (manual)\n")
    elif args.calibrate > 0:
        still_floor = calibrate_still_floor(rx, args.calibrate, args.still_margin)
        print()
    else:
        still_floor = 0.0
        print("  movement-energy gate: OFF (no calibration, no manual floor)\n")

    last_inference_total = 0
    try:
        while True:
            rx.poll()
            if rx.total_samples - last_inference_total < WINDOW_HOP:
                time.sleep(0.005)
                continue
            window = rx.get_latest_window(WINDOW_SIZE)
            if window is None:
                continue
            last_inference_total = rx.total_samples

            X = window.T[None, :, :].astype(np.float32)   # (1, C, T)
            X = norm.transform(X)
            with torch.no_grad():
                pred = int(model(torch.from_numpy(X)).argmax(1).item())
            smoothing.append(pred)
            stable = int(np.bincount(smoothing, minlength=len(labels)).argmax())

            fi = freeze_index(window)
            tp = tremor_power(window)
            energy = movement_energy(window)
            moving = energy > still_floor
            is_freeze = moving and (stable == freeze_idx)     # standing-still gate
            cue.update(is_freeze, fi, time.monotonic())
            state = 'STILL' if not moving else ('FREEZE' if stable == freeze_idx
                                                else 'WALKING')
            print(f"  {state:7s} {labels[stable]:10s} FI={fi:5.2f} "
                  f"E={energy:6.1f}/{still_floor:5.1f} tremor={tp:7.1f}", end='\r')
    except KeyboardInterrupt:
        print("\n  stopped.")
    finally:
        if rx_ctx is not None:
            rx_ctx.__exit__(None, None, None)


if __name__ == '__main__':
    main()
