"""Receivers that turn the board's byte stream into windows of accel samples.

Two transports share one ring buffer:

* :class:`SocketReceiver` — the live, untethered path. The CPX streams over UART
  to an ESP32, which relays it to the laptop over Wi-Fi/TCP. The laptop is the TCP
  *server* and the ESP32 dials in; this receiver accepts that connection. This
  is the **default** live transport.
* :class:`SerialReceiver` — the legacy direct-USB path (CPX tethered to the
  host). Kept for standalone bring-up / a no-ESP32 fallback.

Both speak the same protocol — newline-delimited ``ax,ay,az`` lines going up,
single ``b'C'`` / ``b'S'`` cue bytes coming down — and expose the same API
(:meth:`poll`, :meth:`get_latest_window`, :meth:`send`), so the demos are
transport-agnostic. ``pyserial`` is imported lazily inside the serial context
manager, so the offline trainer (which opens no port) needs no hardware library.
"""
from __future__ import annotations

import socket
import time
from typing import Any

import numpy as np

from .config import (
    BRIDGE_HOST,
    BRIDGE_PORT,
    NUM_AXES,
    SAMPLE_RATE,
    SERIAL_BAUD,
    SERIAL_PORT,
)
from .dsp import AccelFilter, parse_line

__all__ = [
    "StreamReceiver",
    "SerialReceiver",
    "SocketReceiver",
    "ReplayReceiver",
    "make_receiver",
]


class StreamReceiver:
    """Transport-agnostic ring buffer of the most recent accel samples.

    Subclasses open a byte transport in :meth:`__enter__` and implement the three
    hooks :meth:`_read` / :meth:`_write` / :meth:`_close`. Everything else — line
    parsing, the optional online band-pass, and the wrap-around ring buffer —
    lives here and is shared by every transport.

    Usage (via a concrete subclass)::

        with SocketReceiver(filtered=True) as rx:   # or SerialReceiver(...)
            while running:
                rx.poll()
                window = rx.get_latest_window(WINDOW_SIZE)
                if window is not None:
                    ...
    """

    def __init__(self, buffer_seconds: int = 12, filtered: bool = True) -> None:
        self._rxbuf = b""
        self.filt = AccelFilter() if filtered else None
        self.buf = np.zeros((buffer_seconds * SAMPLE_RATE, NUM_AXES), dtype=np.float32)
        self.write_idx = 0
        self.total_samples = 0
        # Optional tap on the RAW (pre-band-pass) milli-g sample. Set by a
        # consumer that wants every sample exactly as the board sent it — e.g.
        # the dashboard's CSV recorder, which must log the same unfiltered mg the
        # worksheet capture did. Called once per accepted sample inside poll(),
        # before self.filt is applied. Kept transport-agnostic (socket/serial/
        # replay all route through poll), and a no-op when left as None.
        self.on_raw_sample = None

    # ── transport hooks — subclasses implement over their handle ─────────────
    def _read(self, n: int) -> bytes:
        """Return up to ``n`` bytes that have arrived; ``b''`` if none are ready."""
        raise NotImplementedError

    def _write(self, data: bytes) -> None:
        """Send raw bytes back to the board (best-effort; never raises)."""
        raise NotImplementedError

    def _close(self) -> None:
        """Release the transport. Default: nothing to do."""

    # ── context manager ──────────────────────────────────────────────────────
    def __enter__(self) -> StreamReceiver:
        return self

    def __exit__(self, *exc: object) -> None:
        self._close()

    # ── shared receive / send / windowing ────────────────────────────────────
    def send(self, command: bytes) -> None:
        """Write a one-byte command back to the board (``b'C'`` on, ``b'S'`` off)."""
        self._write(command)

    def poll(self) -> int:
        """Drain pending bytes into the ring buffer; return # new samples."""
        self._rxbuf += self._read(4096)
        n_new = 0
        while b"\n" in self._rxbuf:
            line, self._rxbuf = self._rxbuf.split(b"\n", 1)
            sample = parse_line(line)
            if sample is None:
                continue
            if self.on_raw_sample is not None:
                self.on_raw_sample(sample)   # raw mg, before the band-pass
            if self.filt is not None:
                sample = self.filt.apply(sample)[0]
            self.buf[self.write_idx] = sample
            self.write_idx = (self.write_idx + 1) % len(self.buf)
            self.total_samples += 1
            n_new += 1
        return n_new

    def get_latest_window(self, window_size: int) -> np.ndarray | None:
        """Return the most recent ``window_size`` samples as ``(window_size, C)``.

        ``None`` until at least ``window_size`` samples have arrived. Handles the
        ring-buffer wrap-around so the returned window is always contiguous in
        time, oldest sample first.
        """
        if self.total_samples < window_size:
            return None
        start = (self.write_idx - window_size) % len(self.buf)
        if start + window_size <= len(self.buf):
            return self.buf[start : start + window_size].copy()
        first = self.buf[start:].copy()
        return np.vstack([first, self.buf[: window_size - len(first)].copy()])


class SerialReceiver(StreamReceiver):
    """Legacy direct-USB path: read the CPX over a USB serial port (``pyserial``).

    Kept for standalone bring-up and as a no-ESP32 fallback; the live garment
    uses :class:`SocketReceiver` instead.
    """

    def __init__(
        self,
        port: str = SERIAL_PORT,
        baud: int = SERIAL_BAUD,
        buffer_seconds: int = 12,
        filtered: bool = True,
    ) -> None:
        super().__init__(buffer_seconds=buffer_seconds, filtered=filtered)
        self.port = port
        self.baud = baud
        # pyserial handle, opened lazily in __enter__. Typed Any because pyserial
        # ships no reliable stubs and the handle is None until the context opens.
        self.ser: Any = None

    def __enter__(self) -> SerialReceiver:
        import serial  # pyserial — imported lazily so the trainer needs no hardware

        self.ser = serial.Serial(self.port, self.baud, timeout=0)
        return self

    def _read(self, n: int) -> bytes:
        return self.ser.read(n) if self.ser is not None else b""

    def _write(self, data: bytes) -> None:
        if self.ser is not None:
            self.ser.write(data)

    def _close(self) -> None:
        if self.ser is not None:
            self.ser.close()
            self.ser = None


class SocketReceiver(StreamReceiver):
    """Live untethered path: accept the ESP32 Wi-Fi bridge's TCP connection.

    The laptop is the *server*: it binds ``(host, port)`` and, in :meth:`__enter__`,
    waits for the ESP32 to dial in (one client). After that the byte stream is
    identical to the serial path, so :meth:`poll` / :meth:`get_latest_window` /
    :meth:`send` behave exactly the same.

    The accepted socket is non-blocking so :meth:`poll` never stalls the
    inference loop. If the bridge drops, the listening socket stays open and the
    next :meth:`poll` transparently re-accepts a reconnecting ESP32.
    """

    def __init__(
        self,
        host: str = BRIDGE_HOST,
        port: int = BRIDGE_PORT,
        buffer_seconds: int = 12,
        filtered: bool = True,
        accept_timeout: float = 30.0,
    ) -> None:
        super().__init__(buffer_seconds=buffer_seconds, filtered=filtered)
        self.host = host
        self.port = port
        self.accept_timeout = accept_timeout
        self._srv: Any = None    # listening socket (server)
        self.sock: Any = None    # the accepted ESP32 connection

    def __enter__(self) -> SocketReceiver:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        srv.settimeout(self.accept_timeout)   # block up to N s for the first client
        conn, _addr = srv.accept()            # waits for the ESP32 to connect
        self._configure(conn)
        srv.setblocking(False)                # later re-accepts are non-blocking
        self._srv = srv
        self.sock = conn
        return self

    @staticmethod
    def _configure(conn: Any) -> None:
        conn.setblocking(False)               # poll() must never stall the loop
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # low latency

    def _accept_if_idle(self) -> None:
        """If the bridge dropped, pick up a reconnecting ESP32 (non-blocking)."""
        if self.sock is not None or self._srv is None:
            return
        try:
            conn, _addr = self._srv.accept()
        except (BlockingIOError, OSError):
            return                            # nobody waiting yet — try again later
        self._configure(conn)
        self.sock = conn

    def _drop(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def _read(self, n: int) -> bytes:
        self._accept_if_idle()
        if self.sock is None:
            return b""
        try:
            data = self.sock.recv(n)
        except BlockingIOError:
            return b""                        # no bytes ready (normal, non-blocking)
        except OSError:
            self._drop()                      # connection errored — re-accept later
            return b""
        if data == b"":                       # peer closed cleanly
            self._drop()
        return data

    def _write(self, data: bytes) -> None:
        if self.sock is None:
            return
        try:
            self.sock.sendall(data)
        except OSError:
            self._drop()

    def _close(self) -> None:
        self._drop()
        if self._srv is not None:
            self._srv.close()
            self._srv = None


class ReplayReceiver(StreamReceiver):
    """Offline demo path: replay a recorded Daphnet file as a live 64 Hz stream.

    No hardware and no network — it reads a Daphnet ``S*R*.txt`` file, drops the
    'not in experiment' rows (annotation 0), and paces the chosen sensor's accel
    out as newline-delimited ``ax,ay,az`` lines in **real time**, so the whole
    pipeline downstream (band-pass → CNN → freeze state → cue → dashboard) runs
    exactly as it does off the live bridge. It loops by default, so a demo can
    run indefinitely while you talk over it.

    The cue bytes the controller sends back (``b'C'`` / ``b'S'``) have no board
    to reach, so :meth:`_write` just tallies them; the dashboard's cue indicator
    still lights from the control loop's own state. Columns and the
    drop-annotation-0 rule mirror ``train_fog.load_daphnet`` so what you see
    replayed is what the model was trained and evaluated on.
    """

    # Daphnet column layout (0-indexed), mirroring train_fog.SENSOR_COLS.
    _SENSOR_COLS = {"ankle": (1, 2, 3), "thigh": (4, 5, 6), "trunk": (7, 8, 9)}
    _ANNOT_COL = 10

    def __init__(
        self,
        path: str,
        sensor: str = "ankle",
        loop: bool = True,
        buffer_seconds: int = 12,
        filtered: bool = True,
    ) -> None:
        super().__init__(buffer_seconds=buffer_seconds, filtered=filtered)
        self.path = path
        self.sensor = sensor
        self.loop = loop
        self._lines: list[bytes] = []   # one pre-formatted ax,ay,az\n per sample
        self._cursor = 0                # samples streamed so far (grows past len)
        self._pending = b""             # formatted-but-not-yet-handed-out bytes
        self._t0 = 0.0                  # wall clock at which replay started
        self.cue_bytes = 0              # cue bytes the controller tried to send

    def __enter__(self) -> ReplayReceiver:
        import os

        if self.sensor not in self._SENSOR_COLS:
            raise ValueError(
                f"unknown replay sensor {self.sensor!r} "
                f"(use {', '.join(self._SENSOR_COLS)})")
        if not os.path.exists(self.path):
            raise FileNotFoundError(
                f"replay file not found: {self.path} "
                f"(point --replay at a Daphnet S*R*.txt)")
        arr = np.loadtxt(self.path, ndmin=2)   # ndmin=2 so a 1-row file stays 2-D
        if arr.shape[1] <= self._ANNOT_COL:
            raise ValueError(
                f"{self.path} is not a Daphnet file "
                f"(need >{self._ANNOT_COL} space-delimited columns)")
        cols = self._SENSOR_COLS[self.sensor]
        sig = arr[arr[:, self._ANNOT_COL] != 0][:, list(cols)]  # drop annot-0 rows
        if len(sig) == 0:
            raise ValueError(f"no in-experiment samples in {self.path}")
        # Pre-format every sample as the firmware's int16 milli-g line, once.
        self._lines = [b"%d,%d,%d\n" % (int(a), int(b), int(c)) for a, b, c in sig]
        self._t0 = time.monotonic()
        return self

    def _read(self, n: int) -> bytes:
        if not self._lines:
            return b""
        # Queue every sample whose 64 Hz slot has elapsed since the last read.
        due = int((time.monotonic() - self._t0) * SAMPLE_RATE)
        while self._cursor < due:
            if self._cursor >= len(self._lines) and not self.loop:
                break
            self._pending += self._lines[self._cursor % len(self._lines)]
            self._cursor += 1
        out, self._pending = self._pending[:n], self._pending[n:]
        return out

    def _write(self, data: bytes) -> None:
        self.cue_bytes += len(data)     # no board downstream — just account for it

    def _close(self) -> None:
        self._lines = []
        self._pending = b""


def make_receiver(
    transport: str = "socket",
    *,
    host: str = BRIDGE_HOST,
    tcp_port: int = BRIDGE_PORT,
    port: str = SERIAL_PORT,
    baud: int = SERIAL_BAUD,
    buffer_seconds: int = 12,
    filtered: bool = True,
    replay_path: str | None = None,
    replay_sensor: str = "ankle",
    replay_loop: bool = True,
) -> StreamReceiver:
    """Build the receiver for the chosen transport.

    ``"socket"`` (default) — the ESP32 Wi-Fi bridge (live, untethered garment).
    ``"serial"``           — the legacy direct-USB tether (standalone bring-up).
    ``"replay"``           — a recorded Daphnet file paced as a live 64 Hz stream
                             (no hardware); needs ``replay_path``.
    """
    if transport == "socket":
        return SocketReceiver(
            host=host, port=tcp_port, buffer_seconds=buffer_seconds, filtered=filtered
        )
    if transport == "serial":
        return SerialReceiver(
            port=port, baud=baud, buffer_seconds=buffer_seconds, filtered=filtered
        )
    if transport == "replay":
        if replay_path is None:
            raise ValueError("transport 'replay' needs replay_path (a Daphnet file)")
        return ReplayReceiver(
            path=replay_path, sensor=replay_sensor, loop=replay_loop,
            buffer_seconds=buffer_seconds, filtered=filtered,
        )
    raise ValueError(
        f"unknown transport {transport!r} (use 'socket', 'serial' or 'replay')")
