"""Tests for fog.streaming — parsing + ring buffer over both transports.

No real hardware: a tiny in-memory ``FakeSerial`` / ``FakeSock`` is injected in
place of the pyserial handle / TCP socket, so parsing, partial-line buffering,
ring-buffer wrap-around and the socket-specific drop/reconnect behaviour are all
exercised with zero hardware.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from fog.config import NUM_AXES, SAMPLE_RATE
from fog.streaming import (
    ReplayReceiver,
    SerialReceiver,
    SocketReceiver,
    make_receiver,
)


class FakeSerial:
    """Minimal stand-in for ``serial.Serial``: a byte queue + a write log."""

    def __init__(self, data: bytes = b"") -> None:
        self._data = bytearray(data)
        self.written = bytearray()

    def read(self, n: int) -> bytes:
        chunk = bytes(self._data[:n])
        del self._data[:n]
        return chunk

    def feed(self, data: bytes) -> None:
        self._data += data

    def write(self, data: bytes) -> None:
        self.written += data

    def close(self) -> None:
        pass


def make_rx(filtered: bool = False, buffer_seconds: int = 12) -> SerialReceiver:
    """A receiver with the fake serial injected (bypassing ``__enter__``)."""
    rx = SerialReceiver(filtered=filtered, buffer_seconds=buffer_seconds)
    rx.ser = FakeSerial()
    return rx


def test_poll_parses_samples() -> None:
    rx = make_rx()
    rx.ser.feed(b"1,2,3\n4,5,6\n")
    assert rx.poll() == 2
    assert rx.total_samples == 2


def test_poll_skips_malformed_lines() -> None:
    rx = make_rx()
    rx.ser.feed(b"1,2,3\nboot banner\n4,5,6\n")
    assert rx.poll() == 2


def test_partial_line_buffered_across_polls() -> None:
    rx = make_rx()
    rx.ser.feed(b"7,8")  # no newline yet
    assert rx.poll() == 0
    rx.ser.feed(b",9\n")  # completes the line
    assert rx.poll() == 1
    win = rx.get_latest_window(1)
    assert win is not None
    np.testing.assert_array_equal(win[0], [7, 8, 9])


def test_get_latest_window_none_until_full() -> None:
    rx = make_rx()
    rx.ser.feed(b"1,1,1\n2,2,2\n3,3,3\n")
    rx.poll()
    assert rx.get_latest_window(5) is None      # only 3 samples so far
    rx.ser.feed(b"4,4,4\n5,5,5\n")
    rx.poll()
    assert rx.get_latest_window(5) is not None


def test_ring_buffer_wraps_in_time_order() -> None:
    # buffer holds 64 samples; push 70 so the buffer wraps, then check the latest
    # 64 come back oldest-first and contiguous (samples 6..69).
    rx = make_rx(buffer_seconds=1)              # 1 s * 64 Hz = 64-sample ring
    assert len(rx.buf) == 64
    rx.ser.feed(b"".join(f"{i},{i},{i}\n".encode() for i in range(70)))
    rx.poll()
    win = rx.get_latest_window(64)
    assert win is not None
    assert win.shape == (64, NUM_AXES)
    np.testing.assert_allclose(win[:, 0], np.arange(6, 70))


def test_send_writes_command() -> None:
    rx = make_rx()
    rx.send(b"C")
    rx.send(b"S")
    assert bytes(rx.ser.written) == b"CS"


# ── SocketReceiver: the ESP32 Wi-Fi-bridge transport ─────────────────────────
class FakeSock:
    """Minimal stand-in for a *non-blocking* TCP socket.

    A non-blocking ``recv`` has three cases that the receiver must handle:
      * bytes ready          → return the next chunk;
      * open but no data yet → raise ``BlockingIOError`` (EAGAIN);
      * peer closed cleanly  → return ``b''``.
    ``feed`` queues bytes; ``close_peer`` flips it into the EOF state.
    """

    def __init__(self, data: bytes = b"") -> None:
        self._data = bytearray(data)
        self.sent = bytearray()
        self._peer_closed = False
        self.closed = False

    def feed(self, data: bytes) -> None:
        self._data += data

    def close_peer(self) -> None:
        self._peer_closed = True

    def recv(self, n: int) -> bytes:
        if self._data:
            chunk = bytes(self._data[:n])
            del self._data[:n]
            return chunk
        if self._peer_closed:
            return b""              # clean EOF
        raise BlockingIOError       # open, nothing ready

    def sendall(self, data: bytes) -> None:
        self.sent += data

    def close(self) -> None:
        self.closed = True


def make_sock_rx(filtered: bool = False, buffer_seconds: int = 12) -> SocketReceiver:
    """A SocketReceiver with the fake socket injected (bypassing ``__enter__``)."""
    rx = SocketReceiver(filtered=filtered, buffer_seconds=buffer_seconds)
    rx.sock = FakeSock()
    rx._srv = None        # no listener → _accept_if_idle is a safe no-op
    return rx


def test_socket_poll_parses_samples() -> None:
    rx = make_sock_rx()
    rx.sock.feed(b"1,2,3\n4,5,6\n")
    assert rx.poll() == 2
    assert rx.total_samples == 2


def test_socket_no_data_is_not_an_error() -> None:
    # recv raises BlockingIOError when the socket is open but idle; poll must
    # swallow it and report zero new samples rather than crash the loop.
    rx = make_sock_rx()
    assert rx.poll() == 0


def test_socket_send_writes_command() -> None:
    rx = make_sock_rx()
    rx.send(b"C")
    rx.send(b"S")
    assert bytes(rx.sock.sent) == b"CS"


def test_socket_peer_close_drops_connection() -> None:
    rx = make_sock_rx()
    rx.sock.feed(b"1,2,3\n")
    assert rx.poll() == 1            # queued sample arrives
    rx.sock.close_peer()  # ESP32 dropped
    assert rx.poll() == 0            # clean EOF → no new samples, no crash
    assert rx.sock is None           # dropped, ready to re-accept a reconnect


def test_make_receiver_picks_transport() -> None:
    # Constructed but not entered, so no port is opened and no socket is bound.
    assert isinstance(make_receiver("serial"), SerialReceiver)
    assert isinstance(make_receiver("socket"), SocketReceiver)
    assert isinstance(
        make_receiver("replay", replay_path="dummy.txt"), ReplayReceiver
    )
    with pytest.raises(ValueError):
        make_receiver("carrier-pigeon")
    with pytest.raises(ValueError):
        make_receiver("replay")            # replay needs a path


# ── ReplayReceiver: replay a Daphnet file as a paced 64 Hz stream ────────────
def _daphnet_row(ax: int, ay: int, az: int, annot: int, t: float = 0.0) -> str:
    """One space-delimited Daphnet line: time, ankle xyz, thigh+trunk, annot.

    Mirrors the real layout — ankle at columns 1-3, annotation at column 10
    (0 = not in experiment, 1 = no-freeze, 2 = freeze).
    """
    return f"{t} {ax} {ay} {az} 0 0 0 0 0 0 {annot}"


def _write_daphnet(tmp_path, rows: list[str]):
    p = tmp_path / "S01R01.txt"
    p.write_text("\n".join(rows) + "\n")
    return str(p)


def test_replay_streams_ankle_and_drops_annot0(tmp_path) -> None:
    # Middle row is annot 0 → must be dropped; the two real rows stream through
    # carrying their ankle (cols 1-3) values, unfiltered so they pass unchanged.
    path = _write_daphnet(tmp_path, [
        _daphnet_row(100, 110, 120, 1),
        _daphnet_row(999, 999, 999, 0),     # not in experiment → dropped
        _daphnet_row(200, 210, 220, 2),
    ])
    rx = ReplayReceiver(path, loop=False, filtered=False).__enter__()
    rx._t0 = time.monotonic() - 1.0          # make every sample due (1 s @ 64 Hz)
    assert rx.poll() == 2                     # the annot-0 row was dropped
    win = rx.get_latest_window(2)
    assert win is not None
    np.testing.assert_array_equal(win[0], [100, 110, 120])
    np.testing.assert_array_equal(win[1], [200, 210, 220])


def test_replay_is_real_time_paced(tmp_path) -> None:
    # Nothing is "due" until wall-clock time elapses, so a poll right at start
    # yields zero samples — this is what keeps the replay at true 64 Hz.
    path = _write_daphnet(tmp_path, [_daphnet_row(1, 2, 3, 1)])
    rx = ReplayReceiver(path, loop=False).__enter__()
    rx._t0 = time.monotonic() + 10.0          # pretend replay starts in the future
    assert rx.poll() == 0


def test_replay_loops_by_default(tmp_path) -> None:
    # Two rows, looping: after ~5 samples' worth of time, 5 should have streamed
    # (rows replay 0,1,0,1,0...), proving the wrap-around.
    path = _write_daphnet(tmp_path, [
        _daphnet_row(10, 0, 0, 1),
        _daphnet_row(20, 0, 0, 2),
    ])
    rx = ReplayReceiver(path, loop=True, filtered=False).__enter__()
    rx._t0 = time.monotonic() - 5.5 / SAMPLE_RATE   # 5 samples due
    assert rx.poll() == 5
    assert rx.total_samples == 5


def test_replay_stops_at_end_when_not_looping(tmp_path) -> None:
    path = _write_daphnet(tmp_path, [
        _daphnet_row(10, 0, 0, 1),
        _daphnet_row(20, 0, 0, 2),
    ])
    rx = ReplayReceiver(path, loop=False, filtered=False).__enter__()
    rx._t0 = time.monotonic() - 10.0          # far more time than the file holds
    assert rx.poll() == 2                     # only the 2 real samples, no wrap
    assert rx.poll() == 0                     # and nothing more ever comes


def test_replay_cue_bytes_are_swallowed(tmp_path) -> None:
    # No board downstream: send() must not raise; the bytes are just tallied.
    path = _write_daphnet(tmp_path, [_daphnet_row(1, 2, 3, 1)])
    rx = ReplayReceiver(path).__enter__()
    rx.send(b"C")
    rx.send(b"S")
    assert rx.cue_bytes == 2


def test_replay_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        ReplayReceiver(str(tmp_path / "nope.txt")).__enter__()


def test_replay_bad_sensor_raises(tmp_path) -> None:
    path = _write_daphnet(tmp_path, [_daphnet_row(1, 2, 3, 1)])
    with pytest.raises(ValueError):
        ReplayReceiver(path, sensor="wrist").__enter__()


def test_replay_rejects_non_daphnet_file(tmp_path) -> None:
    # Too few columns to be a Daphnet record → a clear error, not a crash later.
    p = tmp_path / "bad.txt"
    p.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(ValueError):
        ReplayReceiver(str(p)).__enter__()
