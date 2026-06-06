"""Tests for the closed-loop CueController debounce + episode log (stream_demo).

CueController lives in stream_demo.py, which imports torch at module scope, so
these are marked needs_torch. The logic under test is safety-relevant: a single
twitchy window must NOT buzz the wearer — the cue only fires after ONSET_WINDOWS
consecutive freezes and only clears after OFFSET_WINDOWS consecutive clears. In
dry-run (rx=None) the cue commands are no-ops, so no hardware is needed.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_torch


def test_single_freeze_window_does_not_cue(tmp_path: Path) -> None:
    from stream_demo import CueController

    cue = CueController(rx=None, log_path=str(tmp_path / "ep.csv"))
    cue.update(is_freeze=True, fi=3.0, now=0.0)
    assert cue.cueing is False          # one window < ONSET_WINDOWS (2)


def test_onset_and_offset_debounce(tmp_path: Path) -> None:
    from stream_demo import CueController

    cue = CueController(rx=None, log_path=str(tmp_path / "ep.csv"))
    cue.update(True, 3.0, 0.0)
    cue.update(True, 3.5, 2.0)
    assert cue.cueing is True           # 2 consecutive freezes → cue ON
    cue.update(False, 0.1, 4.0)
    assert cue.cueing is True           # 1 clear < OFFSET_WINDOWS (2)
    cue.update(False, 0.1, 6.0)
    assert cue.cueing is False          # 2 consecutive clears → cue OFF


def test_episode_logged_with_peak_fi_and_duration(tmp_path: Path) -> None:
    from stream_demo import CueController

    log = tmp_path / "ep.csv"
    cue = CueController(rx=None, log_path=str(log))
    cue.update(True, 3.0, 0.0)
    cue.update(True, 3.5, 2.0)          # cue ON, episode starts at now=2.0
    cue.update(False, 0.1, 4.0)
    cue.update(False, 0.1, 6.0)         # cue OFF, episode ends at now=6.0

    rows = list(csv.reader(log.open()))
    assert rows[0] == ["start_unix", "start_iso", "duration_s", "peak_freeze_index"]
    assert len(rows) == 2               # header + exactly one episode
    assert float(rows[1][2]) == pytest.approx(4.0)   # 6.0 - 2.0
    assert float(rows[1][3]) == pytest.approx(3.5)   # peak FI over the run


def test_intermittent_freezes_never_confirm(tmp_path: Path) -> None:
    from stream_demo import CueController

    cue = CueController(rx=None, log_path=str(tmp_path / "ep.csv"))
    # freeze, clear, freeze, clear ... never two in a row → never cues.
    for i in range(8):
        cue.update(is_freeze=(i % 2 == 0), fi=3.0, now=float(i))
        assert cue.cueing is False
