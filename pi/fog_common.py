"""Backward-compatibility shim — the implementation now lives in the ``fog`` package.

Prefer the layered imports going forward (they keep torch out of the signal-
processing path)::

    from fog.dsp import freeze_index, movement_energy   # NumPy + SciPy only
    from fog.metrics import sens_spec                    # NumPy only
    from fog.model import FoGNet                          # pulls in torch

This module re-exports the historical flat API so any existing notebook or
script that did ``from fog_common import ...`` keeps working unchanged. ``FoGNet``
is resolved lazily, so merely importing this shim does **not** drag in torch.
"""
from __future__ import annotations

from fog.config import (  # noqa: F401
    BRIDGE_HOST,
    BRIDGE_PORT,
    CUE_OFF,
    CUE_ON,
    FI_THRESHOLD,
    FREEZE_BAND,
    G_TO_MG,
    LABELS,
    LOCO_BAND,
    NUM_AXES,
    SAMPLE_RATE,
    SERIAL_BAUD,
    SERIAL_PORT,
    TREMOR_BAND,
    WINDOW_HOP,
    WINDOW_SIZE,
)
from fog.dsp import (  # noqa: F401
    AccelFilter,
    band_power,
    filter_offline,
    freeze_index,
    magnitude,
    movement_energy,
    parse_line,
    tremor_power,
    window_signal,
)
from fog.metrics import best_fi_threshold, freeze_index_predict, sens_spec  # noqa: F401
from fog.normalize import Normaliser  # noqa: F401
from fog.streaming import (  # noqa: F401
    SerialReceiver,
    SocketReceiver,
    StreamReceiver,
    make_receiver,
)

# Historical private alias (fog_common exposed `_magnitude`).
_magnitude = magnitude


def __getattr__(name: str) -> object:
    """Resolve ``fog_common.FoGNet`` lazily so this shim never forces a torch import."""
    if name == "FoGNet":
        from fog.model import FoGNet

        return FoGNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
