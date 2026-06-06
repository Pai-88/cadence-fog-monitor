"""fog — toolkit for the Parkinson's closed-loop gait garment (ENGF0031 Scenario 2).

Layered so the signal-processing / serial core never imports torch::

    from fog.dsp import freeze_index, movement_energy   # NumPy + SciPy only
    from fog.metrics import sens_spec                    # NumPy only
    from fog.normalize import Normaliser                 # NumPy only
    from fog.streaming import SocketReceiver             # ESP32 Wi-Fi bridge (TCP)
    from fog.streaming import SerialReceiver             # legacy USB (+ pyserial, lazy)
    from fog.model import FoGNet                          # pulls in torch

``from fog import X`` works for any public name, but torch is imported only when
you actually touch a torch-backed symbol (``FoGNet``) — so ``import fog`` on a
laptop or in a torch-free test environment stays cheap.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from .config import (
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
from .dsp import (
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
from .metrics import best_fi_threshold, freeze_index_predict, sens_spec
from .normalize import Normaliser
from .streaming import (
    SerialReceiver,
    SocketReceiver,
    StreamReceiver,
    make_receiver,
)

if TYPE_CHECKING:  # for type checkers / IDEs only — no torch import at runtime
    from .model import FoGNet

__version__ = "1.0.0"

__all__ = [
    # config
    "SAMPLE_RATE", "NUM_AXES", "LABELS", "WINDOW_SIZE", "WINDOW_HOP",
    "LOCO_BAND", "FREEZE_BAND", "TREMOR_BAND", "FI_THRESHOLD",
    "SERIAL_PORT", "SERIAL_BAUD", "BRIDGE_HOST", "BRIDGE_PORT",
    "CUE_ON", "CUE_OFF", "G_TO_MG",
    # dsp
    "parse_line", "filter_offline", "AccelFilter", "magnitude", "band_power",
    "freeze_index", "tremor_power", "movement_energy", "window_signal",
    # metrics / normalisation / receivers
    "sens_spec", "freeze_index_predict", "best_fi_threshold", "Normaliser",
    "StreamReceiver", "SerialReceiver", "SocketReceiver", "make_receiver",
    # model (lazy)
    "FoGNet",
]


def __getattr__(name: str) -> object:
    """Resolve ``fog.FoGNet`` lazily so importing the package never needs torch."""
    if name == "FoGNet":
        from .model import FoGNet

        return FoGNet
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
