"""Project-wide constants for the Parkinson's closed-loop gait garment.

Single source of truth for everything the firmware, the offline trainer and the
live inference paths must agree on: sample rate, window geometry, spectral bands
and the USB serial protocol. Importing this module pulls in **no third-party
packages at all** (not even NumPy), so it is safe to read from any layer.

These values are mirrored in the firmware (cpx_fog_streamer.ino /
cpx_fog_standalone.ino) and were chosen to match the Daphnet Freezing-of-Gait
dataset, so a model trained offline transfers to the board with no resampling.
"""
from __future__ import annotations

# ── Sampling ────────────────────────────────────────────────────────────────
SAMPLE_RATE: int = 64        # Hz — matches the firmware AND the Daphnet dataset
NUM_AXES: int = 3            # LIS3DH accelerometer: x, y, z

# ── Class map ───────────────────────────────────────────────────────────────
# Index 0 / 1 — the order *is* the label map and must not be reordered.
LABELS: list[str] = ['no_freeze', 'freeze']

# ── Real-time windowing ─────────────────────────────────────────────────────
# 4 s windows, 2 s hop → one decision every 2 s at 50 % overlap. FoG events last
# seconds, so a long window is correct here: it buys spectral resolution for the
# 3-8 Hz freeze band.
WINDOW_SIZE: int = 256       # 4.0 s @ 64 Hz
WINDOW_HOP: int = 128        # 2.0 s @ 64 Hz

# ── Spectral bands (Hz) ─────────────────────────────────────────────────────
# The freeze and locomotor bands are the Moore et al. (2008) definition; the
# tremor band is the classic 4-6 Hz Parkinsonian rest tremor.
LOCO_BAND: tuple[float, float] = (0.5, 3.0)
FREEZE_BAND: tuple[float, float] = (3.0, 8.0)
TREMOR_BAND: tuple[float, float] = (4.0, 6.0)

# ── Freeze-Index baseline threshold ─────────────────────────────────────────
# Deployed operating point for the engineered-feature BASELINE only (the CNN does
# not use this). FI = power(freeze) / power(loco); > threshold ⇒ freeze. The board
# ships 1.815, biased to sensitivity: the clean sens+spec optimum on capture1 is
# 2.10 (specificity 100%), but the lower 1.815 (specificity 92% there) stays
# sensitive on harder real-world gait. train_fog.py still sweeps it per fold.
FI_THRESHOLD: float = 1.815

# ── Link protocol (transport-agnostic payloads) ─────────────────────────────
# The board ⇄ laptop protocol is identical regardless of the wire: newline-delimited
# "ax,ay,az" lines going up, single 'C'/'S' cue bytes coming down.
CUE_ON: bytes = b'C'         # laptop → board: start the vibrotactile cue
CUE_OFF: bytes = b'S'        # laptop → board: stop the cue
SERIAL_BAUD: int = 115200    # UART baud (CPX↔ESP32 link, and the legacy USB tether)

# ── Live transport: Wi-Fi bridge (default) ──────────────────────────────────
# The wearable is untethered: CPX --UART--> ESP32 --Wi-Fi/TCP--> laptop. The laptop is
# the TCP *server* (it has the fixed address) and the ESP32 dials in as a client.
BRIDGE_HOST: str = '0.0.0.0'   # interface the laptop binds (0.0.0.0 = all NICs)
BRIDGE_PORT: int = 8765        # TCP port the ESP32 connects to

# ── Legacy transport: direct USB serial ─────────────────────────────────────
# CPX tethered straight to the host with no ESP32 — kept for standalone bring-up.
SERIAL_PORT: str = '/dev/ttyACM0'   # Linux; Mac /dev/cu.usbmodem*, Windows COMx

# ── Unit conversion ─────────────────────────────────────────────────────────
G_TO_MG: float = 1000.0      # 1 g → 1000 milli-g
