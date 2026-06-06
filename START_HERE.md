# Parkinson's Closed-Loop Gait Garment · Start Here

A non-invasive smart-clothing wearable for Parkinson's disease, built on a
**Circuit Playground Express**. ONE onboard accelerometer drives a **3-in-1**
device that spans three of the four scenario purposes at once:

| Purpose | What it does |
|---|---|
| **Monitoring** | logs 4-6 Hz rest-**tremor** band power over time |
| **Early diagnosis / assessment** | detects **freezing of gait (FoG)** — a 1D-CNN when tethered, the classical Freeze-Index on-board |
| **Treatment delivery** | the instant a freeze starts, a **vibration motor pulses a rhythmic gait cue** — a known therapy that helps PD patients "unfreeze" and step |

The closed loop runs three ways, all sharing one accelerometer in and one motor
out. **Default (untethered):** the CPX sends 64 Hz accel over a short UART link to
a small ESP32, which relays it over Wi-Fi to a laptop; the laptop
runs the CNN and, on a detected freeze, sends a cue command back down the same
path to buzz the motor — that return path is what makes it **closed-loop**, not
just a logger. **Legacy (tethered):** drop the ESP32 and plug the CPX straight
into the host over USB — same byte protocol. **Standalone:** the CPX runs the
classical Freeze-Index detector and drives the cue entirely on-device, no host
required (the full CNN won't fit on the Cortex-M0+). The on-board-CNN (ESP32-S3)
and no-laptop (ESP32 hub) builds live in `firmware/variants/`.

---

## Folder layout

```
scenario2_pd/
├── START_HERE.md                      ← you are here
│
├── firmware/                          ← Arduino sketches (CANONICAL build at top level; alternates in variants/)
│   ├── cpx_fog_streamer/
│   │   └── cpx_fog_streamer.ino       HERO · CPX: 64 Hz accel → UART · 'C'/'S' → motor cue
│   ├── esp32_wifi_bridge/
│   │   └── esp32_wifi_bridge.ino      HERO · ESP32: transparent UART⇄Wi-Fi/TCP bridge to the laptop
│   └── variants/
│       ├── cpx_fog_standalone/        no-host: whole loop ON THE CPX (Freeze-Index + gated cue)
│       ├── cpx_fog_sensor/            CPX sensor end paired with the on-body-CNN variant (S3)
│       ├── esp32_s3_fog/              advanced: runs the CNN ON the ESP32-S3 (its own GPIO18 link)
│       └── esp32_hub/                 no-laptop: ESP32 hosts the dashboard itself + MAX30102 HR/SpO2
│
├── hardware/                          ← physical garment design
│   └── sleeve_layout.py               renders the sleeve component-layout figure (→ .png / .pdf)
│
├── colab/                             ← Google-Colab training pipeline (PyTorch · self-contained)
│   ├── fog_allinone.py                load Daphnet → train CNN → LOSO eval vs. Freeze-Index
│   ├── fog_interpret.py               feature-model interpretability (perm. importance · PDP · SHAP)
│   └── gen_device_coeffs.py           tune FI threshold + emit the C++ filter block for the .ino
│
└── pi/                                ← everything that runs on the host (your laptop)
    ├── fog/                           torch-free core package (firmware-equivalent DSP + metrics)
    │   ├── config.py                  single source of truth for constants (rates · bands · thresholds)
    │   ├── dsp.py                     serial parse · band-pass · freeze index · tremor · windowing
    │   ├── metrics.py                 sensitivity / specificity + Freeze-Index baseline classifier
    │   ├── normalize.py               per-channel z-score (fit on train only)
    │   ├── streaming.py               non-blocking USB-serial receiver with a ring buffer
    │   └── model.py                   FoGNet 1-D CNN — the ONLY module that imports torch
    ├── fog_common.py                  back-compat shim re-exporting the flat fog.* API
    ├── fog_analysis.py                offline analysis + report figures from a trained model
    ├── train_fog.py                   train the freeze classifier on Daphnet (LOSO)
    ├── stream_demo.py                 live inference + closed-loop cue + episode log
    ├── dashboard_server.py            FastAPI + WebSocket dashboard backend
    ├── dashboard.html                 browser dashboard (freeze state · cue · FI · tremor)
    ├── tests/                         pytest suite (DSP · metrics · gate · biquad-equivalence · model …)
    ├── requirements.txt               pinned deps, grouped: core / inference / train / dev
    └── pyproject.toml                 packaging + ruff + mypy + pytest configuration
```

---

## Why this design (the bits that earn marks)

- **It uses real ML on real data.** We can't collect Parkinson's freezing data in
  a week, so the CNN trains on the public **Daphnet Freezing-of-Gait dataset**
  (UCI) — real PD patients, accelerometer-only, sampled at **64 Hz**. The board
  samples at 64 Hz too, so the model transfers with no resampling.
- **The hardware is honest about its limits.** Every kit's board is a Circuit
  Playground *Express* (accel-only LIS3DH, no gyro, no Bluetooth), so we built a
  device that needs **only one accelerometer in and one motor out**. Nothing in
  the design assumes a sensor the kit doesn't have.
- **The accuracy story is rigorous** (this feeds the Accuracy Worksheet directly):
  - **Leave-One-Subject-Out** validation — train on N-1 patients, test on a
    brand-new one. No patient appears in both train and test.
  - We report **sensitivity & specificity**, never accuracy — freezes are rare,
    a missed freeze can mean a fall, and a cue that cries wolf is useless.
  - We score the classic **Freeze-Index** metric (Moore 2008) on the same folds,
    so the worksheet can say "the CNN beats the textbook threshold by X".

---

## Quick start

### Step 1 · Flash the board
1. Arduino IDE 2.x → install **Adafruit Circuit Playground** library (Library Manager)
   and **Adafruit SAMD Boards** (Boards Manager).
2. Open `firmware/cpx_fog_streamer/cpx_fog_streamer.ino`, select board
   **Adafruit Circuit Playground Express**, upload.
3. Wire the vibration motor to pad **A1** through the transistor circuit drawn in
   the sketch header (a pad can't drive a motor directly).
4. Open the Serial Monitor at **115200** — you should see `ax,ay,az` lines streaming.

> **Standalone (no host):** flash `firmware/variants/cpx_fog_standalone/cpx_fog_standalone.ino`
> instead — the board then detects freezes and cues on its own. Stand still and press
> **button A** once to calibrate the movement-energy floor to the wearer.

### Step 2 · Get the data & train
```bash
cd pi
pip install -r requirements.txt   # torch, scipy, numpy, sklearn, pandas … (all groups)
# download Daphnet from the UCI link in train_fog.py, unzip so you have .../dataset/S01R01.txt
python train_fog.py --data ~/Documents/daphnet/dataset --sensor ankle
# → prints per-subject LOSO sensitivity/specificity + the FI baseline,
#   and saves fog_model.pth + fog_norm.npz
```
Copy the printed pooled sensitivity/specificity + confusion matrix straight into
the **Accuracy Worksheet**.

### Step 3 · Run the closed loop
Pick ONE (both bind the same link; default is the ESP32 Wi-Fi bridge):
```bash
# (a) live web dashboard — freeze state, cue indicator, freeze-index & tremor charts
pip install fastapi 'uvicorn[standard]' scipy numpy torch
python dashboard_server.py                      # waits for the ESP32 bridge → open http://localhost:8000/
#   …tethered over USB instead:  python dashboard_server.py --transport serial --port /dev/ttyACM0   (+pip install pyserial)

# (b) headless demo — prints state, buzzes on freeze, logs episodes to fog_episodes.csv
python stream_demo.py                           # default bridge;  --transport serial --port … for USB
```
> Serial port is `/dev/ttyACM0` (Linux), `/dev/cu.usbmodemXXXX` (Mac), `COMx` (Windows).
> Both scripts run print-only / dry if no link is present, so you can demo the
> logic without hardware.

---

## Where this maps onto the deliverables (due Fri 5 June, 5pm)
- **Accuracy worksheet** ← the LOSO sensitivity/specificity + Freeze-Index baseline from `train_fog.py`.
- **Leaflet / advert** ← the 3-in-1 monitor→detect→cue story; real-world anchors are
  Apple Watch's Movement Disorder API, PDMonitor, STAT-ON, and laser/cueing aids like NextStride.
- **Project Reflection quiz** ← **write this entirely yourself. It is Category 1: NO generative
  AI is permitted, and any AI use is Academic Misconduct.** (AI help on the design/code above is
  explicitly allowed; the reflection is not.)
