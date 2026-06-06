"""export_fog_tflite.py — convert the trained PyTorch FoGNet into an int8 TFLite
model for the ESP32-S3 (TensorFlow Lite Micro) and emit the C header + every
constant the firmware needs.

WHY THIS RUNS IN COLAB, NOT THE PROJECT VENV
    Producing a .tflite flatbuffer needs TensorFlow, and TF has no wheel for the
    project's Python 3.14. Google Colab ships a TF-compatible Python, so run this
    there (the repo already uses Colab). Everything that does NOT need TF — the
    filter choice, the SOS coefficients, the z-score stats — was settled locally
    in fog_filter_probe and is baked straight into the sketch.

WHAT IT DOES
    1. rebuild FoGNet in Keras (channels-LAST) and port the trained weights
    2. assert Keras == PyTorch outputs on real windows  (catches any transpose bug)
    3. full-integer int8 quantization with a representative set of real windows
    4. report int8-vs-float sens/spec, under BOTH the training filter and the
       on-device per-window SOS filtfilt the sketch actually runs
    5. write fog_cnn_model.h (C byte array) and print the op list + an arena
       upper bound + the int8 I/O quant params for reference

COLAB SETUP (first cell)
    !pip -q install torch tensorflow scipy numpy
    # Easiest — clone so fog/ + train_fog.py are found automatically:
    !git clone <your-repo-url> /content/scenario2_pd
    # Then add the trained artifacts (fog_model.pth, fog_norm.npz) and the Daphnet
    # data (the folder with S01R01.txt ...) via the Files panel or Drive. Paths
    # auto-detect; override with FOG_MODEL / FOG_NORM / FOG_DATA / FOG_REPO.
    # No git? Upload the CONTENTS of pi/ — the fog/ folder AND train_fog.py —
    # straight into /content.

    Then:  !python export_fog_tflite.py
"""
import glob
import os
import sys

import numpy as np
import scipy.signal as sig
import tensorflow as tf
import torch

# ── locate the deployed `fog` package + train_fog.py (both live in pi/) ──────
def _find_pkg_dir():
    """Dir holding BOTH fog/__init__.py and train_fog.py (the imports below need both)."""
    for c in (os.environ.get("FOG_REPO", ""), "/content",
              "/content/scenario2_pd/pi", "scenario2_pd/pi", "../pi", "pi", "."):
        if c and os.path.isfile(os.path.join(c, "fog", "__init__.py")) \
             and os.path.isfile(os.path.join(c, "train_fog.py")):
            return os.path.abspath(c)
    return None


_pkg = _find_pkg_dir()
if _pkg is None:
    raise SystemExit(
        "Can't find the fog/ package + train_fog.py (both live in pi/). Easiest "
        "fix: !git clone <your-repo-url> /content/scenario2_pd  — or upload the "
        "CONTENTS of pi/ into /content — or set FOG_REPO to the dir holding them.")
if _pkg not in sys.path:
    sys.path.insert(0, _pkg)
print(f"· fog package from: {_pkg}")


def _find_data_dir():
    """Dir directly holding Daphnet S*R*.txt — FOG_DATA, then a recursive search."""
    env = os.environ.get("FOG_DATA", "")
    if env and glob.glob(os.path.join(env, "S*R*.txt")):
        return env
    hits = (glob.glob("/content/**/S*R*.txt", recursive=True)
            or glob.glob("**/S*R*.txt", recursive=True))
    return os.path.dirname(hits[0]) if hits else (env or "/content/dataset")


DATA = _find_data_dir()
MODEL = os.environ.get("FOG_MODEL", "/content/fog_model.pth")
NORM = os.environ.get("FOG_NORM", "/content/fog_norm.npz")
OUT_H = os.environ.get("FOG_OUT", "fog_cnn_model.h")

from fog.config import SAMPLE_RATE, WINDOW_HOP, WINDOW_SIZE  # noqa: E402
from fog.metrics import sens_spec                            # noqa: E402
from fog.model import FoGNet                                 # noqa: E402
from fog.normalize import Normaliser                         # noqa: E402
from train_fog import build_windows, load_daphnet            # noqa: E402

FS = SAMPLE_RATE
NYQ = FS / 2
B_BA, A_BA = sig.butter(4, [0.5 / NYQ, 15.0 / NYQ], btype="band")
SOS = sig.butter(4, [0.5 / NYQ, 15.0 / NYQ], btype="band", output="sos")

# ════════════════════════════════════════════════════════════════════════════
#  1. load the trained PyTorch model + normaliser
# ════════════════════════════════════════════════════════════════════════════
ck = torch.load(MODEL, map_location="cpu", weights_only=False)
arch, labels = ck["arch"], ck["labels"]
pt = FoGNet(num_classes=len(labels), **arch)
pt.load_state_dict(ck["model_state"]); pt.eval()
norm = Normaliser.load(NORM)
print(f"loaded FoGNet {arch}  labels={labels}")


# ════════════════════════════════════════════════════════════════════════════
#  2. rebuild in Keras (channels-last) and port the weights
# ════════════════════════════════════════════════════════════════════════════
def build_keras():
    from tensorflow.keras import Model, layers
    inp = layers.Input((WINDOW_SIZE, arch["num_axes"]), name="accel")   # (256, 3)
    x = layers.Conv1D(arch["c1"], 15, padding="same", name="conv1")(inp)
    x = layers.BatchNormalization(epsilon=1e-5, name="bn1")(x)          # match torch eps
    x = layers.ReLU()(x); x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(arch["c2"], 9, padding="same", name="conv2")(x)
    x = layers.BatchNormalization(epsilon=1e-5, name="bn2")(x)
    x = layers.ReLU()(x); x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(arch["c3"], 5, padding="same", name="conv3")(x)
    x = layers.BatchNormalization(epsilon=1e-5, name="bn3")(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(arch["fc"], name="fc1")(x)
    x = layers.ReLU()(x)
    x = layers.Dense(len(labels), name="fc2")(x)
    out = layers.Softmax(name="prob")(x)        # device gets P(freeze) directly
    return Model(inp, out)


km = build_keras()
sd = {k: v.numpy() for k, v in ck["model_state"].items()}
# conv: torch (O,I,K) → keras (K,I,O);  dense: torch (O,I) → keras (I,O)
km.get_layer("conv1").set_weights([sd["conv1.weight"].transpose(2, 1, 0), sd["conv1.bias"]])
km.get_layer("conv2").set_weights([sd["conv2.weight"].transpose(2, 1, 0), sd["conv2.bias"]])
km.get_layer("conv3").set_weights([sd["conv3.weight"].transpose(2, 1, 0), sd["conv3.bias"]])
for i in (1, 2, 3):                              # bn: [gamma, beta, mean, var]
    km.get_layer(f"bn{i}").set_weights([sd[f"bn{i}.weight"], sd[f"bn{i}.bias"],
                                        sd[f"bn{i}.running_mean"], sd[f"bn{i}.running_var"]])
km.get_layer("fc1").set_weights([sd["fc1.weight"].T, sd["fc1.bias"]])
km.get_layer("fc2").set_weights([sd["fc2.weight"].T, sd["fc2.bias"]])


# ════════════════════════════════════════════════════════════════════════════
#  3. data: windows (training filter) + the on-device per-window SOS filtfilt
# ════════════════════════════════════════════════════════════════════════════
def sos_filtfilt(win, pad=32):
    """Per-window zero-phase SOS filtfilt — the EXACT algorithm the firmware runs
    (odd-reflection pad, DF2T forward, reverse, forward, reverse). win: (W,C)."""
    W, C = win.shape
    out = np.empty_like(win, dtype=np.float64)
    for c in range(C):
        x = win[:, c].astype(np.float64)
        xp = np.concatenate([2 * x[0] - x[pad:0:-1], x, 2 * x[-1] - x[-2:-pad - 2:-1]])
        y = _fwd(xp); y = _fwd(y[::-1])[::-1]
        out[:, c] = y[pad:pad + W]
    return out


def _fwd(x):
    y = x.copy()
    for s in range(SOS.shape[0]):
        b0, b1, b2, _a0, a1, a2 = SOS[s]
        z1 = z2 = 0.0; o = np.empty_like(y)
        for i in range(len(y)):
            xi = y[i]; oi = b0 * xi + z1
            z1 = b1 * xi - a1 * oi + z2; z2 = b2 * xi - a2 * oi; o[i] = oi
        y = o
    return y


subs = load_daphnet(DATA, "ankle")
windows = {s: build_windows(r) for s, r in subs.items()}      # whole-run filtfilt (training)
windows = {s: (X, y) for s, (X, y) in windows.items() if len(y) > 0}
test_s = max(windows, key=lambda s: int((windows[s][1] == 1).sum()))  # most-freeze holdout
X_all = np.concatenate([windows[s][0] for s in windows])              # (N,3,256) training-filtered
X_te, y_te = windows[test_s]
print(f"holdout {test_s}: {len(y_te)} windows, {int(y_te.sum())} freeze")

# the SAME held-out runs re-windowed with the on-device per-window filtfilt
raw = subs[test_s]
Xdev, ydev = [], []
for s_run, ann in raw:
    if len(s_run) < WINDOW_SIZE:
        continue
    n = (len(s_run) - WINDOW_SIZE) // WINDOW_HOP + 1
    for i in range(n):
        sl = slice(i * WINDOW_HOP, i * WINDOW_HOP + WINDOW_SIZE)
        v = ann[sl][ann[sl] > 0]
        if v.size == 0:
            continue
        Xdev.append(sos_filtfilt(s_run[sl]).T.astype(np.float32))
        ydev.append(1 if (v == 2).mean() > 0.5 else 0)
Xdev, ydev = np.stack(Xdev), np.array(ydev)


def to_keras(Xncw):                              # (N,3,256) normalized → (N,256,3)
    return np.transpose(norm.transform(Xncw), (0, 2, 1)).astype(np.float32)


# parity: Keras softmax vs PyTorch softmax on the training-filter holdout
with torch.no_grad():
    pt_prob = torch.softmax(pt(torch.from_numpy(norm.transform(X_te))), 1).numpy()
kr_prob = km.predict(to_keras(X_te), verbose=0)
dmax = float(np.max(np.abs(pt_prob - kr_prob)))
print(f"parity max|keras-pytorch| = {dmax:.2e}", "OK" if dmax < 1e-3 else "!! CHECK PORT")


# ════════════════════════════════════════════════════════════════════════════
#  4. int8 full-integer quantization
# ════════════════════════════════════════════════════════════════════════════
rep = to_keras(X_all)
np.random.seed(0)
rep = rep[np.random.permutation(len(rep))[:400]]


def rep_data():
    for r in rep:
        yield [r[None].astype(np.float32)]


conv = tf.lite.TFLiteConverter.from_keras_model(km)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_data
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8
conv.inference_output_type = tf.int8
tflite_model = conv.convert()
print(f"\nint8 model: {len(tflite_model)} bytes")


# ════════════════════════════════════════════════════════════════════════════
#  5. evaluate the int8 model + report what the sketch needs
# ════════════════════════════════════════════════════════════════════════════
interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
inp_d, out_d = interp.get_input_details()[0], interp.get_output_details()[0]
in_s, in_zp = inp_d["quantization"]
out_s, out_zp = out_d["quantization"]
fi = 1 if labels[1] == "freeze" else labels.index("freeze")


def int8_predict(Xkeras):
    preds = np.empty(len(Xkeras), np.int64)
    for i, x in enumerate(Xkeras):
        q = np.clip(np.round(x / in_s + in_zp), -128, 127).astype(np.int8)
        interp.set_tensor(inp_d["index"], q[None])
        interp.invoke()
        o = interp.get_tensor(out_d["index"])[0].astype(np.float32)
        prob = (o - out_zp) * out_s
        preds[i] = int(prob[fi] >= 0.5)
    return preds


print("\nsens/spec on held-out", test_s, ":")
ps, pp = (torch.softmax(pt(torch.from_numpy(norm.transform(X_te))), 1).argmax(1).numpy()), None
s, sp, _ = sens_spec(y_te, ps); print(f"  float PyTorch  (train filter)   sens={s:.3f} spec={sp:.3f}")
s, sp, _ = sens_spec(y_te, int8_predict(to_keras(X_te)))
print(f"  int8  TFLite   (train filter)   sens={s:.3f} spec={sp:.3f}")
s, sp, _ = sens_spec(ydev, int8_predict(to_keras(Xdev)))
print(f"  int8  TFLite   (ON-DEVICE filt) sens={s:.3f} spec={sp:.3f}   <-- expect this on the S3")

# op list for the MicroMutableOpResolver
ops = sorted({o["op_name"] for o in interp._get_ops_details()})
arena_ub = sum(int(np.prod(t["shape"])) * t["dtype"]().itemsize
               for t in interp.get_tensor_details() if t["shape"].size)
print("\n── paste-ready for esp32_s3_fog.ino ──")
print("ops (MicroMutableOpResolver):", ", ".join(ops))
print(f"arena upper bound  ~{arena_ub/1024:.0f} KB (sketch sets 100 KB; trim via arena_used_bytes())")
print(f"int8 input  scale={in_s:.8g} zero_point={in_zp}")
print(f"int8 output scale={out_s:.8g} zero_point={out_zp}  (read live in the sketch; shown for sanity)")

# write the C header
with open(OUT_H, "w") as f:
    f.write('// Auto-generated by export_fog_tflite.py — int8 FoGNet for TFLM.\n')
    f.write('// Do not edit by hand; re-run the exporter to regenerate.\n')
    f.write("#pragma once\n#include <cstdint>\n\n")
    f.write("alignas(8) const unsigned char g_fog_model[] = {\n")
    for i in range(0, len(tflite_model), 12):
        f.write("  " + ",".join(f"0x{b:02x}" for b in tflite_model[i:i + 12]) + ",\n")
    f.write("};\n")
    f.write(f"const unsigned int g_fog_model_len = {len(tflite_model)};\n")
print(f"\nwrote {OUT_H}  ({len(tflite_model)} bytes)  → copy next to esp32_s3_fog.ino")
