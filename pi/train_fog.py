"""
Train the freeze-of-gait classifier on the Daphnet FoG dataset.

Why Daphnet, and why this evaluation protocol — read before trusting a number:

  * We can't collect real Parkinson's freezing data in a one-week sprint, so we
    train on the public Daphnet Freezing-of-Gait dataset (Bachlin et al. 2010,
    UCI ML repo). It is real accelerometer data from PD patients @ 64 Hz, which
    is exactly our board's sample rate — no resampling, the model transfers.

  * Leave-One-Subject-Out (LOSO). A random window split would leak: windows
    overlap, and a patient's gait is so personal that the model would "recognise
    the person", not the freeze. LOSO trains on N-1 patients and tests on the
    held-out one, every fold — the honest estimate of "works on a NEW patient".

  * We report SENSITIVITY and SPECIFICITY, never accuracy. Freezes are rare, so
    a model that predicts "no freeze" always scores ~90 % accuracy and is
    useless. A missed freeze can mean a fall, so sensitivity is the metric that
    matters; specificity guards against a cue that cries wolf.

  * Freeze-Index baseline. We also score the classic engineered metric
    (Moore 2008) on the same folds, so the report can say "the CNN beats the
    textbook threshold by X" rather than quoting an unanchored number.

Get the data (≈25 MB):
    https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait+data+set
    unzip so that this folder sees  <data>/dataset/S01R01.txt ...

    python train_fog.py --data ~/Documents/daphnet/dataset --sensor ankle
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from fog.config import LABELS, SAMPLE_RATE, WINDOW_HOP, WINDOW_SIZE
from fog.dsp import filter_offline, window_signal
from fog.metrics import best_fi_threshold, freeze_index_predict, sens_spec
from fog.model import FoGNet
from fog.normalize import Normaliser

# Daphnet column layout (0-indexed). Annotation: 0 = drop, 1 = no-freeze, 2 = freeze.
SENSOR_COLS = {'ankle': (1, 2, 3), 'thigh': (4, 5, 6), 'trunk': (7, 8, 9)}
ANNOT_COL   = 10
FREEZE_FRACTION = 0.5   # a window is 'freeze' if >50 % of its valid samples are freeze


# ============================================================
#  DATA
# ============================================================
def load_daphnet(
    data_dir: str, sensor: str
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Return {subject_id: [(signal (T,3), annot (T,)), ...one per run...]}."""
    cols = SENSOR_COLS[sensor]
    files = sorted(glob.glob(os.path.join(data_dir, 'S*R*.txt')))
    if not files:
        raise FileNotFoundError(
            f"No Daphnet S*R*.txt files in {data_dir}. "
            f"Download from the UCI link in this file's docstring and unzip.")
    subjects: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for path in files:
        m = re.search(r'(S\d+)R\d+', os.path.basename(path))
        if m is None:
            continue
        subj = m.group(1)
        arr = np.loadtxt(path)
        sig = arr[:, list(cols)].astype(np.float32)
        annot = arr[:, ANNOT_COL].astype(np.int64)
        subjects.setdefault(subj, []).append((sig, annot))
    return subjects


def build_windows(
    runs: list[tuple[np.ndarray, np.ndarray]]
) -> tuple[np.ndarray, np.ndarray]:
    """List of (signal, annot) runs → (X (N,3,W) filtered, y (N,)).

    Windows are cut WITHIN a run (never across run boundaries), filtered
    zero-phase, then labelled by majority vote over the in-experiment samples.
    Windows that are entirely 'not in experiment' (annot 0) are dropped.
    """
    Xs, ys = [], []
    for sig, annot in runs:
        if len(sig) < WINDOW_SIZE:
            continue
        filt = filter_offline(sig)
        win = window_signal(filt, WINDOW_SIZE, WINDOW_HOP)            # (N,3,W)
        ann = window_signal(annot[:, None].astype(np.float32),
                            WINDOW_SIZE, WINDOW_HOP)[:, 0, :]          # (N,W)
        for i in range(len(win)):
            valid = ann[i][ann[i] > 0]          # samples that are part of the experiment
            if valid.size == 0:
                continue
            freeze_frac = (valid == 2).mean()
            Xs.append(win[i])
            ys.append(1 if freeze_frac > FREEZE_FRACTION else 0)
    if not Xs:
        return np.empty((0, 3, WINDOW_SIZE), np.float32), np.empty((0,), np.int64)
    return np.stack(Xs).astype(np.float32), np.array(ys, dtype=np.int64)


# Metrics (sens_spec) and the Freeze-Index baseline (freeze_index_predict,
# best_fi_threshold) now live in fog.metrics and are imported above — they are
# shared with fog_analysis.py and covered by the test suite.


# ============================================================
#  TRAIN ONE MODEL
# ============================================================
def train_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    device: str,
) -> tuple[FoGNet, Normaliser]:
    norm = Normaliser()
    norm.fit(X_tr)
    tr = DataLoader(TensorDataset(torch.from_numpy(norm.transform(X_tr)),
                                  torch.from_numpy(y_tr)),
                    batch_size=batch_size, shuffle=True)
    va = DataLoader(TensorDataset(torch.from_numpy(norm.transform(X_va)),
                                  torch.from_numpy(y_va)),
                    batch_size=batch_size)

    counts = np.bincount(y_tr, minlength=len(LABELS))
    weights = torch.tensor(len(y_tr) / (len(LABELS) * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    model = FoGNet().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_sens, best_state, wait = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        for X, y in tr:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Early-stop on val SENSITIVITY (a freeze detector that misses freezes
        # is worthless, so we select for recall, not accuracy).
        vp, vy = predict(model, va, device)
        sens, spec, _ = sens_spec(vy, vp)
        score = (0 if np.isnan(sens) else sens) + 0.3 * (0 if np.isnan(spec) else spec)
        if score > best_sens:
            best_sens, wait = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, norm


def predict(
    model: FoGNet, loader: DataLoader, device: str
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for X, y in loader:
            preds.extend(model(X.to(device)).argmax(1).cpu().numpy())
            ys.extend(y.numpy())
    return np.array(preds), np.array(ys)


# ============================================================
#  MAIN — LOSO cross-validation, then a final all-data model to ship
# ============================================================
def main(argv: list[str] | None = None) -> None:
    # argv=None → read sys.argv (CLI). In a notebook (Colab/Jupyter) there is no
    # CLI, so pass an explicit list, e.g. main(['--data', 'daphnet', '--sensor', 'ankle']).
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Daphnet dataset dir (has S01R01.txt ...)')
    parser.add_argument('--sensor', default='ankle', choices=list(SENSOR_COLS))
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--max-folds', type=int, default=0, help='0 = all subjects (full LOSO)')
    parser.add_argument('--output-model', default='fog_model.pth')
    parser.add_argument('--output-norm',  default='fog_norm.npz')
    args = parser.parse_args(argv)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 60)
    print(f"  Freeze-of-Gait Trainer  |  sensor: {args.sensor}  |  device: {device}")
    print("=" * 60)

    subjects = load_daphnet(args.data, args.sensor)
    windows = {s: build_windows(runs) for s, runs in subjects.items()}
    windows = {s: (X, y) for s, (X, y) in windows.items() if len(y) > 0}
    subj_ids = sorted(windows)
    for s in subj_ids:
        X, y = windows[s]
        print(f"  {s}: {len(y):4d} windows | no_freeze={int((y == 0).sum())}  "
              f"freeze={int((y == 1).sum())}")
    print()

    folds = subj_ids if args.max_folds == 0 else subj_ids[:args.max_folds]
    cnn_true: list[int] = []
    cnn_pred: list[int] = []
    base_true: list[int] = []
    base_pred: list[int] = []

    for test_s in folds:
        train_s = [s for s in subj_ids if s != test_s]
        X_tr = np.concatenate([windows[s][0] for s in train_s])
        y_tr = np.concatenate([windows[s][1] for s in train_s])
        X_te, y_te = windows[test_s]

        # Hold out, as the val subject, whichever training subject has the most
        # freeze windows — so early-stopping on sensitivity is meaningful.
        val_s = max(train_s, key=lambda s: int((windows[s][1] == 1).sum()))
        keep = [s for s in train_s if s != val_s]
        X_fit = np.concatenate([windows[s][0] for s in keep])
        y_fit = np.concatenate([windows[s][1] for s in keep])
        X_val, y_val = windows[val_s]

        model, norm = train_model(X_fit, y_fit, X_val, y_val,
                                   args.epochs, args.batch_size, args.lr,
                                   args.patience, device)

        te = DataLoader(TensorDataset(torch.from_numpy(norm.transform(X_te)),
                                      torch.from_numpy(y_te)),
                        batch_size=args.batch_size)
        p, t = predict(model, te, device)
        sens, spec, _ = sens_spec(t, p)
        cnn_true.extend(t)
        cnn_pred.extend(p)

        # Freeze-Index baseline: tune threshold on train windows, score on test.
        thr = best_fi_threshold(X_tr, y_tr)
        bp = freeze_index_predict(X_te, thr)
        bs, bspec, _ = sens_spec(y_te, bp)
        base_true.extend(y_te)
        base_pred.extend(bp)

        print(f"  fold {test_s}:  CNN sens={sens:.2f} spec={spec:.2f}   |   "
              f"FI(thr={thr:.1f}) sens={bs:.2f} spec={bspec:.2f}")

    print("\n" + "=" * 60)
    print("  POOLED LEAVE-ONE-SUBJECT-OUT RESULTS")
    print("=" * 60)
    sens_all, spec_all, cm = sens_spec(np.array(cnn_true), np.array(cnn_pred))
    bs, bsp, _ = sens_spec(np.array(base_true), np.array(base_pred))
    print(f"  CNN (FoGNet)      sensitivity={sens_all:.3f}  specificity={spec_all:.3f}")
    print(f"  Freeze-Index base sensitivity={bs:.3f}  specificity={bsp:.3f}")
    print("\n  CNN confusion (rows=true, cols=pred)  [0=no_freeze, 1=freeze]:")
    print("  " + str(cm).replace("\n", "\n  "))

    # ── Final model on ALL subjects, for deployment ──
    print("\n  Training final model on ALL subjects for deployment ...")
    val_s = max(subj_ids, key=lambda s: int((windows[s][1] == 1).sum()))
    keep = [s for s in subj_ids if s != val_s]
    model, norm = train_model(
        np.concatenate([windows[s][0] for s in keep]),
        np.concatenate([windows[s][1] for s in keep]),
        windows[val_s][0], windows[val_s][1],
        args.epochs, args.batch_size, args.lr, args.patience, device)

    torch.save({
        'model_state': model.state_dict(),
        'arch': model.arch,          # conv/fc widths + dropout → exact-shape reload
        'labels': LABELS,
        'window_size': WINDOW_SIZE,
        'window_hop': WINDOW_HOP,
        'fs': SAMPLE_RATE,
        'sensor': args.sensor,
    }, args.output_model)
    norm.save(args.output_norm)
    print(f"  Saved model      → {args.output_model}")
    print(f"  Saved normaliser → {args.output_norm}")


if __name__ == '__main__':
    main()
