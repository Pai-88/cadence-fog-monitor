"""End-to-end Colab pipeline: train + evaluate the FoG detector on Daphnet.

This is the **self-contained training-notebook source** — every definition the
Colab cells need lives here, with no dependency on the deployed ``fog`` package
(Colab cannot import it without friction). It is therefore the one place that
still pulls in torch; the on-device / laptop code is the torch-free
``fog`` package instead. The signal-processing functions below are deliberately
kept in lock-step with ``fog.dsp`` so the offline and deployed paths agree.

What ``run_all`` does, end to end:
  * load Daphnet, band-pass + window each run, label a window "freeze" when most
    of its in-experiment samples are freeze,
  * train the FoGNet 1-D CNN (class-weighted loss, sensitivity-based early stop),
  * leave-one-subject-out (LOSO) eval against the classical Freeze-Index baseline,
  * optional Optuna hyper-parameter search,
  * save every figure to ``fog_plots/``.

Run as a script for a fast smoke test::

    python fog_allinone.py /path/to/daphnet
"""
import glob
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from torch.utils.data import DataLoader, TensorDataset

sns.set_theme(style="whitegrid", context="talk")
PLOT_DIR = "fog_plots"

# ---- constants (must match the board's firmware AND the Daphnet dataset) ----
SAMPLE_RATE = 64
NUM_AXES    = 3
LABELS      = ['no_freeze', 'freeze']
WINDOW_SIZE = 256          # 4.0 s @ 64 Hz
WINDOW_HOP  = 128          # 2.0 s @ 64 Hz
LOCO_BAND   = (0.5, 3.0)
FREEZE_BAND = (3.0, 8.0)
TREMOR_BAND = (4.0, 6.0)
FI_THRESHOLD = 2.0

SENSOR_COLS = {'ankle': (1, 2, 3), 'thigh': (4, 5, 6), 'trunk': (7, 8, 9)}
ANNOT_COL   = 10           # 0=not-in-experiment, 1=no-freeze, 2=freeze
FREEZE_FRACTION = 0.5

# Temporal debounce / hysteresis — FIXED to match the deployed detector
# (fi_detector.py + firmware), NOT tuned per fold. The garment asserts "freeze"
# only after ONSET consecutive positive windows and releases only after OFFSET
# consecutive clear ones. Reporting the debounced stream shows what the device
# actually emits — isolated false-alarm windows are suppressed.
ONSET_WINDOWS  = 2
OFFSET_WINDOWS = 2

DEFAULT_HP = dict(lr=1e-3, weight_decay=1e-4, dropout=0.3,
                  c1=16, c2=32, c3=64, batch_size=64)


# ---- signal processing ------------------------------------------------------
def filter_offline(x, fs=SAMPLE_RATE):
    """Zero-phase band-pass 0.5-15 Hz: kills gravity DC, keeps freeze+tremor."""
    nyq = fs / 2
    b, a = signal.butter(4, [0.5 / nyq, 15.0 / nyq], btype='band')
    return signal.filtfilt(b, a, x.astype(np.float64), axis=0).astype(np.float32)


def _magnitude(window):
    """(T, C) accel window → (T,) magnitude with its mean (gravity/DC) removed."""
    mag = np.linalg.norm(np.asarray(window, dtype=np.float64), axis=1)
    return mag - mag.mean()


def band_power(sig1d, fs, band):
    """Power within ``band`` Hz of a 1-D signal, via the Welch PSD."""
    sig1d = np.asarray(sig1d, dtype=np.float64)
    nperseg = min(len(sig1d), 256)
    f, pxx = signal.welch(sig1d, fs=fs, nperseg=nperseg)
    lo, hi = band
    mask = (f >= lo) & (f < hi)
    if not mask.any():
        return 0.0
    df = float(f[1] - f[0]) if len(f) > 1 else 1.0
    return float(np.sum(pxx[mask]) * df)


def freeze_index(window, fs=SAMPLE_RATE):
    """Moore (2008) Freeze Index = power(3-8Hz)/power(0.5-3Hz) on accel mag."""
    mag = _magnitude(window)
    return band_power(mag, fs, FREEZE_BAND) / (band_power(mag, fs, LOCO_BAND) + 1e-9)


def tremor_power(window, fs=SAMPLE_RATE):
    """4-6 Hz band power on the accel magnitude — a rest-tremor severity proxy."""
    return band_power(_magnitude(window), fs, TREMOR_BAND)


def window_signal(x, window_size=WINDOW_SIZE, hop=WINDOW_HOP):
    """Slide a window across a (T, C) signal → (N, C, window_size)."""
    n_samples = x.shape[0]
    if n_samples < window_size:
        return np.empty((0, x.shape[1], window_size), dtype=np.float32)
    n = (n_samples - window_size) // hop + 1
    return np.stack([x[i*hop:i*hop+window_size].T for i in range(n)]).astype(np.float32)


class Normaliser:
    """Per-channel standardisation: fit on the training windows, apply to all."""

    def fit(self, X):
        self.mean = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std = (X.std(axis=(0, 2), keepdims=True) + 1e-6).astype(np.float32)

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)


class FoGNet(nn.Module):
    """~18k-param 1D CNN on (3, 256) accel windows. c1/c2/c3/fc/dropout are
    tunable by Optuna; the defaults reproduce the baseline net exactly."""
    def __init__(self, num_classes=len(LABELS), num_axes=NUM_AXES,
                 c1=16, c2=32, c3=64, fc=32, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(num_axes, c1, 15, padding=7); self.bn1 = nn.BatchNorm1d(c1)
        self.conv2 = nn.Conv1d(c1, c2, 9, padding=4);        self.bn2 = nn.BatchNorm1d(c2)
        self.conv3 = nn.Conv1d(c2, c3, 5, padding=2);        self.bn3 = nn.BatchNorm1d(c3)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(c3, fc); self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc, num_classes)

    def forward(self, x):
        x = F.max_pool1d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool1d(F.relu(self.bn2(self.conv2(x))), 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).squeeze(-1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


# ---- data -------------------------------------------------------------------
def load_daphnet(data_dir, sensor):
    """{subject: [(signal (T,3), annot (T,)), ...one per run...]}."""
    cols = SENSOR_COLS[sensor]
    files = sorted(glob.glob(os.path.join(data_dir, 'S*R*.txt')))
    if not files:
        raise FileNotFoundError(f"No Daphnet S*R*.txt in {data_dir}")
    subjects = {}
    for path in files:
        subj = re.search(r'(S\d+)R\d+', os.path.basename(path)).group(1)
        arr = np.loadtxt(path)
        subjects.setdefault(subj, []).append(
            (arr[:, list(cols)].astype(np.float32), arr[:, ANNOT_COL].astype(np.int64)))
    return subjects


def build_windows(runs):
    """Runs → (X (N,3,W) filtered, y (N,)). Window is 'freeze' if >50% of its
    in-experiment samples are freeze; all-not-in-experiment windows dropped."""
    Xs, ys = [], []
    for sig, annot in runs:
        if len(sig) < WINDOW_SIZE:
            continue
        win = window_signal(filter_offline(sig))
        ann = window_signal(annot[:, None].astype(np.float32))[:, 0, :]
        for i in range(len(win)):
            valid = ann[i][ann[i] > 0]
            if valid.size == 0:
                continue
            Xs.append(win[i]); ys.append(1 if (valid == 2).mean() > FREEZE_FRACTION else 0)
    if not Xs:
        return np.empty((0, 3, WINDOW_SIZE), np.float32), np.empty((0,), np.int64)
    return np.stack(Xs).astype(np.float32), np.array(ys, np.int64)


# ---- metrics + Freeze-Index baseline ----------------------------------------
def sens_spec(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else float('nan')
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    return sens, spec, cm


def debounce(pred, onset=ONSET_WINDOWS, offset=OFFSET_WINDOWS):
    """Latching hysteresis matching the deployed detector (fi_detector.py + firmware).

    Walks the 0/1 per-window stream and emits the device's actual freeze flag:
    asserts only after `onset` consecutive positive windows, releases only after
    `offset` consecutive clear ones. Isolated single-window false alarms are
    suppressed -- so the debounced stream is what the garment really buzzes on.

    INTEGRITY: this is a FIXED transform (constants match the firmware), not a
    tunable knob. MUST be applied PER held-out subject, on each LOSO fold's own
    contiguous time-ordered windows -- NEVER across the pooled concatenation of
    all subjects, or you would manufacture transitions at subject boundaries.
    """
    pred = np.asarray(pred, np.int64)
    out = np.zeros_like(pred)
    active = False
    pos = neg = 0
    for i, p in enumerate(pred):
        if p:
            pos += 1; neg = 0
        else:
            neg += 1; pos = 0
        if not active and pos >= onset:
            active = True
        elif active and neg >= offset:
            active = False
        out[i] = 1 if active else 0
    return out


def freeze_index_predict(X, threshold):
    return np.array([1 if freeze_index(X[i].T) > threshold else 0
                     for i in range(len(X))], np.int64)


def best_fi_threshold(X, y):
    fis = np.array([freeze_index(X[i].T) for i in range(len(X))])
    best_t, best_j = FI_THRESHOLD, -1.0
    for t in np.linspace(0.5, 6.0, 45):
        s, sp, _ = sens_spec(y, (fis > t).astype(np.int64))
        j = (0 if np.isnan(s) else s) + (0 if np.isnan(sp) else sp) - 1
        if j > best_j:
            best_j, best_t = j, t
    return best_t


# ---- training ---------------------------------------------------------------
def fit(X_tr, y_tr, hp, epochs, device, X_va=None, y_va=None,
        patience=8, record=False):
    norm = Normaliser(); norm.fit(X_tr)
    tr = DataLoader(TensorDataset(torch.from_numpy(norm.transform(X_tr)),
                                  torch.from_numpy(y_tr)),
                    batch_size=hp['batch_size'], shuffle=True)
    counts = np.bincount(y_tr, minlength=len(LABELS))
    weights = torch.tensor(len(y_tr) / (len(LABELS) * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    model = FoGNet(c1=hp['c1'], c2=hp['c2'], c3=hp['c3'], dropout=hp['dropout']).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {'epoch': [], 'train_loss': [], 'val_sens': [], 'val_spec': []}
    best, best_state, wait = -1.0, None, 0
    for ep in range(epochs):
        model.train(); running = 0.0
        for X, y in tr:
            X, y = X.to(device), y.to(device)
            opt.zero_grad(); loss = criterion(model(X), y)
            loss.backward(); opt.step(); running += loss.item() * len(X)
        sched.step()
        if X_va is not None and len(y_va):
            vp, _ = predict_proba(model, norm, X_va, device)
            s, sp, _ = sens_spec(y_va, vp)
            score = (0 if np.isnan(s) else s) + 0.3 * (0 if np.isnan(sp) else sp)
            if record:
                history['epoch'].append(ep)
                history['train_loss'].append(running / len(y_tr))
                history['val_sens'].append(0.0 if np.isnan(s) else s)
                history['val_spec'].append(0.0 if np.isnan(sp) else sp)
            if score > best:
                best, wait = score, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, norm, (history if record else {})


def predict_proba(model, norm, X, device):
    model.eval(); preds, probs = [], []
    loader = DataLoader(TensorDataset(torch.from_numpy(norm.transform(X))), batch_size=256)
    with torch.no_grad():
        for (xb,) in loader:
            p = torch.softmax(model(xb.to(device)), 1)[:, 1].cpu().numpy()
            probs.extend(p); preds.extend((p >= 0.5).astype(np.int64))
    return np.array(preds), np.array(probs)


def _val_subject(windows, pool):
    return max(pool, key=lambda s: int((windows[s][1] == 1).sum()))


def run_loso(windows, subj_ids, hp, epochs, device, max_folds=0):
    folds = subj_ids if max_folds == 0 else subj_ids[:max_folds]
    cnn_true, cnn_pred, cnn_prob, base_true, base_pred = [], [], [], [], []
    cnn_pred_db, cnn_subject = [], []
    per_subject, fi_rows = [], []
    for test_s in folds:
        train_s = [s for s in subj_ids if s != test_s]
        X_tr = np.concatenate([windows[s][0] for s in train_s])
        y_tr = np.concatenate([windows[s][1] for s in train_s])
        X_te, y_te = windows[test_s]
        val_s = _val_subject(windows, train_s)
        keep = [s for s in train_s if s != val_s]
        model, norm, _ = fit(np.concatenate([windows[s][0] for s in keep]),
                             np.concatenate([windows[s][1] for s in keep]),
                             hp, epochs, device, windows[val_s][0], windows[val_s][1])
        pred, prob = predict_proba(model, norm, X_te, device)
        s, sp, _ = sens_spec(y_te, pred)
        # debounce PER held-out subject (this fold's own time-ordered windows) --
        # never pooled, so no fake transitions are created at subject boundaries.
        pred_db = debounce(pred)
        sd, spd, _ = sens_spec(y_te, pred_db)
        cnn_true.extend(y_te); cnn_pred.extend(pred); cnn_prob.extend(prob)
        cnn_pred_db.extend(pred_db); cnn_subject.extend([test_s] * len(y_te))
        per_subject.append(dict(subject=test_s, method='CNN', sensitivity=s, specificity=sp))
        per_subject.append(dict(subject=test_s, method='CNN+debounce', sensitivity=sd, specificity=spd))
        thr = best_fi_threshold(X_tr, y_tr)
        bp = freeze_index_predict(X_te, thr)
        bs, bsp, _ = sens_spec(y_te, bp)
        base_true.extend(y_te); base_pred.extend(bp)
        per_subject.append(dict(subject=test_s, method='Freeze-Index', sensitivity=bs, specificity=bsp))
        for i in range(len(X_te)):
            fi_rows.append(dict(freeze_index=float(freeze_index(X_te[i].T)), label=LABELS[int(y_te[i])]))
        print(f"  fold {test_s}: CNN sens={s:.2f} spec={sp:.2f} "
              f"(debounced {sd:.2f}/{spd:.2f}) | FI sens={bs:.2f} spec={bsp:.2f}")
    return dict(cnn_true=np.array(cnn_true), cnn_pred=np.array(cnn_pred), cnn_prob=np.array(cnn_prob),
                cnn_pred_db=np.array(cnn_pred_db), cnn_subject=np.array(cnn_subject),
                base_true=np.array(base_true), base_pred=np.array(base_pred),
                per_subject=pd.DataFrame(per_subject), fi=pd.DataFrame(fi_rows))


def save_predictions(res, path=None):
    """Dump the pooled out-of-fold (OOF) CNN predictions so the operating point
    can be chosen later WITHOUT retraining: subject, true label, freeze
    probability, raw 0.5-threshold class, and the debounced (deployed) class.

    INTEGRITY: the ROC/PR curve built from y_prob is threshold-free and always a
    fair picture of the model. If you later want ONE operating threshold for the
    garment, pick it on a separate validation split -- do NOT scan this pooled
    OOF set for the threshold that maximises sens/spec and then quote that same
    set's score: that is tuning on the test data. Never choose per test fold.
    """
    if path is None:
        os.makedirs('fog_plots', exist_ok=True)
        path = os.path.join('fog_plots', 'loso_predictions.csv')
    df = pd.DataFrame(dict(
        subject=res['cnn_subject'],
        y_true=res['cnn_true'],
        y_prob=res['cnn_prob'],
        y_pred_raw=res['cnn_pred'],
        y_pred_debounced=res['cnn_pred_db']))
    df.to_csv(path, index=False)
    print(f"[saved] {path}  ({len(df)} OOF windows)")
    return path


# ---- Optuna -----------------------------------------------------------------
def run_optuna(windows, subj_ids, device, trials, opt_epochs):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    val_s = _val_subject(windows, subj_ids)
    train_s = [s for s in subj_ids if s != val_s]
    X_tr = np.concatenate([windows[s][0] for s in train_s])
    y_tr = np.concatenate([windows[s][1] for s in train_s])
    X_va, y_va = windows[val_s]

    def objective(trial):
        hp = dict(lr=trial.suggest_float('lr', 1e-4, 5e-3, log=True),
                  weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
                  dropout=trial.suggest_float('dropout', 0.1, 0.6),
                  c1=trial.suggest_categorical('c1', [8, 16, 24, 32]),
                  c2=trial.suggest_categorical('c2', [16, 32, 48, 64]),
                  c3=trial.suggest_categorical('c3', [32, 64, 96, 128]),
                  batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]))
        model, norm, _ = fit(X_tr, y_tr, hp, opt_epochs, device, X_va, y_va, patience=opt_epochs)
        pred, _ = predict_proba(model, norm, X_va, device)
        s, sp, _ = sens_spec(y_va, pred)
        return (0 if np.isnan(s) else s) + 0.3 * (0 if np.isnan(sp) else sp)

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    print(f"\n  Optuna best score={study.best_value:.3f}  params={study.best_params}")
    from optuna.visualization.matplotlib import (plot_optimization_history,
        plot_param_importances, plot_parallel_coordinate, plot_slice)
    # Optuna draws its axes small by default; enlarge per-plot so the
    # parallel-coordinate / slice tick labels stop colliding.
    sizes = {"opt_history": (10, 6), "opt_param_importance": (10, 6),
             "opt_parallel": (16, 7), "opt_slice": (18, 6)}
    for fn, name in [(plot_optimization_history, "opt_history"),
                     (plot_param_importances, "opt_param_importance"),
                     (plot_parallel_coordinate, "opt_parallel"),
                     (plot_slice, "opt_slice")]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax = fn(study)
            axes = ax.flatten() if isinstance(ax, np.ndarray) else [ax]
            fig = axes[0].figure
            fig.set_size_inches(*sizes[name])
            for a in axes:
                a.tick_params(labelsize=9)
            fig.suptitle(""); _save(fig, name)
        except Exception as e:
            print(f"    (skipped {name}: {e})")
    return study


# ---- plots ------------------------------------------------------------------
def _save(fig, name):
    # No tight_layout: figures are built with layout='constrained' (which also
    # makes room for suptitles + outside legends), and Optuna's own axes are
    # incompatible with tight_layout. bbox_inches='tight' keeps the PNG clean.
    os.makedirs(PLOT_DIR, exist_ok=True)
    fig.savefig(os.path.join(PLOT_DIR, f"{name}.png"), dpi=120, bbox_inches='tight')
    plt.show(); plt.close(fig)


def plot_training_curve(history):
    if not history.get('epoch'):
        return
    df = pd.DataFrame(history)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    sns.lineplot(data=df, x='epoch', y='train_loss', ax=axL, marker='o', color='#d1495b')
    axL.set(title='Training loss', xlabel='epoch', ylabel='cross-entropy')
    long = df.melt(id_vars='epoch', value_vars=['val_sens', 'val_spec'],
                   var_name='metric', value_name='value')
    long['metric'] = long['metric'].map({'val_sens': 'sensitivity', 'val_spec': 'specificity'})
    sns.lineplot(data=long, x='epoch', y='value', hue='metric', marker='o', ax=axR)
    axR.set(title='Validation metrics', xlabel='epoch', ylabel='', ylim=(0, 1.02))
    sns.move_legend(axR, loc='lower right', frameon=True)
    fig.suptitle('Did it actually learn?'); _save(fig, "training_curve")


def plot_confusion(true, pred, title, name):
    _, _, cm = sens_spec(true, pred)
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    annot = np.array([[f"{cm[i,j]}\n({cmn[i,j]*100:.0f}%)" for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout='constrained')
    sns.heatmap(cmn, annot=annot, fmt='', cmap='rocket_r', vmin=0, vmax=1,
                xticklabels=LABELS, yticklabels=LABELS, cbar_kws={'label': 'row %'},
                linewidths=1, linecolor='white', ax=ax)
    ax.set(title=title, xlabel='predicted', ylabel='true'); _save(fig, name)


def plot_roc_pr(true, prob):
    fpr, tpr, _ = roc_curve(true, prob); roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(true, prob); ap = average_precision_score(true, prob)
    prev = float(np.mean(true))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
    axL.plot(fpr, tpr, color='#0b6e4f', lw=2.5, label=f'AUC = {roc_auc:.3f}')
    axL.plot([0, 1], [0, 1], '--', color='grey', lw=1)
    axL.set(title='ROC', xlabel='1 − specificity', ylabel='sensitivity', xlim=(0, 1), ylim=(0, 1.02))
    axL.legend(loc='lower right')
    axR.plot(rec, prec, color='#d1495b', lw=2.5, label=f'AP = {ap:.3f}')
    axR.axhline(prev, ls='--', color='grey', lw=1, label=f'chance = {prev:.2f}')
    axR.set(title='Precision–Recall', xlabel='recall (sensitivity)', ylabel='precision',
            xlim=(0, 1), ylim=(0, 1.02)); axR.legend(loc='lower left')
    fig.suptitle('Ranking quality, threshold-free'); _save(fig, "roc_pr")


def plot_per_subject(per_subject):
    fig, (axS, axP) = plt.subplots(1, 2, figsize=(16, 6), sharey=True,
                                   layout='constrained')
    sns.barplot(data=per_subject, x='subject', y='sensitivity', hue='method', ax=axS)
    axS.set(title='Sensitivity per held-out subject', ylim=(0, 1.05))
    sns.barplot(data=per_subject, x='subject', y='specificity', hue='method', ax=axP)
    axP.set(title='Specificity per held-out subject', ylabel='', ylim=(0, 1.05))
    # One shared legend below both panels, outside the bars.
    h, l = axS.get_legend_handles_labels()
    for a in (axS, axP):
        if a.legend_:
            a.legend_.remove()
    fig.legend(h, l, title='method', loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Per-patient generalisation (LOSO)'); _save(fig, "per_subject")


def plot_cnn_vs_fi(res):
    s, sp, _ = sens_spec(res['cnn_true'], res['cnn_pred'])
    sd, spd, _ = sens_spec(res['cnn_true'], res['cnn_pred_db'])
    bs, bsp, _ = sens_spec(res['base_true'], res['base_pred'])
    df = pd.DataFrame([dict(method='CNN', metric='sensitivity', value=s),
                       dict(method='CNN', metric='specificity', value=sp),
                       dict(method='CNN+debounce', metric='sensitivity', value=sd),
                       dict(method='CNN+debounce', metric='specificity', value=spd),
                       dict(method='Freeze-Index', metric='sensitivity', value=bs),
                       dict(method='Freeze-Index', metric='specificity', value=bsp)])
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    sns.barplot(data=df, x='metric', y='value', hue='method', ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', padding=3)
    ax.set(title='CNN vs. debounced vs. Freeze-Index (pooled LOSO)', xlabel='', ylabel='', ylim=(0, 1.15))
    sns.move_legend(ax, loc='upper center', bbox_to_anchor=(0.5, -0.06),
                    ncol=3, title='method', frameon=False)
    _save(fig, "cnn_vs_fi")


def plot_fi_distribution(fi):
    if fi.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5), layout='constrained')
    sns.violinplot(data=fi, x='label', y='freeze_index', hue='label', order=LABELS,
                   palette={'no_freeze': '#5b8def', 'freeze': '#d1495b'},
                   cut=0, inner='quartile', legend=False, ax=ax)
    ax.axhline(2.0, ls='--', color='black', lw=1.2, label='FI = 2.0 (Moore 2008)')
    ax.set_yscale('log')
    ax.set(title='Why the Freeze-Index alone struggles', xlabel='true class',
           ylabel='Freeze Index  (log scale)'); ax.legend(loc='upper left')
    _save(fig, "fi_distribution")


# ---- the one entry point ----------------------------------------------------
def run_all(data_dir, sensor='ankle', epochs=25, folds=0,
            trials=25, opt_epochs=10, run_optuna_flag=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device} | sensor: {sensor} | plots → {PLOT_DIR}/")
    subjects = load_daphnet(data_dir, sensor)
    windows = {s: build_windows(r) for s, r in subjects.items()}
    windows = {s: (X, y) for s, (X, y) in windows.items() if len(y) > 0}
    subj_ids = sorted(windows)
    for s in subj_ids:
        y = windows[s][1]
        print(f"  {s}: {len(y):4d} windows | no_freeze={int((y==0).sum())} freeze={int((y==1).sum())}")

    print("\n[1/4] training curve (default hyper-parameters) ...")
    val_s = _val_subject(windows, subj_ids)
    keep = [s for s in subj_ids if s != val_s]
    _, _, history = fit(np.concatenate([windows[s][0] for s in keep]),
                        np.concatenate([windows[s][1] for s in keep]),
                        DEFAULT_HP, epochs, device,
                        windows[val_s][0], windows[val_s][1], record=True)
    plot_training_curve(history)

    print("\n[2/4] leave-one-subject-out evaluation ...")
    res = run_loso(windows, subj_ids, DEFAULT_HP, epochs, device, folds)
    s, sp, cm = sens_spec(res['cnn_true'], res['cnn_pred'])
    sd, spd, _ = sens_spec(res['cnn_true'], res['cnn_pred_db'])
    bs, bsp, _ = sens_spec(res['base_true'], res['base_pred'])
    print(f"\n  POOLED: CNN sens={s:.3f} spec={sp:.3f} | FI sens={bs:.3f} spec={bsp:.3f}")
    print(f"  POOLED debounced (deployed-style, onset={ONSET_WINDOWS}/offset={OFFSET_WINDOWS}): "
          f"sens={sd:.3f} spec={spd:.3f}  <- what the garment actually emits")
    save_predictions(res)
    plot_confusion(res['cnn_true'], res['cnn_pred'], 'CNN — pooled LOSO confusion', 'confusion_cnn')
    plot_confusion(res['cnn_true'], res['cnn_pred_db'],
                   'CNN + debounce — what the garment emits (pooled LOSO)', 'confusion_cnn_debounce')
    plot_confusion(res['base_true'], res['base_pred'], 'Freeze-Index — pooled LOSO confusion', 'confusion_fi')
    plot_roc_pr(res['cnn_true'], res['cnn_prob'])
    plot_per_subject(res['per_subject'])
    plot_cnn_vs_fi(res)
    plot_fi_distribution(res['fi'])

    if run_optuna_flag:
        print(f"\n[3/4] Optuna search ({trials} trials) ...")
        run_optuna(windows, subj_ids, device, trials, opt_epochs)
    print("\n[4/4] done — figures saved in fog_plots/ and shown above.")
    return res


if __name__ == '__main__':
    import sys
    run_all(sys.argv[1], sensor='ankle', epochs=4, folds=2, trials=4, opt_epochs=3)
