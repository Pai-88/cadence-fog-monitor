"""
Interpret the freeze-of-gait training — plots + hyper-parameter search.

This sits ON TOP of the working PyTorch pipeline (train_fog.py / fog_common.py).
It answers three questions a marker will ask about the Accuracy Worksheet:

  1. "Did it actually learn?"     → training curves (loss + val sens/spec).
  2. "How good is it, honestly?"  → LOSO confusion matrix, ROC/PR curves,
                                     per-subject bars, and CNN-vs-FreezeIndex bars.
  3. "Did you tune it, or guess?" → an Optuna study (TPE) over width / dropout /
                                     learning-rate, with its own diagnostic plots.

Everything is saved as a PNG in  fog_plots/  AND shown inline (plt.show), so it
works the same in a terminal and in a Colab cell run with  %run fog_analysis.py.

    python fog_analysis.py --data ~/Documents/daphnet/dataset --sensor ankle

Libraries: seaborn (diagnostic plots), optuna (HPO + its built-in matplotlib
viz), pandas (tidy frames seaborn wants), on top of the existing torch stack.
"""
from __future__ import annotations

import argparse
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader, TensorDataset

from fog.config import LABELS
from fog.dsp import freeze_index
from fog.metrics import best_fi_threshold, freeze_index_predict, sens_spec
from fog.model import FoGNet
from fog.normalize import Normaliser
from train_fog import build_windows, load_daphnet

sns.set_theme(style="whitegrid", context="talk")
PLOT_DIR = "fog_plots"

# The architecture/optimiser defaults that reproduce train_fog.py's model.
DEFAULT_HP = dict(lr=1e-3, weight_decay=1e-4, dropout=0.3,
                  c1=16, c2=32, c3=64, batch_size=64)


# ============================================================
#  A training routine that (a) takes hyper-parameters and
#  (b) records its own history — so we can both tune AND plot.
# ============================================================
def fit(X_tr, y_tr, hp, epochs, device, X_va=None, y_va=None,
        patience=8, record=False):
    """Train one FoGNet with the given hyper-parameter dict.

    Returns (model, normaliser, history). history is {} unless record=True, in
    which case it holds per-epoch train loss + val sensitivity/specificity.
    Early-stops on val score (sens + 0.3*spec) when a validation set is given.
    """
    norm = Normaliser()
    norm.fit(X_tr)
    tr = DataLoader(TensorDataset(torch.from_numpy(norm.transform(X_tr)),
                                  torch.from_numpy(y_tr)),
                    batch_size=hp['batch_size'], shuffle=True)

    counts = np.bincount(y_tr, minlength=len(LABELS))
    weights = torch.tensor(len(y_tr) / (len(LABELS) * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    model = FoGNet(c1=hp['c1'], c2=hp['c2'], c3=hp['c3'],
                   dropout=hp['dropout']).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'],
                                  weight_decay=hp['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {'epoch': [], 'train_loss': [], 'val_sens': [], 'val_spec': []}
    best_score, best_state, wait = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for X, y in tr:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(X)
        scheduler.step()

        if X_va is not None and len(y_va):
            vp, _ = predict_proba(model, norm, X_va, device)
            sens, spec, _ = sens_spec(y_va, vp)
            score = ((0 if np.isnan(sens) else sens)
                     + 0.3 * (0 if np.isnan(spec) else spec))
            if record:
                history['epoch'].append(epoch)
                history['train_loss'].append(running / len(y_tr))
                history['val_sens'].append(0.0 if np.isnan(sens) else sens)
                history['val_spec'].append(0.0 if np.isnan(spec) else spec)
            if score > best_score:
                best_score, wait = score, 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, norm, (history if record else {})


def predict_proba(model, norm, X, device):
    """Return (pred (N,) argmax, prob_freeze (N,) softmax P(freeze))."""
    model.eval()
    preds, probs = [], []
    loader = DataLoader(TensorDataset(torch.from_numpy(norm.transform(X))),
                        batch_size=256)
    with torch.no_grad():
        for (xb,) in loader:
            logits = model(xb.to(device))
            p = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            probs.extend(p)
            preds.extend((p >= 0.5).astype(np.int64))
    return np.array(preds), np.array(probs)


# ============================================================
#  LOSO sweep that also collects everything the plots need
# ============================================================
def _val_subject(windows, pool):
    """The pool subject with the most freeze windows — best early-stop monitor."""
    return max(pool, key=lambda s: int((windows[s][1] == 1).sum()))


def run_loso(windows, subj_ids, hp, epochs, device, max_folds=0):
    """Full (or truncated) LOSO. Returns a dict of everything the plots consume."""
    folds = subj_ids if max_folds == 0 else subj_ids[:max_folds]
    cnn_true, cnn_pred, cnn_prob = [], [], []
    base_true, base_pred = [], []
    per_subject = []          # one row per (subject, method)
    fi_rows = []              # FI value + true label, for the distribution plot

    for test_s in folds:
        train_s = [s for s in subj_ids if s != test_s]
        X_tr = np.concatenate([windows[s][0] for s in train_s])
        y_tr = np.concatenate([windows[s][1] for s in train_s])
        X_te, y_te = windows[test_s]

        val_s = _val_subject(windows, train_s)
        keep = [s for s in train_s if s != val_s]
        model, norm, _ = fit(np.concatenate([windows[s][0] for s in keep]),
                             np.concatenate([windows[s][1] for s in keep]),
                             hp, epochs, device,
                             windows[val_s][0], windows[val_s][1])

        pred, prob = predict_proba(model, norm, X_te, device)
        sens, spec, _ = sens_spec(y_te, pred)
        cnn_true.extend(y_te)
        cnn_pred.extend(pred)
        cnn_prob.extend(prob)
        per_subject.append(dict(subject=test_s, method='CNN',
                                sensitivity=sens, specificity=spec))

        thr = best_fi_threshold(X_tr, y_tr)
        bp = freeze_index_predict(X_te, thr)
        bs, bspec, _ = sens_spec(y_te, bp)
        base_true.extend(y_te)
        base_pred.extend(bp)
        per_subject.append(dict(subject=test_s, method='Freeze-Index',
                                sensitivity=bs, specificity=bspec))

        for i in range(len(X_te)):
            fi_rows.append(dict(freeze_index=float(freeze_index(X_te[i].T)),
                                label=LABELS[int(y_te[i])]))

        print(f"  fold {test_s}: CNN sens={sens:.2f} spec={spec:.2f} | "
              f"FI sens={bs:.2f} spec={bspec:.2f}")

    return dict(
        cnn_true=np.array(cnn_true), cnn_pred=np.array(cnn_pred),
        cnn_prob=np.array(cnn_prob),
        base_true=np.array(base_true), base_pred=np.array(base_pred),
        per_subject=pd.DataFrame(per_subject),
        fi=pd.DataFrame(fi_rows),
    )


# ============================================================
#  OPTUNA — search width / dropout / lr, maximise val sens+0.3*spec
# ============================================================
def run_optuna(windows, subj_ids, device, trials, opt_epochs):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # One fixed train/val split (no LOSO inside the search — that would be far
    # too slow). Val = the most-freeze subject; train = everyone else.
    val_s = _val_subject(windows, subj_ids)
    train_s = [s for s in subj_ids if s != val_s]
    X_tr = np.concatenate([windows[s][0] for s in train_s])
    y_tr = np.concatenate([windows[s][1] for s in train_s])
    X_va, y_va = windows[val_s]

    def objective(trial):
        hp = dict(
            lr=trial.suggest_float('lr', 1e-4, 5e-3, log=True),
            weight_decay=trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True),
            dropout=trial.suggest_float('dropout', 0.1, 0.6),
            c1=trial.suggest_categorical('c1', [8, 16, 24, 32]),
            c2=trial.suggest_categorical('c2', [16, 32, 48, 64]),
            c3=trial.suggest_categorical('c3', [32, 64, 96, 128]),
            batch_size=trial.suggest_categorical('batch_size', [32, 64, 128]),
        )
        model, norm, _ = fit(X_tr, y_tr, hp, opt_epochs, device, X_va, y_va,
                             patience=opt_epochs)
        pred, _ = predict_proba(model, norm, X_va, device)
        sens, spec, _ = sens_spec(y_va, pred)
        return ((0 if np.isnan(sens) else sens)
                + 0.3 * (0 if np.isnan(spec) else spec))

    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    print(f"\n  Optuna best score={study.best_value:.3f}  params={study.best_params}")

    # Optuna's own matplotlib visualisations. Each is wrapped: some need >1
    # trial / >1 completed param to render, so a tiny smoke run won't crash.
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_slice,
    )
    # Optuna draws its axes small by default; enlarge per-plot so the
    # parallel-coordinate / slice tick labels stop colliding.
    sizes = {"opt_history": (10, 6), "opt_param_importance": (10, 6),
             "opt_parallel": (16, 7), "opt_slice": (18, 6)}
    for fn, name in [(plot_optimization_history, "opt_history"),
                     (plot_param_importances,    "opt_param_importance"),
                     (plot_parallel_coordinate,  "opt_parallel"),
                     (plot_slice,                "opt_slice")]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ax = fn(study)
            axes = ax.flatten() if isinstance(ax, np.ndarray) else [ax]
            fig = axes[0].figure
            fig.set_size_inches(*sizes[name])
            for a in axes:
                a.tick_params(labelsize=9)
            fig.suptitle("")
            _save(fig, name)
        except Exception as e:               # noqa: BLE001 — viz is best-effort
            print(f"    (skipped {name}: {e})")
    return study


# ============================================================
#  SEABORN DIAGNOSTIC PLOTS
# ============================================================
def _save(fig, name):
    # No tight_layout: every figure is built with layout='constrained' (which
    # also reserves room for suptitles + outside legends), and Optuna's own axes
    # are incompatible with tight_layout. bbox_inches='tight' trims the PNG
    # margins. Saved to PLOT_DIR *and* shown inline, so it behaves the same in a
    # terminal and in a Colab cell (%run fog_analysis.py).
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, f"{name}.png")
    fig.savefig(path, dpi=120, bbox_inches='tight')
    print(f"  saved {path}")
    plt.show()
    plt.close(fig)


def plot_training_curve(history):
    if not history.get('epoch'):
        print("  (no training history to plot)")
        return
    df = pd.DataFrame(history)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    sns.lineplot(data=df, x='epoch', y='train_loss', ax=axL,
                 marker='o', color='#d1495b')
    axL.set(title='Training loss', xlabel='epoch', ylabel='cross-entropy')

    long = df.melt(id_vars='epoch', value_vars=['val_sens', 'val_spec'],
                   var_name='metric', value_name='value')
    long['metric'] = long['metric'].map({'val_sens': 'sensitivity',
                                         'val_spec': 'specificity'})
    sns.lineplot(data=long, x='epoch', y='value', hue='metric',
                 marker='o', ax=axR)
    axR.set(title='Validation metrics', xlabel='epoch', ylabel='', ylim=(0, 1.02))
    sns.move_legend(axR, loc='lower right', frameon=True)
    fig.suptitle('Did it actually learn?')
    _save(fig, "training_curve")


def plot_confusion(true, pred, title, name):
    _, _, cm = sens_spec(true, pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    annot = np.array([[f"{cm[i, j]}\n({cm_norm[i, j]*100:.0f}%)"
                       for j in range(2)] for i in range(2)])
    fig, ax = plt.subplots(figsize=(6.5, 5.5), layout='constrained')
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='rocket_r', vmin=0, vmax=1,
                xticklabels=LABELS, yticklabels=LABELS, cbar_kws={'label': 'row %'},
                ax=ax, linewidths=1, linecolor='white')
    ax.set(title=title, xlabel='predicted', ylabel='true')
    _save(fig, name)


def plot_roc_pr(true, prob):
    fpr, tpr, _ = roc_curve(true, prob)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(true, prob)
    ap = average_precision_score(true, prob)
    prevalence = float(np.mean(true))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
    axL.plot(fpr, tpr, color='#0b6e4f', lw=2.5, label=f'AUC = {roc_auc:.3f}')
    axL.plot([0, 1], [0, 1], '--', color='grey', lw=1)
    axL.set(title='ROC', xlabel='1 − specificity', ylabel='sensitivity',
            xlim=(0, 1), ylim=(0, 1.02))
    axL.legend(loc='lower right')

    axR.plot(rec, prec, color='#d1495b', lw=2.5, label=f'AP = {ap:.3f}')
    axR.axhline(prevalence, ls='--', color='grey', lw=1,
                label=f'chance = {prevalence:.2f}')
    axR.set(title='Precision–Recall', xlabel='recall (sensitivity)',
            ylabel='precision', xlim=(0, 1), ylim=(0, 1.02))
    axR.legend(loc='lower left')
    fig.suptitle('Ranking quality, threshold-free')
    _save(fig, "roc_pr")


def plot_per_subject(per_subject):
    fig, (axS, axP) = plt.subplots(1, 2, figsize=(16, 6), sharey=True,
                                   layout='constrained')
    sns.barplot(data=per_subject, x='subject', y='sensitivity', hue='method',
                ax=axS)
    axS.set(title='Sensitivity per held-out subject', ylim=(0, 1.05))
    sns.barplot(data=per_subject, x='subject', y='specificity', hue='method',
                ax=axP)
    axP.set(title='Specificity per held-out subject', ylabel='', ylim=(0, 1.05))
    # One shared legend below both panels, outside the bars.
    handles, labels = axS.get_legend_handles_labels()
    for a in (axS, axP):
        if a.legend_:
            a.legend_.remove()
    fig.legend(handles, labels, title='method', loc='outside lower center',
               ncol=2, frameon=False)
    fig.suptitle('Per-patient generalisation (LOSO)')
    _save(fig, "per_subject")


def plot_cnn_vs_fi(res):
    s, sp, _ = sens_spec(res['cnn_true'], res['cnn_pred'])
    bs, bsp, _ = sens_spec(res['base_true'], res['base_pred'])
    df = pd.DataFrame([
        dict(method='CNN', metric='sensitivity', value=s),
        dict(method='CNN', metric='specificity', value=sp),
        dict(method='Freeze-Index', metric='sensitivity', value=bs),
        dict(method='Freeze-Index', metric='specificity', value=bsp),
    ])
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    sns.barplot(data=df, x='metric', y='value', hue='method', ax=ax)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', padding=3)
    ax.set(title='CNN vs. Freeze-Index baseline (pooled LOSO)',
           xlabel='', ylabel='', ylim=(0, 1.15))
    sns.move_legend(ax, loc='upper center', bbox_to_anchor=(0.5, -0.06),
                    ncol=2, title='method', frameon=False)
    _save(fig, "cnn_vs_fi")


def plot_fi_distribution(fi):
    if fi.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5.5), layout='constrained')
    sns.violinplot(data=fi, x='label', y='freeze_index', hue='label',
                   order=LABELS, palette={'no_freeze': '#5b8def',
                                          'freeze': '#d1495b'},
                   cut=0, inner='quartile', legend=False, ax=ax)
    ax.axhline(2.0, ls='--', color='black', lw=1.2, label='FI = 2.0 (Moore 2008)')
    ax.set_yscale('log')
    ax.set(title='Why the Freeze-Index alone struggles',
           xlabel='true class', ylabel='Freeze Index  (log scale)')
    ax.legend(loc='upper left')
    _save(fig, "fi_distribution")


# ============================================================
#  MAIN
# ============================================================
def main(argv=None):
    # argv=None → CLI. In Colab pass a list, or run:  %run fog_analysis.py --data ...
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Daphnet dir (S01R01.txt ...)')
    parser.add_argument('--sensor', default='ankle',
                        choices=['ankle', 'thigh', 'trunk'])
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--folds', type=int, default=0, help='0 = full LOSO')
    parser.add_argument('--trials', type=int, default=25, help='Optuna trials')
    parser.add_argument('--opt-epochs', type=int, default=10,
                        help='epochs per Optuna trial (kept short)')
    parser.add_argument('--no-optuna', action='store_true')
    args = parser.parse_args(argv)

    matplotlib.rcParams['figure.max_open_warning'] = 0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"device: {device}  |  sensor: {args.sensor}  |  plots → {PLOT_DIR}/")

    subjects = load_daphnet(args.data, args.sensor)
    windows = {s: build_windows(runs) for s, runs in subjects.items()}
    windows = {s: (X, y) for s, (X, y) in windows.items() if len(y) > 0}
    subj_ids = sorted(windows)

    # 1) Train ONE model with the default HPs, recording its curve, to show
    #    "did it learn". Train on all-but-most-freeze, validate on most-freeze.
    print("\n[1/4] training curve (default hyper-parameters) ...")
    val_s = _val_subject(windows, subj_ids)
    keep = [s for s in subj_ids if s != val_s]
    _, _, history = fit(np.concatenate([windows[s][0] for s in keep]),
                        np.concatenate([windows[s][1] for s in keep]),
                        DEFAULT_HP, args.epochs, device,
                        windows[val_s][0], windows[val_s][1], record=True)
    plot_training_curve(history)

    # 2) Full LOSO → confusion, ROC/PR, per-subject, CNN-vs-FI, FI distribution.
    print("\n[2/4] leave-one-subject-out evaluation ...")
    res = run_loso(windows, subj_ids, DEFAULT_HP, args.epochs, device, args.folds)
    plot_confusion(res['cnn_true'], res['cnn_pred'],
                   'CNN — pooled LOSO confusion', 'confusion_cnn')
    plot_confusion(res['base_true'], res['base_pred'],
                   'Freeze-Index — pooled LOSO confusion', 'confusion_fi')
    plot_roc_pr(res['cnn_true'], res['cnn_prob'])
    plot_per_subject(res['per_subject'])
    plot_cnn_vs_fi(res)
    plot_fi_distribution(res['fi'])

    # 3) Optuna hyper-parameter search + its diagnostic plots.
    if not args.no_optuna:
        print(f"\n[3/4] Optuna search ({args.trials} trials) ...")
        run_optuna(windows, subj_ids, device, args.trials, args.opt_epochs)

    print("\n[4/4] done — every figure is in fog_plots/ and was shown inline.")


if __name__ == '__main__':
    main()
