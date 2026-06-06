#!/usr/bin/env python3
"""make_cnn_figs.py -- CNN training-mechanics + extra diagnostic figures for the
ML-methods companion (worksheet/ml_methods_companion.tex).

Everything here is HONEST: the conceptual curves are exact mathematics (and are
labelled as schematics), the training curve is a REAL FoGNet run on Daphnet, and
the diagnostic curves are REAL out-of-fold probabilities from the interpretable
9-feature RandomForest under pooled leave-one-subject-out -- the same glass-box
model the companion already uses for the contour / SHAP / PI / PDP figures.

  Group A -- model & training mechanics
    bce_loss.pdf         binary cross-entropy / log-loss            (pure maths)
    sigmoid.pdf          logit -> probability squashing             (pure maths)
    gradient_descent.pdf gradient descent on a convex bowl + the
                         learning-rate effect                       (schematic)
    train_curve.pdf      REAL FoGNet training: cross-entropy loss +
                         validation sensitivity/specificity / epoch (Daphnet)

  Group B -- diagnostics on the 9-feature RandomForest (glass box), pooled LOSO,
             OUT-OF-FOLD probabilities (every window scored by a model that never
             trained on its subject)
    roc_curve.pdf        ROC + AUC, deployed operating point marked
    pr_curve.pdf         precision-recall + average precision, prevalence floor
    calibration.pdf      reliability diagram -- are the probabilities honest?
    score_hist.pdf       P(freeze) distribution: freeze vs no-freeze windows
    per_subject.pdf      per-patient sensitivity & specificity (LOSO spread)
    feature_corr.pdf     9-feature correlation heatmap (why Freeze-Index is redundant)
    feature_dist.pdf     class-conditional distributions of the top features

The CNN architecture itself is drawn in TikZ inside the .tex (vector, crisp).

Usage:
    MPLBACKEND=Agg .venv/bin/python make_cnn_figs.py --data /Users/paing/daphnet/dataset
"""
from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

# Reuse the LaTeX-style matplotlib rcParams + palette + helpers (importing this
# module sets the figure style so every panel matches the rest of the companion).
from make_accuracy_figs import (  # noqa: E402
    FEATURE_NAMES, FREEZE, GRID, MUTE, NAVY, _make_rf, _save, extract_features,
)

GREEN = "#1a7f5a"   # the "deployed boundary" green used in decision_contour
GOLD = "#8a6d00"    # operating-point / optimum accent
DATA_DEFAULT = "/Users/paing/daphnet/dataset"


# ============================================================================
#  GROUP A -- model & training mechanics
# ============================================================================
def plot_bce_loss() -> None:
    """Binary cross-entropy (log-loss) as a function of the predicted prob."""
    p = np.linspace(1e-3, 1 - 1e-3, 500)
    fig, ax = plt.subplots(figsize=(6.3, 4.1), layout="constrained")
    ax.plot(p, -np.log(p), color=FREEZE, lw=2.4,
            label=r"true label $y=1$ (freeze): $-\log\hat p$")
    ax.plot(p, -np.log(1 - p), color=NAVY, lw=2.4,
            label=r"true label $y=0$ (calm): $-\log(1-\hat p)$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 5)
    ax.set_xlabel(r"model's predicted freeze probability $\hat p$")
    ax.set_ylabel("loss for one window (nats)")
    ax.set_title("Binary cross-entropy: the price of being confidently wrong")
    ax.grid(True, color=GRID, lw=0.6)
    ax.legend(fontsize=9, loc="upper center")
    ax.annotate("confident & RIGHT\n$\\to$ loss near 0", xy=(0.92, -np.log(0.92)),
                xytext=(0.5, 0.27), fontsize=8.5, color=MUTE, ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.8))
    ax.annotate("confident & WRONG\n$\\to$ loss blows up", xy=(0.035, -np.log(0.035)),
                xytext=(0.17, 3.55), fontsize=8.5, color=MUTE, ha="left",
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.8))
    _save(fig, "bce_loss")


def plot_sigmoid() -> None:
    """Logit -> probability via the sigmoid (= the 2-class softmax)."""
    z = np.linspace(-6, 6, 400)
    s = 1.0 / (1.0 + np.exp(-z))
    fig, ax = plt.subplots(figsize=(6.3, 4.1), layout="constrained")
    ax.plot(z, s, color=NAVY, lw=2.6)
    ax.axhline(0.5, color=GRID, lw=0.9, ls=":")
    ax.axvline(0.0, color=GRID, lw=0.9, ls=":")
    ax.set_xlim(-6, 6); ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel(r"network score (logit) $z=z_{\mathrm{freeze}}-z_{\mathrm{calm}}$")
    ax.set_ylabel(r"$P(\mathrm{freeze})=\sigma(z)=1/(1+e^{-z})$")
    ax.set_title("Sigmoid / 2-class softmax: a raw score becomes a probability")
    ax.grid(True, color=GRID, lw=0.5)
    ax.annotate("saturates: gradient $\\approx 0$", xy=(4.4, s[np.argmin(np.abs(z - 4.4))]),
                xytext=(0.3, 0.62), fontsize=8.5, color=MUTE,
                arrowprops=dict(arrowstyle="->", color=MUTE, lw=0.8))
    ax.text(-5.7, 0.55, "decide 'freeze'\nonce $P\\geq t$", fontsize=8.5, color=MUTE,
            va="bottom")
    _save(fig, "sigmoid")


def plot_gradient_descent() -> None:
    """Gradient descent on a convex bowl, three learning rates."""
    def f(a, b):
        return 0.18 * a ** 2 + 0.9 * b ** 2

    def grad(a, b):
        return np.array([0.36 * a, 1.8 * b])

    A, B = np.meshgrid(np.linspace(-5, 5, 320), np.linspace(-3, 3, 320))
    Z = f(A, B)
    fig, ax = plt.subplots(figsize=(6.7, 4.4), layout="constrained")
    ax.contour(A, B, Z, levels=np.linspace(0.3, 16, 11), colors=[NAVY],
               linewidths=0.6, alpha=0.45)
    start = np.array([-4.4, 2.5])
    # (learning rate, colour, marker, n_steps, label) -- the too-large run gets
    # fewer steps so its zig-zag stays legible instead of piling up at the centre.
    runs = [(0.05, MUTE, "o", 16, "rate too small -- crawls"),
            (0.5, GREEN, "o", 16, "good rate -- converges"),
            (1.05, FREEZE, "o", 8, "rate too large -- zig-zags")]
    for lr, c, mk, nstep, lab in runs:
        p = start.copy(); xs, ys = [p[0]], [p[1]]
        for _ in range(nstep):
            p = p - lr * grad(*p); xs.append(p[0]); ys.append(p[1])
        ax.plot(xs, ys, "-", marker=mk, color=c, ms=3.2, lw=1.4, label=lab)
    ax.plot(0, 0, "*", color=GOLD, ms=17, label="minimum loss (best weights)",
            zorder=5)
    ax.set_xlim(-5, 5); ax.set_ylim(-3, 3)
    ax.set_xlabel("one weight (schematic)")
    ax.set_ylabel("another weight (schematic)")
    ax.set_title("Gradient descent: follow the slope downhill to the best weights")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    ax.text(0.5, -0.155,
            "Schematic of the optimiser (Cadence uses AdamW). The real loss "
            "surface is ~18000-dimensional; this is a 2-D cartoon.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color=MUTE)
    _save(fig, "gradient_descent")


def _train_curve_history(data_dir: str, sensor: str, epochs: int, seed: int):
    """Run ONE real FoGNet training and record loss + val metrics per epoch."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from fog.config import LABELS
    from fog.metrics import sens_spec
    from fog.model import FoGNet
    from fog.normalize import Normaliser
    from train_fog import build_windows, load_daphnet

    torch.manual_seed(seed); np.random.seed(seed)
    subjects = load_daphnet(data_dir, sensor)
    win = {s: build_windows(r) for s, r in subjects.items()}
    win = {s: (X, y) for s, (X, y) in win.items() if len(y) > 0 and (y == 1).sum() > 0}
    ids = sorted(win)
    # Hold out the subject with the most freeze windows as validation (so val
    # sensitivity is meaningful) -- exactly train_fog's rule.
    val_s = max(ids, key=lambda s: int((win[s][1] == 1).sum()))
    keep = [s for s in ids if s != val_s]
    Xtr = np.concatenate([win[s][0] for s in keep])
    ytr = np.concatenate([win[s][1] for s in keep])
    Xva, yva = win[val_s]

    norm = Normaliser(); norm.fit(Xtr)
    Xtr_t = torch.from_numpy(norm.transform(Xtr)); ytr_t = torch.from_numpy(ytr)
    Xva_t = torch.from_numpy(norm.transform(Xva))
    dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=64, shuffle=True)

    counts = np.bincount(ytr, minlength=len(LABELS))
    w = torch.tensor(len(ytr) / (len(LABELS) * np.maximum(counts, 1)),
                     dtype=torch.float32)
    model = FoGNet()
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    hist = {"epoch": [], "loss": [], "sens": [], "spec": []}
    for ep in range(epochs):
        model.train(); run, ntot = 0.0, 0
        for X, y in dl:
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward(); opt.step()
            run += float(loss.item()) * len(y); ntot += len(y)
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).argmax(1).numpy()
        s, sp, _ = sens_spec(yva, pv)
        hist["epoch"].append(ep + 1); hist["loss"].append(run / ntot)
        hist["sens"].append(0.0 if np.isnan(s) else float(s))
        hist["spec"].append(0.0 if np.isnan(sp) else float(sp))
        print(f"  epoch {ep + 1:2d}  loss={run / ntot:.4f}  "
              f"val_sens={s:.3f}  val_spec={sp:.3f}")
    nparam = sum(p.numel() for p in model.parameters())
    print(f"  [train_curve] val subject={val_s}  train windows={len(ytr)}  "
          f"freeze={int((ytr == 1).sum())}  params={nparam}")
    return hist, val_s, nparam


def plot_train_curve(hist: dict, val_s: str) -> None:
    e = hist["epoch"]
    fig, ax = plt.subplots(figsize=(7.0, 4.3), layout="constrained")
    l1, = ax.plot(e, hist["loss"], color=FREEZE, lw=2.3, marker="o", ms=3.2,
                  label="training loss")
    ax.set_xlabel("epoch  (one full pass over the training windows)")
    ax.set_ylabel("training loss (cross-entropy)", color=FREEZE)
    ax.tick_params(axis="y", labelcolor=FREEZE)
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_xlim(1, e[-1])

    ax2 = ax.twinx()
    l2, = ax2.plot(e, hist["sens"], color=NAVY, lw=2.0, marker="s", ms=3.0,
                   label="validation sensitivity")
    l3, = ax2.plot(e, hist["spec"], color=GREEN, lw=2.0, marker="^", ms=3.0,
                   ls="--", label="validation specificity")
    ax2.set_ylabel("validation metric", color=NAVY)
    ax2.set_ylim(0, 1.02); ax2.tick_params(axis="y", labelcolor=NAVY)
    # Mark the early-stop point: train_fog keeps the checkpoint that maximises
    # val sensitivity + 0.3*specificity (its exact selection score), NOT epoch 30.
    score = [s + 0.3 * sp for s, sp in zip(hist["sens"], hist["spec"])]
    best_i = int(np.argmax(score)); best_ep = hist["epoch"][best_i]
    ax.axvline(best_ep, color=GOLD, lw=1.6, alpha=0.9, zorder=1)
    ax2.plot(best_ep, hist["sens"][best_i], "*", color=GOLD, ms=16, zorder=6)
    # text sits in the clear mid-right band (sens has fallen, spec is high above);
    # arrow points back to the gold star.  No baked-in figure caption -- the
    # interpretation lives in the LaTeX \caption (and a fat one-line caption was
    # what previously widened the tight bbox and shoved the plot sideways).
    ax2.annotate("early stop:\nbest validation\n(this checkpoint ships)",
                 xy=(best_ep, hist["sens"][best_i]), xytext=(12.0, 0.66),
                 fontsize=8.0, color="#6b5200", va="center", ha="left",
                 arrowprops=dict(arrowstyle="->", color=GOLD, lw=1.2))
    ax.set_title("FoGNet learning on Daphnet ankle: loss falls as skill rises")
    # frameless horizontal legend BELOW the axes -> never overlaps the curves.
    ax2.legend(handles=[l1, l2, l3], loc="upper center",
               bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=9)
    _save(fig, "train_curve")


# ============================================================================
#  GROUP B -- diagnostics on the 9-feature RandomForest (REAL out-of-fold)
# ============================================================================
def _oof_rf(data_dir: str, sensor: str):
    """Pooled leave-one-subject-out OOF probabilities from the glass-box RF."""
    from fog.metrics import sens_spec  # noqa: F401  (kept for parity/debug)
    from train_fog import build_windows, load_daphnet

    subjects = load_daphnet(data_dir, sensor)
    feats: dict = {}
    for s, r in subjects.items():
        X, y = build_windows(r)
        if len(y) > 0:
            feats[s] = (extract_features(X), y)
    ids = sorted(feats)

    yt_all, prob_all, owner = [], [], []
    for s in ids:
        Xte, yte = feats[s]
        train = [k for k in ids if k != s]
        Xtr = np.concatenate([feats[k][0] for k in train])
        ytr = np.concatenate([feats[k][1] for k in train])
        clf = _make_rf().fit(Xtr, ytr)
        yt_all.append(yte)
        prob_all.append(clf.predict_proba(Xte)[:, 1])
        owner.append(np.array([s] * len(yte)))
    yt = np.concatenate(yt_all)
    prob = np.concatenate(prob_all)
    owner = np.concatenate(owner)
    X_all = np.concatenate([feats[s][0] for s in ids])
    y_all = np.concatenate([feats[s][1] for s in ids])
    print(f"  [oof] {len(ids)} subjects, {len(yt):,} windows, "
          f"prevalence={yt.mean():.3f}")
    return yt, prob, owner, ids, X_all, y_all


def _youden_threshold(yt: np.ndarray, prob: np.ndarray) -> float:
    grid = np.linspace(0.02, 0.98, 97)
    best_t, best_j = 0.5, -1.0
    for t in grid:
        pred = (prob >= t).astype(int)
        tp = int(((pred == 1) & (yt == 1)).sum()); fn = int(((pred == 0) & (yt == 1)).sum())
        tn = int(((pred == 0) & (yt == 0)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
        se = tp / (tp + fn) if (tp + fn) else 0.0
        sp = tn / (tn + fp) if (tn + fp) else 0.0
        if se + sp - 1 > best_j:
            best_j, best_t = se + sp - 1, t
    return float(best_t)


def plot_roc(yt, prob, t_star) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve
    fpr, tpr, thr = roc_curve(yt, prob)
    auc = roc_auc_score(yt, prob)
    # operating point at t_star
    pred = (prob >= t_star).astype(int)
    op_tpr = pred[yt == 1].mean(); op_fpr = pred[yt == 0].mean()
    fig, ax = plt.subplots(figsize=(5.6, 5.0), layout="constrained")
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.1, ls="--", label="chance (AUC 0.5)")
    ax.plot(fpr, tpr, color=NAVY, lw=2.4, label=f"RF glass-box (AUC = {auc:.2f})")
    ax.plot(op_fpr, op_tpr, "*", color=GOLD, ms=17, zorder=5,
            label=f"deployed point $t\\approx{t_star:.2f}$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("false-positive rate  (1 - specificity)")
    ax.set_ylabel("true-positive rate  (sensitivity)")
    ax.set_title("ROC -- sensitivity vs false alarms across every threshold")
    ax.grid(True, color=GRID, lw=0.5)
    ax.legend(fontsize=8.5, loc="lower right")
    _save(fig, "roc_curve")


def plot_pr(yt, prob, t_star) -> None:
    from sklearn.metrics import average_precision_score, precision_recall_curve
    prec, rec, thr = precision_recall_curve(yt, prob)
    ap = average_precision_score(yt, prob)
    prev = float(yt.mean())
    pred = (prob >= t_star).astype(int)
    op_rec = pred[yt == 1].mean()
    tp = int(((pred == 1) & (yt == 1)).sum()); fp = int(((pred == 1) & (yt == 0)).sum())
    op_prec = tp / (tp + fp) if (tp + fp) else 0.0
    fig, ax = plt.subplots(figsize=(5.6, 5.0), layout="constrained")
    ax.axhline(prev, color=GRID, lw=1.1, ls="--",
               label=f"prevalence floor ({prev * 100:.1f}%)")
    ax.plot(rec, prec, color=NAVY, lw=2.4, label=f"RF glass-box (AP = {ap:.2f})")
    ax.plot(op_rec, op_prec, "*", color=GOLD, ms=17, zorder=5,
            label=f"deployed point $t\\approx{t_star:.2f}$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("recall  (= sensitivity)")
    ax.set_ylabel("precision  (PPV)")
    ax.set_title("Precision-recall -- the honest view under 9.5% prevalence")
    ax.grid(True, color=GRID, lw=0.5)
    ax.legend(fontsize=8.5, loc="upper right")
    _save(fig, "pr_curve")


def plot_calibration(yt, prob) -> None:
    nb = 10
    edges = np.linspace(0, 1, nb + 1)
    idx = np.clip(np.digitize(prob, edges) - 1, 0, nb - 1)
    mean_pred, frac_pos, counts = [], [], []
    for b in range(nb):
        m = idx == b
        if m.sum() == 0:
            continue
        mean_pred.append(prob[m].mean()); frac_pos.append(yt[m].mean())
        counts.append(int(m.sum()))
    mean_pred = np.array(mean_pred); frac_pos = np.array(frac_pos)
    counts = np.array(counts)

    fig, (ax, axh) = plt.subplots(
        2, 1, figsize=(5.8, 5.7), height_ratios=[3, 1], layout="constrained",
        sharex=True)
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2, ls="--", label="perfect calibration")
    # Line ONLY through bins with enough windows to be trustworthy; sparse
    # high-prob bins (<20 windows) are shown as small dots, not joined, so the
    # noisy tail can't masquerade as a real miscalibration.
    dense = counts >= 20
    ax.plot(mean_pred[dense], frac_pos[dense], "-", color=NAVY, lw=2.0, zorder=2,
            label=r"RF glass-box (bins $\geq$ 20 windows)")
    sizes = 25 + 110 * np.sqrt(counts / counts.max())
    ax.scatter(mean_pred, frac_pos, s=sizes, color=NAVY, edgecolors="white",
               linewidths=0.6, zorder=3)
    ax.set_ylabel("observed freeze frequency")
    ax.set_title("Reliability diagram -- do the probabilities mean what they say?")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(True, color=GRID, lw=0.5); ax.legend(fontsize=8.5, loc="upper left")

    allcounts, _ = np.histogram(prob, bins=edges)
    axh.bar((edges[:-1] + edges[1:]) / 2, allcounts, width=0.9 / nb, color=NAVY,
            alpha=0.7)
    axh.set_yscale("log")
    axh.set_xlabel(r"predicted freeze probability $\hat p$")
    axh.set_ylabel("windows\n(log)")
    axh.grid(True, color=GRID, lw=0.5, axis="y")
    _save(fig, "calibration")


def plot_score_hist(yt, prob, t_star) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.3), layout="constrained")
    bins = np.linspace(0, 1, 41)
    ax.hist(prob[yt == 0], bins=bins, density=True, color=NAVY, alpha=0.55,
            label="no-freeze windows")
    ax.hist(prob[yt == 1], bins=bins, density=True, color=FREEZE, alpha=0.6,
            label="freeze windows")
    ax.axvline(t_star, color=GOLD, lw=2.0,
               label=f"deployed threshold $t\\approx{t_star:.2f}$")
    ax.set_xlim(0, 1)
    ax.set_xlabel(r"model's predicted freeze probability $\hat p$")
    ax.set_ylabel("density  (each class scaled to its own area)")
    ax.set_title("Score separation -- freezes are pushed right, calm gait left")
    ax.grid(True, color=GRID, lw=0.5)
    ax.legend(fontsize=8.5, loc="upper right")
    _save(fig, "score_hist")


def plot_per_subject(yt, prob, owner, ids, t_star) -> None:
    pred = (prob >= t_star).astype(int)
    rows = []
    for s in ids:
        m = owner == s
        ys, ps = yt[m], pred[m]
        npos = int((ys == 1).sum()); nneg = int((ys == 0).sum())
        se = (ps[ys == 1].mean() if npos else np.nan)
        sp = (1 - ps[ys == 0].mean() if nneg else np.nan)
        rows.append((s, se, sp, npos))
    rows.sort(key=lambda r: (-(r[3] > 0), r[0]))
    labels = [r[0] for r in rows]
    sens = [r[1] for r in rows]; spec = [r[2] for r in rows]
    x = np.arange(len(labels)); w = 0.4
    fig, ax = plt.subplots(figsize=(7.3, 4.3), layout="constrained")
    ax.bar(x - w / 2, np.nan_to_num(sens), w, color=FREEZE, label="sensitivity")
    ax.bar(x + w / 2, np.nan_to_num(spec), w, color=NAVY, label="specificity")
    for i, r in enumerate(rows):
        if np.isnan(r[1]):
            ax.text(i - w / 2, 0.36, "no\nfreeze\nwindows", ha="center",
                    va="center", fontsize=7.0, color=MUTE)
    ax.axhline(0.71, color=FREEZE, lw=1.0, ls=":", alpha=0.8)
    ax.axhline(0.84, color=NAVY, lw=1.0, ls=":", alpha=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.2); ax.set_ylabel("score on the held-out patient")
    ax.set_xlabel("held-out subject (leave-one-subject-out)")
    ax.set_title("Per-patient spread -- generalisation to a new wearer")
    ax.grid(True, color=GRID, lw=0.5, axis="y")
    ax.legend(fontsize=8.5, loc="upper center", ncol=2, framealpha=0.95)
    ax.text(0.5, -0.16,
            "Dotted lines: pooled sensitivity 0.71 / specificity 0.84. Subjects "
            "with no freeze windows can only score specificity.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8, color=MUTE)
    _save(fig, "per_subject")


def plot_feature_corr(X_all) -> None:
    C = np.corrcoef(X_all.T)
    n = len(FEATURE_NAMES)
    fig, ax = plt.subplots(figsize=(6.6, 5.8), layout="constrained")
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), FEATURE_NAMES, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(n), FEATURE_NAMES, fontsize=8.5)
    for i in range(n):
        for j in range(n):
            v = C[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.6,
                    color="white" if abs(v) > 0.55 else "#222222")
    ax.set_title("Feature correlation -- redundancy the model has to untangle")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Pearson correlation")
    _save(fig, "feature_corr")


def plot_feature_dist(X_all, y_all) -> None:
    # Box plots (quartiles) read the class shift far more clearly than overlaid
    # histograms do under this much class imbalance.
    show = ["jerk_rms", "total_power", "dom_freq"]
    log_axis = {"jerk_rms": True, "total_power": True, "dom_freq": False}
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.8), layout="constrained")
    for ax, name in zip(axes, show):
        k = FEATURE_NAMES.index(name)
        v0 = X_all[y_all == 0, k]; v1 = X_all[y_all == 1, k]
        if log_axis[name]:
            v0 = np.clip(v0, 1e-4, None); v1 = np.clip(v1, 1e-4, None)
            ax.set_yscale("log")
        bp = ax.boxplot([v0, v1], widths=0.55, showfliers=False,
                        patch_artist=True,
                        medianprops=dict(color="black", lw=1.5))
        for patch, c in zip(bp["boxes"], [NAVY, FREEZE]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        ax.set_xticks([1, 2], ["calm", "freeze"])
        ax.set_title(name, fontsize=10.5)
        ax.grid(True, color=GRID, lw=0.4, axis="y")
    axes[0].set_ylabel("feature value")
    fig.suptitle("What separates a freeze window from calm gait  (box = quartiles)",
                 fontsize=12)
    _save(fig, "feature_dist")


# ============================================================================
#  MAIN
# ============================================================================
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DATA_DEFAULT, help="Daphnet dataset dir")
    ap.add_argument("--sensor", default="ankle")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-train", action="store_true",
                    help="skip the (slower) real CNN training curve")
    args = ap.parse_args(argv)

    print("== Group A: training mechanics (conceptual) ==")
    plot_bce_loss()
    plot_sigmoid()
    plot_gradient_descent()

    if not args.skip_train:
        print("== Group A: REAL FoGNet training curve ==")
        hist, val_s, nparam = _train_curve_history(args.data, args.sensor,
                                                   args.epochs, args.seed)
        plot_train_curve(hist, val_s)

    print("== Group B: RF glass-box OOF diagnostics ==")
    yt, prob, owner, ids, X_all, y_all = _oof_rf(args.data, args.sensor)
    t_star = _youden_threshold(yt, prob)
    print(f"  [oof] Youden-optimal pooled threshold t* = {t_star:.3f}")
    plot_roc(yt, prob, t_star)
    plot_pr(yt, prob, t_star)
    plot_calibration(yt, prob)
    plot_score_hist(yt, prob, t_star)
    plot_per_subject(yt, prob, owner, ids, t_star)
    plot_feature_corr(X_all)
    plot_feature_dist(X_all, y_all)
    print("done.")


if __name__ == "__main__":
    main()
