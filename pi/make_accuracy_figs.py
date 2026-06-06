"""
Publication-quality figures for the ENGF0031 Accuracy Worksheet.

Two independent parts, both styled to look like LaTeX (Computer-Modern mathtext +
STIX serif, vector PDF, fonts embedded so the PDFs stay editable):

  Part A — from the pooled LOSO confusion counts you already have (no data needed):
      * confusion_matrix.pdf   counts + row-normalised %
      * metrics_panel.pdf      precision / recall / specificity / NPV / F1 /
                               balanced-accuracy / MCC / accuracy, with values
      * accuracy_caveat.pdf    why accuracy is the WRONG headline here — a
                               do-nothing "always no-freeze" model beats the CNN
                               on accuracy while catching zero freezes

  Part B — from REAL Daphnet (--data), on the interpretable 9-feature RandomForest
           (the engineered-feature model, not the raw-timestep CNN — the point is
           to show *which* clinical features drive a "freeze" call):
      * perm_importance.pdf    drop in balanced accuracy when each feature shuffled
      * partial_dependence.pdf P(freeze) vs. the top-3 features
      * shap_beeswarm.pdf      per-window SHAP contributions
      * shap_bar.pdf           mean |SHAP| per feature
      * results_sheet.pdf      one-page panel: confusion + metrics + importance + SHAP

Usage::

    python make_accuracy_figs.py                      # Part A only (uses the
                                                      # default counts below)
    python make_accuracy_figs.py --tp 607 --fp 1285 --fn 248 --tn 6842
    python make_accuracy_figs.py --data ~/Documents/daphnet/dataset   # + Part B

Every figure is written to  accuracy_figs/  as BOTH .pdf (for LaTeX) and .png.
A ready-to-paste  accuracy_figs/figures.tex  with \\includegraphics + captions is
emitted too.
"""
from __future__ import annotations

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ── LaTeX-style typography (no TeX install needed) ───────────────────────────
mpl.rcParams.update({
    "font.family": "STIXGeneral",     # CM-like serif with full glyph coverage
    "mathtext.fontset": "cm",         # Computer Modern for math
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "normal",
    "font.size": 10.5,
    "axes.edgecolor": "#2b2b2b",
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": 300,
    "figure.dpi": 140,
    "pdf.fonttype": 42,               # embed TrueType → editable vector PDF
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

OUT_DIR = "accuracy_figs"
# Palette — a calm clinical navy with a warm "freeze" accent.
NAVY, FREEZE, MUTE, GRID = "#1f3a63", "#b2182b", "#6b7785", "#d9dee5"


def _save(fig: plt.Figure, name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), bbox_inches="tight")
    print(f"  saved {OUT_DIR}/{name}.pdf  (+.png)")
    plt.close(fig)


# ============================================================================
#  Metrics from the confusion counts
# ============================================================================
def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float]:
    n = tp + fp + fn + tn
    p, r = tp / (tp + fp or 1), tp / (tp + fn or 1)         # precision, recall
    spec, npv = tn / (tn + fp or 1), tn / (tn + fn or 1)
    f1 = 2 * p * r / ((p + r) or 1)
    bal = (r + spec) / 2
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) or 1.0
    mcc = (tp * tn - fp * fn) / denom
    return {
        "Sensitivity (recall)": r, "Specificity": spec, "Precision (PPV)": p,
        "NPV": npv, "F1 score": f1, "Balanced acc.": bal, "MCC": mcc,
        "Accuracy": (tp + tn) / n,
    }


# ── Part A figures ───────────────────────────────────────────────────────────
def plot_confusion(tp: int, fp: int, fn: int, tn: int) -> None:
    cm = np.array([[tn, fp], [fn, tp]], dtype=float)        # rows=true, cols=pred
    row = cm / cm.sum(axis=1, keepdims=True)
    n = int(cm.sum())
    labels = ["no-freeze", "freeze"]

    fig, ax = plt.subplots(figsize=(5.0, 4.4), layout="constrained")
    im = ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            txt = "#ffffff" if row[i, j] > 0.55 else "#10243f"
            ax.text(j, i - 0.10, f"{int(cm[i, j]):,}", ha="center", va="center",
                    fontsize=16, color=txt)
            ax.text(j, i + 0.20, f"{row[i, j] * 100:.1f}%", ha="center",
                    va="center", fontsize=10.5, color=txt)
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Pooled LOSO confusion matrix — FoG-CNN", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("row-normalised rate", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)
    m = compute_metrics(tp, fp, fn, tn)
    ax.text(0.5, -0.30,
            f"$n={n:,}$ windows   ·   sensitivity $={m['Sensitivity (recall)']:.2f}$"
            f"   ·   specificity $={m['Specificity']:.2f}$",
            transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=MUTE)
    _save(fig, "confusion_matrix")


def plot_metrics_panel(tp: int, fp: int, fn: int, tn: int) -> None:
    m = compute_metrics(tp, fp, fn, tn)
    names = list(m)[::-1]                       # accuracy at top, sensitivity low
    vals = [m[k] for k in names]
    # MCC is on [-1, 1]; everything else on [0, 1]. Colour MCC differently and
    # annotate honestly rather than letting it look like a rate.
    colors = [FREEZE if k in ("Precision (PPV)", "F1 score") else
              "#7a7f87" if k == "MCC" else NAVY for k in names]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), layout="constrained")
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors, height=0.62, zorder=3)
    ax.axvline(0, color="#2b2b2b", lw=0.8)
    for yi, v in zip(y, vals, strict=True):
        ax.text(v + (0.012 if v >= 0 else -0.012), float(yi), f"{v:.3f}",
                va="center", ha="left" if v >= 0 else "right", fontsize=10)
    ax.set_yticks(y, names)
    ax.set_xlim(min(0, min(vals)) - 0.05, 1.08)
    ax.set_xlabel("score")
    ax.set_title("Operating-point metrics — FoG-CNN (pooled LOSO)", pad=10)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.text(1.0, -0.16,
            "Precision/F1 (red) are dragged down by class imbalance; "
            "MCC (grey) is on $[-1,1]$.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=MUTE)
    _save(fig, "metrics_panel")


def plot_accuracy_caveat(tp: int, fp: int, fn: int, tn: int) -> None:
    n = tp + fp + fn + tn
    cnn_acc = (tp + tn) / n
    cnn_sens = tp / (tp + fn or 1)
    # "always no-freeze": predicts 0 everywhere → TP=FP=0, FN=freezes, TN=no-freeze.
    triv_acc = (tn + fp) / n          # = prevalence of the majority class
    triv_sens = 0.0

    fig, ax = plt.subplots(figsize=(6.4, 4.4), layout="constrained")
    groups = ["Accuracy", "Sensitivity\n(freezes caught)"]
    x = np.arange(len(groups))
    w = 0.36
    b1 = ax.bar(x - w / 2, [cnn_acc, cnn_sens], w, label="FoG-CNN",
                color=NAVY, zorder=3)
    b2 = ax.bar(x + w / 2, [triv_acc, triv_sens], w,
                label='always "no-freeze" (does nothing)', color="#b8c0cc",
                zorder=3)
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.2f", padding=3, fontsize=10)
    ax.set_xticks(x, groups)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("score")
    ax.set_title("Why accuracy is the wrong headline for freezes", pad=10)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.text(0.5, -0.18,
            "The do-nothing model wins on accuracy yet catches 0 freezes — so we "
            "report sensitivity & specificity.",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color=MUTE)
    _save(fig, "accuracy_caveat")


# ============================================================================
#  Part B — interpretability on the 9-feature RandomForest (REAL data)
# ============================================================================
FEATURE_NAMES = ["freeze_power", "loco_power", "freeze_index", "tremor_power",
                 "total_power", "dom_freq", "mag_rms", "mag_range", "jerk_rms"]


def extract_features(X: np.ndarray) -> np.ndarray:
    """(N, 3, W) accel windows → (N, 9) interpretable feature matrix."""
    import scipy.signal as signal

    from fog.config import FREEZE_BAND, LOCO_BAND, SAMPLE_RATE, TREMOR_BAND
    from fog.dsp import band_power

    feats = []
    for i in range(len(X)):
        w = X[i].T
        mag = np.linalg.norm(w.astype(np.float64), axis=1)
        magc = mag - mag.mean()
        fp = band_power(magc, SAMPLE_RATE, FREEZE_BAND)
        lp = band_power(magc, SAMPLE_RATE, LOCO_BAND)
        tp = band_power(magc, SAMPLE_RATE, TREMOR_BAND)
        total = band_power(magc, SAMPLE_RATE, (0.5, 15.0))
        f, pxx = signal.welch(magc, fs=SAMPLE_RATE, nperseg=min(len(magc), 256))
        feats.append([fp, lp, fp / (lp + 1e-9), tp, total,
                      float(f[np.argmax(pxx)]),
                      float(np.sqrt(np.mean(magc ** 2))),
                      float(mag.max() - mag.min()),
                      float(np.sqrt(np.mean(np.diff(mag) ** 2)))])
    return np.array(feats, dtype=np.float64)


def _make_rf():
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=0, n_jobs=-1)


def _fit_feature_model(data_dir: str, sensor: str) -> dict:
    """Pooled leave-one-subject-out over the 9-feature RF — the *same* protocol
    the CNN was scored under, so the two are directly comparable.

    Every subject is held out once; predictions are pooled into one confusion
    matrix. Permutation importance and SHAP are computed out-of-fold on each
    held-out subject and then aggregated (mean over folds / concatenated), so
    nothing the importance/SHAP "see" was in that fold's training set. A single
    full-data model is fit only for the global partial-dependence panel.
    """
    from sklearn.inspection import permutation_importance

    from fog.metrics import sens_spec
    from train_fog import build_windows, load_daphnet

    subjects = load_daphnet(data_dir, sensor)
    feats: dict = {}
    for s, r in subjects.items():
        X, y = build_windows(r)
        if len(y) > 0:
            feats[s] = (extract_features(X), y)
    subj = sorted(feats)

    yt_all, yp_all, imp_folds, sv_chunks, svx_chunks = [], [], [], [], []
    rng = np.random.default_rng(0)
    for s in subj:
        Xte, yte = feats[s]
        train = [k for k in subj if k != s]
        Xtr = np.concatenate([feats[k][0] for k in train])
        ytr = np.concatenate([feats[k][1] for k in train])
        clf = _make_rf().fit(Xtr, ytr)
        yt_all.append(yte)
        yp_all.append(clf.predict(Xte))
        if (yte == 1).sum() == 0:           # no freezes here → can't score recall
            continue                        # (still counted in the pooled confusion)
        r = permutation_importance(clf, Xte, yte, n_repeats=20, random_state=0,
                                   scoring="balanced_accuracy")
        imp_folds.append(r.importances_mean)
        idx = rng.choice(len(Xte), size=min(60, len(Xte)), replace=False)
        sv_chunks.append(_shap_values_class1(clf, Xte[idx]))
        svx_chunks.append(Xte[idx])

    yt, yp = np.concatenate(yt_all), np.concatenate(yp_all)
    se, sp, _ = sens_spec(yt, yp)
    tp = int(((yt == 1) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())

    X_all = np.concatenate([feats[s][0] for s in subj])
    y_all = np.concatenate([feats[s][1] for s in subj])
    clf_full = _make_rf().fit(X_all, y_all)       # global model: PDP only

    print(f"  9-feature RF — pooled LOSO: sensitivity={se:.3f} specificity={sp:.3f} "
          f"(TP={tp} FP={fp} FN={fn} TN={tn}; {len(subj)} subjects, "
          f"{len(yt):,} windows)")
    return {
        "imp_mean": np.mean(imp_folds, axis=0),
        "imp_std": np.std(imp_folds, axis=0),
        "shap": np.concatenate(sv_chunks),
        "shapX": np.concatenate(svx_chunks),
        "clf_full": clf_full, "X_all": X_all,
        "sens": se, "spec": sp,
    }


def plot_perm_importance(imp_mean, imp_std, sens=None, spec=None) -> np.ndarray:
    order = imp_mean.argsort()
    fig, ax = plt.subplots(figsize=(7.0, 4.8), layout="constrained")
    y = np.arange(len(order))
    ax.barh(y, imp_mean[order], xerr=imp_std[order],
            color=NAVY, height=0.64, error_kw=dict(ecolor="#2b2b2b", capsize=3,
            lw=0.9), zorder=3)
    ax.set_yticks(y, [FEATURE_NAMES[i] for i in order])
    ax.set_xlabel("drop in balanced accuracy when feature is shuffled")
    ax.set_title("Permutation importance — 9-feature RF (pooled LOSO)", pad=10)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if sens is not None:
        ax.text(1.0, -0.15,
                f"out-of-fold, averaged over subjects · RF detector: "
                f"sensitivity $={sens:.2f}$, specificity $={spec:.2f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
                color=MUTE)
    _save(fig, "perm_importance")
    return order[::-1]                       # most-important first


def plot_partial_dependence(clf, X_tr, top_idx) -> None:
    from sklearn.inspection import PartialDependenceDisplay
    fig, axes = plt.subplots(1, len(top_idx), figsize=(4.3 * len(top_idx), 3.8),
                             layout="constrained")
    PartialDependenceDisplay.from_estimator(
        clf, X_tr, features=list(top_idx),
        feature_names=list(FEATURE_NAMES),
        ax=axes, line_kw={"color": FREEZE, "lw": 2.4})
    for a in np.atleast_1d(axes):
        a.grid(color=GRID, lw=0.7)
        a.set_axisbelow(True)
    fig.suptitle("Partial dependence — $P(\\mathrm{freeze})$ vs. top features "
                 "(9-feature RF, full cohort)", fontsize=13)
    _save(fig, "partial_dependence")


def _shap_values_class1(clf, Xs):
    import shap
    clf.n_jobs = 1                     # avoid loky/thread deadlock on macOS
    sv = shap.TreeExplainer(clf).shap_values(Xs, check_additivity=False)
    if isinstance(sv, list):
        return sv[1]
    if getattr(sv, "ndim", 2) == 3:
        return sv[:, :, 1]
    return sv


def plot_shap(sv1, Xs):
    import shap
    feat = list(FEATURE_NAMES)

    shap.summary_plot(sv1, Xs, feature_names=feat, show=False, plot_size=(7.2, 4.8),
                      color_bar_label="feature value")
    fig = plt.gcf()
    fig.suptitle('SHAP — per-window contribution to "freeze" (9-feature RF, '
                 'out-of-fold)', fontsize=13, y=1.02)
    _save(fig, "shap_beeswarm")

    shap.summary_plot(sv1, Xs, feature_names=feat, plot_type="bar", show=False,
                      plot_size=(7.2, 4.6), color=NAVY)
    fig = plt.gcf()
    fig.suptitle("SHAP — mean $|$impact$|$ per feature (9-feature RF)",
                 fontsize=13, y=1.02)
    _save(fig, "shap_bar")
    return sv1, Xs


# ── Combined one-page results sheet (panels we draw ourselves) ───────────────
def plot_results_sheet(tp, fp, fn, tn, perm=None, shap_pack=None,
                       rf_perf=None) -> None:
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(11.0, 8.4))
    gs = GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.26,
                  left=0.07, right=0.97, top=0.83, bottom=0.08)
    fig.suptitle("Freezing-of-gait detection — results summary (pooled LOSO)",
                 fontsize=15, y=0.975)
    sub = ("Top row: the deployed CNN detector.  Bottom row: a simpler 9-feature "
           "RandomForest, shown for interpretability.")
    if rf_perf is not None:
        cnn_se = tp / (tp + fn or 1)
        cnn_sp = tn / (tn + fp or 1)
        sub += (f"\nDetection sensitivity / specificity — CNN: {cnn_se:.2f} / "
                f"{cnn_sp:.2f}   vs   9-feature RF: {rf_perf[0]:.2f} / {rf_perf[1]:.2f}")
    fig.text(0.5, 0.945, sub, ha="center", va="top", fontsize=9.5, color=MUTE)

    # (a) confusion
    ax = fig.add_subplot(gs[0, 0])
    cm = np.array([[tn, fp], [fn, tp]], float)
    row = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(row, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            c = "#ffffff" if row[i, j] > 0.55 else "#10243f"
            ax.text(j, i, f"{int(cm[i, j]):,}\n{row[i, j]*100:.1f}%",
                    ha="center", va="center", color=c, fontsize=11)
    ax.set_xticks([0, 1], ["no-freeze", "freeze"])
    ax.set_yticks([0, 1], ["no-freeze", "freeze"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("(a) CNN — confusion matrix")

    # (b) metrics
    ax = fig.add_subplot(gs[0, 1])
    m = compute_metrics(tp, fp, fn, tn)
    names = list(m)[::-1]
    vals = [m[k] for k in names]
    y = np.arange(len(names))
    ax.barh(y, vals, color=NAVY, height=0.6, zorder=3)
    for yi, v in zip(y, vals, strict=True):
        ax.text(v + 0.012, float(yi), f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(y, names, fontsize=9)
    ax.set_xlim(min(0, min(vals)) - 0.05, 1.12)
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title("(b) CNN — operating-point metrics")

    # (c) permutation importance
    ax = fig.add_subplot(gs[1, 0])
    if perm is not None:
        order, imp_mean, imp_std = perm
        yy = np.arange(len(order))
        ax.barh(yy, imp_mean[order][::-1], xerr=imp_std[order][::-1], color=NAVY,
                height=0.62, error_kw=dict(ecolor="#2b2b2b", capsize=2, lw=0.8),
                zorder=3)
        ax.set_yticks(yy, [FEATURE_NAMES[i] for i in order[::-1]], fontsize=9)
        ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    else:
        ax.text(0.5, 0.5, "run with --data\nfor importance", ha="center",
                va="center", color=MUTE, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("(c) 9-feature RF — permutation importance")

    # (d) SHAP mean |impact|
    ax = fig.add_subplot(gs[1, 1])
    if shap_pack is not None:
        sv1, _ = shap_pack
        mean_abs = np.abs(sv1).mean(axis=0)
        o = mean_abs.argsort()
        yy = np.arange(len(o))
        ax.barh(yy, mean_abs[o], color="#3a6ea5", height=0.62, zorder=3)
        ax.set_yticks(yy, [FEATURE_NAMES[i] for i in o], fontsize=9)
        ax.set_xlabel("mean $|$SHAP$|$")
        ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    else:
        ax.text(0.5, 0.5, "run with --data\nfor SHAP", ha="center", va="center",
                color=MUTE, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title("(d) 9-feature RF — SHAP mean impact")
    _save(fig, "results_sheet")


# ── LaTeX include snippet ────────────────────────────────────────────────────
def write_tex(has_data: bool) -> None:
    figs = [
        ("confusion_matrix", "Pooled leave-one-subject-out confusion matrix for the "
         "FoG-CNN (counts and row-normalised rates)."),
        ("metrics_panel", "Operating-point metrics derived from the pooled LOSO "
         "confusion matrix."),
        ("accuracy_caveat", "Accuracy is misleading under class imbalance: a "
         "do-nothing classifier beats the CNN on accuracy while detecting no "
         "freezes."),
    ]
    if has_data:
        figs += [
            ("perm_importance", "Permutation importance of the nine engineered "
             "features for the interpretable RandomForest, aggregated out-of-fold "
             "over all pooled leave-one-subject-out folds (mean $\\pm$ s.d. of the "
             "drop in balanced accuracy when each feature is shuffled)."),
            ("partial_dependence", "Partial dependence of $P(\\mathrm{freeze})$ on "
             "the three most important features for the 9-feature RandomForest "
             "(fit on the full cohort)."),
            ("shap_beeswarm", "SHAP values for the 9-feature RandomForest: "
             "per-window, out-of-fold contribution of each feature to the "
             "``freeze'' prediction."),
            ("shap_bar", "Mean absolute SHAP value per feature for the 9-feature "
             "RandomForest."),
            ("results_sheet", "One-page results summary: the deployed CNN "
             "(confusion matrix and operating-point metrics, panels a--b) alongside "
             "the interpretable 9-feature RandomForest (permutation importance and "
             "SHAP, panels c--d), all under pooled leave-one-subject-out."),
        ]
    lines = ["% Auto-generated by make_accuracy_figs.py — paste into your report.",
             "% Figures are vector PDFs in accuracy_figs/.", ""]
    for name, cap in figs:
        lines += [
            r"\begin{figure}[htbp]", r"  \centering",
            rf"  \includegraphics[width=0.8\linewidth]{{accuracy_figs/{name}.pdf}}",
            rf"  \caption{{{cap}}}", rf"  \label{{fig:{name}}}",
            r"\end{figure}", ""]
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "figures.tex"), "w") as fh:
        fh.write("\n".join(lines))
    print(f"  saved {OUT_DIR}/figures.tex")


# ============================================================================
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tp", type=int, default=607, help="true positives  (freeze→freeze)")
    ap.add_argument("--fp", type=int, default=1285, help="false positives (no-freeze→freeze)")
    ap.add_argument("--fn", type=int, default=248, help="false negatives (freeze→no-freeze)")
    ap.add_argument("--tn", type=int, default=6842, help="true negatives  (no-freeze→no-freeze)")
    ap.add_argument("--data", default=None, help="Daphnet dir for SHAP/PDP/importance")
    ap.add_argument("--sensor", default="ankle", choices=["ankle", "thigh", "trunk"])
    args = ap.parse_args(argv)

    tp, fp, fn, tn = args.tp, args.fp, args.fn, args.tn
    print(f"Part A — confusion counts: TP={tp} FP={fp} FN={fn} TN={tn} "
          f"(n={tp+fp+fn+tn})")
    plot_confusion(tp, fp, fn, tn)
    plot_metrics_panel(tp, fp, fn, tn)
    plot_accuracy_caveat(tp, fp, fn, tn)

    perm = shap_pack = rf_perf = None
    if args.data:
        print(f"\nPart B — interpretability on real Daphnet ({args.data}) ...")
        res = _fit_feature_model(args.data, args.sensor)
        order = plot_perm_importance(res["imp_mean"], res["imp_std"],
                                     res["sens"], res["spec"])
        plot_partial_dependence(res["clf_full"], res["X_all"], order[:3])
        shap_pack = plot_shap(res["shap"], res["shapX"])
        perm = (order, res["imp_mean"], res["imp_std"])
        rf_perf = (res["sens"], res["spec"])
    else:
        print("\nPart B skipped (no --data). Pass --data <daphnet> for SHAP/PDP/importance.")

    plot_results_sheet(tp, fp, fn, tn, perm=perm, shap_pack=shap_pack, rf_perf=rf_perf)
    write_tex(has_data=bool(args.data))
    print(f"\nDone — figures in {OUT_DIR}/  (PDF for LaTeX, PNG for preview).")


if __name__ == "__main__":
    main()
