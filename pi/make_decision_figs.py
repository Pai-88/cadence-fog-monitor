"""Contour / decision-surface figures for the ML-methods companion PDF.

Three REAL figures, all computed from the authorised Daphnet ankle data via the
interpretable 9-feature RandomForest (the same engineered-feature model that
make_accuracy_figs.py uses for SHAP / permutation importance), so they sit
honestly alongside those plots:

  decision_contour.pdf  -- HOW WE CATEGORISE.  P(freeze) over the (jerk_rms,
                           dom_freq) plane for a 2-feature RF, real windows
                           overlaid, with the 0.3 / 0.5 / 0.7 probability
                           contours drawn (moving that contour = choosing the
                           operating point).
  param_grid.pdf        -- HOW WE FIND THE BEST PARAMETERS (model complexity).
                           Subject-grouped 5-fold CV balanced accuracy over a
                           grid of (max_depth x min_samples_leaf); the star is
                           the cross-validated optimum.
  operating_point.pdf   -- HOW WE FIND THE BEST PARAMETER (decision threshold).
                           Sensitivity / specificity / Youden's J vs. the
                           probability threshold, from pooled leave-one-subject-
                           out out-of-fold scores; the star is max-J -- exactly
                           what colab/pick_threshold.py automates per subject.

Run from the pi/ directory so `fog` and `train_fog` import cleanly:

    MPLBACKEND=Agg .venv/bin/python make_decision_figs.py \
        --data /Users/paing/daphnet/dataset

Outputs go to accuracy_figs/ as both .pdf (LaTeX) and .png (preview), matching
make_accuracy_figs.py.  numpy + matplotlib + scikit-learn only.
"""
from __future__ import annotations

import argparse

import numpy as np

# Importing make_accuracy_figs also applies its LaTeX-style matplotlib rcParams,
# so every figure here matches the rest of the figure set.
import matplotlib.pyplot as plt
from make_accuracy_figs import (
    FEATURE_NAMES, FREEZE, GRID, MUTE, NAVY, _save, extract_features,
)

JERK, DOMF = FEATURE_NAMES.index("jerk_rms"), FEATURE_NAMES.index("dom_freq")


# ── Load real Daphnet and build the pooled 9-feature matrix ──────────────────
def load_features(data_dir: str, sensor: str):
    """{subject: (feat (n,9), y (n,))} plus pooled X, y, integer group ids."""
    from train_fog import build_windows, load_daphnet

    subjects = load_daphnet(data_dir, sensor)
    feats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s, runs in subjects.items():
        X, y = build_windows(runs)
        if len(y):
            feats[s] = (extract_features(X), y)
    subj = sorted(feats)
    X = np.concatenate([feats[s][0] for s in subj])
    y = np.concatenate([feats[s][1] for s in subj])
    g = np.concatenate([np.full(len(feats[s][1]), i) for i, s in enumerate(subj)])
    print(f"  pooled {len(y):,} windows over {len(subj)} subjects "
          f"({int(y.sum()):,} freeze, prevalence {y.mean()*100:.1f}%)")
    return feats, subj, X, y, g


# ── 1.  Decision-region contour — how we categorise ──────────────────────────
def plot_decision_contour(X, y):
    from sklearn.ensemble import RandomForestClassifier

    xi, yi = X[:, JERK], X[:, DOMF]
    # Shallow 2-feature RF with the NATURAL class prior => predict_proba is the
    # honest posterior P(freeze): low across most of the plane (freezes are rare,
    # 9.5%), warm only where freezes concentrate. This panel is a visualisation;
    # the deployed models use all features / the raw signal.
    clf = RandomForestClassifier(n_estimators=400, max_depth=6,
                                 min_samples_leaf=30,
                                 random_state=0, n_jobs=-1).fit(np.c_[xi, yi], y)

    xmax = float(np.percentile(xi, 98))
    ymax = float(min(np.percentile(yi, 97), 9.0))
    gx = np.linspace(0, xmax, 300)
    gy = np.linspace(0, ymax, 300)
    XX, YY = np.meshgrid(gx, gy)
    P = clf.predict_proba(np.c_[XX.ravel(), YY.ravel()])[:, 1].reshape(XX.shape)
    # Gentle smoothing of the blocky RF posterior, purely for legibility: this is
    # a teaching visualisation, so the contours should read as smooth curves
    # rather than piecewise-constant steps. The qualitative story (a warm freeze
    # pocket at high jerk / low dom_freq) is unchanged.
    from scipy.ndimage import gaussian_filter
    P = gaussian_filter(P, sigma=4.0)

    # Finer levels at the low end so the rare-class pocket is visible.
    levels = [0, .05, .1, .15, .2, .25, .3, .4, .5, .7, 1.0]
    fig, ax = plt.subplots(figsize=(7.4, 5.3), layout="constrained")
    cf = ax.contourf(XX, YY, P, levels=levels, cmap="RdBu_r", alpha=0.6, zorder=0)
    # Two teaching anchors drawn as smooth lines and labelled in the LEGEND, not
    # inline, so no probability text ever sits on top of the plotted surface.
    ax.contour(XX, YY, P, levels=[0.1], colors="#1a7f5a",
               linewidths=2.4, zorder=4)
    ax.contour(XX, YY, P, levels=[0.5], colors="#10243f",
               linewidths=1.6, linestyles="--", zorder=4)

    # Subsample the ~8k no-freeze windows so they read as a light haze instead of
    # a dark carpet that hides the contours; every (rarer) freeze window is shown.
    m = y == 0
    rng = np.random.default_rng(0)
    neg = np.where(m)[0]
    show = rng.choice(neg, size=min(1800, neg.size), replace=False)
    ax.scatter(xi[show], yi[show], s=6, c="#33414f", alpha=0.16, linewidths=0,
               zorder=1, label="no-freeze window (sample)")
    ax.scatter(xi[~m], yi[~m], s=15, c=FREEZE, edgecolors="white",
               linewidths=0.3, alpha=0.85, zorder=3, label="freeze window")

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("jerk_rms  (most important feature)")
    ax.set_ylabel("dom_freq  (dominant frequency, Hz)")
    ax.set_title("How a window is categorised — $P(\\mathrm{freeze})$ over two "
                 "features", pad=10)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("$P(\\mathrm{freeze})$  (2-feature RF)", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)
    from matplotlib.lines import Line2D
    handles, labs = ax.get_legend_handles_labels()
    handles += [Line2D([0], [0], color="#1a7f5a", lw=2.4),
                Line2D([0], [0], color="#10243f", lw=1.6, ls="--")]
    labs += ["$t=0.1$ boundary (deployed)", "$t=0.5$ boundary (naive)"]
    leg = ax.legend(handles, labs, loc="upper right", frameon=True,
                    fontsize=8.5, framealpha=0.94)
    leg.set_zorder(6)
    ax.text(1.0, -0.135,
            "Shading is the RF posterior $P(\\mathrm{freeze})$; the deployed "
            "boundary is the LOW green contour ($t\\!\\approx\\!0.1$), not $0.5$.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=MUTE)
    _save(fig, "decision_contour")


# ── 2.  Hyper-parameter landscape — how we find the best parameters ──────────
def plot_param_grid(X, y, g):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold, cross_val_score

    depths = np.array([2, 3, 4, 6, 8, 12, 18])
    leaves = np.array([1, 2, 4, 8, 16, 32, 64])
    gkf = GroupKFold(n_splits=5)
    Z = np.zeros((len(depths), len(leaves)))
    for i, d in enumerate(depths):
        for j, lf in enumerate(leaves):
            clf = RandomForestClassifier(n_estimators=120, max_depth=int(d),
                                         min_samples_leaf=int(lf),
                                         class_weight="balanced",
                                         random_state=0, n_jobs=-1)
            Z[i, j] = cross_val_score(clf, X, y, groups=g, cv=gkf,
                                      scoring="balanced_accuracy",
                                      n_jobs=1).mean()
    bi, bj = np.unravel_index(np.argmax(Z), Z.shape)
    print(f"  grid-search optimum: max_depth={depths[bi]} "
          f"min_samples_leaf={leaves[bj]}  CV bal-acc={Z[bi, bj]:.3f} "
          f"(range {Z.min():.3f}-{Z.max():.3f})")

    LL, DD = np.meshgrid(leaves, depths)
    fig, ax = plt.subplots(figsize=(7.2, 5.1), layout="constrained")
    levels = np.linspace(Z.min(), Z.max(), 13)
    cf = ax.contourf(LL, DD, Z, levels=levels, cmap="viridis")
    cl = ax.contour(LL, DD, Z, levels=levels[::3], colors="white",
                    linewidths=0.6, alpha=0.7)
    ax.clabel(cl, fmt="%.3f", fontsize=7.5, inline=True)
    ax.scatter([leaves[bj]], [depths[bi]], marker="*", s=420, c="#ffd166",
               edgecolors="#10243f", linewidths=1.2, zorder=5,
               label=f"optimum (depth={depths[bi]}, leaf={leaves[bj]})")
    ax.set_xscale("log", base=2)
    ax.set_xticks(leaves)
    ax.set_xticklabels(leaves)
    ax.set_yticks(depths)
    ax.set_xlabel("min_samples_leaf  (more $\\rightarrow$ simpler trees)")
    ax.set_ylabel("max_depth  (more $\\rightarrow$ more complex trees)")
    ax.set_title("How the best parameters are found — grid search over model "
                 "complexity", pad=10)
    cbar = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("CV balanced accuracy", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)
    ax.legend(loc="upper right", frameon=True, fontsize=8.5, framealpha=0.9)
    ax.text(1.0, -0.135,
            "Subject-grouped 5-fold CV on real Daphnet features. Same idea as "
            "pick_threshold.py: scan a grid, keep the held-out optimum.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=MUTE)
    _save(fig, "param_grid")


# ── 3.  Operating-point curve — choosing the decision threshold ──────────────
def plot_operating_point(feats, subj):
    from sklearn.metrics import average_precision_score, roc_auc_score

    from fog.metrics import sens_spec
    from make_accuracy_figs import _make_rf

    prob_all, true_all = [], []
    for s in subj:
        Xte, yte = feats[s]
        Xtr = np.concatenate([feats[k][0] for k in subj if k != s])
        ytr = np.concatenate([feats[k][1] for k in subj if k != s])
        clf = _make_rf().fit(Xtr, ytr)
        prob_all.append(clf.predict_proba(Xte)[:, 1])
        true_all.append(yte)
    prob = np.concatenate(prob_all)
    true = np.concatenate(true_all)
    auc = roc_auc_score(true, prob)
    ap = average_precision_score(true, prob)

    ts = np.linspace(0.02, 0.98, 49)
    sens, spec, you = [], [], []
    for t in ts:
        se, sp, _ = sens_spec(true, (prob >= t).astype(int))
        sens.append(se)
        spec.append(sp)
        you.append(se + sp - 1.0)
    sens, spec, you = map(np.asarray, (sens, spec, you))
    k = int(np.nanargmax(you))
    print(f"  9-feat RF OOF: ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}  "
          f"max Youden J={you[k]:.3f} at t*={ts[k]:.2f} "
          f"(sens={sens[k]:.2f}, spec={spec[k]:.2f})")

    fig, ax = plt.subplots(figsize=(7.3, 4.9), layout="constrained")
    ax.plot(ts, sens, color=FREEZE, lw=2.3, label="sensitivity")
    ax.plot(ts, spec, color=NAVY, lw=2.3, label="specificity")
    ax.plot(ts, you, color="#2a7f62", lw=2.3, ls="--", label="Youden's J")
    ax.axvline(ts[k], color="#6b7785", lw=1.0, ls=":")
    ax.scatter([ts[k]], [you[k]], marker="*", s=340, c="#ffd166",
               edgecolors="#10243f", linewidths=1.1, zorder=5,
               label=f"max J at $t^*={ts[k]:.2f}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("decision threshold on $P(\\mathrm{freeze})$")
    ax.set_ylabel("score")
    ax.set_title("How the best threshold is found — operating-point sweep "
                 "(9-feature RF)", pad=10)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.legend(loc="lower center", ncol=2, frameon=True, fontsize=9,
              framealpha=0.9)
    ax.text(1.0, -0.16,
            f"pooled LOSO, out-of-fold  ·  ROC-AUC $={auc:.2f}$  ·  "
            f"PR-AUC $={ap:.2f}$  ·  threshold-free ranking quality.",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=MUTE)
    _save(fig, "operating_point")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="/Users/paing/daphnet/dataset",
                    help="Daphnet dataset directory")
    ap.add_argument("--sensor", default="ankle",
                    choices=["ankle", "thigh", "trunk"])
    args = ap.parse_args(argv)

    print(f"Loading Daphnet from {args.data} ...")
    feats, subj, X, y, g = load_features(args.data, args.sensor)
    print("decision contour ...")
    plot_decision_contour(X, y)
    print("parameter grid ...")
    plot_param_grid(X, y, g)
    print("operating point ...")
    plot_operating_point(feats, subj)
    print("Done — decision_contour / param_grid / operating_point in accuracy_figs/")


if __name__ == "__main__":
    main()
