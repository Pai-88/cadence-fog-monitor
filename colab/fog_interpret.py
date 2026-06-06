"""Interpretability add-on: permutation importance + partial dependence + SHAP.

These run on an **interpretable engineered-feature model** (a RandomForest over
nine clinically-readable features per window), not the raw-timestep CNN — the
point is to show *which* features drive a "freeze" call, which a black-box CNN
on raw samples cannot. It reuses the loaders, metrics and plot helpers from
``fog_allinone`` so the windows and the held-out-subject split match the CNN
evaluation exactly.

Run as a script::

    python fog_interpret.py /path/to/daphnet
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as signal
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay, permutation_importance

# Reuse everything already defined in the all-in-one training module so the
# windows, labels and held-out split line up with the CNN evaluation.
from fog_allinone import (
    FREEZE_BAND,
    LOCO_BAND,
    SAMPLE_RATE,
    TREMOR_BAND,
    _save,
    _val_subject,
    band_power,
    build_windows,
    load_daphnet,
    sens_spec,
)

# Nine clinically-readable features per 4 s window. All but jerk are computed on
# the accel MAGNITUDE, so they are orientation-invariant (the garment can rotate).
FEATURE_NAMES = ['freeze_power', 'loco_power', 'freeze_index', 'tremor_power',
                 'total_power', 'dom_freq', 'mag_rms', 'mag_range', 'jerk_rms']


def extract_features(X):
    """(N, 3, W) accel windows → (N, 9) interpretable feature matrix."""
    feats = []
    for i in range(len(X)):
        w = X[i].T                                  # (W, 3)
        mag = np.linalg.norm(w.astype(np.float64), axis=1)
        magc = mag - mag.mean()
        fp = band_power(magc, SAMPLE_RATE, FREEZE_BAND)
        lp = band_power(magc, SAMPLE_RATE, LOCO_BAND)
        tp = band_power(magc, SAMPLE_RATE, TREMOR_BAND)
        total = band_power(magc, SAMPLE_RATE, (0.5, 15.0))
        f, pxx = signal.welch(magc, fs=SAMPLE_RATE, nperseg=min(len(magc), 256))
        dom_freq = float(f[np.argmax(pxx)])
        mag_rms = float(np.sqrt(np.mean(magc ** 2)))
        mag_range = float(mag.max() - mag.min())
        jerk_rms = float(np.sqrt(np.mean(np.diff(mag) ** 2)))
        feats.append([fp, lp, fp / (lp + 1e-9), tp, total,
                      dom_freq, mag_rms, mag_range, jerk_rms])
    return np.array(feats, dtype=np.float64)


def plot_perm_importance(clf, X_te, y_te):
    r = permutation_importance(clf, X_te, y_te, n_repeats=20,
                               random_state=0, scoring='balanced_accuracy')
    order = r.importances_mean.argsort()[::-1]
    names = [FEATURE_NAMES[i] for i in order]
    fig, ax = plt.subplots(figsize=(9, 6), layout='constrained')
    sns.barplot(x=r.importances_mean[order], y=names, color='#0b6e4f', ax=ax)
    ax.errorbar(r.importances_mean[order], range(len(order)),
                xerr=r.importances_std[order], fmt='none', ecolor='black', capsize=4)
    ax.set(title='Permutation importance (held-out subject)',
           xlabel='drop in balanced accuracy when feature is shuffled', ylabel='')
    _save(fig, "interp_perm_importance")
    return order


def plot_partial_dependence(clf, X_tr, top_idx):
    fig, ax = plt.subplots(1, len(top_idx), figsize=(6 * len(top_idx), 5),
                           layout='constrained')
    PartialDependenceDisplay.from_estimator(
        clf, X_tr, features=list(top_idx), feature_names=FEATURE_NAMES,
        ax=ax, line_kw={'color': '#d1495b', 'lw': 2.5})
    fig.suptitle('Partial dependence — P(freeze) vs. top features')
    _save(fig, "interp_partial_dependence")


def plot_shap(clf, X_te):
    import shap
    clf.n_jobs = 1               # SHAP's additivity check calls clf.predict; with the
                                 # RF's n_jobs=-1 that forks loky workers after a thread
                                 # pool exists and can deadlock on macOS. Single-thread it.
    Xs = X_te[:400]                                  # cap for speed; exact for trees
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(Xs, check_additivity=False)
    if isinstance(sv, list):                         # older API: [class0, class1]
        sv1 = sv[1]
    elif getattr(sv, 'ndim', 2) == 3:                # newer API: (n, feat, class)
        sv1 = sv[:, :, 1]
    else:
        sv1 = sv
    Xs_df = pd.DataFrame(Xs, columns=FEATURE_NAMES)

    shap.summary_plot(sv1, Xs_df, show=False, plot_size=(9, 6))
    fig = plt.gcf(); fig.suptitle('SHAP — per-window contribution to "freeze"', y=1.02)
    _save(fig, "interp_shap_beeswarm")

    shap.summary_plot(sv1, Xs_df, plot_type='bar', show=False, plot_size=(9, 6))
    fig = plt.gcf(); fig.suptitle('SHAP — mean |impact| per feature', y=1.02)
    _save(fig, "interp_shap_bar")


def run_interpretation(data_dir, sensor='ankle', n_estimators=300):
    print(f"interpretability on engineered features | sensor: {sensor}")
    subjects = load_daphnet(data_dir, sensor)
    windows = {s: build_windows(r) for s, r in subjects.items()}
    windows = {s: (X, y) for s, (X, y) in windows.items() if len(y) > 0}
    subj_ids = sorted(windows)

    # Honest split: hold out the most-freeze subject as the test patient.
    test_s = _val_subject(windows, subj_ids)
    train_s = [s for s in subj_ids if s != test_s]
    X_tr = extract_features(np.concatenate([windows[s][0] for s in train_s]))
    y_tr = np.concatenate([windows[s][1] for s in train_s])
    X_te = extract_features(windows[test_s][0])
    y_te = windows[test_s][1]
    print(f"  train {len(y_tr)} windows ({len(train_s)} subjects) | "
          f"test {len(y_te)} windows (held-out {test_s})")

    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight='balanced',
                                 random_state=0, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    s, sp, _ = sens_spec(y_te, clf.predict(X_te))
    print(f"  RandomForest(features) held-out: sensitivity={s:.3f} specificity={sp:.3f}")

    order = plot_perm_importance(clf, X_te, y_te)
    plot_partial_dependence(clf, X_tr, order[:3])
    plot_shap(clf, X_te)
    print("  done — interp_*.png saved in fog_plots/ and shown above.")


if __name__ == '__main__':
    import sys
    run_interpretation(sys.argv[1])
