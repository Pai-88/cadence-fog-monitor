#!/usr/bin/env python3
"""pick_threshold.py -- choose the FoG detector's operating threshold HONESTLY.

Reads the out-of-fold (OOF) predictions dumped by ``fog_allinone.py``
(``fog_plots/loso_predictions.csv`` with columns
``subject, y_true, y_prob, y_pred_raw, y_pred_debounced``) and reports
sensitivity / specificity as a function of the decision threshold applied to
``y_prob`` (predict "freeze" when ``y_prob >= t``).

WHY THIS SCRIPT EXISTS -- the integrity point
----------------------------------------------
The pooled OOF set is already a fair *leave-one-subject-out* estimate AT A FIXED
threshold (each window was scored by a model that never trained on that subject).
But if you scan every threshold on the whole pooled set and then quote the best
sens/spec from that SAME set, you have tuned on your test data and the number is
optimistic. To stay honest the threshold must be picked on data that is DISJOINT
**by subject** from the data you score it on.

This script does that with NESTED leave-one-subject-out::

    for each held-out subject s:
        pick t* on the OTHER subjects' OOF rows   (maximise Youden's J, or hit a
                                                    target sensitivity),
        then label subject s at that t*.
    pool every subject's labels  ->  honest sens/spec for a NEW patient.

No subject's threshold is ever chosen using that subject's own data, so the
pooled result is an unbiased estimate of "pick the threshold on some patients,
deploy on a new one." The plain pooled sweep is printed too, but ONLY as a
diagnostic trade-off curve -- it is explicitly NOT how you choose the deployed
threshold, and the script labels it as such.

The garment thresholds THEN debounces (assert after ONSET consecutive positive
windows, release after OFFSET clear -- matching fog_allinone.debounce and the
firmware), so metrics are reported on the debounced *deployed* stream by
default. ``--no-debounce`` shows the raw per-window view instead.

Usage::

    python pick_threshold.py                         # default csv + Youden's J
    python pick_threshold.py -c path/to/loso_predictions.csv
    python pick_threshold.py --criterion sens-target --target-sens 0.80
    python pick_threshold.py --no-debounce           # raw per-window metric

Dependencies: numpy + pandas only (no torch / sklearn needed).
"""
import argparse
import os

import numpy as np
import pandas as pd

# Mirrors fog_allinone.ONSET_WINDOWS / OFFSET_WINDOWS and the firmware. Kept here
# as defaults so this tool stays dependency-light; override with --onset/--offset
# if you ever retune the deployed detector (keep all three in sync).
DEFAULT_ONSET = 2
DEFAULT_OFFSET = 2
DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'fog_plots', 'loso_predictions.csv')


def debounce(pred, onset, offset):
    """Latching hysteresis -- identical logic to fog_allinone.debounce.

    Must be applied PER subject on that subject's own time-ordered windows; the
    CSV preserves window order within each subject block, so grouping by subject
    is correct.
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


def sens_spec(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float('nan')
    spec = tn / (tn + fp) if (tn + fp) else float('nan')
    return sens, spec, (tp, fn, tn, fp)


def labels_at(df, subjects, t, onset, offset, do_debounce):
    """Pooled (y_true, y_pred) over `subjects` at threshold t, debounced per
    subject so no spurious transitions are created across subject boundaries."""
    yt, yp = [], []
    for s in subjects:
        sub = df[df.subject == s]                       # already time-ordered
        pred = (sub.y_prob.values >= t).astype(np.int64)
        if do_debounce:
            pred = debounce(pred, onset, offset)
        yt.append(sub.y_true.values.astype(np.int64))
        yp.append(pred)
    return np.concatenate(yt), np.concatenate(yp)


def score(s, sp, criterion, target):
    """Higher is better. Youden's J, or 'reach target sensitivity then maximise
    specificity' (and while below target, rank by how close sensitivity is)."""
    if criterion == 'youden':
        return s + sp - 1.0
    # sens-target
    return sp if s >= target else (s - 1.0)


def best_threshold(df, sel_subjects, grid, criterion, target, onset, offset, do_debounce):
    best_t, best_score = grid[0], -np.inf
    best_ss = (float('nan'), float('nan'))
    for t in grid:
        s, sp, _ = sens_spec(*labels_at(df, sel_subjects, t, onset, offset, do_debounce))
        if np.isnan(s) or np.isnan(sp):
            continue
        sc = score(s, sp, criterion, target)
        if sc > best_score:
            best_score, best_t, best_ss = sc, t, (s, sp)
    return best_t, best_ss


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-c', '--csv', default=DEFAULT_CSV,
                    help='OOF predictions CSV (default: %(default)s)')
    ap.add_argument('--criterion', choices=['youden', 'sens-target'], default='youden',
                    help="threshold objective on the SELECTION subjects "
                         "(default: youden = maximise sensitivity+specificity-1)")
    ap.add_argument('--target-sens', type=float, default=0.80,
                    help='target sensitivity for --criterion sens-target (default: 0.80)')
    ap.add_argument('--onset', type=int, default=DEFAULT_ONSET,
                    help='consecutive positive windows to assert (default: %(default)s)')
    ap.add_argument('--offset', type=int, default=DEFAULT_OFFSET,
                    help='consecutive clear windows to release (default: %(default)s)')
    ap.add_argument('--no-debounce', action='store_true',
                    help='report the raw per-window metric instead of the deployed stream')
    ap.add_argument('--grid', type=int, default=99,
                    help='number of candidate thresholds in (0,1) (default: %(default)s)')
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(
            f"CSV not found: {args.csv}\n"
            "Run fog_allinone.py first (it writes fog_plots/loso_predictions.csv), "
            "or pass the path with -c.")

    df = pd.read_csv(args.csv)
    needed = {'subject', 'y_true', 'y_prob'}
    if not needed.issubset(df.columns):
        raise SystemExit(f"CSV missing columns {needed - set(df.columns)}; got {list(df.columns)}")

    do_debounce = not args.no_debounce
    grid = np.linspace(1.0 / (args.grid + 1), args.grid / (args.grid + 1), args.grid)
    subjects = sorted(df.subject.unique())
    stream = 'debounced (deployed)' if do_debounce else 'raw per-window'
    crit = ("Youden's J" if args.criterion == 'youden'
            else f"sensitivity >= {args.target_sens:.2f}, then max specificity")

    print(f"loaded {len(df)} OOF windows across {len(subjects)} subjects: {', '.join(subjects)}")
    print(f"metric stream : {stream}"
          + (f"  (onset={args.onset}, offset={args.offset})" if do_debounce else ""))
    print(f"selection rule: {crit}")
    if len(subjects) < 3:
        print("WARNING: <3 subjects -> nested LOSO selection set is tiny; treat as a smoke check only.")

    # ---- DIAGNOSTIC sweep on the pooled set (NOT how to pick the threshold) ----
    print("\n" + "=" * 64)
    print("DIAGNOSTIC trade-off curve  (pooled over ALL subjects)")
    print("  -> illustration only. DO NOT pick the threshold from this table:")
    print("     scoring the same pooled set you scanned = tuning on test.")
    print("=" * 64)
    print(f"  {'thr':>5}  {'sens':>6}  {'spec':>6}  {'youden':>7}")
    for t in np.linspace(0.1, 0.9, 9):
        s, sp, _ = sens_spec(*labels_at(df, subjects, t, args.onset, args.offset, do_debounce))
        print(f"  {t:5.2f}  {s:6.3f}  {sp:6.3f}  {s + sp - 1:7.3f}")

    # ---- HONEST nested-LOSO operating point -----------------------------------
    yt_all, yp_all, chosen = [], [], []
    for s in subjects:
        others = [o for o in subjects if o != s]
        t_star, _ = best_threshold(df, others, grid, args.criterion,
                                   args.target_sens, args.onset, args.offset, do_debounce)
        yt, yp = labels_at(df, [s], t_star, args.onset, args.offset, do_debounce)
        yt_all.append(yt); yp_all.append(yp); chosen.append((s, t_star))
    yt_all = np.concatenate(yt_all); yp_all = np.concatenate(yp_all)
    sens, spec, (tp, fn, tn, fp) = sens_spec(yt_all, yp_all)
    thrs = np.array([t for _, t in chosen])

    print("\n" + "=" * 64)
    print("HONEST operating point  (nested LOSO -- threshold picked on OTHER")
    print("subjects, scored on the held-out one).  This is the number to quote.")
    print("=" * 64)
    print("  threshold chosen per held-out subject:")
    for s, t in chosen:
        print(f"    {s}:  t* = {t:.3f}")
    print(f"  median t* = {np.median(thrs):.3f}   (range {thrs.min():.3f} - {thrs.max():.3f})")
    print(f"\n  pooled sensitivity = {sens:.3f}   specificity = {spec:.3f}")
    print(f"  confusion: TP={tp}  FN={fn}  TN={tn}  FP={fp}")

    # ---- what the diagnostic optimism gap looks like (single median thr on all) ---
    med = float(np.median(thrs))
    s_opt, sp_opt, _ = sens_spec(*labels_at(df, subjects, med, args.onset, args.offset, do_debounce))
    print("\n" + "-" * 64)
    print(f"For reference, the SAME median threshold ({med:.3f}) scored on ALL")
    print(f"subjects pooled gives sens={s_opt:.3f} spec={sp_opt:.3f} -- this is the")
    print("optimistic figure; the nested number above is the honest one. The gap")
    print("between them is your tuning-on-test bias.")
    print("-" * 64)
    print("\nDeploy: set the freeze probability threshold (the 0.50 hard-coded in")
    print(f"fog_allinone.predict_proba and the live detector) to ~{med:.2f}.")
    print("Headline the NESTED sens/spec, not the diagnostic sweep.")


if __name__ == '__main__':
    main()
