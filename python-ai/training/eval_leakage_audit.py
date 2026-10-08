"""
Leakage / validity audit for the HOST failure predictor.

This script does not train a deployable model. It answers one question:
is the reported grouped-CV performance evidence of *prediction*, or is it an
artefact of the dataset's structure?

  A. Trivial reference baselines (no learning at all).
  B. The information ceiling of each feature representation, measured by
     exact-duplicate window analysis (identical feature tensor, conflicting
     label => provably unpredictable).
  C. Leave-one-NODE-out cross-validation: train on 9 hosts, test on a host
     whose telemetry the model has never seen.

WHY (C) MATTERS, AND WHAT CHANGED

The grouped CV used for the headline groups windows by (node, temporal
block), so train and test always share all ten hosts. On the OLD dataset that
was fatal: each host had two scheduled events and never recovered, `active`
and `linkUp` were monotone step functions, and `time_since_active` was
effectively a per-host clock — so a model could fit "host k alerts at counter
value K" from other blocks of the same host and score well without learning
anything about failure. LONO removed that and the apparent skill vanished.

On the new dataset that specific artefact is gone: hosts fail and recover
repeatedly (32 failures, 32 recoveries over 10 hosts), the counters reset at
every recovery, and their range is bounded (0..44) instead of growing with
the clock. LONO is still the stronger test, and it is still the number to
quote for "does this transfer to an unseen host" — per-host susceptibility
and the drawn fault mechanism differ between hosts, so a gap is expected;
a *collapse to the trivial baselines* is the failure signature to look for.

Run:  python training/eval_leakage_audit.py [--features observable|legacy]
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "training"))

from data.failure_dataset import (  # noqa: E402
    build_windows, feature_columns, OBSERVABLE_COLUMNS, LEGACY_RAW_FEATURES,
)
from host_cv import (  # noqa: E402
    CSV_PATH, SEED, SEQ_LEN, EPOCHS, seed_everything, train_one,
    pick_threshold, predict_proba, full_metrics,
)

ap = argparse.ArgumentParser()
ap.add_argument("--features", choices=["observable", "legacy"],
                default="observable")
ap.add_argument("--epochs", type=int, default=EPOCHS)
ap.add_argument("--skip-lono", action="store_true")
args = ap.parse_args()

RAW = OBSERVABLE_COLUMNS if args.features == "observable" else LEGACY_RAW_FEATURES
FEATS = feature_columns(RAW)
SUFFIX = "" if args.features == "observable" else "_legacy"

seed_everything(SEED)

X, y, node_ids, event_ids = build_windows(str(CSV_PATH), SEQ_LEN,
                                          raw_features=RAW)
print(f"feature arm={args.features}  windows={X.shape}  "
      f"positives={int(y.sum())}  negatives={int((y == 0).sum())}")
print(f"nodes={sorted(set(node_ids.tolist()))}  features={len(FEATS)}\n")


def report(name, yt, yp, yprob=None):
    m = full_metrics(yt, yp, yprob)
    extra = (f"  PR-AUC={m['pr_auc']:.4f} ROC-AUC={m['roc_auc']:.4f}"
             if "pr_auc" in m else "")
    print(f"  {name:<46s} acc={m['accuracy']*100:6.2f}%  "
          f"prec={m['precision']:.4f}  rec={m['recall']:.4f}  "
          f"f1={m['f1']:.4f}  TN={m['tn']} FP={m['fp']} FN={m['fn']} "
          f"TP={m['tp']}{extra}")
    return m


# --------------------------------------------------------------------------
# A. Trivial reference baselines
# --------------------------------------------------------------------------
print("=" * 108)
print("A. TRIVIAL BASELINES (no model, no training) -- any real model must beat these")
print("=" * 108)
col = {c: i for i, c in enumerate(RAW)}
last = X[:, -1, :]
report("always predict NEGATIVE", y, np.zeros_like(y))
report("always predict POSITIVE", y, np.ones_like(y))
report("rule: linkUp==1 -> alert", y, (last[:, col["linkUp"]] == 1).astype(int))
report("rule: active==1 -> alert", y, (last[:, col["active"]] == 1).astype(int))
report("rule: degraded==1 -> alert", y,
       (last[:, col["degraded"]] == 1).astype(int))
report("rule: active==0 AND linkUp==1 -> alert", y,
       ((last[:, col["active"]] == 0) & (last[:, col["linkUp"]] == 1)).astype(int))
if "cpu" in col:
    report("rule: cpu>90 -> alert", y, (last[:, col["cpu"]] > 90).astype(int))
    report("rule: linkLatencyMs>60 -> alert", y,
           (last[:, col["linkLatencyMs"]] > 60).astype(int))


# --------------------------------------------------------------------------
# B. Information ceiling by exact-duplicate analysis
# --------------------------------------------------------------------------
def ceiling(Xr, yr, label):
    groups = defaultdict(list)
    for i, row in enumerate(Xr.reshape(len(Xr), -1)):
        groups[row.tobytes()].append(i)
    amb_patterns = 0
    amb_pos = 0
    tp = fp = fn = tn = 0
    for idxs in groups.values():
        yy = yr[idxs]
        n1 = int(yy.sum())
        n0 = len(yy) - n1
        if n1 and n0:
            amb_patterns += 1
            amb_pos += n1
        if n1 > n0:
            tp += n1
            fp += n0
        else:
            fn += n1
            tn += n0
    print(f"  {label}")
    print(f"    distinct window patterns : {len(groups)} / {len(Xr)}")
    print(f"    ambiguous patterns       : {amb_patterns} "
          f"(identical features, BOTH labels)")
    print(f"    positives that are provably unpredictable: "
          f"{amb_pos}/{int(yr.sum())} ({amb_pos / max(int(yr.sum()), 1) * 100:.1f}%)")
    print(f"    accuracy-optimal ceiling : acc={(tp + tn) / len(yr) * 100:.2f}%  "
          f"rec={tp / max(tp + fn, 1):.3f}  prec={tp / max(tp + fp, 1):.3f}")
    print()


print()
print("=" * 108)
print("B. INFORMATION CEILING (exact-duplicate window analysis)")
print("=" * 108)
print("  NOTE: with continuous, noisy telemetry exact duplicates are rare, so")
print("  this ceiling is loose (it only catches byte-identical tensors). On the")
print("  old constant-column dataset it was tight and showed 74% of positives")
print("  were unpredictable; treat a high ceiling here as 'no proof of")
print("  impossibility', not as 'proof of learnability'.")
print()

df = pd.read_csv(CSV_PATH)
oXs, oys = [], []
for nid, g in df.groupby("nodeId"):
    g = g.sort_values("time").reset_index(drop=True)
    f = g[OBSERVABLE_COLUMNS].values.astype(np.float32)
    t = g["willFailSoon"].values.astype(np.int64)
    for i in range(len(g) - SEQ_LEN + 1):
        oXs.append(f[i:i + SEQ_LEN])
        oys.append(t[i + SEQ_LEN - 1])
oX, oy = np.stack(oXs), np.array(oys)

ceiling(oX, oy, "RAW 12 observable telemetry columns (no engineering)")
ceiling(X, y, f"{args.features.upper()} arm: {len(FEATS)} engineered causal features")

# Rounded-duplicate variant: byte-identity is too strict once telemetry is
# noisy, so also collapse to 1 decimal place. This is the honest measure of
# "windows a model could not tell apart in practice".
oXr = np.round(oX, 1)
groups = defaultdict(list)
for i, row in enumerate(oXr.reshape(len(oXr), -1)):
    groups[row.tobytes()].append(i)
amb_pos = sum(int(oy[ix].sum()) for ix in groups.values()
              if 0 < int(oy[ix].sum()) < len(ix))
print(f"  raw 12 columns rounded to 0.1: {len(groups)} distinct patterns, "
      f"{amb_pos}/{int(oy.sum())} positives ambiguous "
      f"({amb_pos/max(int(oy.sum()),1)*100:.1f}%)")


# --------------------------------------------------------------------------
# C. Leave-one-NODE-out cross-validation
# --------------------------------------------------------------------------
if args.skip_lono:
    print("\n(LONO skipped)")
    raise SystemExit(0)

print()
print("=" * 108)
print(f"C. LEAVE-ONE-NODE-OUT CV -- train on 9 hosts, test on the unseen 10th")
print("   (threshold selected on a validation slice of the TRAINING hosts only;")
print("    the held-out host contributes nothing to scaler, training, early")
print("    stopping, or threshold selection)")
print("=" * 108)

pooled_p, pooled_t, pooled_prob = [], [], []
per_node = []
for held in sorted(set(node_ids.tolist())):
    te = np.where(node_ids == held)[0]
    tr_all = np.where(node_ids != held)[0]
    if y[te].sum() == 0:
        print(f"  node {held}: no positives, skipped")
        continue
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True,
                                 random_state=SEED + 1)
    i_tr, i_va = next(inner.split(X[tr_all], y[tr_all], event_ids[tr_all]))
    tr, va = tr_all[i_tr], tr_all[i_va]
    assert len(set(node_ids[tr]) & {held}) == 0

    model, std = train_one(X[tr], y[tr], X[va], y[va], epochs=args.epochs)
    thr, _ = pick_threshold(model, std, X[va], y[va], verbose=False)
    pt = predict_proba(model, std, X[te])
    pred = (pt > thr).astype(int)
    m = report(f"held-out node {held} (thr={thr:.2f}, pos={int(y[te].sum())})",
               y[te], pred, pt)
    per_node.append(dict(node=int(held), threshold=float(thr), **m))
    pooled_p.append(pred)
    pooled_t.append(y[te])
    pooled_prob.append(pt)

print()
print("  POOLED leave-one-node-out:")
lono = report("LONO pooled", np.concatenate(pooled_t), np.concatenate(pooled_p),
              np.concatenate(pooled_prob))
print()
print("Interpretation: compare LONO against the (node, temporal-block) grouped-CV")
print("headline in train_failure_predictor.py. Some drop is expected — hosts")
print("differ in susceptibility and in which fault mechanism they drew. What")
print("would invalidate the headline is LONO falling to the section-A baselines.")

import json  # noqa: E402
with open(ROOT / "saved_models" / f"host_lono{SUFFIX}.json", "w") as f:
    json.dump(dict(feature_arm=args.features, per_node=per_node,
                   pooled=lono), f, indent=2)
print(f"\nWrote saved_models/host_lono{SUFFIX}.json")
