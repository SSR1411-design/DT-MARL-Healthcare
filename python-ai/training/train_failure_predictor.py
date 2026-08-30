"""
Host / network failure-prediction training + honest evaluation.

Run:
    python training/train_failure_predictor.py                 # 12-column whitelist
    python training/train_failure_predictor.py --features legacy   # old 7-column list

METHODOLOGY is imported from training/host_cv.py so that every script
reporting a host number uses the identical protocol: grouped CV over
(node, temporal-block) groups, overlap purge, train-only scaler, inner
grouped validation slice for early stopping AND threshold selection, pooled
out-of-fold metrics. Nothing here selects a threshold or stops training using
test labels.

FEATURE WHITELIST — note the change, it matters for interpreting the numbers.
The previous cycle used a 7-column raw block, having dropped cpu / ram /
bandwidth / runningTasks / underAttack because they were CONSTANT in the old
export. The simulator's new latent-wear failure model writes its symptom
overlay onto exactly those channels and none of the 12 observable columns is
constant any more, so the same rule ("keep the columns that vary") readmits
all 12. Both arms are runnable here under an identical protocol; see
data/failure_dataset.py. No audit_* column is admitted in either arm — that
is enforced by assert_no_leakage, not by convention.

HOW TO READ THE OUTPUT. The dataset is ~2.1% positive, so accuracy is not
the headline: always-negative scores ~97.9%. Judge on F1 / PR-AUC / recall /
precision, and check that the model beats BOTH trivial baselines printed at
the end. Out-of-fold probabilities are written to
saved_models/failure_predictor_oof*.npz so that evaluate_lead_time.py can
measure lead time on predictions no model ever trained on.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "training"))

from data.failure_dataset import (  # noqa: E402
    build_windows, save_meta, feature_columns,
    OBSERVABLE_COLUMNS, LEGACY_RAW_FEATURES,
)
from host_cv import (  # noqa: E402
    CSV_PATH, SEED, SEQ_LEN, EPOCHS, N_SPLITS, device, seed_everything,
    grouped_oof, train_one, pick_threshold, predict_proba,
    full_metrics, print_metrics,
)
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--features", choices=["observable", "legacy"],
                default="observable",
                help="observable = all 12 telemetry columns (default); "
                     "legacy = the old 7-column whitelist")
ap.add_argument("--epochs", type=int, default=EPOCHS,
                help="epochs per fold. Only lower it for a smoke test; the "
                     "reported cycle uses the protocol default.")
ap.add_argument("--csv", default=str(CSV_PATH),
                help="telemetry export to train on. Point this at "
                     "simulation/_backup_pre_degradation/failure_history.csv "
                     "to re-measure the PREVIOUS dataset under this exact "
                     "protocol, which is the only apples-to-apples way to "
                     "compare old and new (out-of-fold vs out-of-fold).")
ap.add_argument("--tag", default=None,
                help="artifact suffix override, so a comparison run cannot "
                     "overwrite the deployment model")
args = ap.parse_args()
EPOCHS = args.epochs

RAW = OBSERVABLE_COLUMNS if args.features == "observable" else LEGACY_RAW_FEATURES
FEATS = feature_columns(RAW)
if args.tag is not None:
    SUFFIX = f"_{args.tag}"
else:
    SUFFIX = "" if args.features == "observable" else "_legacy"

seed_everything(SEED)
print("Device:", device)
print(f"Dataset    : {args.csv}")
print(f"Feature arm: {args.features}  "
      f"({len(RAW)} raw -> {len(FEATS)} engineered)")
print("raw columns:", RAW)

X, y, node_ids, event_ids = build_windows(args.csv, SEQ_LEN, raw_features=RAW)
print(f"\nWindows: {X.shape}  positives: {int(y.sum())} "
      f"({y.mean()*100:.2f}%)  blocks: {len(np.unique(event_ids))}  "
      f"nodes: {len(np.unique(node_ids))}")

# --------------------------------------------------------------------------
# Leakage-safe grouped cross-validation -> pooled out-of-fold predictions.
# --------------------------------------------------------------------------

print(f"\n=== Leakage-safe grouped CV ({N_SPLITS} folds, {EPOCHS} epochs) ===")
oof = grouped_oof(X, y, node_ids, event_ids, epochs=EPOCHS)
cov = oof["covered"]
print(f"\nwindows scored out-of-fold: {cov.sum()}/{len(y)} "
      f"({cov.sum()/len(y)*100:.1f}%), "
      f"positives covered: {int(y[cov].sum())}/{int(y.sum())}")

pooled = full_metrics(y[cov], oof["pred"][cov], oof["prob"][cov])
print("\n=== POOLED OUT-OF-FOLD METRICS (headline) ===")
print_metrics("host predictor / pooled OOF", pooled)

# --------------------------------------------------------------------------
# Trivial reference baselines, on the SAME windows and the SAME metric code.
# On a ~98%-negative dataset neither accuracy nor F1 means anything without
# these, and the previous dataset's model was reproduced exactly by the
# one-line 'linkUp==1' rule.
# --------------------------------------------------------------------------

print("\n=== TRIVIAL BASELINES (same windows, same metrics) ===")
baselines = {}
last = X[:, -1, :]                      # last tick of each window
col = {c: i for i, c in enumerate(RAW)}

baselines["always-negative"] = np.zeros_like(y)
baselines["linkUp==1"] = (last[:, col["linkUp"]] == 1).astype(int)
baselines["degraded==1"] = (last[:, col["degraded"]] == 1).astype(int)
if "active" in col:
    baselines["active==1"] = (last[:, col["active"]] == 1).astype(int)

base_metrics = {}
for name, pred in baselines.items():
    # evaluated on the out-of-fold-covered windows so the comparison with the
    # headline is like-for-like (no in-sample vs out-of-fold mixing)
    m = full_metrics(y[cov], pred[cov])
    base_metrics[name] = m
    print_metrics(f"BASELINE {name}", m)

print("\n--- headline vs baselines (F1 is the comparison that matters) ---")
print(f"  {'model':22s} {'acc':>8s} {'prec':>8s} {'rec':>8s} {'F1':>8s}")
print(f"  {'PREDICTOR (OOF)':22s} {pooled['accuracy']*100:7.2f}% "
      f"{pooled['precision']:8.4f} {pooled['recall']:8.4f} {pooled['f1']:8.4f}")
for name, m in base_metrics.items():
    print(f"  {name:22s} {m['accuracy']*100:7.2f}% "
          f"{m['precision']:8.4f} {m['recall']:8.4f} {m['f1']:8.4f}")

# --------------------------------------------------------------------------
# Final deployable model: trained on a 4/5 grouped split, remaining 1/5 used
# ONLY to select the deployment threshold. Deliberately not trained on 100%
# of the windows.
# --------------------------------------------------------------------------

print("\n=== Training final deployable model (4/5 grouped split) ===")
final_inner = StratifiedGroupKFold(n_splits=5, shuffle=True,
                                   random_state=SEED + 2)
i_tr, i_va = next(final_inner.split(X, y, event_ids))
model, std = train_one(X[i_tr], y[i_tr], X[i_va], y[i_va], epochs=EPOCHS)
thr, vf1 = pick_threshold(model, std, X[i_va], y[i_va])
print(f"Final deployment threshold: {thr:.3f} (val F1={vf1:.4f})")

held = full_metrics(y[i_va], (predict_proba(model, std, X[i_va]) > thr).astype(int))
print(f"NOTE: the threshold was tuned on that same 1/5 slice, so its F1 "
      f"({held['f1']:.4f}) is optimistic.\n      Quote the pooled OOF "
      f"numbers above, not this one.")

save_dir = ROOT / "saved_models"
save_dir.mkdir(exist_ok=True)
torch.save(model.state_dict(), save_dir / f"failure_predictor{SUFFIX}.pth")
std.save(save_dir / f"failure_predictor{SUFFIX}_scaler.npz")
save_meta(save_dir / f"failure_predictor{SUFFIX}_meta.json", thr, FEATS, SEQ_LEN)

# Out-of-fold probabilities, for lead-time evaluation on predictions that no
# model saw in training.
np.savez(save_dir / f"failure_predictor{SUFFIX}_oof.npz",
         prob=oof["prob"], pred=oof["pred"], covered=cov,
         y=y, node_ids=node_ids, event_ids=event_ids,
         fold_thresholds=np.array([f["threshold"] for f in oof["folds"]]))

with open(save_dir / f"host_metrics{SUFFIX}.json", "w") as f:
    json.dump(dict(feature_arm=args.features, dataset=args.csv,
                   raw_features=RAW,
                   engineered_features=FEATS, seq_len=SEQ_LEN,
                   epochs=EPOCHS, n_splits=N_SPLITS,
                   pooled_oof=pooled, baselines=base_metrics,
                   folds=oof["folds"], deploy_threshold=float(thr)),
              f, indent=2)

print(f"\nSaved: failure_predictor{SUFFIX}.pth + scaler + meta + OOF probs "
      f"+ host_metrics{SUFFIX}.json")
print("Training complete.")
