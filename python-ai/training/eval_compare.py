"""
Apples-to-apples comparison: ORIGINAL pipeline vs IMPROVED pipeline,
both evaluated under the SAME leakage-safe grouped cross-validation and
pooled out-of-fold metrics. This isolates the effect of the *pipeline*
changes (normalization, pos_weight cap, optimizer, threshold selection) with
the feature set and dataset held fixed.

SUPERSEDED, MOSTLY. This script predates training/host_cv.py, which is now the
single source of truth for the protocol, and it reimplements that protocol
inline — exactly the drift risk host_cv.py exists to remove. For the numbers
quoted in the report, prefer:

    python training/train_failure_predictor.py --features observable
    python training/train_failure_predictor.py --features legacy
    python training/train_failure_predictor.py --features legacy \
        --csv ../simulation/_backup_pre_degradation/failure_history.csv \
        --tag OLDDATA

which cover old-vs-new dataset and old-vs-new feature whitelist under one
protocol, and additionally report PR-AUC / ROC-AUC and assert no train/test
overlap. This file is kept only for the ORIGINAL-hyperparameters arm, which
those scripts do not reproduce.

ORIGINAL = raw 12 telemetry columns, NO normalization, uncapped pos_weight,
           plain Adam lr=3e-4, 20 epochs, fixed 0.5 threshold.
IMPROVED = engineered causal features, train-only normalization, capped
           pos_weight, AdamW+cosine, early stop, val-F1 threshold.

Note on the ORIGINAL arm's feature block: on the OLD export 5 of those 12
columns were constant, which is why the whitelist was cut to 7. On the current
export none are constant, so this arm is no longer "12 features incl. 5
constant" — it is the full observable block with the old hyperparameters.

TRIVIAL baselines are printed alongside, because on a ~97.9%-negative dataset
neither arm's accuracy is interpretable without them.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
from data.failure_dataset import (
    build_windows, Standardizer, purge_overlap, NUM_FEATURES,
    OBSERVABLE_COLUMNS, feature_columns,
)
from models.failure_predictor import FailurePredictor

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CSV = ROOT.parent / "simulation" / "failure_history.csv"
SEQ = 10

# ---- ORIGINAL raw windows (12 features, no engineering) ----
ORIG_FEATS = list(OBSERVABLE_COLUMNS)
df = pd.read_csv(CSV)
oXs, oys, gids, nids = [], [], [], []
for nid, g in df.groupby("nodeId"):
    g = g.sort_values("time").reset_index(drop=True)
    f = g[ORIG_FEATS].values.astype(np.float32); t = g["willFailSoon"].values.astype(np.int64)
    for i in range(len(g) - SEQ + 1):
        end = i + SEQ - 1
        oXs.append(f[i:i+SEQ]); oys.append(t[end]); gids.append(nid*100000+end//SEQ); nids.append(nid)
oX = np.stack(oXs); oy = np.array(oys); og = np.array(gids); onid = np.array(nids)

# ---- IMPROVED engineered windows ----
iX, iy, inid, ig = build_windows(str(CSV), SEQ)

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)


def run_original():
    preds, trues = [], []
    for tr, te in skf.split(oX, oy, og):
        if oy[te].sum() == 0:
            continue
        tr = purge_overlap(tr, te, onid, og)      # SAME purge as IMPROVED
        Xtr = torch.tensor(oX[tr], device=device)           # NO normalization (as original)
        ytr = torch.tensor(oy[tr], dtype=torch.float32, device=device)
        m = FailurePredictor(num_features=12).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=3e-4)      # original optimizer/lr
        pos = max(int(oy[tr].sum()), 1); neg = len(tr) - pos
        pw = torch.tensor(neg / pos, dtype=torch.float32, device=device)   # uncapped ~42x
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        for ep in range(20):                                  # original epochs
            m.train(); perm = torch.randperm(len(Xtr), device=device)
            for s in range(0, len(Xtr), 32):
                idx = perm[s:s+32]; opt.zero_grad()
                crit(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            pt = torch.sigmoid(m(torch.tensor(oX[te], device=device))).cpu().numpy()
        preds.append((pt > 0.5).astype(int)); trues.append(oy[te])   # fixed 0.5
    return np.concatenate(preds), np.concatenate(trues)


def run_improved():
    preds, trues = [], []
    for tr_all, te in skf.split(iX, iy, ig):
        if iy[te].sum() == 0:
            continue
        tr_all = purge_overlap(tr_all, te, inid, ig)   # SAME purge as ORIGINAL
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED+1)
        i_tr, i_va = next(inner.split(iX[tr_all], iy[tr_all], ig[tr_all]))
        tr, va = tr_all[i_tr], tr_all[i_va]
        std = Standardizer().fit(iX[tr])
        Xtr = torch.tensor(std.transform(iX[tr]), device=device)
        ytr = torch.tensor(iy[tr], dtype=torch.float32, device=device)
        Xva = torch.tensor(std.transform(iX[va]), device=device)
        m = FailurePredictor(num_features=NUM_FEATURES).to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=80)
        pos = max(int(iy[tr].sum()), 1); neg = len(tr) - pos
        pw = torch.tensor(min(neg/pos, 8.0), dtype=torch.float32, device=device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        best, bvf = None, -1
        for ep in range(80):
            m.train(); perm = torch.randperm(len(Xtr), device=device)
            for s in range(0, len(Xtr), 64):
                idx = perm[s:s+64]; opt.zero_grad()
                crit(m(Xtr[idx]), ytr[idx]).backward(); opt.step()
            sch.step(); m.eval()
            with torch.no_grad():
                pv = torch.sigmoid(m(Xva)).cpu().numpy()
            vf = max((f1_score(iy[va],(pv>t).astype(int),zero_division=0) for t in np.linspace(.05,.95,19)),default=0)
            if vf >= bvf: bvf, best = vf, {k:v.clone() for k,v in m.state_dict().items()}
        m.load_state_dict(best); m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(Xva)).cpu().numpy()
            pt = torch.sigmoid(m(torch.tensor(std.transform(iX[te]), device=device))).cpu().numpy()
        bt, bf = .5, -1
        for t in np.linspace(.05,.95,91):
            s = f1_score(iy[va],(pv>t).astype(int),zero_division=0)
            if s > bf: bf, bt = s, t
        preds.append((pt > bt).astype(int)); trues.append(iy[te])
    return np.concatenate(preds), np.concatenate(trues)


def report(name, yp, yt):
    cm = confusion_matrix(yt, yp, labels=[0,1])
    print(f"\n[{name}]")
    print(f"  Accuracy : {accuracy_score(yt,yp)*100:.2f}%")
    print(f"  Precision: {precision_score(yt,yp,zero_division=0):.4f}")
    print(f"  Recall   : {recall_score(yt,yp,zero_division=0):.4f}")
    print(f"  F1-score : {f1_score(yt,yp,zero_division=0):.4f}")
    print(f"  Confusion [true x pred]: {cm.tolist()}")
    tn,fp,fn,tp = cm.ravel()
    print(f"  TN={tn} FP={fp} FN={fn} TP={tp}")


print("Trivial reference baselines (no learning) on the same windows:")
report("ALWAYS-NEGATIVE", np.zeros_like(iy), iy)
# Look the column up by NAME. It used to be hardcoded as index 3, which was
# correct only for the old 7-column raw block; under the current engineered
# layout index 3 is `energy`, so the hardcoded version printed a baseline for
# the wrong column entirely.
_ICOL = {c: i for i, c in enumerate(feature_columns(OBSERVABLE_COLUMNS))}
report("RULE linkUp==1", (iX[:, -1, _ICOL["linkUp"]] == 1).astype(int), iy)

print("\nRunning ORIGINAL pipeline under leakage-safe grouped CV...")
report("ORIGINAL", *run_original())
print("\nRunning IMPROVED pipeline under identical grouped CV...")
report("IMPROVED", *run_improved())
