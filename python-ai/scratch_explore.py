"""
Diagnostic-only script (NOT committed to the pipeline).
Goal: measure the honest, leakage-free ceiling for host failure prediction
using proper normalization + purged stratified group cross-validation.
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from models.failure_predictor import FailurePredictor

torch.manual_seed(0)
np.random.seed(0)

CSV = ROOT.parent / "simulation" / "failure_history.csv"
SEQ = 10

ALL_FEATS = ["cpu", "ram", "bandwidth", "energy", "runningTasks",
             "active", "degraded", "linkUp", "linkBandwidthMbps",
             "linkLatencyMs", "linkPacketLoss", "underAttack"]

df = pd.read_csv(CSV)

# Drop zero-variance features (measured on full data — they carry no signal at all)
var = df[ALL_FEATS].var()
FEATS = [c for c in ALL_FEATS if var[c] > 1e-12]
print("Kept features:", FEATS)
print("Dropped (constant):", [c for c in ALL_FEATS if c not in FEATS])

# Build windows, tagging each with node and end index + a temporal block group.
X, y, groups = [], [], []
BLOCK = SEQ  # temporal block size for grouping (limits cross-fold overlap)
for nid, g in df.groupby("nodeId"):
    g = g.sort_values("time").reset_index(drop=True)
    feats = g[FEATS].values.astype(np.float32)
    tgt = g["willFailSoon"].values.astype(np.int64)
    for i in range(len(g) - SEQ + 1):
        end = i + SEQ - 1
        X.append(feats[i:i + SEQ])
        y.append(tgt[end])
        groups.append(nid * 100000 + end // BLOCK)

X = np.stack(X)
y = np.array(y)
groups = np.array(groups)
print("windows", X.shape, "positives", int(y.sum()), "groups", len(np.unique(groups)))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_eval(Xtr, ytr, Xte, yte, epochs=40, use_focal=True):
    nf = Xtr.shape[-1]
    scaler = StandardScaler().fit(Xtr.reshape(-1, nf))
    Xtr = scaler.transform(Xtr.reshape(-1, nf)).reshape(Xtr.shape).astype(np.float32)
    Xte = scaler.transform(Xte.reshape(-1, nf)).reshape(Xte.shape).astype(np.float32)

    model = FailurePredictor(num_features=nf).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    pos = max(int(ytr.sum()), 1)
    neg = len(ytr) - pos
    pw = torch.tensor(min(neg / pos, 15.0), dtype=torch.float32, device=device)

    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)

    bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def focal(logits, target, alpha=0.85, gamma=2.0):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = torch.where(target == 1, p, 1 - p)
        a = torch.where(target == 1, alpha, 1 - alpha)
        return (a * (1 - pt) ** gamma * ce).mean()

    n = len(Xtr_t)
    bs = 64
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]
            opt.zero_grad()
            logits = model(Xtr_t[idx])
            loss = focal(logits, ytr_t[idx]) if use_focal else bce(logits, ytr_t[idx])
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    with torch.no_grad():
        prob_te = torch.sigmoid(model(torch.tensor(Xte, device=device))).cpu().numpy()
        prob_tr = torch.sigmoid(model(Xtr_t)).cpu().numpy()

    # threshold chosen on TRAIN probs to maximize F1 (no test leakage)
    best_t, best_f1 = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(ytr, (prob_tr > t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    pred = (prob_te > best_t).astype(int)
    return pred, yte, best_t


skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
accs, precs, recs, f1s = [], [], [], []
all_cm = np.zeros((2, 2), dtype=int)
for fold, (tr, te) in enumerate(skf.split(X, y, groups)):
    if y[te].sum() == 0:
        print(f"fold {fold}: no positives in test, skip")
        continue
    pred, yte, t = train_eval(X[tr], y[tr], X[te], y[te])
    acc = accuracy_score(yte, pred)
    pr = precision_score(yte, pred, zero_division=0)
    rc = recall_score(yte, pred, zero_division=0)
    f1 = f1_score(yte, pred, zero_division=0)
    cm = confusion_matrix(yte, pred, labels=[0, 1])
    all_cm += cm
    accs.append(acc); precs.append(pr); recs.append(rc); f1s.append(f1)
    print(f"fold {fold}: thr={t:.2f} acc={acc:.3f} prec={pr:.3f} rec={rc:.3f} f1={f1:.3f} "
          f"pos_test={int(yte.sum())} cm={cm.tolist()}")

print("\n=== 5-fold purged stratified group CV ===")
print(f"Accuracy  {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
print(f"Precision {np.mean(precs):.4f}")
print(f"Recall    {np.mean(recs):.4f}")
print(f"F1        {np.mean(f1s):.4f}")
print("Aggregated confusion matrix [rows=true, cols=pred]:\n", all_cm)
