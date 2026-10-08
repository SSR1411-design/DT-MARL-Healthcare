"""Diagnostic v2: engineered causal features + honest val-threshold + purged group CV."""
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

torch.manual_seed(0); np.random.seed(0)
CSV = ROOT.parent / "simulation" / "failure_history.csv"
SEQ = 10
df = pd.read_csv(CSV)
ALL = ["cpu","ram","bandwidth","energy","runningTasks","active","degraded",
       "linkUp","linkBandwidthMbps","linkLatencyMs","linkPacketLoss","underAttack"]
var = df[ALL].var(); BASE = [c for c in ALL if var[c] > 1e-12]


def engineer(g):
    g = g.sort_values("time").reset_index(drop=True)
    X = g[BASE].values.astype(np.float64)
    d = np.diff(X, axis=0, prepend=X[[0]])              # causal per-feature delta
    active = g["active"].values
    linkUp = g["linkUp"].values
    tsa = np.zeros(len(g)); c = 0
    for i in range(len(g)):
        c = 0 if active[i] == 1 else c + 1
        tsa[i] = c
    tsl = np.zeros(len(g)); c = 0                         # time since link last up
    for i in range(len(g)):
        c = 0 if linkUp[i] == 1 else c + 1
        tsl[i] = c
    return np.concatenate([X, d, tsa[:, None], tsl[:, None]], axis=1).astype(np.float32)


Xs, ys, groups = [], [], []
for nid, g in df.groupby("nodeId"):
    F = engineer(g)
    t = g.sort_values("time").reset_index(drop=True)["willFailSoon"].values.astype(np.int64)
    for i in range(len(F) - SEQ + 1):
        end = i + SEQ - 1
        Xs.append(F[i:i + SEQ]); ys.append(t[end]); groups.append(nid * 100000 + end // SEQ)
X = np.stack(Xs); y = np.array(ys); groups = np.array(groups)
print("windows", X.shape, "positives", int(y.sum()))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run(Xtr, ytr, Xva, yva, Xte, yte, epochs=60):
    nf = Xtr.shape[-1]
    sc = StandardScaler().fit(Xtr.reshape(-1, nf))
    tr = sc.transform(Xtr.reshape(-1, nf)).reshape(Xtr.shape).astype(np.float32)
    va = sc.transform(Xva.reshape(-1, nf)).reshape(Xva.shape).astype(np.float32)
    te = sc.transform(Xte.reshape(-1, nf)).reshape(Xte.shape).astype(np.float32)
    m = FailurePredictor(num_features=nf).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos = max(int(ytr.sum()), 1); neg = len(ytr) - pos
    pw = torch.tensor(min(neg / pos, 10.0), dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.tensor(tr, device=device); yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    best_state, best_vf1 = None, -1
    n = len(Xt); bs = 64
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=device)
        for s in range(0, n, bs):
            idx = perm[s:s + bs]; opt.zero_grad()
            loss = crit(m(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        sch.step()
        m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(torch.tensor(va, device=device))).cpu().numpy()
        bf = max((f1_score(yva, (pv > t).astype(int), zero_division=0) for t in np.linspace(0.05,0.95,19)), default=0)
        if bf >= best_vf1:
            best_vf1 = bf; best_state = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        pv = torch.sigmoid(m(torch.tensor(va, device=device))).cpu().numpy()
        pt = torch.sigmoid(m(torch.tensor(te, device=device))).cpu().numpy()
    bt, bf = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(yva, (pv > t).astype(int), zero_division=0)
        if f > bf: bf, bt = f, t
    return (pt > bt).astype(int), yte, bt


skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=1)
folds = list(skf.split(X, y, groups))
A, P, R, F = [], [], [], []; CM = np.zeros((2, 2), int)
for k, (tr, te) in enumerate(folds):
    if y[te].sum() == 0:
        print(f"fold {k}: no test positives, skip"); continue
    # carve validation from train via another stratified group split
    gtr = groups[tr]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=2)
    itr, iva = next(inner.split(X[tr], y[tr], gtr))
    pred, yte, t = run(X[tr][itr], y[tr][itr], X[tr][iva], y[tr][iva], X[te], y[te])
    a=accuracy_score(yte,pred); p=precision_score(yte,pred,zero_division=0)
    r=recall_score(yte,pred,zero_division=0); f=f1_score(yte,pred,zero_division=0)
    cm=confusion_matrix(yte,pred,labels=[0,1]); CM+=cm
    A.append(a);P.append(p);R.append(r);F.append(f)
    print(f"fold {k}: thr={t:.2f} acc={a:.3f} prec={p:.3f} rec={r:.3f} f1={f:.3f} pos={int(yte.sum())} cm={cm.tolist()}")
print("\n=== engineered features, 5-fold purged group CV, val-threshold ===")
print(f"Acc {np.mean(A):.4f}  Prec {np.mean(P):.4f}  Rec {np.mean(R):.4f}  F1 {np.mean(F):.4f}")
print("Aggregate CM [true x pred]:\n", CM)
