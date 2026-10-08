"""Diagnostic v3: compact models vs heavy, engineered features, pooled grouped-CV."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from models.failure_predictor import FailurePredictor

torch.manual_seed(0); np.random.seed(0)
df = pd.read_csv(ROOT.parent / "simulation" / "failure_history.csv")
ALL = ["cpu","ram","bandwidth","energy","runningTasks","active","degraded",
       "linkUp","linkBandwidthMbps","linkLatencyMs","linkPacketLoss","underAttack"]
BASE = [c for c in ALL if df[c].var() > 1e-12]
SEQ = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def engineer(g):
    g = g.sort_values("time").reset_index(drop=True)
    X = g[BASE].values.astype(np.float64)
    d = np.diff(X, axis=0, prepend=X[[0]])
    tsa = np.zeros(len(g)); c = 0
    for i, a in enumerate(g["active"].values):
        c = 0 if a == 1 else c + 1; tsa[i] = c
    tsl = np.zeros(len(g)); c = 0
    for i, a in enumerate(g["linkUp"].values):
        c = 0 if a == 1 else c + 1; tsl[i] = c
    return np.concatenate([X, d, tsa[:, None], tsl[:, None]], axis=1).astype(np.float32)


Xs, ys, groups = [], [], []
for nid, g in df.groupby("nodeId"):
    F = engineer(g)
    t = g.sort_values("time").reset_index(drop=True)["willFailSoon"].values.astype(np.int64)
    for i in range(len(F) - SEQ + 1):
        end = i + SEQ - 1
        Xs.append(F[i:i+SEQ]); ys.append(t[end]); groups.append(nid*100000 + end//SEQ)
X = np.stack(Xs); y = np.array(ys); groups = np.array(groups)


class CompactGRU(nn.Module):
    def __init__(self, nf, hid=32):
        super().__init__()
        self.gru = nn.GRU(nf, hid, batch_first=True, bidirectional=True)
        self.att = nn.Linear(2*hid, 1)
        self.head = nn.Sequential(nn.LayerNorm(2*hid), nn.Linear(2*hid, 32),
                                  nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
    def forward(self, x):
        h, _ = self.gru(x)
        w = torch.softmax(self.att(h), dim=1)
        return self.head((w*h).sum(1)).squeeze(-1)


def train_nn(build, Xtr, ytr, Xva, yva, Xte, epochs=80):
    nf = Xtr.shape[-1]
    sc = StandardScaler().fit(Xtr.reshape(-1, nf))
    f = lambda A: sc.transform(A.reshape(-1, nf)).reshape(A.shape).astype(np.float32)
    tr, va, te = f(Xtr), f(Xva), f(Xte)
    m = build(nf).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pw = torch.tensor(min((len(ytr)-ytr.sum())/max(ytr.sum(),1), 8.0), dtype=torch.float32, device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xt = torch.tensor(tr, device=device); yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    best, bvf = None, -1
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xt), device=device)
        for s in range(0, len(Xt), 64):
            idx = perm[s:s+64]; opt.zero_grad()
            crit(m(Xt[idx]), yt[idx]).backward(); opt.step()
        sch.step(); m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(torch.tensor(va, device=device))).cpu().numpy()
        vf = max((f1_score(yva,(pv>t).astype(int),zero_division=0) for t in np.linspace(.05,.95,19)),default=0)
        if vf >= bvf: bvf, best = vf, {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best); m.eval()
    with torch.no_grad():
        pv = torch.sigmoid(m(torch.tensor(va, device=device))).cpu().numpy()
        pt = torch.sigmoid(m(torch.tensor(te, device=device))).cpu().numpy()
    bt, bf = .5, -1
    for t in np.linspace(.05,.95,91):
        s = f1_score(yva,(pv>t).astype(int),zero_division=0)
        if s>bf: bf,bt = s,t
    return pt, bt


def eval_model(tag, kind):
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=3)
    pooled_pred, pooled_true = [], []
    for tr, te in skf.split(X, y, groups):
        if y[te].sum() == 0: continue
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=4)
        itr, iva = next(inner.split(X[tr], y[tr], groups[tr]))
        Xtr,ytr = X[tr][itr],y[tr][itr]; Xva,yva = X[tr][iva],y[tr][iva]
        if kind == "logistic":
            nf = X.shape[-1]
            sc = StandardScaler().fit(Xtr.reshape(-1,nf))
            flat = lambda A: sc.transform(A.reshape(-1,nf)).reshape(len(A),-1)
            lr = LogisticRegression(max_iter=2000, class_weight="balanced")
            lr.fit(flat(Xtr), ytr)
            pv = lr.predict_proba(flat(Xva))[:,1]; pt = lr.predict_proba(flat(X[te]))[:,1]
            bt,bf=.5,-1
            for t in np.linspace(.05,.95,91):
                s=f1_score(yva,(pv>t).astype(int),zero_division=0)
                if s>bf: bf,bt=s,t
        else:
            build = (lambda nf: CompactGRU(nf)) if kind=="gru" else (lambda nf: FailurePredictor(num_features=nf))
            pt, bt = train_nn(build, Xtr, ytr, Xva, yva, X[te])
        pooled_pred.append((pt>bt).astype(int)); pooled_true.append(y[te])
    yp=np.concatenate(pooled_pred); yt=np.concatenate(pooled_true)
    cm=confusion_matrix(yt,yp,labels=[0,1])
    print(f"{tag:22s} acc={accuracy_score(yt,yp):.4f} prec={precision_score(yt,yp,zero_division=0):.3f} "
          f"rec={recall_score(yt,yp,zero_division=0):.3f} f1={f1_score(yt,yp,zero_division=0):.3f} cm={cm.tolist()}")


eval_model("Logistic(flatten)", "logistic")
eval_model("CompactGRU+attn", "gru")
eval_model("Heavy Transf+BiLSTM", "heavy")
