"""Experiment: robust threshold via inner-CV OOF on train; report pooled test metrics."""
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from data.failure_dataset import build_windows, Standardizer, NUM_FEATURES
from models.failure_predictor import FailurePredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X, y, nid, gid = build_windows(str(ROOT.parent / "simulation" / "failure_history.csv"), 10)


def fit(Xtr, ytr, Xva, yva, seed, epochs=80):
    torch.manual_seed(seed)
    std = Standardizer().fit(Xtr)
    Xt = torch.tensor(std.transform(Xtr), device=device)
    yt = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xv = torch.tensor(std.transform(Xva), device=device)
    m = FailurePredictor(num_features=NUM_FEATURES).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    pos = max(int(ytr.sum()), 1); neg = len(ytr) - pos
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min(neg/pos, 8.0), device=device))
    best, bvf = None, -1
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(Xt), device=device)
        for s in range(0, len(Xt), 64):
            idx = perm[s:s+64]; opt.zero_grad(); crit(m(Xt[idx]), yt[idx]).backward(); opt.step()
        sch.step(); m.eval()
        with torch.no_grad():
            pv = torch.sigmoid(m(Xv)).cpu().numpy()
        vf = max((f1_score(yva,(pv>t).astype(int),zero_division=0) for t in np.linspace(.05,.95,19)),default=0)
        if vf >= bvf: bvf, best = vf, {k:v.clone() for k,v in m.state_dict().items()}
    m.load_state_dict(best); m.eval()
    return m, std


def prob(m, std, Xa):
    with torch.no_grad():
        return torch.sigmoid(m(torch.tensor(std.transform(Xa), device=device))).cpu().numpy()


def outer(threshold_mode):
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    P, T = [], []
    for tr_all, te in skf.split(X, y, gid):
        if y[te].sum() == 0: continue
        # inner grouped CV: collect OOF train probs to choose a stable threshold
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=43)
        oof_prob = np.full(len(tr_all), np.nan);
        last_model = None
        splits = list(inner.split(X[tr_all], y[tr_all], gid[tr_all]))
        for itr, iva in splits:
            m, std = fit(X[tr_all][itr], y[tr_all][itr], X[tr_all][iva], y[tr_all][iva], seed=42)
            oof_prob[iva] = prob(m, std, X[tr_all][iva])
        yoof = y[tr_all]
        if threshold_mode == "oof":
            bt, bf = .5, -1
            for t in np.linspace(.05,.95,91):
                f = f1_score(yoof,(oof_prob>t).astype(int),zero_division=0)
                if f > bf: bf, bt = f, t
        else:
            bt = 0.5
        # final model on all train, apply chosen threshold to test
        i_tr, i_va = splits[0]
        m, std = fit(X[tr_all], y[tr_all], X[tr_all][i_va], y[tr_all][i_va], seed=42)
        pt = prob(m, std, X[te])
        P.append((pt > bt).astype(int)); T.append(y[te])
    yp, yt = np.concatenate(P), np.concatenate(T)
    cm = confusion_matrix(yt, yp, labels=[0,1])
    print(f"[threshold={threshold_mode}] acc={accuracy_score(yt,yp)*100:.2f}% "
          f"prec={precision_score(yt,yp,zero_division=0):.3f} rec={recall_score(yt,yp,zero_division=0):.3f} "
          f"f1={f1_score(yt,yp,zero_division=0):.3f} cm={cm.tolist()}")


outer("oof")
outer("half")
