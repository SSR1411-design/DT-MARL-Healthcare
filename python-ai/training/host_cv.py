"""
The host failure-prediction evaluation PROTOCOL, in one place.

Every script that reports a host-predictor number imports from here, so that
"the same evaluation methodology" is enforced by construction rather than by
three copies of the same loop drifting apart.

The protocol, unchanged from the finalisation cycle:

  1. Windows are grouped into (node, temporal-block) groups; the outer split
     is StratifiedGroupKFold over those groups.
  2. After the split, purge_overlap drops every TRAINING window that shares
     any tick with a TEST window of the same node. Grouping alone is not
     enough: adjacent sliding windows overlap by SEQ_LEN-1 ticks.
  3. The scaler is fit on TRAINING windows only, then applied to val/test.
  4. An inner grouped split carves a validation slice out of the training
     windows. It is used for early stopping AND for threshold selection.
     Test labels are never involved in either.
  5. Headline metrics are POOLED OUT-OF-FOLD: every window is scored exactly
     once, by a model that never saw that window's block.

Ranking metrics (PR-AUC / ROC-AUC) are computed on the pooled out-of-fold
probabilities. Those come from K different models, so the pooled score
mixes K calibrations; it is the standard cross-validated estimate and is
reported as such, not as a single model's calibrated score.
"""

from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score, roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.failure_dataset import Standardizer, purge_overlap  # noqa: E402
from models.failure_predictor import FailurePredictor  # noqa: E402

# ---- protocol constants (identical to the finalisation cycle) ----
SEED = 42
SEQ_LEN = 10
EPOCHS = 80
BATCH = 64
POS_WEIGHT_CAP = 8.0
N_SPLITS = 5
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)
EARLYSTOP_GRID = np.linspace(0.05, 0.95, 19)

CSV_PATH = ROOT.parent / "simulation" / "failure_history.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_one(Xtr, ytr, Xva, yva, epochs=EPOCHS, num_features=None):
    """
    Train one model. Scaler is fit on Xtr ONLY. Best checkpoint is the epoch
    with the highest achievable validation F1 (early stopping on val, never
    on test). Returns (model, fitted_standardizer).
    """
    num_features = num_features or Xtr.shape[-1]

    std = Standardizer().fit(Xtr)
    Xtr_n = torch.tensor(std.transform(Xtr), device=device)
    Xva_n = torch.tensor(std.transform(Xva), device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)

    model = FailurePredictor(num_features=num_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    pos = max(int(ytr.sum()), 1)
    neg = len(ytr) - pos
    pw = torch.tensor(min(neg / pos, POS_WEIGHT_CAP),
                      dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    best_state, best_vf1 = None, -1.0
    n = len(Xtr_n)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, BATCH):
            idx = perm[s:s + BATCH]
            opt.zero_grad()
            loss = criterion(model(Xtr_n[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva_n)).cpu().numpy()
        vf1 = max(
            (f1_score(yva, (pv > t).astype(int), zero_division=0)
             for t in EARLYSTOP_GRID),
            default=0.0,
        )
        if vf1 >= best_vf1:
            best_vf1 = vf1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    return model, std


def predict_proba(model, std, X, chunk=4096):
    """Sigmoid scores for X, chunked so a 15k-window scan fits in VRAM."""
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            xb = torch.tensor(std.transform(X[s:s + chunk]), device=device)
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def pick_threshold(model, std, Xva, yva, verbose=True):
    """
    Threshold that maximises F1 on the VALIDATION slice. Never sees test
    labels. Warns when the winner sits on a grid boundary, which means F1 was
    still improving as the threshold left the searched range — a symptom of
    positive scores barely separated from negative ones.
    """
    pv = predict_proba(model, std, Xva)
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLD_GRID:
        f = f1_score(yva, (pv > t).astype(int), zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    if verbose and best_t in (THRESHOLD_GRID[0], THRESHOLD_GRID[-1]):
        print(f"  WARNING: threshold pinned at search-grid boundary "
              f"({best_t:.2f}); val F1={best_f1:.3f}. Scores poorly "
              f"separated - distinct probabilities: "
              f"{len(np.unique(np.round(pv, 4)))}")
    return best_t, best_f1


def grouped_oof(X, y, node_ids, event_ids, n_splits=N_SPLITS, epochs=EPOCHS,
                seed=SEED, verbose=True):
    """
    Run the leakage-safe grouped CV and return pooled out-of-fold results.

    Returns dict with:
      prob      — (N,) out-of-fold probability per window (NaN if uncovered)
      pred      — (N,) out-of-fold 0/1 per window (-1 if uncovered)
      covered   — (N,) bool mask of windows that got an out-of-fold score
      folds     — per-fold bookkeeping (threshold, sizes, purge count)
    """
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                               random_state=seed)

    prob = np.full(len(y), np.nan, dtype=np.float64)
    pred = np.full(len(y), -1, dtype=np.int64)
    folds = []

    for fold, (tr_all, te) in enumerate(skf.split(X, y, event_ids)):
        if y[te].sum() == 0:
            if verbose:
                print(f"fold {fold}: no positives in test split, skipped")
            continue

        before = len(tr_all)
        tr_all = purge_overlap(tr_all, te, node_ids, event_ids)

        inner = StratifiedGroupKFold(n_splits=4, shuffle=True,
                                     random_state=seed + 1)
        i_tr, i_va = next(inner.split(X[tr_all], y[tr_all], event_ids[tr_all]))
        tr, va = tr_all[i_tr], tr_all[i_va]

        # Hard assertion, not a comment: no test window may share a tick with
        # any training or validation window of the same node.
        assert_no_overlap(tr, te, node_ids, event_ids)
        assert_no_overlap(va, te, node_ids, event_ids)

        model, std = train_one(X[tr], y[tr], X[va], y[va], epochs=epochs)
        thr, vf1 = pick_threshold(model, std, X[va], y[va])

        pt = predict_proba(model, std, X[te])
        prob[te] = pt
        pred[te] = (pt > thr).astype(int)

        folds.append(dict(fold=fold, threshold=float(thr), val_f1=float(vf1),
                          n_train=len(tr), n_val=len(va), n_test=len(te),
                          purged=before - len(tr_all),
                          pos_test=int(y[te].sum())))
        if verbose:
            print(f"fold {fold}: thr={thr:.2f} "
                  f"acc={accuracy_score(y[te], pred[te]):.4f} "
                  f"prec={precision_score(y[te], pred[te], zero_division=0):.3f} "
                  f"rec={recall_score(y[te], pred[te], zero_division=0):.3f} "
                  f"pos_test={int(y[te].sum())} purged={before - len(tr_all)}")

    covered = pred >= 0
    return dict(prob=prob, pred=pred, covered=covered, folds=folds)


def assert_no_overlap(train_idx, test_idx, node_ids, event_ids):
    """Fail loudly if any train window overlaps a test window of same node."""
    test_blocks = {}
    for j in test_idx:
        test_blocks.setdefault(node_ids[j], set()).add(event_ids[j])
    for i in train_idx:
        for tb in test_blocks.get(node_ids[i], ()):
            if abs(event_ids[i] - tb) <= 1:
                raise AssertionError(
                    f"train/test overlap: node {node_ids[i]} "
                    f"block {event_ids[i]} vs test block {tb}")


def full_metrics(y_true, y_pred, y_prob=None):
    """
    The metric set the host predictor is judged on. Accuracy is included but
    is NOT the headline: at a ~2% positive rate the always-negative baseline
    scores ~98%, so accuracy is only interpretable next to `base_rate_acc`.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    n = len(y_true)
    m = dict(
        n=n,
        positives=int(y_true.sum()),
        positive_rate=float(y_true.mean()),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        tn=tn, fp=fp, fn=fn, tp=tp,
        pos_pred_rate=float((np.asarray(y_pred) == 1).mean()),
        base_rate_acc=float(1.0 - y_true.mean()),
    )
    if y_prob is not None and len(np.unique(y_true)) > 1:
        m["pr_auc"] = float(average_precision_score(y_true, y_prob))
        m["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        m["pr_auc_baseline"] = float(y_true.mean())   # random-ranking PR-AUC
    return m


def print_metrics(name, m):
    print(f"\n[{name}]  n={m['n']}  positives={m['positives']} "
          f"({m['positive_rate']*100:.2f}%)")
    print(f"  Accuracy        : {m['accuracy']*100:6.2f}%   "
          f"(always-negative would score {m['base_rate_acc']*100:.2f}%)")
    print(f"  Precision       : {m['precision']:.4f}")
    print(f"  Recall          : {m['recall']:.4f}")
    print(f"  F1-score        : {m['f1']:.4f}")
    if "pr_auc" in m:
        print(f"  PR-AUC          : {m['pr_auc']:.4f}   "
              f"(random ranking = {m['pr_auc_baseline']:.4f})")
        print(f"  ROC-AUC         : {m['roc_auc']:.4f}   (random = 0.5000)")
    print(f"  Pos. pred. rate : {m['pos_pred_rate']*100:.2f}%  "
          f"(alerts raised on this share of all windows)")
    print(f"  Confusion       : TN={m['tn']} FP={m['fp']} "
          f"FN={m['fn']} TP={m['tp']}")
    print(f"                    [[{m['tn']:6d} {m['fp']:5d}]   rows=true 0/1")
    print(f"                     [{m['fn']:6d} {m['tp']:5d}]]  cols=pred 0/1")
