"""
predicted_failure_risk provider.

The MARL state receives a CONTINUOUS risk value per node per tick. Nothing in
this module thresholds, rounds, or binarises it — the 0/1 decision only ever
happens inside the classification-metric code in `training/`, never here.

TERMINOLOGY. The value is the sigmoid of the predictor's single output logit.
It is an UNCALIBRATED score in [0, 1]: no Platt scaling, isotonic regression
or temperature fitting has been performed, and the out-of-fold scores pool
five folds with five different operating points. It is therefore called
`predicted_failure_risk` throughout, never "failure probability" and never
"calibrated probability".

THREE SOURCES

  "oof"   (default) — the leakage-safe out-of-fold sigmoid scores saved by
          training/train_failure_predictor.py. Every window was scored by a
          model that never trained on that window's temporal block, so the
          risk the agent sees is an honest held-out estimate for all 1491
          windows of all 10 nodes. This is the right source for research
          results.

  "model" — live inference with saved_models/failure_predictor.pth. That
          checkpoint is the deployment model refit on everything, so it is
          IN SAMPLE for ~4/5 of this trace and its risk values are
          optimistic. Provided for the deployment path, flagged in the log.

  "zero"  — the risk channel forced to 0.0 everywhere. The ablation that
          answers "does the policy actually use the prediction, or would it
          behave identically without it?".

ALIGNMENT. Window i of node n ends at tick index i + sequence_length - 1, so
risk is defined for tick indices >= sequence_length - 1 and undefined (NaN)
before that. `first_valid_tick` exposes this and the environment refuses to
start an episode earlier.
"""

from pathlib import Path
import json
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from data.failure_dataset import (  # noqa: E402
    OBSERVABLE_COLUMNS, build_windows, Standardizer,
)


class RiskProvider:
    """
    Dense (n_nodes, n_ticks) matrix of continuous predicted_failure_risk,
    plus a matching `uncertainty` matrix.

    The uncertainty matrix is a RESERVED SLOT for Sprint 6.5. In Sprint 6 it
    is identically zero and no code branches on it: it exists so that adding
    real predictive uncertainty later is a change of one provider, not a
    redesign of the state representation. It is exposed as an observation
    channel now precisely so the observation dimension does not change later.
    """

    def __init__(self, risk, uncertainty, first_valid_tick, source, notes):
        self.risk = risk                        # (N, T) float32, NaN before start
        self.uncertainty = uncertainty          # (N, T) float32
        self.first_valid_tick = int(first_valid_tick)
        self.source = source
        self.notes = notes
        self.calibrated = False                 # never claim otherwise

    def at(self, node: int, tick: int) -> float:
        v = self.risk[node, tick]
        return 0.0 if np.isnan(v) else float(v)

    def uncertainty_at(self, node: int, tick: int) -> float:
        v = self.uncertainty[node, tick]
        return 0.0 if np.isnan(v) else float(v)

    def summary(self) -> str:
        m = self.risk[:, self.first_valid_tick:]
        return (f"risk_source={self.source} calibrated={self.calibrated} "
                f"min={np.nanmin(m):.4f} max={np.nanmax(m):.4f} "
                f"mean={np.nanmean(m):.4f} "
                f"distinct={len(np.unique(np.round(m, 4)))} "
                f"first_valid_tick={self.first_valid_tick}")


def _empty(n_nodes, n_ticks):
    return np.full((n_nodes, n_ticks), np.nan, dtype=np.float32)


def build_risk_provider(cfg) -> RiskProvider:
    """Build the provider described by `cfg` (an EnvConfig)."""
    source = cfg.risk_source

    if source == "zero":
        # Needs the trace geometry only.
        import pandas as pd
        df = pd.read_csv(cfg.trace_csv, usecols=["time", "nodeId"])
        n = df.nodeId.nunique()
        t = df.time.nunique()
        z = np.zeros((n, t), dtype=np.float32)
        return RiskProvider(z, np.zeros_like(z), cfg.sequence_length - 1,
                            "zero",
                            "ABLATION: risk channel forced to 0 everywhere.")

    # Both real sources need the exact same window geometry the predictor was
    # trained on, so they are rebuilt with the same function.
    X, y, node_ids, event_ids, end_times = build_windows(
        cfg.trace_csv, cfg.sequence_length,
        raw_features=OBSERVABLE_COLUMNS, return_times=True)

    uniq_nodes = np.sort(np.unique(node_ids))
    uniq_times = np.sort(np.unique(end_times))
    node_pos = {int(v): i for i, v in enumerate(uniq_nodes)}

    # Tick index space is the FULL trace clock, not just window ends.
    import pandas as pd
    all_times = np.sort(pd.read_csv(cfg.trace_csv, usecols=["time"]).time.unique())
    tick_pos = {round(float(v), 4): i for i, v in enumerate(all_times)}

    risk = _empty(len(uniq_nodes), len(all_times))

    if source == "oof":
        d = np.load(cfg.oof_npz)
        prob, covered = d["prob"], d["covered"]
        if len(prob) != len(y):
            raise ValueError(
                f"out-of-fold artifact has {len(prob)} windows but the trace "
                f"produces {len(y)} — regenerate with "
                f"training/train_failure_predictor.py before using risk_source=oof")
        # Ordering integrity: the npz stores no times, so prove the row order
        # matches by checking the label / node / block vectors element-wise.
        for name, a, b in (("y", d["y"], y),
                           ("node_ids", d["node_ids"], node_ids),
                           ("event_ids", d["event_ids"], event_ids)):
            if not np.array_equal(a, b):
                raise ValueError(
                    f"out-of-fold artifact does not align with the trace "
                    f"({name} differs) — regenerate it")
        if not covered.all():
            raise ValueError("out-of-fold artifact has uncovered windows")
        scores = prob.astype(np.float32)
        notes = ("Leakage-safe out-of-fold sigmoid scores: every window scored "
                 "by a model that never trained on its temporal block. "
                 "UNCALIBRATED (pools 5 folds).")
    elif source == "model":
        scores = _live_inference(cfg, X)
        notes = ("Live inference with the refit deployment checkpoint. IN "
                 "SAMPLE for ~4/5 of this trace; risk values are optimistic. "
                 "UNCALIBRATED sigmoid of a single logit.")
    else:
        raise ValueError(f"unknown risk_source: {source!r}")

    for k in range(len(scores)):
        r = node_pos[int(node_ids[k])]
        c = tick_pos[round(float(end_times[k]), 4)]
        risk[r, c] = scores[k]

    first_valid = cfg.sequence_length - 1
    if np.isnan(risk[:, first_valid:]).any():            # pragma: no cover
        raise ValueError("risk matrix has holes after the first valid tick")

    return RiskProvider(risk, np.zeros_like(risk), first_valid, source, notes)


def _live_inference(cfg, X):
    """Sigmoid of the saved checkpoint's logit, using the saved scaler."""
    import torch
    from models.failure_predictor import FailurePredictor

    meta = json.load(open(cfg.meta_path))
    if meta["num_features"] != X.shape[-1]:
        raise ValueError(
            f"checkpoint expects {meta['num_features']} features, trace "
            f"produces {X.shape[-1]}")

    std = Standardizer.load(cfg.scaler_path)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FailurePredictor(num_features=X.shape[-1]).to(dev)
    model.load_state_dict(torch.load(cfg.model_path, map_location=dev))
    model.eval()

    out = []
    with torch.no_grad():
        for s in range(0, len(X), 4096):
            xb = torch.tensor(std.transform(X[s:s + 4096]), device=dev)
            # The model returns a single LOGIT; sigmoid is applied here and
            # only here. Documented because the checkpoint itself is not a
            # probability model.
            out.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


__all__ = ["RiskProvider", "build_risk_provider"]
