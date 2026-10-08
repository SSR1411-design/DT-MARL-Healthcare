"""
Loader for the recorded Digital Twin trace exported by the Java simulation.

The Java run writes `simulation/failure_history.csv`: one row per (tick, node)
with the 12 observable telemetry channels, the `willFailSoon` label, and an
audit_* block. This module loads ONLY the observable channels into dense
(n_nodes, n_ticks) matrices and refuses to load anything else — the label and
the audit columns are dropped at parse time so that no later code can reach
them by accident.

What the environment uses each matrix for:

  * `active`                  -> the environment's availability physics. A
                                 host the simulator recorded as down at tick t
                                 is down at tick t. This is the transition
                                 function, NOT an observation: an agent never
                                 sees active[j, t'] for t' > t.
  * everything else           -> observation channels at the current tick.

`assert_causal_slice` is the runtime guard: it re-reads the CSV with all rows
after a cut tick perturbed and checks that the observation matrices up to the
cut are bit-identical.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from data.failure_dataset import (  # noqa: E402
    OBSERVABLE_COLUMNS, FORBIDDEN_COLUMNS, assert_no_leakage,
)

# The channels the environment is allowed to see. Deliberately identical to
# the predictor's whitelist, so "observable" means one thing in this repo.
TRACE_CHANNELS = list(OBSERVABLE_COLUMNS)
assert_no_leakage(TRACE_CHANNELS)


class Trace:
    """Dense observable telemetry, indexed [node, tick]."""

    def __init__(self, channels, times, node_ids):
        self.channels = channels                # dict name -> (N, T) float32
        self.times = times                      # (T,) simulation clock
        self.node_ids = node_ids                # (N,) node id per row index
        self.n_nodes = len(node_ids)
        self.n_ticks = len(times)

    # -- accessors ---------------------------------------------------------

    def ch(self, name: str) -> np.ndarray:
        return self.channels[name]

    def is_up(self, node: int, tick: int) -> bool:
        return bool(self.channels["active"][node, tick] >= 0.5)

    def up_during(self, node: int, tick0: int, tick1: int) -> bool:
        """
        True only if the node was recorded active for EVERY tick in
        [tick0, tick1). A decision step spans several recorded ticks, and a
        host that went down for any of them was genuinely unavailable — so
        sub-sampling the trace cannot hide an outage.
        """
        a = self.channels["active"][node, tick0:max(tick1, tick0 + 1)]
        return bool(a.size and np.all(a >= 0.5))

    def any_down_during(self, node: int, tick0: int, tick1: int) -> bool:
        a = self.channels["active"][node, tick0:max(tick1, tick0 + 1)]
        return bool(a.size and np.any(a < 0.5))


def load_trace(csv_path, perturb_after_tick=None, rng_seed=0) -> Trace:
    """
    Load the observable trace.

    `perturb_after_tick` is a TEST HOOK: every observable value at a tick
    index strictly greater than the given index is replaced with noise. Used
    by the no-future-information test — if any observation at or before the
    cut changes, the environment is reading ahead.
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in TRACE_CHANNELS if c not in df.columns]
    if missing:
        raise ValueError(f"trace is missing observable columns: {missing}")

    # Hard drop of the label and every audit column before anything else
    # touches the frame.
    keep = ["time", "nodeId"] + TRACE_CHANNELS
    dropped = [c for c in df.columns if c not in keep]
    forbidden_kept = [c for c in keep
                      if c in FORBIDDEN_COLUMNS and c not in ("time", "nodeId")]
    if forbidden_kept:                                   # pragma: no cover
        raise ValueError(f"refusing to load forbidden columns: {forbidden_kept}")
    df = df[keep].copy()

    node_ids = np.sort(df.nodeId.unique())
    times = np.sort(df.time.unique())
    n, t = len(node_ids), len(times)

    node_pos = {int(v): i for i, v in enumerate(node_ids)}
    tick_pos = {float(v): i for i, v in enumerate(times)}

    ri = df.nodeId.map(node_pos).values
    ci = df.time.map(tick_pos).values

    channels = {}
    for name in TRACE_CHANNELS:
        m = np.full((n, t), np.nan, dtype=np.float32)
        m[ri, ci] = df[name].values.astype(np.float32)
        if np.isnan(m).any():
            # Forward-fill along time so a missing (node, tick) cell cannot
            # silently become a NaN observation.
            for i in range(n):
                row = m[i]
                bad = np.isnan(row)
                if bad.any():
                    good = np.flatnonzero(~bad)
                    if not good.size:
                        row[:] = 0.0
                    else:
                        row[bad] = np.interp(np.flatnonzero(bad), good, row[good])
        channels[name] = m

    if perturb_after_tick is not None:
        rng = np.random.default_rng(rng_seed)
        cut = int(perturb_after_tick)
        for name, m in channels.items():
            if cut + 1 < t:
                m[:, cut + 1:] = rng.random(
                    (n, t - cut - 1)).astype(np.float32) * 100.0

    tr = Trace(channels, times.astype(np.float64), node_ids.astype(np.int64))
    if dropped:
        tr.dropped_columns = dropped
    return tr


def load_failure_events(log_path, types=("HOST_FAILURE",)) -> pd.DataFrame:
    """
    Ground-truth failure events. FOR EVALUATION ONLY.

    Used after an episode to answer "was this migration preemptive, i.e. did
    it move a task off a node that later actually failed?". The environment
    never reads this during `step`, and it is not an observation, a reward
    term, or a policy input. Any function that consumes it lives in the
    metrics path.
    """
    log = pd.read_csv(log_path)
    ev = log[log.type.isin(types)][["time", "nodeId", "type"]]
    return ev.sort_values("time").reset_index(drop=True)


__all__ = ["Trace", "load_trace", "load_failure_events", "TRACE_CHANNELS"]
