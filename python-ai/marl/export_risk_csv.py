"""
Export the per-(node, tick) predicted_failure_risk table the Java simulation
reads through RiskCsvPredictionGateway.

    python marl/export_risk_csv.py                    # oof scores (default)
    python marl/export_risk_csv.py --risk-source zero # the ablation
    python marl/export_risk_csv.py --out /tmp/r.csv

WHAT THIS IS FOR. The Java side has a PredictionGateway seam that Sprint 5
filled with an inert placeholder returning NEUTRAL for every node. This script
writes the file that lets a real predicted risk cross that seam, so the Java
simulation can be run with prediction available to it.

WHY A FILE AND NOT A LIVE CALL. The predictor is a PyTorch model; the
simulation is a JVM process. A file is the smallest honest coupling that does
not require embedding one runtime in the other, and it keeps the exported
values auditable after the fact. The cost is that this is a REPLAY of scores
computed over an already-recorded trace, not closed-loop co-simulation — see
the SCOPE note printed at the end of the run.

CAUSALITY. Row (n, t) carries the score of the window ENDING at tick t, so it
is computed from observations at or before t and nothing later. The Java
gateway does a step-hold on the most recent row at or before the current clock
and never interpolates forward, so no future value can be read at time t. Ticks
before `first_valid_tick` (the predictor needs sequence_length observations
before it can score anything) are OMITTED rather than filled, because inventing
a warm-up value would be fabricating a prediction.

The `prediction_uncertainty` column is written as an explicit zero. It is a
RESERVED SLOT for Sprint 6.5, not an estimate: no uncertainty quantification
has been performed, and the risk itself is an UNCALIBRATED sigmoid pooling five
cross-validation folds. It is exported so the schema does not change when a
real uncertainty estimate arrives.

DEFAULT SOURCE is "oof": every window scored by a model that never trained on
its temporal block. `--risk-source model` uses the refit deployment checkpoint,
which is IN SAMPLE for ~4/5 of this trace and therefore optimistic; the script
says so on stdout and stamps it into the file header comment.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.config import EnvConfig, REPO                     # noqa: E402
from marl.risk_provider import build_risk_provider          # noqa: E402

DEFAULT_OUT = REPO / "simulation" / "predicted_risk.csv"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Export predicted_failure_risk for the Java simulation")
    p.add_argument("--risk-source", choices=["oof", "model", "zero"],
                   default="oof")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--trace", default=None,
                   help="override the Digital Twin trace CSV")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = EnvConfig()
    cfg.risk_source = args.risk_source
    if args.trace:
        cfg.trace_csv = args.trace

    provider = build_risk_provider(cfg)
    times = np.sort(pd.read_csv(cfg.trace_csv, usecols=["time"]).time.unique())
    n_nodes, n_ticks = provider.risk.shape
    if n_ticks != len(times):
        raise SystemExit(
            f"risk matrix has {n_ticks} ticks but the trace has {len(times)} "
            f"— refusing to write a misaligned export")

    first = provider.first_valid_tick

    print("=" * 74)
    print("EXPORT predicted_failure_risk -> Java")
    print("=" * 74)
    print(f"  trace        : {cfg.trace_csv}")
    print(f"  {provider.summary()}")
    print(f"  notes        : {provider.notes}")

    # Rows ordered by time then node. The gateway requires one shared ascending
    # time axis, so every valid tick emits a row for every node.
    rows = []
    for c in range(first, n_ticks):
        t = float(times[c])
        for n in range(n_nodes):
            v = provider.risk[n, c]
            if np.isnan(v):
                raise SystemExit(
                    f"hole in the risk matrix at node {n}, tick {c} "
                    f"(t={t}) — regenerate the out-of-fold artifact")
            rows.append((n, t, float(v), 0.0))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["nodeId", "time",
                                     "predicted_failure_risk",
                                     "prediction_uncertainty"])
    df.to_csv(out, index=False, float_format="%.6f")

    print(f"  nodes        : {n_nodes}")
    print(f"  ticks        : {n_ticks - first} exported of {n_ticks} "
          f"(ticks 0..{first - 1} omitted: predictor warm-up, "
          f"sequence_length={cfg.sequence_length})")
    print(f"  time range   : {times[first]:.1f}s .. {times[-1]:.1f}s")
    print(f"  rows         : {len(df)}")
    print(f"  written      : {out}")
    print()
    print("  SCOPE. This is a REPLAY of scores computed over an "
          "already-recorded trace,")
    print("  not closed-loop co-simulation: the simulation that produced the "
          "trace did not")
    print("  itself see these predictions. Values are UNCALIBRATED "
          "predicted_failure_risk,")
    print("  not probabilities, and prediction_uncertainty is a reserved zero, "
          "not an estimate.")
    print()
    print("  The Java side picks this up automatically via "
          "PredictionGateways.fromDefaultLocation;")
    print("  delete the file to restore the inert Sprint 5 gateway exactly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
