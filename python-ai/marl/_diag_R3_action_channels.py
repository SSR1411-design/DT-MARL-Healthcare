#!/usr/bin/env python
"""
SPRINT 7 R3 -- EXPLORATORY ONLY. Which ACTION CHANNEL carries the risk
response?

THIS IS NOT A PREREGISTERED METRIC AND CANNOT CHANGE THE R3 GO/NO-GO DECISION.
R3's preregistered outcome is NO-GO on P1/P2/P3/P4 and stays NO-GO regardless of
anything measured here. This probe exists because of a specific, narrow gap.

THE GAP. Every preregistered R3 metric is defined on ONE of the four actions:

  P1  P(EDGE | risk>=0.6) - P(EDGE | risk<0.2)     on RANDOM
  P2  the same difference                          on UNION
  P3  Spearman(risk, P(EDGE))                      on RANDOM
  P4  ||g_real|| / ||g_shuffled||, where the reference direction `synth` is
      +1 on MIGRATE_EDGE / -1 on STAY at high risk
  P5  cos(g_real, g_synth), same EDGE-based reference direction

So the whole family scores exactly one hypothesis: "risk response == raise
MIGRATE_TO_NEIGHBOR_EDGE". A policy that responds to predicted_failure_risk by
routing to MIGRATE_TO_CLOUD instead would be scored as NO RESPONSE by all five.
That is a live possibility for R3 specifically, because R3's greedy argmax
selects MIGRATE_EDGE ZERO times at high risk on both neutral state sets while
selecting MIGRATE_CLOUD 56/216 (RANDOM) and 288/3592 (UNION) times.

The existing artifact cannot settle it: SPRINT_7_RUNG2_75_matched_states_*.json
records p_edge at BOTH risk levels but p_stay only at the high-risk level, and
never records p_cloud or p_preempt at either. So the CLOUD channel's risk
response has never been measured for any arm, in any rung.

WHAT THIS ADDS. The same difference-of-means, on the same two neutral state
sets, with the same decision rule, the same risk cut points, the same episode
clustering and the same bootstrap -- computed for ALL FOUR actions instead of
one. Nothing about the methodology changes; the metric is simply not truncated
to a single column any more.

WHY A NEW FILE RATHER THAN AN EDIT. _diag_rung2_75_matched_states.py produced
the preregistered P1/P2/P3 numbers. Editing it to emit more columns would mean
the artifact backing a preregistered decision was written by a script that no
longer exists in that form. This file imports that script's own helpers --
eval_starts, trajectory, random_trajectory, union_source, probs_at,
cluster_ci_diff -- so the state sets are constructed by the SAME code, not by a
reimplementation.

READ IT AS: a question about the next hypothesis, not as evidence about R3's
success. Sum(Delta_a) == 0 identically over the four actions, so a positive
Delta on one channel is always somebody else's negative Delta; the interesting
quantity is WHICH channel absorbs the mass that leaves STAY.

Writes SPRINT_7_R3_action_channels.json. Reads checkpoints, writes no
parameters, constructs no optimiser, touches no production module.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[0]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from marl._diag_rung0 import load_agent_and_cfg, OUT_DIR          # noqa: E402
from marl._diag_rung2_75_matched_states import (                  # noqa: E402
    ACTION_NAMES, MODELS, EXTRA,
    eval_starts, trajectory, random_trajectory, union_source,
    probs_at, cluster_ci_diff, spearman,
)

TAG = "SPRINT_7_R3"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    # every default below is copied from _diag_rung2_75_matched_states.parse_args
    # so the state sets are bit-identical to the ones P1/P2/P3 were scored on.
    p.add_argument("--clusters", type=int, default=32)
    p.add_argument("--start-seed", type=int, default=20260825)
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--random-seed", type=int, default=31337)
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 R3 -- EXPLORATORY: which ACTION CHANNEL carries the risk")
    print("        response?  The preregistered P1/P2/P3 measure only the")
    print("        MIGRATE_EDGE column of this same table.")
    print("=" * 78)
    print("  NOT a preregistered metric. Cannot and does not change R3's")
    print("  NO-GO. Reported as an exploratory observation only.")

    agents, cfgs = {}, {}
    for tag, fn in {**MODELS, **EXTRA}.items():
        a, _, c = load_agent_and_cfg(str(OUT_DIR / fn), args.device, "eval")
        agents[tag], cfgs[tag] = a, c
    ALL = list(MODELS) + list(EXTRA)

    starts, env0 = eval_starts(cfgs["A0"], args.clusters, args.start_seed)
    print(f"\n  eval window [{env0._min_start}, {env0._max_start}]  "
          f"starts {len(starts)}  arms {', '.join(ALL)}")

    # Identical construction to the preregistered probe: A0's and R2's greedy
    # trajectories generate the state sources, RANDOM is uniform-legal, UNION is
    # the pool. R3 is scored, never a source -- promoting it would redefine the
    # sets the thresholds were calibrated on.
    src_A0 = trajectory(agents["A0"], cfgs["A0"], starts)
    src_R2 = trajectory(agents["R2"], cfgs["R2"], starts)
    sources = {
        "RANDOM": random_trajectory(cfgs["A0"], starts, args.random_seed),
        "UNION": union_source(src_A0, src_R2),
    }

    out = {}
    for src, tr in sources.items():
        dec = tr["mask"].sum(-1) > 1.0
        eps_dec = np.repeat(tr["ep"][:, None], tr["n_agents"], axis=1)[dec]
        r_all = tr["risk"][dec]
        lo_m, hi_m = r_all < 0.2, r_all >= 0.6
        print(f"\n-- {src}  decision {int(dec.sum())}  "
              f"risk<0.2 {int(lo_m.sum())}  risk>=0.6 {int(hi_m.sum())} "
              + "-" * 12)
        print(f"   {'arm':>4} {'action':>16} {'P|lo':>8} {'P|hi':>8} "
              f"{'Delta':>9} {'CI95':>20} {'z':>7} {'rho':>8}")
        cell = dict(source=src, n_decision=int(dec.sum()),
                    n_lo=int(lo_m.sum()), n_hi=int(hi_m.sum()), evaluated={})
        for who in ALL:
            P = probs_at(agents[who], tr["obs"], tr["mask"])
            Pd = P[dec]
            per = {}
            for k, name in enumerate(ACTION_NAMES):
                col = Pd[:, k]
                d = float(col[hi_m].mean() - col[lo_m].mean())
                ci = cluster_ci_diff(col, hi_m, lo_m, eps_dec, args.boot,
                                     np.random.default_rng(11))
                rho = spearman(r_all, col)
                per[name] = dict(p_lo=float(col[lo_m].mean()),
                                 p_hi=float(col[hi_m].mean()),
                                 delta=d, delta_cluster=ci,
                                 spearman_risk_vs_p=rho)
                z = ci.get("z") if isinstance(ci, dict) else None
                lo_ci, hi_ci = (ci.get("ci95") or [None, None])[:2] \
                    if isinstance(ci, dict) else (None, None)
                cis = (f"[{lo_ci:+.4f}, {hi_ci:+.4f}]"
                       if lo_ci is not None else " " * 20)
                zs = f"{z:+.2f}" if isinstance(z, (int, float)) else "   n/a"
                print(f"   {who:>4} {name:>16} {col[lo_m].mean():8.4f} "
                      f"{col[hi_m].mean():8.4f} {d:+9.4f} {cis:>20} "
                      f"{zs:>7} {rho:+8.4f}")
            # sum(Delta) == 0 identically; assert it so a silent indexing bug
            # cannot masquerade as a finding. Tolerance is 1e-6, not 1e-9:
            # probs_at builds the softmax in float32 and these are means over
            # ~5k entries, so ~2e-9 of accumulation drift is expected and is not
            # what this guard is for. Any real mis-indexing would be O(0.01+).
            tot = sum(v["delta"] for v in per.values())
            assert abs(tot) < 1e-6, f"Delta sum {tot} != 0 for {who}/{src}"
            cell["evaluated"][who] = dict(per_action=per, delta_sum=float(tot))
        out[src] = cell

    print("\n" + "=" * 78)
    print("SUMMARY -- risk response Delta by channel (RANDOM / UNION)")
    print("=" * 78)
    print(f"  {'arm':>4} " + " ".join(
        f"{n[:9]:>10}" for n in ACTION_NAMES) + "   set")
    for src in sources:
        for who in ALL:
            per = out[src]["evaluated"][who]["per_action"]
            print(f"  {who:>4} " + " ".join(
                f"{per[n]['delta']:+10.4f}" for n in ACTION_NAMES)
                + f"   {src}")
    print("\n  read: the preregistered P1/P2 are the MIGRATE_EDGE column only.")
    print("        Sum over the four columns is 0 by construction, so the")
    print("        question is which channel absorbs what leaves STAY.")

    blob = dict(
        probe=f"{TAG}_action_channels",
        status="EXPLORATORY -- not preregistered, does not enter the R3 "
               "GO/NO-GO decision, which is NO-GO on P1/P2/P3/P4",
        what="risk response Delta = P(a|risk>=0.6) - P(a|risk<0.2) for ALL "
             "FOUR actions on the same neutral state sets, same decision rule, "
             "same cut points, same episode clustering and bootstrap as the "
             "preregistered P1/P2/P3",
        why="P1/P2/P3 and the P4/P5 reference direction are all defined on "
            "MIGRATE_EDGE alone, so a policy that answers risk with "
            "MIGRATE_CLOUD scores as unresponsive on every one of them. R3's "
            "greedy argmax never selects MIGRATE_EDGE at high risk but does "
            "select MIGRATE_CLOUD, which makes the alternative live.",
        provenance="state sets built by _diag_rung2_75_matched_states' own "
                   "eval_starts / trajectory / random_trajectory / "
                   "union_source with that script's default seeds, imported "
                   "rather than reimplemented; that script is unmodified",
        identity="sum over the four Deltas is 0 for every arm and set; "
                 "asserted at runtime",
        per_source=out,
    )
    p = OUT_DIR / f"{TAG}_action_channels_{args.tag}.json"
    p.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n  wrote {p}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
