#!/usr/bin/env python
"""
SPRINT 7 RUNG 2.5 -- item E addendum: an R2-NATIVE deviation set.

WHY THIS EXISTS.

  Item E reused Rung 0's exact 583 states / 745 forced replays, as the rung
  brief asked ("the same 745 deviation pairs from Rung 0 where possible").
  Reusing them turned out to be only partly possible, and the measurement
  that proves it is:

      ref_action still equals the baseline action   A0 100.0%   R2  67.6%
      every D1-recorded legal action still legal    A0 100.0%   R2  47.0%

  Rung 0 and Rung 1 both ran on the A0 checkpoint, so their rows were
  self-consistent by construction. R2 is a DIFFERENT policy: at the same
  (start, step, agent) it has usually walked the episode somewhere else, so
  32.4% of D1 rows carry a stale reference action and 53.0% force at least one
  action that R2's own mask marks illegal at that state.

  That does NOT invalidate the arm-vs-arm or raw-vs-paired contrasts in item E
  -- every arm was scored on identical rows, so those comparisons are matched.
  It does mean the ABSOLUTE sign-agreement level and the per-action means on
  Rung 0's rows are provisional for R2.

  This probe therefore rebuilds the deviation set from R2's OWN greedy
  baselines, using Rung 0's selection rule verbatim (pools by the action greedy
  actually chose, risk buckets, evenly spaced take() with no RNG), and records
  R2's own legal mask so every forced action is genuinely legal.

No training, no gradient, no production edit. Writes
SPRINT_7_RUNG2_5_native_dev_<tag>.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[0]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from marl._diag_rung0 import (                                   # noqa: E402
    load_agent_and_cfg, _replay, episode_starts, risk_bucket, OUT_DIR,
    BUCKETS, MEASURED, IDX_HAS_TASK,
    ACTION_STAY, ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD,
)
from marl._diag_rung1_critic import (                            # noqa: E402
    collect_baselines, deviation_pass, agreement, per_action, noise_floor,
    _dump, _ikeys,
)

TAG = "SPRINT_7_RUNG2_5"
CAPS = {"A": 150, "B": 80, "C": 60}   # the caps that reproduce D1's own mix


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(OUT_DIR / "mappo_R2_mc_target.pth"))
    p.add_argument("--device", default="cpu")
    p.add_argument("--window", default="train", choices=("train", "eval"))
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--tag", default="R2")
    return p.parse_args(argv)


def build_native_rows(env, base, starts):
    """Rung 0's pool/bucket/take rule, verbatim, on THIS agent's baselines."""
    pools = {(p, b): [] for p in ("A", "B", "C") for b in BUCKETS}
    for s in starts:
        rec = base[s]["rec"]
        for t in range(len(rec["act"])):
            for i in range(env.n_agents):
                if rec["obs"][t, i, IDX_HAS_TASK] < 0.5:
                    continue
                m = rec["mask"][t, i]
                if m.sum() < 1.5:
                    continue
                a0 = int(rec["act"][t, i])
                pool = {ACTION_STAY: "A", ACTION_MIGRATE_CLOUD: "B",
                        ACTION_MIGRATE_EDGE: "C"}.get(a0)
                if pool is None:
                    continue
                r = float(rec["risk"][t, i])
                legal = [int(a) for a in MEASURED if m[a] > 0.5]
                pools[(pool, risk_bucket(r))].append(
                    dict(pool=pool, bucket=risk_bucket(r), start=int(s),
                         step=int(t), agent=int(i), risk=r, ref_action=a0,
                         legal=legal))

    def take(pool, k):
        """Evenly spaced, no RNG: the selection cannot be cherry-picked."""
        if not pool or k <= 0:
            return []
        idx = np.linspace(0, len(pool) - 1, min(k, len(pool))).astype(int)
        return [pool[t] for t in sorted(set(idx.tolist()))]

    rows = []
    print("\n  pool sizes (available -> sampled):")
    for key in sorted(pools):
        sel = take(pools[key], CAPS[key[0]])
        print(f"    pool {key[0]} risk_{key[1]:<3s}: "
              f"{len(pools[key]):6d} -> {len(sel):4d}")
        rows.extend(sel)
    # keep only rows that actually admit a deviation
    rows = [r for r in rows if len([a for a in r["legal"]
                                    if a != r["ref_action"]]) >= 1]
    return rows, {f"{k[0]}_{k[1]}": len(v) for k, v in sorted(pools.items())}


def paired_agreement(rows, arm, truth_key):
    """
    Sign agreement of the PAIRED estimator gae(a) - gae(a_ref) against
    a_true = Q(a) - Q(a_ref).

    The paired form is the estimator that actually matches the truth's
    definition: both are differences taken at the SAME state, so any
    per-state offset in V(s) cancels. The raw form does not cancel it.
    """
    out = {}
    for b in list(BUCKETS) + ["all"]:
        R = rows if b == "all" else [r for r in rows if r["bucket"] == b]
        T, Gr, Gp = [], [], []
        for r in R:
            g = _ikeys(r["gae"][arm])
            ref = int(r["ref_action"])
            if ref not in g:
                continue
            for k, v in _ikeys(r[truth_key]).items():
                if k == ref:
                    continue
                T.append(v); Gr.append(g[k]); Gp.append(g[k] - g[ref])
        T = np.asarray(T, np.float64)
        if T.size < 3:
            out[b] = dict(n=int(T.size)); continue
        Gr = np.asarray(Gr, np.float64); Gp = np.asarray(Gp, np.float64)
        nz = np.abs(T) > 1e-9
        out[b] = dict(
            n=int(T.size), n_sign_comparable=int(nz.sum()),
            sign_agreement_raw=float(np.mean(np.sign(T[nz]) ==
                                             np.sign(Gr[nz]))),
            sign_agreement_paired=float(np.mean(np.sign(T[nz]) ==
                                                np.sign(Gp[nz]))),
            pearson_raw=float(np.corrcoef(T, Gr)[0, 1]),
            pearson_paired=float(np.corrcoef(T, Gp)[0, 1]),
            true_mean=float(T.mean()),
            gae_raw_mean=float(Gr.mean()), gae_paired_mean=float(Gp.mean()),
            gae_raw_sd=float(Gr.std()), gae_paired_sd=float(Gp.std()),
        )
    return out


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 RUNG 2.5 -- item E addendum: R2-NATIVE deviation set")
    print("        (no training, no gradient, no production edit)")
    print("=" * 78)
    agent, extra, cfg = load_agent_and_cfg(args.model, args.device, args.window)
    print(f"  model        : {Path(args.model).name}")
    print(f"  critic_target: {agent.cfg.critic_target}")

    env, starts, base, seed_of, replica = collect_baselines(
        agent, cfg, args.episodes)
    lens = sorted(int(base[s]["n_steps"]) for s in starts)
    n_tr = sum(1 for v in lens if v >= int(cfg.env.episode_steps))
    print(f"  greedy baseline lengths: {lens}")
    print(f"  truncated (>= {cfg.env.episode_steps}): {n_tr}/{len(lens)}")

    rows, pool_sizes = build_native_rows(env, base, starts)
    print(f"  native rows with >=1 deviation: {len(rows)}")

    critics = {"C0": agent.critic}
    t0 = time.time()
    devrows, mismatch, runs = deviation_pass(rows, critics, agent, cfg,
                                             base, seed_of, env)
    print(f"  usable states {len(devrows)}, forced replays {runs}, "
          f"mismatches {mismatch}  ({time.time() - t0:.0f}s)")

    summary, paired = {}, {}
    for truth in ("a_true_team", "a_true_own"):
        summary[truth] = {arm: dict(
            agreement=agreement(devrows, arm, truth),
            per_action_hi=per_action(devrows, arm, truth, "hi"),
            per_action_lo=per_action(devrows, arm, truth, "lo"),
        ) for arm in critics}
        paired[truth] = {arm: paired_agreement(devrows, arm, truth)
                         for arm in critics}

    print("\n-- per-action means, HIGH-RISK bucket -------------------------")
    for truth in ("a_true_team", "a_true_own"):
        pa = summary[truth]["C0"]["per_action_hi"]
        print(f"  [{truth}] ordering_by_true={pa['ordering_by_true']}")
        print(f"  {'':11s} ordering_by_gae ={pa['ordering_by_gae']}")
        for an, v in pa["actions"].items():
            if not v.get("n"):
                continue
            t, g = v["true"], v["gae"]
            se_t = t["sd"] / np.sqrt(v["n"]); se_g = g["sd"] / np.sqrt(v["n"])
            print(f"     {an:24s} n={v['n']:4d}  true {t['mean']:+8.4f} "
                  f"+-{se_t:.4f} (t={t['mean'] / max(se_t, 1e-12):+6.2f})   "
                  f"gae {g['mean']:+8.4f} +-{se_g:.4f} "
                  f"(t={g['mean'] / max(se_g, 1e-12):+6.2f})")

    print("\n-- raw vs paired sign agreement -------------------------------")
    for truth in ("a_true_team", "a_true_own"):
        print(f"  [{truth}]")
        for b in ("hi", "lo", "all"):
            p = paired[truth]["C0"].get(b, {})
            if "sign_agreement_raw" not in p:
                continue
            print(f"    {b:3s} n={p['n_sign_comparable']:4d}  raw "
                  f"{p['sign_agreement_raw']:.4f}  paired "
                  f"{p['sign_agreement_paired']:.4f}  delta "
                  f"{p['sign_agreement_paired'] - p['sign_agreement_raw']:+.4f}")

    blob = dict(
        probe=f"{TAG}_native_dev_{args.tag}",
        what="deviation set rebuilt from THIS checkpoint's own greedy "
             "baselines, so ref_action and the legal mask are self-consistent; "
             "fixes the 67.6%/47.0% staleness of Rung 0's D1 rows under R2",
        model=str(args.model), critic_target=agent.cfg.critic_target,
        why_not_rung0_rows=dict(
            d1_ref_action_matches_baseline_A0=1.0,
            d1_ref_action_matches_baseline_R2=0.676,
            d1_legal_all_still_legal_A0=1.0,
            d1_legal_all_still_legal_R2=0.470,
            note="arm-vs-arm and raw-vs-paired contrasts in item E are "
                 "matched-row and therefore unaffected; absolute levels and "
                 "per-action means on D1 rows are provisional for R2",
        ),
        episodes=args.episodes, caps=CAPS, pool_sizes=pool_sizes,
        greedy_baseline_lengths=lens, greedy_baselines_truncated=int(n_tr),
        deviation=dict(n_rows_built=len(rows), n_states=len(devrows),
                       forced_replays=runs, replay_mismatches=mismatch,
                       gae_replica_check=replica),
        summary=summary, paired=paired,
        noise_floor={arm: noise_floor(devrows, arm) for arm in critics},
        rows=devrows,
    )
    _dump(blob, OUT_DIR / f"{TAG}_native_dev_{args.tag}.json")
    print(f"\n  wrote {OUT_DIR / (TAG + '_native_dev_' + args.tag + '.json')}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
