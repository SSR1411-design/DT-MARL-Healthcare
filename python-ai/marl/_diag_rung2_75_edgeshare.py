#!/usr/bin/env python
"""
SPRINT 7 RUNG 2.75 -- item A: reconcile the high-risk EDGE-share discrepancy,
and item B (static half): a per-risk-bucket policy census under ONE definition.

THE DISCREPANCY.

  Rung 2 headline      : high-risk EDGE share 2.7% -> 8.1% (A0 -> R2)
  Rung 2.5 census      : R2 3.7%, A0 6.6%   (opposite direction)

  Both numbers are real. They are not the same quantity. Inspecting the code
  rather than guessing, the production metric is built in
  rollout.run_episode (shared by train.py and evaluate.py):

      risks    = [env.risk_at(i) for i in range(env.n_agents)]
      decision = masks.sum(axis=1) > 1          # NO has_task filter
      b        = min(int(risks[i] * 5), 4)      # 5 fixed buckets
      table[b, a] += 1                          # decision entries only

  and evaluate.py reports it for policy "mappo-greedy" over
  rollout.episode_starts on the HELD-OUT window [491, 698].

  The Rung 2.5 saturation census instead used obs[..., 12] with a >0.50
  threshold, an additional has_task filter, STOCHASTIC actions, and the
  TRAINING start window. Five differences at once, on a DISJOINT region of
  the trace.

  obs[..., 12] IS env.risk_at(i) exactly (env.py:463), so the risk SOURCE is
  not one of the differences. The other four are.

WHAT THIS PROBE DOES.

  Replays each checkpoint once per (window, policy) cell, recording every
  decision entry: risk, action, has_task, n_legal, and the full action
  distribution. The ladder below then changes ONE definitional element at a
  time, so the discrepancy is attributed rather than explained away:

      D0  eval window,  greedy,      mask-only, risk>=0.6   <- production
      D1  TRAIN window, greedy,      mask-only, risk>=0.6
      D2  train window, greedy,      mask-only, risk>0.50
      D3  train window, greedy,      +has_task, risk>0.50
      D4  train window, STOCHASTIC,  +has_task, risk>0.50   <- Rung 2.5

  D0 must reproduce 2.75% / 8.07% from the eval artifacts exactly, or the
  replica is wrong and nothing downstream is trustworthy.

No training. No gradient. No production edit. Writes
SPRINT_7_RUNG2_75_edgeshare.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[0]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from marl._diag_rung0 import (                                   # noqa: E402
    load_agent_and_cfg, _replay, OUT_DIR, IDX_HAS_TASK,
    ACTION_STAY, ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD,
)
from marl.env import DTMarlEnv                                   # noqa: E402
from marl.mappo import MASK_FILL                                 # noqa: E402
from marl.rollout import episode_starts as prod_episode_starts   # noqa: E402

TAG = "SPRINT_7_RUNG2_75"
IDX_LOCAL_RISK = 12          # evaluate.py's own constant; == env.risk_at(i)
ACTION_NAMES = ["STAY", "MIGRATE_EDGE", "MIGRATE_CLOUD", "PREEMPT_REROUTE"]

MODELS = {
    "A0": "mappo_A0_cpu_repro.pth",
    "R2": "mappo_R2_mc_target.pth",
}
# the two figures this probe must reproduce, read straight from the eval JSONs
PROD_EVAL = {
    "A0": "mappo_A0_cpu_repro_eval.json",
    "R2": "mappo_R2_mc_target_eval.json",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=8,
                   help="production evaluate.py used 8; keep it for D0 parity")
    p.add_argument("--stoch-seeds", type=int, default=3,
                   help="stochastic replays are sampled; repeat to get a spread")
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


# ----------------------------------------------------------------------
# production metric, recomputed from the frozen eval artifacts
# ----------------------------------------------------------------------

def production_reference():
    """Recompute the Rung 2 headline from risk_action_table, no replay."""
    out = {}
    for tag, fn in PROD_EVAL.items():
        d = json.loads((OUT_DIR / fn).read_text())
        t = np.asarray(d["risk_action_table"], np.int64)
        hi = t[3:].sum(axis=0)                    # buckets 0.6-0.8 and 0.8-1.0
        out[tag] = dict(
            source=fn,
            eval_start_window=d["eval_start_window"],
            eval_starts=d["eval_starts"],
            n_eval_starts=len(d["eval_starts"]),
            risk_action_table=t.tolist(),
            highrisk_counts={ACTION_NAMES[a]: int(hi[a]) for a in range(len(hi))},
            highrisk_n=int(hi.sum()),
            highrisk_EDGE_share=float(hi[ACTION_MIGRATE_EDGE] / max(hi.sum(), 1)),
            all_decision_entries=int(t.sum()),
        )
    return out


# ----------------------------------------------------------------------
# one replay cell -> flat per-decision-entry records
# ----------------------------------------------------------------------

def collect_cell(agent, cfg, window, sample, n_eps, torch_seed=None):
    """
    Replay `n_eps` production-spaced starts on `window` and flatten every
    (step, agent) pair into records. Recorded per entry:
        risk (env.risk_at, == obs[...,12]), action, has_task, n_legal,
        full action probabilities, max prob, entropy, legal mask.
    """
    cfg = cfg  # already windowed by load_agent_and_cfg
    env = DTMarlEnv(cfg.env, cfg.reward)
    starts = prod_episode_starts(env, n_eps)      # production's own spacing
    rec = dict(risk=[], act=[], has_task=[], n_legal=[], maxp=[], ent=[],
               p_stay=[], p_edge=[], p_cloud=[], p_rr=[], ep=[], step=[],
               legal_edge=[], legal_cloud=[])
    lens = []
    for j, s in enumerate(starts):
        if torch_seed is not None:
            torch.manual_seed(torch_seed + j)
        r = _replay(env, agent, s, seed=j, sample=sample)["rec"]
        T = r["act"].shape[0]
        lens.append(int(T))
        with torch.no_grad():
            ob = torch.as_tensor(r["obs"], dtype=torch.float32,
                                 device=agent.device)
            mk = torch.as_tensor(r["mask"], dtype=torch.float32,
                                 device=agent.device)
            P = np.empty((T, env.n_agents, env.n_actions), np.float64)
            for i in range(env.n_agents):
                # same forward + same MASK_FILL as act/act_greedy/masked_dist
                logits = agent.actor.logits(i, ob[:, i]).masked_fill(
                    ~mk[:, i].bool(), MASK_FILL)
                P[:, i] = torch.softmax(logits, dim=-1).cpu().numpy()
        ent = -(P * np.log(np.clip(P, 1e-12, None))).sum(-1)
        for t in range(T):
            for i in range(env.n_agents):
                m = r["mask"][t, i]
                if m.sum() < 1.5:                 # production's decision rule
                    continue
                rec["risk"].append(float(r["risk"][t, i]))
                rec["act"].append(int(r["act"][t, i]))
                rec["has_task"].append(
                    float(r["obs"][t, i, IDX_HAS_TASK]) >= 0.5)
                rec["n_legal"].append(int(m.sum()))
                rec["maxp"].append(float(P[t, i].max()))
                rec["ent"].append(float(ent[t, i]))
                rec["p_stay"].append(float(P[t, i, ACTION_STAY]))
                rec["p_edge"].append(float(P[t, i, ACTION_MIGRATE_EDGE]))
                rec["p_cloud"].append(float(P[t, i, ACTION_MIGRATE_CLOUD]))
                rec["p_rr"].append(float(P[t, i, 3]))
                rec["legal_edge"].append(bool(m[ACTION_MIGRATE_EDGE] > 0.5))
                rec["legal_cloud"].append(bool(m[ACTION_MIGRATE_CLOUD] > 0.5))
                rec["ep"].append(int(j))
                rec["step"].append(int(t))
    out = {k: np.asarray(v) for k, v in rec.items()}
    out["_starts"] = np.asarray(starts)
    out["_lens"] = np.asarray(lens)
    return out


def edge_share(rec, thr, use_bucket, has_task_filter):
    """
    thr            : risk threshold
    use_bucket     : True  -> production's min(int(risk*5),4) >= 3 rule
                     False -> plain risk > thr
    has_task_filter: add the Rung 2.5 has_task requirement
    """
    risk = rec["risk"]
    if use_bucket:
        m = np.minimum((risk * 5).astype(int), 4) >= 3
    else:
        m = risk > thr
    if has_task_filter:
        m = m & rec["has_task"]
    n = int(m.sum())
    if n == 0:
        return dict(n=0, EDGE=0, share=None)
    act = rec["act"][m]
    cnt = {ACTION_NAMES[a]: int((act == a).sum()) for a in range(4)}
    k = cnt["MIGRATE_EDGE"]
    # Wilson 95% interval: with n this small the point estimate alone misleads
    z = 1.959964
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return dict(n=n, EDGE=k, share=float(p), counts=cnt,
                wilson95=[float(max(0.0, ctr - hw)), float(min(1.0, ctr + hw))])


def bucket_census(rec, edges=(0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.01)):
    """Item B (static): what the POLICY says, per risk bucket, one definition."""
    risk, out = rec["risk"], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (risk >= lo) & (risk < hi)
        n = int(m.sum())
        if n == 0:
            out.append(dict(lo=lo, hi=hi, n=0))
            continue
        out.append(dict(
            lo=float(lo), hi=float(hi), n=n,
            taken_EDGE_share=float((rec["act"][m] == ACTION_MIGRATE_EDGE).mean()),
            taken_STAY_share=float((rec["act"][m] == ACTION_STAY).mean()),
            p_edge_mean=float(rec["p_edge"][m].mean()),
            p_edge_p50=float(np.percentile(rec["p_edge"][m], 50)),
            p_edge_p95=float(np.percentile(rec["p_edge"][m], 95)),
            p_stay_mean=float(rec["p_stay"][m].mean()),
            maxp_mean=float(rec["maxp"][m].mean()),
            maxp_p50=float(np.percentile(rec["maxp"][m], 50)),
            frac_maxp_gt_099=float((rec["maxp"][m] > 0.99).mean()),
            entropy_mean=float(rec["ent"][m].mean()),
            frac_entropy_lt_001=float((rec["ent"][m] < 0.01).mean()),
            frac_legal_edge=float(rec["legal_edge"][m].mean()),
            n_legal_mean=float(rec["n_legal"][m].mean()),
        ))
    return out


def saturated_on_what(rec, thr=0.5):
    """
    THE decisive item-B question: when the policy is saturated at high risk,
    WHICH action is it saturated on? Saturation on STAY is a plasticity trap;
    saturation on EDGE would be desirable convergence.
    """
    m = (rec["risk"] > thr) & rec["has_task"]
    if m.sum() == 0:
        return dict(n=0)
    sat = m & (rec["maxp"] > 0.99)
    out = dict(n_highrisk=int(m.sum()), n_saturated=int(sat.sum()),
               frac_saturated=float(sat.sum() / m.sum()))
    if sat.sum():
        P = np.stack([rec["p_stay"][sat], rec["p_edge"][sat],
                      rec["p_cloud"][sat], rec["p_rr"][sat]], axis=1)
        arg = P.argmax(axis=1)
        out["saturated_argmax"] = {
            ACTION_NAMES[a]: int((arg == a).sum()) for a in range(4)}
        out["saturated_argmax_share"] = {
            ACTION_NAMES[a]: float((arg == a).mean()) for a in range(4)}
    return out


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 RUNG 2.75 -- item A: EDGE-share reconciliation")
    print("        (no training, no gradient, no production edit)")
    print("=" * 78)

    prod = production_reference()
    print("\n-- production reference, recomputed from the eval artifacts -----")
    for tag, p in prod.items():
        print(f"  {tag}: window {p['eval_start_window']} "
              f"starts {p['n_eval_starts']}  highrisk n={p['highrisk_n']:5d}  "
              f"EDGE {p['highrisk_EDGE_share']:.4f}")
    print(f"  Rung 2 headline reproduced: "
          f"A0 {prod['A0']['highrisk_EDGE_share']:.4f} -> "
          f"R2 {prod['R2']['highrisk_EDGE_share']:.4f}")

    cells, ladder, census, sat = {}, {}, {}, {}
    for tag, fn in MODELS.items():
        print(f"\n-- {tag} ({fn}) " + "-" * (58 - len(tag) - len(fn)))
        t0 = time.time()
        for window in ("eval", "train"):
            agent, extra, cfg = load_agent_and_cfg(
                str(OUT_DIR / fn), args.device, window)
            g = collect_cell(agent, cfg, window, False, args.episodes)
            cells[(tag, window, "greedy")] = g
            print(f"   {window:5s} greedy    : {len(g['risk']):6d} decision "
                  f"entries, lens {g['_lens'].tolist()}")
            reps = []
            for k in range(args.stoch_seeds):
                reps.append(collect_cell(agent, cfg, window, True,
                                         args.episodes, torch_seed=1000 * (k + 1)))
            cells[(tag, window, "stoch")] = reps
            print(f"   {window:5s} stochastic: "
                  f"{[len(r['risk']) for r in reps]} entries "
                  f"over {args.stoch_seeds} seeds")
        print(f"   ({time.time() - t0:.0f}s)")

        # ---------------- the ladder ----------------
        L = {}
        L["D0_prod_eval_greedy_maskonly_bucket>=0.6"] = edge_share(
            cells[(tag, "eval", "greedy")], 0.6, True, False)
        L["D1_TRAINwindow_greedy_maskonly_bucket>=0.6"] = edge_share(
            cells[(tag, "train", "greedy")], 0.6, True, False)
        L["D2_train_greedy_maskonly_risk>0.50"] = edge_share(
            cells[(tag, "train", "greedy")], 0.50, False, False)
        L["D3_train_greedy_HASTASK_risk>0.50"] = edge_share(
            cells[(tag, "train", "greedy")], 0.50, False, True)
        d4 = [edge_share(r, 0.50, False, True)
              for r in cells[(tag, "train", "stoch")]]
        sh = [x["share"] for x in d4 if x["share"] is not None]
        L["D4_train_STOCHASTIC_HASTASK_risk>0.50"] = dict(
            per_seed=d4, share_mean=float(np.mean(sh)) if sh else None,
            share_min=float(np.min(sh)) if sh else None,
            share_max=float(np.max(sh)) if sh else None,
            n_total=int(sum(x["n"] for x in d4)),
            EDGE_total=int(sum(x["EDGE"] for x in d4)))
        # extra cell: the eval window under Rung 2.5's own definition
        L["X_evalwindow_greedy_HASTASK_risk>0.50"] = edge_share(
            cells[(tag, "eval", "greedy")], 0.50, False, True)
        ladder[tag] = L

        census[tag] = dict(
            train_greedy=bucket_census(cells[(tag, "train", "greedy")]),
            eval_greedy=bucket_census(cells[(tag, "eval", "greedy")]),
            train_stoch_seed0=bucket_census(cells[(tag, "train", "stoch")][0]),
        )
        sat[tag] = dict(
            train_greedy=saturated_on_what(cells[(tag, "train", "greedy")]),
            train_stoch_seed0=saturated_on_what(
                cells[(tag, "train", "stoch")][0]),
            eval_greedy=saturated_on_what(cells[(tag, "eval", "greedy")]),
        )

    # ---------------- report ----------------
    print("\n" + "=" * 78)
    print("ITEM A -- definitional ladder, high-risk EDGE share")
    print("=" * 78)
    keys = list(ladder["A0"].keys())
    print(f"  {'rung':<44s} {'A0':>18s} {'R2':>18s}")
    for k in keys:
        row = []
        for tag in ("A0", "R2"):
            v = ladder[tag][k]
            if "per_seed" in v:
                row.append(f"{v['share_mean']:.4f} n={v['n_total']}"
                           if v["share_mean"] is not None else "n/a")
            else:
                row.append(f"{v['share']:.4f} n={v['n']}"
                           if v["share"] is not None else f"n/a n={v['n']}")
        print(f"  {k:<44s} {row[0]:>18s} {row[1]:>18s}")

    print("\n  D0 parity check against the frozen eval artifacts:")
    for tag in ("A0", "R2"):
        got = ladder[tag]["D0_prod_eval_greedy_maskonly_bucket>=0.6"]
        exp = prod[tag]
        ok = (got["n"] == exp["highrisk_n"] and
              got["EDGE"] == exp["highrisk_counts"]["MIGRATE_EDGE"])
        print(f"    {tag}: replica n={got['n']} EDGE={got['EDGE']}  "
              f"artifact n={exp['highrisk_n']} "
              f"EDGE={exp['highrisk_counts']['MIGRATE_EDGE']}  "
              f"{'MATCH' if ok else '*** MISMATCH ***'}")

    print("\n" + "=" * 78)
    print("ITEM B (static) -- policy census by risk bucket, train/greedy")
    print("=" * 78)
    for tag in ("A0", "R2"):
        print(f"\n  [{tag}] train window, greedy")
        print(f"    {'bucket':<12s} {'n':>6s} {'p_edge':>8s} {'p_stay':>8s} "
              f"{'maxp':>7s} {'>0.99':>7s} {'ent':>7s} {'tookE':>7s} "
              f"{'legalE':>7s}")
        for b in census[tag]["train_greedy"]:
            if not b["n"]:
                print(f"    {b['lo']:.2f}-{b['hi']:.2f}    {0:>6d}")
                continue
            print(f"    {b['lo']:.2f}-{b['hi']:.2f} {b['n']:>7d} "
                  f"{b['p_edge_mean']:>8.4f} {b['p_stay_mean']:>8.4f} "
                  f"{b['maxp_mean']:>7.4f} {b['frac_maxp_gt_099']:>7.4f} "
                  f"{b['entropy_mean']:>7.4f} {b['taken_EDGE_share']:>7.4f} "
                  f"{b['frac_legal_edge']:>7.4f}")

    print("\n  saturation at high risk -- WHICH action is it saturated on?")
    for tag in ("A0", "R2"):
        for cell in ("train_greedy", "train_stoch_seed0"):
            s = sat[tag][cell]
            if not s.get("n_highrisk"):
                continue
            print(f"    {tag} {cell:<18s} n_hi={s['n_highrisk']:5d} "
                  f"saturated {s['frac_saturated']:.4f} "
                  f"({s['n_saturated']}) argmax="
                  f"{s.get('saturated_argmax', {})}")

    blob = dict(
        probe=f"{TAG}_edgeshare",
        what="reconcile the Rung 2 (2.7%->8.1%) vs Rung 2.5 (R2 3.7%, A0 6.6%) "
             "high-risk EDGE-share discrepancy by changing ONE definitional "
             "element at a time; plus a per-risk-bucket policy census under a "
             "single fixed definition",
        production_metric_definition=dict(
            source="marl/rollout.py:run_episode (shared by train.py and "
                   "evaluate.py, so training and evaluation cannot diverge)",
            risk_source="env.risk_at(i); identical to obs[...,12] (env.py:463)",
            decision_rule="masks.sum(axis=1) > 1  -- NO has_task filter",
            bucketing="min(int(risk*5), 4); 'high risk' = buckets 3+4 "
                      "i.e. risk >= 0.6",
            policy="MappoPolicy(greedy=True) -> act_greedy (mappo.py:569)",
            starts="rollout.episode_starts over the HELD-OUT eval window "
                   "[491, 698], 8 episodes",
        ),
        rung2_5_metric_definition=dict(
            source="marl/_diag_rung2_5_actor_stall.py saturation_census",
            risk_source="obs[...,12] (same values)",
            decision_rule="masks.sum > 1 AND obs[...,15] >= 0.5 (has_task)",
            threshold="risk > 0.50",
            policy="stochastic (agent.act)",
            starts="TRAINING window, 8 episodes",
        ),
        differences=["start window (eval [491,698] vs train [9,491]) -- "
                     "DISJOINT regions of the trace",
                     "threshold (bucket >= 0.6 vs > 0.50)",
                     "has_task filter (absent vs present)",
                     "greedy vs stochastic action selection"],
        production_reference=prod,
        ladder=ladder, bucket_census=census, saturation_target=sat,
        episodes=args.episodes, stoch_seeds=args.stoch_seeds,
    )
    p = OUT_DIR / f"{TAG}_edgeshare_{args.tag}.json"
    p.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n  wrote {p}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
