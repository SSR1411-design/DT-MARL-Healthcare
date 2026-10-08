#!/usr/bin/env python
"""
SPRINT 7 RUNG 2.75 -- item A, second addendum: CLUSTER-AWARE inference on the
high-risk EDGE share.

WHY THIS EXISTS. A FLAW IN MY OWN POWER ADDENDUM, NOT IN A PRIOR RUNG.

  _diag_rung2_75_edgeshare_power.py reported, under Rung 2.5's own definition:

      rung2_5_on_record   A0 14/212   vs R2 10/269    z = -1.44  p = 0.149  ns
      D5s_pooled_8_seeds  A0 150/1534 vs R2 92/1727   z = -4.84  p < 0.0001 SIG

  Taken at face value the second line says R2 really is worse. It should NOT be
  taken at face value, for two reasons I have to state before quoting it:

  1. CLUSTERING. The two-proportion z test assumes n independent Bernoulli
     trials. A "trial" here is one (timestep, agent) decision entry. Entries
     within an episode share the same trace segment, the same failure
     realisation, and the same 10 agents; entries at adjacent timesteps are
     near-duplicates. The effective sample size is closer to the number of
     EPISODES (8) than to the number of entries (1534). Naive SEs are therefore
     too small and |z| is inflated by roughly sqrt(design effect).

  2. NO NEW TRACE. All 8 torch seeds in D5s replay the SAME 8 start ticks
     [158, 202, 32, 228, 343, 164, 277, 205]. Extra seeds resample the policy's
     action noise, not the trace. So n grew ~7x while the underlying episode
     sample did not grow at all. That is precisely the construction that turns
     a null into a "p < 0.0001".

  Corroborating evidence that start ticks dominate: the main ladder's D4 cell is
  Rung 2.5's definition with EVENLY SPACED starts instead of train.py's RNG
  draw, and it reads A0 0.0865 / R2 0.0855 (z = -0.07, p = 0.95) -- a dead tie
  -- while D5s on the RNG starts reads A0 0.0978 / R2 0.0533. Same definition,
  same window, different 8 episodes, opposite conclusion.

WHAT THIS DOES.

  Draws K start ticks from the TRAIN window under an RNG seed independent of
  TRAIN_SEED, replays each one under BOTH arms with the SAME torch seed (so
  episodes are MATCHED PAIRS), and then reports the same contrast three ways:

    naive     pooled two-proportion z over decision entries  (the inflated one)
    cluster   episode as the unit: paired per-episode share difference, t and
              sign test, plus a cluster bootstrap over episode PAIRS
    deff      design effect = (cluster-robust SE / naive SE)^2, i.e. exactly how
              much the naive test overstates its own precision

  Nothing is trained. No gradient. No production edit. Offline replay only.

Writes SPRINT_7_RUNG2_75_edgeshare_cluster.json.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[0]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from marl._diag_rung0 import (                                   # noqa: E402
    load_agent_and_cfg, _replay, OUT_DIR, ACTION_MIGRATE_EDGE, ACTION_STAY,
)
from marl.env import DTMarlEnv                                   # noqa: E402

TAG = "SPRINT_7_RUNG2_75"
HI_RISK = 0.50
MODELS = {"A0": "mappo_A0_cpu_repro.pth", "R2": "mappo_R2_mc_target.pth"}

# The two figures on record, each reproduced EXACTLY by this rung's main ladder,
# so neither is in question as an arithmetic matter. What is in question is the
# SE they were implicitly quoted with. naive_z is the pooled two-proportion z
# over decision entries from _diag_rung2_75_edgeshare_power.py.
ON_RECORD = {
    "rung2_5": dict(A0=14 / 212, R2=10 / 269, naive_z=-1.44,
                    source="SPRINT_7_RUNG2_5_actor_stall.json "
                           "saturation_census (A0 14/212, R2 10/269)"),
    "production": dict(A0=15 / 546, R2=31 / 384, naive_z=+3.69,
                       source="Rung 2 headline 'high-risk EDGE share "
                              "2.7% -> 8.1%' (A0 15/546, R2 31/384)"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--cell", default="rung2_5",
                   choices=("rung2_5", "production"),
                   help="rung2_5   = TRAIN window, stochastic, risk > 0.50, "
                        "mask-only (Rung 2.5's census); "
                        "production = EVAL window, GREEDY, bucket >= 0.6 "
                        "(rollout.run_episode, i.e. the Rung 2 headline)")
    p.add_argument("--clusters", type=int, default=32,
                   help="number of DISTINCT start ticks (the real sample unit)")
    p.add_argument("--start-seed", type=int, default=20260825,
                   help="RNG for the start ticks; deliberately NOT TRAIN_SEED, "
                        "so this is an independent draw from the same window")
    p.add_argument("--torch-seed", type=int, default=4242,
                   help="action-noise seed; identical for both arms per episode")
    p.add_argument("--n-original", type=int, default=8,
                   help="episodes the on-record figure used; used to convert the "
                        "measured between-episode SD into that figure's SE")
    p.add_argument("--boot", type=int, default=20000)
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, ctr - hw), min(1.0, ctr + hw)]


def naive_two_prop(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (p2 - p1) / se if se > 0 else None
    return dict(k1=k1, n1=n1, k2=k2, n2=n2, p1=p1, p2=p2, diff=p2 - p1,
                se_naive=se, z=z,
                p_value=2 * (1 - _phi(abs(z))) if z is not None else None,
                significant_at_05=bool(z is not None
                                       and 2 * (1 - _phi(abs(z))) < 0.05),
                wilson1=wilson(k1, n1), wilson2=wilson(k2, n2))


def per_episode(agent, cfg, starts, torch_seed, cell="rung2_5", thr=HI_RISK):
    """
    Per-episode high-risk EDGE counts, so the episode can be the unit of
    inference. Two selection rules, each verbatim from its own source:

      cell == "rung2_5"    decision = mask.sum(-1) > 1.5, high risk =
                           env.risk_at(i) > 0.50, STOCHASTIC actions
                           (_diag_rung2_5_actor_stall.saturation_census)
      cell == "production" decision = masks.sum(axis=1) > 1, high risk =
                           min(int(risk*5), 4) >= 3, GREEDY actions
                           (rollout.run_episode, shared by train and evaluate)
    """
    env = DTMarlEnv(cfg.env, cfg.reward)
    sample = (cell == "rung2_5")
    rows = []
    for j, s in enumerate(starts):
        torch.manual_seed(torch_seed + j)
        r = _replay(env, agent, int(s), j, record=True, sample=sample)
        rec = r["rec"]
        rk, ac = [], []
        for i in range(env.n_agents):
            legal = rec["mask"][:, i, :].sum(-1) > 1.5
            rk.append(np.asarray(rec["risk"])[legal, i])
            ac.append(np.asarray(rec["act"])[legal, i])
        rk = np.concatenate(rk); ac = np.concatenate(ac)
        hi = (rk > thr) if cell == "rung2_5" else (
            np.minimum((rk * 5).astype(int), 4) >= 3)
        n = int(hi.sum())
        k = int((ac[hi] == ACTION_MIGRATE_EDGE).sum())
        rows.append(dict(
            start=int(s), episode=j, n_steps=int(r["n_steps"]),
            truncated=bool(r["n_steps"] >= env.cfg.episode_steps),
            n_entries=int(rk.size), n_highrisk=n, EDGE=k,
            share=float(k / n) if n else None,
            STAY=int((ac[hi] == ACTION_STAY).sum()) if n else 0,
        ))
    return rows


def cluster_inference(a_rows, r_rows, n_boot, rng):
    """
    Episode is the cluster. Starts are matched, so pair by episode index.
    Three cluster-level readings, all reported, none selected after the fact.
    """
    pairs = [(a, r) for a, r in zip(a_rows, r_rows)]
    both = [(a, r) for a, r in pairs
            if a["n_highrisk"] > 0 and r["n_highrisk"] > 0]

    # (i) paired per-episode share difference -- the simplest cluster reading
    d = np.array([r["share"] - a["share"] for a, r in both])
    n = d.size
    t = float(d.mean() / (d.std(ddof=1) / math.sqrt(n))) if n > 1 and d.std(ddof=1) > 0 else None
    n_pos = int((d > 0).sum()); n_neg = int((d < 0).sum())
    # exact two-sided sign test
    m = n_pos + n_neg
    if m:
        tailk = min(n_pos, n_neg)
        p_sign = 2 * sum(math.comb(m, i) for i in range(tailk + 1)) / (2 ** m)
        p_sign = min(1.0, p_sign)
    else:
        p_sign = None

    # (ii) cluster bootstrap over episode PAIRS on the POOLED share difference
    ka = np.array([a["EDGE"] for a, _ in pairs], float)
    na = np.array([a["n_highrisk"] for a, _ in pairs], float)
    kr = np.array([r["EDGE"] for _, r in pairs], float)
    nr = np.array([r["n_highrisk"] for _, r in pairs], float)
    obs = (kr.sum() / max(nr.sum(), 1)) - (ka.sum() / max(na.sum(), 1))
    boots = np.empty(n_boot)
    N = len(pairs)
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        sa, sb = na[idx].sum(), nr[idx].sum()
        boots[b] = ((kr[idx].sum() / sb if sb else np.nan)
                    - (ka[idx].sum() / sa if sa else np.nan))
    boots = boots[np.isfinite(boots)]
    se_cluster = float(boots.std(ddof=1))
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    z_cluster = float(obs / se_cluster) if se_cluster > 0 else None

    # (iii) the naive test on the very same data, for the comparison
    naive = naive_two_prop(int(ka.sum()), int(na.sum()),
                           int(kr.sum()), int(nr.sum()))
    deff = (float((se_cluster / naive["se_naive"]) ** 2)
            if naive["se_naive"] > 0 else None)

    return dict(
        n_clusters=N, n_clusters_usable_for_paired=n,
        pooled_A0=dict(EDGE=int(ka.sum()), n=int(na.sum()),
                       share=float(ka.sum() / max(na.sum(), 1))),
        pooled_R2=dict(EDGE=int(kr.sum()), n=int(nr.sum()),
                       share=float(kr.sum() / max(nr.sum(), 1))),
        paired_episode_diff=dict(
            mean=float(d.mean()) if n else None,
            sd=float(d.std(ddof=1)) if n > 1 else None,
            se=float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
            t=t, p_t=2 * (1 - _phi(abs(t))) if t is not None else None,
            n_R2_higher=n_pos, n_A0_higher=n_neg, p_sign_exact=p_sign,
            per_episode=d.tolist(),
        ),
        cluster_bootstrap=dict(
            n_boot=int(boots.size), observed_diff=float(obs),
            se=se_cluster, ci95=ci, z=z_cluster,
            p_value=2 * (1 - _phi(abs(z_cluster))) if z_cluster else None,
            significant_at_05=bool(ci[0] > 0 or ci[1] < 0),
        ),
        naive_same_data=naive,
        design_effect=deff,
        effective_n_A0=float(na.sum() / deff) if deff else None,
        effective_n_R2=float(nr.sum() / deff) if deff else None,
    )


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 RUNG 2.75 -- item A addendum 2: CLUSTER-AWARE inference")
    print("        episode is the sample unit, not the decision entry")
    print("=" * 78)

    # start ticks: an independent draw from the SAME window as the cell, matched
    # across arms so episodes pair up
    window = "train" if args.cell == "rung2_5" else "eval"
    a0, _, cfg0 = load_agent_and_cfg(
        str(OUT_DIR / MODELS["A0"]), args.device, window)
    env0 = DTMarlEnv(cfg0.env, cfg0.reward)
    rng_s = np.random.default_rng(args.start_seed)
    starts = sorted({int(rng_s.integers(env0._min_start, env0._max_start + 1))
                     for _ in range(args.clusters * 3)})
    rng_s2 = np.random.default_rng(args.start_seed + 1)
    starts = sorted(rng_s2.choice(starts, size=min(args.clusters, len(starts)),
                                  replace=False).tolist())
    print(f"\n  cell             : {args.cell}  "
          f"({'stochastic, risk>0.50' if args.cell == 'rung2_5' else 'GREEDY, bucket>=0.6'})")
    print(f"  {window} window     : [{env0._min_start}, {env0._max_start}]")
    print(f"  distinct starts  : {len(starts)}  {starts}")
    print(f"  torch seed       : {args.torch_seed} (identical for both arms)")

    per_arm, ep_sd = {}, {}
    for tag, fn in MODELS.items():
        agent, _, cfg = load_agent_and_cfg(str(OUT_DIR / fn), args.device,
                                           window)
        rows = per_episode(agent, cfg, starts, args.torch_seed, args.cell)
        per_arm[tag] = rows
        sh = [r["share"] for r in rows if r["share"] is not None]
        ep_sd[tag] = float(np.std(sh, ddof=1)) if len(sh) > 1 else None
        print(f"\n  {tag}: {len(rows)} episodes, "
              f"{sum(r['n_highrisk'] for r in rows)} high-risk entries, "
              f"{sum(r['EDGE'] for r in rows)} EDGE")
        print(f"      episode share  mean {np.mean(sh):.4f}  "
              f"sd {np.std(sh, ddof=1):.4f}  "
              f"range {np.min(sh):.4f}-{np.max(sh):.4f}  "
              f"({len(sh)} episodes with high-risk entries)")
        print(f"      truncated {sum(r['truncated'] for r in rows)}/{len(rows)}"
              f"   mean length {np.mean([r['n_steps'] for r in rows]):.1f}")

    ci = cluster_inference(per_arm["A0"], per_arm["R2"], args.boot,
                           np.random.default_rng(11))

    # --- what the measured between-episode SD implies for an n-episode figure ---
    # The on-record numbers (Rung 2 headline, Rung 2.5 census) are each a single
    # draw from a sample of n_original episodes. Their sampling SE is therefore
    # sd_between_episodes / sqrt(n_original) -- NOT the binomial SE over decision
    # entries, which is what both of those write-ups implicitly used. This
    # converts the SD measured HERE (32 episodes, so it is actually estimable)
    # into the SE that the on-record figure should have been quoted with.
    n0 = max(1, args.n_original)
    se_n0 = {t: (s / math.sqrt(n0) if s is not None else None)
             for t, s in ep_sd.items()}
    se_diff_n0 = (math.sqrt(se_n0["A0"] ** 2 + se_n0["R2"] ** 2)
                  if None not in se_n0.values() else None)
    on_record = ON_RECORD[args.cell]
    obs_diff = on_record["R2"] - on_record["A0"]
    z_corrected = (obs_diff / se_diff_n0) if se_diff_n0 else None
    episode_se = dict(
        n_original=n0,
        sd_between_episodes=ep_sd,
        se_of_an_n_original_episode_estimate=se_n0,
        se_of_the_A0_R2_difference=se_diff_n0,
        on_record_figure=on_record,
        on_record_diff=obs_diff,
        z_with_episode_level_se=z_corrected,
        p_with_episode_level_se=(2 * (1 - _phi(abs(z_corrected)))
                                 if z_corrected is not None else None),
        naive_z_on_record=on_record["naive_z"],
        note="the naive z treats correlated decision entries from only "
             f"{n0} episodes as independent trials; the episode-level z uses "
             "the between-episode SD measured on this rung's "
             f"{len(starts)} episodes",
    )
    print("\n" + "=" * 78)
    print(f"WHAT THE ON-RECORD {args.cell} FIGURE SHOULD HAVE BEEN QUOTED WITH")
    print("=" * 78)
    print(f"  on record: A0 {on_record['A0']:.4f}  R2 {on_record['R2']:.4f}"
          f"  diff {obs_diff:+.4f}   (naive z {on_record['naive_z']:+.2f})")
    print(f"  between-episode SD (measured here, {len(starts)} episodes): "
          f"A0 {ep_sd['A0']:.4f}  R2 {ep_sd['R2']:.4f}")
    print(f"  => SE of an {n0}-episode estimate: A0 {se_n0['A0']:.4f}  "
          f"R2 {se_n0['R2']:.4f}   SE(diff) {se_diff_n0:.4f}")
    print(f"  => episode-level z = {z_corrected:+.2f}  "
          f"p = {episode_se['p_with_episode_level_se']:.4f}  "
          f"{'SIG' if episode_se['p_with_episode_level_se'] < 0.05 else 'ns'}")

    print("\n" + "=" * 78)
    print("A0 vs R2, SAME data, three inferences")
    print("=" * 78)
    nv = ci["naive_same_data"]
    print(f"  pooled                 A0 {nv['k1']}/{nv['n1']} = {nv['p1']:.4f}"
          f"   R2 {nv['k2']}/{nv['n2']} = {nv['p2']:.4f}"
          f"   diff {nv['diff']:+.4f}")
    print(f"  naive  (entry as unit) SE {nv['se_naive']:.5f}  z {nv['z']:+.2f}"
          f"  p {nv['p_value']:.4f}  "
          f"{'SIG' if nv['significant_at_05'] else 'ns'}")
    cb = ci["cluster_bootstrap"]
    print(f"  cluster bootstrap      SE {cb['se']:.5f}  z {cb['z']:+.2f}"
          f"  p {cb['p_value']:.4f}  CI95 "
          f"[{cb['ci95'][0]:+.4f}, {cb['ci95'][1]:+.4f}]  "
          f"{'SIG' if cb['significant_at_05'] else 'ns'}")
    pe = ci["paired_episode_diff"]
    print(f"  paired per-episode     mean {pe['mean']:+.4f} "
          f"(sd {pe['sd']:.4f}, n={ci['n_clusters_usable_for_paired']})  "
          f"t {pe['t']:+.2f}  p {pe['p_t']:.4f}")
    print(f"  sign test              R2 higher in {pe['n_R2_higher']} "
          f"episodes, A0 higher in {pe['n_A0_higher']}  "
          f"p_exact {pe['p_sign_exact']:.4f}")
    print(f"\n  DESIGN EFFECT = {ci['design_effect']:.2f}"
          f"   -> effective n is A0 {ci['effective_n_A0']:.0f} / "
          f"R2 {ci['effective_n_R2']:.0f}, not "
          f"{nv['n1']} / {nv['n2']}")
    print(f"  the naive test overstates its precision by "
          f"{math.sqrt(ci['design_effect']):.2f}x in SE terms")

    blob = dict(
        probe=f"{TAG}_edgeshare_cluster",
        what="cluster-aware re-inference of the high-risk EDGE-share contrast; "
             "corrects a flaw in MY OWN _diag_rung2_75_edgeshare_power.py, "
             "which treated correlated decision entries as independent trials "
             "and pooled 8 torch seeds over the same 8 start ticks",
        flaw_being_corrected=dict(
            where="_diag_rung2_75_edgeshare_power.py, tests.D5s_pooled_8_seeds",
            what_it_reported="A0 150/1534 vs R2 92/1727, z = -4.84, p < 0.0001",
            why_it_is_not_trustworthy=[
                "decision entries are clustered within episodes (same trace "
                "segment, same failure realisation, 10 agents, adjacent "
                "timesteps) -- they are not independent Bernoulli trials",
                "all 8 torch seeds replay the SAME 8 start ticks, so n grew "
                "~7x while the episode sample did not grow at all",
                "the main ladder's D4 cell is the same definition on EVENLY "
                "SPACED starts and reads a dead tie (A0 0.0865 / R2 0.0855, "
                "p = 0.95), so the conclusion flips with the choice of 8 "
                "episodes",
            ],
            resolution="episode is the unit of inference; starts are matched "
                       "across arms and drawn independently of TRAIN_SEED",
        ),
        definition=dict(
            cell=args.cell,
            source=("Rung 2.5 saturation_census accounting, per episode"
                    if args.cell == "rung2_5" else
                    "rollout.run_episode accounting (the Rung 2 headline), "
                    "per episode"),
            window=window,
            decision_rule=("mask.sum(-1) > 1.5  (no has_task filter)"
                           if args.cell == "rung2_5"
                           else "masks.sum(axis=1) > 1"),
            threshold=("env.risk_at(i) > 0.50" if args.cell == "rung2_5"
                       else "min(int(risk*5), 4) >= 3, i.e. risk >= 0.6"),
            policy=("stochastic, torch.manual_seed(torch_seed + "
                    "episode_index), identical seed for both arms"
                    if args.cell == "rung2_5" else
                    "GREEDY argmax (deterministic; torch seed is irrelevant "
                    "and the two arms see identical traces)"),
            starts=f"independent RNG draw from the {window.upper()} window",
        ),
        n_clusters=len(starts), start_ticks=starts,
        start_seed=args.start_seed, torch_seed=args.torch_seed,
        per_episode=per_arm, inference=ci, episode_level_se=episode_se,
    )
    p = OUT_DIR / f"{TAG}_edgeshare_cluster_{args.cell}_{args.tag}.json"
    p.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n  wrote {p}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
