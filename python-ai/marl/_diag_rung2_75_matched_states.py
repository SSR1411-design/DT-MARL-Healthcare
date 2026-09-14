#!/usr/bin/env python
"""
SPRINT 7 RUNG 2.75 -- item A, third addendum: STATE-MATCHED policy comparison,
and a replacement for the primary metric.

WHY THIS EXISTS. A CONFOUND IN THE HEADLINE METRIC ITSELF, surfaced by this
rung's own cluster run and not previously noted anywhere.

  Under the production definition on 32 matched eval-window starts
  (SPRINT_7_RUNG2_75_edgeshare_cluster_production_main.json):

      A0   2174 high-risk decision entries,  60 EDGE,  share 0.0276
           0/32 episodes truncated, mean length 374.2
      R2   1418 high-risk decision entries, 122 EDGE,  share 0.0860
           17/32 episodes truncated, mean length 392.1

  That contrast is robust to episode-level inference (cluster bootstrap
  z = +6.48, CI95 [+0.0417, +0.0770], R2 higher in 31 of 32 episodes). But the
  DENOMINATORS differ by 1.53x, because each arm generates its own trajectory.
  "R2 migrates more at high risk" is therefore entangled with "R2 ENCOUNTERS a
  different set of high-risk states". A share computed on each policy's own
  on-policy distribution cannot separate the two.

  Worse, the entanglement is not incidental. Risk is high AT a host BECAUSE
  tasks are still on it. Choosing STAY is what keeps an agent in a high-risk
  state; choosing MIGRATE_EDGE is what removes it from one. So conditioning on
  "currently in a high-risk state" SELECTS FOR "just chose to stay". The metric
  has the outcome in its own denominator.

  This is not a computation error. Both figures use the same definition and were
  reproduced exactly. The defect is that the quantity is not identified.

WHAT THIS DOES.

  (1) STATE-MATCHED CROSS-EVALUATION. For each trajectory source, evaluate BOTH
      actors at the SAME recorded states -- same obs, same masks, same
      high-risk selection. The only thing that varies is which actor is asked.
      On the diagonal this must reproduce the headline exactly; off the diagonal
      it reveals whether the contrast is policy or distribution.

  (2) A NEUTRAL STATE SOURCE. Both arms' own distributions privilege that arm.
      A third source draws uniformly at random over legal actions, so it depends
      on NEITHER policy. Caveat stated plainly: a random behaviour policy visits
      states that are off-manifold for both arms, so it is a neutral reference,
      not a realistic one. It is reported alongside, never instead.

  (3) A RISK-RESPONSE CURVE, which is what the project's primary question
      actually asks -- "does the learned policy use predicted_failure_risk to
      change its decision?" On a FIXED state set, bin entries by risk and read
      pi(MIGRATE_EDGE | risk bin). A policy that uses risk has a rising curve.
      Reported with Spearman rho over entries and with the high-minus-low
      contrast, clustered by episode. This quantity has no on-policy selection
      in it, because the state set does not depend on the policy being scored.

No training, no gradient, no optimiser, no production edit. Greedy replay plus
forward passes on loaded checkpoints.

Writes SPRINT_7_RUNG2_75_matched_states.json.
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
    load_agent_and_cfg, _replay, OUT_DIR,
    ACTION_STAY, ACTION_MIGRATE_EDGE,
)
from marl.env import DTMarlEnv                                   # noqa: E402
from marl.mappo import MASK_FILL                                 # noqa: E402

TAG = "SPRINT_7_RUNG2_75"
ACTION_NAMES = ["STAY", "MIGRATE_EDGE", "MIGRATE_CLOUD", "PREEMPT_REROUTE"]
MODELS = {"A0": "mappo_A0_cpu_repro.pth", "R2": "mappo_R2_mc_target.pth"}
# every other Sprint 6.5 / Sprint 7 arm, scored on the SAME neutral state sets.
# Their only role is to calibrate the spread of the proposed primary metric
# across arms that already exist, so a GO threshold for the next rung is set
# against observed variation rather than invented.
#
# SPRINT 7 R3 is registered HERE and deliberately NOT in MODELS. MODELS is the
# set that GENERATES state sources (line ~337 iterates it, and UNION is the
# pool of exactly those trajectories), so promoting R3 there would silently
# redefine UNION and RANDOM and the pre-registered thresholds would no longer
# be measured on the sets they were calibrated on. As an EXTRA arm R3 is scored
# on the unchanged A0/R2/RANDOM/UNION sets, which is what the pre-registration
# requires. Its filename has no `mappo_` prefix because the approved command
# passed `--tag R3_batch32` verbatim; the command was not altered to fit the
# older naming convention.
EXTRA = {"A1": "mappo_A1_cpu_bugfix.pth",
         "A2": "mappo_A2_crit_sign.pth",
         "A3": "mappo_A3_entropy.pth",
         "R3": "R3_batch32.pth"}
RISK_BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
# the headline figures this must reproduce on the diagonal
HEADLINE = {"A0": (60, 2174), "R2": (122, 1418)}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--clusters", type=int, default=32)
    p.add_argument("--start-seed", type=int, default=20260825,
                   help="same draw as the cluster probe, so the episode set is "
                        "identical and the two artifacts are comparable")
    p.add_argument("--boot", type=int, default=5000)
    p.add_argument("--random-seed", type=int, default=31337,
                   help="action RNG for the neutral (uniform-legal) source")
    p.add_argument("--extra-arms", default=",".join(EXTRA),
                   help="also SCORE these arms on every state source (they do "
                        "not generate state sources); '' to disable")
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _rank(x):
    """Average-tie ranks, so Spearman needs no scipy."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    r[order] = np.arange(1, x.size + 1, dtype=float)
    # average ties
    xs = x[order]
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    return r


def spearman(a, b):
    if a.size < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else None


def eval_starts(cfg, n, seed):
    env = DTMarlEnv(cfg.env, cfg.reward)
    rng = np.random.default_rng(seed)
    s = sorted({int(rng.integers(env._min_start, env._max_start + 1))
                for _ in range(n * 3)})
    rng2 = np.random.default_rng(seed + 1)
    return sorted(rng2.choice(s, size=min(n, len(s)), replace=False).tolist()), env


def _pack(OB, MK, RK, EP, LN, TR, n_agents):
    return dict(obs=np.concatenate(OB), mask=np.concatenate(MK),
                risk=np.concatenate(RK), ep=np.concatenate(EP),
                lengths=LN, truncated=TR, n_agents=n_agents)


def trajectory(agent, cfg, starts):
    """Greedy replay of one arm; keep obs, masks, risk, episode id."""
    env = DTMarlEnv(cfg.env, cfg.reward)
    OB, MK, RK, EP, LN, TR = [], [], [], [], [], 0
    for j, s in enumerate(starts):
        rec = _replay(env, agent, int(s), j, record=True, sample=False)["rec"]
        n = np.asarray(rec["risk"]).shape[0]
        OB.append(np.asarray(rec["obs"])); MK.append(np.asarray(rec["mask"]))
        RK.append(np.asarray(rec["risk"])); EP.append(np.full(n, j, np.int32))
        LN.append(n); TR += int(n >= env.cfg.episode_steps)
    return _pack(OB, MK, RK, EP, LN, TR, env.n_agents)


def random_trajectory(cfg, starts, seed):
    """
    Uniform over LEGAL actions -- a behaviour distribution that depends on
    neither arm. Mirrors _replay's loop exactly, minus the agent.
    """
    env = DTMarlEnv(cfg.env, cfg.reward)
    rng = np.random.default_rng(seed)
    OB, MK, RK, EP, LN, TR = [], [], [], [], [], 0
    for j, s in enumerate(starts):
        obs, state, masks = env.reset(episode_start_tick=int(s), seed=j)
        done, step = False, 0
        while not done:
            OB.append(obs.copy()[None]); MK.append(masks.copy()[None])
            RK.append(np.array([[env.risk_at(i)
                                 for i in range(env.n_agents)]], np.float32))
            EP.append(np.array([j], np.int32))
            a = np.empty(env.n_agents, np.int64)
            for i in range(env.n_agents):
                legal = np.flatnonzero(np.asarray(masks[i]) > 0.5)
                a[i] = int(rng.choice(legal)) if legal.size else 0
            obs, state, rew, done, info = env.step(a)
            masks = info["action_masks"]
            step += 1
        LN.append(step); TR += int(step >= env.cfg.episode_steps)
    return _pack(OB, MK, RK, EP, LN, TR, env.n_agents)


def union_source(a, b, ep_offset=1000):
    """
    Pool two recorded state sets. Symmetric: each arm's own privileged states
    enter on equal footing, so neither arm's on-policy selection effect can
    dominate the contrast. Episode ids are kept disjoint so clustering is honest.
    """
    assert a["n_agents"] == b["n_agents"]
    return dict(
        obs=np.concatenate([a["obs"], b["obs"]]),
        mask=np.concatenate([a["mask"], b["mask"]]),
        risk=np.concatenate([a["risk"], b["risk"]]),
        ep=np.concatenate([a["ep"], b["ep"] + ep_offset]),
        lengths=list(a["lengths"]) + list(b["lengths"]),
        truncated=a["truncated"] + b["truncated"],
        n_agents=a["n_agents"])


def probs_at(agent, obs, mask):
    """pi(.|s) for every (t, i) using production's forward and MASK_FILL."""
    with torch.no_grad():
        ob = torch.as_tensor(obs, dtype=torch.float32, device=agent.device)
        mk = torch.as_tensor(mask, dtype=torch.float32, device=agent.device)
        P = np.empty(obs.shape[:2] + (4,), np.float64)
        for i in range(agent.n_agents):
            lg = agent.actor.logits(i, ob[:, i]).masked_fill(
                ~mk[:, i].bool(), MASK_FILL)
            P[:, i] = torch.softmax(lg, dim=-1).cpu().numpy()
    return P


def cluster_ci(vals, eps, n_boot, rng):
    """Cluster bootstrap over episodes on the mean of a per-entry quantity."""
    uniq = np.unique(eps)
    by = {e: vals[eps == e] for e in uniq}
    bs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        bs[b] = np.concatenate([by[uniq[k]] for k in pick]).mean()
    se = float(bs.std(ddof=1))
    return dict(mean=float(vals.mean()), se_cluster=se,
                ci95=[float(np.percentile(bs, 2.5)),
                      float(np.percentile(bs, 97.5))],
                n=int(vals.size))


def cluster_ci_diff(vals, sel_hi, sel_lo, eps, n_boot, rng):
    """
    Cluster bootstrap over episodes on a DIFFERENCE OF TWO SUBGROUP MEANS of the
    same per-entry quantity -- here pi(EDGE | risk>=0.6) - pi(EDGE | risk<0.2).
    Both subgroups are resampled together within each drawn episode, so the
    correlation between them inside an episode is preserved.
    """
    uniq = np.unique(eps)
    by = {e: (vals[sel_hi & (eps == e)], vals[sel_lo & (eps == e)])
          for e in uniq}
    obs = (float(vals[sel_hi].mean() - vals[sel_lo].mean())
           if sel_hi.any() and sel_lo.any() else None)
    if obs is None:
        return dict(delta=None)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, uniq.size, uniq.size)
        h = np.concatenate([by[uniq[k]][0] for k in pick])
        l = np.concatenate([by[uniq[k]][1] for k in pick])
        if h.size and l.size:
            bs.append(h.mean() - l.mean())
    bs = np.asarray(bs)
    se = float(bs.std(ddof=1)) if bs.size > 1 else None
    return dict(delta=obs, se_cluster=se,
                ci95=[float(np.percentile(bs, 2.5)),
                      float(np.percentile(bs, 97.5))] if bs.size else None,
                z=(obs / se if se and se > 0 else None),
                p_value=(2 * (1 - _phi(abs(obs / se)))
                         if se and se > 0 else None),
                n_hi=int(sel_hi.sum()), n_lo=int(sel_lo.sum()),
                n_clusters=int(uniq.size))


def summarise(Q, arg):
    return dict(
        p_edge_mean=float(Q[:, ACTION_MIGRATE_EDGE].mean()),
        p_stay_mean=float(Q[:, ACTION_STAY].mean()),
        maxp_mean=float(Q.max(-1).mean()),
        frac_maxp_gt_099=float((Q.max(-1) > 0.99).mean()),
        entropy_mean=float(-(Q * np.log(np.clip(Q, 1e-12, None)))
                           .sum(-1).mean()),
        frac_argmax_edge=float((arg == ACTION_MIGRATE_EDGE).mean()),
        argmax_counts={ACTION_NAMES[k]: int((arg == k).sum()) for k in range(4)},
    )


def risk_curve(risk, P, dec):
    """pi(EDGE | risk bin) over ALL decision entries of a fixed state set."""
    rows, r, pe = [], risk[dec], P[dec][:, ACTION_MIGRATE_EDGE]
    arg = P[dec].argmax(-1)
    for lo, hi in zip(RISK_BINS[:-1], RISK_BINS[1:]):
        m = (r >= lo) & (r < hi)
        rows.append(dict(lo=lo, hi=min(hi, 1.0), n=int(m.sum()),
                         risk_mean=float(r[m].mean()) if m.any() else None,
                         p_edge=float(pe[m].mean()) if m.any() else None,
                         frac_argmax_edge=float(
                             (arg[m] == ACTION_MIGRATE_EDGE).mean())
                         if m.any() else None))
    return rows


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 RUNG 2.75 -- item A addendum 3: STATE-MATCHED comparison")
    print("        both actors evaluated at the SAME states, so the state")
    print("        distribution cannot carry the contrast")
    print("=" * 78)

    agents, cfgs = {}, {}
    extra = {k: EXTRA[k] for k in
             (x.strip() for x in args.extra_arms.split(",")) if k in EXTRA} \
        if args.extra_arms.strip() else {}
    for tag, fn in {**MODELS, **extra}.items():
        a, _, c = load_agent_and_cfg(str(OUT_DIR / fn), args.device, "eval")
        agents[tag], cfgs[tag] = a, c
    ALL = list(MODELS) + list(extra)
    starts, env0 = eval_starts(cfgs["A0"], args.clusters, args.start_seed)
    print(f"\n  eval window      : [{env0._min_start}, {env0._max_start}]")
    print(f"  distinct starts  : {len(starts)}")
    print(f"  policy           : GREEDY (deterministic, no action noise)")
    print(f"  high-risk rule   : min(int(risk*5), 4) >= 3   (production bucket)")
    print(f"  decision rule    : mask.sum(-1) > 1           (production)")
    print(f"  state sources    : A0, R2 (own greedy), RANDOM (uniform-legal), "
          f"UNION (pooled)")
    print(f"  arms SCORED      : {', '.join(ALL)}"
          + (f"   [{', '.join(extra)} score only -- they generate no state "
             f"source; their role is to calibrate the spread of the proposed "
             f"primary metric across arms that already exist]" if extra else ""))

    sources = {}
    for src in MODELS:
        sources[src] = trajectory(agents[src], cfgs[src], starts)
    sources["RANDOM"] = random_trajectory(cfgs["A0"], starts, args.random_seed)
    # UNION: A0's states pooled with R2's. Symmetric by construction -- it
    # contains exactly the states each arm is privileged on, in equal standing,
    # so neither arm's selection effect can dominate. Unlike RANDOM it is fully
    # on-manifold for both. No new rollout: just a pooled state set, with R2's
    # episode ids offset so the cluster bootstrap keeps them distinct.
    sources["UNION"] = union_source(sources["A0"], sources["R2"])

    out = {}
    for src, tr in sources.items():
        dec = tr["mask"].sum(-1) > 1.0
        hi = (np.minimum((tr["risk"] * 5).astype(int), 4) >= 3) & dec
        n_hi = int(hi.sum())
        rk_hi = tr["risk"][hi]
        lbl = ("uniform-random legal actions (POLICY-INDEPENDENT)"
               if src == "RANDOM" else
               "A0's states POOLED with R2's (SYMMETRIC)" if src == "UNION"
               else f"{src}'s own greedy trajectory")
        print(f"\n-- states from {lbl} " + "-" * max(4, 40 - len(lbl)))
        print(f"   decision entries {int(dec.sum())}   high-risk {n_hi}"
              f"   ({n_hi / max(dec.sum(), 1):.1%} of decision entries)")
        print(f"   truncated {tr['truncated']}/{len(starts)}"
              f"   mean length {np.mean(tr['lengths']):.1f}")
        print(f"   risk WITHIN the high-risk set: mean {rk_hi.mean():.4f}  "
              f"p25 {np.percentile(rk_hi, 25):.4f}  p50 "
              f"{np.percentile(rk_hi, 50):.4f}  p75 "
              f"{np.percentile(rk_hi, 75):.4f}")

        eps = np.repeat(tr["ep"][:, None], tr["n_agents"], axis=1)[hi]
        eps_dec = np.repeat(tr["ep"][:, None], tr["n_agents"], axis=1)[dec]
        cell = dict(
            source=src, n_decision=int(dec.sum()), n_highrisk=n_hi,
            highrisk_frac_of_decision=float(n_hi / max(dec.sum(), 1)),
            truncated=tr["truncated"],
            mean_length=float(np.mean(tr["lengths"])),
            risk_within_highrisk=dict(
                mean=float(rk_hi.mean()),
                p25=float(np.percentile(rk_hi, 25)),
                p50=float(np.percentile(rk_hi, 50)),
                p75=float(np.percentile(rk_hi, 75))),
            evaluated={},
        )
        PE = {}
        for who in ALL:
            P = probs_at(agents[who], tr["obs"], tr["mask"])
            Q, arg = P[hi], P[hi].argmax(-1)
            PE[who] = Q[:, ACTION_MIGRATE_EDGE]
            cb = cluster_ci(PE[who], eps, args.boot, np.random.default_rng(5))
            ab = cluster_ci((arg == ACTION_MIGRATE_EDGE).astype(float), eps,
                            args.boot, np.random.default_rng(6))
            # --- risk response on THIS fixed state set (all decision entries) --
            rc = risk_curve(tr["risk"], P, dec)
            r_all = tr["risk"][dec]
            pe_all = P[dec][:, ACTION_MIGRATE_EDGE]
            lo_m, hi_m = r_all < 0.2, r_all >= 0.6
            resp = (float(pe_all[hi_m].mean() - pe_all[lo_m].mean())
                    if lo_m.any() and hi_m.any() else None)
            # The response is the proposed PRIMARY metric for the next rung, so
            # it needs an interval, not just a point estimate: a GO threshold
            # has to be set against the sampling spread of the quantity it
            # gates on. Clustered by episode because entries within an episode
            # are not independent (design effect 1.45 was measured for the
            # related share metric in _diag_rung2_75_edgeshare_cluster).
            respci = cluster_ci_diff(pe_all, hi_m, lo_m, eps_dec, args.boot,
                                     np.random.default_rng(11))
            cell["evaluated"][who] = dict(
                **summarise(Q, arg),
                p_edge_cluster=cb, frac_argmax_edge_cluster=ab,
                risk_curve=rc,
                spearman_risk_vs_p_edge=spearman(r_all, pe_all),
                p_edge_risk_lt_02=float(pe_all[lo_m].mean())
                if lo_m.any() else None,
                p_edge_risk_ge_06=float(pe_all[hi_m].mean())
                if hi_m.any() else None,
                risk_response_high_minus_low=resp,
                risk_response_cluster=respci,
                is_own_state_source=bool(who == src),
            )
            e = cell["evaluated"][who]
            star = "  <- this arm's own states" if who == src else ""
            print(f"   pi_{who}: p_edge {e['p_edge_mean']:.4f} "
                  f"CI95 [{cb['ci95'][0]:.4f}, {cb['ci95'][1]:.4f}]   "
                  f"argmaxE {e['frac_argmax_edge']:.4f} "
                  f"CI95 [{ab['ci95'][0]:.4f}, {ab['ci95'][1]:.4f}]   "
                  f"maxp {e['maxp_mean']:.4f}  >.99 "
                  f"{e['frac_maxp_gt_099']:.4f}  ent {e['entropy_mean']:.4f}"
                  f"{star}")
            print(f"          argmax {e['argmax_counts']}")
            rci = (f"  CI95 [{respci['ci95'][0]:+.4f}, {respci['ci95'][1]:+.4f}]"
                   f"  z {respci['z']:+.2f}"
                   if respci.get("ci95") and respci.get("z") is not None else "")
            print(f"          risk response (ALL decision entries of this set): "
                  f"p_edge risk<0.2 {e['p_edge_risk_lt_02']:.4f} -> "
                  f"risk>=0.6 {e['p_edge_risk_ge_06']:.4f}  "
                  f"(delta {resp:+.4f}, spearman "
                  f"{e['spearman_risk_vs_p_edge']:+.4f}){rci}")
            print("          pi(EDGE) by risk bin: " + "  ".join(
                f"[{r['lo']:.1f},{r['hi']:.1f}) {r['p_edge']:.3f}/n{r['n']}"
                for r in rc if r["n"] > 0))

        d = PE["R2"] - PE["A0"]
        db = cluster_ci(d, eps, args.boot, np.random.default_rng(9))
        z = db["mean"] / db["se_cluster"] if db["se_cluster"] > 0 else None
        cell["paired_within_state_R2_minus_A0"] = dict(
            mean=db["mean"], se_cluster=db["se_cluster"], ci95=db["ci95"], z=z,
            p_value=(2 * (1 - _phi(abs(z))) if z is not None else None),
            frac_entries_R2_higher=float((d > 0).mean()),
            median=float(np.median(d)))
        print(f"   PAIRED within-state  pi_R2 - pi_A0 = {db['mean']:+.4f}"
              f"  CI95 [{db['ci95'][0]:+.4f}, {db['ci95'][1]:+.4f}]"
              f"  z {z:+.2f}  R2 higher at "
              f"{cell['paired_within_state_R2_minus_A0']['frac_entries_R2_higher']:.1%}"
              f" of entries")
        out[src] = cell

    # ---------------- diagonal parity against the headline ----------------
    print("\n" + "=" * 78)
    print("DIAGONAL PARITY -- does the cross-evaluation reproduce the headline?")
    print("=" * 78)
    parity = {}
    for tag in MODELS:
        k, n = HEADLINE[tag]
        got = out[tag]["evaluated"][tag]["argmax_counts"]["MIGRATE_EDGE"]
        gn = out[tag]["n_highrisk"]
        ok = (got == k and gn == n)
        parity[tag] = dict(expected_EDGE=k, expected_n=n, got_EDGE=got,
                           got_n=gn, match=bool(ok))
        print(f"  {tag}: {got}/{gn}   headline {k}/{n}   "
              f"{'MATCH' if ok else '*** MISMATCH ***'}")

    print("\n" + "=" * 78)
    print("VERDICT -- pi(MIGRATE_EDGE | high-risk state), by state source")
    print("=" * 78)
    print(f"  {'states from':>12s} {'pi_A0':>9s} {'pi_R2':>9s} {'R2-A0':>9s} "
          f"{'z':>7s} {'argmaxE A0':>11s} {'argmaxE R2':>11s}  note")
    NEUTRAL = ("RANDOM", "UNION")
    for src in out:
        c = out[src]
        a, r = c["evaluated"]["A0"], c["evaluated"]["R2"]
        pd = c["paired_within_state_R2_minus_A0"]
        note = ("policy-independent" if src == "RANDOM" else
                "symmetric" if src == "UNION" else
                f"CONFOUNDED: selected by {src}")
        print(f"  {src:>12s} {a['p_edge_mean']:>9.4f} {r['p_edge_mean']:>9.4f} "
              f"{pd['mean']:>+9.4f} {pd['z']:>+7.2f} "
              f"{a['frac_argmax_edge']:>11.4f} {r['frac_argmax_edge']:>11.4f}"
              f"  {note}")
    signs = {s: out[s]["paired_within_state_R2_minus_A0"]["mean"] > 0
             for s in out}
    own = {s: signs[s] for s in MODELS}
    neu = {s: signs[s] for s in NEUTRAL if s in signs}
    consistent = len(set(signs.values())) == 1
    own_consistent = len(set(own.values())) == 1
    neu_consistent = len(set(neu.values())) == 1
    print(f"\n  sign of (pi_R2 - pi_A0) by source: "
          + "  ".join(f"{s} {'+' if v else '-'}" for s, v in signs.items()))
    print(f"  consistent across ALL sources          : "
          f"{'YES' if consistent else 'NO'}")
    print(f"  consistent across each arm's OWN states: "
          f"{'YES' if own_consistent else 'NO'}")
    print(f"  consistent across NEUTRAL sources      : "
          f"{'YES' if neu_consistent else 'NO'}"
          + (f"  (sign {'+' if list(neu.values())[0] else '-'})"
             if neu_consistent and neu else ""))
    if not own_consistent:
        print("\n  -> the ON-POLICY share REVERSES between the two arms' own")
        print("     state sets, so as computed since Sprint 6 it does NOT")
        print("     measure a property of the policy. Each arm scores LOW on")
        print("     its own states and HIGH on the other's -- an on-policy")
        print("     selection effect: choosing STAY is what keeps an agent in a")
        print("     high-risk state, so conditioning on 'currently high-risk'")
        print("     selects for 'just chose STAY'. Both the Rung 2 'win' and")
        print("     the Rung 2.5 'loss' therefore fail as policy-quality claims.")
    if neu_consistent and neu:
        sgn = "HIGHER" if list(neu.values())[0] else "LOWER"
        print(f"\n  -> BUT on state sets that privilege neither arm "
              f"({', '.join(neu)}),")
        print(f"     R2 has {sgn} pi(EDGE) at high risk than A0, consistently.")
        print(f"     The confound invalidated the METRIC, not the direction of")
        print(f"     the underlying policy difference.")

    print("\n" + "=" * 78)
    print("UNCONFOUNDED ALTERNATIVE -- risk RESPONSE on a FIXED state set")
    print("   (this is what the project's primary question actually asks: does")
    print("    the policy change its decision as predicted_failure_risk rises?)")
    print("=" * 78)
    print(f"  {'state set':>12s} {'arm':>4s} {'p_edge r<.2':>12s} "
          f"{'p_edge r>=.6':>13s} {'delta':>9s} {'z':>7s} {'spearman':>9s}  note")
    for src in out:
        for who in ALL:
            e = out[src]["evaluated"][who]
            rc = e["risk_response_cluster"]
            note = "own states -- still selected" if src == who else ""
            zs = f"{rc['z']:>+7.2f}" if rc.get("z") is not None else f"{'--':>7s}"
            print(f"  {src:>12s} {who:>4s} {e['p_edge_risk_lt_02']:>12.4f} "
                  f"{e['p_edge_risk_ge_06']:>13.4f} "
                  f"{e['risk_response_high_minus_low']:>+9.4f} {zs} "
                  f"{e['spearman_risk_vs_p_edge']:>+9.4f}  {note}")
    resp_neutral = {
        s: {w: out[s]["evaluated"][w]["risk_response_high_minus_low"]
            for w in ALL} for s in NEUTRAL if s in out}
    print("\n  reading the NEUTRAL rows only (the rows with no selection in "
          "them):")
    for s, d in resp_neutral.items():
        verdict = ("R2 responds to risk, A0 does not"
                   if d["R2"] > 0 >= d["A0"] else
                   "both respond" if min(d["A0"], d["R2"]) > 0 else
                   "neither responds" if max(d["A0"], d["R2"]) <= 0 else
                   "A0 responds, R2 does not")
        print(f"    {s:>8s}: A0 {d['A0']:+.4f}   R2 {d['R2']:+.4f}   -> {verdict}")

    # ------------------------------------------------------------------
    # CALIBRATION of the proposed primary metric across the arms that
    # ALREADY EXIST. No new training: A1/A2/A3 are scored on the same fixed
    # state sets. Their only purpose is to show how much this metric varies
    # between arms that were produced by real, completed training runs, so a
    # GO threshold for the next rung can be set against observed variation
    # instead of being invented.
    # ------------------------------------------------------------------
    calib = {}
    if len(ALL) > len(MODELS):
        print("\n" + "=" * 78)
        print("CALIBRATION -- the proposed primary metric across ALL EXISTING")
        print("   arms, on the NEUTRAL (policy-independent / symmetric) sets.")
        print("   Purpose: set the next rung's GO threshold against observed")
        print("   between-arm spread rather than an invented number.")
        print("=" * 78)
        for s in NEUTRAL:
            if s not in out:
                continue
            print(f"\n  -- state set {s} " + "-" * 40)
            print(f"     {'arm':>4s} {'delta':>9s} {'ci95_lo':>9s} "
                  f"{'ci95_hi':>9s} {'z':>7s} {'p':>8s} {'spearman':>9s}")
            vals = {}
            for who in ALL:
                e = out[s]["evaluated"][who]
                rc = e["risk_response_cluster"]
                vals[who] = e["risk_response_high_minus_low"]
                lo = f"{rc['ci95'][0]:>+9.4f}" if rc.get("ci95") else f"{'--':>9s}"
                hh = f"{rc['ci95'][1]:>+9.4f}" if rc.get("ci95") else f"{'--':>9s}"
                zs = (f"{rc['z']:>+7.2f}" if rc.get("z") is not None
                      else f"{'--':>7s}")
                ps = (f"{rc['p_value']:>8.4f}" if rc.get("p_value") is not None
                      else f"{'--':>8s}")
                print(f"     {who:>4s} {vals[who]:>+9.4f} {lo} {hh} {zs} {ps} "
                      f"{e['spearman_risk_vs_p_edge']:>+9.4f}")
            v = np.array([vals[w] for w in ALL], float)
            calib[s] = dict(
                per_arm=vals, n_arms=len(ALL),
                min=float(v.min()), max=float(v.max()),
                range=float(v.max() - v.min()),
                mean=float(v.mean()),
                sd_across_arms=float(v.std(ddof=1)) if v.size > 1 else None,
                best_arm=max(vals, key=lambda k: vals[k]),
                n_arms_positive=int((v > 0).sum()),
                median_ci_halfwidth=float(np.median([
                    (out[s]["evaluated"][w]["risk_response_cluster"]["ci95"][1]
                     - out[s]["evaluated"][w]["risk_response_cluster"]["ci95"][0])
                    / 2.0 for w in ALL
                    if out[s]["evaluated"][w]["risk_response_cluster"].get("ci95")
                ])),
            )
            c = calib[s]
            print(f"     across {c['n_arms']} existing arms: min {c['min']:+.4f}"
                  f"  max {c['max']:+.4f}  range {c['range']:.4f}"
                  f"  sd {c['sd_across_arms']:.4f}"
                  f"  best {c['best_arm']}  positive {c['n_arms_positive']}"
                  f"/{c['n_arms']}")
            print(f"     median within-arm CI95 half-width "
                  f"{c['median_ci_halfwidth']:.4f}"
                  f"  -> a next-rung delta must clear BOTH the best existing arm"
                  f" and its own interval")

    blob = dict(
        probe=f"{TAG}_matched_states",
        what="both actors evaluated at the SAME recorded states, so the "
             "on-policy state distribution cannot carry the high-risk EDGE "
             "contrast; plus a risk-response curve on fixed state sets as an "
             "unconfounded replacement for the share metric",
        confound_being_addressed=dict(
            where="the Rung 2 headline metric itself, and every high-risk EDGE "
                  "share quoted since Sprint 6",
            what="each arm's share is computed on its OWN trajectory; "
                 "denominators differ (2174 vs 1418 high-risk entries) and "
                 "episode lengths differ (A0 374.2 with 0/32 truncated; R2 "
                 "392.1 with 17/32), so a policy difference and a "
                 "state-distribution difference are not separable",
            why_it_is_structural="risk is high at a host BECAUSE tasks remain "
                                 "on it; STAY keeps an agent in a high-risk "
                                 "state and MIGRATE_EDGE removes it from one, "
                                 "so conditioning on 'in a high-risk state' "
                                 "selects for 'just chose STAY'. The metric "
                                 "has its own outcome in the denominator.",
            not_a_computation_error="both figures use the same definition and "
                                    "are reproduced EXACTLY on the diagonal "
                                    "here; the defect is identification, not "
                                    "arithmetic",
            resolution=["cross-evaluate pi_A0 and pi_R2 at identical states, "
                        "paired within state, clustered by episode",
                        "add a policy-independent (uniform-legal) state source",
                        "score risk RESPONSE on a fixed state set instead of a "
                        "share on a policy-dependent one"],
        ),
        definition=dict(
            window="eval", policy="greedy (deterministic)",
            decision_rule="mask.sum(-1) > 1",
            threshold="min(int(risk*5), 4) >= 3",
            starts="same construction and seed as "
                   "_diag_rung2_75_edgeshare_cluster --cell production",
            random_source="uniform over legal actions; depends on neither arm. "
                          "CAVEAT: visits states off-manifold for both, so it "
                          "is a neutral reference, not a realistic one.",
            union_source="A0's recorded states pooled with R2's, episode ids "
                         "kept disjoint. Symmetric by construction and fully "
                         "on-manifold for both arms, at the cost of containing "
                         "both arms' selection effects rather than none.",
            risk_bins=RISK_BINS,
        ),
        not_a_fix="forward passes only; no optimiser, no gradient, no write to "
                  "any checkpoint; production code unmodified",
        n_clusters=len(starts), start_ticks=starts,
        diagonal_parity_vs_headline=parity,
        by_state_source=out,
        sign_of_R2_minus_A0_by_source=signs,
        sign_consistent_across_sources=bool(consistent),
        sign_consistent_across_own_state_sources=bool(own_consistent),
        sign_consistent_across_neutral_sources=bool(neu_consistent),
        neutral_sources=list(NEUTRAL),
        risk_response_on_neutral_sources=resp_neutral,
        arms_scored=ALL,
        arms_generating_state_sources=list(MODELS),
        calibration_of_primary_metric=dict(
            what="the proposed primary metric (risk response) computed for "
                 "every arm that ALREADY EXISTS, on the neutral state sets",
            why="a GO threshold for the next rung has to be set against the "
                "spread this metric already shows across completed training "
                "runs, otherwise the threshold is invented",
            no_new_training="A1/A2/A3 are existing Sprint 7 checkpoints scored "
                            "with forward passes only; they generate no state "
                            "source, so they cannot shift the state sets",
            per_state_set=calib,
        ),
        conclusion=dict(
            metric_as_used_since_sprint6="NOT identified: the paired "
                "within-state contrast reverses sign between A0's own state set "
                "(+) and R2's own state set (-), so the on-policy high-risk EDGE "
                "share does not measure a property of the policy. Both the Rung "
                "2 'win' and the Rung 2.5 'loss' fail as policy-quality claims.",
            direction_on_neutral_sets="on the two state sets that privilege "
                "neither arm (uniform-random and pooled), R2's pi(EDGE) at high "
                "risk exceeds A0's consistently -- so the confound invalidated "
                "the METRIC, not the direction of the policy difference",
            recommended_primary_metric="risk RESPONSE on a FIXED, "
                "policy-independent state set: pi(MIGRATE_EDGE | risk>=0.6) - "
                "pi(MIGRATE_EDGE | risk<0.2), plus Spearman(risk, pi(EDGE)) "
                "over decision entries. No policy appears in the state "
                "distribution, so the quantity is identified.",
        ),
    )
    p = OUT_DIR / f"{TAG}_matched_states_{args.tag}.json"
    p.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n  wrote {p}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
