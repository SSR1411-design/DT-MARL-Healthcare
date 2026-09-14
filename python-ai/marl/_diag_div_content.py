#!/usr/bin/env python
"""
SPRINT 7 DIVERGENCE DIAGNOSTIC — part 4 of 4: WHAT WAS IN THE UPDATE?

OFFLINE. Rebuilds each arm's own on-policy training buffer with the EXISTING
`_diag_rung2_75_coherence.build` machinery, then characterises the CONTENT of a
PPO update: risk occupancy, action frequencies, reward/return/advantage
distributions, critic error and TD residuals per risk bucket, within-update
episode heterogeneity, and the exact minibatch decomposition. No optimiser is
constructed, no parameter is written, no production module is touched.

SCOPE NOTE, AND IT IS A HARD ONE. The pre-registration forbids using R3's own
trajectory to define a state set for a POLICY-QUALITY comparison, and the fixed
RANDOM/UNION sets stay exactly as defined. Nothing here scores policy quality.
Every quantity below is a property of the DATA THE OPTIMISER CONSUMED — the
update content — which is precisely what candidate mechanism 1 asks for. Where a
number could be misread as a quality comparison (e.g. the selected-EDGE rate at
high risk on an arm's own rollout) it is labelled ON-OWN-DISTRIBUTION and is not
compared against the locked P1/P2 thresholds.

THE MEASUREMENT THAT MOTIVATES THIS PROBE. `mappo.py:363-366` normalises the
advantage over the decision-entry pool of the WHOLE buffer:

    adv <- (adv - mean(adv[dec])) / (std(adv[dec]) + 1e-8)

The per-sample actor gradient weight is therefore not the raw advantage but its
z-score against a pool whose size is `rollout_episodes`-dependent. R2 pools 8
episodes, R3 pools 32. So the SAME environment event can enter R2's update with a
different gradient weight than it enters R3's, with no change to the reward
function, the critic, or the estimator. `z_hi_EDGE` below is that weight for the
one channel the behavioural metric is about, and it is reported alongside the
pool statistics that produce it.

Sections
  C1  occupancy and action frequency by risk bucket        (candidate 1)
  C2  reward / return / critic error / TD residual         (candidates 1, 4)
  C3  advantage distribution, tails, and calibration       (candidates 1, 4)
  C4  within-update episode heterogeneity and mass         (candidate 2)
  C5  exact minibatch decomposition and reuse accounting   (candidate 3)

Writes SPRINT_7_DIV_content_<tag>.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[0]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from marl._diag_rung0 import load_agent_and_cfg, OUT_DIR          # noqa: E402
from marl._diag_rung2_75_coherence import build                   # noqa: E402
from marl._diag_rung2_75_matched_states import ACTION_NAMES       # noqa: E402
from marl._diag_div_geometry import HI, LO                        # noqa: E402

TAG = "SPRINT_7_DIV"

# (label, checkpoint, rollout_episodes override or None = the arm's own)
# R2_b32 is the MATCHED-BATCH control: the same policy, the same start-tick
# stream, a 32-episode buffer. It separates "what a 32-episode buffer does to the
# statistics" from "what R3's parameters do", which the D3/D4 census cannot --
# that probe reads rollout_episodes from the checkpoint config and ignores
# --episodes, so `--tag R2_b32` there reproduced the 8-episode numbers exactly.
CELLS = [
    ("A0",     "mappo_A0_cpu_repro.pth", None),
    ("R2",     "mappo_R2_mc_target.pth", None),
    ("R2_b32", "mappo_R2_mc_target.pth", 32),
    ("R3_best", "R3_batch32_best.pth",   None),
    ("R3",     "R3_batch32.pth",         None),
]
I_STAY = ACTION_NAMES.index("STAY")
I_EDGE = ACTION_NAMES.index("MIGRATE_EDGE")
I_CLOUD = ACTION_NAMES.index("MIGRATE_CLOUD")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


def q(x, ps=(1, 5, 25, 50, 75, 95, 99)):
    if x.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


def stat(x):
    if x.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan")}
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std())}


def ev(ret, val):
    """Explained variance, the standard 1 - Var(residual)/Var(target)."""
    if ret.size < 2:
        return float("nan")
    vt = float(ret.var())
    return float("nan") if vt < 1e-12 else float(1.0 - (ret - val).var() / vt)


# ------------------------------------------------------------------ per cell

def measure(label, ckpt, n_eps_override, device):
    agent, extra, cfg = load_agent_and_cfg(str(OUT_DIR / ckpt), device, "train")
    own_eps = int(extra["config"]["train"]["rollout_episodes"])
    n_eps = own_eps if n_eps_override is None else int(n_eps_override)
    target = getattr(agent.cfg, "critic_target", "lambda")

    buf, risk, starts, trunc = build(agent, cfg, n_eps)
    T = len(buf)
    n_ag = agent.n_agents
    g, lam = agent.cfg.gamma, agent.cfg.gae_lambda

    dec = buf.decision[:T] > 0.5
    act = buf.act[:T]
    rew = buf.rew[:T]
    val = buf.val[:T]
    cont = buf.cont[:T, 0]
    boot = buf.boot[:T]

    adv_raw, ret_gae = agent.compute_gae(buf)
    ret = agent.compute_mc_returns(buf) if target == "mc" else ret_gae

    # exactly mappo.py:363-366
    pool = adv_raw[dec]
    p_mean, p_std = float(pool.mean()), float(pool.std())
    adv_n = (adv_raw - p_mean) / (p_std + 1e-8)

    # TD residual, exactly compute_gae's delta
    nxt = np.empty_like(val)
    for t in range(T):
        nxt[t] = val[t + 1] if (cont[t] > 0.5 and t + 1 < T) else boot[t]
    td = rew + g * nxt - val

    # episode index per timestep: cont == 0 closes an episode
    ep = np.zeros(T, np.int64)
    e = 0
    for t in range(T):
        ep[t] = e
        if cont[t] <= 0.5:
            e += 1
    EP = np.broadcast_to(ep[:, None], (T, n_ag))
    n_ep_seen = int(e)

    hi = dec & (risk >= HI)
    lo = dec & (risk < LO)
    hi50 = dec & (risk >= 0.50)
    mid = dec & ~hi & ~lo
    hiE = hi & (act == I_EDGE)
    loE = lo & (act == I_EDGE)

    R = {"label": label, "checkpoint": ckpt, "own_rollout_episodes": own_eps,
         "rollout_episodes_used": n_eps, "critic_target": target,
         "start_ticks": [int(s) for s in starts],
         "n_truncated_episodes": int(sum(bool(t) for t in trunc)),
         "gamma": float(g), "gae_lambda": float(lam)}

    # -------------------------------------------------- C1 occupancy / actions
    c1 = {"timesteps": T, "agents": n_ag, "entries": T * n_ag,
          "episodes_in_buffer": n_ep_seen,
          "decision_entries": int(dec.sum()),
          "decision_frac": float(dec.mean()),
          "hi_decision": int(hi.sum()), "lo_decision": int(lo.sum()),
          "mid_decision": int(mid.sum()),
          "hi50_decision": int(hi50.sum()),
          "hi_share_of_decision": float(hi.sum() / max(1, dec.sum())),
          "steps_per_episode_mean": T / max(1, n_ep_seen)}
    for nm, sel in (("all", dec), ("hi", hi), ("lo", lo)):
        tot = max(1, int(sel.sum()))
        for ai, an in enumerate(ACTION_NAMES):
            c1[f"sel_{an}_{nm}"] = int((sel & (act == ai)).sum())
            c1[f"selfrac_{an}_{nm}"] = float((sel & (act == ai)).sum() / tot)
        c1[f"legal_CLOUD_{nm}"] = int((sel & (buf.mask[:T, :, I_CLOUD] > 0.5)).sum())
    # ON-OWN-DISTRIBUTION selection contrast. NOT a policy-quality metric and
    # NOT comparable to the locked P1/P2, which live on the fixed eval sets.
    c1["own_dist_selfrac_EDGE_hi_minus_lo"] = (c1["selfrac_MIGRATE_EDGE_hi"]
                                               - c1["selfrac_MIGRATE_EDGE_lo"])
    R["C1_occupancy_actions"] = c1

    # -------------------------------------------- C2 reward / critic / TD
    c2 = {}
    for nm, sel in (("all", dec), ("hi", hi), ("lo", lo),
                    ("hi_EDGE", hiE), ("lo_EDGE", loE)):
        c2[nm] = {"reward": stat(rew[sel]), "return": stat(ret[sel]),
                  "value": stat(val[sel]),
                  "value_err": stat(ret[sel] - val[sel]),
                  "abs_value_err": stat(np.abs(ret[sel] - val[sel])),
                  "td_residual": stat(td[sel]),
                  "abs_td_residual": stat(np.abs(td[sel])),
                  "explained_var": ev(ret[sel], val[sel])}
    R["C2_reward_critic_td"] = c2

    # ------------------------------------- C3 advantage distribution / calib
    c3 = {"pool_mean_raw": p_mean, "pool_std_raw": p_std,
          "normalisation_scale": 1.0 / (p_std + 1e-8),
          "pool_n": int(dec.sum())}
    for nm, sel in (("all", dec), ("hi", hi), ("lo", lo)):
        c3[nm] = {"raw": stat(adv_raw[sel]), "raw_q": q(adv_raw[sel]),
                  "norm": stat(adv_n[sel]), "norm_q": q(adv_n[sel]),
                  "frac_positive": float((adv_raw[sel] > 0).mean())
                  if sel.sum() else float("nan")}
    # calibration: the normalised advantage IS the per-sample gradient weight
    cal = {}
    for nm, sel in (("hi", hi), ("lo", lo)):
        for ai, an in enumerate(ACTION_NAMES):
            s = sel & (act == ai)
            cal[f"{nm}_{an}"] = {"n": int(s.sum()),
                                 "mean_adv_norm": float(adv_n[s].mean())
                                 if s.sum() else float("nan"),
                                 "mean_adv_raw": float(adv_raw[s].mean())
                                 if s.sum() else float("nan"),
                                 "frac_positive": float((adv_n[s] > 0).mean())
                                 if s.sum() else float("nan")}
    c3["calibration_by_bucket_action"] = cal
    c3["z_hi_EDGE"] = cal["hi_MIGRATE_EDGE"]["mean_adv_norm"]
    c3["z_hi_STAY"] = cal["hi_STAY"]["mean_adv_norm"]
    c3["z_lo_EDGE"] = cal["lo_MIGRATE_EDGE"]["mean_adv_norm"]
    c3["z_hi_EDGE_minus_hi_STAY"] = c3["z_hi_EDGE"] - c3["z_hi_STAY"]
    # gradient-weight share: |adv_n| mass, since the PG scales linearly in adv
    m_all = float(np.abs(adv_n[dec]).sum())
    for nm, sel in (("hi", hi), ("lo", lo), ("hi_EDGE", hiE), ("lo_EDGE", loE)):
        c3[f"absadv_mass_share_{nm}"] = (float(np.abs(adv_n[sel]).sum()) / m_all
                                         if m_all > 0 else float("nan"))
    c3["signed_adv_mass_share_hi_EDGE"] = (float(adv_n[hiE].sum()) / m_all
                                           if m_all > 0 else float("nan"))
    R["C3_advantage"] = c3

    # --------------------------------- C4 within-update episode heterogeneity
    uniq = np.arange(n_ep_seen)
    per = {"hi_decision": [], "dec": [], "hi_EDGE": [], "mean_adv_norm": [],
           "mean_reward": [], "absadv_mass_hi_EDGE": [], "steps": []}
    for eid in uniq:
        m = EP == eid
        per["steps"].append(int((ep == eid).sum()))
        per["dec"].append(int((m & dec).sum()))
        per["hi_decision"].append(int((m & hi).sum()))
        per["hi_EDGE"].append(int((m & hiE).sum()))
        per["mean_adv_norm"].append(float(adv_n[m & dec].mean())
                                    if (m & dec).sum() else float("nan"))
        per["mean_reward"].append(float(rew[m & dec].mean())
                                  if (m & dec).sum() else float("nan"))
        per["absadv_mass_hi_EDGE"].append(float(np.abs(adv_n[m & hiE]).sum()))
    mass = np.asarray(per["absadv_mass_hi_EDGE"])
    pr = float(mass.sum() ** 2 / (mass ** 2).sum()) if (mass ** 2).sum() > 0 \
        else float("nan")
    hiE_arr = np.asarray(per["hi_EDGE"], float)
    ma = np.asarray(per["mean_adv_norm"], float)
    c4 = {"n_episodes": n_ep_seen, "per_episode": per,
          "hi_EDGE_per_episode_mean": float(hiE_arr.mean()),
          "hi_EDGE_per_episode_std": float(hiE_arr.std()),
          "frac_episodes_zero_hi_EDGE": float((hiE_arr == 0).mean()),
          "hi_EDGE_mass_participation_ratio": pr,
          "hi_EDGE_mass_PR_over_n_episodes": pr / max(1, n_ep_seen),
          "between_episode_sd_mean_adv_norm": float(np.nanstd(ma)),
          "between_episode_cv_hi_decision":
              float(np.std(per["hi_decision"])
                    / max(1e-9, np.mean(per["hi_decision"]))),
          "start_tick_span": int(max(starts) - min(starts)) if starts else 0}
    R["C4_within_update_mixing"] = c4

    # ------------------------------ C5 exact minibatch decomposition / reuse
    n_mb = max(1, agent.cfg.minibatches)
    mb_size = max(1, T // n_mb)
    bounds = list(range(0, T, mb_size))
    sizes = [min(mb_size, T - b) for b in bounds]
    rng = np.random.default_rng(0)          # mappo.py:370, re-seeded per update
    mbs = []
    for _ in range(agent.cfg.ppo_epochs):
        order = rng.permutation(T)
        for b, s in zip(bounds, sizes):
            idx = order[b:b + mb_size]
            d = dec[idx]
            mbs.append({"size_t": int(idx.size),
                        "decision": int(d.sum()),
                        "hi": int(hi[idx].sum()),
                        "hi_EDGE": int(hiE[idx].sum()),
                        "absadv_mass_hi_EDGE": float(np.abs(adv_n[idx][hiE[idx]]).sum()),
                        "absadv_mass_dec": float(np.abs(adv_n[idx][d]).sum())})
    hiE_mb = np.asarray([m["hi_EDGE"] for m in mbs], float)
    share = np.asarray([m["absadv_mass_hi_EDGE"] / m["absadv_mass_dec"]
                        if m["absadv_mass_dec"] > 0 else np.nan for m in mbs])
    c5 = {"minibatches_cfg": int(n_mb), "ppo_epochs": int(agent.cfg.ppo_epochs),
          "T": T, "mb_size": mb_size, "chunks_per_epoch": len(bounds),
          "chunk_sizes": sizes, "tail_size": sizes[-1],
          "tail_fraction_of_T": sizes[-1] / T,
          "has_degenerate_tail": len(bounds) > n_mb,
          "minibatches_per_update": len(mbs),
          "timestep_rows_consumed_per_update": int(agent.cfg.ppo_epochs * T),
          "gradient_passes_per_env_decision": int(agent.cfg.ppo_epochs),
          "hi_EDGE_per_mb_mean": float(hiE_mb.mean()),
          "hi_EDGE_per_mb_min": float(hiE_mb.min()),
          "frac_mb_zero_hi_EDGE": float((hiE_mb == 0).mean()),
          "absadv_share_hi_EDGE_per_mb_mean": float(np.nanmean(share)),
          "absadv_share_hi_EDGE_per_mb_std": float(np.nanstd(share)),
          "per_minibatch": mbs}
    R["C5_minibatch"] = c5
    return R


# --------------------------------------------------------------------- print

def show(rows):
    def line(title, keys, fmt, get):
        print(f"\n  {title}")
        print(f"  {'metric':<38s}" + "".join(f"{r['label']:>11s}" for r in rows))
        for k, lab in keys:
            vs = [get(r, k) for r in rows]
            print(f"  {lab:<38s}" + "".join(
                (f"{v:>11{fmt}}" if isinstance(v, float) else f"{v:>11}")
                for v in vs))

    print("\n" + "=" * 78)
    print("C1  UPDATE OCCUPANCY AND ACTION FREQUENCY (each arm's own buffer)")
    print("=" * 78)
    line("size and risk occupancy", [
        ("rollout_episodes_used", "rollout_episodes"),
        ("timesteps", "timesteps T"),
        ("decision_entries", "decision entries"),
        ("decision_frac", "decision_frac"),
        ("hi_decision", f"hi (risk>={HI}) decision"),
        ("lo_decision", f"lo (risk<{LO}) decision"),
        ("hi_share_of_decision", "hi share of decision"),
    ], ".4f", lambda r, k: r["C1_occupancy_actions"].get(
        k, r.get(k)))
    line("selected-action fraction, ALL decision entries", [
        ("selfrac_STAY_all", "STAY"),
        ("selfrac_MIGRATE_EDGE_all", "MIGRATE_EDGE"),
        ("selfrac_MIGRATE_CLOUD_all", "MIGRATE_CLOUD"),
    ], ".4f", lambda r, k: r["C1_occupancy_actions"][k])
    line("ON-OWN-DISTRIBUTION selected EDGE (not P1/P2)", [
        ("selfrac_MIGRATE_EDGE_hi", "at high risk"),
        ("selfrac_MIGRATE_EDGE_lo", "at low risk"),
        ("own_dist_selfrac_EDGE_hi_minus_lo", "hi - lo"),
    ], ".4f", lambda r, k: r["C1_occupancy_actions"][k])

    print("\n" + "=" * 78)
    print("C2  REWARD, CRITIC ERROR AND TD RESIDUAL BY RISK BUCKET")
    print("=" * 78)
    for b in ("all", "hi", "lo"):
        line(f"bucket = {b}", [
            ("reward.mean", "mean reward"),
            ("return.mean", "mean return (arm's target)"),
            ("value.mean", "mean V(s)"),
            ("value_err.mean", "mean (ret - V)"),
            ("abs_value_err.mean", "mean |ret - V|"),
            ("explained_var", "explained variance"),
            ("td_residual.mean", "mean TD residual"),
            ("abs_td_residual.mean", "mean |TD residual|"),
        ], ".4f", lambda r, k, b=b: (
            r["C2_reward_critic_td"][b][k.split(".")[0]][k.split(".")[1]]
            if "." in k else r["C2_reward_critic_td"][b][k]))

    print("\n" + "=" * 78)
    print("C3  ADVANTAGE DISTRIBUTION AND PER-SAMPLE GRADIENT WEIGHT")
    print("=" * 78)
    line("normalisation pool (mappo.py:363-366)", [
        ("pool_n", "pool size (decision entries)"),
        ("pool_mean_raw", "pool mean raw adv"),
        ("pool_std_raw", "pool std raw adv"),
        ("normalisation_scale", "1/(std+1e-8)"),
    ], ".4f", lambda r, k: r["C3_advantage"][k])
    line("raw GAE by bucket", [
        ("hi.raw.std", "hi raw std"),
        ("lo.raw.std", "lo raw std"),
        ("hi.raw.mean", "hi raw mean"),
        ("lo.raw.mean", "lo raw mean"),
        ("hi.raw_q.p99", "hi raw p99"),
        ("hi.raw_q.p1", "hi raw p1"),
    ], ".4f", lambda r, k: _dig(r["C3_advantage"], k))
    line("NORMALISED advantage = gradient weight", [
        ("z_hi_EDGE", "z(hi, EDGE)"),
        ("z_hi_STAY", "z(hi, STAY)"),
        ("z_hi_EDGE_minus_hi_STAY", "z(hi,EDGE) - z(hi,STAY)"),
        ("z_lo_EDGE", "z(lo, EDGE)"),
        ("absadv_mass_share_hi", "|adv| mass share hi"),
        ("absadv_mass_share_hi_EDGE", "|adv| mass share hi&EDGE"),
        ("signed_adv_mass_share_hi_EDGE", "signed mass share hi&EDGE"),
    ], ".4f", lambda r, k: r["C3_advantage"][k])

    print("\n" + "=" * 78)
    print("C4  WITHIN-UPDATE EPISODE HETEROGENEITY")
    print("=" * 78)
    line("", [
        ("n_episodes", "episodes pooled per update"),
        ("start_tick_span", "start-tick span"),
        ("hi_EDGE_per_episode_mean", "hi&EDGE per episode (mean)"),
        ("hi_EDGE_per_episode_std", "hi&EDGE per episode (sd)"),
        ("frac_episodes_zero_hi_EDGE", "frac episodes with 0 hi&EDGE"),
        ("hi_EDGE_mass_participation_ratio", "hi&EDGE mass PR (eff. eps)"),
        ("hi_EDGE_mass_PR_over_n_episodes", "PR / n_episodes"),
        ("between_episode_sd_mean_adv_norm", "between-ep sd of mean z"),
        ("between_episode_cv_hi_decision", "between-ep CV of hi count"),
    ], ".4f", lambda r, k: r["C4_within_update_mixing"][k])

    print("\n" + "=" * 78)
    print("C5  EXACT MINIBATCH DECOMPOSITION AND REUSE")
    print("=" * 78)
    line("", [
        ("T", "T"),
        ("mb_size", "mb_size = T//4"),
        ("chunks_per_epoch", "chunks per epoch"),
        ("tail_size", "tail chunk size"),
        ("tail_fraction_of_T", "tail / T"),
        ("minibatches_per_update", "minibatches per update"),
        ("timestep_rows_consumed_per_update", "timestep rows per update"),
        ("gradient_passes_per_env_decision", "grad passes per decision"),
        ("hi_EDGE_per_mb_mean", "hi&EDGE per minibatch (mean)"),
        ("hi_EDGE_per_mb_min", "hi&EDGE per minibatch (min)"),
        ("frac_mb_zero_hi_EDGE", "frac minibatches with 0"),
        ("absadv_share_hi_EDGE_per_mb_mean", "|adv| share hi&EDGE per mb"),
    ], ".4f", lambda r, k: r["C5_minibatch"][k])


def _dig(d, path):
    for k in path.split("."):
        d = d[k]
    return d


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 DIVERGENCE — part 4: UPDATE CONTENT / DATA DISTRIBUTION")
    print("        offline; each arm's own on-policy buffer via the existing")
    print("        _diag_rung2_75_coherence.build; no optimiser, no writes")
    print("=" * 78)
    rows = []
    for label, ckpt, ov in CELLS:
        print(f"\n  building {label:<8s} ({ckpt}"
              f"{'' if ov is None else f', forced {ov} episodes'}) ...")
        r = measure(label, ckpt, ov, args.device)
        print(f"    T={r['C1_occupancy_actions']['timesteps']}  "
              f"decision={r['C1_occupancy_actions']['decision_entries']}  "
              f"hi={r['C1_occupancy_actions']['hi_decision']}  "
              f"episodes={r['C4_within_update_mixing']['n_episodes']}")
        rows.append(r)
    show(rows)
    out = {"probe": "DIV-4 update content / data distribution",
           "hi_cut": HI, "lo_cut": LO, "args": vars(args),
           "cells": {r["label"]: r for r in rows}}
    p = OUT_DIR / f"{TAG}_content_{args.tag}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
