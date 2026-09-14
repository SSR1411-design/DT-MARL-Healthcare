#!/usr/bin/env python
"""
SPRINT 7 DIVERGENCE DIAGNOSTIC — part 5 of 5: IS THE HIGH-RISK LEARNING SIGNAL
AN ESTIMATOR WHOSE *VARIANCE* IS THE THING THAT CHANGED?

OFFLINE. Builds ONE large on-policy buffer per policy with the existing
`_diag_rung2_75_coherence.build`, then re-derives the update statistics that
production would have computed on an 8-episode and on a 32-episode buffer by
SUBSAMPLING EPISODES from that population. Nothing is trained, no parameter is
written, no production module is touched.

WHY THIS PROBE EXISTS. `SPRINT_7_DIV_content_*` ran a matched-batch control that
the D3/D4 census cannot run (that probe reads `rollout_episodes` from the
checkpoint config and ignores `--episodes`). Holding R2's PARAMETERS fixed and
changing only the buffer from 8 to 32 episodes:

    z(hi,EDGE) - z(hi,STAY)      R2 @ 8 eps  +0.9080      R2 @ 32 eps  +0.1769
    z(hi,EDGE) alone             R2 @ 8 eps  +0.2988      R2 @ 32 eps  -0.1461

where z is the NORMALISED advantage, i.e. literally the per-sample coefficient
the actor loss puts on log pi(a|s) (`mappo.py:363-366` normalises over the whole
buffer's decision pool, so the pool is `rollout_episodes`-dependent). The same
policy, the same nested start-tick stream, no critic change, no estimator change
— and the high-risk EDGE-versus-STAY drive falls 5.1x and z(hi,EDGE) changes
sign. That is a difference between R2 and R3 that exists in the DATA, before any
parameter moves.

But an 8-episode buffer contains only ~10 high-risk EDGE entries, so +0.9080 may
simply be a draw from a wide sampling distribution rather than a signal a bigger
buffer destroys. Those two readings have opposite implications and this probe
separates them, using the only instrument that can: the sampling distribution of
the statistic itself, at both buffer sizes, from real data, with the policy held
fixed.

THE STATISTIC THAT MATTERS. In `mappo.py` the actor loss is
`-(min(s1,s2) * dec).sum() / dec.sum()`, so at ratio == 1 the coefficient
multiplying `log pi(EDGE | high-risk state)`, summed over the buffer, is

    channel_drive(hi,EDGE) = (n_hi_EDGE / n_decision) * mean z(hi,EDGE)

That product — not z alone — is the per-update push on the channel P1/P2 measure.
Both factors are `rollout_episodes`-dependent and both are reported.

SUBSAMPLING IS EXACT, NOT APPROXIMATE. Episodes are drawn by
`training_start_ticks`, whose rng is re-seeded from `TRAIN_SEED`, so the first 8
start ticks of a 128-episode stream ARE R2's 8 and the first 32 ARE R3's 32
(already verified in `SPRINT_7_DIV_logs_*` section 2). GAE never crosses an
episode boundary (`cont == 0` zeroes the trace, `mappo.py:305`), so an episode's
per-entry advantages are identical whether it sits in an 8-episode or a
128-episode buffer. Selecting an episode subset therefore reproduces exactly the
advantages production would have had, and only the NORMALISATION POOL changes —
which is the effect under study. Pool mean/std are recomputed per subsample for
that reason.

EXPLORATORY. This probe was designed after seeing part 4's matched-batch numbers.
It changes no locked verdict and sets no threshold.

Writes SPRINT_7_DIV_variance_<tag>.json.
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
from marl._diag_rung2_75_coherence import build                   # noqa: E402
from marl._diag_rung2_75_matched_states import ACTION_NAMES       # noqa: E402
from marl._diag_div_geometry import HI, LO                        # noqa: E402

TAG = "SPRINT_7_DIV"
CELLS = [("R2", "mappo_R2_mc_target.pth", 8),
         ("R3", "R3_batch32.pth", 32)]
I_STAY = ACTION_NAMES.index("STAY")
I_EDGE = ACTION_NAMES.index("MIGRATE_EDGE")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--population", type=int, default=128,
                   help="episodes in the population buffer (nested superset of "
                        "both arms' native buffers)")
    p.add_argument("--sizes", default="8,32",
                   help="buffer sizes whose sampling distribution to derive")
    p.add_argument("--draws", type=int, default=4000)
    p.add_argument("--seed", type=int, default=90210)
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


def pack(agent, cfg, n_eps):
    """Population buffer -> flat per-entry arrays plus the episode index."""
    buf, risk, starts, trunc = build(agent, cfg, n_eps)
    T = len(buf)
    adv_raw, _ = agent.compute_gae(buf)
    dec = buf.decision[:T] > 0.5
    act = buf.act[:T]
    cont = buf.cont[:T, 0]
    ep = np.zeros(T, np.int64)
    e = 0
    for t in range(T):
        ep[t] = e
        if cont[t] <= 0.5:
            e += 1
    EP = np.broadcast_to(ep[:, None], act.shape)
    return {"adv": adv_raw.reshape(-1), "dec": dec.reshape(-1),
            "act": act.reshape(-1), "risk": risk.reshape(-1),
            "ep": EP.reshape(-1).copy(), "n_ep": int(e),
            "starts": [int(s) for s in starts], "T": T}


def stats_for(P, eps):
    """
    The update statistics production would compute on a buffer made of exactly
    `eps`. Normalisation pool is recomputed, as `mappo.py:363-366` does.
    """
    m = np.isin(P["ep"], eps)
    dec = m & P["dec"]
    nd = int(dec.sum())
    if nd < 2:
        return None
    pool = P["adv"][dec]
    mu, sd = float(pool.mean()), float(pool.std())
    z = (P["adv"] - mu) / (sd + 1e-8)
    hi = dec & (P["risk"] >= HI)
    lo = dec & (P["risk"] < LO)
    hiE = hi & (P["act"] == I_EDGE)
    hiS = hi & (P["act"] == I_STAY)
    loE = lo & (P["act"] == I_EDGE)
    f = lambda s: float(z[s].mean()) if s.sum() else np.nan
    zhe, zhs = f(hiE), f(hiS)
    return {"n_decision": nd, "n_hi": int(hi.sum()), "n_hi_EDGE": int(hiE.sum()),
            "n_hi_STAY": int(hiS.sum()), "pool_mean_raw": mu, "pool_std_raw": sd,
            "z_hi_EDGE": zhe, "z_hi_STAY": zhs, "z_lo_EDGE": f(loE),
            "contrast": zhe - zhs,
            "channel_drive_hi_EDGE": (hiE.sum() / nd) * zhe,
            "channel_drive_hi_STAY": (hiS.sum() / nd) * zhs,
            "raw_mean_hi": float(P["adv"][hi].mean()) if hi.sum() else np.nan}


KEYS = ["z_hi_EDGE", "z_hi_STAY", "contrast", "channel_drive_hi_EDGE",
        "z_lo_EDGE", "n_hi_EDGE", "pool_std_raw", "raw_mean_hi"]


def summarise(vals):
    a = np.asarray(vals, float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std()),
            "p2_5": float(np.percentile(a, 2.5)),
            "p50": float(np.percentile(a, 50)),
            "p97_5": float(np.percentile(a, 97.5)),
            "frac_positive": float((a > 0).mean()),
            "min": float(a.min()), "max": float(a.max())}


def main(argv=None):
    args = parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(",")]
    print("=" * 78)
    print("SPRINT 7 DIVERGENCE — part 5: SAMPLING DISTRIBUTION OF THE HIGH-RISK")
    print("        LEARNING SIGNAL AT 8 vs 32 EPISODES  (policy held fixed)")
    print("        EXPLORATORY. Offline; no optimiser, no writes.")
    print("=" * 78)
    out = {"probe": "DIV-5 estimator variance of the high-risk channel",
           "exploratory": True, "hi_cut": HI, "lo_cut": LO,
           "args": vars(args), "cells": {}}

    for label, ckpt, native in CELLS:
        agent, extra, cfg = load_agent_and_cfg(str(OUT_DIR / ckpt),
                                               args.device, "train")
        print(f"\n{'='*78}\n  {label}   ({ckpt})   native rollout_episodes="
              f"{extra['config']['train']['rollout_episodes']}\n{'='*78}")
        print(f"  building the {args.population}-episode population buffer ...")
        P = pack(agent, cfg, args.population)
        print(f"    T={P['T']}  episodes={P['n_ep']}  "
              f"decision={int(P['dec'].sum())}")
        assert P["n_ep"] >= max(sizes), "population smaller than a requested size"

        pop = stats_for(P, np.arange(P["n_ep"]))
        nat = stats_for(P, np.arange(native))     # the arm's ACTUAL buffer
        print(f"\n  {'quantity':<26s}{'population':>13s}{'native@'+str(native):>13s}")
        for k in KEYS:
            print(f"  {k:<26s}{pop[k]:>13.4f}{nat[k]:>13.4f}")

        rng = np.random.default_rng(args.seed)
        cell = {"population": pop, "native": nat, "native_size": native,
                "population_episodes": P["n_ep"], "start_ticks": P["starts"],
                "sampling": {}}
        for n in sizes:
            draws = {k: [] for k in KEYS}
            for _ in range(args.draws):
                eps = rng.choice(P["n_ep"], size=n, replace=False)
                s = stats_for(P, eps)
                if s is None:
                    continue
                for k in KEYS:
                    draws[k].append(s[k])
            cell["sampling"][n] = {k: summarise(v) for k, v in draws.items()}

        print(f"\n  sampling distribution over {args.draws} episode subsets "
              f"drawn WITHOUT replacement")
        for k in KEYS:
            print(f"\n  {k}")
            print(f"    {'n_eps':<7s}{'mean':>10s}{'sd':>10s}{'2.5%':>10s}"
                  f"{'50%':>10s}{'97.5%':>10s}{'frac>0':>9s}")
            for n in sizes:
                s = cell["sampling"][n][k]
                print(f"    {n:<7d}{s['mean']:>10.4f}{s['sd']:>10.4f}"
                      f"{s['p2_5']:>10.4f}{s['p50']:>10.4f}"
                      f"{s['p97_5']:>10.4f}{s['frac_positive']:>9.4f}")
            s8, s32 = cell["sampling"][sizes[0]][k], cell["sampling"][sizes[-1]][k]
            if s32["sd"] > 0:
                print(f"    sd ratio {sizes[0]}/{sizes[-1]} = "
                      f"{s8['sd']/s32['sd']:.3f}   "
                      f"(pure 1/sqrt(n) prediction "
                      f"{np.sqrt(sizes[-1]/sizes[0]):.3f})")
            # where does the arm's ACTUAL buffer sit in the n=native law?
            if native in cell["sampling"]:
                sn = cell["sampling"][native][k]
                if sn["sd"] > 0 and not np.isnan(nat[k]):
                    print(f"    native@{native} value {nat[k]:+.4f} sits at "
                          f"{(nat[k]-sn['mean'])/sn['sd']:+.2f} sd of the "
                          f"n={native} law")
        out["cells"][label] = cell

    p = OUT_DIR / f"{TAG}_variance_{args.tag}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
