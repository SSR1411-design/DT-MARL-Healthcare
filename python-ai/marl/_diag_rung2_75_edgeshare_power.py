#!/usr/bin/env python
"""
SPRINT 7 RUNG 2.75 -- item A, closing addendum.

The main ladder (_diag_rung2_75_edgeshare.py) reproduced the Rung 2 headline
EXACTLY (A0 15/546 = 0.0275, R2 31/384 = 0.0807) and attributed the drift to
the start window and to greedy-vs-stochastic. It did NOT reproduce Rung 2.5's
"R2 3.7% / A0 6.6%". This file finds out why, and answers the only question
that actually matters about that pair of numbers: are they distinguishable?

TWO ERRORS IN MY OWN EARLIER STATEMENT OF THE RUNG 2.5 DEFINITION, corrected
here from the source rather than from the write-up:

  1. `_diag_rung2_5_actor_stall.saturation_census` filters decision entries on
     `msk.sum(-1) > 1.5` ONLY. It applies NO has_task filter. I had listed one.
  2. It draws start ticks with `training_start_ticks` =
     `np.random.default_rng(TRAIN_SEED).integers(min_start, max_start+1)` --
     train.py's own draw, i.e. the literal first 8 training episodes -- NOT
     `rollout.episode_starts`, which is evenly spaced. So the START TICKS
     THEMSELVES differ from every cell in the main ladder.

  Neither error changes a conclusion (ladder step D2->D3 showed the has_task
  filter is a bit-exact no-op at high risk), but the definition on record was
  wrong and is corrected rather than quietly restated.

WHAT THIS DOES.

  D5   exact reproduction of the Rung 2.5 census: train.py RNG start ticks,
       torch.manual_seed(TRAIN_SEED + j), stochastic, mask-only, risk > 0.50.
       Must land on A0 14/212 and R2 10/269.
  D5s  the same definition under 8 further torch seeds, to show the spread the
       single-seed number was drawn from.
  POW  two-proportion tests + Wilson intervals on every A0-vs-R2 contrast in
       the ladder, because the Rung 2.5 contrast rests on 14 vs 10 events.

No training. No gradient. No production edit.
Writes SPRINT_7_RUNG2_75_edgeshare_power.json.
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
from marl._diag_rung2_5_targets import (                         # noqa: E402
    training_start_ticks, TRAIN_SEED,
)
from marl.env import DTMarlEnv                                   # noqa: E402

TAG = "SPRINT_7_RUNG2_75"
HI_RISK = 0.50               # _diag_rung2_5_actor_stall.HI_RISK
MODELS = {"A0": "mappo_A0_cpu_repro.pth", "R2": "mappo_R2_mc_target.pth"}
# the figures on record in SPRINT_7_RUNG2_5_actor_stall.json
RUNG2_5 = {"A0": dict(n=212, EDGE=14, share=0.0660377358490566),
           "R2": dict(n=269, EDGE=10, share=0.03717472118959108)}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--extra-seeds", type=int, default=8)
    p.add_argument("--ladder", default=str(
        OUT_DIR / f"{TAG}_edgeshare_main.json"))
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


def wilson(k, n, z=1.959964):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, ctr - hw), min(1.0, ctr + hw)]


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_prop(k1, n1, k2, n2):
    """Pooled two-proportion z test, two-sided. No scipy dependency."""
    if n1 == 0 or n2 == 0:
        return None
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se < 1e-15:
        return dict(p1=p1, p2=p2, diff=p2 - p1, z=None, p_value=None)
    z = (p2 - p1) / se
    return dict(p1=p1, p2=p2, diff=p2 - p1, z=z,
                p_value=2 * (1 - _phi(abs(z))),
                significant_at_05=bool(2 * (1 - _phi(abs(z))) < 0.05),
                n1=n1, n2=n2, k1=k1, k2=k2,
                wilson1=wilson(k1, n1), wilson2=wilson(k2, n2))


def min_detectable(n1, n2, p1, power=0.80, alpha=0.05):
    """
    Smallest p2 > p1 this sample size can detect. Answers "was the Rung 2.5
    contrast even capable of resolving the difference it reported?"
    """
    za, zb = 1.959964, 0.8416212
    lo, hi = p1, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        p = (p1 * n1 + mid * n2) / (n1 + n2)
        se0 = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
        se1 = math.sqrt(p1 * (1 - p1) / n1 + mid * (1 - mid) / n2)
        if se1 < 1e-15:
            lo = mid; continue
        if (mid - p1 - za * se0) / se1 >= zb:
            hi = mid
        else:
            lo = mid
    return hi


def census_cell(agent, cfg, starts, torch_base, thr=HI_RISK):
    """
    saturation_census's own accounting, verbatim: decision entries are
    mask.sum > 1.5, high risk is env.risk_at(i) > thr, no has_task filter.
    """
    env = DTMarlEnv(cfg.env, cfg.reward)
    risks, acts, lens, trunc = [], [], [], 0
    for j, s in enumerate(starts):
        torch.manual_seed(torch_base + j)
        r = _replay(env, agent, int(s), j, record=True, sample=True)
        rec = r["rec"]
        lens.append(int(r["n_steps"]))
        trunc += int(r["n_steps"] >= env.cfg.episode_steps)
        for i in range(env.n_agents):
            legal = rec["mask"][:, i, :].sum(-1) > 1.5
            risks.append(np.asarray(rec["risk"])[legal, i])
            acts.append(np.asarray(rec["act"])[legal, i])
    risks = np.concatenate(risks); acts = np.concatenate(acts)
    hi = risks > thr
    n = int(hi.sum())
    k = int((acts[hi] == ACTION_MIGRATE_EDGE).sum())
    return dict(n_decision_entries=int(risks.size), n_highrisk=n, EDGE=k,
                share=float(k / n) if n else None, wilson95=wilson(k, n),
                STAY_share=float((acts[hi] == ACTION_STAY).mean())
                if n else None,
                episode_lengths=lens, truncated=trunc)


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 RUNG 2.75 -- item A addendum: reproduce Rung 2.5's own")
    print("        census, then test whether its contrast is resolvable")
    print("=" * 78)

    out = {}
    for tag, fn in MODELS.items():
        agent, extra, cfg = load_agent_and_cfg(
            str(OUT_DIR / fn), args.device, "train")
        env = DTMarlEnv(cfg.env, cfg.reward)
        starts = training_start_ticks(env, args.episodes)
        print(f"\n-- {tag}: train.py RNG start ticks {starts}")
        d5 = census_cell(agent, cfg, starts, TRAIN_SEED)
        ref = RUNG2_5[tag]
        ok = (d5["n_highrisk"] == ref["n"] and d5["EDGE"] == ref["EDGE"])
        print(f"   D5  n_hi={d5['n_highrisk']:4d} EDGE={d5['EDGE']:3d} "
              f"share={d5['share']:.4f}   "
              f"on record n={ref['n']} EDGE={ref['EDGE']} "
              f"share={ref['share']:.4f}   "
              f"{'MATCH' if ok else '*** MISMATCH ***'}")
        reps = []
        for k in range(args.extra_seeds):
            c = census_cell(agent, cfg, starts, 7_000_000 + 1000 * k)
            reps.append(c)
            print(f"   D5s seed {k}: n_hi={c['n_highrisk']:4d} "
                  f"EDGE={c['EDGE']:3d} share="
                  f"{c['share']:.4f}" if c["share"] is not None else "n/a")
        sh = [c["share"] for c in reps if c["share"] is not None]
        agg_n = sum(c["n_highrisk"] for c in reps)
        agg_k = sum(c["EDGE"] for c in reps)
        out[tag] = dict(
            D5_exact=d5, D5_matches_record=bool(ok), rung2_5_on_record=ref,
            D5s=reps,
            D5s_share_mean=float(np.mean(sh)) if sh else None,
            D5s_share_sd=float(np.std(sh, ddof=1)) if len(sh) > 1 else None,
            D5s_share_min=float(np.min(sh)) if sh else None,
            D5s_share_max=float(np.max(sh)) if sh else None,
            D5s_pooled=dict(n=agg_n, EDGE=agg_k,
                            share=float(agg_k / agg_n) if agg_n else None,
                            wilson95=wilson(agg_k, agg_n)),
        )
        print(f"   D5s over {args.extra_seeds} seeds: share "
              f"{out[tag]['D5s_share_mean']:.4f} "
              f"(sd {out[tag]['D5s_share_sd']:.4f}, "
              f"range {out[tag]['D5s_share_min']:.4f}"
              f"-{out[tag]['D5s_share_max']:.4f})   pooled "
              f"{agg_k}/{agg_n} = {out[tag]['D5s_pooled']['share']:.4f}")

    # ---------------- power ----------------
    print("\n" + "=" * 78)
    print("IS THE RUNG 2.5 CONTRAST RESOLVABLE?  (A0 vs R2, two-proportion)")
    print("=" * 78)
    tests = {}
    tests["rung2_5_on_record"] = two_prop(
        RUNG2_5["A0"]["EDGE"], RUNG2_5["A0"]["n"],
        RUNG2_5["R2"]["EDGE"], RUNG2_5["R2"]["n"])
    tests["D5_reproduced"] = two_prop(
        out["A0"]["D5_exact"]["EDGE"], out["A0"]["D5_exact"]["n_highrisk"],
        out["R2"]["D5_exact"]["EDGE"], out["R2"]["D5_exact"]["n_highrisk"])
    tests["D5s_pooled_8_seeds"] = two_prop(
        out["A0"]["D5s_pooled"]["EDGE"], out["A0"]["D5s_pooled"]["n"],
        out["R2"]["D5s_pooled"]["EDGE"], out["R2"]["D5s_pooled"]["n"])

    lad = json.loads(Path(args.ladder).read_text())["ladder"]
    for key in lad["A0"]:
        a, r = lad["A0"][key], lad["R2"][key]
        if "per_seed" in a:
            ka = sum(x["EDGE"] for x in a["per_seed"])
            na = sum(x["n"] for x in a["per_seed"])
            kr = sum(x["EDGE"] for x in r["per_seed"])
            nr = sum(x["n"] for x in r["per_seed"])
        else:
            ka, na, kr, nr = a["EDGE"], a["n"], r["EDGE"], r["n"]
        tests[f"ladder_{key}"] = two_prop(ka, na, kr, nr)

    print(f"  {'contrast':<46s} {'A0':>12s} {'R2':>12s} {'z':>7s} "
          f"{'p':>8s} {'sig':>5s}")
    for k, t in tests.items():
        if t is None:
            continue
        print(f"  {k:<46s} {t['k1']:>4d}/{t['n1']:<7d} "
              f"{t['k2']:>4d}/{t['n2']:<7d} "
              f"{t['z']:>+7.2f} {t['p_value']:>8.4f} "
              f"{'YES' if t['significant_at_05'] else 'no':>5s}")

    md = min_detectable(RUNG2_5["A0"]["n"], RUNG2_5["R2"]["n"],
                        RUNG2_5["A0"]["share"])
    print(f"\n  at Rung 2.5's own sample size (n={RUNG2_5['A0']['n']} vs "
          f"{RUNG2_5['R2']['n']}) and A0 = {RUNG2_5['A0']['share']:.4f},")
    print(f"  the smallest R2 share detectable at 80% power / alpha=0.05 is "
          f"{md:.4f}")
    print(f"  -- i.e. that census could only have resolved an INCREASE past "
          f"{md:.1%}, and")
    print(f"     it reported a DECREASE to "
          f"{RUNG2_5['R2']['share']:.4f} with 10 events.")

    blob = dict(
        probe=f"{TAG}_edgeshare_power",
        what="exact reproduction of the Rung 2.5 high-risk EDGE census, plus "
             "two-proportion tests on every A0-vs-R2 contrast in the ladder",
        corrections_to_my_earlier_statement_of_the_rung2_5_definition=[
            "saturation_census applies NO has_task filter -- decision entries "
            "are mask.sum(-1) > 1.5 only. I had listed a has_task filter. "
            "Ladder step D2->D3 shows it is a bit-exact no-op at high risk "
            "(n=581/373 unchanged), so no conclusion moves.",
            "saturation_census uses training_start_ticks (train.py's own "
            "np.random.default_rng(TRAIN_SEED) draw = the literal first 8 "
            "training episodes), NOT rollout.episode_starts. The start ticks "
            "themselves therefore differ from every main-ladder cell. This is "
            "a SIXTH difference, not one of the five I listed.",
        ],
        rung2_5_definition_verified_from_source=dict(
            file="marl/_diag_rung2_5_actor_stall.py:288-347",
            decision_rule="msk[:, i, :].sum(-1) > 1.5",
            threshold="rk[legal, i] > HI_RISK, HI_RISK = 0.50",
            policy="_replay(..., sample=True) with torch.manual_seed"
                   "(TRAIN_SEED + j)",
            starts="training_start_ticks(env, 8)",
        ),
        per_arm=out, tests=tests,
        min_detectable_R2_share_at_rung2_5_n=md,
        ladder_source=str(args.ladder),
    )
    p = OUT_DIR / f"{TAG}_edgeshare_power_{args.tag}.json"
    p.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n  wrote {p}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
