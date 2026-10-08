#!/usr/bin/env python
"""
SPRINT 7 DIVERGENCE DIAGNOSTIC — part 3 of 3: IS THE HIGH-RISK DISPLACEMENT
DEFICIT REAL, AND WHAT SHAPE DOES IT HAVE?

OFFLINE. Reads checkpoints, rebuilds the FIXED sets with the same imported
builders as part 2, writes no parameters, touches no production module.

WHAT PART 2 FOUND. Functional displacement from the shared theta_0, projected
onto R2's displacement direction, on UNION:

                     low-risk (n=70621)      high-risk (n=3592)
    R3   proj_frac         0.7124                  0.2147
    A0   proj_frac         0.5965                  0.3458

The ORDERING INVERTS between regimes: R3 is CLOSER to R2 than A0 is in the
low-risk bulk, and FARTHER from R2 than A0 is at high risk. Same sign and same
inversion on RANDOM. Two things must be settled before that is believed:

  E1 IS IT REAL? proj_frac and cosine are ratios, not means, so the existing
     cluster_ci helper does not apply. But both numerator and denominator are
     SUMS over entries, so per-episode partial sums are sufficient statistics:

        proj = (sum_e a_e) / (sum_e b_e),   a_e = sum_{t in e} <v_t, r_t>
        cos  = (sum_e a_e) / sqrt(sum_e c_e * sum_e b_e)

     with b_e = sum ||r_t||^2 and c_e = sum ||v_t||^2. A bootstrap replicate is
     then just a resample of the per-episode triples -- exact, and no replicate
     vector is ever materialised. (Same sufficiency trick as
     _diag_rung3_bootstrap.) Clusters are EPISODES, which is the unit
     _diag_rung2_75_matched_states already treats as independent; UNION keeps
     A0's and R2's episode ids disjoint so pooling does not merge clusters.

  E2 WHAT SHAPE? A low proj_frac at high risk has two very different causes.
     Decompose each arm's high-risk displacement into

        Dp_t = u + (Dp_t - u),      u = mean over high-risk entries of Dp_t

     a UNIFORM shift u common to every high-risk state, plus STATE-VARYING
     residual. `uniform_share` = ||u||*sqrt(n) / ||Dp|| separates:
       - a policy that raises/lowers the same actions at EVERY high-risk state
         (uniform_share near 1) -- it changes aggregate probabilities without
         re-ranking anything, so it can move a mean-difference metric like P1/P2
         while never changing which action is arg-max;
       - a policy that RE-RANKS actions differently at different high-risk
         states (uniform_share well below 1).
     This is a direct, quantitative test of the already-recorded asymmetry that
     R2 flips the greedy arg-max to MIGRATE_EDGE at 46.3% of high-risk states
     while R3 flips it at 0. arg-max flip counts are recomputed here alongside,
     so the two measurements can be read against each other.

E2 IS EXPLORATORY. It was chosen after seeing E1's numbers, is not
pre-registered, and cannot change any locked R3 verdict. E1's CIs are reported
so the part-2 finding can be judged, not to license a new threshold.

Writes SPRINT_7_DIV_shape_<tag>.json.
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

from marl.mappo import MAPPO, MappoConfig                          # noqa: E402
from marl._diag_rung0 import load_agent_and_cfg, OUT_DIR           # noqa: E402
from marl._diag_rung2_75_matched_states import (                   # noqa: E402
    ACTION_NAMES, eval_starts, trajectory, random_trajectory,
    union_source, probs_at,
)
from marl._diag_div_geometry import (                              # noqa: E402
    ARM_FILES, REF, HI, LO, WINDOW, EXPECT,
)

TAG = "SPRINT_7_DIV"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--clusters", type=int, default=32)
    p.add_argument("--start-seed", type=int, default=20260825)
    p.add_argument("--random-seed", type=int, default=31337)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--boot-seed", type=int, default=7)
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


# ------------------------------------------------------------------ bootstrap

def suff(v, r, ep):
    """Per-episode sufficient statistics for <v,r>, ||r||^2, ||v||^2."""
    uniq = np.unique(ep)
    a = np.array([float((v[ep == e] * r[ep == e]).sum()) for e in uniq])
    b = np.array([float((r[ep == e] ** 2).sum()) for e in uniq])
    c = np.array([float((v[ep == e] ** 2).sum()) for e in uniq])
    return a, b, c, uniq


def boot_ci(a, b, c, n_boot, rng):
    """Cluster bootstrap CIs for proj_frac and cosine from the suff. stats."""
    n = a.size
    idx = rng.integers(0, n, size=(n_boot, n))
    A, B, C = a[idx].sum(1), b[idx].sum(1), c[idx].sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        pf = A / B
        cs = A / np.sqrt(C * B)
    q = lambda x: (float(np.nanpercentile(x, 2.5)),
                   float(np.nanpercentile(x, 97.5)))
    return {"proj_frac_point": float(a.sum() / b.sum()),
            "proj_frac_ci95": q(pf),
            "cosine_point": float(a.sum() / np.sqrt(c.sum() * b.sum())),
            "cosine_ci95": q(cs), "n_clusters": int(n)}


def boot_ci_diff(aH, bH, cH, aL, bL, cL, n_boot, rng):
    """
    CI on proj_frac(hi) - proj_frac(lo) with the SAME resampled episodes on both
    sides, so the two regimes stay paired within a replicate. Episodes are the
    shared cluster index; an episode contributes its hi entries and its lo
    entries together, which is what makes the difference honest.
    """
    n = aH.size
    idx = rng.integers(0, n, size=(n_boot, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = aH[idx].sum(1) / bH[idx].sum(1) - aL[idx].sum(1) / bL[idx].sum(1)
    return {"point": float(aH.sum() / bH.sum() - aL.sum() / bL.sum()),
            "ci95": (float(np.nanpercentile(d, 2.5)),
                     float(np.nanpercentile(d, 97.5))),
            "frac_replicates_negative": float(np.mean(d < 0))}


# ------------------------------------------------------------------- main

def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.boot_seed)
    res = {"probe": "DIV-3 reality and shape of the high-risk deficit",
           "ref_arm": REF, "args": vars(args)}

    arms, cfgs, extras = {}, {}, {}
    for k, f in ARM_FILES.items():
        a, e, c = load_agent_and_cfg(str(OUT_DIR / f), args.device, WINDOW)
        arms[k], extras[k], cfgs[k] = a, e, c
    ck = torch.load(str(OUT_DIR / ARM_FILES[REF]), map_location="cpu",
                    weights_only=False)
    seed = ck["extra"]["config"]["train"]["seed"]
    z0 = MAPPO(arms[REF].n_agents, arms[REF].obs_dim, arms[REF].state_dim,
               MappoConfig(**ck["extra"]["config"]["mappo"]),
               device=args.device, seed=seed)

    starts, _ = eval_starts(cfgs["A0"], args.clusters, args.start_seed)
    src = {"RANDOM": random_trajectory(cfgs["A0"], starts, args.random_seed)}
    gen = {k: trajectory(arms[k], cfgs[k], starts) for k in ("A0", "R2")}
    src["UNION"] = union_source(gen["A0"], gen["R2"])
    for k, S in list(gen.items()) + [("UNION", src["UNION"])]:
        d = S["mask"].sum(-1) > 1.5
        got = {"dec": int(d.sum()), "hi": int((d & (S["risk"] >= HI)).sum())}
        assert got == EXPECT[k], f"{k}: {got} != {EXPECT[k]}"
    print("  state sets verified against the pre-registered cardinalities: OK")

    ie = ACTION_NAMES.index("MIGRATE_EDGE")
    out = {}
    for name, S in src.items():
        dec = S["mask"].sum(-1) > 1.5
        hi = dec & (S["risk"] >= HI)
        lo = dec & (S["risk"] < LO)
        # episode id broadcast to (t, i) so every entry carries its cluster
        EPI = np.broadcast_to(S["ep"][:, None], S["risk"].shape)
        P0 = probs_at(z0, S["obs"], S["mask"])
        P = {k: probs_at(arms[k], S["obs"], S["mask"]) for k in ARM_FILES}
        DP = {k: P[k] - P0 for k in ARM_FILES}
        R = DP[REF]

        print("\n" + "=" * 78)
        print(f"  {name}:  hi n={int(hi.sum())}  lo n={int(lo.sum())}  "
              f"clusters={len(np.unique(S['ep']))}")
        print("=" * 78)

        # ---------------- E1: is the deficit real? ----------------
        print(f"\n  E1  cluster-bootstrap CIs ({args.boot} replicates, "
              f"clusters = episodes)")
        print(f"  {'arm':<8s} {'regime':<7s} {'proj_frac':>10s} "
              f"{'95% CI':>20s} {'cosine':>8s} {'95% CI':>20s}")
        e1 = {}
        for k in ARM_FILES:
            if k == REF or k == "R2_best":
                continue
            e1[k] = {}
            st = {}
            for reg, sel in (("hi", hi), ("lo", lo)):
                v, r, ep = DP[k][sel], R[sel], EPI[sel]
                st[reg] = suff(v, r, ep)
                ci = boot_ci(*st[reg][:3], args.boot,
                             np.random.default_rng(args.boot_seed))
                e1[k][reg] = ci
                print(f"  {k:<8s} {reg:<7s} {ci['proj_frac_point']:>10.4f} "
                      f"[{ci['proj_frac_ci95'][0]:+.4f},"
                      f"{ci['proj_frac_ci95'][1]:+.4f}] "
                      f"{ci['cosine_point']:>8.4f} "
                      f"[{ci['cosine_ci95'][0]:+.4f},"
                      f"{ci['cosine_ci95'][1]:+.4f}]")
            # paired hi-lo difference on the same resampled episodes
            dh = boot_ci_diff(*st["hi"][:3], *st["lo"][:3], args.boot,
                              np.random.default_rng(args.boot_seed + 1))
            e1[k]["hi_minus_lo"] = dh
            print(f"  {k:<8s} {'hi-lo':<7s} {dh['point']:>10.4f} "
                  f"[{dh['ci95'][0]:+.4f},{dh['ci95'][1]:+.4f}]   "
                  f"frac replicates < 0 = {dh['frac_replicates_negative']:.4f}")

        # ---------------- E2: what shape? ----------------
        print(f"\n  E2  EXPLORATORY  uniform shift vs state-varying re-ranking")
        print(f"  {'arm':<8s} {'regime':<7s} {'||Dp||':>9s} {'uniform':>9s} "
              f"{'share':>7s} {'argmax->EDGE':>13s} {'argmax flips':>13s}")
        e2 = {}
        for k in list(ARM_FILES) + ["theta0"]:
            e2[k] = {}
            D = np.zeros_like(R) if k == "theta0" else DP[k]
            Pk = P0 if k == "theta0" else P[k]
            for reg, sel in (("hi", hi), ("lo", lo)):
                d = D[sel]
                n = d.shape[0]
                u = d.mean(0)
                nrm = float(np.linalg.norm(d))
                un = float(np.linalg.norm(u) * np.sqrt(n))
                # arg-max under the mask (MASK_FILL already applied in probs_at,
                # so masked actions carry ~0 probability and cannot win)
                am = Pk[sel].argmax(-1)
                am0 = P0[sel].argmax(-1)
                e2[k][reg] = {"norm": nrm, "uniform_norm": un,
                              "uniform_share": un / nrm if nrm else float("nan"),
                              "argmax_EDGE": int((am == ie).sum()),
                              "argmax_flips_vs_theta0": int((am != am0).sum()),
                              "n": int(n)}
                r_ = e2[k][reg]
                print(f"  {k:<8s} {reg:<7s} {nrm:>9.4f} {un:>9.4f} "
                      f"{r_['uniform_share']:>7.4f} "
                      f"{r_['argmax_EDGE']:>7d}/{n:<5d} "
                      f"{r_['argmax_flips_vs_theta0']:>13d}")

        out[name] = {"E1_bootstrap": e1, "E2_shape": e2,
                     "n_hi": int(hi.sum()), "n_lo": int(lo.sum())}

    res["results"] = out
    p = OUT_DIR / f"{TAG}_shape_{args.tag}.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
