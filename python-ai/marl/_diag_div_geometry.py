#!/usr/bin/env python
"""
SPRINT 7 DIVERGENCE DIAGNOSTIC — part 2 of 3: DISPLACEMENT GEOMETRY.

OFFLINE. Loads checkpoints, reconstructs the shared initial policy, and replays
episodes to rebuild the FIXED state sets. Trains nothing, writes no parameters,
constructs no optimiser for training, and touches no production module.

THE QUESTION THIS ANSWERS. `SPRINT_7_DIV_logs_*` found that three quantities of
independent provenance — actor entropy, `decision_frac`, and the multi-agent
contention counter `infeasible` — all place R3's ENDPOINT at roughly R2's
UPDATE 25-30 of 75, while R2 and R3 have identical LR schedules, identical
update counts (75), and R3 took MORE Adam steps (1440 vs 1392). Yet the
cumulative within-update movement proxy is nearly identical (sum|k1| 0.1533 vs
0.1528). Equal per-update movement with a nearer endpoint can only mean the
steps CANCEL more in R3. This probe measures net displacement directly, in the
two spaces where it is well defined:

  PART A  PARAMETER SPACE. theta_0 is exactly reconstructable: MAPPO.__init__
          calls torch.manual_seed(seed) at mappo.py:218 IMMEDIATELY before
          constructing the actor and critic, and every architecture field in
          `mappo_cfg` is identical across arms (asserted here). So all arms
          share one theta_0 and displacements are directly comparable.

  PART B  FUNCTION SPACE. Parameter distance is not behaviour: an actor can move
          far in a flat direction. The behaviourally meaningful displacement is
          how much pi(.|s) moved, measured on states chosen independently of the
          arms. Uses the UNCHANGED RANDOM (policy-independent, uniform-legal)
          and UNION (A0's states POOLED with R2's) sets, built by importing
          _diag_rung2_75_matched_states' OWN builders rather than
          reimplementing them. R3 IS NOT PROMOTED INTO `MODELS` -- it is scored
          on the unchanged sets exactly as the pre-registration requires, so
          UNION remains A0-pooled-with-R2 and no threshold is re-based.

THE DISCRIMINATING MEASUREMENT. For each arm, form the functional displacement
vector Dp = pi_arm(.|s) - pi_0(.|s) over decision entries, then decompose R3's
against R2's:

    proj_fraction = <Dp_R3, Dp_R2> / ||Dp_R2||^2

If R3 is simply an EARLIER R2 -- the same path, walked less far -- then
cos(Dp_R3, Dp_R2) is high and proj_fraction is well below 1, AND both hold
EQUALLY in the high-risk and low-risk regimes. If instead R3 tracked the
majority direction faithfully and under-moved specifically where the minority
signal lives, then proj_fraction on LOW-risk entries is near 1 while on
HIGH-risk entries it is much smaller. Those two outcomes are different
mechanisms and this is the measurement that separates them. Both are reported
per risk regime for exactly that reason.

READ PART A's cosines WITH CARE: in ~233k actor dimensions two independent
random directions have cosine ~1/sqrt(d) ~ 0.002, so a "small" cosine of 0.05
is still ~24x chance. A same-dimension random-direction reference is printed
alongside every cosine so the scale is explicit.

Writes SPRINT_7_DIV_geometry_<tag>.json.
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

TAG = "SPRINT_7_DIV"

# `final` and `_best` are the ONLY policy snapshots that exist for these runs;
# no intermediate per-update checkpoint was ever written. Scoring both gives two
# points per arm on the displacement/response curve instead of one.
ARM_FILES = {
    "A0":      "mappo_A0_cpu_repro.pth",
    "R2":      "mappo_R2_mc_target.pth",
    "R2_best": "mappo_R2_mc_target_best.pth",
    "R3":      "R3_batch32.pth",
    "R3_best": "R3_batch32_best.pth",
}
REF = "R2"          # the arm whose displacement direction defines the axis
HI, LO = 0.6, 0.2   # the pre-registered risk cut points, unchanged
# The pre-registered probe evaluates on the HELD-OUT window
# (_diag_rung2_75_matched_states.main line 331 passes window="eval"), so the
# fixed sets must be rebuilt there and nowhere else. Building them in the train
# window instead yields UNION=70791/hi=3501 rather than 74237/3592 -- close
# enough to look right, which is exactly why these counts are asserted.
WINDOW = "eval"
# Documented cardinalities of the sets the pre-registered P1/P2/P3 were measured
# on: SPRINT_7_RUNG2_75_matched_states_main.json (n_decision per source) and
# HEADLINE/action-channel counts for the high-risk subsets.
EXPECT = {"A0": {"dec": 43519, "hi": 2174},
          "R2": {"dec": 30718, "hi": 1418},
          "UNION": {"dec": 74237, "hi": 3592}}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    # defaults copied verbatim from _diag_rung2_75_matched_states.parse_args so
    # the state sets are bit-identical to the ones the pre-registered P1/P2 used
    p.add_argument("--clusters", type=int, default=32)
    p.add_argument("--start-seed", type=int, default=20260825)
    p.add_argument("--random-seed", type=int, default=31337)
    p.add_argument("--skip-function", action="store_true",
                   help="parameter-space geometry only (no episode replay)")
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


# ====================================================================== utils

def flat_actor(agent):
    return torch.cat([p.detach().reshape(-1)
                      for p in agent.actor.parameters()]).double()


def flat_critic(agent):
    return torch.cat([p.detach().reshape(-1)
                      for p in agent.critic.parameters()]).double()


def per_agent_flat(agent):
    """One flat vector per agent. Requires separate (unshared) actors."""
    out = []
    for i in range(agent.n_agents):
        mod = agent.actor.nets[i] if hasattr(agent.actor, "nets") else None
        if mod is None:
            return None
        out.append(torch.cat([p.detach().reshape(-1)
                              for p in mod.parameters()]).double())
    return out


def cos(a, b):
    na, nb = float(a.norm()), float(b.norm())
    if na == 0 or nb == 0:
        return float("nan")
    return float((a @ b) / (na * nb))


def decomp(v, ref):
    """Project v on ref: returns (proj_fraction, cosine, residual_fraction)."""
    nr = float(ref.norm())
    if nr == 0:
        return float("nan"), float("nan"), float("nan")
    pf = float((v @ ref) / (nr ** 2))
    resid = v - pf * ref
    return pf, cos(v, ref), float(resid.norm()) / nr


# ============================================================= PART A  params

def part_a(args, res):
    arms, cfgs, extras = {}, {}, {}
    for k, f in ARM_FILES.items():
        a, e, c = load_agent_and_cfg(str(OUT_DIR / f), args.device, WINDOW)
        arms[k], extras[k], cfgs[k] = a, e, c

    # --- theta_0 reconstruction, and the assertions that make it legitimate ---
    arch = ["actor_hidden", "critic_hidden", "separate_actors"]
    base = extras[REF]["config"]["mappo"]
    for k, e in extras.items():
        for f in arch:
            assert e["config"]["mappo"][f] == base[f], \
                f"architecture field {f} differs in {k}: cannot share theta_0"
    seeds = {k: e["config"]["train"]["seed"] for k, e in extras.items()}
    assert len(set(seeds.values())) == 1, f"seeds differ across arms: {seeds}"
    seed = list(seeds.values())[0]

    dims = (arms[REF].n_agents, arms[REF].obs_dim, arms[REF].state_dim)
    mk = lambda: MAPPO(*dims, MappoConfig(**base), device=args.device, seed=seed)
    z0, z0b = mk(), mk()
    det_a = float((flat_actor(z0) - flat_actor(z0b)).abs().max())
    det_c = float((flat_critic(z0) - flat_critic(z0b)).abs().max())
    assert det_a == 0.0 and det_c == 0.0, "theta_0 is not seed-deterministic"
    t0a, t0c = flat_actor(z0), flat_critic(z0)
    d_actor, d_critic = t0a.numel(), t0c.numel()

    print(f"  seed (all arms)          : {seed}")
    print(f"  theta_0 reproducibility  : actor max|diff| {det_a:.1e}, "
          f"critic {det_c:.1e}  (two independent constructions)")
    print(f"  actor params             : {d_actor}   critic params: {d_critic}")
    print(f"  chance cosine 1/sqrt(d)  : actor {1/np.sqrt(d_actor):.5f}   "
          f"critic {1/np.sqrt(d_critic):.5f}")

    # --- displacements ---
    D = {k: {"actor": flat_actor(a) - t0a, "critic": flat_critic(a) - t0c}
         for k, a in arms.items()}
    print(f"\n  {'arm':<8s} {'||dTheta_actor||':>17s} {'||dTheta_critic||':>18s}"
          f" {'actor/R2':>9s}")
    ref_n = float(D[REF]["actor"].norm())
    out = {"seed": seed, "d_actor": d_actor, "d_critic": d_critic,
           "theta0_determinism_max_abs_diff": [det_a, det_c],
           "chance_cosine_actor": 1 / float(np.sqrt(d_actor)),
           "displacement": {}}
    for k in ARM_FILES:
        na, nc = float(D[k]["actor"].norm()), float(D[k]["critic"].norm())
        print(f"  {k:<8s} {na:>17.4f} {nc:>18.4f} {na/ref_n:>9.4f}")
        out["displacement"][k] = {"actor_norm": na, "critic_norm": nc,
                                  "actor_norm_over_ref": na / ref_n}

    # --- projection of each arm onto R2's actor direction ---
    print(f"\n  decomposition against {REF}'s ACTOR displacement direction")
    print(f"  {'arm':<8s} {'proj_frac':>10s} {'cosine':>9s} {'resid_frac':>11s}")
    out["actor_decomposition_vs_ref"] = {}
    for k in ARM_FILES:
        pf, c, rf = decomp(D[k]["actor"], D[REF]["actor"])
        print(f"  {k:<8s} {pf:>10.4f} {c:>9.4f} {rf:>11.4f}")
        out["actor_decomposition_vs_ref"][k] = {
            "proj_fraction": pf, "cosine": c, "residual_fraction": rf}

    # --- pairwise cosines ---
    print(f"\n  pairwise cosine of ACTOR displacement")
    ks = list(ARM_FILES)
    print("           " + "".join(f"{k:>9s}" for k in ks))
    out["actor_cosine_matrix"] = {}
    for k in ks:
        row = [cos(D[k]["actor"], D[j]["actor"]) for j in ks]
        out["actor_cosine_matrix"][k] = dict(zip(ks, row))
        print(f"  {k:<8s} " + "".join(f"{x:>9.4f}" for x in row))

    # --- per-agent displacement: is the deficit uniform across the 10 actors? ---
    pa0 = per_agent_flat(z0)
    if pa0 is not None:
        print(f"\n  per-agent ||dTheta_actor||  (10 independent actor MLPs)")
        print(f"  {'arm':<8s}" + "".join(f"{i:>8d}" for i in range(len(pa0))))
        out["per_agent_actor_norm"] = {}
        for k in ARM_FILES:
            pk = per_agent_flat(arms[k])
            v = [float((pk[i] - pa0[i]).norm()) for i in range(len(pa0))]
            out["per_agent_actor_norm"][k] = v
            print(f"  {k:<8s}" + "".join(f"{x:>8.3f}" for x in v))
        r2v = out["per_agent_actor_norm"][REF]
        r3v = out["per_agent_actor_norm"]["R3"]
        rat = [b / a if a else float("nan") for a, b in zip(r2v, r3v)]
        print(f"  {'R3/R2':<8s}" + "".join(f"{x:>8.3f}" for x in rat))
        out["per_agent_R3_over_R2"] = rat

    # --- Adam state: how big is a step at the END of training? ---
    print(f"\n  Adam actor state at the final checkpoint")
    print(f"  {'arm':<8s} {'steps':>7s} {'lr_saved':>10s} {'||m||':>10s} "
          f"{'||update||':>11s} {'||upd||/||dTheta||':>19s}")
    out["adam_actor"] = {}
    for k, f in ARM_FILES.items():
        ck = torch.load(str(OUT_DIR / f), map_location="cpu", weights_only=False)
        st = ck["opt_actor"]
        g = st["param_groups"][0]
        lr, b1, b2, eps = g["lr"], g["betas"][0], g["betas"][1], g["eps"]
        steps, ms, us = None, [], []
        for _, s in sorted(st["state"].items()):
            t = float(s["step"]) if not torch.is_tensor(s["step"]) \
                else float(s["step"].item())
            steps = t
            m, v = s["exp_avg"].double(), s["exp_avg_sq"].double()
            mh = m / (1 - b1 ** t)
            vh = v / (1 - b2 ** t)
            ms.append(m.reshape(-1))
            us.append((lr * mh / (vh.sqrt() + eps)).reshape(-1))
        mn = float(torch.cat(ms).norm())
        un = float(torch.cat(us).norm())
        dn = float(D[k]["actor"].norm())
        print(f"  {k:<8s} {steps:>7.0f} {lr:>10.3e} {mn:>10.3e} {un:>11.4f} "
              f"{un/dn:>19.5f}")
        out["adam_actor"][k] = {"steps": steps, "lr_saved": lr,
                                "grad_moment_norm": mn, "update_norm": un,
                                "update_over_displacement": un / dn}

    res["part_a_parameter_space"] = out
    return arms, cfgs, out


# =========================================================== PART B  function

def part_b(args, arms, cfgs, res):
    """
    Functional displacement on the UNCHANGED fixed state sets. Source sets are
    generated from A0 and R2 only (MODELS), exactly as the pre-registration
    requires -- R3 is scored, never a generator.
    """
    starts, _ = eval_starts(cfgs["A0"], args.clusters, args.start_seed)
    print(f"  episodes per source      : {len(starts)}  "
          f"(start-seed {args.start_seed})")

    src = {}
    src["RANDOM"] = random_trajectory(cfgs["A0"], starts, args.random_seed)
    gen = {k: trajectory(arms[k], cfgs[k], starts) for k in ("A0", "R2")}
    src["UNION"] = union_source(gen["A0"], gen["R2"])

    # Self-verification: the rebuilt sets must have the SAME cardinality as the
    # ones the pre-registered numbers were measured on, or the comparison is
    # being made against a different population than the thresholds were set on.
    chk = {}
    for k, S in list(gen.items()) + [("UNION", src["UNION"])]:
        d = S["mask"].sum(-1) > 1.5
        got = {"dec": int(d.sum()), "hi": int((d & (S["risk"] >= HI)).sum())}
        chk[k] = {"got": got, "expected": EXPECT[k]}
        ok = got == EXPECT[k]
        print(f"  {k:<6s} decision {got['dec']:>6d} (expect "
              f"{EXPECT[k]['dec']:>6d})   hi {got['hi']:>5d} (expect "
              f"{EXPECT[k]['hi']:>5d})   {'OK' if ok else 'MISMATCH'}")
        assert ok, (f"{k} state set does not match the pre-registered "
                    f"cardinality: got {got}, expected {EXPECT[k]}")
    res["state_set_verification"] = chk

    # the shared initial policy, as a behaving agent
    base = res["part_a_parameter_space"]
    ck = torch.load(str(OUT_DIR / ARM_FILES[REF]), map_location="cpu",
                    weights_only=False)
    z0 = MAPPO(arms[REF].n_agents, arms[REF].obs_dim, arms[REF].state_dim,
               MappoConfig(**ck["extra"]["config"]["mappo"]),
               device=args.device, seed=base["seed"])

    out = {}
    for name, S in src.items():
        dec = (S["mask"].sum(-1) > 1.5)
        risk = S["risk"]
        hi = dec & (risk >= HI)
        lo = dec & (risk < LO)
        P0 = probs_at(z0, S["obs"], S["mask"])
        print(f"\n  --- {name} ---")
        print(f"  decision entries {int(dec.sum())}   hi(risk>={HI}) "
              f"{int(hi.sum())}   lo(risk<{LO}) {int(lo.sum())}")

        # Reproduce the pre-registered risk-response Delta itself (P1 on RANDOM,
        # P2 on UNION) so the population can be checked against the locked
        # numbers before any new quantity computed on it is believed.
        ie = ACTION_NAMES.index("MIGRATE_EDGE")
        print(f"  reproduced Delta = P(EDGE|hi) - P(EDGE|lo):")
        delta = {}
        for k in ARM_FILES:
            P = probs_at(arms[k], S["obs"], S["mask"])
            delta[k] = float(P[hi][:, ie].mean() - P[lo][:, ie].mean())
        delta["theta0"] = float(P0[hi][:, ie].mean() - P0[lo][:, ie].mean())
        for k, v in delta.items():
            print(f"      {k:<8s} {v:+.4f}")

        DP = {}
        for k in ARM_FILES:
            DP[k] = probs_at(arms[k], S["obs"], S["mask"]) - P0

        rows = {}
        for regime, sel in (("all", dec), ("hi", hi), ("lo", lo)):
            n = int(sel.sum())
            if n == 0:
                continue
            ref = torch.as_tensor(DP[REF][sel].reshape(-1)).double()
            print(f"\n  regime={regime} (n={n})")
            print(f"  {'arm':<8s} {'mean_TV':>9s} {'||Dp||':>10s} "
                  f"{'proj_frac':>10s} {'cosine':>9s} {'resid_frac':>11s} "
                  f"{'d_p_EDGE':>9s} {'d_p_STAY':>9s}")
            rows[regime] = {}
            for k in ARM_FILES:
                d = DP[k][sel]
                v = torch.as_tensor(d.reshape(-1)).double()
                tv = float(0.5 * np.abs(d).sum(-1).mean())
                pf, c, rf = decomp(v, ref)
                de = float(d[..., ACTION_NAMES.index("MIGRATE_EDGE")].mean())
                ds = float(d[..., ACTION_NAMES.index("STAY")].mean())
                print(f"  {k:<8s} {tv:>9.5f} {float(v.norm()):>10.4f} "
                      f"{pf:>10.4f} {c:>9.4f} {rf:>11.4f} {de:>9.5f} "
                      f"{ds:>9.5f}")
                rows[regime][k] = {"mean_TV": tv, "norm": float(v.norm()),
                                   "proj_fraction": pf, "cosine": c,
                                   "residual_fraction": rf,
                                   "mean_dp_EDGE": de, "mean_dp_STAY": ds}
            rows[regime]["_n"] = n
        out[name] = rows
        out[name]["_reproduced_delta"] = delta

        # The headline contrast: proj_fraction hi vs lo for R3.
        if "hi" in rows and "lo" in rows:
            for k in ("R3", "R3_best", "A0"):
                ph, pl = rows["hi"][k]["proj_fraction"], rows["lo"][k]["proj_fraction"]
                print(f"  ==> {k:<8s} proj_frac  lo={pl:+.4f}  hi={ph:+.4f}  "
                      f"hi-lo={ph-pl:+.4f}")

    res["part_b_function_space"] = {
        "state_sets": {k: {"entries": int(v["mask"].shape[0]),
                           "decision_entries": int((v["mask"].sum(-1) > 1.5).sum())}
                       for k, v in src.items()},
        "n_episodes": len(starts), "hi_cut": HI, "lo_cut": LO,
        "results": out}


# ==================================================================== main

def main(argv=None):
    args = parse_args(argv)
    res = {"probe": "DIV-2 displacement geometry", "ref_arm": REF,
           "arms": ARM_FILES, "args": vars(args)}
    hdr = lambda t: print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

    hdr("PART A  PARAMETER-SPACE DISPLACEMENT FROM THE SHARED theta_0")
    arms, cfgs, _ = part_a(args, res)

    if not args.skip_function:
        hdr("PART B  FUNCTION-SPACE DISPLACEMENT ON THE UNCHANGED FIXED SETS")
        part_b(args, arms, cfgs, res)

    p = OUT_DIR / f"{TAG}_geometry_{args.tag}.json"
    p.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
