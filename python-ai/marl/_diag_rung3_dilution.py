#!/usr/bin/env python
"""
SPRINT 7 RUNG 3 -- is the useful high-risk actor signal DILUTED by the low-risk
bulk?  OFFLINE ONLY.

PRE-REGISTERED IN saved_models/marl/SPRINT_7_RUNG3_PREREGISTRATION.md
(md5 972dc323bc4a8d98ca0c5ad3273540ad), written and hash-pinned BEFORE this file
existed. Read that first; every threshold applied here is fixed there.

WHAT THIS TESTS. Sprint 7 R3 falsified the variance hypothesis: a 4x rollout
batch demonstrably raised actor-gradient SNR (real/shuffled 2.0275, z +4.37,
p 0.0000) and the risk response got WORSE. R3's report proposed a compositional
successor: high-risk decision entries are 3.3-6.5% of the total, the gradient
RESTRICTED to them points the right way in every arm (cos(g_hi, synth) =
+0.913 / +0.536 / +0.451), and the full gradient's direction is set by the ~95%
low-risk remainder.

THE EXACT DECOMPOSITION. At a checkpoint's own parameters the actor policy
gradient (entropy term excluded -- it is advantage-independent) is

    g = -(1/D) * SUM_{j in decision}  A_j * grad log pi(a_j | s_j)

which is LINEAR in A. Zeroing A outside a subset therefore yields exactly that
subset's partial sum, with the SAME 1/D normaliser. Partitioning the decision
entries by the high-risk rule gives

    g_full = g_hi + g_lo        EXACTLY

asserted numerically at runtime. Every quantity reported is a function of
(g_hi, g_lo, g_synth).

TWO DIFFERENT HYPOTHESES, MEASURED SEPARATELY -- the brief is explicit that
these must not be conflated:

  (A) the high-risk gradient CONTAINS a useful directional signal
      -> cos(g_hi, g_synth) > 0, and beats a cardinality-matched null
  (B) the high-risk gradient is LARGE ENOUGH TO MATERIALLY CONTROL the update
      -> ||g_hi||/||g_full||, and cos(g_full, g_lo) ~ 1 would mean it is NOT

A can hold while B fails. That combination IS the dilution hypothesis. If B
holds, dilution is not the constraint.

SCALING CONVENTION, deliberately different from the Rung 2.75 coherence probe.
That probe rescaled every condition to MATCHED L2 advantage mass, because it was
asking about coherence at fixed mass. This rung is asking about mass itself, so
g_full / g_hi / g_lo are left on their NATURAL scale -- rescaling them would
destroy the very quantity under test. Only `synth` is mass-matched (to g_full's
advantage mass), following the existing convention, so that ||g_synth|| stays
comparable across arms.

  This is NOT a redefinition of the published number. The coherence probe's
  adv_hi_m is exactly k_hi * adv_hi for a positive scalar k_hi, so its gradient
  is k_hi * g_hi and cos(g_hi_matched, g_synth) == cos(g_hi, g_synth)
  identically. The cos(g_hi, g_synth) values here are therefore directly
  comparable to the +0.913 / +0.536 / +0.451 disclosed in prereg S6. Only the
  NORM ||g_hi|| differs by that scalar, and the norms this rung reports are the
  natural-scale ones, which the coherence probe never reported.

THREE STATE SOURCES.
  OWN     the arm's own stochastic rollout, TRAIN window, training_start_ticks
          -- identical to _diag_rung2_75_coherence.build. Ratio is EXACTLY 1 at
          the arm's own parameters, so the vanilla-PG form is exact. This is the
          production-faithful source.
  RANDOM  uniform-legal actions, EVAL window -- identical to matched-states.
          Depends on no arm.
  UNION   A0-greedy pooled with R2-greedy, EVAL window -- identical to
          matched-states. On-manifold, symmetric.

The train/eval distinction is not cosmetic: load_agent_and_cfg's `window`
argument sets cfg.env.start_frac_lo/hi and nothing else. OWN uses the TRAIN
window because that is where the coherence probe measured and where training
actually happened; RANDOM/UNION use the EVAL window because that is where P1/P2
were scored. Each source keeps the window its own machinery already used.

On RANDOM and UNION the actions come from a FOREIGN behaviour policy, so the PPO
ratio is not 1 and what is computed there is a COUNTERFACTUAL gradient -- "what
the actor gradient would be if these were the sampled entries" -- not the
production update. Declared in the pre-registration, labelled in the output, and
no conclusion is drawn from RANDOM/UNION alone.

REUSE, NOT REIMPLEMENTATION. The state sets come from
_diag_rung2_75_matched_states' own eval_starts / random_trajectory / trajectory
/ union_source; the buffer from _diag_rung2_5_actor_stall.build_buffer_r2; the
gradient and mass matching from _diag_rung2_75_coherence.flat_pg_grad /
match_mass / cos. Those modules are imported unmodified. Because the existing
constructors DISCARD the act/rew/state/cont fields a gradient needs, this file
re-runs the identical rollout loop keeping them, and then ASSERTS its
obs/mask/risk arrays are bit-identical to the existing constructors' output,
including against union_source for UNION. Those assertions are the guarantee
that the state sets were not silently redefined.

NOT AN INTERVENTION. No optimiser is constructed, no parameter is written, no
checkpoint is saved, no production module is touched, no weighting is applied to
any real update. The 2x/4x/8x/16x sensitivity is arithmetic on two frozen
vectors.

Writes SPRINT_7_RUNG3_dilution_<tag>.json.
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

from marl.config import (                                          # noqa: E402
    ACTION_STAY, ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD,
)
from marl._diag_rung0 import load_agent_and_cfg, _replay, OUT_DIR   # noqa: E402
from marl._diag_rung2_5_actor_stall import build_buffer_r2          # noqa: E402
from marl._diag_rung2_5_targets import (                            # noqa: E402
    training_start_ticks, TRAIN_SEED,
)
from marl._diag_rung2_75_coherence import (                        # noqa: E402
    flat_pg_grad, match_mass, cos, HI_RISK,
)
from marl._diag_rung2_75_matched_states import (                   # noqa: E402
    eval_starts, random_trajectory, trajectory, union_source,
)
from marl.env import DTMarlEnv                                     # noqa: E402
from marl.mappo import masked_dist                                 # noqa: E402

TAG = "SPRINT_7_RUNG3"
ARMS = {"A0": "mappo_A0_cpu_repro.pth",
        "R2": "mappo_R2_mc_target.pth",
        "R3": "R3_batch32.pth"}
WEIGHTS = [1, 2, 4, 8, 16]
# the behavioural fact D4 must reproduce: UNION risk response Delta, R3 report S6
UNION_DELTA = {"R2": 0.1579, "R3": 0.1242, "A0": 0.0191}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=32,
                   help="OWN-source measurement batch. 32 is the PRE-REGISTERED "
                        "default: R3 report S13.2 showed the 8-episode buffer "
                        "is underpowered and reverses one published rank order. "
                        "Run with 8 as the estimator-noise control.")
    p.add_argument("--perms", type=int, default=200,
                   help="cardinality-matched random-subset null draws")
    p.add_argument("--start-seed", type=int, default=20260825,
                   help="matched-states' own default, so RANDOM/UNION are the "
                        "same sets P1/P2 were scored on")
    p.add_argument("--random-seed", type=int, default=31337)
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# Recording rollouts. Identical loops to the existing constructors, keeping the
# fields a gradient needs. Verified bit-identical against them at runtime.
# --------------------------------------------------------------------------

def record_own(agent, cfg_train, n_eps):
    """
    The arm's own stochastic buffer -- EXACTLY _diag_rung2_75_coherence.build:
    TRAIN-window cfg, training_start_ticks, torch.manual_seed(TRAIN_SEED + j),
    sample=True. Reproduced here rather than imported because build() also
    constructs the buffer and discards the per-episode recs this rung needs.
    """
    env = DTMarlEnv(cfg_train.env, cfg_train.reward)
    starts = training_start_ticks(env, n_eps)
    recs, trunc = [], []
    for j, s in enumerate(starts):
        torch.manual_seed(TRAIN_SEED + j)
        r = _replay(env, agent, int(s), j, record=True, sample=True)
        recs.append(r["rec"])
        trunc.append(bool(r["n_steps"] >= env.cfg.episode_steps))
    return recs, trunc, [int(s) for s in starts]


def record_greedy(agent, cfg_eval, starts):
    """Greedy replay keeping the full rec. Same _replay call as trajectory()."""
    env = DTMarlEnv(cfg_eval.env, cfg_eval.reward)
    recs, trunc = [], []
    for j, s in enumerate(starts):
        r = _replay(env, agent, int(s), j, record=True, sample=False)
        recs.append(r["rec"])
        trunc.append(bool(r["n_steps"] >= env.cfg.episode_steps))
    return recs, trunc


def record_random(cfg_eval, starts, seed):
    """
    Uniform over LEGAL actions, keeping state/act/rew/cont so a buffer can be
    built. The loop, the RNG construction and the per-agent rng.choice(legal)
    call are copied from _diag_rung2_75_matched_states.random_trajectory so the
    trajectory is identical; verified by assertion in main().
    """
    env = DTMarlEnv(cfg_eval.env, cfg_eval.reward)
    rng = np.random.default_rng(seed)
    recs, trunc = [], []
    for j, s in enumerate(starts):
        obs, state, masks = env.reset(episode_start_tick=int(s), seed=j)
        R = {k: [] for k in ("obs", "state", "act", "logp", "rew", "mask",
                             "cont", "risk")}
        done, step = False, 0
        while not done:
            R["obs"].append(obs.copy())
            R["state"].append(state.copy())
            R["mask"].append(masks.copy())
            R["risk"].append(np.array([env.risk_at(i)
                                       for i in range(env.n_agents)],
                                      np.float32))
            a = np.empty(env.n_agents, np.int64)
            for i in range(env.n_agents):
                legal = np.flatnonzero(np.asarray(masks[i]) > 0.5)
                a[i] = int(rng.choice(legal)) if legal.size else 0
            R["act"].append(a.copy())
            # logp is unused downstream: the vanilla-PG form recomputes
            # log pi from the agent and never reads the behaviour logp.
            R["logp"].append(np.zeros(env.n_agents, np.float32))
            obs, state, rew, done, info = env.step(a)
            R["rew"].append(rew.copy())
            R["cont"].append(0.0 if done else 1.0)
            masks = info["action_masks"]
            step += 1
        rec = {k: np.asarray(v) for k, v in R.items()}
        rec["boot_state"] = state
        recs.append(rec)
        trunc.append(bool(step >= env.cfg.episode_steps))
    return recs, trunc


def _concat(recs, key):
    return np.concatenate([np.asarray(r[key]) for r in recs], axis=0)


# --------------------------------------------------------------------------


def cached_grad_maker(agent, obs, act, mask, dec_t):
    """
    flat_pg_grad recomputes the actor forward pass on every call. This rung
    needs ~205 gradients per cell, so the forward pass and its graph are built
    ONCE and reused with retain_graph=True. Mathematically identical to
    flat_pg_grad -- verified against it on g_full in every cell, asserted
    to 1e-6.
    """
    lp = []
    for i in range(agent.n_agents):
        d = masked_dist(agent.actor.logits(i, obs[:, i, :]), mask[:, i, :])
        lp.append(d.log_prob(act[:, i]))
    lp = torch.stack(lp, dim=1)
    denom = dec_t.sum().clamp(min=1.0)
    params = list(agent.actor.parameters())

    def G(a_np):
        a = torch.as_tensor(a_np, device=lp.device)
        pg = -(lp * a * dec_t).sum() / denom
        gs = torch.autograd.grad(pg, params, allow_unused=True,
                                 retain_graph=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                          for g, p in zip(gs, params)])
    return G


def build_synth(hi, a_np, like, plus_action):
    """+1 on `plus_action` / -1 on STAY at high-risk entries, 0 elsewhere."""
    s = np.zeros_like(like)
    s[hi & (a_np == plus_action)] = +1.0
    s[hi & (a_np == ACTION_STAY)] = -1.0
    return s


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 RUNG 3 -- DILUTION: is the high-risk actor signal drowned")
    print("        by the low-risk bulk?   g_full = g_hi + g_lo, exactly.")
    print(f"        OFFLINE ONLY. OWN batch = {args.episodes} episodes")
    print("=" * 78)
    print("  pre-registration: SPRINT_7_RUNG3_PREREGISTRATION.md")
    print("                    md5 972dc323bc4a8d98ca0c5ad3273540ad")
    print("  (A) does g_hi contain useful direction?  (B) does g_hi control the")
    print("      update?   These are DIFFERENT and are scored separately.")

    agents, cfg_tr, cfg_ev = {}, {}, {}
    for tag, fn in ARMS.items():
        pth = str(OUT_DIR / fn)
        a, _, c_ev = load_agent_and_cfg(pth, args.device, "eval")
        _, _, c_tr = load_agent_and_cfg(pth, args.device, "train")
        agents[tag], cfg_ev[tag], cfg_tr[tag] = a, c_ev, c_tr

    # ---- state sources -----------------------------------------------------
    starts, env0 = eval_starts(cfg_ev["A0"], 32, args.start_seed)
    print(f"\n  EVAL window [{env0._min_start}, {env0._max_start}]  "
          f"greedy/random starts {len(starts)}")

    print("  building RANDOM (uniform-legal) with full recording ...")
    rnd_recs, rnd_trunc = record_random(cfg_ev["A0"], starts, args.random_seed)
    print("  building A0-greedy and R2-greedy with full recording ...")
    a0_recs, a0_trunc = record_greedy(agents["A0"], cfg_ev["A0"], starts)
    r2_recs, r2_trunc = record_greedy(agents["R2"], cfg_ev["R2"], starts)

    # ---- PROVE the state sets were not silently redefined ------------------
    print("\n  verifying the recorded sets are BIT-IDENTICAL to the existing")
    print("  matched-states constructors (guards against silent redefinition):")
    checks = {}
    ref_rnd = random_trajectory(cfg_ev["A0"], starts, args.random_seed)
    ref_a0 = trajectory(agents["A0"], cfg_ev["A0"], starts)
    ref_r2 = trajectory(agents["R2"], cfg_ev["R2"], starts)
    ref_union = union_source(ref_a0, ref_r2)
    union_recs = a0_recs + r2_recs
    union_trunc = a0_trunc + r2_trunc
    for nm, recs, ref in (("RANDOM", rnd_recs, ref_rnd),
                          ("A0-greedy", a0_recs, ref_a0),
                          ("R2-greedy", r2_recs, ref_r2),
                          ("UNION", union_recs, ref_union)):
        ok = {}
        for k in ("obs", "mask", "risk"):
            mine = _concat(recs, k)
            ok[k] = bool(mine.shape == ref[k].shape
                         and np.array_equal(mine, ref[k]))
        checks[nm] = ok
        print(f"     {nm:>10}: " + "  ".join(
            f"{k} {'IDENTICAL' if v else '*** DIFFERS ***'}"
            for k, v in ok.items()))
        assert all(ok.values()), (f"{nm} state set differs from the existing "
                                 f"constructor -- refusing to continue")
    print(f"     UNION verified against union_source(A0, R2): "
          f"{len(a0_recs)} + {len(r2_recs)} episodes")

    neutral = {"RANDOM": (rnd_recs, rnd_trunc),
               "UNION": (union_recs, union_trunc)}

    out, rows = {}, []
    for arm in ARMS:
        agent = agents[arm]
        own_recs, own_trunc, own_starts = record_own(agent, cfg_tr[arm],
                                                    args.episodes)
        sources = {"OWN": (own_recs, own_trunc)}
        sources.update(neutral)
        out[arm] = {"own_starts": own_starts}
        for src, (recs, trunc) in sources.items():
            onpol = (src == "OWN")
            print(f"\n-- {arm} / {src} "
                  + "-" * max(4, 58 - len(arm) - len(src)))
            if not onpol:
                print("   COUNTERFACTUAL gradient: actions are from a foreign")
                print("   behaviour policy, so the PPO ratio is not 1 here.")
            buf = build_buffer_r2(agent, recs, trunc)
            T = len(buf)
            risk = _concat(recs, "risk")[:T]
            dev = agent.device
            obs = torch.as_tensor(buf.obs[:T], device=dev)
            act = torch.as_tensor(buf.act[:T], device=dev)
            mask = torch.as_tensor(buf.mask[:T], device=dev)
            dec_t = torch.as_tensor(buf.decision[:T], device=dev)
            dec = np.asarray(buf.decision[:T]) > 0.5
            a_np = np.asarray(buf.act[:T])

            # advantage: production convention, normalised over DECISION
            # entries only, exactly mappo.py:360-366
            adv_raw, _ = agent.compute_gae(buf)
            adv = adv_raw.copy()
            adv = (adv - adv[dec].mean()) / (adv[dec].std() + 1e-8)

            hi = (risk > HI_RISK) & dec
            lo = (~(risk > HI_RISK)) & dec
            n_dec, n_hi, n_lo = int(dec.sum()), int(hi.sum()), int(lo.sum())
            assert n_hi + n_lo == n_dec
            # the OTHER high-risk rule used elsewhere in this project, reported
            # for transparency; NOT used for the partition (see prereg S3)
            n_hi_bucket = int(((np.minimum((risk * 5).astype(int), 4) >= 3)
                               & dec).sum())
            frac_hi = n_hi / max(n_dec, 1)
            print(f"   T={T}  decision={n_dec}  high-risk(risk>0.50)={n_hi} "
                  f"({frac_hi:.2%})  [bucket rule >=0.6 would give "
                  f"{n_hi_bucket}]")

            G = cached_grad_maker(agent, obs, act, mask, dec_t)
            adv_hi = np.where(hi, adv, 0.0).astype(adv.dtype)
            adv_lo = np.where(lo, adv, 0.0).astype(adv.dtype)
            g_full, g_hi, g_lo = G(adv), G(adv_hi), G(adv_lo)

            # ---- exactness assertions -------------------------------------
            g_ref = flat_pg_grad(agent, obs, act, mask, dec_t,
                                 torch.as_tensor(adv, device=dev))
            cache_err = float((g_full - g_ref).norm() / g_ref.norm())
            add_err = float((g_full - (g_hi + g_lo)).norm() / g_full.norm())
            print(f"   EXACTNESS  cached-vs-flat_pg_grad {cache_err:.2e}   "
                  f"g_full-(g_hi+g_lo) {add_err:.2e}")
            assert cache_err < 1e-6, f"cached gradient differs: {cache_err}"
            assert add_err < 1e-5, f"decomposition not additive: {add_err}"

            # ---- reference directions -------------------------------------
            syn_e = build_synth(hi, a_np, adv, ACTION_MIGRATE_EDGE)
            syn_c = build_synth(hi, a_np, adv, ACTION_MIGRATE_CLOUD)
            syn_e_m, k_e = match_mass(syn_e, adv, dec)
            syn_c_m, k_c = match_mass(syn_c, adv, dec)
            n_e = int((syn_e > 0).sum()); n_c = int((syn_c > 0).sum())
            n_s = int((syn_e < 0).sum())
            assert k_e > 0.0, ("synth_EDGE has empty support -- cannot form "
                              "the reference direction")
            g_syn = G(syn_e_m)
            g_syn_c = G(syn_c_m) if k_c > 0.0 else None
            print(f"   synth support: EDGE(+) {n_e}  CLOUD(+) {n_c}  "
                  f"STAY(-) {n_s}")

            nf, nh, nl = (float(g_full.norm()), float(g_hi.norm()),
                          float(g_lo.norm()))
            c_full = cos(g_full, g_syn)
            c_hi = cos(g_hi, g_syn)
            c_lo = cos(g_lo, g_syn)
            c_hi_lo = cos(g_hi, g_lo)
            c_full_lo = cos(g_full, g_lo)
            c_full_hi = cos(g_full, g_hi)
            c_hi_cloud = cos(g_hi, g_syn_c) if g_syn_c is not None else None
            print(f"   ||g_full|| {nf:.5e}   ||g_hi|| {nh:.5e}   "
                  f"||g_lo|| {nl:.5e}")
            print(f"   MASS   hi/full {nh / nf:.4f}   lo/full {nl / nf:.4f}   "
                  f"hi/lo {nh / nl:.4f}")
            print(f"   COS    (full,syn) {c_full:+.5f}   (hi,syn) {c_hi:+.5f}"
                  f"   (lo,syn) {c_lo:+.5f}")
            print(f"          (hi,lo) {c_hi_lo:+.5f}   (full,lo) "
                  f"{c_full_lo:+.5f}   (full,hi) {c_full_hi:+.5f}")
            if c_hi_cloud is not None:
                print(f"          (hi, synth_CLOUD) {c_hi_cloud:+.5f}   "
                      f"<- action-channel control")

            # ---- cardinality-matched random-subset NULL (brief item 7) ----
            # Draw n_hi decision entries uniformly from ALL decision entries.
            # Asks: is the high-risk set special vs an arbitrary set of the
            # same size? DIFFERENT from Rung 2.75's null, which permuted
            # advantage VALUES over a fixed support.
            idx = np.flatnonzero(dec.reshape(-1))
            flat = adv.reshape(-1)
            rng = np.random.default_rng(7)
            ncos, nnorm = [], []
            for _ in range(args.perms):
                pick = rng.choice(idx, size=n_hi, replace=False)
                a = np.zeros_like(flat)
                a[pick] = flat[pick]
                gs = G(a.reshape(adv.shape))
                c = cos(gs, g_syn)
                if c is not None:
                    ncos.append(c)
                nnorm.append(float(gs.norm()))
            ncos = np.asarray(ncos); nnorm = np.asarray(nnorm)
            p_cos_one = float((ncos >= c_hi).mean())
            z_cos = (float((c_hi - ncos.mean()) / ncos.std(ddof=1))
                     if ncos.size > 1 and ncos.std(ddof=1) > 0 else None)
            p_norm_one = float((nnorm >= nh).mean())
            zs = f"{z_cos:+.2f}" if z_cos is not None else "n/a"
            print(f"   NULL (n={args.perms} random subsets of size {n_hi})")
            print(f"      cos(g_subset, g_synth) {ncos.mean():+.5f} +/- "
                  f"{ncos.std(ddof=1):.5f}   [{ncos.min():+.4f}, "
                  f"{ncos.max():+.4f}]")
            print(f"      cos(g_hi,g_synth) {c_hi:+.5f}  vs null: z {zs}, "
                  f"one-sided p {p_cos_one:.4f}")
            print(f"      ||g_subset|| {nnorm.mean():.5e} +/- "
                  f"{nnorm.std(ddof=1):.2e}   ||g_hi|| p {p_norm_one:.4f}")

            # ---- analytical sensitivity g(w) = w*g_hi + g_lo --------------
            # Arithmetic on two frozen vectors. NOT applied to any update.
            sens = {}
            for w in WEIGHTS:
                gw = w * g_hi + g_lo
                ngw = float(gw.norm())
                sens[w] = dict(
                    cos_vs_synth=cos(gw, g_syn),
                    cos_vs_baseline=cos(gw, g_full),
                    norm_ratio=ngw / nf,
                    mass_share_hi=(w * nh) / ngw,
                )
            dot_hi = float(torch.dot(g_hi, g_syn))
            dot_lo = float(torch.dot(g_lo, g_syn))
            w_star = (-dot_lo / dot_hi) if abs(dot_hi) > 1e-30 else None
            print("   SENSITIVITY  g(w) = w*g_hi + g_lo   (analytic, offline)")
            print(f"      {'w':>4} {'cos(g(w),syn)':>14} "
                  f"{'cos(g(w),g(1))':>15} {'||g(w)||/||g(1)||':>18}")
            for w in WEIGHTS:
                s = sens[w]
                print(f"      {w:>4} {s['cos_vs_synth']:>+14.5f} "
                      f"{s['cos_vs_baseline']:>+15.5f} "
                      f"{s['norm_ratio']:>18.4f}")
            ws = f"{w_star:+.3f}" if w_star is not None else "n/a"
            print(f"      <g_hi,syn> {dot_hi:+.4e}  <g_lo,syn> {dot_lo:+.4e}"
                  f"  -> alignment numerator flips sign at w* = {ws}")

            out[arm][src] = dict(
                source=src, on_policy=onpol, ratio_is_one=onpol,
                counterfactual=not onpol,
                window="train" if onpol else "eval",
                T=T, n_decision=n_dec, n_highrisk=n_hi, n_lowrisk=n_lo,
                n_highrisk_bucket_rule=n_hi_bucket,
                highrisk_frac_of_decision=frac_hi,
                exactness=dict(cached_vs_flat_pg_grad_rel=cache_err,
                               additivity_rel=add_err),
                grad_norms=dict(full=nf, hi=nh, lo=nl),
                mass=dict(hi_over_full=nh / nf, lo_over_full=nl / nf,
                          hi_over_lo=nh / nl),
                cosines=dict(full_vs_synth=c_full, hi_vs_synth=c_hi,
                             lo_vs_synth=c_lo, hi_vs_lo=c_hi_lo,
                             full_vs_lo=c_full_lo, full_vs_hi=c_full_hi,
                             hi_vs_synth_CLOUD=c_hi_cloud),
                synth=dict(scale_edge=k_e, scale_cloud=k_c,
                           n_edge_pos=n_e, n_cloud_pos=n_c, n_stay_neg=n_s),
                null_cardinality_matched=dict(
                    n_draws=int(args.perms), subset_size=n_hi,
                    cos_mean=float(ncos.mean()),
                    cos_sd=float(ncos.std(ddof=1)),
                    cos_min=float(ncos.min()), cos_max=float(ncos.max()),
                    cos_p95=float(np.percentile(ncos, 95)),
                    z_cos=z_cos, p_one_sided_cos=p_cos_one,
                    norm_mean=float(nnorm.mean()),
                    norm_sd=float(nnorm.std(ddof=1)),
                    p_one_sided_norm=p_norm_one,
                    definition="n_hi decision entries drawn uniformly from ALL "
                               "decision entries, arm's own advantages kept on "
                               "the drawn support; tests whether the HIGH-RISK "
                               "SET is special versus an arbitrary set of the "
                               "same cardinality"),
                sensitivity={str(w): sens[w] for w in WEIGHTS},
                sensitivity_w_star=w_star,
                dots=dict(g_hi_dot_synth=dot_hi, g_lo_dot_synth=dot_lo),
            )
            rows.append((arm, src, frac_hi, nh / nf, nh / nl, c_full, c_hi,
                         c_lo, c_hi_lo, c_full_lo, p_cos_one))

    # ---------------------------------------------------------------- summary
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  {'arm':>3} {'src':>7} {'hi%':>6} {'|hi|/|full|':>11} "
          f"{'|hi|/|lo|':>10} {'cos(full,s)':>11} {'cos(hi,s)':>10} "
          f"{'cos(lo,s)':>10} {'cos(hi,lo)':>11} {'cos(full,lo)':>12} "
          f"{'nullp':>6}")
    for (arm, src, fh, mhf, mhl, cf, ch, cl, chl, cfl, p) in rows:
        print(f"  {arm:>3} {src:>7} {fh:>6.2%} {mhf:>11.4f} {mhl:>10.4f} "
              f"{cf:>+11.5f} {ch:>+10.5f} {cl:>+10.5f} {chl:>+11.5f} "
              f"{cfl:>+12.5f} {p:>6.3f}")
    print("\n  (A) useful direction?  -> cos(hi,s) > 0 and nullp < 0.05")
    print("  (B) controls update?   -> |hi|/|full| large, cos(full,lo) far "
          "from 1")
    print("  dilution needs (A) TRUE and (B) FALSE.")
    print("  cos(hi,lo) < 0 would be COMPETING gradients, not dilution.")

    blob = dict(
        probe=f"{TAG}_dilution",
        status="OFFLINE DIAGNOSTIC. No training, no production modification, "
               "no weighting applied to any real update.",
        preregistration=dict(
            file="SPRINT_7_RUNG3_PREREGISTRATION.md",
            md5="972dc323bc4a8d98ca0c5ad3273540ad",
            note="written and hash-pinned before this probe existed"),
        hypothesis="H2: the high-risk actor signal is a 3-6% minority whose "
                   "correct direction is drowned by the ~95% low-risk bulk",
        two_hypotheses=dict(
            A="g_hi CONTAINS useful direction -> cos(g_hi,g_synth) > 0 beating "
              "a cardinality-matched null",
            B="g_hi is LARGE ENOUGH TO CONTROL the update -> "
              "||g_hi||/||g_full|| and cos(g_full,g_lo); dilution requires A "
              "true and B false"),
        decomposition="g_full = g_hi + g_lo exactly (the PG is linear in A and "
                      "both terms carry the same 1/D normaliser); asserted "
                      "numerically per cell",
        scaling_convention="g_full/g_hi/g_lo left on their NATURAL scale "
                           "because mass IS the quantity under test; only "
                           "synth is mass-matched, per the Rung 2.75 "
                           "convention. cos(g_hi,g_synth) is UNCHANGED by that "
                           "difference (scaling g_hi by a positive scalar "
                           "leaves the cosine identical), so it is directly "
                           "comparable to the coherence probe's published "
                           "values; only ||g_hi|| differs, and the coherence "
                           "probe never reported the natural-scale norm.",
        high_risk_rule=f"risk > {HI_RISK} (the coherence probe's HI_RISK, used "
                       f"for the gradient partition). n_highrisk_bucket_rule "
                       f"reports what the matched-states bucket rule "
                       f"(min(int(risk*5),4)>=3, i.e. risk>=0.6) would give; "
                       f"that rule is NOT used for the partition.",
        window_note="OWN uses the TRAIN window and training_start_ticks "
                    "(coherence-probe convention, and where training actually "
                    "happened); RANDOM/UNION use the EVAL window and "
                    "eval_starts (matched-states convention, where P1/P2 were "
                    "scored). load_agent_and_cfg's window argument changes "
                    "only cfg.env.start_frac_lo/hi.",
        counterfactual_caveat="on RANDOM and UNION the actions come from a "
                              "foreign behaviour policy, so the PPO ratio is "
                              "not 1 and the gradient is counterfactual, not "
                              "the production update. OWN is the only source "
                              "where the vanilla-PG identity is exact.",
        state_set_verification=checks,
        provenance="state sets from _diag_rung2_75_matched_states' own "
                   "eval_starts/random_trajectory/trajectory/union_source; "
                   "buffer from _diag_rung2_5_actor_stall.build_buffer_r2; "
                   "gradient and mass matching from "
                   "_diag_rung2_75_coherence. All imported unmodified. "
                   "Recorded rollouts asserted bit-identical to the existing "
                   "constructors on obs/mask/risk.",
        measurement=dict(episodes=args.episodes, perms=args.perms,
                         start_seed=args.start_seed,
                         random_seed=args.random_seed,
                         weights=WEIGHTS, device=args.device),
        union_delta_to_explain=UNION_DELTA,
        per_arm=out,
    )
    p = OUT_DIR / f"{TAG}_dilution_{args.tag}.json"
    p.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n  wrote {p}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
