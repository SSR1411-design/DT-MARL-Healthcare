#!/usr/bin/env python
"""
SPRINT 7 RUNG 2.75 -- items B/C bridge: is the advantage signal WEAK or
INCOHERENT?

WHY THIS EXISTS. The plasticity probe (_diag_rung2_75_plasticity.py) produced a
number I can only interpret by comparing across two scripts, which is exactly
the kind of cross-artifact inference this project has already been burned by.
The comparison is:

  production advantage : normalised to mean 0 / sd 1 over DECISION entries
                         (mappo.py:360-366), so mean|A| ~ 0.8 at EVERY decision
                         entry -- dense.
  synthetic advantage  : +/-1 at high-risk entries only, i.e. 212/3452 = 6.1%
                         of A0's decision entries, so mean|A| ~ 0.061 -- sparse.
                         13x LESS advantage mass.
  measured d(theta)    : production A0 0.010596 per update
                         (SPRINT_7_RUNG2_5_actor_stall.json)
                         synthetic  A0 0.012537 per update -- the SAME, in fact
                         slightly larger, on 1/13th of the mass.

That comparison SUGGESTS production's gradient contributions cancel rather than
add. It does not establish it: the two numbers come from different scripts and
the advantage vectors differ in support, scale and sparsity all at once.

THIS FILE MEASURES IT DIRECTLY, one buffer, one set of starting parameters, one
backward pass per condition, no optimiser step at all.

  At the checkpoint's own parameters the PPO ratio is EXACTLY 1 (new_lp ==
  old_lp), so s1 = s2 = A and min(s1, s2) = A. The actor's policy-gradient loss
  is therefore exactly the vanilla one:

      g(A) = -(1/denom) * sum_over_decision_entries  A_j * grad log pi(a_j|s_j)

  g is LINEAR in A. So comparing g under different A vectors of MATCHED L2 norm
  is a clean, assumption-free measurement of how much of A's magnitude survives
  summation -- i.e. of coherence, not of scale.

FOUR CONDITIONS, all at matched L2 advantage mass over decision entries:

  real      production's own GAE, normalised exactly as mappo.update does
  synth     +M on MIGRATE_EDGE / -M on STAY at risk > 0.50, 0 elsewhere; the
            perfectly-coherent reference direction
  shuffled  real's values randomly PERMUTED among decision entries. Preserves
            the marginal distribution of A exactly and destroys only the
            state-action correspondence. This is the NULL: whatever gradient
            norm survives here is what pure noise of that magnitude produces.
  real_hi   real, zeroed outside high-risk entries. Isolates whether the signal
            AT the states the research question is about points the right way,
            independently of what it does elsewhere.

REPORTED, per arm:

  ||g|| per condition, at matched mass -> how much magnitude survives summation
  ||g_real|| / ||g_shuffled||          -> coherence gain over its own null.
                                          ~1.0 means the real advantage carries
                                          no more usable state-action
                                          information than a permutation of
                                          itself.
  cos(g_real, g_synth)                 -> does the real signal push the policy
                                          in the direction that would raise
                                          MIGRATE_EDGE at high risk? Signed:
                                          negative means it actively pushes the
                                          other way.
  cos(g_shuffled, g_synth)             -> the null distribution for that cosine,
                                          so the real cosine can be read against
                                          chance rather than against zero.

NOT A FIX AND NOT AN INTERVENTION. No optimiser is constructed, no parameter is
written, no checkpoint is saved, production code is untouched. This is one
backward pass per condition on a loaded checkpoint.

Writes SPRINT_7_RUNG2_75_coherence.json.
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
from marl._diag_rung2_5_actor_stall import build_buffer_r2       # noqa: E402
from marl._diag_rung2_5_targets import (                         # noqa: E402
    training_start_ticks, TRAIN_SEED,
)
from marl.env import DTMarlEnv                                   # noqa: E402
from marl.mappo import masked_dist                              # noqa: E402

TAG = "SPRINT_7_RUNG2_75"
HI_RISK = 0.50
MODELS = {"A0": "mappo_A0_cpu_repro.pth", "R2": "mappo_R2_mc_target.pth"}
# d(theta) per update measured in Rung 2.5's replica, for the scale check
RUNG2_5_DTHETA = {"A0": 0.010596, "R2": 0.005430}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--perms", type=int, default=200,
                   help="permutation draws for the shuffled null")
    # SPRINT 7 R3. Additive arm registration, following the same `name=file`
    # convention the stepcollapse probe already uses. It defaults to EMPTY so
    # MODELS -- and therefore the Rung 2.75 `main` artifact -- stays exactly
    # reproducible. Every arm in this probe is computed independently from its
    # own checkpoint and its own buffer, so adding an arm cannot perturb any
    # other arm's numbers; only the summary table gains a row.
    p.add_argument("--also", default="",
                   help="comma-separated extra arms as name=checkpoint_file")
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


def build(agent, cfg, n_eps):
    """The real 8-episode stochastic buffer, plus per-entry risk."""
    env = DTMarlEnv(cfg.env, cfg.reward)
    starts = training_start_ticks(env, n_eps)
    recs, trunc = [], []
    for j, s in enumerate(starts):
        torch.manual_seed(TRAIN_SEED + j)
        r = _replay(env, agent, int(s), j, record=True, sample=True)
        recs.append(r["rec"])
        trunc.append(bool(r["n_steps"] >= env.cfg.episode_steps))
    buf = build_buffer_r2(agent, recs, trunc)
    risk = np.concatenate([np.asarray(r["risk"]) for r in recs], axis=0)
    return buf, risk, [int(s) for s in starts], trunc


def flat_pg_grad(agent, obs, act, mask, dec, adv):
    """
    The exact full-batch actor policy gradient at the checkpoint's parameters,
    flattened. Entropy term deliberately EXCLUDED: it is advantage-independent,
    so including it would add the same vector to every condition and inflate
    every cosine toward 1.
    """
    lp = []
    for i in range(agent.n_agents):
        d = masked_dist(agent.actor.logits(i, obs[:, i, :]), mask[:, i, :])
        lp.append(d.log_prob(act[:, i]))
    lp = torch.stack(lp, dim=1)
    denom = dec.sum().clamp(min=1.0)
    pg = -(lp * adv * dec).sum() / denom
    gs = torch.autograd.grad(pg, list(agent.actor.parameters()),
                             allow_unused=True, retain_graph=False)
    return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1)
                      for g, p in zip(gs, agent.actor.parameters())])


def match_mass(a, ref, sel):
    """Rescale `a` so its L2 norm over `sel` equals `ref`'s. a is modified copy."""
    na = float(np.linalg.norm(a[sel]))
    nr = float(np.linalg.norm(ref[sel]))
    if na < 1e-12:
        return a.copy(), 0.0
    k = nr / na
    return a * k, k


def cos(u, v):
    du, dv = float(u.norm()), float(v.norm())
    if du < 1e-20 or dv < 1e-20:
        return None
    return float(torch.dot(u, v) / (du * dv))


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 RUNG 2.75 -- items B/C bridge: WEAK signal or INCOHERENT")
    print("        signal?  gradient norms at MATCHED advantage mass")
    print("        (one backward pass per condition; no optimiser, no step)")
    print("=" * 78)

    arms = dict(MODELS)
    for kv in (x for x in args.also.split(",") if x.strip()):
        k, v = kv.split("=", 1)
        arms[k.strip()] = v.strip()

    out = {}
    for tag, fn in arms.items():
        print(f"\n-- {tag} ({fn}) " + "-" * (56 - len(tag) - len(fn)))
        agent, _, cfg = load_agent_and_cfg(str(OUT_DIR / fn), args.device,
                                           "train")
        buf, risk, starts, trunc = build(agent, cfg, args.episodes)
        T = len(buf)
        dev = agent.device
        obs = torch.as_tensor(buf.obs[:T], device=dev)
        act = torch.as_tensor(buf.act[:T], device=dev)
        mask = torch.as_tensor(buf.mask[:T], device=dev)
        dec_t = torch.as_tensor(buf.decision[:T], device=dev)
        dec = np.asarray(buf.decision[:T]) > 0.5
        a_np = np.asarray(buf.act[:T])
        hi = (np.asarray(risk[:T]) > HI_RISK) & dec

        # ---- production's own advantage, normalised exactly as mappo does ----
        # compute_gae returns (adv, ret); mappo.update takes adv from here in
        # BOTH arms -- only the critic's regression target differs under
        # critic_target == "mc", and the critic is irrelevant to this probe.
        adv_raw, _ = agent.compute_gae(buf)
        adv_real = adv_raw.copy()
        m, s = adv_real[dec].mean(), adv_real[dec].std()
        adv_real = (adv_real - m) / (s + 1e-8)
        print(f"   T={T}  decision={int(dec.sum())}  high-risk={int(hi.sum())}"
              f"  ({hi.sum() / max(dec.sum(), 1):.1%} of decision entries)")
        print(f"   adv pre-norm  mean {adv_raw.mean():+.4f} sd "
              f"{adv_raw.std():.4f}   post-norm over decision entries "
              f"mean {adv_real[dec].mean():+.2e} sd {adv_real[dec].std():.4f}")

        # ---- the perfectly-coherent reference direction ----
        adv_syn = np.zeros_like(adv_real)
        adv_syn[hi & (a_np == ACTION_MIGRATE_EDGE)] = +1.0
        adv_syn[hi & (a_np == ACTION_STAY)] = -1.0
        n_pos = int((adv_syn > 0).sum()); n_neg = int((adv_syn < 0).sum())

        # ---- real restricted to high-risk entries ----
        adv_hi = np.where(hi, adv_real, 0.0).astype(adv_real.dtype)

        # ---- match L2 mass over decision entries to `real` ----
        adv_syn_m, k_syn = match_mass(adv_syn, adv_real, dec)
        adv_hi_m, k_hi = match_mass(adv_hi, adv_real, dec)
        mass_real = float(np.linalg.norm(adv_real[dec]))
        print(f"   L2 mass over decision entries (all matched to real "
              f"{mass_real:.2f}):  synth x{k_syn:.2f}  real_hi x{k_hi:.2f}")
        print(f"   synth support: {n_pos} EDGE(+) / {n_neg} STAY(-) entries")

        def G(a):
            return flat_pg_grad(agent, obs, act, mask, dec_t,
                                torch.as_tensor(a, device=dev))

        g_real = G(adv_real)
        g_syn = G(adv_syn_m)
        g_hi = G(adv_hi_m)

        # ---- the null: permute real's values among decision entries ----
        idx = np.flatnonzero(dec.reshape(-1))
        base = adv_real.reshape(-1)
        rng = np.random.default_rng(7)
        norms, coss = [], []
        for b in range(args.perms):
            a = base.copy()
            a[idx] = a[rng.permutation(idx)]
            gp = G(a.reshape(adv_real.shape))
            norms.append(float(gp.norm()))
            c = cos(gp, g_syn)
            if c is not None:
                coss.append(c)
        norms = np.asarray(norms); coss = np.asarray(coss)

        nr, ns, nh = float(g_real.norm()), float(g_syn.norm()), float(g_hi.norm())
        c_rs, c_hs = cos(g_real, g_syn), cos(g_hi, g_syn)
        # where does the real cosine sit in the permutation null?
        p_cos = float((np.abs(coss) >= abs(c_rs)).mean()) if coss.size else None
        p_norm = float((norms >= nr).mean()) if norms.size else None
        z_norm = float((nr - norms.mean()) / norms.std(ddof=1)) \
            if norms.size > 1 and norms.std(ddof=1) > 0 else None

        print(f"\n   ||g|| at matched mass")
        print(f"      real                 {nr:.6e}")
        print(f"      synth (coherent)     {ns:.6e}    "
              f"real/synth = {nr / ns:.4f}")
        print(f"      real_hi              {nh:.6e}")
        print(f"      shuffled NULL        {norms.mean():.6e} "
              f"+/- {norms.std(ddof=1):.2e}  "
              f"[{norms.min():.3e}, {norms.max():.3e}]")
        zs = f"{z_norm:+.2f}" if z_norm is not None else "n/a"
        ps = f"{p_norm:.4f}" if p_norm is not None else "n/a"
        print(f"      real/shuffled        {nr / norms.mean():.4f}"
              f"   (z {zs}, perm p {ps})")
        print(f"\n   direction: does the real gradient point the way that would")
        print(f"              raise MIGRATE_EDGE at high risk?")
        print(f"      cos(g_real,  g_synth) = {c_rs:+.5f}")
        print(f"      cos(g_real_hi, g_synth) = {c_hs:+.5f}")
        print(f"      NULL cos(g_shuffled, g_synth) = {coss.mean():+.5f} "
              f"+/- {coss.std(ddof=1):.5f}   "
              f"|null| 95th pct {np.percentile(np.abs(coss), 95):.5f}")
        print(f"      two-sided permutation p for cos(g_real, g_synth) = "
              f"{p_cos:.4f}")

        out[tag] = dict(
            T=T, start_ticks=starts, truncated=sum(trunc),
            n_decision=int(dec.sum()), n_highrisk=int(hi.sum()),
            highrisk_frac_of_decision=float(hi.sum() / max(dec.sum(), 1)),
            adv_pre_norm=dict(mean=float(adv_raw.mean()),
                              sd=float(adv_raw.std())),
            adv_post_norm_decision=dict(mean=float(adv_real[dec].mean()),
                                        sd=float(adv_real[dec].std())),
            mass_real_L2=mass_real, synth_scale=k_syn, real_hi_scale=k_hi,
            synth_support=dict(n_edge_pos=n_pos, n_stay_neg=n_neg),
            grad_norms=dict(real=nr, synth_coherent=ns, real_highrisk_only=nh,
                            shuffled_null_mean=float(norms.mean()),
                            shuffled_null_sd=float(norms.std(ddof=1)),
                            shuffled_null_min=float(norms.min()),
                            shuffled_null_max=float(norms.max())),
            ratios=dict(real_over_synth=nr / ns,
                        real_over_shuffled=nr / float(norms.mean()),
                        real_hi_over_synth=nh / ns,
                        z_vs_shuffled_null=z_norm,
                        perm_p_norm=p_norm),
            cosines=dict(real_vs_synth=c_rs, real_hi_vs_synth=c_hs,
                         shuffled_vs_synth_mean=float(coss.mean()),
                         shuffled_vs_synth_sd=float(coss.std(ddof=1)),
                         shuffled_abs_p95=float(np.percentile(np.abs(coss), 95)),
                         perm_p_two_sided=p_cos, n_perms=int(coss.size)),
            scale_check=dict(
                # RUNG2_5_DTHETA only ever covered A0 and R2, the two arms
                # Rung 2.5's replica measured. An arm added via --also has no
                # such replica, so this is None rather than a fabricated
                # number; every other field is computed from the arm's own
                # checkpoint and is unaffected.
                dtheta_per_update_rung2_5=RUNG2_5_DTHETA.get(tag),
                note="for reading ||g|| against the movement it actually "
                     "produced in production"),
        )

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY -- at MATCHED advantage mass")
    print("=" * 78)
    print(f"  {'':4s} {'real/synth':>11s} {'real/shuf':>10s} "
          f"{'cos(real,syn)':>14s} {'cos(hi,syn)':>12s} {'null|cos|p95':>13s}")
    for tag in arms:
        o = out[tag]
        print(f"  {tag:4s} {o['ratios']['real_over_synth']:>11.4f} "
              f"{o['ratios']['real_over_shuffled']:>10.4f} "
              f"{o['cosines']['real_vs_synth']:>+14.5f} "
              f"{o['cosines']['real_hi_vs_synth']:>+12.5f} "
              f"{o['cosines']['shuffled_abs_p95']:>13.5f}")
    print("\n  read: real/synth << 1 means magnitude is being lost to")
    print("        cancellation. real/shuf ~ 1 means the real advantage")
    print("        carries no more usable state-action information than a")
    print("        permutation of its own values. cos(real,syn) <= 0 means the")
    print("        real signal does not push toward MIGRATE_EDGE at high risk.")

    blob = dict(
        probe=f"{TAG}_coherence",
        what="is the actor's advantage signal WEAK or INCOHERENT? full-batch "
             "policy-gradient norms and directions under four advantage "
             "vectors of MATCHED L2 mass, at the checkpoint's own parameters",
        why=dict(
            motivation="the plasticity probe moved theta as far with a sparse "
                       "+/-1 synthetic advantage (mean|A| ~ 0.061) as "
                       "production moved it with a dense unit-variance one "
                       "(mean|A| ~ 0.8), which SUGGESTS cancellation; this "
                       "measures it on one buffer instead of inferring it "
                       "across two scripts",
            exactness="at the checkpoint's own parameters new_lp == old_lp so "
                      "ratio == 1 exactly, min(s1,s2) == A, and the PPO actor "
                      "loss reduces exactly to the vanilla policy gradient, "
                      "which is LINEAR in A -- so matched-mass comparison is "
                      "assumption-free",
            entropy_excluded="the entropy term is advantage-independent; "
                             "including it would add an identical vector to "
                             "every condition and inflate every cosine",
        ),
        conditions=dict(
            real="production's compute_gae, normalised over decision entries "
                 "exactly as mappo.py:360-366",
            synth="+1 on MIGRATE_EDGE / -1 on STAY at risk > 0.50, 0 elsewhere, "
                  "then rescaled to real's L2 mass; the perfectly-coherent "
                  "reference direction",
            shuffled="real's own values PERMUTED among decision entries; "
                     "preserves the marginal distribution of A exactly and "
                     "destroys only the state-action correspondence -- the null",
            real_hi="real zeroed outside high-risk entries, rescaled to real's "
                    "L2 mass",
        ),
        not_a_fix=[
            "no optimiser is constructed and no parameter is written",
            "synth presupposes MIGRATE_EDGE is correct at high risk, which a "
            "training run cannot know; it is a reference DIRECTION, not a "
            "proposed target",
            "production code is unmodified",
        ],
        buffer=dict(episodes=args.episodes,
                    starts="training_start_ticks (train.py's own RNG draw)",
                    policy="stochastic, torch.manual_seed(TRAIN_SEED + j)"),
        n_perms=args.perms, per_arm=out,
    )
    p = OUT_DIR / f"{TAG}_coherence_{args.tag}.json"
    p.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n  wrote {p}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
