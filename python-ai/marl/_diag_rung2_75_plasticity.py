#!/usr/bin/env python
"""
SPRINT 7 RUNG 2.75 -- item B, dynamic half: is the actor still PLASTIC?

The brief asks, verbatim, "whether the policy is actually capable of moving away
from STAY when the advantage signal is favorable". No existing artifact answers
that: every recorded gradient was driven by the critic's OWN advantage, whose
sign at high risk is at or below chance (item C). So a favourable signal has
never actually been presented to this actor, and its absence of movement is
unexplained -- it could be an incapable policy or an uninformative signal.

THIS PROBE SEPARATES THE TWO. It presents a KNOWN-FAVOURABLE advantage and
measures the response.

  A_synth(entry) = +M  if risk > 0.50 and the taken action was MIGRATE_EDGE
                   -M  if risk > 0.50 and the taken action was STAY
                    0  otherwise

  That is the *cleanest possible* learning signal for the behaviour we want:
  reward the migrations that were taken at high risk, penalise the stays. It is
  injected in place of the critic's advantage and nothing else changes -- same
  clipped surrogate, same entropy bonus, same clip_eps, same max_grad_norm, same
  optimisers, same minibatching, same ppo_epochs. Then pi(.|s) is re-measured at
  the SAME states (the buffer's own stored obs and masks, not a fresh rollout),
  so the before/after difference is a pure policy-response measurement.

  THIS IS NOT A PROPOSED FIX AND IS NOT AN INTERVENTION. A_synth is built from
  labels that a training run does not have (it needs to know that EDGE is the
  right answer). It exists only to place an upper bound on what the actor COULD
  do if the signal were perfect. Every step is taken on a deepcopy that is
  never saved; production code is untouched; nothing is trained.

======================================================================
CORRECTION TO THIS PROBE'S OWN FIRST RUN -- A FLAW I FOUND MYSELF
======================================================================

The first version of this file loaded the checkpoint with
`load_agent_and_cfg(..., "train")`, which RESTORES THE SAVED OPTIMISER STATE,
including its learning rate. `train.py` uses `anneal_lr = True`, so at the end
of a 75-update run the actor lr has decayed linearly to

    lr_actor_final = 7e-4 * (1/75) = 9.333333333333316e-06

which is exactly what the checkpoints carry (verified: `opt_actor` state length
60, `agent.cfg.lr_actor = 0.0007`). The probe therefore took all of its updates
at 1/75th of the learning rate that training actually used, and its headline
conclusion -- "the actor barely moves even under a perfect signal" -- was
largely an artifact of the lr schedule, not a statement about the actor's
capacity. It did NOT answer the brief's capability question.

Per the brief's rule ("If an existing diagnostic is flawed, document the flaw
and create the smallest corrected diagnostic rather than silently changing
methodology"), the original condition is REPRODUCED here as a named cell rather
than deleted, and two corrected cells are added alongside it:

  anneal_end     lr as loaded from the checkpoint (9.33e-06), Adam state kept.
                 Reproduces the first run EXACTLY, for the record.
  lr_full        lr = cfg.lr_actor = 7e-4, i.e. the START-of-training rate,
                 Adam state kept. This is the cell that answers the brief:
                 what the actor can do at the learning rate the run used when
                 clip_frac was still healthy (0.108 A0 / 0.163 R2).
  lr_full_fresh  lr = 7e-4 with Adam's exponential moving averages CLEARED, so
                 a stale second-moment estimate accumulated over 1500 annealed
                 steps cannot suppress the step size. Controls for optimiser
                 state as distinct from policy geometry.

WHAT ELSE THE SAME FLAW TOUCHES, recorded so it is not re-derived later:
  - `_diag_rung2_75_mbtail.py`: its tail-vs-substantive gradient/step RATIOS
    survive, because lr is matched WITHIN each comparison. Its ABSOLUTE d(theta)
    magnitudes inherit the caveat.
  - the Rung 2.5 actor-stall replica: its gradient norms survive (they are
    computed before the optimiser step, so lr-independent), but its
    `param_delta` and `clip_frac = 0.0` describe only the annealed END state of
    training, not training.

READING THE RESULT.

  If pi(EDGE) at high risk rises substantially under A_synth, the actor is
  plastic and the stall is a SIGNAL problem -> the offset finding (item C) is
  the live hypothesis.
  If pi(EDGE) barely moves even under a perfect signal AT THE FULL LEARNING
  RATE, the actor is functionally frozen and the stall is a PLASTICITY problem
  -> saturation / optimiser state is the live hypothesis.
  Both arms get the identical probe at identical magnitudes and identical cells,
  so the answer is a matched contrast, not an absolute.

  M is deliberately NOT renormalised. Production would standardise the advantage
  over decision entries, which would rescale a sparse +/-M pattern by a factor
  that depends on how many EDGE entries each arm happens to have -- i.e. the
  normaliser would confound the very contrast being measured. Fixing M instead
  makes the response per unit of advantage comparable across arms. M = 1.0 is
  the headline; M = 3.0 (~= the real post-normalisation adv_std of 2.94 / 2.65)
  checks that the response is monotone rather than saturated at M = 1.

ALSO REPORTED, and requiring no optimiser at all:

  attenuation = ||e_a - pi(.|s)||_2, the exact logit-space norm of
  d log pi(a|s)/d logits. It is the state-dependent factor multiplying A_hat in
  the policy gradient, and it goes to 0 as pi(a|s) -> 1. Measured per arm at
  high-risk entries, it is the analytic version of the same question.

Writes SPRINT_7_RUNG2_75_plasticity.json.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
from marl.mappo import masked_dist, MASK_FILL                    # noqa: E402

TAG = "SPRINT_7_RUNG2_75"
HI_RISK = 0.50
ACTION_NAMES = ["STAY", "MIGRATE_EDGE", "MIGRATE_CLOUD", "PREEMPT_REROUTE"]
MODELS = {"A0": "mappo_A0_cpu_repro.pth", "R2": "mappo_R2_mc_target.pth"}

# lr cell -> (lr source, reset Adam moving averages?)
CELLS = {
    "anneal_end":    dict(lr="as_loaded",  reset_adam=False,
                          note="reproduces this probe's flawed first run "
                               "EXACTLY: the checkpoint's fully-annealed lr "
                               "(7e-4 * 1/75 = 9.33e-06), Adam state kept"),
    "lr_full":       dict(lr="cfg",        reset_adam=False,
                          note="cfg.lr_actor = 7e-4, the rate training used "
                               "while clip_frac was still healthy; Adam state "
                               "kept. THIS is the cell that answers the "
                               "brief's capability question."),
    "lr_full_fresh": dict(lr="cfg",        reset_adam=True,
                          note="7e-4 with Adam's moving averages cleared, so a "
                               "stale second moment from 1500 annealed steps "
                               "cannot suppress the step"),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--updates", type=int, default=10,
                   help="consecutive synthetic-advantage updates at M=1")
    p.add_argument("--probe-at", default="1,2,3,5,10",
                   help="measure pi after these update counts")
    p.add_argument("--cells", default="anneal_end,lr_full,lr_full_fresh",
                   help="comma-separated subset of " + ",".join(CELLS))
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


# ----------------------------------------------------------------------

def build(agent, cfg, n_eps):
    """The real 8-episode stochastic buffer, plus the risk of every entry."""
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


def opt_lrs(opt):
    return [float(g["lr"]) for g in opt.param_groups]


def adam_state_size(opt):
    return int(sum(1 for _ in opt.state.values()))


def apply_cell(work, cell):
    """
    Set the deepcopy's actor lr and optionally clear Adam's moving averages.
    Returns what was actually applied, so the artifact records it rather than
    asserting it.
    """
    spec = CELLS[cell]
    before = opt_lrs(work.opt_actor)
    if spec["lr"] == "cfg":
        for g in work.opt_actor.param_groups:
            g["lr"] = float(work.cfg.lr_actor)
    n_state_before = adam_state_size(work.opt_actor)
    if spec["reset_adam"]:
        work.opt_actor.state.clear()
    return dict(cell=cell, lr_before=before, lr_after=opt_lrs(work.opt_actor),
                adam_state_entries_before=n_state_before,
                adam_state_entries_after=adam_state_size(work.opt_actor),
                reset_adam=spec["reset_adam"], note=spec["note"])


def policy_probs(agent, obs, mask):
    """pi(.|s) for every (t, i), using production's forward and MASK_FILL."""
    with torch.no_grad():
        T = obs.shape[0]
        P = np.empty((T, agent.n_agents, 4), np.float64)
        for i in range(agent.n_agents):
            lg = agent.actor.logits(i, obs[:, i]).masked_fill(
                ~mask[:, i].bool(), MASK_FILL)
            P[:, i] = torch.softmax(lg, dim=-1).cpu().numpy()
    return P


def census(P, act, sel):
    """Policy census over a boolean (T, n_agents) selection of entries."""
    if sel.sum() == 0:
        return dict(n=0)
    Q = P[sel]                      # (n, 4)
    a = act[sel]
    e = -(Q * np.log(np.clip(Q, 1e-12, None))).sum(-1)
    arg = Q.argmax(-1)
    return dict(
        n=int(sel.sum()),
        p_edge_mean=float(Q[:, ACTION_MIGRATE_EDGE].mean()),
        p_edge_p95=float(np.percentile(Q[:, ACTION_MIGRATE_EDGE], 95)),
        p_stay_mean=float(Q[:, ACTION_STAY].mean()),
        maxp_mean=float(Q.max(-1).mean()),
        frac_maxp_gt_099=float((Q.max(-1) > 0.99).mean()),
        entropy_mean=float(e.mean()),
        argmax_counts={ACTION_NAMES[k]: int((arg == k).sum()) for k in range(4)},
        frac_argmax_edge=float((arg == ACTION_MIGRATE_EDGE).mean()),
        taken_edge_share=float((a == ACTION_MIGRATE_EDGE).mean()),
        attenuation_mean=float(np.mean([
            np.linalg.norm(np.eye(4)[int(aa)] - qq) for qq, aa in zip(Q, a)])),
    )


def synth_update(agent, buf, risk, M, thr=HI_RISK):
    """
    One production-shaped PPO update with the CRITIC'S ADVANTAGE REPLACED by the
    synthetic one. Everything else is mappo.MAPPO.update verbatim in structure:
    same clipped surrogate, entropy bonus, minibatch loop (tail included), grad
    clipping and optimiser. The critic is deliberately NOT updated -- the
    question is about the actor.
    """
    cfg = agent.cfg
    T = len(buf)
    dev = agent.device
    obs = torch.as_tensor(buf.obs[:T], device=dev)
    act = torch.as_tensor(buf.act[:T], device=dev)
    old_lp = torch.as_tensor(buf.logp[:T], device=dev)
    mask = torch.as_tensor(buf.mask[:T], device=dev)
    dec = torch.as_tensor(buf.decision[:T], device=dev)

    a_np = np.asarray(buf.act[:T])
    hi = np.asarray(risk[:T]) > thr
    A = np.zeros_like(a_np, np.float32)
    A[hi & (a_np == ACTION_MIGRATE_EDGE)] = +M
    A[hi & (a_np == ACTION_STAY)] = -M
    adv = torch.as_tensor(A, device=dev)

    n_mb = max(1, cfg.minibatches)
    mb_size = max(1, T // n_mb)
    rng = np.random.default_rng(0)
    clipped, steps, gsum, rmax = 0, 0, 0.0, 1.0
    for _ in range(cfg.ppo_epochs):
        order = rng.permutation(T)
        for start in range(0, T, mb_size):
            idx_np = order[start:start + mb_size]
            if idx_np.size == 0:
                continue
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=dev)
            new_lp, ent = [], []
            for i in range(agent.n_agents):
                d = masked_dist(agent.actor.logits(i, obs[idx, i, :]),
                                mask[idx, i, :])
                new_lp.append(d.log_prob(act[idx, i]))
                ent.append(d.entropy())
            new_lp = torch.stack(new_lp, dim=1)
            ent = torch.stack(ent, dim=1)
            d_mb = dec[idx]
            denom = d_mb.sum().clamp(min=1.0)
            ratio = torch.exp(new_lp - old_lp[idx])
            s1 = ratio * adv[idx]
            s2 = torch.clamp(ratio, 1.0 - cfg.clip_eps,
                             1.0 + cfg.clip_eps) * adv[idx]
            pg = -(torch.min(s1, s2) * d_mb).sum() / denom
            loss = pg + cfg.entropy_coef * (-(ent * d_mb).sum() / denom)
            agent.opt_actor.zero_grad(set_to_none=True)
            loss.backward()
            g = float(np.sqrt(sum(float(p.grad.pow(2).sum())
                                  for p in agent.actor.parameters()
                                  if p.grad is not None)))
            nn.utils.clip_grad_norm_(agent.actor.parameters(),
                                     cfg.max_grad_norm)
            agent.opt_actor.step()
            with torch.no_grad():
                sel = d_mb > 0.5
                cf = float(((ratio - 1).abs() > cfg.clip_eps)[sel]
                           .float().mean())
                if sel.any():
                    rmax = max(rmax, float(ratio[sel].max()))
            clipped += cf; gsum += g; steps += 1
    return dict(n_steps=steps, mean_grad_norm_preclip=gsum / max(steps, 1),
                mean_clip_frac=clipped / max(steps, 1), ratio_max=rmax,
                n_pos=int((A > 0).sum()), n_neg=int((A < 0).sum()),
                n_zero=int((A == 0).sum()))


def pdelta(a, b):
    return float(np.sqrt(sum(float((x - y).pow(2).sum())
                             for x, y in zip(a.actor.parameters(),
                                             b.actor.parameters()))))


def main(argv=None):
    args = parse_args(argv)
    probe_at = sorted({int(x) for x in args.probe_at.split(",")})
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    bad = [c for c in cells if c not in CELLS]
    if bad:
        raise SystemExit(f"unknown cell(s) {bad}; choose from {list(CELLS)}")
    print("=" * 78)
    print("SPRINT 7 RUNG 2.75 -- item B dynamic: actor PLASTICITY under a")
    print("        KNOWN-FAVOURABLE synthetic advantage (diagnostic upper")
    print("        bound; deepcopy only, never saved, not a proposed fix)")
    print("=" * 78)
    print("  CORRECTED: the first run of this probe used the checkpoint's")
    print("  fully-annealed lr (9.33e-06 = 7e-4/75) because load_agent_and_cfg")
    print("  restores optimiser state. That condition is reproduced below as")
    print("  cell 'anneal_end'; cells 'lr_full' and 'lr_full_fresh' answer the")
    print("  brief's capability question at the training learning rate.")
    print(f"  cells: {', '.join(cells)}")

    out = {}
    for tag, fn in MODELS.items():
        print(f"\n" + "=" * 78)
        print(f"-- {tag} ({fn}) " + "-" * (56 - len(tag) - len(fn)))
        agent, _, cfg = load_agent_and_cfg(str(OUT_DIR / fn), args.device,
                                           "train")
        buf, risk, starts, trunc = build(agent, cfg, args.episodes)
        T = len(buf)
        dev = agent.device
        obs = torch.as_tensor(buf.obs[:T], device=dev)
        mask = torch.as_tensor(buf.mask[:T], device=dev)
        act = np.asarray(buf.act[:T])
        dec = np.asarray(buf.decision[:T]) > 0.5
        hi = (np.asarray(risk[:T]) > HI_RISK) & dec
        lr_loaded = opt_lrs(agent.opt_actor)
        print(f"   T={T}  decision entries={int(dec.sum())}  "
              f"high-risk={int(hi.sum())}  truncated={sum(trunc)}/{len(trunc)}")
        print(f"   lr as loaded from checkpoint : {lr_loaded[0]:.6e}  "
              f"({len(lr_loaded)} param groups)")
        print(f"   cfg.lr_actor (training start): "
              f"{float(agent.cfg.lr_actor):.6e}   ratio "
              f"{lr_loaded[0] / float(agent.cfg.lr_actor):.6f}")
        print(f"   Adam state entries           : "
              f"{adam_state_size(agent.opt_actor)}")

        P0 = policy_probs(agent, obs, mask)
        base = census(P0, act, hi)
        print(f"   BEFORE  p_edge {base['p_edge_mean']:.4f}  "
              f"p_stay {base['p_stay_mean']:.4f}  "
              f"maxp {base['maxp_mean']:.4f}  >.99 {base['frac_maxp_gt_099']:.4f}"
              f"  ent {base['entropy_mean']:.4f}  "
              f"argmaxE {base['frac_argmax_edge']:.4f}  "
              f"atten {base['attenuation_mean']:.4f}")
        print(f"           argmax {base['argmax_counts']}")

        arm = dict(T=T, start_ticks=starts, truncated=sum(trunc),
                   n_decision=int(dec.sum()), n_highrisk=int(hi.sum()),
                   lr_as_loaded=lr_loaded[0],
                   lr_cfg_training_start=float(agent.cfg.lr_actor),
                   lr_anneal_ratio=lr_loaded[0] / float(agent.cfg.lr_actor),
                   adam_state_entries=adam_state_size(agent.opt_actor),
                   before=base, before_all_decision=census(P0, act, dec),
                   cells={})

        for cell in cells:
            print(f"\n   [cell {cell}] {CELLS[cell]['note']}")
            cell_out = dict(applied=None, by_M={})
            for M in (1.0, 3.0):
                n_up = args.updates if M == 1.0 else 1
                work = copy.deepcopy(agent)
                applied = apply_cell(work, cell)
                cell_out["applied"] = applied
                if M == 1.0:
                    print(f"      lr {applied['lr_before'][0]:.6e} -> "
                          f"{applied['lr_after'][0]:.6e}   adam entries "
                          f"{applied['adam_state_entries_before']} -> "
                          f"{applied['adam_state_entries_after']}")
                traj = []
                for u in range(1, n_up + 1):
                    st = synth_update(work, buf, risk, M)
                    if u in probe_at or u == n_up:
                        P = policy_probs(work, obs, mask)
                        c = census(P, act, hi)
                        traj.append(dict(
                            update=u, M=M, step_stats=st, census=c,
                            param_delta_from_start=pdelta(agent, work),
                            d_p_edge=c["p_edge_mean"] - base["p_edge_mean"],
                            d_p_stay=c["p_stay_mean"] - base["p_stay_mean"],
                            d_entropy=c["entropy_mean"] - base["entropy_mean"],
                            d_frac_argmax_edge=c["frac_argmax_edge"]
                            - base["frac_argmax_edge"]))
                        print(f"      M={M:.0f} after {u:2d} update(s): "
                              f"p_edge {c['p_edge_mean']:.4f} "
                              f"({traj[-1]['d_p_edge']:+.4f})  "
                              f"p_stay {c['p_stay_mean']:.4f} "
                              f"({traj[-1]['d_p_stay']:+.4f})  "
                              f"ent {c['entropy_mean']:.4f}  "
                              f"argmaxE {c['frac_argmax_edge']:.4f} "
                              f"({traj[-1]['d_frac_argmax_edge']:+.4f})  "
                              f"|dtheta| {traj[-1]['param_delta_from_start']:.4e}"
                              f"  clip {st['mean_clip_frac']:.3f}  rmax "
                              f"{st['ratio_max']:.4f}")
                cell_out["by_M"][f"M{M:g}"] = dict(
                    n_updates=n_up, trajectory=traj,
                    n_pos_entries=traj[0]["step_stats"]["n_pos"],
                    n_neg_entries=traj[0]["step_stats"]["n_neg"])
            arm["cells"][cell] = cell_out
        out[tag] = arm

    # ---------------- matched contrast, per cell ----------------
    print("\n" + "=" * 78)
    print("MATCHED CONTRAST -- response per unit of PERFECT advantage,")
    print("                    by learning-rate cell (M=1, 10 updates)")
    print("=" * 78)
    print(f"  {'cell':>14s} {'arm':>4s} {'lr':>10s} {'p_edge before':>14s} "
          f"{'after':>9s} {'delta':>9s} {'argmaxE':>16s} {'|dtheta|':>10s}")
    contrast = {}
    for cell in cells:
        contrast[cell] = {}
        for tag in MODELS:
            a = out[tag]
            cc = a["cells"][cell]
            last = cc["by_M"]["M1"]["trajectory"][-1]
            b0 = a["before"]["p_edge_mean"]
            b1 = last["census"]["p_edge_mean"]
            contrast[cell][tag] = dict(
                lr=cc["applied"]["lr_after"][0],
                p_edge_before=b0, p_edge_after=b1, delta=b1 - b0,
                fold=float(b1 / b0) if b0 > 0 else None,
                attenuation_before=a["before"]["attenuation_mean"],
                param_delta=last["param_delta_from_start"],
                frac_argmax_edge_before=a["before"]["frac_argmax_edge"],
                frac_argmax_edge_after=last["census"]["frac_argmax_edge"],
                frac_maxp_gt_099_before=a["before"]["frac_maxp_gt_099"],
                frac_maxp_gt_099_after=last["census"]["frac_maxp_gt_099"],
                mean_clip_frac_last=last["step_stats"]["mean_clip_frac"],
                ratio_max_last=last["step_stats"]["ratio_max"],
            )
            c = contrast[cell][tag]
            print(f"  {cell:>14s} {tag:>4s} {c['lr']:>10.3e} {b0:>14.4f} "
                  f"{b1:>9.4f} {c['delta']:>+9.4f} "
                  f"{c['frac_argmax_edge_before']:>7.4f}->"
                  f"{c['frac_argmax_edge_after']:<8.4f} "
                  f"{c['param_delta']:>10.3e}")

    print("\n  CAPABILITY VERDICT (cell lr_full, the one that answers the brief)")
    if "lr_full" in contrast:
        for tag in MODELS:
            c = contrast["lr_full"][tag]
            ae = contrast.get("anneal_end", {}).get(tag)
            gain = (f"  ({c['delta'] / ae['delta']:.1f}x the annealed cell)"
                    if ae and abs(ae["delta"]) > 1e-12 else "")
            print(f"    {tag}: pi(EDGE|high risk) "
                  f"{c['p_edge_before']:.4f} -> {c['p_edge_after']:.4f} "
                  f"({c['delta']:+.4f}){gain}")
            print(f"        argmax==EDGE {c['frac_argmax_edge_before']:.4f} -> "
                  f"{c['frac_argmax_edge_after']:.4f};  saturated >.99 "
                  f"{c['frac_maxp_gt_099_before']:.4f} -> "
                  f"{c['frac_maxp_gt_099_after']:.4f};  clip_frac "
                  f"{c['mean_clip_frac_last']:.3f}, ratio_max "
                  f"{c['ratio_max_last']:.4f}")

    blob = dict(
        probe=f"{TAG}_plasticity",
        what="can the actor move away from STAY at high risk when the advantage "
             "signal is KNOWN-FAVOURABLE? diagnostic upper bound on actor "
             "plasticity; deepcopy only, never saved; NOT a proposed fix",
        self_caught_flaw_in_first_run=dict(
            what="load_agent_and_cfg restores the saved optimiser state, so the "
                 "first run of this probe took every update at the checkpoint's "
                 "FULLY-ANNEALED actor lr of 9.333333333333316e-06 = 7e-4 * "
                 "1/75, i.e. 1/75th of the rate training used",
            consequence="its headline ('the actor barely moves even under a "
                        "perfect signal') was largely an artifact of the lr "
                        "schedule and did NOT answer the brief's capability "
                        "question",
            handling="the flawed condition is REPRODUCED as cell 'anneal_end' "
                     "rather than deleted; corrected cells 'lr_full' and "
                     "'lr_full_fresh' are added alongside, per the brief's rule "
                     "on flawed diagnostics",
            what_else_it_touches=[
                "_diag_rung2_75_mbtail.py: tail-vs-substantive RATIOS survive "
                "(lr matched within each comparison); absolute d(theta) does not",
                "Rung 2.5 actor-stall replica: gradient norms survive "
                "(pre-step, lr-independent); param_delta and clip_frac = 0.0 "
                "describe only the annealed END state of training",
            ],
        ),
        cells=CELLS,
        not_a_fix=[
            "A_synth is built from labels a training run does not possess (it "
            "presupposes MIGRATE_EDGE is correct at high risk), so it cannot be "
            "an intervention -- only an upper bound on achievable movement",
            "the critic is not updated and nothing is written to disk",
            "production code is unmodified; this file only reads checkpoints",
            "changing lr HERE is a diagnostic condition on a deepcopy, not a "
            "proposed hyperparameter change",
        ],
        synthetic_advantage=dict(
            rule="+M if risk>0.50 and taken action == MIGRATE_EDGE; "
                 "-M if risk>0.50 and taken action == STAY; 0 otherwise",
            not_renormalised="production standardises the advantage over "
                             "decision entries; a sparse +/-M pattern would be "
                             "rescaled by an arm-dependent factor, confounding "
                             "the contrast. M is fixed instead so the response "
                             "is per unit of advantage.",
            magnitudes=[1.0, 3.0],
            reference="real post-normalisation adv_std was 2.94 (A0) / 2.65 "
                      "(R2) in SPRINT_7_RUNG2_5_actor_stall.json",
        ),
        attenuation_definition="||e_a - pi(.|s)||_2 = exact logit-space norm of "
                               "d log pi(a|s)/d logits; the state factor "
                               "multiplying A_hat in the policy gradient",
        buffer=dict(episodes=args.episodes,
                    starts="training_start_ticks (train.py's own RNG draw)",
                    policy="stochastic, torch.manual_seed(TRAIN_SEED + j)"),
        per_arm=out, contrast_by_cell=contrast,
    )
    p = OUT_DIR / f"{TAG}_plasticity_{args.tag}.json"
    p.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n  wrote {p}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
