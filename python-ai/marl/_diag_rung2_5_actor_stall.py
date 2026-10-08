#!/usr/bin/env python
"""
SPRINT 7 RUNG 2.5 -- items G and H.

G. ACTOR STALL ANALYSIS -- attribution only. The R2 actor is NOT changed.
   Explains why clip_frac and k3 went to exactly 0, why entropy fell
   0.860 -> 0.138, and whether that is convergence, policy collapse, or
   insufficient gradient signal.

H. R2 REPLICA FIX -- an R2-AWARE diagnostic replica, so that the actor-side
   freeze is established independently of the stale Rung-0 replica.

   The old probe reported replica ok=False with max|d| 7.46. TWO independent
   causes, both in Rung-0 diagnostic code and NEITHER in production:

     1. _diag_rung0.py:1009 hardcodes `agent.compute_gae(buf)` for the critic
        target, so the replica regresses the critic on the LAMBDA-return even
        when the checkpoint was trained with critic_target="mc".

     2. _diag_rung0.py:276 calls `buf.set_bootstrap(buf.ptr - 1, boot)` without
        the `truncated` argument, which defaults to True. Every episode in a
        replica buffer is therefore flagged as a time-limit truncation even
        when it ended naturally. compute_gae ignores that flag, so the old
        probe was unaffected; compute_mc_returns READS it, so an MC replica
        built on top of it would bootstrap on every episode.

   Both are fixed here, locally. _diag_rung0.py is NOT modified -- Rung 0's
   published artifacts must stay reproducible.

Runs NO PPO training in the production sense: every update is taken on a
deepcopy of the agent, nothing is ever saved, and no production file changes.

Writes SPRINT_7_RUNG2_5_actor_stall.json.
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[0]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from marl.config import ACTION_NAMES, ACTION_MIGRATE_EDGE, ACTION_STAY  # noqa: E402
from marl.mappo import RolloutBuffer, masked_dist                    # noqa: E402
from marl._diag_rung0 import (                                       # noqa: E402
    load_agent_and_cfg, _replay, _values, OUT_DIR, LEGACY_HIGH_RISK,
)
from marl._diag_rung2_5_targets import (                             # noqa: E402
    build_env, training_start_ticks, dist, TRAIN_SEED,
)

TAG = "SPRINT_7_RUNG2_5"
R2_MODEL = OUT_DIR / "mappo_R2_mc_target.pth"
A0_MODEL = OUT_DIR / "mappo_A0_cpu_repro.pth"
HI_RISK = 0.50


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(R2_MODEL))
    p.add_argument("--compare", default=str(A0_MODEL))
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=8,
                   help="episodes per rollout buffer (train.py used 8)")
    p.add_argument("--window", default="train", choices=("train", "eval"))
    return p.parse_args(argv)


# ======================================================================
# H: an R2-aware buffer + replica update
# ======================================================================

def build_buffer_r2(agent, recs, trunc_flags):
    """
    Like _diag_rung0.build_buffer, but passes the REAL truncation flag to
    set_bootstrap instead of letting it default to True.
    """
    T = sum(len(r["act"]) for r in recs)
    buf = RolloutBuffer(T, agent.n_agents, agent.obs_dim, agent.state_dim)
    for r, tr in zip(recs, trunc_flags):
        v = _values(agent, r["state"])
        boot = agent.value(r["boot_state"])
        for t in range(len(r["act"])):
            buf.add(r["obs"][t], r["state"][t], r["act"][t], r["logp"][t],
                    v[t], r["rew"][t], r["mask"][t], r["cont"][t])
        buf.set_bootstrap(buf.ptr - 1, boot, bool(tr))
    return buf


def replica_update_r2(agent, buf, step_optimisers=True):
    """
    Replica of MAPPO.update that HONOURS cfg.critic_target, so an MC-trained
    checkpoint is replicated with the MC target. Additive diagnostics only:
    k3, pre-clip gradient norms, per-minibatch traces, and a decomposition of
    the actor gradient into its policy-gradient and entropy-bonus parts.
    """
    cfg = agent.cfg
    T = len(buf)
    adv_np, ret_np = agent.compute_gae(buf)
    used = "lambda"
    if getattr(cfg, "critic_target", "lambda") == "mc":
        ret_np = agent.compute_mc_returns(buf)
        used = "mc"

    dev = agent.device
    obs = torch.as_tensor(buf.obs[:T], device=dev)
    state = torch.as_tensor(buf.state[:T], device=dev)
    act = torch.as_tensor(buf.act[:T], device=dev)
    old_lp = torch.as_tensor(buf.logp[:T], device=dev)
    old_v = torch.as_tensor(buf.val[:T], device=dev)
    mask = torch.as_tensor(buf.mask[:T], device=dev)
    dec = torch.as_tensor(buf.decision[:T], device=dev)
    adv = torch.as_tensor(adv_np, device=dev)
    ret = torch.as_tensor(ret_np, device=dev)

    if cfg.normalise_advantages:
        sel = dec > 0.5
        if sel.any():
            a = adv[sel]
            adv = (adv - a.mean()) / (a.std() + 1e-8)

    n_mb = max(1, cfg.minibatches)
    mb_size = max(1, T // n_mb)
    rng = np.random.default_rng(0)
    stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "clip_frac": 0.0, "n": 0}
    trace = []

    def gnorm(params):
        tot = 0.0
        for p in params:
            if p.grad is not None:
                tot += float(p.grad.detach().pow(2).sum())
        return float(np.sqrt(tot))

    for epoch in range(cfg.ppo_epochs):
        order = rng.permutation(T)
        for mb, start in enumerate(range(0, T, mb_size)):
            idx = torch.as_tensor(order[start:start + mb_size],
                                  dtype=torch.long, device=dev)
            if idx.numel() == 0:
                continue
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
            ent_loss = -(ent * d_mb).sum() / denom

            # --- G: decompose the actor gradient BEFORE the real step -----
            agent.opt_actor.zero_grad(set_to_none=True)
            pg.backward(retain_graph=True)
            g_pg = gnorm(agent.actor.parameters())
            agent.opt_actor.zero_grad(set_to_none=True)
            (cfg.entropy_coef * ent_loss).backward(retain_graph=True)
            g_ent = gnorm(agent.actor.parameters())

            actor_loss = pg + cfg.entropy_coef * ent_loss
            agent.opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            g_actor_pre = float(nn.utils.clip_grad_norm_(
                agent.actor.parameters(), cfg.max_grad_norm))
            if step_optimisers:
                agent.opt_actor.step()

            v = agent.critic(state[idx])
            v_clipped = old_v[idx] + torch.clamp(
                v - old_v[idx], -cfg.value_clip_eps, cfg.value_clip_eps)
            vl = torch.max(F.mse_loss(v, ret[idx], reduction="none"),
                           F.mse_loss(v_clipped, ret[idx], reduction="none"))
            critic_loss = cfg.value_coef * vl.mean()
            agent.opt_critic.zero_grad(set_to_none=True)
            critic_loss.backward()
            g_critic_pre = float(nn.utils.clip_grad_norm_(
                agent.critic.parameters(), cfg.max_grad_norm))
            if step_optimisers:
                agent.opt_critic.step()

            with torch.no_grad():
                k1 = ((old_lp[idx] - new_lp) * d_mb).sum() / denom
                lr_ = new_lp - old_lp[idx]
                r_ = torch.exp(lr_)
                k3 = (((r_ - 1.0) - lr_) * d_mb).sum() / denom
                cf = (((ratio - 1.0).abs() > cfg.clip_eps).float()
                      * d_mb).sum() / denom
                rr = ratio[d_mb > 0.5]
                # how much HIGH-RISK EDGE data is in this minibatch?
                rsel = d_mb > 0.5
                hr = rsel & (torch.as_tensor(
                    buf.obs[:T][:, :, 12], device=dev)[idx] > HI_RISK)
                hr_edge = hr & (act[idx] == ACTION_MIGRATE_EDGE)

            stats["actor_loss"] += float(pg.item())
            stats["critic_loss"] += float(critic_loss.item())
            stats["entropy"] += float(-ent_loss.item())
            stats["approx_kl"] += float(k1.item())
            stats["clip_frac"] += float(cf.item())
            stats["n"] += 1
            trace.append(dict(
                epoch=epoch, minibatch=mb, size=int(idx.numel()),
                decision_entries=float(denom.item()),
                policy_loss=float(pg.item()),
                value_loss=float(critic_loss.item()),
                entropy=float(-ent_loss.item()),
                approx_kl_k1=float(k1.item()), kl_k3=float(k3.item()),
                clip_frac=float(cf.item()),
                actor_grad_norm_preclip=g_actor_pre,
                actor_grad_norm_pg_only=g_pg,
                actor_grad_norm_entropy_only=g_ent,
                critic_grad_norm_preclip=g_critic_pre,
                n_highrisk_decision=int(hr.sum().item()),
                n_highrisk_EDGE=int(hr_edge.sum().item()),
                ratio_mean=float(rr.mean().item()) if rr.numel() else float("nan"),
                ratio_min=float(rr.min().item()) if rr.numel() else float("nan"),
                ratio_max=float(rr.max().item()) if rr.numel() else float("nan"),
            ))

    n = max(1, stats.pop("n"))
    out = {k: v / n for k, v in stats.items()}
    out["critic_target_used"] = used
    out["adv_mean"] = float(adv_np.mean())
    out["adv_std"] = float(adv_np.std())
    out["decision_frac"] = float(buf.decision[:T].mean())
    out["kl_k3"] = float(np.mean([t["kl_k3"] for t in trace]))
    out["actor_grad_norm_preclip_mean"] = float(
        np.mean([t["actor_grad_norm_preclip"] for t in trace]))
    out["actor_grad_norm_pg_only_mean"] = float(
        np.mean([t["actor_grad_norm_pg_only"] for t in trace]))
    out["actor_grad_norm_entropy_only_mean"] = float(
        np.mean([t["actor_grad_norm_entropy_only"] for t in trace]))
    out["n_minibatches"] = len(trace)
    out["n_minibatches_zero_highrisk"] = int(
        sum(1 for t in trace if t["n_highrisk_decision"] == 0))
    out["n_minibatches_zero_highrisk_EDGE"] = int(
        sum(1 for t in trace if t["n_highrisk_EDGE"] == 0))
    out["highrisk_per_minibatch"] = dist([t["n_highrisk_decision"] for t in trace])
    out["highrisk_EDGE_per_minibatch"] = dist([t["n_highrisk_EDGE"] for t in trace])
    out["_trace"] = trace
    return out


def param_delta(a_before, a_after):
    """L2 norm of the actor / critic parameter change, and the max |delta|."""
    def collect(m):
        return {k: v.detach().cpu().numpy().copy()
                for k, v in m.state_dict().items()}
    out = {}
    for name, mb, ma in (("actor", a_before.actor, a_after.actor),
                         ("critic", a_before.critic, a_after.critic)):
        b, c = collect(mb), collect(ma)
        sq, mx, npar = 0.0, 0.0, 0
        for k in b:
            d = c[k] - b[k]
            sq += float((d ** 2).sum())
            mx = max(mx, float(np.abs(d).max()) if d.size else 0.0)
            npar += int(d.size)
        out[name] = dict(l2_norm=float(np.sqrt(sq)), max_abs=mx,
                         n_params=npar,
                         rms=float(np.sqrt(sq / max(npar, 1))))
    return out


# ======================================================================
# G: policy saturation at the final checkpoint
# ======================================================================

def saturation_census(agent, cfg, n_eps, label):
    """
    Action-probability geometry at DECISION entries. A saturated softmax has
    max prob -> 1 and per-entry entropy -> 0, which makes d(log pi)/d(theta)
    vanish -- the mechanism that pins ratio at 1, hence clip_frac and k3 at 0.
    """
    env = build_env(cfg)
    starts = training_start_ticks(env, n_eps)
    recs, trunc = [], []
    for j, s in enumerate(starts):
        torch.manual_seed(TRAIN_SEED + j)
        r = _replay(env, agent, s, j, record=True, sample=True)
        recs.append(r["rec"])
        trunc.append(bool(r["n_steps"] >= env.cfg.episode_steps))

    maxp, entr, risks, acts, is_hi = [], [], [], [], []
    for rec in recs:
        obs = torch.as_tensor(np.asarray(rec["obs"]), dtype=torch.float32,
                              device=agent.device)
        msk = torch.as_tensor(np.asarray(rec["mask"]), dtype=torch.float32,
                              device=agent.device)
        rk = np.asarray(rec["risk"])
        ac = np.asarray(rec["act"])
        with torch.no_grad():
            for i in range(agent.n_agents):
                d = masked_dist(agent.actor.logits(i, obs[:, i, :]), msk[:, i, :])
                p = d.probs.cpu().numpy()
                e = d.entropy().cpu().numpy()
                legal = msk[:, i, :].sum(-1).cpu().numpy() > 1.5
                maxp.append(p.max(-1)[legal])
                entr.append(e[legal])
                risks.append(rk[legal, i])
                acts.append(ac[legal, i])
                is_hi.append(rk[legal, i] > HI_RISK)
    maxp = np.concatenate(maxp); entr = np.concatenate(entr)
    risks = np.concatenate(risks); acts = np.concatenate(acts)
    is_hi = np.concatenate(is_hi)

    out = dict(
        label=label, n_decision_entries=int(maxp.size),
        max_prob=dist(maxp), per_entry_entropy=dist(entr),
        frac_maxprob_gt_0_99=float((maxp > 0.99).mean()),
        frac_maxprob_gt_0_999=float((maxp > 0.999).mean()),
        frac_entropy_lt_0_01=float((entr < 0.01).mean()),
        mean_entropy_all=float(entr.mean()),
        n_highrisk_decision=int(is_hi.sum()),
        highrisk=dict(
            max_prob=dist(maxp[is_hi]) if is_hi.any() else dict(n=0),
            per_entry_entropy=dist(entr[is_hi]) if is_hi.any() else dict(n=0),
            EDGE_share=float((acts[is_hi] == ACTION_MIGRATE_EDGE).mean())
            if is_hi.any() else None,
            STAY_share=float((acts[is_hi] == ACTION_STAY).mean())
            if is_hi.any() else None,
            n_EDGE=int((acts[is_hi] == ACTION_MIGRATE_EDGE).sum())
            if is_hi.any() else 0,
        ),
        episode_lengths=dist([len(r["act"]) for r in recs]),
        truncated_episodes=int(sum(trunc)), n_episodes=len(recs),
    )
    return out, recs, trunc


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 RUNG 2.5 -- items G (actor stall) and H (R2-aware replica)")
    print("       attribution only; the R2 actor is NOT changed")
    print("=" * 78)

    blob = dict(probe=f"{TAG}_actor_stall",
                what="attribution of the R2 actor stall, plus an R2-aware "
                     "replica that establishes the freeze independently of "
                     "the stale Rung-0 lambda-target replica",
                old_replica_defects=[
                    "_diag_rung0.py:1009 hardcodes compute_gae, so the critic "
                    "target is the lambda-return even for an MC checkpoint",
                    "_diag_rung0.py:276 calls set_bootstrap without the "
                    "truncated argument, which defaults to True, flagging "
                    "every replica episode as a time-limit truncation",
                ],
                models={}, saturation={}, replica={}, param_delta={})

    for tag, path in (("R2", args.model), ("A0", args.compare)):
        if not Path(path).exists():
            print(f"  [{tag}] MISSING: {path} -- skipped")
            continue
        agent, extra, cfg = load_agent_and_cfg(path, args.device, args.window)
        blob["models"][tag] = dict(
            path=str(path), critic_target=getattr(agent.cfg, "critic_target",
                                                  "lambda"),
            entropy_coef=float(agent.cfg.entropy_coef),
            clip_eps=float(agent.cfg.clip_eps),
            minibatches=int(agent.cfg.minibatches),
            ppo_epochs=int(agent.cfg.ppo_epochs))

        print(f"\n===== {tag}: {Path(path).name} "
              f"(critic_target={blob['models'][tag]['critic_target']}) =====")
        print("-- G: policy saturation at decision entries ------------------")
        sat, recs, trunc = saturation_census(agent, cfg, args.episodes, tag)
        blob["saturation"][tag] = sat
        print(f"  decision entries          : {sat['n_decision_entries']}")
        print(f"  mean per-entry entropy    : {sat['mean_entropy_all']:.4f} "
              f"(max possible ln4 = 1.3863)")
        print(f"  max prob   mean/p50/p95   : {sat['max_prob']['mean']:.4f} / "
              f"{sat['max_prob']['p50']:.4f} / {sat['max_prob']['p95']:.4f}")
        print(f"  frac max prob > 0.99      : {sat['frac_maxprob_gt_0_99']:.4f}")
        print(f"  frac max prob > 0.999     : {sat['frac_maxprob_gt_0_999']:.4f}")
        print(f"  frac entry entropy < 0.01 : {sat['frac_entropy_lt_0_01']:.4f}")
        print(f"  high-risk decision entries: {sat['n_highrisk_decision']} "
              f"(EDGE n={sat['highrisk']['n_EDGE']}, "
              f"share={sat['highrisk']['EDGE_share']})")
        print(f"  episodes truncated        : {sat['truncated_episodes']}"
              f"/{sat['n_episodes']}")

        print("-- H: R2-aware replica update on a deepcopy -----------------")
        buf = build_buffer_r2(agent, recs, trunc)
        probe = copy.deepcopy(agent)
        before = copy.deepcopy(agent)
        res = replica_update_r2(probe, buf, step_optimisers=True)
        pd = param_delta(before, probe)
        blob["replica"][tag] = {k: v for k, v in res.items() if k != "_trace"}
        blob["replica"][tag]["_trace"] = res["_trace"]
        blob["param_delta"][tag] = pd
        print(f"  critic target used        : {res['critic_target_used']}")
        print(f"  entropy                   : {res['entropy']:.6f}")
        print(f"  approx_kl (k1)            : {res['approx_kl']:+.8f}")
        print(f"  kl_k3                     : {res['kl_k3']:.8f}")
        print(f"  clip_frac                 : {res['clip_frac']:.6f}")
        print(f"  adv_std                   : {res['adv_std']:.6f}")
        print(f"  actor grad norm  total    : "
              f"{res['actor_grad_norm_preclip_mean']:.6e}")
        print(f"                   pg only  : "
              f"{res['actor_grad_norm_pg_only_mean']:.6e}")
        print(f"              entropy only  : "
              f"{res['actor_grad_norm_entropy_only_mean']:.6e}")
        print(f"  minibatches               : {res['n_minibatches']}  "
              f"zero high-risk: {res['n_minibatches_zero_highrisk']}  "
              f"zero high-risk EDGE: "
              f"{res['n_minibatches_zero_highrisk_EDGE']}")
        print(f"  high-risk EDGE / minibatch: "
              f"mean {res['highrisk_EDGE_per_minibatch']['mean']:.2f} "
              f"min {res['highrisk_EDGE_per_minibatch']['min']:.0f} "
              f"max {res['highrisk_EDGE_per_minibatch']['max']:.0f}")
        print(f"  ACTOR param delta  L2     : {pd['actor']['l2_norm']:.6e} "
              f"(rms {pd['actor']['rms']:.3e}, max|d| {pd['actor']['max_abs']:.3e})")
        print(f"  CRITIC param delta L2     : {pd['critic']['l2_norm']:.6e} "
              f"(rms {pd['critic']['rms']:.3e}, max|d| {pd['critic']['max_abs']:.3e})")

    path = OUT_DIR / f"{TAG}_actor_stall.json"
    with open(path, "w") as f:
        json.dump(blob, f, indent=2, default=float)
    print(f"\n  wrote {path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
