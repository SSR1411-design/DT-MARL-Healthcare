#!/usr/bin/env python
"""
SPRINT 7 RUNG 2.75 -- item D: the minibatch tail, examined as a CANDIDATE only.

NOT FIXED HERE. Production code is untouched. Nothing is trained. The optimiser
steps taken below are on a DEEPCOPY of the checkpoint and are never saved.

THE DEFECT, from the source (mappo.py:368-377):

    n_mb    = max(1, self.cfg.minibatches)      # 4
    mb_size = max(1, T // n_mb)                 # floor division
    for start in range(0, T, mb_size):          # <- not range(n_mb)
        idx = order[start : start + mb_size]

  Write T = 4q + r with q = T//4 and 0 <= r < 4. The starts are 0, q, 2q, 3q,
  and -- whenever r > 0 -- 4q as well, because 4q < T. The next start 5q
  exceeds T for any q >= 4. So the loop yields FIVE chunks of sizes

      q, q, q, q, r        with r = T mod 4  in  {1, 2, 3}

  i.e. a fifth, degenerate minibatch of ONE TO THREE TIMESTEPS. It is not
  skipped (`if idx.numel() == 0: continue` only catches r == 0, where the chunk
  does not exist anyway), and it is not down-weighted: the losses are MEANS
  over each chunk's own decision entries,

      pg = -(min(s1, s2) * d_mb).sum() / d_mb.sum().clamp(min=1)

  so a 1-timestep chunk produces a gradient of the same nominal SCALE as a
  q-timestep chunk, is clipped to the same max_grad_norm, and receives its own
  full Adam step. With ppo_epochs = 4 that is 4 of every 20 optimiser steps per
  update -- 20% of all steps -- driven by <= 3 timesteps of data.

DOES T mod 4 != 0 ACTUALLY HAPPEN IN TRAINING? Yes.

  train.py sizes the buffer at rollout_episodes * episode_steps = 8 * 400 and
  calls buf.clear() after each update, so T is the SUM OF EIGHT ACTUAL EPISODE
  LENGTHS. Episodes end early whenever every task reaches a terminal state
  (env.py:594), and run_R2_mc_target_train.log records

      "episode ends: 377 time-limit truncation, 223 true terminal"

  so 223/600 = 37.2% of R2's training episodes ended BEFORE step 400 and T was
  below 3200 for essentially every update. Rung 2.5's replica happened to use
  8 x 400 = 3200 exactly, which IS divisible by 4, which is why the tail never
  appeared there and why its per-minibatch counters read n_minibatches = 16.

WHAT IS MEASURED:

  D1  the chunk arithmetic, enumerated over the realistic range of T
  D2  the empirical episode-length distribution on the REAL training start
      ticks, under each checkpoint, and the induced distribution of T mod 4
  D3  the measurement the brief asks for: per-chunk gradient norm, clip
      engagement, parameter-update norm, and advantage statistics, for
      SUBSTANTIVE chunks vs the DEGENERATE tail -- on both arms, with r forced
      to 1, 2 and 3 by trimming the buffer, so the comparison is matched
  D4  the falsification test the brief demands: a defect shared by A0 and R2
      cannot explain a DIFFERENCE between A0 and R2. Measured, not assumed.

Writes SPRINT_7_RUNG2_75_mbtail.json.
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

from marl._diag_rung0 import load_agent_and_cfg, _replay, OUT_DIR  # noqa: E402
from marl._diag_rung2_5_actor_stall import build_buffer_r2         # noqa: E402
from marl._diag_rung2_5_targets import (                           # noqa: E402
    training_start_ticks, TRAIN_SEED,
)
from marl.env import DTMarlEnv                                     # noqa: E402
from marl.mappo import masked_dist                                 # noqa: E402

TAG = "SPRINT_7_RUNG2_75"
MODELS = {"A0": "mappo_A0_cpu_repro.pth", "R2": "mappo_R2_mc_target.pth"}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=8,
                   help="rollout_episodes; train.py uses 8")
    p.add_argument("--length-episodes", type=int, default=40,
                   help="how many real training start ticks to replay for D2")
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


# ----------------------------------------------------------------------
# D1: the arithmetic, no model needed
# ----------------------------------------------------------------------

def chunk_sizes(T, n_mb=4):
    mb = max(1, T // n_mb)
    return [len(range(s, min(s + mb, T))) for s in range(0, T, mb)]


def d1_arithmetic(n_mb=4, cap=3200):
    rows = []
    for r in range(n_mb):
        T = cap - r                      # cap is divisible by 4, so T%4 == -r%4
        cs = chunk_sizes(T, n_mb)
        rows.append(dict(T=T, T_mod_n_mb=T % n_mb, n_chunks=len(cs),
                         chunk_sizes=cs, tail_size=cs[-1] if len(cs) > n_mb
                         else None))
    # and a sweep, to show the rule holds and is not an artefact of T near cap
    sweep = {}
    for T in range(2000, 3201):
        cs = chunk_sizes(T, n_mb)
        key = (len(cs), cs[-1] if len(cs) > n_mb else 0)
        sweep.setdefault(str(key), 0)
        sweep[str(key)] += 1
    return dict(
        examples=rows, sweep_2000_to_3200_by_n_chunks_and_tail=sweep,
        rule="T = n_mb*q + r  =>  chunks are [q]*n_mb + ([r] if r>0 else []); "
             "tail size == T mod n_mb",
        optimiser_steps_per_update=dict(
            n_mb=n_mb, ppo_epochs=4,
            steps_when_divisible=4 * n_mb,
            steps_when_not=4 * (n_mb + 1),
            frac_of_steps_that_are_degenerate=4 / (4 * (n_mb + 1)),
        ),
    )


# ----------------------------------------------------------------------
# D2: real episode lengths -> distribution of T and T mod 4
# ----------------------------------------------------------------------

def d2_lengths(agent, cfg, n_eps, rollout_eps, n_mb=4):
    env = DTMarlEnv(cfg.env, cfg.reward)
    starts = training_start_ticks(env, n_eps)
    lens = []
    for j, s in enumerate(starts):
        torch.manual_seed(TRAIN_SEED + j)
        r = _replay(env, agent, int(s), j, record=False, sample=True)
        lens.append(int(r["n_steps"]))
    lens = np.asarray(lens)
    full = int(cfg.env.episode_steps)
    blocks = [int(lens[i:i + rollout_eps].sum())
              for i in range(0, len(lens) - rollout_eps + 1, rollout_eps)]
    mods = [b % n_mb for b in blocks]
    return dict(
        n_episodes_replayed=len(lens),
        start_ticks=[int(s) for s in starts],
        episode_lengths=lens.tolist(),
        episode_steps_limit=full,
        n_truncated=int((lens >= full).sum()),
        n_terminated_early=int((lens < full).sum()),
        frac_terminated_early=float((lens < full).mean()),
        mean_length=float(lens.mean()), min_length=int(lens.min()),
        blocks_of_8=blocks, T_mod_4=mods,
        frac_blocks_with_tail=float(np.mean([m != 0 for m in mods]))
        if mods else None,
        tail_sizes=[m for m in mods if m != 0],
        caveat="replayed under the FINAL checkpoint, not the historical "
               "mid-training policies, so this is the induced distribution "
               "of T for THIS policy on the REAL start ticks -- an estimate "
               "of how often the tail fired, not the historical trace",
    )


# ----------------------------------------------------------------------
# D3: per-chunk gradient / update contribution
# ----------------------------------------------------------------------

def gnorm(params):
    tot = 0.0
    for p in params:
        if p.grad is not None:
            tot += float(p.grad.detach().pow(2).sum())
    return float(np.sqrt(tot))


def pnorm_delta(before, after):
    tot = 0.0
    for a, b in zip(before, after):
        tot += float((a - b).pow(2).sum())
    return float(np.sqrt(tot))


def instrumented_update(agent, buf, T_override=None):
    """
    MAPPO.update's minibatch loop, per-chunk instrumented. Honours
    cfg.critic_target. Optimiser steps are real but taken on a deepcopy.
    """
    cfg = agent.cfg
    T = len(buf) if T_override is None else int(T_override)
    # GAE is computed over the FULL buffer and then sliced, so the advantage
    # values on the retained indices are bit-identical across every trim. The
    # trim changes ONLY the chunking, which is exactly the variable under test.
    adv_np, ret_np = agent.compute_gae(buf)
    if getattr(cfg, "critic_target", "lambda") == "mc":
        ret_np = agent.compute_mc_returns(buf)
    dev = agent.device
    obs = torch.as_tensor(buf.obs[:T], device=dev)
    state = torch.as_tensor(buf.state[:T], device=dev)
    act = torch.as_tensor(buf.act[:T], device=dev)
    old_lp = torch.as_tensor(buf.logp[:T], device=dev)
    old_v = torch.as_tensor(buf.val[:T], device=dev)
    mask = torch.as_tensor(buf.mask[:T], device=dev)
    dec = torch.as_tensor(buf.decision[:T], device=dev)
    adv = torch.as_tensor(adv_np[:T], device=dev)
    ret = torch.as_tensor(ret_np[:T], device=dev)

    if cfg.normalise_advantages:
        sel = dec > 0.5
        if sel.any():
            a = adv[sel]
            adv = (adv - a.mean()) / (a.std() + 1e-8)

    n_mb = max(1, cfg.minibatches)
    mb_size = max(1, T // n_mb)
    rng = np.random.default_rng(0)
    trace = []
    for ep in range(cfg.ppo_epochs):
        order = rng.permutation(T)
        for ci, start in enumerate(range(0, T, mb_size)):
            idx_np = order[start:start + mb_size]
            if idx_np.size == 0:
                continue
            idx = torch.as_tensor(idx_np, dtype=torch.long, device=dev)
            before = [p.detach().clone() for p in agent.actor.parameters()]

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
            actor_loss = pg + cfg.entropy_coef * ent_loss

            agent.opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            g_pre = gnorm(agent.actor.parameters())
            nn.utils.clip_grad_norm_(agent.actor.parameters(),
                                     cfg.max_grad_norm)
            g_post = gnorm(agent.actor.parameters())
            agent.opt_actor.step()
            after = [p.detach().clone() for p in agent.actor.parameters()]

            a_mb = adv[idx][d_mb > 0.5]
            trace.append(dict(
                epoch=ep, chunk=ci, start=int(start),
                n_timesteps=int(idx_np.size),
                is_tail=bool(ci >= n_mb),
                n_decision_entries=int(d_mb.sum().item()),
                grad_norm_preclip=g_pre, grad_norm_postclip=g_post,
                clip_engaged=bool(g_pre > cfg.max_grad_norm),
                clip_shrink=float(g_post / g_pre) if g_pre > 0 else None,
                param_delta_l2=pnorm_delta(before, after),
                adv_mean=float(a_mb.mean().item()) if a_mb.numel() else None,
                adv_std=float(a_mb.std().item()) if a_mb.numel() > 1 else None,
                adv_absmax=float(a_mb.abs().max().item())
                if a_mb.numel() else None,
                pg_loss=float(pg.item()), entropy=float(ent.mean().item()),
            ))
    return dict(T=T, mb_size=mb_size, n_mb_configured=n_mb,
                n_chunks_per_epoch=len(range(0, T, mb_size)),
                tail_size=T % n_mb, trace=trace)


def summarise(res):
    tr = res["trace"]
    sub = [x for x in tr if not x["is_tail"]]
    tail = [x for x in tr if x["is_tail"]]

    def agg(rows, key):
        v = [x[key] for x in rows if x.get(key) is not None]
        return dict(n=len(v), mean=float(np.mean(v)) if v else None,
                    sd=float(np.std(v, ddof=1)) if len(v) > 1 else None,
                    min=float(np.min(v)) if v else None,
                    max=float(np.max(v)) if v else None)
    out = dict(T=res["T"], mb_size=res["mb_size"], tail_size=res["tail_size"],
               n_chunks_per_epoch=res["n_chunks_per_epoch"],
               n_optimiser_steps=len(tr),
               n_substantive=len(sub), n_tail=len(tail),
               frac_steps_degenerate=float(len(tail) / max(len(tr), 1)))
    for key in ("n_timesteps", "n_decision_entries", "grad_norm_preclip",
                "grad_norm_postclip", "param_delta_l2", "adv_std",
                "adv_absmax", "pg_loss"):
        out[f"substantive_{key}"] = agg(sub, key)
        out[f"tail_{key}"] = agg(tail, key)
    out["substantive_clip_engaged_frac"] = float(
        np.mean([x["clip_engaged"] for x in sub])) if sub else None
    out["tail_clip_engaged_frac"] = float(
        np.mean([x["clip_engaged"] for x in tail])) if tail else None
    if sub and tail:
        sg = out["substantive_grad_norm_preclip"]["mean"]
        tg = out["tail_grad_norm_preclip"]["mean"]
        sp = out["substantive_param_delta_l2"]["mean"]
        tp = out["tail_param_delta_l2"]["mean"]
        out["tail_over_substantive_grad_norm"] = (
            float(tg / sg) if sg else None)
        out["tail_over_substantive_param_delta"] = (
            float(tp / sp) if sp else None)
        out["tail_data_fraction"] = float(
            out["tail_n_timesteps"]["mean"] / out["substantive_n_timesteps"]["mean"])
    return out


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 RUNG 2.75 -- item D: minibatch tail as a CANDIDATE")
    print("        not fixed; steps taken on a deepcopy and never saved")
    print("=" * 78)

    d1 = d1_arithmetic()
    print("\n-- D1  chunk arithmetic (n_mb = 4) ----------------------------")
    for r in d1["examples"]:
        print(f"   T={r['T']:5d}  T mod 4 = {r['T_mod_n_mb']}  "
              f"chunks={r['n_chunks']}  sizes={r['chunk_sizes'][:4]}"
              f"{'+[' + str(r['chunk_sizes'][-1]) + ']' if r['n_chunks'] > 4 else ''}")
    o = d1["optimiser_steps_per_update"]
    print(f"   optimiser steps/update: {o['steps_when_divisible']} if T%4==0, "
          f"else {o['steps_when_not']} "
          f"-> {o['frac_of_steps_that_are_degenerate']:.1%} degenerate")

    out = {}
    for tag, fn in MODELS.items():
        print(f"\n-- {tag} ({fn}) " + "-" * (58 - len(tag) - len(fn)))
        agent, extra, cfg = load_agent_and_cfg(
            str(OUT_DIR / fn), args.device, "train")

        d2 = d2_lengths(agent, cfg, args.length_episodes, args.episodes)
        print(f"   D2  {d2['n_episodes_replayed']} real start ticks: "
              f"lengths mean {d2['mean_length']:.1f} min {d2['min_length']}, "
              f"early-terminating {d2['frac_terminated_early']:.1%}")
        print(f"       T per block of 8: {d2['blocks_of_8']}")
        print(f"       T mod 4         : {d2['T_mod_4']}  "
              f"-> tail fires in {d2['frac_blocks_with_tail']:.0%} of blocks")

        # build ONE real 8-episode buffer, exactly as train.py would
        env = DTMarlEnv(cfg.env, cfg.reward)
        starts = training_start_ticks(env, args.episodes)
        recs, trunc = [], []
        for j, s in enumerate(starts):
            torch.manual_seed(TRAIN_SEED + j)
            r = _replay(env, agent, int(s), j, record=True, sample=True)
            recs.append(r["rec"])
            trunc.append(bool(r["n_steps"] >= env.cfg.episode_steps))
        buf = build_buffer_r2(agent, recs, trunc)
        T_full = len(buf)
        print(f"   D3  real buffer T = {T_full} (T mod 4 = {T_full % 4})")

        cells = {}
        for trim in range(4):
            # trimming 0..3 timesteps forces every residue r = T mod 4 in turn,
            # so "substantive vs tail" is compared at matched data and matched
            # starting parameters
            T = T_full - trim
            work = copy.deepcopy(agent)
            res = instrumented_update(work, buf, T_override=T)
            s = summarise(res)
            cells[f"T={T}_mod{T % 4}"] = s
            tg = s.get("tail_grad_norm_preclip", {}).get("mean")
            tp = s.get("tail_param_delta_l2", {}).get("mean")
            print(f"       T={T:5d} mod4={T % 4}  chunks/epoch="
                  f"{s['n_chunks_per_epoch']}  steps={s['n_optimiser_steps']}"
                  f"  tail={s['n_tail']}")
            print(f"              substantive: {s['substantive_n_timesteps']['mean']:.0f} steps"
                  f"  |g|={s['substantive_grad_norm_preclip']['mean']:.4e}"
                  f"  clip {s['substantive_clip_engaged_frac']:.2f}"
                  f"  dtheta={s['substantive_param_delta_l2']['mean']:.4e}")
            if s["n_tail"]:
                print(f"              TAIL       : {s['tail_n_timesteps']['mean']:.0f} steps"
                      f"  |g|={tg:.4e}"
                      f"  clip {s['tail_clip_engaged_frac']:.2f}"
                      f"  dtheta={tp:.4e}")
                print(f"              tail/substantive: data "
                      f"{s['tail_data_fraction']:.4f}x   "
                      f"|g| {s['tail_over_substantive_grad_norm']:.3f}x   "
                      f"dtheta {s['tail_over_substantive_param_delta']:.3f}x")
        out[tag] = dict(d2_lengths=d2, T_full=T_full, cells=cells)

    # ---------------- D4: can a SHARED defect explain a DIFFERENCE? -------
    print("\n" + "=" * 78)
    print("D4  falsification: the tail is shared by both arms")
    print("=" * 78)
    shared = {}
    for tag in MODELS:
        d2 = out[tag]["d2_lengths"]
        shared[tag] = dict(
            frac_blocks_with_tail=d2["frac_blocks_with_tail"],
            frac_episodes_early=d2["frac_terminated_early"],
            mean_T=float(np.mean(d2["blocks_of_8"])) if d2["blocks_of_8"] else None,
        )
        print(f"   {tag}: tail fires in {d2['frac_blocks_with_tail']:.0%} of "
              f"blocks, {d2['frac_terminated_early']:.0%} of episodes end early")
    print("   -> if both arms are exposed at a similar rate, the tail cannot")
    print("      be the cause of an A0-vs-R2 DIFFERENCE, only of a shared")
    print("      degradation. Stated as a measurement, not an assumption.")

    blob = dict(
        probe=f"{TAG}_mbtail",
        what="minibatch tail examined as a candidate cause of R2 saturation; "
             "NOT fixed, no production change, steps on a deepcopy only",
        defect_source="marl/mappo.py:368-377  mb_size = max(1, T // n_mb) "
                      "then for start in range(0, T, mb_size)",
        why_rung2_5_missed_it="its replica buffer was 8 x 400 = 3200, which IS "
                              "divisible by 4, so no tail chunk existed and "
                              "n_minibatches read 16",
        training_geometry=dict(
            rollout_episodes=8, episodes=600, updates=75,
            minibatches=4, ppo_epochs=4,
            buffer_cleared_each_update=True,
            T_equals="sum of the 8 actual episode lengths in the block",
            r2_training_log="run_R2_mc_target_train.log:94  'episode ends: "
                            "377 time-limit truncation, 223 true terminal' "
                            "-> 37.2% of episodes ended before step 400",
            a0_training_log="run_A0_cpu_repro.log predates the counter (it was "
                            "added with the Rung 2 critic_target flag), so A0's "
                            "historical split is not on record; D2 estimates it",
        ),
        D1_arithmetic=d1, per_arm=out, D4_shared_exposure=shared,
    )
    p = OUT_DIR / f"{TAG}_mbtail_{args.tag}.json"
    p.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n  wrote {p}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
