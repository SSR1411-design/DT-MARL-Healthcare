"""
Train MAPPO on the DT-MARL environment (Sprint 6, testing step 5).

    python marl/train.py                       # default configuration
    python marl/train.py --episodes 200
    python marl/train.py --risk-source zero    # the no-prediction ablation
    python marl/train.py --tag myrun

Saves, into saved_models/marl/:

    <tag>.pth              actor(s) + critic + optimiser state + geometry
    <tag>_config.json      the FULL configuration that produced the run
    <tag>_history.csv      per-episode reward and metrics
    <tag>_updates.csv      per-update PPO diagnostics

Training uses episode starts drawn from the FIRST `start_frac_hi` fraction of
the trace; evaluate.py uses the remainder. Reproducible from `--seed` alone.
"""

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.config import Sprint6Config, resolve_device, ACTION_NAMES  # noqa: E402
from marl.env import DTMarlEnv                                        # noqa: E402
from marl.mappo import MAPPO, RolloutBuffer, MappoPolicy              # noqa: E402
from marl.rollout import run_episodes, episode_starts                 # noqa: E402

TRAIN_FRAC = 0.7          # episode-start window given to training


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train MAPPO on DT-MARL")
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--rollout-episodes", type=int, default=None)
    p.add_argument("--episode-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--lr-actor", type=float, default=None)
    p.add_argument("--lr-critic", type=float, default=None)
    p.add_argument("--entropy-coef", type=float, default=None)
    # Sprint 6.5 arm A2. The migration charge is scaled by
    # (1 + w_criticality_migration * severity), i.e. moving a CRITICAL patient's
    # task costs more than moving a routine one - the only criticality term in
    # the reward that pushes against protecting critical patients (loss, SLA and
    # exposure all push toward it via w_criticality). Sprint 6 never felt this
    # because the term was dead; A1 activated it and the severity/relocation
    # correlation stayed negative. Exposed as a coefficient so the direction can
    # be tested rather than asserted.
    p.add_argument("--w-criticality-migration", type=float, default=None)
    p.add_argument("--risk-source", choices=["oof", "model", "zero"], default=None)
    # SPRINT 7 RUNG 2. Critic regression target only; the actor's GAE is
    # unaffected. Default None -> config default "lambda" -> A0 behaviour.
    p.add_argument("--critic-target", choices=["lambda", "mc"], default=None,
                   help="critic regression target: lambda-return (A0 control) "
                        "or within-episode Monte-Carlo return (R2-MC)")
    p.add_argument("--device", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--log-every", type=int, default=None)
    # Sprint 6.5 ablation arms. --legacy-sprint6 restores BOTH defects the
    # Sprint 6 baseline contained, so arm A0 is re-runnable from this codebase;
    # the two individual flags let each fix be attributed separately.
    p.add_argument("--legacy-sprint6", action="store_true",
                   help="reproduce the defective Sprint 6 environment (A0)")
    p.add_argument("--legacy-fixed-apply-order", action="store_true")
    p.add_argument("--legacy-dead-migration-criticality", action="store_true")
    return p.parse_args(argv)


def apply_args(cfg, a):
    if a.episodes is not None:        cfg.train.episodes = a.episodes
    if a.rollout_episodes is not None: cfg.train.rollout_episodes = a.rollout_episodes
    if a.episode_steps is not None:   cfg.env.episode_steps = a.episode_steps
    if a.seed is not None:            cfg.train.seed = cfg.env.seed = a.seed
    if a.lr_actor is not None:        cfg.mappo.lr_actor = a.lr_actor
    if a.lr_critic is not None:       cfg.mappo.lr_critic = a.lr_critic
    if a.entropy_coef is not None:    cfg.mappo.entropy_coef = a.entropy_coef
    if a.w_criticality_migration is not None:
        cfg.reward.w_criticality_migration = a.w_criticality_migration
    if a.risk_source is not None:     cfg.env.risk_source = a.risk_source
    if a.critic_target is not None:   cfg.mappo.critic_target = a.critic_target
    if a.device is not None:          cfg.train.device = a.device
    if a.tag is not None:             cfg.train.tag = a.tag
    if a.log_every is not None:       cfg.train.log_every = a.log_every
    if a.legacy_sprint6:
        cfg.env.legacy_fixed_apply_order = True
        cfg.env.legacy_dead_migration_criticality = True
    if a.legacy_fixed_apply_order:    cfg.env.legacy_fixed_apply_order = True
    if a.legacy_dead_migration_criticality:
        cfg.env.legacy_dead_migration_criticality = True
    return cfg


def main(argv=None):
    args = parse_args(argv)
    cfg = apply_args(Sprint6Config(), args)

    # deterministic training window
    cfg.env.start_frac_lo, cfg.env.start_frac_hi = 0.0, TRAIN_FRAC

    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed % (2 ** 32))
    device = resolve_device(cfg.train.device)

    env = DTMarlEnv(cfg.env, cfg.reward)
    agent = MAPPO(env.n_agents, env.obs_dim, env.state_dim, cfg.mappo,
                  device=device, seed=cfg.train.seed)

    out = Path(cfg.train.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = cfg.train.tag

    print("=" * 78)
    print("SPRINT 6 — MAPPO TRAINING")
    print("=" * 78)
    print(env.describe())
    print()
    print(agent.describe())
    print(f"  independence      : {agent.assert_actors_independent()}")
    print()
    print(f"  episodes          : {cfg.train.episodes} "
          f"({cfg.train.rollout_episodes} per PPO update = "
          f"{cfg.train.episodes // cfg.train.rollout_episodes} updates)")
    print(f"  train start window: ticks [{env._min_start}, {env._max_start}] "
          f"(first {TRAIN_FRAC:.0%} of the usable range; evaluate.py uses the rest)")
    print(f"  seed              : {cfg.train.seed}")
    print("=" * 78)

    cap = cfg.train.rollout_episodes * cfg.env.episode_steps
    buf = RolloutBuffer(cap, env.n_agents, env.obs_dim, env.state_dim)

    hist_path = out / f"{tag}_history.csv"
    upd_path = out / f"{tag}_updates.csv"
    hf = open(hist_path, "w", newline="")
    hw = csv.writer(hf)
    hw.writerow(["episode", "start_tick", "reward", "success_rate", "lost",
                 "critical_lost", "relocations", "preemptive", "sla",
                 "protected", "energy", "infeasible"])
    uf = open(upd_path, "w", newline="")
    uw = csv.writer(uf)
    uw.writerow(["update", "episode", "mean_reward", "actor_loss",
                 "critic_loss", "entropy", "approx_kl", "clip_frac",
                 "adv_std", "value_mean", "decision_frac", "lr_scale",
                 "explained_var"])

    rng = np.random.default_rng(cfg.train.seed)
    ep_rewards, recent, update_id = [], [], 0
    # How each episode ended. Sprint 7 Rung 1 found 0/16 diagnostic episodes hit
    # the step limit, so the MC target was pure MC there; this records whether
    # that also holds during training, since it decides how much of the target
    # is critic-free.
    n_trunc, n_term = 0, 0
    last_stats = {}
    best_mean, best_path = -np.inf, None
    t_start = time.time()
    agent.train_mode()

    n_updates = max(1, cfg.train.episodes // cfg.train.rollout_episodes)

    for ep in range(1, cfg.train.episodes + 1):
        start = int(rng.integers(env._min_start, env._max_start + 1))
        obs, state, masks = env.reset(episode_start_tick=start)
        done = False
        step_in_ep = 0
        while not done:
            a, lp = agent.act(obs, masks)
            v = agent.value(state)
            nobs, nstate, rew, done, info = env.step(a)
            buf.add(obs, state, a, lp, v, rew, masks, 0.0 if done else 1.0)
            if done:
                # An episode ends EITHER at the step limit — a genuine
                # truncation, where real return remains but is unobserved, so
                # bootstrap — OR because every task reached a terminal state
                # (env.py:594), where there is nothing further to predict.
                # compute_gae bootstraps in BOTH cases exactly as before, so the
                # A0 control is bit-identical; this flag is read only by the MC
                # critic target.
                truncated = bool(env.step_idx >= env.cfg.episode_steps)
                buf.set_bootstrap(buf.ptr - 1, agent.value(nstate), truncated)
                n_trunc += int(truncated)
                n_term += int(not truncated)
            obs, state, masks = nobs, nstate, info["action_masks"]
            step_in_ep += 1

        m = env.episode_metrics()
        ep_rewards.append(m["episode_reward"])
        recent.append(m["episode_reward"])
        hw.writerow([ep, start, f"{m['episode_reward']:.4f}",
                     f"{m['task_success_rate']:.4f}", m["lost"],
                     m["failed_critical_tasks"], m["relocations"],
                     m["preemptive_relocations"], m["sla_violations"],
                     m["tasks_protected_before_failure"],
                     f"{m['energy_cost']:.2f}", m["infeasible_actions"]])
        # Flushed every episode, not left to the buffer. Three diagnostic runs
        # were killed mid-training and each lost its entire history file, which
        # is the one artifact you actually need when a run is going wrong.
        hf.flush()

        if ep % cfg.train.rollout_episodes == 0:
            update_id += 1
            frac = 1.0 - (update_id - 1) / n_updates
            agent.set_lr_scale(frac)
            stats = agent.update(buf)
            last_stats = stats
            mean_r = float(np.mean(recent))
            uw.writerow([update_id, ep, f"{mean_r:.4f}",
                         f"{stats['actor_loss']:.6f}",
                         f"{stats['critic_loss']:.6f}",
                         f"{stats['entropy']:.4f}",
                         f"{stats['approx_kl']:.6f}",
                         f"{stats['clip_frac']:.4f}",
                         f"{stats['adv_std']:.4f}",
                         f"{stats['value_mean']:.4f}",
                         f"{stats['decision_frac']:.4f}",
                         f"{frac:.4f}",
                         f"{stats['explained_var']:.4f}"])
            uf.flush()
            buf.clear()
            recent = []
            if mean_r > best_mean:
                best_mean = mean_r
                best_path = agent.save(
                    out / f"{tag}_best.pth",
                    extra=dict(config=cfg.to_dict(), episode=ep,
                               mean_reward=mean_r, kind="best"))

        if ep % cfg.train.log_every == 0 or ep == 1:
            w = ep_rewards[-cfg.train.log_every:]
            print(f"  ep {ep:4d}/{cfg.train.episodes}  "
                  f"reward {np.mean(w):+9.2f}  "
                  f"success {m['task_success_rate']:.3f}  "
                  f"lost {m['lost']:2.0f}  reloc {m['relocations']:3.0f}  "
                  f"preem {m['preemptive_relocations']:3.0f}  "
                  f"prot {m['tasks_protected_before_failure']:3.0f}  "
                  f"ev {last_stats.get('explained_var', float('nan')):+.2f}  "
                  f"ent {last_stats.get('entropy', float('nan')):.3f}  "
                  f"({time.time() - t_start:5.0f}s)")

    hf.close()
    uf.close()

    final_path = agent.save(
        out / f"{tag}.pth",
        extra=dict(config=cfg.to_dict(), episode=cfg.train.episodes,
                   mean_reward=float(np.mean(ep_rewards[-20:])), kind="final"))
    with open(out / f"{tag}_config.json", "w") as f:
        json.dump({"config": cfg.to_dict(),
                   "train_start_window": [env._min_start, env._max_start],
                   "train_frac": TRAIN_FRAC,
                   "device": device,
                   "episodes": cfg.train.episodes,
                   "wall_time_s": round(time.time() - t_start, 1),
                   "risk": {"source": env.risk.source,
                            "calibrated": env.risk.calibrated,
                            "notes": env.risk.notes}}, f, indent=2)

    print("-" * 78)
    first = float(np.mean(ep_rewards[:20]))
    last = float(np.mean(ep_rewards[-20:]))
    print(f"  critic target     : {cfg.mappo.critic_target}"
          f"   (episode ends: {n_trunc} time-limit truncation, "
          f"{n_term} true terminal)")
    print(f"  first 20 episodes : {first:+.2f}")
    print(f"  last  20 episodes : {last:+.2f}   (change {last - first:+.2f})")
    print(f"  best update mean  : {best_mean:+.2f}")
    print(f"  wall time         : {time.time() - t_start:.0f}s")
    print(f"  saved             : {final_path}")
    print(f"                      {best_path}")
    print(f"                      {hist_path}")
    print(f"                      {upd_path}")

    # quick in-window greedy check (the reported numbers come from evaluate.py)
    agent.eval()
    starts = episode_starts(env, 5)
    agg, _, _, _, _ = run_episodes(env, MappoPolicy(agent, "mappo-greedy"),
                                   starts, collect_actions=True)
    print(f"  greedy, TRAIN window, 5 episodes: reward "
          f"{agg['episode_reward']:+.2f}  success {agg['task_success_rate']:.3f}"
          f"  reloc {agg['relocations']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
