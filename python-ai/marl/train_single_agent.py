"""Manual-only training entry point for Sprint 9's independent PPO baseline.

This command is intentionally inert unless ``--allow-training`` is supplied.
Sprint 9 implementation and validation must never accidentally train a model.
When a later approved study runs it, checkpoints and real update losses are
written only under ``saved_models/single_agent/`` with a caller-selected tag.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.config import resolve_device
from marl.env import DTMarlEnv
from marl.single_agent_ppo import SingleAgentPPO, SingleAgentPPOConfig, SingleAgentRolloutBuffer
from marl.sprint9_benchmarks import load_validated_r2_config


TRAIN_FRACTION = 0.70
DEFAULT_OUTPUT_DIR = ROOT / "saved_models" / "single_agent"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Manual-only Sprint 9 single-agent PPO trainer")
    parser.add_argument("--allow-training", action="store_true",
                        help="required explicit acknowledgement before any training begins")
    parser.add_argument("--episodes", type=int, default=None,
                        help="required number of real environment episodes")
    parser.add_argument("--rollout-episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--risk-source", choices=("oof", "zero"), default="oof")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tag", default="single_agent_ppo")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args(argv)


def _require_explicit_training(args) -> None:
    if not args.allow_training:
        raise SystemExit(
            "Training is disabled by default. To run a separately approved study, "
            "pass --allow-training and an explicit --episodes value.")
    if args.episodes is None or args.episodes < 1:
        raise SystemExit("--episodes must be a positive explicit value when training is allowed")
    if args.rollout_episodes < 1:
        raise SystemExit("--rollout-episodes must be positive")


def _artifact_paths(output_dir: Path, tag: str) -> dict:
    return {
        "checkpoint": output_dir / f"{tag}.pth",
        "best": output_dir / f"{tag}_best.pth",
        "history": output_dir / f"{tag}_history.csv",
        "updates": output_dir / f"{tag}_updates.csv",
        "config": output_dir / f"{tag}_config.json",
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    _require_explicit_training(args)

    # Reuse the validated R2 *environment* configuration, never its MAPPO
    # weights or architecture. This makes a later approved training run share
    # the exact task, topology, trace, and reward settings used for evaluation.
    cfg = load_validated_r2_config()
    cfg.env.start_frac_lo, cfg.env.start_frac_hi = 0.0, TRAIN_FRACTION
    cfg.env.random_episode_start = True
    cfg.env.risk_source = args.risk_source
    cfg.env.seed = args.seed
    device = resolve_device(args.device)
    np.random.seed(args.seed % (2 ** 32))
    torch.manual_seed(args.seed)

    env = DTMarlEnv(cfg.env, cfg.reward)
    agent_cfg = SingleAgentPPOConfig()
    agent = SingleAgentPPO(env.state_dim, env.n_agents, agent_cfg,
                           device=device, seed=args.seed)
    output_dir = Path(args.out_dir).resolve()
    paths = _artifact_paths(output_dir, args.tag)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite a single-agent artifact: "
            + ", ".join(str(path) for path in existing))
    output_dir.mkdir(parents=True, exist_ok=True)

    capacity = args.rollout_episodes * cfg.env.episode_steps
    buffer = SingleAgentRolloutBuffer(capacity, env.state_dim, env.n_agents)
    starts = np.random.default_rng(args.seed)
    update_number = 0
    best_reward = -np.inf
    recent_rewards = []
    began = time.time()

    with open(paths["history"], "w", newline="", encoding="utf-8") as history_file, \
            open(paths["updates"], "w", newline="", encoding="utf-8") as update_file:
        history = csv.DictWriter(history_file, fieldnames=[
            "episode", "start_tick", "team_reward", "task_success_rate", "completed",
            "lost", "relocations", "recovery_attempts", "sla_violations", "energy_cost",
        ])
        updates = csv.DictWriter(update_file, fieldnames=[
            "update", "episode", "mean_reward", "actor_loss", "critic_loss", "entropy",
            "approx_kl", "clip_frac", "adv_mean", "adv_std", "value_mean",
            "joint_action_dimensions", "explained_var", "lr_scale",
        ])
        history.writeheader()
        updates.writeheader()
        for episode in range(1, args.episodes + 1):
            start = int(starts.integers(env._min_start, env._max_start + 1))
            observations, state, masks = env.reset(episode_start_tick=start,
                                                   seed=args.seed + episode)
            del observations
            done = False
            while not done:
                actions, log_probability = agent.act(state, masks)
                value = agent.value(state)
                _, next_state, rewards, done, info = env.step(actions)
                team_reward = float(np.sum(rewards))
                buffer.add(state, actions, log_probability, value, team_reward, masks,
                           0.0 if done else 1.0)
                if done:
                    buffer.set_bootstrap(agent.value(next_state))
                state, masks = next_state, info["action_masks"]

            metrics = env.episode_metrics()
            recent_rewards.append(float(metrics["episode_reward"]))
            history.writerow({
                "episode": episode, "start_tick": start,
                "team_reward": metrics["episode_reward"],
                "task_success_rate": metrics["task_success_rate"],
                "completed": metrics["completed"], "lost": metrics["lost"],
                "relocations": metrics["relocations"],
                "recovery_attempts": metrics["recovery_attempts"],
                "sla_violations": metrics["sla_violations"], "energy_cost": metrics["energy_cost"],
            })
            history_file.flush()

            should_update = (episode % args.rollout_episodes == 0 or episode == args.episodes)
            if should_update:
                update_number += 1
                total_updates = int(np.ceil(args.episodes / args.rollout_episodes))
                lr_scale = 1.0 - (update_number - 1) / max(total_updates, 1)
                agent.set_lr_scale(lr_scale)
                stats = agent.update(buffer)
                updates.writerow({
                    "update": update_number, "episode": episode,
                    "mean_reward": float(np.mean(recent_rewards)), **stats,
                    "lr_scale": lr_scale,
                })
                update_file.flush()
                mean_reward = float(np.mean(recent_rewards))
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    agent.save(paths["best"], extra={"episode": episode,
                                                        "mean_reward": mean_reward,
                                                        "kind": "best"})
                recent_rewards = []
                buffer.clear()
            if episode == 1 or episode % args.log_every == 0:
                print(f"episode {episode}/{args.episodes}: reward={metrics['episode_reward']:+.3f} "
                      f"success={metrics['task_success_rate']:.3f}")

    agent.save(paths["checkpoint"], extra={"episodes": args.episodes, "kind": "final"})
    with open(paths["config"], "w", encoding="utf-8") as handle:
        json.dump({
            "training_kind": "single_agent_ppo", "environment_config": cfg.to_dict(),
            "single_agent_ppo_config": agent_cfg.__dict__, "train_fraction": TRAIN_FRACTION,
            "seed": args.seed, "device": device, "episodes": args.episodes,
            "wall_time_s": round(time.time() - began, 3),
        }, handle, indent=2)
    print(f"single-agent PPO training completed in {time.time() - began:.1f}s")
    for label, path in paths.items():
        print(f"  {label}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
