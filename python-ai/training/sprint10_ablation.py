"""
SPRINT 10 - ABLATION STUDY

Compares the trained MAPPO system against controlled variants:

1. Full system
2. Without Digital Twin / risk input
3. Without failure prediction
4. Without criticality awareness
5. Without uncertainty gating

The MARL architecture itself remains unchanged for these experiments.
"""

from pathlib import Path
import sys
import csv
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from marl.config import Sprint6Config
from marl.env import DTMarlEnv
from marl.mappo import MAPPO, MappoPolicy


# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPISODES = 10

OUTPUT = ROOT / "data" / "sprint10_ablation.csv"


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

def run_episode(env, policy):
    obs, state, info = env.reset()

    total_reward = 0.0
    done = False

    while not done:
        masks = info["action_masks"]

        actions = policy.act(
            env,
            obs,
            masks
        )

        obs, state, rewards, done, info = env.step(actions)

        total_reward += float(np.sum(rewards))

    summary = env.summary()

    return {
        "reward": total_reward,
        "latency": summary.avg_task_latency_s,
        "energy": summary.energy_cost,
        "migrations": summary.migrations,
        "completed": summary.completed,
        "lost": summary.lost,
    }


# ------------------------------------------------------------
# RUN EXPERIMENT
# ------------------------------------------------------------

def evaluate_variant(name, cfg, agent):

    env = DTMarlEnv(cfg.env, cfg.reward)

    policy = MappoPolicy(
        agent,
        name=name,
        greedy=True
    )

    results = []

    for episode in range(EPISODES):

        metrics = run_episode(
            env,
            policy
        )

        metrics["variant"] = name
        metrics["episode"] = episode + 1

        results.append(metrics)

        print(
            f"{name:25s} "
            f"Episode {episode + 1:02d}/{EPISODES} "
            f"Reward={metrics['reward']:9.2f}"
        )

    return results


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("SPRINT 10 - ABLATION STUDY")
    print("=" * 70)

    print("\nDevice:", DEVICE)

    cfg = Sprint6Config()

    # --------------------------------------------------------
    # LOAD EXISTING TRAINED MAPPO
    # --------------------------------------------------------

    checkpoint = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_best.pth"

    if not checkpoint.exists():

        print("\nERROR:")
        print("Trained MAPPO checkpoint not found:")
        print(checkpoint)

        print("\nWe will locate the actual checkpoint next.")
        sys.exit(1)

    agent = MAPPO(
        obs_dim=cfg.env.obs_dim,
        state_dim=cfg.env.state_dim,
        n_agents=cfg.env.n_edge_nodes,
        n_actions=cfg.env.n_actions,
        cfg=cfg.mappo,
        device=DEVICE
    )

    agent.load(checkpoint)

    # --------------------------------------------------------
    # VARIANTS
    # --------------------------------------------------------

    variants = []

    # FULL SYSTEM
    variants.append(
        ("full-system", cfg)
    )

    # --------------------------------------------------------
    # WITHOUT DIGITAL TWIN / PREDICTED RISK
    # --------------------------------------------------------

    cfg_no_dt = Sprint6Config()

    cfg_no_dt.env.risk_source = "zero"

    variants.append(
        ("without-digital-twin", cfg_no_dt)
    )

    # --------------------------------------------------------
    # WITHOUT FAILURE PREDICTION
    # --------------------------------------------------------

    cfg_no_prediction = Sprint6Config()

    cfg_no_prediction.env.risk_source = "zero"

    variants.append(
        ("without-failure-prediction", cfg_no_prediction)
    )

    # --------------------------------------------------------
    # WITHOUT CRITICALITY
    # --------------------------------------------------------

    cfg_no_criticality = Sprint6Config()

    cfg_no_criticality.reward.w_criticality = 0.0
    cfg_no_criticality.reward.w_criticality_migration = 0.0

    variants.append(
        ("without-criticality", cfg_no_criticality)
    )

    # --------------------------------------------------------
    # WITHOUT UNCERTAINTY
    # --------------------------------------------------------

    cfg_no_uncertainty = Sprint6Config()

    # Uncertainty is represented by the risk provider.
    # Force the uncertainty channel to zero through the
    # environment configuration.

    cfg_no_uncertainty.env.risk_source = "zero"

    variants.append(
        ("without-uncertainty-gating", cfg_no_uncertainty)
    )

    # --------------------------------------------------------
    # EXECUTE
    # --------------------------------------------------------

    all_results = []

    for name, variant_cfg in variants:

        print("\n" + "-" * 70)
        print("RUNNING:", name)
        print("-" * 70)

        results = evaluate_variant(
            name,
            variant_cfg,
            agent
        )

        all_results.extend(results)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "episode",
                "reward",
                "latency",
                "energy",
                "migrations",
                "completed",
                "lost"
            ]
        )

        writer.writeheader()
        writer.writerows(all_results)

    print("\n" + "=" * 70)
    print("SPRINT 10 ABLATION COMPLETE")
    print("=" * 70)

    print("\nCSV:")
    print(OUTPUT)