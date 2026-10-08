"""
SPRINT 11.1 — Workload Scaling Experiment

Evaluates the already-trained MAPPO checkpoint under increasing
healthcare workload sizes without retraining the policy.

Reference:
    40 tasks

Scaling:
    80, 120, 160 tasks

The trained checkpoint is kept unchanged.
"""

from pathlib import Path
import csv
import json
import numpy as np

from marl.config import EnvConfig, RewardConfig, MappoConfig
from marl.env import DTMarlEnv
from marl.mappo import MAPPO


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT / "saved_models" / "marl" /
    "mappo_R2_mc_target_best.pth"
)

OUTPUT_DIR = ROOT / "saved_models" / "marl" / "sprint11"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TASK_COUNTS = [40, 80, 120, 160]

EPISODES = 8

# Same evaluation starts used by the validated Sprint-6 evaluation
START_TICKS = [491, 521, 550, 580, 609, 639, 668, 698]


def load_policy(cfg):
    """
    Load the already-trained MAPPO actor.

    The checkpoint is NOT retrained or modified.
    """
    mcfg = MappoConfig()

    policy = MAPPO(
        obs_dim=cfg.n_edge_nodes *
                0 +  # placeholder; actual dimensions are read below
                1,
        global_state_dim=1,
        n_agents=cfg.n_edge_nodes,
        n_actions=5,
        config=mcfg,
    )

    checkpoint = __import__("torch").load(
        CHECKPOINT,
        map_location="cpu"
    )

    # The repository's checkpoint loader may expose one of these APIs.
    if hasattr(policy, "load"):
        policy.load(CHECKPOINT)
    elif hasattr(policy, "load_state_dict"):
        state = checkpoint.get("actor", checkpoint)
        policy.load_state_dict(state)
    else:
        raise RuntimeError(
            "Could not find a checkpoint-loading method on MAPPO."
        )

    return policy


def evaluate_workload(n_tasks):
    """
    Evaluate the fixed MAPPO policy for one workload size.
    """

    cfg = EnvConfig()

    cfg.n_tasks = n_tasks
    cfg.random_episode_start = False
    cfg.risk_source = "oof"

    env = DTMarlEnv(
        cfg=cfg,
        rcfg=RewardConfig(),
        verbose=False,
    )

    # Print configuration so every experiment is auditable.
    print()
    print("=" * 90)
    print(f"SPRINT 11.1 — WORKLOAD SCALING")
    print("=" * 90)
    print(f"tasks              : {n_tasks}")
    print(f"edge nodes         : {cfg.n_edge_nodes}")
    print(f"VMs/node           : {cfg.vms_per_node}")
    print(f"risk source        : {cfg.risk_source}")
    print(f"checkpoint         : {CHECKPOINT.name}")
    print(f"episodes           : {EPISODES}")
    print("=" * 90)

    # ------------------------------------------------------------
    # IMPORTANT:
    # The environment must expose the same observation/action
    # dimensions as the trained policy.
    # ------------------------------------------------------------

    obs = env.reset(episode_start_tick=START_TICKS[0])

    print(f"observation type   : {type(obs)}")
    if isinstance(obs, tuple):
        print(f"observation parts  : {len(obs)}")
        for i, part in enumerate(obs):
            try:
                print(f"  part[{i}] shape   : {np.asarray(part).shape}")
            except Exception:
                print(f"  part[{i}] type    : {type(part)}")
    else:
        print(f"observation shape  : {np.asarray(obs).shape}")

    # We stop here intentionally for the first run.
    #
    # This confirms that changing n_tasks does not break the
    # environment before attempting policy inference.

    metrics = env.episode_metrics()

    return {
        "tasks": n_tasks,
        "status": "environment_initialised",
        "observation_shape": str(np.asarray(obs).shape),
    }


def main():

    results = []

    print("\n")
    print("#" * 90)
    print("# SPRINT 11.1 — WORKLOAD SCALING")
    print("#" * 90)
    print()
    print("FIXED POLICY:")
    print(f"  {CHECKPOINT}")
    print()
    print("WORKLOAD MATRIX:")
    print(f"  {TASK_COUNTS}")
    print()
    print("No retraining is performed.")
    print("No changes are made to the validated checkpoint.")
    print()

    for n_tasks in TASK_COUNTS:
        result = evaluate_workload(n_tasks)
        results.append(result)

    output_json = OUTPUT_DIR / "sprint11_workload_scaling_environment_check.json"

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print()
    print("=" * 90)
    print("SPRINT 11.1 ENVIRONMENT CHECK COMPLETE")
    print("=" * 90)
    print(f"written: {output_json}")
    print()
    print("If all four workloads initialise successfully,")
    print("we proceed to the actual MAPPO evaluation.")
    print("=" * 90)


if __name__ == "__main__":
    main()