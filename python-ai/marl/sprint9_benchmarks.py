"""Common, reproducible Sprint 9 baseline construction and evaluation.

The module deliberately has no MAPPO import and never opens the validated
MAPPO checkpoint.  It reconstructs the validated R2 *environment* settings
from the saved JSON configuration, then makes the baseline-specific changes
explicitly.  This keeps workload, trace, topology, task model, reward terms,
and held-out deterministic starts comparable without reusing a learned policy.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.baseline import DetectionRecoveryPolicy, ReactiveThresholdPolicy
from marl.config import EnvConfig, MappoConfig, RewardConfig, Sprint6Config
from marl.env import DTMarlEnv
from marl.rollout import episode_starts, run_episode


VALIDATED_R2_CONFIG = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_config.json"
HELD_OUT_START_FRACTION = 0.70
BASELINE_REACTIVE = "reactive-no-digital-twin"
BASELINE_DETECTION = "detection-without-prediction"
BASELINE_SINGLE_AGENT = "single-agent-ppo"
SUPPORTED_BASELINES = (
    BASELINE_REACTIVE,
    BASELINE_DETECTION,
    BASELINE_SINGLE_AGENT,
)


@dataclass(frozen=True)
class BaselineMetadata:
    """Provenance that accompanies every Sprint 9 output row."""

    baseline_name: str
    risk_source: str
    uses_digital_twin: bool
    uses_prediction: bool
    uses_uncertainty: bool
    uses_marl: bool
    seed: int
    node_count: int
    task_count: int
    recovery_mode: str
    training_loss_status: str
    checkpoint: str

    def as_row(self) -> dict:
        return asdict(self)


@dataclass
class BaselineRun:
    """A policy/environment pair plus immutable configuration provenance."""

    environment: DTMarlEnv
    policy: object
    metadata: BaselineMetadata


def load_validated_r2_config(config_path: Path = VALIDATED_R2_CONFIG) -> Sprint6Config:
    """Load R2's recorded configuration without loading any model weights.

    Evaluation uses the disjoint held-out trace range, matching the validated
    evaluation convention even if the saved training config was written with a
    narrower training window.
    """
    with open(config_path, encoding="utf-8") as handle:
        raw = json.load(handle)["config"]
    cfg = Sprint6Config(
        env=EnvConfig(**raw["env"]),
        reward=RewardConfig(**raw["reward"]),
        mappo=MappoConfig(**raw["mappo"]),
    )
    cfg.env.start_frac_lo = HELD_OUT_START_FRACTION
    cfg.env.start_frac_hi = 1.0
    cfg.env.random_episode_start = False
    return cfg


def disable_prediction_and_risk(cfg: Sprint6Config) -> Sprint6Config:
    """Apply the required non-predictive baseline configuration in one place."""
    cfg.env.risk_source = "zero"
    cfg.env.dest_w_risk = 0.0
    cfg.reward.P_risk_expose = 0.0
    return cfg


def _metadata(name: str, cfg: Sprint6Config, seed: int,
              checkpoint: Optional[Path] = None) -> BaselineMetadata:
    if name == BASELINE_REACTIVE:
        return BaselineMetadata(
            baseline_name=name, risk_source="zero", uses_digital_twin=False,
            uses_prediction=False, uses_uncertainty=False, uses_marl=False,
            seed=int(seed), node_count=cfg.env.n_edge_nodes,
            task_count=cfg.env.n_tasks, recovery_mode="none",
            training_loss_status="N/A (heuristic baseline)", checkpoint="",
        )
    if name == BASELINE_DETECTION:
        return BaselineMetadata(
            baseline_name=name, risk_source="zero", uses_digital_twin=False,
            uses_prediction=False, uses_uncertainty=False, uses_marl=False,
            seed=int(seed), node_count=cfg.env.n_edge_nodes,
            task_count=cfg.env.n_tasks,
            recovery_mode="observed failure -> immediate requeue/restart",
            training_loss_status="N/A (heuristic baseline)", checkpoint="",
        )
    if name == BASELINE_SINGLE_AGENT:
        return BaselineMetadata(
            baseline_name=name, risk_source=cfg.env.risk_source,
            uses_digital_twin=True, uses_prediction=(cfg.env.risk_source != "zero"),
            uses_uncertainty=False, uses_marl=True, seed=int(seed),
            node_count=cfg.env.n_edge_nodes, task_count=cfg.env.n_tasks,
            recovery_mode="environment default", training_loss_status=(
                "single-agent PPO update CSV required after manual training"),
            checkpoint="" if checkpoint is None else str(checkpoint),
        )
    raise ValueError(f"unknown Sprint 9 baseline: {name}")


def build_baseline(name: str, *, seed: int = 20260818,
                   single_agent_checkpoint: Optional[str | Path] = None,
                   config_path: Path = VALIDATED_R2_CONFIG) -> BaselineRun:
    """Build a Sprint 9 baseline with a shared R2 replay environment.

    ``single_agent_checkpoint`` is accepted only for the independently trained
    single-agent PPO implementation.  Passing the MAPPO checkpoint is rejected
    by the single-agent loader because it has a distinct checkpoint kind.
    """
    cfg = load_validated_r2_config(config_path)
    checkpoint = (None if single_agent_checkpoint is None
                  else Path(single_agent_checkpoint))

    if name == BASELINE_REACTIVE:
        disable_prediction_and_risk(cfg)
        env = DTMarlEnv(cfg.env, cfg.reward)
        policy = ReactiveThresholdPolicy()
    elif name == BASELINE_DETECTION:
        disable_prediction_and_risk(cfg)
        env = DTMarlEnv(cfg.env, cfg.reward, detection_recovery=True)
        policy = DetectionRecoveryPolicy()
    elif name == BASELINE_SINGLE_AGENT:
        if checkpoint is None:
            raise ValueError(
                "single-agent-ppo evaluation requires a separately trained "
                "single-agent checkpoint; Sprint 9 does not use MAPPO weights")
        from marl.single_agent_ppo import SingleAgentPPO, SingleAgentPolicy

        env = DTMarlEnv(cfg.env, cfg.reward)
        agent = SingleAgentPPO.load(checkpoint, device="cpu")
        if agent.state_dim != env.state_dim or agent.n_nodes != env.n_agents:
            raise ValueError("single-agent checkpoint is incompatible with the R2 environment")
        policy = SingleAgentPolicy(agent, greedy=True)
    else:
        raise ValueError(f"unknown Sprint 9 baseline {name!r}; choose {SUPPORTED_BASELINES}")

    return BaselineRun(env, policy, _metadata(name, cfg, seed, checkpoint))


def deterministic_starts(run: BaselineRun, episodes: int) -> list[int]:
    """The exact held-out starts shared by every Sprint 9 baseline."""
    if episodes < 1:
        raise ValueError("episodes must be at least one")
    return episode_starts(run.environment, episodes, seed=run.metadata.seed)


def run_baseline(run: BaselineRun, starts: Sequence[int], *, observer=None) -> list[dict]:
    """Evaluate one policy on caller-provided deterministic starts.

    All metric values returned are the environment's real trace-driven values.
    The optional observer is an instrumentation-only hook used by
    :mod:`training.sprint9_metrics`; it cannot choose actions.
    """
    rows = []
    for episode, start in enumerate(starts):
        metrics, _, _, _ = run_episode(
            run.environment, run.policy, int(start),
            seed=run.metadata.seed + episode, observer=observer)
        rows.append(metrics)
    return rows


__all__ = [
    "BASELINE_REACTIVE", "BASELINE_DETECTION", "BASELINE_SINGLE_AGENT",
    "SUPPORTED_BASELINES", "VALIDATED_R2_CONFIG", "BaselineMetadata",
    "BaselineRun", "load_validated_r2_config", "disable_prediction_and_risk",
    "build_baseline", "deterministic_starts", "run_baseline",
]
