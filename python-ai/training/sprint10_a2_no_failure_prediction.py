"""Sprint 10 A2: evaluate frozen MAPPO with failure prediction disabled.

This runner is deliberately an *evaluation-only* experiment.  It loads the
validated R2 checkpoint read-only, reconstructs the recorded R2 environment,
and changes only the three prediction/risk paths:

* the per-node risk provider is hard-zeroed;
* risk is removed from destination selection; and
* resident-risk exposure is removed from the reward.

The Digital Twin trace remains the environment's recorded telemetry and
transition source.  It is not re-run, modified, or used as a prediction
artifact.  The experiment always uses the eight fixed held-out starts from the
validated R2 evaluation protocol.  No training entry point is imported.

Run from ``python-ai`` after the implementation has passed validation::

    python training/sprint10_a2_no_failure_prediction.py

Outputs are written only below ``data/sprint10/a2_no_failure_prediction``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.config import ACTION_NAMES, EnvConfig, MappoConfig, RewardConfig, Sprint6Config
from marl.env import COMPLETED, LOST, DTMarlEnv
from marl.mappo import MAPPO, MappoPolicy
from marl.rollout import episode_starts, run_episode


EXPERIMENT_NAME = "a2-no-failure-prediction"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "sprint10" / "a2_no_failure_prediction"
VALIDATED_CHECKPOINT = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_best.pth"
VALIDATED_CONFIG = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_config.json"

# Kept alongside the Sprint 9 validation guard.  A mismatch is an integrity
# failure, not a reason to silently evaluate a different MAPPO policy.
VALIDATED_CHECKPOINT_SHA256 = "F7B06FF8AE3D9C3288004101A8329F203D09FC5E9A82D88957DAF696C24A8BEE"
HELD_OUT_START_FRACTION = 0.70
EVALUATION_SEED = 20260818
FIXED_HELD_OUT_STARTS = (491, 521, 550, 580, 609, 639, 668, 698)


METADATA_FIELDS = [
    "run_id", "experiment", "episode", "episode_start_tick", "risk_source",
    "prediction_used", "uncertainty_used", "oof_loaded", "live_predictor_loaded",
    "uses_digital_twin_trace", "uses_marl", "seed", "node_count", "task_count",
    "checkpoint", "checkpoint_sha256", "validated_config", "validated_config_sha256",
    "protocol",
]
NODE_TICK_FIELDS = [
    "run_id", "experiment", "episode", "episode_start_tick", "decision_step",
    "recorded_tick", "sim_time_s", "tick_duration_s", "node_id", "link_latency_ms",
    "power_w", "energy_interval_j", "cpu_utilization_pct", "ram_utilization_pct",
    "bandwidth_utilization_pct", "running_tasks_telemetry", "active", "degraded",
    "link_up", "link_bandwidth_mbps", "link_packet_loss_pct", "under_attack",
    "prediction_used", "uncertainty_used",
]
STEP_REWARD_FIELDS = [
    "run_id", "experiment", "episode", "episode_start_tick", "decision_step",
    "recorded_tick_start", "recorded_tick_end", "agent_id", "requested_action",
    "requested_action_name", "executed_action", "executed_action_name", "agent_reward",
    "team_reward", "completed_delta", "lost_delta", "sla_delta", "migration_cost",
    "energy_reward_cost", "progress_fraction", "infeasible_action", "focus_task_id",
]
TASK_EVENT_FIELDS = [
    "run_id", "experiment", "episode", "episode_start_tick", "decision_step",
    "recorded_tick", "sim_time_s", "event_type", "task_id", "patient_id",
    "prior_state", "terminal_state", "node_id", "destination_node_id", "reward_owner",
    "arrival_step", "start_step", "finish_step", "lost_step", "task_latency_s",
    "task_latency_available", "migrations", "reroutes", "hsi", "clinical_severity",
    "priority_at_event", "deadline_breached",
]
TASK_METADATA_FIELDS = [
    "run_id", "experiment", "episode", "episode_start_tick", "task_id", "patient_id",
    "hsi", "vitals_instability", "age_years", "clinical_severity",
    "priority_at_episode_start", "priority_at_episode_end", "arrival_time_s", "deadline_s",
    "final_state", "final_node_id", "migrations", "reroutes",
]
MIGRATION_FIELDS = [
    "run_id", "experiment", "episode", "episode_start_tick", "decision_step",
    "recorded_tick", "sim_time_s", "task_id", "patient_id", "clinical_severity",
    "priority", "action", "action_name", "source_node_id", "destination_node_id",
    "migration_cost", "latency_steps", "preemptive", "source_symptomatic",
    "source_risk", "destination_risk", "source_failed_within_window", "task_survived",
    "outcome_scope",
]
RECOVERY_FIELDS = [
    "run_id", "experiment", "episode", "episode_start_tick", "detection_step",
    "failure_window_start_tick", "failure_window_end_tick", "detection_tick",
    "detection_time_s", "task_id", "patient_id", "source_node_id", "destination_node_id",
    "destination_observed_healthy", "recovery_tick", "recovery_time_s",
    "recovery_latency_s", "recovery_succeeded", "reason",
]
PREDICTION_USAGE_FIELDS = [
    "run_id", "experiment", "episode", "episode_start_tick", "risk_source",
    "prediction_used", "uncertainty_used", "oof_loaded", "live_predictor_loaded",
    "policy_risk_channels_zero", "destination_risk_weight", "risk_exposure_penalty",
    "offline_only_note",
]
EPISODE_FIELDS = [
    "run_id", "experiment", "episode", "episode_start_tick", "episode_end_tick",
    "episode_start_time_s", "episode_end_time_s", "decision_steps", "episode_reward",
    "task_success_rate", "completed", "lost", "unfinished", "avg_task_latency_s",
    "energy_reward_cost", "migrations", "migrations_edge", "migrations_cloud", "reroutes",
    "relocations", "preemptive_relocations", "reactive_relocations",
    "tasks_protected_before_failure", "tasks_lost_on_resident_host",
    "tasks_lost_in_flight", "sla_violations", "failed_critical_tasks",
    "critical_success_rate", "infeasible_actions", "recovery_attempts",
    "recoveries_succeeded", "recoveries_failed", "mean_recovery_latency_s",
    "mean_link_latency_ms", "mean_power_w", "mean_cpu_utilization_pct",
    "mean_ram_utilization_pct", "mean_bandwidth_utilization_pct", "mean_hsi", "min_hsi",
    "max_hsi", "mean_priority_at_start", "mean_priority_at_end", "risk_source",
    "prediction_used", "uncertainty_used",
]

FILE_SPECS = {
    "metadata": ("sprint10_a2_metadata.csv", METADATA_FIELDS),
    "node_ticks": ("sprint10_a2_node_ticks.csv", NODE_TICK_FIELDS),
    "step_rewards": ("sprint10_a2_step_rewards.csv", STEP_REWARD_FIELDS),
    "task_events": ("sprint10_a2_task_events.csv", TASK_EVENT_FIELDS),
    "task_metadata": ("sprint10_a2_task_metadata.csv", TASK_METADATA_FIELDS),
    "migrations": ("sprint10_a2_migrations.csv", MIGRATION_FIELDS),
    "recovery_events": ("sprint10_a2_recovery_events.csv", RECOVERY_FIELDS),
    "prediction_usage": ("sprint10_a2_prediction_usage.csv", PREDICTION_USAGE_FIELDS),
    "episode_metrics": ("sprint10_a2_episode_metrics.csv", EPISODE_FIELDS),
}
INTEGRITY_FILENAME = "sprint10_a2_integrity.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def protected_output_hashes() -> dict[str, str]:
    """Fingerprint every existing Sprint 8/9 output without writing to it."""
    data_root = ROOT / "data"
    paths = []
    if data_root.exists():
        paths.extend(path for path in data_root.glob("sprint8*") if path.is_file())
        legacy = data_root / "LEGACY_SYNTHETIC_INVALID_NOT_FOR_RESEARCH_sprint8_metrics.csv"
        if legacy.exists():
            paths.append(legacy)
        sprint9 = data_root / "sprint9"
        if sprint9.exists():
            paths.extend(path for path in sprint9.rglob("*") if path.is_file())
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in sorted(set(paths))}


def load_validated_r2_config() -> Sprint6Config:
    """Read the validated R2 configuration, then apply only A2 switches."""
    with open(VALIDATED_CONFIG, encoding="utf-8") as handle:
        raw = json.load(handle)["config"]
    cfg = Sprint6Config(
        env=EnvConfig(**raw["env"]),
        reward=RewardConfig(**raw["reward"]),
        mappo=MappoConfig(**raw["mappo"]),
    )
    cfg.env.start_frac_lo = HELD_OUT_START_FRACTION
    cfg.env.start_frac_hi = 1.0
    cfg.env.random_episode_start = False

    # A2's complete mechanical removal of failure prediction/risk.
    cfg.env.risk_source = "zero"
    cfg.env.dest_w_risk = 0.0
    cfg.reward.P_risk_expose = 0.0
    return cfg


def risk_observation_indices(env: DTMarlEnv) -> tuple[int, ...]:
    """Local and neighbour risk feature positions in the fixed 48-D observation."""
    local_risk = 12
    neighbours_begin = 15 + 8
    return (local_risk,) + tuple(neighbours_begin + 5 * index
                                 for index in range(env.topology.degree))


def assert_a2_risk_disabled(env: DTMarlEnv,
                            observations: np.ndarray | None = None) -> None:
    """Fail fast if an A2 environment can consume prediction/risk anywhere.

    This check is deliberately side-effect free.  In particular, it must not
    reset ``env``: rollout observers invoke it after ``run_episode`` has reset
    the environment to a held-out start, and resetting here would replace that
    start with the first trace tick.
    """
    if env.cfg.risk_source != "zero" or env.risk.source != "zero":
        raise AssertionError("A2 requires risk_source='zero' in config and provider")
    if env.cfg.dest_w_risk != 0.0:
        raise AssertionError("A2 destination risk weight must be exactly zero")
    if env.rcfg.P_risk_expose != 0.0:
        raise AssertionError("A2 risk-exposure reward penalty must be exactly zero")
    if not np.array_equal(env.risk.risk, np.zeros_like(env.risk.risk)):
        raise AssertionError("A2 risk provider is not hard-zeroed")
    if not np.array_equal(env.risk.uncertainty, np.zeros_like(env.risk.uncertainty)):
        raise AssertionError("A2 uncertainty channel is not zero")

    if observations is not None:
        indices = risk_observation_indices(env)
        if not np.array_equal(
                observations[:, indices], np.zeros((env.n_agents, len(indices)))):
            raise AssertionError("A2 policy observation contains nonzero risk")


@dataclass
class A2Run:
    environment: DTMarlEnv
    policy: MappoPolicy
    checkpoint: Path
    checkpoint_sha256: str
    config_sha256: str
    starts: tuple[int, ...]


def build_a2_run(device: str = "cpu") -> A2Run:
    """Construct frozen MAPPO + the strictly non-predictive A2 environment."""
    if not VALIDATED_CHECKPOINT.exists():
        raise FileNotFoundError(f"validated MAPPO checkpoint missing: {VALIDATED_CHECKPOINT}")
    if not VALIDATED_CONFIG.exists():
        raise FileNotFoundError(f"validated R2 config missing: {VALIDATED_CONFIG}")
    checkpoint_hash = sha256(VALIDATED_CHECKPOINT)
    if checkpoint_hash != VALIDATED_CHECKPOINT_SHA256:
        raise AssertionError("validated MAPPO checkpoint SHA-256 does not match the protected value")

    cfg = load_validated_r2_config()
    env = DTMarlEnv(cfg.env, cfg.reward)
    # This pre-rollout reset verifies the zero-risk observation shape without
    # affecting any evaluated episode: each run_episode call resets again to
    # its fixed held-out start.
    observations, _, _ = env.reset(episode_start_tick=env._min_start,
                                   seed=EVALUATION_SEED)
    assert_a2_risk_disabled(env, observations)
    starts = tuple(episode_starts(env, len(FIXED_HELD_OUT_STARTS)))
    if starts != FIXED_HELD_OUT_STARTS:
        raise AssertionError(
            f"A2 held-out starts changed: got {starts}, expected {FIXED_HELD_OUT_STARTS}")

    agent, _ = MAPPO.load(VALIDATED_CHECKPOINT, device=device)
    agent.eval()
    if agent.obs_dim != env.obs_dim or agent.state_dim != env.state_dim:
        raise AssertionError("validated MAPPO dimensions do not match the R2 replay environment")
    if agent.n_agents != env.n_agents:
        raise AssertionError("validated MAPPO agent count does not match the R2 replay environment")
    return A2Run(env, MappoPolicy(agent, EXPERIMENT_NAME, greedy=True),
                 VALIDATED_CHECKPOINT, checkpoint_hash, sha256(VALIDATED_CONFIG), starts)


class CsvBundle:
    """A2-only output writer; it refuses to overwrite any existing result by default."""

    def __init__(self, output_dir: Path, overwrite: bool = False):
        self.output_dir = output_dir
        self.paths = {key: output_dir / spec[0] for key, spec in FILE_SPECS.items()}
        self.integrity_path = output_dir / INTEGRITY_FILENAME
        existing = [path for path in [*self.paths.values(), self.integrity_path] if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                "Sprint 10 A2 outputs already exist: " + ", ".join(path.name for path in existing)
                + ". Use --overwrite only to replace this A2 output directory.")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.files = {}
        self.writers = {}
        self.counts = {key: 0 for key in FILE_SPECS}
        for key, path in self.paths.items():
            handle = open(path, "w", newline="", encoding="utf-8")
            self.files[key] = handle
            self.writers[key] = csv.DictWriter(handle, fieldnames=FILE_SPECS[key][1],
                                                extrasaction="raise")
            self.writers[key].writeheader()

    def write(self, key: str, row: Mapping[str, object]) -> None:
        self.writers[key].writerow(row)
        self.counts[key] += 1

    def close(self) -> None:
        for handle in self.files.values():
            handle.close()


class A2Observer:
    """Read-only rollout observer that exports actual trace and environment events."""

    def __init__(self, bundle: CsvBundle, run: A2Run, run_id: str):
        self.bundle = bundle
        self.run = run
        self.run_id = run_id
        self.episode = 0
        self.consumed_starts: list[int] = []
        self.task_before: Dict[int, dict] = {}
        self.priority_start: Dict[int, float] = {}
        self.telemetry: Dict[str, float] = {}
        self.telemetry_samples = 0

    def _base(self, env: DTMarlEnv) -> dict:
        return dict(run_id=self.run_id, experiment=EXPERIMENT_NAME,
                    episode=self.episode, episode_start_tick=env.t0)

    def on_episode_start(self, env: DTMarlEnv, start_tick: int, seed: int,
                         policy_name: str) -> None:
        if policy_name != EXPERIMENT_NAME:
            raise AssertionError("A2 rollout used an unexpected policy adapter")
        if self.episode >= len(self.run.starts):
            raise AssertionError("A2 rollout exceeded the fixed held-out start protocol")
        expected_start = self.run.starts[self.episode]
        if start_tick != expected_start:
            raise AssertionError(
                f"A2 episode {self.episode + 1} requested start {start_tick}, "
                f"expected {expected_start}")
        if start_tick != env.t0:
            raise AssertionError("rollout start tick differs from environment state")
        assert_a2_risk_disabled(env)
        # Keep this second check adjacent to the non-mutating risk assertion.
        # It turns any future validation-side reset into an immediate failure
        # before a row can claim a misleading episode start.
        if start_tick != env.t0:
            raise AssertionError("A2 risk validation changed the environment start tick")
        self.consumed_starts.append(env.t0)
        self.episode += 1
        self.priority_start = {
            task.spec.task_id: float(task.spec.priority_at(0.0)) for task in env.tasks
        }
        self.telemetry = {key: 0.0 for key in (
            "link_latency_ms", "power_w", "cpu_utilization_pct",
            "ram_utilization_pct", "bandwidth_utilization_pct")}
        self.telemetry_samples = 0
        self.bundle.write("metadata", {
            **self._base(env), "risk_source": "zero", "prediction_used": 0,
            "uncertainty_used": 0, "oof_loaded": 0, "live_predictor_loaded": 0,
            "uses_digital_twin_trace": 1, "uses_marl": 1, "seed": int(seed),
            "node_count": env.n_agents, "task_count": env.cfg.n_tasks,
            "checkpoint": str(self.run.checkpoint),
            "checkpoint_sha256": self.run.checkpoint_sha256,
            "validated_config": str(VALIDATED_CONFIG),
            "validated_config_sha256": self.run.config_sha256,
            "protocol": "validated R2 held-out starts; frozen greedy MAPPO; A2 risk disabled",
        })
        self.bundle.write("prediction_usage", {
            **self._base(env), "risk_source": "zero", "prediction_used": 0,
            "uncertainty_used": 0, "oof_loaded": 0, "live_predictor_loaded": 0,
            "policy_risk_channels_zero": 1, "destination_risk_weight": env.cfg.dest_w_risk,
            "risk_exposure_penalty": env.rcfg.P_risk_expose,
            "offline_only_note": (
                "A2 loads no OOF or live predictor. Risk is hard-zeroed in the policy "
                "observation; destination-risk scoring and risk-exposure reward are disabled."),
        })

    def before_step(self, env: DTMarlEnv, actions: np.ndarray,
                    observations: np.ndarray, action_masks: np.ndarray) -> None:
        del actions, action_masks
        assert_a2_risk_disabled(env, observations)
        step = env.step_idx
        start, end = env.tick(step), env.tick(step + 1)
        self.task_before = {
            task.spec.task_id: dict(state=task.state, node=task.node,
                                    destination=task.dest, reward_owner=task.reward_owner)
            for task in env.tasks
        }
        for tick in range(start, end):
            duration_s = float(env.trace.times[tick + 1] - env.trace.times[tick])
            for node in range(env.n_agents):
                power_w = float(env.trace.ch("energy")[node, tick])
                row = {
                    **self._base(env), "decision_step": step, "recorded_tick": tick,
                    "sim_time_s": float(env.trace.times[tick]), "tick_duration_s": duration_s,
                    "node_id": node,
                    "link_latency_ms": float(env.trace.ch("linkLatencyMs")[node, tick]),
                    "power_w": power_w, "energy_interval_j": power_w * duration_s,
                    "cpu_utilization_pct": float(env.trace.ch("cpu")[node, tick]),
                    "ram_utilization_pct": float(env.trace.ch("ram")[node, tick]),
                    "bandwidth_utilization_pct": float(env.trace.ch("bandwidth")[node, tick]),
                    "running_tasks_telemetry": float(env.trace.ch("runningTasks")[node, tick]),
                    "active": int(env.trace.ch("active")[node, tick] >= 0.5),
                    "degraded": int(env.trace.ch("degraded")[node, tick] >= 0.5),
                    "link_up": int(env.trace.ch("linkUp")[node, tick] >= 0.5),
                    "link_bandwidth_mbps": float(env.trace.ch("linkBandwidthMbps")[node, tick]),
                    "link_packet_loss_pct": float(env.trace.ch("linkPacketLoss")[node, tick]),
                    "under_attack": int(env.trace.ch("underAttack")[node, tick] >= 0.5),
                    "prediction_used": 0, "uncertainty_used": 0,
                }
                self.bundle.write("node_ticks", row)
                for key in self.telemetry:
                    self.telemetry[key] += float(row[key])
                self.telemetry_samples += 1

    def after_step(self, env: DTMarlEnv, actions: np.ndarray, rewards: np.ndarray,
                   done: bool, info: Mapping[str, object]) -> None:
        del done
        step = env.step_idx - 1
        start, end = env.tick(step), env.tick()
        team_reward = float(np.sum(rewards))
        for agent, event in enumerate(info["events"]):
            requested, executed = int(actions[agent]), int(event["action"])
            self.bundle.write("step_rewards", {
                **self._base(env), "decision_step": step, "recorded_tick_start": start,
                "recorded_tick_end": end, "agent_id": agent, "requested_action": requested,
                "requested_action_name": ACTION_NAMES[requested], "executed_action": executed,
                "executed_action_name": ACTION_NAMES[executed],
                "agent_reward": float(rewards[agent]), "team_reward": team_reward,
                "completed_delta": int(event["completed"]), "lost_delta": int(event["lost"]),
                "sla_delta": int(event["sla"]), "migration_cost": float(event["migration_cost"]),
                "energy_reward_cost": float(event["energy"]),
                "progress_fraction": float(event["progress"]),
                "infeasible_action": int(bool(event["infeasible"])),
                "focus_task_id": "" if event["task_id"] < 0 else int(event["task_id"]),
            })
        for task in env.tasks:
            before = self.task_before[task.spec.task_id]
            if task.state not in (COMPLETED, LOST) or before["state"] in (COMPLETED, LOST):
                continue
            latency, latency_available = "", 0
            if task.state == COMPLETED:
                latency = ((task.finish_step - task.arrival_step) * env.dt
                           + task.wan_latency_ms / 1000.0)
                latency_available = 1
            self.bundle.write("task_events", {
                **self._base(env), "decision_step": step, "recorded_tick": end,
                "sim_time_s": float(env.trace.times[end]), "event_type": task.state,
                "task_id": task.spec.task_id, "patient_id": task.spec.patient_id,
                "prior_state": before["state"], "terminal_state": task.state,
                "node_id": task.node,
                "destination_node_id": task.dest if task.dest >= 0 else "",
                "reward_owner": task.reward_owner, "arrival_step": task.arrival_step,
                "start_step": task.start_step if task.start_step >= 0 else "",
                "finish_step": task.finish_step if task.finish_step >= 0 else "",
                "lost_step": task.lost_step if task.lost_step >= 0 else "",
                "task_latency_s": latency, "task_latency_available": latency_available,
                "migrations": task.migrations, "reroutes": task.reroutes,
                "hsi": float(task.spec.attributes["hsi"]),
                "clinical_severity": float(task.spec.severity),
                "priority_at_event": float(task.spec.priority_at(env.elapsed_s)),
                "deadline_breached": int(task.deadline_breached),
            })

    def on_episode_end(self, env: DTMarlEnv, metrics: Mapping[str, object]) -> None:
        for migration in env.migrations:
            tick = env.tick(migration.step)
            self.bundle.write("migrations", {
                **self._base(env), "decision_step": migration.step, "recorded_tick": tick,
                "sim_time_s": float(env.trace.times[tick]), "task_id": migration.task_id,
                "patient_id": migration.patient_id, "clinical_severity": migration.severity,
                "priority": migration.priority, "action": migration.action,
                "action_name": migration.action_name, "source_node_id": migration.source_node,
                "destination_node_id": migration.dest_node, "migration_cost": migration.cost,
                "latency_steps": migration.latency_steps, "preemptive": int(migration.preemptive),
                "source_symptomatic": int(migration.source_symptomatic),
                "source_risk": migration.source_risk, "destination_risk": migration.dest_risk,
                "source_failed_within_window": int(bool(migration.source_failed_within_window)),
                "task_survived": int(bool(migration.task_survived)),
                "outcome_scope": "post_episode_evaluation_only",
            })
        # A2 leaves the validated R2 recovery semantics unchanged.  This file
        # is intentionally header-only when no real recovery record exists.
        for recovery in env.recovery_records:
            healthy = (recovery.succeeded and recovery.destination_node >= 0
                       and env.trace.is_up(recovery.destination_node, recovery.recovery_tick))
            self.bundle.write("recovery_events", {
                **self._base(env), "detection_step": recovery.detection_step,
                "failure_window_start_tick": recovery.failure_window_start_tick,
                "failure_window_end_tick": recovery.failure_window_end_tick,
                "detection_tick": recovery.detection_tick,
                "detection_time_s": recovery.detection_time_s, "task_id": recovery.task_id,
                "patient_id": recovery.patient_id, "source_node_id": recovery.source_node,
                "destination_node_id": recovery.destination_node if recovery.destination_node >= 0 else "",
                "destination_observed_healthy": int(healthy),
                "recovery_tick": recovery.recovery_tick, "recovery_time_s": recovery.recovery_time_s,
                "recovery_latency_s": recovery.recovery_latency_s,
                "recovery_succeeded": int(recovery.succeeded), "reason": recovery.reason,
            })
        end_tick = env.tick()
        hsi = np.asarray([task.spec.attributes["hsi"] for task in env.tasks], dtype=float)
        priorities = np.asarray(
            [task.spec.priority_at(env.elapsed_s) for task in env.tasks], dtype=float)
        priority_start = np.asarray(list(self.priority_start.values()), dtype=float)
        mean = lambda key: self.telemetry[key] / max(self.telemetry_samples, 1)
        self.bundle.write("episode_metrics", {
            **self._base(env), "episode_end_tick": end_tick,
            "episode_start_time_s": float(env.trace.times[env.t0]),
            "episode_end_time_s": float(env.trace.times[end_tick]),
            "decision_steps": metrics["steps"], "episode_reward": metrics["episode_reward"],
            "task_success_rate": metrics["task_success_rate"], "completed": metrics["completed"],
            "lost": metrics["lost"], "unfinished": metrics["unfinished"],
            "avg_task_latency_s": metrics["avg_task_latency_s"],
            "energy_reward_cost": metrics["energy_cost"], "migrations": metrics["migrations"],
            "migrations_edge": metrics["migrations_edge"],
            "migrations_cloud": metrics["migrations_cloud"], "reroutes": metrics["reroutes"],
            "relocations": metrics["relocations"],
            "preemptive_relocations": metrics["preemptive_relocations"],
            "reactive_relocations": metrics["reactive_relocations"],
            "tasks_protected_before_failure": metrics["tasks_protected_before_failure"],
            "tasks_lost_on_resident_host": metrics["tasks_lost_on_resident_host"],
            "tasks_lost_in_flight": metrics["tasks_lost_in_flight"],
            "sla_violations": metrics["sla_violations"],
            "failed_critical_tasks": metrics["failed_critical_tasks"],
            "critical_success_rate": metrics["critical_success_rate"],
            "infeasible_actions": metrics["infeasible_actions"],
            "recovery_attempts": metrics["recovery_attempts"],
            "recoveries_succeeded": metrics["recoveries_succeeded"],
            "recoveries_failed": metrics["recoveries_failed"],
            "mean_recovery_latency_s": metrics["mean_recovery_latency_s"],
            "mean_link_latency_ms": mean("link_latency_ms"), "mean_power_w": mean("power_w"),
            "mean_cpu_utilization_pct": mean("cpu_utilization_pct"),
            "mean_ram_utilization_pct": mean("ram_utilization_pct"),
            "mean_bandwidth_utilization_pct": mean("bandwidth_utilization_pct"),
            "mean_hsi": float(hsi.mean()), "min_hsi": float(hsi.min()),
            "max_hsi": float(hsi.max()), "mean_priority_at_start": float(priority_start.mean()),
            "mean_priority_at_end": float(priorities.mean()), "risk_source": "zero",
            "prediction_used": 0, "uncertainty_used": 0,
        })
        for task in env.tasks:
            attrs = task.spec.attributes
            self.bundle.write("task_metadata", {
                **self._base(env), "task_id": task.spec.task_id, "patient_id": task.spec.patient_id,
                "hsi": float(attrs["hsi"]), "vitals_instability": float(attrs["vitalsInstability"]),
                "age_years": float(attrs["age"]), "clinical_severity": float(task.spec.severity),
                "priority_at_episode_start": self.priority_start[task.spec.task_id],
                "priority_at_episode_end": float(task.spec.priority_at(env.elapsed_s)),
                "arrival_time_s": task.spec.arrival_time, "deadline_s": task.spec.deadline,
                "final_state": task.state, "final_node_id": task.node,
                "migrations": task.migrations, "reroutes": task.reroutes,
            })


def write_integrity_manifest(bundle: CsvBundle, run: A2Run,
                             protected_before: Mapping[str, str],
                             protected_after: Mapping[str, str],
                             consumed_starts: Sequence[int]) -> None:
    if protected_before != protected_after:
        raise AssertionError("Sprint 8/9 outputs changed during A2 evaluation")
    checkpoint_after = sha256(run.checkpoint)
    if checkpoint_after != run.checkpoint_sha256:
        raise AssertionError("validated MAPPO checkpoint changed during A2 evaluation")
    if tuple(consumed_starts) != run.starts:
        raise AssertionError(
            f"A2 environment consumed starts {list(consumed_starts)}, "
            f"expected {list(run.starts)}")
    payload = {
        "experiment": EXPERIMENT_NAME,
        "checkpoint": str(run.checkpoint),
        "checkpoint_sha256_before": run.checkpoint_sha256,
        "checkpoint_sha256_after": checkpoint_after,
        "validated_checkpoint_sha256": VALIDATED_CHECKPOINT_SHA256,
        "validated_config": str(VALIDATED_CONFIG),
        "validated_config_sha256": run.config_sha256,
        "held_out_starts": list(run.starts),
        "consumed_starts": list(consumed_starts),
        "a2_configuration": {
            "risk_source": run.environment.cfg.risk_source,
            "destination_risk_weight": run.environment.cfg.dest_w_risk,
            "risk_exposure_penalty": run.environment.rcfg.P_risk_expose,
            "prediction_used": False,
            "oof_loaded": False,
            "live_predictor_loaded": False,
        },
        "protected_sprint8_sprint9_before": dict(protected_before),
        "protected_sprint8_sprint9_after": dict(protected_after),
        "protected_outputs_unchanged": True,
        "rows": bundle.counts,
    }
    with open(bundle.integrity_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_fixed_start_protocol(run: A2Run, observer: A2Observer) -> None:
    """Execute and prove the fixed held-out resets in their validated order."""
    for episode, start in enumerate(run.starts):
        run_episode(run.environment, run.policy, start_tick=start,
                    seed=EVALUATION_SEED + episode, observer=observer)
    if tuple(observer.consumed_starts) != run.starts:
        raise AssertionError(
            f"A2 environment consumed starts {observer.consumed_starts}, "
            f"expected {list(run.starts)}")


def run_pipeline(args: argparse.Namespace) -> CsvBundle:
    """Run exactly the validated fixed-start protocol; no training occurs."""
    protected_before = protected_output_hashes()
    run = build_a2_run(device=args.device)
    bundle = CsvBundle(Path(args.out_dir).resolve(), overwrite=args.overwrite)
    observer = A2Observer(bundle, run, args.run_id or EXPERIMENT_NAME)
    try:
        run_fixed_start_protocol(run, observer)
        protected_after = protected_output_hashes()
        write_integrity_manifest(bundle, run, protected_before, protected_after,
                                 observer.consumed_starts)
    except Exception:
        bundle.close()
        raise
    bundle.close()
    return bundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sprint 10 A2: fixed-start, no-failure-prediction MAPPO evaluation")
    parser.add_argument("--device", default="cpu",
                        help="MAPPO inference device; default cpu")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="A2-only result directory")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace only existing Sprint 10 A2 result files")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = run_pipeline(args)
    print("SPRINT 10 A2 real evaluation export complete")
    print(f"  checkpoint (verified read-only): {VALIDATED_CHECKPOINT}")
    print(f"  held-out starts:                 {list(FIXED_HELD_OUT_STARTS)}")
    print(f"  output:                          {Path(args.out_dir).resolve()}")
    for key, path in bundle.paths.items():
        print(f"  {path.name:<38s} {bundle.counts[key]:>8d} rows")
    print("  scope: frozen MAPPO inference only; no training or checkpoint writes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
