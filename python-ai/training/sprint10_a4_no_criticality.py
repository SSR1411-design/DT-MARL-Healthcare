"""Sprint 10 A4: inference-only R2 counterfactual without criticality awareness.

This evaluator replays the validated R2 trace and uses the frozen greedy MAPPO
checkpoint.  It changes only clinical-criticality influence: the two clinical
task observation channels are hard-zeroed, focus/VM/admission ordering uses a
stable non-clinical FIFO key, and the two criticality reward multipliers are
zero.  Patient/task records remain intact for outcome reporting.

Do not run this file until the implementation has been reviewed and approved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.config import ACTION_NAMES
from marl.env import DTMarlEnv
from marl.mappo import MAPPO, MappoPolicy
from marl.rollout import episode_starts, run_episode
from marl.sprint9_benchmarks import load_validated_r2_config
from training.sprint9_metrics import (
    MIGRATION_FIELDS as SPRINT9_MIGRATION_FIELDS,
    NODE_TICK_FIELDS as SPRINT9_NODE_TICK_FIELDS,
    PREDICTION_USAGE_FIELDS as SPRINT9_PREDICTION_USAGE_FIELDS,
    RECOVERY_FIELDS as SPRINT9_RECOVERY_FIELDS,
    STEP_REWARD_FIELDS as SPRINT9_STEP_REWARD_FIELDS,
    TASK_EVENT_FIELDS as SPRINT9_TASK_EVENT_FIELDS,
    TASK_METADATA_FIELDS as SPRINT9_TASK_METADATA_FIELDS,
    Sprint9Observer,
    _EpisodeTelemetry,
)


EXPERIMENT_NAME = "a4-no-criticality"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "sprint10" / "a4_no_criticality"
VALIDATED_CHECKPOINT = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_best.pth"
VALIDATED_CONFIG = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_config.json"
OOF_PREDICTION_ARTIFACT = ROOT / "saved_models" / "failure_predictor_oof.npz"

VALIDATED_CHECKPOINT_SHA256 = "F7B06FF8AE3D9C3288004101A8329F203D09FC5E9A82D88957DAF696C24A8BEE"
VALIDATED_CONFIG_SHA256 = "E37BEA225B810F412708547C951670385F8D0A429D74DD945D0EEF0EA2B07DCD"
OOF_PREDICTION_ARTIFACT_SHA256 = "F19B1FB5EFFDBC77498BB35C8C82B96BD091027F551CC0575A948F30EFB66950"
HELD_OUT_START_FRACTION = 0.70
EVALUATION_SEED = 20260818
FIXED_HELD_OUT_STARTS = (491, 521, 550, 580, 609, 639, 668, 698)
UNCERTAINTY_GATING_PROTOCOL = "reserved-zero-observation-channel; no-active-R2-gate"
CRITICALITY_SCHEDULER = "fifo-arrival-step-then-task-id"
INTEGRITY_FILENAME = "sprint10_a4_integrity.json"
README_FILENAME = "README.md"

# The fixed R2 observation is: 15 node fields, then 8 focus-task fields.
# The clinical severity and Sprint-5 priority entries must remain present for
# checkpoint compatibility, but contain no clinical signal in A4.
TASK_SEVERITY_OBSERVATION_INDEX = 16
TASK_PRIORITY_OBSERVATION_INDEX = 17

METADATA_FIELDS = [
    "run_id", "baseline_name", "experiment", "episode", "episode_start_tick", "risk_source",
    "uses_digital_twin", "uses_prediction", "uses_uncertainty", "uses_marl",
    "seed", "node_count", "task_count", "recovery_mode", "training_loss_status",
    "checkpoint", "checkpoint_sha256", "ablation_id", "decision_mechanism",
    "mappo_action_selection_used", "uses_digital_twin_trace",
    "criticality_awareness_enabled", "criticality_observation_features_zero",
    "criticality_scheduler", "criticality_weight", "criticality_migration_weight",
    "uncertainty_component_preserved", "uncertainty_gating_protocol",
    "validated_r2_config", "validated_r2_config_sha256", "oof_prediction_artifact_sha256",
]
PREDICTION_USAGE_FIELDS = [
    *SPRINT9_PREDICTION_USAGE_FIELDS,
    "oof_prediction_artifact_sha256", "destination_risk_weight",
    "risk_exposure_penalty", "criticality_weight", "criticality_migration_weight",
    "criticality_observation_features_zero", "uncertainty_component_preserved",
    "uncertainty_gating_protocol",
]
EPISODE_FIELDS = [
    "run_id", "baseline_name", "episode", "episode_start_tick", "episode_end_tick",
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
    "prediction_used", "uncertainty_used", "training_loss_status",
]

FILE_SPECS = {
    "metadata": ("sprint10_a4_metadata.csv", METADATA_FIELDS),
    "node_ticks": ("sprint10_a4_node_ticks.csv", SPRINT9_NODE_TICK_FIELDS),
    "step_rewards": ("sprint10_a4_step_rewards.csv", SPRINT9_STEP_REWARD_FIELDS),
    "task_events": ("sprint10_a4_task_events.csv", SPRINT9_TASK_EVENT_FIELDS),
    "task_metadata": ("sprint10_a4_task_metadata.csv", SPRINT9_TASK_METADATA_FIELDS),
    "migrations": ("sprint10_a4_migrations.csv", SPRINT9_MIGRATION_FIELDS),
    "recovery_events": ("sprint10_a4_recovery_events.csv", SPRINT9_RECOVERY_FIELDS),
    "prediction_usage": ("sprint10_a4_prediction_usage.csv", PREDICTION_USAGE_FIELDS),
    "episode_metrics": ("sprint10_a4_episode_metrics.csv", EPISODE_FIELDS),
}


def sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest without modifying ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _hash_paths(paths: Sequence[Path]) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in sorted(set(paths))
    }


def protected_sprint8_sprint9_hashes() -> dict[str, str]:
    """Fingerprint all existing Sprint 8/9 evidence read-only."""
    data_root = ROOT / "data"
    paths: list[Path] = []
    if data_root.exists():
        for candidate in data_root.glob("sprint8*"):
            if candidate.is_file():
                paths.append(candidate)
            elif candidate.is_dir():
                paths.extend(path for path in candidate.rglob("*") if path.is_file())
        legacy = data_root / "LEGACY_SYNTHETIC_INVALID_NOT_FOR_RESEARCH_sprint8_metrics.csv"
        if legacy.exists():
            paths.append(legacy)
        sprint9 = data_root / "sprint9"
        if sprint9.exists():
            paths.extend(path for path in sprint9.rglob("*") if path.is_file())
    return _hash_paths(paths)


def _protected_sprint10_hashes(prefix: str) -> dict[str, str]:
    sprint10 = ROOT / "data" / "sprint10"
    if not sprint10.exists():
        return {}
    paths: list[Path] = []
    for candidate in sprint10.glob(f"{prefix}*"):
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(path for path in candidate.rglob("*") if path.is_file())
    return _hash_paths(paths)


def protected_a2_hashes() -> dict[str, str]:
    return _protected_sprint10_hashes("a2")


def protected_a3_hashes() -> dict[str, str]:
    return _protected_sprint10_hashes("a3")


def _require_digest(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise AssertionError(f"{description} SHA-256 does not match its protected value")
    return actual


def verify_protected_inputs() -> dict[str, str]:
    """Verify every frozen R2/model/prediction input before or after evaluation."""
    return {
        "validated_r2_checkpoint": _require_digest(
            VALIDATED_CHECKPOINT, VALIDATED_CHECKPOINT_SHA256,
            "validated R2 MAPPO checkpoint"),
        "validated_r2_config": _require_digest(
            VALIDATED_CONFIG, VALIDATED_CONFIG_SHA256, "validated R2 configuration"),
        "oof_prediction_artifact": _require_digest(
            OOF_PREDICTION_ARTIFACT, OOF_PREDICTION_ARTIFACT_SHA256,
            "OOF failure-prediction artifact"),
    }


class NoCriticalityDTMarlEnv(DTMarlEnv):
    """R2 replay environment with only clinical-criticality influence removed.

    The trace, risk provider, failure/recovery behavior, task generator and
    output task attributes all remain inherited from ``DTMarlEnv``.  The
    ordering below is a stable FIFO tie-breaker over ordinary arrival metadata;
    it deliberately does not inspect severity, priority, vitals, HSI, age, or
    deadline urgency.
    """

    @staticmethod
    def _neutral_task_key(task_index: int, tasks) -> tuple[int, int]:
        task = tasks[task_index]
        return (int(task.arrival_step), int(task.spec.task_id))

    def _refresh_derived(self):
        self._focus = []
        for node in range(self.n_agents):
            live = [index for index in self.residents[node] if self.tasks[index].live]
            self._focus.append(
                min(live, key=lambda index: self._neutral_task_key(index, self.tasks))
                if live else -1)

    def _admit_arrivals(self, step: int):
        arriving = [index for index, task in enumerate(self.tasks)
                    if task.state == "PENDING" and task.arrival_step <= step]
        for index in sorted(arriving, key=lambda item: self._neutral_task_key(item, self.tasks)):
            candidates = [node for node in range(self.n_agents)
                          if self.trace.is_up(node, min(self.tick(step), self.trace.n_ticks - 1))
                          and self._occupancy(node) < self.node_capacity]
            if candidates:
                self.place_task(index, min(candidates, key=lambda node: (self._occupancy(node), node)))

    def _assign_running(self, step: int):
        for node in range(self.n_agents):
            live = [index for index in self.residents[node] if self.tasks[index].live]
            for rank, index in enumerate(sorted(
                    live, key=lambda item: self._neutral_task_key(item, self.tasks))):
                task = self.tasks[index]
                task.state = "RUNNING" if rank < self.cfg.vms_per_node else "QUEUED"
                if task.state == "RUNNING" and task.start_step < 0:
                    task.start_step = step
        live = [index for index in self.cloud_residents if self.tasks[index].live]
        for rank, index in enumerate(sorted(
                live, key=lambda item: self._neutral_task_key(item, self.tasks))):
            task = self.tasks[index]
            task.state = "RUNNING" if rank < self.cfg.cloud_slots else "QUEUED"
            if task.state == "RUNNING" and task.start_step < 0:
                task.start_step = step

    def _observations(self) -> np.ndarray:
        observations = super()._observations()
        observations[:, (TASK_SEVERITY_OBSERVATION_INDEX, TASK_PRIORITY_OBSERVATION_INDEX)] = 0.0
        return observations


def assert_a4_controls(env: DTMarlEnv, observations: np.ndarray | None = None) -> None:
    """Fail closed unless only A4's clinical-criticality paths are disabled."""
    if not isinstance(env, NoCriticalityDTMarlEnv):
        raise AssertionError("A4 requires its no-criticality R2 environment")
    if env.cfg.risk_source != "oof" or env.risk.source != "oof":
        raise AssertionError("A4 must retain the R2 OOF failure-prediction provider")
    if env.cfg.dest_w_risk <= 0.0 or env.rcfg.P_risk_expose <= 0.0:
        raise AssertionError("A4 must retain R2 prediction-driven destination/reward paths")
    if env.rcfg.w_criticality != 0.0 or env.rcfg.w_criticality_migration != 0.0:
        raise AssertionError("A4 criticality reward multipliers must be exactly zero")
    if env.cfg.start_frac_lo != HELD_OUT_START_FRACTION or env.cfg.start_frac_hi != 1.0:
        raise AssertionError("A4 must use the validated R2 held-out start window")
    if env.cfg.random_episode_start:
        raise AssertionError("A4 must consume caller-specified held-out starts in order")
    if not np.isfinite(env.risk.risk[:, env.risk.first_valid_tick:]).all():
        raise AssertionError("A4 OOF prediction matrix is incomplete")
    if not np.any(env.risk.risk[:, env.risk.first_valid_tick:] > 0.0):
        raise AssertionError("A4 OOF risk provider has no active prediction signal")
    if not np.array_equal(env.risk.uncertainty, np.zeros_like(env.risk.uncertainty)):
        raise AssertionError("validated R2 uncertainty contract unexpectedly changed")
    if observations is not None:
        if observations.shape != (env.n_agents, env.obs_dim):
            raise AssertionError("A4 environment observation geometry changed")
        if not np.array_equal(
                observations[:, (TASK_SEVERITY_OBSERVATION_INDEX, TASK_PRIORITY_OBSERVATION_INDEX)],
                np.zeros((env.n_agents, 2))):
            raise AssertionError("A4 MAPPO observation contains clinical-criticality signal")
        if not np.array_equal(observations[:, 13], np.zeros(env.n_agents)):
            raise AssertionError("validated R2 reserved uncertainty channel changed")


class VerifiedMappoPolicy(MappoPolicy):
    """The sole A4 controller: frozen, greedy, multi-agent MAPPO."""

    def __init__(self, agent: MAPPO):
        if not isinstance(agent, MAPPO):
            raise TypeError("A4 requires the validated MAPPO controller")
        super().__init__(agent, name=EXPERIMENT_NAME, greedy=True)
        self.action_selection_calls = 0

    def act(self, env, obs, masks) -> np.ndarray:
        if not isinstance(self.agent, MAPPO) or not self.greedy:
            raise AssertionError("A4 policy is not frozen greedy MAPPO")
        actions = super().act(env, obs, masks)
        if actions.shape != (env.n_agents,) or not np.issubdtype(actions.dtype, np.integer):
            raise AssertionError("MAPPO did not emit the environment joint action")
        if not np.all(masks[np.arange(env.n_agents), actions]):
            raise AssertionError("MAPPO selected a masked action")
        self.action_selection_calls += 1
        return actions


@dataclass(frozen=True)
class A4Metadata:
    checkpoint_sha256: str
    validated_r2_config_sha256: str
    oof_prediction_artifact_sha256: str
    seed: int
    node_count: int
    task_count: int
    baseline_name: str = EXPERIMENT_NAME
    risk_source: str = "oof"
    uses_digital_twin: bool = True
    uses_prediction: bool = True
    uses_uncertainty: bool = False
    uses_marl: bool = True
    recovery_mode: str = "environment default"
    training_loss_status: str = "inference-only; no A4 retraining"

    def as_row(self) -> dict[str, object]:
        return {
            "baseline_name": self.baseline_name,
            "experiment": EXPERIMENT_NAME,
            "risk_source": self.risk_source,
            "uses_digital_twin": self.uses_digital_twin,
            "uses_prediction": self.uses_prediction,
            "uses_uncertainty": self.uses_uncertainty,
            "uses_marl": self.uses_marl,
            "seed": self.seed,
            "node_count": self.node_count,
            "task_count": self.task_count,
            "recovery_mode": self.recovery_mode,
            "training_loss_status": self.training_loss_status,
            "checkpoint": str(VALIDATED_CHECKPOINT),
            "checkpoint_sha256": self.checkpoint_sha256,
            "ablation_id": "A4",
            "decision_mechanism": "frozen_greedy_mappo",
            "mappo_action_selection_used": 1,
            "uses_digital_twin_trace": 1,
            "criticality_awareness_enabled": 0,
            "criticality_observation_features_zero": 1,
            "criticality_scheduler": CRITICALITY_SCHEDULER,
            "criticality_weight": 0.0,
            "criticality_migration_weight": 0.0,
            "uncertainty_component_preserved": 1,
            "uncertainty_gating_protocol": UNCERTAINTY_GATING_PROTOCOL,
            "validated_r2_config": str(VALIDATED_CONFIG),
            "validated_r2_config_sha256": self.validated_r2_config_sha256,
            "oof_prediction_artifact_sha256": self.oof_prediction_artifact_sha256,
        }


@dataclass
class A4Run:
    environment: NoCriticalityDTMarlEnv
    policy: VerifiedMappoPolicy
    metadata: A4Metadata
    starts: tuple[int, ...]
    input_hashes: Mapping[str, str]


def build_a4_run(device: str = "cpu") -> A4Run:
    """Reconstruct R2, disable only clinical-criticality influence, load MAPPO."""
    input_hashes = verify_protected_inputs()
    cfg = load_validated_r2_config(VALIDATED_CONFIG)
    cfg.reward.w_criticality = 0.0
    cfg.reward.w_criticality_migration = 0.0
    env = NoCriticalityDTMarlEnv(cfg.env, cfg.reward)
    observations, _, _ = env.reset(episode_start_tick=env._min_start, seed=EVALUATION_SEED)
    assert_a4_controls(env, observations)
    starts = tuple(episode_starts(env, len(FIXED_HELD_OUT_STARTS)))
    if starts != FIXED_HELD_OUT_STARTS:
        raise AssertionError(f"A4 held-out starts changed: got {starts}, expected {FIXED_HELD_OUT_STARTS}")
    agent, _ = MAPPO.load(VALIDATED_CHECKPOINT, device=device)
    agent.eval()
    if agent.obs_dim != env.obs_dim or agent.state_dim != env.state_dim or agent.n_agents != env.n_agents:
        raise AssertionError("validated MAPPO dimensions do not match the A4 R2 replay environment")
    metadata = A4Metadata(
        checkpoint_sha256=input_hashes["validated_r2_checkpoint"],
        validated_r2_config_sha256=input_hashes["validated_r2_config"],
        oof_prediction_artifact_sha256=input_hashes["oof_prediction_artifact"],
        seed=EVALUATION_SEED, node_count=env.n_agents, task_count=env.cfg.n_tasks)
    return A4Run(env, VerifiedMappoPolicy(agent), metadata, starts, input_hashes)


class A4CsvBundle:
    """A4-only output writer that never opens an existing output directory."""

    def __init__(self, output_dir: Path):
        if output_dir.exists():
            raise FileExistsError(
                f"Sprint 10 A4 output directory already exists: {output_dir}. "
                "A4 never overwrites or reuses an output directory.")
        self.output_dir = output_dir
        self.paths = {key: output_dir / filename for key, (filename, _) in FILE_SPECS.items()}
        self.integrity_path = output_dir / INTEGRITY_FILENAME
        self.readme_path = output_dir / README_FILENAME
        output_dir.mkdir(parents=True, exist_ok=False)
        self.files = {}
        self.writers = {}
        self.counts = {key: 0 for key in FILE_SPECS}
        for key, path in self.paths.items():
            handle = open(path, "w", newline="", encoding="utf-8")
            self.files[key] = handle
            self.writers[key] = csv.DictWriter(handle, fieldnames=FILE_SPECS[key][1], extrasaction="raise")
            self.writers[key].writeheader()

    def write(self, key: str, row: Mapping[str, object]) -> None:
        self.writers[key].writerow(row)
        self.counts[key] += 1

    def close(self) -> None:
        for handle in self.files.values():
            handle.close()


class A4Observer(Sprint9Observer):
    """Read-only exporter with A4 control and start-order assertions."""

    def __init__(self, bundle: A4CsvBundle, run: A4Run, run_id: str):
        super().__init__(bundle, run, run_id)
        self.consumed_starts: list[int] = []

    def on_episode_start(self, env: DTMarlEnv, start_tick: int, seed: int, policy_name: str) -> None:
        if policy_name != EXPERIMENT_NAME or not isinstance(self.run.policy, VerifiedMappoPolicy):
            raise AssertionError("A4 rollout did not use verified frozen MAPPO")
        if self.episode >= len(self.run.starts):
            raise AssertionError("A4 rollout exceeded the fixed held-out start protocol")
        expected = self.run.starts[self.episode]
        if start_tick != expected or env.t0 != expected:
            raise AssertionError(f"A4 episode {self.episode + 1} consumed {env.t0}, expected {expected}")
        assert_a4_controls(env)
        if env.t0 != expected:
            raise AssertionError("A4 control validation changed the environment start tick")
        self.episode += 1
        self.consumed_starts.append(env.t0)
        self.priority_start = {task.spec.task_id: float(task.spec.priority_at(0.0)) for task in env.tasks}
        self.telemetry = _EpisodeTelemetry({key: 0.0 for key in (
            "link_latency_ms", "power_w", "cpu_utilization_pct",
            "ram_utilization_pct", "bandwidth_utilization_pct")})
        self.bundle.write("metadata", {
            "run_id": self.run_id, **self.metadata.as_row(), "episode": self.episode,
            "episode_start_tick": env.t0, "seed": int(seed),
        })
        self.bundle.write("prediction_usage", {
            **self._base(env), "risk_source": "oof", "prediction_used": 1,
            "uncertainty_used": 0, "oof_loaded": 1, "live_predictor_loaded": 0,
            "offline_only_note": (
                "Leakage-safe OOF risk remains active; R2 uncertainty remains its reserved-zero "
                "channel with no active gate. A4 only removes clinical-criticality influence."),
            "oof_prediction_artifact_sha256": self.metadata.oof_prediction_artifact_sha256,
            "destination_risk_weight": env.cfg.dest_w_risk,
            "risk_exposure_penalty": env.rcfg.P_risk_expose,
            "criticality_weight": env.rcfg.w_criticality,
            "criticality_migration_weight": env.rcfg.w_criticality_migration,
            "criticality_observation_features_zero": 1,
            "uncertainty_component_preserved": 1,
            "uncertainty_gating_protocol": UNCERTAINTY_GATING_PROTOCOL,
        })

    def before_step(self, env: DTMarlEnv, actions: np.ndarray,
                    observations: np.ndarray, action_masks: np.ndarray) -> None:
        assert_a4_controls(env, observations)
        super().before_step(env, actions, observations, action_masks)

    def on_episode_end(self, env: DTMarlEnv, metrics: Mapping[str, object]) -> None:
        for migration in env.migrations:
            tick = env.tick(migration.step)
            self.bundle.write("migrations", {
                **self._base(env), "decision_step": migration.step, "recorded_tick": tick,
                "sim_time_s": float(env.trace.times[tick]), "task_id": migration.task_id,
                "patient_id": migration.patient_id, "clinical_severity": migration.severity,
                "priority": migration.priority, "action": migration.action,
                "action_name": ACTION_NAMES[migration.action], "source_node_id": migration.source_node,
                "destination_node_id": migration.dest_node, "migration_cost": migration.cost,
                "latency_steps": migration.latency_steps, "preemptive": int(migration.preemptive),
                "source_symptomatic": int(migration.source_symptomatic),
                "source_risk": migration.source_risk, "destination_risk": migration.dest_risk,
                "source_failed_within_window": int(bool(migration.source_failed_within_window)),
                "task_survived": int(bool(migration.task_survived)),
                "outcome_scope": "post_episode_evaluation_only",
            })
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
                "destination_observed_healthy": int(healthy), "recovery_tick": recovery.recovery_tick,
                "recovery_time_s": recovery.recovery_time_s,
                "recovery_latency_s": recovery.recovery_latency_s,
                "recovery_succeeded": int(recovery.succeeded), "reason": recovery.reason,
            })
        end_tick = env.tick()
        hsi = np.asarray([task.spec.attributes["hsi"] for task in env.tasks], dtype=float)
        priorities = np.asarray([task.spec.priority_at(env.elapsed_s) for task in env.tasks], dtype=float)
        priority_start = np.asarray(list(self.priority_start.values()), dtype=float)
        mean = lambda key: self.telemetry.sums[key] / max(self.telemetry.samples, 1)
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
            "mean_hsi": float(hsi.mean()), "min_hsi": float(hsi.min()), "max_hsi": float(hsi.max()),
            "mean_priority_at_start": float(priority_start.mean()),
            "mean_priority_at_end": float(priorities.mean()), "risk_source": self.metadata.risk_source,
            "prediction_used": int(self.metadata.uses_prediction),
            "uncertainty_used": int(self.metadata.uses_uncertainty),
            "training_loss_status": self.metadata.training_loss_status,
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


def run_fixed_start_protocol(run: A4Run, observer: A4Observer) -> None:
    """Execute exactly one real inference episode for every validated start."""
    for episode, start in enumerate(run.starts):
        run_episode(run.environment, run.policy, start_tick=start,
                    seed=EVALUATION_SEED + episode, observer=observer)
    if tuple(observer.consumed_starts) != run.starts:
        raise AssertionError(f"A4 consumed {observer.consumed_starts}, expected {list(run.starts)}")
    if run.policy.action_selection_calls <= 0:
        raise AssertionError("A4 exported no frozen MAPPO action selections")


def _assert_a4_output_isolation(output_dir: Path) -> None:
    sprint10_root = (ROOT / "data" / "sprint10").resolve()
    try:
        output_dir.relative_to(sprint10_root)
    except ValueError as exc:
        raise ValueError("A4 outputs must stay below data/sprint10") from exc
    if output_dir.parent != sprint10_root or not output_dir.name.startswith("a4_"):
        raise ValueError("A4 output directory must be a new data/sprint10/a4_* directory")


def _write_readme(bundle: A4CsvBundle, run: A4Run) -> None:
    bundle.readme_path.write_text(
        "# Sprint 10 A4 — No Criticality Awareness\\n\\n"
        "This bundle is an inference-only A4 evaluation of the frozen greedy R2 MAPPO policy. "
        "The Digital Twin trace, OOF failure prediction, R2 task/failure generation, and "
        "uncertainty contract remain unchanged.\\n\\n"
        "A4 preserves clinical values only as post-episode evidence. It zeroes the clinical "
        "severity and priority MAPPO inputs, uses FIFO arrival/task-id ordering rather than "
        "clinical priority for admission/focus/VM scheduling, and sets both criticality reward "
        "multipliers to zero.\\n\\n"
        f"Frozen R2 MAPPO checkpoint SHA-256: `{run.metadata.checkpoint_sha256}`\\n",
        encoding="utf-8")


def write_integrity_manifest(bundle: A4CsvBundle, run: A4Run,
                             sprint8_sprint9_before: Mapping[str, str],
                             sprint8_sprint9_after: Mapping[str, str],
                             a2_before: Mapping[str, str], a2_after: Mapping[str, str],
                             a3_before: Mapping[str, str], a3_after: Mapping[str, str],
                             consumed_starts: Sequence[int]) -> None:
    """Seal A4 only when frozen inputs and all prior evidence remained unchanged."""
    if sprint8_sprint9_before != sprint8_sprint9_after:
        raise AssertionError("Sprint 8/9 evidence changed during A4 evaluation")
    if a2_before != a2_after:
        raise AssertionError("completed A2 evidence changed during A4 evaluation")
    if a3_before != a3_after:
        raise AssertionError("completed A3 evidence changed during A4 evaluation")
    if tuple(consumed_starts) != run.starts:
        raise AssertionError(f"A4 consumed {list(consumed_starts)}, expected {list(run.starts)}")
    input_after = verify_protected_inputs()
    if dict(run.input_hashes) != input_after:
        raise AssertionError("a protected R2 checkpoint/config/prediction input changed")
    artifact_hashes = {path.name: sha256(path) for path in [*bundle.paths.values(), bundle.readme_path]}
    payload = {
        "experiment": EXPERIMENT_NAME,
        "inference_only": True,
        "held_out_starts": list(run.starts),
        "consumed_starts": list(consumed_starts),
        "frozen_r2_mappo_checkpoint": str(VALIDATED_CHECKPOINT),
        "frozen_r2_mappo_checkpoint_sha256_before": run.input_hashes["validated_r2_checkpoint"],
        "frozen_r2_mappo_checkpoint_sha256_after": input_after["validated_r2_checkpoint"],
        "validated_r2_config": str(VALIDATED_CONFIG),
        "validated_r2_config_sha256_before": run.input_hashes["validated_r2_config"],
        "validated_r2_config_sha256_after": input_after["validated_r2_config"],
        "oof_prediction_artifact": str(OOF_PREDICTION_ARTIFACT),
        "oof_prediction_artifact_sha256_before": run.input_hashes["oof_prediction_artifact"],
        "oof_prediction_artifact_sha256_after": input_after["oof_prediction_artifact"],
        "a4_configuration": {
            "decision_mechanism": "frozen_greedy_mappo",
            "mappo_action_selection_used": True,
            "mappo_action_selection_calls": run.policy.action_selection_calls,
            "digital_twin_enabled": True,
            "failure_prediction_enabled": True,
            "risk_source": run.environment.cfg.risk_source,
            "oof_loaded": True,
            "destination_risk_weight": run.environment.cfg.dest_w_risk,
            "risk_exposure_penalty": run.environment.rcfg.P_risk_expose,
            "criticality_awareness_enabled": False,
            "criticality_observation_features_zero": True,
            "criticality_scheduler": CRITICALITY_SCHEDULER,
            "criticality_weight": run.environment.rcfg.w_criticality,
            "criticality_migration_weight": run.environment.rcfg.w_criticality_migration,
            "uncertainty_component_preserved": True,
            "uncertainty_gating_protocol": UNCERTAINTY_GATING_PROTOCOL,
        },
        "protected_sprint8_sprint9_before": dict(sprint8_sprint9_before),
        "protected_sprint8_sprint9_after": dict(sprint8_sprint9_after),
        "protected_a2_before": dict(a2_before), "protected_a2_after": dict(a2_after),
        "protected_a3_before": dict(a3_before), "protected_a3_after": dict(a3_after),
        "protected_outputs_unchanged": True,
        "artifact_sha256": artifact_hashes,
        "rows": bundle.counts,
    }
    with open(bundle.integrity_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_pipeline(args: argparse.Namespace) -> A4CsvBundle:
    """Run the real A4 evaluation. Tests never call this function."""
    output_dir = Path(args.out_dir).resolve()
    _assert_a4_output_isolation(output_dir)
    sprint8_sprint9_before = protected_sprint8_sprint9_hashes()
    a2_before, a3_before = protected_a2_hashes(), protected_a3_hashes()
    run = build_a4_run(device=args.device)
    bundle = A4CsvBundle(output_dir)
    observer = A4Observer(bundle, run, args.run_id or EXPERIMENT_NAME)
    try:
        run_fixed_start_protocol(run, observer)
        sprint8_sprint9_after = protected_sprint8_sprint9_hashes()
        a2_after, a3_after = protected_a2_hashes(), protected_a3_hashes()
        bundle.close()
        _write_readme(bundle, run)
        write_integrity_manifest(
            bundle, run, sprint8_sprint9_before, sprint8_sprint9_after,
            a2_before, a2_after, a3_before, a3_after, observer.consumed_starts)
    except Exception:
        bundle.close()
        raise
    return bundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 10 A4 inference-only no-criticality evaluation")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = run_pipeline(args)
    print("SPRINT 10 A4 real no-criticality evaluation complete")
    print(f"  output: {bundle.output_dir}")
    for key, path in bundle.paths.items():
        print(f"  {path.name:<38s} {bundle.counts[key]:>8d} rows")
    print("  scope: inference-only; frozen MAPPO; criticality influence disabled")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
