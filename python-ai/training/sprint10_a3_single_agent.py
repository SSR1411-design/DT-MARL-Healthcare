"""Sprint 10 A3: evaluate the validated single-agent PPO without MAPPO.

This is an inference-only ablation.  It replays the validated R2 held-out
trace window with the unchanged Digital Twin, OOF risk provider, destination
risk scoring, criticality-weighted reward, and task/environment configuration.
The only decision-mechanism change is that the frozen Sprint 9 centralized
single-agent PPO replaces MAPPO's per-agent action selection.

The existing R2 uncertainty channel is preserved exactly: it is a reserved,
zero-valued observation feature and the validated R2 code has no uncertainty
gate.  A3 must not invent one merely to label this ablation as uncertainty
aware.

Run from ``python-ai`` only after implementation tests have passed::

    python training/sprint10_a3_single_agent.py \
        --out-dir data/sprint10/a3_single_agent
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

from marl.env import DTMarlEnv
from marl.rollout import episode_starts, run_episode
from marl.single_agent_ppo import SingleAgentPPO, SingleAgentPolicy
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


EXPERIMENT_NAME = "a3-single-agent-ppo"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "sprint10" / "a3_single_agent"
VALIDATED_R2_CHECKPOINT = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_best.pth"
VALIDATED_R2_CONFIG = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_config.json"
SINGLE_AGENT_CHECKPOINT = (
    ROOT / "saved_models" / "single_agent" / "sprint9" / "sprint9_single_agent_best.pth")
SINGLE_AGENT_CONFIG = (
    ROOT / "saved_models" / "single_agent" / "sprint9" / "sprint9_single_agent_config.json")
OOF_PREDICTION_ARTIFACT = ROOT / "saved_models" / "failure_predictor_oof.npz"

VALIDATED_R2_CHECKPOINT_SHA256 = "F7B06FF8AE3D9C3288004101A8329F203D09FC5E9A82D88957DAF696C24A8BEE"
VALIDATED_R2_CONFIG_SHA256 = "E37BEA225B810F412708547C951670385F8D0A429D74DD945D0EEF0EA2B07DCD"
SINGLE_AGENT_CHECKPOINT_SHA256 = "223BF33C14CEF2F5EBA93C5F3697E8DDB880E4C56E5A348D82656AF4F1750CB5"
SINGLE_AGENT_CONFIG_SHA256 = "23AD6E51ADD40CDB5784940D757B69EE2D31FC12995C729FD94E740A491CF438"
OOF_PREDICTION_ARTIFACT_SHA256 = "F19B1FB5EFFDBC77498BB35C8C82B96BD091027F551CC0575A948F30EFB66950"

HELD_OUT_START_FRACTION = 0.70
EVALUATION_SEED = 20260818
FIXED_HELD_OUT_STARTS = (491, 521, 550, 580, 609, 639, 668, 698)
UNCERTAINTY_GATING_PROTOCOL = "reserved-zero-observation-channel; no-active-R2-gate"
INTEGRITY_FILENAME = "sprint10_a3_integrity.json"
README_FILENAME = "README.md"


# The Sprint 9 exporter already contains the project's trace/event instrumentation
# for the validated single-agent controller.  A3 retains those schemas and adds
# explicit, auditable policy/control provenance to its per-episode metadata.
METADATA_FIELDS = [
    "run_id", "baseline_name", "episode", "episode_start_tick", "risk_source",
    "uses_digital_twin", "uses_prediction", "uses_uncertainty", "uses_marl",
    "seed", "node_count", "task_count", "recovery_mode", "training_loss_status",
    "checkpoint", "ablation_id", "decision_mechanism", "single_agent_policy_used",
    "mappo_action_selection_used", "uses_digital_twin_trace",
    "criticality_awareness_enabled", "uncertainty_component_preserved",
    "uncertainty_gating_protocol", "validated_r2_config",
    "validated_r2_config_sha256", "single_agent_config",
    "single_agent_config_sha256", "single_agent_checkpoint_sha256",
    "oof_prediction_artifact_sha256",
]
PREDICTION_USAGE_FIELDS = [
    *SPRINT9_PREDICTION_USAGE_FIELDS,
    "oof_prediction_artifact_sha256", "destination_risk_weight",
    "risk_exposure_penalty", "criticality_weight", "criticality_migration_weight",
    "uncertainty_component_preserved", "uncertainty_gating_protocol",
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
    "metadata": ("sprint10_a3_metadata.csv", METADATA_FIELDS),
    "node_ticks": ("sprint10_a3_node_ticks.csv", SPRINT9_NODE_TICK_FIELDS),
    "step_rewards": ("sprint10_a3_step_rewards.csv", SPRINT9_STEP_REWARD_FIELDS),
    "task_events": ("sprint10_a3_task_events.csv", SPRINT9_TASK_EVENT_FIELDS),
    "task_metadata": ("sprint10_a3_task_metadata.csv", SPRINT9_TASK_METADATA_FIELDS),
    "migrations": ("sprint10_a3_migrations.csv", SPRINT9_MIGRATION_FIELDS),
    "recovery_events": ("sprint10_a3_recovery_events.csv", SPRINT9_RECOVERY_FIELDS),
    "prediction_usage": ("sprint10_a3_prediction_usage.csv", PREDICTION_USAGE_FIELDS),
    "episode_metrics": ("sprint10_a3_episode_metrics.csv", EPISODE_FIELDS),
}


def sha256(path: Path) -> str:
    """Return an uppercase SHA-256 digest without modifying ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def protected_output_hashes() -> dict[str, str]:
    """Fingerprint every existing Sprint 8/9 result output read-only."""
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
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in sorted(set(paths))
    }


def protected_a2_hashes() -> dict[str, str]:
    """Fingerprint completed A2 result bundles so A3 cannot change them silently."""
    sprint10 = ROOT / "data" / "sprint10"
    if not sprint10.exists():
        return {}
    paths: list[Path] = []
    for candidate in sprint10.glob("a2*"):
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(path for path in candidate.rglob("*") if path.is_file())
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in sorted(paths)
    }


def _require_digest(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise AssertionError(f"{description} SHA-256 does not match its protected value")
    return actual


def verify_protected_inputs() -> dict[str, str]:
    """Verify every model/config/prediction input A3 is allowed to consume."""
    return {
        "validated_r2_checkpoint": _require_digest(
            VALIDATED_R2_CHECKPOINT, VALIDATED_R2_CHECKPOINT_SHA256,
            "validated R2 MAPPO checkpoint"),
        "validated_r2_config": _require_digest(
            VALIDATED_R2_CONFIG, VALIDATED_R2_CONFIG_SHA256, "validated R2 configuration"),
        "single_agent_checkpoint": _require_digest(
            SINGLE_AGENT_CHECKPOINT, SINGLE_AGENT_CHECKPOINT_SHA256,
            "Sprint 9 single-agent PPO checkpoint"),
        "single_agent_config": _require_digest(
            SINGLE_AGENT_CONFIG, SINGLE_AGENT_CONFIG_SHA256,
            "Sprint 9 single-agent PPO configuration"),
        "oof_prediction_artifact": _require_digest(
            OOF_PREDICTION_ARTIFACT, OOF_PREDICTION_ARTIFACT_SHA256,
            "OOF failure-prediction artifact"),
    }


def _load_single_agent_config() -> dict:
    with open(SINGLE_AGENT_CONFIG, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("training_kind") != "single_agent_ppo":
        raise AssertionError("single-agent configuration does not identify the validated PPO")
    if "single_agent_ppo_config" not in config:
        raise AssertionError("single-agent configuration lacks its PPO settings")
    if config.get("train_fraction") != HELD_OUT_START_FRACTION:
        raise AssertionError("single-agent configuration does not record the disjoint R2 split")
    with open(VALIDATED_R2_CONFIG, encoding="utf-8") as handle:
        r2 = json.load(handle)["config"]
    recorded_environment = config.get("environment_config", {})
    if (recorded_environment.get("env") != r2["env"]
            or recorded_environment.get("reward") != r2["reward"]):
        raise AssertionError(
            "single-agent PPO was not recorded against the validated R2 environment/reward")
    return config


def assert_a3_controls(env: DTMarlEnv, observations: np.ndarray | None = None) -> None:
    """Prove that A3 changes no R2 system component other than its controller."""
    if env.cfg.risk_source != "oof" or env.risk.source != "oof":
        raise AssertionError("A3 must retain the R2 OOF failure-prediction provider")
    if env.cfg.dest_w_risk <= 0.0 or env.rcfg.P_risk_expose <= 0.0:
        raise AssertionError("A3 must retain R2 prediction-driven destination/reward paths")
    if env.rcfg.w_criticality <= 0.0 or env.rcfg.w_criticality_migration <= 0.0:
        raise AssertionError("A3 must retain R2 criticality-aware rewards")
    if env.cfg.legacy_dead_migration_criticality:
        raise AssertionError("A3 must retain the validated criticality-aware migration path")
    if env.cfg.start_frac_lo != HELD_OUT_START_FRACTION or env.cfg.start_frac_hi != 1.0:
        raise AssertionError("A3 must use the validated R2 held-out start window")
    if env.cfg.random_episode_start:
        raise AssertionError("A3 must consume caller-specified held-out starts in order")
    if not np.isfinite(env.risk.risk[:, env.risk.first_valid_tick:]).all():
        raise AssertionError("A3 OOF prediction matrix is incomplete")
    if not np.any(env.risk.risk[:, env.risk.first_valid_tick:] > 0.0):
        raise AssertionError("A3 OOF risk provider has no active prediction signal")
    # This is intentionally a preservation check, not a new uncertainty mechanism.
    if not np.array_equal(env.risk.uncertainty, np.zeros_like(env.risk.uncertainty)):
        raise AssertionError("validated R2 uncertainty contract unexpectedly changed")
    if observations is not None:
        if observations.shape != (env.n_agents, env.obs_dim):
            raise AssertionError("A3 environment observation geometry changed")
        # Index 13 is the documented local uncertainty slot in the fixed R2 observation.
        if not np.array_equal(observations[:, 13], np.zeros(env.n_agents)):
            raise AssertionError("validated R2 reserved uncertainty channel changed")


class VerifiedSingleAgentPolicy(SingleAgentPolicy):
    """The sole A3 action adapter; it has no MAPPO fallback or delegation path."""

    def __init__(self, agent: SingleAgentPPO):
        if not isinstance(agent, SingleAgentPPO):
            raise TypeError("A3 requires the project's SingleAgentPPO controller")
        super().__init__(agent, name=EXPERIMENT_NAME, greedy=True)
        self.action_selection_calls = 0

    def act(self, env, obs, masks) -> np.ndarray:
        if not isinstance(self.agent, SingleAgentPPO):
            raise AssertionError("A3 policy was replaced with a non-single-agent controller")
        actions = super().act(env, obs, masks)
        if actions.shape != (env.n_agents,) or not np.issubdtype(actions.dtype, np.integer):
            raise AssertionError("single-agent PPO did not emit the environment joint action")
        if not np.all(masks[np.arange(env.n_agents), actions]):
            raise AssertionError("single-agent PPO selected a masked action")
        self.action_selection_calls += 1
        return actions


@dataclass(frozen=True)
class A3Metadata:
    """Per-episode provenance, including explicit evidence of MARL removal."""

    checkpoint: str
    checkpoint_sha256: str
    validated_r2_config_sha256: str
    single_agent_config_sha256: str
    oof_prediction_artifact_sha256: str
    seed: int
    node_count: int
    task_count: int

    # Sprint9Observer consumes these as attributes while exporting real events.
    # Keep the values in one place so metadata and per-row provenance cannot diverge.
    baseline_name: str = EXPERIMENT_NAME
    risk_source: str = "oof"
    uses_digital_twin: bool = True
    uses_prediction: bool = True
    uses_uncertainty: bool = False
    uses_marl: bool = False
    recovery_mode: str = "environment default"
    training_loss_status: str = "inference-only; no A3 retraining"

    def as_row(self) -> dict[str, object]:
        return {
            "baseline_name": self.baseline_name,
            "risk_source": self.risk_source,
            "uses_digital_twin": self.uses_digital_twin,
            "uses_prediction": self.uses_prediction,
            # R2 has a reserved observation field, but no gating decision path.
            "uses_uncertainty": self.uses_uncertainty,
            "uses_marl": self.uses_marl,
            "seed": self.seed,
            "node_count": self.node_count,
            "task_count": self.task_count,
            "recovery_mode": self.recovery_mode,
            "training_loss_status": self.training_loss_status,
            "checkpoint": self.checkpoint,
            "ablation_id": "A3",
            "decision_mechanism": "single_agent_ppo",
            "single_agent_policy_used": 1,
            "mappo_action_selection_used": 0,
            "uses_digital_twin_trace": 1,
            "criticality_awareness_enabled": 1,
            "uncertainty_component_preserved": 1,
            "uncertainty_gating_protocol": UNCERTAINTY_GATING_PROTOCOL,
            "validated_r2_config": str(VALIDATED_R2_CONFIG),
            "validated_r2_config_sha256": self.validated_r2_config_sha256,
            "single_agent_config": str(SINGLE_AGENT_CONFIG),
            "single_agent_config_sha256": self.single_agent_config_sha256,
            "single_agent_checkpoint_sha256": self.checkpoint_sha256,
            "oof_prediction_artifact_sha256": self.oof_prediction_artifact_sha256,
        }


@dataclass
class A3Run:
    environment: DTMarlEnv
    policy: VerifiedSingleAgentPolicy
    metadata: A3Metadata
    starts: tuple[int, ...]
    input_hashes: Mapping[str, str]


def build_a3_run(device: str = "cpu") -> A3Run:
    """Reconstruct the validated R2 environment and load only single-agent PPO."""
    input_hashes = verify_protected_inputs()
    _load_single_agent_config()
    cfg = load_validated_r2_config(VALIDATED_R2_CONFIG)
    env = DTMarlEnv(cfg.env, cfg.reward)
    observations, _, _ = env.reset(episode_start_tick=env._min_start, seed=EVALUATION_SEED)
    assert_a3_controls(env, observations)
    starts = tuple(episode_starts(env, len(FIXED_HELD_OUT_STARTS)))
    if starts != FIXED_HELD_OUT_STARTS:
        raise AssertionError(f"A3 held-out starts changed: got {starts}, expected {FIXED_HELD_OUT_STARTS}")

    agent = SingleAgentPPO.load(SINGLE_AGENT_CHECKPOINT, device=device)
    agent.eval()
    if agent.state_dim != env.state_dim or agent.n_nodes != env.n_agents:
        raise AssertionError("single-agent checkpoint dimensions do not match the R2 environment")
    if agent.n_actions != env.n_actions:
        raise AssertionError("single-agent checkpoint action definition does not match R2")
    metadata = A3Metadata(
        checkpoint=str(SINGLE_AGENT_CHECKPOINT),
        checkpoint_sha256=input_hashes["single_agent_checkpoint"],
        validated_r2_config_sha256=input_hashes["validated_r2_config"],
        single_agent_config_sha256=input_hashes["single_agent_config"],
        oof_prediction_artifact_sha256=input_hashes["oof_prediction_artifact"],
        seed=EVALUATION_SEED,
        node_count=env.n_agents,
        task_count=env.cfg.n_tasks,
    )
    return A3Run(env, VerifiedSingleAgentPolicy(agent), metadata, starts, input_hashes)


class A3CsvBundle:
    """A3-only writer that refuses to replace any prior A3 evaluation bundle."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.paths = {key: output_dir / filename for key, (filename, _) in FILE_SPECS.items()}
        self.integrity_path = output_dir / INTEGRITY_FILENAME
        self.readme_path = output_dir / README_FILENAME
        existing = [
            path for path in [*self.paths.values(), self.integrity_path, self.readme_path]
            if path.exists()
        ]
        if output_dir.exists():
            existing.extend(path for path in output_dir.glob("sprint10_a3_*") if path not in existing)
        if existing:
            names = ", ".join(sorted({path.name for path in existing}))
            raise FileExistsError(
                f"Sprint 10 A3 output directory already contains an evaluation bundle: {names}. "
                "Choose a new empty A3 directory; this runner never overwrites results.")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.files = {}
        self.writers = {}
        self.counts = {key: 0 for key in FILE_SPECS}
        for key, path in self.paths.items():
            handle = open(path, "w", newline="", encoding="utf-8")
            self.files[key] = handle
            self.writers[key] = csv.DictWriter(
                handle, fieldnames=FILE_SPECS[key][1], extrasaction="raise")
            self.writers[key].writeheader()

    def write(self, key: str, row: Mapping[str, object]) -> None:
        self.writers[key].writerow(row)
        self.counts[key] += 1

    def close(self) -> None:
        for handle in self.files.values():
            handle.close()


class A3Observer(Sprint9Observer):
    """Read-only trace exporter with A3 start and policy-mechanism guards."""

    def __init__(self, bundle: A3CsvBundle, run: A3Run, run_id: str):
        super().__init__(bundle, run, run_id)
        self.consumed_starts: list[int] = []

    def on_episode_start(self, env: DTMarlEnv, start_tick: int, seed: int,
                         policy_name: str) -> None:
        if policy_name != EXPERIMENT_NAME or not isinstance(self.run.policy, VerifiedSingleAgentPolicy):
            raise AssertionError("A3 rollout did not use the verified single-agent PPO adapter")
        if self.episode >= len(self.run.starts):
            raise AssertionError("A3 rollout exceeded the fixed held-out start protocol")
        expected = self.run.starts[self.episode]
        if start_tick != expected or env.t0 != expected:
            raise AssertionError(
                f"A3 episode {self.episode + 1} consumed {env.t0}, expected held-out start {expected}")
        assert_a3_controls(env)
        if env.t0 != expected:
            raise AssertionError("A3 component validation changed the environment start tick")
        self.episode += 1
        self.consumed_starts.append(env.t0)
        self.priority_start = {
            task.spec.task_id: float(task.spec.priority_at(0.0)) for task in env.tasks
        }
        self.telemetry = _EpisodeTelemetry({
            key: 0.0 for key in (
                "link_latency_ms", "power_w", "cpu_utilization_pct",
                "ram_utilization_pct", "bandwidth_utilization_pct")
        })
        self.bundle.write("metadata", {
            "run_id": self.run_id, **self.metadata.as_row(), "episode": self.episode,
            "episode_start_tick": env.t0, "seed": int(seed),
        })
        self.bundle.write("prediction_usage", {
            **self._base(env), "risk_source": "oof", "prediction_used": 1,
            "uncertainty_used": 0, "oof_loaded": 1, "live_predictor_loaded": 0,
            "offline_only_note": (
                "Leakage-safe OOF risk artifact is loaded by the unchanged R2 provider; "
                "no live predictor is used. The R2 uncertainty slot is reserved-zero and "
                "has no active gating branch."),
            "oof_prediction_artifact_sha256": self.metadata.oof_prediction_artifact_sha256,
            "destination_risk_weight": env.cfg.dest_w_risk,
            "risk_exposure_penalty": env.rcfg.P_risk_expose,
            "criticality_weight": env.rcfg.w_criticality,
            "criticality_migration_weight": env.rcfg.w_criticality_migration,
            "uncertainty_component_preserved": 1,
            "uncertainty_gating_protocol": UNCERTAINTY_GATING_PROTOCOL,
        })

    def before_step(self, env: DTMarlEnv, actions: np.ndarray,
                    observations: np.ndarray, action_masks: np.ndarray) -> None:
        assert_a3_controls(env, observations)
        super().before_step(env, actions, observations, action_masks)

    def on_episode_end(self, env: DTMarlEnv, metrics: Mapping[str, object]) -> None:
        """Export actual environment outcomes, including R2 criticality metrics."""
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
            "mean_hsi": float(hsi.mean()), "min_hsi": float(hsi.min()),
            "max_hsi": float(hsi.max()), "mean_priority_at_start": float(priority_start.mean()),
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


def run_fixed_start_protocol(run: A3Run, observer: A3Observer) -> None:
    """Execute exactly one real inference episode for every validated held-out start."""
    for episode, start in enumerate(run.starts):
        run_episode(run.environment, run.policy, start_tick=start,
                    seed=EVALUATION_SEED + episode, observer=observer)
    if tuple(observer.consumed_starts) != run.starts:
        raise AssertionError(
            f"A3 environment consumed starts {observer.consumed_starts}, expected {list(run.starts)}")
    if run.policy.action_selection_calls <= 0:
        raise AssertionError("A3 exported no single-agent PPO action selections")


def _assert_a3_output_isolation(output_dir: Path) -> None:
    sprint10_root = (ROOT / "data" / "sprint10").resolve()
    try:
        output_dir.relative_to(sprint10_root)
    except ValueError as exc:
        raise ValueError("A3 outputs must stay below data/sprint10") from exc
    if output_dir.parent != sprint10_root or not output_dir.name.startswith("a3_"):
        raise ValueError("A3 output directory must be a new data/sprint10/a3_* directory")


def _write_readme(bundle: A3CsvBundle, run: A3Run) -> None:
    bundle.readme_path.write_text(
        "# Sprint 10 A3 — Single-Agent PPO Ablation\n\n"
        "This bundle contains real, inference-only evaluation traces for the eight fixed R2 "
        "held-out starts. It uses the validated Sprint 9 centralized single-agent PPO checkpoint "
        "for action selection; MAPPO is not loaded or called for actions.\n\n"
        "The R2 Digital Twin trace, OOF failure prediction, destination-risk weighting, "
        "criticality-aware reward, task generation, and environment dynamics are retained. "
        "R2 uncertainty remains its documented reserved, zero-valued observation channel; "
        "there is no active uncertainty-gating branch to add or remove.\n\n"
        f"Single-agent checkpoint SHA-256: `{run.metadata.checkpoint_sha256}`\n\n"
        f"Frozen R2 MAPPO checkpoint SHA-256: `{VALIDATED_R2_CHECKPOINT_SHA256}`\n",
        encoding="utf-8")


def write_integrity_manifest(bundle: A3CsvBundle, run: A3Run,
                             protected_before: Mapping[str, str],
                             protected_after: Mapping[str, str],
                             protected_a2_before: Mapping[str, str],
                             protected_a2_after: Mapping[str, str],
                             consumed_starts: Sequence[int]) -> None:
    """Seal the completed bundle only if no protected input/output changed."""
    if protected_before != protected_after:
        raise AssertionError("Sprint 8/9 outputs changed during A3 evaluation")
    if protected_a2_before != protected_a2_after:
        raise AssertionError("completed A2 outputs changed during A3 evaluation")
    if tuple(consumed_starts) != run.starts:
        raise AssertionError(
            f"A3 environment consumed starts {list(consumed_starts)}, expected {list(run.starts)}")
    input_after = verify_protected_inputs()
    if dict(run.input_hashes) != input_after:
        raise AssertionError("a protected A3 checkpoint, configuration, or prediction input changed")
    artifact_hashes = {
        path.name: sha256(path) for path in [*bundle.paths.values(), bundle.readme_path]
    }
    payload = {
        "experiment": EXPERIMENT_NAME,
        "held_out_starts": list(run.starts),
        "consumed_starts": list(consumed_starts),
        "single_agent_checkpoint": str(SINGLE_AGENT_CHECKPOINT),
        "single_agent_checkpoint_sha256_before": run.metadata.checkpoint_sha256,
        "single_agent_checkpoint_sha256_after": input_after["single_agent_checkpoint"],
        "single_agent_config": str(SINGLE_AGENT_CONFIG),
        "single_agent_config_sha256": input_after["single_agent_config"],
        "frozen_r2_mappo_checkpoint": str(VALIDATED_R2_CHECKPOINT),
        "frozen_r2_mappo_checkpoint_sha256_before": run.input_hashes["validated_r2_checkpoint"],
        "frozen_r2_mappo_checkpoint_sha256_after": input_after["validated_r2_checkpoint"],
        "validated_r2_config": str(VALIDATED_R2_CONFIG),
        "validated_r2_config_sha256": input_after["validated_r2_config"],
        "oof_prediction_artifact": str(OOF_PREDICTION_ARTIFACT),
        "oof_prediction_artifact_sha256": input_after["oof_prediction_artifact"],
        "a3_configuration": {
            "decision_mechanism": "single_agent_ppo",
            "single_agent_policy_used": True,
            "single_agent_action_selection_calls": run.policy.action_selection_calls,
            "mappo_action_selection_used": False,
            "digital_twin_enabled": True,
            "failure_prediction_enabled": True,
            "risk_source": run.environment.cfg.risk_source,
            "oof_loaded": True,
            "destination_risk_weight": run.environment.cfg.dest_w_risk,
            "risk_exposure_penalty": run.environment.rcfg.P_risk_expose,
            "criticality_awareness_enabled": True,
            "criticality_weight": run.environment.rcfg.w_criticality,
            "criticality_migration_weight": run.environment.rcfg.w_criticality_migration,
            "uncertainty_component_preserved": True,
            "uncertainty_gating_protocol": UNCERTAINTY_GATING_PROTOCOL,
        },
        "protected_sprint8_sprint9_before": dict(protected_before),
        "protected_sprint8_sprint9_after": dict(protected_after),
        "protected_a2_before": dict(protected_a2_before),
        "protected_a2_after": dict(protected_a2_after),
        "protected_outputs_unchanged": True,
        "artifact_sha256": artifact_hashes,
        "rows": bundle.counts,
        "inference_only": True,
    }
    with open(bundle.integrity_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_pipeline(args: argparse.Namespace) -> A3CsvBundle:
    """Run the real A3 evaluation. This function is intentionally not called by tests."""
    output_dir = Path(args.out_dir).resolve()
    _assert_a3_output_isolation(output_dir)
    protected_before = protected_output_hashes()
    protected_a2_before = protected_a2_hashes()
    run = build_a3_run(device=args.device)
    bundle = A3CsvBundle(output_dir)
    observer = A3Observer(bundle, run, args.run_id or EXPERIMENT_NAME)
    try:
        run_fixed_start_protocol(run, observer)
        protected_after = protected_output_hashes()
        protected_a2_after = protected_a2_hashes()
        bundle.close()
        _write_readme(bundle, run)
        write_integrity_manifest(
            bundle, run, protected_before, protected_after,
            protected_a2_before, protected_a2_after, observer.consumed_starts)
    except Exception:
        bundle.close()
        raise
    return bundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 10 A3 inference-only single-agent PPO evaluation")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = run_pipeline(args)
    print("SPRINT 10 A3 real single-agent evaluation complete")
    print(f"  output: {bundle.output_dir}")
    for key, path in bundle.paths.items():
        print(f"  {path.name:<38s} {bundle.counts[key]:>8d} rows")
    print("  scope: inference-only; MAPPO action selection was not used")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
