"""Sprint 10 A1: frozen-R2 counterfactual without Digital Twin decision context.

R2's recorded CloudSim trace has two deliberately separate roles.  It is the
immutable transition source for availability/failures/throughput/energy, and it
also exposes current telemetry to the controller.  A1 removes only the latter
role: direct trace telemetry is neutralised before frozen MAPPO inference and
trace link latency is made a constant in edge-destination ranking.  The trace
continues to provide environment physics and feasibility guards, so A1 cannot
place a task on a host the recorded system says is down.

OOF risk is intentionally *not* a Digital Twin substitute in this ablation:
it remains the distinct validated failure-prediction mechanism.  Likewise,
criticality, the reserved-zero uncertainty contract, all task scheduling, and
the frozen greedy multi-agent MAPPO controller remain R2-identical.

Do not run this module until implementation review is complete.  It is an
inference-only runner and refuses every pre-existing A1 output directory.
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

from marl.config import ACTION_NAMES, CLOUD_NODE_ID
from marl.destination import CandidateView
from marl.env import DTMarlEnv
from marl.mappo import MAPPO, MappoPolicy
from marl.rollout import episode_starts, run_episode
from marl.sprint9_benchmarks import load_validated_r2_config
from training.sprint10_a5_no_uncertainty import A5Observer
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


EXPERIMENT_NAME = "a1-no-digital-twin"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "sprint10" / "a1_no_digital_twin"
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
INTEGRITY_FILENAME = "sprint10_a1_integrity.json"
README_FILENAME = "README.md"

# R2's 48-D observation is [local 15][task 8][four neighbours x 5][cloud 3][context 2].
# The predictor's risk channels, task context, capacity/load context, cloud
# constants, and episode context are deliberately excluded from this list.
LOCAL_TELEMETRY_OBSERVATION_INDICES = tuple(range(12))
NEIGHBOUR_TELEMETRY_OBSERVATION_INDICES = tuple(
    23 + 5 * neighbour + offset
    for neighbour in range(4)
    for offset in (1, 3, 4)  # observed availability, CPU, link latency
)
DIRECT_DIGITAL_TWIN_OBSERVATION_INDICES = (
    LOCAL_TELEMETRY_OBSERVATION_INDICES + NEIGHBOUR_TELEMETRY_OBSERVATION_INDICES
)
DIGITAL_TWIN_MECHANISM_FOUND = (
    "DTMarlEnv._observations places the current trace's 12 local telemetry fields at "
    "indices 0-11 and each neighbour's observed availability, CPU, and link-latency "
    "fields at indices 24/26/27 plus five per neighbour; DTMarlEnv._candidate also "
    "passes trace-derived link latency to DestinationSelector. Trace active remains "
    "availability physics/feasibility, trace energy remains the recorded reward outcome, "
    "and trace symptoms are migration-audit classification rather than policy input."
)
DIGITAL_TWIN_A1_CHANGE = (
    "A1 hard-zeros only the direct trace telemetry observation slots and supplies a "
    "constant zero edge link-latency term to destination ranking. It preserves the "
    "48-D geometry, OOF risk, capacity/load state, and recorded active availability as "
    "an environment-physics feasibility constraint; it does not replace either path "
    "with a heuristic."
)
DIGITAL_TWIN_ABLATION_PROTOCOL = (
    "direct-trace-policy-telemetry-zeroed; edge-destination-trace-latency-neutral; "
    "trace-physics-and-availability-feasibility-retained"
)

METADATA_FIELDS = [
    "run_id", "baseline_name", "experiment", "episode", "episode_start_tick", "risk_source",
    "uses_digital_twin", "uses_prediction", "uses_uncertainty", "uses_marl",
    "seed", "node_count", "task_count", "recovery_mode", "training_loss_status",
    "checkpoint", "checkpoint_sha256", "ablation_id", "decision_mechanism",
    "mappo_action_selection_used", "uses_digital_twin_trace",
    "digital_twin_decision_context_enabled", "digital_twin_policy_observation_indices_zero",
    "digital_twin_destination_link_latency_neutralized",
    "digital_twin_availability_retained_as_physics_constraint",
    "digital_twin_observation_geometry_preserved", "digital_twin_mechanism_found",
    "digital_twin_a1_change", "digital_twin_ablation_protocol",
    "criticality_awareness_enabled", "criticality_weight", "criticality_migration_weight",
    "uncertainty_component_preserved", "uncertainty_gating_protocol",
    "validated_r2_config", "validated_r2_config_sha256", "oof_prediction_artifact_sha256",
]
PREDICTION_USAGE_FIELDS = [
    *SPRINT9_PREDICTION_USAGE_FIELDS,
    "oof_prediction_artifact_sha256", "destination_risk_weight", "risk_exposure_penalty",
    "criticality_weight", "criticality_migration_weight",
    "digital_twin_decision_context_enabled", "digital_twin_policy_observation_indices_zero",
    "digital_twin_destination_link_latency_neutralized",
    "digital_twin_availability_retained_as_physics_constraint",
    "digital_twin_observation_geometry_preserved", "digital_twin_mechanism_found",
    "digital_twin_a1_change", "digital_twin_ablation_protocol",
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
    "metadata": ("sprint10_a1_metadata.csv", METADATA_FIELDS),
    "node_ticks": ("sprint10_a1_node_ticks.csv", SPRINT9_NODE_TICK_FIELDS),
    "step_rewards": ("sprint10_a1_step_rewards.csv", SPRINT9_STEP_REWARD_FIELDS),
    "task_events": ("sprint10_a1_task_events.csv", SPRINT9_TASK_EVENT_FIELDS),
    "task_metadata": ("sprint10_a1_task_metadata.csv", SPRINT9_TASK_METADATA_FIELDS),
    "migrations": ("sprint10_a1_migrations.csv", SPRINT9_MIGRATION_FIELDS),
    "recovery_events": ("sprint10_a1_recovery_events.csv", SPRINT9_RECOVERY_FIELDS),
    "prediction_usage": ("sprint10_a1_prediction_usage.csv", PREDICTION_USAGE_FIELDS),
    "episode_metrics": ("sprint10_a1_episode_metrics.csv", EPISODE_FIELDS),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _hash_paths(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
            for path in sorted(set(paths))}


def protected_sprint8_sprint9_hashes() -> dict[str, str]:
    """Fingerprint all completed Sprint 8/9 evidence without opening it for write."""
    paths: list[Path] = []
    data_root = ROOT / "data"
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
    root = ROOT / "data" / "sprint10"
    if not root.exists():
        return {}
    paths: list[Path] = []
    for candidate in root.glob(f"{prefix}*"):
        if candidate.is_file():
            paths.append(candidate)
        elif candidate.is_dir():
            paths.extend(path for path in candidate.rglob("*") if path.is_file())
    return _hash_paths(paths)


def protected_a2_hashes() -> dict[str, str]:
    return _protected_sprint10_hashes("a2")


def protected_a3_hashes() -> dict[str, str]:
    return _protected_sprint10_hashes("a3")


def protected_a4_hashes() -> dict[str, str]:
    return _protected_sprint10_hashes("a4")


def protected_a5_hashes() -> dict[str, str]:
    return _protected_sprint10_hashes("a5")


def _require_digest(path: Path, expected: str, description: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{description} is missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise AssertionError(f"{description} SHA-256 does not match its protected value")
    return actual


def verify_protected_inputs() -> dict[str, str]:
    return {
        "validated_r2_checkpoint": _require_digest(
            VALIDATED_CHECKPOINT, VALIDATED_CHECKPOINT_SHA256, "validated R2 MAPPO checkpoint"),
        "validated_r2_config": _require_digest(
            VALIDATED_CONFIG, VALIDATED_CONFIG_SHA256, "validated R2 configuration"),
        "oof_prediction_artifact": _require_digest(
            OOF_PREDICTION_ARTIFACT, OOF_PREDICTION_ARTIFACT_SHA256,
            "OOF failure-prediction artifact"),
    }


class NoDigitalTwinDecisionDTMarlEnv(DTMarlEnv):
    """R2 replay physics with direct Digital Twin decision context neutralised.

    ``active`` intentionally remains the actual recorded value in candidates:
    it is an availability constraint of the transition system and action mask,
    not a preference signal.  Candidate risk, free capacity, and runtime load
    are also inherited.  Only trace link latency, which otherwise ranks live
    edge destinations, is made a constant.
    """

    def _observations(self) -> np.ndarray:
        observations = super()._observations()
        observations[:, DIRECT_DIGITAL_TWIN_OBSERVATION_INDICES] = 0.0
        return observations

    def _candidate(self, node: int, step: int | None = None,
                   risk_override: float | None = None) -> CandidateView:
        candidate = super()._candidate(node, step, risk_override)
        if node == CLOUD_NODE_ID:
            return candidate
        return CandidateView(
            node_id=candidate.node_id,
            risk=candidate.risk,
            observed_up=candidate.observed_up,
            free_capacity_fraction=candidate.free_capacity_fraction,
            load_fraction=candidate.load_fraction,
            link_latency_norm=0.0,
            is_cloud=False,
        )


def assert_a1_controls(env: DTMarlEnv, observations: np.ndarray | None = None) -> None:
    """Fail closed unless A1 removes precisely the direct DT decision inputs."""
    if not isinstance(env, NoDigitalTwinDecisionDTMarlEnv):
        raise AssertionError("A1 requires its no-Digital-Twin-decision R2 environment")
    if env.cfg.risk_source != "oof" or env.risk.source != "oof":
        raise AssertionError("A1 must retain the R2 OOF failure-prediction provider")
    if env.cfg.dest_w_risk <= 0.0 or env.rcfg.P_risk_expose <= 0.0:
        raise AssertionError("A1 must retain R2 prediction-driven destination/reward paths")
    if env.rcfg.w_criticality <= 0.0 or env.rcfg.w_criticality_migration <= 0.0:
        raise AssertionError("A1 must retain active R2 criticality paths")
    if env.cfg.start_frac_lo != HELD_OUT_START_FRACTION or env.cfg.start_frac_hi != 1.0:
        raise AssertionError("A1 must use the validated R2 held-out start window")
    if env.cfg.random_episode_start:
        raise AssertionError("A1 must consume caller-specified held-out starts in order")
    if not np.isfinite(env.risk.risk[:, env.risk.first_valid_tick:]).all():
        raise AssertionError("A1 OOF prediction matrix is incomplete")
    if not np.any(env.risk.risk[:, env.risk.first_valid_tick:] > 0.0):
        raise AssertionError("A1 OOF provider has no active prediction signal")
    if not np.array_equal(env.risk.uncertainty, np.zeros_like(env.risk.uncertainty)):
        raise AssertionError("A1 changed the frozen R2 reserved-zero uncertainty contract")
    if observations is not None:
        if observations.shape != (env.n_agents, env.obs_dim):
            raise AssertionError("A1 observation geometry changed")
        if not np.array_equal(
                observations[:, DIRECT_DIGITAL_TWIN_OBSERVATION_INDICES],
                np.zeros((env.n_agents, len(DIRECT_DIGITAL_TWIN_OBSERVATION_INDICES)))):
            raise AssertionError("A1 policy observation retains direct Digital Twin telemetry")
        if not np.array_equal(observations[:, 13], np.zeros(env.n_agents)):
            raise AssertionError("A1 changed the R2 uncertainty observation contract")


class VerifiedMappoPolicy(MappoPolicy):
    """The sole A1 controller: frozen greedy multi-agent MAPPO."""

    def __init__(self, agent: MAPPO):
        if not isinstance(agent, MAPPO):
            raise TypeError("A1 requires the validated MAPPO controller")
        super().__init__(agent, name=EXPERIMENT_NAME, greedy=True)
        self.action_selection_calls = 0

    def act(self, env, obs, masks) -> np.ndarray:
        if not isinstance(self.agent, MAPPO) or not self.greedy:
            raise AssertionError("A1 policy is not frozen greedy MAPPO")
        actions = super().act(env, obs, masks)
        if actions.shape != (env.n_agents,) or not np.issubdtype(actions.dtype, np.integer):
            raise AssertionError("MAPPO did not emit an A1 joint action")
        if not np.all(masks[np.arange(env.n_agents), actions]):
            raise AssertionError("MAPPO selected a masked A1 action")
        self.action_selection_calls += 1
        return actions


@dataclass(frozen=True)
class A1Metadata:
    checkpoint_sha256: str
    validated_r2_config_sha256: str
    oof_prediction_artifact_sha256: str
    seed: int
    node_count: int
    task_count: int
    baseline_name: str = EXPERIMENT_NAME
    risk_source: str = "oof"
    uses_digital_twin: bool = False
    uses_prediction: bool = True
    uses_uncertainty: bool = False
    uses_marl: bool = True
    recovery_mode: str = "environment default"
    training_loss_status: str = "inference-only; no A1 retraining"

    def as_row(self) -> dict[str, object]:
        return {
            "baseline_name": self.baseline_name, "experiment": EXPERIMENT_NAME,
            "risk_source": self.risk_source, "uses_digital_twin": self.uses_digital_twin,
            "uses_prediction": self.uses_prediction, "uses_uncertainty": self.uses_uncertainty,
            "uses_marl": self.uses_marl, "seed": self.seed, "node_count": self.node_count,
            "task_count": self.task_count, "recovery_mode": self.recovery_mode,
            "training_loss_status": self.training_loss_status,
            "checkpoint": str(VALIDATED_CHECKPOINT), "checkpoint_sha256": self.checkpoint_sha256,
            "ablation_id": "A1", "decision_mechanism": "frozen_greedy_mappo",
            "mappo_action_selection_used": 1, "uses_digital_twin_trace": 1,
            "digital_twin_decision_context_enabled": 0,
            "digital_twin_policy_observation_indices_zero": json.dumps(
                DIRECT_DIGITAL_TWIN_OBSERVATION_INDICES),
            "digital_twin_destination_link_latency_neutralized": 1,
            "digital_twin_availability_retained_as_physics_constraint": 1,
            "digital_twin_observation_geometry_preserved": 1,
            "digital_twin_mechanism_found": DIGITAL_TWIN_MECHANISM_FOUND,
            "digital_twin_a1_change": DIGITAL_TWIN_A1_CHANGE,
            "digital_twin_ablation_protocol": DIGITAL_TWIN_ABLATION_PROTOCOL,
            "criticality_awareness_enabled": 1,
            "criticality_weight": "R2-config-retained",
            "criticality_migration_weight": "R2-config-retained",
            "uncertainty_component_preserved": 1,
            "uncertainty_gating_protocol": UNCERTAINTY_GATING_PROTOCOL,
            "validated_r2_config": str(VALIDATED_CONFIG),
            "validated_r2_config_sha256": self.validated_r2_config_sha256,
            "oof_prediction_artifact_sha256": self.oof_prediction_artifact_sha256,
        }


@dataclass
class A1Run:
    environment: NoDigitalTwinDecisionDTMarlEnv
    policy: VerifiedMappoPolicy
    metadata: A1Metadata
    starts: tuple[int, ...]
    input_hashes: Mapping[str, str]


def build_a1_run(device: str = "cpu") -> A1Run:
    """Reconstruct frozen R2 and remove only direct DT decision contribution."""
    input_hashes = verify_protected_inputs()
    cfg = load_validated_r2_config(VALIDATED_CONFIG)
    env = NoDigitalTwinDecisionDTMarlEnv(cfg.env, cfg.reward)
    observations, _, _ = env.reset(episode_start_tick=env._min_start, seed=EVALUATION_SEED)
    assert_a1_controls(env, observations)
    starts = tuple(episode_starts(env, len(FIXED_HELD_OUT_STARTS)))
    if starts != FIXED_HELD_OUT_STARTS:
        raise AssertionError(f"A1 held-out starts changed: got {starts}, expected {FIXED_HELD_OUT_STARTS}")
    agent, _ = MAPPO.load(VALIDATED_CHECKPOINT, device=device)
    agent.eval()
    if agent.obs_dim != env.obs_dim or agent.state_dim != env.state_dim or agent.n_agents != env.n_agents:
        raise AssertionError("validated MAPPO dimensions do not match the A1 R2 replay environment")
    metadata = A1Metadata(
        checkpoint_sha256=input_hashes["validated_r2_checkpoint"],
        validated_r2_config_sha256=input_hashes["validated_r2_config"],
        oof_prediction_artifact_sha256=input_hashes["oof_prediction_artifact"],
        seed=EVALUATION_SEED, node_count=env.n_agents, task_count=env.cfg.n_tasks)
    return A1Run(env, VerifiedMappoPolicy(agent), metadata, starts, input_hashes)


class A1CsvBundle:
    """A1-only output writer that never opens an existing evidence directory."""

    def __init__(self, output_dir: Path):
        if output_dir.exists():
            raise FileExistsError(
                f"Sprint 10 A1 output directory already exists: {output_dir}. "
                "A1 never overwrites or reuses evidence.")
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


class A1Observer(A5Observer):
    """Read-only A1 exporter; it records trace evidence without restoring it to policy input."""

    def __init__(self, bundle: A1CsvBundle, run: A1Run, run_id: str):
        super().__init__(bundle, run, run_id)
        self.consumed_starts: list[int] = []

    def _a1_row(self) -> dict[str, object]:
        return {
            "digital_twin_decision_context_enabled": 0,
            "digital_twin_policy_observation_indices_zero": json.dumps(
                DIRECT_DIGITAL_TWIN_OBSERVATION_INDICES),
            "digital_twin_destination_link_latency_neutralized": 1,
            "digital_twin_availability_retained_as_physics_constraint": 1,
            "digital_twin_observation_geometry_preserved": 1,
            "digital_twin_mechanism_found": DIGITAL_TWIN_MECHANISM_FOUND,
            "digital_twin_a1_change": DIGITAL_TWIN_A1_CHANGE,
            "digital_twin_ablation_protocol": DIGITAL_TWIN_ABLATION_PROTOCOL,
            "uncertainty_component_preserved": 1,
            "uncertainty_gating_protocol": UNCERTAINTY_GATING_PROTOCOL,
        }

    def on_episode_start(self, env: DTMarlEnv, start_tick: int, seed: int, policy_name: str) -> None:
        if policy_name != EXPERIMENT_NAME or not isinstance(self.run.policy, VerifiedMappoPolicy):
            raise AssertionError("A1 rollout did not use verified frozen MAPPO")
        if self.episode >= len(self.run.starts):
            raise AssertionError("A1 rollout exceeded the fixed held-out protocol")
        expected = self.run.starts[self.episode]
        if start_tick != expected or env.t0 != expected:
            raise AssertionError(f"A1 episode {self.episode + 1} consumed {env.t0}, expected {expected}")
        assert_a1_controls(env)
        self.episode += 1
        self.consumed_starts.append(env.t0)
        self.priority_start = {task.spec.task_id: float(task.spec.priority_at(0.0)) for task in env.tasks}
        self.telemetry = _EpisodeTelemetry({key: 0.0 for key in (
            "link_latency_ms", "power_w", "cpu_utilization_pct", "ram_utilization_pct",
            "bandwidth_utilization_pct")})
        self.bundle.write("metadata", {
            "run_id": self.run_id, **self.metadata.as_row(), "episode": self.episode,
            "episode_start_tick": env.t0, "seed": int(seed),
            "criticality_weight": env.rcfg.w_criticality,
            "criticality_migration_weight": env.rcfg.w_criticality_migration,
        })
        self.bundle.write("prediction_usage", {
            **self._base(env), "risk_source": "oof", "prediction_used": 1,
            "uncertainty_used": 0, "oof_loaded": 1, "live_predictor_loaded": 0,
            "offline_only_note": (
                "OOF risk remains active. A1 disables direct Digital Twin decision context, "
                "not failure prediction or the trace-backed environment."),
            "oof_prediction_artifact_sha256": self.metadata.oof_prediction_artifact_sha256,
            "destination_risk_weight": env.cfg.dest_w_risk,
            "risk_exposure_penalty": env.rcfg.P_risk_expose,
            "criticality_weight": env.rcfg.w_criticality,
            "criticality_migration_weight": env.rcfg.w_criticality_migration,
            **self._a1_row(),
        })

    def before_step(self, env: DTMarlEnv, actions: np.ndarray,
                    observations: np.ndarray, action_masks: np.ndarray) -> None:
        assert_a1_controls(env, observations)
        # Do not call A5Observer.before_step: it asserts A5's environment type.
        Sprint9Observer.before_step(self, env, actions, observations, action_masks)


def run_fixed_start_protocol(run: A1Run, observer: A1Observer) -> None:
    for episode, start in enumerate(run.starts):
        run_episode(run.environment, run.policy, start_tick=start,
                    seed=EVALUATION_SEED + episode, observer=observer)
    if tuple(observer.consumed_starts) != run.starts:
        raise AssertionError(f"A1 consumed {observer.consumed_starts}, expected {list(run.starts)}")
    if run.policy.action_selection_calls <= 0:
        raise AssertionError("A1 exported no frozen MAPPO action selections")


def _assert_a1_output_isolation(output_dir: Path) -> None:
    root = (ROOT / "data" / "sprint10").resolve()
    try:
        output_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("A1 outputs must stay below data/sprint10") from exc
    if output_dir.parent != root or not output_dir.name.startswith("a1_"):
        raise ValueError("A1 output directory must be a new data/sprint10/a1_* directory")


def _write_readme(bundle: A1CsvBundle, run: A1Run) -> None:
    bundle.readme_path.write_text(
        "# Sprint 10 A1 — No Digital Twin Decision Context\n\n"
        "This inference-only counterfactual retains the frozen greedy R2 MAPPO policy, OOF "
        "failure prediction, criticality, reserved-zero uncertainty contract, task/failure "
        "generation, and all trace-backed environment physics. Trace exports remain evidence; "
        "they are not restored to MAPPO observations or edge destination ranking.\n\n"
        f"R2 mechanism found: {DIGITAL_TWIN_MECHANISM_FOUND}\n\n"
        f"A1 change: {DIGITAL_TWIN_A1_CHANGE}\n\n"
        f"Frozen R2 MAPPO checkpoint SHA-256: `{run.metadata.checkpoint_sha256}`\n",
        encoding="utf-8")


def write_integrity_manifest(bundle: A1CsvBundle, run: A1Run,
                             sprint8_sprint9_before: Mapping[str, str],
                             sprint8_sprint9_after: Mapping[str, str],
                             a2_before: Mapping[str, str], a2_after: Mapping[str, str],
                             a3_before: Mapping[str, str], a3_after: Mapping[str, str],
                             a4_before: Mapping[str, str], a4_after: Mapping[str, str],
                             a5_before: Mapping[str, str], a5_after: Mapping[str, str],
                             consumed_starts: Sequence[int]) -> None:
    protected_after = verify_protected_inputs()
    if protected_after != run.input_hashes:
        raise AssertionError("a frozen R2 input changed during A1 evaluation")
    if not (dict(sprint8_sprint9_before) == dict(sprint8_sprint9_after)
            and dict(a2_before) == dict(a2_after) and dict(a3_before) == dict(a3_after)
            and dict(a4_before) == dict(a4_after) and dict(a5_before) == dict(a5_after)):
        raise AssertionError("protected Sprint 8/9 or A2-A5 evidence changed during A1")
    artifact_hashes = {path.name: sha256(path) for path in bundle.paths.values()}
    artifact_hashes[README_FILENAME] = sha256(bundle.readme_path)
    payload = {
        "experiment": EXPERIMENT_NAME, "inference_only": True,
        "held_out_starts": list(FIXED_HELD_OUT_STARTS), "consumed_starts": list(consumed_starts),
        "frozen_r2_mappo_checkpoint": str(VALIDATED_CHECKPOINT),
        "frozen_r2_mappo_checkpoint_sha256_before": run.input_hashes["validated_r2_checkpoint"],
        "frozen_r2_mappo_checkpoint_sha256_after": protected_after["validated_r2_checkpoint"],
        "validated_r2_config": str(VALIDATED_CONFIG),
        "validated_r2_config_sha256_before": run.input_hashes["validated_r2_config"],
        "validated_r2_config_sha256_after": protected_after["validated_r2_config"],
        "oof_prediction_artifact": str(OOF_PREDICTION_ARTIFACT),
        "oof_prediction_artifact_sha256_before": run.input_hashes["oof_prediction_artifact"],
        "oof_prediction_artifact_sha256_after": protected_after["oof_prediction_artifact"],
        "digital_twin_mechanism_found": DIGITAL_TWIN_MECHANISM_FOUND,
        "digital_twin_a1_change": DIGITAL_TWIN_A1_CHANGE,
        "a1_configuration": {
            "decision_mechanism": "frozen_greedy_mappo",
            "mappo_action_selection_used": run.policy.action_selection_calls > 0,
            "digital_twin_decision_context_enabled": False,
            "digital_twin_trace_retained_for_environment_physics": True,
            "digital_twin_policy_observation_indices_zero": list(DIRECT_DIGITAL_TWIN_OBSERVATION_INDICES),
            "digital_twin_destination_link_latency_neutralized": True,
            "digital_twin_availability_retained_as_physics_constraint": True,
            "digital_twin_observation_geometry_preserved": True,
            "failure_prediction_enabled": True, "risk_source": "oof", "oof_loaded": True,
            "destination_risk_weight": run.environment.cfg.dest_w_risk,
            "risk_exposure_penalty": run.environment.rcfg.P_risk_expose,
            "criticality_awareness_enabled": True,
            "criticality_weight": run.environment.rcfg.w_criticality,
            "criticality_migration_weight": run.environment.rcfg.w_criticality_migration,
            "uncertainty_component_preserved": True,
            "uncertainty_gating_protocol": UNCERTAINTY_GATING_PROTOCOL,
        },
        "protected_sprint8_sprint9_before": dict(sprint8_sprint9_before),
        "protected_sprint8_sprint9_after": dict(sprint8_sprint9_after),
        "protected_a2_before": dict(a2_before), "protected_a2_after": dict(a2_after),
        "protected_a3_before": dict(a3_before), "protected_a3_after": dict(a3_after),
        "protected_a4_before": dict(a4_before), "protected_a4_after": dict(a4_after),
        "protected_a5_before": dict(a5_before), "protected_a5_after": dict(a5_after),
        "protected_outputs_unchanged": True, "artifact_sha256": artifact_hashes,
        "rows": bundle.counts,
    }
    with open(bundle.integrity_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def run_pipeline(args: argparse.Namespace) -> A1CsvBundle:
    """Run the real A1 evaluation. Regression tests never call this function."""
    output_dir = Path(args.out_dir).resolve()
    _assert_a1_output_isolation(output_dir)
    sprint8_sprint9_before = protected_sprint8_sprint9_hashes()
    a2_before, a3_before = protected_a2_hashes(), protected_a3_hashes()
    a4_before, a5_before = protected_a4_hashes(), protected_a5_hashes()
    run = build_a1_run(device=args.device)
    bundle = A1CsvBundle(output_dir)
    observer = A1Observer(bundle, run, args.run_id or EXPERIMENT_NAME)
    try:
        run_fixed_start_protocol(run, observer)
        sprint8_sprint9_after = protected_sprint8_sprint9_hashes()
        a2_after, a3_after = protected_a2_hashes(), protected_a3_hashes()
        a4_after, a5_after = protected_a4_hashes(), protected_a5_hashes()
        bundle.close()
        _write_readme(bundle, run)
        write_integrity_manifest(
            bundle, run, sprint8_sprint9_before, sprint8_sprint9_after,
            a2_before, a2_after, a3_before, a3_after, a4_before, a4_after,
            a5_before, a5_after, observer.consumed_starts)
    except Exception:
        bundle.close()
        raise
    return bundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sprint 10 A1 inference-only no-Digital-Twin evaluation")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = run_pipeline(args)
    print("SPRINT 10 A1 real no-Digital-Twin evaluation complete")
    print(f"  output: {bundle.output_dir}")
    for key, path in bundle.paths.items():
        print(f"  {path.name:<38s} {bundle.counts[key]:>8d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
