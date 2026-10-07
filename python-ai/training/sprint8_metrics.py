"""
Sprint 8: export research metrics from the recorded Digital Twin and an
existing MAPPO evaluation. This module never trains a model or writes a
checkpoint. It deliberately keeps source granularities separate:

* node ticks come directly from the recorded Digital Twin trace;
* task, migration, step-reward and episode rows come from one real evaluation
  rollout of an existing checkpoint;
* recovery rows come from the recorded simulator event log;
* failure-prediction rows come from leakage-safe OOF artifacts; and
* PPO losses are copied, with provenance, from an existing update log.

The legacy Sprint 8 script fabricated values with ``random.*``. That path is
replaced here; no synthetic metric is emitted by this module.

Example (from python-ai/):
    python training/sprint8_metrics.py --episodes 1 --overwrite
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.failure_dataset import OBSERVABLE_COLUMNS, build_windows  # noqa: E402
from marl.config import ACTION_NAMES, EnvConfig, RewardConfig, Sprint6Config, resolve_device  # noqa: E402
from marl.env import COMPLETED, LOST, DTMarlEnv  # noqa: E402
from marl.mappo import MAPPO, MappoPolicy  # noqa: E402
from marl.rollout import episode_starts, run_episode  # noqa: E402
from host_cv import full_metrics  # noqa: E402


DEFAULT_MODEL = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_best.pth"
DEFAULT_OUTPUT_DIR = ROOT / "data"
DEFAULT_TRAIN_FRAC = 0.7  # Must match marl/train.py and marl/evaluate.py.


NODE_TICK_FIELDS = [
    "run_id", "policy", "episode", "episode_start_tick", "decision_step",
    "recorded_tick", "sim_time_s", "tick_duration_s", "node_id",
    "link_latency_ms", "power_w", "energy_interval_j",
    "cpu_utilization_pct", "ram_utilization_pct", "bandwidth_utilization_pct",
    "running_tasks_telemetry", "active", "degraded", "link_up",
    "link_bandwidth_mbps", "link_packet_loss_pct", "under_attack",
    "predicted_failure_risk", "prediction_uncertainty_reserved",
]

STEP_REWARD_FIELDS = [
    "run_id", "policy", "episode", "episode_start_tick", "decision_step",
    "recorded_tick_start", "recorded_tick_end", "agent_id",
    "requested_action", "requested_action_name", "executed_action",
    "executed_action_name", "agent_reward", "team_reward", "completed_delta",
    "lost_delta", "sla_delta", "migration_cost", "energy_reward_cost",
    "progress_fraction", "infeasible_action", "focus_task_id",
]

TASK_EVENT_FIELDS = [
    "run_id", "policy", "episode", "episode_start_tick", "decision_step",
    "recorded_tick", "sim_time_s", "event_type", "task_id", "patient_id",
    "prior_state", "terminal_state", "node_id", "destination_node_id",
    "reward_owner", "arrival_step", "start_step", "finish_step", "lost_step",
    "task_latency_s", "task_latency_available", "migrations", "reroutes",
    "hsi", "clinical_severity", "priority_at_event", "deadline_breached",
]

TASK_METADATA_FIELDS = [
    "run_id", "policy", "episode", "episode_start_tick", "task_id", "patient_id",
    "hsi", "vitals_instability", "age_years", "clinical_severity",
    "priority_at_episode_start", "priority_at_episode_end", "arrival_time_s",
    "deadline_s", "final_state", "final_node_id", "migrations", "reroutes",
]

MIGRATION_FIELDS = [
    "run_id", "policy", "episode", "episode_start_tick", "decision_step",
    "recorded_tick", "sim_time_s", "task_id", "patient_id", "clinical_severity",
    "priority", "action", "action_name", "source_node_id", "destination_node_id",
    "migration_cost", "latency_steps", "preemptive", "source_symptomatic",
    "source_risk", "destination_risk", "source_failed_within_window",
    "task_survived", "outcome_scope",
]

RECOVERY_FIELDS = [
    "node_id", "failure_time_s", "failure_tick", "recovery_time_s",
    "recovery_tick", "recovery_duration_s", "failure_event_type",
    "recovery_event_type", "source_csv",
]

PREDICTION_WINDOW_FIELDS = [
    "node_id", "window_end_time_s", "recorded_tick", "predicted_failure_risk",
    "oof_predicted_label", "will_fail_soon_label", "prediction_correct",
    "event_id", "prediction_source", "policy_observable",
]

PREDICTION_METRIC_FIELDS = [
    "scope", "metric", "value", "n_windows", "positive_windows",
    "prediction_source", "policy_observable",
]

EPISODE_FIELDS = [
    "run_id", "policy", "episode", "episode_start_tick", "episode_end_tick",
    "episode_start_time_s", "episode_end_time_s", "decision_steps", "episode_reward",
    "task_success_rate", "completed", "lost", "unfinished", "avg_task_latency_s",
    "energy_reward_cost", "migrations", "migrations_edge", "migrations_cloud",
    "reroutes", "relocations", "preemptive_relocations", "reactive_relocations",
    "tasks_protected_before_failure", "sla_violations", "mean_link_latency_ms",
    "mean_power_w", "mean_cpu_utilization_pct", "mean_ram_utilization_pct",
    "mean_bandwidth_utilization_pct", "host_failures_started_in_window",
    "host_recoveries_completed_in_window", "mean_recovery_duration_s",
    "mean_hsi", "min_hsi", "max_hsi", "mean_priority_at_start",
    "mean_priority_at_end", "prediction_uncertainty_status",
]

TRAINING_LOSS_FIELDS = [
    "source_csv", "update", "episode", "mean_reward", "actor_loss", "critic_loss",
    "entropy", "approx_kl", "clip_frac", "adv_std", "value_mean",
    "decision_frac", "lr_scale", "explained_var",
]

FILE_SPECS = {
    "node_ticks": ("sprint8_node_ticks.csv", NODE_TICK_FIELDS),
    "step_rewards": ("sprint8_step_rewards.csv", STEP_REWARD_FIELDS),
    "task_events": ("sprint8_task_events.csv", TASK_EVENT_FIELDS),
    "task_metadata": ("sprint8_task_metadata.csv", TASK_METADATA_FIELDS),
    "migrations": ("sprint8_migrations.csv", MIGRATION_FIELDS),
    "recovery": ("sprint8_recovery.csv", RECOVERY_FIELDS),
    "prediction_windows": ("sprint8_prediction_windows.csv", PREDICTION_WINDOW_FIELDS),
    "prediction_metrics": ("sprint8_prediction_metrics.csv", PREDICTION_METRIC_FIELDS),
    "episode_metrics": ("sprint8_episode_metrics.csv", EPISODE_FIELDS),
    "training_loss": ("sprint8_training_loss.csv", TRAINING_LOSS_FIELDS),
}


@dataclass(frozen=True)
class RecoveryPair:
    """One recorded HOST_FAILURE -> HOST_RECOVERED pair for one node."""

    node_id: int
    failure_time_s: float
    recovery_time_s: float
    failure_tick: int
    recovery_tick: int

    @property
    def duration_s(self) -> float:
        return self.recovery_time_s - self.failure_time_s


class CsvBundle:
    """CSV writers with a fixed schema and explicit overwrite protection."""

    def __init__(self, output_dir: Path, overwrite: bool = False):
        self.output_dir = output_dir
        self.paths = {key: output_dir / name for key, (name, _) in FILE_SPECS.items()}
        existing = [path for path in self.paths.values() if path.exists()]
        if existing and not overwrite:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"Sprint 8 output already exists ({names}). Use --overwrite only "
                "when intentionally regenerating it from the same real sources.")
        output_dir.mkdir(parents=True, exist_ok=True)
        self._files = {}
        self.writers = {}
        self.counts = {key: 0 for key in FILE_SPECS}
        for key, path in self.paths.items():
            _, fields = FILE_SPECS[key]
            handle = open(path, "w", newline="", encoding="utf-8")
            self._files[key] = handle
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            self.writers[key] = writer

    def write(self, key: str, row: Mapping[str, object]) -> None:
        self.writers[key].writerow(row)
        self.counts[key] += 1

    def close(self) -> None:
        for handle in self._files.values():
            handle.close()


def _time_to_tick(trace_times: np.ndarray, time_s: float) -> int:
    """Resolve an event time to the exact recorded trace tick."""
    tick = int(np.searchsorted(trace_times, time_s))
    if tick < len(trace_times) and np.isclose(trace_times[tick], time_s, atol=1e-6):
        return tick
    if tick > 0 and np.isclose(trace_times[tick - 1], time_s, atol=1e-6):
        return tick - 1
    raise ValueError(f"event time {time_s} is absent from the Digital Twin trace")


def load_recovery_pairs(event_log_csv: str, trace_times: np.ndarray) -> List[RecoveryPair]:
    """Pair each actual host failure with the next actual recovery on that node."""
    events = []
    with open(event_log_csv, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["type"] in {"HOST_FAILURE", "HOST_RECOVERED"}:
                events.append((float(row["time"]), int(row["nodeId"]), row["type"]))
    events.sort()

    pending: Dict[int, float] = {}
    pairs: List[RecoveryPair] = []
    for time_s, node_id, event_type in events:
        if event_type == "HOST_FAILURE":
            if node_id in pending:
                raise ValueError(
                    f"host {node_id} failed again at {time_s} before recovering "
                    f"from {pending[node_id]}")
            pending[node_id] = time_s
        else:
            failure_time = pending.pop(node_id, None)
            if failure_time is None:
                raise ValueError(f"host {node_id} recovered at {time_s} without a prior failure")
            pairs.append(RecoveryPair(
                node_id=node_id,
                failure_time_s=failure_time,
                recovery_time_s=time_s,
                failure_tick=_time_to_tick(trace_times, failure_time),
                recovery_tick=_time_to_tick(trace_times, time_s),
            ))
    if pending:
        raise ValueError(f"unpaired host failures in {event_log_csv}: {sorted(pending)}")
    return pairs


def load_oof_predictions(cfg: EnvConfig) -> tuple[List[dict], dict]:
    """Load only the saved, leakage-safe out-of-fold prediction evidence."""
    artifact = np.load(cfg.oof_npz)
    _, labels, node_ids, event_ids, end_times = build_windows(
        cfg.trace_csv, cfg.sequence_length, raw_features=OBSERVABLE_COLUMNS,
        return_times=True)
    for key, actual, saved in (
        ("y", labels, artifact["y"]),
        ("node_ids", node_ids, artifact["node_ids"]),
        ("event_ids", event_ids, artifact["event_ids"]),
    ):
        if not np.array_equal(actual, saved):
            raise ValueError(f"OOF artifact no longer aligns with the trace ({key})")

    covered = artifact["covered"].astype(bool)
    if not covered.all():
        raise ValueError("OOF artifact has uncovered windows; refusing partial metrics")
    probs = artifact["prob"].astype(float)
    predictions = artifact["pred"].astype(int)
    trace_times = np.unique(np.loadtxt(cfg.trace_csv, delimiter=",", skiprows=1,
                                       usecols=0))

    rows = []
    for i in range(len(labels)):
        label = int(labels[i])
        pred = int(predictions[i])
        rows.append({
            "node_id": int(node_ids[i]),
            "window_end_time_s": float(end_times[i]),
            "recorded_tick": _time_to_tick(trace_times, float(end_times[i])),
            "predicted_failure_risk": float(probs[i]),
            "oof_predicted_label": pred,
            "will_fail_soon_label": label,
            "prediction_correct": int(pred == label),
            "event_id": int(event_ids[i]),
            "prediction_source": "leakage_safe_out_of_fold",
            "policy_observable": "offline_evaluation_only",
        })
    metrics = full_metrics(labels, predictions, probs)
    return rows, metrics


def infer_training_loss_csv(model_path: Path, supplied: str | None) -> Path:
    """Find the existing update log; never create a loss through evaluation."""
    if supplied:
        path = Path(supplied)
    else:
        stem = model_path.stem
        candidates = [model_path.parent / f"{stem}_updates.csv"]
        if stem.endswith("_best"):
            candidates.append(model_path.parent / f"{stem[:-5]}_updates.csv")
        path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not path.exists():
        raise FileNotFoundError(
            f"No existing MAPPO update CSV at {path}. Supply --training-loss-csv; "
            "Sprint 8 will not calculate a new loss during evaluation.")
    return path


def copy_training_loss(source: Path, bundle: CsvBundle) -> None:
    """Copy actual MAPPO update diagnostics with source provenance."""
    with open(source, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in TRAINING_LOSS_FIELDS[1:] if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"training update log lacks required fields: {missing}")
        for row in reader:
            bundle.write("training_loss", {
                "source_csv": str(source),
                **{field: row[field] for field in TRAINING_LOSS_FIELDS[1:]},
            })


class Sprint8Observer:
    """Evaluation-only observer used by ``marl.rollout.run_episode``."""

    def __init__(self, bundle: CsvBundle, run_id: str,
                 recovery_pairs: Sequence[RecoveryPair]):
        self.bundle = bundle
        self.run_id = run_id
        self.recovery_pairs = list(recovery_pairs)
        self.episode = 0
        self.policy_name = ""
        self.node_sums: Dict[str, float] = {}
        self.node_samples = 0
        self.task_before: Dict[int, dict] = {}
        self.priority_start: Dict[int, float] = {}

    def on_episode_start(self, env: DTMarlEnv, start_tick: int, seed: int,
                         policy_name: str) -> None:
        del seed  # Seeding remains an environment concern, not a metric.
        self.episode += 1
        self.policy_name = policy_name
        self.node_sums = {key: 0.0 for key in (
            "link_latency_ms", "power_w", "cpu_utilization_pct",
            "ram_utilization_pct", "bandwidth_utilization_pct",
        )}
        self.node_samples = 0
        self.task_before = {}
        self.priority_start = {
            task.spec.task_id: float(task.spec.priority_at(0.0)) for task in env.tasks
        }
        if start_tick != env.t0:
            raise AssertionError("rollout start tick differs from environment state")

    def before_step(self, env: DTMarlEnv, actions: np.ndarray,
                    observations: np.ndarray, action_masks: np.ndarray) -> None:
        del actions, observations, action_masks
        decision_step = env.step_idx
        tick_start = env.tick(decision_step)
        tick_end = env.tick(decision_step + 1)
        self.task_before = {
            task.spec.task_id: {
                "state": task.state,
                "node": task.node,
                "destination": task.dest,
                "reward_owner": task.reward_owner,
            }
            for task in env.tasks
        }

        # One action spans cfg.ticks_per_step trace ticks. Export every source
        # tick so this CSV retains its promised raw-tick granularity.
        for recorded_tick in range(tick_start, tick_end):
            duration_s = float(env.trace.times[recorded_tick + 1]
                               - env.trace.times[recorded_tick])
            for node_id in range(env.n_agents):
                power_w = float(env.trace.ch("energy")[node_id, recorded_tick])
                row = {
                    "run_id": self.run_id,
                    "policy": self.policy_name,
                    "episode": self.episode,
                    "episode_start_tick": env.t0,
                    "decision_step": decision_step,
                    "recorded_tick": recorded_tick,
                    "sim_time_s": float(env.trace.times[recorded_tick]),
                    "tick_duration_s": duration_s,
                    "node_id": node_id,
                    "link_latency_ms": float(env.trace.ch("linkLatencyMs")[node_id, recorded_tick]),
                    "power_w": power_w,
                    "energy_interval_j": power_w * duration_s,
                    "cpu_utilization_pct": float(env.trace.ch("cpu")[node_id, recorded_tick]),
                    "ram_utilization_pct": float(env.trace.ch("ram")[node_id, recorded_tick]),
                    "bandwidth_utilization_pct": float(env.trace.ch("bandwidth")[node_id, recorded_tick]),
                    "running_tasks_telemetry": float(env.trace.ch("runningTasks")[node_id, recorded_tick]),
                    "active": int(env.trace.ch("active")[node_id, recorded_tick] >= 0.5),
                    "degraded": int(env.trace.ch("degraded")[node_id, recorded_tick] >= 0.5),
                    "link_up": int(env.trace.ch("linkUp")[node_id, recorded_tick] >= 0.5),
                    "link_bandwidth_mbps": float(env.trace.ch("linkBandwidthMbps")[node_id, recorded_tick]),
                    "link_packet_loss_pct": float(env.trace.ch("linkPacketLoss")[node_id, recorded_tick]),
                    "under_attack": int(env.trace.ch("underAttack")[node_id, recorded_tick] >= 0.5),
                    "predicted_failure_risk": float(env.risk.at(node_id, recorded_tick)),
                    "prediction_uncertainty_reserved": float(env.risk.uncertainty_at(node_id, recorded_tick)),
                }
                self.bundle.write("node_ticks", row)
                for key in self.node_sums:
                    self.node_sums[key] += float(row[key])
                self.node_samples += 1

    def after_step(self, env: DTMarlEnv, actions: np.ndarray, rewards: np.ndarray,
                   done: bool, info: Mapping[str, object]) -> None:
        del done
        decision_step = env.step_idx - 1
        tick_start = env.tick(decision_step)
        tick_end = env.tick()
        events = info["events"]
        team_reward = float(np.sum(rewards))
        for agent_id, event in enumerate(events):
            requested = int(actions[agent_id])
            executed = int(event["action"])
            focus_id = int(event["task_id"])
            self.bundle.write("step_rewards", {
                "run_id": self.run_id,
                "policy": self.policy_name,
                "episode": self.episode,
                "episode_start_tick": env.t0,
                "decision_step": decision_step,
                "recorded_tick_start": tick_start,
                "recorded_tick_end": tick_end,
                "agent_id": agent_id,
                "requested_action": requested,
                "requested_action_name": ACTION_NAMES[requested],
                "executed_action": executed,
                "executed_action_name": ACTION_NAMES[executed],
                "agent_reward": float(rewards[agent_id]),
                "team_reward": team_reward,
                "completed_delta": int(event["completed"]),
                "lost_delta": int(event["lost"]),
                "sla_delta": int(event["sla"]),
                "migration_cost": float(event["migration_cost"]),
                "energy_reward_cost": float(event["energy"]),
                "progress_fraction": float(event["progress"]),
                "infeasible_action": int(bool(event["infeasible"])),
                "focus_task_id": focus_id if focus_id >= 0 else "",
            })

        for task in env.tasks:
            before = self.task_before[task.spec.task_id]
            if task.state not in (COMPLETED, LOST) or before["state"] in (COMPLETED, LOST):
                continue
            latency_s = ""
            available = 0
            if task.state == COMPLETED:
                latency_s = ((task.finish_step - task.arrival_step) * env.dt
                             + task.wan_latency_ms / 1000.0)
                available = 1
            self.bundle.write("task_events", {
                "run_id": self.run_id,
                "policy": self.policy_name,
                "episode": self.episode,
                "episode_start_tick": env.t0,
                "decision_step": decision_step,
                "recorded_tick": tick_end,
                "sim_time_s": float(env.trace.times[tick_end]),
                "event_type": task.state,
                "task_id": task.spec.task_id,
                "patient_id": task.spec.patient_id,
                "prior_state": before["state"],
                "terminal_state": task.state,
                "node_id": task.node,
                "destination_node_id": task.dest if task.dest >= 0 else "",
                "reward_owner": task.reward_owner,
                "arrival_step": task.arrival_step,
                "start_step": task.start_step if task.start_step >= 0 else "",
                "finish_step": task.finish_step if task.finish_step >= 0 else "",
                "lost_step": task.lost_step if task.lost_step >= 0 else "",
                "task_latency_s": latency_s,
                "task_latency_available": available,
                "migrations": task.migrations,
                "reroutes": task.reroutes,
                "hsi": float(task.spec.attributes["hsi"]),
                "clinical_severity": float(task.spec.severity),
                "priority_at_event": float(task.spec.priority_at(env.elapsed_s)),
                "deadline_breached": int(task.deadline_breached),
            })

    def on_episode_end(self, env: DTMarlEnv, metrics: Mapping[str, object]) -> None:
        # episode_metrics() has already called finalise_migration_outcomes().
        for migration in env.migrations:
            recorded_tick = env.tick(migration.step)
            self.bundle.write("migrations", {
                "run_id": self.run_id,
                "policy": self.policy_name,
                "episode": self.episode,
                "episode_start_tick": env.t0,
                "decision_step": migration.step,
                "recorded_tick": recorded_tick,
                "sim_time_s": float(env.trace.times[recorded_tick]),
                "task_id": migration.task_id,
                "patient_id": migration.patient_id,
                "clinical_severity": migration.severity,
                "priority": migration.priority,
                "action": migration.action,
                "action_name": migration.action_name,
                "source_node_id": migration.source_node,
                "destination_node_id": migration.dest_node,
                "migration_cost": migration.cost,
                "latency_steps": migration.latency_steps,
                "preemptive": int(migration.preemptive),
                "source_symptomatic": int(migration.source_symptomatic),
                "source_risk": migration.source_risk,
                "destination_risk": migration.dest_risk,
                "source_failed_within_window": int(bool(migration.source_failed_within_window)),
                "task_survived": int(bool(migration.task_survived)),
                "outcome_scope": "post_episode_evaluation_only",
            })

        end_tick = env.tick()
        start_time = float(env.trace.times[env.t0])
        end_time = float(env.trace.times[end_tick])
        failures = [pair for pair in self.recovery_pairs
                    if start_time <= pair.failure_time_s < end_time]
        recoveries = [pair for pair in self.recovery_pairs
                      if start_time <= pair.recovery_time_s < end_time]
        hsi = np.asarray([task.spec.attributes["hsi"] for task in env.tasks], dtype=float)
        priority_end = np.asarray(
            [task.spec.priority_at(env.elapsed_s) for task in env.tasks], dtype=float)
        priority_start = np.asarray(list(self.priority_start.values()), dtype=float)
        mean = lambda key: self.node_sums[key] / max(self.node_samples, 1)
        self.bundle.write("episode_metrics", {
            "run_id": self.run_id,
            "policy": self.policy_name,
            "episode": self.episode,
            "episode_start_tick": env.t0,
            "episode_end_tick": end_tick,
            "episode_start_time_s": start_time,
            "episode_end_time_s": end_time,
            "decision_steps": metrics["steps"],
            "episode_reward": metrics["episode_reward"],
            "task_success_rate": metrics["task_success_rate"],
            "completed": metrics["completed"],
            "lost": metrics["lost"],
            "unfinished": metrics["unfinished"],
            "avg_task_latency_s": metrics["avg_task_latency_s"],
            "energy_reward_cost": metrics["energy_cost"],
            "migrations": metrics["migrations"],
            "migrations_edge": metrics["migrations_edge"],
            "migrations_cloud": metrics["migrations_cloud"],
            "reroutes": metrics["reroutes"],
            "relocations": metrics["relocations"],
            "preemptive_relocations": metrics["preemptive_relocations"],
            "reactive_relocations": metrics["reactive_relocations"],
            "tasks_protected_before_failure": metrics["tasks_protected_before_failure"],
            "sla_violations": metrics["sla_violations"],
            "mean_link_latency_ms": mean("link_latency_ms"),
            "mean_power_w": mean("power_w"),
            "mean_cpu_utilization_pct": mean("cpu_utilization_pct"),
            "mean_ram_utilization_pct": mean("ram_utilization_pct"),
            "mean_bandwidth_utilization_pct": mean("bandwidth_utilization_pct"),
            "host_failures_started_in_window": len(failures),
            "host_recoveries_completed_in_window": len(recoveries),
            "mean_recovery_duration_s": (
                float(np.mean([pair.duration_s for pair in recoveries])) if recoveries else ""),
            "mean_hsi": float(hsi.mean()),
            "min_hsi": float(hsi.min()),
            "max_hsi": float(hsi.max()),
            "mean_priority_at_start": float(priority_start.mean()),
            "mean_priority_at_end": float(priority_end.mean()),
            "prediction_uncertainty_status": "reserved_zero_not_an_estimate",
        })

        for task in env.tasks:
            attrs = task.spec.attributes
            self.bundle.write("task_metadata", {
                "run_id": self.run_id,
                "policy": self.policy_name,
                "episode": self.episode,
                "episode_start_tick": env.t0,
                "task_id": task.spec.task_id,
                "patient_id": task.spec.patient_id,
                "hsi": float(attrs["hsi"]),
                "vitals_instability": float(attrs["vitalsInstability"]),
                "age_years": float(attrs["age"]),
                "clinical_severity": float(task.spec.severity),
                "priority_at_episode_start": self.priority_start[task.spec.task_id],
                "priority_at_episode_end": float(task.spec.priority_at(env.elapsed_s)),
                "arrival_time_s": task.spec.arrival_time,
                "deadline_s": task.spec.deadline,
                "final_state": task.state,
                "final_node_id": task.node,
                "migrations": task.migrations,
                "reroutes": task.reroutes,
            })


def build_evaluation_env(model_path: Path, device: str) -> tuple[DTMarlEnv, MappoPolicy]:
    """Load a validated checkpoint for greedy evaluation without modifying it."""
    if not model_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {model_path}")
    agent, extra = MAPPO.load(model_path, device=resolve_device(device))
    agent.eval()
    cfg = Sprint6Config()
    saved = extra.get("config", {})
    if saved:
        cfg.env = EnvConfig(**saved["env"])
        cfg.reward = RewardConfig(**saved["reward"])
    cfg.env.start_frac_lo, cfg.env.start_frac_hi = DEFAULT_TRAIN_FRAC, 1.0
    env = DTMarlEnv(cfg.env, cfg.reward)
    return env, MappoPolicy(agent, "mappo-greedy")


def write_static_sources(bundle: CsvBundle, env: DTMarlEnv,
                         recovery_pairs: Sequence[RecoveryPair]) -> None:
    """Write source-derived rows that do not depend on a policy rollout."""
    for pair in recovery_pairs:
        bundle.write("recovery", {
            "node_id": pair.node_id,
            "failure_time_s": pair.failure_time_s,
            "failure_tick": pair.failure_tick,
            "recovery_time_s": pair.recovery_time_s,
            "recovery_tick": pair.recovery_tick,
            "recovery_duration_s": pair.duration_s,
            "failure_event_type": "HOST_FAILURE",
            "recovery_event_type": "HOST_RECOVERED",
            "source_csv": str(env.cfg.event_log_csv),
        })
    prediction_rows, prediction_metrics = load_oof_predictions(env.cfg)
    for row in prediction_rows:
        bundle.write("prediction_windows", row)
    for metric, value in prediction_metrics.items():
        bundle.write("prediction_metrics", {
            "scope": "pooled_leakage_safe_oof",
            "metric": metric,
            "value": value,
            "n_windows": prediction_metrics["n"],
            "positive_windows": prediction_metrics["positives"],
            "prediction_source": "leakage_safe_out_of_fold",
            "policy_observable": "offline_evaluation_only",
        })


def run_pipeline(args: argparse.Namespace) -> tuple[CsvBundle, Path]:
    if args.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    model_path = Path(args.model).resolve()
    env, policy = build_evaluation_env(model_path, args.device)
    loss_csv = infer_training_loss_csv(model_path, args.training_loss_csv)
    bundle = CsvBundle(Path(args.out_dir).resolve(), overwrite=args.overwrite)
    try:
        recovery_pairs = load_recovery_pairs(env.cfg.event_log_csv, env.trace.times)
        write_static_sources(bundle, env, recovery_pairs)
        copy_training_loss(loss_csv, bundle)

        observer = Sprint8Observer(bundle, run_id=args.run_id or model_path.stem,
                                    recovery_pairs=recovery_pairs)
        starts = episode_starts(env, args.episodes)
        for episode_index, start_tick in enumerate(starts):
            run_episode(env, policy, start_tick=start_tick,
                        seed=args.seed + episode_index,
                        observer=observer)
    except Exception:
        bundle.close()
        raise
    bundle.close()
    return bundle, loss_csv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Sprint 8 metrics from real DT, OOF, and MAPPO artifacts")
    parser.add_argument("--model", default=str(DEFAULT_MODEL),
                        help="existing checkpoint used only for greedy evaluation")
    parser.add_argument("--episodes", type=int, default=1,
                        help="number of fixed held-out evaluation episodes (default: 1)")
    parser.add_argument("--seed", type=int, default=20261002,
                        help="fixed evaluation reset seed; does not create telemetry")
    parser.add_argument("--device", default="cpu",
                        help="MAPPO inference device, e.g. cpu or auto")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR),
                        help="directory for the distinct Sprint 8 CSV files")
    parser.add_argument("--training-loss-csv", default=None,
                        help="existing MAPPO *_updates.csv to copy with provenance")
    parser.add_argument("--run-id", default=None,
                        help="stable identifier written to evaluation-derived CSVs")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace only prior Sprint 8 CSV exports in --out-dir")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle, loss_csv = run_pipeline(args)
    print("SPRINT 8 real metrics export complete")
    print(f"  checkpoint (read only): {Path(args.model).resolve()}")
    print(f"  training losses copied: {loss_csv}")
    print(f"  output directory:       {Path(args.out_dir).resolve()}")
    for key, path in bundle.paths.items():
        print(f"  {path.name:<38s} {bundle.counts[key]:>8d} rows")
    print("  scope: greedy evaluation only; no training and no checkpoint writes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
