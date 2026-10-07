"""Trace-backed Sprint 9 metric exports for the three baseline definitions.

This exporter records one actual evaluation rollout.  It never calls a
predictor for the two non-predictive baselines and it never manufactures
measurements: node telemetry comes from the Digital Twin trace and event rows
come from the live ``DTMarlEnv`` task lifecycle.

Examples (run from ``python-ai``)::

    python training/sprint9_metrics.py --baseline reactive-no-digital-twin
    python training/sprint9_metrics.py --baseline detection-without-prediction

The single-agent option intentionally needs its own separately trained
checkpoint and is not usable until the later manual training phase.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.config import ACTION_NAMES
from marl.env import COMPLETED, LOST, DTMarlEnv
from marl.rollout import run_episode
from marl.sprint9_benchmarks import (
    BASELINE_DETECTION,
    SUPPORTED_BASELINES,
    BaselineRun,
    build_baseline,
    deterministic_starts,
)


DEFAULT_OUTPUT_ROOT = ROOT / "data" / "sprint9"

METADATA_FIELDS = [
    "run_id", "baseline_name", "episode", "episode_start_tick", "risk_source",
    "uses_digital_twin", "uses_prediction", "uses_uncertainty", "uses_marl",
    "seed", "node_count", "task_count", "recovery_mode", "training_loss_status",
    "checkpoint",
]
NODE_TICK_FIELDS = [
    "run_id", "baseline_name", "episode", "episode_start_tick", "decision_step",
    "recorded_tick", "sim_time_s", "tick_duration_s", "node_id", "link_latency_ms",
    "power_w", "energy_interval_j", "cpu_utilization_pct", "ram_utilization_pct",
    "bandwidth_utilization_pct", "running_tasks_telemetry", "active", "degraded",
    "link_up", "link_bandwidth_mbps", "link_packet_loss_pct", "under_attack",
    "prediction_used", "uncertainty_used",
]
STEP_REWARD_FIELDS = [
    "run_id", "baseline_name", "episode", "episode_start_tick", "decision_step",
    "recorded_tick_start", "recorded_tick_end", "agent_id", "requested_action",
    "requested_action_name", "executed_action", "executed_action_name", "agent_reward",
    "team_reward", "completed_delta", "lost_delta", "sla_delta", "migration_cost",
    "energy_reward_cost", "progress_fraction", "infeasible_action", "focus_task_id",
]
TASK_EVENT_FIELDS = [
    "run_id", "baseline_name", "episode", "episode_start_tick", "decision_step",
    "recorded_tick", "sim_time_s", "event_type", "task_id", "patient_id",
    "prior_state", "terminal_state", "node_id", "destination_node_id", "reward_owner",
    "arrival_step", "start_step", "finish_step", "lost_step", "task_latency_s",
    "task_latency_available", "migrations", "reroutes", "hsi", "clinical_severity",
    "priority_at_event", "deadline_breached",
]
TASK_METADATA_FIELDS = [
    "run_id", "baseline_name", "episode", "episode_start_tick", "task_id", "patient_id",
    "hsi", "vitals_instability", "age_years", "clinical_severity",
    "priority_at_episode_start", "priority_at_episode_end", "arrival_time_s", "deadline_s",
    "final_state", "final_node_id", "migrations", "reroutes",
]
MIGRATION_FIELDS = [
    "run_id", "baseline_name", "episode", "episode_start_tick", "decision_step",
    "recorded_tick", "sim_time_s", "task_id", "patient_id", "clinical_severity",
    "priority", "action", "action_name", "source_node_id", "destination_node_id",
    "migration_cost", "latency_steps", "preemptive", "source_symptomatic",
    "source_risk", "destination_risk", "source_failed_within_window", "task_survived",
    "outcome_scope",
]
RECOVERY_FIELDS = [
    "run_id", "baseline_name", "episode", "episode_start_tick", "detection_step",
    "failure_window_start_tick", "failure_window_end_tick", "detection_tick",
    "detection_time_s", "task_id", "patient_id", "source_node_id", "destination_node_id",
    "destination_observed_healthy", "recovery_tick", "recovery_time_s",
    "recovery_latency_s", "recovery_succeeded", "reason",
]
PREDICTION_USAGE_FIELDS = [
    "run_id", "baseline_name", "episode", "episode_start_tick", "risk_source",
    "prediction_used", "uncertainty_used", "oof_loaded", "live_predictor_loaded",
    "offline_only_note",
]
EPISODE_FIELDS = [
    "run_id", "baseline_name", "episode", "episode_start_tick", "episode_end_tick",
    "episode_start_time_s", "episode_end_time_s", "decision_steps", "episode_reward",
    "task_success_rate", "completed", "lost", "unfinished", "avg_task_latency_s",
    "energy_reward_cost", "migrations", "migrations_edge", "migrations_cloud", "reroutes",
    "relocations", "preemptive_relocations", "reactive_relocations", "sla_violations",
    "recovery_attempts", "recoveries_succeeded", "recoveries_failed",
    "mean_recovery_latency_s", "mean_link_latency_ms", "mean_power_w",
    "mean_cpu_utilization_pct", "mean_ram_utilization_pct",
    "mean_bandwidth_utilization_pct", "mean_hsi", "min_hsi", "max_hsi",
    "mean_priority_at_start", "mean_priority_at_end", "risk_source", "prediction_used",
    "uncertainty_used", "training_loss_status",
]

FILE_SPECS = {
    "metadata": ("sprint9_metadata.csv", METADATA_FIELDS),
    "node_ticks": ("sprint9_node_ticks.csv", NODE_TICK_FIELDS),
    "step_rewards": ("sprint9_step_rewards.csv", STEP_REWARD_FIELDS),
    "task_events": ("sprint9_task_events.csv", TASK_EVENT_FIELDS),
    "task_metadata": ("sprint9_task_metadata.csv", TASK_METADATA_FIELDS),
    "migrations": ("sprint9_migrations.csv", MIGRATION_FIELDS),
    "recovery_events": ("sprint9_recovery_events.csv", RECOVERY_FIELDS),
    "prediction_usage": ("sprint9_prediction_usage.csv", PREDICTION_USAGE_FIELDS),
    "episode_metrics": ("sprint9_episode_metrics.csv", EPISODE_FIELDS),
}


class CsvBundle:
    """Fixed schemas with opt-in replacement of only prior Sprint 9 outputs."""

    def __init__(self, output_dir: Path, overwrite: bool = False):
        self.output_dir = output_dir
        self.paths = {key: output_dir / spec[0] for key, spec in FILE_SPECS.items()}
        existing = [path for path in self.paths.values() if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(
                "Sprint 9 outputs already exist: " + ", ".join(p.name for p in existing)
                + ". Use --overwrite only to regenerate Sprint 9 files.")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.files = {}
        self.writers = {}
        self.counts = {key: 0 for key in FILE_SPECS}
        for key, path in self.paths.items():
            fields = FILE_SPECS[key][1]
            handle = open(path, "w", newline="", encoding="utf-8")
            self.files[key] = handle
            self.writers[key] = csv.DictWriter(handle, fieldnames=fields,
                                                extrasaction="raise")
            self.writers[key].writeheader()

    def write(self, key: str, row: Mapping[str, object]) -> None:
        self.writers[key].writerow(row)
        self.counts[key] += 1

    def close(self) -> None:
        for handle in self.files.values():
            handle.close()


@dataclass
class _EpisodeTelemetry:
    sums: Dict[str, float]
    samples: int = 0


class Sprint9Observer:
    """Read-only instrumentation hook shared by the two heuristic baselines."""

    def __init__(self, bundle: CsvBundle, run: BaselineRun, run_id: str):
        self.bundle = bundle
        self.run = run
        self.run_id = run_id
        self.episode = 0
        self.task_before: Dict[int, dict] = {}
        self.priority_start: Dict[int, float] = {}
        self.telemetry = _EpisodeTelemetry({})

    @property
    def metadata(self):
        return self.run.metadata

    def _base(self, env: DTMarlEnv) -> dict:
        return dict(run_id=self.run_id, baseline_name=self.metadata.baseline_name,
                    episode=self.episode, episode_start_tick=env.t0)

    def on_episode_start(self, env: DTMarlEnv, start_tick: int, seed: int,
                         policy_name: str) -> None:
        del policy_name
        self.episode += 1
        if start_tick != env.t0:
            raise AssertionError("rollout start tick differs from environment state")
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
            **self._base(env), "risk_source": self.metadata.risk_source,
            "prediction_used": int(self.metadata.uses_prediction),
            "uncertainty_used": int(self.metadata.uses_uncertainty), "oof_loaded": 0,
            "live_predictor_loaded": 0,
            "offline_only_note": (
                "No OOF windows or live predictor are loaded for this baseline; "
                "risk is hard-zeroed.") if not self.metadata.uses_prediction else
                "Evaluation policy configuration determines prediction use.",
        })

    def before_step(self, env: DTMarlEnv, actions: np.ndarray,
                    observations: np.ndarray, action_masks: np.ndarray) -> None:
        del actions, observations, action_masks
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
                    "prediction_used": int(self.metadata.uses_prediction),
                    "uncertainty_used": int(self.metadata.uses_uncertainty),
                }
                self.bundle.write("node_ticks", row)
                for key in self.telemetry.sums:
                    self.telemetry.sums[key] += float(row[key])
                self.telemetry.samples += 1

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
            latency = ""
            latency_available = 0
            if task.state == COMPLETED:
                latency = ((task.finish_step - task.arrival_step) * env.dt
                           + task.wan_latency_ms / 1000.0)
                latency_available = 1
            self.bundle.write("task_events", {
                **self._base(env), "decision_step": step, "recorded_tick": end,
                "sim_time_s": float(env.trace.times[end]), "event_type": task.state,
                "task_id": task.spec.task_id, "patient_id": task.spec.patient_id,
                "prior_state": before["state"], "terminal_state": task.state,
                "node_id": task.node, "destination_node_id": (
                    task.dest if task.dest >= 0 else ""), "reward_owner": task.reward_owner,
                "arrival_step": task.arrival_step,
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
        # run_episode calls episode_metrics before this hook, so migration
        # outcome fields were annotated only for post-episode analysis.
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
            "sla_violations": metrics["sla_violations"],
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
            "mean_priority_at_end": float(priorities.mean()),
            "risk_source": self.metadata.risk_source,
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


def run_pipeline(args: argparse.Namespace) -> CsvBundle:
    run = build_baseline(args.baseline, seed=args.seed,
                         single_agent_checkpoint=args.single_agent_checkpoint)
    starts = deterministic_starts(run, args.episodes)
    output_dir = Path(args.out_dir).resolve()
    bundle = CsvBundle(output_dir, overwrite=args.overwrite)
    observer = Sprint9Observer(bundle, run, args.run_id or args.baseline)
    try:
        for episode, start in enumerate(starts):
            run_episode(run.environment, run.policy, int(start),
                        seed=args.seed + episode, observer=observer)
    except Exception:
        bundle.close()
        raise
    bundle.close()
    return bundle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export real Sprint 9 baseline evaluation metrics")
    parser.add_argument("--baseline", choices=SUPPORTED_BASELINES,
                        default=BASELINE_DETECTION)
    parser.add_argument("--episodes", type=int, default=1,
                        help="bounded count of deterministic held-out episodes")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--out-dir", default=None,
                        help="default: data/sprint9/<baseline>")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--single-agent-checkpoint", default=None,
                        help="only for a separately trained single-agent PPO checkpoint")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace only existing Sprint 9 files in --out-dir")
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.out_dir is None:
        args.out_dir = str(DEFAULT_OUTPUT_ROOT / args.baseline)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    bundle = run_pipeline(args)
    print("SPRINT 9 real baseline metrics export complete")
    print(f"  baseline: {args.baseline}")
    print(f"  output:   {Path(args.out_dir).resolve()}")
    for key, path in bundle.paths.items():
        print(f"  {path.name:<35s} {bundle.counts[key]:>8d} rows")
    print("  scope: bounded evaluation only; no training and no checkpoint writes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
