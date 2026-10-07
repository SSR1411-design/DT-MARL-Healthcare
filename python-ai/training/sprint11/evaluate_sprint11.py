"""Evaluate one completed, independently seeded Sprint 11 trial plan.

This program is deliberately single-trial and refuses a blocked plan.  It is
not a matrix launcher: repeated windows are nested measurements within the
same simulator trial and are labelled as such in the resulting metadata.
At present it can execute non-predictive baselines only.  MAPPO plans remain
blocked until a trace-aligned OOF artifact and compatible checkpoint are
available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.baseline import (DetectionRecoveryPolicy, NoMigrationPolicy,
                           RandomPolicy, ReactiveThresholdPolicy)
from marl.config import EnvConfig, RewardConfig
from marl.env import COMPLETED, DTMarlEnv
from marl.rollout import episode_starts, run_episode
from training.sprint11.schema import (SchemaError, load_json, sha256_file,
                                      validate_result_record)


BASELINE_FACTORIES = {
    "static-no-migration": lambda seed: (NoMigrationPolicy(), False),
    "static": lambda seed: (NoMigrationPolicy(), False),
    "reactive-threshold": lambda seed: (ReactiveThresholdPolicy(), False),
    "reactive-no-digital-twin": lambda seed: (ReactiveThresholdPolicy(), False),
    "detection-recovery-no-prediction": lambda seed: (DetectionRecoveryPolicy(), True),
    "detection-without-prediction": lambda seed: (DetectionRecoveryPolicy(), True),
    "random-legal": lambda seed: (RandomPolicy(seed=seed), False),
}


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _finite_mean(values: list[float], label: str) -> float:
    if not values or not all(math.isfinite(value) for value in values):
        raise SchemaError(f"{label} is unavailable; it must not be zero-imputed")
    return float(mean(values))


def _priority_bin(priority: float) -> str:
    if priority < 1 / 3:
        return "low"
    if priority < 2 / 3:
        return "medium"
    return "high"


def _episode_physical_metrics(env: DTMarlEnv) -> dict[str, float]:
    """Integrate only recorded Java telemetry; no synthetic utilization data."""
    start, end = env.t0, env.tick()
    if end <= start or end > env.trace.n_ticks:
        raise SchemaError("invalid evaluation trace interval")
    times = env.trace.times[start:end]
    if len(times) < 2:
        interval_s = np.full(len(times), env.tick_dt, dtype=float)
    else:
        interval_s = np.diff(np.append(times, times[-1] + env.tick_dt))
    power = env.trace.ch("energy")[:, start:end]
    energy_j = float(np.sum(power * interval_s[np.newaxis, :]))
    return {
        "physical_energy_j": energy_j,
        "mean_cpu_utilization_pct": float(np.mean(env.trace.ch("cpu")[:, start:end])),
        "mean_ram_utilization_pct": float(np.mean(env.trace.ch("ram")[:, start:end])),
        "mean_bandwidth_utilization_pct": float(
            np.mean(env.trace.ch("bandwidth")[:, start:end])),
    }


def _priority_response_rows(env: DTMarlEnv) -> list[dict[str, Any]]:
    by_bin: dict[str, list[float]] = defaultdict(list)
    for task in env.tasks:
        if task.state != COMPLETED:
            continue
        latency = (task.finish_step - task.arrival_step) * env.dt + task.wan_latency_ms / 1000.0
        by_bin[_priority_bin(float(task.spec.priority_at(env.elapsed_s)))].append(float(latency))
    return [{"priority_bin": name, "mean_response_time_s": float(mean(values)),
             "completed_task_observations": len(values)}
            for name, values in sorted(by_bin.items())]


def load_plan(path: Path) -> dict[str, Any]:
    plan = load_json(path)
    if not isinstance(plan, dict) or plan.get("record_type") != "trial_plan":
        raise SchemaError("expected a Sprint 11 trial_plan.json")
    return plan


def build_baseline_environment(plan: Mapping[str, Any]) -> tuple[DTMarlEnv, Any]:
    if plan.get("policy_kind") != "baseline":
        raise SchemaError("only baseline plans can run until aligned OOF risk is available")
    if plan.get("plan_status") != "ready_for_simulator_only":
        raise SchemaError("trial plan is blocked; resolve every recorded blocker first")
    policy_name = str(plan["policy"])
    if policy_name not in BASELINE_FACTORIES:
        raise SchemaError("unsupported baseline policy: " + policy_name)
    output = Path(str(plan["simulator_output_directory"]))
    trace_path = output / "failure_history.csv"
    event_path = output / "failure_log.csv"
    if not trace_path.is_file() or not event_path.is_file():
        raise FileNotFoundError("completed isolated simulator trace is required before evaluation")
    cfg = EnvConfig()
    cfg.trace_csv = str(trace_path)
    cfg.event_log_csv = str(event_path)
    cfg.risk_source = "zero"
    cfg.dest_w_risk = 0.0
    cfg.n_edge_nodes = int(plan["edge_node_count"])
    cfg.n_tasks = int(plan["task_count"])
    cfg.n_patients = int(plan["patient_count"])
    cfg.seed = int(plan["python_seed"])
    cfg.random_episode_start = False
    # This is a baseline evaluation, not a train/test claim.  Windows are
    # deterministic nested observations within the independently seeded trial.
    cfg.start_frac_lo, cfg.start_frac_hi = 0.0, 1.0
    reward = RewardConfig()
    reward.P_risk_expose = 0.0
    policy, detection_recovery = BASELINE_FACTORIES[policy_name](int(plan["python_seed"]))
    return DTMarlEnv(cfg, reward, detection_recovery=detection_recovery), policy


def evaluate_plan(plan_path: Path, episodes: int, result_path: Path | None = None) -> Path:
    if episodes < 1:
        raise SchemaError("episodes must be positive")
    plan = load_plan(plan_path)
    env, policy = build_baseline_environment(plan)
    starts = episode_starts(env, episodes, seed=int(plan["python_seed"]))
    metrics_by_episode: list[dict[str, Any]] = []
    physical_by_episode: list[dict[str, float]] = []
    episode_ends: list[int] = []
    priority_values: dict[str, list[float]] = defaultdict(list)
    distributions: list[dict[str, Any]] = []
    for index, start in enumerate(starts):
        metrics, _, _, _ = run_episode(env, policy, start, seed=int(plan["python_seed"]) + index)
        physical = _episode_physical_metrics(env)
        metrics_by_episode.append(metrics)
        physical_by_episode.append(physical)
        episode_ends.append(env.tick())
        for row in _priority_response_rows(env):
            priority_values[row["priority_bin"]].append(row["mean_response_time_s"])
        hsi = [float(task.spec.attributes["hsi"]) for task in env.tasks]
        distributions.extend([
            {"metric": "avg_task_latency_s", "value": metrics["avg_task_latency_s"],
             "sample_unit": "nested_evaluation_window"},
            {"metric": "physical_energy_j", "value": physical["physical_energy_j"],
             "sample_unit": "nested_evaluation_window"},
            {"metric": "episode_reward", "value": metrics["episode_reward"],
             "sample_unit": "nested_evaluation_window"},
            *[{"metric": "HSI", "value": value,
               "sample_unit": "task_within_nested_evaluation_window"} for value in hsi],
        ])
    trace_path = Path(str(plan["simulator_output_directory"])) / "failure_history.csv"
    result = {
        "schema_version": plan["schema_version"],
        "record_type": "trial_result",
        "scenario_id": plan["scenario_id"],
        "trial_id": plan["trial_id"],
        "policy": plan["policy"],
        "policy_kind": plan["policy_kind"],
        "simulator_seed": plan["simulator_seed"],
        "python_seed": plan["python_seed"],
        "host_count": plan["host_count"],
        "edge_node_count": plan["edge_node_count"],
        "device_count": plan["device_count"],
        "task_count": plan["task_count"],
        "patient_count": plan["patient_count"],
        "bandwidth_mbps": plan["bandwidth_mbps"],
        "failure_mode": plan["failure_mode"],
        "failure_intensity": plan["failure_intensity"],
        "failure_parameters": plan["failure_parameters"],
        "trace_id": trace_path.name,
        "trace_path": str(trace_path.resolve()),
        "trace_hash": sha256_file(trace_path),
        "risk_artifact_id": "",
        "risk_artifact_hash": "",
        "risk_alignment_status": "unavailable",
        "checkpoint_id": "",
        "checkpoint_hash": "",
        "checkpoint_compatibility": "not_applicable",
        "episode_start": min(starts),
        "episode_end": max(episode_ends),
        "episode_starts": starts,
        "evaluation_protocol": (
            f"{len(starts)} deterministic nested evaluation windows within one "
            "independently seeded simulator trial; windows are not independent trials"),
        "uncertainty_gate_status": "not_available_reserved_zero_interface",
        "metrics": {
            "task_success_rate": _finite_mean([m["task_success_rate"] for m in metrics_by_episode], "task_success_rate"),
            "avg_task_latency_s": _finite_mean([m["avg_task_latency_s"] for m in metrics_by_episode], "avg_task_latency_s"),
            "physical_energy_j": _finite_mean([m["physical_energy_j"] for m in physical_by_episode], "physical_energy_j"),
            "energy_reward_cost": _finite_mean([m["energy_cost"] for m in metrics_by_episode], "energy_reward_cost"),
            "mean_cpu_utilization_pct": _finite_mean([m["mean_cpu_utilization_pct"] for m in physical_by_episode], "mean_cpu_utilization_pct"),
            "mean_ram_utilization_pct": _finite_mean([m["mean_ram_utilization_pct"] for m in physical_by_episode], "mean_ram_utilization_pct"),
            "mean_bandwidth_utilization_pct": _finite_mean([m["mean_bandwidth_utilization_pct"] for m in physical_by_episode], "mean_bandwidth_utilization_pct"),
            "episode_reward": _finite_mean([m["episode_reward"] for m in metrics_by_episode], "episode_reward"),
            "HSI": _finite_mean([float(task.spec.attributes["hsi"]) for task in env.tasks], "HSI"),
        },
        "priority_response_time_s": [
            {"priority_bin": key, "mean_response_time_s": _finite_mean(values, f"response time {key}"),
             "nested_window_count": len(values)}
            for key, values in sorted(priority_values.items())
        ],
        "distribution_observations": distributions,
    }
    validate_result_record(result, require_files=True)
    target = result_path or plan_path.with_name("trial_result.json")
    _write_new_json(target, result)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-plan", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=5,
                        help="nested deterministic windows; not independent trials")
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    print(f"Wrote validated baseline trial result: {evaluate_plan(args.trial_plan, args.episodes, args.result)}")


if __name__ == "__main__":
    main()
