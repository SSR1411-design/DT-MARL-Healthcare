"""Read-only integrity validation for a completed Sprint 10 A5 export."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.config import ACTION_NAMES, EnvConfig
from marl.trace import load_trace
from training.sprint10_a5_no_uncertainty import (
    DEFAULT_OUTPUT_DIR,
    EPISODE_FIELDS,
    EXPERIMENT_NAME,
    FILE_SPECS,
    FIXED_HELD_OUT_STARTS,
    INTEGRITY_FILENAME,
    OOF_PREDICTION_ARTIFACT,
    OOF_PREDICTION_ARTIFACT_SHA256,
    README_FILENAME,
    UNCERTAINTY_A5_CHANGE,
    UNCERTAINTY_GATING_PROTOCOL,
    UNCERTAINTY_MECHANISM_FOUND,
    UNCERTAINTY_OBSERVATION_INDEX,
    VALIDATED_CHECKPOINT,
    VALIDATED_CHECKPOINT_SHA256,
    VALIDATED_CONFIG,
    VALIDATED_CONFIG_SHA256,
    build_a5_run,
    protected_a2_hashes,
    protected_a3_hashes,
    protected_a4_hashes,
    protected_sprint8_sprint9_hashes,
    sha256,
)


def fail(message: str) -> None:
    raise AssertionError(message)


def close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
        fail(f"{context}: got {actual}, expected {expected}")


def read_csvs(output_dir: Path) -> dict[str, list[dict[str, str]]]:
    rows: dict[str, list[dict[str, str]]] = {}
    required_nonempty = {
        "metadata", "node_ticks", "step_rewards", "task_metadata",
        "prediction_usage", "episode_metrics",
    }
    for key, (filename, fields) in FILE_SPECS.items():
        path = output_dir / filename
        if not path.is_file():
            fail(f"missing A5 CSV: {path}")
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fields:
                fail(f"{path.name} has an unexpected header")
            rows[key] = list(reader)
        if key in required_nonempty and not rows[key]:
            fail(f"{path.name} has no actual A5 evaluation rows")
    if FILE_SPECS["episode_metrics"][1] != EPISODE_FIELDS:
        fail("A5 episode schema was unexpectedly replaced")
    readme = output_dir / README_FILENAME
    if not readme.is_file() or not readme.read_text(encoding="utf-8").strip():
        fail("missing or empty A5 README")
    return rows


def _starts(entries: Sequence[Mapping[str, str]]) -> list[int]:
    return [int(row["episode_start_tick"]) for row in entries]


def validate_fixed_starts(rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    expected = list(FIXED_HELD_OUT_STARTS)
    metrics = rows["episode_metrics"]
    if len(metrics) != len(expected):
        fail(f"A5 must have exactly {len(expected)} episodes, found {len(metrics)}")
    if _starts(metrics) != expected:
        fail("A5 episode metrics do not use the validated R2 held-out starts")
    if [int(row["episode"]) for row in metrics] != list(range(1, len(expected) + 1)):
        fail("A5 episode numbering is not the fixed eight-episode protocol")
    for key in ("metadata", "prediction_usage"):
        if len(rows[key]) != len(expected) or _starts(rows[key]) != expected:
            fail(f"{key} does not prove ordered consumption of all eight starts")
    valid = {(index + 1, start) for index, start in enumerate(expected)}
    for key, entries in rows.items():
        for row in entries:
            if (int(row["episode"]), int(row["episode_start_tick"])) not in valid:
                fail(f"{key} contains a row outside the fixed A5 episode/start protocol")


def validate_controls(rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    for row in rows["metadata"]:
        if row["baseline_name"] != EXPERIMENT_NAME or row["experiment"] != EXPERIMENT_NAME:
            fail("A5 metadata identifies the wrong experiment")
        if row["ablation_id"] != "A5" or row["decision_mechanism"] != "frozen_greedy_mappo":
            fail("A5 metadata does not identify frozen greedy MAPPO")
        if row["mappo_action_selection_used"] != "1" or row["uses_marl"] != "True":
            fail("A5 metadata does not prove MAPPO action selection")
        if row["uses_digital_twin"] != "True" or row["uses_digital_twin_trace"] != "1":
            fail("A5 metadata does not preserve the Digital Twin trace")
        if row["uses_prediction"] != "True" or row["risk_source"] != "oof":
            fail("A5 metadata does not preserve R2 OOF failure prediction")
        if (row["criticality_awareness_enabled"] != "1"
                or float(row["criticality_weight"]) <= 0.0
                or float(row["criticality_migration_weight"]) <= 0.0):
            fail("A5 metadata does not retain active R2 criticality behavior")
        if (row["uses_uncertainty"] != "False" or row["uncertainty_gating_enabled"] != "0"
                or int(row["uncertainty_observation_index"]) != UNCERTAINTY_OBSERVATION_INDEX
                or row["uncertainty_observation_geometry_preserved"] != "1"
                or row["uncertainty_mechanism_found"] != UNCERTAINTY_MECHANISM_FOUND
                or row["uncertainty_a5_change"] != UNCERTAINTY_A5_CHANGE
                or row["uncertainty_gating_protocol"] != UNCERTAINTY_GATING_PROTOCOL):
            fail("A5 metadata does not document the reserved/inactive uncertainty counterfactual")
        if Path(row["checkpoint"]).resolve() != VALIDATED_CHECKPOINT.resolve():
            fail("A5 metadata references a nonvalidated MAPPO checkpoint")
        if row["checkpoint_sha256"] != VALIDATED_CHECKPOINT_SHA256:
            fail("A5 metadata has the wrong MAPPO checkpoint hash")
        if row["validated_r2_config_sha256"] != VALIDATED_CONFIG_SHA256:
            fail("A5 metadata has the wrong R2 configuration hash")
        if row["oof_prediction_artifact_sha256"] != OOF_PREDICTION_ARTIFACT_SHA256:
            fail("A5 metadata has the wrong OOF prediction artifact hash")
    for row in rows["prediction_usage"]:
        if (row["risk_source"] != "oof" or row["prediction_used"] != "1"
                or row["uncertainty_used"] != "0" or row["oof_loaded"] != "1"
                or row["live_predictor_loaded"] != "0"):
            fail("A5 prediction provenance is not internally consistent")
        if row["oof_prediction_artifact_sha256"] != OOF_PREDICTION_ARTIFACT_SHA256:
            fail("A5 prediction provenance has the wrong OOF artifact")
        if (float(row["destination_risk_weight"]) <= 0.0
                or float(row["risk_exposure_penalty"]) <= 0.0
                or float(row["criticality_weight"]) <= 0.0
                or float(row["criticality_migration_weight"]) <= 0.0):
            fail("A5 disabled an R2 prediction or criticality path")
        if (row["uncertainty_gating_enabled"] != "0"
                or int(row["uncertainty_observation_index"]) != UNCERTAINTY_OBSERVATION_INDEX
                or row["uncertainty_observation_geometry_preserved"] != "1"
                or row["uncertainty_mechanism_found"] != UNCERTAINTY_MECHANISM_FOUND
                or row["uncertainty_a5_change"] != UNCERTAINTY_A5_CHANGE
                or row["uncertainty_gating_protocol"] != UNCERTAINTY_GATING_PROTOCOL):
            fail("A5 uncertainty provenance is inconsistent")
    for key in ("node_ticks", "episode_metrics"):
        for row in rows[key]:
            if row["prediction_used"] != "1" or row["uncertainty_used"] != "0":
                fail(f"{key} does not preserve prediction/uncertainty provenance")
            if key == "episode_metrics" and row["risk_source"] != "oof":
                fail("A5 episode metric row has a non-OOF risk source")


def validate_observation_geometry_and_checkpoint() -> None:
    """Construct only; do not step or evaluate the environment."""
    run = build_a5_run(device="cpu")
    observations, _, _ = run.environment.reset(episode_start_tick=run.starts[0], seed=7)
    if observations.shape != (run.environment.n_agents, run.environment.obs_dim):
        fail("A5 observation geometry is invalid")
    if run.policy.agent.obs_dim != run.environment.obs_dim:
        fail("frozen MAPPO does not match A5 observation geometry")
    if run.policy.agent.state_dim != run.environment.state_dim:
        fail("frozen MAPPO does not match A5 centralized-state geometry")
    if not np.array_equal(observations[:, UNCERTAINTY_OBSERVATION_INDEX],
                          np.zeros(run.environment.n_agents)):
        fail("A5 checkpoint-compatible uncertainty slot is not zero")


def validate_row_counts(rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    metrics_by_episode = {int(row["episode"]): row for row in rows["episode_metrics"]}
    metadata_by_episode = {int(row["episode"]): row for row in rows["metadata"]}
    for episode, metrics in metrics_by_episode.items():
        node_count = int(metadata_by_episode[episode]["node_count"])
        task_count = int(metadata_by_episode[episode]["task_count"])
        steps = int(metrics["decision_steps"])
        ordinary_migrations = int(metrics["migrations"])
        reroutes = int(metrics["reroutes"])
        relocations = int(metrics["relocations"])
        if ordinary_migrations != int(metrics["migrations_edge"]) + int(metrics["migrations_cloud"]):
            fail(f"episode {episode} ordinary migrations do not equal edge plus cloud migrations")
        if relocations != ordinary_migrations + reroutes:
            fail(f"episode {episode} relocations do not equal migrations plus reroutes")
        for key, expected in (("node_ticks", steps * 2 * node_count),
                              ("step_rewards", steps * node_count),
                              ("task_metadata", task_count),
                              ("migrations", relocations),
                              ("recovery_events", int(metrics["recovery_attempts"]))):
            actual = sum(int(row["episode"]) == episode for row in rows[key])
            if actual != expected:
                fail(f"{key} row count for episode {episode} is {actual}, expected {expected}")
        if int(metrics["episode_end_tick"]) != int(metrics["episode_start_tick"]) + steps * 2:
            fail(f"episode {episode} end tick does not match its real decision-step duration")
    if len(rows["migrations"]) != sum(int(row["relocations"]) for row in rows["episode_metrics"]):
        fail("migrations CSV row count does not equal total relocations")
    for row in rows["task_events"]:
        if row["event_type"] not in {"COMPLETED", "LOST"} or row["terminal_state"] != row["event_type"]:
            fail("A5 task-event row is not a consistent terminal lifecycle event")
    for row in rows["migrations"]:
        action = int(row["action"])
        if not (0 <= action < len(ACTION_NAMES)) or row["action_name"] != ACTION_NAMES[action]:
            fail("A5 migration row has an invalid action/action-name pair")


def validate_trace_metrics(rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    trace = load_trace(EnvConfig().trace_csv)
    positions = {int(node_id): index for index, node_id in enumerate(trace.node_ids)}
    for row in rows["node_ticks"]:
        tick, node_id = int(row["recorded_tick"]), int(row["node_id"])
        if node_id not in positions or not (0 <= tick < trace.n_ticks - 1):
            fail("A5 node telemetry lies outside the Digital Twin trace")
        node = positions[node_id]
        close(float(row["sim_time_s"]), float(trace.times[tick]), "node telemetry time")
        duration = float(trace.times[tick + 1] - trace.times[tick])
        close(float(row["tick_duration_s"]), duration, "node telemetry duration")
        for field, channel in (("link_latency_ms", "linkLatencyMs"), ("power_w", "energy"),
                               ("cpu_utilization_pct", "cpu"), ("ram_utilization_pct", "ram"),
                               ("bandwidth_utilization_pct", "bandwidth")):
            close(float(row[field]), float(trace.ch(channel)[node, tick]),
                  f"{field} at node={node_id}, tick={tick}")
        close(float(row["energy_interval_j"]), float(row["power_w"]) * duration,
              "node energy interval")


def validate_integrity(output_dir: Path, rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    path = output_dir / INTEGRITY_FILENAME
    if not path.is_file():
        fail(f"missing A5 integrity manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != EXPERIMENT_NAME or payload.get("inference_only") is not True:
        fail("A5 integrity manifest does not identify an inference-only A5 evaluation")
    if (payload.get("held_out_starts") != list(FIXED_HELD_OUT_STARTS)
            or payload.get("consumed_starts") != list(FIXED_HELD_OUT_STARTS)):
        fail("A5 manifest has the wrong ordered held-out start protocol")
    for key, path_value, expected in (
        ("frozen_r2_mappo_checkpoint", VALIDATED_CHECKPOINT, VALIDATED_CHECKPOINT_SHA256),
        ("validated_r2_config", VALIDATED_CONFIG, VALIDATED_CONFIG_SHA256),
        ("oof_prediction_artifact", OOF_PREDICTION_ARTIFACT, OOF_PREDICTION_ARTIFACT_SHA256),
    ):
        if Path(payload.get(key, "")).resolve() != path_value.resolve():
            fail(f"A5 manifest references a wrong {key}")
        if (payload.get(f"{key}_sha256_before") != expected
                or payload.get(f"{key}_sha256_after") != expected
                or sha256(path_value) != expected):
            fail(f"protected A5 input changed: {key}")
    if (payload.get("uncertainty_mechanism_found") != UNCERTAINTY_MECHANISM_FOUND
            or payload.get("uncertainty_a5_change") != UNCERTAINTY_A5_CHANGE):
        fail("A5 manifest does not describe the inspected uncertainty mechanism and change")
    controls = payload.get("a5_configuration", {})
    required = {
        "decision_mechanism": "frozen_greedy_mappo", "mappo_action_selection_used": True,
        "digital_twin_enabled": True, "failure_prediction_enabled": True,
        "risk_source": "oof", "oof_loaded": True, "criticality_awareness_enabled": True,
        "uncertainty_gating_enabled": False,
        "uncertainty_observation_index": UNCERTAINTY_OBSERVATION_INDEX,
        "uncertainty_observation_geometry_preserved": True,
        "uncertainty_mechanism_found": UNCERTAINTY_MECHANISM_FOUND,
        "uncertainty_a5_change": UNCERTAINTY_A5_CHANGE,
        "uncertainty_gating_protocol": UNCERTAINTY_GATING_PROTOCOL,
    }
    if any(controls.get(key) != value for key, value in required.items()):
        fail("A5 manifest does not prove required component controls")
    if (controls.get("mappo_action_selection_calls", 0) <= 0
            or controls.get("destination_risk_weight", 0.0) <= 0.0
            or controls.get("risk_exposure_penalty", 0.0) <= 0.0
            or controls.get("criticality_weight", 0.0) <= 0.0
            or controls.get("criticality_migration_weight", 0.0) <= 0.0):
        fail("A5 manifest records inactive MAPPO/prediction/criticality control")
    for before_key, after_key, current, label in (
        ("protected_sprint8_sprint9_before", "protected_sprint8_sprint9_after",
         protected_sprint8_sprint9_hashes, "Sprint 8/9"),
        ("protected_a2_before", "protected_a2_after", protected_a2_hashes, "A2"),
        ("protected_a3_before", "protected_a3_after", protected_a3_hashes, "A3"),
        ("protected_a4_before", "protected_a4_after", protected_a4_hashes, "A4"),
    ):
        before, after = payload.get(before_key), payload.get(after_key)
        if before != after or current() != after:
            fail(f"{label} evidence changed")
    if payload.get("protected_outputs_unchanged") is not True:
        fail("A5 manifest does not affirm protected-output integrity")
    artifact_hashes = payload.get("artifact_sha256", {})
    for _, (filename, _) in FILE_SPECS.items():
        if artifact_hashes.get(filename) != sha256(output_dir / filename):
            fail(f"A5 artifact hash mismatch: {filename}")
    if artifact_hashes.get(README_FILENAME) != sha256(output_dir / README_FILENAME):
        fail("A5 artifact hash mismatch: README")
    if payload.get("rows") != {key: len(value) for key, value in rows.items()}:
        fail("A5 manifest row counts do not match the actual CSV rows")


def validate_inference_source() -> None:
    """Static guards against training, overwrite, or fabricated uncertainty logic."""
    source = (ROOT / "training" / "sprint10_a5_no_uncertainty.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    forbidden_methods = {"save", "train_mode", "backward"}
    forbidden_uncertainty_names = {"uncertainty_threshold", "uncertainty_gate", "uncertainty_weight"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
        elif isinstance(node, ast.Attribute):
            if node.attr in forbidden_methods:
                fail(f"A5 evaluator invokes prohibited training/checkpoint method {node.attr!r}")
            if (node.attr == "update" and isinstance(node.value, ast.Name)
                    and node.value.id in {"agent", "policy", "run"}):
                fail("A5 evaluator invokes a policy-update path")
            if isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr == "random":
                fail("A5 runner accesses np.random")
        elif isinstance(node, ast.Name) and node.id in forbidden_uncertainty_names:
            fail("A5 runner introduces a fabricated uncertainty heuristic")
    if any(name == "random" or name.startswith("random.") for name in imports):
        fail("A5 runner imports random")
    if "marl.mappo" not in imports:
        fail("A5 runner does not import the MAPPO implementation")
    if "--overwrite" in source:
        fail("A5 runner exposes an overwrite option")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a completed Sprint 10 A5 export")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.out_dir).resolve()
    rows = read_csvs(output_dir)
    validate_fixed_starts(rows)
    validate_controls(rows)
    validate_observation_geometry_and_checkpoint()
    validate_row_counts(rows)
    validate_trace_metrics(rows)
    validate_integrity(output_dir, rows)
    validate_inference_source()
    print("SPRINT 10 A5 validation passed")
    for key, entries in rows.items():
        print(f"  {key:<20s} {len(entries):>8d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
