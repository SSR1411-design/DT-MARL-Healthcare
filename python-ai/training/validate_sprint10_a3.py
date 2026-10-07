"""Read-only integrity validation for a completed Sprint 10 A3 export.

This validator never trains, evaluates, repairs, or writes an artifact.  It
checks that the result bundle came from the independent single-agent PPO while
all R2 environment controls, held-out starts, and protected research outputs
remain intact.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.config import ACTION_NAMES, EnvConfig
from marl.trace import load_trace
from training.sprint10_a3_single_agent import (
    DEFAULT_OUTPUT_DIR,
    EPISODE_FIELDS,
    EXPERIMENT_NAME,
    FILE_SPECS,
    FIXED_HELD_OUT_STARTS,
    INTEGRITY_FILENAME,
    OOF_PREDICTION_ARTIFACT,
    OOF_PREDICTION_ARTIFACT_SHA256,
    README_FILENAME,
    SINGLE_AGENT_CHECKPOINT,
    SINGLE_AGENT_CHECKPOINT_SHA256,
    SINGLE_AGENT_CONFIG,
    SINGLE_AGENT_CONFIG_SHA256,
    UNCERTAINTY_GATING_PROTOCOL,
    VALIDATED_R2_CHECKPOINT,
    VALIDATED_R2_CHECKPOINT_SHA256,
    VALIDATED_R2_CONFIG,
    VALIDATED_R2_CONFIG_SHA256,
    protected_a2_hashes,
    protected_output_hashes,
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
            fail(f"missing A3 CSV: {path}")
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fields:
                fail(f"{path.name} has an unexpected header")
            rows[key] = list(reader)
        if key in required_nonempty and not rows[key]:
            fail(f"{path.name} has no actual A3 evaluation rows")
    if FILE_SPECS["episode_metrics"][1] != EPISODE_FIELDS:
        fail("A3 criticality episode schema was unexpectedly replaced")
    readme = output_dir / README_FILENAME
    if not readme.is_file() or not readme.read_text(encoding="utf-8").strip():
        fail("missing or empty A3 README")
    return rows


def _starts(entries: Sequence[Mapping[str, str]], key: str) -> list[int]:
    return [int(row["episode_start_tick"]) for row in entries]


def validate_fixed_starts(rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    expected = list(FIXED_HELD_OUT_STARTS)
    metrics = rows["episode_metrics"]
    if len(metrics) != len(expected):
        fail(f"A3 must have exactly {len(expected)} episodes, found {len(metrics)}")
    if _starts(metrics, "episode_metrics") != expected:
        fail("A3 episode metrics do not use the validated R2 held-out starts")
    if [int(row["episode"]) for row in metrics] != list(range(1, len(expected) + 1)):
        fail("A3 episode numbering is not the fixed eight-episode protocol")
    for key in ("metadata", "prediction_usage"):
        if len(rows[key]) != len(expected) or _starts(rows[key], key) != expected:
            fail(f"{key} does not prove ordered consumption of all eight starts")
    valid = {(index + 1, start) for index, start in enumerate(expected)}
    for key, entries in rows.items():
        for row in entries:
            if (int(row["episode"]), int(row["episode_start_tick"])) not in valid:
                fail(f"{key} contains a row outside the fixed A3 episode/start protocol")


def validate_single_agent_and_controls(rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    for row in rows["metadata"]:
        if row["baseline_name"] != EXPERIMENT_NAME or row["ablation_id"] != "A3":
            fail("A3 metadata identifies the wrong experiment")
        if row["decision_mechanism"] != "single_agent_ppo":
            fail("A3 metadata does not identify the single-agent PPO controller")
        if row["single_agent_policy_used"] != "1" or row["mappo_action_selection_used"] != "0":
            fail("A3 metadata does not prove MAPPO action selection was removed")
        if row["uses_marl"] != "False":
            fail("A3 metadata claims multi-agent coordination remains enabled")
        if row["uses_digital_twin"] != "True" or row["uses_digital_twin_trace"] != "1":
            fail("A3 metadata does not preserve the Digital Twin trace")
        if row["uses_prediction"] != "True" or row["risk_source"] != "oof":
            fail("A3 metadata does not preserve the R2 OOF failure prediction")
        if row["criticality_awareness_enabled"] != "1":
            fail("A3 metadata does not preserve criticality awareness")
        if (row["uncertainty_component_preserved"] != "1"
                or row["uncertainty_gating_protocol"] != UNCERTAINTY_GATING_PROTOCOL):
            fail("A3 metadata does not preserve the validated R2 uncertainty contract")
        if Path(row["checkpoint"]).resolve() != SINGLE_AGENT_CHECKPOINT.resolve():
            fail("A3 metadata references a nonvalidated single-agent PPO checkpoint")
        if row["single_agent_checkpoint_sha256"] != SINGLE_AGENT_CHECKPOINT_SHA256:
            fail("A3 metadata has the wrong single-agent checkpoint hash")
        if row["validated_r2_config_sha256"] != VALIDATED_R2_CONFIG_SHA256:
            fail("A3 metadata has the wrong R2 configuration hash")
        if row["single_agent_config_sha256"] != SINGLE_AGENT_CONFIG_SHA256:
            fail("A3 metadata has the wrong single-agent configuration hash")
        if row["oof_prediction_artifact_sha256"] != OOF_PREDICTION_ARTIFACT_SHA256:
            fail("A3 metadata has the wrong OOF prediction artifact hash")
    for row in rows["prediction_usage"]:
        if (row["risk_source"] != "oof" or row["prediction_used"] != "1"
                or row["uncertainty_used"] != "0" or row["oof_loaded"] != "1"
                or row["live_predictor_loaded"] != "0"):
            fail("A3 prediction provenance is not internally consistent")
        if row["oof_prediction_artifact_sha256"] != OOF_PREDICTION_ARTIFACT_SHA256:
            fail("A3 prediction provenance has the wrong OOF artifact")
        if float(row["destination_risk_weight"]) <= 0.0 or float(row["risk_exposure_penalty"]) <= 0.0:
            fail("A3 disabled an R2 prediction-dependent decision/reward path")
        if (float(row["criticality_weight"]) <= 0.0
                or float(row["criticality_migration_weight"]) <= 0.0):
            fail("A3 disabled R2 criticality weighting")
        if (row["uncertainty_component_preserved"] != "1"
                or row["uncertainty_gating_protocol"] != UNCERTAINTY_GATING_PROTOCOL):
            fail("A3 uncertainty provenance is inconsistent")
    for key in ("node_ticks", "episode_metrics"):
        for row in rows[key]:
            # Node telemetry has no risk_source column; both tables still must document use.
            if row["prediction_used"] != "1" or row["uncertainty_used"] != "0":
                fail(f"{key} does not preserve prediction/uncertainty provenance")
            if key == "episode_metrics" and row["risk_source"] != "oof":
                fail("A3 episode metric row has a non-OOF risk source")


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
        if ordinary_migrations != (
                int(metrics["migrations_edge"]) + int(metrics["migrations_cloud"])):
            fail(f"episode {episode} ordinary migrations do not equal edge plus cloud migrations")
        if relocations != ordinary_migrations + reroutes:
            fail(
                f"episode {episode} relocations {relocations} do not equal ordinary "
                f"migrations {ordinary_migrations} plus reroutes {reroutes}")
        expected_node_ticks = steps * 2 * node_count  # R2 ticks_per_step is protected at 2.
        expected_step_rewards = steps * node_count
        for key, expected in (("node_ticks", expected_node_ticks),
                              ("step_rewards", expected_step_rewards),
                              ("task_metadata", task_count),
                              # This CSV exports every env.migrations record: ordinary
                              # migrations plus PREEMPTIVE_REROUTE relocation records.
                              ("migrations", relocations),
                              ("recovery_events", int(metrics["recovery_attempts"]))):
            actual = sum(int(row["episode"]) == episode for row in rows[key])
            if actual != expected:
                fail(f"{key} row count for episode {episode} is {actual}, expected {expected}")
        end_tick = int(metrics["episode_start_tick"]) + steps * 2
        if int(metrics["episode_end_tick"]) != end_tick:
            fail(f"episode {episode} end tick does not match its real decision-step duration")
    expected_total_relocations = sum(
        int(metrics["relocations"]) for metrics in rows["episode_metrics"])
    if len(rows["migrations"]) != expected_total_relocations:
        fail(
            f"migrations CSV has {len(rows['migrations'])} rows, expected total "
            f"relocations {expected_total_relocations}")
    for row in rows["task_events"]:
        if row["event_type"] not in {"COMPLETED", "LOST"}:
            fail("A3 task-event row is not a real terminal lifecycle event")
        if row["terminal_state"] != row["event_type"]:
            fail("A3 task-event terminal state is inconsistent")
    for row in rows["migrations"]:
        action = int(row["action"])
        if not (0 <= action < len(ACTION_NAMES)) or row["action_name"] != ACTION_NAMES[action]:
            fail("A3 migration row has an invalid action/action-name pair")


def validate_trace_metrics(rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    trace = load_trace(EnvConfig().trace_csv)
    positions = {int(node_id): index for index, node_id in enumerate(trace.node_ids)}
    for row in rows["node_ticks"]:
        tick, node_id = int(row["recorded_tick"]), int(row["node_id"])
        if node_id not in positions or not (0 <= tick < trace.n_ticks - 1):
            fail("A3 node telemetry lies outside the Digital Twin trace")
        node = positions[node_id]
        close(float(row["sim_time_s"]), float(trace.times[tick]), "node telemetry time")
        duration = float(trace.times[tick + 1] - trace.times[tick])
        close(float(row["tick_duration_s"]), duration, "node telemetry duration")
        for field, channel in (
            ("link_latency_ms", "linkLatencyMs"), ("power_w", "energy"),
            ("cpu_utilization_pct", "cpu"), ("ram_utilization_pct", "ram"),
            ("bandwidth_utilization_pct", "bandwidth"),
        ):
            close(float(row[field]), float(trace.ch(channel)[node, tick]),
                  f"{field} at node={node_id}, tick={tick}")
        close(float(row["energy_interval_j"]), float(row["power_w"]) * duration,
              "node energy interval")


def validate_integrity(output_dir: Path, rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    path = output_dir / INTEGRITY_FILENAME
    if not path.is_file():
        fail(f"missing A3 integrity manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != EXPERIMENT_NAME or payload.get("inference_only") is not True:
        fail("A3 integrity manifest does not identify an inference-only A3 evaluation")
    if payload.get("held_out_starts") != list(FIXED_HELD_OUT_STARTS):
        fail("A3 manifest has the wrong held-out start protocol")
    if payload.get("consumed_starts") != list(FIXED_HELD_OUT_STARTS):
        fail("A3 manifest does not prove ordered start consumption")
    if Path(payload.get("single_agent_checkpoint", "")).resolve() != SINGLE_AGENT_CHECKPOINT.resolve():
        fail("A3 manifest references a wrong single_agent_checkpoint")
    if (payload.get("single_agent_checkpoint_sha256_before") != SINGLE_AGENT_CHECKPOINT_SHA256
            or payload.get("single_agent_checkpoint_sha256_after") != SINGLE_AGENT_CHECKPOINT_SHA256
            or sha256(SINGLE_AGENT_CHECKPOINT) != SINGLE_AGENT_CHECKPOINT_SHA256):
        fail("protected single-agent PPO checkpoint changed")
    for key, path_value, expected in (
        ("single_agent_config", SINGLE_AGENT_CONFIG, SINGLE_AGENT_CONFIG_SHA256),
        ("validated_r2_config", VALIDATED_R2_CONFIG, VALIDATED_R2_CONFIG_SHA256),
        ("oof_prediction_artifact", OOF_PREDICTION_ARTIFACT, OOF_PREDICTION_ARTIFACT_SHA256),
    ):
        if Path(payload.get(key, "")).resolve() != path_value.resolve():
            fail(f"A3 manifest references a wrong {key}")
        digest_key = f"{key}_sha256"
        if payload.get(digest_key) != expected:
            fail(f"A3 manifest has the wrong {digest_key}")
        if sha256(path_value) != expected:
            fail(f"protected input changed after the A3 evaluation: {key}")
    if Path(payload.get("frozen_r2_mappo_checkpoint", "")).resolve() != VALIDATED_R2_CHECKPOINT.resolve():
        fail("A3 manifest references the wrong frozen R2 MAPPO checkpoint")
    if (payload.get("frozen_r2_mappo_checkpoint_sha256_before") != VALIDATED_R2_CHECKPOINT_SHA256
            or payload.get("frozen_r2_mappo_checkpoint_sha256_after") != VALIDATED_R2_CHECKPOINT_SHA256
            or sha256(VALIDATED_R2_CHECKPOINT) != VALIDATED_R2_CHECKPOINT_SHA256):
        fail("frozen R2 MAPPO checkpoint changed")
    controls = payload.get("a3_configuration", {})
    required_controls = {
        "decision_mechanism": "single_agent_ppo",
        "single_agent_policy_used": True,
        "mappo_action_selection_used": False,
        "digital_twin_enabled": True,
        "failure_prediction_enabled": True,
        "risk_source": "oof",
        "oof_loaded": True,
        "criticality_awareness_enabled": True,
        "uncertainty_component_preserved": True,
        "uncertainty_gating_protocol": UNCERTAINTY_GATING_PROTOCOL,
    }
    if any(controls.get(key) != value for key, value in required_controls.items()):
        fail("A3 manifest does not prove the required controller/component controls")
    if (controls.get("single_agent_action_selection_calls", 0) <= 0
            or controls.get("destination_risk_weight", 0.0) <= 0.0
            or controls.get("risk_exposure_penalty", 0.0) <= 0.0
            or controls.get("criticality_weight", 0.0) <= 0.0
            or controls.get("criticality_migration_weight", 0.0) <= 0.0):
        fail("A3 manifest records disabled action, prediction, or criticality control")
    before, after = (payload.get("protected_sprint8_sprint9_before"),
                     payload.get("protected_sprint8_sprint9_after"))
    if before != after or payload.get("protected_outputs_unchanged") is not True:
        fail("Sprint 8/9 outputs changed during A3 evaluation")
    if protected_output_hashes() != after:
        fail("Sprint 8/9 outputs changed after A3 integrity recording")
    a2_before, a2_after = payload.get("protected_a2_before"), payload.get("protected_a2_after")
    if a2_before != a2_after or protected_a2_hashes() != a2_after:
        fail("protected completed A2 outputs changed")
    artifact_hashes = payload.get("artifact_sha256", {})
    expected_paths = [*FILE_SPECS.values()]
    del expected_paths  # Names are resolved from the actual bundle below.
    for _, (filename, _) in FILE_SPECS.items():
        if artifact_hashes.get(filename) != sha256(output_dir / filename):
            fail(f"A3 artifact hash mismatch: {filename}")
    if artifact_hashes.get(README_FILENAME) != sha256(output_dir / README_FILENAME):
        fail("A3 artifact hash mismatch: README")
    if payload.get("rows") != {key: len(value) for key, value in rows.items()}:
        fail("A3 manifest row counts do not match the actual CSV rows")


def validate_inference_source() -> None:
    """Static guard: the A3 runner has no random metric or MAPPO action path."""
    source = (ROOT / "training" / "sprint10_a3_single_agent.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden_methods = {"save", "train_mode", "backward"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "random" or alias.name.startswith("marl.mappo") for alias in node.names):
                fail("A3 runner imports random or the MAPPO implementation")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "random" or (node.module and node.module.startswith("marl.mappo")):
                fail("A3 runner imports random or the MAPPO implementation")
        elif isinstance(node, ast.Attribute):
            if node.attr in forbidden_methods:
                fail(f"A3 evaluator invokes prohibited training/checkpoint method {node.attr!r}")
            if (node.attr == "update" and isinstance(node.value, ast.Name)
                    and node.value.id in {"agent", "policy", "run"}):
                fail("A3 evaluator invokes a policy-update path")
            if isinstance(node.value, ast.Name) and node.value.id == "np" and node.attr == "random":
                fail("A3 runner accesses np.random")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a completed Sprint 10 A3 export")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.out_dir).resolve()
    rows = read_csvs(output_dir)
    validate_fixed_starts(rows)
    validate_single_agent_and_controls(rows)
    validate_row_counts(rows)
    validate_trace_metrics(rows)
    validate_integrity(output_dir, rows)
    validate_inference_source()
    print("SPRINT 10 A3 validation passed")
    for key, entries in rows.items():
        print(f"  {key:<20s} {len(entries):>8d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
