"""Read-only provenance validation for Sprint 10 A2 result files.

This validator does not train, evaluate, or write files.  Run it only after
``sprint10_a2_no_failure_prediction.py`` has completed an A2 export.
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

from marl.config import EnvConfig
from marl.trace import load_trace
from training.sprint10_a2_no_failure_prediction import (
    DEFAULT_OUTPUT_DIR,
    EXPERIMENT_NAME,
    FILE_SPECS,
    FIXED_HELD_OUT_STARTS,
    INTEGRITY_FILENAME,
    VALIDATED_CHECKPOINT,
    VALIDATED_CHECKPOINT_SHA256,
    protected_output_hashes,
    sha256,
)


def fail(message: str) -> None:
    raise AssertionError(message)


def close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
        fail(f"{context}: got {actual}, expected {expected}")


def read_csvs(output_dir: Path) -> dict[str, list[dict[str, str]]]:
    rows = {}
    required_nonempty = {"metadata", "node_ticks", "step_rewards", "task_metadata",
                         "prediction_usage", "episode_metrics"}
    for key, (filename, fields) in FILE_SPECS.items():
        path = output_dir / filename
        if not path.exists():
            fail(f"missing A2 CSV: {path}")
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != fields:
                fail(f"{path.name} has unexpected header")
            rows[key] = list(reader)
        if key in required_nonempty and not rows[key]:
            fail(f"{path.name} has no actual A2 rows")
    return rows


def validate_no_prediction(rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    for key in ("metadata", "prediction_usage", "episode_metrics", "node_ticks"):
        for row in rows[key]:
            if row.get("risk_source", "zero") != "zero":
                fail(f"{key} records a nonzero risk source")
            if row["prediction_used"] != "0" or row["uncertainty_used"] != "0":
                fail(f"{key} records prediction or uncertainty use")
    for row in rows["metadata"]:
        if row["oof_loaded"] != "0" or row["live_predictor_loaded"] != "0":
            fail("A2 metadata records a loaded predictor")
    for row in rows["prediction_usage"]:
        if (row["oof_loaded"] != "0" or row["live_predictor_loaded"] != "0"
                or row["policy_risk_channels_zero"] != "1"):
            fail("A2 prediction provenance does not prove zero policy risk channels")
        close(float(row["destination_risk_weight"]), 0.0, "destination risk weight")
        close(float(row["risk_exposure_penalty"]), 0.0, "risk exposure penalty")
    for row in rows["migrations"]:
        close(float(row["source_risk"]), 0.0, "migration source risk")
        close(float(row["destination_risk"]), 0.0, "migration destination risk")


def validate_fixed_starts(rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    expected = list(FIXED_HELD_OUT_STARTS)
    actual = [int(row["episode_start_tick"]) for row in rows["episode_metrics"]]
    if actual != expected:
        fail(f"A2 starts {actual} do not equal validated R2 starts {expected}")
    episodes = [int(row["episode"]) for row in rows["episode_metrics"]]
    if episodes != list(range(1, len(expected) + 1)):
        fail("A2 episode numbering is not the fixed eight-episode protocol")
    for key in ("metadata", "prediction_usage"):
        if [int(row["episode_start_tick"]) for row in rows[key]] != expected:
            fail(f"{key} does not use the fixed held-out starts")


def validate_trace_metrics(rows: Mapping[str, Sequence[Mapping[str, str]]]) -> None:
    trace = load_trace(EnvConfig().trace_csv)
    node_positions = {int(node_id): position for position, node_id in enumerate(trace.node_ids)}
    for row in rows["node_ticks"]:
        tick, node_id = int(row["recorded_tick"]), int(row["node_id"])
        if node_id not in node_positions or not (0 <= tick < trace.n_ticks - 1):
            fail("node telemetry row is outside the recorded Digital Twin trace")
        node = node_positions[node_id]
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


def validate_integrity(output_dir: Path) -> None:
    path = output_dir / INTEGRITY_FILENAME
    if not path.exists():
        fail(f"missing A2 integrity manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("experiment") != EXPERIMENT_NAME:
        fail("integrity manifest has the wrong experiment name")
    if Path(payload.get("checkpoint", "")).resolve() != VALIDATED_CHECKPOINT.resolve():
        fail("integrity manifest references a nonvalidated MAPPO checkpoint")
    for field in ("checkpoint_sha256_before", "checkpoint_sha256_after",
                  "validated_checkpoint_sha256"):
        if payload.get(field) != VALIDATED_CHECKPOINT_SHA256:
            fail(f"integrity manifest has an invalid {field}")
    if sha256(VALIDATED_CHECKPOINT) != VALIDATED_CHECKPOINT_SHA256:
        fail("validated MAPPO checkpoint content changed")
    if payload.get("held_out_starts") != list(FIXED_HELD_OUT_STARTS):
        fail("integrity manifest has the wrong held-out start protocol")
    if payload.get("consumed_starts") != list(FIXED_HELD_OUT_STARTS):
        fail("integrity manifest does not prove the environment consumed each held-out start")
    a2 = payload.get("a2_configuration", {})
    if (a2.get("risk_source") != "zero" or a2.get("destination_risk_weight") != 0.0
            or a2.get("risk_exposure_penalty") != 0.0 or a2.get("prediction_used") is not False
            or a2.get("oof_loaded") is not False or a2.get("live_predictor_loaded") is not False):
        fail("integrity manifest does not prove A2 risk removal")
    before = payload.get("protected_sprint8_sprint9_before")
    after = payload.get("protected_sprint8_sprint9_after")
    if before != after or payload.get("protected_outputs_unchanged") is not True:
        fail("Sprint 8/9 output hashes changed during A2 evaluation")
    if protected_output_hashes() != after:
        fail("Sprint 8/9 outputs changed after the A2 integrity manifest was written")


def validate_no_random_metric_generation() -> None:
    source = (ROOT / "training" / "sprint10_a2_no_failure_prediction.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(alias.name == "random" for alias in node.names):
            fail("A2 runner imports random")
        if isinstance(node, ast.ImportFrom) and node.module == "random":
            fail("A2 runner imports from random")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "np" and node.attr == "random":
                fail("A2 runner accesses np.random")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an existing Sprint 10 A2 export")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.out_dir).resolve()
    rows = read_csvs(output_dir)
    validate_no_prediction(rows)
    validate_fixed_starts(rows)
    validate_trace_metrics(rows)
    validate_integrity(output_dir)
    validate_no_random_metric_generation()
    print("SPRINT 10 A2 validation passed")
    for key, (_, _) in FILE_SPECS.items():
        print(f"  {key:<20s} {len(rows[key]):>8d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
