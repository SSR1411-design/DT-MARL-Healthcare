"""Lightweight provenance checks for CSVs exported by ``sprint8_metrics``.

This verifier does not train a policy and does not load or modify a checkpoint.
It checks already-exported CSV values against their recorded source artifacts.

Example (from python-ai/):
    python training/validate_sprint8_metrics.py --out-dir data
"""

from __future__ import annotations

import argparse
import ast
import csv
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.config import EnvConfig  # noqa: E402
from marl.trace import load_trace  # noqa: E402
from sprint8_metrics import (  # noqa: E402
    FILE_SPECS,
    infer_training_loss_csv,
    load_oof_predictions,
    load_recovery_pairs,
)


def _fail(message: str) -> None:
    raise AssertionError(message)


def _close(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
        _fail(f"{context}: got {actual}, expected {expected}")


def _read_csv(output_dir: Path, key: str) -> list[dict[str, str]]:
    filename, expected_fields = FILE_SPECS[key]
    path = output_dir / filename
    if not path.exists():
        _fail(f"missing Sprint 8 CSV: {path}")
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            _fail(f"{path.name} has unexpected header: {reader.fieldnames}")
        rows = list(reader)
    if not rows:
        _fail(f"{path.name} has a header but no real rows")
    return rows


def _validate_node_ticks(rows: Sequence[Mapping[str, str]], cfg: EnvConfig) -> None:
    trace = load_trace(cfg.trace_csv)
    node_positions = {int(node_id): position
                      for position, node_id in enumerate(trace.node_ids)}
    for row in rows:
        tick = int(row["recorded_tick"])
        node_id = int(row["node_id"])
        if not (0 <= tick < trace.n_ticks - 1):
            _fail(f"node tick outside source trace: tick={tick}")
        if node_id not in node_positions:
            _fail(f"node {node_id} is absent from source trace")
        node = node_positions[node_id]
        _close(float(row["sim_time_s"]), float(trace.times[tick]), "node tick time")
        duration = float(trace.times[tick + 1] - trace.times[tick])
        _close(float(row["tick_duration_s"]), duration, "node tick duration")
        for column, channel in (
            ("link_latency_ms", "linkLatencyMs"),
            ("power_w", "energy"),
            ("cpu_utilization_pct", "cpu"),
            ("ram_utilization_pct", "ram"),
            ("bandwidth_utilization_pct", "bandwidth"),
            ("running_tasks_telemetry", "runningTasks"),
            ("link_bandwidth_mbps", "linkBandwidthMbps"),
            ("link_packet_loss_pct", "linkPacketLoss"),
        ):
            _close(float(row[column]), float(trace.ch(channel)[node, tick]),
                   f"{column} at node={node_id}, tick={tick}")
        expected_energy_j = float(trace.ch("energy")[node, tick]) * duration
        _close(float(row["energy_interval_j"]), expected_energy_j,
               f"energy_interval_j at node={node_id}, tick={tick}")
        for column, channel in (("active", "active"), ("degraded", "degraded"),
                                ("link_up", "linkUp"), ("under_attack", "underAttack")):
            expected = int(trace.ch(channel)[node, tick] >= 0.5)
            if int(row[column]) != expected:
                _fail(f"{column} at node={node_id}, tick={tick} does not match trace")


def _validate_recovery(rows: Sequence[Mapping[str, str]], cfg: EnvConfig) -> None:
    trace = load_trace(cfg.trace_csv)
    expected = load_recovery_pairs(cfg.event_log_csv, trace.times)
    if len(rows) != len(expected):
        _fail(f"recovery row count {len(rows)} != {len(expected)} source pairs")
    for row, pair in zip(rows, expected):
        if int(row["node_id"]) != pair.node_id:
            _fail("recovery node order does not match failure_log.csv")
        for column, value in (
            ("failure_time_s", pair.failure_time_s),
            ("recovery_time_s", pair.recovery_time_s),
            ("recovery_duration_s", pair.duration_s),
        ):
            _close(float(row[column]), value, f"recovery {column}")
        if int(row["failure_tick"]) != pair.failure_tick:
            _fail("recovery failure tick does not match source log")
        if int(row["recovery_tick"]) != pair.recovery_tick:
            _fail("recovery tick does not match source log")


def _validate_predictions(window_rows: Sequence[Mapping[str, str]],
                          metric_rows: Sequence[Mapping[str, str]],
                          cfg: EnvConfig) -> None:
    expected_rows, expected_metrics = load_oof_predictions(cfg)
    if len(window_rows) != len(expected_rows):
        _fail("prediction-window count does not match the leakage-safe OOF artifact")
    for actual, expected in zip(window_rows, expected_rows):
        for field in ("node_id", "recorded_tick", "oof_predicted_label",
                      "will_fail_soon_label", "prediction_correct", "event_id"):
            if int(actual[field]) != int(expected[field]):
                _fail(f"OOF prediction field {field} does not match its source artifact")
        for field in ("window_end_time_s", "predicted_failure_risk"):
            _close(float(actual[field]), float(expected[field]), f"OOF {field}")
        if actual["prediction_source"] != "leakage_safe_out_of_fold":
            _fail("prediction CSV is missing OOF provenance")
        if actual["policy_observable"] != "offline_evaluation_only":
            _fail("offline OOF labels were incorrectly marked policy-observable")

    actual_metrics = {row["metric"]: float(row["value"]) for row in metric_rows}
    if set(actual_metrics) != set(expected_metrics):
        _fail("prediction metric names do not match host_cv.full_metrics")
    for metric, expected in expected_metrics.items():
        _close(actual_metrics[metric], float(expected), f"OOF metric {metric}")


def _validate_migrations(migration_rows: Sequence[Mapping[str, str]],
                         episode_rows: Sequence[Mapping[str, str]]) -> None:
    by_episode = Counter(int(row["episode"]) for row in migration_rows)
    for row in episode_rows:
        episode = int(row["episode"])
        relocations = int(float(row["relocations"]))
        if by_episode[episode] != relocations:
            _fail(
                f"episode {episode}: {by_episode[episode]} MigrationRecord rows "
                f"!= environment relocations={relocations}")
    unknown_episodes = set(by_episode) - {int(row["episode"]) for row in episode_rows}
    if unknown_episodes:
        _fail(f"migration rows refer to unknown episodes: {sorted(unknown_episodes)}")


def _validate_training_loss(rows: Sequence[Mapping[str, str]], model: Path,
                            supplied: str | None) -> None:
    source = infer_training_loss_csv(model, supplied).resolve()
    if any(Path(row["source_csv"]).resolve() != source for row in rows):
        _fail("training-loss rows do not retain their real update-log provenance")
    with open(source, newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    if len(rows) != len(source_rows):
        _fail("training-loss row count differs from the existing update log")
    for actual, expected in zip(rows, source_rows):
        for field in FILE_SPECS["training_loss"][1][1:]:
            if actual[field] != expected[field]:
                _fail(f"training loss field {field} was altered during export")


def _validate_no_synthetic_path() -> None:
    exporter = Path(__file__).with_name("sprint8_metrics.py").read_text(encoding="utf-8")
    tree = ast.parse(exporter)
    random_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            random_names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            random_names.append(node.module)
        elif isinstance(node, ast.Name) and node.id == "random":
            random_names.append("random")
    if any(name == "random" or name.startswith("random.") for name in random_names):
        _fail("Sprint 8 exporter still contains a random metric-generation path")
    if "sprint8_metrics.csv" in {filename for filename, _ in FILE_SPECS.values()}:
        _fail("legacy synthetic sprint8_metrics.csv was reintroduced as research output")
    legacy_name = "LEGACY_SYNTHETIC_INVALID_NOT_FOR_RESEARCH_sprint8_metrics.csv"
    if (ROOT / "data" / "sprint8_metrics.csv").exists():
        _fail("legacy synthetic CSV must be renamed outside the research output path")
    if not (ROOT / "data" / legacy_name).exists():
        _fail("renamed legacy synthetic CSV is missing; preserve it for audit provenance")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate real Sprint 8 CSV provenance")
    parser.add_argument("--out-dir", default=str(ROOT / "data"))
    parser.add_argument("--model", default=str(ROOT / "saved_models" / "marl" /
                                                "mappo_R2_mc_target_best.pth"))
    parser.add_argument("--training-loss-csv", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.out_dir).resolve()
    cfg = EnvConfig()
    rows = {key: _read_csv(output_dir, key) for key in FILE_SPECS}
    _validate_node_ticks(rows["node_ticks"], cfg)
    _validate_recovery(rows["recovery"], cfg)
    _validate_predictions(rows["prediction_windows"], rows["prediction_metrics"], cfg)
    _validate_migrations(rows["migrations"], rows["episode_metrics"])
    _validate_training_loss(rows["training_loss"], Path(args.model).resolve(),
                            args.training_loss_csv)
    _validate_no_synthetic_path()
    print("SPRINT 8 CSV validation passed")
    for key, exported in rows.items():
        print(f"  {FILE_SPECS[key][0]:<38s} {len(exported):>8d} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
