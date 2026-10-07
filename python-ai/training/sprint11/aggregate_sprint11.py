"""Aggregate completed Sprint 11 trial summaries without pseudoreplication.

One row in ``trial_result.json`` is one scenario × independently seeded
simulator trial × policy summary.  Multiple evaluation windows inside it are
not promoted to independent samples.  Confidence intervals are therefore over
independent simulator trials only.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sprint11.schema import (REQUIRED_RESULT_METRICS, SchemaError,
                                      load_json, validate_result_record)


DEFAULT_TRIAL_ROOT = ROOT / "data" / "sprint11" / "trials"
DEFAULT_AGGREGATE_ROOT = ROOT / "data" / "sprint11" / "aggregated"

T975_BY_DF = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
    7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
    13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
    19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
    25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

AGGREGATE_FIELDS = [
    "scenario_id", "policy", "policy_kind", "host_count", "edge_node_count",
    "device_count", "task_count", "patient_count", "bandwidth_mbps",
    "failure_intensity", "metric", "mean", "stddev", "sample_count",
    "standard_error", "ci95_low", "ci95_high", "ci95_status", "trial_ids",
]

PRIORITY_FIELDS = [
    "scenario_id", "policy", "policy_kind", "priority_bin", "mean_response_time_s",
    "stddev", "sample_count", "standard_error", "ci95_low", "ci95_high",
    "ci95_status", "trial_ids",
]

OBSERVATION_FIELDS = [
    "scenario_id", "policy", "policy_kind", "trial_id", "metric", "value", "sample_unit",
]


def t_critical_95(sample_count: int) -> float:
    if sample_count < 2:
        raise SchemaError("a 95% confidence interval needs at least two independent trials")
    degrees = sample_count - 1
    return T975_BY_DF.get(degrees, 1.96)


def summary(values: Iterable[float]) -> dict[str, float | int | str | None]:
    values = [float(v) for v in values]
    if not values:
        raise SchemaError("cannot aggregate an empty metric group")
    n = len(values)
    avg = mean(values)
    if n < 2:
        return {"mean": avg, "stddev": None, "sample_count": n,
                "standard_error": None, "ci95_low": None, "ci95_high": None,
                "ci95_status": "not_available_n_lt_2"}
    sd = stdev(values)
    se = sd / math.sqrt(n)
    half_width = t_critical_95(n) * se
    return {"mean": avg, "stddev": sd, "sample_count": n,
            "standard_error": se, "ci95_low": avg - half_width,
            "ci95_high": avg + half_width, "ci95_status": "t_interval_95"}


def load_results(trial_root: Path, *, require_files: bool = False) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for path in sorted(trial_root.rglob("trial_result.json")):
        record = load_json(path)
        if not isinstance(record, dict):
            raise SchemaError(f"result must be an object: {path}")
        validate_result_record(record, require_files=require_files)
        key = (str(record["scenario_id"]), int(record["trial_id"]), str(record["policy"]))
        if key in seen:
            raise SchemaError(f"duplicate independent trial result: {key}")
        seen.add(key)
        record["_source_path"] = str(path)
        records.append(record)
    return records


def aggregate_records(records: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metric_groups: dict[tuple[Any, ...], list[tuple[int, float]]] = defaultdict(list)
    priority_groups: dict[tuple[str, str, str, str], list[tuple[int, float]]] = defaultdict(list)
    observations: list[dict[str, Any]] = []
    for record in records:
        shared = tuple(record[key] for key in (
            "scenario_id", "policy", "policy_kind", "host_count", "edge_node_count",
            "device_count", "task_count", "patient_count", "bandwidth_mbps", "failure_intensity"))
        for metric in REQUIRED_RESULT_METRICS:
            metric_groups[shared + (metric,)].append((int(record["trial_id"]), float(record["metrics"][metric])))
        for item in record.get("priority_response_time_s", []):
            if not isinstance(item, Mapping):
                raise SchemaError("priority_response_time_s entries must be objects")
            priority = str(item["priority_bin"])
            value = float(item["mean_response_time_s"])
            if not math.isfinite(value) or value < 0:
                raise SchemaError("priority response time must be finite and non-negative")
            priority_groups[(str(record["scenario_id"]), str(record["policy"]),
                             str(record["policy_kind"]), priority)].append((int(record["trial_id"]), value))
        for item in record.get("distribution_observations", []):
            if not isinstance(item, Mapping):
                raise SchemaError("distribution_observations entries must be objects")
            metric = str(item.get("metric", ""))
            value = item.get("value")
            if metric not in {"avg_task_latency_s", "physical_energy_j", "HSI", "episode_reward"}:
                raise SchemaError("unsupported histogram metric")
            value = float(value)
            if not math.isfinite(value):
                raise SchemaError("histogram observation must be finite")
            observations.append({
                "scenario_id": record["scenario_id"], "policy": record["policy"],
                "policy_kind": record["policy_kind"], "trial_id": record["trial_id"],
                "metric": metric, "value": value,
                "sample_unit": str(item.get("sample_unit", "unspecified")),
            })
    aggregate_rows: list[dict[str, Any]] = []
    for key, trials_and_values in sorted(metric_groups.items()):
        *shared, metric = key
        trial_ids, values = zip(*trials_and_values)
        aggregate_rows.append({
            **dict(zip(AGGREGATE_FIELDS[:10], shared)), "metric": metric,
            **summary(values), "trial_ids": ";".join(map(str, sorted(trial_ids))),
        })
    priority_rows: list[dict[str, Any]] = []
    for key, trials_and_values in sorted(priority_groups.items()):
        trial_ids, values = zip(*trials_and_values)
        statistics = summary(values)
        priority_rows.append({
            **dict(zip(PRIORITY_FIELDS[:4], key)),
            "mean_response_time_s": statistics.pop("mean"),
            **statistics,
            "trial_ids": ";".join(map(str, sorted(trial_ids))),
        })
    return aggregate_rows, priority_rows, observations


def _write_csv_new(path: Path, fields: list[str], rows: Iterable[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite aggregate: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_aggregates(output_dir: Path, aggregate_rows: list[dict[str, Any]],
                     priority_rows: list[dict[str, Any]], observations: list[dict[str, Any]]) -> None:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite aggregate directory: {output_dir}")
    output_dir.mkdir(parents=True)
    try:
        _write_csv_new(output_dir / "trial_metric_aggregates.csv", AGGREGATE_FIELDS, aggregate_rows)
        _write_csv_new(output_dir / "priority_response_aggregates.csv", PRIORITY_FIELDS, priority_rows)
        _write_csv_new(output_dir / "distribution_observations.csv", OBSERVATION_FIELDS, observations)
    except Exception:
        # Directory existed only for this invocation and has no historical data.
        for path in output_dir.glob("*"):
            path.unlink()
        output_dir.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-root", type=Path, default=DEFAULT_TRIAL_ROOT)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="new directory, e.g. data/sprint11/aggregated/run-001")
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args()
    records = load_results(args.trial_root, require_files=args.verify_files)
    if not records:
        parser.error("no completed trial_result.json files found; no aggregate was written")
    aggregates, priorities, observations = aggregate_records(records)
    write_aggregates(args.output_dir, aggregates, priorities, observations)
    print(f"Aggregated {len(records)} independent trial-policy summaries into {args.output_dir}")


if __name__ == "__main__":
    main()
