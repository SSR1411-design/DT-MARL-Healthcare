"""Read-only integrity validation for Sprint 11 scenario and trial artifacts.

The validator deliberately treats provenance as experimental data.  It rejects
duplicate independent trials, geometry-incompatible MAPPO evaluations,
unaligned OOF reuse, zero-imputed unavailable uncertainty results, and CIs
created from fewer than two independent simulator trials.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sprint11.aggregate_sprint11 import AGGREGATE_FIELDS, load_results
from training.sprint11.schema import (Scenario, SchemaError, canonical_json,
                                      load_json, sha256_bytes)


DEFAULT_DATA_ROOT = ROOT / "data" / "sprint11"


def validate_scenario_manifests(scenario_root: Path) -> dict[str, Mapping[str, Any]]:
    manifests: dict[str, Mapping[str, Any]] = {}
    if not scenario_root.exists():
        return manifests
    for path in sorted(scenario_root.rglob("scenario_manifest.json")):
        manifest = load_json(path)
        if not isinstance(manifest, Mapping) or manifest.get("record_type") != "scenario_manifest":
            raise SchemaError(f"invalid scenario manifest: {path}")
        scenario = Scenario.from_mapping(manifest)
        if manifest.get("scenario_id") != scenario.scenario_id:
            raise SchemaError(f"scenario id/hash mismatch: {path}")
        expected_config_hash = sha256_bytes(canonical_json(scenario.canonical_config()).encode("utf-8"))
        if manifest.get("configuration_sha256") != expected_config_hash:
            raise SchemaError(f"configuration hash mismatch: {path}")
        if manifest.get("effective_failure_parameters") != scenario.effective_failure_parameters:
            raise SchemaError(f"effective progressive-wear configuration mismatch: {path}")
        if scenario.scenario_id in manifests:
            raise SchemaError(f"duplicate scenario_id in manifests: {scenario.scenario_id}")
        manifests[scenario.scenario_id] = manifest
    return manifests


def _same(value: Any, expected: Any) -> bool:
    if isinstance(value, float) or isinstance(expected, float):
        return math.isclose(float(value), float(expected), rel_tol=0.0, abs_tol=1e-12)
    return value == expected


def validate_trial_results(trial_root: Path, manifests: Mapping[str, Mapping[str, Any]],
                           *, verify_files: bool = False) -> int:
    records = load_results(trial_root, require_files=verify_files) if trial_root.exists() else []
    trace_by_trial: dict[tuple[str, int], tuple[str, str]] = {}
    trace_scenarios: dict[str, set[str]] = defaultdict(set)
    risk_trace_by_artifact: dict[str, str] = {}
    for record in records:
        scenario_id = str(record["scenario_id"])
        if scenario_id not in manifests:
            raise SchemaError(f"result references an unknown scenario: {scenario_id}")
        manifest = manifests[scenario_id]
        for field in ("host_count", "edge_node_count", "device_count", "task_count",
                      "patient_count", "bandwidth_mbps", "failure_mode", "failure_intensity"):
            if not _same(record[field], manifest[field]):
                raise SchemaError(f"result scenario metadata disagrees on {field}: {scenario_id}")
        if record["failure_parameters"] != manifest["effective_failure_parameters"]:
            raise SchemaError(f"result failure parameters are not the scenario's exact effective parameters")
        trial_key = (scenario_id, int(record["trial_id"]))
        trace = (str(record["trace_id"]), str(record["trace_hash"]))
        if trial_key in trace_by_trial and trace_by_trial[trial_key] != trace:
            raise SchemaError("policies in the same simulator trial use different traces")
        trace_by_trial[trial_key] = trace
        trace_scenarios[trace[1]].add(scenario_id)
        if record["risk_alignment_status"] == "aligned":
            risk_hash = str(record["risk_artifact_hash"])
            prior_trace = risk_trace_by_artifact.setdefault(risk_hash, str(record["trace_hash"]))
            if prior_trace != record["trace_hash"]:
                raise SchemaError("one OOF risk artifact is reused for different traces")
        if "uncertainty_gating_error_rate" in record["metrics"]:
            raise SchemaError("uncertainty gating metric is prohibited without a genuine estimator")
    contaminated = {trace: ids for trace, ids in trace_scenarios.items() if len(ids) > 1}
    if contaminated:
        raise SchemaError("trace hash reused across scenarios: " + repr(contaminated))
    return len(records)


def validate_aggregate_files(aggregate_root: Path) -> int:
    checked = 0
    if not aggregate_root.exists():
        return checked
    for path in sorted(aggregate_root.rglob("trial_metric_aggregates.csv")):
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or set(AGGREGATE_FIELDS) - set(reader.fieldnames):
                raise SchemaError(f"aggregate CSV schema mismatch: {path}")
            seen: set[tuple[str, str, str]] = set()
            for row in reader:
                key = (row["scenario_id"], row["policy"], row["metric"])
                if key in seen:
                    raise SchemaError(f"duplicate aggregate metric row: {key}")
                seen.add(key)
                n = int(row["sample_count"])
                if n < 1:
                    raise SchemaError("aggregate sample_count must be positive")
                status = row["ci95_status"]
                if n == 1:
                    if status != "not_available_n_lt_2" or row["ci95_low"] or row["ci95_high"]:
                        raise SchemaError("confidence interval fabricated from one trial")
                elif status not in {"t_interval_95", "t_interval_95_approx_df_gt_30"}:
                    raise SchemaError("multi-trial aggregate has no valid confidence interval")
                else:
                    if float(row["ci95_low"]) > float(row["ci95_high"]):
                        raise SchemaError("aggregate confidence bounds reversed")
                checked += 1
    return checked


def validate_all(data_root: Path = DEFAULT_DATA_ROOT, *, verify_files: bool = False) -> dict[str, int]:
    manifests = validate_scenario_manifests(data_root / "scenarios")
    records = validate_trial_results(data_root / "trials", manifests, verify_files=verify_files)
    aggregates = validate_aggregate_files(data_root / "aggregated")
    return {"scenario_manifests": len(manifests), "trial_results": records,
            "aggregate_rows": aggregates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--verify-files", action="store_true",
                        help="hash referenced trace/checkpoint/risk files when paths are recorded")
    parser.add_argument("--report", type=Path,
                        help="optional new JSON report path; never overwrites")
    args = parser.parse_args()
    result = validate_all(args.data_root, verify_files=args.verify_files)
    if args.report:
        if args.report.exists():
            parser.error(f"refusing to overwrite report: {args.report}")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Sprint 11 validation passed:")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
