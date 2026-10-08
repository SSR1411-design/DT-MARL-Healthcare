"""Plan, never implicitly execute, a Sprint 11 scenario × trial × policy run.

The generated plan is an immutable request for a future simulator/evaluation
job.  It includes seed provenance and explicit blockers for OOF alignment and
MAPPO geometry.  ``--execute`` is intentionally not implemented: execution is
a separately reviewed campaign action, not a side effect of planning.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sprint11 import SCHEMA_VERSION
from training.sprint11.schema import (R2_HOST_COUNT, Scenario, SchemaError,
                                      load_json, sha256_file, trial_seed_pair)
from training.sprint11.scenario_generator import _properties


DEFAULT_TRIAL_ROOT = ROOT / "data" / "sprint11" / "trials"
R2_CHECKPOINT = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_best.pth"
R2_CONFIG = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_config.json"

POLICY_KINDS = {"baseline", "frozen_r2_mappo", "scale_specific_mappo"}


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_new_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_manifest(path: Path) -> tuple[Mapping[str, Any], Scenario]:
    manifest = load_json(path)
    if not isinstance(manifest, Mapping) or manifest.get("record_type") != "scenario_manifest":
        raise SchemaError("expected an immutable Sprint 11 scenario manifest")
    scenario = Scenario.from_mapping(manifest)
    if manifest.get("scenario_id") != scenario.scenario_id:
        raise SchemaError("scenario manifest id does not match its configuration")
    return manifest, scenario


def checkpoint_geometry(config_path: Path) -> int:
    document = load_json(config_path)
    try:
        return int(document["config"]["env"]["n_edge_nodes"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SchemaError(f"cannot determine MAPPO geometry from {config_path}") from exc


def policy_preflight(scenario: Scenario, policy: str, policy_kind: str,
                     checkpoint_path: Path | None = None,
                     checkpoint_config: Path | None = None) -> dict[str, Any]:
    """Return explicit policy compatibility; never load model weights."""
    if policy_kind not in POLICY_KINDS:
        raise SchemaError(f"unsupported policy_kind: {policy_kind}")
    if policy_kind == "baseline":
        return {
            "checkpoint_id": "", "checkpoint_path": "", "checkpoint_hash": "",
            "checkpoint_compatibility": "not_applicable", "checkpoint_host_count": None,
            "evaluation_blockers": [],
        }
    checkpoint_path = checkpoint_path or R2_CHECKPOINT
    checkpoint_config = checkpoint_config or R2_CONFIG
    if not checkpoint_path.is_file() or not checkpoint_config.is_file():
        raise SchemaError("MAPPO plan requires a checkpoint and companion config JSON")
    host_geometry = checkpoint_geometry(checkpoint_config)
    compatible = host_geometry == scenario.host_count
    if policy_kind == "frozen_r2_mappo" and scenario.host_count != R2_HOST_COUNT:
        compatible = False
    blockers: list[str] = []
    if not compatible:
        blockers.append(
            f"checkpoint geometry is {host_geometry} hosts but scenario requires {scenario.host_count}")
    return {
        "checkpoint_id": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_hash": sha256_file(checkpoint_path),
        "checkpoint_compatibility": "compatible" if compatible else "incompatible",
        "checkpoint_host_count": host_geometry,
        "evaluation_blockers": blockers,
    }


def build_trial_plan(manifest: Mapping[str, Any], scenario: Scenario, trial_id: int,
                     policy: str, policy_kind: str,
                     checkpoint_path: Path | None = None,
                     checkpoint_config: Path | None = None) -> dict[str, Any]:
    if not policy.strip():
        raise SchemaError("policy name is required")
    if trial_id > scenario.trials:
        raise SchemaError(f"trial_id {trial_id} exceeds planned trials={scenario.trials}")
    seeds = trial_seed_pair(scenario.scenario_id, trial_id)
    preflight = policy_preflight(scenario, policy, policy_kind,
                                 checkpoint_path, checkpoint_config)
    blockers = list(preflight.pop("evaluation_blockers"))
    # New simulator traces have no valid OOF file until a trace-aligned OOF
    # procedure is deliberately run.  A plan therefore cannot pretend that
    # an historic artifact applies to a future seed.
    if policy_kind != "baseline":
        blockers.append("scenario-specific aligned OOF risk artifact is not yet available")
    plan = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "trial_plan",
        "plan_status": "blocked" if blockers else "ready_for_simulator_only",
        "scenario_id": scenario.scenario_id,
        "trial_id": trial_id,
        "policy": policy,
        "policy_kind": policy_kind,
        "simulator_seed": seeds["simulator_seed"],
        "python_seed": seeds["python_seed"],
        "host_count": scenario.host_count,
        "edge_node_count": scenario.edge_node_count,
        "device_count": scenario.device_count,
        "task_count": scenario.task_count,
        "patient_count": scenario.patient_count,
        "bandwidth_mbps": scenario.bandwidth_mbps,
        "failure_mode": scenario.failure_mode,
        "failure_intensity": scenario.failure_intensity,
        "failure_parameters": scenario.effective_failure_parameters,
        "scenario_manifest_sha256": manifest.get("configuration_sha256", ""),
        "risk_artifact_status": "unavailable_until_trace_aligned_oof_regeneration",
        "uncertainty_gate_status": "not_available_reserved_zero_interface",
        "evaluation_protocol": "trial_summary; evaluation windows nested within independent simulator trial",
        "simulator_properties_template": "simulator.properties with simulator_seed replaced by this plan's simulator_seed",
        "evaluation_blockers": blockers,
        **preflight,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return plan


def materialize_trial_plan(manifest_path: Path, trial_id: int, policy: str,
                           policy_kind: str, output_root: Path = DEFAULT_TRIAL_ROOT,
                           checkpoint_path: Path | None = None,
                           checkpoint_config: Path | None = None) -> Path:
    manifest, scenario = load_manifest(manifest_path)
    plan = build_trial_plan(manifest, scenario, trial_id, policy, policy_kind,
                            checkpoint_path, checkpoint_config)
    safe_policy = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in policy)
    directory = output_root / scenario.scenario_id / f"trial-{trial_id:03d}" / safe_policy
    if directory.exists():
        raise FileExistsError(f"refusing to overwrite trial directory: {directory}")
    directory.mkdir(parents=True)
    trial_output = directory / "simulator_output"
    plan["simulator_output_directory"] = str(trial_output.resolve())
    properties = _properties(manifest, trial_output.resolve()).replace(
        "simulator_seed=TRIAL_SEED_REQUIRED",
        f"simulator_seed={plan['simulator_seed']}")
    plan["simulator_properties_path"] = str((directory / "simulator.properties").resolve())
    _write_new(directory / "trial_plan.json", plan)
    _write_new_text(directory / "simulator.properties", properties)
    return directory / "trial_plan.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-manifest", required=True, type=Path)
    parser.add_argument("--trial-id", required=True, type=int)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--policy-kind", required=True, choices=sorted(POLICY_KINDS))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_TRIAL_ROOT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-config", type=Path)
    parser.add_argument("--execute", action="store_true",
                        help="rejected: this infrastructure pass never runs a simulator")
    args = parser.parse_args()
    if args.execute:
        parser.error("--execute is intentionally disabled; run a reviewed simulator command manually")
    plan = materialize_trial_plan(args.scenario_manifest, args.trial_id, args.policy,
                                  args.policy_kind, args.output_root, args.checkpoint,
                                  args.checkpoint_config)
    print(f"Created immutable trial plan: {plan}")


if __name__ == "__main__":
    main()
