"""Strict, dependency-light schemas for Sprint 11 experiment provenance.

The campaign has two separate identities:

* a *scenario* is an immutable physical-simulator configuration; and
* a *trial* is one independently seeded realization of that scenario.

Evaluation windows inside a trial are intentionally not independent trials.
This module does not touch the simulator or any historical artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from training.sprint11 import SCHEMA_VERSION


class SchemaError(ValueError):
    """Raised when a scenario or result is scientifically unsafe to use."""


FAILURE_MODE = "progressive_stochastic_wear"
R2_HOST_COUNT = 10

# Values below are the active progressive-wear values in SimulationManager,
# not the older HostDegradationConfig constructor defaults.
REQUIRED_FAILURE_PARAMETERS = frozenset({
    "fault_onset_probability_per_tick",
    "background_wear_per_tick",
    "fault_episode_wear_per_tick",
    "wear_noise_sigma",
    "susceptibility_min",
    "susceptibility_max",
    "severity_min",
    "severity_max",
    "hazard_scale",
    "hazard_shape",
    "abrupt_failure_probability",
    "repair_ticks_min",
    "repair_ticks_max",
})

RATE_SCALED_FAILURE_PARAMETERS = frozenset({
    "fault_onset_probability_per_tick",
    "background_wear_per_tick",
    "fault_episode_wear_per_tick",
    "hazard_scale",
})

DEFAULT_FAILURE_PARAMETERS: dict[str, float | int] = {
    "fault_onset_probability_per_tick": 0.0018,
    "background_wear_per_tick": 0.00015,
    "fault_episode_wear_per_tick": 0.010,
    "wear_noise_sigma": 0.45,
    "susceptibility_min": 0.6,
    "susceptibility_max": 1.8,
    "severity_min": 0.5,
    "severity_max": 2.0,
    "hazard_scale": 0.010,
    "hazard_shape": 4.0,
    "abrupt_failure_probability": 0.15,
    "abrupt_wear_multiplier": 40.0,
    "degrading_wear_threshold": 0.25,
    "critical_wear_threshold": 0.70,
    "repair_ticks_min": 12,
    "repair_ticks_max": 45,
    "imperfect_repair_retention": 0.30,
}

REQUIRED_RESULT_METRICS = (
    "task_success_rate",
    "avg_task_latency_s",
    "physical_energy_j",
    "energy_reward_cost",
    "mean_cpu_utilization_pct",
    "mean_ram_utilization_pct",
    "mean_bandwidth_utilization_pct",
    "episode_reward",
    "HSI",
)

RESULT_REQUIRED_FIELDS = frozenset({
    "schema_version", "record_type", "scenario_id", "trial_id", "policy",
    "policy_kind", "simulator_seed", "python_seed", "host_count",
    "edge_node_count", "device_count", "task_count", "patient_count",
    "bandwidth_mbps", "failure_mode", "failure_intensity", "failure_parameters", "trace_id",
    "trace_hash", "risk_artifact_id", "risk_artifact_hash",
    "risk_alignment_status", "checkpoint_id", "checkpoint_hash",
    "checkpoint_compatibility", "episode_start", "episode_end",
    "evaluation_protocol", "metrics", "uncertainty_gate_status",
})


def canonical_json(value: Any) -> str:
    """Return a canonical JSON representation suitable for hashing."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_scenario_id(config: Mapping[str, Any]) -> str:
    """Calculate the identifier from physical configuration, never labels."""
    payload = dict(config)
    payload.pop("scenario_id", None)
    payload.pop("description", None)
    payload.pop("planned_policies", None)
    payload.pop("notes", None)
    payload.pop("trials", None)  # trial count is campaign design, not physics.
    return "s11-" + sha256_bytes(canonical_json(payload).encode("utf-8"))[:16]


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise SchemaError(f"{label} must be finite")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SchemaError(f"{label} must be a positive integer")
    return int(value)


def effective_failure_parameters(base: Mapping[str, Any], intensity: float) -> dict[str, float | int]:
    """Apply a documented multiplier only to stochastic wear/hazard rates.

    Thresholds, ranges, repair physics, and abrupt-episode fraction are not
    changed by intensity.  This preserves the progressive physical mechanism
    instead of substituting a Bernoulli host-kill process.
    """
    effective: dict[str, float | int] = {}
    for key, raw in base.items():
        value = _finite_number(raw, f"failure_parameters.{key}")
        if key in RATE_SCALED_FAILURE_PARAMETERS:
            value *= intensity
        effective[key] = int(value) if key in {"repair_ticks_min", "repair_ticks_max"} else value
    return effective


def validate_failure_parameters(base: Mapping[str, Any], intensity: float) -> dict[str, float | int]:
    missing = REQUIRED_FAILURE_PARAMETERS - set(base)
    if missing:
        raise SchemaError("failure_parameters missing: " + ", ".join(sorted(missing)))
    unknown = set(base) - set(DEFAULT_FAILURE_PARAMETERS)
    if unknown:
        raise SchemaError("unsupported failure_parameters: " + ", ".join(sorted(unknown)))
    effective = effective_failure_parameters(base, intensity)
    for key, value in effective.items():
        if key in {"repair_ticks_min", "repair_ticks_max"}:
            if int(value) != value or value <= 0:
                raise SchemaError(f"failure_parameters.{key} must be a positive integer")
        elif value < 0:
            raise SchemaError(f"failure_parameters.{key} must be non-negative")
    if not 0 <= float(effective["fault_onset_probability_per_tick"]) <= 1:
        raise SchemaError("effective fault-onset probability must be in [0, 1]")
    if not 0 <= float(effective["abrupt_failure_probability"]) <= 1:
        raise SchemaError("abrupt_failure_probability must be in [0, 1]")
    if float(effective["susceptibility_min"]) <= 0 or float(effective["susceptibility_min"]) > float(effective["susceptibility_max"]):
        raise SchemaError("susceptibility range must be positive and ordered")
    if float(effective["severity_min"]) <= 0 or float(effective["severity_min"]) > float(effective["severity_max"]):
        raise SchemaError("severity range must be positive and ordered")
    if float(effective["hazard_shape"]) <= 0:
        raise SchemaError("hazard_shape must be positive")
    if int(effective["repair_ticks_min"]) > int(effective["repair_ticks_max"]):
        raise SchemaError("repair duration range must be ordered")
    for key in ("degrading_wear_threshold", "critical_wear_threshold", "imperfect_repair_retention"):
        if key in effective and not 0 <= float(effective[key]) <= 1:
            raise SchemaError(f"failure_parameters.{key} must be in [0, 1]")
    if ("degrading_wear_threshold" in effective and "critical_wear_threshold" in effective
            and float(effective["degrading_wear_threshold"]) >= float(effective["critical_wear_threshold"])):
        raise SchemaError("degrading_wear_threshold must be below critical_wear_threshold")
    return effective


@dataclass(frozen=True)
class Scenario:
    """Validated physical scenario definition before trial-specific seeds."""

    host_count: int
    edge_node_count: int
    device_count: int
    task_count: int
    patient_count: int
    bandwidth_mbps: float
    failure_intensity: float
    trials: int
    failure_parameters: dict[str, float | int]
    failure_mode: str = FAILURE_MODE
    description: str = ""
    planned_policies: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "Scenario":
        required = {
            "host_count", "edge_node_count", "device_count", "task_count",
            "patient_count", "bandwidth_mbps", "failure_intensity", "trials",
            "failure_parameters",
        }
        missing = required - set(raw)
        if missing:
            raise SchemaError("scenario missing: " + ", ".join(sorted(missing)))
        host_count = _positive_int(raw["host_count"], "host_count")
        edge_node_count = _positive_int(raw["edge_node_count"], "edge_node_count")
        if host_count != edge_node_count:
            raise SchemaError(
                "current simulator is 1:1 host-to-edge-node; host_count must equal edge_node_count")
        task_count = _positive_int(raw["task_count"], "task_count")
        patient_count = _positive_int(raw["patient_count"], "patient_count")
        if patient_count > task_count:
            raise SchemaError("patient_count cannot exceed task_count")
        bandwidth = _finite_number(raw["bandwidth_mbps"], "bandwidth_mbps")
        if bandwidth <= 0:
            raise SchemaError("bandwidth_mbps must be positive physical bandwidth")
        intensity = _finite_number(raw["failure_intensity"], "failure_intensity")
        if intensity <= 0:
            raise SchemaError("failure_intensity must be positive")
        if raw.get("failure_mode", FAILURE_MODE) != FAILURE_MODE:
            raise SchemaError("Sprint 11 only accepts progressive_stochastic_wear failure mode")
        if not isinstance(raw["failure_parameters"], Mapping):
            raise SchemaError("failure_parameters must be an object")
        params = {key: raw["failure_parameters"][key]
                  for key in raw["failure_parameters"]}
        validate_failure_parameters(params, intensity)
        policies = raw.get("planned_policies", [])
        if not isinstance(policies, list) or not all(isinstance(p, str) for p in policies):
            raise SchemaError("planned_policies must be a list of strings")
        return cls(
            host_count=host_count, edge_node_count=edge_node_count,
            device_count=_positive_int(raw["device_count"], "device_count"),
            task_count=task_count, patient_count=patient_count, bandwidth_mbps=bandwidth,
            failure_intensity=intensity, trials=_positive_int(raw["trials"], "trials"),
            failure_parameters=params, failure_mode=FAILURE_MODE,
            description=str(raw.get("description", "")),
            planned_policies=tuple(policies),
        )

    @property
    def effective_failure_parameters(self) -> dict[str, float | int]:
        return validate_failure_parameters(self.failure_parameters, self.failure_intensity)

    def canonical_config(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "host_count": self.host_count,
            "edge_node_count": self.edge_node_count,
            "device_count": self.device_count,
            "task_count": self.task_count,
            "patient_count": self.patient_count,
            "bandwidth_mbps": self.bandwidth_mbps,
            "failure_mode": self.failure_mode,
            "failure_intensity": self.failure_intensity,
            "failure_parameters": self.failure_parameters,
        }

    @property
    def scenario_id(self) -> str:
        return deterministic_scenario_id(self.canonical_config())

    def manifest_config(self) -> dict[str, Any]:
        config = self.canonical_config()
        config.update({
            "scenario_id": self.scenario_id,
            "trials": self.trials,
            "description": self.description,
            "planned_policies": list(self.planned_policies),
            "effective_failure_parameters": self.effective_failure_parameters,
        })
        return config


def derive_seed(scenario_id: str, trial_id: int, stream: str) -> int:
    """Derive a stable non-zero positive 63-bit seed for one random stream."""
    if trial_id < 1:
        raise SchemaError("trial_id must start at 1")
    if not stream:
        raise SchemaError("seed stream name is required")
    digest = hashlib.sha256(f"{scenario_id}|{trial_id}|{stream}".encode("utf-8")).digest()
    # Reserve a small positive range because Java derives manager streams by
    # adding 1..3 to this value.
    return (int.from_bytes(digest[:8], "big") % ((1 << 63) - 4)) + 1


def trial_seed_pair(scenario_id: str, trial_id: int) -> dict[str, int]:
    return {
        "simulator_seed": derive_seed(scenario_id, trial_id, "simulator"),
        "python_seed": derive_seed(scenario_id, trial_id, "python-evaluation"),
    }


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"invalid JSON in {path}: {exc}") from exc


def validate_result_record(record: Mapping[str, Any], *, require_files: bool = False) -> None:
    """Validate one completed trial-policy summary, without mutating anything."""
    missing = RESULT_REQUIRED_FIELDS - set(record)
    if missing:
        raise SchemaError("result missing fields: " + ", ".join(sorted(missing)))
    if record["schema_version"] != SCHEMA_VERSION or record["record_type"] != "trial_result":
        raise SchemaError("result has an unsupported schema_version or record_type")
    if record["failure_mode"] != FAILURE_MODE:
        raise SchemaError("result did not use progressive stochastic wear")
    host_count = _positive_int(record["host_count"], "host_count")
    if host_count != _positive_int(record["edge_node_count"], "edge_node_count"):
        raise SchemaError("result host/edge-node geometry is inconsistent")
    task_count = _positive_int(record["task_count"], "task_count")
    if _positive_int(record["patient_count"], "patient_count") > task_count:
        raise SchemaError("result patient_count cannot exceed task_count")
    if _positive_int(record["device_count"], "device_count") < 1:
        raise SchemaError("result device_count must be positive")
    if _finite_number(record["bandwidth_mbps"], "bandwidth_mbps") <= 0:
        raise SchemaError("result bandwidth_mbps must be positive")
    if _finite_number(record["failure_intensity"], "failure_intensity") <= 0:
        raise SchemaError("result failure_intensity must be positive")
    if not isinstance(record["failure_parameters"], Mapping):
        raise SchemaError("result failure_parameters must be an object")
    # Result records carry the already-effective parameters used by Java;
    # applying intensity a second time would falsify the provenance check.
    validate_failure_parameters(record["failure_parameters"], 1.0)
    trial_id = _positive_int(record["trial_id"], "trial_id")
    expected = trial_seed_pair(str(record["scenario_id"]), trial_id)
    for field, value in expected.items():
        if record[field] != value:
            raise SchemaError(f"{field} does not match deterministic seed provenance")
    for field in ("trace_hash",):
        if not isinstance(record[field], str) or len(record[field]) != 64:
            raise SchemaError(f"{field} must be a SHA-256 hex digest")
    risk_status = record["risk_alignment_status"]
    if risk_status == "aligned":
        if not isinstance(record["risk_artifact_hash"], str) or len(record["risk_artifact_hash"]) != 64:
            raise SchemaError("aligned OOF risk must have an artifact SHA-256")
        if record.get("risk_trace_hash") != record["trace_hash"]:
            raise SchemaError("aligned OOF risk must attest to the exact trace hash")
    elif risk_status == "unavailable":
        if record["risk_artifact_hash"] not in ("", None) or record["risk_artifact_id"] not in ("", None):
            raise SchemaError("unavailable OOF risk cannot be zero-imputed or given an artifact")
    else:
        raise SchemaError("risk_alignment_status must be 'aligned' or 'unavailable'")
    kind = record["policy_kind"]
    compatibility = record["checkpoint_compatibility"]
    if kind == "frozen_r2_mappo":
        if int(record["host_count"]) != R2_HOST_COUNT or compatibility != "compatible":
            raise SchemaError("frozen R2 MAPPO is only compatible with the 10-node geometry")
        if not record["checkpoint_hash"]:
            raise SchemaError("MAPPO result needs a checkpoint hash")
    elif kind == "scale_specific_mappo":
        if compatibility != "compatible" or not record["checkpoint_hash"]:
            raise SchemaError("scale-specific MAPPO needs a compatible checkpoint hash")
        if int(record.get("checkpoint_host_count", -1)) != int(record["host_count"]):
            raise SchemaError("scale-specific checkpoint host geometry does not match scenario")
    elif kind == "baseline":
        if compatibility != "not_applicable":
            raise SchemaError("baseline must not claim MAPPO checkpoint compatibility")
    else:
        raise SchemaError("unknown policy_kind")
    if not isinstance(record["metrics"], Mapping):
        raise SchemaError("metrics must be an object")
    missing_metrics = set(REQUIRED_RESULT_METRICS) - set(record["metrics"])
    if missing_metrics:
        raise SchemaError("missing required metrics: " + ", ".join(sorted(missing_metrics)))
    bounds = {
        "task_success_rate": (0.0, 1.0), "physical_energy_j": (0.0, None),
        "energy_reward_cost": (0.0, None), "avg_task_latency_s": (0.0, None),
        "mean_cpu_utilization_pct": (0.0, 100.0),
        "mean_ram_utilization_pct": (0.0, 100.0),
        "mean_bandwidth_utilization_pct": (0.0, 100.0), "HSI": (0.0, 1.0),
    }
    for name in REQUIRED_RESULT_METRICS:
        value = _finite_number(record["metrics"][name], f"metrics.{name}")
        lo, hi = bounds.get(name, (None, None))
        if lo is not None and value < lo or hi is not None and value > hi:
            raise SchemaError(f"metrics.{name} is outside its physically valid range")
    if record["uncertainty_gate_status"] != "not_available_reserved_zero_interface":
        raise SchemaError("uncertainty gate metrics require a separately validated estimator")
    if require_files:
        for path_key, hash_key in (("trace_path", "trace_hash"), ("checkpoint_path", "checkpoint_hash")):
            if record.get(path_key):
                path = Path(record[path_key])
                if not path.is_file() or sha256_file(path) != record[hash_key]:
                    raise SchemaError(f"{path_key} does not match its recorded hash")
        if risk_status == "aligned" and record.get("risk_artifact_path"):
            path = Path(record["risk_artifact_path"])
            if not path.is_file() or sha256_file(path) != record["risk_artifact_hash"]:
                raise SchemaError("risk artifact does not match its recorded hash")
