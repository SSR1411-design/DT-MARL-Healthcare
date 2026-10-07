"""Generate immutable Sprint 11 scenario manifests and Java properties.

No simulator is started by this tool.  It only writes new files beneath an
explicit Sprint 11 output root and refuses to replace an existing scenario.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.sprint11 import SCHEMA_VERSION
from training.sprint11.schema import (Scenario, SchemaError, canonical_json,
                                      load_json, sha256_bytes)


DEFAULT_SCENARIO_ROOT = ROOT / "data" / "sprint11" / "scenarios"


def _write_new(path: Path, text: str) -> None:
    """Atomically create a file, refusing overwrite including a race."""
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {path}")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        # Link creates a destination only when it does not already exist.
        os.link(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def serialize_java_properties_path(path: str | os.PathLike[str]) -> str:
    """Serialize a filesystem path safely for Java ``Properties.load``.

    A raw Windows backslash is an escape introducer in a Java properties file:
    for example ``\\trials`` begins with a tab escape.  Java accepts forward
    slashes for Windows drive paths (``D:/...``), so normalising separators is
    unambiguous and avoids every backslash escape without altering the actual
    filesystem target.
    """
    value = os.fspath(path)
    if not isinstance(value, str):
        raise TypeError("Java properties paths must be text paths")
    if any(character in value for character in ("\r", "\n", "\t", "\f")):
        raise ValueError("filesystem path contains a control character unsafe for Java properties")
    return value.replace("\\", "/")


def _properties(manifest: Mapping[str, Any], output_directory: str | os.PathLike[str]) -> str:
    params = manifest["effective_failure_parameters"]
    pairs: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": manifest["scenario_id"],
        "simulator_seed": "TRIAL_SEED_REQUIRED",
        "host_count": manifest["host_count"],
        "edge_node_count": manifest["edge_node_count"],
        "device_count": manifest["device_count"],
        "task_count": manifest["task_count"],
        "patient_count": manifest["patient_count"],
        "bandwidth_mbps": manifest["bandwidth_mbps"],
        "failure_mode": manifest["failure_mode"],
        "failure_intensity": manifest["failure_intensity"],
        "risk_alignment_status": "unavailable",
        "output_directory": serialize_java_properties_path(output_directory),
    }
    for key, value in params.items():
        pairs[f"failure.{key}"] = value
    return "\n".join(f"{key}={value}" for key, value in sorted(pairs.items())) + "\n"


def iter_scenarios(document: Mapping[str, Any]) -> Iterable[Scenario]:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise SchemaError(f"scenario matrix must use schema_version {SCHEMA_VERSION}")
    items = document.get("scenarios")
    if not isinstance(items, list) or not items:
        raise SchemaError("scenario matrix requires a non-empty scenarios list")
    seen: set[str] = set()
    for raw in items:
        if not isinstance(raw, Mapping):
            raise SchemaError("each scenario must be an object")
        scenario = Scenario.from_mapping(raw)
        if scenario.scenario_id in seen:
            raise SchemaError(f"duplicate scenario configuration: {scenario.scenario_id}")
        seen.add(scenario.scenario_id)
        yield scenario


def materialize_scenarios(matrix_path: Path, output_root: Path = DEFAULT_SCENARIO_ROOT) -> list[Path]:
    """Create one isolated immutable directory per validated configuration."""
    document = load_json(matrix_path)
    if not isinstance(document, Mapping):
        raise SchemaError("scenario matrix must be a JSON object")
    scenarios = list(iter_scenarios(document))
    output_root.mkdir(parents=True, exist_ok=True)
    planned = [(scenario, output_root / scenario.scenario_id) for scenario in scenarios]
    existing = [directory for _, directory in planned if directory.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite scenario directories: " +
                             ", ".join(str(p) for p in existing))
    created: list[Path] = []
    try:
        for scenario, directory in planned:
            directory.mkdir()
            created.append(directory)
            manifest = scenario.manifest_config()
            manifest.update({
                "record_type": "scenario_manifest",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_matrix_sha256": sha256_bytes(matrix_path.read_bytes()),
                "simulator_contract": "simulation Sprint11ScenarioConfig properties",
                "status": "planned_not_executed",
            })
            manifest["configuration_sha256"] = sha256_bytes(
                canonical_json(scenario.canonical_config()).encode("utf-8"))
            _write_new(directory / "scenario_manifest.json",
                       json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            _write_new(directory / "simulator.properties",
                       _properties(manifest, directory / "TRIAL_OUTPUT_REQUIRED"))
    except Exception:
        # Clean up only brand-new empty/incomplete directories made in this call.
        for directory in reversed(created):
            shutil.rmtree(directory)
        raise
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path,
                        default=DEFAULT_SCENARIO_ROOT / "sprint11_scenario_matrix.json")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_SCENARIO_ROOT)
    args = parser.parse_args()
    paths = materialize_scenarios(args.matrix, args.output_root)
    print("Created immutable Sprint 11 scenario manifests:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
