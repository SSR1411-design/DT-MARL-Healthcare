"""Create a read-only, reproducible analysis of completed Sprint 10 ablations.

This module intentionally reads only six episode-metrics CSV files.  It never
imports an environment, model, checkpoint, or prediction artifact, and it only
writes the requested files in ``data/sprint10/ablation_analysis`` (or an
explicitly supplied analysis directory).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "sprint10" / "ablation_analysis"
ANALYSIS_VERSION = "1.0"


@dataclass(frozen=True)
class ExperimentSpec:
    """An immutable reference to one existing evaluation export."""

    experiment: str
    description: str
    relative_source: str
    expected_rows: int

    def source_path(self, root: Path) -> Path:
        return root / Path(self.relative_source)


EXPERIMENTS: tuple[ExperimentSpec, ...] = (
    ExperimentSpec(
        "R2 baseline",
        "Validated R2 MAPPO baseline (legacy Sprint 8 episode schema).",
        "data/sprint8_episode_metrics.csv",
        1,
    ),
    ExperimentSpec(
        "A1 No Digital Twin",
        "A1 ablation: Digital Twin decision context removed.",
        "data/sprint10/a1_no_digital_twin/sprint10_a1_episode_metrics.csv",
        8,
    ),
    ExperimentSpec(
        "A2 No Failure Prediction",
        "A2 ablation: failure prediction removed.",
        "data/sprint10/a2_no_failure_prediction_rerun/sprint10_a2_episode_metrics.csv",
        8,
    ),
    ExperimentSpec(
        "A3 Single-Agent PPO / No MARL",
        "A3 ablation: single-agent PPO replaces MARL action selection.",
        "data/sprint10/a3_single_agent/sprint10_a3_episode_metrics.csv",
        8,
    ),
    ExperimentSpec(
        "A4 No Criticality",
        "A4 ablation: clinical criticality awareness removed.",
        "data/sprint10/a4_no_criticality/sprint10_a4_episode_metrics.csv",
        8,
    ),
    ExperimentSpec(
        "A5 No Uncertainty",
        "A5 ablation: uncertainty interface removed.",
        "data/sprint10/a5_no_uncertainty/sprint10_a5_episode_metrics.csv",
        8,
    ),
)

# These fields have the exact same CSV names and measurement meaning in every
# source.  Protocol/identifier columns such as ``episode`` and end ticks are
# deliberately excluded: their arithmetic mean is not a study outcome.
COMPARABLE_METRICS: tuple[str, ...] = (
    "decision_steps",
    "episode_reward",
    "task_success_rate",
    "completed",
    "lost",
    "unfinished",
    "avg_task_latency_s",
    "energy_reward_cost",
    "migrations",
    "migrations_edge",
    "migrations_cloud",
    "reroutes",
    "relocations",
    "preemptive_relocations",
    "reactive_relocations",
    "tasks_protected_before_failure",
    "sla_violations",
    "mean_link_latency_ms",
    "mean_power_w",
    "mean_cpu_utilization_pct",
    "mean_ram_utilization_pct",
    "mean_bandwidth_utilization_pct",
    "mean_hsi",
    "min_hsi",
    "max_hsi",
    "mean_priority_at_start",
    "mean_priority_at_end",
)

# These fields exist in each completed Sprint 10 ablation but not in the R2
# legacy export.  They remain separate rather than receiving invented R2
# values or being mapped onto differently named legacy fields.
SPRINT10_ONLY_METRICS: tuple[str, ...] = (
    "tasks_lost_on_resident_host",
    "tasks_lost_in_flight",
    "failed_critical_tasks",
    "critical_success_rate",
    "infeasible_actions",
    "recovery_attempts",
    "recoveries_succeeded",
    "recoveries_failed",
    "mean_recovery_latency_s",
    "prediction_used",
    "uncertainty_used",
)

IDENTITY_COLUMNS: tuple[str, ...] = (
    "experiment",
    "description",
    "n_episodes",
    "episode_start_ticks",
)
MEANS_COLUMNS: tuple[str, ...] = (
    *IDENTITY_COLUMNS,
    *COMPARABLE_METRICS,
    *SPRINT10_ONLY_METRICS,
)
COMPARABLE_COLUMNS: tuple[str, ...] = (*IDENTITY_COLUMNS, *COMPARABLE_METRICS)

R2_COMPARISON_NOTE = (
    "descriptive only; not an aligned 8-episode comparison."
)


def sha256(path: Path) -> str:
    """Return a content digest without altering *path*."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_episode_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read an existing CSV as text values, preserving source NaN spellings."""

    if not path.is_file():
        raise FileNotFoundError(f"required source CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"source CSV has no header: {path}")
        rows = list(reader)
    return list(reader.fieldnames), rows


def _number(value: str, *, context: str) -> float:
    """Parse a source numeric value without treating blanks or NaN as zero."""

    if value is None or not value.strip():
        raise ValueError(f"missing numeric value for {context}")
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"non-numeric value for {context}: {value!r}") from error


def arithmetic_mean(rows: Sequence[Mapping[str, str]], metric: str, source: Path) -> float:
    """Compute an arithmetic mean; a source NaN remains a NaN result.

    Propagating NaN is intentional.  It prevents a missing recovery latency,
    for example, from becoming a false zero or from being silently excluded.
    """

    if not rows:
        raise ValueError(f"cannot calculate {metric} from zero rows in {source}")
    values = [
        _number(row[metric], context=f"{source.name}:{metric}")
        for row in rows
    ]
    if any(math.isnan(value) for value in values):
        return math.nan
    if any(math.isinf(value) for value in values):
        raise ValueError(f"infinite numeric value for {metric} in {source}")
    return math.fsum(values) / len(values)


def format_number(value: float | None) -> str:
    """Give generated CSVs a stable representation, including source NaN."""

    if value is None:
        return ""
    if math.isnan(value):
        return "NaN"
    return format(value, ".17g")


def numeric_columns(headers: Sequence[str], rows: Sequence[Mapping[str, str]], path: Path) -> list[str]:
    """Identify source columns whose every episode value parses numerically."""

    numeric: list[str] = []
    for header in headers:
        try:
            for row in rows:
                _number(row[header], context=f"{path.name}:{header}")
        except ValueError:
            continue
        numeric.append(header)
    return numeric


def episode_starts(rows: Sequence[Mapping[str, str]], source: Path) -> list[int]:
    return [
        int(_number(row["episode_start_tick"], context=f"{source.name}:episode_start_tick"))
        for row in rows
    ]


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _require_metrics(
    headers: Iterable[str], metrics: Iterable[str], source: Path, category: str
) -> None:
    missing = [metric for metric in metrics if metric not in headers]
    if missing:
        raise ValueError(
            f"{source} is missing {category} metric(s): {', '.join(missing)}"
        )


def inspect_sources(root: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load and validate the six sources, returning facts used for analysis."""

    inspected: list[dict[str, Any]] = []
    before_hashes: dict[str, str] = {}
    for spec in EXPERIMENTS:
        source = spec.source_path(root)
        headers, rows = read_episode_csv(source)
        if len(rows) != spec.expected_rows:
            raise ValueError(
                f"{source} has {len(rows)} rows; expected {spec.expected_rows} for "
                f"{spec.experiment}"
            )
        _require_metrics(headers, COMPARABLE_METRICS, source, "comparable")
        if spec.experiment == "R2 baseline":
            present_sprint10 = [metric for metric in SPRINT10_ONLY_METRICS if metric in headers]
            if present_sprint10:
                raise ValueError(
                    f"legacy R2 source unexpectedly contains Sprint-10-only metric(s): "
                    f"{', '.join(present_sprint10)}"
                )
        else:
            _require_metrics(headers, SPRINT10_ONLY_METRICS, source, "Sprint-10-only")
        relative = _relative(source, root)
        before_hashes[relative] = sha256(source)
        inspected.append(
            {
                "spec": spec,
                "source": source,
                "headers": headers,
                "rows": rows,
                "starts": episode_starts(rows, source),
                "numeric_columns": numeric_columns(headers, rows, source),
            }
        )
    return inspected, before_hashes


def build_mean_rows(inspected: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Calculate the requested means from the in-memory read-only source data."""

    result: list[dict[str, str]] = []
    for item in inspected:
        spec: ExperimentSpec = item["spec"]
        source: Path = item["source"]
        rows: list[dict[str, str]] = item["rows"]
        headers: list[str] = item["headers"]
        mean_row: dict[str, str] = {
            "experiment": spec.experiment,
            "description": spec.description,
            "n_episodes": str(len(rows)),
            "episode_start_ticks": ";".join(str(start) for start in item["starts"]),
        }
        for metric in COMPARABLE_METRICS:
            mean_row[metric] = format_number(arithmetic_mean(rows, metric, source))
        for metric in SPRINT10_ONLY_METRICS:
            # An empty R2 cell records an absent source field.  It is not an
            # imputed zero and it is not a fabricated R2 episode value.
            mean_row[metric] = (
                format_number(arithmetic_mean(rows, metric, source))
                if metric in headers
                else ""
            )
        result.append(mean_row)
    return result


def schema_metadata(inspected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_headers = [set(item["headers"]) for item in inspected]
    common = set.intersection(*all_headers)
    union = set.union(*all_headers)
    r2_headers = set(inspected[0]["headers"])
    sprint10_headers = set.union(*(set(item["headers"]) for item in inspected[1:]))
    return {
        "common_exact_columns_all_six": sorted(common),
        "columns_only_in_r2_legacy_schema": sorted(r2_headers - sprint10_headers),
        "columns_present_in_sprint10_schema_but_absent_from_r2": sorted(
            sprint10_headers - r2_headers
        ),
        "columns_by_experiment": {
            item["spec"].experiment: item["headers"] for item in inspected
        },
        "columns_missing_by_experiment": {
            item["spec"].experiment: sorted(union - set(item["headers"]))
            for item in inspected
        },
        "semantic_fields_not_renamed": {
            "mean_recovery_duration_s": (
                "R2 legacy-only field; it is not renamed to "
                "Sprint-10 mean_recovery_latency_s."
            ),
            "prediction_uncertainty_status": (
                "R2 legacy-only provenance field; it is not renamed to "
                "Sprint-10 prediction_used or uncertainty_used."
            ),
            "host_failures_started_in_window and host_recoveries_completed_in_window": (
                "R2 legacy-only event-window counters with no exact Sprint-10 "
                "field match."
            ),
            "baseline_name and experiment": (
                "Sprint-10 provenance labels differ by export; the analysis uses "
                "the fixed experiment labels above rather than conflating them."
            ),
        },
    }


def build_metadata(
    root: Path,
    inspected: Sequence[Mapping[str, Any]],
    source_hashes_before: Mapping[str, str],
    source_hashes_after: Mapping[str, str],
) -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for item in inspected:
        spec: ExperimentSpec = item["spec"]
        source: Path = item["source"]
        relative = _relative(source, root)
        sources[spec.experiment] = {
            "source_file": relative,
            "row_count": len(item["rows"]),
            "expected_row_count": spec.expected_rows,
            "episode_starts": item["starts"],
            "columns": item["headers"],
            "numeric_columns": item["numeric_columns"],
            "source_sha256_before_analysis": source_hashes_before[relative],
            "source_sha256_after_analysis": source_hashes_after[relative],
        }
    return {
        "analysis_name": "Sprint 10 completed-ablation analysis",
        "analysis_version": ANALYSIS_VERSION,
        "analysis_scope": (
            "CSV-only analysis. No model inference, training, checkpoint writes, "
            "environment execution, or evaluation regeneration was performed."
        ),
        "r2_comparison_status": R2_COMPARISON_NOTE,
        "r2_one_episode_limitation": {
            "row_count": 1,
            "episode_starts": sources["R2 baseline"]["episode_starts"],
            "statement": (
                "R2 contains one legacy-schema episode. A1-A5 each contain eight "
                "held-out episodes; no episode alignment, imputation, or percentage "
                "improvement versus R2 is calculated."
            ),
        },
        "expected_row_counts": {
            spec.experiment: spec.expected_rows for spec in EXPERIMENTS
        },
        "sources": sources,
        "schema_differences": schema_metadata(inspected),
        "comparable_metrics": list(COMPARABLE_METRICS),
        "comparability_rule": (
            "A comparable metric must be numeric in every source and retain the "
            "same exact CSV field name; semantically different fields are not renamed."
        ),
        "sprint10_only_metrics": list(SPRINT10_ONLY_METRICS),
        "sprint10_only_reporting": (
            "Reported in ablation_results_means.csv for A1-A5 only. Empty R2 cells "
            "mean the legacy source has no such field; they are not zeroes or imputed "
            "values."
        ),
        "nan_handling": (
            "A source NaN propagates to the corresponding generated mean as NaN; "
            "NaN is never converted to zero or dropped from the arithmetic mean."
        ),
        "source_hashes_before_analysis": dict(source_hashes_before),
        "source_hashes_after_analysis": dict(source_hashes_after),
        "source_files_unchanged_during_analysis": source_hashes_before == source_hashes_after,
    }


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sprint10_only_readme_table(mean_rows: Sequence[Mapping[str, str]]) -> str:
    headers = ["experiment", *SPRINT10_ONLY_METRICS]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in mean_rows:
        if row["experiment"] == "R2 baseline":
            continue
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")
    return "\n".join(lines)


def build_readme(mean_rows: Sequence[Mapping[str, str]]) -> str:
    source_rows = "\n".join(
        f"| {spec.experiment} | `{spec.relative_source}` | {spec.expected_rows} |"
        for spec in EXPERIMENTS
    )
    return f"""# Sprint 10 ablation analysis

This directory is generated by `training/analyze_sprint10_ablation.py`. It is
an analysis-only artifact: the script reads six completed episode-metrics CSVs
and does not run models, training, environments, checkpoints, or predictions.

## Inputs inspected

| Experiment | Source CSV | Rows |
| --- | --- | ---: |
{source_rows}

R2 has one legacy-schema episode at tick 491. A1-A5 each have eight held-out
episodes at ticks 491, 521, 550, 580, 609, 639, 668, and 698. Accordingly, R2
comparisons are **{R2_COMPARISON_NOTE}** No R2 episode rows are aligned,
fabricated, or used for percentage-improvement claims.

## Generated files

- `ablation_results_means.csv` contains the arithmetic means for all shared
  metrics plus a separately labelled Sprint-10-only section. R2 fields absent
  from its source are empty, not zero-filled.
- `ablation_results_comparable.csv` contains only exact-name, numeric metrics
  that appear in all six sources. It does not rename legacy fields to create a
  comparison.
- `ablation_results_metadata.json` records source digests, schemas, row counts,
  episode starts, comparability decisions, and NaN handling.

## Sprint-10-only metric means (A1-A5)

`mean_recovery_latency_s` is `NaN` in the source exports and remains `NaN` here;
it was not converted to zero or excluded from the mean.

{_sprint10_only_readme_table(mean_rows)}

## Reproduce and validate

From `python-ai`:

```powershell
python training/analyze_sprint10_ablation.py
python training/validate_sprint10_ablation_analysis.py
python -m unittest training.test_sprint10_ablation_analysis
```

The analysis command refuses to overwrite an existing nonempty analysis
directory. Use a different `--output-dir` for an independent reproduction.
"""


def ensure_new_output_directory(output_dir: Path) -> None:
    """Avoid overwriting any pre-existing analysis artifacts by default."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty analysis directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def create_analysis(root: Path, output_dir: Path) -> dict[str, Any]:
    """Generate analysis artifacts, verifying sources stayed byte-identical."""

    root = root.resolve()
    output_dir = output_dir.resolve()
    inspected, hashes_before = inspect_sources(root)
    mean_rows = build_mean_rows(inspected)
    hashes_after = {
        _relative(item["source"], root): sha256(item["source"])
        for item in inspected
    }
    if hashes_before != hashes_after:
        raise RuntimeError("a source evaluation CSV changed during read-only analysis")

    source_paths = {item["source"].resolve() for item in inspected}
    artifact_paths = {
        output_dir / "ablation_results_means.csv",
        output_dir / "ablation_results_comparable.csv",
        output_dir / "ablation_results_metadata.json",
        output_dir / "README.md",
    }
    if source_paths & {path.resolve() for path in artifact_paths}:
        raise RuntimeError("an analysis output path collides with a source evaluation CSV")

    ensure_new_output_directory(output_dir)
    metadata = build_metadata(root, inspected, hashes_before, hashes_after)
    _write_csv(output_dir / "ablation_results_means.csv", MEANS_COLUMNS, mean_rows)
    _write_csv(
        output_dir / "ablation_results_comparable.csv",
        COMPARABLE_COLUMNS,
        [{column: row[column] for column in COMPARABLE_COLUMNS} for row in mean_rows],
    )
    with (output_dir / "ablation_results_metadata.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    (output_dir / "README.md").write_text(build_readme(mean_rows), encoding="utf-8", newline="\n")
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="python-ai root containing data/ (default: script project root)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="new directory for generated analysis artifacts",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else root / DEFAULT_OUTPUT_DIR.relative_to(ROOT)
    metadata = create_analysis(root, output_dir)
    print(f"Created read-only ablation analysis: {output_dir}")
    print(
        "Source rows: "
        + ", ".join(
            f"{name}={entry['row_count']}" for name, entry in metadata["sources"].items()
        )
    )
    print("R2 comparison status: " + metadata["r2_comparison_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
