"""Validate generated, read-only Sprint 10 ablation-analysis artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.analyze_sprint10_ablation import (  # noqa: E402
    ANALYSIS_VERSION,
    COMPARABLE_COLUMNS,
    COMPARABLE_METRICS,
    DEFAULT_OUTPUT_DIR,
    EXPERIMENTS,
    MEANS_COLUMNS,
    R2_COMPARISON_NOTE,
    SPRINT10_ONLY_METRICS,
    arithmetic_mean,
    episode_starts,
    read_episode_csv,
    sha256,
)


REQUIRED_ARTIFACTS = (
    "ablation_results_means.csv",
    "ablation_results_comparable.csv",
    "ablation_results_metadata.json",
    "README.md",
)


def fail(message: str) -> None:
    raise AssertionError(message)


def read_generated_csv(path: Path, expected_header: Sequence[str]) -> list[dict[str, str]]:
    if not path.is_file():
        fail(f"missing generated analysis table: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(expected_header):
            fail(f"unexpected header in {path.name}")
        rows = list(reader)
    if len(rows) != len(EXPERIMENTS):
        fail(f"{path.name} has {len(rows)} rows; expected {len(EXPERIMENTS)}")
    return rows


def _is_within(candidate: Path, parent: Path) -> bool:
    try:
        candidate.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _assert_output_does_not_overlap_sources(root: Path, output_dir: Path) -> None:
    for spec in EXPERIMENTS:
        source = spec.source_path(root)
        if source.resolve() == output_dir.resolve() or _is_within(source, output_dir):
            fail(
                "analysis output directory contains a source evaluation CSV; "
                "refusing an output/source overlap"
            )
        for artifact in REQUIRED_ARTIFACTS:
            if source.resolve() == (output_dir / artifact).resolve():
                fail("a generated analysis artifact overwrote a source evaluation CSV")


def _parse_generated_number(value: str, context: str) -> float:
    if value == "":
        fail(f"unexpected unavailable value for {context}")
    try:
        return float(value)
    except ValueError as error:
        fail(f"non-numeric generated value for {context}: {value!r}")
        raise error  # unreachable, for type checkers


def _assert_same_number(actual_text: str, expected: float | None, context: str) -> None:
    if expected is None:
        if actual_text != "":
            fail(f"{context} should be empty because the source field is absent")
        return
    actual = _parse_generated_number(actual_text, context)
    if math.isnan(expected):
        if not math.isnan(actual):
            fail(f"{context} must preserve source NaN, got {actual_text!r}")
        return
    if math.isnan(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
        fail(f"{context}: got {actual_text!r}, expected arithmetic mean {expected!r}")


def _validate_metadata_and_sources(root: Path, output_dir: Path) -> dict[str, object]:
    metadata_path = output_dir / "ablation_results_metadata.json"
    if not metadata_path.is_file():
        fail(f"missing generated metadata: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    if metadata.get("analysis_version") != ANALYSIS_VERSION:
        fail("analysis metadata has an unexpected version")
    if metadata.get("r2_comparison_status") != R2_COMPARISON_NOTE:
        fail("metadata does not mark R2 as descriptive-only")
    if metadata.get("comparable_metrics") != list(COMPARABLE_METRICS):
        fail("metadata comparable metric list does not match the validated schema")
    if metadata.get("sprint10_only_metrics") != list(SPRINT10_ONLY_METRICS):
        fail("metadata Sprint-10-only metric list does not match the validated schema")
    if metadata.get("source_files_unchanged_during_analysis") is not True:
        fail("metadata does not prove the source CSVs were unchanged during analysis")

    sources = metadata.get("sources")
    if not isinstance(sources, dict) or set(sources) != {spec.experiment for spec in EXPERIMENTS}:
        fail("metadata source inventory is incomplete")
    expected_row_counts = metadata.get("expected_row_counts")
    if expected_row_counts != {spec.experiment: spec.expected_rows for spec in EXPERIMENTS}:
        fail("metadata expected row counts are wrong")

    before_hashes = metadata.get("source_hashes_before_analysis")
    after_hashes = metadata.get("source_hashes_after_analysis")
    if not isinstance(before_hashes, dict) or before_hashes != after_hashes:
        fail("source before/after hashes do not prove read-only analysis")

    for spec in EXPERIMENTS:
        source = spec.source_path(root)
        if not source.is_file():
            fail(f"missing required source CSV: {source}")
        headers, rows = read_episode_csv(source)
        source_metadata = sources[spec.experiment]
        if not isinstance(source_metadata, dict):
            fail(f"invalid metadata entry for {spec.experiment}")
        relative = spec.relative_source.replace("\\", "/")
        if source_metadata.get("source_file") != relative:
            fail(f"metadata references the wrong source path for {spec.experiment}")
        if len(rows) != spec.expected_rows or source_metadata.get("row_count") != spec.expected_rows:
            fail(f"unexpected source row count for {spec.experiment}")
        starts = episode_starts(rows, source)
        if source_metadata.get("episode_starts") != starts:
            fail(f"metadata episode starts do not match {spec.experiment} source")
        current_hash = sha256(source)
        if before_hashes.get(relative) != current_hash:
            fail(f"source evaluation CSV was modified after analysis: {source}")
        if source_metadata.get("source_sha256_before_analysis") != current_hash:
            fail(f"source digest mismatch for {spec.experiment}")
        if source_metadata.get("source_sha256_after_analysis") != current_hash:
            fail(f"source post-analysis digest mismatch for {spec.experiment}")
        if source_metadata.get("columns") != headers:
            fail(f"metadata source schema does not match {spec.experiment}")
    return metadata


def _validate_means_against_sources(root: Path, means_rows: Sequence[Mapping[str, str]]) -> None:
    for spec, mean_row in zip(EXPERIMENTS, means_rows, strict=True):
        source = spec.source_path(root)
        headers, rows = read_episode_csv(source)
        if mean_row["experiment"] != spec.experiment:
            fail(f"means row order/identity is wrong for {spec.experiment}")
        if mean_row["n_episodes"] != str(spec.expected_rows):
            fail(f"n_episodes does not match source for {spec.experiment}")
        expected_starts = ";".join(str(value) for value in episode_starts(rows, source))
        if mean_row["episode_start_ticks"] != expected_starts:
            fail(f"episode_start_ticks does not match source for {spec.experiment}")
        if not mean_row["description"].strip():
            fail(f"description is empty for {spec.experiment}")
        for metric in COMPARABLE_METRICS:
            _assert_same_number(
                mean_row[metric],
                arithmetic_mean(rows, metric, source),
                f"{spec.experiment}:{metric}",
            )
        for metric in SPRINT10_ONLY_METRICS:
            expected = arithmetic_mean(rows, metric, source) if metric in headers else None
            _assert_same_number(mean_row[metric], expected, f"{spec.experiment}:{metric}")


def _validate_comparable_table(
    means_rows: Sequence[Mapping[str, str]], comparable_rows: Sequence[Mapping[str, str]]
) -> None:
    for means_row, comparable_row in zip(means_rows, comparable_rows, strict=True):
        for column in COMPARABLE_COLUMNS:
            if comparable_row[column] != means_row[column]:
                fail(f"comparable table differs from means table for {means_row['experiment']}:{column}")
        extra_metrics = set(comparable_row) - set(COMPARABLE_COLUMNS)
        if extra_metrics:
            fail(f"comparable table includes noncomparable columns: {sorted(extra_metrics)}")


def _validate_readme(output_dir: Path) -> None:
    path = output_dir / "README.md"
    if not path.is_file():
        fail("missing generated analysis README")
    text = path.read_text(encoding="utf-8")
    for phrase in (
        R2_COMPARISON_NOTE,
        "No R2 episode rows are aligned",
        "mean_recovery_latency_s",
        "an analysis-only artifact",
    ):
        if phrase not in text:
            fail(f"README is missing required disclosure: {phrase}")


def validate_analysis_artifacts(root: Path, output_dir: Path) -> None:
    """Raise ``AssertionError`` if generated results diverge from their sources."""

    root = root.resolve()
    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        fail(f"analysis output directory does not exist: {output_dir}")
    _assert_output_does_not_overlap_sources(root, output_dir)
    missing_artifacts = [name for name in REQUIRED_ARTIFACTS if not (output_dir / name).is_file()]
    if missing_artifacts:
        fail("missing generated analysis artifacts: " + ", ".join(missing_artifacts))
    _validate_metadata_and_sources(root, output_dir)
    means_rows = read_generated_csv(output_dir / "ablation_results_means.csv", MEANS_COLUMNS)
    comparable_rows = read_generated_csv(
        output_dir / "ablation_results_comparable.csv", COMPARABLE_COLUMNS
    )
    _validate_means_against_sources(root, means_rows)
    _validate_comparable_table(means_rows, comparable_rows)
    _validate_readme(output_dir)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else root / DEFAULT_OUTPUT_DIR.relative_to(ROOT)
    validate_analysis_artifacts(root, output_dir)
    print("Sprint 10 ablation-analysis validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
