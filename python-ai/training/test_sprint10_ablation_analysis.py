"""Regression tests for the CSV-only Sprint 10 ablation analysis."""

from __future__ import annotations

import csv
import math
import shutil
import tempfile
import unittest
from pathlib import Path

from training.analyze_sprint10_ablation import (
    COMPARABLE_COLUMNS,
    EXPERIMENTS,
    MEANS_COLUMNS,
    SPRINT10_ONLY_METRICS,
    create_analysis,
    sha256,
)
from training.validate_sprint10_ablation_analysis import validate_analysis_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Sprint10AblationAnalysisTests(unittest.TestCase):
    """Use copied CSV fixtures so no existing evaluation evidence is touched."""

    def make_fixture_root(self, temporary_directory: Path) -> Path:
        for spec in EXPERIMENTS:
            source = spec.source_path(PROJECT_ROOT)
            destination = temporary_directory / spec.relative_source
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        return temporary_directory

    def test_generates_validated_tables_without_modifying_source_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_fixture_root(Path(directory))
            source_hashes_before = {
                spec.relative_source: sha256(spec.source_path(root)) for spec in EXPERIMENTS
            }
            output_dir = root / "data" / "sprint10" / "ablation_analysis"

            metadata = create_analysis(root, output_dir)
            validate_analysis_artifacts(root, output_dir)

            self.assertTrue(metadata["source_files_unchanged_during_analysis"])
            self.assertEqual(
                source_hashes_before,
                {spec.relative_source: sha256(spec.source_path(root)) for spec in EXPERIMENTS},
            )
            with (output_dir / "ablation_results_means.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, list(MEANS_COLUMNS))
                means_rows = list(reader)
            self.assertEqual(len(means_rows), 6)
            r2_row = means_rows[0]
            for metric in SPRINT10_ONLY_METRICS:
                self.assertEqual(r2_row[metric], "")
            for row in means_rows[1:]:
                self.assertTrue(math.isnan(float(row["mean_recovery_latency_s"])))

            with (output_dir / "ablation_results_comparable.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                self.assertEqual(csv.DictReader(handle).fieldnames, list(COMPARABLE_COLUMNS))

    def test_validator_detects_a_changed_source_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_fixture_root(Path(directory))
            output_dir = root / "data" / "sprint10" / "ablation_analysis"
            create_analysis(root, output_dir)

            # This touches only a disposable copied fixture. It demonstrates that
            # the validator catches an altered source rather than accepting it.
            r2_copy = EXPERIMENTS[0].source_path(root)
            with r2_copy.open("a", encoding="utf-8", newline="") as handle:
                handle.write("\n")

            with self.assertRaisesRegex(AssertionError, "modified after analysis"):
                validate_analysis_artifacts(root, output_dir)


if __name__ == "__main__":
    unittest.main(verbosity=2)
