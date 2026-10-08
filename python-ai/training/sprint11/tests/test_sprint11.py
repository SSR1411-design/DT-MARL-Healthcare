from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path, PureWindowsPath

from training.sprint11.aggregate_sprint11 import aggregate_records, summary
from training.sprint11.plot_sprint11 import GRAPH_SPECS, validate_aggregate_rows
from training.sprint11.scenario_generator import (_properties, materialize_scenarios,
                                                  serialize_java_properties_path)
from training.sprint11.schema import (
    DEFAULT_FAILURE_PARAMETERS,
    SchemaError,
    Scenario,
    trial_seed_pair,
    validate_result_record,
)
from training.sprint11.trial_runner import build_trial_plan


def scenario_raw(**changes):
    raw = {
        "host_count": 10,
        "edge_node_count": 10,
        "device_count": 10,
        "task_count": 40,
        "patient_count": 10,
        "bandwidth_mbps": 100.0,
        "failure_intensity": 1.0,
        "trials": 5,
        "failure_parameters": copy.deepcopy(DEFAULT_FAILURE_PARAMETERS),
    }
    raw.update(changes)
    return raw


def valid_result() -> dict:
    scenario = Scenario.from_mapping(scenario_raw())
    seeds = trial_seed_pair(scenario.scenario_id, 1)
    return {
        "schema_version": "sprint11.v1",
        "record_type": "trial_result",
        "scenario_id": scenario.scenario_id,
        "trial_id": 1,
        "policy": "reactive-baseline",
        "policy_kind": "baseline",
        **seeds,
        "host_count": 10,
        "edge_node_count": 10,
        "device_count": 10,
        "task_count": 40,
        "patient_count": 10,
        "bandwidth_mbps": 100.0,
        "failure_mode": "progressive_stochastic_wear",
        "failure_intensity": 1.0,
        "failure_parameters": scenario.effective_failure_parameters,
        "trace_id": "test-trace",
        "trace_hash": "a" * 64,
        "risk_artifact_id": "",
        "risk_artifact_hash": "",
        "risk_alignment_status": "unavailable",
        "checkpoint_id": "",
        "checkpoint_hash": "",
        "checkpoint_compatibility": "not_applicable",
        "episode_start": 500,
        "episode_end": 900,
        "evaluation_protocol": "unit-test trial summary",
        "uncertainty_gate_status": "not_available_reserved_zero_interface",
        "metrics": {
            "task_success_rate": 0.75,
            "avg_task_latency_s": 1.5,
            "physical_energy_j": 20.0,
            "energy_reward_cost": 2.0,
            "mean_cpu_utilization_pct": 50.0,
            "mean_ram_utilization_pct": 40.0,
            "mean_bandwidth_utilization_pct": 30.0,
            "episode_reward": -1.0,
            "HSI": 0.4,
        },
    }


class Sprint11SchemaTests(unittest.TestCase):
    def test_scenario_hash_is_deterministic_and_excludes_description(self):
        first = Scenario.from_mapping(scenario_raw(description="first wording"))
        second = Scenario.from_mapping(scenario_raw(description="second wording"))
        self.assertEqual(first.scenario_id, second.scenario_id)
        altered = Scenario.from_mapping(scenario_raw(bandwidth_mbps=50.0))
        self.assertNotEqual(first.scenario_id, altered.scenario_id)

    def test_scenario_validation_rejects_bad_geometry_and_ranges(self):
        with self.assertRaises(SchemaError):
            Scenario.from_mapping(scenario_raw(host_count=5, edge_node_count=10))
        invalid = scenario_raw()
        invalid["failure_parameters"]["repair_ticks_min"] = 50
        invalid["failure_parameters"]["repair_ticks_max"] = 12
        with self.assertRaises(SchemaError):
            Scenario.from_mapping(invalid)

    def test_failure_intensity_scales_only_documented_progressive_rates(self):
        scenario = Scenario.from_mapping(scenario_raw(failure_intensity=1.5))
        effective = scenario.effective_failure_parameters
        self.assertAlmostEqual(effective["fault_onset_probability_per_tick"], 0.0027)
        self.assertAlmostEqual(effective["fault_episode_wear_per_tick"], 0.015)
        self.assertAlmostEqual(effective["hazard_scale"], 0.015)
        self.assertEqual(effective["repair_ticks_min"], 12)
        self.assertEqual(effective["abrupt_failure_probability"], 0.15)

    def test_seed_generation_is_stable_and_independent(self):
        scenario = Scenario.from_mapping(scenario_raw())
        seeds = [trial_seed_pair(scenario.scenario_id, trial) for trial in range(1, 6)]
        self.assertEqual(seeds[0], trial_seed_pair(scenario.scenario_id, 1))
        self.assertEqual(len({seed["simulator_seed"] for seed in seeds}), 5)
        self.assertEqual(len({seed["python_seed"] for seed in seeds}), 5)
        self.assertNotEqual(seeds[0]["simulator_seed"], seeds[0]["python_seed"])

    def test_generator_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            matrix = root / "matrix.json"
            matrix.write_text(json.dumps({"schema_version": "sprint11.v1", "scenarios": [scenario_raw()]}), encoding="utf-8")
            paths = materialize_scenarios(matrix, root / "scenarios")
            self.assertEqual(len(paths), 1)
            self.assertTrue((paths[0] / "simulator.properties").is_file())
            with self.assertRaises(FileExistsError):
                materialize_scenarios(matrix, root / "scenarios")

    def test_windows_path_is_safe_in_java_properties(self):
        windows_path = PureWindowsPath(
            "D:\\MY_GURL\\RESEARCH_PAPER\\DT-MARL-Healthcare\\python-ai\\data\\sprint11\\trials\\"
            "s11-d15e537cb76b5836\\trial-002\\reactive-no-digital-twin\\simulator_output")
        serialized = serialize_java_properties_path(windows_path)
        self.assertEqual(
            serialized,
            "D:/MY_GURL/RESEARCH_PAPER/DT-MARL-Healthcare/python-ai/data/sprint11/trials/"
            "s11-d15e537cb76b5836/trial-002/reactive-no-digital-twin/simulator_output")
        self.assertNotIn("\\", serialized)
        self.assertFalse(any(character in serialized for character in ("\r", "\n", "\t", "\f")))
        self.assertTrue(PureWindowsPath(serialized).is_absolute())

        manifest = Scenario.from_mapping(scenario_raw()).manifest_config()
        properties = _properties(manifest, windows_path)
        output_value = next(line.split("=", 1)[1] for line in properties.splitlines()
                            if line.startswith("output_directory="))
        self.assertEqual(output_value, serialized)

    def test_result_metadata_validation_and_oof_provenance_detection(self):
        result = valid_result()
        validate_result_record(result)
        misaligned = copy.deepcopy(result)
        misaligned.update({
            "risk_alignment_status": "aligned",
            "risk_artifact_id": "oof.npz",
            "risk_artifact_hash": "b" * 64,
            "risk_trace_hash": "c" * 64,
        })
        with self.assertRaisesRegex(SchemaError, "exact trace hash"):
            validate_result_record(misaligned)

    def test_checkpoint_compatibility_detection(self):
        result = valid_result()
        result.update({
            "policy_kind": "frozen_r2_mappo",
            "host_count": 5,
            "edge_node_count": 5,
            "checkpoint_id": "r2.pth",
            "checkpoint_hash": "d" * 64,
            "checkpoint_compatibility": "compatible",
        })
        with self.assertRaisesRegex(SchemaError, "frozen R2 MAPPO"):
            validate_result_record(result)

    def test_aggregation_withholds_single_trial_confidence_interval(self):
        one = summary([2.0])
        self.assertEqual(one["ci95_status"], "not_available_n_lt_2")
        self.assertIsNone(one["ci95_low"])
        two = summary([2.0, 4.0])
        self.assertEqual(two["sample_count"], 2)
        self.assertGreater(two["ci95_high"], two["mean"])

    def test_aggregate_records_uses_one_value_per_independent_trial(self):
        first = valid_result()
        second = valid_result()
        second["trial_id"] = 2
        second.update(trial_seed_pair(second["scenario_id"], 2))
        second["trace_hash"] = "e" * 64
        second["metrics"]["episode_reward"] = 3.0
        rows, priority_rows, observations = aggregate_records([first, second])
        reward = next(row for row in rows if row["metric"] == "episode_reward")
        self.assertEqual(reward["sample_count"], 2)
        self.assertEqual(reward["mean"], 1.0)
        self.assertEqual(reward["ci95_status"], "t_interval_95")
        self.assertEqual(priority_rows, [])
        self.assertEqual(observations, [])

    def test_trial_plan_blocks_incompatible_checkpoint_and_unaligned_oof(self):
        scenario = Scenario.from_mapping(scenario_raw(host_count=5, edge_node_count=5))
        manifest = scenario.manifest_config()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "model.pth"
            checkpoint.write_bytes(b"synthetic checkpoint for preflight only")
            config = root / "config.json"
            config.write_text(json.dumps({"config": {"env": {"n_edge_nodes": 10}}}),
                              encoding="utf-8")
            plan = build_trial_plan(manifest, scenario, 1, "R2", "frozen_r2_mappo",
                                    checkpoint, config)
        self.assertEqual(plan["checkpoint_compatibility"], "incompatible")
        self.assertEqual(plan["risk_artifact_status"],
                         "unavailable_until_trace_aligned_oof_regeneration")
        self.assertEqual(plan["plan_status"], "blocked")

    def test_graph_input_validation_rejects_single_trial_ci(self):
        bad = [{"mean": "0.5", "sample_count": "1", "ci95_status": "t_interval_95",
                "ci95_low": "0.4", "ci95_high": "0.6"}]
        with self.assertRaisesRegex(ValueError, "single trial"):
            validate_aggregate_rows(bad)
        valid = [{"mean": "0.5", "sample_count": "1", "ci95_status": "not_available_n_lt_2",
                  "ci95_low": "", "ci95_high": ""}]
        validate_aggregate_rows(valid)

    def test_task_count_scale_graph_specs_are_available(self):
        expected = {
            "latency_vs_task_count": {
                "metric": "avg_task_latency_s",
                "x": "task_count",
                "xlabel": "Healthcare task count",
                "ylabel": "Mean task latency (s)",
            },
            "energy_vs_task_count": {
                "metric": "physical_energy_j",
                "x": "task_count",
                "xlabel": "Healthcare task count",
                "ylabel": "Physical energy (J)",
            },
            "success_vs_task_count": {
                "metric": "task_success_rate",
                "x": "task_count",
                "xlabel": "Healthcare task count",
                "ylabel": "Task success rate (%)",
            },
        }
        for graph, specification in expected.items():
            self.assertIn(graph, GRAPH_SPECS)
            self.assertEqual(GRAPH_SPECS[graph], specification)


if __name__ == "__main__":
    unittest.main()
