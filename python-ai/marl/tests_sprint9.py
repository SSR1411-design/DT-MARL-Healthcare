"""Deterministic, no-training regression tests for Sprint 9 baselines.

Run from the repository root:
    python python-ai/marl/tests_sprint9.py
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.single_agent_ppo import SingleAgentPPO
from marl.sprint9_benchmarks import (
    BASELINE_DETECTION,
    BASELINE_REACTIVE,
    build_baseline,
    deterministic_starts,
)
from training.validate_sprint9_baselines import (
    SINGLE_AGENT_CHECKPOINT,
    exercise_detection_only_recovery,
    validate_no_fabricated_metric_source,
    validate_non_predictive_construction,
    validate_protected_artifacts,
    validate_single_agent_checkpoint_and_config,
    validate_single_agent_evaluation_source,
    validate_single_agent_export_bundle,
)


class Sprint9BaselineTests(unittest.TestCase):
    def test_protected_artifacts_and_zero_risk_construction(self):
        validate_protected_artifacts()
        reactive, detection = validate_non_predictive_construction()
        self.assertEqual(reactive.environment.risk.source, "zero")
        self.assertEqual(detection.environment.risk.source, "zero")
        self.assertFalse(reactive.metadata.uses_prediction)
        self.assertFalse(detection.metadata.uses_uncertainty)

    def test_identical_held_out_starts(self):
        reactive = build_baseline(BASELINE_REACTIVE, seed=101)
        detection = build_baseline(BASELINE_DETECTION, seed=101)
        self.assertEqual(deterministic_starts(reactive, 3),
                         deterministic_starts(detection, 3))
        self.assertGreaterEqual(deterministic_starts(reactive, 1)[0],
                                reactive.environment._min_start)

    def test_detection_recovery_is_post_failure_and_healthy(self):
        record = exercise_detection_only_recovery()
        self.assertTrue(record.succeeded)
        self.assertEqual(record.detection_tick, record.recovery_tick)
        self.assertEqual(record.recovery_latency_s, 0.0)

    def test_single_agent_ppo_is_factored_and_legal(self):
        run = build_baseline(BASELINE_REACTIVE)
        _, state, masks = run.environment.reset(
            episode_start_tick=run.environment._min_start, seed=17)
        agent = SingleAgentPPO(run.environment.state_dim, run.environment.n_agents, seed=17)
        action, logp = agent.act(state, masks)
        greedy = agent.act_greedy(state, masks)
        self.assertEqual(action.shape, (run.environment.n_agents,))
        self.assertEqual(greedy.shape, (run.environment.n_agents,))
        self.assertTrue(np.isfinite(logp))
        self.assertTrue(np.all(masks[np.arange(run.environment.n_agents), action]))
        self.assertEqual(agent.n_actions, 4)
        self.assertEqual(agent.n_nodes, 10)

    def test_single_agent_checkpoint_round_trip_is_own_format(self):
        run = build_baseline(BASELINE_REACTIVE)
        _, state, masks = run.environment.reset(
            episode_start_tick=run.environment._min_start, seed=23)
        agent = SingleAgentPPO(run.environment.state_dim, run.environment.n_agents, seed=23)
        expected = agent.act_greedy(state, masks)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "single_agent_test.pth"
            agent.save(path, extra={"test_only": True})
            restored = SingleAgentPPO.load(path)
            np.testing.assert_array_equal(restored.act_greedy(state, masks), expected)
            self.assertEqual(restored.state_dim, run.environment.state_dim)
            self.assertEqual(restored.n_nodes, run.environment.n_agents)

    def test_metric_exporter_has_no_random_metric_source(self):
        validate_no_fabricated_metric_source()

    def test_single_agent_checkpoint_and_evaluator_are_independent(self):
        validate_single_agent_checkpoint_and_config()
        validate_single_agent_evaluation_source()

    def test_final_single_agent_bundle_has_common_held_out_starts(self):
        bundle = ROOT / "data" / "sprint9" / "final" / "single-agent-ppo"
        metadata = validate_single_agent_export_bundle(bundle)
        self.assertEqual(len(metadata), 20)
        self.assertTrue(all(row["checkpoint"] for row in metadata))
        self.assertTrue(SINGLE_AGENT_CHECKPOINT.is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
