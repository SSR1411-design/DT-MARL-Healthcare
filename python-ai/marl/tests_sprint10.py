"""Bounded, non-evaluation tests for Sprint 10 A2 through A5.

Run from ``python-ai`` with ``python -m unittest marl.tests_sprint10``.  These
tests load configuration, trace geometry, and the frozen checkpoint only; they
never execute an environment step, train a policy, or write any result CSV.
"""

from __future__ import annotations

import ast
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from marl.destination import CandidateView
from marl.env import DTMarlEnv
from marl.sprint9_benchmarks import load_validated_r2_config as load_frozen_r2_config
from training.sprint10_a1_no_digital_twin import (
    A1CsvBundle,
    A1Observer,
    DIRECT_DIGITAL_TWIN_OBSERVATION_INDICES,
    FIXED_HELD_OUT_STARTS as A1_FIXED_HELD_OUT_STARTS,
    OOF_PREDICTION_ARTIFACT as A1_OOF_PREDICTION_ARTIFACT,
    OOF_PREDICTION_ARTIFACT_SHA256 as A1_OOF_PREDICTION_ARTIFACT_SHA256,
    VALIDATED_CHECKPOINT as A1_VALIDATED_CHECKPOINT,
    VALIDATED_CHECKPOINT_SHA256 as A1_VALIDATED_CHECKPOINT_SHA256,
    VALIDATED_CONFIG as A1_VALIDATED_CONFIG,
    VALIDATED_CONFIG_SHA256 as A1_VALIDATED_CONFIG_SHA256,
    VerifiedMappoPolicy as A1VerifiedMappoPolicy,
    assert_a1_controls,
    build_a1_run,
    protected_a2_hashes as a1_protected_a2_hashes,
    protected_a3_hashes as a1_protected_a3_hashes,
    protected_a4_hashes as a1_protected_a4_hashes,
    protected_a5_hashes,
    protected_sprint8_sprint9_hashes as a1_protected_sprint8_sprint9_hashes,
    run_fixed_start_protocol as run_a1_fixed_start_protocol,
)
from training.sprint10_a2_no_failure_prediction import (
    A2Observer,
    FIXED_HELD_OUT_STARTS,
    VALIDATED_CHECKPOINT,
    VALIDATED_CHECKPOINT_SHA256,
    assert_a2_risk_disabled,
    build_a2_run,
    load_validated_r2_config,
    protected_output_hashes,
    risk_observation_indices,
    run_fixed_start_protocol,
    sha256,
)
from training.sprint10_a3_single_agent import (
    A3Observer,
    FIXED_HELD_OUT_STARTS as A3_FIXED_HELD_OUT_STARTS,
    SINGLE_AGENT_CHECKPOINT,
    SINGLE_AGENT_CHECKPOINT_SHA256,
    VALIDATED_R2_CHECKPOINT,
    VALIDATED_R2_CHECKPOINT_SHA256,
    assert_a3_controls,
    build_a3_run,
    protected_a2_hashes,
    protected_output_hashes as a3_protected_output_hashes,
    run_fixed_start_protocol as run_a3_fixed_start_protocol,
)
from training.sprint10_a4_no_criticality import (
    A4CsvBundle,
    A4Observer,
    FIXED_HELD_OUT_STARTS as A4_FIXED_HELD_OUT_STARTS,
    TASK_PRIORITY_OBSERVATION_INDEX,
    TASK_SEVERITY_OBSERVATION_INDEX,
    VALIDATED_CHECKPOINT as A4_VALIDATED_CHECKPOINT,
    VALIDATED_CHECKPOINT_SHA256 as A4_VALIDATED_CHECKPOINT_SHA256,
    VerifiedMappoPolicy,
    assert_a4_controls,
    build_a4_run,
    protected_a2_hashes as a4_protected_a2_hashes,
    protected_a3_hashes,
    protected_sprint8_sprint9_hashes,
    run_fixed_start_protocol as run_a4_fixed_start_protocol,
)
from training.sprint10_a5_no_uncertainty import (
    A5CsvBundle,
    A5Observer,
    FIXED_HELD_OUT_STARTS as A5_FIXED_HELD_OUT_STARTS,
    OOF_PREDICTION_ARTIFACT as A5_OOF_PREDICTION_ARTIFACT,
    OOF_PREDICTION_ARTIFACT_SHA256 as A5_OOF_PREDICTION_ARTIFACT_SHA256,
    UNCERTAINTY_A5_CHANGE,
    UNCERTAINTY_MECHANISM_FOUND,
    UNCERTAINTY_OBSERVATION_INDEX,
    VALIDATED_CHECKPOINT as A5_VALIDATED_CHECKPOINT,
    VALIDATED_CHECKPOINT_SHA256 as A5_VALIDATED_CHECKPOINT_SHA256,
    VALIDATED_CONFIG as A5_VALIDATED_CONFIG,
    VALIDATED_CONFIG_SHA256 as A5_VALIDATED_CONFIG_SHA256,
    VerifiedMappoPolicy as A5VerifiedMappoPolicy,
    assert_a5_controls,
    build_a5_run,
    protected_a2_hashes as a5_protected_a2_hashes,
    protected_a3_hashes as a5_protected_a3_hashes,
    protected_a4_hashes,
    protected_sprint8_sprint9_hashes as a5_protected_sprint8_sprint9_hashes,
    run_fixed_start_protocol as run_a5_fixed_start_protocol,
)


class Sprint10A1Tests(unittest.TestCase):
    """A1 construction/action guards; no A1 episode is evaluated or exported."""

    def setUp(self):
        self.checkpoint_before = sha256(A1_VALIDATED_CHECKPOINT)
        self.config_before = sha256(A1_VALIDATED_CONFIG)
        self.oof_before = sha256(A1_OOF_PREDICTION_ARTIFACT)
        self.sprint8_sprint9_before = a1_protected_sprint8_sprint9_hashes()
        self.a2_before = a1_protected_a2_hashes()
        self.a3_before = a1_protected_a3_hashes()
        self.a4_before = a1_protected_a4_hashes()
        self.a5_before = protected_a5_hashes()

    def tearDown(self):
        self.assertEqual(sha256(A1_VALIDATED_CHECKPOINT), self.checkpoint_before)
        self.assertEqual(sha256(A1_VALIDATED_CONFIG), self.config_before)
        self.assertEqual(sha256(A1_OOF_PREDICTION_ARTIFACT), self.oof_before)
        self.assertEqual(a1_protected_sprint8_sprint9_hashes(), self.sprint8_sprint9_before)
        self.assertEqual(a1_protected_a2_hashes(), self.a2_before)
        self.assertEqual(a1_protected_a3_hashes(), self.a3_before)
        self.assertEqual(a1_protected_a4_hashes(), self.a4_before)
        self.assertEqual(protected_a5_hashes(), self.a5_before)

    def test_a1_disables_only_direct_digital_twin_decision_context(self):
        cfg = load_frozen_r2_config(A1_VALIDATED_CONFIG)
        reference = DTMarlEnv(cfg.env, cfg.reward)
        reference_obs, reference_state, _ = reference.reset(
            episode_start_tick=A1_FIXED_HELD_OUT_STARTS[0], seed=7)
        run = build_a1_run(device="cpu")
        observations, state, _ = run.environment.reset(
            episode_start_tick=A1_FIXED_HELD_OUT_STARTS[0], seed=7)
        assert_a1_controls(run.environment, observations)

        expected_obs = reference_obs.copy()
        expected_obs[:, DIRECT_DIGITAL_TWIN_OBSERVATION_INDICES] = 0.0
        self.assertTrue(np.array_equal(observations, expected_obs))
        expected_state = reference_state.copy()
        for agent in range(run.environment.n_agents):
            offset = agent * run.environment.obs_dim
            expected_state[offset + np.asarray(DIRECT_DIGITAL_TWIN_OBSERVATION_INDICES)] = 0.0
        self.assertTrue(np.array_equal(state, expected_state))

        a1_candidate = run.environment._candidate(1)
        r2_candidate = reference._candidate(1)
        self.assertEqual(a1_candidate.risk, r2_candidate.risk)
        self.assertEqual(a1_candidate.observed_up, r2_candidate.observed_up)
        self.assertEqual(a1_candidate.free_capacity_fraction, r2_candidate.free_capacity_fraction)
        self.assertEqual(a1_candidate.load_fraction, r2_candidate.load_fraction)
        self.assertEqual(a1_candidate.link_latency_norm, 0.0)

        # Changing only direct trace telemetry in this test-local replay cannot
        # re-enter the frozen MAPPO observation.  OOF risk and runtime task
        # state are already materialized independently, so no artifact changes.
        for channel in ("cpu", "ram", "bandwidth", "energy", "runningTasks", "active",
                        "degraded", "linkUp", "linkBandwidthMbps", "linkLatencyMs",
                        "linkPacketLoss", "underAttack"):
            run.environment.trace.ch(channel).fill(99.0)
        self.assertTrue(np.array_equal(observations, run.environment._observations()))
        self.assertEqual(run.environment._candidate(1).link_latency_norm, 0.0)

    def test_a1_keeps_failure_prediction_and_destination_risk_scoring_active(self):
        run = build_a1_run(device="cpu")
        env = run.environment
        self.assertEqual(env.risk.source, "oof")
        self.assertTrue(np.any(env.risk.risk[:, env.risk.first_valid_tick:] > 0.0))
        self.assertGreater(env.cfg.dest_w_risk, 0.0)
        low = CandidateView(1, 0.0, True, 0.5, 0.25, 0.0)
        high = CandidateView(1, 1.0, True, 0.5, 0.25, 0.0)
        self.assertGreater(env.selector.score(low), env.selector.score(high))
        self.assertGreater(env.rcfg.P_risk_expose, 0.0)

    def test_a1_keeps_criticality_and_the_reserved_uncertainty_contract(self):
        cfg = load_frozen_r2_config(A1_VALIDATED_CONFIG)
        reference = DTMarlEnv(cfg.env, cfg.reward)
        reference_obs, _, _ = reference.reset(episode_start_tick=A1_FIXED_HELD_OUT_STARTS[0], seed=7)
        run = build_a1_run(device="cpu")
        observations, _, _ = run.environment.reset(episode_start_tick=A1_FIXED_HELD_OUT_STARTS[0], seed=7)
        self.assertGreater(run.environment.rcfg.w_criticality, 0.0)
        self.assertGreater(run.environment.rcfg.w_criticality_migration, 0.0)
        events = [run.environment._blank_event() for _ in range(run.environment.n_agents)]
        events[0].update(completed=1, severity=0.0)
        plain = run.environment._rewards(events)
        events[0]["severity"] = 10.0
        self.assertFalse(np.array_equal(plain, run.environment._rewards(events)))
        self.assertTrue(np.array_equal(run.environment.risk.uncertainty, reference.risk.uncertainty))
        self.assertTrue(np.array_equal(observations[:, 13], reference_obs[:, 13]))
        self.assertTrue(np.array_equal(observations[:, 13], np.zeros(run.environment.n_agents)))

    def test_a1_uses_frozen_greedy_mappo_with_checkpoint_compatible_geometry(self):
        run = build_a1_run(device="cpu")
        observations, state, masks = run.environment.reset(
            episode_start_tick=run.starts[0], seed=7)
        self.assertEqual(run.starts, A1_FIXED_HELD_OUT_STARTS)
        self.assertEqual(run.policy.agent.obs_dim, run.environment.obs_dim)
        self.assertEqual(run.policy.agent.state_dim, run.environment.state_dim)
        self.assertEqual(state.shape, (run.environment.state_dim,))
        self.assertIsInstance(run.policy, A1VerifiedMappoPolicy)
        with mock.patch.object(run.policy.agent, "act_greedy", wraps=run.policy.agent.act_greedy) as act:
            actions = run.policy.act(run.environment, observations, masks)
        self.assertEqual(actions.shape, (run.environment.n_agents,))
        self.assertTrue(np.all(masks[np.arange(run.environment.n_agents), actions]))
        self.assertEqual(act.call_count, 1)
        self.assertEqual(run.policy.action_selection_calls, 1)

    def test_a1_fixed_start_protocol_consumes_each_reset_once_in_exact_order(self):
        class MemoryBundle:
            def __init__(self):
                self.rows = {}

            def write(self, key, row):
                self.rows.setdefault(key, []).append(dict(row))

        run = build_a1_run(device="cpu")
        bundle = MemoryBundle()
        observer = A1Observer(bundle, run, "bounded-a1-start-regression")
        actual_environment_starts = []

        def bounded_run_episode(env, policy, start_tick, seed, observer):
            env.reset(episode_start_tick=start_tick, seed=seed)
            actual_environment_starts.append(env.t0)
            observer.on_episode_start(env, start_tick, seed, policy.name)
            policy.action_selection_calls += 1

        with mock.patch("training.sprint10_a1_no_digital_twin.run_episode",
                        side_effect=bounded_run_episode):
            run_a1_fixed_start_protocol(run, observer)

        self.assertEqual(actual_environment_starts, list(A1_FIXED_HELD_OUT_STARTS))
        self.assertEqual(observer.consumed_starts, list(A1_FIXED_HELD_OUT_STARTS))
        self.assertEqual(
            [row["episode_start_tick"] for row in bundle.rows["metadata"]],
            list(A1_FIXED_HELD_OUT_STARTS))
        self.assertEqual(
            [row["episode_start_tick"] for row in bundle.rows["prediction_usage"]],
            list(A1_FIXED_HELD_OUT_STARTS))

    def test_a1_runner_is_inference_only_and_refuses_existing_evidence_directory(self):
        source = (Path(__file__).resolve().parents[1] / "training" /
                  "sprint10_a1_no_digital_twin.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        forbidden_methods = {"save", "train_mode", "backward"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            elif isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, forbidden_methods)
                if node.attr == "update" and isinstance(node.value, ast.Name):
                    self.assertNotIn(node.value.id, {"agent", "policy", "run"})
                if isinstance(node.value, ast.Name) and node.value.id == "np":
                    self.assertNotEqual(node.attr, "random")
        self.assertFalse(any(name == "random" or name.startswith("random.") for name in imports))
        self.assertIn("marl.mappo", imports)
        self.assertNotIn("--overwrite", source)
        self.assertEqual(sha256(A1_VALIDATED_CHECKPOINT), A1_VALIDATED_CHECKPOINT_SHA256)
        self.assertEqual(sha256(A1_VALIDATED_CONFIG), A1_VALIDATED_CONFIG_SHA256)
        self.assertEqual(sha256(A1_OOF_PREDICTION_ARTIFACT), A1_OOF_PREDICTION_ARTIFACT_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                A1CsvBundle(Path(directory))


class Sprint10A2Tests(unittest.TestCase):
    def setUp(self):
        self.checkpoint_before = sha256(VALIDATED_CHECKPOINT)
        self.outputs_before = protected_output_hashes()

    def tearDown(self):
        self.assertEqual(sha256(VALIDATED_CHECKPOINT), self.checkpoint_before)
        self.assertEqual(protected_output_hashes(), self.outputs_before)

    def test_a2_config_disables_all_risk_paths(self):
        cfg = load_validated_r2_config()
        self.assertEqual(cfg.env.risk_source, "zero")
        self.assertEqual(cfg.env.dest_w_risk, 0.0)
        self.assertEqual(cfg.reward.P_risk_expose, 0.0)

    def test_a2_constructs_without_loading_oof_artifact(self):
        with mock.patch("marl.risk_provider.np.load", side_effect=AssertionError(
                "A2 attempted to load the OOF prediction artifact")):
            run = build_a2_run(device="cpu")
            observations, _, masks = run.environment.reset(
                episode_start_tick=run.starts[0], seed=7)
            run.policy.act(run.environment, observations, masks)
        assert_a2_risk_disabled(run.environment)
        self.assertEqual(run.environment.risk.source, "zero")

    def test_policy_observation_has_zero_local_and_neighbour_risk(self):
        cfg = load_validated_r2_config()
        env = DTMarlEnv(cfg.env, cfg.reward)
        observations, _, masks = env.reset(episode_start_tick=env._min_start, seed=7)
        indices = risk_observation_indices(env)
        self.assertTrue(np.array_equal(
            observations[:, indices], np.zeros((env.n_agents, len(indices)))))

        # The frozen MAPPO adapter receives exactly the same zero-risk tensor.
        run = build_a2_run(device="cpu")
        run.policy.act(env, observations, masks)
        self.assertTrue(np.array_equal(
            observations[:, indices], np.zeros((env.n_agents, len(indices)))))

    def test_destination_score_is_invariant_to_risk(self):
        cfg = load_validated_r2_config()
        env = DTMarlEnv(cfg.env, cfg.reward)
        low = CandidateView(1, 0.0, True, 0.5, 0.25, 0.1)
        high = CandidateView(1, 1.0, True, 0.5, 0.25, 0.1)
        self.assertEqual(env.selector.score(low), env.selector.score(high))

    def test_reward_does_not_query_risk_when_exposure_is_disabled(self):
        cfg = load_validated_r2_config()
        env = DTMarlEnv(cfg.env, cfg.reward)
        env.reset(episode_start_tick=env._min_start, seed=7)
        events = [env._blank_event() for _ in range(env.n_agents)]
        events[0]["stayed_resident"] = True

        def unexpected_risk_query(*args, **kwargs):
            raise AssertionError("A2 reward queried risk despite a zero exposure penalty")

        env.risk_at = unexpected_risk_query
        env._rewards(events)

    def test_frozen_checkpoint_and_fixed_held_out_protocol(self):
        run = build_a2_run(device="cpu")
        self.assertEqual(run.checkpoint, VALIDATED_CHECKPOINT)
        self.assertEqual(run.checkpoint_sha256, VALIDATED_CHECKPOINT_SHA256)
        self.assertEqual(run.starts, FIXED_HELD_OUT_STARTS)

    def test_fixed_start_protocol_consumes_each_environment_reset_in_order(self):
        """Regression for a validator reset that formerly reused tick 491."""
        class MemoryBundle:
            def __init__(self):
                self.rows = {}

            def write(self, key, row):
                self.rows.setdefault(key, []).append(dict(row))

        run = build_a2_run(device="cpu")
        bundle = MemoryBundle()
        observer = A2Observer(bundle, run, "bounded-start-regression")
        actual_environment_starts = []

        # Substitute only the costly step loop.  This preserves the production
        # reset -> observer-start boundary where the original bug occurred.
        def bounded_run_episode(env, policy, start_tick, seed, observer):
            env.reset(episode_start_tick=start_tick, seed=seed)
            actual_environment_starts.append(env.t0)
            observer.on_episode_start(env, start_tick, seed, policy.name)

        with mock.patch(
                "training.sprint10_a2_no_failure_prediction.run_episode",
                side_effect=bounded_run_episode):
            run_fixed_start_protocol(run, observer)

        self.assertEqual(actual_environment_starts, list(FIXED_HELD_OUT_STARTS))
        self.assertEqual(observer.consumed_starts, list(FIXED_HELD_OUT_STARTS))
        self.assertEqual(
            [row["episode_start_tick"] for row in bundle.rows["metadata"]],
            list(FIXED_HELD_OUT_STARTS))
        self.assertEqual(
            [row["episode_start_tick"] for row in bundle.rows["prediction_usage"]],
            list(FIXED_HELD_OUT_STARTS))

    def test_runner_has_no_random_metric_generation_path(self):
        source_path = Path(__file__).resolve().parents[1] / "training" / \
            "sprint10_a2_no_failure_prediction.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                forbidden.extend(alias.name for alias in node.names
                                 if alias.name == "random" or alias.name.startswith("random."))
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module == "random" or node.module.startswith("random."):
                    forbidden.append(node.module)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "np" and node.attr == "random":
                    forbidden.append("np.random")
        self.assertEqual(forbidden, [])


class Sprint10A3Tests(unittest.TestCase):
    """A3 guards construct and select a controller, but never step or export it."""

    def setUp(self):
        self.r2_checkpoint_before = sha256(VALIDATED_R2_CHECKPOINT)
        self.single_agent_checkpoint_before = sha256(SINGLE_AGENT_CHECKPOINT)
        self.sprint8_sprint9_before = a3_protected_output_hashes()
        self.a2_before = protected_a2_hashes()

    def tearDown(self):
        self.assertEqual(sha256(VALIDATED_R2_CHECKPOINT), self.r2_checkpoint_before)
        self.assertEqual(sha256(SINGLE_AGENT_CHECKPOINT), self.single_agent_checkpoint_before)
        self.assertEqual(a3_protected_output_hashes(), self.sprint8_sprint9_before)
        self.assertEqual(protected_a2_hashes(), self.a2_before)

    def test_a3_reconstructs_r2_controls_and_fixed_eight_starts(self):
        run = build_a3_run(device="cpu")
        observations, _, _ = run.environment.reset(
            episode_start_tick=run.starts[0], seed=7)
        assert_a3_controls(run.environment, observations)
        self.assertEqual(run.starts, A3_FIXED_HELD_OUT_STARTS)
        self.assertEqual(run.environment.risk.source, "oof")
        self.assertGreater(run.environment.cfg.dest_w_risk, 0.0)
        self.assertGreater(run.environment.rcfg.P_risk_expose, 0.0)
        self.assertGreater(run.environment.rcfg.w_criticality, 0.0)
        self.assertGreater(run.environment.rcfg.w_criticality_migration, 0.0)
        self.assertTrue(np.array_equal(
            run.environment.risk.uncertainty,
            np.zeros_like(run.environment.risk.uncertainty)))

    def test_a3_policy_uses_single_agent_ppo_and_never_calls_mappo_adapter(self):
        from marl.mappo import MappoPolicy

        run = build_a3_run(device="cpu")
        observations, _, masks = run.environment.reset(
            episode_start_tick=run.starts[0], seed=7)
        with mock.patch.object(MappoPolicy, "act", side_effect=AssertionError(
                "A3 attempted to invoke MAPPO action selection")):
            actions = run.policy.act(run.environment, observations, masks)
        self.assertEqual(actions.shape, (run.environment.n_agents,))
        self.assertTrue(np.all(masks[np.arange(run.environment.n_agents), actions]))
        self.assertEqual(run.policy.action_selection_calls, 1)
        self.assertEqual(run.metadata.checkpoint_sha256, SINGLE_AGENT_CHECKPOINT_SHA256)

    def test_a3_fixed_start_protocol_consumes_each_reset_in_order(self):
        class MemoryBundle:
            def __init__(self):
                self.rows = {}

            def write(self, key, row):
                self.rows.setdefault(key, []).append(dict(row))

        run = build_a3_run(device="cpu")
        bundle = MemoryBundle()
        observer = A3Observer(bundle, run, "bounded-a3-start-regression")
        actual_environment_starts = []

        def bounded_run_episode(env, policy, start_tick, seed, observer):
            env.reset(episode_start_tick=start_tick, seed=seed)
            actual_environment_starts.append(env.t0)
            observer.on_episode_start(env, start_tick, seed, policy.name)
            policy.action_selection_calls += 1

        with mock.patch(
                "training.sprint10_a3_single_agent.run_episode",
                side_effect=bounded_run_episode):
            run_a3_fixed_start_protocol(run, observer)

        self.assertEqual(actual_environment_starts, list(A3_FIXED_HELD_OUT_STARTS))
        self.assertEqual(observer.consumed_starts, list(A3_FIXED_HELD_OUT_STARTS))
        self.assertEqual(
            [row["episode_start_tick"] for row in bundle.rows["metadata"]],
            list(A3_FIXED_HELD_OUT_STARTS))
        self.assertEqual(
            [row["episode_start_tick"] for row in bundle.rows["prediction_usage"]],
            list(A3_FIXED_HELD_OUT_STARTS))

    def test_a3_source_is_inference_only_and_has_no_mappo_import(self):
        source = (Path(__file__).resolve().parents[1] / "training" /
                  "sprint10_a3_single_agent.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        forbidden_methods = {"save", "train_mode", "backward"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            elif isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, forbidden_methods)
                if node.attr == "update" and isinstance(node.value, ast.Name):
                    self.assertNotIn(node.value.id, {"agent", "policy", "run"})
                if isinstance(node.value, ast.Name) and node.value.id == "np":
                    self.assertNotEqual(node.attr, "random")
        self.assertFalse(any(name == "random" or name.startswith("marl.mappo")
                             for name in imports))
        self.assertEqual(sha256(VALIDATED_R2_CHECKPOINT), VALIDATED_R2_CHECKPOINT_SHA256)


class Sprint10A4Tests(unittest.TestCase):
    """A4 construction/action checks only; no environment step or export occurs."""

    def setUp(self):
        self.checkpoint_before = sha256(A4_VALIDATED_CHECKPOINT)
        self.sprint8_sprint9_before = protected_sprint8_sprint9_hashes()
        self.a2_before = a4_protected_a2_hashes()
        self.a3_before = protected_a3_hashes()

    def tearDown(self):
        self.assertEqual(sha256(A4_VALIDATED_CHECKPOINT), self.checkpoint_before)
        self.assertEqual(protected_sprint8_sprint9_hashes(), self.sprint8_sprint9_before)
        self.assertEqual(a4_protected_a2_hashes(), self.a2_before)
        self.assertEqual(protected_a3_hashes(), self.a3_before)

    def test_a4_disables_every_clinical_criticality_reward_and_policy_input(self):
        run = build_a4_run(device="cpu")
        observations, _, _ = run.environment.reset(
            episode_start_tick=run.starts[0], seed=7)
        assert_a4_controls(run.environment, observations)
        self.assertEqual(run.environment.rcfg.w_criticality, 0.0)
        self.assertEqual(run.environment.rcfg.w_criticality_migration, 0.0)
        self.assertTrue(np.array_equal(
            observations[:, (TASK_SEVERITY_OBSERVATION_INDEX, TASK_PRIORITY_OBSERVATION_INDEX)],
            np.zeros((run.environment.n_agents, 2))))

        events = [run.environment._blank_event() for _ in range(run.environment.n_agents)]
        events[0].update(stayed_resident=True, severity=0.0, migration_severity=0.0)
        reference_reward = run.environment._rewards(events)
        events[0].update(severity=1_000_000.0, migration_severity=1_000_000.0)
        self.assertTrue(np.array_equal(reference_reward, run.environment._rewards(events)))

    def test_a4_scheduler_never_calls_priority_order_and_observations_are_invariant(self):
        run = build_a4_run(device="cpu")
        env = run.environment
        with mock.patch("marl.env.priority_order_key", side_effect=AssertionError(
                "A4 scheduling consulted clinical priority")):
            observations, _, _ = env.reset(episode_start_tick=run.starts[0], seed=7)
        focus_before = list(env._focus)
        for task in env.tasks:
            task.spec.severity += 1000.0
        env._refresh_derived()
        observations_after = env._observations()
        self.assertEqual(env._focus, focus_before)
        self.assertTrue(np.array_equal(
            observations[:, (TASK_SEVERITY_OBSERVATION_INDEX, TASK_PRIORITY_OBSERVATION_INDEX)],
            observations_after[:, (TASK_SEVERITY_OBSERVATION_INDEX, TASK_PRIORITY_OBSERVATION_INDEX)]))

    def test_a4_retains_digital_twin_prediction_and_frozen_greedy_mappo(self):
        run = build_a4_run(device="cpu")
        observations, _, masks = run.environment.reset(
            episode_start_tick=run.starts[0], seed=7)
        self.assertEqual(run.environment.trace.n_nodes, run.environment.n_agents)
        self.assertEqual(run.environment.risk.source, "oof")
        self.assertGreater(run.environment.cfg.dest_w_risk, 0.0)
        self.assertGreater(run.environment.rcfg.P_risk_expose, 0.0)
        self.assertIsInstance(run.policy, VerifiedMappoPolicy)
        with mock.patch.object(run.policy.agent, "act_greedy", wraps=run.policy.agent.act_greedy) as act:
            actions = run.policy.act(run.environment, observations, masks)
        self.assertEqual(actions.shape, (run.environment.n_agents,))
        self.assertTrue(np.all(masks[np.arange(run.environment.n_agents), actions]))
        self.assertEqual(act.call_count, 1)
        self.assertEqual(run.policy.action_selection_calls, 1)

    def test_a4_fixed_start_protocol_consumes_each_reset_exactly_once_in_order(self):
        class MemoryBundle:
            def __init__(self):
                self.rows = {}

            def write(self, key, row):
                self.rows.setdefault(key, []).append(dict(row))

        run = build_a4_run(device="cpu")
        bundle = MemoryBundle()
        observer = A4Observer(bundle, run, "bounded-a4-start-regression")
        actual_environment_starts = []

        def bounded_run_episode(env, policy, start_tick, seed, observer):
            env.reset(episode_start_tick=start_tick, seed=seed)
            actual_environment_starts.append(env.t0)
            observer.on_episode_start(env, start_tick, seed, policy.name)
            policy.action_selection_calls += 1

        with mock.patch("training.sprint10_a4_no_criticality.run_episode",
                        side_effect=bounded_run_episode):
            run_a4_fixed_start_protocol(run, observer)

        self.assertEqual(actual_environment_starts, list(A4_FIXED_HELD_OUT_STARTS))
        self.assertEqual(observer.consumed_starts, list(A4_FIXED_HELD_OUT_STARTS))
        self.assertEqual(
            [row["episode_start_tick"] for row in bundle.rows["metadata"]],
            list(A4_FIXED_HELD_OUT_STARTS))
        self.assertEqual(
            [row["episode_start_tick"] for row in bundle.rows["prediction_usage"]],
            list(A4_FIXED_HELD_OUT_STARTS))

    def test_a4_runner_is_inference_only_and_has_no_overwrite_option(self):
        source = (Path(__file__).resolve().parents[1] / "training" /
                  "sprint10_a4_no_criticality.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = []
        forbidden_methods = {"save", "train_mode", "backward"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            elif isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, forbidden_methods)
                if node.attr == "update" and isinstance(node.value, ast.Name):
                    self.assertNotIn(node.value.id, {"agent", "policy", "run"})
                if isinstance(node.value, ast.Name) and node.value.id == "np":
                    self.assertNotEqual(node.attr, "random")
        self.assertFalse(any(name == "random" or name.startswith("random.") for name in imports))
        self.assertIn("marl.mappo", imports)
        self.assertNotIn("--overwrite", source)
        self.assertEqual(sha256(A4_VALIDATED_CHECKPOINT), A4_VALIDATED_CHECKPOINT_SHA256)

    def test_a4_refuses_an_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                A4CsvBundle(Path(directory))


class Sprint10A5Tests(unittest.TestCase):
    """A5 construction/action checks only; no A5 episode is evaluated or exported."""

    def setUp(self):
        self.checkpoint_before = sha256(A5_VALIDATED_CHECKPOINT)
        self.config_before = sha256(A5_VALIDATED_CONFIG)
        self.oof_before = sha256(A5_OOF_PREDICTION_ARTIFACT)
        self.sprint8_sprint9_before = a5_protected_sprint8_sprint9_hashes()
        self.a2_before = a5_protected_a2_hashes()
        self.a3_before = a5_protected_a3_hashes()
        self.a4_before = protected_a4_hashes()

    def tearDown(self):
        self.assertEqual(sha256(A5_VALIDATED_CHECKPOINT), self.checkpoint_before)
        self.assertEqual(sha256(A5_VALIDATED_CONFIG), self.config_before)
        self.assertEqual(sha256(A5_OOF_PREDICTION_ARTIFACT), self.oof_before)
        self.assertEqual(a5_protected_sprint8_sprint9_hashes(), self.sprint8_sprint9_before)
        self.assertEqual(a5_protected_a2_hashes(), self.a2_before)
        self.assertEqual(a5_protected_a3_hashes(), self.a3_before)
        self.assertEqual(protected_a4_hashes(), self.a4_before)

    def test_a5_removes_only_the_inactive_reserved_uncertainty_interface(self):
        run = build_a5_run(device="cpu")
        observations, _, _ = run.environment.reset(
            episode_start_tick=run.starts[0], seed=7)
        assert_a5_controls(run.environment, observations)
        self.assertEqual(run.environment.obs_dim, run.policy.agent.obs_dim)
        self.assertTrue(np.array_equal(
            observations[:, UNCERTAINTY_OBSERVATION_INDEX],
            np.zeros(run.environment.n_agents)))
        self.assertTrue(np.array_equal(
            run.environment.risk.uncertainty,
            np.zeros_like(run.environment.risk.uncertainty)))

        # The A5 interface does not consult the provider: even a deliberately
        # non-R2 in-memory value cannot enter the checkpoint input.  This does
        # not mutate any source artifact and the run object is test-local.
        run.environment.risk.uncertainty.fill(0.75)
        observations_after = run.environment._observations()
        self.assertEqual(run.environment.uncertainty_at(0), 0.0)
        self.assertTrue(np.array_equal(
            observations_after[:, UNCERTAINTY_OBSERVATION_INDEX],
            np.zeros(run.environment.n_agents)))

    def test_a5_is_observation_mask_and_greedy_action_equivalent_to_r2(self):
        """An inactive zero slot has no behavioral counterfactual to fabricate."""
        cfg = load_frozen_r2_config(A5_VALIDATED_CONFIG)
        reference = DTMarlEnv(cfg.env, cfg.reward)
        reference_obs, reference_state, reference_masks = reference.reset(
            episode_start_tick=A5_FIXED_HELD_OUT_STARTS[0], seed=7)
        run = build_a5_run(device="cpu")
        a5_obs, a5_state, a5_masks = run.environment.reset(
            episode_start_tick=A5_FIXED_HELD_OUT_STARTS[0], seed=7)
        self.assertTrue(np.array_equal(a5_obs, reference_obs))
        self.assertTrue(np.array_equal(a5_state, reference_state))
        self.assertTrue(np.array_equal(a5_masks, reference_masks))
        a5_actions = run.policy.act(run.environment, a5_obs, a5_masks)
        r2_actions = run.policy.act(reference, reference_obs, reference_masks)
        self.assertTrue(np.array_equal(a5_actions, r2_actions))

    def test_a5_retains_digital_twin_prediction_criticality_and_frozen_mappo(self):
        run = build_a5_run(device="cpu")
        observations, _, masks = run.environment.reset(
            episode_start_tick=run.starts[0], seed=7)
        self.assertEqual(run.environment.trace.n_nodes, run.environment.n_agents)
        self.assertEqual(run.environment.risk.source, "oof")
        self.assertGreater(run.environment.cfg.dest_w_risk, 0.0)
        self.assertGreater(run.environment.rcfg.P_risk_expose, 0.0)
        self.assertGreater(run.environment.rcfg.w_criticality, 0.0)
        self.assertGreater(run.environment.rcfg.w_criticality_migration, 0.0)
        events = [run.environment._blank_event() for _ in range(run.environment.n_agents)]
        events[0].update(completed=1, severity=0.0)
        unweighted = run.environment._rewards(events)
        events[0]["severity"] = 10.0
        self.assertFalse(np.array_equal(unweighted, run.environment._rewards(events)))
        self.assertIsInstance(run.policy, A5VerifiedMappoPolicy)
        with mock.patch.object(run.policy.agent, "act_greedy", wraps=run.policy.agent.act_greedy) as act:
            actions = run.policy.act(run.environment, observations, masks)
        self.assertEqual(actions.shape, (run.environment.n_agents,))
        self.assertTrue(np.all(masks[np.arange(run.environment.n_agents), actions]))
        self.assertEqual(act.call_count, 1)
        self.assertEqual(run.policy.action_selection_calls, 1)

    def test_a5_fixed_start_protocol_consumes_each_reset_exactly_once_in_order(self):
        class MemoryBundle:
            def __init__(self):
                self.rows = {}

            def write(self, key, row):
                self.rows.setdefault(key, []).append(dict(row))

        run = build_a5_run(device="cpu")
        bundle = MemoryBundle()
        observer = A5Observer(bundle, run, "bounded-a5-start-regression")
        actual_environment_starts = []

        def bounded_run_episode(env, policy, start_tick, seed, observer):
            env.reset(episode_start_tick=start_tick, seed=seed)
            actual_environment_starts.append(env.t0)
            observer.on_episode_start(env, start_tick, seed, policy.name)
            policy.action_selection_calls += 1

        with mock.patch("training.sprint10_a5_no_uncertainty.run_episode",
                        side_effect=bounded_run_episode):
            run_a5_fixed_start_protocol(run, observer)

        self.assertEqual(actual_environment_starts, list(A5_FIXED_HELD_OUT_STARTS))
        self.assertEqual(observer.consumed_starts, list(A5_FIXED_HELD_OUT_STARTS))
        self.assertEqual(
            [row["episode_start_tick"] for row in bundle.rows["metadata"]],
            list(A5_FIXED_HELD_OUT_STARTS))
        self.assertEqual(
            [row["episode_start_tick"] for row in bundle.rows["prediction_usage"]],
            list(A5_FIXED_HELD_OUT_STARTS))

    def test_a5_source_is_inference_only_without_a_fake_uncertainty_heuristic_or_overwrite(self):
        source = (Path(__file__).resolve().parents[1] / "training" /
                  "sprint10_a5_no_uncertainty.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        class_node = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef)
                          and node.name == "NoUncertaintyGatingDTMarlEnv")
        method = next(node for node in class_node.body
                      if isinstance(node, ast.FunctionDef) and node.name == "uncertainty_at")
        self.assertEqual(len(method.body), 1)
        self.assertIsInstance(method.body[0], ast.Return)
        self.assertIsInstance(method.body[0].value, ast.Constant)
        self.assertEqual(method.body[0].value.value, 0.0)
        forbidden_methods = {"save", "train_mode", "backward"}
        forbidden_names = {"uncertainty_threshold", "uncertainty_gate", "uncertainty_weight"}
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
            elif isinstance(node, ast.Attribute):
                self.assertNotIn(node.attr, forbidden_methods)
                if node.attr == "update" and isinstance(node.value, ast.Name):
                    self.assertNotIn(node.value.id, {"agent", "policy", "run"})
                if isinstance(node.value, ast.Name) and node.value.id == "np":
                    self.assertNotEqual(node.attr, "random")
            elif isinstance(node, ast.Name):
                self.assertNotIn(node.id, forbidden_names)
        self.assertFalse(any(name == "random" or name.startswith("random.") for name in imports))
        self.assertIn("marl.mappo", imports)
        self.assertNotIn("--overwrite", source)
        self.assertEqual(UNCERTAINTY_MECHANISM_FOUND.count("no reward"), 1)
        self.assertIn("without reading the provider", UNCERTAINTY_A5_CHANGE)
        self.assertEqual(sha256(A5_VALIDATED_CHECKPOINT), A5_VALIDATED_CHECKPOINT_SHA256)
        self.assertEqual(sha256(A5_VALIDATED_CONFIG), A5_VALIDATED_CONFIG_SHA256)
        self.assertEqual(sha256(A5_OOF_PREDICTION_ARTIFACT), A5_OOF_PREDICTION_ARTIFACT_SHA256)

    def test_a5_refuses_an_existing_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                A5CsvBundle(Path(directory))


if __name__ == "__main__":
    unittest.main()
