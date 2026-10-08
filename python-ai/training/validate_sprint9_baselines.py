"""Safety and provenance validation for Sprint 9 baseline implementation.

The validator is intentionally read-only.  It checks the protected MAPPO and
Sprint 8 artifacts against their pre-Sprint-9 SHA-256 values, proves the two
heuristics construct zero-risk environments, exercises one observed-failure
recovery without a predictor, and optionally validates exported CSV bundles.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import inspect
import math
import sys
import json
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.baseline import DetectionRecoveryPolicy, ReactiveThresholdPolicy
from marl.env import PENDING
from marl.single_agent_ppo import SingleAgentPPO
from marl.sprint9_benchmarks import (
    BASELINE_DETECTION,
    BASELINE_REACTIVE,
    BASELINE_SINGLE_AGENT,
    build_baseline,
    deterministic_starts,
)
from training.sprint9_metrics import FILE_SPECS


MAPPO_CHECKPOINT = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_best.pth"
SINGLE_AGENT_CHECKPOINT = (
    ROOT / "saved_models" / "single_agent" / "sprint9" / "sprint9_single_agent_best.pth")
SINGLE_AGENT_CONFIG = (
    ROOT / "saved_models" / "single_agent" / "sprint9" / "sprint9_single_agent_config.json")
MAPPO_SHA256 = "F7B06FF8AE3D9C3288004101A8329F203D09FC5E9A82D88957DAF696C24A8BEE"
SPRINT8_SHA256 = {
    "sprint8_episode_metrics.csv": "D66CF8807A1868A72A4CCA59748931271AC5F8AE5F13754D3527B9BC5D967B96",
    "sprint8_migrations.csv": "23CF25050D11EA7264DA2582CC0AF005E32F4D508FD88D67507D69AE6454D951",
    "sprint8_node_ticks.csv": "0C3E2AA4ABFE6E9FA21E7ADC29418886A296ABB8DBA0358E0D99C161BD5D1D18",
    "sprint8_prediction_metrics.csv": "6002DF82AEA246D6B97CA298B491D34F59EB1836406AFED6BB9539617CEC8953",
    "sprint8_prediction_windows.csv": "8918D999C42E8D98DF66A82C15088489985A66F2E44635EEB0A70146C5FFA33E",
    "sprint8_recovery.csv": "793608CE7DF3AD5F8FFACC6C0DC6E1BB2348A4D1828879D7320930270055A833",
    "sprint8_step_rewards.csv": "18D62065C72E595DFF96DF1BCE418EFFC003A1AADDE446D3B0C824262EF7EC6F",
    "sprint8_task_events.csv": "B9699C38BD013C896168AD8FB3C92E25B8085F7A7FDE4B657D1D5DD493229CB5",
    "sprint8_task_metadata.csv": "AEF99FFF1D23B468050195CE89C316CD43FD1B62820BF7CF52AF5247F7FD6144",
    "sprint8_training_loss.csv": "4F4CF63A2D932400D1ED8BDD2CD32C6462D1B2E13CFDD119923D08F0ED67D179",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def validate_protected_artifacts() -> None:
    if sha256(MAPPO_CHECKPOINT) != MAPPO_SHA256:
        raise AssertionError("validated MAPPO checkpoint content changed")
    for filename, expected in SPRINT8_SHA256.items():
        path = ROOT / "data" / filename
        if not path.exists():
            raise AssertionError(f"validated Sprint 8 output is missing: {path}")
        if sha256(path) != expected:
            raise AssertionError(f"validated Sprint 8 output changed: {path.name}")


def validate_non_predictive_construction() -> tuple:
    """Zero risk must be an enforced construction property, not a label."""
    # The zero provider exits before OOF's np.load.  Turn any accidental load
    # into an error to prove the non-predictive paths do not consume it.
    with mock.patch("marl.risk_provider.np.load", side_effect=AssertionError(
            "a zero-risk baseline attempted to load a prediction artifact")):
        reactive = build_baseline(BASELINE_REACTIVE)
        detection = build_baseline(BASELINE_DETECTION)
    for run in (reactive, detection):
        assert run.environment.risk.source == "zero"
        assert run.environment.cfg.dest_w_risk == 0.0
        assert run.environment.rcfg.P_risk_expose == 0.0
        assert not run.metadata.uses_prediction
        assert not run.metadata.uses_uncertainty
    assert detection.environment.detection_recovery
    assert deterministic_starts(reactive, 3) == deterministic_starts(detection, 3)

    # Policy source must not call the predictor. Environment observations still
    # contain the zero-valued risk channel for fixed geometry compatibility.
    assert "risk_at(" not in inspect.getsource(ReactiveThresholdPolicy.act)
    assert "risk_at(" not in inspect.getsource(DetectionRecoveryPolicy.act)
    return reactive, detection


def exercise_detection_only_recovery():
    """Run exactly one forced real trace outage transition; no prediction is used."""
    run = build_baseline(BASELINE_DETECTION)
    env = run.environment
    selected = None
    for start in range(env._min_start, env._max_start + 1, env.cfg.ticks_per_step):
        end = start + env.cfg.ticks_per_step
        if end >= env.trace.n_ticks:
            continue
        for source in range(env.n_agents):
            if not env.trace.is_up(source, start):
                continue
            if not env.trace.any_down_during(source, start, end):
                continue
            if any(env.trace.is_up(dest, end) for dest in range(env.n_agents)
                   if dest != source):
                selected = (start, end, source)
                break
        if selected is not None:
            break
    if selected is None:
        raise AssertionError("held-out trace has no observable outage with a healthy peer")
    start, end, source = selected
    env.reset(episode_start_tick=start, seed=7)
    env.residents = [[] for _ in range(env.n_agents)]
    env.cloud_residents = []
    env.inbound[:] = 0
    for task in env.tasks:
        task.state, task.node, task.dest, task.reward_owner = PENDING, -1, -1, -1
        task.remaining_mi = task.spec.length_mi
        task.start_step = task.finish_step = task.lost_step = -1
    env.place_task(0, source)
    env._assign_running(0)
    env._refresh_derived()
    env.step(np.zeros(env.n_agents, dtype=np.int64))
    if not env.recovery_records:
        raise AssertionError("observed host failure did not create a recovery record")
    record = env.recovery_records[0]
    assert record.failure_window_start_tick == start
    assert record.failure_window_end_tick == end
    assert record.detection_tick == end == record.recovery_tick
    assert env.trace.any_down_during(record.source_node, start, end)
    assert record.succeeded
    assert record.destination_node != source
    assert env.trace.is_up(record.destination_node, record.recovery_tick)
    assert math.isclose(record.recovery_latency_s, 0.0, abs_tol=0.0)
    assert env.tasks[0].state != "LOST"
    return record


def validate_no_fabricated_metric_source() -> None:
    source = ROOT / "training" / "sprint9_metrics.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name == "random" for alias in node.names):
                raise AssertionError("Sprint 9 metrics imports random")
        if isinstance(node, ast.ImportFrom) and node.module == "random":
            raise AssertionError("Sprint 9 metrics imports random")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == "random":
                raise AssertionError("Sprint 9 metrics accesses random")


def validate_single_agent_evaluation_source() -> None:
    """Evaluation must be inference/export only, never a PPO training path."""
    source = ROOT / "training" / "sprint9_metrics.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    forbidden_methods = {"save", "update", "train_mode", "backward"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in forbidden_methods:
            raise AssertionError(
                f"Sprint 9 evaluator must not call training/checkpoint method {node.attr!r}")


def validate_single_agent_checkpoint_and_config() -> None:
    """Prove the evaluated controller is a distinct, trained PPO artifact."""
    if not SINGLE_AGENT_CHECKPOINT.is_file():
        raise AssertionError(f"single-agent checkpoint is missing: {SINGLE_AGENT_CHECKPOINT}")
    if not SINGLE_AGENT_CONFIG.is_file():
        raise AssertionError(f"single-agent config is missing: {SINGLE_AGENT_CONFIG}")
    with open(SINGLE_AGENT_CONFIG, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("training_kind") != "single_agent_ppo":
        raise AssertionError("single-agent config has the wrong training kind")
    if "single_agent_ppo_config" not in config:
        raise AssertionError("single-agent config lacks its independent PPO configuration")
    if config.get("train_fraction") != 0.7:
        raise AssertionError("single-agent config does not record the disjoint training fraction")

    # The loader rejects MAPPO checkpoint kinds before constructing a controller.
    from marl.single_agent_ppo import CHECKPOINT_KIND, SingleAgentPPO

    agent = SingleAgentPPO.load(SINGLE_AGENT_CHECKPOINT, device="cpu")
    if CHECKPOINT_KIND != "sprint9_single_agent_ppo":
        raise AssertionError("unexpected single-agent checkpoint kind constant")
    if agent.n_actions != 4 or agent.n_nodes != 10 or agent.state_dim < 1:
        raise AssertionError("single-agent checkpoint has incompatible controller dimensions")
    source = (ROOT / "marl" / "single_agent_ppo.py").read_text(encoding="utf-8")
    if "marl.mappo" in source or "from marl import mappo" in source:
        raise AssertionError("single-agent PPO implementation imports MAPPO")


def validate_single_agent_construction() -> None:
    run = build_baseline(BASELINE_REACTIVE)
    _, state, masks = run.environment.reset(
        episode_start_tick=run.environment._min_start, seed=11)
    agent = SingleAgentPPO(run.environment.state_dim, run.environment.n_agents, seed=11)
    actions, joint_logprob = agent.act(state, masks)
    greedy = agent.act_greedy(state, masks)
    assert actions.shape == (run.environment.n_agents,)
    assert greedy.shape == (run.environment.n_agents,)
    assert np.isfinite(joint_logprob)
    assert np.all(masks[np.arange(run.environment.n_agents), actions])
    source = (ROOT / "marl" / "single_agent_ppo.py").read_text(encoding="utf-8")
    assert "from marl.mappo" not in source
    assert "import marl.mappo" not in source


def _read_csv(path: Path, key: str) -> list[dict]:
    expected = FILE_SPECS[key][1]
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected:
            raise AssertionError(f"{path.name} does not have its fixed Sprint 9 header")
        return list(reader)


def validate_export_bundle(directory: Path, expected_baseline: str) -> list[dict]:
    """Validate actual rows, including raw node telemetry and recovery causality."""
    rows = {}
    for key, (filename, _) in FILE_SPECS.items():
        path = directory / filename
        if not path.exists():
            raise AssertionError(f"missing Sprint 9 output: {path}")
        rows[key] = _read_csv(path, key)
    required_nonempty = ("metadata", "node_ticks", "step_rewards", "task_metadata",
                         "prediction_usage", "episode_metrics")
    for key in required_nonempty:
        if not rows[key]:
            raise AssertionError(f"{key} unexpectedly has no real evaluation rows")
    metadata = rows["metadata"]
    if any(row["baseline_name"] != expected_baseline for row in metadata):
        raise AssertionError("CSV metadata identifies the wrong baseline")
    for key, entries in rows.items():
        if any(row["baseline_name"] != expected_baseline for row in entries):
            raise AssertionError(f"{key} identifies the wrong baseline")
    if expected_baseline in (BASELINE_REACTIVE, BASELINE_DETECTION):
        for row in metadata:
            if row["risk_source"] != "zero" or row["uses_prediction"] != "False" \
                    or row["uses_uncertainty"] != "False":
                raise AssertionError("heuristic baseline metadata does not prove zero-risk execution")
        for row in rows["prediction_usage"]:
            if row["prediction_used"] != "0" or row["uncertainty_used"] != "0" \
                    or row["oof_loaded"] != "0" or row["live_predictor_loaded"] != "0":
                raise AssertionError("non-predictive baseline logged prediction usage")

    # Compare every recorded node metric to its trace source, not just a sample.
    run = build_baseline(
        expected_baseline,
        single_agent_checkpoint=(SINGLE_AGENT_CHECKPOINT
                                 if expected_baseline == BASELINE_SINGLE_AGENT else None),
    )
    trace = run.environment.trace
    for row in rows["node_ticks"]:
        node, tick = int(row["node_id"]), int(row["recorded_tick"])
        for csv_name, channel in (("link_latency_ms", "linkLatencyMs"),
                                  ("power_w", "energy"), ("cpu_utilization_pct", "cpu"),
                                  ("ram_utilization_pct", "ram"),
                                  ("bandwidth_utilization_pct", "bandwidth")):
            if not math.isclose(float(row[csv_name]), float(trace.ch(channel)[node, tick]),
                                rel_tol=1e-9, abs_tol=1e-9):
                raise AssertionError(f"node telemetry differs from trace: {csv_name}")
        duration = float(row["tick_duration_s"])
        if not math.isclose(float(row["energy_interval_j"]), float(row["power_w"]) * duration,
                            rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("energy interval is not documented power x trace duration")

    if expected_baseline == BASELINE_DETECTION:
        if not rows["recovery_events"]:
            raise AssertionError("detection baseline smoke export has no recovery records")
        for row in rows["recovery_events"]:
            start, end = int(row["failure_window_start_tick"]), int(row["failure_window_end_tick"])
            source = int(row["source_node_id"])
            if not trace.any_down_during(source, start, end):
                raise AssertionError("recovery row is not linked to an observed host failure")
            if int(row["recovery_succeeded"]):
                destination = int(row["destination_node_id"])
                if not trace.is_up(destination, int(row["recovery_tick"])):
                    raise AssertionError("successful recovery chose an unhealthy node")
                if int(row["destination_observed_healthy"]) != 1:
                    raise AssertionError("successful recovery did not log destination health")
            if not math.isclose(float(row["recovery_latency_s"]), 0.0, abs_tol=0.0):
                raise AssertionError("detection baseline did not use immediate recovery semantics")
    return metadata


def validate_single_agent_export_bundle(directory: Path) -> list[dict]:
    """Validate the 20-episode inference-only Single-Agent PPO export."""
    metadata = validate_export_bundle(directory, BASELINE_SINGLE_AGENT)
    if len(metadata) != 20:
        raise AssertionError(f"single-agent bundle must contain 20 episodes, found {len(metadata)}")
    episodes = [int(row["episode"]) for row in metadata]
    if episodes != list(range(1, 21)):
        raise AssertionError("single-agent metadata does not contain episodes 1 through 20")
    starts = [int(row["episode_start_tick"]) for row in metadata]
    expected_run = build_baseline(
        BASELINE_SINGLE_AGENT, single_agent_checkpoint=SINGLE_AGENT_CHECKPOINT)
    if starts != deterministic_starts(expected_run, 20):
        raise AssertionError("single-agent bundle does not use the 20 deterministic held-out starts")
    checkpoint = SINGLE_AGENT_CHECKPOINT.resolve()
    for row in metadata:
        recorded = Path(row["checkpoint"])
        resolved = recorded.resolve() if recorded.is_absolute() else (ROOT / recorded).resolve()
        if resolved != checkpoint:
            raise AssertionError("single-agent metadata references the wrong checkpoint")
        if row["training_loss_status"] != "single-agent PPO update CSV required after manual training":
            raise AssertionError("single-agent metadata does not preserve evaluation-only provenance")
    return metadata


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate Sprint 9 baseline safety and provenance")
    parser.add_argument("--reactive-dir", default=None,
                        help="optional exported reactive baseline CSV directory")
    parser.add_argument("--detection-dir", default=None,
                        help="optional exported detection baseline CSV directory")
    parser.add_argument("--single-agent-dir", default=None,
                        help="optional exported Single-Agent PPO CSV directory")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if bool(args.reactive_dir) != bool(args.detection_dir):
        raise SystemExit("supply both --reactive-dir and --detection-dir, or neither")
    validate_protected_artifacts()
    validate_non_predictive_construction()
    exercise_detection_only_recovery()
    validate_no_fabricated_metric_source()
    validate_single_agent_evaluation_source()
    validate_single_agent_construction()
    validate_single_agent_checkpoint_and_config()
    single_agent = None
    if args.reactive_dir:
        reactive = validate_export_bundle(Path(args.reactive_dir), BASELINE_REACTIVE)
        detection = validate_export_bundle(Path(args.detection_dir), BASELINE_DETECTION)
        reactive_starts = [row["episode_start_tick"] for row in reactive]
        detection_starts = [row["episode_start_tick"] for row in detection]
        if reactive_starts != detection_starts:
            raise AssertionError("baselines did not use identical deterministic evaluation starts")
        print(f"validated exported bundles at common starts: {reactive_starts}")
    if args.single_agent_dir:
        single_agent = validate_single_agent_export_bundle(Path(args.single_agent_dir))
        single_agent_starts = [row["episode_start_tick"] for row in single_agent]
        if args.reactive_dir and single_agent_starts != reactive_starts:
            raise AssertionError("single-agent PPO did not use the common deterministic evaluation starts")
        print(f"validated Single-Agent PPO bundle at held-out starts: {single_agent_starts}")
    print("SPRINT 9 BASELINE VALIDATION PASSED")
    print(f"  MAPPO checkpoint SHA-256 unchanged: {MAPPO_SHA256}")
    print("  Sprint 8 validated outputs unchanged")
    print("  Baseline 1 and 3 are mechanically zero-risk; Baseline 3 recovery is post-failure")
    print("  Single-Agent PPO checkpoint/config are independent of MAPPO; evaluation is inference-only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
