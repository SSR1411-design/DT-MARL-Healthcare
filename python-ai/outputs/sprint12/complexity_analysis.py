"""Reproducible, checkpoint-preserving Sprint 12 complexity analysis.

This program intentionally reads existing artifacts only.  It does not train,
evaluate an environment, step an optimizer, or save a model.  Its only writes
are new Sprint 12 reports in the selected output directory.

The FLOP convention is explicit: each dense-layer multiply-accumulate (MAC)
is reported separately and as two floating-point operations, with one extra
addition for each bias.  Tanh, masks, softmax/categorical operations, tensor
conversion, and Python control flow are excluded because their FLOP counts are
implementation-dependent.  Runtime is measured separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn


OUT_DIR = Path(__file__).resolve().parent
PYTHON_AI = OUT_DIR.parents[1]
REPO = PYTHON_AI.parent
if str(PYTHON_AI) not in sys.path:
    sys.path.insert(0, str(PYTHON_AI))

from marl.mappo import MAPPO  # noqa: E402
from marl.single_agent_ppo import SingleAgentPPO  # noqa: E402


MAPPO_CHECKPOINT = PYTHON_AI / "saved_models" / "marl" / "mappo_R2_mc_target_best.pth"
SINGLE_AGENT_CHECKPOINT = (
    PYTHON_AI / "saved_models" / "single_agent" / "sprint9" /
    "sprint9_single_agent_best.pth"
)
MAPPO_TRAINING_LOG = PYTHON_AI / "run_R2_mc_target_train.log"
MAPPO_TRAINING_CONFIG = PYTHON_AI / "saved_models" / "marl" / "mappo_R2_mc_target_config.json"
SINGLE_AGENT_TRAINING_CONFIG = (
    PYTHON_AI / "saved_models" / "single_agent" / "sprint9" /
    "sprint9_single_agent_config.json"
)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 without modifying *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(REPO).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def checkpoint_inventory() -> list[dict[str, Any]]:
    """Inventory every model-like checkpoint under saved_models exactly."""
    paths = sorted(
        path for path in (PYTHON_AI / "saved_models").rglob("*")
        if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".ckpt"}
    )
    return [file_record(path) for path in paths]


def trainable_parameter_summary(module: nn.Module) -> dict[str, Any]:
    tensors = list(module.parameters())
    dtype_counts: dict[str, int] = {}
    for tensor in tensors:
        dtype = str(tensor.dtype)
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + tensor.numel()
    total = sum(tensor.numel() for tensor in tensors)
    raw_bytes = sum(tensor.numel() * tensor.element_size() for tensor in tensors)
    return {
        "trainable_parameters": total,
        "raw_parameter_bytes": raw_bytes,
        "dtypes": dtype_counts,
        "requires_grad": all(tensor.requires_grad for tensor in tensors),
    }


def linear_operation_summary(module: nn.Module, multiplier: int = 1) -> dict[str, int]:
    """Return the dense linear-operation count for one fixed batch element.

    ``multiplier`` captures repeated identical rows, e.g. MAPPO's critic
    evaluates one global state for each agent identifier.
    """
    linears = [layer for layer in module.modules() if isinstance(layer, nn.Linear)]
    macs = multiplier * sum(layer.in_features * layer.out_features for layer in linears)
    bias_additions = multiplier * sum(
        layer.out_features for layer in linears if layer.bias is not None)
    return {
        "linear_macs": macs,
        "linear_fma_flops": 2 * macs,
        "bias_additions": bias_additions,
        "linear_flops_including_bias": 2 * macs + bias_additions,
    }


def format_bytes(value: int) -> str:
    return f"{value:,} B ({value / 1024 ** 2:.3f} MiB)"


def format_time_us(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def timing_summary(
    fn: Callable[[], Any], *, warmup: int, repeats: int, iterations: int
) -> dict[str, Any]:
    """Time a CPU-only callable in independent batches of fixed length."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        samples_us: list[float] = []
        for _ in range(repeats):
            began = time.perf_counter_ns()
            for _ in range(iterations):
                fn()
            elapsed_ns = time.perf_counter_ns() - began
            samples_us.append(elapsed_ns / iterations / 1_000.0)
    return {
        "warmup_iterations": warmup,
        "repeats": repeats,
        "iterations_per_repeat": iterations,
        "total_measured_calls": repeats * iterations,
        "mean_us": statistics.fmean(samples_us),
        "std_us": statistics.stdev(samples_us) if len(samples_us) > 1 else 0.0,
        "median_us": statistics.median(samples_us),
        "p95_us": float(np.percentile(np.asarray(samples_us), 95)),
        "raw_batch_means_us": samples_us,
    }


def benchmark_inference(
    mappo: MAPPO,
    single: SingleAgentPPO,
    *, warmup: int,
    repeats: int,
    iterations: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Benchmark frozen neural execution only; no simulator is involved."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        mappo.eval()
        single.eval()
        mappo_obs = np.linspace(
            -1.0, 1.0, mappo.n_agents * mappo.obs_dim, dtype=np.float32
        ).reshape(mappo.n_agents, mappo.obs_dim)
        mappo_masks = np.ones((mappo.n_agents, 4), dtype=np.float32)
        mappo_state = np.linspace(-0.5, 0.5, mappo.state_dim, dtype=np.float32)
        mappo_obs_tensor = torch.from_numpy(mappo_obs).unsqueeze(0)
        mappo_state_tensor = torch.from_numpy(mappo_state).unsqueeze(0)

        single_state = np.linspace(
            -1.0, 1.0, single.state_dim, dtype=np.float32
        )
        single_masks = np.ones((single.n_nodes, single.n_actions), dtype=bool)
        single_state_tensor = torch.from_numpy(single_state).unsqueeze(0)

        results = {
            "mappo_actor_forward_all_agents": timing_summary(
                lambda: mappo.actor(mappo_obs_tensor), warmup=warmup,
                repeats=repeats, iterations=iterations),
            "mappo_greedy_policy_decision_all_agents": timing_summary(
                lambda: mappo.act_greedy(mappo_obs, mappo_masks), warmup=warmup,
                repeats=repeats, iterations=iterations),
            "mappo_critic_forward_all_agent_values": timing_summary(
                lambda: mappo.critic(mappo_state_tensor), warmup=warmup,
                repeats=repeats, iterations=iterations),
            "single_agent_actor_forward": timing_summary(
                lambda: single.actor(single_state_tensor), warmup=warmup,
                repeats=repeats, iterations=iterations),
            "single_agent_greedy_policy_decision": timing_summary(
                lambda: single.act_greedy(single_state, single_masks), warmup=warmup,
                repeats=repeats, iterations=iterations),
            "single_agent_critic_forward": timing_summary(
                lambda: single.critic(single_state_tensor), warmup=warmup,
                repeats=repeats, iterations=iterations),
        }
    finally:
        torch.set_num_threads(previous_threads)
    protocol = {
        "scope": "neural model only; excludes environment, simulator, trace I/O, and checkpoint loading",
        "device": "cpu",
        "torch_threads_during_measurement": 1,
        "input": "fixed deterministic float32 linspace tensors with actual checkpoint dimensions; all action masks legal",
        "measurement_clock": "time.perf_counter_ns",
        "statistic": "per-call time calculated within each fixed-size batch; summary statistics are across batches",
        "warmup_iterations": warmup,
        "repeats": repeats,
        "iterations_per_repeat": iterations,
    }
    return protocol, results


def load_training_evidence() -> dict[str, Any]:
    with MAPPO_TRAINING_CONFIG.open(encoding="utf-8") as handle:
        mappo_config = json.load(handle)
    with SINGLE_AGENT_TRAINING_CONFIG.open(encoding="utf-8") as handle:
        single_config = json.load(handle)
    mappo_train = mappo_config["config"]["train"]
    single_updates = SINGLE_AGENT_TRAINING_CONFIG.with_name(
        "sprint9_single_agent_updates.csv"
    )
    with single_updates.open(newline="", encoding="utf-8") as handle:
        single_update_count = sum(1 for _ in csv.DictReader(handle))
    return {
        "mappo_r2": {
            "duration_s": mappo_config["wall_time_s"],
            "duration_source": MAPPO_TRAINING_CONFIG.relative_to(REPO).as_posix(),
            "corroborating_log": MAPPO_TRAINING_LOG.relative_to(REPO).as_posix(),
            "episodes": mappo_train["episodes"],
            "steps_per_episode": mappo_config["config"]["env"]["episode_steps"],
            "environment_steps": (
                mappo_train["episodes"] *
                mappo_config["config"]["env"]["episode_steps"]
            ),
            "updates": mappo_train["episodes"] // mappo_train["rollout_episodes"],
            "device_recorded": mappo_config["device"],
            "measurement_status": "observed historical run time (not estimated)",
        },
        "single_agent_ppo": {
            "duration_s": single_config["wall_time_s"],
            "duration_source": SINGLE_AGENT_TRAINING_CONFIG.relative_to(REPO).as_posix(),
            "episodes": single_config["episodes"],
            "steps_per_episode": single_config["environment_config"]["env"]["episode_steps"],
            "environment_steps": (
                single_config["episodes"] *
                single_config["environment_config"]["env"]["episode_steps"]
            ),
            "updates": single_update_count,
            "device_recorded": single_config["device"],
            "measurement_status": "observed historical run time (not estimated)",
        },
    }


def make_results(args: argparse.Namespace) -> dict[str, Any]:
    if not MAPPO_CHECKPOINT.is_file() or not SINGLE_AGENT_CHECKPOINT.is_file():
        raise FileNotFoundError("required validated Sprint 9/12 checkpoint is missing")

    mappo, mappo_extra = MAPPO.load(MAPPO_CHECKPOINT, device="cpu")
    single = SingleAgentPPO.load(SINGLE_AGENT_CHECKPOINT, device="cpu")
    mappo.eval()
    single.eval()

    mappo_actor = trainable_parameter_summary(mappo.actor)
    mappo_critic = trainable_parameter_summary(mappo.critic)
    mappo_actor_each = mappo_actor["trainable_parameters"] // mappo.n_agents
    mappo_actor_each_bytes = mappo_actor["raw_parameter_bytes"] // mappo.n_agents
    critic_eye = mappo.critic.eye

    single_actor = trainable_parameter_summary(single.actor)
    single_critic = trainable_parameter_summary(single.critic)

    mappo_flops = {
        "actor_per_agent_forward": linear_operation_summary(mappo.actor.net(0)),
        "actor_one_decentralized_decision_all_agents": linear_operation_summary(
            mappo.actor.net(0), multiplier=mappo.n_agents),
        "critic_per_agent_value": linear_operation_summary(mappo.critic.net),
        "critic_one_centralized_state_forward_all_agent_values": linear_operation_summary(
            mappo.critic.net, multiplier=mappo.n_agents),
    }
    mappo_flops["actor_plus_critic_one_state"] = {
        key: (
            mappo_flops["actor_one_decentralized_decision_all_agents"][key] +
            mappo_flops["critic_one_centralized_state_forward_all_agent_values"][key]
        )
        for key in mappo_flops["actor_per_agent_forward"]
    }
    single_flops = {
        "actor_forward": linear_operation_summary(single.actor.net),
        "critic_forward": linear_operation_summary(single.critic.net),
    }
    single_flops["actor_plus_critic_one_state"] = {
        key: single_flops["actor_forward"][key] + single_flops["critic_forward"][key]
        for key in single_flops["actor_forward"]
    }

    inference_protocol, inference = benchmark_inference(
        mappo, single, warmup=args.warmup, repeats=args.repeats,
        iterations=args.iterations)
    training = load_training_evidence()
    mappo_file = file_record(MAPPO_CHECKPOINT)
    single_file = file_record(SINGLE_AGENT_CHECKPOINT)

    results: dict[str, Any] = {
        "sprint": "12",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_scope": (
            "checkpoint-based neural-model analysis and an isolated inference benchmark; "
            "no training, simulator run, evaluation matrix, optimizer step, or source checkpoint write"
        ),
        "environment": {
            "python": sys.version.replace("\n", " "),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "platform": platform.platform(),
            "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        },
        "checkpoint_inventory": checkpoint_inventory(),
        "source_files": {
            "mappo_definition": "python-ai/marl/mappo.py",
            "mappo_configuration": "python-ai/marl/config.py",
            "single_agent_ppo_definition": "python-ai/marl/single_agent_ppo.py",
            "rule_baselines": "python-ai/marl/baseline.py",
            "baseline_factory": "python-ai/marl/sprint9_benchmarks.py",
            "mappo_training": "python-ai/marl/train.py",
            "single_agent_training": "python-ai/marl/train_single_agent.py",
            "mappo_evaluation": "python-ai/marl/evaluate.py",
            "sprint9_baseline_documentation": "docs/SPRINT_9_BASELINES.md",
        },
        "mappo": {
            "checkpoint": mappo_file,
            "checkpoint_extra": mappo_extra,
            "architecture": {
                "n_agents": mappo.n_agents,
                "actor_parameter_sharing": "none: separate_actors=True; 10 independent actor MLPs",
                "actor": {
                    "input_observation_dim": mappo.obs_dim,
                    "hidden_layers": mappo.cfg.actor_hidden,
                    "activation": "Tanh after each hidden Linear layer",
                    "output_actions": 4,
                },
                "critic": {
                    "centralized_global_state_dim": mappo.state_dim,
                    "agent_id_one_hot_dim": mappo.n_agents,
                    "effective_input_dim": mappo.state_dim + mappo.n_agents,
                    "hidden_layers": mappo.cfg.critic_hidden,
                    "activation": "Tanh after each hidden Linear layer",
                    "output": "one value per (state, agent-id); forward(state) emits all 10 values",
                },
            },
            "parameters": {
                "actor_per_agent": {
                    "trainable_parameters": mappo_actor_each,
                    "raw_parameter_bytes": mappo_actor_each_bytes,
                    "dtype": "torch.float32",
                },
                "actors_all_agents": mappo_actor,
                "centralized_critic": mappo_critic,
                "total_trainable_parameters": (
                    mappo_actor["trainable_parameters"] +
                    mappo_critic["trainable_parameters"]
                ),
                "total_raw_parameter_bytes": (
                    mappo_actor["raw_parameter_bytes"] +
                    mappo_critic["raw_parameter_bytes"]
                ),
                "excluded_non_trainable_buffer": {
                    "name": "CentralisedCritic.eye",
                    "elements": critic_eye.numel(),
                    "bytes": critic_eye.numel() * critic_eye.element_size(),
                    "reason": "registered one-hot identity buffer, not a trainable parameter",
                },
            },
            "flops": mappo_flops,
            "model_size": {
                "serialized_checkpoint": mappo_file,
                "raw_trainable_parameter_memory": (
                    mappo_actor["raw_parameter_bytes"] + mappo_critic["raw_parameter_bytes"]
                ),
                "deployment_actor_only_raw_parameter_memory": mappo_actor["raw_parameter_bytes"],
                "dtype": "float32 (4 bytes per trainable parameter)",
                "serialization_note": (
                    "The .pth file also contains Adam optimizer states and metadata; it is not a "
                    "deployment-weight-only artifact."
                ),
            },
        },
        "single_agent_ppo": {
            "checkpoint": single_file,
            "architecture": {
                "parameter_sharing": "one centralized actor, not ten actor replicas",
                "actor": {
                    "input_global_state_dim": single.state_dim,
                    "hidden_layers": single.cfg.actor_hidden,
                    "activation": "Tanh after each hidden Linear layer",
                    "output": f"{single.n_nodes} x {single.n_actions} factored action logits",
                },
                "critic": {
                    "input_global_state_dim": single.state_dim,
                    "hidden_layers": single.cfg.critic_hidden,
                    "activation": "Tanh after each hidden Linear layer",
                    "output": "one scalar team value",
                },
            },
            "parameters": {
                "actor": single_actor,
                "critic": single_critic,
                "total_trainable_parameters": (
                    single_actor["trainable_parameters"] + single_critic["trainable_parameters"]
                ),
                "total_raw_parameter_bytes": (
                    single_actor["raw_parameter_bytes"] + single_critic["raw_parameter_bytes"]
                ),
            },
            "flops": single_flops,
            "model_size": {
                "serialized_checkpoint": single_file,
                "raw_trainable_parameter_memory": (
                    single_actor["raw_parameter_bytes"] + single_critic["raw_parameter_bytes"]
                ),
                "deployment_actor_only_raw_parameter_memory": single_actor["raw_parameter_bytes"],
                "dtype": "float32 (4 bytes per trainable parameter)",
                "serialization_note": (
                    "The .pth file also contains Adam optimizer states and metadata; it is not a "
                    "deployment-weight-only artifact."
                ),
            },
        },
        "flop_convention": {
            "definition": "dense Linear MACs plus 2 FLOPs per MAC and one addition per bias",
            "excluded": [
                "Tanh nonlinearities", "action masking", "argmax/softmax/categorical sampling",
                "tensor conversion", "Python-loop overhead", "memory traffic",
            ],
            "status": "analytical from actual checkpoint architecture; not a runtime measurement",
        },
        "training": training,
        "inference_benchmark_protocol": inference_protocol,
        "inference_benchmark": inference,
        "baseline_scope": {
            "implemented_sprint9_baselines": [
                "reactive-no-digital-twin (ReactiveThresholdPolicy)",
                "single-agent-ppo (SingleAgentPPO)",
                "detection-without-prediction (DetectionRecoveryPolicy)",
            ],
            "additional_evaluation_references": [
                "static-no-migration (NoMigrationPolicy)",
                "random-legal (RandomPolicy)",
                "risk-threshold (RiskThresholdPolicy; heuristic reference, not Sprint 9 baseline)",
            ],
            "not_scheduling_baselines": (
                "python-ai/models contains failure-prediction models (TransformerEncoder, BiLSTM, "
                "FailurePredictor, HTCF); they are upstream predictors, not MAPPO scheduling baselines."
            ),
        },
    }
    return results


def comparison_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    mappo = results["mappo"]
    single = results["single_agent_ppo"]
    timing = results["inference_benchmark"]
    train = results["training"]
    return [
        {
            "algorithm": "MAPPO (validated R2)",
            "category": "learned MARL",
            "parameter_count": mappo["parameters"]["total_trainable_parameters"],
            "parameter_count_status": "measured from loaded trainable tensors",
            "actor_inference_flops": mappo["flops"]["actor_one_decentralized_decision_all_agents"]["linear_flops_including_bias"],
            "flop_status": "analytical dense Linear FLOPs; exclusions documented",
            "serialized_checkpoint_bytes": mappo["checkpoint"]["bytes"],
            "raw_parameter_bytes": mappo["model_size"]["raw_trainable_parameter_memory"],
            "training_duration_s": train["mappo_r2"]["duration_s"],
            "training_scope": "600 episodes; 240,000 environment steps; 75 updates; CPU",
            "inference_mean_us": timing["mappo_greedy_policy_decision_all_agents"]["mean_us"],
            "inference_std_us": timing["mappo_greedy_policy_decision_all_agents"]["std_us"],
            "inference_scope": "CPU greedy actor decision for all 10 agents; no simulator",
            "complexity": "Actor decision O(N_a*(d_o*h_1+h_1*h_2+h_2*A)); critic O(N_a*((d_s+N_a)*g_1+g_1*g_2+g_2))",
            "provenance": mappo["checkpoint"]["path"],
        },
        {
            "algorithm": "Single-agent PPO", "category": "learned centralized baseline",
            "parameter_count": single["parameters"]["total_trainable_parameters"],
            "parameter_count_status": "measured from loaded trainable tensors",
            "actor_inference_flops": single["flops"]["actor_forward"]["linear_flops_including_bias"],
            "flop_status": "analytical dense Linear FLOPs; exclusions documented",
            "serialized_checkpoint_bytes": single["checkpoint"]["bytes"],
            "raw_parameter_bytes": single["model_size"]["raw_trainable_parameter_memory"],
            "training_duration_s": train["single_agent_ppo"]["duration_s"],
            "training_scope": "400 episodes; 160,000 environment steps; 50 updates; CUDA",
            "inference_mean_us": timing["single_agent_greedy_policy_decision"]["mean_us"],
            "inference_std_us": timing["single_agent_greedy_policy_decision"]["std_us"],
            "inference_scope": "CPU greedy centralized policy decision; no simulator",
            "complexity": "O(d_s*h_1+h_1*h_2+h_2*(N_a*A))",
            "provenance": single["checkpoint"]["path"],
        },
        {
            "algorithm": "Reactive no-digital-twin", "category": "rule baseline",
            "parameter_count": "N/A", "parameter_count_status": "no trainable model",
            "actor_inference_flops": "N/A", "flop_status": "non-neural branch/control work; no FLOP measurement",
            "serialized_checkpoint_bytes": "N/A", "raw_parameter_bytes": "N/A",
            "training_duration_s": "N/A", "training_scope": "not required (heuristic)",
            "inference_mean_us": "N/A", "inference_std_us": "N/A",
            "inference_scope": "no isolated benchmark; simulator-free neural timing is not applicable",
            "complexity": "O(N_a*A) worst case for per-agent legal-action preference scan",
            "provenance": "python-ai/marl/baseline.py::ReactiveThresholdPolicy",
        },
        {
            "algorithm": "Detection without prediction", "category": "rule baseline + environment hook",
            "parameter_count": "N/A", "parameter_count_status": "no trainable model",
            "actor_inference_flops": "N/A", "flop_status": "non-neural; recovery belongs to environment transition",
            "serialized_checkpoint_bytes": "N/A", "raw_parameter_bytes": "N/A",
            "training_duration_s": "N/A", "training_scope": "not required (heuristic)",
            "inference_mean_us": "N/A", "inference_std_us": "N/A",
            "inference_scope": "no isolated benchmark; does not perform neural inference",
            "complexity": "Policy O(N_a); recovery transition cost depends on environment healthy-node search",
            "provenance": "python-ai/marl/baseline.py::DetectionRecoveryPolicy",
        },
        {
            "algorithm": "Static no-migration", "category": "evaluation reference",
            "parameter_count": "N/A", "parameter_count_status": "no trainable model",
            "actor_inference_flops": "N/A", "flop_status": "non-neural fixed action",
            "serialized_checkpoint_bytes": "N/A", "raw_parameter_bytes": "N/A",
            "training_duration_s": "N/A", "training_scope": "not required (heuristic)",
            "inference_mean_us": "N/A", "inference_std_us": "N/A",
            "inference_scope": "not benchmarked; no neural model",
            "complexity": "O(N_a) to allocate/emit one STAY action per agent",
            "provenance": "python-ai/marl/baseline.py::NoMigrationPolicy",
        },
        {
            "algorithm": "Random legal", "category": "evaluation reference",
            "parameter_count": "N/A", "parameter_count_status": "no trainable model",
            "actor_inference_flops": "N/A", "flop_status": "non-neural RNG and mask scan",
            "serialized_checkpoint_bytes": "N/A", "raw_parameter_bytes": "N/A",
            "training_duration_s": "N/A", "training_scope": "not required (heuristic)",
            "inference_mean_us": "N/A", "inference_std_us": "N/A",
            "inference_scope": "not benchmarked; RNG timing would be implementation-specific",
            "complexity": "O(N_a*A) to find legal actions and sample", "provenance": "python-ai/marl/baseline.py::RandomPolicy",
        },
        {
            "algorithm": "Risk threshold", "category": "heuristic reference (not Sprint 9 baseline)",
            "parameter_count": "N/A", "parameter_count_status": "no trainable model",
            "actor_inference_flops": "N/A", "flop_status": "non-neural threshold and legal-action preference scan",
            "serialized_checkpoint_bytes": "N/A", "raw_parameter_bytes": "N/A",
            "training_duration_s": "N/A", "training_scope": "not required (heuristic)",
            "inference_mean_us": "N/A", "inference_std_us": "N/A",
            "inference_scope": "not benchmarked; separate upstream risk-prediction cost is out of scope",
            "complexity": "O(N_a*A) worst case", "provenance": "python-ai/marl/baseline.py::RiskThresholdPolicy",
        },
    ]


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def write_report(results: dict[str, Any], comparison: list[dict[str, Any]], out_dir: Path) -> None:
    m = results["mappo"]
    s = results["single_agent_ppo"]
    mt = results["inference_benchmark"]
    tr = results["training"]
    md: list[str] = [
        "# Sprint 12 — Computational Complexity Analysis",
        "",
        "## Scope and provenance",
        "",
        "This is a checkpoint-based analysis of the validated R2 MAPPO policy and the independently trained Sprint 9 single-agent PPO baseline. It did **not** retrain MAPPO, run the Sprint 11 matrix, execute the simulator, alter prior-sprint data, or overwrite a checkpoint. FLOPs are analytical dense-layer counts; timing is a new isolated CPU-only neural benchmark.",
        "",
        f"- Generated: `{results['generated_at_utc']}`",
        f"- MAPPO checkpoint: `{m['checkpoint']['path']}` (`{m['checkpoint']['sha256']}`, {format_bytes(m['checkpoint']['bytes'])})",
        f"- Single-agent PPO checkpoint: `{s['checkpoint']['path']}` (`{s['checkpoint']['sha256']}`, {format_bytes(s['checkpoint']['bytes'])})",
        "- All source checkpoint hashes, sizes, and exact paths are included in `sprint12_complexity_results.json` under `checkpoint_inventory`.",
        "",
        "## Phase 1 — read-only audit",
        "",
        "### A. Model, evaluation, and timing source files",
        "",
        markdown_table(["Purpose", "Exact file(s)"], [
            ["MAPPO actor/critic, rollout, save/load", "`python-ai/marl/mappo.py` (`MultiAgentActor`, `CentralisedCritic`, `MAPPO`)"],
            ["MAPPO architecture/training configuration", "`python-ai/marl/config.py` (`MappoConfig`, `TrainConfig`)"],
            ["MAPPO training", "`python-ai/marl/train.py`"],
            ["MAPPO evaluation and older reference policies", "`python-ai/marl/evaluate.py`"],
            ["Rule policies", "`python-ai/marl/baseline.py`"],
            ["Sprint 9 baseline factory/evaluation", "`python-ai/marl/sprint9_benchmarks.py`"],
            ["Independent centralized PPO", "`python-ai/marl/single_agent_ppo.py`"],
            ["Single-agent PPO training", "`python-ai/marl/train_single_agent.py`"],
            ["Sprint 9 baseline protocol", "`docs/SPRINT_9_BASELINES.md`"],
            ["Existing MAPPO timing evidence", "`python-ai/run_R2_mc_target_train.log`; `python-ai/saved_models/marl/mappo_R2_mc_target_config.json`"],
            ["Existing single-agent timing evidence", "`python-ai/saved_models/single_agent/sprint9/sprint9_single_agent_config.json`"],
        ]),
        "",
        "`python-ai/models/transformer.py`, `bilstm.py`, `failure_predictor.py`, and `htcf.py` define upstream failure-prediction models. They are not scheduling-policy baselines and are outside the MAPPO scheduling-complexity comparison.",
        "",
        "### B. Checkpoints",
        "",
        "The full exact inventory of every `.pth`, `.pt`, and `.ckpt` file under `python-ai/saved_models/` is machine-readable in `sprint12_complexity_results.json`. The two scheduling checkpoints used for numerical analysis are listed above. This avoids ambiguity between the validated R2 checkpoint and other historical / trajectory checkpoints.",
        "",
        "### C–E. What existing artifacts support, and limitations",
        "",
        markdown_table(["Checklist item", "Existing-artifact status", "Sprint 12 action / limitation"], [
            ["Parameter count", "Directly available from source/checkpoint", "Computed from loaded trainable tensors."],
            ["Model size", "Directly available", "Observed serialized bytes and derived raw float32 parameter memory are separated."],
            ["FLOPs", "No prior artifact", "Analytically derived from actual dense layers; not presented as runtime."],
            ["Training time", "Directly available", "Historical wall times come from saved R2 and single-agent configuration/log artifacts."],
            ["Inference time", "No prior artifact", "New fixed-input, CPU-only benchmark; excludes simulator/runtime physics."],
            ["Baseline comparison", "Mixed", "Learned single-agent baseline is measured; rule baselines legitimately have no parameters/checkpoints and no fabricated FLOP/timing values."],
        ]),
        "",
        "Scientific limitations: timing is hardware, PyTorch version, CPU-thread, and implementation dependent; it is not a simulation-step or end-to-end latency. Tanh/masking/argmax/softmax operations are not included in analytical dense FLOPs. Historical MAPPO training says `device=cpu`; historical single-agent training says `device=cuda`, but neither artifact records the exact training CPU/GPU model.",
        "",
        "## MAPPO architecture and parameters",
        "",
        f"- Actors: {m['architecture']['n_agents']} **independent** MLPs, each `48 → 128 → 128 → 4`, with Tanh after hidden layers. There is no actor parameter sharing.",
        f"- Critic: one shared centralized MLP, `(489 global-state + 10 agent-ID) → 256 → 256 → 1`, evaluated once for each agent ID; `forward(state)` returns 10 values.",
        "",
        markdown_table(["Component", "Trainable parameters", "Raw parameter memory", "Dtype"], [
            ["One actor", f"{m['parameters']['actor_per_agent']['trainable_parameters']:,}", format_bytes(m['parameters']['actor_per_agent']['raw_parameter_bytes']), "float32"],
            ["All 10 independent actors", f"{m['parameters']['actors_all_agents']['trainable_parameters']:,}", format_bytes(m['parameters']['actors_all_agents']['raw_parameter_bytes']), "float32"],
            ["Centralized critic", f"{m['parameters']['centralized_critic']['trainable_parameters']:,}", format_bytes(m['parameters']['centralized_critic']['raw_parameter_bytes']), "float32"],
            ["Total trainable", f"{m['parameters']['total_trainable_parameters']:,}", format_bytes(m['parameters']['total_raw_parameter_bytes']), "float32"],
        ]),
        "",
        f"The critic’s `eye` one-hot buffer ({m['parameters']['excluded_non_trainable_buffer']['elements']} float32 elements / {m['parameters']['excluded_non_trainable_buffer']['bytes']} B) is explicitly excluded from the trainable count. Optimizer state is also excluded.",
        "",
        "## Model size",
        "",
        markdown_table(["Model", "Serialized checkpoint", "Raw trainable parameters", "Deployment actor-only raw parameters", "Meaning"], [
            ["MAPPO R2", format_bytes(m['checkpoint']['bytes']), format_bytes(m['model_size']['raw_trainable_parameter_memory']), format_bytes(m['model_size']['deployment_actor_only_raw_parameter_memory']), "`.pth` includes optimizer + metadata; raw is weights only"],
            ["Single-agent PPO", format_bytes(s['checkpoint']['bytes']), format_bytes(s['model_size']['raw_trainable_parameter_memory']), format_bytes(s['model_size']['deployment_actor_only_raw_parameter_memory']), "`.pth` includes optimizer + metadata; raw is weights only"],
        ]),
        "",
        "All analyzed trainable tensors are float32, so raw parameter memory is exactly 4 bytes × parameter count. Serialized file size must not be interpreted as deployment-only model weight memory.",
        "",
        "## FLOPs (analytical)",
        "",
        "Convention: dense linear MACs are reported alongside `2 × MACs + bias additions` FLOPs. This is an analytical operation count for one batch/state and excludes Tanh, action masking, softmax/categorical operations, memory traffic, and framework overhead.",
        "",
        markdown_table(["MAPPO computation", "MACs", "FLOPs incl. bias", "Interpretation"], [
            ["Actor, one agent", f"{m['flops']['actor_per_agent_forward']['linear_macs']:,}", f"{m['flops']['actor_per_agent_forward']['linear_flops_including_bias']:,}", "one local-observation policy MLP"],
            ["Actor, 10-agent decentralized decision", f"{m['flops']['actor_one_decentralized_decision_all_agents']['linear_macs']:,}", f"{m['flops']['actor_one_decentralized_decision_all_agents']['linear_flops_including_bias']:,}", "deployed policy computation"],
            ["Critic, one agent value", f"{m['flops']['critic_per_agent_value']['linear_macs']:,}", f"{m['flops']['critic_per_agent_value']['linear_flops_including_bias']:,}", "one `(state, agent-ID)` row"],
            ["Critic, `forward(state)` → 10 values", f"{m['flops']['critic_one_centralized_state_forward_all_agent_values']['linear_macs']:,}", f"{m['flops']['critic_one_centralized_state_forward_all_agent_values']['linear_flops_including_bias']:,}", "centralized learning/value evaluation"],
            ["Actor + critic, one state", f"{m['flops']['actor_plus_critic_one_state']['linear_macs']:,}", f"{m['flops']['actor_plus_critic_one_state']['linear_flops_including_bias']:,}", "meaningful only when both are run"],
        ]),
        "",
        markdown_table(["Single-agent PPO computation", "MACs", "FLOPs incl. bias"], [
            ["Actor, factored 10 × 4 output", f"{s['flops']['actor_forward']['linear_macs']:,}", f"{s['flops']['actor_forward']['linear_flops_including_bias']:,}"],
            ["Critic, one global value", f"{s['flops']['critic_forward']['linear_macs']:,}", f"{s['flops']['critic_forward']['linear_flops_including_bias']:,}"],
            ["Actor + critic, one state", f"{s['flops']['actor_plus_critic_one_state']['linear_macs']:,}", f"{s['flops']['actor_plus_critic_one_state']['linear_flops_including_bias']:,}"],
        ]),
        "",
        "## Training time (existing evidence)",
        "",
        markdown_table(["Run", "Duration", "Work represented", "Recorded device", "Evidence"], [
            ["Validated R2 MAPPO", f"{tr['mappo_r2']['duration_s']:.1f} s", f"{tr['mappo_r2']['episodes']} episodes × {tr['mappo_r2']['steps_per_episode']} steps = {tr['mappo_r2']['environment_steps']:,} environment steps; {tr['mappo_r2']['updates']} updates", tr['mappo_r2']['device_recorded'], f"`{tr['mappo_r2']['duration_source']}`; log rounds to 1,586 s"],
            ["Single-agent PPO baseline", f"{tr['single_agent_ppo']['duration_s']:.3f} s", f"{tr['single_agent_ppo']['episodes']} episodes × {tr['single_agent_ppo']['steps_per_episode']} steps = {tr['single_agent_ppo']['environment_steps']:,} environment steps; {tr['single_agent_ppo']['updates']} updates", tr['single_agent_ppo']['device_recorded'], f"`{tr['single_agent_ppo']['duration_source']}`"],
        ]),
        "",
        "These are observed historical run times, not scaling extrapolations. They include their respective training-loop/environment costs and must not be compared as pure neural compute benchmarks because device contexts differ.",
        "",
        "## Isolated inference / decision time",
        "",
        f"Protocol: {results['inference_benchmark_protocol']['repeats']} batches × {results['inference_benchmark_protocol']['iterations_per_repeat']} calls after {results['inference_benchmark_protocol']['warmup_iterations']} warmups; CPU with one PyTorch thread; fixed float32 inputs at the real checkpoint dimensions. No environment, trace, I/O, or checkpoint loading is timed.",
        "",
        markdown_table(["Callable", "Mean µs", "SD µs", "Median µs", "P95 µs", "What is timed"], [
            ["MAPPO actor forward", format_time_us(mt['mappo_actor_forward_all_agents']['mean_us']), format_time_us(mt['mappo_actor_forward_all_agents']['std_us']), format_time_us(mt['mappo_actor_forward_all_agents']['median_us']), format_time_us(mt['mappo_actor_forward_all_agents']['p95_us']), "all 10 actor MLPs only"],
            ["MAPPO greedy decision", format_time_us(mt['mappo_greedy_policy_decision_all_agents']['mean_us']), format_time_us(mt['mappo_greedy_policy_decision_all_agents']['std_us']), format_time_us(mt['mappo_greedy_policy_decision_all_agents']['median_us']), format_time_us(mt['mappo_greedy_policy_decision_all_agents']['p95_us']), "production-style `act_greedy`, including masks/argmax"],
            ["MAPPO critic forward", format_time_us(mt['mappo_critic_forward_all_agent_values']['mean_us']), format_time_us(mt['mappo_critic_forward_all_agent_values']['std_us']), format_time_us(mt['mappo_critic_forward_all_agent_values']['median_us']), format_time_us(mt['mappo_critic_forward_all_agent_values']['p95_us']), "all 10 centralized values"],
            ["Single-agent actor forward", format_time_us(mt['single_agent_actor_forward']['mean_us']), format_time_us(mt['single_agent_actor_forward']['std_us']), format_time_us(mt['single_agent_actor_forward']['median_us']), format_time_us(mt['single_agent_actor_forward']['p95_us']), "centralized 10 × 4 logits"],
            ["Single-agent greedy decision", format_time_us(mt['single_agent_greedy_policy_decision']['mean_us']), format_time_us(mt['single_agent_greedy_policy_decision']['std_us']), format_time_us(mt['single_agent_greedy_policy_decision']['median_us']), format_time_us(mt['single_agent_greedy_policy_decision']['p95_us']), "production-style `act_greedy`, including masks/argmax"],
            ["Single-agent critic forward", format_time_us(mt['single_agent_critic_forward']['mean_us']), format_time_us(mt['single_agent_critic_forward']['std_us']), format_time_us(mt['single_agent_critic_forward']['median_us']), format_time_us(mt['single_agent_critic_forward']['p95_us']), "one centralized value"],
        ]),
        "",
        "## Paper-ready complexity comparison",
        "",
        markdown_table(["Algorithm", "Trainable parameters", "Actor decision FLOPs", "Serialized size", "Training time", "CPU inference mean", "Complexity"], [
            [row['algorithm'], str(row['parameter_count']), str(row['actor_inference_flops']), str(row['serialized_checkpoint_bytes']), str(row['training_duration_s']), (format_time_us(row['inference_mean_us']) if isinstance(row['inference_mean_us'], float) else str(row['inference_mean_us'])), row['complexity']]
            for row in comparison
        ]),
        "",
        "`N/A` means no model, checkpoint, or scientifically appropriate isolated measurement exists; it is not a zero-valued experimental result. The full table with status/provenance columns is `sprint12_complexity_comparison.csv`.",
        "",
        "## Validation and preservation",
        "",
        "- The analysis script loads checkpoints with `map_location='cpu'`, calls `eval()`, and never calls a training/update method.",
        "- SHA-256 fingerprints were recorded for every discovered checkpoint and specifically for the two analyzed scheduling checkpoints.",
        "- Prior sprint directories were read only. The only new artifacts are in `python-ai/outputs/sprint12/`.",
        "- No Sprint 11 command or experiment matrix was invoked.",
    ]
    (out_dir / "SPRINT12_COMPLEXITY_ANALYSIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_outputs(results: dict[str, Any], out_dir: Path, *, overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        out_dir / "sprint12_complexity_results.json",
        out_dir / "sprint12_complexity_comparison.csv",
        out_dir / "SPRINT12_COMPLEXITY_ANALYSIS.md",
    ]
    existing = [target for target in targets if target.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise FileExistsError(f"refusing to overwrite Sprint 12 outputs: {names}; use --overwrite")

    comparison = comparison_rows(results)
    with (out_dir / "sprint12_complexity_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    with (out_dir / "sprint12_complexity_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    write_report(results, comparison, out_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR,
                        help="new Sprint 12 output directory (default: script directory)")
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace only prior files in the chosen Sprint 12 output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or args.repeats < 2 or args.iterations < 1:
        raise ValueError("warmup must be >= 0, repeats >= 2, and iterations >= 1")
    results = make_results(args)
    write_outputs(results, args.out_dir.resolve(), overwrite=args.overwrite)
    print(f"Wrote Sprint 12 analysis to {args.out_dir.resolve()}")
    print("MAPPO checkpoint SHA-256:", results["mappo"]["checkpoint"]["sha256"])
    print("Single-agent checkpoint SHA-256:", results["single_agent_ppo"]["checkpoint"]["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
