"""Run the missing full-MAPPO evaluation on Sprint 9's 20 held-out starts.

This is a deliberately separate, inference-only evaluation tool.  It does not
alter any Sprint 9 baseline CSV or R2 artifact.  Its output is compatible with
the Sprint 11 baseline-comparison plot only after independently exported
Sprint 9 baseline rows with the same protocol and starts are assembled beside
it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marl.env import DTMarlEnv
from marl.mappo import MAPPO, MappoPolicy
from marl.rollout import run_episodes
from marl.sprint9_benchmarks import (BASELINE_REACTIVE, build_baseline,
                                     deterministic_starts,
                                     load_validated_r2_config)
from training.sprint11.schema import sha256_file


R2_CHECKPOINT = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_best.pth"
R2_CONFIG = ROOT / "saved_models" / "marl" / "mappo_R2_mc_target_config.json"
PROTOCOL = "sprint9_matched_20_deterministic_held_out_starts"
FIELDS = [
    "policy", "metric", "mean", "stddev", "sample_count", "evaluation_protocol",
    "held_out_starts_hash", "alignment_status", "checkpoint_id", "checkpoint_hash",
]


def starts_hash(starts: list[int]) -> str:
    return hashlib.sha256(json.dumps(starts, separators=(",", ":")).encode("utf-8")).hexdigest()


def _write_new_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite comparison artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate(output: Path, *, device: str = "cpu") -> Path:
    if not R2_CHECKPOINT.is_file() or not R2_CONFIG.is_file():
        raise FileNotFoundError("frozen R2 checkpoint/config is missing")
    # Build the reactive baseline only to obtain its authoritative Sprint 9
    # held-out starts.  We do not run the baseline or modify its outputs.
    reference = build_baseline(BASELINE_REACTIVE, config_path=R2_CONFIG)
    starts = deterministic_starts(reference, 20)
    cfg = load_validated_r2_config(R2_CONFIG)
    env = DTMarlEnv(cfg.env, cfg.reward)
    if starts != deterministic_starts(reference, 20):
        raise AssertionError("MAPPO and Sprint 9 baseline starts are not identical")
    agent, _ = MAPPO.load(R2_CHECKPOINT, device=device)
    if agent.n_agents != env.n_agents or agent.obs_dim != env.obs_dim or agent.state_dim != env.state_dim:
        raise ValueError("frozen R2 MAPPO geometry is incompatible with the Sprint 9 environment")
    aggregate, _, _, _, _ = run_episodes(env, MappoPolicy(agent, "frozen-r2-mappo"), starts)
    start_digest = starts_hash(starts)
    checkpoint_digest = sha256_file(R2_CHECKPOINT)
    metrics = {
        "episode_reward": (aggregate["episode_reward"], aggregate["episode_reward_std"]),
        "task_success_rate": (aggregate["task_success_rate"], aggregate["task_success_rate_std"]),
        "avg_task_latency_s": (aggregate["avg_task_latency_s"], aggregate["avg_task_latency_s_std"]),
        "energy_reward_cost": (aggregate["energy_cost"], aggregate["energy_cost_std"]),
    }
    rows = [{
        "policy": "frozen-r2-mappo", "metric": metric, "mean": avg, "stddev": sd,
        "sample_count": len(starts), "evaluation_protocol": PROTOCOL,
        "held_out_starts_hash": start_digest, "alignment_status": "aligned",
        "checkpoint_id": R2_CHECKPOINT.name, "checkpoint_hash": checkpoint_digest,
    } for metric, (avg, sd) in metrics.items()]
    _write_new_csv(output, rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new comparison CSV, e.g. data/sprint11/aggregated/sprint9-mappo.csv")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(f"Wrote aligned MAPPO comparison rows: {evaluate(args.output, device=args.device)}")


if __name__ == "__main__":
    main()
