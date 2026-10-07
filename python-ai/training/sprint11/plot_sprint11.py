"""Render data-driven, publication-oriented Sprint 11 figures.

No graph contains experimental constants.  Every series is read from a
validated aggregate or explicitly aligned comparison file; output is refused
when input is missing, statistically malformed, or protocol-incompatible.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


GRAPH_SPECS: dict[str, dict[str, str]] = {
    "latency_vs_device_count": {"metric": "avg_task_latency_s", "x": "device_count", "xlabel": "IoMT device count", "ylabel": "Mean task latency (s)"},
    "energy_vs_device_count": {"metric": "physical_energy_j", "x": "device_count", "xlabel": "IoMT device count", "ylabel": "Physical energy (J)"},
    "success_vs_device_count": {"metric": "task_success_rate", "x": "device_count", "xlabel": "IoMT device count", "ylabel": "Task success rate (%)"},
    "latency_vs_task_count": {"metric": "avg_task_latency_s", "x": "task_count", "xlabel": "Healthcare task count", "ylabel": "Mean task latency (s)"},
    "energy_vs_task_count": {"metric": "physical_energy_j", "x": "task_count", "xlabel": "Healthcare task count", "ylabel": "Physical energy (J)"},
    "success_vs_task_count": {"metric": "task_success_rate", "x": "task_count", "xlabel": "Healthcare task count", "ylabel": "Task success rate (%)"},
    "success_vs_bandwidth": {"metric": "task_success_rate", "x": "bandwidth_mbps", "xlabel": "Physical link bandwidth (Mbps)", "ylabel": "Task success rate (%)"},
    "cpu_utilization_vs_task_count": {"metric": "mean_cpu_utilization_pct", "x": "task_count", "xlabel": "Healthcare task count", "ylabel": "Mean CPU utilization (%)"},
    "ram_utilization_vs_task_count": {"metric": "mean_ram_utilization_pct", "x": "task_count", "xlabel": "Healthcare task count", "ylabel": "Mean RAM utilization (%)"},
    "bandwidth_utilization_vs_task_count": {"metric": "mean_bandwidth_utilization_pct", "x": "task_count", "xlabel": "Healthcare task count", "ylabel": "Mean bandwidth utilization (%)"},
}
HISTOGRAM_METRICS = {
    "latency_histogram": ("avg_task_latency_s", "Task latency (s)"),
    "physical_energy_histogram": ("physical_energy_j", "Physical energy (J)"),
    "HSI_histogram": ("HSI", "Health Severity Index"),
    "reward_histogram": ("episode_reward", "Episode reward"),
}


def _matplotlib():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 160, "savefig.dpi": 300, "font.size": 10,
        "axes.grid": True, "grid.alpha": 0.28, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })
    return plt


def _read_csv(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"input CSV does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} is missing required columns: {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{path} contains no generated observations")
    return rows


def _float(value: str, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def validate_aggregate_rows(rows: Iterable[Mapping[str, str]]) -> None:
    for row in rows:
        n = int(row["sample_count"])
        if n < 1:
            raise ValueError("sample_count must be positive")
        _float(row["mean"], "mean")
        status = row["ci95_status"]
        low, high = row["ci95_low"], row["ci95_high"]
        if n == 1:
            if status != "not_available_n_lt_2" or low or high:
                raise ValueError("a single trial must not have a confidence interval")
        elif status not in {"t_interval_95", "t_interval_95_approx_df_gt_30"}:
            raise ValueError("multiple trials require a valid 95% confidence interval")
        elif _float(low, "ci95_low") > _float(high, "ci95_high"):
            raise ValueError("confidence interval bounds are reversed")


def _save_figure(fig: Any, output_dir: Path, name: str, metadata: Mapping[str, Any]) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{name}.{suffix}" for suffix in ("png", "svg", "pdf")]
    manifest = output_dir / f"{name}.json"
    existing = [path for path in paths + [manifest] if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite graph artifact(s): " + ", ".join(map(str, existing)))
    for path in paths:
        fig.savefig(path, bbox_inches="tight")
    manifest.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def plot_scale_graph(kind: str, aggregate_csv: Path, output_dir: Path) -> list[Path]:
    if kind not in GRAPH_SPECS:
        raise ValueError(f"unknown scale graph: {kind}")
    spec = GRAPH_SPECS[kind]
    fields = {"scenario_id", "policy", "policy_kind", "metric", "mean", "sample_count",
              "ci95_low", "ci95_high", "ci95_status", spec["x"], "host_count",
              "task_count", "device_count", "bandwidth_mbps", "failure_intensity"}
    rows = _read_csv(aggregate_csv, fields)
    validate_aggregate_rows(rows)
    rows = [row for row in rows if row["metric"] == spec["metric"]]
    if len(rows) < 2 or len({row[spec["x"]] for row in rows}) < 2:
        raise ValueError(f"{kind} requires at least two generated {spec['x']} values")
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    controls = [name for name in ("host_count", "task_count", "device_count", "bandwidth_mbps", "failure_intensity") if name != spec["x"]]
    for row in rows:
        groups[(row["policy"],) + tuple(row[key] for key in controls)].append(row)
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    for group, values in sorted(groups.items()):
        values.sort(key=lambda row: _float(row[spec["x"]], spec["x"]))
        x = [_float(row[spec["x"]], spec["x"]) for row in values]
        raw_y = [_float(row["mean"], "mean") for row in values]
        y = [100 * item if spec["metric"] == "task_success_rate" else item for item in raw_y]
        err = []
        for row, avg in zip(values, y):
            if row["ci95_status"] == "not_available_n_lt_2":
                err.append(0.0)
            else:
                lo = _float(row["ci95_low"], "ci95_low")
                hi = _float(row["ci95_high"], "ci95_high")
                multiplier = 100 if spec["metric"] == "task_success_rate" else 1
                err.append(max(abs(avg - multiplier * lo), abs(multiplier * hi - avg)))
        n_values = sorted({row["sample_count"] for row in values})
        control_label = ", ".join(f"{name.replace('_count', '')}={value}" for name, value in zip(controls, group[1:]))
        label = f"{group[0]} (n={','.join(n_values)}; {control_label})"
        ax.errorbar(x, y, yerr=err, marker="o", capsize=3, linewidth=1.8, label=label)
    ax.set_xlabel(spec["xlabel"])
    ax.set_ylabel(spec["ylabel"])
    ax.set_title(kind.replace("_", " ").title())
    ax.legend(fontsize=8)
    return _save_figure(fig, output_dir, kind, {
        "graph": kind, "input": str(aggregate_csv), "metric": spec["metric"],
        "independent_sample_unit": "simulator trial", "ci": "95% t interval when n >= 2",
    })


def plot_priority_response(priority_csv: Path, output_dir: Path) -> list[Path]:
    fields = {"scenario_id", "policy", "priority_bin", "mean_response_time_s", "sample_count",
              "ci95_low", "ci95_high", "ci95_status"}
    rows = _read_csv(priority_csv, fields)
    # Translate to aggregate field names for shared validation.
    validate_aggregate_rows([{**row, "mean": row["mean_response_time_s"]} for row in rows])
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    by_policy: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_policy[row["policy"]].append(row)
    for policy, values in sorted(by_policy.items()):
        values.sort(key=lambda row: row["priority_bin"])
        x = [row["priority_bin"] for row in values]
        y = [_float(row["mean_response_time_s"], "mean_response_time_s") for row in values]
        errors = [0.0 if row["ci95_status"] == "not_available_n_lt_2" else
                  max(yv - _float(row["ci95_low"], "ci95_low"),
                      _float(row["ci95_high"], "ci95_high") - yv)
                  for row, yv in zip(values, y)]
        ax.errorbar(x, y, yerr=errors, marker="o", capsize=3, linewidth=1.8,
                    label=f"{policy} (n={','.join(sorted({r['sample_count'] for r in values}))})")
    ax.set_xlabel("Clinical priority bin")
    ax.set_ylabel("Mean response time (s)")
    ax.set_title("Response time vs priority")
    ax.legend()
    return _save_figure(fig, output_dir, "response_time_vs_priority", {
        "graph": "response_time_vs_priority", "input": str(priority_csv),
        "independent_sample_unit": "simulator trial",
    })


def plot_histogram(kind: str, observation_csv: Path, output_dir: Path) -> list[Path]:
    if kind not in HISTOGRAM_METRICS:
        raise ValueError(f"unknown histogram: {kind}")
    metric, xlabel = HISTOGRAM_METRICS[kind]
    rows = _read_csv(observation_csv, {"scenario_id", "policy", "metric", "value", "sample_unit"})
    rows = [row for row in rows if row["metric"] == metric]
    if not rows:
        raise ValueError(f"no generated observations for {metric}")
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    by_policy: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_policy[row["policy"]].append(_float(row["value"], "value"))
    for policy, values in sorted(by_policy.items()):
        ax.hist(values, bins="auto", alpha=0.55, edgecolor="white", label=f"{policy} (n={len(values)})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Observation count")
    ax.set_title(kind.replace("_", " ").title())
    ax.legend()
    return _save_figure(fig, output_dir, kind, {
        "graph": kind, "input": str(observation_csv), "metric": metric,
        "sample_unit": sorted({row["sample_unit"] for row in rows}),
    })


def plot_training(kind: str, training_csv: Path, output_dir: Path) -> list[Path]:
    if kind not in {"reward_vs_training_episode", "actor_critic_loss_vs_training_update"}:
        raise ValueError(f"unknown training graph: {kind}")
    required = {"episode", "episode_reward"} if kind == "reward_vs_training_episode" else {"update", "actor_loss", "critic_loss"}
    rows = _read_csv(training_csv, required)
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    if kind == "reward_vs_training_episode":
        rows.sort(key=lambda row: _float(row["episode"], "episode"))
        ax.plot([_float(row["episode"], "episode") for row in rows],
                [_float(row["episode_reward"], "episode_reward") for row in rows], linewidth=1.4)
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Episode reward")
    else:
        rows.sort(key=lambda row: _float(row["update"], "update"))
        x = [_float(row["update"], "update") for row in rows]
        ax.plot(x, [_float(row["actor_loss"], "actor_loss") for row in rows], label="Actor loss")
        ax.plot(x, [_float(row["critic_loss"], "critic_loss") for row in rows], label="Critic loss")
        ax.set_xlabel("Training update")
        ax.set_ylabel("Loss")
        ax.legend()
    ax.set_title(kind.replace("_", " ").title())
    return _save_figure(fig, output_dir, kind, {"graph": kind, "input": str(training_csv)})


def plot_sprint9_comparison(comparison_csv: Path, output_dir: Path, metric: str) -> list[Path]:
    required = {"policy", "metric", "mean", "evaluation_protocol", "held_out_starts_hash", "alignment_status", "sample_count"}
    rows = _read_csv(comparison_csv, required)
    rows = [row for row in rows if row["metric"] == metric]
    if not rows:
        raise ValueError(f"comparison contains no {metric} rows")
    protocols = {row["evaluation_protocol"] for row in rows}
    starts = {row["held_out_starts_hash"] for row in rows}
    if len(protocols) != 1 or len(starts) != 1 or any(row["alignment_status"] != "aligned" for row in rows):
        raise ValueError("Sprint 9 comparison requires exactly matched protocol and held-out starts")
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    labels = [row["policy"] for row in rows]
    values = [_float(row["mean"], "mean") for row in rows]
    ax.bar(range(len(rows)), values)
    ax.set_xticks(range(len(rows)), labels, rotation=20, ha="right")
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title("Aligned Sprint 9 baseline comparison")
    return _save_figure(fig, output_dir, "sprint9_baseline_comparison", {
        "graph": "sprint9_baseline_comparison", "input": str(comparison_csv),
        "metric": metric, "evaluation_protocol": next(iter(protocols)),
        "held_out_starts_hash": next(iter(starts)), "alignment_status": "aligned",
    })


def plot_uncertainty_gate_error_rate(input_csv: Path, output_dir: Path) -> list[Path]:
    rows = _read_csv(input_csv, {"policy", "error_rate", "genuine_uncertainty_estimator", "gate_definition"})
    if any(row["genuine_uncertainty_estimator"].lower() != "true" for row in rows):
        raise ValueError("NOT AVAILABLE: current reserved-zero interface is not a genuine uncertainty gate")
    plt = _matplotlib()
    fig, ax = plt.subplots(figsize=(7.3, 4.5))
    ax.bar([row["policy"] for row in rows], [_float(row["error_rate"], "error_rate") for row in rows])
    ax.set_ylabel("Uncertainty-gating error rate")
    ax.set_title("Uncertainty-gating error rate")
    return _save_figure(fig, output_dir, "uncertainty_gate_error_rate", {"graph": "uncertainty_gate_error_rate", "input": str(input_csv)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", choices=sorted(set(GRAPH_SPECS) | set(HISTOGRAM_METRICS) | {
        "response_time_vs_priority", "reward_vs_training_episode",
        "actor_critic_loss_vs_training_update", "sprint9_baseline_comparison",
        "uncertainty_gate_error_rate"}))
    parser.add_argument("--aggregate-csv", type=Path)
    parser.add_argument("--priority-csv", type=Path)
    parser.add_argument("--observation-csv", type=Path)
    parser.add_argument("--training-csv", type=Path)
    parser.add_argument("--comparison-csv", type=Path)
    parser.add_argument("--uncertainty-csv", type=Path)
    parser.add_argument("--comparison-metric", default="task_success_rate")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.graph in GRAPH_SPECS:
        if args.aggregate_csv is None:
            parser.error("--aggregate-csv is required")
        paths = plot_scale_graph(args.graph, args.aggregate_csv, args.output_dir)
    elif args.graph == "response_time_vs_priority":
        if args.priority_csv is None:
            parser.error("--priority-csv is required")
        paths = plot_priority_response(args.priority_csv, args.output_dir)
    elif args.graph in HISTOGRAM_METRICS:
        if args.observation_csv is None:
            parser.error("--observation-csv is required")
        paths = plot_histogram(args.graph, args.observation_csv, args.output_dir)
    elif args.graph in {"reward_vs_training_episode", "actor_critic_loss_vs_training_update"}:
        if args.training_csv is None:
            parser.error("--training-csv is required")
        paths = plot_training(args.graph, args.training_csv, args.output_dir)
    elif args.graph == "sprint9_baseline_comparison":
        if args.comparison_csv is None:
            parser.error("--comparison-csv is required")
        paths = plot_sprint9_comparison(args.comparison_csv, args.output_dir, args.comparison_metric)
    else:
        if args.uncertainty_csv is None:
            parser.error("--uncertainty-csv is required")
        paths = plot_uncertainty_gate_error_rate(args.uncertainty_csv, args.output_dir)
    print("Wrote:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
