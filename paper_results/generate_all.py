"""Generate the additive DT-MARL-Healthcare publication-results package.

This script reads completed, immutable experiment artifacts only.  It does not
import the MARL environment, load checkpoints, train, evaluate, or write into
the historical artifact tree.  All values that are not direct JSON/CSV fields
are labelled as derived in the generated output matrix.

Run from the repository root, for example:
    python paper_results/generate_all.py --output-root paper_results

The default writes a separately regenerable package under
paper_results/generated/.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
PAPER_ROOT = REPO / "paper_results"
MARL = REPO / "python-ai" / "saved_models" / "marl"
PREDICTOR = REPO / "python-ai" / "saved_models"
SIMULATION = REPO / "simulation"

S65_ARMS = [
    ("A0", "mappo_A0_cpu_repro_eval.json", "Sprint 6 legacy reproduction"),
    ("A1", "mappo_A1_cpu_bugfix_eval.json", "Both Sprint 6.5 bug fixes"),
    ("A2", "mappo_A2_crit_sign_eval.json", "A1 + w_criticality_migration=0"),
    ("A3", "mappo_A3_entropy_eval.json", "A1 + entropy_coef=0.05"),
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

COLORS = {
    "mappo": "#377eb8",
    "threshold": "#e66101",
    "random": "#4daf4a",
    "reactive": "#984ea3",
    "static": "#777777",
    "a0": "#377eb8",
    "a1": "#4daf4a",
    "a2": "#984ea3",
    "a3": "#e66101",
    "random_set": "#377eb8",
    "union_set": "#e66101",
    "low": "#377eb8",
    "high": "#e66101",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def ensure_sources(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Required completed artifact(s) missing:\n" + "\n".join(missing))


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a dependency-free Markdown table without silently changing values."""
    clean = frame.copy()
    clean = clean.fillna("")
    for column in clean.columns:
        clean[column] = clean[column].map(lambda value: str(value).replace("|", "\\|"))
    header = "| " + " | ".join(clean.columns) + " |"
    divider = "|" + "|".join(["---"] * len(clean.columns)) + "|"
    body = ["| " + " | ".join(row) + " |" for row in clean.astype(str).itertuples(index=False, name=None)]
    return "\n".join([header, divider, *body])


def write_table(frame: pd.DataFrame, name: str, tables_dir: Path) -> None:
    frame.to_csv(tables_dir / f"{name}.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    (tables_dir / f"{name}.md").write_text(markdown_table(frame) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, name: str, figures_dir: Path) -> None:
    png = figures_dir / f"{name}.png"
    pdf = figures_dir / f"{name}.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    # Validation during generation: a malformed PNG fails here.
    mpimg.imread(png)


def format_float(value: Any, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return f"{float(value):.{digits}f}"


def outcome_direction(correlation: float) -> str:
    if correlation > 0:
        return "positive"
    if correlation < 0:
        return "negative"
    return "zero"


def load_inputs() -> dict[str, Any]:
    paths = {
        "host_metrics": PREDICTOR / "host_metrics.json",
        "lead_time": PREDICTOR / "host_lead_time.csv",
        "mappo_config": MARL / "mappo_config.json",
        "mappo_history": MARL / "mappo_history.csv",
        "mappo_updates": MARL / "mappo_updates.csv",
        "mappo_eval": MARL / "mappo_eval.json",
        "phase5": MARL / "SPRINT_7_PHASE5_risk_trajectory_main.json",
        "s65_report": MARL / "SPRINT_6_5_REPORT.md",
        "phase5_report": MARL / "SPRINT_7_PHASE5_REPORT.md",
        "audit": MARL / "SPRINT_7_FINAL_SYNTHESIS_AUDIT.md",
        "sim_manager": REPO / "simulation" / "src" / "main" / "java" / "com" / "dtmarl" / "simulation" / "SimulationManager.java",
    }
    for arm, filename, _ in S65_ARMS:
        paths[f"s65_{arm}"] = MARL / filename
    ensure_sources(paths.values())
    return {
        "paths": paths,
        "host_metrics": read_json(paths["host_metrics"]),
        "lead_time": pd.read_csv(paths["lead_time"]),
        "config": read_json(paths["mappo_config"]),
        "history": pd.read_csv(paths["mappo_history"]),
        "updates": pd.read_csv(paths["mappo_updates"]),
        "evaluation": read_json(paths["mappo_eval"]),
        "phase5": read_json(paths["phase5"]),
        "s65": {arm: read_json(paths[f"s65_{arm}"]) for arm, _, _ in S65_ARMS},
        "sim_manager_text": paths["sim_manager"].read_text(encoding="utf-8"),
    }


def build_failure_data(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pooled = data["host_metrics"]["pooled_oof"]
    rows = [
        {"measure_group": "pooled_oof", "metric": key, "value": value, "source_artifact": "python-ai/saved_models/host_metrics.json"}
        for key, value in pooled.items()
        if isinstance(value, (int, float))
    ]
    for baseline, metrics in data["host_metrics"]["baselines"].items():
        for metric in ("accuracy", "precision", "recall", "f1"):
            rows.append(
                {
                    "measure_group": f"baseline:{baseline}",
                    "metric": metric,
                    "value": metrics[metric],
                    "source_artifact": "python-ai/saved_models/host_metrics.json",
                }
            )
    performance = pd.DataFrame(rows)

    lead = data["lead_time"].copy()
    lead["lead_seconds"] = pd.to_numeric(lead["lead"], errors="coerce")
    lead["run_lead_seconds"] = pd.to_numeric(lead["run_lead"], errors="coerce")
    lead["source_artifact"] = "python-ai/saved_models/host_lead_time.csv"

    verdict_counts = lead.groupby("verdict", dropna=False).size().reset_index(name="n_events")
    verdict_counts["source_artifact"] = "python-ai/saved_models/host_lead_time.csv"
    return performance, lead, verdict_counts


def build_evaluation_data(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy, metrics in data["evaluation"]["policies"].items():
        rows.append(
            {
                "policy": policy,
                "episode_reward": metrics["episode_reward"],
                "task_success_rate": metrics["task_success_rate"],
                "relocations": metrics["relocations"],
                "tasks_protected_before_failure": metrics["tasks_protected_before_failure"],
                "lost": metrics["lost"],
                "failed_critical_tasks": metrics["failed_critical_tasks"],
                "n_episodes": metrics["n_episodes"],
                "source_artifact": "python-ai/saved_models/marl/mappo_eval.json",
            }
        )
    return pd.DataFrame(rows)


def build_s65_data(data: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, filename, change in S65_ARMS:
        payload = data["s65"][arm]
        learned = payload["policies"]["mappo-greedy"]
        zero = payload["ablation_risk_zero"]
        rows.append(
            {
                "arm": arm,
                "change": change,
                "risk_sensitivity_span": payload["probe_risk"]["p_relocate_span"],
                "risk_sensitivity_correlation": payload["probe_risk"]["corr_risk_p_relocate"],
                "criticality_span": payload["probe_severity"]["p_relocate_span"],
                "criticality_correlation": payload["probe_severity"]["corr_severity_p_relocate"],
                "criticality_direction": outcome_direction(payload["probe_severity"]["corr_severity_p_relocate"]),
                "protected_with_risk": learned["tasks_protected_before_failure"],
                "protected_risk_zero": zero["tasks_protected_before_failure"],
                "lost_with_risk": learned["lost"],
                "lost_risk_zero": zero["lost"],
                "relocations_with_risk": learned["relocations"],
                "relocations_risk_zero": zero["relocations"],
                "reward_with_risk": learned["episode_reward"],
                "source_artifact": f"python-ai/saved_models/marl/{filename}",
            }
        )
    output = pd.DataFrame(rows)
    output["bc_risk_sweep_span"] = 0.3935
    output["bc_source"] = "python-ai/saved_models/marl/SPRINT_6_5_REPORT.md §7"
    return output


def build_trajectory_data(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    phase5 = data["phase5"]
    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    dynamics_rows: list[dict[str, Any]] = []
    source_sizes = phase5["state_sets"]
    for item in phase5["trajectory"]:
        stats = item["recorded_stats"] or {}
        for source in ("RANDOM", "UNION"):
            metrics = item["sources"][source]
            new = metrics.get("_NEW_not_in_frozen_artifact", {})
            rows.append(
                {
                    "update": item["update"],
                    "episode": item["episode"],
                    "checkpoint": item["file"],
                    "checkpoint_md5": item["md5"],
                    "lr_scale": item["lr_scale"],
                    "state_set": source,
                    "n_decision": source_sizes[source]["n_decision"],
                    "n_highrisk": source_sizes[source]["n_highrisk"],
                    "n_lowrisk": source_sizes[source]["n_risk_lt_02"],
                    "delta_edge": metrics["risk_response_high_minus_low"],
                    "p_edge_high_risk": metrics["p_edge_risk_ge_06"],
                    "p_edge_low_risk": metrics["p_edge_risk_lt_02"],
                    "greedy_edge_high_risk": metrics["frac_argmax_edge"],
                    "greedy_edge_low_risk": new.get("argmax_edge_risk_lt_02"),
                    "stochastic_entropy_high_risk": metrics["entropy_mean"],
                    "stochastic_entropy_all_decision": new.get("entropy_mean_all_decision"),
                    "max_probability_all_decision": new.get("maxp_mean_all_decision"),
                    "fraction_max_probability_gt_099_all_decision": new.get("frac_maxp_gt_099_all_decision"),
                    "source_artifact": "python-ai/saved_models/marl/SPRINT_7_PHASE5_risk_trajectory_main.json",
                }
            )
            for action, delta in new.get("risk_response_by_action", {}).items():
                action_rows.append(
                    {
                        "update": item["update"],
                        "state_set": source,
                        "action": action,
                        "high_minus_low_probability": delta,
                        "source_artifact": "python-ai/saved_models/marl/SPRINT_7_PHASE5_risk_trajectory_main.json",
                    }
                )
        if stats:
            dynamics_rows.append(
                {
                    "update": item["update"],
                    "episode": item["episode"],
                    "actor_loss": stats["actor_loss"],
                    "critic_loss": stats["critic_loss"],
                    "entropy": stats["entropy"],
                    "approx_kl": stats["approx_kl"],
                    "clip_frac": stats["clip_frac"],
                    "explained_var": stats["explained_var"],
                    "decision_frac": stats["decision_frac"],
                    "source_artifact": "python-ai/saved_models/marl/SPRINT_7_PHASE5_risk_trajectory_main.json (recorded_stats)",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(action_rows), pd.DataFrame(dynamics_rows)


def build_complexity_data(data: dict[str, Any]) -> pd.DataFrame:
    config = data["config"]["config"]
    pth = MARL / "mappo.pth"
    best = MARL / "mappo_best.pth"
    rows = [
        ("actors", "number of separate actors", 10, "count", "mappo_config.json / run_mappo_train.log", "documented"),
        ("actor", "parameters per actor", 23300, "parameters", "run_mappo_train.log", "documented"),
        ("actor", "parameters across actors", 233000, "parameters", "run_mappo_train.log", "documented"),
        ("critic", "centralised critic parameters", 194049, "parameters", "run_mappo_train.log", "documented"),
        ("combined", "relevant trainable parameters", 427049, "parameters", "233000 + 194049", "derived sum"),
        ("actor", "hidden layers", "[128, 128]", "layer widths", "mappo_config.json", "documented"),
        ("critic", "hidden layers", "[256, 256]", "layer widths", "mappo_config.json", "documented"),
        ("actor", "actions per actor", 4, "actions", "mappo_config.json / run_mappo_train.log", "documented"),
        ("checkpoint", "mappo.pth file size", pth.stat().st_size, "bytes", "mappo.pth", "observed file size"),
        ("checkpoint", "mappo_best.pth file size", best.stat().st_size, "bytes", "mappo_best.pth", "observed file size"),
        ("complexity", "FLOPs", "unavailable", "", "no completed artifact provides a FLOP measurement", "not measured"),
    ]
    return pd.DataFrame(rows, columns=["component", "measure", "value", "unit", "source", "value_status"])


def build_configuration_table(data: dict[str, Any]) -> pd.DataFrame:
    config = data["config"]["config"]
    metadata = read_json(PREDICTOR / "failure_predictor_meta.json")
    source = data["sim_manager_text"]
    sim_seed = re.search(r"SIM_SEED\s*=\s*(\d+)L", source)
    max_seconds = re.search(r"MAX_SIMULATION_SECONDS\s*=\s*([0-9.]+)", source)
    device_count = re.search(r"DEVICE_COUNT\s*=\s*(\d+)", source)
    rows = [
        ("Simulation", "SIM_SEED", sim_seed.group(1) if sim_seed else "not retained", "SimulationManager.java"),
        ("Simulation", "maximum simulation seconds", max_seconds.group(1) if max_seconds else "not retained", "SimulationManager.java"),
        ("Simulation", "device count", device_count.group(1) if device_count else "not retained", "SimulationManager.java"),
        ("Failure predictor", "sequence length", metadata["sequence_length"], "failure_predictor_meta.json"),
        ("Failure predictor", "feature count", metadata["num_features"], "failure_predictor_meta.json"),
        ("Failure predictor", "threshold", metadata["threshold"], "failure_predictor_meta.json"),
        ("MAPPO", "seed", config["train"]["seed"], "mappo_config.json"),
        ("MAPPO", "device", config["train"]["device"], "mappo_config.json"),
        ("MAPPO", "training episodes", config["train"]["episodes"], "mappo_config.json"),
        ("MAPPO", "rollout episodes/update", config["train"]["rollout_episodes"], "mappo_config.json"),
        ("MAPPO", "PPO updates", config["train"]["episodes"] // config["train"]["rollout_episodes"], "derived from mappo_config.json"),
        ("MAPPO", "episode steps", config["env"]["episode_steps"], "mappo_config.json"),
        ("MAPPO", "gamma", config["mappo"]["gamma"], "mappo_config.json"),
        ("MAPPO", "gae lambda", config["mappo"]["gae_lambda"], "mappo_config.json"),
        ("MAPPO", "risk source", config["env"]["risk_source"], "mappo_config.json"),
        ("MAPPO", "actors", "10 separate MLP(48 -> [128, 128] -> 4)", "run_mappo_train.log"),
        ("MAPPO", "centralised critic", "MLP(489+10 -> [256, 256] -> 1)", "run_mappo_train.log"),
        ("MAPPO", "action space", "STAY; MIGRATE_TO_NEIGHBOR_EDGE; MIGRATE_TO_CLOUD; PREEMPTIVE_REROUTE", "run_mappo_train.log"),
        ("Evaluation", "fixed starts", "[491, 521, 550, 580, 609, 639, 668, 698]", "mappo_eval.json"),
    ]
    return pd.DataFrame(rows, columns=["system_component", "setting", "value", "source_artifact"])


def build_failure_table(performance: pd.DataFrame, lead: pd.DataFrame, verdicts: pd.DataFrame) -> pd.DataFrame:
    metrics = performance[performance["measure_group"] == "pooled_oof"].set_index("metric")["value"]
    early = lead[lead["verdict"] == "EARLY_WARNING"]
    rows = [
        ("Pooled OOF", "observations", int(metrics["n"]), "host_metrics.json"),
        ("Pooled OOF", "positive windows", int(metrics["positives"]), "host_metrics.json"),
        ("Pooled OOF", "PR-AUC", metrics["pr_auc"], "host_metrics.json"),
        ("Pooled OOF", "ROC-AUC", metrics["roc_auc"], "host_metrics.json"),
        ("Pooled OOF", "F1 at threshold 0.18", metrics["f1"], "host_metrics.json"),
        ("Pooled OOF", "precision at threshold 0.18", metrics["precision"], "host_metrics.json"),
        ("Pooled OOF", "recall at threshold 0.18", metrics["recall"], "host_metrics.json"),
        ("Lead time", "EARLY_WARNING events", int(len(early)), "host_lead_time.csv"),
        ("Lead time", "mean lead seconds (EARLY_WARNING)", early["lead_seconds"].mean(), "host_lead_time.csv"),
        ("Lead time", "median lead seconds (EARLY_WARNING)", early["lead_seconds"].median(), "host_lead_time.csv"),
    ]
    for row in verdicts.itertuples(index=False):
        rows.append(("Lead time", f"{row.verdict} events", row.n_events, "host_lead_time.csv"))
    return pd.DataFrame(rows, columns=["result_group", "metric", "value", "source_artifact"])


def build_mechanism_table(trajectory: pd.DataFrame, dynamics: pd.DataFrame) -> pd.DataFrame:
    rows: list[tuple[str, str, str, str]] = []
    for state_set in ("RANDOM", "UNION"):
        segment = trajectory[trajectory["state_set"] == state_set].sort_values("update")
        first, final = segment.iloc[0], segment.iloc[-1]
        rows.extend(
            [
                (f"{state_set} p_hi", f"{first.p_edge_high_risk:.4f} -> {final.p_edge_high_risk:.4f}", "Phase 5 trajectory", "direct field across u000-u075"),
                (f"{state_set} p_lo", f"{first.p_edge_low_risk:.4f} -> {final.p_edge_low_risk:.4f}", "Phase 5 trajectory", "direct field across u000-u075"),
                (f"{state_set} Delta_EDGE", f"{first.delta_edge:+.4f} -> {final.delta_edge:+.4f}", "Phase 5 trajectory", "derived as p_hi - p_lo; stored directly"),
                (f"{state_set} greedy high-risk EDGE", f"{first.greedy_edge_high_risk:.4f} -> {final.greedy_edge_high_risk:.4f}", "Phase 5 trajectory", "direct argmax field"),
            ]
        )
    ordered_dynamics = dynamics.sort_values("update").reset_index(drop=True)
    zero_from = next(
        int(ordered_dynamics.loc[index, "update"])
        for index in range(len(ordered_dynamics))
        if ordered_dynamics.loc[index:, "clip_frac"].eq(0).all()
    )
    rows.extend(
        [
            ("R2 actor trust-region activity", f"clip_frac is zero from u{zero_from:03d}", "Phase 5 report / trajectory", "temporal ordering only; n=1 arm"),
            ("Low-risk dilution", "cos(g_full, g_lo) = +0.9099 to +0.9994 in 9/9 cells", "Final synthesis audit", "supported; not an intervention result"),
            ("Net Delta_EDGE interpretation", "159% RANDOM / 218% UNION of net growth attributed to low-risk suppression", "Final synthesis audit", "net trajectory result; a later u042-u062 acquisition phase is also documented"),
            ("R2 vs R3 explanation", "unresolved", "Final synthesis audit", "descriptive differences are not a causal account"),
        ]
    )
    return pd.DataFrame(rows, columns=["finding", "quantitative_evidence", "source_artifact", "interpretation_limit"])


def build_evidence_table() -> pd.DataFrame:
    rows = [
        ("OOF risk signal is leakage-safe but uncalibrated", "failure_dataset.py safeguards; mappo_config.json risk metadata", "SUPPORTED"),
        ("risk-threshold@0.18 outperforms Sprint 6 MAPPO on held-out protection", "mappo_eval.json: 30.25 vs 3.25 protected", "SUPPORTED"),
        ("A0 reproduces Sprint 6 recorded behavior", "A0 and original history/update artifacts byte-identical; matching held-out histogram", "SUPPORTED"),
        ("Sprint 6.5 improvement in protection is risk targeting", "zero-risk ablations retain most protection while migration volume differs", "RULED OUT"),
        ("R2 fixes the full behavioral outcome", "R2 improves diagnostic sign error but held-out task performance is -0.46", "RULED OUT"),
        ("Net Delta_EDGE growth is high-risk acquisition", "Phase 5: p_hi ends below initialization; low-risk suppression dominates net growth", "RULED OUT"),
        ("u042-u062 contains a high-risk acquisition phase", "Phase 5 / final audit: p_hi rises on both fixed sets during this interval", "SUPPORTED"),
        ("Low-risk dilution sets the update direction", "Rung 3: cos(g_full, g_lo) +0.9099 to +0.9994 in 9/9 cells", "SUPPORTED"),
        ("Softmax saturation is the primary causal mechanism", "Saturation follows Delta_EDGE timing and lower saturation in R3 did not improve behavior", "PARTIALLY SUPPORTED / DOWNGRADED"),
        ("Why R2 exceeds R3", "Final synthesis audit explicitly retains no causal account", "UNRESOLVED"),
        ("Multi-seed generalization", "Only one retained seed per arm for the central ladder", "NOT MEASURED"),
        ("Sprints 8-12 results", "No completed artifacts", "NOT MEASURED"),
        ("FLOPs or execution-complexity result", "No completed artifact supplies a FLOP measurement", "NOT MEASURED"),
    ]
    return pd.DataFrame(rows, columns=["finding", "evidence", "status"])


def figure_1_failure_prediction(performance: pd.DataFrame, verdicts: pd.DataFrame, figures: Path) -> None:
    pooled = performance[performance["measure_group"] == "pooled_oof"].set_index("metric")["value"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    quality_names = ["PR-AUC", "ROC-AUC", "F1", "Precision", "Recall"]
    quality_values = [pooled["pr_auc"], pooled["roc_auc"], pooled["f1"], pooled["precision"], pooled["recall"]]
    bars = axes[0].bar(quality_names, quality_values, color=COLORS["mappo"])
    axes[0].axhline(pooled["pr_auc_baseline"], color="black", linewidth=1, linestyle="--", label="positive-rate baseline")
    axes[0].set_ylim(0, 1.0)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Pooled OOF predictor performance\n(n=14,910 windows; 320 positives)")
    axes[0].tick_params(axis="x", rotation=25)
    axes[0].legend(frameon=False, loc="upper right")
    for bar, value in zip(bars, quality_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    order = ["EARLY_WARNING", "STANDING_ALARM", "MISSED"]
    counts = [int(verdicts.loc[verdicts["verdict"].eq(label), "n_events"].sum()) for label in order]
    bars = axes[1].bar(order, counts, color=[COLORS["random"], COLORS["threshold"], COLORS["static"]])
    axes[1].set_ylabel("Failure events (n)")
    axes[1].set_title("Recorded failure-warning outcomes")
    axes[1].tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.3, str(value), ha="center", va="bottom", fontsize=8)
    fig.suptitle("Figure 1. Host-failure prediction evidence", y=1.03, fontsize=11)
    save_figure(fig, "figure_1_failure_prediction_performance", figures)


def figure_2_lead_time(lead: pd.DataFrame, figures: Path) -> None:
    early = lead.loc[lead["verdict"].eq("EARLY_WARNING") & lead["lead_seconds"].notna(), "lead_seconds"].sort_values().to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    y = np.arange(1, len(early) + 1) / len(early)
    ax.step(early, y, where="post", color=COLORS["mappo"], linewidth=2)
    ax.scatter(early, y, color=COLORS["mappo"], s=18, zorder=3)
    ax.set_xlabel("Lead time before failure (s)")
    ax.set_ylabel("Empirical cumulative probability")
    ax.set_ylim(0, 1.03)
    ax.set_title(f"Figure 2. EARLY_WARNING lead-time distribution (n={len(early)})")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "figure_2_failure_warning_lead_time", figures)


def figure_3_training_reward(history: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    ax.plot(history["episode"], history["reward"], color=COLORS["mappo"], linewidth=1.0)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward")
    ax.set_title("Figure 3. Sprint 6 MAPPO training reward (all 600 episodes)")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "figure_3_mappo_training_reward", figures)


def figure_4_learning_dynamics(updates: pd.DataFrame, figures: Path) -> None:
    metrics = [
        ("entropy", "Entropy"),
        ("explained_var", "Explained variance"),
        ("approx_kl", "Approximate KL (signed k1)"),
        ("clip_frac", "Clip fraction"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 5.4), sharex=True)
    for ax, (column, label) in zip(axes.flat, metrics):
        ax.plot(updates["update"], updates[column], color=COLORS["mappo"], linewidth=1.4)
        if column == "approx_kl":
            ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        ax.set_title(label)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("PPO update (n=75)")
    fig.suptitle("Figure 4. Sprint 6 MAPPO learning dynamics", y=1.01, fontsize=11)
    fig.tight_layout()
    save_figure(fig, "figure_4_mappo_learning_dynamics", figures)


def figure_5_evaluation(evaluation: pd.DataFrame, figures: Path) -> None:
    order = ["mappo-greedy", "static-no-migration", "random-legal", "reactive-threshold", "risk-threshold@0.18", "risk-threshold@0.5"]
    frame = evaluation.set_index("policy").loc[order].reset_index()
    labels = [label.replace("risk-threshold@", "risk @") for label in frame["policy"]]
    colors = [COLORS["mappo"], COLORS["static"], COLORS["random"], COLORS["reactive"], COLORS["threshold"], COLORS["threshold"]]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    axes[0].barh(labels, frame["episode_reward"], color=colors)
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Episode reward")
    axes[0].set_xlabel("Mean over 8 held-out episodes")
    axes[1].barh(labels, frame["tasks_protected_before_failure"], color=colors, label="Protected")
    axes[1].set_title("Tasks protected before failure")
    axes[1].set_xlabel("Mean count")
    axes[2].barh(labels, frame["lost"], color=colors)
    axes[2].set_title("Tasks lost")
    axes[2].set_xlabel("Mean count")
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("Figure 5. Held-out Sprint 6 policy comparison", y=1.03, fontsize=11)
    fig.tight_layout()
    save_figure(fig, "figure_5_mappo_vs_baseline_evaluation", figures)


def figure_6_s65_risk_sensitivity(s65: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    positions = np.arange(len(s65))
    bars = ax.bar(positions, s65["risk_sensitivity_span"], color=[COLORS[x.lower()] for x in s65["arm"]])
    ax.axhline(0.3935, color="black", linewidth=1.2, linestyle="--", label="Behaviour-cloned threshold ceiling (0.3935)")
    ax.set_xticks(positions, s65["arm"])
    ax.set_xlabel("Sprint 6.5 arm")
    ax.set_ylabel("P(relocate) span over risk sweep")
    ax.set_title("Figure 6. Sprint 6.5 risk-sensitivity comparison")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    for bar, corr in zip(bars, s65["risk_sensitivity_correlation"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003, f"r={corr:.3f}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, "figure_6_sprint65_risk_sensitivity", figures)


def figure_7_s65_zero_risk(s65: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    positions = np.arange(len(s65))
    for position, row in enumerate(s65.itertuples(index=False)):
        ax.plot([position - 0.12, position + 0.12], [row.protected_with_risk, row.protected_risk_zero], color="black", linewidth=1, zorder=1)
        ax.scatter(position - 0.12, row.protected_with_risk, color=COLORS[row.arm.lower()], s=50, zorder=2, label="With risk" if position == 0 else None)
        ax.scatter(position + 0.12, row.protected_risk_zero, facecolors="white", edgecolors=COLORS[row.arm.lower()], linewidths=1.8, s=50, zorder=2, label="Risk=0" if position == 0 else None)
    ax.set_xticks(positions, s65["arm"])
    ax.set_xlabel("Sprint 6.5 arm")
    ax.set_ylabel("Tasks protected before failure")
    ax.set_title("Figure 7. Zero-risk ablation: physical protection (8 held-out episodes/arm)")
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "figure_7_sprint65_zero_risk_ablation", figures)


def figure_8_delta_trajectory(trajectory: pd.DataFrame, figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.0))
    for source, color in (("RANDOM", COLORS["random_set"]), ("UNION", COLORS["union_set"])):
        frame = trajectory[trajectory["state_set"] == source]
        ax.plot(frame["update"], frame["delta_edge"], label=source, color=color, linewidth=1.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("R2 update (u000-u075)")
    ax.set_ylabel(r"$\Delta_{EDGE}=\pi(EDGE|risk\geq0.6)-\pi(EDGE|risk<0.2)$")
    ax.set_title("Figure 8. Sprint 7 R2 Δ_EDGE trajectory on frozen state sets")
    ax.legend(title="State set", frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, "figure_8_sprint7_delta_edge_trajectory", figures)


def figure_9_high_low_probability(trajectory: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    for ax, source in zip(axes, ("RANDOM", "UNION")):
        frame = trajectory[trajectory["state_set"] == source]
        ax.plot(frame["update"], frame["p_edge_high_risk"], color=COLORS["high"], label=r"$p_{hi}=\pi(EDGE|risk\geq0.6)$")
        ax.plot(frame["update"], frame["p_edge_low_risk"], color=COLORS["low"], label=r"$p_{lo}=\pi(EDGE|risk<0.2)$")
        ax.set_title(f"{source} frozen set")
        ax.set_xlabel("R2 update")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean stochastic EDGE probability")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Figure 9. High-risk and low-risk EDGE probability across the complete R2 trajectory", y=1.03, fontsize=11)
    fig.tight_layout()
    save_figure(fig, "figure_9_sprint7_high_low_edge_probability", figures)


def figure_10_temporal_dynamics(trajectory: pd.DataFrame, dynamics: pd.DataFrame, figures: Path) -> None:
    panels = [
        ("clip_frac", "Clip fraction"),
        ("entropy", "On-policy entropy"),
        ("explained_var", "Explained variance"),
        ("approx_kl", "Approximate KL (signed k1)"),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(8.0, 9.0), sharex=True)
    for ax, (column, label) in zip(axes[:4], panels):
        ax.plot(dynamics["update"], dynamics[column], color=COLORS["mappo"], linewidth=1.4)
        if column == "approx_kl":
            ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.25)
    for source, color in (("RANDOM", COLORS["random_set"]), ("UNION", COLORS["union_set"])):
        frame = trajectory[trajectory["state_set"] == source]
        axes[4].plot(frame["update"], frame["delta_edge"], color=color, label=f"Δ_EDGE {source}", linewidth=1.5)
    axes[4].axhline(0, color="black", linewidth=0.8)
    axes[4].set_ylabel("Δ_EDGE")
    axes[4].set_xlabel("R2 update")
    axes[4].legend(frameon=False, ncol=2)
    axes[4].grid(axis="y", alpha=0.25)
    fig.suptitle("Figure 10. Sprint 7 diagnostic temporal evidence (ordering, not causality)", y=0.995, fontsize=11)
    fig.tight_layout()
    save_figure(fig, "figure_10_sprint7_temporal_learning_dynamics", figures)


def figure_11_stochastic_greedy(trajectory: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharey=True)
    for ax, source in zip(axes, ("RANDOM", "UNION")):
        frame = trajectory[trajectory["state_set"] == source]
        ax.plot(frame["update"], frame["p_edge_high_risk"], color=COLORS["high"], label="Stochastic π(EDGE | high risk)", linewidth=1.7)
        ax.plot(frame["update"], frame["greedy_edge_high_risk"], color=COLORS["mappo"], linestyle="--", label="Greedy argmax EDGE rate", linewidth=1.7)
        n_hi = int(frame["n_highrisk"].iloc[0])
        ax.set_title(f"{source} (high-risk n={n_hi})")
        ax.set_xlabel("R2 update")
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Rate across high-risk entries")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle("Figure 11. Stochastic versus greedy high-risk EDGE behaviour", y=1.03, fontsize=11)
    fig.tight_layout()
    save_figure(fig, "figure_11_stochastic_vs_greedy_edge", figures)


def add_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str, width: float = 0.21, height: float = 0.11) -> None:
    x, y = xy
    box = plt.Rectangle((x, y), width, height, facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.18)
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=8, wrap=True)


def figure_12_mechanism_summary(figures: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    ax.set_axis_off()
    established = COLORS["random"]
    ruled = COLORS["threshold"]
    unresolved = COLORS["static"]
    chain = [
        (0.05, 0.72, "Failure predictor\nLeakage-safe OOF risk;\nPR-AUC 0.2863", established),
        (0.29, 0.72, "Risk signal in\ntrace-driven replay\n(uncalibrated)", established),
        (0.53, 0.72, "Sprint 6 MAPPO\n35.875 relocations;\n3.25 protected", established),
        (0.77, 0.72, "Threshold baseline\n36.25 relocations;\n30.25 protected", established),
        (0.29, 0.43, "Sprint 6.5\nA0-A3: low sweep span;\nzero-risk protection retained", established),
        (0.53, 0.43, "Sprint 7 trajectory\nΔ_EDGE mostly low-risk\nsuppression overall", established),
        (0.77, 0.43, "Later evidence\nlow-risk dilution\nsupported", established),
    ]
    for x, y, text, color in chain:
        add_box(ax, (x, y), text, color)
    arrows = [
        ((0.26, 0.775), (0.29, 0.775)),
        ((0.50, 0.775), (0.53, 0.775)),
        ((0.74, 0.775), (0.77, 0.775)),
        ((0.875, 0.72), (0.40, 0.54)),
        ((0.50, 0.485), (0.53, 0.485)),
        ((0.74, 0.485), (0.77, 0.485)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "black"})
    ax.text(0.05, 0.27, "Established findings", fontsize=10, fontweight="bold")
    ax.text(0.05, 0.21, "• The threshold rule outperformed MAPPO on held-out protection at comparable migration volume.", fontsize=8)
    ax.text(0.05, 0.16, "• A0 reproduced the retained Sprint 6 behavior; the R2 trajectory was scored on frozen sets.", fontsize=8)
    ax.text(0.05, 0.11, "• Net Δ_EDGE growth is mostly low-risk suppression; u042-u062 includes a documented high-risk acquisition phase.", fontsize=8)
    ax.text(0.05, 0.03, "Ruled out / downgraded: reward absence, representation incapacity, simple variance explanation, and saturation as the primary cause.", fontsize=8, color=ruled)
    ax.text(0.05, -0.03, "Unresolved: causal account of R2 > R3; multi-seed generalization; FLOPs/execution-complexity claims.", fontsize=8, color=unresolved)
    ax.set_xlim(0, 1.03)
    ax.set_ylim(-0.08, 1.0)
    fig.suptitle("Figure 12. Completed evidence chain for DT-MARL-Healthcare", y=0.98, fontsize=11)
    save_figure(fig, "figure_12_final_mechanism_summary", figures)


def build_output_matrix() -> str:
    rows = [
        ("A-F1", "Figure 1: Failure prediction performance", "How strong is the retained host-failure predictor evidence?", "host_metrics.json; host_lead_time.csv", "Pooled OOF PR-AUC, ROC-AUC, F1/precision/recall; warning verdict counts", "Derived", "Flatten existing JSON and count existing CSV verdicts", "Descriptive", "Single completed predictor run; OOF risk uncalibrated", "figures/figure_1_failure_prediction_performance.png"),
        ("A-F2", "Figure 2: Failure-warning lead time", "How early were recorded EARLY_WARNING events?", "host_lead_time.csv", "lead (seconds), EARLY_WARNING", "Derived", "ECDF over non-null EARLY_WARNING lead values", "Descriptive", "Event-level file; n is plotted", "figures/figure_2_failure_warning_lead_time.png"),
        ("B-T1", "Table 1: System / experiment configuration", "What completed apparatus produced the evidence?", "SimulationManager.java; failure_predictor_meta.json; mappo_config.json; run_mappo_train.log", "Seeds, simulator, predictor, MAPPO settings", "Mixed", "Direct extraction; PPO update count derived as 600/8", "Descriptive", "Exact historical Maven invocation not retained", "tables/table_1_system_experiment_configuration.{csv,md}"),
        ("C-F3", "Figure 3: MAPPO training reward", "What did the retained Sprint 6 training trajectory record?", "mappo_history.csv", "episode, reward", "Already exists", "Plot all 600 recorded rows without smoothing", "Descriptive", "Episode-600 log/CSV discrepancy retained in summary", "figures/figure_3_mappo_training_reward.png"),
        ("C-F4", "Figure 4: MAPPO learning dynamics", "How did key PPO statistics evolve in Sprint 6?", "mappo_updates.csv", "entropy, explained_var, approx_kl, clip_frac", "Already exists", "Four separate panels, one common update axis", "Descriptive", "No causal interpretation", "figures/figure_4_mappo_learning_dynamics.png"),
        ("D-F5", "Figure 5: MAPPO vs baseline evaluation", "Did MAPPO beat existing held-out policies?", "mappo_eval.json", "reward, protected, lost", "Already exists", "Flatten policy JSON; preserve 8-episode means", "Comparative", "Reward comparison is within Sprint 6 evaluation only", "figures/figure_5_mappo_vs_baseline_evaluation.png"),
        ("D-T4", "Table 4: Held-out evaluation", "What exact held-out policy values were recorded?", "mappo_eval.json", "reward, relocations, protected, lost, success; MAPPO action counts", "Mixed", "Flatten policies and append MAPPO action counts", "Comparative", "Action counts are only recorded for MAPPO", "tables/table_4_held_out_evaluation.{csv,md}"),
        ("E-F6", "Figure 6: Sprint 6.5 risk sensitivity", "Did A0-A3 acquire meaningful risk sensitivity?", "mappo_A{0,1,2,3}_*_eval.json; SPRINT_6_5_REPORT.md", "probe_risk.p_relocate_span; correlation; BC span 0.3935", "Mixed", "Extract arm probes; report BC ceiling from report §7", "Diagnostic", "One seed per arm; BC ceiling is report-only", "figures/figure_6_sprint65_risk_sensitivity.png"),
        ("E-F7", "Figure 7: Sprint 6.5 zero-risk ablation", "Did protection depend on the risk input?", "mappo_A{0,1,2,3}_*_eval.json", "tasks_protected_before_failure, ablation_risk_zero", "Already exists", "Paired arm-level plot of with-risk and risk=0 physical protection", "Diagnostic", "Ablation changes three components; reward is not isolating", "figures/figure_7_sprint65_zero_risk_ablation.png"),
        ("E-T5", "Table 5: Sprint 6.5 A0-A3", "What did the controlled arms record?", "A0-A3 eval JSON; SPRINT_6_5_REPORT.md", "risk span/correlation, criticality direction, zero-risk protection, physical outcomes", "Mixed", "Direct fields; direction derived from correlation sign", "Diagnostic", "Rewards are not comparable across all arms", "tables/table_5_sprint65_results.{csv,md}"),
        ("F-F8", "Figure 8: Sprint 7 Δ_EDGE trajectory", "How did Δ_EDGE evolve over all stored R2 checkpoints?", "SPRINT_7_PHASE5_risk_trajectory_main.json", "risk_response_high_minus_low, u000-u075", "Already exists", "Flatten all 76 records for RANDOM and UNION", "Diagnostic", "Frozen state sets; one arm", "figures/figure_8_sprint7_delta_edge_trajectory.png"),
        ("F-F9", "Figure 9: High- vs low-risk EDGE probability", "Which Δ_EDGE component changed?", "SPRINT_7_PHASE5_risk_trajectory_main.json", "p_edge_risk_ge_06, p_edge_risk_lt_02", "Already exists", "Plot both fields across all 76 checkpoints", "Diagnostic", "Mean probability, not an outcome metric", "figures/figure_9_sprint7_high_low_edge_probability.png"),
        ("F-F10", "Figure 10: Temporal learning evidence", "What ordering do recorded training statistics and Δ_EDGE show?", "Phase 5 trajectory recorded_stats", "clip_frac, entropy, explained_var, approx_kl, Δ_EDGE", "Already exists", "Separate panels aligned by update", "Diagnostic", "Temporal ordering is explicitly non-causal", "figures/figure_10_sprint7_temporal_learning_dynamics.png"),
        ("F-F11", "Figure 11: Stochastic vs greedy EDGE", "How do mean probability and greedy action channels differ?", "Phase 5 trajectory", "p_edge_risk_ge_06, frac_argmax_edge", "Already exists", "Plot complete high-risk series by frozen state set", "Diagnostic", "Different behavioral channels; do not substitute one for the other", "figures/figure_11_stochastic_vs_greedy_edge.png"),
        ("F-F12", "Figure 12: Final mechanism summary", "What evidence chain is completed and what remains open?", "Sprint 6.5 report; Sprint 7 Phase 5 report; final synthesis audit", "documented findings/statuses", "Derived", "Evidence-only conceptual diagram", "Diagnostic synthesis", "Not a new causal model", "figures/figure_12_final_mechanism_summary.png"),
        ("F-T6", "Table 6: Sprint 7 mechanism evidence", "What are the strongest completed trajectory and mechanism readings?", "Phase 5 trajectory; final synthesis audit", "trajectory endpoints, clip timing, dilution evidence", "Mixed", "Direct trajectory endpoints; report-stated findings", "Diagnostic", "R2/R3 causal difference remains unresolved", "tables/table_6_sprint7_mechanism_evidence.{csv,md}"),
        ("G-T3", "Table 3: MAPPO architecture and complexity", "What model scale is directly documented?", "run_mappo_train.log; mappo_config.json; checkpoint files", "parameter counts, layers, checkpoint byte sizes", "Mixed", "Direct values; total parameters is a stated sum", "Descriptive", "FLOPs unavailable", "tables/table_3_mappo_architecture_complexity.{csv,md}"),
        ("H-T7", "Table 7: Final evidence status", "Which project claims are supported, ruled out, or open?", "final synthesis audit and cited artifacts", "finding/evidence/status", "Mixed", "Transcribe documented status with source-linked evidence", "Synthesis", "Statuses are not new experiments", "tables/table_7_final_evidence_status.{csv,md}"),
        ("H-S1", "Publication Results Summary", "What can the paper safely state?", "All output sources above", "completed quantitative findings and limitations", "Derived", "Traceable narrative synthesis", "Synthesis", "No multi-seed or Sprints 8-12 claims", "PAPER_RESULTS_SUMMARY.md"),
    ]
    columns = ["Output ID", "Proposed paper figure/table", "Research question", "Source artifact(s)", "Exact metric(s)", "Value status", "Computation performed", "Type", "Limitations", "Final output filename"]
    frame = pd.DataFrame(rows, columns=columns)
    grouped = []
    section_titles = [
        ("A-", "A. Failure prediction"), ("B-", "B. System / architecture"), ("C-", "C. MAPPO training"),
        ("D-", "D. Sprint 6 baseline evaluation"), ("E-", "E. Sprint 6.5 ablation/diagnostics"),
        ("F-", "F. Sprint 7 mechanism analysis"), ("G-", "G. Model complexity"), ("H-", "H. Overall research summary"),
    ]
    for prefix, title in section_titles:
        grouped.append(f"## {title}\n")
        grouped.append(markdown_table(frame[frame["Output ID"].str.startswith(prefix)]) + "\n")
    return "# Paper Output Matrix\n\nAll outputs are additive derivations from completed artifacts. `Value status` distinguishes direct fields from derived values; unavailable metrics are named rather than substituted.\n\n" + "\n".join(grouped)


def build_results_summary(failure_table: pd.DataFrame, evaluation: pd.DataFrame, s65: pd.DataFrame, complexity: pd.DataFrame) -> str:
    pooled = failure_table.set_index("metric")["value"]
    eval_index = evaluation.set_index("policy")
    a0 = s65.set_index("arm").loc["A0"]
    return f"""# Publication Results Summary

## 1. Failure Prediction

- The retained pooled out-of-fold predictor result has PR-AUC {float(pooled['PR-AUC']):.4f}, ROC-AUC {float(pooled['ROC-AUC']):.4f}, and F1 {float(pooled['F1 at threshold 0.18']):.4f} at threshold 0.18 (n={int(pooled['observations']):,} windows; {int(pooled['positive windows'])} positives).
- The lead-time figure uses only recorded `EARLY_WARNING` events from `host_lead_time.csv`; it reports its sample size directly.
- The OOF risk used by MAPPO is leakage-safe but uncalibrated. This is a predictor-performance result, not a calibrated clinical probability claim.

## 2. MAPPO Baseline

- The completed Sprint 6 MAPPO held-out result is reward {eval_index.loc['mappo-greedy', 'episode_reward']:.4f}, protected {eval_index.loc['mappo-greedy', 'tasks_protected_before_failure']:.2f}, lost {eval_index.loc['mappo-greedy', 'lost']:.3f}, and success rate {eval_index.loc['mappo-greedy', 'task_success_rate']:.5f}.
- `risk-threshold@0.18` records reward {eval_index.loc['risk-threshold@0.18', 'episode_reward']:.4f}, protected {eval_index.loc['risk-threshold@0.18', 'tasks_protected_before_failure']:.2f}, and lost {eval_index.loc['risk-threshold@0.18', 'lost']:.3f} at comparable relocation volume ({eval_index.loc['mappo-greedy', 'relocations']:.3f} versus {eval_index.loc['risk-threshold@0.18', 'relocations']:.3f}).
- The Sprint 6 training plot preserves all 600 recorded rewards without smoothing. The known episode-600 log/CSV discrepancy remains documented in `SPRINT_6_REPORT.md` and is not reconciled here.

## 3. Sprint 6.5

- A0-A3 risk-sweep spans are {', '.join(f"{row.arm}={row.risk_sensitivity_span:.4f}" for row in s65.itertuples(index=False))}; the report-recorded behaviour-cloning ceiling is 0.3935.
- The physical-protection zero-risk ablation is plotted as paired observations, not as a causal reward comparison. For A0, protection is {a0.protected_with_risk:.3f} with risk and {a0.protected_risk_zero:.3f} with risk forced to zero.
- The arm rewards are not compared across all arms because the reward conventions differ; the tables retain this limitation.

## 4. Sprint 7

- The R2 trajectory contains all 76 stored checkpoints (u000-u075), scored on frozen RANDOM (4,975 decision entries; 216 high-risk) and UNION (74,237; 3,592 high-risk) sets.
- `Δ_EDGE` is plotted with its two components. The final audit concludes that its net growth is mostly low-risk suppression, while preserving the documented u042-u062 high-risk acquisition phase.
- The stochastic mean-probability and greedy argmax channels are shown separately; they are not interchangeable.
- The aligned training-dynamics figure is temporal/diagnostic evidence only. It does not claim that clip fraction, entropy, explained variance, or approximate KL caused the behaviour change.

## 5. Complexity

- The training log documents 23,300 parameters per actor, 233,000 across 10 separate actors, and 194,049 in the centralised critic; 427,049 is their arithmetic sum.
- Checkpoint byte sizes are reported as file observations. FLOPs and execution-complexity results are not available and are not estimated.

## 6. Strongest Paper Findings

1. The leakage-safe OOF host-failure predictor achieved PR-AUC {float(pooled['PR-AUC']):.4f} on {int(pooled['observations']):,} windows.
2. On held-out Sprint 6 starts, the threshold baseline protected {eval_index.loc['risk-threshold@0.18', 'tasks_protected_before_failure']:.2f} tasks versus MAPPO's {eval_index.loc['mappo-greedy', 'tasks_protected_before_failure']:.2f} at nearly equal relocation volume.
3. The A0 control reproduces the retained Sprint 6 action histogram and has byte-identical history/update artifacts.
4. Sprint 6.5 risk-sweep spans remain far below the report-recorded 0.3935 behaviour-cloning ceiling.
5. Zero-risk ablations show that observed arm-level protection was not established as risk-targeted protection.
6. Across the complete R2 trajectory, `Δ_EDGE` must be interpreted through both `p_hi` and `p_lo`; net growth is mostly low-risk suppression.
7. The final audit records low-risk dilution as supported and retains the causal explanation for R2 > R3 as unresolved.

## 7. Claims We MUST NOT Make

- Multi-seed generalization: the principal arms retain one seed each.
- A causal claim from temporal precedence in the R2 trajectory.
- That net `Δ_EDGE` growth alone demonstrates high-risk acquisition or self-healing.
- That the later Sprint 7 mechanism was established by Sprint 6 alone.
- Reward comparisons across arms with different reward conventions.
- FLOPs, runtime-complexity, or execution-cost claims not recorded in completed artifacts.
- Any result for Sprints 8-12; no completed evidence exists for those stages.
- A reconciliation of the Sprint 6 episode-600 log/CSV discrepancy.
"""


def generate(output_root: Path) -> list[Path]:
    output_root = output_root.resolve()
    if not output_root.is_relative_to(PAPER_ROOT.resolve()):
        raise ValueError("--output-root must be inside paper_results/ to protect original artifacts")
    figures = output_root / "figures"
    tables = output_root / "tables"
    derived = output_root / "data"
    for directory in (figures, tables, derived):
        directory.mkdir(parents=True, exist_ok=True)

    data = load_inputs()
    performance, lead, verdicts = build_failure_data(data)
    evaluation = build_evaluation_data(data)
    s65 = build_s65_data(data)
    trajectory, action_channels, dynamics = build_trajectory_data(data)
    complexity = build_complexity_data(data)
    configuration = build_configuration_table(data)
    failure_table = build_failure_table(performance, lead, verdicts)
    mechanism_table = build_mechanism_table(trajectory, dynamics)
    evidence_table = build_evidence_table()

    history = data["history"].copy()
    history["source_artifact"] = "python-ai/saved_models/marl/mappo_history.csv"
    updates = data["updates"].copy()
    updates["source_artifact"] = "python-ai/saved_models/marl/mappo_updates.csv"
    action_counts = pd.DataFrame(
        [{"action": action, "count": count, "source_artifact": "python-ai/saved_models/marl/mappo_eval.json"} for action, count in data["evaluation"]["action_histogram"].items()]
    )

    datasets = {
        "failure_prediction_summary.csv": performance,
        "failure_lead_times.csv": lead,
        "failure_warning_verdicts.csv": verdicts,
        "mappo_training_curve.csv": history,
        "mappo_update_metrics.csv": updates,
        "evaluation_comparison.csv": evaluation,
        "mappo_action_counts.csv": action_counts,
        "sprint65_ablation.csv": s65,
        "sprint7_trajectory.csv": trajectory,
        "sprint7_action_channels.csv": action_channels,
        "sprint7_training_dynamics.csv": dynamics,
        "complexity_summary.csv": complexity,
    }
    for filename, frame in datasets.items():
        frame.to_csv(derived / filename, index=False, quoting=csv.QUOTE_MINIMAL)

    held_out_table = evaluation.copy()
    held_out_table["action_counts_note"] = "Recorded for mappo-greedy only: STAY=9832; MIGRATE_TO_NEIGHBOR_EDGE=77; MIGRATE_TO_CLOUD=842; PREEMPTIVE_REROUTE=0"
    write_table(configuration, "table_1_system_experiment_configuration", tables)
    write_table(failure_table, "table_2_failure_prediction_results", tables)
    write_table(complexity, "table_3_mappo_architecture_complexity", tables)
    write_table(held_out_table, "table_4_held_out_evaluation", tables)
    write_table(s65, "table_5_sprint65_results", tables)
    write_table(mechanism_table, "table_6_sprint7_mechanism_evidence", tables)
    write_table(evidence_table, "table_7_final_evidence_status", tables)

    figure_1_failure_prediction(performance, verdicts, figures)
    figure_2_lead_time(lead, figures)
    figure_3_training_reward(history, figures)
    figure_4_learning_dynamics(updates, figures)
    figure_5_evaluation(evaluation, figures)
    figure_6_s65_risk_sensitivity(s65, figures)
    figure_7_s65_zero_risk(s65, figures)
    figure_8_delta_trajectory(trajectory, figures)
    figure_9_high_low_probability(trajectory, figures)
    figure_10_temporal_dynamics(trajectory, dynamics, figures)
    figure_11_stochastic_greedy(trajectory, figures)
    figure_12_mechanism_summary(figures)

    (output_root / "PAPER_OUTPUT_MATRIX.md").write_text(build_output_matrix(), encoding="utf-8")
    (output_root / "PAPER_RESULTS_SUMMARY.md").write_text(
        build_results_summary(failure_table, evaluation, s65, complexity), encoding="utf-8"
    )
    return sorted(path for path in output_root.rglob("*") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create additive publication outputs from completed DT-MARL artifacts.")
    parser.add_argument(
        "--output-root",
        default=str(PAPER_ROOT / "generated"),
        help="directory inside paper_results/ for derived figures, tables, data, and Markdown",
    )
    args = parser.parse_args()
    created = generate(Path(args.output_root))
    print(f"Generated {len(created)} additive publication-result files in {Path(args.output_root).resolve()}")


if __name__ == "__main__":
    main()
