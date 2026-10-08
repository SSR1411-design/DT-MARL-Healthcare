from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

FIGURES.mkdir(parents=True, exist_ok=True)


def save(fig, filename):
    png = FIGURES / f"{filename}.png"
    pdf = FIGURES / f"{filename}.pdf"

    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Created: {png}")
    print(f"Created: {pdf}")


# ============================================================
# FIGURE 1 — Failure Prediction Performance
# ============================================================

df = pd.read_csv(DATA / "failure_prediction_summary.csv")

metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]

pooled = df[df["measure_group"] == "pooled_oof"]

available = [
    m for m in metrics
    if m in pooled["metric"].values
]

if available:
    values = [
        pooled.loc[pooled["metric"] == m, "value"].iloc[0]
        for m in available
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(available, values)

    ax.set_title("Failure Prediction Performance")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)

    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center")

    save(fig, "paper_fig_failure_prediction_performance")


# ============================================================
# FIGURE 2 — Failure Warning Lead Times
# ============================================================

df = pd.read_csv(DATA / "failure_lead_times.csv")

valid = df[df["verdict"] == "EARLY_WARNING"]["lead"].dropna()

if not valid.empty:
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(valid, bins=10)

    ax.axvline(
        valid.mean(),
        linestyle="--",
        label=f"Mean = {valid.mean():.2f}"
    )

    ax.set_title("Early-Warning Lead-Time Distribution")
    ax.set_xlabel("Lead Time")
    ax.set_ylabel("Number of Failure Events")
    ax.legend()

    save(fig, "paper_fig_failure_warning_lead_time")


# ============================================================
# FIGURE 3 — MAPPO Training Reward
# ============================================================

df = pd.read_csv(DATA / "mappo_training_curve.csv")

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    df["episode"],
    df["reward"],
    alpha=0.35,
    linewidth=1
)

rolling = df["reward"].rolling(25).mean()

ax.plot(
    df["episode"],
    rolling,
    linewidth=2,
    label="25-episode rolling mean"
)

ax.set_title("MAPPO Training Reward")
ax.set_xlabel("Episode")
ax.set_ylabel("Reward")
ax.legend()

save(fig, "paper_fig_mappo_training_reward")


# ============================================================
# FIGURE 4 — MAPPO Training Dynamics
# ============================================================

df = pd.read_csv(DATA / "mappo_update_metrics.csv")

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    df["update"],
    df["explained_var"],
    linewidth=2,
    label="Explained Variance"
)

ax.plot(
    df["update"],
    df["decision_frac"],
    linewidth=2,
    label="Decision Fraction"
)

ax.set_title("MAPPO Training Dynamics")
ax.set_xlabel("Update")
ax.set_ylabel("Value")
ax.legend()

save(fig, "paper_fig_mappo_training_dynamics")


# ============================================================
# FIGURE 5 — Held-Out Policy Evaluation
# ============================================================

df = pd.read_csv(DATA / "evaluation_comparison.csv")

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(df))

ax.bar(
    x,
    df["task_success_rate"]
)

ax.set_xticks(list(x))
ax.set_xticklabels(
    df["policy"],
    rotation=30,
    ha="right"
)

ax.set_title("Held-Out Policy Evaluation")
ax.set_ylabel("Task Success Rate")
ax.set_ylim(0, 1)

for i, value in enumerate(df["task_success_rate"]):
    ax.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

save(fig, "paper_fig_held_out_policy_evaluation")


# ============================================================
# FIGURE 6 — Sprint 6.5 Ablation
# ============================================================

df = pd.read_csv(DATA / "sprint65_ablation.csv")

fig, ax = plt.subplots(figsize=(10, 6))

x = range(len(df))

width = 0.35

ax.bar(
    [i - width / 2 for i in x],
    df["protected_with_risk"],
    width,
    label="With Risk"
)

ax.bar(
    [i + width / 2 for i in x],
    df["protected_risk_zero"],
    width,
    label="Risk = 0"
)

ax.set_xticks(list(x))
ax.set_xticklabels(
    df["arm"],
    rotation=20,
    ha="right"
)

ax.set_title("Sprint 6.5 Ablation: Protected Tasks")
ax.set_ylabel("Tasks Protected")
ax.legend()

save(fig, "paper_fig_sprint65_ablation")


# ============================================================
# FIGURE 7 — Sprint 7 Risk Mechanism
# ============================================================

df = pd.read_csv(DATA / "sprint7_trajectory.csv")

# Keep UNION state only for the main mechanism trajectory.
union = df[df["state_set"] == "UNION"].copy()

if not union.empty:

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        union["update"],
        union["delta_edge"],
        marker="o",
        markersize=3,
        linewidth=1.5
    )

    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_title("Sprint 7: Risk-Conditioned Edge Preference")
    ax.set_xlabel("Update")
    ax.set_ylabel("High-Risk − Low-Risk Edge Probability")

    save(fig, "paper_fig_sprint7_delta_edge_trajectory")


# ============================================================
# FIGURE 8 — Sprint 7 High vs Low Risk Edge Probability
# ============================================================

if not union.empty:

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        union["update"],
        union["p_edge_high_risk"],
        linewidth=2,
        label="High-risk edge"
    )

    ax.plot(
        union["update"],
        union["p_edge_low_risk"],
        linewidth=2,
        label="Low-risk edge"
    )

    ax.set_title("Sprint 7: Edge Selection Probability")
    ax.set_xlabel("Update")
    ax.set_ylabel("Probability")
    ax.legend()

    save(fig, "paper_fig_sprint7_high_low_probability")


# ============================================================
# FIGURE 9 — Action Distribution
# ============================================================

df = pd.read_csv(DATA / "mappo_action_counts.csv")

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    df["action"],
    df["count"]
)

ax.set_title("MAPPO Action Distribution")
ax.set_ylabel("Number of Actions")

ax.tick_params(axis="x", rotation=25)

save(fig, "paper_fig_mappo_action_distribution")


print()
print("=" * 70)
print("PAPER FIGURE GENERATION COMPLETE")
print("=" * 70)
print()
print(f"Figures saved to:")
print(FIGURES)