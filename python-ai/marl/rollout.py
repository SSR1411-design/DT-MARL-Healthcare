"""
Roll out and evaluate a policy on the DT-MARL environment.

Shared by train.py (for its periodic greedy checkpoints) and evaluate.py (for
the reported numbers), so training and evaluation can never diverge in how a
metric is computed.

`run_episodes` always uses the SAME list of episode start ticks for every
policy it is given, which is what makes the baseline comparison apples to
apples: identical failure sequences, identical arrival schedule, identical
initial placement. The only difference between two rows of the results table
is the relocation decision.
"""

import numpy as np

METRIC_KEYS = [
    "episode_reward", "task_success_rate", "sla_violation_rate",
    "sla_violations", "relocations", "migrations", "migrations_edge",
    "migrations_cloud", "reroutes", "preemptive_relocations",
    "reactive_relocations", "avg_task_latency_s", "energy_cost",
    "failed_critical_tasks", "critical_success_rate",
    "tasks_protected_before_failure", "tasks_lost_on_resident_host",
    "tasks_lost_in_flight", "completed", "lost", "unfinished",
    "infeasible_actions",
]


def episode_starts(env, n, seed=0):
    """Evenly spaced, deterministic episode starts across the env's window."""
    lo, hi = env._min_start, env._max_start
    if n == 1 or hi <= lo:
        return [lo]
    return [int(round(lo + k * (hi - lo) / (n - 1))) for k in range(n)]


def run_episode(env, policy, start_tick, seed=None, collect_actions=False):
    """One episode. Returns (metrics, action_histogram, risk_action_table)."""
    obs, state, masks = env.reset(episode_start_tick=start_tick, seed=seed)
    if hasattr(policy, "reset"):
        policy.reset()
    hist = np.zeros(env.n_actions, dtype=np.int64)
    # (risk bucket, action) contingency over DECISION steps only — the
    # behaviour probe for "do agents act differently as risk changes?"
    n_buckets = 5
    table = np.zeros((n_buckets, env.n_actions), dtype=np.int64)
    # (severity bucket, action) — "do agents act differently as criticality
    # changes?"
    sev_table = np.zeros((2, env.n_actions), dtype=np.int64)
    done = False
    while not done:
        risks = [env.risk_at(i) for i in range(env.n_agents)]
        sevs = []
        for i in range(env.n_agents):
            t = env.focus_task(i)
            sevs.append(t.spec.severity if t is not None else -1.0)
        decision = masks.sum(axis=1) > 1
        actions = policy.act(env, obs, masks)
        if collect_actions:
            for i in range(env.n_agents):
                if not decision[i]:
                    continue
                a = int(actions[i])
                hist[a] += 1
                b = min(int(risks[i] * n_buckets), n_buckets - 1)
                table[b, a] += 1
                if sevs[i] >= 0.0:
                    sev_table[1 if sevs[i] >= 0.5 else 0, a] += 1
        obs, state, rew, done, info = env.step(actions)
        masks = info["action_masks"]
    return env.episode_metrics(), hist, table, sev_table


def run_episodes(env, policy, starts, seed=0, collect_actions=False):
    """Aggregate `run_episode` over a fixed list of starts."""
    rows, hist, table, sev_table = [], None, None, None
    for j, s in enumerate(starts):
        m, h, tb, sv = run_episode(env, policy, s, seed=seed + j,
                                   collect_actions=collect_actions)
        rows.append(m)
        hist = h if hist is None else hist + h
        table = tb if table is None else table + tb
        sev_table = sv if sev_table is None else sev_table + sv
    agg = {}
    for k in METRIC_KEYS:
        vals = np.array([float(r[k]) for r in rows], dtype=np.float64)
        agg[k] = float(np.nanmean(vals))
        agg[k + "_std"] = float(np.nanstd(vals))
    agg["n_episodes"] = len(rows)
    agg["starts"] = list(starts)
    return agg, rows, hist, table, sev_table


def format_row(name, agg):
    return (f"{name:<24s} {agg['episode_reward']:>9.2f} "
            f"{agg['task_success_rate']:>7.3f} "
            f"{agg['lost']:>6.1f} "
            f"{agg['failed_critical_tasks']:>6.1f} "
            f"{agg['relocations']:>7.1f} "
            f"{agg['preemptive_relocations']:>7.1f} "
            f"{agg['reactive_relocations']:>7.1f} "
            f"{agg['tasks_protected_before_failure']:>7.1f} "
            f"{agg['sla_violations']:>6.1f} "
            f"{agg['avg_task_latency_s']:>8.1f} "
            f"{agg['energy_cost']:>9.1f}")


HEADER = (f"{'policy':<24s} {'reward':>9s} {'success':>7s} {'lost':>6s} "
          f"{'critLo':>6s} {'reloc':>7s} {'preem':>7s} {'react':>7s} "
          f"{'prot':>7s} {'sla':>6s} {'latency':>8s} {'energy':>9s}")


__all__ = ["run_episode", "run_episodes", "episode_starts", "METRIC_KEYS",
           "format_row", "HEADER"]
