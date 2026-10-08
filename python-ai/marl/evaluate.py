"""
Evaluate trained MAPPO agents with exploration DISABLED (Sprint 6, testing
steps 6-10).

    python marl/evaluate.py                        # evaluate saved_models/marl/mappo.pth
    python marl/evaluate.py --model .../mappo_best.pth
    python marl/evaluate.py --episodes 8

WHAT IS REPORTED, and why each item is here

  * The ten metrics the brief asks for: average episode reward, task success
    rate, SLA violation rate, migration count, preemptive migration count,
    reactive migration count, average task latency, energy cost, failed
    critical tasks, tasks protected before failure.
  * The same ten for every baseline, on the IDENTICAL episode start ticks, so
    the only difference between two rows is the relocation decision.
  * Two behaviour probes: does the policy change what it does when
    predicted_failure_risk changes, and when task criticality changes.
  * A save/load round-trip check.

THE HELD-OUT WINDOW. Episode starts are drawn from the LAST (1 - TRAIN_FRAC)
of the usable trace; training used the first TRAIN_FRAC. The windows are
disjoint in start tick but they DO overlap in coverage, because an 800-tick
episode cannot be carved out of a 1500-tick trace without overlap. This is a
real limitation of evaluating on a single recorded simulation and is reported
as one rather than glossed over.

THE COMPARISON THAT MATTERS. `risk-threshold` is a one-line rule on the same
risk channel the policy sees. If MAPPO does not beat it on something, the
honest conclusion is that a threshold suffices, not that MAPPO learned
something. That row is therefore printed in every table and discussed
explicitly in the verdict.

BEHAVIOUR PROBES ARE COUNTERFACTUALS ON THE OBSERVATION, not on the
environment. Real decision-time observations are collected, then a single
feature (local risk, or focus-task severity) is overwritten across a grid and
the action distribution is recomputed. Everything else - neighbours, capacity,
deadlines - is held fixed, so the measured change is attributable to that one
feature and not to a different world.
"""

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.config import (                                        # noqa: E402
    Sprint6Config, EnvConfig, RewardConfig, MappoConfig, resolve_device,
    ACTION_NAMES, N_ACTIONS, ACTION_STAY, ACTION_MIGRATE_EDGE,
    ACTION_MIGRATE_CLOUD, ACTION_PREEMPTIVE_REROUTE,
)
from marl.env import DTMarlEnv                                   # noqa: E402
from marl.mappo import MAPPO, MappoPolicy                        # noqa: E402
from marl.rollout import (                                       # noqa: E402
    run_episodes, episode_starts, format_row, HEADER, METRIC_KEYS,
)
from marl.baseline import (                                      # noqa: E402
    NoMigrationPolicy, RandomPolicy, ReactiveThresholdPolicy,
    RiskThresholdPolicy,
)
from marl.train import TRAIN_FRAC                                # noqa: E402

# observation indices (see env.py::_observations)
IDX_LOCAL_RISK = 12
IDX_UNCERTAINTY = 13
IDX_HAS_TASK = 15
IDX_SEVERITY = 16

RISK_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
SEV_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate trained MAPPO")
    p.add_argument("--model", default=None,
                   help="checkpoint path (default: <out_dir>/<tag>.pth)")
    p.add_argument("--episodes", type=int, default=8,
                   help="evaluation episodes (fixed, evenly spaced starts)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="results JSON path")
    p.add_argument("--skip-ablation", action="store_true")
    return p.parse_args(argv)


# ==========================================================================
# behaviour probes
# ==========================================================================

def collect_decision_obs(env, policy, starts, limit=400):
    """
    Real decision-time (agent, observation, mask) triples from greedy rollouts.
    Only agents that hold a focus task AND have >= 2 legal actions, because
    those are the only states where the policy's output means anything.
    """
    samples = []
    for j, s in enumerate(starts):
        obs, state, masks = env.reset(episode_start_tick=s, seed=j)
        done = False
        while not done and len(samples) < limit:
            for i in range(env.n_agents):
                if masks[i].sum() > 1 and obs[i, IDX_HAS_TASK] > 0.5:
                    samples.append((i, obs[i].copy(), masks[i].copy()))
            obs, state, _, done, info = env.step(policy.act(env, obs, masks))
            masks = info["action_masks"]
        if len(samples) >= limit:
            break
    return samples


def sweep(agent, samples, index, grid, n_agents, obs_dim):
    """
    Overwrite observation feature `index` with each value in `grid` and return
    the mean action distribution (grid x n_actions), plus the per-agent mean
    P(relocate) so we can see whether individual actors differ.
    """
    out = np.zeros((len(grid), N_ACTIONS), np.float64)
    per_agent = np.zeros((len(grid), n_agents), np.float64)
    counts = np.zeros(n_agents, np.int64)
    for i, _, _ in samples:
        counts[i] += 1
    for gi, val in enumerate(grid):
        acc = np.zeros(N_ACTIONS, np.float64)
        acc_a = np.zeros(n_agents, np.float64)
        for (i, o, m) in samples:
            o2 = o.copy()
            o2[index] = val
            batch = np.zeros((n_agents, obs_dim), np.float32)
            mb = np.zeros((n_agents, N_ACTIONS), np.float32)
            batch[i] = o2
            mb[i] = m
            mb[mb.sum(axis=1) == 0, ACTION_STAY] = 1.0    # keep other rows legal
            p = agent.action_probs(batch, mb)[i]
            acc += p
            acc_a[i] += 1.0 - p[ACTION_STAY]
        out[gi] = acc / max(len(samples), 1)
        per_agent[gi] = acc_a / np.maximum(counts, 1)
    return out, per_agent, counts


def print_sweep(title, grid, label, probs, note):
    print(f"\n  {title}")
    print(f"    {label:>8s}  " + "  ".join(f"{ACTION_NAMES[a][:9]:>9s}"
                                           for a in range(N_ACTIONS))
          + "   P(relocate)")
    for gi, v in enumerate(grid):
        rel = 1.0 - probs[gi, ACTION_STAY]
        print(f"    {v:8.2f}  " + "  ".join(f"{probs[gi, a]:9.3f}"
                                            for a in range(N_ACTIONS))
              + f"   {rel:9.3f}")
    span = (1.0 - probs[:, ACTION_STAY]).max() - (1.0 - probs[:, ACTION_STAY]).min()
    mono = float(np.corrcoef(grid, 1.0 - probs[:, ACTION_STAY])[0, 1])
    print(f"    P(relocate) span = {span:.3f}, "
          f"corr(feature, P(relocate)) = {mono:+.3f}")
    print(f"    {note}")
    return span, mono


def print_contingency(title, row_labels, table, note):
    print(f"\n  {title}")
    print(f"    {'bucket':>12s}  "
          + "  ".join(f"{ACTION_NAMES[a][:9]:>9s}" for a in range(N_ACTIONS))
          + f"  {'n':>6s}  {'%reloc':>7s}")
    for r, lab in enumerate(row_labels):
        n = table[r].sum()
        rel = (n - table[r, ACTION_STAY]) / n if n else float("nan")
        print(f"    {lab:>12s}  "
              + "  ".join(f"{table[r, a]:9d}" for a in range(N_ACTIONS))
              + f"  {n:6d}  {rel:7.3f}")
    print(f"    {note}")


# ==========================================================================
# main
# ==========================================================================

def main(argv=None):
    args = parse_args(argv)
    base = Sprint6Config()
    model_path = Path(args.model) if args.model else \
        Path(base.train.out_dir) / f"{base.train.tag}.pth"
    if not model_path.exists():
        print(f"no checkpoint at {model_path} — run marl/train.py first")
        return 1

    device = resolve_device(args.device)
    agent, extra = MAPPO.load(model_path, device=device)
    agent.eval()                                  # exploration disabled

    # Rebuild the environment from the configuration SAVED IN THE CHECKPOINT,
    # so evaluation cannot silently use different physics than training.
    saved = extra.get("config", {})
    cfg = Sprint6Config()
    if saved:
        cfg.env = EnvConfig(**saved["env"])
        cfg.reward = RewardConfig(**saved["reward"])
    # ... but force the HELD-OUT start window.
    cfg.env.start_frac_lo, cfg.env.start_frac_hi = TRAIN_FRAC, 1.0

    env = DTMarlEnv(cfg.env, cfg.reward)
    starts = episode_starts(env, args.episodes)

    print("=" * 100)
    print("SPRINT 6 — MAPPO EVALUATION (exploration disabled)")
    print("=" * 100)
    print(f"  checkpoint        : {model_path.name}  "
          f"(kind={extra.get('kind', '?')}, episode={extra.get('episode', '?')}, "
          f"train mean reward={extra.get('mean_reward', float('nan')):+.2f})")
    print(f"  risk source       : {env.risk.source}  "
          f"(calibrated={env.risk.calibrated})")
    print(f"  eval start window : ticks [{env._min_start}, {env._max_start}] "
          f"= last {1 - TRAIN_FRAC:.0%} of usable trace (training used the first "
          f"{TRAIN_FRAC:.0%})")
    print(f"  episodes          : {len(starts)} at fixed starts {starts}")
    print(f"  episode length    : {cfg.env.episode_steps} steps "
          f"x {env.dt:.0f}s = {cfg.env.episode_steps * env.dt:.0f}s")
    print(f"  agents            : {env.n_agents} (one per edge node)")
    print("=" * 100)

    # ------------------------------------------------------------------
    # 1. MAPPO + baselines on identical starts
    # ------------------------------------------------------------------
    mappo_pol = MappoPolicy(agent, "mappo-greedy")
    policies = [
        mappo_pol,
        NoMigrationPolicy(),
        RandomPolicy(seed=11),
        ReactiveThresholdPolicy(),
        RiskThresholdPolicy(threshold=0.18),
        RiskThresholdPolicy(threshold=0.5),
    ]

    print("\nPOLICY COMPARISON — identical episode starts, identical failures\n")
    print(HEADER)
    print("-" * len(HEADER))
    results, tables = {}, {}
    for pol in policies:
        agg, rows, hist, rtab, stab = run_episodes(
            env, pol, starts, collect_actions=True)
        results[pol.name] = agg
        tables[pol.name] = (hist, rtab, stab)
        print(format_row(pol.name, agg))
    print("-" * len(HEADER))
    print("  lost/critLo/reloc/preem/react/prot/sla are per-episode means; "
          "latency in s; energy in Wh-equivalent trace units")

    m = results["mappo-greedy"]

    # ------------------------------------------------------------------
    # 2. the ten required metrics, spelled out with dispersion
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("REQUIRED METRICS — MAPPO, exploration disabled, "
          f"mean +/- sd over {len(starts)} episodes")
    print("=" * 100)
    required = [
        ("average episode reward", "episode_reward", "{:+.3f}"),
        ("task success rate", "task_success_rate", "{:.4f}"),
        ("SLA violation rate", "sla_violation_rate", "{:.4f}"),
        ("migration count (total relocations)", "relocations", "{:.2f}"),
        ("  of which preemptive", "preemptive_relocations", "{:.2f}"),
        ("  of which reactive", "reactive_relocations", "{:.2f}"),
        ("average task latency (s)", "avg_task_latency_s", "{:.2f}"),
        ("energy cost", "energy_cost", "{:.1f}"),
        ("failed critical tasks", "failed_critical_tasks", "{:.2f}"),
        ("tasks protected before failure", "tasks_protected_before_failure",
         "{:.2f}"),
    ]
    for label, key, fmt in required:
        print(f"  {label:<38s} {fmt.format(m[key]):>10s}  "
              f"+/- {fmt.format(m[key + '_std']).lstrip('+'):>9s}")
    print("\n  supporting breakdown")
    for label, key, fmt in [
        ("edge migrations", "migrations_edge", "{:.2f}"),
        ("cloud migrations", "migrations_cloud", "{:.2f}"),
        ("preemptive reroutes", "reroutes", "{:.2f}"),
        ("tasks completed", "completed", "{:.2f}"),
        ("tasks lost", "lost", "{:.2f}"),
        ("  lost on the resident host", "tasks_lost_on_resident_host", "{:.2f}"),
        ("  lost in flight (bad destination)", "tasks_lost_in_flight", "{:.2f}"),
        ("tasks unfinished at horizon", "unfinished", "{:.2f}"),
        ("critical-task success rate", "critical_success_rate", "{:.4f}"),
        ("infeasible actions attempted", "infeasible_actions", "{:.2f}"),
    ]:
        print(f"  {label:<38s} {fmt.format(m[key]):>10s}")

    # ------------------------------------------------------------------
    # 3. behaviour probe: does risk change behaviour?
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("BEHAVIOUR PROBE 1 — does predicted_failure_risk change the policy?")
    print("=" * 100)
    samples = collect_decision_obs(env, mappo_pol, starts, limit=300)
    print(f"  {len(samples)} real decision-time observations collected "
          f"(agents holding a task with >= 2 legal actions)")
    risk_probs, risk_per_agent, counts = sweep(
        agent, samples, IDX_LOCAL_RISK, RISK_GRID, env.n_agents, env.obs_dim)
    risk_span, risk_corr = print_sweep(
        "counterfactual: local predicted_failure_risk overwritten, "
        "everything else held fixed",
        RISK_GRID, "risk", risk_probs,
        "a policy that ignored the prediction would show a flat column")
    active = [i for i in range(env.n_agents) if counts[i] >= 5]
    print(f"    per-agent P(relocate) at risk=0.0 -> 0.99 "
          f"(agents with >= 5 samples):")
    for i in active:
        print(f"      agent {i:2d} (n={counts[i]:3d}): "
              f"{risk_per_agent[0, i]:.3f} -> {risk_per_agent[-1, i]:.3f} "
              f"(delta {risk_per_agent[-1, i] - risk_per_agent[0, i]:+.3f})")

    print_contingency(
        "empirical: actions actually taken, bucketed by the node's real risk",
        ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"],
        tables["mappo-greedy"][1],
        "counts over decision steps only; buckets are unevenly populated "
        "because high risk is rare in the trace")

    # ------------------------------------------------------------------
    # 4. behaviour probe: does criticality change behaviour?
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("BEHAVIOUR PROBE 2 — does patient criticality change the policy?")
    print("=" * 100)
    sev_probs, sev_per_agent, _ = sweep(
        agent, samples, IDX_SEVERITY, SEV_GRID, env.n_agents, env.obs_dim)
    sev_span, sev_corr = print_sweep(
        "counterfactual: focus-task clinical severity overwritten",
        SEV_GRID, "severity", sev_probs,
        "severity enters the reward only as a multiplier on loss/SLA/migration, "
        "so its effect should be smaller than risk's and need not be monotone")
    print_contingency(
        "empirical: actions taken, split by severity",
        ["sev < 0.5", "sev >= 0.5"], tables["mappo-greedy"][2],
        "the Sprint 5 severity distribution centres near 0.5, so both cells "
        "are populated")

    # ------------------------------------------------------------------
    # 5. zero-risk ablation
    # ------------------------------------------------------------------
    abl = None
    if not args.skip_ablation:
        print("\n" + "=" * 100)
        print("ABLATION — the same trained policy with the risk channel forced to 0")
        print("=" * 100)
        cfg0 = Sprint6Config()
        cfg0.env = EnvConfig(**saved["env"]) if saved else cfg0.env
        cfg0.reward = RewardConfig(**saved["reward"]) if saved else cfg0.reward
        cfg0.env.start_frac_lo, cfg0.env.start_frac_hi = TRAIN_FRAC, 1.0
        cfg0.env.risk_source = "zero"
        env0 = DTMarlEnv(cfg0.env, cfg0.reward)
        starts0 = episode_starts(env0, args.episodes)
        abl, _, _, _, _ = run_episodes(env0, mappo_pol, starts0,
                                       collect_actions=True)
        print(HEADER)
        print("-" * len(HEADER))
        print(format_row("mappo (real risk)", m))
        print(format_row("mappo (risk=0)", abl))
        print("-" * len(HEADER))
        print("  CAVEAT: zeroing risk changes THREE things at once — the "
              "policy's input, the exposure term of the reward, and the "
              "destination selector's score. The reward columns are therefore "
              "not directly comparable; read `prot`, `lost` and `reloc`, which "
              "are physical outcomes.")
        print(f"  protected before failure: {m['tasks_protected_before_failure']:.2f} "
              f"-> {abl['tasks_protected_before_failure']:.2f}")
        print(f"  tasks lost              : {m['lost']:.2f} -> {abl['lost']:.2f}")
        print(f"  relocations             : {m['relocations']:.2f} "
              f"-> {abl['relocations']:.2f}")

    # ------------------------------------------------------------------
    # 6. save / load round trip
    # ------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("SAVE / LOAD ROUND TRIP (testing step 10)")
    print("=" * 100)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "roundtrip.pth"
        agent.save(p, extra=dict(config=cfg.to_dict(), kind="roundtrip"))
        agent2, extra2 = MAPPO.load(p, device=device)
        agent2.eval()
        obs, state, masks = env.reset(episode_start_tick=starts[0], seed=0)
        same_actions, same_probs, checked = True, True, 0
        for _ in range(60):
            a1 = agent.act_greedy(obs, masks)
            a2 = agent2.act_greedy(obs, masks)
            p1 = agent.action_probs(obs, masks)
            p2 = agent2.action_probs(obs, masks)
            same_actions &= bool(np.array_equal(a1, a2))
            same_probs &= bool(np.allclose(p1, p2, atol=0))
            checked += 1
            obs, state, _, done, info = env.step(a1)
            masks = info["action_masks"]
            if done:
                break
        params_equal = all(
            torch.equal(x, y) for x, y in
            zip(agent.actor.state_dict().values(),
                agent2.actor.state_dict().values()))
        crit_equal = all(
            torch.equal(x, y) for x, y in
            zip(agent.critic.state_dict().values(),
                agent2.critic.state_dict().values()))
        v1 = agent.value(state)
        v2 = agent2.value(state)
        print(f"  actor parameters bit-identical after reload : {params_equal}")
        print(f"  critic parameters bit-identical             : {crit_equal}")
        print(f"  greedy actions identical over {checked:2d} steps      : "
              f"{same_actions}")
        print(f"  action probabilities exactly equal          : {same_probs}")
        print(f"  centralised value identical                 : "
              f"{bool(np.array_equal(v1, v2))}")
        print(f"  config recovered from checkpoint            : "
              f"{'config' in extra2}")
        roundtrip_ok = (params_equal and crit_equal and same_actions
                        and same_probs and "config" in extra2)

    # ------------------------------------------------------------------
    # 7. verdict — stated against the threshold rule, not against nothing
    # ------------------------------------------------------------------
    rt = results["risk-threshold@0.18"]
    st = results["static-no-migration"]
    print("\n" + "=" * 100)
    print("VERDICT")
    print("=" * 100)
    print(f"  {'':<34s} {'MAPPO':>10s} {'risk-thr':>10s} {'static':>10s}")
    for label, key, better in [
        ("episode reward", "episode_reward", "high"),
        ("task success rate", "task_success_rate", "high"),
        ("failed critical tasks", "failed_critical_tasks", "low"),
        ("tasks protected", "tasks_protected_before_failure", "high"),
        ("relocations (cost)", "relocations", "low"),
        ("SLA violations", "sla_violations", "low"),
        ("energy cost", "energy_cost", "low"),
    ]:
        win = ("MAPPO" if ((m[key] > rt[key]) == (better == "high"))
               else "risk-threshold")
        print(f"  {label:<34s} {m[key]:>10.3f} {rt[key]:>10.3f} "
              f"{st[key]:>10.3f}   better: {win}")
    print("\n  A threshold on the risk channel is a one-line rule. Any claim "
          "for MAPPO has to be a claim against THAT row, not against the "
          "static baseline.")

    beats_rt = [k for _, k, b in [
        ("", "episode_reward", "high"), ("", "task_success_rate", "high"),
        ("", "failed_critical_tasks", "low"),
        ("", "relocations", "low"), ("", "sla_violations", "low"),
        ("", "energy_cost", "low")]
        if (m[k] > rt[k]) == (b == "high")]
    print(f"  MAPPO is ahead of risk-threshold on {len(beats_rt)}/6 axes: "
          f"{beats_rt}")

    # ------------------------------------------------------------------
    # write results
    # ------------------------------------------------------------------
    out_path = Path(args.out) if args.out else \
        Path(cfg.train.out_dir) / f"{model_path.stem}_eval.json"
    payload = {
        "checkpoint": str(model_path),
        "device": device,
        "eval_start_window": [env._min_start, env._max_start],
        "eval_starts": starts,
        "train_frac": TRAIN_FRAC,
        "exploration": "disabled (greedy argmax over legal actions)",
        "risk": {"source": env.risk.source,
                 "calibrated": env.risk.calibrated,
                 "notes": env.risk.notes},
        "policies": {k: {kk: v[kk] for kk in v if kk != "starts"}
                     for k, v in results.items()},
        "ablation_risk_zero": ({kk: abl[kk] for kk in abl if kk != "starts"}
                               if abl else None),
        "probe_risk": {"grid": RISK_GRID,
                       "action_probs": risk_probs.tolist(),
                       "p_relocate_span": risk_span,
                       "corr_risk_p_relocate": risk_corr,
                       "n_samples": len(samples)},
        "probe_severity": {"grid": SEV_GRID,
                           "action_probs": sev_probs.tolist(),
                           "p_relocate_span": sev_span,
                           "corr_severity_p_relocate": sev_corr},
        "risk_action_table": tables["mappo-greedy"][1].tolist(),
        "severity_action_table": tables["mappo-greedy"][2].tolist(),
        "action_histogram": {ACTION_NAMES[a]: int(tables["mappo-greedy"][0][a])
                             for a in range(N_ACTIONS)},
        "save_load_roundtrip_ok": roundtrip_ok,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    csv_path = out_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["policy"] + METRIC_KEYS)
        for k, v in results.items():
            w.writerow([k] + [f"{v[kk]:.6f}" for kk in METRIC_KEYS])
        if abl:
            w.writerow(["mappo-greedy-risk0"]
                       + [f"{abl[kk]:.6f}" for kk in METRIC_KEYS])

    print(f"\n  written: {out_path}")
    print(f"           {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
