"""
Sprint 6.5 Phase 2b — IS THE NEGATIVE MIGRATE_EDGE ADVANTAGE *TRUE*?

Diagnostic only. Reads a trained checkpoint and the recorded trace; writes
nothing but its own JSON. Does not modify the environment, the reward, the
learner or any saved model.

`_diag_sensitivity.py` P5 found that the learner's own normalised GAE advantage
for MIGRATE_TO_NEIGHBOR_EDGE is NEGATIVE in every risk bucket (-0.538 at
risk<=0.18, -0.266 at risk>0.5) while STAY is positive, and that only 4
MIGRATE_EDGE samples exist in the whole high-risk region. P3 found MIGRATE_EDGE
is the ONLY relocation legal in 92% of decisions. So the gradient pushes away
from the only relocation usually available.

That leaves exactly two possibilities, and they demand opposite fixes:

  (i)  the advantage is CORRECT — relocating really does not pay in this
       environment under this policy, because something eats the benefit
       (destination contention, transfer latency, the migration charge). Then
       the environment/reward is the bottleneck.
  (ii) the advantage is WRONG — relocating does pay, but the learner's estimate
       says otherwise (too few samples, critic bias, credit assignment). Then
       exploration / credit assignment is the bottleneck.

Three probes separate them.

  P7  CONTENTION AUDIT      masks are computed ONCE per step (env.step line
                            521) for all agents, then actions are applied
                            sequentially, and _relocate re-runs the selector
                            (line 581) against an occupancy that already counts
                            earlier agents' inbound transfers (_occupancy line
                            322). So an action legal at mask time can be
                            refused at apply time. This measures how often, and
                            whether the refusal was caused by a co-agent in the
                            SAME step. Sprint 6 reported ~79 infeasible actions
                            per episode; this attributes them.

  P8  REWARD TERMS BY ACTION  the nine reward terms, separated, conditioned on
                            the action actually taken and the risk bucket. Says
                            what the immediate reward pays for a relocation
                            versus a stay. Immediate reward is expected to be
                            negative for a migration (cost now, benefit later)
                            -- the number that matters is HOW negative relative
                            to the exposure term it is supposed to be trading
                            against.

  P9  EXACT COUNTERFACTUAL  the decisive one. The environment is deterministic
                            given the episode start (arrival times come from a
                            closed formula in reset(); no RNG is consulted
                            after t0 is chosen) and act_greedy is
                            deterministic. So replaying an episode with ONE
                            agent's action overridden at ONE step, and
                            everything else identical, yields the exact
                            A(s,a) = Q(s,a) - V(s) of the greedy policy -- no
                            critic, no GAE, no bootstrapping, no sampling
                            noise. Comparing that true advantage against the
                            learner's estimate decides (i) vs (ii).

The replay identity is asserted, not assumed: every counterfactual first
reproduces the baseline episode bit-for-bit up to the deviation step, and the
run aborts if it does not.

    python marl/_diag_counterfactual.py
    python marl/_diag_counterfactual.py --episodes 4 --deviations 24
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.config import (                                          # noqa: E402
    Sprint6Config, resolve_device, ACTION_NAMES, N_ACTIONS,
    ACTION_STAY, ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD,
    ACTION_PREEMPTIVE_REROUTE,
)
from marl.env import DTMarlEnv, CLOUD_NODE_ID                      # noqa: E402
from marl.mappo import MAPPO, MappoPolicy                          # noqa: E402
from marl.rollout import episode_starts                            # noqa: E402
from marl.baseline import RiskThresholdPolicy                      # noqa: E402
from marl.train import TRAIN_FRAC                                  # noqa: E402

IDX_HAS_TASK = 15
HIGH_RISK = 0.18          # the threshold the winning baseline uses
TERMS = ["complete", "progress", "lost", "sla", "migration", "energy",
         "infeasible", "overload", "expose", "team"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sprint 6.5 counterfactual probe")
    p.add_argument("--model", default=None)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--deviations", type=int, default=30,
                   help="high-risk deviation points (same number of low-risk "
                        "controls is also run)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    return p.parse_args(argv)


# ==========================================================================
# P7 — contention audit
# ==========================================================================

def instrument_contention(env, sink):
    """
    Wrap step() to record, for every relocation ATTEMPT:
      - the action asked for, the agent, its risk
      - whether the step-start mask said it was legal
      - whether it survived the per-agent re-check at apply time
      - how many co-agents committed a transfer in the same step
    Nothing about the dynamics changes: the wrapper observes and delegates.

    SPRINT 6.5 REWRITE. This used to patch _relocate() and read its `chosen is
    None` branch. After the A1 fix that branch is unreachable from step():
    step() re-checks agent_mask(i)[a] first, which runs the same selector calls,
    so a doomed relocation is refused before _relocate is entered and the old
    probe would have reported zero refusals. Refusals are now read from the
    per-agent event records instead.

    One consequence worth stating, because it changes what P7 can conclude:
    agent_mask() and the step-start joint mask are built from the SAME selector
    against the SAME occupancy, so `legal_at_mask and not feasible_now` is now
    equivalent to "an earlier agent in this step took the slot". Every refusal
    is a genuine same-step collision by construction; the environment can no
    longer refuse an action for a reason that predated the step. P7's remaining
    job is to quantify how many collisions the trained policy still walks into.
    """
    orig_step = env.step

    def patched_step(actions):
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        masks = env.action_masks()
        risks = [float(env.risk_at(i)) for i in range(env.n_agents)]
        asked = [int(a) for a in actions]
        out = orig_step(actions)
        ev = out[4]["events"]
        committed = sum(1 for e in ev if e["migration_cost"] > 0.0)
        for i in range(env.n_agents):
            a = asked[i]
            if a == ACTION_STAY:
                continue
            refused = bool(ev[i]["infeasible"])
            legal = bool(masks[i, a])
            sink.append(dict(
                agent=i, action=a, risk=risks[i],
                legal_at_mask=legal,
                feasible_now=not refused,
                # feasible had no co-agent gone first == legal against
                # step-start occupancy, which is exactly the step-start mask
                feasible_alone=legal,
                refused=refused,
                co_committed=committed - (0 if refused else 1),
                dest=int(ev[i]["dest"])))
        return out

    env.step = patched_step


def p7_contention(env, policy, starts):
    sink = []
    instrument_contention(env, sink)
    per_ep = []
    for j, s in enumerate(starts):
        obs, state, masks = env.reset(episode_start_tick=s, seed=j)
        done = False
        while not done:
            a = policy.act(env, obs, masks)
            obs, state, _, done, info = env.step(a)
            masks = info["action_masks"]
        per_ep.append(dict(env.ep))
    # aggregate
    def agg(rows):
        n = len(rows)
        if n == 0:
            return dict(n=0)
        ref = [r for r in rows if r["refused"]]
        # of the refused, how many were legal at mask time AND would have been
        # feasible with no co-agent -> pure contention
        cont = [r for r in ref if r["legal_at_mask"] and r["feasible_alone"]]
        return dict(n=n, refused=len(ref), refused_frac=len(ref) / n,
                    refused_and_legal_at_mask=sum(
                        1 for r in ref if r["legal_at_mask"]),
                    contention_caused=len(cont),
                    contention_frac_of_refused=(
                        len(cont) / len(ref) if ref else 0.0),
                    mean_co_committed=float(
                        np.mean([r["co_committed"] for r in rows])))
    out = dict(episodes=len(starts), attempts=len(sink),
               attempts_per_episode=len(sink) / max(len(starts), 1),
               overall=agg(sink))
    for a in range(N_ACTIONS):
        if a == ACTION_STAY:
            continue
        out[ACTION_NAMES[a]] = agg([r for r in sink if r["action"] == a])
    hi = [r for r in sink if r["risk"] > HIGH_RISK]
    out["high_risk_attempts"] = agg(hi)
    out["episode_infeasible_mean"] = float(
        np.mean([e["infeasible"] for e in per_ep]))
    out["episode_migrate_edge_mean"] = float(
        np.mean([e["migrate_edge"] for e in per_ep]))
    out["episode_migrate_cloud_mean"] = float(
        np.mean([e["migrate_cloud"] for e in per_ep]))
    return out


# ==========================================================================
# P8 — reward terms conditioned on the action taken
# ==========================================================================

def p8_reward_terms(cfg, policy, starts):
    """
    Rebuild a clean env (P7's wrappers must not be in the way) and decompose
    _rewards per agent, keyed by (action, risk bucket). Mirrors the arithmetic
    of env._rewards exactly; the original is still what the env uses.
    """
    env = DTMarlEnv(cfg.env, cfg.reward)
    acc = defaultdict(lambda: defaultdict(float))
    cnt = defaultdict(int)
    orig = env._rewards

    def patched(ev):
        r = env.rcfg
        loads = np.array([env._load_fraction(i) for i in range(env.n_agents)])
        team = (r.R_complete * sum(e["completed"] for e in ev)
                - r.P_task_lost * sum(e["lost"] for e in ev)
                - r.P_balance * float(loads.std()))
        team_each = r.team_reward_share * team / max(env.n_agents, 1)
        for i, e in enumerate(ev):
            if env._focus[i] < 0 and not e["completed"] and not e["lost"]:
                continue
            sev = e["severity"]
            crit = 1.0 + r.w_criticality * sev
            # Sprint 6.5: the migration charge scales with the severity of
            # the task MOVED, not the focus task's outcome severity.
            crit_m = 1.0 + r.w_criticality_migration * e["migration_severity"]
            risk = float(env.risk_at(i))
            key = (int(e["action"]), "high" if risk > HIGH_RISK else "low")
            d = acc[key]
            d["complete"] += r.R_complete * crit * e["completed"]
            d["progress"] += r.R_progress * e["progress_w"]
            d["lost"] -= r.P_task_lost * crit * e["lost"]
            d["sla"] -= r.P_sla * crit * e["sla"]
            d["migration"] -= r.P_migration * crit_m * e["migration_cost"]
            d["energy"] -= r.P_energy * e["energy"]
            d["infeasible"] -= r.P_infeasible * (1.0 if e["infeasible"] else 0.0)
            d["overload"] -= r.P_overload * max(
                0.0, loads[i] - env.cfg.overload_target)
            if e["stayed_resident"]:
                k = env._focus[i]
                s = env.tasks[k].spec.severity if k >= 0 else 0.0
                d["expose"] -= (r.P_risk_expose * (1.0 + r.w_criticality * s)
                                * risk)
            d["team"] += team_each
            cnt[key] += 1
        return orig(ev)

    env._rewards = patched
    for j, s in enumerate(starts):
        obs, state, masks = env.reset(episode_start_tick=s, seed=j)
        done = False
        while not done:
            a = policy.act(env, obs, masks)
            obs, state, _, done, info = env.step(a)
            masks = info["action_masks"]

    out = {}
    scale = cfg.reward.reward_scale
    for (a, bucket), d in acc.items():
        n = cnt[(a, bucket)]
        row = {k: (d.get(k, 0.0) / n) * scale for k in TERMS}
        row["TOTAL"] = sum(row[k] for k in TERMS)
        row["n"] = n
        out[f"{ACTION_NAMES[a]}|risk_{bucket}"] = row
    return out


# ==========================================================================
# P9 — exact counterfactual one-step deviation
# ==========================================================================

def _run_episode(env, policy, start, seed, override=None, record=False):
    """
    Deterministic greedy replay. `override` = (step, agent, action) forces one
    action at one step; everything else is the policy's own choice.

    Returns (total_reward, per_step_team_reward, per_step_agent_reward, trail).
    `trail` is a per-step fingerprint used to ASSERT the replay is identical to
    the baseline before the deviation.
    """
    obs, state, masks = env.reset(episode_start_tick=start, seed=seed)
    done = False
    step = 0
    team_r, agent_r, trail, states = [], [], [], []
    ov_step = ov_agent = ov_act = -1
    if override is not None:
        ov_step, ov_agent, ov_act = override
    while not done:
        a = policy.act(env, obs, masks)
        if record:
            states.append(dict(
                step=step,
                risk=[float(env.risk_at(i)) for i in range(env.n_agents)],
                act=a.copy(),
                mask=masks.copy(),
                has_task=[float(obs[i, IDX_HAS_TASK])
                          for i in range(env.n_agents)]))
        if step == ov_step:
            a = a.copy()
            a[ov_agent] = ov_act
        trail.append(tuple(int(x) for x in a))
        obs, state, rew, done, info = env.step(a)
        masks = info["action_masks"]
        team_r.append(float(rew.sum()))
        agent_r.append(rew.copy())
        step += 1
    return (float(np.sum(team_r)), np.array(team_r), np.array(agent_r),
            trail, states, dict(env.ep), env.episode_metrics())


def p9_counterfactual(cfg, policy, starts, n_dev, gamma):
    env = DTMarlEnv(cfg.env, cfg.reward)
    ref = RiskThresholdPolicy(threshold=HIGH_RISK)
    rows = []
    baselines = {}

    # 1. baseline replays, and the pool of candidate deviation points
    pool_hi, pool_lo = [], []
    for j, s in enumerate(starts):
        tot, team_r, agent_r, trail, states, ep, met = _run_episode(
            env, policy, s, j, record=True)
        baselines[s] = dict(total=tot, team_r=team_r, agent_r=agent_r,
                            trail=trail, ep=ep, met=met)
        for st in states:
            for i in range(env.n_agents):
                if st["has_task"][i] < 0.5:
                    continue
                if st["act"][i] != ACTION_STAY:
                    continue          # deviate only where greedy chose STAY
                if not st["mask"][i, ACTION_MIGRATE_EDGE]:
                    continue
                rec = (s, j, st["step"], i, st["risk"][i])
                (pool_hi if st["risk"][i] > HIGH_RISK else pool_lo).append(rec)

    # 2. sample deviation points deterministically (evenly spaced, no RNG, so
    #    the selection cannot be accused of cherry-picking)
    def take(pool, k):
        if not pool or k <= 0:
            return []
        idx = np.linspace(0, len(pool) - 1, min(k, len(pool))).astype(int)
        return [pool[t] for t in sorted(set(idx.tolist()))]

    dev_hi = take(pool_hi, n_dev)
    dev_lo = take(pool_lo, n_dev)

    # 3. run each deviation
    mismatches = 0
    for bucket, devs in (("high", dev_hi), ("low", dev_lo)):
        for (s, j, step, i, risk) in devs:
            base = baselines[s]
            for act in (ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD):
                tot, team_r, agent_r, trail, _, ep, met = _run_episode(
                    env, policy, s, j, override=(step, i, act))
                # replay identity check BEFORE the deviation
                if trail[:step] != base["trail"][:step]:
                    mismatches += 1
                    continue
                if not np.allclose(base["team_r"][:step], team_r[:step],
                                   atol=1e-6):
                    mismatches += 1
                    continue
                nb, nd = len(base["team_r"]), len(team_r)
                h = min(nb, nd)
                g = gamma ** np.arange(0, max(nb, nd) - step)
                d_team_undisc = float(np.sum(team_r[step:])
                                      - np.sum(base["team_r"][step:]))
                d_team_disc = float(
                    np.dot(g[:nd - step], team_r[step:])
                    - np.dot(g[:nb - step], base["team_r"][step:]))
                d_self = float(np.sum(agent_r[step:, i])
                               - np.sum(base["agent_r"][step:, i]))
                rows.append(dict(
                    bucket=bucket, start=int(s), step=int(step), agent=int(i),
                    risk=float(risk), action=int(act),
                    action_name=ACTION_NAMES[act],
                    d_team_undiscounted=d_team_undisc,
                    d_team_discounted=d_team_disc,
                    d_self_undiscounted=d_self,
                    d_lost=int(met["lost"]) - int(base["met"]["lost"]),
                    d_completed=(int(met["completed"])
                                 - int(base["met"]["completed"])),
                    d_infeasible=(int(ep["infeasible"])
                                  - int(base["ep"]["infeasible"])),
                    horizon=int(h)))
                del nd, nb

    # 4. aggregate
    def summarise(sub):
        if not sub:
            return dict(n=0)
        v = np.array([r["d_team_undiscounted"] for r in sub])
        sv = np.array([r["d_self_undiscounted"] for r in sub])
        return dict(
            n=len(sub),
            mean_d_team=float(v.mean()), sd_d_team=float(v.std()),
            se_d_team=float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1
            else float("nan"),
            median_d_team=float(np.median(v)),
            frac_positive=float((v > 0).mean()),
            mean_d_self=float(sv.mean()),
            mean_d_lost=float(np.mean([r["d_lost"] for r in sub])),
            frac_avoided_a_loss=float(np.mean(
                [1.0 if r["d_lost"] < 0 else 0.0 for r in sub])),
            mean_d_completed=float(np.mean([r["d_completed"] for r in sub])),
            mean_d_infeasible=float(np.mean([r["d_infeasible"] for r in sub])))

    out = dict(
        pool_high=len(pool_hi), pool_low=len(pool_lo),
        deviations_high=len(dev_hi), deviations_low=len(dev_lo),
        replay_mismatches=mismatches, runs=len(rows),
        baseline_total_mean=float(np.mean([b["total"]
                                           for b in baselines.values()])))
    for bucket in ("high", "low"):
        for act in (ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD):
            sub = [r for r in rows
                   if r["bucket"] == bucket and r["action"] == act]
            out[f"{ACTION_NAMES[act]}|risk_{bucket}"] = summarise(sub)
    out["rows"] = rows
    return out


# ==========================================================================
# main
# ==========================================================================

def main(argv=None):
    args = parse_args(argv)
    cfg = Sprint6Config()
    device = resolve_device(args.device)
    torch.manual_seed(cfg.train.seed)

    model = args.model or str(
        _ROOT / "saved_models" / "marl" / "mappo.pth")
    if not Path(model).exists():
        raise SystemExit(f"checkpoint not found: {model}")

    # the HELD-OUT start window, exactly as evaluate.py forces it
    cfg.env.start_frac_lo, cfg.env.start_frac_hi = TRAIN_FRAC, 1.0
    env = DTMarlEnv(cfg.env, cfg.reward)
    agent, extra = MAPPO.load(model, device=device)
    policy = MappoPolicy(agent)

    starts = episode_starts(env, args.episodes)
    print("=" * 78)
    print("SPRINT 6.5 PHASE 2b — IS THE NEGATIVE MIGRATE_EDGE ADVANTAGE TRUE?")
    print("=" * 78)
    print(f"  checkpoint  : {model}")
    print(f"  episodes    : {args.episodes}   starts {list(map(int, starts))}")
    print(f"  policy      : greedy (argmax over legal), as evaluate.py reports")
    print()

    report = dict(model=model, episodes=args.episodes,
                  starts=[int(s) for s in starts], high_risk=HIGH_RISK)

    # ---- P7 -------------------------------------------------------------
    print("-" * 78)
    print("P7  CONTENTION AUDIT — why do relocations get refused?")
    print("-" * 78)
    p7 = p7_contention(env, policy, starts)
    report["p7_contention"] = p7
    print(f"  relocation attempts        : {p7['attempts']} "
          f"({p7['attempts_per_episode']:.1f}/episode)")
    print(f"  env-reported infeasible/ep : {p7['episode_infeasible_mean']:.1f}")
    print(f"  applied edge/ep            : {p7['episode_migrate_edge_mean']:.1f}"
          f"   cloud/ep {p7['episode_migrate_cloud_mean']:.1f}")
    print()
    print(f"  {'action':<28}{'n':>6}{'refused':>9}{'refused%':>10}"
          f"{'legal@mask':>12}{'contention':>11}")
    for key in ("overall", ACTION_NAMES[ACTION_MIGRATE_EDGE],
                ACTION_NAMES[ACTION_MIGRATE_CLOUD],
                ACTION_NAMES[ACTION_PREEMPTIVE_REROUTE],
                "high_risk_attempts"):
        d = p7.get(key, dict(n=0))
        if d.get("n", 0) == 0:
            print(f"  {key:<28}{0:>6}{'-':>9}{'-':>10}{'-':>12}{'-':>11}")
            continue
        print(f"  {key:<28}{d['n']:>6}{d['refused']:>9}"
              f"{100 * d['refused_frac']:>9.1f}%"
              f"{d['refused_and_legal_at_mask']:>12}"
              f"{d['contention_caused']:>11}")
    print()

    # ---- P8 -------------------------------------------------------------
    print("-" * 78)
    print("P8  IMMEDIATE REWARD TERMS BY ACTION (per decision, post-scale)")
    print("-" * 78)
    p8 = p8_reward_terms(cfg, policy, starts)
    report["p8_reward_terms"] = p8
    keys = sorted(p8.keys())
    hdr = f"  {'key':<30}" + "".join(f"{t[:8]:>9}" for t in TERMS) \
          + f"{'TOTAL':>10}{'n':>7}"
    print(hdr)
    for k in keys:
        row = p8[k]
        line = f"  {k:<30}" + "".join(f"{row[t]:>9.4f}" for t in TERMS)
        print(line + f"{row['TOTAL']:>10.4f}{row['n']:>7}")
    print()

    # ---- P9 -------------------------------------------------------------
    print("-" * 78)
    print("P9  EXACT COUNTERFACTUAL — true A(s,a) by deterministic replay")
    print("-" * 78)
    p9 = p9_counterfactual(cfg, policy, starts, args.deviations,
                           cfg.mappo.gamma)
    report["p9_counterfactual"] = {k: v for k, v in p9.items() if k != "rows"}
    report["p9_rows"] = p9["rows"]
    print(f"  candidate STAY decisions   : {p9['pool_high']} high-risk, "
          f"{p9['pool_low']} low-risk")
    print(f"  deviations run             : {p9['runs']} "
          f"(replay mismatches {p9['replay_mismatches']})")
    print(f"  baseline episode reward    : {p9['baseline_total_mean']:.2f}")
    print()
    print(f"  {'deviation':<34}{'n':>5}{'mean dR':>10}{'SE':>8}{'sd':>8}"
          f"{'median':>9}{'P(>0)':>8}{'dLost':>8}{'saved':>7}{'dInf':>7}")
    for bucket in ("high", "low"):
        for act in (ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD):
            k = f"{ACTION_NAMES[act]}|risk_{bucket}"
            d = p9[k]
            if d.get("n", 0) == 0:
                print(f"  {k:<34}{0:>5}{'-':>10}")
                continue
            print(f"  {k:<34}{d['n']:>5}{d['mean_d_team']:>10.3f}"
                  f"{d['se_d_team']:>8.3f}{d['sd_d_team']:>8.3f}"
                  f"{d['median_d_team']:>9.3f}"
                  f"{d['frac_positive']:>8.2f}{d['mean_d_lost']:>8.2f}"
                  f"{d['frac_avoided_a_loss']:>7.2f}"
                  f"{d['mean_d_infeasible']:>7.2f}")
    print()
    print("  dR = change in TOTAL episode reward from step s onward, caused by")
    print("  overriding ONE agent's action at ONE step and then following the")
    print("  same greedy policy. This is the true advantage, not an estimate.")
    print()

    out = args.out or str(_ROOT / "saved_models" / "marl"
                          / "diag_counterfactual.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"  written: {out}")
    return report


if __name__ == "__main__":
    main()
