"""
Throwaway diagnostic: is the completion reward VISIBLE inside the discount
horizon that MAPPO's advantages are computed over?

Three training runs have now degraded monotonically (gamma 0.99, gamma 0.95, and
gamma 0.95 again after the reward-attribution fix). The learner plumbing checks
out: `episode_reward` is literally sum(per-agent rewards), the buffer pairs
(obs_t, a_t, r_t, v(s_t)) correctly, GAE bootstraps at the time limit, and
advantages are normalised over decision entries only. So PPO is descending the
very quantity being reported.

The remaining structural explanation is HORIZON BLINDNESS. A task takes ~150
decision steps to finish. If the completion bonus lands 150 steps after the
decision that earns it, then at gamma=0.95 it is discounted by 0.95^150 ~ 5e-4
and simply does not exist as far as the advantage is concerned. Every term that
IS visible within ~20 steps (exposure, migration cost, energy, overload) is a
COST. A short-horizon-optimal agent therefore minimises the number of tasks it
is accountable for -- which is exactly the observed completions 30 -> 20 and
losses 6 -> 16.

This measures that instead of assuming it:

  1. the actual distribution of task lifetime in decision steps, and the
     resulting discount factor applied to the completion bonus;
  2. for each reward term, its DISCOUNTED contribution to the return-from-t
     averaged over decision steps -- i.e. the composition of the signal the
     advantage is actually built from -- at several gammas.

If completion is the largest positive UNDISCOUNTED term but a negligible share
of the DISCOUNTED return, the hypothesis is confirmed and no amount of
coefficient tuning at fixed gamma will fix it.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl.config import Sprint6Config                      # noqa: E402
from marl.env import DTMarlEnv, COMPLETED                  # noqa: E402
from marl.baseline import RandomPolicy, NoMigrationPolicy   # noqa: E402
from marl.rollout import episode_starts                     # noqa: E402

TERMS = ["complete", "progress", "lost", "sla", "migration", "energy",
         "infeasible", "overload", "expose", "team"]
GAMMAS = [0.95, 0.99, 0.997, 0.999]


def instrument(env):
    """
    Wrap _rewards so every term is recorded PER STEP PER AGENT, not merely
    accumulated. The per-step resolution is the whole point: the question is
    when each term lands relative to the decision that caused it.
    """
    log = {k: [] for k in TERMS}
    r = env.rcfg
    orig = env._rewards

    def patched(ev):
        loads = np.array([env._load_fraction(i) for i in range(env.n_agents)])
        team = (r.R_complete * sum(e["completed"] for e in ev)
                - r.P_task_lost * sum(e["lost"] for e in ev)
                - r.P_balance * float(loads.std()))
        team_each = r.team_reward_share * team / max(env.n_agents, 1)
        row = {k: np.zeros(env.n_agents, np.float64) for k in TERMS}
        for i, e in enumerate(ev):
            sev = e["severity"]
            crit = 1.0 + r.w_criticality * sev
            crit_m = 1.0 + r.w_criticality_migration * sev
            row["complete"][i] = r.R_complete * crit * e["completed"]
            row["progress"][i] = r.R_progress * e["progress_w"]
            row["lost"][i] = -r.P_task_lost * crit * e["lost"]
            row["sla"][i] = -r.P_sla * crit * e["sla"]
            row["migration"][i] = -r.P_migration * crit_m * e["migration_cost"]
            row["energy"][i] = -r.P_energy * e["energy"]
            row["infeasible"][i] = -r.P_infeasible * (1.0 if e["infeasible"] else 0.0)
            row["overload"][i] = -r.P_overload * max(
                0.0, loads[i] - env.cfg.overload_target)
            if e["stayed_resident"]:
                k = env._focus[i]
                s = env.tasks[k].spec.severity if k >= 0 else 0.0
                row["expose"][i] = -(r.P_risk_expose * (1.0 + r.w_criticality * s)
                                     * env.risk_at(i))
            row["team"][i] = team_each
        for k in TERMS:
            log[k].append(row[k])
        return orig(ev)

    env._rewards = patched
    return log


def discounted_from_t(x, gamma):
    """
    G[t] = sum_{k>=t} gamma^(k-t) * x[k], computed backwards. x is [T, n_agents].
    """
    T = x.shape[0]
    g = np.zeros_like(x)
    acc = np.zeros(x.shape[1])
    for t in range(T - 1, -1, -1):
        acc = x[t] + gamma * acc
        g[t] = acc
    return g


def run(name, policy, starts, cfg):
    env = DTMarlEnv(cfg.env, cfg.reward)
    log = instrument(env)
    lifetimes, per_ep_terms, dec_masks = [], [], []
    acts = np.zeros(4, np.int64)
    extra = {k: [] for k in ("lost_in_flight", "lost_on_resident_host",
                             "completed", "lost", "unfinished")}

    for j, s in enumerate(starts):
        for k in TERMS:
            log[k].clear()
        obs, state, masks = env.reset(episode_start_tick=s, seed=1000 + j)
        decs, done, step = [], False, 0
        while not done:
            decs.append(masks.sum(axis=1) > 1)
            a = policy.act(env, obs, masks)
            for x in np.asarray(a).ravel():
                acts[int(x)] += 1
            obs, state, rew, done, info = env.step(a)
            masks = info["action_masks"]
            step += 1
        per_ep_terms.append({k: np.array(log[k]) for k in TERMS})
        dec_masks.append(np.array(decs))
        for k in ("lost_in_flight", "lost_on_resident_host", "completed", "lost"):
            extra[k].append(env.ep[k])
        extra["unfinished"].append(
            len(env.tasks) - env.ep["completed"] - env.ep["lost"])
        # task lifetime: steps from first placement to completion
        for t in env.tasks:
            if t.state == COMPLETED and t.start_step >= 0:
                lifetimes.append(t.finish_step - t.arrival_step)

    print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
    tot_a = max(acts.sum(), 1)
    print("  action mix   : " + "  ".join(
        f"{n}={acts[i]}({acts[i] / tot_a:.1%})"
        for i, n in enumerate(["STAY", "MIG_EDGE", "MIG_CLOUD", "REROUTE"])))
    print("  outcomes/ep  : " + "  ".join(
        f"{k}={np.mean(v):.1f}" for k, v in extra.items()))
    lf = np.array(lifetimes, float)
    if lf.size:
        print(f"  task lifetime (arrival -> completion), decision steps: "
              f"n={lf.size}  mean={lf.mean():.0f}  median={np.median(lf):.0f}  "
              f"p10={np.percentile(lf, 10):.0f}  p90={np.percentile(lf, 90):.0f}")
        print(f"  discount applied to the completion bonus at the MEDIAN "
              f"lifetime ({np.median(lf):.0f} steps):")
        for g in GAMMAS:
            print(f"      gamma={g:<6} -> {g ** np.median(lf):.6f}  "
                  f"(bonus 10.0 x crit 2.0 = 20.0 seen as "
                  f"{20.0 * g ** np.median(lf):.4f})")

    print(f"\n  reward-term composition, mean per episode "
          f"(x reward_scale={cfg.reward.reward_scale})")
    sc = cfg.reward.reward_scale
    undisc = {k: float(np.mean([e[k].sum() for e in per_ep_terms])) * sc
              for k in TERMS}
    tot_u = sum(undisc.values())
    print(f"    {'term':<11} {'UNDISCOUNTED':>14}", end="")
    for g in GAMMAS:
        print(f" {'g=' + str(g):>12}", end="")
    print()

    disc = {g: {} for g in GAMMAS}
    for g in GAMMAS:
        for k in TERMS:
            vals = []
            for e, dm in zip(per_ep_terms, dec_masks):
                G = discounted_from_t(e[k], g)
                n = min(G.shape[0], dm.shape[0])
                sel = dm[:n]
                if sel.any():
                    vals.append(float(G[:n][sel].mean()))
            disc[g][k] = float(np.mean(vals)) if vals else 0.0

    for k in TERMS:
        print(f"    {k:<11} {undisc[k]:>+14.2f}", end="")
        for g in GAMMAS:
            print(f" {disc[g][k]:>+12.3f}", end="")
        print()
    print(f"    {'TOTAL':<11} {tot_u:>+14.2f}", end="")
    for g in GAMMAS:
        print(f" {sum(disc[g].values()):>+12.3f}", end="")
    print()

    print("\n  share of the POSITIVE signal that is doing-the-work reward:")
    for g in [None] + GAMMAS:
        src = undisc if g is None else disc[g]
        lbl = "undiscounted" if g is None else f"gamma={g}"
        pos = sum(v for v in src.values() if v > 0)
        c, p = src["complete"], src["progress"]
        print(f"      {lbl:<14} complete {c / pos if pos else 0:.4f}"
              f"  progress {p / pos if pos else 0:.4f}"
              f"   ({c:+.3f} + {p:+.3f} of {pos:+.3f} positive)")
    return undisc, disc


def main():
    cfg = Sprint6Config()
    cfg.env.start_frac_lo, cfg.env.start_frac_hi = 0.0, 0.7
    probe = DTMarlEnv(cfg.env, cfg.reward)
    starts = episode_starts(probe, 6)
    print(f"starts = {starts}   episode_steps = {cfg.env.episode_steps}   "
          f"dt = {probe.dt}s")
    run("NO-MIGRATION (mask-legal stay)", NoMigrationPolicy(), starts, cfg)
    run("RANDOM-LEGAL", RandomPolicy(seed=3), starts, cfg)

    # A trained checkpoint, if one was named. This is the case that matters:
    # if the learner has HIGHER discounted and LOWER undiscounted return than the
    # baselines, the two objectives disagree somewhere the two fixed policies of
    # tests_env.py::t15 do not probe, and the table below says which term buys it.
    for ck in sys.argv[1:]:
        from marl.mappo import MAPPO, MappoPolicy               # noqa: E402
        agent = MAPPO(probe.n_agents, probe.obs_dim, probe.state_dim,
                      cfg.mappo, device="cpu", seed=0)
        agent.load(ck)
        agent.eval()
        run(f"TRAINED (greedy) {Path(ck).name}", MappoPolicy(agent, "ck"),
            starts, cfg)


if __name__ == "__main__":
    main()
