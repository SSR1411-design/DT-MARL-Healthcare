"""
Throwaway diagnostic: decompose episode reward into its component terms for
several policies on IDENTICAL episode starts.

Reward went DOWN monotonically in two training runs with different discount
horizons, and the PPO update / GAE / normalisation code all read correctly. So
the remaining explanation is that the reward as specified is genuinely maximised
by behaviour that lowers the reported episode reward -- i.e. one term is being
farmed at the expense of the others. This finds which one.
"""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from marl.config import Sprint6Config, EnvConfig, ACTION_NAMES, N_ACTIONS
from marl.env import DTMarlEnv
from marl.baseline import RandomPolicy, NoMigrationPolicy
from marl.mappo import MAPPO
from marl.rollout import episode_starts

TERMS = ["complete", "lost", "sla", "migration", "energy", "infeasible",
         "overload", "expose", "team"]


def instrument(env):
    """Wrap _rewards so each term is accumulated separately."""
    acc = {k: 0.0 for k in TERMS}
    r = env.rcfg
    orig = env._rewards

    def patched(ev):
        loads = np.array([env._load_fraction(i) for i in range(env.n_agents)])
        team = (r.R_complete * sum(e["completed"] for e in ev)
                - r.P_task_lost * sum(e["lost"] for e in ev)
                - r.P_balance * float(loads.std()))
        team_each = r.team_reward_share * team / max(env.n_agents, 1)
        for i, e in enumerate(ev):
            sev = e["severity"]
            crit = 1.0 + r.w_criticality * sev
            # Sprint 6.5: the migration charge scales with the severity of
            # the task MOVED, not the focus task's outcome severity.
            crit_m = 1.0 + r.w_criticality_migration * e["migration_severity"]
            acc["complete"] += r.R_complete * crit * e["completed"]
            acc["lost"] -= r.P_task_lost * crit * e["lost"]
            acc["sla"] -= r.P_sla * crit * e["sla"]
            acc["migration"] -= r.P_migration * crit_m * e["migration_cost"]
            acc["energy"] -= r.P_energy * e["energy"]
            acc["infeasible"] -= r.P_infeasible * (1.0 if e["infeasible"] else 0.0)
            acc["overload"] -= r.P_overload * max(0.0, loads[i] - env.cfg.overload_target)
            if e["stayed_resident"]:
                k = env._focus[i]
                s = env.tasks[k].spec.severity if k >= 0 else 0.0
                acc["expose"] -= (r.P_risk_expose * (1.0 + r.w_criticality * s)
                                  * env.risk_at(i))
            acc["team"] += team_each
        return orig(ev)

    env._rewards = patched
    return acc


class Greedy:
    def __init__(self, agent): self.agent = agent
    def act(self, env, obs, masks): return self.agent.act_greedy(obs, masks)


class AlwaysStay:
    def act(self, env, obs, masks):
        return np.zeros(env.n_agents, dtype=np.int64)


def run(name, policy, starts, cfg):
    env = DTMarlEnv(cfg.env, cfg.reward)
    acc = instrument(env)
    rows, hist = [], np.zeros(N_ACTIONS, np.int64)
    for j, s in enumerate(starts):
        obs, state, masks = env.reset(episode_start_tick=s, seed=1000 + j)
        done = False
        while not done:
            dec = masks.sum(axis=1) > 1
            a = policy.act(env, obs, masks)
            for i in range(env.n_agents):
                if dec[i]:
                    hist[int(a[i])] += 1
            obs, state, rew, done, info = env.step(a)
            masks = info["action_masks"]
        rows.append(env.episode_metrics())
    n = len(starts)
    scale = cfg.reward.reward_scale / n
    m = {k: float(np.mean([r[k] for r in rows]))
         for k in rows[0] if isinstance(rows[0][k], (int, float, np.floating))}
    print(f"\n{name}")
    print(f"  reward {m['episode_reward']:+9.2f}   success {m['task_success_rate']:.3f}"
          f"   lost {m['lost']:5.1f}   unfin {m['unfinished']:5.1f}"
          f"   reloc {m['relocations']:6.1f}   prot {m['tasks_protected_before_failure']:5.1f}")
    tot = sum(acc.values()) * scale
    parts = " ".join(f"{k}={acc[k]*scale:+8.2f}" for k in TERMS)
    print(f"  terms  {parts}")
    print(f"  sum of terms {tot:+.2f}  (reported {m['episode_reward']:+.2f})")
    h = hist.sum()
    if h:
        print("  decision actions: " + "  ".join(
            f"{ACTION_NAMES[k][:12]}={hist[k]/h:.3f}" for k in range(N_ACTIONS)))
    return m, {k: acc[k] * scale for k in TERMS}


def main():
    ckpt = Path(__file__).resolve().parents[1] / "saved_models" / "marl" / "mappo_gamma99_failed_best.pth"
    if not ckpt.exists():
        cands = sorted((ckpt.parent).glob("mappo_gamma99_failed*.pth"))
        ckpt = cands[0] if cands else None
    cfg = Sprint6Config()
    saved = None
    if ckpt is not None and ckpt.exists():
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        cj = ckpt.parent / "mappo_gamma99_failed_config.json"
        if cj.exists():
            import json
            cfg.env = EnvConfig(**json.loads(cj.read_text())["config"]["env"])
    cfg.env.start_frac_lo, cfg.env.start_frac_hi = 0.0, 0.7

    probe = DTMarlEnv(cfg.env, cfg.reward)
    starts = episode_starts(probe, 6)
    print(f"starts = {starts}")

    run("ALWAYS-STAY", AlwaysStay(), starts, cfg)
    run("NO-MIGRATION (mask-legal stay)", NoMigrationPolicy(), starts, cfg)
    run("RANDOM-LEGAL", RandomPolicy(seed=3), starts, cfg)

    if saved is not None:
        agent, _extra = MAPPO.load(ckpt, device="cpu")
        run(f"TRAINED (failed run: {ckpt.name})", Greedy(agent), starts, cfg)


if __name__ == "__main__":
    main()
