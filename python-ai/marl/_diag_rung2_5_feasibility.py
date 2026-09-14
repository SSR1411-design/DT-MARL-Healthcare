"""
SPRINT 7 RUNG 2.5 -- stage 0: FEASIBILITY of continuation replay.

Answers, empirically rather than by assumption, the four questions the rung
brief demands be settled before any continuation target is built:

  Q1. Can the environment state be reproduced exactly from the episode
      start tick + action history?
  Q2. Can step_idx safely be continued beyond cfg.episode_steps in a
      diagnostic-only clone?
  Q3. Are there hidden state variables that make observation-only replay
      insufficient?
  Q4. Does deterministic replay already exist in the Rung-0/Rung-1 machinery?

Nothing here trains, and nothing here writes a production artifact.

Run:  python marl/_diag_rung2_5_feasibility.py
"""

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.config import Sprint6Config       # noqa: E402
from marl.env import DTMarlEnv, TERMINAL     # noqa: E402
from marl.mappo import MAPPO                 # noqa: E402

R2 = _ROOT / "saved_models" / "marl" / "mappo_R2_mc_target.pth"
TRAIN_FRAC = 0.7


def build_env(cfg):
    cfg.env.start_frac_lo, cfg.env.start_frac_hi = 0.0, TRAIN_FRAC
    return DTMarlEnv(cfg.env, cfg.reward)


def greedy_actions(agent, obs, masks):
    return agent.act_greedy(obs, masks)


def rollout(env, agent, start, cont_limit=0, record_actions=None):
    """
    Replay one episode greedily. If cont_limit > 0, keep stepping past
    cfg.episode_steps for at most cont_limit extra steps, stopping early if
    every task is terminal or the trace runs out.

    Returns dict with the action history, per-step team reward, and how the
    episode/continuation ended.
    """
    obs, state, masks = env.reset(episode_start_tick=start)
    horizon = env.cfg.episode_steps
    acts, rews = [], []
    if cont_limit > 0:
        env.cfg.episode_steps = horizon + cont_limit   # diagnostic clone only

    done = False
    ended_at_horizon = None
    trace_exhausted = False
    while True:
        if record_actions is not None and len(acts) < len(record_actions):
            a = record_actions[len(acts)]
        else:
            a = greedy_actions(agent, obs, masks)
        # would the NEXT tick fall outside the recorded trace?
        if env.tick(env.step_idx + 1) >= env.trace.n_ticks:
            trace_exhausted = True
            break
        obs, state, rew, done, info = env.step(a)
        acts.append(np.asarray(a).copy())
        rews.append(np.asarray(rew).copy())
        masks = info["action_masks"]
        if env.step_idx == horizon and ended_at_horizon is None:
            ended_at_horizon = bool(done)
        if done:
            break

    env.cfg.episode_steps = horizon        # always restore
    return dict(acts=acts, rews=np.array(rews), n=len(acts),
                done=done, ended_at_horizon=ended_at_horizon,
                trace_exhausted=trace_exhausted,
                all_terminal=all(t.state in TERMINAL for t in env.tasks))


def main():
    print("=" * 78)
    print("SPRINT 7 RUNG 2.5 -- stage 0: continuation-replay feasibility")
    print("=" * 78)

    cfg = Sprint6Config()
    env = build_env(cfg)
    agent, extra = MAPPO.load(R2, device="cpu")
    agent.eval()

    print(f"  trace ticks        : {env.trace.n_ticks}")
    print(f"  ticks_per_step     : {env.cfg.ticks_per_step}")
    print(f"  episode_steps      : {env.cfg.episode_steps}")
    print(f"  start window       : [{env._min_start}, {env._max_start}]")
    span = env.cfg.episode_steps * env.cfg.ticks_per_step
    print(f"  episode tick span  : {span}")
    print(f"  headroom at t0=min : "
          f"{(env.trace.n_ticks - 1 - (env._min_start + span)) // env.cfg.ticks_per_step} steps")
    print(f"  headroom at t0=max : "
          f"{(env.trace.n_ticks - 1 - (env._max_start + span)) // env.cfg.ticks_per_step} steps")

    # ---- Q1 / Q3: determinism of replay from start tick alone -----------
    print("\n-- Q1/Q3: is replay from the start tick alone exact? ---------")
    start = int((env._min_start + env._max_start) // 2)
    r1 = rollout(env, agent, start)
    r2 = rollout(env, agent, start)
    same_a = all(np.array_equal(x, y) for x, y in zip(r1["acts"], r2["acts"])) \
        and r1["n"] == r2["n"]
    same_r = np.array_equal(r1["rews"], r2["rews"])
    print(f"  two greedy replays of t0={start}: n={r1['n']}/{r2['n']} "
          f"actions_identical={same_a} rewards_identical={same_r}")

    # replay driven by the RECORDED action history (no policy calls)
    r3 = rollout(env, agent, start, record_actions=r1["acts"])
    same_forced = np.array_equal(r3["rews"], r1["rews"]) and r3["n"] == r1["n"]
    print(f"  forced-action replay reproduces rewards: {same_forced}")

    # a DIFFERENT episode in between must not perturb the replay (no RNG carry)
    _ = rollout(env, agent, env._min_start)
    r4 = rollout(env, agent, start)
    print(f"  replay after an intervening episode still identical: "
          f"{np.array_equal(r4['rews'], r1['rews'])}")

    # ---- Q2: continuation past the horizon ------------------------------
    print("\n-- Q2: can step_idx continue past cfg.episode_steps? ---------")
    for t0 in (env._min_start, start, env._max_start):
        base = rollout(env, agent, t0)
        headroom = (env.trace.n_ticks - 1
                    - (t0 + env.cfg.episode_steps * env.cfg.ticks_per_step)) \
            // env.cfg.ticks_per_step
        cont = rollout(env, agent, t0, cont_limit=max(headroom, 0))
        print(f"  t0={t0:4d}: base n={base['n']:3d} "
              f"ended_at_horizon_done={base['done']}  |  "
              f"cont n={cont['n']:3d} extra={cont['n'] - base['n']:3d} "
              f"all_terminal={cont['all_terminal']} "
              f"trace_exhausted={cont['trace_exhausted']} "
              f"headroom={headroom}")
        # the continuation must agree with the base run on the first n steps
        pre = all(np.array_equal(base["acts"][i], cont["acts"][i])
                  for i in range(min(base["n"], cont["n"])))
        print(f"           continuation is a strict extension of base: {pre}")

    print("\n-- Q4: deterministic replay already in Rung-0 machinery? -----")
    src = (_ROOT / "marl" / "_diag_rung0.py").read_text(encoding="utf-8")
    print(f"  _diag_rung0._replay exists      : {'def _replay' in src}")
    print(f"  it supports sample=False (greedy): {'sample=False' in src}")
    print(f"  it records actions per step      : {'acts' in src or 'action' in src}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
