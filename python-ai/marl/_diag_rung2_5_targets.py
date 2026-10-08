"""
SPRINT 7 RUNG 2.5 -- stages 1-3: truncation census, target decomposition,
and continuation-MC construction.

Answers brief items A (target decomposition), B (truncation stratification),
C (critic dependence / continuation replay) and D (three-target comparison).

METHOD AND ITS LIMITS -- stated up front because they bound every number here.

  * The environment is EXACTLY reproducible from the episode start tick alone.
    Verified in _diag_rung2_5_feasibility.py: env.py's only RNG use is
    random episode-start selection (env.py:192,219,224), build_patient_tasks
    is a closed formula, and everything else is trace replay. Two greedy
    replays of the same t0 agree bit-for-bit, and a forced-action replay
    reproduces the reward stream exactly.

  * cfg.episode_steps is NOT mutated to enable continuation. It enters the
    OBSERVATION as prog = step_idx / episode_steps (env.py:445) and the global
    state (env.py:524), so raising it perturbs every observation and therefore
    the policy's own actions -- the feasibility probe measured exactly that
    (continuation stopped being a strict extension of the base episode).
    Instead the `done` flag is IGNORED and stepping simply continues, which
    leaves steps 0..399 bit-identical to the real episode.

  * CONSEQUENCE, and the main honesty caveat of item C: beyond step 400,
    prog > 1.0, which is out of distribution for both actor and critic. The
    continuation REWARDS are true environment rewards, but the continuation
    ACTIONS are chosen on OOD observations. We therefore do not report a
    single "exact" continuation return; we bracket it with several
    continuation action policies (greedy / all-STAY) and report the spread.

  * The 377/223 truncation split came from a policy that was CHANGING across
    training, sampled stochastically. Only one R2 policy exists on disk (best
    and final are bit-identical), so this replays the FINAL FROZEN policy over
    the training start distribution. It is not a reconstruction of the actual
    training rollouts, and the split is not expected to match exactly.

  * train.py classified truncation as (step_idx >= episode_steps) ALONE. When
    all tasks went terminal exactly at step 400 both conditions held and the
    episode was still bootstrapped, even though no return remained. That class
    is counted separately here as `both`.

Writes SPRINT_7_RUNG2_5_targets.json. Trains nothing; touches no production file.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.config import Sprint6Config, ACTION_STAY      # noqa: E402
from marl.env import DTMarlEnv, TERMINAL                # noqa: E402
from marl.mappo import MAPPO                            # noqa: E402

OUT = _ROOT / "saved_models" / "marl"
R2 = OUT / "mappo_R2_mc_target.pth"
A0 = OUT / "mappo_A0_cpu_repro.pth"
TRAIN_FRAC = 0.7
TRAIN_SEED = 20260818
IDX_RISK = 12
HI_RISK, LO_RISK = 0.50, 0.10


# ---------------------------------------------------------------- helpers
def build_env(cfg):
    cfg.env.start_frac_lo, cfg.env.start_frac_hi = 0.0, TRAIN_FRAC
    return DTMarlEnv(cfg.env, cfg.reward)


def training_start_ticks(env, n_episodes, seed=TRAIN_SEED):
    """The EXACT start ticks train.py drew, reproduced from its own rng."""
    rng = np.random.default_rng(seed)
    return [int(rng.integers(env._min_start, env._max_start + 1))
            for _ in range(n_episodes)]


def all_terminal(env):
    return all(t.state in TERMINAL for t in env.tasks)


def replay_with_continuation(env, agent, start, sample, cont_policy=None,
                             torch_seed=None, force_steps=20):
    """
    Replay one episode to the 400-step horizon, then -- only if the episode is
    still live at the horizon -- keep stepping to estimate the true remaining
    return. `done` is ignored rather than episode_steps being changed.

    cont_policy: None -> no continuation. "greedy" -> policy argmax on the OOD
    observation. "stay" -> all agents STAY (a deliberately passive lower
    reference for the continuation reward stream).
    """
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    obs, state, masks = env.reset(episode_start_tick=start)
    horizon = env.cfg.episode_steps
    n = env.n_agents

    rew, risk, obs_l, state_l, mask_l = [], [], [], [], []
    acts = []
    step = 0
    fired_done = None
    while True:
        if sample:
            a, _ = agent.act(obs, masks)
        else:
            a = agent.act_greedy(obs, masks)
        obs_l.append(obs.copy()); state_l.append(state.copy())
        mask_l.append(masks.copy()); acts.append(np.asarray(a).copy())
        risk.append(np.array([env.risk_at(i) for i in range(n)], np.float32))
        obs, state, r, done, info = env.step(a)
        rew.append(np.asarray(r).copy())
        masks = info["action_masks"]
        step += 1
        if done:
            fired_done = True
            break
    n_in = step
    at_horizon = int(env.step_idx) >= horizon
    term_at_end = all_terminal(env)
    # train.py's own rule, reproduced exactly:
    trunc_flag_trainpy = bool(env.step_idx >= horizon)
    cls = ("term" if not at_horizon else ("both" if term_at_end else "trunc"))

    boot_state = state.copy()          # V(s_T) is read from this
    cont_rew = []
    cont_ended = "not_attempted"
    if cont_policy is not None and cls == "trunc":
        while True:
            if env.tick(env.step_idx + 1) >= env.trace.n_ticks:
                cont_ended = "trace_exhausted"
                break
            if all_terminal(env):
                cont_ended = "terminal"
                break
            if cont_policy == "stay":
                a = np.full(n, ACTION_STAY, dtype=np.int64)
            else:
                a = agent.act_greedy(obs, masks)
            obs, state, r, done, info = env.step(a)
            cont_rew.append(np.asarray(r).copy())
            masks = info["action_masks"]
        if cont_ended == "not_attempted":
            cont_ended = "terminal" if all_terminal(env) else "trace_exhausted"

    # For the `both` class the true remaining return is ASSUMED to be zero
    # (every task already terminal). Do not assume it -- step anyway, with the
    # all-terminal break disabled, and measure whether any reward still flows.
    both_probe = None
    if cls == "both" and cont_policy is not None:
        pr = []
        for _ in range(force_steps):
            if env.tick(env.step_idx + 1) >= env.trace.n_ticks:
                break
            a = (np.full(n, ACTION_STAY, dtype=np.int64) if cont_policy == "stay"
                 else agent.act_greedy(obs, masks))
            obs, state, r, done, info = env.step(a)
            pr.append(np.asarray(r).copy())
            masks = info["action_masks"]
        both_probe = np.array(pr) if pr else np.zeros((0, n), np.float32)

    return dict(
        start=start, n_in=n_in, cls=cls, at_horizon=at_horizon,
        term_at_end=term_at_end, trunc_flag_trainpy=trunc_flag_trainpy,
        rew=np.array(rew), risk=np.array(risk), obs=np.array(obs_l),
        state=np.array(state_l), mask=np.array(mask_l), act=np.array(acts),
        boot_state=boot_state, cont_rew=np.array(cont_rew) if cont_rew
        else np.zeros((0, n), np.float32), cont_ended=cont_ended,
        cont_steps=len(cont_rew), both_probe=both_probe,
        headroom=(env.trace.n_ticks - 1 - (start + horizon * env.cfg.ticks_per_step))
        // env.cfg.ticks_per_step)


def discounted_tail(rew, gamma, tail=None):
    """G[t] = sum_{k>=t} gamma^(k-t) rew[k]  (+ gamma^(T-t) * tail)."""
    T = rew.shape[0]
    out = np.zeros_like(rew, dtype=np.float64)
    run = np.zeros(rew.shape[1], np.float64) if tail is None else tail.astype(np.float64)
    for t in reversed(range(T)):
        run = rew[t].astype(np.float64) + gamma * run
        out[t] = run
    return out


def dist(x):
    x = np.asarray(x, np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0)
    q = np.percentile(x, [0, 5, 25, 50, 75, 95, 100])
    return dict(n=int(x.size), mean=float(x.mean()), sd=float(x.std()),
                min=float(q[0]), p5=float(q[1]), p25=float(q[2]),
                p50=float(q[3]), p75=float(q[4]), p95=float(q[5]),
                max=float(q[6]))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--model", default=str(R2))
    p.add_argument("--sample", action="store_true", default=True,
                   help="stochastic replay, matching how train.py acted")
    p.add_argument("--greedy", dest="sample", action="store_false")
    p.add_argument("--cont", default="greedy", choices=["greedy", "stay", "none"])
    p.add_argument("--tag", default="R2_mc_target")
    a = p.parse_args(argv)

    cfg = Sprint6Config()
    env = build_env(cfg)
    agent, extra = MAPPO.load(Path(a.model), device="cpu")
    agent.eval()
    gamma = agent.cfg.gamma

    print("=" * 78)
    print("SPRINT 7 RUNG 2.5 -- stages 1-3: truncation + target decomposition")
    print("=" * 78)
    print(f"  model            : {a.model}")
    print(f"  critic_target    : {agent.cfg.critic_target}   gamma={gamma}")
    print(f"  replay           : {'stochastic (matches train.py)' if a.sample else 'greedy'}")
    print(f"  continuation     : {a.cont}")
    print(f"  episodes         : {a.episodes} (start ticks reproduced from "
          f"train.py rng seed {TRAIN_SEED})")

    starts = training_start_ticks(env, a.episodes)
    cont_policy = None if a.cont == "none" else a.cont

    eps = []
    for k, s in enumerate(starts):
        r = replay_with_continuation(env, agent, s, a.sample,
                                     cont_policy=cont_policy,
                                     torch_seed=TRAIN_SEED + k)
        eps.append(r)
        if (k + 1) % 25 == 0:
            print(f"    {k + 1}/{a.episodes} replayed")

    # ---------------- stage 1: truncation census ----------------------
    counts = {c: sum(1 for e in eps if e["cls"] == c)
              for c in ("term", "both", "trunc")}
    print("\n-- stage 1: episode-end classification -----------------------")
    print(f"  term  (all tasks terminal BEFORE step 400)        : {counts['term']}")
    print(f"  both  (terminal AND at step 400; train.py bootstrapped): {counts['both']}")
    print(f"  trunc (live at step 400; bootstrap justified)     : {counts['trunc']}")
    n_boot = counts["both"] + counts["trunc"]
    print(f"  => train.py would flag truncated for {n_boot}/{len(eps)} "
          f"({100 * n_boot / len(eps):.1f}%)")

    # ---------------- stage 2: target decomposition -------------------
    # agent.value(state) is EXACTLY what train.py called for the bootstrap
    # (train.py:188 buf.set_bootstrap(..., agent.value(nstate), truncated)),
    # so the decomposition below reproduces the real target, not a lookalike.
    boot_vals = np.array([np.asarray(agent.value(e["boot_state"])).ravel()
                          for e in eps], np.float64)

    rows = []
    for e, bv in zip(eps, boot_vals):
        T = e["n_in"]
        bootstrapped = e["trunc_flag_trainpy"]
        tail = np.asarray(bv, np.float64).ravel()
        if tail.size == 1:
            tail = np.repeat(tail, e["rew"].shape[1])
        g_rew = discounted_tail(e["rew"], gamma, None)
        g_r2 = discounted_tail(e["rew"], gamma,
                               tail if bootstrapped else None)
        boot_contrib = g_r2 - g_rew
        # continuation target where available
        if e["cont_steps"] > 0:
            full = np.concatenate([e["rew"], e["cont_rew"]], axis=0)
            g_cont_full = discounted_tail(full, gamma, None)
            g_cont = g_cont_full[:T]
        else:
            g_cont = g_rew.copy() if e["cls"] != "trunc" else None
        rows.append(dict(e=e, g_rew=g_rew, g_r2=g_r2, boot=boot_contrib,
                         g_cont=g_cont, boot_V=tail, T=T,
                         tail_true=(discounted_tail(e["cont_rew"], gamma,
                                                    None)[0]
                                    if e["cont_steps"] > 0
                                    else np.zeros_like(tail))))

    # ------ THE central number for item F.1/F.2 ------------------------
    # At the truncation boundary the R2 target substituted V(s_T) for the true
    # remaining discounted return. Measure both and take the residual.
    print("\n-- V(s_T) vs the TRUE remaining return at the boundary -------")
    boundary = {}
    for cls in ("trunc", "both"):
        sel = [r for r in rows if r["e"]["cls"] == cls
               and (cls == "both" or r["e"]["cont_ended"] == "terminal")]
        if not sel:
            boundary[cls] = dict(n_episodes=0)
            continue
        V = np.concatenate([r["boot_V"].ravel() for r in sel])
        Rt = np.concatenate([r["tail_true"].ravel() for r in sel])
        boundary[cls] = dict(
            n_episodes=len(sel), n_entries=int(V.size),
            V_sT=dist(V), true_remaining_return=dist(Rt),
            residual_V_minus_true=dist(V - Rt),
            abs_residual=dist(np.abs(V - Rt)),
            frac_V_explained=(float(np.mean(np.abs(Rt)) / max(np.mean(np.abs(V)), 1e-9))),
        )
        print(f"  [{cls:5s}] n_ep={len(sel):3d}  "
              f"V(s_T) {dist(V)['mean']:+.4f} (sd {dist(V)['sd']:.4f})  "
              f"true tail {dist(Rt)['mean']:+.4f} (sd {dist(Rt)['sd']:.4f})  "
              f"mean|V-true| {dist(np.abs(V - Rt))['mean']:.4f}")
    # did any reward flow AFTER every task was already terminal?
    bp = [r["e"]["both_probe"] for r in rows
          if r["e"]["cls"] == "both" and r["e"]["both_probe"] is not None]
    if bp:
        tot = float(sum(float(np.abs(x).sum()) for x in bp))
        boundary["both_post_terminal_reward_abs_sum"] = tot
        boundary["both_post_terminal_steps_probed"] = int(sum(len(x) for x in bp))
        print(f"  post-terminal probe: {int(sum(len(x) for x in bp))} forced "
              f"steps across {len(bp)} 'both' episodes, "
              f"sum|reward| = {tot:.6f}"
              f"{'  -> tail really is zero' if tot < 1e-6 else '  -> NONZERO'}")
    out_boundary = boundary

    def at(idx_fn, field):
        vals = []
        for r in rows:
            if not r["e"]["trunc_flag_trainpy"]:
                continue
            t = idx_fn(r["T"])
            vals.append(r[field][t])
        return np.array(vals)

    print("\n-- stage 2: bootstrap contribution to the R2 target ----------")
    print("   (bootstrapped episodes only; per-agent entries pooled)")
    print("   NOTE: signed means are near zero because rewards are two-sided,")
    print("   so mean|.| is the magnitude that matters, not mean.")
    for lab, fn in (("episode start   t=0", lambda T: 0),
                    ("midpoint     t=T/2", lambda T: T // 2),
                    ("just before  t=T-1", lambda T: T - 1)):
        b = at(fn, "boot"); tgt = at(fn, "g_r2"); rw = at(fn, "g_rew")
        frac = np.abs(b) / np.maximum(np.abs(b) + np.abs(rw), 1e-9)
        print(f"  {lab}: mean|boot| {np.abs(b).mean():7.3f}  "
              f"mean|reward-only| {np.abs(rw).mean():7.3f}  "
              f"mean|target| {np.abs(tgt).mean():7.3f}  "
              f"|boot|share mean {frac.mean():.3f} "
              f"p50 {np.percentile(frac, 50):.3f} "
              f"p95 {np.percentile(frac, 95):.3f}")

    out = dict(
        rung="2.5", stage="1-3", model=a.model,
        critic_target=agent.cfg.critic_target, gamma=gamma,
        replay="stochastic" if a.sample else "greedy",
        continuation_policy=a.cont, episodes=len(eps),
        train_seed_for_starts=TRAIN_SEED,
        method_caveats=[
            "final frozen R2 policy replayed over the training start "
            "distribution; NOT a reconstruction of the actual training "
            "rollouts (per-episode torch RNG states were never saved)",
            "cfg.episode_steps deliberately NOT mutated: prog=step_idx/"
            "episode_steps enters obs (env.py:445) and global state "
            "(env.py:524); continuation ignores the done flag instead",
            "beyond step 400 prog>1.0, so continuation ACTIONS are chosen on "
            "out-of-distribution observations; continuation REWARDS are true "
            "environment rewards",
        ],
        end_classification=counts,
        trainpy_would_bootstrap=n_boot,
        episode_lengths=dist([e["n_in"] for e in eps]),
        headroom_steps=dist([e["headroom"] for e in eps]),
    )

    # ---- stratified stats (item B) -----------------------------------
    strat = {}
    for cls in ("term", "both", "trunc"):
        sel = [r for r in rows if r["e"]["cls"] == cls]
        if not sel:
            strat[cls] = dict(count=0)
            continue
        risk_all = np.concatenate([r["e"]["risk"].ravel() for r in sel])
        dec = np.concatenate([(r["e"]["mask"].sum(-1) > 1.5).ravel() for r in sel])
        rk = np.concatenate([r["e"]["risk"].ravel() for r in sel])
        act = np.concatenate([r["e"]["act"].ravel() for r in sel])
        hi = dec & (rk > HI_RISK)
        strat[cls] = dict(
            count=len(sel),
            length=dist([r["T"] for r in sel]),
            episode_return=dist([float(r["e"]["rew"].sum()) for r in sel]),
            target_t0=dist([r["g_r2"][0] for r in sel]),
            reward_only_t0=dist([r["g_rew"][0] for r in sel]),
            boot_V=dist([float(np.mean(r["boot_V"])) for r in sel]),
            boot_contrib_t0=dist([r["boot"][0] for r in sel]),
            risk=dist(risk_all),
            frac_decision_entries=float(dec.mean()),
            n_highrisk_decision=int(hi.sum()),
            highrisk_EDGE_share=(float((act[hi] == 1).mean())
                                 if hi.sum() else None),
            highrisk_STAY_share=(float((act[hi] == 0).mean())
                                 if hi.sum() else None),
            cont_ended={k: sum(1 for r in sel if r["e"]["cont_ended"] == k)
                        for k in ("terminal", "trace_exhausted",
                                  "not_attempted")},
            cont_steps=dist([r["e"]["cont_steps"] for r in sel]),
        )
    out["stratified"] = strat

    # ---- bootstrap distributions at three depths --------------------
    depth = {}
    for lab, fn in (("t0", lambda T: 0), ("mid", lambda T: T // 2),
                    ("last", lambda T: T - 1)):
        b = at(fn, "boot"); rw = at(fn, "g_rew"); tg = at(fn, "g_r2")
        frac = np.abs(b) / np.maximum(np.abs(b) + np.abs(rw), 1e-9)
        depth[lab] = dict(bootstrap=dist(b), reward_only=dist(rw),
                          total_target=dist(tg), abs_bootstrap_share=dist(frac),
                          mean_abs_bootstrap=float(np.abs(b).mean()),
                          mean_abs_reward_only=float(np.abs(rw).mean()),
                          mean_abs_target=float(np.abs(tg).mean()),
                          signed_share=dist(b / np.where(np.abs(tg) < 1e-9,
                                                         np.nan, tg)))
    out["bootstrap_by_depth"] = depth
    out["boundary_V_vs_true_tail"] = out_boundary

    # ---- item D: three-target comparison ----------------------------
    comp = {}
    have = [r for r in rows if r["e"]["cls"] == "trunc" and r["g_cont"] is not None]
    exact = [r for r in have if r["e"]["cont_ended"] == "terminal"]
    print("\n-- stage 3: continuation-MC availability ---------------------")
    print(f"  truncated episodes                     : {counts['trunc']}")
    print(f"  continuation attempted                 : {len(have)}")
    print(f"  reached GENUINE terminal (exact)       : {len(exact)}")
    print(f"  censored by trace end (bounds only)    : {len(have) - len(exact)}")
    if exact:
        t1 = np.concatenate([r["g_r2"].ravel() for r in exact])
        t2 = np.concatenate([r["g_rew"].ravel() for r in exact])
        t3 = np.concatenate([r["g_cont"].ravel() for r in exact])
        def cc(x, y):
            m = np.isfinite(x) & np.isfinite(y)
            return float(np.corrcoef(x[m], y[m])[0, 1]) if m.sum() > 2 else None
        comp = dict(
            n_states=int(t1.size), n_episodes_exact=len(exact),
            R2_target=dist(t1), reward_only_target=dist(t2),
            continuation_MC_target=dist(t3),
            mad_R2_vs_cont=dist(np.abs(t1 - t3)),
            mad_rewardonly_vs_cont=dist(np.abs(t2 - t3)),
            corr_R2_cont=cc(t1, t3), corr_rewardonly_cont=cc(t2, t3),
            mean_signed_R2_minus_cont=float(np.mean(t1 - t3)),
            cont_steps=dist([r["e"]["cont_steps"] for r in exact]),
        )
        print(f"  MAD(R2 target, continuation-MC)        : "
              f"{comp['mad_R2_vs_cont']['mean']:.4f}")
        print(f"  MAD(reward-only, continuation-MC)      : "
              f"{comp['mad_rewardonly_vs_cont']['mean']:.4f}")
        print(f"  mean signed (R2 - continuation)        : "
              f"{comp['mean_signed_R2_minus_cont']:+.4f}")
        print(f"  corr(R2, cont)={comp['corr_R2_cont']:.4f}  "
              f"corr(reward-only, cont)={comp['corr_rewardonly_cont']:.4f}")
    out["three_target_comparison"] = comp

    path = OUT / f"SPRINT_7_RUNG2_5_targets_{a.tag}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\n  wrote {path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
