"""
Sprint 7 Rung 0 — MEASUREMENT ONLY. Four probes, no training, no fix.

This module is diagnostic. It imports the production code and reads saved
checkpoints; it NEVER writes a checkpoint, NEVER steps an optimiser on a loaded
model (every PPO probe runs on a `deepcopy`), and NEVER modifies the
environment, the reward, the action space, the observation space, the cloud
capacity, the risk threshold or the MAPPO architecture. All output goes to new
files named `diag_S7_D*.json`. No Sprint 6 / Sprint 6.5 artifact is touched.

  D1  TRUE vs GAE ADVANTAGE, on identical (state, action) pairs.
      Sprint 6.5's p9 measured only where greedy chose STAY and deviated only
      to EDGE/CLOUD, so `A(STAY)` was zero by construction and n was 30. Here:

        * three pools -- states where greedy chose STAY, where it chose CLOUD,
          and where it chose EDGE -- so every one of the three actions is
          measured as both the reference and the deviation. STAY is no longer
          degenerate.
        * for each sampled state, every legal action in {STAY, EDGE, CLOUD} is
          forced in a separate replay, giving the exact Q(s,a) of the greedy
          policy for all of them. Under a DETERMINISTIC policy
          A(s,a) = Q(s,a) - V(s) = Q(s,a) - Q(s, a_greedy) exactly -- no critic,
          no bootstrap, no sampling noise, no expectation to approximate.
        * the learner's own GAE advantage is computed AT THE SAME ENTRY on the
          trajectory in which that action was actually taken, i.e. exactly what
          PPO would have used had it sampled that action. That is the honest
          pairing; comparing a counterfactual's true value against the
          baseline's GAE would compare two different things.
        * per-state ORDERING AGREEMENT (does argmax_a A_gae equal argmax_a
          A_true?) is the headline: it answers the Rung 0 question directly
          instead of comparing two bucket means with n=4.

      Replay identity is asserted three ways before any deviation is counted:
      the action trail, the per-step team reward, and the full observation
      tensor must match the baseline bit-for-bit up to the deviation step.

  D2  GAE HORIZON vs the actual payoff delay, and the critic residual.
      The exact statement available here, and the one that matters: a reward
      difference arriving K steps after the decision enters the TRUE discounted
      return with weight gamma^K but enters GAE's observed-reward path with
      weight (gamma*lambda)^K. The ratio is lambda^K. Everything else must
      arrive through V. D2 measures K empirically from D1's replays and reports
      the resulting split, plus std(ret - V) restricted to high-risk decision
      states so the signal can be compared against the noise floor.

      One free exact measurement falls out of D1: under a deterministic policy
      the true advantage of the action actually taken is identically zero, so
      any non-zero GAE at a greedy baseline entry IS estimator error. Its
      spread is the noise floor, measured rather than inferred.

  D3  HIGH-RISK SAMPLE CENSUS per PPO update, without retraining.
      Update 1 is exactly reconstructible: a fresh MAPPO seeded 20260818 IS the
      policy that generated update 1's rollout, and the episode starts come from
      `np.random.default_rng(20260818)`. That reconstruction is VERIFIED against
      the recorded per-episode rewards in the A0 history CSV, so the census is
      bit-exact and provably so. The best and final checkpoints give two more
      points; those are the post-update weights re-rolled on the historical
      start ticks, which is one update off and is labelled as such. The 75
      recorded updates in `_updates.csv` supply the real trace for everything
      that was logged at the time.

      The per-minibatch counts are exact, not estimated: `MAPPO.update` builds
      its partition from `np.random.default_rng(0)`, so the same permutation is
      reproduced here and the high-risk EDGE entries are counted per minibatch.

  D4  PPO UPDATE INSTRUMENTATION, dry-run on a deepcopy.
      Adds the k3 KL estimator ((r-1) - log r; non-negative, low variance)
      alongside the existing k1 (mean(old_lp - new_lp), which is why the logged
      approx_kl goes slightly negative). Records actor and critic gradient
      norms PRE-clip, policy loss, value loss, entropy, k1, k3, clip fraction
      and explained variance per minibatch. Nothing here is wired into
      training; `mappo.py` is not modified.

      It also answers item C's real question -- gradient small, or large but
      cancelling? -- by decomposing the full-batch policy gradient at ratio == 1
      into the high-risk-EDGE part, the high-risk-non-EDGE part and the
      low-risk-EDGE part, and reporting their norms and pairwise cosines.

      Fidelity is proved, not assumed: the replica and the real
      `MAPPO.update` are run on two deepcopies of the same agent with the same
      buffer and their reported statistics compared.

    python marl/_diag_rung0.py --probe d1
    python marl/_diag_rung0.py --probe d3
    python marl/_diag_rung0.py --probe d4
"""

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.config import (                                          # noqa: E402
    Sprint6Config, EnvConfig, RewardConfig, MappoConfig, resolve_device,
    ACTION_NAMES, N_ACTIONS,
    ACTION_STAY, ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD,
    ACTION_PREEMPTIVE_REROUTE,
)
from marl.env import DTMarlEnv                                     # noqa: E402
from marl.mappo import MAPPO, RolloutBuffer, masked_dist           # noqa: E402
from marl.rollout import episode_starts                            # noqa: E402
from marl.train import TRAIN_FRAC                                  # noqa: E402

IDX_RISK = 12
IDX_HAS_TASK = 15
LEGACY_HIGH_RISK = 0.18       # the threshold Sprint 6.5 bucketed on
MEASURED = (ACTION_STAY, ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD)
OUT_DIR = _ROOT / "saved_models" / "marl"
DEFAULT_MODEL = OUT_DIR / "mappo_A0_cpu_repro.pth"

# Rung 0 stratification, exactly as specified: risk < 0.10, risk > 0.50, and
# whatever naturally occurs in between. No synthetic middle-risk state is
# created anywhere in this file.
def risk_bucket(r):
    if r < 0.10:
        return "lo"
    if r > 0.50:
        return "hi"
    return "mid"


BUCKETS = ("lo", "mid", "hi")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sprint 7 Rung 0 diagnostics")
    p.add_argument("--probe", choices=["d1", "d2", "d3", "d4", "all"],
                   default="all")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--device", default="cpu")
    p.add_argument("--window", choices=["train", "eval"], default="train",
                   help="episode-start window for D1/D2. 'train' is the "
                        "distribution PPO's gradient was actually computed on")
    p.add_argument("--episodes", type=int, default=16,
                   help="D1 baseline episodes (multiples of 8 mirror one "
                        "PPO batch each for advantage normalisation)")
    p.add_argument("--deviations", type=int, default=150,
                   help="D1 states per (pool, risk bucket)")
    p.add_argument("--pool-b-cap", type=int, default=80)
    p.add_argument("--pool-c-cap", type=int, default=40)
    p.add_argument("--tag", default="")
    return p.parse_args(argv)


# ==========================================================================
# shared: configuration restored from the checkpoint, replay, GAE replica
# ==========================================================================

def load_agent_and_cfg(model_path, device, window="train"):
    """Restore the EXACT env/reward/mappo configuration the checkpoint was
    trained with, the way evaluate.py does. Nothing is defaulted silently."""
    agent, extra = MAPPO.load(model_path, device=device)
    saved = extra.get("config", {})
    cfg = Sprint6Config()
    if saved:
        cfg.env = EnvConfig(**saved["env"])
        cfg.reward = RewardConfig(**saved["reward"])
        cfg.mappo = MappoConfig(**saved["mappo"])
        for k, v in saved.get("train", {}).items():
            if hasattr(cfg.train, k):
                setattr(cfg.train, k, v)
    if window == "train":
        cfg.env.start_frac_lo, cfg.env.start_frac_hi = 0.0, TRAIN_FRAC
    else:
        cfg.env.start_frac_lo, cfg.env.start_frac_hi = TRAIN_FRAC, 1.0
    return agent, extra, cfg


def _replay(env, agent, start, seed, override=None, record=True, sample=False):
    """
    One episode. `override=(step, agent, action)` forces a single action at a
    single step; everything else is the policy's own choice.

    Greedy replay is deterministic: arrival times come from a closed formula in
    reset() and act_greedy is an argmax, so no RNG is consulted after t0. That
    is what makes the counterfactual exact.
    """
    obs, state, masks = env.reset(episode_start_tick=start, seed=seed)
    ov_step, ov_agent, ov_act = (-1, -1, -1) if override is None else override
    done, step = False, 0
    team_r, trail = [], []
    n = env.n_agents
    R = {k: [] for k in ("obs", "state", "act", "logp", "rew", "mask",
                         "cont", "risk")}
    last_state = state
    while not done:
        if sample:
            a, lp = agent.act(obs, masks)
        else:
            a, lp = agent.act_greedy(obs, masks), np.zeros(n, np.float32)
        if step == ov_step:
            a = a.copy()
            a[ov_agent] = int(ov_act)
            lp = lp.copy()
            lp[ov_agent] = np.nan          # forced, not sampled: no valid logp
        trail.append(tuple(int(x) for x in a))
        if record:
            R["obs"].append(obs.copy())
            R["state"].append(state.copy())
            R["act"].append(a.copy())
            R["logp"].append(lp.copy())
            R["mask"].append(masks.copy())
            R["risk"].append(np.array([env.risk_at(i) for i in range(n)],
                                      np.float32))
        nobs, nstate, rew, done, info = env.step(a)
        if record:
            R["rew"].append(rew.copy())
            R["cont"].append(0.0 if done else 1.0)
        team_r.append(float(rew.sum()))
        obs, state, masks = nobs, nstate, info["action_masks"]
        last_state = nstate
        step += 1

    out = dict(total=float(np.sum(team_r)), team_r=np.array(team_r),
               trail=trail, ep=dict(env.ep), met=env.episode_metrics(),
               n_steps=step)
    if record:
        out["rec"] = {k: np.asarray(v) for k, v in R.items()}
        out["rec"]["boot_state"] = last_state
    return out


def _values(agent, states):
    with torch.no_grad():
        s = torch.as_tensor(np.asarray(states), dtype=torch.float32,
                            device=agent.device)
        return agent.critic(s).cpu().numpy()


def gae_single(agent, rec):
    """
    Exact replica of MAPPO.compute_gae for ONE episode.

    Per-episode is not an approximation: compute_gae's recursion is
    `last = delta + gamma*lambda*cont*last` and cont == 0 on an episode's last
    step, so the trace never crosses an episode boundary inside a multi-episode
    buffer. Verified against agent.compute_gae in verify_gae_replica().
    """
    g, lam = agent.cfg.gamma, agent.cfg.gae_lambda
    v = _values(agent, rec["state"])
    boot = agent.value(rec["boot_state"])
    T = len(v)
    adv = np.zeros_like(v)
    last = np.zeros(v.shape[1], np.float32)
    for t in reversed(range(T)):
        cont = float(rec["cont"][t])
        next_v = v[t + 1] if (cont > 0.5 and t + 1 < T) else boot
        delta = rec["rew"][t] + g * next_v - v[t]
        last = delta + g * lam * cont * last
        adv[t] = last
    return adv, adv + v, v


def build_buffer(agent, recs):
    """A RolloutBuffer filled exactly as train.py fills it, from replays."""
    T = sum(len(r["act"]) for r in recs)
    buf = RolloutBuffer(T, agent.n_agents, agent.obs_dim, agent.state_dim)
    for r in recs:
        v = _values(agent, r["state"])
        boot = agent.value(r["boot_state"])
        for t in range(len(r["act"])):
            buf.add(r["obs"][t], r["state"][t], r["act"][t], r["logp"][t],
                    v[t], r["rew"][t], r["mask"][t], r["cont"][t])
        buf.set_bootstrap(buf.ptr - 1, boot)
    return buf


def verify_gae_replica(agent, rec):
    """gae_single must equal MAPPO.compute_gae on the same episode."""
    buf = build_buffer(agent, [rec])
    adv_ref, ret_ref = agent.compute_gae(buf)
    adv, ret, _ = gae_single(agent, rec)
    return dict(
        max_abs_adv_diff=float(np.max(np.abs(adv_ref - adv))),
        max_abs_ret_diff=float(np.max(np.abs(ret_ref - ret))),
        ok=bool(np.allclose(adv_ref, adv, atol=1e-5)
                and np.allclose(ret_ref, ret, atol=1e-5)))


def _rank(x):
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), np.float64)
    r[order] = np.arange(len(x), dtype=np.float64)
    # average ties, so the rank correlation is the standard one
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
    return (sums / cnt)[inv]


def _spearman(a, b):
    """Rank correlation without a scipy dependency."""
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank(a), _rank(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _stats(v):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return dict(n=0)
    d = dict(n=int(v.size), mean=float(v.mean()), sd=float(v.std()),
             median=float(np.median(v)),
             frac_positive=float((v > 0).mean()))
    d["se"] = (float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1
               else float("nan"))
    return d


# ==========================================================================
# D1 + D2 — one replay pass, two reports
# ==========================================================================

def probe_d1_d2(args, agent, extra, cfg):
    g, lam = agent.cfg.gamma, agent.cfg.gae_lambda
    env = DTMarlEnv(cfg.env, cfg.reward)
    starts = episode_starts(env, args.episodes)
    t0 = time.time()

    print(f"  window        : {args.window} ticks "
          f"[{env._min_start}, {env._max_start}]")
    print(f"  baselines     : {args.episodes} episodes, starts "
          f"{list(map(int, starts))}")

    # ---- 1. baselines, fully recorded -----------------------------------
    base, seed_of = {}, {}
    for j, s in enumerate(starts):
        seed_of[s] = j
        base[s] = _replay(env, agent, s, j, record=True)
    print(f"  baseline mean reward {np.mean([b['total'] for b in base.values()]):+.3f}"
          f"   ({time.time() - t0:.0f}s)")

    replica = verify_gae_replica(agent, base[starts[0]]["rec"])
    print(f"  gae replica vs MAPPO.compute_gae: ok={replica['ok']} "
          f"max|dadv|={replica['max_abs_adv_diff']:.2e}")

    # ---- 2. batch-level advantage normalisation, mirroring update() -----
    # PPO normalises over the whole rollout batch (rollout_episodes episodes),
    # over decision entries only. Reproduce those constants per block so the
    # normalised numbers are on the scale the learner actually saw.
    rpe = int(cfg.train.rollout_episodes)
    blocks = [starts[i:i + rpe] for i in range(0, len(starts), rpe)]
    norm_of, block_of = {}, {}
    baseline_gae, baseline_ret, baseline_val = {}, {}, {}
    for bi, blk in enumerate(blocks):
        recs = [base[s]["rec"] for s in blk]
        buf = build_buffer(agent, recs)
        adv, ret = agent.compute_gae(buf)
        dec = buf.decision[:len(buf)] > 0.5
        a = adv[dec]
        mu, sd = float(a.mean()), float(a.std())
        for s in blk:
            norm_of[s] = (mu, sd)
            block_of[s] = bi
        off = 0
        for s in blk:
            n = len(base[s]["rec"]["act"])
            baseline_gae[s] = adv[off:off + n].copy()
            baseline_ret[s] = ret[off:off + n].copy()
            baseline_val[s] = buf.val[off:off + n].copy()
            off += n
        print(f"  block {bi}: {len(blk)} eps, T={len(buf)}, "
              f"decision_frac={float(buf.decision[:len(buf)].mean()):.4f}, "
              f"adv_mu={mu:+.4f} adv_sd={sd:.4f}")

    # ---- 3. candidate pools, split by the action greedy actually chose ---
    # Pool A: greedy chose STAY   -> forces EDGE / CLOUD  (Sprint 6.5's pool)
    # Pool B: greedy chose CLOUD  -> forces STAY / EDGE   (STAY becomes real)
    # Pool C: greedy chose EDGE   -> forces STAY / CLOUD
    pools = {("A", b): [] for b in BUCKETS}
    pools.update({("B", b): [] for b in BUCKETS})
    pools.update({("C", b): [] for b in BUCKETS})
    preempt_legal = 0
    census = {("A", b): 0 for b in BUCKETS}
    for s in starts:
        rec = base[s]["rec"]
        for t in range(len(rec["act"])):
            for i in range(env.n_agents):
                if rec["obs"][t, i, IDX_HAS_TASK] < 0.5:
                    continue
                m = rec["mask"][t, i]
                if m.sum() < 1.5:
                    continue
                if m[ACTION_PREEMPTIVE_REROUTE] > 0.5:
                    preempt_legal += 1
                a0 = int(rec["act"][t, i])
                pool = {ACTION_STAY: "A", ACTION_MIGRATE_CLOUD: "B",
                        ACTION_MIGRATE_EDGE: "C"}.get(a0)
                if pool is None:
                    continue
                r = float(rec["risk"][t, i])
                pools[(pool, risk_bucket(r))].append((s, t, i, r, a0))

    caps = {"A": args.deviations, "B": args.pool_b_cap, "C": args.pool_c_cap}

    def take(pool, k):
        """Evenly spaced, no RNG: the selection cannot be cherry-picked."""
        if not pool or k <= 0:
            return []
        idx = np.linspace(0, len(pool) - 1, min(k, len(pool))).astype(int)
        return [pool[t] for t in sorted(set(idx.tolist()))]

    selected = {key: take(v, caps[key[0]]) for key, v in pools.items()}
    print("\n  pool sizes (available -> sampled):")
    for key in sorted(pools):
        print(f"    pool {key[0]} risk_{key[1]:<3s}: "
              f"{len(pools[key]):6d} -> {len(selected[key]):4d}")
    print(f"    PREEMPTIVE_REROUTE legal in {preempt_legal} decision entries")

    # ---- 4. forced replays ----------------------------------------------
    rows, mismatch = [], 0
    n_dev_runs = sum(len(v) for v in selected.values()) * 2
    print(f"\n  forced replays to run: ~{n_dev_runs}")
    done_runs = 0
    for (pool, bucket), states in sorted(selected.items()):
        for (s, step, i, risk, a0) in states:
            b = base[s]
            brec = b["rec"]
            legal = [a for a in MEASURED if brec["mask"][step, i, a] > 0.5]
            q_u, q_d, gae_raw, series, dsteps = {}, {}, {}, {}, {}
            # the reference arm is the baseline itself -- no replay needed.
            # Episode length is NOT fixed (an episode ends when every task has
            # completed or been lost), so a deviation can lengthen or shorten
            # it. Rewards after an episode ends are zero by definition, so the
            # two return streams are zero-padded to a common length; that
            # leaves each arm's own Q untouched and makes the per-step
            # difference well defined.
            tb = np.asarray(b["team_r"][step:], dtype=np.float64)
            q_u[a0] = float(tb.sum())
            q_d[a0] = float(np.dot(g ** np.arange(len(tb)), tb))
            gae_raw[a0] = float(baseline_gae[s][step, i])
            dsteps[a0] = 0
            bad = False
            for a in legal:
                if a == a0:
                    continue
                d = _replay(env, agent, s, seed_of[s], record=True,
                            override=(step, i, a))
                done_runs += 1
                # ---- replay identity, asserted three ways ----------------
                if (d["trail"][:step] != b["trail"][:step]
                        or not np.allclose(d["team_r"][:step],
                                           b["team_r"][:step], atol=1e-6)
                        or not np.allclose(d["rec"]["obs"][:step],
                                           brec["obs"][:step], atol=1e-6)):
                    mismatch += 1
                    bad = True
                    break
                td = np.asarray(d["team_r"][step:], dtype=np.float64)
                L = max(len(tb), len(td))
                gam = g ** np.arange(L)
                tb_p = np.pad(tb, (0, L - len(tb)))
                td_p = np.pad(td, (0, L - len(td)))
                q_u[a] = float(td.sum())
                q_d[a] = float(np.dot(gam[:len(td)], td))
                adv_d, _, _ = gae_single(agent, d["rec"])
                gae_raw[a] = float(adv_d[step, i])
                series[a] = td_p - tb_p
                dsteps[a] = int(d["n_steps"] - b["n_steps"])
            if bad:
                continue
            if bad:
                continue

            mu, sd = norm_of[s]
            row = dict(
                pool=pool, bucket=bucket, start=int(s), step=int(step),
                agent=int(i), risk=risk, ref_action=int(a0),
                legal=[int(a) for a in legal],
                q_undisc={int(k): v for k, v in q_u.items()},
                q_disc={int(k): v for k, v in q_d.items()},
                # exact true advantage under the deterministic policy:
                # A(s,a) = Q(s,a) - V(s) = Q(s,a) - Q(s, a_greedy)
                a_true_undisc={int(k): v - q_u[a0] for k, v in q_u.items()},
                a_true_disc={int(k): v - q_d[a0] for k, v in q_d.items()},
                gae_raw={int(k): v for k, v in gae_raw.items()},
                gae_norm={int(k): (v - mu) / (sd + 1e-8)
                          for k, v in gae_raw.items()},
                # everything re-expressed relative to STAY, so the three-way
                # ordering is comparable across pools
                vs_stay_undisc=({int(k): v - q_u[ACTION_STAY]
                                 for k, v in q_u.items()}
                                if ACTION_STAY in q_u else None),
                critic_v=float(baseline_val[s][step, i]),
                critic_ret=float(baseline_ret[s][step, i]),
                block=block_of[s],
                baseline_steps=int(b["n_steps"]),
                d_episode_steps={int(k): v for k, v in dsteps.items()},
            )
            # ---- payoff-delay decomposition (feeds D2) ------------------
            if ACTION_MIGRATE_EDGE in series:
                row["payoff_edge"] = _payoff_profile(series[ACTION_MIGRATE_EDGE],
                                                    g, lam)
            if ACTION_MIGRATE_CLOUD in series:
                row["payoff_cloud"] = _payoff_profile(
                    series[ACTION_MIGRATE_CLOUD], g, lam)
            if ACTION_STAY in series:
                row["payoff_stay"] = _payoff_profile(series[ACTION_STAY],
                                                     g, lam)
            rows.append(row)
        print(f"    pool {pool} risk_{bucket}: {len(states)} states done "
              f"({done_runs} replays, {time.time() - t0:.0f}s)")

    print(f"\n  replay mismatches: {mismatch} / {done_runs}")

    d1 = _summarise_d1(rows, mismatch, done_runs, pools, selected, agent,
                       args, extra, replica, preempt_legal)
    d2 = _summarise_d2(rows, base, baseline_gae, baseline_ret, baseline_val,
                       starts, agent, args, extra)
    return d1, d2


def _payoff_profile(d, g, lam):
    """
    Where in time the counterfactual's payoff actually lands, and how much of
    it GAE's observed-reward path can see.

    A reward difference at offset k enters the true discounted return with
    weight g**k and enters GAE_t with weight (g*lam)**k. So the ratio of the
    two weighted sums is exactly the fraction of the payoff GAE receives as
    REWARD; the remainder has to arrive through the critic's V estimates.
    """
    k = np.arange(len(d))
    wg = g ** k
    wgl = (g * lam) ** k
    true_disc = float(np.dot(wg, d))
    gae_disc = float(np.dot(wgl, d))
    cum = np.cumsum(wg * d)
    fin = cum[-1] if len(cum) else 0.0

    def reach(frac):
        if abs(fin) < 1e-9:
            return -1
        hit = np.nonzero(np.abs(cum) >= frac * abs(fin))[0]
        return int(hit[0]) if hit.size else -1

    nz = np.nonzero(np.abs(d) > 1e-9)[0]
    h_eff = int(round(1.0 / (1.0 - g * lam)))
    return dict(
        undisc=float(d.sum()), true_disc=true_disc, gae_reward_path=gae_disc,
        gae_capture=(gae_disc / true_disc if abs(true_disc) > 1e-9
                     else float("nan")),
        first_nonzero_k=int(nz[0]) if nz.size else -1,
        last_nonzero_k=int(nz[-1]) if nz.size else -1,
        argmax_abs_k=int(np.argmax(np.abs(d))) if len(d) else -1,
        k50=reach(0.5), k90=reach(0.9), k95=reach(0.95),
        frac_within_h_eff=(float(cum[min(h_eff, len(cum) - 1)] / fin)
                           if abs(fin) > 1e-9 else float("nan")),
        horizon=int(len(d)))


def _summarise_d1(rows, mismatch, runs, pools, selected, agent, args, extra,
                  replica, preempt_legal):
    out = dict(
        probe="D1", what="true counterfactual advantage vs the learner's GAE, "
                         "identical (state, action) pairs, greedy policy",
        model=args.model, window=args.window,
        checkpoint_kind=extra.get("kind"), checkpoint_episode=extra.get("episode"),
        policy="greedy (argmax over legal) -- deterministic, so "
               "A(s,a)=Q(s,a)-Q(s,a_greedy) is EXACT",
        gamma=agent.cfg.gamma, gae_lambda=agent.cfg.gae_lambda,
        episodes=args.episodes, replay_mismatches=mismatch,
        forced_replays=runs, gae_replica_check=replica,
        preemptive_legal_entries=preempt_legal,
        stratification="risk<0.10 | 0.10<=risk<=0.50 (natural only) | risk>0.50",
        pool_sizes={f"{k[0]}_{k[1]}": len(v) for k, v in sorted(pools.items())},
        sampled_sizes={f"{k[0]}_{k[1]}": len(v)
                       for k, v in sorted(selected.items())},
    )

    # ---- per-cell true advantage and GAE, with every sample count -------
    cells = {}
    for pool in ("A", "B", "C"):
        for bucket in BUCKETS:
            sub = [r for r in rows if r["pool"] == pool and r["bucket"] == bucket]
            cell = dict(n_states=len(sub))
            for a in MEASURED:
                tu = [r["a_true_undisc"][a] for r in sub if a in r["a_true_undisc"]]
                td = [r["a_true_disc"][a] for r in sub if a in r["a_true_disc"]]
                gr = [r["gae_raw"][a] for r in sub if a in r["gae_raw"]]
                gn = [r["gae_norm"][a] for r in sub if a in r["gae_norm"]]
                cell[ACTION_NAMES[a]] = dict(
                    a_true_undiscounted=_stats(tu),
                    a_true_discounted=_stats(td),
                    gae_raw=_stats(gr), gae_normalised=_stats(gn),
                    is_reference=int(sum(1 for r in sub
                                         if r["ref_action"] == a)))
            cells[f"pool{pool}_risk_{bucket}"] = cell
    out["cells"] = cells

    # ---- Q relative to STAY, pooled across pools ------------------------
    vs_stay = {}
    for bucket in BUCKETS:
        sub = [r for r in rows if r["bucket"] == bucket
               and r["vs_stay_undisc"] is not None]
        d = dict(n_states=len(sub))
        for a in MEASURED:
            d[ACTION_NAMES[a]] = _stats([r["vs_stay_undisc"][a] for r in sub
                                         if a in r["vs_stay_undisc"]])
        # true ordering by mean
        means = {a: np.mean([r["vs_stay_undisc"][a] for r in sub
                             if a in r["vs_stay_undisc"]])
                 for a in MEASURED
                 if any(a in r["vs_stay_undisc"] for r in sub)}
        d["ordering_true"] = [ACTION_NAMES[a] for a in
                              sorted(means, key=lambda k: -means[k])]
        vs_stay[f"risk_{bucket}"] = d
    out["q_relative_to_stay"] = vs_stay

    # ---- THE HEADLINE: per-state ordering and sign agreement ------------
    agree = {}
    for bucket in list(BUCKETS) + ["all"]:
        sub = [r for r in rows if bucket == "all" or r["bucket"] == bucket]
        n_pair, n_sign, n_ord, n_ord2 = 0, 0, 0, 0
        tv, gv = [], []
        for r in sub:
            acts = [a for a in MEASURED
                    if a in r["a_true_undisc"] and a in r["gae_raw"]]
            for a in acts:
                if a == r["ref_action"]:
                    continue           # A_true == 0 by construction: no sign
                n_pair += 1
                t, q = r["a_true_undisc"][a], r["gae_raw"][a]
                tv.append(t)
                gv.append(q)
                if np.sign(t) == np.sign(q):
                    n_sign += 1
            if len(acts) >= 2:
                n_ord += 1
                bt = max(acts, key=lambda a: r["a_true_undisc"][a])
                bg = max(acts, key=lambda a: r["gae_raw"][a])
                if bt == bg:
                    n_ord2 += 1
        c = (float(np.corrcoef(tv, gv)[0, 1]) if len(tv) > 2
             and np.std(tv) > 0 and np.std(gv) > 0 else float("nan"))
        rk = _spearman(tv, gv)
        agree[f"risk_{bucket}"] = dict(
            n_deviation_pairs=n_pair,
            sign_agreement=(n_sign / n_pair) if n_pair else float("nan"),
            n_states_with_2plus_measured=n_ord,
            argmax_agreement=(n_ord2 / n_ord) if n_ord else float("nan"),
            pearson_true_vs_gae=c, spearman_true_vs_gae=rk)
    out["agreement"] = agree

    # ---- legacy 0.18 split, for continuity with Sprint 6.5's +2.671 -----
    legacy = {}
    for name, sel in (("risk_gt_0.18", lambda r: r["risk"] > LEGACY_HIGH_RISK),
                      ("risk_le_0.18", lambda r: r["risk"] <= LEGACY_HIGH_RISK)):
        sub = [r for r in rows if r["pool"] == "A" and sel(r)]
        legacy[name] = {ACTION_NAMES[a]: dict(
            a_true_undiscounted=_stats([r["a_true_undisc"][a] for r in sub
                                        if a in r["a_true_undisc"]]),
            gae_raw=_stats([r["gae_raw"][a] for r in sub if a in r["gae_raw"]]))
            for a in (ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD)}
    out["legacy_0_18_split_poolA_only"] = legacy
    out["rows"] = rows
    return out


def _summarise_d2(rows, base, baseline_gae, baseline_ret, baseline_val, starts,
                  agent, args, extra):
    g, lam = agent.cfg.gamma, agent.cfg.gae_lambda
    gl = g * lam
    out = dict(
        probe="D2", what="how much of the delayed payoff is inside the GAE "
                         "horizon, and how big the critic residual is",
        model=args.model, window=args.window,
        gamma=g, gae_lambda=lam, gamma_lambda=gl,
        effective_gae_horizon_steps=1.0 / (1.0 - gl),
        gae_95pct_mass_horizon_steps=float(np.log(0.05) / np.log(gl)),
        gamma_95pct_mass_horizon_steps=float(np.log(0.05) / np.log(g)),
    )

    # ---- empirical decision -> payoff delay -----------------------------
    delay = {}
    for key, field in (("high_risk_EDGE", "payoff_edge"),
                       ("high_risk_CLOUD", "payoff_cloud"),
                       ("high_risk_STAY", "payoff_stay")):
        for bucket in ("hi", "lo", "mid"):
            sub = [r[field] for r in rows
                   if field in r and r["bucket"] == bucket]
            if not sub:
                continue
            delay[f"{key.split('_')[-1]}_risk_{bucket}"] = dict(
                n=len(sub),
                undisc=_stats([x["undisc"] for x in sub]),
                true_discounted=_stats([x["true_disc"] for x in sub]),
                gae_reward_path=_stats([x["gae_reward_path"] for x in sub]),
                gae_capture_fraction=_stats(
                    [x["gae_capture"] for x in sub
                     if np.isfinite(x["gae_capture"])]),
                first_nonzero_k=_stats([x["first_nonzero_k"] for x in sub
                                        if x["first_nonzero_k"] >= 0]),
                argmax_abs_k=_stats([x["argmax_abs_k"] for x in sub
                                     if x["argmax_abs_k"] >= 0]),
                k50=_stats([x["k50"] for x in sub if x["k50"] >= 0]),
                k90=_stats([x["k90"] for x in sub if x["k90"] >= 0]),
                k95=_stats([x["k95"] for x in sub if x["k95"] >= 0]),
                frac_within_effective_horizon=_stats(
                    [x["frac_within_h_eff"] for x in sub
                     if np.isfinite(x["frac_within_h_eff"])]))
    out["payoff_delay"] = delay

    # ---- the analytic split, stated exactly -----------------------------
    out["analytic_reward_path_weight"] = {
        f"K={K}": dict(true_weight_gamma_K=float(g ** K),
                       gae_weight_gammalambda_K=float(gl ** K),
                       ratio_lambda_K=float(lam ** K))
        for K in (1, 20, 50, 100, 150, 167, 200, 300, 363, 373, 400)}

    # ---- critic residual, stratified ------------------------------------
    ret = np.concatenate([baseline_ret[s] for s in starts])
    val = np.concatenate([baseline_val[s] for s in starts])
    adv = np.concatenate([baseline_gae[s] for s in starts])
    risk = np.concatenate([base[s]["rec"]["risk"] for s in starts])
    mask = np.concatenate([base[s]["rec"]["mask"] for s in starts])
    dec = mask.sum(axis=-1) > 1.5
    res = ret - val

    def block(sel, name):
        if sel.sum() == 0:
            return dict(n=0)
        r, v, a = ret[sel], val[sel], adv[sel]
        var_r = float(np.var(r))
        return dict(n=int(sel.sum()),
                    explained_var=(1.0 - float(np.var(r - v)) / var_r
                                   if var_r > 1e-12 else float("nan")),
                    residual_sd=float(np.std(r - v)),
                    residual_mean_abs=float(np.mean(np.abs(r - v))),
                    ret_sd=float(np.std(r)), value_mean=float(np.mean(v)),
                    adv_sd=float(np.std(a)), adv_mean=float(np.mean(a)),
                    adv_mean_abs=float(np.mean(np.abs(a))))

    strat = dict(
        all_entries=block(np.ones_like(dec, bool), "all"),
        decision_entries=block(dec, "dec"),
        decision_risk_gt_0_50=block(dec & (risk > 0.50), "hi"),
        decision_risk_gt_0_18=block(dec & (risk > LEGACY_HIGH_RISK), "hi18"),
        decision_risk_in_0_10_0_50=block(dec & (risk >= 0.10) & (risk <= 0.50),
                                         "mid"),
        decision_risk_lt_0_10=block(dec & (risk < 0.10), "lo"),
    )
    out["critic_residual"] = strat

    # ---- the free exact measurement -------------------------------------
    # Under a DETERMINISTIC policy the true advantage of the action actually
    # taken is identically zero. So on a greedy rollout, every non-zero GAE at
    # a decision entry is pure estimator error. Its spread is the noise floor
    # against which the +2.671 signal has to be resolved.
    out["estimator_noise_floor_greedy"] = dict(
        note="under a deterministic policy A(s, a_taken) == 0 exactly, so any "
             "non-zero GAE at a greedy baseline decision entry IS estimator "
             "error. sd below is a measured noise floor, not an inference.",
        decision_entries=dict(
            n=int(dec.sum()), sd=float(np.std(adv[dec])),
            mean=float(np.mean(adv[dec])),
            mean_abs=float(np.mean(np.abs(adv[dec]))),
            p95_abs=float(np.percentile(np.abs(adv[dec]), 95))),
        decision_risk_gt_0_50=dict(
            n=int((dec & (risk > 0.50)).sum()),
            sd=float(np.std(adv[dec & (risk > 0.50)]))
            if (dec & (risk > 0.50)).sum() else float("nan"),
            mean_abs=float(np.mean(np.abs(adv[dec & (risk > 0.50)])))
            if (dec & (risk > 0.50)).sum() else float("nan")))
    return out


# ==========================================================================
# D3 — per-update high-risk sample census, no retraining
# ==========================================================================

def training_starts(env, seed, episodes):
    """The EXACT episode-start sequence train.py draws, from --seed alone."""
    rng = np.random.default_rng(seed)
    return [int(rng.integers(env._min_start, env._max_start + 1))
            for _ in range(episodes)]


def census_of(buf, recs, cfg, agent):
    """
    Every count Rung 0 asked for, plus the per-minibatch counts, which are
    EXACT rather than estimated: MAPPO.update partitions with
    np.random.default_rng(0), reproduced here.
    """
    T = len(buf)
    risk = np.concatenate([r["risk"] for r in recs])[:T]
    act = buf.act[:T]
    dec = buf.decision[:T] > 0.5
    mask = buf.mask[:T]

    hi = risk > 0.50
    hi18 = risk > LEGACY_HIGH_RISK
    lo = risk < 0.10
    mid = (risk >= 0.10) & (risk <= 0.50)

    def n(x):
        return int(np.asarray(x).sum())

    c = dict(
        steps=T, agents=int(buf.n), total_entries=T * int(buf.n),
        decision_entries=n(dec), decision_frac=float(dec.mean()),
        risk_gt_0_50_all=n(hi), risk_gt_0_18_all=n(hi18),
        risk_lt_0_10_all=n(lo), risk_in_0_10_0_50_all=n(mid),
        risk_gt_0_50_decision=n(dec & hi), risk_gt_0_18_decision=n(dec & hi18),
        risk_lt_0_10_decision=n(dec & lo),
        risk_in_0_10_0_50_decision=n(dec & mid),
    )
    for a in range(N_ACTIONS):
        sel = dec & (act == a)
        c[f"selected_{ACTION_NAMES[a]}"] = n(sel)
        c[f"selected_{ACTION_NAMES[a]}_risk_gt_0_50"] = n(sel & hi)
        c[f"selected_{ACTION_NAMES[a]}_risk_gt_0_18"] = n(sel & hi18)
        c[f"selected_{ACTION_NAMES[a]}_risk_lt_0_10"] = n(sel & lo)
        c[f"legal_{ACTION_NAMES[a]}_decision"] = n(dec & (mask[:, :, a] > 0.5))
        c[f"legal_{ACTION_NAMES[a]}_decision_risk_gt_0_50"] = n(
            dec & hi & (mask[:, :, a] > 0.5))

    # ---- how many useful samples actually reach each minibatch ----------
    useful = dec & hi & (act == ACTION_MIGRATE_EDGE)
    useful18 = dec & hi18 & (act == ACTION_MIGRATE_EDGE)
    n_mb = max(1, agent.cfg.minibatches)
    mb_size = max(1, T // n_mb)
    rng = np.random.default_rng(0)          # exactly what update() uses
    per_mb, per_mb18, denoms, dilution = [], [], [], []
    for _ in range(agent.cfg.ppo_epochs):
        order = rng.permutation(T)
        for start in range(0, T, mb_size):
            idx = order[start:start + mb_size]
            if idx.size == 0:
                continue
            d = float(dec[idx].sum())
            k = int(useful[idx].sum())
            per_mb.append(k)
            per_mb18.append(int(useful18[idx].sum()))
            denoms.append(d)
            dilution.append(k / max(d, 1.0))
    c["useful_highrisk_EDGE_total"] = n(useful)
    c["useful_highrisk_0_18_EDGE_total"] = n(useful18)
    c["minibatches_per_update"] = len(per_mb)
    c["useful_per_minibatch"] = dict(
        mean=float(np.mean(per_mb)), min=int(np.min(per_mb)),
        max=int(np.max(per_mb)),
        frac_minibatches_with_zero=float(np.mean(np.array(per_mb) == 0)),
        counts=[int(x) for x in per_mb])
    c["useful_0_18_per_minibatch"] = dict(
        mean=float(np.mean(per_mb18)), min=int(np.min(per_mb18)),
        max=int(np.max(per_mb18)),
        frac_minibatches_with_zero=float(np.mean(np.array(per_mb18) == 0)))
    c["decision_entries_per_minibatch_mean"] = float(np.mean(denoms))
    c["gradient_weight_share_of_useful_samples"] = dict(
        mean=float(np.mean(dilution)), max=float(np.max(dilution)),
        note="each entry's actor loss is weighted 1/denom where denom is the "
             "decision entries in its minibatch; this is the share of the "
             "actor gradient contributed by high-risk EDGE entries")
    return c


def rollout_block(env, agent, starts, torch_seed):
    """Sampling rollout -- TRAINING conditions, not greedy evaluation."""
    torch.manual_seed(torch_seed)
    recs, rewards = [], []
    for j, s in enumerate(starts):
        r = _replay(env, agent, s, j, record=True, sample=True)
        recs.append(r["rec"])
        rewards.append(r["met"]["episode_reward"])
    return recs, rewards


def probe_d3(args, cfg_model_agent, cfg, extra):
    seed = int(cfg.train.seed)
    rpe = int(cfg.train.rollout_episodes)
    env = DTMarlEnv(cfg.env, cfg.reward)
    all_starts = training_starts(env, seed, int(cfg.train.episodes))

    out = dict(
        probe="D3",
        what="per-update census of decision states, risk bands and action "
             "selections, and how many high-risk EDGE samples reach each "
             "minibatch. NO RETRAINING.",
        seed=seed, rollout_episodes=rpe, episodes=int(cfg.train.episodes),
        train_start_window=[int(env._min_start), int(env._max_start)],
        minibatches=agent_cfg_int(cfg_model_agent, "minibatches"),
        ppo_epochs=agent_cfg_int(cfg_model_agent, "ppo_epochs"),
        snapshots={},
    )

    # ---- the recorded trace: real data for all 75 updates ---------------
    upd = OUT_DIR / (Path(args.model).stem + "_updates.csv")
    if upd.exists():
        with open(upd) as f:
            rec = list(csv.DictReader(f))
        out["recorded_updates_csv"] = str(upd)
        out["recorded_updates"] = [
            {k: (float(v) if k not in ("update", "episode") else int(v))
             for k, v in r.items()} for r in rec]

    # ---- snapshot 1: update 1, EXACTLY reconstructible -------------------
    # train.py does torch.manual_seed(seed) then constructs MAPPO(seed=seed).
    # Reproducing that gives the identical initial actor AND the identical
    # torch RNG state entering episode 1, so the rollout is bit-exact. Proven
    # below against the recorded per-episode rewards.
    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    fresh = MAPPO(env.n_agents, env.obs_dim, env.state_dim, cfg.mappo,
                  device=str(cfg_model_agent.device), seed=seed)
    fresh.train_mode()
    starts1 = all_starts[:rpe]
    recs1, rew1 = [], []
    for j, s in enumerate(starts1):
        r = _replay(env, fresh, s, j, record=True, sample=True)
        recs1.append(r["rec"])
        rew1.append(float(r["met"]["episode_reward"]))
    buf1 = build_buffer(fresh, recs1)
    c1 = census_of(buf1, recs1, cfg, fresh)

    # verification against the historical record
    hist = OUT_DIR / (Path(args.model).stem + "_history.csv")
    verify = dict(available=hist.exists())
    if hist.exists():
        with open(hist) as f:
            rows = list(csv.DictReader(f))
        rec_start = [int(r["start_tick"]) for r in rows[:rpe]]
        rec_rew = [float(r["reward"]) for r in rows[:rpe]]
        verify.update(
            recorded_start_ticks=rec_start, reproduced_start_ticks=starts1,
            starts_match=bool(rec_start == list(starts1)),
            recorded_rewards=rec_rew, reproduced_rewards=rew1,
            max_abs_reward_diff=float(np.max(np.abs(
                np.array(rec_rew) - np.array(rew1)))),
            bit_exact=bool(rec_start == list(starts1)
                           and np.allclose(rec_rew, rew1, atol=1e-3)))
    c1["reproduction_check"] = verify
    c1["episode_rewards"] = rew1
    c1["fidelity"] = ("EXACT: this is the policy and the RNG state that "
                      "generated update 1's rollout")
    out["snapshots"]["update_01_init"] = c1
    out["_buffers"] = {"update_01_init": (buf1, recs1, fresh)}
    print(f"  update 01 (init) : bit_exact={verify.get('bit_exact')} "
          f"useful_highrisk_EDGE={c1['useful_highrisk_EDGE_total']}")

    # ---- snapshots 2-3: best and final checkpoints -----------------------
    for label, path in (("best", OUT_DIR / (Path(args.model).stem + "_best.pth")),
                        ("final", Path(args.model))):
        if not path.exists():
            continue
        a2, x2 = MAPPO.load(str(path), device=str(cfg_model_agent.device))
        a2.train_mode()
        ep = int(x2.get("episode", cfg.train.episodes))
        upd_id = max(1, ep // rpe)
        starts_k = all_starts[max(0, ep - rpe):ep]
        recs_k, rew_k = rollout_block(env, a2, starts_k, seed + 1000 + upd_id)
        buf_k = build_buffer(a2, recs_k)
        ck = census_of(buf_k, recs_k, cfg, a2)
        ck["checkpoint"] = str(path)
        ck["checkpoint_episode"] = ep
        ck["approximates_update"] = upd_id
        ck["episode_rewards"] = rew_k
        ck["start_ticks"] = starts_k
        ck["fidelity"] = (
            "APPROXIMATE: these are the POST-update weights re-rolled on the "
            "historical start ticks, i.e. one PPO update later than the policy "
            "that produced update %d's rollout, and on a fresh torch RNG "
            "stream. Not bit-exact. Retraining was out of scope." % upd_id)
        out["snapshots"][f"update_{upd_id:02d}_{label}"] = ck
        out["_buffers"][f"update_{upd_id:02d}_{label}"] = (buf_k, recs_k, a2)
        print(f"  update {upd_id:02d} ({label:5s}): "
              f"useful_highrisk_EDGE={ck['useful_highrisk_EDGE_total']}  "
              f"per_mb_mean={ck['useful_per_minibatch']['mean']:.2f}  "
              f"zero_mb_frac={ck['useful_per_minibatch']['frac_minibatches_with_zero']:.2f}")
    return out


def agent_cfg_int(agent, name):
    return int(getattr(agent.cfg, name))


# ==========================================================================
# D4 — PPO update instrumentation, dry-run on deepcopies
# ==========================================================================

def instrumented_update(agent, buf, step_optimisers=True):
    """
    Byte-for-byte replica of MAPPO.update, plus diagnostics. The ONLY
    behavioural differences from production are additive measurements:
    the k3 KL estimator, pre-clip gradient norms (clip_grad_norm_ already
    returns them; production discards the value) and per-minibatch traces.
    The optimisation itself -- losses, clipping, ordering, RNG -- is identical.

    Always call on a deepcopy. Nothing here is ever saved.
    """
    cfg = agent.cfg
    T = len(buf)
    adv_np, ret_np = agent.compute_gae(buf)

    obs = torch.as_tensor(buf.obs[:T], device=agent.device)
    state = torch.as_tensor(buf.state[:T], device=agent.device)
    act = torch.as_tensor(buf.act[:T], device=agent.device)
    old_lp = torch.as_tensor(buf.logp[:T], device=agent.device)
    old_v = torch.as_tensor(buf.val[:T], device=agent.device)
    mask = torch.as_tensor(buf.mask[:T], device=agent.device)
    dec = torch.as_tensor(buf.decision[:T], device=agent.device)
    adv = torch.as_tensor(adv_np, device=agent.device)
    ret = torch.as_tensor(ret_np, device=agent.device)

    if cfg.normalise_advantages:
        sel = dec > 0.5
        if sel.any():
            a = adv[sel]
            adv = (adv - a.mean()) / (a.std() + 1e-8)

    n_mb = max(1, cfg.minibatches)
    mb_size = max(1, T // n_mb)
    rng = np.random.default_rng(0)
    stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
             "approx_kl": 0.0, "clip_frac": 0.0, "n": 0}
    trace = []

    def actor_grad_norm():
        tot = 0.0
        for p in agent.actor.parameters():
            if p.grad is not None:
                tot += float(p.grad.detach().pow(2).sum())
        return float(np.sqrt(tot))

    for epoch in range(cfg.ppo_epochs):
        order = rng.permutation(T)
        for mb, start in enumerate(range(0, T, mb_size)):
            idx = torch.as_tensor(order[start:start + mb_size],
                                  dtype=torch.long, device=agent.device)
            if idx.numel() == 0:
                continue

            new_lp, ent = [], []
            for i in range(agent.n_agents):
                d = masked_dist(agent.actor.logits(i, obs[idx, i, :]),
                                mask[idx, i, :])
                new_lp.append(d.log_prob(act[idx, i]))
                ent.append(d.entropy())
            new_lp = torch.stack(new_lp, dim=1)
            ent = torch.stack(ent, dim=1)

            d_mb = dec[idx]
            denom = d_mb.sum().clamp(min=1.0)

            ratio = torch.exp(new_lp - old_lp[idx])
            s1 = ratio * adv[idx]
            s2 = torch.clamp(ratio, 1.0 - cfg.clip_eps,
                             1.0 + cfg.clip_eps) * adv[idx]
            pg = -(torch.min(s1, s2) * d_mb).sum() / denom
            ent_loss = -(ent * d_mb).sum() / denom
            actor_loss = pg + cfg.entropy_coef * ent_loss

            agent.opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            g_actor_pre = float(nn.utils.clip_grad_norm_(
                agent.actor.parameters(), cfg.max_grad_norm))
            if step_optimisers:
                agent.opt_actor.step()

            v = agent.critic(state[idx])
            v_clipped = old_v[idx] + torch.clamp(
                v - old_v[idx], -cfg.value_clip_eps, cfg.value_clip_eps)
            vl = torch.max(F.mse_loss(v, ret[idx], reduction="none"),
                           F.mse_loss(v_clipped, ret[idx], reduction="none"))
            critic_loss = cfg.value_coef * vl.mean()

            agent.opt_critic.zero_grad(set_to_none=True)
            critic_loss.backward()
            g_critic_pre = float(nn.utils.clip_grad_norm_(
                agent.critic.parameters(), cfg.max_grad_norm))
            if step_optimisers:
                agent.opt_critic.step()

            with torch.no_grad():
                # k1: what production logs. Unbiased in expectation but high
                # variance and frequently NEGATIVE on a sample, which is why
                # the recorded approx_kl reaches -0.000033.
                k1 = ((old_lp[idx] - new_lp) * d_mb).sum() / denom
                # k3: (r - 1) - log r. Non-negative by construction, same
                # expectation, far lower variance. Schulman's estimator.
                lr_ = new_lp - old_lp[idx]
                r_ = torch.exp(lr_)
                k3 = (((r_ - 1.0) - lr_) * d_mb).sum() / denom
                cf = (((ratio - 1.0).abs() > cfg.clip_eps).float()
                      * d_mb).sum() / denom
                rr = ratio[d_mb > 0.5]

            stats["actor_loss"] += float(pg.item())
            stats["critic_loss"] += float(critic_loss.item())
            stats["entropy"] += float(-ent_loss.item())
            stats["approx_kl"] += float(k1.item())
            stats["clip_frac"] += float(cf.item())
            stats["n"] += 1
            trace.append(dict(
                epoch=epoch, minibatch=mb, size=int(idx.numel()),
                decision_entries=float(denom.item()),
                policy_loss=float(pg.item()),
                value_loss=float(critic_loss.item()),
                entropy=float(-ent_loss.item()),
                approx_kl_k1=float(k1.item()), kl_k3=float(k3.item()),
                clip_frac=float(cf.item()),
                actor_grad_norm_preclip=g_actor_pre,
                critic_grad_norm_preclip=g_critic_pre,
                actor_grad_norm_postclip=actor_grad_norm(),
                actor_grad_clipped=bool(g_actor_pre > cfg.max_grad_norm),
                ratio_mean=float(rr.mean().item()) if rr.numel() else float("nan"),
                ratio_sd=float(rr.std().item()) if rr.numel() > 1 else float("nan"),
                ratio_min=float(rr.min().item()) if rr.numel() else float("nan"),
                ratio_max=float(rr.max().item()) if rr.numel() else float("nan"),
            ))

    n = max(1, stats.pop("n"))
    out = {k: v / n for k, v in stats.items()}
    out["adv_mean"] = float(adv_np.mean())
    out["adv_std"] = float(adv_np.std())
    out["value_mean"] = float(buf.val[:T].mean())
    out["decision_frac"] = float(buf.decision[:T].mean())
    with torch.no_grad():
        r_flat = ret.reshape(-1)
        v_now = agent.critic(state).reshape(-1)
        var_r = torch.var(r_flat)
        out["explained_var"] = (float(1.0 - torch.var(r_flat - v_now) / var_r)
                                if float(var_r) > 1e-12 else float("nan"))
    out["kl_k3"] = float(np.mean([t["kl_k3"] for t in trace]))
    out["actor_grad_norm_preclip_mean"] = float(
        np.mean([t["actor_grad_norm_preclip"] for t in trace]))
    out["critic_grad_norm_preclip_mean"] = float(
        np.mean([t["critic_grad_norm_preclip"] for t in trace]))
    out["_trace"] = trace
    return out


def gradient_decomposition(agent, buf, recs):
    """
    Item C's real question: is the actor gradient SMALL, or LARGE BUT
    CANCELLING?

    Measured at ratio == 1 (before any optimiser step), on the FULL batch, so
    the clipped surrogate reduces to the plain policy gradient. The batch loss
    is restricted to a subset of entries and its gradient norm recorded; the
    pairwise cosines then say whether the high-risk EDGE gradient points with
    or against the aggregate.
    """
    cfg = agent.cfg
    T = len(buf)
    adv_np, _ = agent.compute_gae(buf)
    obs = torch.as_tensor(buf.obs[:T], device=agent.device)
    act = torch.as_tensor(buf.act[:T], device=agent.device)
    old_lp = torch.as_tensor(buf.logp[:T], device=agent.device)
    mask = torch.as_tensor(buf.mask[:T], device=agent.device)
    dec = torch.as_tensor(buf.decision[:T], device=agent.device)
    adv = torch.as_tensor(adv_np, device=agent.device)
    if cfg.normalise_advantages:
        sel = dec > 0.5
        if sel.any():
            a = adv[sel]
            adv = (adv - a.mean()) / (a.std() + 1e-8)

    risk = torch.as_tensor(np.concatenate([r["risk"] for r in recs])[:T],
                           device=agent.device)
    d = dec > 0.5
    hi = risk > 0.50
    e = act == ACTION_MIGRATE_EDGE
    subsets = {
        "all_decision": d,
        "highrisk_EDGE": d & hi & e,
        "highrisk_not_EDGE": d & hi & (~e),
        "lowrisk_EDGE": d & (risk < 0.10) & e,
        "lowrisk_not_EDGE": d & (risk < 0.10) & (~e),
    }

    new_lp, ent = [], []
    for i in range(agent.n_agents):
        dist = masked_dist(agent.actor.logits(i, obs[:, i, :]), mask[:, i, :])
        new_lp.append(dist.log_prob(act[:, i]))
        ent.append(dist.entropy())
    new_lp = torch.stack(new_lp, dim=1)
    ent = torch.stack(ent, dim=1)
    ratio = torch.exp(new_lp - old_lp)

    denom_all = d.float().sum().clamp(min=1.0)
    params = [p for p in agent.actor.parameters()]
    flat = {}
    counts = {}
    for name, sel in subsets.items():
        w = sel.float()
        counts[name] = int(sel.sum())
        if counts[name] == 0:
            flat[name] = None
            continue
        # same 1/denom weighting production uses, so the norms are comparable
        # to the real gradient rather than rescaled per subset
        loss = -(torch.min(ratio * adv,
                           torch.clamp(ratio, 1 - cfg.clip_eps,
                                       1 + cfg.clip_eps) * adv) * w).sum() \
            / denom_all
        gs = torch.autograd.grad(loss, params, retain_graph=True,
                                 allow_unused=True)
        flat[name] = torch.cat([(torch.zeros_like(p) if gi is None else gi)
                                .reshape(-1) for p, gi in zip(params, gs)])
    ent_loss = -(ent * d.float()).sum() / denom_all
    gs = torch.autograd.grad(cfg.entropy_coef * ent_loss, params,
                             retain_graph=True, allow_unused=True)
    flat["entropy_term"] = torch.cat(
        [(torch.zeros_like(p) if gi is None else gi).reshape(-1)
         for p, gi in zip(params, gs)])
    counts["entropy_term"] = int(d.sum())

    out = dict(
        note="gradients of the PPO policy loss at ratio==1, full batch, each "
             "restricted to a subset of entries but divided by the SAME denom "
             "production uses, so norms are directly comparable and additive",
        entries=counts,
        norms={k: (float(v.norm().item()) if v is not None else None)
               for k, v in flat.items()})
    cos = {}
    keys = [k for k, v in flat.items() if v is not None]
    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            a, b = flat[k1], flat[k2]
            na, nb = float(a.norm()), float(b.norm())
            cos[f"{k1}|{k2}"] = (float((a @ b).item() / (na * nb))
                                 if na > 1e-12 and nb > 1e-12 else float("nan"))
    out["cosine"] = cos
    if flat.get("highrisk_EDGE") is not None and flat.get("all_decision") is not None:
        na = float(flat["all_decision"].norm())
        out["highrisk_EDGE_share_of_aggregate_norm"] = (
            float(flat["highrisk_EDGE"].norm()) / na if na > 1e-12
            else float("nan"))
    return out


def probe_d4(args, buffers, cfg):
    out = dict(
        probe="D4",
        what="PPO update instrumentation: k3 KL added as a DIAGNOSTIC, grad "
             "norms, losses, entropy, clip fraction, explained variance. "
             "mappo.py NOT modified; every measurement runs on a deepcopy and "
             "no checkpoint is written.",
        k1_definition="mean(old_logp - new_logp) over decision entries "
                      "(what production logs; sample-negative, high variance)",
        k3_definition="mean((r-1) - log r) over decision entries, "
                      "r = exp(new_logp - old_logp) (non-negative, low variance)",
        snapshots={})
    for label, (buf, recs, agent) in buffers.items():
        # fidelity: real update vs replica, two deepcopies, same buffer
        a_real = copy.deepcopy(agent)
        a_rep = copy.deepcopy(agent)
        real = a_real.update(buf)
        rep = instrumented_update(a_rep, buf, step_optimisers=True)
        keys = ["actor_loss", "critic_loss", "entropy", "approx_kl",
                "clip_frac", "adv_mean", "adv_std", "value_mean",
                "decision_frac", "explained_var"]
        diff = {k: abs(float(real[k]) - float(rep[k])) for k in keys}
        # gradient decomposition on a third copy, no steps taken
        a_grad = copy.deepcopy(agent)
        decomp = gradient_decomposition(a_grad, buf, recs)

        trace = rep.pop("_trace")
        out["snapshots"][label] = dict(
            reported=rep, real_update_reported=({k: float(real[k]) for k in keys}),
            replica_fidelity=dict(max_abs_diff=max(diff.values()), per_key=diff,
                                  ok=bool(max(diff.values()) < 1e-4)),
            first_minibatch=trace[0], last_minibatch=trace[-1],
            trace=trace,
            gradient_decomposition=decomp,
            kl_k1_vs_k3=dict(
                k1_mean=float(np.mean([t["approx_kl_k1"] for t in trace])),
                k1_min=float(np.min([t["approx_kl_k1"] for t in trace])),
                k1_negative_minibatches=int(sum(
                    1 for t in trace if t["approx_kl_k1"] < 0)),
                k3_mean=float(np.mean([t["kl_k3"] for t in trace])),
                k3_min=float(np.min([t["kl_k3"] for t in trace])),
                k3_negative_minibatches=int(sum(
                    1 for t in trace if t["kl_k3"] < 0)),
                n_minibatches=len(trace)),
        )
        print(f"  {label:22s} replica ok={out['snapshots'][label]['replica_fidelity']['ok']} "
              f"(max|d|={max(diff.values()):.2e})  "
              f"grad_actor={rep['actor_grad_norm_preclip_mean']:.4f}  "
              f"k1={rep['approx_kl']:+.6f}  k3={rep['kl_k3']:.6f}  "
              f"clip={rep['clip_frac']:.4f}")
        print(f"    grad norms: {decomp['norms']}")
        print(f"    entries   : {decomp['entries']}")
    return out


# ==========================================================================
# main
# ==========================================================================

def _dump(obj, path):
    def enc(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        raise TypeError(type(o))
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=enc)
    print(f"  wrote {path}")


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.device)
    agent, extra, cfg = load_agent_and_cfg(args.model, device, args.window)
    suffix = f"_{args.tag}" if args.tag else ""

    print("=" * 78)
    print("SPRINT 7 RUNG 0 — MEASUREMENT ONLY (no training, no fix)")
    print("=" * 78)
    print(f"  checkpoint    : {args.model}")
    print(f"  kind={extra.get('kind')} episode={extra.get('episode')} "
          f"train mean reward={extra.get('mean_reward', float('nan')):+.2f}")
    print(f"  gamma={agent.cfg.gamma} lambda={agent.cfg.gae_lambda} "
          f"minibatches={agent.cfg.minibatches} epochs={agent.cfg.ppo_epochs}")
    print(f"  device        : {device}")
    print()

    if args.probe in ("d1", "d2", "all"):
        print("-" * 78)
        print("D1 + D2 — true vs GAE advantage; GAE horizon and critic residual")
        print("-" * 78)
        d1, d2 = probe_d1_d2(args, agent, extra, cfg)
        _dump(d1, OUT_DIR / f"diag_S7_D1_advantage_fidelity{suffix}.json")
        _dump(d2, OUT_DIR / f"diag_S7_D2_horizon_residual{suffix}.json")

    if args.probe in ("d3", "d4", "all"):
        print("-" * 78)
        print("D3 — high-risk sample census per PPO update (no retraining)")
        print("-" * 78)
        d3 = probe_d3(args, agent, cfg, extra)
        buffers = d3.pop("_buffers")
        _dump(d3, OUT_DIR / f"diag_S7_D3_sample_census{suffix}.json")

        if args.probe in ("d4", "all"):
            print("-" * 78)
            print("D4 — PPO update instrumentation (dry-run on deepcopies)")
            print("-" * 78)
            d4 = probe_d4(args, buffers, cfg)
            _dump(d4, OUT_DIR / f"diag_S7_D4_ppo_update{suffix}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
