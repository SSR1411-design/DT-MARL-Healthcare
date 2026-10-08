#!/usr/bin/env python
"""
SPRINT 7 - RUNG 1: offline critic diagnostic.

Question: is the centralised critic the source of the wrong-signed learning
target measured in Rung 0?

This file is a DIAGNOSTIC. It is clearly separated from production MAPPO:
  * it imports production code and Rung 0's replay machinery, and modifies
    neither;
  * it never writes a production checkpoint and never steps a production
    optimiser -- every fitted critic is a FRESH module, and every agent used
    for evaluation is a copy.deepcopy with its .critic swapped;
  * it does not touch reward, environment, action space, observation space,
    risk predictor, cloud_slots, actor architecture, gamma/lambda, or any PPO
    hyperparameter;
  * it runs NO PPO training. No actor gradient is ever taken.
  * every artifact it writes is named SPRINT_7_RUNG1_*.

Reuse: the deviation pairs are read straight out of Rung 0's
diag_S7_D1_advantage_fidelity.json, so the state/action pairs, pools and
stratification are IDENTICAL to Rung 0 by construction, not by re-derivation.

Arms (each differs from C1 in exactly ONE thing):
  C0  production critic as loaded                          [reference]
  C1  fresh critic, MC discounted-return targets, plain MSE,
      fit to convergence with held-out early stopping      [can it fit?]
  C2  = C1 but lambda-return targets recomputed from the critic being trained
                                                           [target construction]
  C3  = C1 but production's clipped value loss (value_clip_eps)
                                                           [value clipping]

A note on the Rung 0 confound this file resolves. Rung 0 compared a TEAM-level
true advantage (built from team_r = rew.sum()) against a PER-AGENT GAE
(adv[step, i]). Rewards from env.step are per-agent, and a migration moves the
task -- and its reward -- to another node. So a team gain can be a per-agent
loss, which would flip the sign with no critic fault at all. Every true
advantage below is therefore reported under BOTH attributions.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[0]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from marl.config import (                                          # noqa: E402
    ACTION_NAMES, ACTION_STAY, ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD,
)
from marl.env import DTMarlEnv                                     # noqa: E402
from marl.mappo import CentralisedCritic                           # noqa: E402
from marl.rollout import episode_starts                            # noqa: E402

from marl._diag_rung0 import (                                     # noqa: E402
    load_agent_and_cfg, _replay, _values, gae_single, build_buffer,
    risk_bucket, _stats, _spearman, verify_gae_replica,
    IDX_RISK, LEGACY_HIGH_RISK, MEASURED, OUT_DIR, DEFAULT_MODEL, BUCKETS,
)

D1_ARTIFACT = OUT_DIR / "diag_S7_D1_advantage_fidelity.json"
TAG = "SPRINT_7_RUNG1"
ARMS = ("C0", "C1", "C2", "C3")
ARM_WHAT = {
    "C0": "production critic as loaded (reference; reproduces Rung 0)",
    "C1": "fresh critic, MC discounted-return targets, plain MSE, "
          "fit to convergence with held-out early stopping",
    "C2": "= C1 but lambda-return targets recomputed from the critic being "
          "trained (isolates target construction)",
    "C3": "= C1 but production's clipped value loss at value_clip_eps "
          "(isolates value clipping)",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sprint 7 Rung 1 offline critic diagnostic")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--device", default="cpu")
    p.add_argument("--window", default="train", choices=("train", "eval"))
    p.add_argument("--episodes", type=int, default=16,
                   help="baseline episodes; must match Rung 0 for apples-to-apples")
    p.add_argument("--val-episodes", type=int, default=4,
                   help="held-out episodes for critic early stopping")
    p.add_argument("--max-epochs", type=int, default=600)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="default is production's lr_critic")
    p.add_argument("--phase", default="all",
                   choices=("data", "fit", "deviations", "tail", "reagg", "all"))
    p.add_argument("--max-pairs", type=int, default=0,
                   help="0 = all pairs from the D1 artifact (apples-to-apples)")
    return p.parse_args(argv)


# ==========================================================================
# phase A -- baseline data, identical starts/seeds to Rung 0 D1/D2
# ==========================================================================

def collect_baselines(agent, cfg, episodes):
    env = DTMarlEnv(cfg.env, cfg.reward)
    starts = episode_starts(env, episodes)
    base, seed_of = {}, {}
    t0 = time.time()
    for j, s in enumerate(starts):
        seed_of[s] = j
        base[s] = _replay(env, agent, s, j, record=True)
    print(f"  window     : {cfg.env.start_frac_lo:.2f}-{cfg.env.start_frac_hi:.2f} "
          f"ticks [{env._min_start}, {env._max_start}]")
    print(f"  starts     : {list(map(int, starts))}")
    print(f"  baseline mean reward {np.mean([b['total'] for b in base.values()]):+.4f}"
          f"  ({time.time() - t0:.0f}s)")
    replica = verify_gae_replica(agent, base[starts[0]]["rec"])
    print(f"  gae replica vs MAPPO.compute_gae: ok={replica['ok']} "
          f"max|dadv|={replica['max_abs_adv_diff']:.2e}")
    return env, starts, base, seed_of, replica


def mc_returns(rec, gamma, n_steps, episode_steps):
    """
    Monte-Carlo discounted per-agent return. EXACT and critic-free when the
    episode ends naturally (no reward exists past the end, so the tail is 0).

    Returns (G, truncated). `truncated` is True if the episode hit the step
    limit, in which case pure MC understates the tail and the row is flagged
    rather than silently bootstrapped -- bootstrapping would reintroduce the
    very critic dependence this target exists to avoid.
    """
    rew = np.asarray(rec["rew"], dtype=np.float64)      # (T, n_agents)
    T = rew.shape[0]
    G = np.zeros_like(rew)
    run = np.zeros(rew.shape[1], dtype=np.float64)
    for t in reversed(range(T)):
        run = rew[t] + gamma * run
        G[t] = run
    return G.astype(np.float32), bool(n_steps >= episode_steps)


def build_dataset(agent, base, starts, cfg, val_episodes):
    """States, MC targets, risk, per-episode index. Split by EPISODE."""
    g = agent.cfg.gamma
    ep_steps = int(cfg.env.episode_steps)
    S, Y, R, EP = [], [], [], []
    trunc = []
    for k, s in enumerate(starts):
        rec = base[s]["rec"]
        G, tr = mc_returns(rec, g, base[s]["n_steps"], ep_steps)
        S.append(np.asarray(rec["state"], np.float32))
        Y.append(G)
        R.append(np.asarray(rec["risk"], np.float32))
        EP.append(np.full(len(G), k, np.int32))
        trunc.append(tr)
    S = np.concatenate(S); Y = np.concatenate(Y)
    R = np.concatenate(R); EP = np.concatenate(EP)
    n_ep = len(starts)
    val_ids = set(range(n_ep - val_episodes, n_ep))
    is_val = np.isin(EP, list(val_ids))
    print(f"  dataset    : {S.shape[0]} timesteps x {Y.shape[1]} agents, "
          f"state_dim={S.shape[1]}")
    print(f"  split      : train {int((~is_val).sum())} / val {int(is_val.sum())} "
          f"timesteps (val = episodes {sorted(val_ids)})")
    print(f"  truncated episodes (hit step limit {ep_steps}): "
          f"{int(np.sum(trunc))} / {n_ep}"
          + ("  -> MC targets are EXACT" if not any(trunc)
             else "  -> WARNING: MC understates tail on those"))
    return dict(state=S, mc=Y, risk=R, ep=EP, is_val=is_val,
                truncated=[bool(t) for t in trunc])


# ==========================================================================
# phase B -- offline critic fitting
# ==========================================================================

def lambda_targets(critic, base, starts, agent):
    """
    Production's target construction: ret = adv + V, where adv is GAE(lambda)
    bootstrapped off the critic's OWN predictions. Recomputed from whatever
    critic is passed in, episode by episode, in the same order build_dataset
    concatenates them.
    """
    probe = copy.deepcopy(agent)
    probe.critic = critic
    out = []
    for s in starts:
        _, ret, _ = gae_single(probe, base[s]["rec"])
        out.append(np.asarray(ret, np.float32))
    return np.concatenate(out)


def fit_critic(arm, data, agent, cfg, base, starts, args, seed=0):
    """
    Fit a FRESH critic of the production architecture. No production module and
    no production optimiser is touched.
    """
    torch.manual_seed(seed)
    dev = torch.device(args.device)
    critic = CentralisedCritic(agent.state_dim, agent.n_agents,
                              agent.cfg.critic_hidden).to(dev)
    opt = torch.optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)

    S = torch.as_tensor(data["state"], device=dev)
    tr = ~data["is_val"]
    va = data["is_val"]
    idx_tr = np.flatnonzero(tr)
    Y_mc = torch.as_tensor(data["mc"], device=dev)
    clip_eps = float(agent.cfg.value_clip_eps)

    best = (np.inf, -1, None)
    rng = np.random.default_rng(seed)
    hist = []
    for epoch in range(args.max_epochs):
        # ---- targets -----------------------------------------------------
        if arm == "C2":
            # production's construction: recompute from the CURRENT critic
            with torch.no_grad():
                Y = torch.as_tensor(
                    lambda_targets(copy.deepcopy(critic), base, starts, agent),
                    device=dev)
        else:
            Y = Y_mc
        # ---- C3 snapshots old_v per epoch, then rate-limits movement -----
        if arm == "C3":
            with torch.no_grad():
                old_v = critic(S).detach()

        perm = rng.permutation(idx_tr)
        critic.train()
        for k in range(0, len(perm), args.batch):
            mb = torch.as_tensor(perm[k:k + args.batch], dtype=torch.long,
                                 device=dev)
            v = critic(S[mb])
            if arm == "C3":
                vc = old_v[mb] + torch.clamp(v - old_v[mb], -clip_eps, clip_eps)
                loss = torch.max((v - Y[mb]) ** 2, (vc - Y[mb]) ** 2).mean()
            else:
                loss = ((v - Y[mb]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), agent.cfg.max_grad_norm)
            opt.step()

        # ---- early stopping on held-out MC MSE (same yardstick for all) --
        critic.eval()
        with torch.no_grad():
            v_all = critic(S)
            mse_va = float(((v_all[va] - Y_mc[va]) ** 2).mean())
            mse_tr = float(((v_all[tr] - Y_mc[tr]) ** 2).mean())
        hist.append(dict(epoch=epoch, train_mse=mse_tr, val_mse=mse_va))
        if mse_va < best[0] - 1e-6:
            best = (mse_va, epoch,
                    copy.deepcopy(critic.state_dict()))
        elif epoch - best[1] >= args.patience:
            break

    critic.load_state_dict(best[2])
    critic.eval()
    print(f"    {arm}: best val MC-MSE {best[0]:.4f} at epoch {best[1]} "
          f"({len(hist)} epochs run)")
    return critic, dict(best_val_mc_mse=best[0], best_epoch=best[1],
                        epochs_run=len(hist), history_tail=hist[-5:])


# ==========================================================================
# phase C -- residual vs risk
# ==========================================================================

def _ols(x, y):
    x = np.asarray(x, np.float64); y = np.asarray(y, np.float64)
    if len(x) < 3 or x.std() < 1e-12:
        return dict(slope=None, intercept=None, pearson=None, n=int(len(x)))
    A = np.vstack([x, np.ones_like(x)]).T
    sl, ic = np.linalg.lstsq(A, y, rcond=None)[0]
    return dict(slope=float(sl), intercept=float(ic),
                pearson=float(np.corrcoef(x, y)[0, 1]), n=int(len(x)))


def _ev(y, p):
    y = np.asarray(y, np.float64); p = np.asarray(p, np.float64)
    vy = y.var()
    return float(1.0 - (y - p).var() / vy) if vy > 1e-12 else None


def residual_analysis(critics, data, agent):
    """
    residual = MC return - V(s,i), the honest (critic-free-target) measure of
    critic bias. Rung 0's residual was production's GAE, i.e. ret - V where
    ret = adv + V, which is self-referential; both are reported.
    """
    dev = torch.device(agent.device)
    S = torch.as_tensor(data["state"], device=dev)
    mc = data["mc"].astype(np.float64)
    risk = data["risk"].astype(np.float64)
    va = data["is_val"]
    out = {}
    for arm, critic in critics.items():
        with torch.no_grad():
            V = critic(S).cpu().numpy().astype(np.float64)
        res = mc - V
        rec = dict(what=ARM_WHAT[arm])
        rec["mse_train"] = float(((res[~va]) ** 2).mean())
        rec["mse_val"] = float(((res[va]) ** 2).mean())
        rec["mae_train"] = float(np.abs(res[~va]).mean())
        rec["mae_val"] = float(np.abs(res[va]).mean())
        rec["explained_var_train"] = _ev(mc[~va].ravel(), V[~va].ravel())
        rec["explained_var_val"] = _ev(mc[va].ravel(), V[va].ravel())
        rec["value_mean"] = float(V.mean())
        rec["mc_mean"] = float(mc.mean())
        rec["mc_sd"] = float(mc.std())
        # ---- stratified residual --------------------------------------
        strat = {}
        for b in BUCKETS:
            if b == "lo":
                m = risk < 0.10
            elif b == "hi":
                m = risk > 0.50
            else:
                m = (risk >= 0.10) & (risk <= 0.50)
            if m.sum() == 0:
                strat[b] = dict(n=0)
                continue
            strat[b] = dict(n=int(m.sum()),
                            residual_mean=float(res[m].mean()),
                            residual_sd=float(res[m].std()),
                            residual_mean_abs=float(np.abs(res[m]).mean()),
                            value_mean=float(V[m].mean()),
                            mc_mean=float(mc[m].mean()),
                            explained_var=_ev(mc[m], V[m]))
        rec["residual_by_risk"] = strat
        lo = strat["lo"].get("residual_mean")
        hi = strat["hi"].get("residual_mean")
        rec["residual_lo_to_hi_swing"] = (None if lo is None or hi is None
                                         else float(hi - lo))
        # ---- the headline: slope of residual on risk -------------------
        rec["residual_vs_risk_ols_all"] = _ols(risk.ravel(), res.ravel())
        hm = risk > 0.50
        rec["residual_vs_risk_ols_highrisk_only"] = _ols(risk[hm], res[hm])
        rec["residual_risk_spearman"] = float(_spearman(risk.ravel(), res.ravel()))
        out[arm] = rec
    return out


# ==========================================================================
# phase D -- deviation pairs, exactly Rung 0's set
# ==========================================================================

def load_d1_rows(path, max_pairs=0):
    d1 = json.loads(Path(path).read_text())
    rows = d1["rows"]
    if max_pairs:
        rows = rows[:max_pairs]
    return rows, d1


def deviation_pass(rows, critics, agent, cfg, base, seed_of, env):
    """
    One replay pass over Rung 0's exact state/action pairs. For each pair the
    GAE is recomputed under EVERY arm in the same pass, and the true advantage
    is measured under BOTH reward attributions.
    """
    g = agent.cfg.gamma
    probes = {}
    for arm, critic in critics.items():
        p = copy.deepcopy(agent)
        p.critic = critic
        probes[arm] = p

    # baseline GAE per arm, per start, cached once
    base_gae = {arm: {} for arm in critics}
    for s in base:
        for arm, p in probes.items():
            adv, _, _ = gae_single(p, base[s]["rec"])
            base_gae[arm][s] = adv

    out, mismatch, runs = [], 0, 0
    t0 = time.time()
    for n_row, row in enumerate(rows):
        s = int(row["start"]); step = int(row["step"]); i = int(row["agent"])
        a0 = int(row["ref_action"])
        legal = [int(a) for a in row["legal"] if int(a) in MEASURED]
        b = base[s]; brec = b["rec"]
        tb = np.asarray(b["team_r"][step:], np.float64)
        ab = np.asarray(brec["rew"][step:, i], np.float64)

        q_team, q_own, gae = {}, {}, {arm: {} for arm in critics}
        q_team[a0] = float(np.dot(g ** np.arange(len(tb)), tb))
        q_own[a0] = float(np.dot(g ** np.arange(len(ab)), ab))
        for arm in critics:
            gae[arm][a0] = float(base_gae[arm][s][step, i])

        bad = False
        for a in legal:
            if a == a0:
                continue
            d = _replay(env, agent, s, seed_of[s], record=True,
                        override=(step, i, a))
            runs += 1
            if (d["trail"][:step] != b["trail"][:step]
                    or not np.allclose(d["team_r"][:step], b["team_r"][:step],
                                       atol=1e-6)
                    or not np.allclose(d["rec"]["obs"][:step],
                                       brec["obs"][:step], atol=1e-6)):
                mismatch += 1
                bad = True
                break
            td = np.asarray(d["team_r"][step:], np.float64)
            ad = np.asarray(d["rec"]["rew"][step:, i], np.float64)
            q_team[a] = float(np.dot(g ** np.arange(len(td)), td))
            q_own[a] = float(np.dot(g ** np.arange(len(ad)), ad))
            for arm, p in probes.items():
                adv_d, _, _ = gae_single(p, d["rec"])
                gae[arm][a] = float(adv_d[step, i])
        if bad:
            continue

        out.append(dict(
            pool=row["pool"], bucket=row["bucket"], start=s, step=step,
            agent=i, risk=float(row["risk"]), ref_action=a0,
            legal=legal,
            a_true_team={int(k): v - q_team[a0] for k, v in q_team.items()},
            a_true_own={int(k): v - q_own[a0] for k, v in q_own.items()},
            q_team={int(k): v for k, v in q_team.items()},
            q_own={int(k): v for k, v in q_own.items()},
            gae={arm: {int(k): v for k, v in gae[arm].items()}
                 for arm in critics},
        ))
        if (n_row + 1) % 100 == 0:
            print(f"    {n_row + 1}/{len(rows)} states, {runs} replays, "
                  f"{time.time() - t0:.0f}s")
    print(f"  replay mismatches: {mismatch} / {runs}")
    return out, mismatch, runs


def _ikeys(d):
    """
    Normalise an action-keyed dict to int keys.

    In memory these dicts are built with int keys; after a json round-trip the
    same dicts come back with str keys.  The aggregators below are run both
    ways (live in phase D, and again by --phase reagg over the saved
    artifact), so every lookup goes through here.  Getting this wrong is
    silent: a str/int mismatch yields an empty selection, not an error.
    """
    return {int(k): v for k, v in d.items()}


def agreement(rows, arm, truth_key):
    """Sign agreement / rank correlation of an arm's GAE against a truth."""
    res = {}
    for b in list(BUCKETS) + ["all"]:
        R = rows if b == "all" else [r for r in rows if r["bucket"] == b]
        T, G = [], []
        for r in R:
            g = _ikeys(r["gae"][arm])
            for k, v in _ikeys(r[truth_key]).items():
                if k == int(r["ref_action"]):
                    continue
                T.append(v); G.append(g[k])
        if len(T) < 3:
            res[b] = dict(n_deviation_pairs=len(T))
            continue
        T = np.asarray(T, np.float64); G = np.asarray(G, np.float64)
        nz = (np.abs(T) > 1e-9)
        res[b] = dict(
            n_deviation_pairs=int(len(T)),
            sign_agreement=float(np.mean(np.sign(T[nz]) == np.sign(G[nz])))
            if nz.any() else None,
            n_sign_comparable=int(nz.sum()),
            pearson=float(np.corrcoef(T, G)[0, 1]),
            spearman=float(_spearman(T, G)),
            true_mean=float(T.mean()), gae_mean=float(G.mean()),
            true_frac_pos=float((T > 0).mean()),
            gae_frac_pos=float((G > 0).mean()),
        )
    return res


def per_action(rows, arm, truth_key, bucket):
    """EDGE / STAY / CLOUD estimates, and the resulting ordering."""
    R = [r for r in rows if r["bucket"] == bucket]
    out = {}
    for a in MEASURED:
        T, G = [], []
        for r in R:
            truth = _ikeys(r[truth_key])
            if a == int(r["ref_action"]) or a not in truth:
                continue
            T.append(truth[a]); G.append(_ikeys(r["gae"][arm])[a])
        if not T:
            out[ACTION_NAMES[a]] = dict(n=0)
            continue
        out[ACTION_NAMES[a]] = dict(n=len(T),
                                    true=_stats(T), gae=_stats(G))
    # ordering by mean GAE, over actions with data
    have = [(a, out[ACTION_NAMES[a]]) for a in MEASURED
            if out[ACTION_NAMES[a]].get("n")]
    order_gae = [ACTION_NAMES[a] for a, v in
                 sorted(have, key=lambda kv: -kv[1]["gae"]["mean"])]
    order_true = [ACTION_NAMES[a] for a, v in
                  sorted(have, key=lambda kv: -kv[1]["true"]["mean"])]
    return dict(actions=out, ordering_by_gae=order_gae,
                ordering_by_true=order_true)


def noise_floor(rows, arm):
    """
    Under a deterministic policy the true advantage of the action actually
    taken is identically 0, so any non-zero GAE there is measured estimator
    error. Same calibration constant Rung 0 used.
    """
    out = {}
    for b in list(BUCKETS) + ["all"]:
        R = rows if b == "all" else [r for r in rows if r["bucket"] == b]
        v = [_ikeys(r["gae"][arm])[int(r["ref_action"])] for r in R
             if int(r["ref_action"]) in _ikeys(r["gae"][arm])]
        out[b] = _stats(v) if v else dict(n=0)
    return out


# ==========================================================================
# phase E -- the minibatch tail issue, ISOLATED from the critic experiment
# ==========================================================================

def tail_minibatch_report(agent):
    """
    Characterise `mb_size = T // n_mb` + `range(0, T, mb_size)`. Measurement
    and documentation only -- NOT fixed, and deliberately kept in its own
    artifact so it cannot confound the critic diagnosis.
    """
    n_mb = max(1, int(agent.cfg.minibatches))
    rows = []
    for T in range(3000, 3300):
        mb = max(1, T // n_mb)
        starts = list(range(0, T, mb))
        sizes = [min(mb, T - s) for s in starts]
        rows.append(dict(T=T, mb_size=mb, n_chunks=len(starts),
                         tail_size=sizes[-1] if len(starts) > n_mb else None))
    extra = [r for r in rows if r["n_chunks"] > n_mb]
    tails = [r["tail_size"] for r in extra]
    d4 = OUT_DIR / "diag_S7_D4_ppo_update.json"
    observed = {}
    if d4.exists():
        j = json.loads(d4.read_text())
        for lbl, v in j.get("snapshots", {}).items():
            tr = v.get("trace", [])
            if not tr:
                continue
            sz = np.array([t["size"] for t in tr])
            gn = np.array([t["actor_grad_norm_preclip"] for t in tr])
            cl = np.array([bool(t["actor_grad_clipped"]) for t in tr])
            small = sz < 10
            observed[lbl] = dict(
                n_minibatch_steps=int(len(tr)),
                distinct_sizes=sorted(set(int(x) for x in sz)),
                n_degenerate=int(small.sum()),
                degenerate_sizes=[int(x) for x in sz[small]],
                degenerate_grad_norms=[float(x) for x in gn[small]],
                degenerate_clipped=int(cl[small].sum()),
                substantive_grad_norm_mean=float(gn[~small].mean()) if (~small).any() else None,
                substantive_grad_norm_max=float(gn[~small].max()) if (~small).any() else None,
                ratio_degenerate_to_substantive=(
                    float(gn[small].max() / gn[~small].mean())
                    if small.any() and (~small).any() and gn[~small].mean() > 0 else None),
            )
    return dict(
        what="mb_size = T // n_mb then range(0, T, mb_size) yields n_mb + 1 "
             "chunks whenever n_mb does not divide T; the extra tail chunk has "
             "T mod mb_size timesteps",
        status="MEASURED AND DOCUMENTED ONLY -- NOT FIXED, and isolated from "
               "the Rung 1 critic experiment so it cannot confound it",
        configured_minibatches=n_mb,
        ppo_epochs=int(agent.cfg.ppo_epochs),
        scanned_T_range=[3000, 3299],
        frac_T_producing_extra_chunk=float(len(extra) / len(rows)),
        tail_size_min=int(min(tails)) if tails else None,
        tail_size_max=int(max(tails)) if tails else None,
        tail_size_median=float(np.median(tails)) if tails else None,
        observed_in_rung0_D4=observed,
        note="A tail chunk of 1-2 timesteps yields an actor gradient ~40-46x "
             "the substantive minibatches' and is gradient-clipped, so it "
             "contributes a clipped near-single-sample step at 1 in every "
             "n_mb+1 actor updates.",
    )


# ==========================================================================
# main
# ==========================================================================

def _dump(obj, path):
    Path(path).write_text(json.dumps(obj, indent=1, default=float))
    print(f"  wrote {path}")


def build_summary(devrows, verbose=True):
    """Aggregate the deviation rows into the Rung 1 success-test statistics."""
    summary = {}
    for arm in ARMS:
        summary[arm] = dict(
            what=ARM_WHAT[arm],
            agreement_vs_team_truth=agreement(devrows, arm, "a_true_team"),
            agreement_vs_own_truth=agreement(devrows, arm, "a_true_own"),
            noise_floor=noise_floor(devrows, arm),
            per_action_highrisk_team=per_action(devrows, arm, "a_true_team", "hi"),
            per_action_highrisk_own=per_action(devrows, arm, "a_true_own", "hi"),
            per_action_lowrisk_team=per_action(devrows, arm, "a_true_team", "lo"),
        )
        if verbose:
            a = summary[arm]["agreement_vs_team_truth"]["hi"]
            o = summary[arm]["agreement_vs_own_truth"]["hi"]
            print(f"    {arm}: hi-risk sign agreement  team-truth "
                  f"{a.get('sign_agreement')}  own-truth {o.get('sign_agreement')}"
                  f"   gae_mean {a.get('gae_mean'):+.4f}")
    return summary


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 - RUNG 1: offline critic diagnostic (NO PPO TRAINING)")
    print("=" * 78)

    agent, extra, cfg = load_agent_and_cfg(args.model, args.device, args.window)
    print(f"  model      : {Path(args.model).name}")
    print(f"  gamma={agent.cfg.gamma} lambda={agent.cfg.gae_lambda} "
          f"critic_hidden={agent.cfg.critic_hidden} "
          f"lr_critic={agent.cfg.lr_critic} "
          f"value_clip_eps={agent.cfg.value_clip_eps}")

    if args.phase == "tail":
        _dump(tail_minibatch_report(agent),
              OUT_DIR / f"{TAG}_minibatch_tail_issue.json")
        return

    if args.phase == "reagg":
        # Re-aggregate the already-saved deviation rows.  The rows hold every
        # per-arm GAE and both truth attributions, so the success-test
        # statistics are a pure function of them -- no replay, no refit, and
        # therefore no risk of the numbers drifting from the recorded run.
        path = OUT_DIR / f"{TAG}_deviation_agreement.json"
        blob = json.loads(path.read_text())
        print(f"\n-- re-aggregating {len(blob['rows'])} rows from {path.name}")
        blob["summary"] = build_summary(blob["rows"])
        blob["reaggregated"] = (
            "summary recomputed from the saved rows after fixing an int/str "
            "action-key mismatch that made per_action and noise_floor select "
            "empty sets in the original phase-D run; agreement/* were "
            "unaffected and are reproduced identically"
        )
        _dump(blob, path)
        return

    print("\n-- phase A: baselines (identical starts/seeds to Rung 0) ------")
    env, starts, base, seed_of, replica = collect_baselines(
        agent, cfg, args.episodes)
    data = build_dataset(agent, base, starts, cfg, args.val_episodes)

    print("\n-- phase B: offline critic fits ------------------------------")
    critics = {"C0": agent.critic}
    fitinfo = {"C0": dict(note="production critic as loaded; not refitted")}
    for arm in ("C1", "C2", "C3"):
        c, info = fit_critic(arm, data, agent, cfg, base, starts, args)
        critics[arm], fitinfo[arm] = c, info

    print("\n-- phase C: residual vs risk ---------------------------------")
    resid = residual_analysis(critics, data, agent)
    for arm in ARMS:
        r = resid[arm]
        s = r["residual_vs_risk_ols_all"]
        st = r["residual_by_risk"]
        print(f"    {arm}: val EV {r['explained_var_val']:+.4f}  "
              f"val MSE {r['mse_val']:8.3f}  "
              f"resid lo {st['lo']['residual_mean']:+7.3f} "
              f"hi {st['hi']['residual_mean']:+7.3f}  "
              f"slope {s['slope']:+8.4f}")

    out_c = dict(
        probe="SPRINT_7_RUNG1", what="offline critic diagnostic",
        model=str(args.model), window=args.window,
        arms=ARM_WHAT,
        gamma=float(agent.cfg.gamma), gae_lambda=float(agent.cfg.gae_lambda),
        value_clip_eps=float(agent.cfg.value_clip_eps),
        critic_hidden=list(agent.cfg.critic_hidden),
        lr_used=float(args.lr),
        episodes=args.episodes, val_episodes=args.val_episodes,
        starts=[int(s) for s in starts],
        gae_replica_check=replica,
        truncated_episodes=data["truncated"],
        target_note="MC discounted per-agent return; exact and critic-free "
                    "when episodes end naturally",
        fit=fitinfo, residual=resid,
    )
    _dump(out_c, OUT_DIR / f"{TAG}_critic_fit_and_residual.json")

    if args.phase in ("data", "fit"):
        return

    print("\n-- phase D: deviation pairs (Rung 0's exact set) -------------")
    rows_d1, d1meta = load_d1_rows(D1_ARTIFACT, args.max_pairs)
    print(f"  reusing {len(rows_d1)} states from {D1_ARTIFACT.name}")
    devrows, mismatch, runs = deviation_pass(
        rows_d1, critics, agent, cfg, base, seed_of, env)

    summary = build_summary(devrows)

    _dump(dict(
        probe="SPRINT_7_RUNG1_deviations",
        what="Rung 0's exact 583 states / 745 pairs re-scored under every arm",
        reused_artifact=str(D1_ARTIFACT.name),
        n_states=len(devrows), replay_mismatches=mismatch, forced_replays=runs,
        attribution_note="a_true_team uses team_r (Rung 0's convention); "
                         "a_true_own uses agent i's OWN reward stream, which "
                         "is what per-agent GAE actually estimates",
        arms=ARM_WHAT, summary=summary, rows=devrows,
    ), OUT_DIR / f"{TAG}_deviation_agreement.json")

    _dump(tail_minibatch_report(agent),
          OUT_DIR / f"{TAG}_minibatch_tail_issue.json")
    print("\ndone. No PPO training was run; no production file was modified.")


if __name__ == "__main__":
    main()
