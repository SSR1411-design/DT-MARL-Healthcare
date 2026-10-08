#!/usr/bin/env python
"""
SPRINT 7 RUNG 2.5 -- item E: offline advantage / sign test.

QUESTION: would replacing R2's truncation bootstrap with a genuine
continuation return be expected to move the advantage in the correct
direction at high risk?

This is an OFFLINE CAUSAL DIAGNOSTIC. It is NOT an online result. No PPO
training is run, no actor gradient is ever taken, no production module or
checkpoint is modified, and every fitted critic is a FRESH module.

WHY THE EXPERIMENT IS SHAPED THIS WAY -- the central Rung 2.5 finding.

  Rung 1 declared its MC arm (C1) critic-free, and its own artifact records
  truncated_episodes = [False] x 16. That is true, and it is exactly the
  problem: Rung 1 built its dataset from GREEDY replays (_diag_rung0._replay
  defaults to sample=False), and greedy episodes drive every task terminal
  before step 400. So Rung 1 validated a target in a regime containing ZERO
  truncations.

  R2's ONLINE training sampled actions (train.py calls agent.act), which keeps
  tasks alive to the horizon: 377/600 episodes truncated. On those, the "MC"
  target bootstrapped V(s_T).

  Therefore R2 did not implement the target Rung 1 validated, and the
  difference is invisible on Rung 1's own dataset. On Rung 0's exact deviation
  pairs the three targets below are IDENTICAL by construction, so evaluating
  the target choice there is a structural null -- reported as such rather than
  dressed up as a measurement.

  The resolution used here: the arms differ in the CRITIC-FIT dataset, which
  is drawn from stochastic, truncation-bearing rollouts matching what R2
  actually trained on. The EVALUATION set stays Rung 0's exact 583 states /
  745 forced replays, so sign agreement remains directly comparable to Rung 0
  and Rung 1. Deviation replays remain GREEDY and therefore exact
  counterfactuals (A(s, a_greedy) == 0 identically, preserving Rung 0's
  noise-floor identity).

ARMS
  C0      the R2 critic exactly as loaded from the checkpoint   [reference]
  M_R2    fresh critic, R2's ACTUAL target: reward-to-end + gamma^(T-t)*V(s_T)
          on truncated episodes                          [what R2 really did]
  M_cont  fresh critic, continuation-MC: reward accumulated through GENUINE
          terminal completion past the 400-step boundary   [critic-free ideal]
  M_rew   fresh critic, reward-only to the truncation boundary, no bootstrap
                                          [Rung 1's C1 as literally written]

All three fresh arms are fit with plain MSE and identical optimiser, schedule
and seed. They are early-stopped on a COMMON held-out yardstick -- the
continuation-MC target -- so no arm is scored on its own target.

Writes SPRINT_7_RUNG2_5_signtest.json.
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
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from marl.mappo import CentralisedCritic                        # noqa: E402
from marl._diag_rung0 import (                                  # noqa: E402
    load_agent_and_cfg, _replay, gae_single, OUT_DIR, BUCKETS,
)
from marl._diag_rung1_critic import (                           # noqa: E402
    collect_baselines, deviation_pass, agreement, per_action, noise_floor,
    residual_analysis, load_d1_rows, _ols, _ev, _dump, D1_ARTIFACT,
)
from marl._diag_rung2_5_targets import (                        # noqa: E402
    build_env, training_start_ticks, replay_with_continuation,
    discounted_tail, dist, TRAIN_SEED,
)

TAG = "SPRINT_7_RUNG2_5"
R2_MODEL = OUT_DIR / "mappo_R2_mc_target.pth"
ARMS = ("C0", "M_R2", "M_cont", "M_rew")
ARM_WHAT = {
    "C0": "the R2 critic exactly as loaded from the checkpoint (reference)",
    "M_R2": "fresh critic, R2's ACTUAL target: reward-to-end + "
            "gamma^(T-t)*V(s_T) bootstrap on truncated episodes",
    "M_cont": "fresh critic, continuation-MC: reward accumulated through "
              "GENUINE terminal completion past the 400-step boundary "
              "(critic-free)",
    "M_rew": "fresh critic, reward-only to the truncation boundary, no "
             "bootstrap (Rung 1's C1 as literally written)",
}


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(R2_MODEL))
    p.add_argument("--device", default="cpu")
    p.add_argument("--window", default="train", choices=("train", "eval"))
    p.add_argument("--fit-episodes", type=int, default=40,
                   help="stochastic truncation-bearing episodes for the fit")
    p.add_argument("--val-episodes", type=int, default=8)
    p.add_argument("--max-epochs", type=int, default=600)
    p.add_argument("--patience", type=int, default=40)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-pairs", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--phase", default="all", choices=("data", "fit", "all"))
    return p.parse_args(argv)


# ======================================================================
# the fit dataset: stochastic rollouts that actually truncate
# ======================================================================

def build_fit_dataset(agent, cfg, n_eps, val_eps):
    """
    Stochastic rollouts over the training start window, with continuation past
    the horizon for truncated episodes. Produces the three targets.

    Episodes whose continuation is censored by the end of the trace are
    EXCLUDED from the dataset rather than approximated: a censored
    continuation-MC target would silently understate the tail, which is the
    exact failure mode this rung exists to expose.
    """
    env = build_env(cfg)
    starts = training_start_ticks(env, n_eps * 2)   # oversample; some drop out
    gamma = agent.cfg.gamma
    S, Y_r2, Y_cont, Y_rew, R, EP = [], [], [], [], [], []
    kept, dropped, cls_count = 0, 0, {}
    for k, s in enumerate(starts):
        if kept >= n_eps:
            break
        e = replay_with_continuation(env, agent, s, sample=True,
                                     cont_policy="greedy",
                                     torch_seed=TRAIN_SEED + k)
        cls_count[e["cls"]] = cls_count.get(e["cls"], 0) + 1
        if e["cls"] == "trunc" and e["cont_ended"] != "terminal":
            dropped += 1
            continue
        T = e["n_in"]
        V_sT = np.asarray(agent.value(e["boot_state"]), np.float64).ravel()
        g_rew = discounted_tail(e["rew"], gamma, None)
        if e["trunc_flag_trainpy"]:
            g_r2 = discounted_tail(e["rew"], gamma, V_sT)
        else:
            g_r2 = g_rew.copy()
        if e["cont_steps"] > 0:
            full = np.concatenate([e["rew"], e["cont_rew"]], axis=0)
            g_cont = discounted_tail(full, gamma, None)[:T]
        else:
            g_cont = g_rew.copy()     # nothing remained: MC is already exact
        S.append(np.asarray(e["state"], np.float32))
        Y_r2.append(g_r2.astype(np.float32))
        Y_cont.append(g_cont.astype(np.float32))
        Y_rew.append(g_rew.astype(np.float32))
        R.append(np.asarray(e["risk"], np.float32))
        EP.append(np.full(T, kept, np.int32))
        kept += 1

    S = np.concatenate(S); R = np.concatenate(R); EP = np.concatenate(EP)
    Y = dict(M_R2=np.concatenate(Y_r2), M_cont=np.concatenate(Y_cont),
             M_rew=np.concatenate(Y_rew))
    val_ids = set(range(kept - val_eps, kept))
    is_val = np.isin(EP, list(val_ids))
    n_trunc = cls_count.get("trunc", 0) + cls_count.get("both", 0)
    print(f"  fit dataset : {S.shape[0]} timesteps x {Y['M_cont'].shape[1]} "
          f"agents, state_dim={S.shape[1]}")
    print(f"  episodes    : kept {kept}, dropped {dropped} "
          f"(continuation censored by trace end)")
    print(f"  end classes : {cls_count}  -> bootstrapped {n_trunc}")
    print(f"  split       : train {int((~is_val).sum())} / "
          f"val {int(is_val.sum())} timesteps")
    for a in ("M_R2", "M_cont", "M_rew"):
        print(f"    target {a:7s}: mean {Y[a].mean():+.4f} sd {Y[a].std():.4f} "
              f"mean|.| {np.abs(Y[a]).mean():.4f}")
    d_r2 = np.abs(Y["M_R2"] - Y["M_cont"])
    d_rw = np.abs(Y["M_rew"] - Y["M_cont"])
    print(f"  MAD vs continuation-MC: R2 target {d_r2.mean():.4f}   "
          f"reward-only {d_rw.mean():.4f}")
    return dict(state=S, targets=Y, risk=R, ep=EP, is_val=is_val,
                n_episodes=kept, dropped=dropped, cls_count=cls_count,
                mad_R2_vs_cont=float(d_r2.mean()),
                mad_rew_vs_cont=float(d_rw.mean()))


def fit_critic_split(arm, data, agent, args, seed=0):
    """
    Fit a fresh critic on arm's OWN target, early-stopped on the COMMON
    continuation-MC yardstick so no arm is scored on its own target.
    """
    torch.manual_seed(seed)
    dev = torch.device(args.device)
    critic = CentralisedCritic(agent.state_dim, agent.n_agents,
                               agent.cfg.critic_hidden).to(dev)
    opt = torch.optim.Adam(critic.parameters(), lr=args.lr, eps=1e-5)
    S = torch.as_tensor(data["state"], device=dev)
    Y = torch.as_tensor(data["targets"][arm], device=dev)
    Y_ref = torch.as_tensor(data["targets"]["M_cont"], device=dev)
    tr, va = ~data["is_val"], data["is_val"]
    idx_tr = np.flatnonzero(tr)
    best = (np.inf, -1, None)
    rng = np.random.default_rng(seed)
    hist = []
    for epoch in range(args.max_epochs):
        perm = rng.permutation(idx_tr)
        critic.train()
        for k in range(0, len(perm), args.batch):
            mb = torch.as_tensor(perm[k:k + args.batch], dtype=torch.long,
                                 device=dev)
            loss = ((critic(S[mb]) - Y[mb]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(),
                                     agent.cfg.max_grad_norm)
            opt.step()
        critic.eval()
        with torch.no_grad():
            v_all = critic(S)
            mse_va = float(((v_all[va] - Y_ref[va]) ** 2).mean())
            mse_tr = float(((v_all[tr] - Y_ref[tr]) ** 2).mean())
        hist.append(dict(epoch=epoch, train_mse=mse_tr, val_mse=mse_va))
        if mse_va < best[0] - 1e-6:
            best = (mse_va, epoch, copy.deepcopy(critic.state_dict()))
        elif epoch - best[1] >= args.patience:
            break
    critic.load_state_dict(best[2])
    critic.eval()
    print(f"    {arm}: best val continuation-MC MSE {best[0]:.4f} at epoch "
          f"{best[1]} ({len(hist)} epochs)")
    return critic, dict(best_val_cont_mse=best[0], best_epoch=best[1],
                        epochs_run=len(hist), history_tail=hist[-5:])


def residuals_vs_risk(critics, data, agent):
    """residual = continuation-MC return - V(s,i), stratified by risk."""
    dev = torch.device(agent.device)
    S = torch.as_tensor(data["state"], device=dev)
    y = data["targets"]["M_cont"].astype(np.float64)
    risk = data["risk"].astype(np.float64)
    va = data["is_val"]
    out = {}
    for arm, critic in critics.items():
        with torch.no_grad():
            V = critic(S).cpu().numpy().astype(np.float64)
        res = y - V
        rec = dict(what=ARM_WHAT[arm],
                   mse_train=float((res[~va] ** 2).mean()),
                   mse_val=float((res[va] ** 2).mean()),
                   explained_var_train=_ev(y[~va].ravel(), V[~va].ravel()),
                   explained_var_val=_ev(y[va].ravel(), V[va].ravel()),
                   value_mean=float(V.mean()))
        strat = {}
        for b in BUCKETS:
            m = (risk < 0.10) if b == "lo" else \
                (risk > 0.50) if b == "hi" else \
                ((risk >= 0.10) & (risk <= 0.50))
            if m.sum() == 0:
                strat[b] = dict(n=0); continue
            strat[b] = dict(n=int(m.sum()),
                            residual_mean=float(res[m].mean()),
                            residual_sd=float(res[m].std()),
                            residual_mean_abs=float(np.abs(res[m]).mean()),
                            value_mean=float(V[m].mean()),
                            target_mean=float(y[m].mean()),
                            explained_var=_ev(y[m], V[m]))
        rec["residual_by_risk"] = strat
        lo, hi = strat["lo"].get("residual_mean"), strat["hi"].get("residual_mean")
        rec["residual_lo_to_hi_swing"] = (None if lo is None or hi is None
                                          else float(hi - lo))
        rec["residual_vs_risk_ols_all"] = _ols(risk.ravel(), res.ravel())
        hm = risk > 0.50
        rec["residual_vs_risk_ols_highrisk_only"] = _ols(risk[hm], res[hm])
        out[arm] = rec
    return out


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 RUNG 2.5 -- item E: offline advantage / sign test")
    print("       (NO PPO TRAINING; offline causal diagnostic only)")
    print("=" * 78)
    agent, extra, cfg = load_agent_and_cfg(args.model, args.device, args.window)
    print(f"  model        : {Path(args.model).name}")
    print(f"  critic_target: {agent.cfg.critic_target}  gamma={agent.cfg.gamma} "
          f"lambda={agent.cfg.gae_lambda}")

    print("\n-- phase A: fit dataset (stochastic, truncation-bearing) ------")
    data = build_fit_dataset(agent, cfg, args.fit_episodes, args.val_episodes)
    if args.phase == "data":
        _dump(dict(probe=f"{TAG}_signtest_data", dataset={
            k: v for k, v in data.items() if k not in
            ("state", "targets", "risk", "ep", "is_val")}),
            OUT_DIR / f"{TAG}_signtest_data.json")
        return 0

    print("\n-- phase B: fit fresh critics (common held-out yardstick) -----")
    critics, fitinfo = {}, {}
    for arm in ("M_R2", "M_cont", "M_rew"):
        c, info = fit_critic_split(arm, data, agent, args, seed=args.seed)
        critics[arm] = c
        fitinfo[arm] = info
    critics["C0"] = agent.critic          # reference: R2 as loaded
    fitinfo["C0"] = dict(note="not fitted; the R2 checkpoint's own critic")

    print("\n-- phase C: residual vs risk (target = continuation-MC) -------")
    resid = residuals_vs_risk(critics, data, agent)
    for arm in ARMS:
        r = resid[arm]
        hi = r["residual_by_risk"]["hi"]
        print(f"    {arm:7s}: EV_val {str(r['explained_var_val'])[:7]:>7s}  "
              f"hi-risk residual {hi.get('residual_mean', float('nan')):+.4f}  "
              f"lo->hi swing {r['residual_lo_to_hi_swing']:+.4f}  "
              f"hi-only slope "
              f"{str(r['residual_vs_risk_ols_highrisk_only']['slope'])[:8]}")

    print("\n-- phase D: deviation pairs (Rung 0's EXACT set) --------------")
    rows, d1 = load_d1_rows(D1_ARTIFACT, args.max_pairs)
    print(f"  loaded {len(rows)} states from {D1_ARTIFACT.name}")
    env, starts, base, seed_of, replica = collect_baselines(
        agent, cfg, int(d1["episodes"]))
    # Prove the structural null: do ANY of these greedy baselines truncate?
    lens = {int(s): int(base[s]["n_steps"]) for s in starts}
    n_tr = sum(1 for v in lens.values() if v >= int(cfg.env.episode_steps))
    print(f"  greedy baseline episode lengths: {sorted(lens.values())}")
    print(f"  of these, truncated (>= {cfg.env.episode_steps} steps): {n_tr}")
    print("  => on Rung 0's own pairs the three targets coincide exactly;")
    print("     the arms differ ONLY through their fitted critics.")

    devrows, mismatch, runs = deviation_pass(rows, critics, agent, cfg,
                                             base, seed_of, env)
    print(f"  usable states {len(devrows)}, forced replays {runs}, "
          f"mismatches {mismatch}")

    summary = {}
    for truth in ("a_true_team", "a_true_own"):
        summary[truth] = {}
        for arm in ARMS:
            summary[truth][arm] = dict(
                agreement=agreement(devrows, arm, truth),
                per_action_hi=per_action(devrows, arm, truth, "hi"),
                per_action_lo=per_action(devrows, arm, truth, "lo"),
            )
    nf = {arm: noise_floor(devrows, arm) for arm in ARMS}

    print("\n-- sign agreement, high-risk bucket --------------------------")
    for truth in ("a_true_team", "a_true_own"):
        print(f"  [{truth}]")
        for arm in ARMS:
            a = summary[truth][arm]["agreement"]
            hi = a.get("hi", {}); al = a.get("all", {})
            print(f"    {arm:7s} hi {str(hi.get('sign_agreement'))[:6]:>6s} "
                  f"(n={hi.get('n_sign_comparable')})   "
                  f"all {str(al.get('sign_agreement'))[:6]:>6s} "
                  f"(n={al.get('n_sign_comparable')})   "
                  f"spearman_hi {str(hi.get('spearman'))[:7]}")

    blob = dict(
        probe=f"{TAG}_signtest",
        what="offline advantage/sign test: does a critic-free continuation-MC "
             "target move the high-risk advantage sign in the correct "
             "direction? OFFLINE CAUSAL DIAGNOSTIC, NOT an online result.",
        model=str(args.model), arms=ARM_WHAT,
        structural_null=dict(
            finding="Rung 1's dataset and Rung 0's deviation pairs contain "
                    "ZERO truncated episodes, so M_R2, M_cont and M_rew are "
                    "identical targets on them; the target choice can only be "
                    "probed through a truncation-bearing fit dataset",
            rung1_truncated_episodes_all_false=True,
            greedy_baseline_lengths=lens,
            greedy_baselines_truncated=int(n_tr),
            episode_steps=int(cfg.env.episode_steps),
        ),
        fit_dataset={k: v for k, v in data.items()
                     if k not in ("state", "targets", "risk", "ep", "is_val")},
        fit=fitinfo, residual=resid,
        deviation=dict(n_states=len(devrows), forced_replays=runs,
                       replay_mismatches=mismatch,
                       gae_replica_check=replica),
        summary=summary, noise_floor=nf,
        attribution_note="team = team_r.sum() (Rung 0 convention); "
                         "own = rew[step:, i]. Both reported.",
        rows=devrows,
    )
    _dump(blob, OUT_DIR / f"{TAG}_signtest.json")
    print(f"\n  wrote {OUT_DIR / (TAG + '_signtest.json')}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
