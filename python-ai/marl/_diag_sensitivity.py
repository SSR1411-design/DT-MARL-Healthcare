"""
Sprint 6.5 Phase 2 — WHERE DOES THE RISK SIGNAL DIE?

Diagnostic only. Reads a trained checkpoint and the recorded trace; writes
nothing except its own JSON report. It does not modify the environment, the
reward, the learner or any saved model.

The Sprint 6 result to explain: MAPPO scores 20.62 while a one-line rule
("relocate when predicted_failure_risk > 0.18") scores 76.50 under the SAME
reward, on the SAME episodes, with the SAME action set and the SAME masks. So
the objective demonstrably pays for the behaviour; the learner is not finding
it. These six probes localise the failure.

  P1  OBSERVATION SUPPORT     what the policy actually sees. Per-feature mean,
                              sd and range over real decision-time states. A
                              feature the policy never sees vary cannot be
                              used, and a probe grid wider than the training
                              support measures extrapolation, not behaviour.
  P2  LOGIT SENSITIVITY       d(logit_a)/d(obs_j) by autograd at real states,
                              multiplied by that feature's REAL sd. That
                              product is the logit swing the feature actually
                              produces in this environment, which is the only
                              scale on which "risk is underrepresented" means
                              anything. Ranked over all 48 features.
  P3  ACTION AUTHORITY        how often each action is even legal at a
                              decision, and how often the argmax is forced.
                              Tests cause E (masking / destination logic
                              removing the actor's decision authority).
  P4  DECISION OPPORTUNITY    how many (agent, step) pairs per episode are
                              high-risk decisions at all, and what each policy
                              does on exactly those. Tests cause F (the
                              distribution is too sparse to learn from) and
                              quantifies the exploration budget in cause G.
  P5  ADVANTAGE AUDIT         GAE advantages from a real on-policy rollout,
                              bucketed by (risk, action). If the advantage of
                              relocating at high risk is not positive in the
                              data the learner sees, the gradient never points
                              the right way regardless of exploration. Tests
                              causes C/D/H at the level where they act.
  P6  CAPACITY PROBE          behaviour-clone risk-threshold@0.18 into the
                              SAME actor architecture from the SAME
                              observations. If that fits, the representation
                              and the network can express the target policy and
                              the failure is credit assignment / exploration,
                              not representation. Tests causes A/B/J directly
                              and is the one probe that can exonerate them.

    python marl/_diag_sensitivity.py
    python marl/_diag_sensitivity.py --episodes 4 --model .../mappo_best.pth
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

from marl.config import (                                          # noqa: E402
    Sprint6Config, EnvConfig, RewardConfig, resolve_device, ACTION_NAMES,
    N_ACTIONS, ACTION_STAY, ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD,
    ACTION_PREEMPTIVE_REROUTE,
)
from marl.env import DTMarlEnv                                     # noqa: E402
from marl.mappo import MAPPO, MappoPolicy, RolloutBuffer, masked_dist  # noqa: E402
from marl.rollout import episode_starts                            # noqa: E402
from marl.baseline import RiskThresholdPolicy, NoMigrationPolicy   # noqa: E402
from marl.train import TRAIN_FRAC                                  # noqa: E402

IDX_LOCAL_RISK = 12
IDX_HAS_TASK = 15
IDX_SEVERITY = 16

# Names for all 48 observation slots, in _observations() order. Kept here
# rather than in env.py because this is the only consumer that needs them and
# adding a parallel list to the env invites the two drifting apart.
NB_OFFSETS = [-2, -1, 1, 2]
FEATURE_NAMES = (
    ["cpu", "ram", "bandwidth", "energy", "runningTasks", "active",
     "degraded", "linkUp", "linkBwMbps", "linkLatency", "linkPktLoss",
     "underAttack", "RISK", "uncertainty", "freeFrac"]
    + ["task_present", "SEVERITY", "priority", "progress", "timeToDeadline",
       "migrationsUsed", "isRunning", "residentFrac"]
    + [f"nb{o:+d}_{n}" for o in NB_OFFSETS
       for n in ("risk", "active", "freeFrac", "cpu", "lat")]
    + ["cloud_freeFrac", "cloud_lat", "cloud_risk0"]
    + ["episodeProgress", "agentId"]
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Sprint 6.5 sensitivity diagnosis")
    p.add_argument("--model", default=None)
    p.add_argument("--episodes", type=int, default=6)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None)
    p.add_argument("--bc-steps", type=int, default=600,
                   help="gradient steps for the P6 capacity probe")
    return p.parse_args(argv)


# ==========================================================================
# shared collection
# ==========================================================================

def collect_states(env, policy, starts, cap=6000):
    """
    Every decision-time (agent, obs, mask) triple a policy actually visits,
    plus the action it took and the reference action risk-threshold@0.18 would
    have taken from the identical state. Collected under ONE policy so the
    state distribution is the policy's own, which is the distribution its
    gradient is computed on.
    """
    ref = RiskThresholdPolicy(threshold=0.18)
    rows = []
    per_ep_highrisk, per_ep_decisions = [], []
    for j, s in enumerate(starts):
        obs, state, masks = env.reset(episode_start_tick=s, seed=j)
        done = False
        hr = dec = 0
        while not done:
            a = policy.act(env, obs, masks)
            a_ref = ref.act(env, obs, masks)
            for i in range(env.n_agents):
                if masks[i].sum() < 1.5 or obs[i, IDX_HAS_TASK] < 0.5:
                    continue
                dec += 1
                r = float(env.risk_at(i))
                if r > 0.18:
                    hr += 1
                if len(rows) < cap:
                    rows.append(dict(
                        agent=i, obs=obs[i].copy(), mask=masks[i].copy(),
                        act=int(a[i]), ref=int(a_ref[i]), risk=r,
                        severity=float(obs[i, IDX_SEVERITY])))
            obs, state, _, done, info = env.step(a)
            masks = info["action_masks"]
        per_ep_highrisk.append(hr)
        per_ep_decisions.append(dec)
    return rows, per_ep_decisions, per_ep_highrisk


# ==========================================================================
# P1 / P2
# ==========================================================================

def p1_support(rows):
    X = np.stack([r["obs"] for r in rows])
    return dict(mean=X.mean(0), sd=X.std(0), lo=X.min(0), hi=X.max(0), X=X)


def p2_logit_sensitivity(agent, rows, n_max=400):
    """
    Mean |d(logit_a - logit_STAY)/d(obs_j)| by autograd, per feature, over
    real decision states. Relative to STAY because only differences between
    logits affect the softmax; the absolute level is unidentifiable.
    """
    obs_dim = rows[0]["obs"].shape[0]
    grad = np.zeros((N_ACTIONS, obs_dim))
    cnt = np.zeros(N_ACTIONS)
    sub = rows[:n_max]
    for r in sub:
        o = torch.tensor(r["obs"][None, :], dtype=torch.float32,
                         requires_grad=True)
        lg = agent.actor.logits(r["agent"], o)[0]
        for a in range(N_ACTIONS):
            if a == ACTION_STAY:
                continue
            if agent.actor.net(r["agent"]) is None:      # pragma: no cover
                continue
            g, = torch.autograd.grad(lg[a] - lg[ACTION_STAY], o,
                                     retain_graph=True)
            grad[a] += np.abs(g[0].numpy())
            cnt[a] += 1
    for a in range(N_ACTIONS):
        if cnt[a]:
            grad[a] /= cnt[a]
    return grad, len(sub)


# ==========================================================================
# P3
# ==========================================================================

def p3_authority(rows):
    legal = np.zeros(N_ACTIONS)
    forced = 0
    n_legal_hist = np.zeros(N_ACTIONS + 1)
    for r in rows:
        m = r["mask"].astype(bool)
        legal += m
        k = int(m.sum())
        n_legal_hist[k] += 1
        if k <= 1:
            forced += 1
    n = max(len(rows), 1)
    return dict(legal_frac=(legal / n), forced_frac=forced / n,
                n_legal_hist=n_legal_hist / n, n=n)


# ==========================================================================
# P5
# ==========================================================================

def p5_advantages(env, agent, starts, cfg):
    """
    On-policy rollout with SAMPLING (training conditions), GAE computed exactly
    as mappo.update() does, then advantages bucketed by (risk bucket, action).
    Normalised the same way the update normalises them, because that is the
    number the policy gradient actually multiplies.
    """
    cap = len(starts) * cfg.env.episode_steps
    buf = RolloutBuffer(cap, env.n_agents, env.obs_dim, env.state_dim)
    risk_log = np.zeros((cap, env.n_agents), np.float32)
    for j, s in enumerate(starts):
        obs, state, masks = env.reset(episode_start_tick=s, seed=j)
        done = False
        while not done:
            a, lp = agent.act(obs, masks)
            v = agent.value(state)
            risk_log[buf.ptr] = [env.risk_at(i) for i in range(env.n_agents)]
            nobs, nstate, rew, done, info = env.step(a)
            buf.add(obs, state, a, lp, v, rew, masks, 0.0 if done else 1.0)
            if done:
                buf.set_bootstrap(buf.ptr - 1, agent.value(nstate))
            obs, state, masks = nobs, nstate, info["action_masks"]

    T = len(buf)
    adv, ret = agent.compute_gae(buf)
    dec = buf.decision[:T] > 0.5
    if dec.any():
        a_sel = adv[dec]
        adv_n = (adv - a_sel.mean()) / (a_sel.std() + 1e-8)
    else:                                                # pragma: no cover
        adv_n = adv

    edges = [0.0, 0.18, 0.5, 1.01]
    labels = ["risk<=0.18", "0.18-0.5", "risk>0.5"]
    table = np.full((len(labels), N_ACTIONS), np.nan)
    counts = np.zeros((len(labels), N_ACTIONS), np.int64)
    rk = risk_log[:T]
    ac = buf.act[:T]
    for b in range(len(labels)):
        sel_b = (rk > edges[b]) if b else (rk <= edges[1])
        if b:
            sel_b &= rk <= edges[b + 1]
        for a in range(N_ACTIONS):
            sel = dec & sel_b & (ac == a)
            counts[b, a] = int(sel.sum())
            if sel.any():
                table[b, a] = float(adv_n[sel].mean())
    return dict(table=table, counts=counts, labels=labels,
                adv_std_raw=float(adv[dec].std()) if dec.any() else float("nan"),
                decision_frac=float(dec.mean()), T=T)


# ==========================================================================
# P6
# ==========================================================================

def p6_capacity(rows, obs_dim, n_agents, hidden, steps, device="cpu", seed=0):
    """
    Behaviour-clone risk-threshold@0.18 into a FRESH copy of the same actor
    architecture, from the same observations, with the same masks. Pure
    supervised learning: no reward, no advantage, no exploration.

    This isolates one question. If the fresh actors reach high accuracy, then
    (obs_dim -> hidden -> 4) with these features CAN express "relocate when
    risk is high", the risk feature is legible enough to drive it, and the
    Sprint 6 failure is therefore NOT representation or architecture. If they
    cannot, it is.
    """
    from marl.mappo import MultiAgentActor
    torch.manual_seed(seed)
    net = MultiAgentActor(n_agents, obs_dim, hidden, N_ACTIONS,
                          separate=True).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    by_agent = {}
    for r in rows:
        by_agent.setdefault(r["agent"], []).append(r)

    packs = {}
    for i, rs in by_agent.items():
        packs[i] = (
            torch.tensor(np.stack([r["obs"] for r in rs]), dtype=torch.float32),
            torch.tensor(np.stack([r["mask"] for r in rs]), dtype=torch.float32),
            torch.tensor(np.array([r["ref"] for r in rs]), dtype=torch.long),
            np.array([r["risk"] for r in rs]),
        )

    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = 0.0
        for i, (o, m, y, _) in packs.items():
            d = masked_dist(net.logits(i, o), m)
            loss = loss - d.log_prob(y).mean()
        loss.backward()
        opt.step()

    # accuracy overall and specifically where it matters
    tot = corr = 0
    hr_tot = hr_corr = 0
    reloc_tot = reloc_corr = 0
    with torch.no_grad():
        for i, (o, m, y, rk) in packs.items():
            lg = net.logits(i, o).masked_fill(~m.bool(), -1e8)
            pred = lg.argmax(-1).numpy()
            yv = y.numpy()
            tot += len(yv); corr += int((pred == yv).sum())
            hr = rk > 0.18
            hr_tot += int(hr.sum()); hr_corr += int((pred[hr] == yv[hr]).sum())
            rl = yv != ACTION_STAY
            reloc_tot += int(rl.sum())
            reloc_corr += int((pred[rl] == yv[rl]).sum())

    # risk sweep on the CLONED policy — same probe as evaluate.py, so the two
    # numbers are directly comparable.
    grid = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    p_rel = []
    with torch.no_grad():
        for val in grid:
            acc = []
            for i, (o, m, _, _) in packs.items():
                o2 = o.clone()
                o2[:, IDX_LOCAL_RISK] = val
                d = masked_dist(net.logits(i, o2), m)
                acc.append((1.0 - d.probs[:, ACTION_STAY]).numpy())
            p_rel.append(float(np.concatenate(acc).mean()))
    return dict(acc=corr / max(tot, 1),
                acc_highrisk=hr_corr / max(hr_tot, 1),
                acc_relocate=reloc_corr / max(reloc_tot, 1),
                n=tot, n_highrisk=hr_tot, n_relocate=reloc_tot,
                sweep_grid=grid, sweep_p_relocate=p_rel,
                sweep_span=max(p_rel) - min(p_rel),
                final_loss=float(loss.item()))


# ==========================================================================
# main
# ==========================================================================

def main(argv=None):
    args = parse_args(argv)
    base = Sprint6Config()
    model_path = Path(args.model) if args.model else \
        Path(base.train.out_dir) / f"{base.train.tag}.pth"
    if not model_path.exists():
        print(f"no checkpoint at {model_path}")
        return 1

    device = resolve_device(args.device)
    agent, extra = MAPPO.load(model_path, device=device)
    agent.eval()
    saved = extra.get("config", {})
    cfg = Sprint6Config()
    if saved:
        cfg.env = EnvConfig(**saved["env"])
        cfg.reward = RewardConfig(**saved["reward"])
        cfg.mappo.__dict__.update(saved.get("mappo", {}))
    cfg.env.start_frac_lo, cfg.env.start_frac_hi = TRAIN_FRAC, 1.0

    env = DTMarlEnv(cfg.env, cfg.reward)
    starts = episode_starts(env, args.episodes)
    pol = MappoPolicy(agent, "mappo-greedy")

    print("=" * 100)
    print("SPRINT 6.5 PHASE 2 — WHERE DOES THE RISK SIGNAL DIE?")
    print("=" * 100)
    print(f"  checkpoint : {model_path.name}")
    print(f"  episodes   : {len(starts)} at {starts}  (held-out window)")
    print(f"  risk       : {env.risk.summary()}")
    print("=" * 100)

    rows, dec_per_ep, hr_per_ep = collect_states(env, pol, starts)
    print(f"\n  collected {len(rows)} decision states under the trained policy")

    # ---------------- P1 ----------------
    print("\n" + "=" * 100)
    print("P1  OBSERVATION SUPPORT — what the policy actually sees")
    print("=" * 100)
    sup = p1_support(rows)
    order = np.argsort(-sup["sd"])
    print(f"    {'feature':>18s} {'mean':>9s} {'sd':>9s} {'min':>9s} {'max':>9s}")
    print("    " + "-" * 58)
    for j in order[:12]:
        print(f"    {FEATURE_NAMES[j]:>18s} {sup['mean'][j]:9.4f} "
              f"{sup['sd'][j]:9.4f} {sup['lo'][j]:9.4f} {sup['hi'][j]:9.4f}")
    print("    ...")
    for j in (IDX_LOCAL_RISK, IDX_SEVERITY):
        rank = int(np.where(order == j)[0][0]) + 1
        print(f"    {FEATURE_NAMES[j]:>18s} {sup['mean'][j]:9.4f} "
              f"{sup['sd'][j]:9.4f} {sup['lo'][j]:9.4f} {sup['hi'][j]:9.4f}"
              f"   <-- sd rank {rank}/{len(FEATURE_NAMES)}")
    n_dead = int((sup["sd"] < 1e-6).sum())
    print(f"\n    {n_dead} of {len(FEATURE_NAMES)} features have sd < 1e-6 "
          f"(constant at every decision the policy visits)")
    print(f"    features held constant: "
          f"{[FEATURE_NAMES[j] for j in np.flatnonzero(sup['sd'] < 1e-6)]}")

    # ---------------- P2 ----------------
    print("\n" + "=" * 100)
    print("P2  LOGIT SENSITIVITY — d(logit - logit_STAY)/d(feature), "
          "x the feature's REAL sd")
    print("=" * 100)
    grad, n_grad = p2_logit_sensitivity(agent, rows)
    print(f"    over {n_grad} real decision states, "
          f"averaged over the 10 independent actors that visited them")
    eff = grad * sup["sd"][None, :]        # logit swing per 1 sd of real variation
    for a in (ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD):
        e = eff[a]
        rank = np.argsort(-e)
        print(f"\n    {ACTION_NAMES[a]}  (top 8 drivers by |dlogit| x sd)")
        for j in rank[:8]:
            print(f"      {FEATURE_NAMES[j]:>18s}  |grad|={grad[a, j]:8.4f}  "
                  f"sd={sup['sd'][j]:6.4f}  swing={e[j]:8.5f}")
        r_rank = int(np.where(rank == IDX_LOCAL_RISK)[0][0]) + 1
        s_rank = int(np.where(rank == IDX_SEVERITY)[0][0]) + 1
        print(f"      {'RISK':>18s}  |grad|={grad[a, IDX_LOCAL_RISK]:8.4f}  "
              f"sd={sup['sd'][IDX_LOCAL_RISK]:6.4f}  "
              f"swing={e[IDX_LOCAL_RISK]:8.5f}   <-- rank {r_rank}/48, "
              f"{e[IDX_LOCAL_RISK] / max(e[rank[0]], 1e-12):.3%} of the top driver")
        print(f"      {'SEVERITY':>18s}  |grad|={grad[a, IDX_SEVERITY]:8.4f}  "
              f"sd={sup['sd'][IDX_SEVERITY]:6.4f}  "
              f"swing={e[IDX_SEVERITY]:8.5f}   <-- rank {s_rank}/48")
    print("\n    total swing available to move the STAY/MIGRATE margin:")
    for a in (ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD):
        print(f"      {ACTION_NAMES[a]:>26s}  sum |dlogit| x sd = "
              f"{eff[a].sum():.4f}  (risk contributes "
              f"{eff[a, IDX_LOCAL_RISK] / max(eff[a].sum(), 1e-12):.2%})")

    # ---------------- P3 ----------------
    print("\n" + "=" * 100)
    print("P3  ACTION AUTHORITY — is the actor allowed to decide?")
    print("=" * 100)
    au = p3_authority(rows)
    print(f"    over {au['n']} decision states (agent holds a task, "
          f">= 2 legal actions)")
    for a in range(N_ACTIONS):
        print(f"      {ACTION_NAMES[a]:>26s} legal in {au['legal_frac'][a]:6.1%} "
              f"of them")
    print(f"      {'#legal actions = 2':>26s} {au['n_legal_hist'][2]:6.1%}")
    print(f"      {'#legal actions = 3':>26s} {au['n_legal_hist'][3]:6.1%}")
    print(f"      {'#legal actions = 4':>26s} {au['n_legal_hist'][4]:6.1%}")

    # ---------------- P4 ----------------
    print("\n" + "=" * 100)
    print("P4  DECISION OPPORTUNITY — how often is there a high-risk call to make?")
    print("=" * 100)
    hr = [r for r in rows if r["risk"] > 0.18]
    print(f"    decisions per episode              : "
          f"{np.mean(dec_per_ep):8.1f}")
    print(f"    of which risk > 0.18               : "
          f"{np.mean(hr_per_ep):8.1f}  "
          f"({np.mean(hr_per_ep) / max(np.mean(dec_per_ep), 1e-9):.2%})")
    print(f"    agent-steps per episode            : "
          f"{cfg.env.episode_steps * env.n_agents:8d}")
    print(f"    high-risk decisions / agent-step    : "
          f"{np.mean(hr_per_ep) / (cfg.env.episode_steps * env.n_agents):8.5f}")
    if hr:
        mine = np.array([r["act"] for r in hr])
        theirs = np.array([r["ref"] for r in hr])
        print(f"\n    on those {len(hr)} high-risk decision states:")
        print(f"      {'action':>26s} {'MAPPO':>8s} {'risk-thr@0.18':>14s}")
        for a in range(N_ACTIONS):
            print(f"      {ACTION_NAMES[a]:>26s} {int((mine == a).sum()):8d} "
                  f"{int((theirs == a).sum()):14d}")
        print(f"      {'P(relocate)':>26s} "
              f"{float((mine != ACTION_STAY).mean()):8.3f} "
              f"{float((theirs != ACTION_STAY).mean()):14.3f}")
        agree = float((mine == theirs).mean())
        print(f"      agreement with the reference rule: {agree:.3f}")
    lo = [r for r in rows if r["risk"] <= 0.18]
    if lo:
        mine = np.array([r["act"] for r in lo])
        theirs = np.array([r["ref"] for r in lo])
        print(f"\n    on the {len(lo)} low-risk decision states, P(relocate): "
              f"MAPPO {float((mine != ACTION_STAY).mean()):.3f}  "
              f"risk-thr {float((theirs != ACTION_STAY).mean()):.3f}")

    # ---------------- P5 ----------------
    print("\n" + "=" * 100)
    print("P5  ADVANTAGE AUDIT — does the learning signal point at migration "
          "when risk is high?")
    print("=" * 100)
    adv = p5_advantages(env, agent, starts[:max(2, len(starts) // 2)], cfg)
    print(f"    on-policy rollout, {adv['T']} timesteps, "
          f"decision_frac={adv['decision_frac']:.4f}, "
          f"raw adv sd={adv['adv_std_raw']:.4f}")
    print(f"    normalised advantage, mean by (risk bucket, action taken); "
          f"n in parentheses\n")
    print(f"      {'bucket':>12s}  " + "  ".join(
        f"{ACTION_NAMES[a][:12]:>18s}" for a in range(N_ACTIONS)))
    for b, lab in enumerate(adv["labels"]):
        cells = []
        for a in range(N_ACTIONS):
            v, n = adv["table"][b, a], adv["counts"][b, a]
            cells.append("        --  (0)" if n == 0
                         else f"{v:+9.3f} ({n:5d})")
        print(f"      {lab:>12s}  " + "  ".join(f"{c:>18s}" for c in cells))
    print("\n    read: a POSITIVE cell means taking that action in that bucket")
    print("    came out better than the critic expected, so PPO pushes toward "
          "it.")

    # ---------------- P6 ----------------
    print("\n" + "=" * 100)
    print("P6  CAPACITY PROBE — can this architecture express the rule at all?")
    print("=" * 100)
    bc = p6_capacity(rows, env.obs_dim, env.n_agents, cfg.mappo.actor_hidden,
                     args.bc_steps, device=device, seed=cfg.train.seed)
    print(f"    behaviour-cloned risk-threshold@0.18 into a FRESH "
          f"{env.n_agents} x MLP({env.obs_dim} -> "
          f"{cfg.mappo.actor_hidden} -> {N_ACTIONS})")
    print(f"    {args.bc_steps} full-batch Adam steps, no reward, no RL\n")
    print(f"      overall action accuracy            : {bc['acc']:.4f}  "
          f"(n={bc['n']})")
    print(f"      accuracy where risk > 0.18         : "
          f"{bc['acc_highrisk']:.4f}  (n={bc['n_highrisk']})")
    print(f"      accuracy where the rule relocates  : "
          f"{bc['acc_relocate']:.4f}  (n={bc['n_relocate']})")
    print(f"      final NLL                          : {bc['final_loss']:.4f}")
    print(f"\n      risk sweep on the CLONE (same probe as evaluate.py):")
    for g, p in zip(bc["sweep_grid"], bc["sweep_p_relocate"]):
        print(f"        risk={g:4.2f}  P(relocate)={p:.4f}")
    print(f"      span = {bc['sweep_span']:.4f}")

    # ---------------- write ----------------
    out = Path(args.out) if args.out else \
        Path(cfg.train.out_dir) / "diag_sensitivity.json"
    payload = dict(
        checkpoint=str(model_path), episodes=starts,
        n_decision_states=len(rows),
        p1_feature_names=FEATURE_NAMES,
        p1_mean=sup["mean"].tolist(), p1_sd=sup["sd"].tolist(),
        p1_lo=sup["lo"].tolist(), p1_hi=sup["hi"].tolist(),
        p2_grad_abs=grad.tolist(), p2_swing=eff.tolist(),
        p3_legal_frac=au["legal_frac"].tolist(),
        p3_n_legal_hist=au["n_legal_hist"].tolist(),
        p4_decisions_per_ep=float(np.mean(dec_per_ep)),
        p4_highrisk_per_ep=float(np.mean(hr_per_ep)),
        p4_n_highrisk_states=len(hr),
        p5_labels=adv["labels"], p5_adv=adv["table"].tolist(),
        p5_counts=adv["counts"].tolist(),
        p5_decision_frac=adv["decision_frac"],
        p6=bc,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\n  written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
