#!/usr/bin/env python
"""
SPRINT 7 DIVERGENCE DIAGNOSTIC — part 6: IS R3's HIGH-RISK CRITIC DEFICIT A
PROPERTY OF THE CRITIC, OR OF THE BUFFER IT WAS MEASURED ON?

OFFLINE. Loads checkpoints, builds on-policy buffers with the existing
`_diag_rung2_75_coherence.build`, evaluates critics under `torch.no_grad`.
Trains nothing, writes no parameters, touches no production module.

WHY THIS PROBE EXISTS. `SPRINT_7_DIV_content_*` section C2 found that R3 is the
only arm whose HIGH-RISK explained variance is worse than its LOW-RISK:

    explained variance      all       hi(>=0.6)   lo(<0.2)
    R2      (8 episodes)   0.7808      0.8155      0.7814     hi > lo
    R2_b32  (32 episodes)  0.7497      0.7924      0.7467     hi > lo
    R3_best (32 episodes)  0.7214      0.7175      0.7211     ~equal
    R3      (32 episodes)  0.7056      0.6083      0.7113     hi < lo

But each arm was scored on ITS OWN rollout, so critic quality and state
distribution are confounded: R3's critic may be fine and merely measured on
harder states. The R2_b32 cell already shows the inversion is not created by the
buffer SIZE. It does not show that it is not created by the buffer's CONTENT.

THE IDENTIFYING DESIGN. Score BOTH critics against the SAME regression target on
the SAME buffer, for both buffers — a 2x2:

                        buffer R2@32          buffer R3@32
    critic R2         (diagonal, = C2)      off-distribution
    critic R3          off-distribution    (diagonal, = C2)

Within a column the target, the states, the risk labels and the entry counts are
byte-identical, so an explained-variance difference down a column is a critic
difference and nothing else. Reading BOTH columns keeps the off-distribution
penalty symmetric: each critic is scored once at home and once away.

WHY THE TARGET IS SHARED. `compute_mc_returns` (mappo.py:310-338) is a function of
`buf.rew`, `buf.cont` and `buf.trunc` only, plus `buf.boot` at genuine time-limit
truncations — its docstring says so explicitly and that is the reason the "mc"
target exists. So for a FIXED buffer the target is one array, computed once, and
both critics regress against it. The one residual critic dependence is `boot` at
truncated episodes; it enters through the COLLECTING agent's critic, is therefore
identical for both critics in a column, and the truncation count is reported so
its weight can be judged.

BUFFER SIZE IS HELD AT 32 FOR BOTH so that the estimator's sampling variance
(quantified in `SPRINT_7_DIV_variance_*`) is matched across cells and cannot be
mistaken for a critic effect.

EXPLORATORY. Designed after seeing part 4's C2 numbers. It sets no threshold and
changes no locked verdict. It is descriptive of the critics as they exist in the
checkpoints; it establishes no causal claim about training, which would require
training and is out of scope.

Writes SPRINT_7_DIV_critic_<tag>.json.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[0]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from marl._diag_rung0 import load_agent_and_cfg, OUT_DIR          # noqa: E402
from marl._diag_rung2_75_coherence import build                   # noqa: E402
from marl._diag_rung2_75_matched_states import ACTION_NAMES       # noqa: E402
from marl._diag_div_geometry import HI, LO                        # noqa: E402

TAG = "SPRINT_7_DIV"
ARMS = [("R2", "mappo_R2_mc_target.pth"),
        ("R3", "R3_batch32.pth")]
I_EDGE = ACTION_NAMES.index("MIGRATE_EDGE")
I_STAY = ACTION_NAMES.index("STAY")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=32,
                   help="rollout size, held EQUAL for every cell on purpose")
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--boot-seed", type=int, default=11)
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


def ev(target, pred):
    """1 - var(target - pred) / var(target); the production definition."""
    t = np.asarray(target, np.float64).reshape(-1)
    p = np.asarray(pred, np.float64).reshape(-1)
    vt = t.var()
    return float(1.0 - (t - p).var() / vt) if vt > 1e-12 else float("nan")


# --------------------------------------------------------------- bootstrap
# Explained variance is 1 - var(err)/var(target): a ratio of sums of squares, so
# per-episode partial sums (n, sum e, sum e^2, sum t, sum t^2) are SUFFICIENT
# statistics and a cluster-bootstrap replicate is just a resample of the
# per-episode tuples. No replicate array is ever materialised. Same trick as
# _diag_div_shape's proj_frac/cosine bootstrap; clusters are episodes.

def suff_ev(target, pred, sel, ep):
    t = target[sel].astype(np.float64)
    e = t - pred[sel].astype(np.float64)
    g = ep[sel]
    uniq = np.unique(g)
    S = np.zeros((uniq.size, 5))
    for i, u in enumerate(uniq):
        m = g == u
        tt, ee = t[m], e[m]
        S[i] = (tt.size, ee.sum(), (ee ** 2).sum(), tt.sum(), (tt ** 2).sum())
    return S, uniq


def ev_from_suff(S):
    n = S[..., 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        ve = S[..., 2] / n - (S[..., 1] / n) ** 2
        vt = S[..., 4] / n - (S[..., 3] / n) ** 2
        return 1.0 - ve / vt


def boot_delta_ev(Sa, Sb, n_boot, rng):
    """
    CI on ev(a) - ev(b) with the SAME resampled episodes on both sides, so the
    two critics stay paired inside a replicate. Sa/Sb must share the row order
    (same episode index), which holds because both are built from the same
    selection on the same buffer.
    """
    n = Sa.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    d = ev_from_suff(Sa[idx].sum(1)) - ev_from_suff(Sb[idx].sum(1))
    return {"point": float(ev_from_suff(Sa.sum(0)) - ev_from_suff(Sb.sum(0))),
            "ci95": (float(np.nanpercentile(d, 2.5)),
                     float(np.nanpercentile(d, 97.5))),
            "frac_replicates_positive": float(np.nanmean(d > 0)),
            "n_clusters": int(n)}


def boot_delta_of_delta(SaL, SbL, SaH, SbH, n_boot, rng):
    """CI on [ev_a - ev_b]_lo - [ev_a - ev_b]_hi, all four on one resample."""
    n = SaL.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    dl = ev_from_suff(SaL[idx].sum(1)) - ev_from_suff(SbL[idx].sum(1))
    dh = ev_from_suff(SaH[idx].sum(1)) - ev_from_suff(SbH[idx].sum(1))
    d = dl - dh
    pt = ((ev_from_suff(SaL.sum(0)) - ev_from_suff(SbL.sum(0)))
          - (ev_from_suff(SaH.sum(0)) - ev_from_suff(SbH.sum(0))))
    return {"point": float(pt),
            "ci95": (float(np.nanpercentile(d, 2.5)),
                     float(np.nanpercentile(d, 97.5))),
            "frac_replicates_positive": float(np.nanmean(d > 0)),
            "n_clusters": int(n)}


def score(target, pred, sel):
    t = target[sel]
    p = pred[sel]
    if t.size < 2:
        return {"n": int(t.size)}
    e = t - p
    return {"n": int(t.size), "explained_var": ev(t, p),
            "mean_target": float(t.mean()), "sd_target": float(t.std()),
            "mean_pred": float(p.mean()), "sd_pred": float(p.std()),
            "mean_err": float(e.mean()), "mean_abs_err": float(np.abs(e).mean()),
            "rmse": float(np.sqrt((e ** 2).mean()))}


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 DIVERGENCE — part 6: MATCHED-BUFFER CRITIC COMPARISON")
    print("        both critics vs the SAME target on the SAME states")
    print("        EXPLORATORY. Offline; no optimiser, no writes.")
    print("=" * 78)

    agents, cfgs = {}, {}
    for k, f in ARMS:
        a, e, c = load_agent_and_cfg(str(OUT_DIR / f), args.device, "train")
        agents[k], cfgs[k] = a, c
        print(f"  loaded {k:<4s} {f:<26s} native rollout_episodes="
              f"{e['config']['train']['rollout_episodes']}  "
              f"critic_target={e['config']['train'].get('critic_target')}")

    out = {"probe": "DIV-6 matched-buffer critic comparison", "exploratory": True,
           "hi_cut": HI, "lo_cut": LO, "args": vars(args), "columns": {}}

    for owner, _ in ARMS:
        print("\n" + "=" * 78)
        print(f"  COLUMN: buffer collected by {owner} at {args.episodes} episodes")
        print("=" * 78)
        buf, risk, starts, trunc = build(agents[owner], cfgs[owner],
                                         args.episodes)
        T = len(buf)
        # target: a function of rew/cont/trunc (+ boot at truncations) only, so
        # it is ONE array for the column, shared by both critics
        target = agents[owner].compute_mc_returns(buf)
        state = torch.as_tensor(np.asarray(buf.state[:T]), dtype=torch.float32,
                                device=args.device)
        dec = buf.decision[:T] > 0.5
        act = buf.act[:T]
        hi = dec & (risk >= HI)
        lo = dec & (risk < LO)
        # episode id per timestep, broadcast to (t, agent): cont == 0 marks an
        # episode's last step (mappo.py:305), the same rule compute_gae uses
        contf = buf.cont[:T, 0]
        ept = np.zeros(T, np.int64)
        e_ = 0
        for t in range(T):
            ept[t] = e_
            if contf[t] <= 0.5:
                e_ += 1
        ep = np.broadcast_to(ept[:, None], dec.shape)
        sels = {"all_entries": np.ones_like(dec, bool), "decision": dec,
                "hi": hi, "lo": lo,
                "hi_EDGE": hi & (act == I_EDGE), "hi_STAY": hi & (act == I_STAY),
                "lo_EDGE": lo & (act == I_EDGE)}
        n_trunc = int(np.asarray(trunc).sum()) if trunc is not None else -1
        print(f"  T={T}  entries={dec.size}  decision={int(dec.sum())}  "
              f"hi={int(hi.sum())}  lo={int(lo.sum())}")
        print(f"  truncated episodes (target uses boot there) = {n_trunc}"
              f" of {args.episodes}")
        print(f"  start ticks = {[int(s) for s in starts]}")

        col = {"owner": owner, "T": T, "n_truncated_episodes": n_trunc,
               "start_ticks": [int(s) for s in starts],
               "counts": {k: int(v.sum()) for k, v in sels.items()},
               "critics": {}}

        preds = {}
        for k, _ in ARMS:
            with torch.no_grad():
                preds[k] = agents[k].critic(state).cpu().numpy()
            assert preds[k].shape == target.shape, \
                f"{k}: {preds[k].shape} != {target.shape}"
            col["critics"][k] = {r: score(target, preds[k], s)
                                 for r, s in sels.items()}

        # the diagonal cell must reproduce the arm's own C2 number
        print(f"\n  explained variance   (target is IDENTICAL across the two rows)")
        print(f"  {'critic':<9s}" + "".join(f"{r:>13s}" for r in sels))
        for k, _ in ARMS:
            mark = "  <- diagonal" if k == owner else ""
            print(f"  {k:<9s}" + "".join(
                f"{col['critics'][k][r].get('explained_var', float('nan')):>13.4f}"
                for r in sels) + mark)
        print(f"\n  mean |target - V|")
        print(f"  {'critic':<9s}" + "".join(f"{r:>13s}" for r in sels))
        for k, _ in ARMS:
            print(f"  {k:<9s}" + "".join(
                f"{col['critics'][k][r].get('mean_abs_err', float('nan')):>13.4f}"
                for r in sels))
        print(f"\n  mean (target - V)   [sign = systematic under/over-valuation]")
        print(f"  {'critic':<9s}" + "".join(f"{r:>13s}" for r in sels))
        for k, _ in ARMS:
            print(f"  {k:<9s}" + "".join(
                f"{col['critics'][k][r].get('mean_err', float('nan')):>+13.4f}"
                for r in sels))

        # the quantity the actor's advantage sign actually depends on: does the
        # critic separate high-risk from low-risk states the way the returns do?
        print(f"\n  regime separation on this column   "
              f"(hi minus lo; target gap is the truth)")
        tg = (col["critics"][ARMS[0][0]]["hi"]["mean_target"]
              - col["critics"][ARMS[0][0]]["lo"]["mean_target"])
        print(f"    target  hi-lo = {tg:+.4f}")
        col["target_hi_minus_lo"] = tg
        for k, _ in ARMS:
            c = col["critics"][k]
            pg = c["hi"]["mean_pred"] - c["lo"]["mean_pred"]
            print(f"    {k:<4s} V  hi-lo = {pg:+.4f}   "
                  f"gap error = {pg - tg:+.4f}   "
                  f"hi ev - lo ev = "
                  f"{c['hi']['explained_var'] - c['lo']['explained_var']:+.4f}")
            c["V_hi_minus_lo"] = pg
            c["V_gap_error"] = pg - tg
            c["ev_hi_minus_lo"] = (c["hi"]["explained_var"]
                                   - c["lo"]["explained_var"])

        # ---- HOME ADVANTAGE, with a cluster bootstrap over episodes ----
        # The identified quantity: on THIS buffer, how much better does the
        # critic that collected it fit, relative to the other arm's critic?
        # Reported per regime, because the whole question is whether the home
        # advantage is regime-uniform or concentrated in the low-risk bulk.
        other = [k for k, _ in ARMS if k != owner][0]
        print(f"\n  HOME ADVANTAGE  ev({owner}'s critic) - ev({other}'s critic) "
              f"on {owner}'s own buffer")
        print(f"  cluster bootstrap, {args.boot} replicates, clusters = "
              f"episodes, critics paired within a replicate")
        S = {}
        for reg in ("hi", "lo", "hi_EDGE", "lo_EDGE", "decision"):
            sa, ua = suff_ev(target, preds[owner], sels[reg], ep)
            sb, ub = suff_ev(target, preds[other], sels[reg], ep)
            assert np.array_equal(ua, ub), f"{reg}: cluster index mismatch"
            S[reg] = (sa, sb)
        col["home_advantage"] = {}
        print(f"    {'regime':<10s}{'delta ev':>10s}{'95% CI':>22s}"
              f"{'frac>0':>9s}{'clusters':>10s}")
        for reg, (sa, sb) in S.items():
            b = boot_delta_ev(sa, sb, args.boot,
                              np.random.default_rng(args.boot_seed))
            col["home_advantage"][reg] = b
            print(f"    {reg:<10s}{b['point']:>+10.4f}"
                  f"   [{b['ci95'][0]:+.4f},{b['ci95'][1]:+.4f}]"
                  f"{b['frac_replicates_positive']:>9.4f}{b['n_clusters']:>10d}")
        dod = boot_delta_of_delta(*S["lo"], *S["hi"], args.boot,
                                  np.random.default_rng(args.boot_seed + 1))
        col["home_advantage_lo_minus_hi"] = dod
        print(f"    {'lo - hi':<10s}{dod['point']:>+10.4f}"
              f"   [{dod['ci95'][0]:+.4f},{dod['ci95'][1]:+.4f}]"
              f"{dod['frac_replicates_positive']:>9.4f}{dod['n_clusters']:>10d}")
        print(f"    ^ positive = the home critic's advantage is CONCENTRATED in "
              f"the low-risk bulk")
        out["columns"][owner] = col

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 78)
    print("  2x2 SUMMARY: hi explained variance MINUS lo explained variance")
    print("  (negative = the critic fits high-risk states WORSE than low-risk)")
    print("=" * 78)
    print(f"  {'critic':<9s}" + "".join(f"{'buffer '+o:>18s}" for o, _ in ARMS))
    for k, _ in ARMS:
        print(f"  {k:<9s}" + "".join(
            f"{out['columns'][o]['critics'][k]['ev_hi_minus_lo']:>+18.4f}"
            for o, _ in ARMS))
    print(f"\n  {'critic':<9s}" + "".join(
        f"{'hi ev @'+o:>18s}" for o, _ in ARMS))
    for k, _ in ARMS:
        print(f"  {k:<9s}" + "".join(
            f"{out['columns'][o]['critics'][k]['hi']['explained_var']:>18.4f}"
            for o, _ in ARMS))

    print("\n" + "=" * 78)
    print("  HOME ADVANTAGE BY REGIME  (own critic minus other critic, on own")
    print("  buffer). Read DOWN: is the arm's critic better than its rival's")
    print("  everywhere, or only in the low-risk bulk?")
    print("=" * 78)
    print(f"  {'owner':<9s}{'hi':>12s}{'lo':>12s}{'lo - hi':>12s}"
          f"{'lo-hi CI':>24s}")
    for o, _ in ARMS:
        c = out["columns"][o]
        d = c["home_advantage_lo_minus_hi"]
        print(f"  {o:<9s}{c['home_advantage']['hi']['point']:>+12.4f}"
              f"{c['home_advantage']['lo']['point']:>+12.4f}"
              f"{d['point']:>+12.4f}"
              f"   [{d['ci95'][0]:+.4f},{d['ci95'][1]:+.4f}]")

    p = OUT_DIR / f"{TAG}_critic_{args.tag}.json"
    p.write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
