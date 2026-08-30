#!/usr/bin/env python
"""
SPRINT 7 RUNG 2.75 -- item C: do raw per-state advantage OFFSETS suppress PPO
learning?

DIAGNOSIS ONLY. No centred/matched advantage is implemented anywhere in this
file. Nothing is trained. No production code is touched.

THE SETUP (why an "offset" is even well defined).

  Under a greedy baseline the truth used by every Sprint 7 rung is a difference
  taken at ONE state:

      A_true(s, a) = Q(s, a) - Q(s, a_ref)      =>   A_true(s, a_ref) == 0

  So for the reference action the true advantage is EXACTLY zero, and whatever
  GAE reports there is, by construction, pure estimator error. Define

      c(s) := gae(s, a_ref)          <- the per-state offset ("noise floor")

  and decompose the estimator PPO actually consumes:

      gae(s, a) = c(s) + [gae(s, a) - gae(s, a_ref)]
                = c(s) +      paired(s, a)

  `paired` is the matched estimator (Rung 2.5); `c(s)` is the contaminant. The
  actor's surrogate uses the RAW left-hand side.

WHAT IS MEASURED, on the 547 R2-NATIVE rows only (Rung 2.5's item E addendum,
already on disk -- no re-derivation, no new deviation set):

  C1  raw vs paired vs true: level, SD, MAE, correlation, regression slope
  C2  variance decomposition Var(raw) = Var(c) + Var(paired) + 2Cov, per bucket
  C3  action ORDERING recovered within each state, raw vs paired (Kendall tau)
  C4  signal-to-offset ratio: SD(c) against SD(paired) and SD(A_true)
  C5  the PPO link. The gradient of the clipped surrogate w.r.t. the logits is

          d(pg)/d(logit_b) = -A_hat * (1{b=a} - pi_b)      (unclipped region)

      so the per-entry push is |A_hat| * ||e_a - pi||. For an ON-POLICY entry
      whose action is the greedy one, A_true == 0 and therefore A_hat == c(s)
      exactly: the whole update on that entry is offset. This measures what
      share of the batch's total policy-gradient magnitude is offset-driven,
      and -- separately -- how much of it survives saturation.

  C6  what global advantage normalisation can and cannot remove: PPO subtracts
      ONE batch mean over decision entries, so only E[c] goes; the within-state
      dispersion SD(c) stays. Quantified per bucket.

Writes SPRINT_7_RUNG2_75_offset.json.
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

from marl._diag_rung0 import (                                   # noqa: E402
    load_agent_and_cfg, _replay, OUT_DIR, BUCKETS,
    ACTION_STAY, ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD,
)
from marl.env import DTMarlEnv                                   # noqa: E402
from marl.mappo import MASK_FILL                                 # noqa: E402
from marl.rollout import episode_starts                          # noqa: E402

TAG = "SPRINT_7_RUNG2_75"
ACTION_NAMES = ["STAY", "MIGRATE_EDGE", "MIGRATE_CLOUD", "PREEMPT_REROUTE"]
NATIVE = "SPRINT_7_RUNG2_5_native_dev_R2.json"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--native", default=str(OUT_DIR / NATIVE),
                   help="Rung 2.5 item-E addendum rows (R2-native, n=547)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--arm", default="C0", help="R2's own deployed critic")
    p.add_argument("--tag", default="R2")
    return p.parse_args(argv)


def _ik(d):
    return {int(k): float(v) for k, v in d.items()}


def _stats(x):
    x = np.asarray(x, np.float64)
    if x.size == 0:
        return dict(n=0)
    return dict(n=int(x.size), mean=float(x.mean()),
                sd=float(x.std(ddof=1)) if x.size > 1 else 0.0,
                median=float(np.median(x)),
                p05=float(np.percentile(x, 5)),
                p95=float(np.percentile(x, 95)),
                mean_abs=float(np.abs(x).mean()))


def _pearson(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.size < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _ols(x, y):
    """slope of y on x; 1.0 would mean the estimator is correctly scaled."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.size < 3 or x.std() < 1e-12:
        return None
    A = np.vstack([x, np.ones_like(x)]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return dict(slope=float(m), intercept=float(b))


def _kendall(a, b):
    """Kendall tau-b on short vectors -- O(n^2) is fine, n<=3 per state."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    n = a.size
    if n < 2:
        return None
    conc = disc = ta = tb = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                ta += 1; tb += 1
            elif da == 0:
                ta += 1
            elif db == 0:
                tb += 1
            elif np.sign(da) == np.sign(db):
                conc += 1
            else:
                disc += 1
    n0 = n * (n - 1) / 2
    den = np.sqrt(max(n0 - ta, 0) * max(n0 - tb, 0))
    return float((conc - disc) / den) if den > 0 else None


# ----------------------------------------------------------------------
# policy at each row's state -- 8 greedy replays, nothing more
# ----------------------------------------------------------------------

def policy_at_rows(agent, cfg, rows, episodes=8):
    """
    pi(.|s) and the softmax attenuation factor ||e_a - pi|| at every row.

    ||e_a - pi|| is the EXACT logit-space norm of d log pi(a|s)/d logits, so it
    is the state-dependent factor multiplying A_hat in the policy gradient. It
    goes to 0 as pi(a|s) -> 1, which is the saturation channel from item B.
    """
    env = DTMarlEnv(cfg.env, cfg.reward)
    starts = episode_starts(env, episodes)
    need = sorted({int(r["start"]) for r in rows})
    missing = [s for s in need if s not in set(int(x) for x in starts)]
    cache = {}
    for j, s in enumerate(starts):
        rec = _replay(env, agent, int(s), j, record=True)["rec"]
        with torch.no_grad():
            ob = torch.as_tensor(rec["obs"], dtype=torch.float32,
                                 device=agent.device)
            mk = torch.as_tensor(rec["mask"], dtype=torch.float32,
                                 device=agent.device)
            T = ob.shape[0]
            P = np.empty((T, env.n_agents, env.n_actions), np.float64)
            for i in range(env.n_agents):
                lg = agent.actor.logits(i, ob[:, i]).masked_fill(
                    ~mk[:, i].bool(), MASK_FILL)
                P[:, i] = torch.softmax(lg, dim=-1).cpu().numpy()
        cache[int(s)] = dict(P=P, act=rec["act"], T=T)
    out = []
    for r in rows:
        c = cache.get(int(r["start"]))
        if c is None or int(r["step"]) >= c["T"]:
            out.append(None); continue
        t, i = int(r["step"]), int(r["agent"])
        pi = c["P"][t, i]
        a_ref = int(r["ref_action"])
        e = np.zeros_like(pi); e[a_ref] = 1.0
        out.append(dict(
            pi=pi.tolist(),
            pi_ref=float(pi[a_ref]),
            maxp=float(pi.max()),
            argmax=int(pi.argmax()),
            entropy=float(-(pi * np.log(np.clip(pi, 1e-12, None))).sum()),
            attenuation=float(np.linalg.norm(e - pi)),
            greedy_action=int(c["act"][t, i]),
        ))
    return out, missing, [int(s) for s in starts]


# ----------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 RUNG 2.75 -- item C: per-state advantage OFFSET vs PPO")
    print("        DIAGNOSIS ONLY -- no advantage change is implemented")
    print("=" * 78)

    nat = json.loads(Path(args.native).read_text())
    rows = nat["rows"]
    arm = args.arm
    print(f"  native rows : {len(rows)}  from {Path(args.native).name}")
    print(f"  model       : {nat['model']}  critic_target={nat['critic_target']}")
    print(f"  arm         : {arm} (the checkpoint's own deployed critic)")

    agent, extra, cfg = load_agent_and_cfg(nat["model"], args.device, "train")
    pol, missing, starts = policy_at_rows(agent, cfg, rows, nat["episodes"])
    print(f"  greedy replays: {starts}  (rows missing a replay: {len(missing)})")

    # ---------------- flatten to per-(state, deviating action) records -----
    recs = []
    for r, p in zip(rows, pol):
        g = _ik(r["gae"][arm])
        ref = int(r["ref_action"])
        if ref not in g:
            continue
        c = g[ref]                              # the per-state offset
        for truth_key, tname in (("a_true_team", "team"), ("a_true_own", "own")):
            tv = _ik(r[truth_key])
            for a, t in tv.items():
                if a == ref or a not in g:
                    continue
                recs.append(dict(
                    truth=tname, bucket=r["bucket"], risk=float(r["risk"]),
                    action=a, ref=ref, a_true=t, raw=g[a],
                    paired=g[a] - c, offset=c,
                    start=int(r["start"]), step=int(r["step"]),
                    agent=int(r["agent"]),
                    attenuation=(p or {}).get("attenuation"),
                    pi_ref=(p or {}).get("pi_ref"),
                    maxp=(p or {}).get("maxp"),
                ))
    print(f"  (state, deviating action) records: {len(recs)} "
          f"({len(recs) // 2} per truth definition)")

    # ---------------- C1/C2/C4/C6 -----------------------------------------
    est = {}
    for tname in ("team", "own"):
        est[tname] = {}
        for b in list(BUCKETS) + ["all"]:
            R = [x for x in recs if x["truth"] == tname
                 and (b == "all" or x["bucket"] == b)]
            if len(R) < 3:
                est[tname][b] = dict(n=len(R)); continue
            T = np.array([x["a_true"] for x in R])
            RAW = np.array([x["raw"] for x in R])
            PA = np.array([x["paired"] for x in R])
            OF = np.array([x["offset"] for x in R])
            nz = np.abs(T) > 1e-9
            vr, vc, vp = RAW.var(ddof=1), OF.var(ddof=1), PA.var(ddof=1)
            est[tname][b] = dict(
                n=len(R),
                true=_stats(T), raw=_stats(RAW), paired=_stats(PA),
                offset=_stats(OF),
                mae_raw=float(np.abs(RAW - T).mean()),
                mae_paired=float(np.abs(PA - T).mean()),
                corr_raw=_pearson(T, RAW), corr_paired=_pearson(T, PA),
                ols_raw=_ols(T, RAW), ols_paired=_ols(T, PA),
                sign_raw=float(np.mean(np.sign(T[nz]) == np.sign(RAW[nz])))
                if nz.sum() else None,
                sign_paired=float(np.mean(np.sign(T[nz]) == np.sign(PA[nz])))
                if nz.sum() else None,
                # C2: Var(raw) = Var(offset) + Var(paired) + 2 Cov
                var_decomp=dict(
                    var_raw=float(vr), var_offset=float(vc),
                    var_paired=float(vp),
                    cov2=float(2 * np.cov(OF, PA, ddof=1)[0, 1]),
                    offset_share_of_var=float(vc / vr) if vr > 0 else None,
                    paired_share_of_var=float(vp / vr) if vr > 0 else None,
                ),
                # C4: signal-to-offset
                sd_ratio_offset_over_paired=float(
                    np.sqrt(vc) / np.sqrt(vp)) if vp > 0 else None,
                sd_ratio_offset_over_true=float(
                    np.sqrt(vc) / T.std(ddof=1)) if T.std(ddof=1) > 0 else None,
                # C6: what one global mean subtraction can remove
                offset_mean_removable=float(OF.mean()),
                offset_sd_irreducible=float(np.sqrt(vc)),
                corr_offset_risk=_pearson([x["risk"] for x in R], OF),
            )

    print("\n-- C1/C4  estimator vs truth, HIGH-RISK bucket ----------------")
    for tname in ("team", "own"):
        e = est[tname]["hi"]
        if not e.get("n", 0) >= 3:
            continue
        print(f"  [{tname}] n={e['n']}")
        print(f"    true    mean {e['true']['mean']:+8.4f}  sd {e['true']['sd']:7.4f}")
        print(f"    raw     mean {e['raw']['mean']:+8.4f}  sd {e['raw']['sd']:7.4f}"
              f"   MAE {e['mae_raw']:7.4f}  r {str(e['corr_raw'])[:7]:>7s}"
              f"  sign {e['sign_raw']:.4f}")
        print(f"    paired  mean {e['paired']['mean']:+8.4f}  sd {e['paired']['sd']:7.4f}"
              f"   MAE {e['mae_paired']:7.4f}  r {str(e['corr_paired'])[:7]:>7s}"
              f"  sign {e['sign_paired']:.4f}")
        print(f"    offset  mean {e['offset']['mean']:+8.4f}  sd {e['offset']['sd']:7.4f}"
              f"   SD(offset)/SD(paired) = "
              f"{e['sd_ratio_offset_over_paired']:.2f}x")
        v = e["var_decomp"]
        print(f"    Var(raw)={v['var_raw']:.3f} = Var(offset) {v['var_offset']:.3f} "
              f"({v['offset_share_of_var']:.1%}) + Var(paired) "
              f"{v['var_paired']:.3f} ({v['paired_share_of_var']:.1%}) "
              f"+ 2Cov {v['cov2']:+.3f}")

    # ---------------- C3: within-state ordering ---------------------------
    order = {}
    for tname in ("team", "own"):
        order[tname] = {}
        for b in list(BUCKETS) + ["all"]:
            per_state = {}
            for x in recs:
                if x["truth"] != tname or (b != "all" and x["bucket"] != b):
                    continue
                per_state.setdefault((x["start"], x["step"], x["agent"]), []
                                     ).append(x)
            tr, tp, n_multi, exact_r, exact_p = [], [], 0, 0, 0
            for k, xs in per_state.items():
                # include the reference action: truth 0, raw = offset, paired 0
                T = [0.0] + [x["a_true"] for x in xs]
                RW = [xs[0]["offset"]] + [x["raw"] for x in xs]
                PA = [0.0] + [x["paired"] for x in xs]
                if len(T) < 2:
                    continue
                n_multi += 1
                kr, kp = _kendall(T, RW), _kendall(T, PA)
                if kr is not None:
                    tr.append(kr); exact_r += int(kr > 0.999)
                if kp is not None:
                    tp.append(kp); exact_p += int(kp > 0.999)
            order[tname][b] = dict(
                n_states=n_multi,
                kendall_raw=_stats(tr), kendall_paired=_stats(tp),
                frac_exact_order_raw=float(exact_r / max(n_multi, 1)),
                frac_exact_order_paired=float(exact_p / max(n_multi, 1)),
                note="the reference action is INCLUDED in each ranking; that "
                     "is where raw carries the offset and paired carries 0",
            )

    print("\n-- C3  within-state action ORDERING (ref action included) ------")
    for tname in ("team", "own"):
        for b in ("hi", "lo", "all"):
            o = order[tname][b]
            if not o["kendall_raw"].get("n"):
                continue
            print(f"  [{tname}] {b:3s} states={o['n_states']:4d}  "
                  f"tau raw {o['kendall_raw']['mean']:+.4f}  "
                  f"paired {o['kendall_paired']['mean']:+.4f}   "
                  f"exact-order raw {o['frac_exact_order_raw']:.4f}  "
                  f"paired {o['frac_exact_order_paired']:.4f}")

    # ---------------- C5: the PPO policy-gradient link --------------------
    # on-policy entries: action == greedy == ref, so A_true == 0 exactly and
    # A_hat == offset. push = |A_hat| * ||e_a - pi||.
    pg = {}
    for b in list(BUCKETS) + ["all"]:
        S = [(r, p) for r, p in zip(rows, pol)
             if p is not None and (b == "all" or r["bucket"] == b)]
        if not S:
            pg[b] = dict(n=0); continue
        c = np.array([_ik(r["gae"][arm])[int(r["ref_action"])] for r, _ in S])
        at = np.array([p["attenuation"] for _, p in S])
        pi_ref = np.array([p["pi_ref"] for _, p in S])
        push_off = np.abs(c) * at
        # the paired signal that a matched estimator WOULD have supplied, for
        # scale only -- it is not applied anywhere
        pair_scale = []
        for r, _ in S:
            g = _ik(r["gae"][arm]); ref = int(r["ref_action"])
            d = [abs(g[a] - g[ref]) for a in g if a != ref]
            pair_scale.append(float(np.mean(d)) if d else 0.0)
        pair_scale = np.array(pair_scale)
        pg[b] = dict(
            n=len(S),
            offset_abs=_stats(np.abs(c)),
            attenuation=_stats(at),
            pi_ref=_stats(pi_ref),
            push_offset_driven=_stats(push_off),
            push_if_matched=_stats(pair_scale * at),
            ratio_offset_push_over_matched_push=float(
                push_off.sum() / max((pair_scale * at).sum(), 1e-12)),
            frac_saturated_pi_ref_gt_099=float((pi_ref > 0.99).mean()),
            frac_attenuation_lt_0_02=float((at < 0.02).mean()),
            share_of_total_push_from_unsaturated=float(
                push_off[at >= 0.02].sum() / max(push_off.sum(), 1e-12)),
            note="A_true(ref) == 0 by construction, so on an on-policy entry "
                 "whose action is the greedy one the ENTIRE policy-gradient "
                 "push is the offset. Valid under the greedy-baseline truth "
                 "definition used in Rungs 0-2.5; in stochastic training the "
                 "sampled action is not always greedy, so read this as the "
                 "greedy-entry subset, not the whole batch.",
        )

    print("\n-- C5  PPO policy-gradient link -------------------------------")
    print(f"  {'bucket':<6s} {'n':>5s} {'|offset|':>9s} {'atten':>8s} "
          f"{'pi_ref':>8s} {'push_off':>9s} {'push_match':>10s} "
          f"{'off/match':>10s} {'sat>.99':>8s}")
    for b in ("lo", "mid", "hi", "all"):
        d = pg[b]
        if not d.get("n"):
            continue
        print(f"  {b:<6s} {d['n']:>5d} {d['offset_abs']['mean']:>9.4f} "
              f"{d['attenuation']['mean']:>8.4f} {d['pi_ref']['mean']:>8.4f} "
              f"{d['push_offset_driven']['mean']:>9.4f} "
              f"{d['push_if_matched']['mean']:>10.4f} "
              f"{d['ratio_offset_push_over_matched_push']:>10.2f} "
              f"{d['frac_saturated_pi_ref_gt_099']:>8.4f}")

    blob = dict(
        probe=f"{TAG}_offset_{args.tag}",
        what="do raw per-state advantage offsets suppress PPO learning? "
             "diagnosis only; no centred/matched advantage implemented",
        definition=dict(
            offset="c(s) := gae(s, a_ref); A_true(s, a_ref) == 0 exactly under "
                   "a greedy baseline, so this is pure estimator error",
            decomposition="gae(s,a) = c(s) + paired(s,a)",
            attenuation="||e_a - pi(.|s)||_2 = exact logit-space norm of "
                        "d log pi(a|s)/d logits; the state factor multiplying "
                        "A_hat in the policy gradient",
            ppo_uses="the RAW form (mappo.py:394-402, s1 = ratio * adv[idx])",
        ),
        source=dict(native=str(args.native), model=nat["model"],
                    critic_target=nat["critic_target"], arm=arm,
                    n_rows=len(rows), n_records=len(recs),
                    greedy_starts=starts, rows_without_replay=len(missing)),
        C1_C2_C4_C6_estimators=est,
        C3_ordering=order,
        C5_policy_gradient=pg,
        rung2_5_noise_floor_for_crosscheck=nat["noise_floor"],
    )
    p = OUT_DIR / f"{TAG}_offset_{args.tag}.json"
    p.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n  wrote {p}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
