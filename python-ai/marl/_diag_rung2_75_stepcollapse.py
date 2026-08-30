#!/usr/bin/env python
"""
SPRINT 7 RUNG 2.75 -- item B, the part no previous diagnostic covered:
WHY did the actor's step utilisation collapse DURING TRAINING?

WHY THIS EXISTS -- A METHODOLOGICAL FLAW SHARED BY EVERY PRIOR ACTOR DIAGNOSTIC.

Every actor-stall measurement in this project (Rung 2.5's replica, this rung's
plasticity, coherence and mbtail probes) was taken at the FINAL checkpoint. At
that point clip_frac is 0.0000 and ratio_max reaches 3-9% of the permitted trust
region, and the natural reading is "the actor cannot move". But the training CSVs
say something different:

    clip_frac  A0  0.1081 at update 1  ->  0.0000 at update 75
               R2  0.1625 at update 1  ->  0.0000 at update 75

The actor STARTED with a healthy trust region and lost it. Every checkpoint-time
probe therefore measured the BOTTOM of a collapse, and could not see the collapse
itself. This file measures the collapse, from the per-update CSVs only.

THE DECOMPOSITION.

  train.py anneals the learning rate linearly, lr_scale: 1.0 -> 0.0133 over 75
  updates. Adam's step is approximately lr-sized regardless of gradient
  magnitude, so the PPO ratio deviation per update scales roughly with lr_scale:

      |ratio - 1| ~ lr_scale * ||grad log pi|| = lr_scale * ||e_a - pi|| * ...

  So a collapse in clip_frac has (at least) two candidate causes, and they are
  separable using the schedule, which is known exactly:

      SCHEDULE   lr_scale falls by construction -- 75x end to end.
      SHARPENING pi(a|s) -> 1 makes ||e_a - pi|| -> 0, shrinking the ratio
                 movement produced PER UNIT of lr. This shows up in
                 clip_frac / lr_scale, which is lr-free by construction.

  If clip_frac / lr_scale is flat, the collapse is entirely the schedule and the
  remedy is an lr question. If clip_frac / lr_scale falls, the actor is losing
  step utilisation for reasons the schedule does not explain, and raising lr can
  only recover the schedule's share.

  approx_kl / lr_scale^2 is reported as an independent check with a different
  power of lr (KL is second order in the step, clip_frac is first order in it).

NO MODEL IS LOADED. NO ROLLOUT IS RUN. This reads two CSVs that training already
wrote. It cannot disturb anything.

Writes SPRINT_7_RUNG2_75_stepcollapse.json.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[0]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from marl._diag_rung0 import OUT_DIR                             # noqa: E402

TAG = "SPRINT_7_RUNG2_75"
CSVS = {"A0": "mappo_A0_cpu_repro_updates.csv",
        "R2": "mappo_R2_mc_target_updates.csv"}
EPS = 1e-9


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--also", default="",
                   help="comma-separated extra arms as name=csvfile")
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


def _rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(x.size, float)
    r[order] = np.arange(1, x.size + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < xs.size:
        j = i
        while j + 1 < xs.size and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    return r


def spearman(a, b):
    if a.size < 3:
        return None
    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else None


def quartiles(v):
    n = v.size
    return [float(v[i * n // 4:(i + 1) * n // 4].mean()) for i in range(4)]


def ols(y, X):
    """Least squares with intercept; returns coefficients and R^2."""
    A = np.column_stack([np.ones(y.size)] + list(X))
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ b
    ss = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - ((y - yh) ** 2).sum() / ss) if ss > 0 else None
    return [float(x) for x in b], r2


def load(fn):
    rows = list(csv.DictReader(open(OUT_DIR / fn)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def analyse(tag, fn):
    d = load(fn)
    cf, kl, ls = d["clip_frac"], d["approx_kl"], d["lr_scale"]
    ent, n = d["entropy"], cf.size
    u1 = cf / np.maximum(ls, EPS)                 # lr-free, first order in step
    u2 = kl / np.maximum(ls ** 2, EPS)            # lr-free, second order
    q_cf, q_ls, q_u1, q_ent = (quartiles(x) for x in (cf, ls, u1, ent))

    def ratio(q):
        return float(q[0] / q[3]) if q[3] > 0 else None

    r_cf, r_ls, r_u1 = ratio(q_cf), ratio(q_ls), ratio(q_u1)
    # ---------------------------------------------------------------------
    # The attribution must FACTORISE exactly, and with r_u1 it does not:
    # u1 = clip_frac/lr_scale is averaged PER UPDATE, and
    # mean(cf/ls) != mean(cf)/mean(ls). The first version of this file used
    # r_u1 as the sharpening factor and its own product_check caught the
    # discrepancy (A0: 6.7 x 21.1 = 140.6 vs a total of 105.4). The exactly
    # factorising sharpening term uses the RATIO OF QUARTILE MEANS:
    #     r_cf = (q_cf[0]/q_cf[3])
    #          = (q_ls[0]/q_ls[3]) * ((q_cf[0]/q_ls[0]) / (q_cf[3]/q_ls[3]))
    # which is an identity. r_u1 is retained alongside as the honest average of
    # the lr-free series; the headline attribution uses the identity.
    # ---------------------------------------------------------------------
    r_sharp = (float((q_cf[0] / q_ls[0]) / (q_cf[3] / q_ls[3]))
               if q_cf[3] > 0 and q_ls[0] > 0 and q_ls[3] > 0 else None)
    check = (float(r_ls * r_sharp) if (r_ls and r_sharp) else None)
    # when was the trust region last actually engaged?
    eng = np.flatnonzero(cf > 0.01)
    # log-log fit: does entropy explain the lr-free residual?
    ok = (u1 > 0)
    coef, r2 = (ols(np.log(u1[ok]), [np.log(np.maximum(ent[ok], EPS))])
                if ok.sum() > 3 else (None, None))
    out = dict(
        source_csv=fn, n_updates=int(n),
        clip_frac=dict(first=float(cf[0]), last=float(cf[-1]),
                       max=float(cf.max()), quartile_means=q_cf,
                       collapse_Q1_over_Q4=r_cf),
        lr_scale=dict(first=float(ls[0]), last=float(ls[-1]),
                      quartile_means=q_ls, collapse_Q1_over_Q4=r_ls),
        clip_frac_over_lr_scale=dict(
            what="clip_frac with the annealing schedule divided out; lr-free "
                 "by construction, so any remaining collapse is NOT the "
                 "schedule",
            quartile_means=q_u1, collapse_Q1_over_Q4=r_u1,
            caveat="this is mean(cf/ls) per quartile, which does NOT factorise "
                   "exactly against mean(cf)/mean(ls); the headline attribution "
                   "uses the exactly-factorising form instead"),
        approx_kl_over_lr_scale_sq=dict(
            what="independent check with a different power of lr (KL is second "
                 "order in the step, clip_frac first order)",
            quartile_means=quartiles(u2),
            collapse_Q1_over_Q4=(float(quartiles(u2)[0] / quartiles(u2)[3])
                                 if quartiles(u2)[3] > 0 else None)),
        entropy=dict(first=float(ent[0]), last=float(ent[-1]),
                     quartile_means=q_ent, collapse_Q1_over_Q4=ratio(q_ent)),
        attribution=dict(
            total_clip_frac_collapse=r_cf,
            explained_by_lr_schedule=r_ls,
            explained_by_sharpening=r_sharp,
            lr_free_series_mean_ratio=r_u1,
            product_check=check,
            product_matches_total=(abs(check - r_cf) / r_cf < 0.02
                                   if (check and r_cf) else None),
            identity="total = schedule x sharpening, exactly, by construction",
            dominant=("sharpening" if (r_sharp and r_ls and r_sharp > r_ls)
                      else "lr schedule" if (r_sharp and r_ls) else None),
        ),
        trust_region_engagement=dict(
            n_updates_clip_frac_gt_001=int(eng.size),
            frac_of_run=float(eng.size / n),
            last_update_engaged=int(d["update"][eng[-1]]) if eng.size else None,
            first_update_never_engaged_after=(int(d["update"][eng[-1]]) + 1
                                              if eng.size else 1),
        ),
        entropy_vs_lr_free_utilisation=dict(
            spearman=spearman(ent, u1),
            loglog_slope_on_entropy=(coef[1] if coef else None),
            loglog_r2=r2,
            reading="a positive slope near 1 means the lr-free loss of step "
                    "utilisation tracks the policy's own sharpening, which is "
                    "what ||e_a - pi|| -> 0 predicts",
        ),
        counterfactual_lr_held_at_full=dict(
            what="clip_frac that the FINAL quartile would have shown if "
                 "lr_scale had stayed at 1.0, holding the measured lr-free "
                 "utilisation fixed",
            value=float(q_cf[3] / q_ls[3]) if q_ls[3] > 0 else None,
            vs_first_quartile_clip_frac=float(q_cf[0] / q_ls[0])
            if q_ls[0] > 0 else None,
            shortfall_factor=r_sharp,
            reading="this is the ceiling on what an lr-only intervention can "
                    "recover: it restores the schedule's factor and nothing "
                    "else, leaving the sharpening factor untouched",
        ),
    )
    return out, d


def main(argv=None):
    args = parse_args(argv)
    arms = dict(CSVS)
    for kv in (x for x in args.also.split(",") if x.strip()):
        k, v = kv.split("=", 1)
        arms[k.strip()] = v.strip()

    print("=" * 78)
    print("SPRINT 7 RUNG 2.75 -- item B: WHY the actor's trust region")
    print("        collapsed DURING TRAINING (schedule vs sharpening)")
    print("        CSVs only -- no model, no rollout, nothing loaded")
    print("=" * 78)
    print("  every prior actor diagnostic in this project measured the FINAL")
    print("  checkpoint, i.e. the BOTTOM of this collapse, and so could not")
    print("  distinguish 'the actor cannot move' from 'the actor was annealed")
    print("  and sharpened until it stopped moving'.")

    out = {}
    for tag, fn in arms.items():
        try:
            res, d = analyse(tag, fn)
        except FileNotFoundError:
            print(f"\n-- {tag}: {fn} NOT FOUND, skipped")
            continue
        out[tag] = res
        a = res["attribution"]
        print(f"\n-- {tag} ({fn}, {res['n_updates']} updates) " + "-" * 8)
        print(f"   clip_frac        {res['clip_frac']['first']:.4f} -> "
              f"{res['clip_frac']['last']:.4f}   quartiles "
              + " ".join(f"{x:.5f}" for x in res['clip_frac']['quartile_means']))
        print(f"   lr_scale         {res['lr_scale']['first']:.4f} -> "
              f"{res['lr_scale']['last']:.4f}   quartiles "
              + " ".join(f"{x:.5f}" for x in res['lr_scale']['quartile_means']))
        print(f"   entropy          {res['entropy']['first']:.4f} -> "
              f"{res['entropy']['last']:.4f}   quartiles "
              + " ".join(f"{x:.5f}" for x in res['entropy']['quartile_means']))
        print(f"   clip_frac/lr_scale (lr-free)          quartiles "
              + " ".join(f"{x:.5f}" for x in
                         res['clip_frac_over_lr_scale']['quartile_means']))
        print(f"\n   ATTRIBUTION of the clip_frac collapse (Q1 mean / Q4 mean)")
        print(f"      total                       {a['total_clip_frac_collapse']:8.1f}x")
        print(f"      due to the lr SCHEDULE      {a['explained_by_lr_schedule']:8.1f}x")
        print(f"      due to SHARPENING (lr-free) {a['explained_by_sharpening']:8.1f}x")
        print(f"      product check               {a['product_check']:8.1f}x"
              f"   matches total: {a['product_matches_total']}")
        print(f"      DOMINANT CAUSE: {a['dominant']}")
        e = res["trust_region_engagement"]
        print(f"   trust region engaged (clip_frac>0.01) in "
              f"{e['n_updates_clip_frac_gt_001']}/{res['n_updates']} updates "
              f"({e['frac_of_run']:.0%}); last at update "
              f"{e['last_update_engaged']}")
        ev = res["entropy_vs_lr_free_utilisation"]
        print(f"   lr-free utilisation vs entropy: spearman "
              f"{ev['spearman']:+.4f}   log-log slope "
              f"{ev['loglog_slope_on_entropy']:+.3f} (R2 {ev['loglog_r2']:.3f})")
        cfa = res["counterfactual_lr_held_at_full"]
        print(f"   CEILING on an lr-only fix: final-quartile clip_frac at "
              f"lr_scale=1 would be {cfa['value']:.5f} vs "
              f"{cfa['vs_first_quartile_clip_frac']:.5f} early "
              f"({cfa['shortfall_factor']:.1f}x short)")

    print("\n" + "=" * 78)
    print("SUMMARY -- what an lr-only intervention could and could not recover")
    print("=" * 78)
    print(f"  {'arm':>4s} {'total':>9s} {'schedule':>9s} {'sharpen':>9s} "
          f"{'dominant':>13s} {'lr-only ceiling':>16s}")
    for tag, r in out.items():
        a, c = r["attribution"], r["counterfactual_lr_held_at_full"]
        print(f"  {tag:>4s} {a['total_clip_frac_collapse']:>8.1f}x "
              f"{a['explained_by_lr_schedule']:>8.1f}x "
              f"{a['explained_by_sharpening']:>8.1f}x "
              f"{str(a['dominant']):>13s} {c['shortfall_factor']:>15.1f}x")
    print("\n  read: 'sharpen' is the factor that the annealing schedule does")
    print("        NOT explain. Where it exceeds 'schedule', raising the")
    print("        learning rate addresses the smaller of the two causes.")

    blob = dict(
        probe=f"{TAG}_stepcollapse",
        what="attribution of the actor's trust-region collapse during training "
             "into the known lr annealing schedule versus policy sharpening, "
             "using only the per-update CSVs training already wrote",
        flaw_it_corrects=dict(
            what="every prior actor-stall diagnostic in this project measured "
                 "the FINAL checkpoint, where clip_frac is 0.0000",
            why_that_misleads="clip_frac STARTS healthy (0.1081 A0 / 0.1625 R2) "
                              "and collapses over the run, so a final-checkpoint "
                              "measurement observes the bottom of a collapse and "
                              "cannot distinguish an incapable actor from an "
                              "annealed and sharpened one",
            handling="no prior artifact is altered; this adds the training-time "
                     "measurement they were all missing",
        ),
        method=dict(
            identity="Adam's step is approximately lr-sized regardless of "
                     "gradient magnitude, so the PPO ratio deviation is first "
                     "order in lr_scale and clip_frac/lr_scale is lr-free by "
                     "construction",
            schedule_is_known_exactly="train.py anneals lr_scale linearly "
                                      "1.0 -> 1/75, and lr_scale is logged per "
                                      "update, so the schedule's contribution "
                                      "needs no estimation",
            independent_check="approx_kl/lr_scale^2 has a different power of lr; "
                              "if both lr-free series collapse, the residual is "
                              "not an artifact of the assumed power",
            caveat="Adam's step is only approximately lr-sized -- with grad "
                   "clipping at max_grad_norm 0.5 and bias correction the "
                   "proportionality is not exact, so the split is an "
                   "attribution, not an identity. The two independent powers of "
                   "lr are reported precisely so this can be judged.",
        ),
        not_a_fix="reads two CSVs; loads no model, runs no rollout, writes no "
                  "checkpoint, and proposes no hyperparameter change",
        per_arm=out,
    )
    p = OUT_DIR / f"{TAG}_stepcollapse_{args.tag}.json"
    p.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n  wrote {p}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
