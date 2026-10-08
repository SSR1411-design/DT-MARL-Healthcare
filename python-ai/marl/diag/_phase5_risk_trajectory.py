"""
Sprint 7 PHASE 5 -- WHEN during R2 training does the risk-conditioned
(self-healing) EDGE response emerge, strengthen, plateau, or disappear?

READ-ONLY with respect to every existing artifact. This file writes exactly one
new JSON artifact and modifies nothing.

WHAT THIS DOES
--------------
Sprint 7 has, until now, only ever measured the risk response at ONE point in
R2's history: the final checkpoint. Phase 4 replicated R2 bit-exactly and saved
all 76 update boundaries (`saved_models/marl/R2_trajectory/R2_trajectory_u000
..u075.pth`), so the same response can now be read as a function of update
number. This script does that and nothing else.

The response is the metric that was already pre-registered and already measured
for A0/A1/A2/A3/R2/R3 in `SPRINT_7_RUNG2_75_matched_states_main.json`:

    Delta_EDGE = pi(MIGRATE_EDGE | risk >= 0.6) - pi(MIGRATE_EDGE | risk < 0.2)

taken over ALL decision entries (`mask.sum(-1) > 1`) of a FIXED state set.

WHY A NEW FILE RATHER THAN AN EDIT
----------------------------------
`_diag_rung2_75_matched_states.py` produced the frozen endpoint artifact that
every Sprint 7 comparison is anchored to. Editing it -- even additively -- would
put the frozen numbers behind changed code. So this file IMPORTS that script's
own helpers (`eval_starts`, `trajectory`, `random_trajectory`, `union_source`,
`probs_at`, `cluster_ci`, `cluster_ci_diff`, `spearman`, `risk_curve`,
`summarise`) and calls them with the unchanged default parameters. The state sets
are therefore constructed by the SAME code, not by a reimplementation. This
mirrors the precedent set by `_diag_R3_action_channels.py`.

STATE SETS ARE NOT REDEFINED
----------------------------
  RANDOM = random_trajectory(cfg_A0, starts, 31337)      policy-independent
  UNION  = union_source(traj(A0_final), traj(R2_final))  built from FINAL models

Both are fixed constants across all 76 checkpoints. UNION is deliberately built
from the FINAL A0 and R2 actors -- exactly as the frozen artifact built it -- and
is NOT rebuilt per checkpoint. Only the actor being scored varies. R2's own
per-checkpoint on-policy state distribution is never used as a state set: if it
were, a change in Delta_EDGE could be a change in which states were visited
rather than a change in the policy's risk conditioning.

SELF-TEST (parity, not a new success criterion)
-----------------------------------------------
u075 is the end of training and must therefore reproduce the frozen R2 row of
`SPRINT_7_RUNG2_75_matched_states_main.json` field for field. Same helper code,
same state sets, same bootstrap seeds (5/6/11), same --boot. A mismatch is
reported as a FAILURE of this script; nothing is adjusted to make it pass.

WHAT IS NEW HERE AND FLAGGED AS SUCH
------------------------------------
The frozen artifact records the EDGE channel only. The four-action channel probe
(`_diag_R3_action_channels.py`) exists as code but its artifact
`SPRINT_7_R3_action_channels.json` was never written, so the CLOUD response is
NOT a pre-existing measurement for any arm. It is computed here (free -- the same
`P` array already holds all four columns) and is labelled
`_NEW_not_in_frozen_artifact` in the output. It is descriptive only.

Usage:
    python -m marl.diag._phase5_risk_trajectory                  # all 76
    python -m marl.diag._phase5_risk_trajectory --only 0,75      # parity check
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (str(_ROOT), str(_ROOT / "marl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from marl._diag_rung0 import (                                    # noqa: E402
    load_agent_and_cfg, OUT_DIR, ACTION_STAY, ACTION_MIGRATE_EDGE)
from marl._diag_rung2_75_matched_states import (                  # noqa: E402
    ACTION_NAMES, MODELS, RISK_BINS,
    eval_starts, trajectory, random_trajectory, union_source,
    probs_at, cluster_ci, cluster_ci_diff, spearman, risk_curve, summarise)
from marl.mappo import MAPPO                                      # noqa: E402

TAG = "SPRINT_7_PHASE5"
TRAJ_DIR = OUT_DIR / "R2_trajectory"
MANIFEST = TRAJ_DIR / "SPRINT_7_P4_trajectory_manifest.jsonl"
FROZEN = OUT_DIR / "SPRINT_7_RUNG2_75_matched_states_main.json"

# per-update quantities already recorded by production training, read from the
# Phase 4 manifest. NOT recomputed here.
RECORDED = ("actor_loss", "critic_loss", "entropy", "approx_kl", "clip_frac",
            "adv_mean", "adv_std", "value_mean", "decision_frac",
            "explained_var")


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--clusters", type=int, default=32,
                   help="UNCHANGED from the frozen artifact")
    p.add_argument("--start-seed", type=int, default=20260825,
                   help="UNCHANGED from the frozen artifact")
    p.add_argument("--boot", type=int, default=5000,
                   help="UNCHANGED from the frozen artifact; parity at u075 "
                        "requires this value")
    p.add_argument("--random-seed", type=int, default=31337,
                   help="UNCHANGED from the frozen artifact")
    p.add_argument("--only", default="",
                   help="comma list of update numbers to score (default: all)")
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


# --------------------------------------------------------------------------
# per-checkpoint scoring on ONE fixed state set
# --------------------------------------------------------------------------
def score(agent, pre, boot):
    """
    All quantities for one actor on one FIXED state set.

    Field names for everything that also exists in the frozen artifact are
    identical to the frozen artifact's, so parity is a field-by-field compare.
    """
    tr = pre["tr"]
    P = probs_at(agent, tr["obs"], tr["mask"])
    hi, dec = pre["hi"], pre["dec"]
    Q, arg = P[hi], P[hi].argmax(-1)
    pe_hi_subset = Q[:, ACTION_MIGRATE_EDGE]

    cb = cluster_ci(pe_hi_subset, pre["eps_hi"], boot, np.random.default_rng(5))
    ab = cluster_ci((arg == ACTION_MIGRATE_EDGE).astype(float),
                    pre["eps_hi"], boot, np.random.default_rng(6))

    r_all = pre["r_all"]
    Pd = P[dec]
    pe_all = Pd[:, ACTION_MIGRATE_EDGE]
    lo_m, hi_m = pre["lo_m"], pre["hi_m"]
    resp = (float(pe_all[hi_m].mean() - pe_all[lo_m].mean())
            if lo_m.any() and hi_m.any() else None)
    respci = cluster_ci_diff(pe_all, hi_m, lo_m, pre["eps_dec"], boot,
                             np.random.default_rng(11))

    cell = dict(
        # ---- fields that exist verbatim in the frozen endpoint artifact ----
        **summarise(Q, arg),
        p_edge_cluster=cb, frac_argmax_edge_cluster=ab,
        risk_curve=risk_curve(tr["risk"], P, dec),
        spearman_risk_vs_p_edge=spearman(r_all, pe_all),
        p_edge_risk_lt_02=float(pe_all[lo_m].mean()) if lo_m.any() else None,
        p_edge_risk_ge_06=float(pe_all[hi_m].mean()) if hi_m.any() else None,
        risk_response_high_minus_low=resp,
        risk_response_cluster=respci,
    )

    # ---- NEW, not present in the frozen artifact. Descriptive only. ----
    argd = Pd.argmax(-1)
    new = dict(
        # all four action channels, same two risk cuts. The frozen artifact
        # records EDGE only; SPRINT_7_R3_action_channels.json does not exist.
        risk_response_by_action={
            ACTION_NAMES[k]: float(Pd[hi_m, k].mean() - Pd[lo_m, k].mean())
            for k in range(4)},
        p_action_risk_lt_02={ACTION_NAMES[k]: float(Pd[lo_m, k].mean())
                             for k in range(4)},
        p_action_risk_ge_06={ACTION_NAMES[k]: float(Pd[hi_m, k].mean())
                             for k in range(4)},
        # greedy (argmax) response, same two cuts, over all decision entries
        argmax_edge_risk_lt_02=float((argd[lo_m] == ACTION_MIGRATE_EDGE).mean()),
        argmax_edge_risk_ge_06=float((argd[hi_m] == ACTION_MIGRATE_EDGE).mean()),
        argmax_response_by_action={
            ACTION_NAMES[k]: float((argd[hi_m] == k).mean()
                                   - (argd[lo_m] == k).mean())
            for k in range(4)},
        # saturation over ALL decision entries of the fixed set (the frozen
        # artifact's entropy/maxp are over the high-risk subset only)
        entropy_mean_all_decision=float(
            -(Pd * np.log(np.clip(Pd, 1e-12, None))).sum(-1).mean()),
        maxp_mean_all_decision=float(Pd.max(-1).mean()),
        frac_maxp_gt_099_all_decision=float((Pd.max(-1) > 0.99).mean()),
        p_edge_mean_all_decision=float(pe_all.mean()),
        p_stay_mean_all_decision=float(Pd[:, ACTION_STAY].mean()),
    )
    new["argmax_edge_response"] = (new["argmax_edge_risk_ge_06"]
                                   - new["argmax_edge_risk_lt_02"])
    cell["_NEW_not_in_frozen_artifact"] = new
    return cell


def precompute(tr):
    """Masks / cluster ids / risk cuts for a fixed state set. Done once."""
    dec = tr["mask"].sum(-1) > 1.0
    hi = (np.minimum((tr["risk"] * 5).astype(int), 4) >= 3) & dec
    epsmat = np.repeat(tr["ep"][:, None], tr["n_agents"], axis=1)
    r_all = tr["risk"][dec]
    return dict(tr=tr, dec=dec, hi=hi,
                eps_hi=epsmat[hi], eps_dec=epsmat[dec],
                r_all=r_all, lo_m=r_all < 0.2, hi_m=r_all >= 0.6)


# --------------------------------------------------------------------------
# descriptive temporal summaries -- NOT causal, NOT thresholds
# --------------------------------------------------------------------------
def frac_of_range_crossings(upd, val, fracs=(0.5, 0.9)):
    """
    First update at which a series has completed a given fraction of its own
    total observed excursion from its first value to its extreme value.

    Purely descriptive shorthand for "when did most of this quantity's movement
    happen". It is NOT a GO/NO-GO threshold and nothing is gated on it.
    """
    v = np.asarray([np.nan if x is None else float(x) for x in val], float)
    u = np.asarray(upd, int)
    ok = ~np.isnan(v)
    if ok.sum() < 3:
        return {f"p{int(f * 100)}": None for f in fracs}
    v, u = v[ok], u[ok]
    v0 = v[0]
    j = int(np.argmax(np.abs(v - v0)))
    span = v[j] - v0
    out = {}
    for f in fracs:
        if abs(span) < 1e-12:
            out[f"p{int(f * 100)}"] = None
            continue
        tgt = v0 + f * span
        reach = (v >= tgt) if span > 0 else (v <= tgt)
        out[f"p{int(f * 100)}"] = int(u[np.argmax(reach)]) if reach.any() else None
    return out


def block_means(upd, val, n_blocks=5):
    v = [np.nan if x is None else float(x) for x in val]
    idx = [i for i, x in enumerate(v) if not np.isnan(x)]
    if not idx:
        return []
    v = np.asarray([v[i] for i in idx]); u = np.asarray([upd[i] for i in idx])
    bounds = np.array_split(np.arange(v.size), n_blocks)
    return [dict(u_lo=int(u[b[0]]), u_hi=int(u[b[-1]]),
                 mean=float(v[b].mean()), sd=float(v[b].std(ddof=1))
                 if b.size > 1 else None) for b in bounds if b.size]


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    print("=" * 78)
    print("SPRINT 7 PHASE 5 -- risk response as a function of update number")
    print("  metric : Delta_EDGE = pi(EDGE | risk>=0.6) - pi(EDGE | risk<0.2)")
    print("  states : RANDOM and UNION, rebuilt with the frozen artifact's")
    print("           own code and unchanged parameters. NOT redefined.")
    print("=" * 78)

    # ---- final models: needed only to construct the frozen state sets ----
    agents, cfgs = {}, {}
    for tag, fn in MODELS.items():
        a, _, c = load_agent_and_cfg(str(OUT_DIR / fn), args.device, "eval")
        agents[tag], cfgs[tag] = a, c
    starts, env0 = eval_starts(cfgs["A0"], args.clusters, args.start_seed)
    print(f"\n  eval window     : [{env0._min_start}, {env0._max_start}]")
    print(f"  distinct starts : {len(starts)}  first {starts[:6]}")

    print("  building fixed state sets (once) ...", flush=True)
    src_A0 = trajectory(agents["A0"], cfgs["A0"], starts)
    src_R2 = trajectory(agents["R2"], cfgs["R2"], starts)
    sources = {
        "RANDOM": random_trajectory(cfgs["A0"], starts, args.random_seed),
        "UNION": union_source(src_A0, src_R2),
    }
    pre = {}
    for s, tr in sources.items():
        pre[s] = precompute(tr)
        print(f"    {s:7s} n_decision {int(pre[s]['dec'].sum()):6d}"
              f"  n_highrisk {int(pre[s]['hi'].sum()):5d}"
              f"  n_lo {int(pre[s]['lo_m'].sum()):6d}"
              f"  n_hi_cut {int(pre[s]['hi_m'].sum()):5d}"
              f"  truncated {tr['truncated']}"
              f"  mean_len {np.mean(tr['lengths']):.1f}")

    # ---- already-recorded per-update training stats (Phase 4 manifest) ----
    man = {}
    with open(MANIFEST, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                man[int(r["update"])] = r
    print(f"\n  manifest records : {len(man)}  "
          f"(u000 has stats=None by construction: it precedes update 1)")

    # ---- which checkpoints ----
    want = (sorted(int(x) for x in args.only.split(",") if x.strip())
            if args.only.strip() else sorted(man))
    print(f"  checkpoints to score : {len(want)}"
          + (f"  {want}" if len(want) <= 12 else f"  u{want[0]:03d}..u{want[-1]:03d}"))

    # ---- the sweep ----
    rows = []
    for n, u in enumerate(want):
        fp = TRAJ_DIR / man[u]["file"]
        agent, _extra = MAPPO.load(str(fp), device=args.device)
        rec = dict(update=u, file=man[u]["file"], md5=man[u]["md5"],
                   episode=man[u]["episode"], lr_scale=man[u]["lr_scale"],
                   recorded_stats=man[u]["stats"], sources={})
        for s in sources:
            rec["sources"][s] = score(agent, pre[s], args.boot)
        rows.append(rec)
        r_ = rec["sources"]["RANDOM"]; v_ = rec["sources"]["UNION"]
        print(f"  u{u:03d} ep{man[u]['episode']:3d}  "
              f"RANDOM d {r_['risk_response_high_minus_low']:+.4f} "
              f"(lo {r_['p_edge_risk_lt_02']:.4f} hi {r_['p_edge_risk_ge_06']:.4f} "
              f"argmaxE_hs {r_['frac_argmax_edge']:.4f})   "
              f"UNION d {v_['risk_response_high_minus_low']:+.4f} "
              f"(lo {v_['p_edge_risk_lt_02']:.4f} hi {v_['p_edge_risk_ge_06']:.4f} "
              f"argmaxE_hs {v_['frac_argmax_edge']:.4f})   "
              f"[{n + 1}/{len(want)} {time.time() - t0:.0f}s]", flush=True)

    # ---------------- parity self-test against the frozen artifact ----------
    print("\n" + "=" * 78)
    print("PARITY SELF-TEST -- u075 must reproduce the frozen R2 endpoint row")
    print("  reference: SPRINT_7_RUNG2_75_matched_states_main.json")
    print("=" * 78)
    parity = {"checked": False, "all_pass": None, "fields": []}
    have75 = [r for r in rows if r["update"] == 75]
    if not FROZEN.exists():
        print("  SKIPPED: frozen artifact not found")
    elif not have75:
        print("  SKIPPED: u075 was not scored in this invocation")
    else:
        fz = json.loads(FROZEN.read_text(encoding="utf-8"))
        mine = have75[0]
        parity["checked"] = True
        ok_all = True
        FIELDS = ["risk_response_high_minus_low", "p_edge_risk_lt_02",
                  "p_edge_risk_ge_06", "spearman_risk_vs_p_edge",
                  "frac_argmax_edge", "entropy_mean", "p_edge_mean",
                  "maxp_mean", "frac_maxp_gt_099", "p_stay_mean"]
        for s in ("RANDOM", "UNION"):
            ref = fz["by_state_source"][s]
            m = mine["sources"][s]
            # state-set shape must match exactly (integers)
            for nm, got, exp in (
                ("n_decision", int(pre[s]["dec"].sum()), ref["n_decision"]),
                ("n_highrisk", int(pre[s]["hi"].sum()), ref["n_highrisk"]),
                ("truncated", sources[s]["truncated"], ref["truncated"]),
            ):
                good = got == exp
                ok_all &= good
                parity["fields"].append(dict(source=s, field=nm, got=got,
                                             expected=exp, delta=None,
                                             exact=good, pass_=good))
                print(f"  {'PASS' if good else 'FAIL'}  {s:6s} {nm:32s} "
                      f"{got} vs {exp}")
            rr = ref["evaluated"]["R2"]
            for f in FIELDS:
                got, exp = m[f], rr[f]
                d = abs(got - exp)
                exact = (got == exp)
                good = d <= 1e-9
                ok_all &= good
                parity["fields"].append(dict(source=s, field=f, got=got,
                                             expected=exp, delta=d,
                                             exact=exact, pass_=good))
                print(f"  {'PASS' if good else 'FAIL'}  {s:6s} {f:32s} "
                      f"{got:+.10f} vs {exp:+.10f}  |d| {d:.3e}"
                      f"{'  (bit-exact)' if exact else ''}")
            # bootstrap reproduction: same seeds, same boot count
            for f in ("delta", "se_cluster", "z"):
                got = m["risk_response_cluster"].get(f)
                exp = rr["risk_response_cluster"].get(f)
                if got is None or exp is None:
                    continue
                d = abs(got - exp)
                good = d <= 1e-9
                ok_all &= good
                parity["fields"].append(dict(source=s,
                                             field=f"risk_response_cluster.{f}",
                                             got=got, expected=exp, delta=d,
                                             exact=(got == exp), pass_=good))
                print(f"  {'PASS' if good else 'FAIL'}  {s:6s} "
                      f"{'risk_response_cluster.' + f:32s} "
                      f"{got:+.10f} vs {exp:+.10f}  |d| {d:.3e}")
        parity["all_pass"] = bool(ok_all)
        print(f"\n  PARITY: {'ALL PASS' if ok_all else 'FAILED'}"
              f"   ({sum(1 for f in parity['fields'] if f['pass_'])}"
              f"/{len(parity['fields'])} fields)")

    # ---------------- descriptive temporal summaries ------------------------
    upd = [r["update"] for r in rows]
    series = {}
    for s in sources:
        for f in ("risk_response_high_minus_low", "p_edge_risk_ge_06",
                  "p_edge_risk_lt_02", "spearman_risk_vs_p_edge",
                  "frac_argmax_edge", "entropy_mean", "maxp_mean",
                  "frac_maxp_gt_099"):
            series[f"{s}.{f}"] = [r["sources"][s][f] for r in rows]
        for f in ("argmax_edge_response", "entropy_mean_all_decision",
                  "frac_maxp_gt_099_all_decision", "p_edge_mean_all_decision"):
            series[f"{s}.{f}"] = [
                r["sources"][s]["_NEW_not_in_frozen_artifact"][f] for r in rows]
        series[f"{s}.risk_response_CLOUD"] = [
            r["sources"][s]["_NEW_not_in_frozen_artifact"]
            ["risk_response_by_action"]["MIGRATE_CLOUD"] for r in rows]
    for f in RECORDED:
        series[f"recorded.{f}"] = [
            (r["recorded_stats"] or {}).get(f) for r in rows]

    temporal = {}
    for k, v in series.items():
        vv = np.asarray([np.nan if x is None else float(x) for x in v], float)
        ok = ~np.isnan(vv)
        uu = np.asarray(upd, int)[ok]
        temporal[k] = dict(
            n=int(ok.sum()),
            first=float(vv[ok][0]) if ok.any() else None,
            last=float(vv[ok][-1]) if ok.any() else None,
            min=float(np.nanmin(vv)) if ok.any() else None,
            max=float(np.nanmax(vv)) if ok.any() else None,
            argmax_update=int(uu[np.argmax(vv[ok])]) if ok.any() else None,
            argmin_update=int(uu[np.argmin(vv[ok])]) if ok.any() else None,
            spearman_vs_update=(spearman(uu.astype(float), vv[ok])
                                if ok.sum() >= 3 else None),
            frac_of_range=frac_of_range_crossings(uu.tolist(), vv[ok].tolist()),
            blocks5=block_means(uu.tolist(), vv[ok].tolist()),
        )

    print("\n" + "=" * 78)
    print("DESCRIPTIVE TEMPORAL SUMMARY  (ordering only -- NOT causal, and the")
    print("  n=75 updates are NOT independent replicates)")
    print("=" * 78)
    print(f"  {'series':46s} {'first':>9s} {'last':>9s} {'max':>9s} "
          f"{'@u':>4s} {'rho_u':>7s} {'p50':>4s} {'p90':>4s}")
    for k, t in temporal.items():
        if t["n"] < 3:
            continue
        print(f"  {k:46s} {t['first']:+9.4f} {t['last']:+9.4f} "
              f"{t['max']:+9.4f} {t['argmax_update']:4d} "
              f"{(t['spearman_vs_update'] if t['spearman_vs_update'] is not None else float('nan')):+7.3f} "
              f"{str(t['frac_of_range']['p50']):>4s} "
              f"{str(t['frac_of_range']['p90']):>4s}")

    # ---------------- persist -----------------------------------------------
    out = dict(
        tag=TAG, variant=args.tag,
        question=("WHEN during R2 training does the risk-conditioned EDGE "
                  "response emerge, strengthen, plateau or disappear, and "
                  "which already-recorded training quantities temporally "
                  "precede that change?"),
        metric=("Delta_EDGE = pi(MIGRATE_EDGE | risk>=0.6) - "
                "pi(MIGRATE_EDGE | risk<0.2), over all decision entries "
                "(mask.sum(-1)>1) of a FIXED state set"),
        params=dict(device=args.device, clusters=args.clusters,
                    start_seed=args.start_seed, boot=args.boot,
                    random_seed=args.random_seed,
                    starts=starts,
                    eval_window=[int(env0._min_start), int(env0._max_start)]),
        provenance=dict(
            state_set_code="marl/_diag_rung2_75_matched_states.py (imported, "
                           "not modified)",
            union_built_from=["mappo_A0_cpu_repro.pth (FINAL)",
                              "mappo_R2_mc_target.pth (FINAL)"],
            union_is_fixed_across_checkpoints=True,
            r2_own_onpolicy_distribution_used=False,
            checkpoints=str(TRAJ_DIR.relative_to(OUT_DIR.parent)),
            manifest=str(MANIFEST.name),
            frozen_reference=str(FROZEN.name),
            recorded_stats_source="Phase 4 manifest (production training logs); "
                                  "not recomputed",
            bootstrap_seeds=dict(p_edge=5, frac_argmax_edge=6,
                                 risk_response_diff=11),
        ),
        state_sets={s: dict(n_decision=int(pre[s]["dec"].sum()),
                            n_highrisk=int(pre[s]["hi"].sum()),
                            n_risk_lt_02=int(pre[s]["lo_m"].sum()),
                            n_risk_ge_06=int(pre[s]["hi_m"].sum()),
                            truncated=sources[s]["truncated"],
                            mean_length=float(np.mean(sources[s]["lengths"])),
                            n_clusters=int(np.unique(sources[s]["ep"]).size),
                            risk_bins=RISK_BINS)
                    for s in sources},
        parity=parity,
        temporal=temporal,
        trajectory=rows,
        elapsed_s=round(time.time() - t0, 1),
    )
    dst = OUT_DIR / f"{TAG}_risk_trajectory_{args.tag}.json"
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n  wrote {dst}")
    print(f"  elapsed {out['elapsed_s']:.0f}s")
    return 0 if (parity["all_pass"] is not False) else 1


if __name__ == "__main__":
    sys.exit(main())
