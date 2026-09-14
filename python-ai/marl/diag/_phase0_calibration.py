"""Sprint 7 PHASE 0/1 -- reconstruction audit. READ-ONLY.

Two questions, both answered from frozen artifacts. No training, no production
edit, no checkpoint written, no state set redefined.

Q1 (RULE 9). Every Sprint 7 mechanism instrument is scored against the
    pre-registered behavioural metric  Delta = pi(EDGE|risk>=0.6) - pi(EDGE|risk<0.2)
    on the fixed policy-independent sets. An instrument that orders the arms
    backwards cannot be used to choose between mechanisms.

Q2. The per-state advantage offset c(s) := gae(s, a_ref), whose variance
    decomposition motivated R3, is characterised on R2's own native deviation
    rows: is it shared across agents at a tick (=> an implementable online
    centring exists) or agent-idiosyncratic (=> it does not)?
"""
import json, statistics as st
from collections import defaultdict
from pathlib import Path

M = Path(__file__).resolve().parents[2] / "saved_models" / "marl"


def spearman(x, y):
    def rk(v):
        s = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for j, i in enumerate(s): r[i] = j + 1
        return r
    a, b = rk(x), rk(y); n = len(x); ma = sum(a) / n; mb = sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = sum((a[i] - ma) ** 2 for i in range(n)) ** .5
    db = sum((b[i] - mb) ** 2 for i in range(n)) ** .5
    return num / (da * db)


def jload(name):
    with open(M / name) as f:
        return json.load(f)


# ---- Q1 -------------------------------------------------------------------
beh = {s: jload("SPRINT_7_DIV_geometry_eval.json")["part_b_function_space"]
       ["results"][s]["_reproduced_delta"] for s in ("RANDOM", "UNION")}
sh = jload("SPRINT_7_DIV_shape_main.json")["results"]
dil = jload("SPRINT_7_RUNG3_dilution_main.json")["per_arm"]
d4 = jload("SPRINT_7_RUNG3_verdict_main.json")["criteria"]["D4"]["detail"]

instr = []
for s in ("RANDOM", "UNION"):
    e1 = sh[s]["E1_bootstrap"]
    instr += [
        ("E1 proj_frac hi-lo", s, {a: e1[a]["hi_minus_lo"]["point"] for a in ("A0", "R3")}),
        ("E1 proj_frac at hi", s, {a: e1[a]["hi"]["proj_frac_point"] for a in ("A0", "R3")}),
        ("cos(g_full,g_synth)", s, {a: dil[a][s]["cosines"]["full_vs_synth"] for a in ("A0", "R2", "R3")}),
        ("cos(g_hi,g_lo)", s, {a: dil[a][s]["cosines"]["hi_vs_lo"] for a in ("A0", "R2", "R3")}),
    ]
instr += [("||g_hi||/||g_lo||", "OWN", d4["||g_hi||/||g_lo||"]["values"]),
          ("cos(g_full,g_synth)", "OWN", d4["cos(g_full,g_synth)"]["values"])]

cal, rhos = [], []
for name, src, vals in instr:
    arms = [a for a in ("A0", "R2", "R3") if a in vals]
    x = [vals[a] for a in arms]
    row = {"instrument": name, "measured_on": src, "arms": arms,
           "values": {a: vals[a] for a in arms}}
    for bs in ("RANDOM", "UNION"):
        r = spearman(x, [beh[bs][a] for a in arms])
        row[f"spearman_vs_behaviour_{bs}"] = r
        rhos.append(r)
    cal.append(row)

# ---- Q2 -------------------------------------------------------------------
rows = jload("SPRINT_7_RUNG2_5_native_dev_R2.json")["rows"]
c_of = lambda r: r["gae"]["C0"][str(r["ref_action"])]

grp = defaultdict(list)
for r in rows:
    grp[(r["start"], r["step"])].append(r)
multi = [v for v in grp.values() if len(v) > 1]
spread = [max(c_of(r) for r in v) - min(c_of(r) for r in v) for v in multi]
allc = [c_of(r) for r in rows]

per_bucket, flips = {}, {}
for b in ("lo", "mid", "hi"):
    cs = [c_of(r) for r in rows if r["bucket"] == b]
    if cs:
        per_bucket[b] = {"n": len(cs), "mean_c": st.mean(cs), "sd_c": st.pstdev(cs),
                         "frac_abs_c_gt_1": sum(1 for v in cs if abs(v) > 1) / len(cs)}
    nf = tot = 0; ratio = []
    for r in rows:
        if r["bucket"] != b: continue
        c = c_of(r)
        for a in r["a_true_own"]:
            if int(a) == r["ref_action"]: continue
            raw = r["gae"]["C0"][a]; paired = raw - c
            tot += 1
            nf += (raw > 0) != (paired > 0)
            if abs(paired) > 1e-9: ratio.append(abs(c) / abs(paired))
    if tot:
        flips[b] = {"deviation_pairs": tot, "sign_flips": nf, "flip_rate": nf / tot,
                    "median_abs_c_over_abs_paired": st.median(ratio)}

out = {
    "probe": "SPRINT_7_PHASE0_calibration",
    "what": "read-only reconstruction audit of Sprint 7; no training, no production edit",
    "Q1_rule9_calibration": {
        "behavioural_metric": "Delta = pi(EDGE|risk>=0.6) - pi(EDGE|risk<0.2), fixed policy-independent sets",
        "behaviour": beh,
        "behaviour_order": "R2 > R3 > A0 on both sets",
        "per_instrument": cal,
        "n_readings": len(rhos),
        "n_negative": sum(1 for r in rhos if r < 0),
        "n_positive": sum(1 for r in rhos if r > 0),
        "mean_spearman": sum(rhos) / len(rhos),
        "caveat": "n=2 or n=3 arms per instrument, so no single rho is significant "
                  "(permutation p >= 1/6 at n=3). The readings share arms and all derive "
                  "from terminal checkpoints, so they are NOT independent and the sign "
                  "count must not be read as p = 2^-20.",
    },
    "Q2_per_state_offset": {
        "definition": "c(s) := gae(s, a_ref); a_true(s,a_ref) == 0 by construction, so c is pure estimator error",
        "source": "SPRINT_7_RUNG2_5_native_dev_R2.json rows (R2's own greedy baselines)",
        "n_rows": len(rows),
        "shared_across_agents_at_same_tick": {
            "n_groups_total": len(grp),
            "n_groups_with_2plus_agents": len(multi),
            "mean_within_group_spread_of_c": st.mean(spread) if spread else None,
            "max_within_group_spread_of_c": max(spread) if spread else None,
            "overall_sd_of_c": st.pstdev(allc),
            "verdict": "agent-idiosyncratic: within-tick spread EXCEEDS the overall sd, "
                       "so cross-agent or per-tick centring would not remove c(s)",
        },
        "per_bucket": per_bucket,
        "sign_flips_raw_vs_paired": flips,
    },
}
p = M / "SPRINT_7_PHASE0_calibration.json"
with open(p, "w") as f:
    json.dump(out, f, indent=1)
print(f"wrote {p}")
print(f"Q1: {out['Q1_rule9_calibration']['n_readings']} readings, "
      f"{out['Q1_rule9_calibration']['n_negative']} negative, "
      f"mean rho {out['Q1_rule9_calibration']['mean_spearman']:+.3f}")
print(f"Q2: within-tick spread {out['Q2_per_state_offset']['shared_across_agents_at_same_tick']['mean_within_group_spread_of_c']:.4f} "
      f"vs overall sd {out['Q2_per_state_offset']['shared_across_agents_at_same_tick']['overall_sd_of_c']:.4f}")
for b, v in flips.items():
    print(f"    {b}: flip_rate {v['flip_rate']:.3f}  median |c|/|paired| {v['median_abs_c_over_abs_paired']:.2f}")
