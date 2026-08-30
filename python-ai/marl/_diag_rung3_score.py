#!/usr/bin/env python
"""
SPRINT 7 RUNG 3 -- mechanical scorer for the pre-registered criteria D1-D5.

Reads SPRINT_7_RUNG3_dilution_<tag>.json and applies the thresholds exactly as
written in SPRINT_7_RUNG3_PREREGISTRATION.md (md5
972dc323bc4a8d98ca0c5ad3273540ad) S7-S8. Every threshold below is transcribed
from that file; none is computed from the data.

The point of a separate scorer is that the verdict is produced by code, from the
JSON, with the thresholds hard-coded -- so applying the rule cannot drift toward
the result. It prints the per-cell pass/fail grid alongside each verdict so the
scoring is auditable rather than asserted.

Reports only. Writes SPRINT_7_RUNG3_verdict_<tag>.json.
"""

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[0]
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from marl._diag_rung0 import OUT_DIR                              # noqa: E402

ARMS = ["A0", "R2", "R3"]
SOURCES = ["OWN", "RANDOM", "UNION"]

# ---- thresholds, transcribed verbatim from the pre-registration -----------
T_D1A_MIN_COS = 0.0        # S7 D1a: cos(g_hi,g_synth) > 0, all 9 cells
T_D1B_ALPHA = 0.05         # S7 D1b: one-sided p < 0.05
T_D1B_MIN_CELLS = 7        # S7 D1b: at least 7 of 9
T_D2A_MAX_FRAC = 0.10      # S7 D2a: n_hi/n_dec < 0.10, all 9 cells
T_D2B_MAX_RATIO = 0.50     # S7 D2b: ||g_hi||/||g_full|| < 0.50
T_D2B_MIN_CELLS = 7        # S7 D2b: at least 7 of 9
T_D2C_MIN_COS = 0.90       # S7 D2c: cos(g_full,g_lo) > 0.90
T_D2C_MIN_CELLS = 7        # S7 D2c: at least 7 of 9
T_D3_MAX_COS = -0.10       # S7 D3: cos(g_hi,g_lo) < -0.10
T_D3_MIN_CELLS = 5         # S7 D3: at least 5 of 9
T_D5_MATERIAL = 0.90       # S7 D5: "materially change" = cos(g(w),g(1)) < 0.90
T_D5_BETTER = 0.10         # S7 D5: "align better" = +0.10 over cos(g(1),synth)
D4_ORDER = ["R2", "R3", "A0"]   # S7 D4: must reproduce R2 > R3 > A0 on OWN


def cells(per_arm):
    for arm in ARMS:
        for src in SOURCES:
            yield arm, src, per_arm[arm][src]


def grid(per_arm, fn):
    """{(arm,src): (value, bool)} for a predicate returning (value, ok)."""
    return {(a, s): fn(c) for a, s, c in cells(per_arm)}


def show(name, g, fmt="{:+.5f}"):
    n = sum(1 for v, ok in g.values() if ok)
    print(f"  {name}: {n}/9 cells")
    for arm in ARMS:
        row = []
        for src in SOURCES:
            v, ok = g[(arm, src)]
            vs = "n/a" if v is None else fmt.format(v)
            row.append(f"{src[:3]} {vs} {'PASS' if ok else 'fail'}")
        print(f"     {arm}: " + " | ".join(row))
    return n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    args = ap.parse_args(argv)

    src_path = OUT_DIR / f"SPRINT_7_RUNG3_dilution_{args.tag}.json"
    blob = json.loads(src_path.read_text())
    per = blob["per_arm"]
    print("=" * 78)
    print(f"SPRINT 7 RUNG 3 -- SCORING {src_path.name}")
    print("=" * 78)
    print(f"  prereg {blob['preregistration']['file']} "
          f"md5 {blob['preregistration']['md5']}")
    print(f"  OWN batch {blob['measurement']['episodes']} episodes, "
          f"null draws {blob['measurement']['perms']}")

    # ---------------------------------------------------------------- D1
    print("\n" + "-" * 78)
    print("D1 -- hypothesis (A): does g_hi CONTAIN useful directional signal?")
    print("-" * 78)
    g1a = grid(per, lambda c: (c["cosines"]["hi_vs_synth"],
                              c["cosines"]["hi_vs_synth"] is not None
                              and c["cosines"]["hi_vs_synth"] > T_D1A_MIN_COS))
    n1a = show(f"D1a cos(g_hi,g_synth) > {T_D1A_MIN_COS} (need 9/9)", g1a)
    d1a = n1a == 9

    g1b = grid(per, lambda c: (c["null_cardinality_matched"]["p_one_sided_cos"],
                              c["null_cardinality_matched"]["p_one_sided_cos"]
                              < T_D1B_ALPHA))
    n1b = show(f"D1b vs cardinality-matched null, p < {T_D1B_ALPHA} "
               f"(need {T_D1B_MIN_CELLS}/9)", g1b, "{:.4f}")
    d1b = n1b >= T_D1B_MIN_CELLS
    d1 = d1a and d1b
    print(f"\n  D1a {'PASS' if d1a else 'FAIL'}   D1b {'PASS' if d1b else 'FAIL'}"
          f"   ==> D1 {'PASS' if d1 else 'FAIL'}")

    # ---------------------------------------------------------------- D2
    print("\n" + "-" * 78)
    print("D2 -- hypothesis (B): is the update DOMINATED by low-risk mass?")
    print("     (D2 pass = g_hi does NOT control the update)")
    print("-" * 78)
    g2a = grid(per, lambda c: (c["highrisk_frac_of_decision"],
                              c["highrisk_frac_of_decision"] < T_D2A_MAX_FRAC))
    n2a = show(f"D2a n_hi/n_dec < {T_D2A_MAX_FRAC} (need 9/9)", g2a, "{:.4f}")
    d2a = n2a == 9

    g2b = grid(per, lambda c: (c["mass"]["hi_over_full"],
                              c["mass"]["hi_over_full"] < T_D2B_MAX_RATIO))
    n2b = show(f"D2b ||g_hi||/||g_full|| < {T_D2B_MAX_RATIO} "
               f"(need {T_D2B_MIN_CELLS}/9)", g2b, "{:.4f}")
    d2b = n2b >= T_D2B_MIN_CELLS

    g2c = grid(per, lambda c: (c["cosines"]["full_vs_lo"],
                              c["cosines"]["full_vs_lo"] is not None
                              and c["cosines"]["full_vs_lo"] > T_D2C_MIN_COS))
    n2c = show(f"D2c cos(g_full,g_lo) > {T_D2C_MIN_COS} "
               f"(need {T_D2C_MIN_CELLS}/9)  <- OPERATIVE", g2c)
    d2c = n2c >= T_D2C_MIN_CELLS
    d2 = d2a and d2b and d2c
    print(f"\n  D2a {'PASS' if d2a else 'FAIL'}   D2b {'PASS' if d2b else 'FAIL'}"
          f"   D2c {'PASS' if d2c else 'FAIL'}   ==> D2 "
          f"{'PASS' if d2 else 'FAIL'}")
    if not d2c:
        print("  D2c failed: g_hi ALREADY materially controls the update, so")
        print("  dilution is not the mechanism regardless of D2a/D2b.")

    # ---------------------------------------------------------------- D3
    print("\n" + "-" * 78)
    print("D3 -- COMPETING-GRADIENT interference (distinct from dilution)")
    print("-" * 78)
    g3 = grid(per, lambda c: (c["cosines"]["hi_vs_lo"],
                             c["cosines"]["hi_vs_lo"] is not None
                             and c["cosines"]["hi_vs_lo"] < T_D3_MAX_COS))
    n3 = show(f"D3 cos(g_hi,g_lo) < {T_D3_MAX_COS} (need {T_D3_MIN_CELLS}/9)",
              g3)
    d3 = n3 >= T_D3_MIN_CELLS
    print(f"\n  ==> D3 {'PASS' if d3 else 'FAIL'}"
          f"{'  (low-risk gradient ACTIVELY OPPOSES high-risk)' if d3 else ''}")

    # ---------------------------------------------------------------- D4
    print("\n" + "-" * 78)
    print("D4 -- does the decomposition EXPLAIN the A0/R2/R3 ordering?")
    print(f"     behavioural fact to reproduce on OWN: "
          f"{' > '.join(D4_ORDER)}")
    print(f"     UNION Delta: " + "  ".join(
        f"{k} {v:+.4f}" for k, v in blob["union_delta_to_explain"].items()))
    print("-" * 78)
    d4_detail = {}
    for name, get in (("||g_hi||/||g_lo||", lambda c: c["mass"]["hi_over_lo"]),
                      ("||g_hi||/||g_full||",
                       lambda c: c["mass"]["hi_over_full"]),
                      ("cos(g_full,g_synth)",
                       lambda c: c["cosines"]["full_vs_synth"])):
        vals = {a: get(per[a]["OWN"]) for a in ARMS}
        ok = all(v is not None for v in vals.values()) and (
            vals[D4_ORDER[0]] > vals[D4_ORDER[1]] > vals[D4_ORDER[2]])
        order = " > ".join(sorted(ARMS, key=lambda a: -vals[a]))
        d4_detail[name] = dict(values=vals, observed_order=order,
                               reproduces=bool(ok))
        print(f"  {name:>22}: " + "  ".join(f"{a} {vals[a]:+.4f}"
                                            for a in ARMS)
              + f"   -> {order}   {'PASS' if ok else 'fail'}")
    d4 = any(v["reproduces"] for v in d4_detail.values())
    print(f"\n  ==> D4 {'PASS' if d4 else 'FAIL'}")
    if not d4:
        print("  No measured quantity reproduces the behavioural ordering, so")
        print("  the decomposition does not account for the R3 degradation.")

    # ---------------------------------------------------------------- D5
    print("\n" + "-" * 78)
    print("D5 -- analytical sensitivity (DESCRIPTIVE; does NOT gate the verdict)")
    print(f"     'materially change' = cos(g(w),g(1)) < {T_D5_MATERIAL}")
    print(f"     'align better'      = cos(g(w),synth) > cos(g(1),synth) "
          f"+ {T_D5_BETTER}")
    print("-" * 78)
    weights = [str(w) for w in blob["measurement"]["weights"]]
    d5 = {}
    print(f"  {'cell':>10} {'w*':>9} " + " ".join(f"{'w=' + w:>9}"
                                                 for w in weights)
          + "   (cos vs synth; base at w=1)")
    for arm, src, c in cells(per):
        s = c["sensitivity"]
        base = s["1"]["cos_vs_synth"]
        mat = [w for w in weights if s[w]["cos_vs_baseline"] is not None
               and s[w]["cos_vs_baseline"] < T_D5_MATERIAL]
        bet = [w for w in weights if s[w]["cos_vs_synth"] is not None
               and base is not None
               and s[w]["cos_vs_synth"] > base + T_D5_BETTER]
        ws = c["sensitivity_w_star"]
        d5[f"{arm}/{src}"] = dict(
            w_star=ws, base_cos=base,
            materially_changes_at=mat, aligns_better_at=bet,
            cos_vs_synth={w: s[w]["cos_vs_synth"] for w in weights},
            cos_vs_baseline={w: s[w]["cos_vs_baseline"] for w in weights},
            norm_ratio={w: s[w]["norm_ratio"] for w in weights})
        wss = "n/a" if ws is None else f"{ws:+.3f}"
        print(f"  {arm + '/' + src:>10} {wss:>9} "
              + " ".join(f"{s[w]['cos_vs_synth']:>+9.4f}" for w in weights)
              + f"   material@{mat or '-'} better@{bet or '-'}")
    n_better16 = sum(1 for v in d5.values() if "16" in v["aligns_better_at"])
    n_sign = sum(1 for v in d5.values()
                 if v["w_star"] is not None and 0 < v["w_star"] <= 16)
    print(f"\n  cells where w=16 aligns better by >{T_D5_BETTER}: "
          f"{n_better16}/9")
    print(f"  cells whose alignment numerator flips sign at some w in (0,16]: "
          f"{n_sign}/9")

    # ---------------------------------------------------------------- verdict
    print("\n" + "=" * 78)
    if not d1:
        verdict = "FALSIFIED"
        why = ("the high-risk gradient carries no reliable directional signal "
               "(D1 failed)")
    elif d3:
        verdict = "COMPETING-GRADIENT INTERFERENCE"
        why = ("g_hi is aligned but g_lo actively OPPOSES it; this is reported "
               "separately and is NOT simple dilution")
    elif d2:
        verdict = "SUPPORTED (simple dilution)"
        why = ("g_hi is aligned and beats its null, yet is small and the "
               "direction PPO follows is set by the low-risk bulk")
    else:
        verdict = "NOT SUPPORTED"
        why = ("g_hi already materially controls the update, so dilution "
               "cannot be the constraint")
    explains = d4
    if not explains:
        verdict += " -- DOES NOT EXPLAIN R3"
        why += ("; and no measured quantity reproduces the observed "
                "A0/R2/R3 ordering, so this mechanism does not account for "
                "the degradation it was invoked to explain")
    go = d1 and d2 and d4
    print(f"VERDICT: {verdict}")
    print(f"  because {why}")
    print(f"\nGO for a future TRAINING rung requires D1 AND D2 AND D4:")
    print(f"  D1 {'PASS' if d1 else 'FAIL'}  D2 {'PASS' if d2 else 'FAIL'}  "
          f"D4 {'PASS' if d4 else 'FAIL'}   ==> "
          f"{'GO' if go else 'NO-GO for a training rung'}")
    if not go:
        print("  D5 magnitudes may inform a future design but cannot create a "
              "GO (prereg S7 D5).")
    print("=" * 78)

    out = dict(
        scored_file=src_path.name,
        preregistration=blob["preregistration"],
        measurement=blob["measurement"],
        thresholds=dict(
            D1a_min_cos=T_D1A_MIN_COS, D1b_alpha=T_D1B_ALPHA,
            D1b_min_cells=T_D1B_MIN_CELLS, D2a_max_frac=T_D2A_MAX_FRAC,
            D2b_max_ratio=T_D2B_MAX_RATIO, D2b_min_cells=T_D2B_MIN_CELLS,
            D2c_min_cos=T_D2C_MIN_COS, D2c_min_cells=T_D2C_MIN_CELLS,
            D3_max_cos=T_D3_MAX_COS, D3_min_cells=T_D3_MIN_CELLS,
            D5_material=T_D5_MATERIAL, D5_better=T_D5_BETTER,
            D4_order=D4_ORDER),
        criteria=dict(
            D1=dict(passed=bool(d1), D1a=dict(passed=bool(d1a), cells=n1a,
                                              need=9),
                    D1b=dict(passed=bool(d1b), cells=n1b,
                             need=T_D1B_MIN_CELLS)),
            D2=dict(passed=bool(d2),
                    D2a=dict(passed=bool(d2a), cells=n2a, need=9),
                    D2b=dict(passed=bool(d2b), cells=n2b,
                             need=T_D2B_MIN_CELLS),
                    D2c=dict(passed=bool(d2c), cells=n2c,
                             need=T_D2C_MIN_CELLS, operative=True)),
            D3=dict(passed=bool(d3), cells=n3, need=T_D3_MIN_CELLS),
            D4=dict(passed=bool(d4), detail=d4_detail),
            D5=dict(gates_verdict=False, detail=d5,
                    n_cells_better_at_w16=n_better16,
                    n_cells_sign_flip_within_16=n_sign)),
        verdict=verdict,
        verdict_reason=why,
        explains_R3=bool(explains),
        go_for_training_rung=bool(go),
    )
    p = OUT_DIR / f"SPRINT_7_RUNG3_verdict_{args.tag}.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
