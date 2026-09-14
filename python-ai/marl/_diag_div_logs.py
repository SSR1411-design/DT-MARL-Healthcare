"""
SPRINT 7 DIVERGENCE DIAGNOSTIC — part 1 of 3: LOG-ONLY R2 vs R3 comparison.

OFFLINE. Reads only artifacts already on disk (`*_updates.csv`,
`*_history.csv`, `*_config.json`). Trains nothing, loads no checkpoint, runs no
environment. Every number here is derived from files written by the original
training runs, so this probe is exactly reproducible from the repository state.

WHY THIS PROBE EXISTS. `SPRINT_7_R3_REPORT.md` asks an unresolved question: R3
got healthier optimisation mechanics (higher estimator SNR, far less sharpening
collapse, zero saturation, higher entropy, more plasticity) yet a significantly
WORSE risk response than R2. This probe looks for quantities that genuinely
DIFFER between the two runs at the rollout/update level, using the identity that
makes the comparison exact:

  * `train.py:156` draws one episode start tick per episode from
    `np.random.default_rng(cfg.train.seed)`. Both arms used seed 20260818, so
    R2's 600 start ticks are the first 600 of R3's 2400. VERIFIED here, asserted.

  * `env.reset(episode_start_tick=...)` does NOT touch `env._rng` (env.py:217-226;
    `_rng` is read only when `random_episode_start` is used, which train.py
    bypasses). Everything else in an episode -- arrivals, deadlines, patient
    specs, the failure schedule read from the trace -- is a deterministic
    function of the start tick. So EPISODE e OF R2 AND EPISODE e OF R3 ARE THE
    SAME ENVIRONMENT REALISATION and differ only in the policy's actions.
    That makes `history.csv` a 600-pair MATCHED experiment, not two samples.

  * `n_updates = episodes // rollout_episodes` (train.py:168) is 75 in both, and
    `frac = 1 - (update_id-1)/n_updates` (train.py:211) depends only on the
    update index, so the LR schedule is identical. VERIFIED here.

  * R3's update v pools episodes 32(v-1)+1..32v, which is exactly the episode
    span of R2's updates 4v-3..4v. So the two runs can be aligned two ways --
    by UPDATE INDEX (matches LR) or by EPISODE INDEX (matches experience) -- and
    they cannot both be matched at once. Both alignments are reported.

READ THE OUTPUT AS DESCRIPTIVE. This is one run per arm, so per-update
differences have no per-arm noise model. Where a "sustained divergence" onset is
reported the rule is stated inline and the result is labelled EXPLORATORY.
"""

import argparse
import csv
import json
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "saved_models" / "marl"

ARMS = {
    "A0": "mappo_A0_cpu_repro",
    "R2": "mappo_R2_mc_target",
    "R3": "R3_batch32",
}

UPD_METRICS = ["mean_reward", "actor_loss", "critic_loss", "entropy",
               "approx_kl", "clip_frac", "adv_std", "value_mean",
               "decision_frac", "lr_scale", "explained_var"]

HIST_METRICS = ["reward", "success_rate", "lost", "critical_lost",
                "relocations", "preemptive", "sla", "protected", "energy",
                "infeasible"]


# ---------------------------------------------------------------- io helpers

def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def load_arm(tag):
    u = _rows(OUT / f"{tag}_updates.csv")
    h = _rows(OUT / f"{tag}_history.csv")
    cfg = json.loads((OUT / f"{tag}_config.json").read_text())
    return {"tag": tag, "upd": u, "hist": h, "cfg": cfg}


def flatten(d, pre=""):
    out = {}
    for k, v in d.items():
        key = f"{pre}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


# ---------------------------------------------------------------- stats

def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs, ddof=1):
    xs = [x for x in xs if not math.isnan(x)]
    n = len(xs)
    if n <= ddof:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - ddof))


def paired_stats(a, b):
    """Paired difference b - a over matched indices. Exact matching required."""
    d = [x - y for x, y in zip(b, a) if not (math.isnan(x) or math.isnan(y))]
    n = len(d)
    m = mean(d)
    s = sd(d)
    se = s / math.sqrt(n) if n > 1 and not math.isnan(s) else float("nan")
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    # sign test, normal approximation on the non-tied pairs
    nt = pos + neg
    z_sign = ((pos - nt / 2) / math.sqrt(nt / 4)) if nt > 0 else float("nan")
    return {"n": n, "mean_diff": m, "sd_diff": s, "se_diff": se,
            "z_paired": (m / se) if se and not math.isnan(se) else float("nan"),
            "n_pos": pos, "n_neg": neg, "n_tie": n - nt, "z_sign": z_sign}


def quartile_means(xs):
    n = len(xs)
    q = [xs[i * n // 4:(i + 1) * n // 4] for i in range(4)]
    return [mean(x) for x in q]


def sustained_onset(diff, ref_sd, k=2.0, run=5):
    """
    EXPLORATORY onset rule, stated so it can be argued with: the first index i
    such that indices i..i+run-1 all have the same sign and all exceed k*ref_sd
    in magnitude. `ref_sd` is supplied by the caller (the sd of the difference
    over an early reference window), which is a crude noise proxy -- with one
    run per arm no honest noise model exists.
    """
    if math.isnan(ref_sd) or ref_sd <= 0:
        return None
    n = len(diff)
    for i in range(n - run + 1):
        w = diff[i:i + run]
        if any(math.isnan(x) for x in w):
            continue
        if all(x > k * ref_sd for x in w) or all(x < -k * ref_sd for x in w):
            return i + 1          # 1-based update/episode index
    return None


# ---------------------------------------------------------------- sections

def s1_config(a, b, res):
    fa, fb = flatten(a["cfg"]), flatten(b["cfg"])
    keys = sorted(set(fa) | set(fb))
    diff = {k: [fa.get(k, "<absent>"), fb.get(k, "<absent>")]
            for k in keys if fa.get(k, "<absent>") != fb.get(k, "<absent>")}
    # Fields that are pure provenance, not experimental configuration.
    provenance = {"wall_time_s", "episodes", "config.train.episodes",
                  "config.train.tag"}
    substantive = {k: v for k, v in diff.items()
                   if k.split(".")[-1] not in {"wall_time_s"} and k not in provenance}
    print(f"  config fields compared      : {len(keys)}")
    print(f"  differing                   : {len(diff)}")
    for k, (x, y) in diff.items():
        print(f"      {k:<44s} {str(x):>22s}  ->  {str(y)}")
    res["config"] = {"n_fields": len(keys), "diff": diff,
                     "n_substantive_excl_provenance": len(substantive)}


def s2_episode_identity(a, b, res):
    ha, hb = a["hist"], b["hist"]
    na = len(ha)
    ta = [r["start_tick"] for r in ha]
    tb = [r["start_tick"] for r in hb]
    nested = ta == tb[:na]
    assert nested, "start-tick nesting FAILED -- the matched-pair premise is void"
    cols = ["start_tick"] + HIST_METRICS
    first_diff = None
    for i in range(na):
        if [ha[i][c] for c in cols] != [hb[i][c] for c in cols]:
            first_diff = i + 1
            break
    print(f"  start ticks nested (R2 == R3[:600])          : {nested}")
    print(f"  identical episode prefix length             : "
          f"{(first_diff - 1) if first_diff else na} episodes")
    print(f"  FIRST divergent episode                     : {first_diff}")
    ra = a["cfg"]["config"]["train"]["rollout_episodes"]
    rb = b["cfg"]["config"]["train"]["rollout_episodes"]
    print(f"  rollout_episodes                            : {ra} vs {rb}")
    print(f"  -> first update lands after episode          : {ra} vs {rb}")
    print(f"  -> so divergence must begin at episode       : {ra + 1}")
    res["episode_identity"] = {
        "nested": nested, "identical_prefix": (first_diff - 1) if first_diff else na,
        "first_divergent_episode": first_diff,
        "rollout_episodes": [ra, rb],
        "predicted_first_divergent_episode": ra + 1,
        "prediction_holds": first_diff == ra + 1,
    }


def s3_lr(a, b, res):
    la = [_f(r["lr_scale"]) for r in a["upd"]]
    lb = [_f(r["lr_scale"]) for r in b["upd"]]
    assert len(la) == len(lb), "update counts differ"
    m = max(abs(x - y) for x, y in zip(la, lb))
    print(f"  updates                     : {len(la)} vs {len(lb)}")
    print(f"  max |lr_scale_a - lr_scale_b|: {m:.2e}")
    print(f"  lr_scale range              : {la[0]:.4f} -> {la[-1]:.4f}")
    res["lr"] = {"n_updates": len(la), "max_abs_lr_diff": m,
                 "lr_first": la[0], "lr_last": la[-1]}


def s4_update_aligned(a, b, res):
    """Align by UPDATE INDEX -- matches the LR schedule, mismatches experience."""
    out = {}
    print(f"  {'metric':<15s} {'A Q1':>9s} {'A Q4':>9s} {'B Q1':>9s} {'B Q4':>9s} "
          f"{'B-A mean':>10s} {'onset':>6s}")
    for k in UPD_METRICS:
        xa = [_f(r[k]) for r in a["upd"]]
        xb = [_f(r[k]) for r in b["upd"]]
        d = [y - x for x, y in zip(xa, xb)]
        ref = sd(d[:10])
        onset = sustained_onset(d, ref)
        qa, qb = quartile_means(xa), quartile_means(xb)
        print(f"  {k:<15s} {qa[0]:>9.4f} {qa[3]:>9.4f} {qb[0]:>9.4f} {qb[3]:>9.4f} "
              f"{mean(d):>10.4f} {str(onset):>6s}")
        out[k] = {"a_quartile_means": qa, "b_quartile_means": qb,
                  "mean_diff": mean(d), "ref_sd_first10": ref,
                  "sustained_onset_update": onset,
                  "a_first": xa[0], "a_last": xa[-1],
                  "b_first": xb[0], "b_last": xb[-1],
                  "diff_per_update": d}
    res["update_aligned"] = out


def s5_experience_aligned(a, b, res):
    """
    Align by EPISODE INDEX. R3's update v spans the same episodes as R2's
    updates 4v-3..4v, so compare R3's update v against the MEAN of that block.
    Only defined for v <= 75/ratio.
    """
    ra = a["cfg"]["config"]["train"]["rollout_episodes"]
    rb = b["cfg"]["config"]["train"]["rollout_episodes"]
    ratio = rb // ra
    assert rb % ra == 0, "non-integer rollout ratio; block alignment undefined"
    nv = len(a["upd"]) // ratio
    out = {"ratio": ratio, "n_blocks": nv, "metrics": {}}
    print(f"  rollout ratio {ratio}x -> B update v <-> mean of A updates "
          f"{ratio}v-{ratio - 1}..{ratio}v, v=1..{nv}")
    print(f"  {'metric':<15s} {'A block mean':>13s} {'B value mean':>13s} {'B-A':>10s}")
    for k in UPD_METRICS:
        xa = [_f(r[k]) for r in a["upd"]]
        xb = [_f(r[k]) for r in b["upd"]]
        ba, bb = [], []
        for v in range(1, nv + 1):
            blk = xa[(v - 1) * ratio: v * ratio]
            ba.append(mean(blk))
            bb.append(xb[v - 1])
        d = [y - x for x, y in zip(ba, bb)]
        print(f"  {k:<15s} {mean(ba):>13.4f} {mean(bb):>13.4f} {mean(d):>10.4f}")
        out["metrics"][k] = {"a_block_means": ba, "b_values": bb,
                             "mean_a": mean(ba), "mean_b": mean(bb),
                             "mean_diff": mean(d)}
    # update density per episode of experience
    print()
    print(f"  updates applied by episode 600 : A={len(a['upd'])}  "
          f"B={600 // rb}  (ratio {len(a['upd']) / max(1, 600 // rb):.2f}x)")
    out["updates_by_episode_600"] = {"a": len(a["upd"]), "b": 600 // rb}
    res["experience_aligned"] = out


def s6_matched_pairs(a, b, res):
    """The exact 600-pair matched comparison. Same env realisation per index."""
    ha, hb = a["hist"], b["hist"]
    n = min(len(ha), len(hb))
    out = {"n_pairs": n, "metrics": {}}
    print(f"  matched pairs (same start tick, same env realisation): {n}")
    print(f"  {'metric':<14s} {'A mean':>10s} {'B mean':>10s} {'B-A':>10s} "
          f"{'z_paired':>9s} {'+/-/=':>13s} {'z_sign':>7s} {'onset ep':>8s}")
    for k in HIST_METRICS:
        xa = [_f(ha[i][k]) for i in range(n)]
        xb = [_f(hb[i][k]) for i in range(n)]
        st = paired_stats(xa, xb)
        d = [y - x for x, y in zip(xa, xb)]
        # reference window: the identical prefix has diff exactly 0, so use the
        # first 50 DIVERGENT episodes as the noise proxy instead.
        pre = res["episode_identity"]["identical_prefix"]
        ref = sd(d[pre:pre + 50])
        onset = sustained_onset(d, ref)
        onset_ep = onset if onset is None else onset
        print(f"  {k:<14s} {mean(xa):>10.4f} {mean(xb):>10.4f} "
              f"{st['mean_diff']:>10.4f} {st['z_paired']:>9.2f} "
              f"{st['n_pos']:>4d}/{st['n_neg']:>4d}/{st['n_tie']:>3d} "
              f"{st['z_sign']:>7.2f} {str(onset_ep):>8s}")
        out["metrics"][k] = dict(st, mean_a=mean(xa), mean_b=mean(xb),
                                 ref_sd=ref, sustained_onset_episode=onset_ep)
    res["matched_pairs"] = out


def s7_adv_pool(a, b, res):
    """
    `adv_std` (mappo.py:439) is the std of the RAW pre-normalisation GAE over
    ALL entries in the buffer. It is the divisor the actor's advantages are
    scaled by (mappo.py:360-366 normalises over decision entries, so the two are
    not identical, but they move together).

    If the extra spread in B's adv_std were pure estimation noise, B's pooled
    estimate over `ratio`x more episodes should be LESS variable across updates
    by 1/sqrt(ratio). If instead the extra spread is real between-episode
    variance entering a wider pool, B's across-update variability will NOT
    shrink that way. adv_std also trends downward over training, so the
    comparison is made on residuals after removing a centred moving-average
    trend (window 9) -- stated so the detrending is not hidden.
    """
    def detrend(xs, w=9):
        n, r = len(xs), []
        for i in range(n):
            lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
            blk = [x for x in xs[lo:hi] if not math.isnan(x)]
            r.append(xs[i] - (sum(blk) / len(blk) if blk else float("nan")))
        return r

    ra = a["cfg"]["config"]["train"]["rollout_episodes"]
    rb = b["cfg"]["config"]["train"]["rollout_episodes"]
    out = {}
    for key in ["adv_std", "decision_frac", "critic_loss", "explained_var",
                "entropy", "approx_kl", "clip_frac"]:
        xa = [_f(r[key]) for r in a["upd"]]
        xb = [_f(r[key]) for r in b["upd"]]
        sa, sb = sd(detrend(xa)), sd(detrend(xb))
        out[key] = {"mean_a": mean(xa), "mean_b": mean(xb),
                    "detrended_sd_a": sa, "detrended_sd_b": sb,
                    "observed_ratio_b_over_a": (sb / sa) if sa else float("nan"),
                    "pure_noise_prediction": 1.0 / math.sqrt(rb / ra)}
        print(f"  {key:<15s} mean {mean(xa):>9.4f} / {mean(xb):>9.4f}   "
              f"detrended sd {sa:>8.4f} / {sb:>8.4f}   "
              f"ratio {out[key]['observed_ratio_b_over_a']:>6.3f} "
              f"(pure-noise prediction {out[key]['pure_noise_prediction']:.3f})")
    res["variability"] = out


def s8_movement_proxies(a, b, res):
    """
    Cumulative distribution-space movement proxies from the logs. `approx_kl` is
    the k1 estimator mean(old_lp - new_lp) over decision entries (mappo.py:426),
    averaged over that update's minibatch steps. It is a per-update, per-step
    average, so a sum over updates is only a proxy for total movement -- it is
    reported as such, not as a KL.
    """
    out = {}
    for name, arm in (("a", a), ("b", b)):
        kl = [_f(r["approx_kl"]) for r in arm["upd"]]
        cf = [_f(r["clip_frac"]) for r in arm["upd"]]
        eng = [i + 1 for i, x in enumerate(cf) if x > 0]
        out[name] = {
            "sum_k1": sum(x for x in kl if not math.isnan(x)),
            "sum_abs_k1": sum(abs(x) for x in kl if not math.isnan(x)),
            "n_negative_k1_updates": sum(1 for x in kl if x < 0),
            "sum_clip_frac": sum(x for x in cf if not math.isnan(x)),
            "n_updates_clip_engaged": len(eng),
            "last_update_clip_engaged": eng[-1] if eng else None,
            "entropy_first": _f(arm["upd"][0]["entropy"]),
            "entropy_last": _f(arm["upd"][-1]["entropy"]),
        }
        print(f"  {arm['tag']:<22s} sum_k1 {out[name]['sum_k1']:>9.4f}  "
              f"sum|k1| {out[name]['sum_abs_k1']:>8.4f}  "
              f"k1<0 on {out[name]['n_negative_k1_updates']:>3d} updates  "
              f"clip engaged {out[name]['n_updates_clip_engaged']:>3d}/75 "
              f"(last {out[name]['last_update_clip_engaged']})  "
              f"entropy {out[name]['entropy_first']:.4f}->{out[name]['entropy_last']:.4f}")
    res["movement_proxies"] = out


def s9_progress_clock(a, b, res):
    """
    If B is simply LESS FAR ALONG THE SAME PATH than A, then B's final state
    should match A's state at some earlier update. Test that by locating, for
    each monotone-ish progress metric, the A-update whose value is closest to
    B's FINAL value. A consistent early index across independent metrics is
    evidence for under-convergence; a scattered one is evidence against.
    """
    out = {}
    print(f"  {'metric':<15s} {'B final':>10s} {'closest A update':>17s} "
          f"{'A value':>10s}")
    for k in ["entropy", "critic_loss", "explained_var", "adv_std",
              "decision_frac", "mean_reward", "value_mean"]:
        xa = [_f(r[k]) for r in a["upd"]]
        bf = _f(b["upd"][-1][k])
        best, bi = None, None
        for i, v in enumerate(xa):
            if math.isnan(v):
                continue
            e = abs(v - bf)
            if best is None or e < best:
                best, bi = e, i + 1
        print(f"  {k:<15s} {bf:>10.4f} {str(bi):>17s} {xa[bi - 1]:>10.4f}")
        out[k] = {"b_final": bf, "closest_a_update": bi,
                  "a_value": xa[bi - 1] if bi else None}
    idx = [v["closest_a_update"] for v in out.values()
           if v["closest_a_update"] is not None]
    print(f"  -> matched A-update indices: {sorted(idx)}  "
          f"median {sorted(idx)[len(idx) // 2]}")
    res["progress_clock"] = {"per_metric": out, "indices": idx,
                             "median_index": sorted(idx)[len(idx) // 2]}


# ---------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--a", default="R2", help="control arm key")
    p.add_argument("--b", default="R3", help="treatment arm key")
    p.add_argument("--tag", default="main")
    args = p.parse_args(argv)

    a, b = load_arm(ARMS[args.a]), load_arm(ARMS[args.b])
    res = {"probe": "DIV-1 logs-only",
           "a": {"key": args.a, "tag": a["tag"]},
           "b": {"key": args.b, "tag": b["tag"]},
           "source_files": {
               "a_updates": str(OUT / f"{a['tag']}_updates.csv"),
               "b_updates": str(OUT / f"{b['tag']}_updates.csv"),
               "a_history": str(OUT / f"{a['tag']}_history.csv"),
               "b_history": str(OUT / f"{b['tag']}_history.csv")}}

    hdr = lambda t: print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)

    hdr(f"S1  CONFIGURATION INVARIANTS  {args.a} vs {args.b}")
    s1_config(a, b, res)

    hdr("S2  EPISODE-LEVEL IDENTITY AND EXACT DIVERGENCE ONSET")
    s2_episode_identity(a, b, res)

    hdr("S3  LR SCHEDULE IDENTITY")
    s3_lr(a, b, res)

    hdr("S4  UPDATE-INDEX-ALIGNED TRAJECTORIES (LR matched, experience NOT)")
    s4_update_aligned(a, b, res)

    hdr("S5  EPISODE-ALIGNED TRAJECTORIES (experience matched, LR NOT)")
    s5_experience_aligned(a, b, res)

    hdr("S6  MATCHED-PAIR BEHAVIOURAL DIVERGENCE (exact, same env per index)")
    s6_matched_pairs(a, b, res)

    hdr("S7  ACROSS-UPDATE VARIABILITY vs THE 1/sqrt(N) PURE-NOISE PREDICTION")
    s7_adv_pool(a, b, res)

    hdr("S8  CUMULATIVE MOVEMENT PROXIES")
    s8_movement_proxies(a, b, res)

    hdr("S9  IS B SIMPLY LESS FAR ALONG A'S PATH?  (progress-clock matching)")
    s9_progress_clock(a, b, res)

    path = OUT / f"SPRINT_7_DIV_logs_{args.tag}.json"
    path.write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
