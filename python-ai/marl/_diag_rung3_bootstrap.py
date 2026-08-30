#!/usr/bin/env python
"""
SPRINT 7 RUNG 3 -- CLUSTER BOOTSTRAP over episodes for the dilution quantities.

This delivers the third estimator-noise measurement pre-registered in
SPRINT_7_RUNG3_PREREGISTRATION.md S9 ("8ep vs 32ep stability of every cosine;
the cardinality-matched null's spread; cluster-bootstrap over the 32 episodes").
The first two come from the two _diag_rung3_dilution runs (--episodes 32 and
--episodes 8) and its null. This file supplies the third.

It answers a question the point estimates cannot: how much of the arm-to-arm
spread in cos(g_hi, g_synth) and cos(g_full, g_lo) is real, and how much is
episode-sampling noise. It does NOT introduce a new success metric -- the
verdict is scored by _diag_rung3_score.py from the point estimates alone, per
the pre-registration. This is a precision statement about those numbers.

WHY THIS IS EXACT AND CHEAP. The actor PG is linear in the advantage, so an
episode's contribution is well defined:

    g_hi = (1/D) * SUM_e  h_e        where h_e = -SUM_{j in e, hi} A_j grad lp_j

A cluster-bootstrap replicate that draws episodes with multiplicities m_e is
therefore exactly  g_hi(m) = (1/D_m) * SUM_e m_e h_e.  Every quantity reported
here is either a cosine or a ratio of norms, and BOTH are invariant to the
scalar 1/D_m -- so the changing denominator cannot bias them, and it drops out.

That leaves only inner products of  SUM_e m_e (.)  vectors, and

    <SUM_e m_e a_e, SUM_f m_f b_f> = m^T (A B^T) m

so the SIX Gram matrices among {h_e, l_e, s_e} -- each only n_eps x n_eps --
determine every replicate exactly. No replicate gradient vector is ever formed,
which is what makes 10,000 replicates cost nothing. Verified against the
main probe: SUM_e h_e / D is asserted to reproduce g_hi to 1e-5.

The per-episode gradients are computed on PER-EPISODE graphs, not by masking the
advantage on the full-buffer graph. Both are mathematically identical, but the
per-episode route costs about ONE full-buffer forward+backward per vector type
in total (the episode lengths sum to T), instead of one per episode.

DISCLOSED APPROXIMATION. The advantage standardisation (subtract mean, divide by
sd over decision entries) is held FIXED at its full-sample value across
replicates. A replicate does not re-standardise on its own resample. This is the
usual fixed-standardisation cluster bootstrap; re-standardising per replicate
would make the contributions non-linear in the resample and destroy the exact
Gram-matrix identity above. GAE itself needs no such caveat -- it is computed
within an episode and so is unaffected by which other episodes are drawn.

OFFLINE ONLY. No training, no production modification, no parameter written.

Writes SPRINT_7_RUNG3_bootstrap_<tag>.json.
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

from marl.config import ACTION_MIGRATE_EDGE                        # noqa: E402
from marl._diag_rung0 import load_agent_and_cfg, OUT_DIR            # noqa: E402
from marl._diag_rung2_5_actor_stall import build_buffer_r2          # noqa: E402
from marl._diag_rung2_75_coherence import match_mass, HI_RISK       # noqa: E402
from marl._diag_rung2_75_matched_states import eval_starts          # noqa: E402
# the recorders and the exact grad/synth constructors this rung already uses --
# imported, not reimplemented, so the two probes cannot drift apart
from marl._diag_rung3_dilution import (                             # noqa: E402
    ARMS, record_own, record_greedy, record_random, _concat,
    cached_grad_maker, build_synth,
)

TAG = "SPRINT_7_RUNG3"


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--boot", type=int, default=10000)
    p.add_argument("--start-seed", type=int, default=20260825)
    p.add_argument("--random-seed", type=int, default=31337)
    p.add_argument("--boot-seed", type=int, default=4242)
    p.add_argument("--tag", default="main")
    return p.parse_args(argv)


def quad(G, M):
    """Row-wise quadratic forms m^T G m for every row m of M."""
    return np.einsum("bi,ij,bj->b", M, G, M)


def ratio_ci(num, den, lo=2.5, hi=97.5):
    """Percentile CI of sqrt(num)/sqrt(den) guarding against zero norms."""
    ok = (num > 0) & (den > 0)
    r = np.sqrt(num[ok]) / np.sqrt(den[ok])
    return (float(np.percentile(r, lo)), float(np.percentile(r, hi)),
            float(r.mean()), int(ok.sum()))


def cos_from(gxy, gxx, gyy):
    ok = (gxx > 0) & (gyy > 0)
    c = np.full(gxy.shape, np.nan)
    c[ok] = gxy[ok] / np.sqrt(gxx[ok] * gyy[ok])
    return c


def summarise(c, label):
    v = c[~np.isnan(c)]
    if v.size == 0:
        print(f"      {label:>22}: n/a")
        return None
    d = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
             ci_lo=float(np.percentile(v, 2.5)),
             ci_hi=float(np.percentile(v, 97.5)),
             frac_positive=float((v > 0).mean()), n_valid=int(v.size))
    print(f"      {label:>22}: {d['mean']:+.5f} +/- {d['sd']:.5f}  "
          f"95% [{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}]  "
          f"P(>0) {d['frac_positive']:.4f}")
    return d


def main(argv=None):
    args = parse_args(argv)
    print("=" * 78)
    print("SPRINT 7 RUNG 3 -- CLUSTER BOOTSTRAP over episodes")
    print("        exact by linearity: every replicate is a quadratic form in")
    print("        the episode-multiplicity vector. OFFLINE ONLY.")
    print("=" * 78)
    print(f"  replicates {args.boot}   OWN batch {args.episodes} episodes")
    print("  cosines and norm RATIOS are invariant to the replicate's 1/D,")
    print("  so the changing denominator cannot bias them.")

    agents, cfg_tr, cfg_ev = {}, {}, {}
    for tag, fn in ARMS.items():
        pth = str(OUT_DIR / fn)
        a, _, c_ev = load_agent_and_cfg(pth, args.device, "eval")
        _, _, c_tr = load_agent_and_cfg(pth, args.device, "train")
        agents[tag], cfg_ev[tag], cfg_tr[tag] = a, c_ev, c_tr

    starts, _ = eval_starts(cfg_ev["A0"], 32, args.start_seed)
    print("\n  rebuilding the same three sources ...")
    rnd_recs, rnd_trunc = record_random(cfg_ev["A0"], starts, args.random_seed)
    a0_recs, a0_trunc = record_greedy(agents["A0"], cfg_ev["A0"], starts)
    r2_recs, r2_trunc = record_greedy(agents["R2"], cfg_ev["R2"], starts)
    neutral = {"RANDOM": (rnd_recs, rnd_trunc),
               "UNION": (a0_recs + r2_recs, a0_trunc + r2_trunc)}

    rng_boot = np.random.default_rng(args.boot_seed)
    out = {}
    for arm in ARMS:
        agent = agents[arm]
        own_recs, own_trunc, _ = record_own(agent, cfg_tr[arm], args.episodes)
        sources = {"OWN": (own_recs, own_trunc)}
        sources.update(neutral)
        out[arm] = {}
        for src, (recs, trunc) in sources.items():
            print(f"\n-- {arm} / {src} "
                  + "-" * max(4, 58 - len(arm) - len(src)))
            buf = build_buffer_r2(agent, recs, trunc)
            T = len(buf)
            risk = _concat(recs, "risk")[:T]
            dev = agent.device
            dec = np.asarray(buf.decision[:T]) > 0.5
            a_np = np.asarray(buf.act[:T])

            # advantage: production convention, standardised over decision
            # entries on the FULL sample and then held fixed (see docstring)
            adv_raw, _ = agent.compute_gae(buf)
            adv = adv_raw.copy()
            adv = (adv - adv[dec].mean()) / (adv[dec].std() + 1e-8)

            hi = (risk > HI_RISK) & dec
            lo = (~(risk > HI_RISK)) & dec
            syn_e = build_synth(hi, a_np, adv, ACTION_MIGRATE_EDGE)
            syn_m, k_e = match_mass(syn_e, adv, dec)
            assert k_e > 0.0, "synth_EDGE has empty support"

            adv_hi = np.where(hi, adv, 0.0).astype(adv.dtype)
            adv_lo = np.where(lo, adv, 0.0).astype(adv.dtype)

            # ---- per-episode contributions on per-episode graphs ----------
            bounds, o = [], 0
            for r in recs:
                n = len(r["act"])
                bounds.append((o, o + n))
                o += n
            assert o == T, f"episode lengths sum to {o}, buffer has {T}"
            n_eps = len(bounds)
            H, L, S = [], [], []
            for (s0, s1) in bounds:
                ob = torch.as_tensor(buf.obs[s0:s1], device=dev)
                ac = torch.as_tensor(buf.act[s0:s1], device=dev)
                mk = torch.as_tensor(buf.mask[s0:s1], device=dev)
                dt = torch.as_tensor(buf.decision[s0:s1], device=dev)
                # The decision mask is applied by cached_grad_maker itself, in
                # the production convention. Its internal 1/denom is then undone
                # by multiplying by that same denom, leaving the UNNORMALISED
                # contribution -SUM_{j in e, dec} A_j grad lp_j. This matters:
                # the denominator must NOT vary across episodes, or the h_e
                # would each carry a different scale and the linearity identity
                # this whole method rests on would be false. `sc` mirrors the
                # maker's clamp(min=1.0) exactly so the two cancel identically.
                # float64 from the start: the Gram matrices are inner products
                # of 233k-dim vectors, where float32 would leave ~1e-5 relative
                # error in every bootstrap cosine.
                Ge = cached_grad_maker(agent, ob, ac, mk, dt)
                sc = max(float(dt.sum()), 1.0)
                H.append((Ge(adv_hi[s0:s1]) * sc).double())
                L.append((Ge(adv_lo[s0:s1]) * sc).double())
                S.append((Ge(syn_m[s0:s1]) * sc).double())
            H = torch.stack(H); L = torch.stack(L); S = torch.stack(S)

            # ---- exactness: do the pieces reassemble the whole gradient? ---
            obs = torch.as_tensor(buf.obs[:T], device=dev)
            act = torch.as_tensor(buf.act[:T], device=dev)
            mask = torch.as_tensor(buf.mask[:T], device=dev)
            dec_t = torch.as_tensor(buf.decision[:T], device=dev)
            G = cached_grad_maker(agent, obs, act, mask, dec_t)
            D = float(dec.sum())
            g_hi_ref, g_lo_ref = G(adv_hi), G(adv_lo)
            e_hi = float(((H.sum(0) / D).float() - g_hi_ref).norm()
                         / g_hi_ref.norm())
            e_lo = float(((L.sum(0) / D).float() - g_lo_ref).norm()
                         / g_lo_ref.norm())
            print(f"   n_eps={n_eps}  T={T}  decision={int(D)}   "
                  f"reassembly err  hi {e_hi:.2e}  lo {e_lo:.2e}")
            assert e_hi < 1e-5 and e_lo < 1e-5, (
                f"per-episode contributions do not reassemble: "
                f"hi {e_hi}, lo {e_lo}")

            # ---- six Gram matrices -> every replicate exactly --------------
            g = {k: v.numpy() for k, v in dict(
                HH=H @ H.T, HL=H @ L.T, HS=H @ S.T,
                LL=L @ L.T, LS=L @ S.T, SS=S @ S.T).items()}
            del H, L, S

            M = np.zeros((args.boot, n_eps))
            for b in range(args.boot):
                pick = rng_boot.integers(0, n_eps, size=n_eps)
                M[b] = np.bincount(pick, minlength=n_eps)
            q = {k: quad(v, M) for k, v in g.items()}
            hh, ll, ss, hl, hs, ls_ = (q["HH"], q["LL"], q["SS"], q["HL"],
                                       q["HS"], q["LS"])
            ff = hh + 2 * hl + ll          # ||g_full||^2
            fl = hl + ll                   # <g_full, g_lo>
            fs = hs + ls_                  # <g_full, g_synth>

            print(f"   BOOTSTRAP ({args.boot} episode resamples)")
            b = dict(
                cos_hi_synth=summarise(cos_from(hs, hh, ss),
                                       "cos(g_hi,g_synth)"),
                cos_full_synth=summarise(cos_from(fs, ff, ss),
                                         "cos(g_full,g_synth)"),
                cos_full_lo=summarise(cos_from(fl, ff, ll),
                                      "cos(g_full,g_lo)"),
                cos_hi_lo=summarise(cos_from(hl, hh, ll), "cos(g_hi,g_lo)"),
            )
            r_hf = ratio_ci(hh, ff)
            r_hl = ratio_ci(hh, ll)
            print(f"      {'||g_hi||/||g_full||':>22}: {r_hf[2]:.5f}  "
                  f"95% [{r_hf[0]:.5f}, {r_hf[1]:.5f}]")
            print(f"      {'||g_hi||/||g_lo||':>22}: {r_hl[2]:.5f}  "
                  f"95% [{r_hl[0]:.5f}, {r_hl[1]:.5f}]")

            out[arm][src] = dict(
                n_episodes=n_eps, T=T, n_decision=int(D),
                reassembly_err=dict(hi=e_hi, lo=e_lo),
                bootstrap=b,
                mass_hi_over_full=dict(mean=r_hf[2], ci_lo=r_hf[0],
                                       ci_hi=r_hf[1], n_valid=r_hf[3]),
                mass_hi_over_lo=dict(mean=r_hl[2], ci_lo=r_hl[0],
                                     ci_hi=r_hl[1], n_valid=r_hl[3]),
            )

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 78)
    print("SUMMARY -- cluster-bootstrap 95% CIs (episode resampling)")
    print("=" * 78)
    print(f"  {'cell':>10} {'cos(g_hi,synth)':>30} {'cos(g_full,g_lo)':>30} "
          f"{'P(cos(hi,s)>0)':>15}")
    for arm in ARMS:
        for src in ("OWN", "RANDOM", "UNION"):
            c = out[arm][src]["bootstrap"]
            a1, a2 = c["cos_hi_synth"], c["cos_full_lo"]
            f1 = (f"{a1['mean']:+.4f} [{a1['ci_lo']:+.4f},{a1['ci_hi']:+.4f}]"
                  if a1 else "n/a")
            f2 = (f"{a2['mean']:+.4f} [{a2['ci_lo']:+.4f},{a2['ci_hi']:+.4f}]"
                  if a2 else "n/a")
            pp = f"{a1['frac_positive']:.4f}" if a1 else "n/a"
            print(f"  {arm + '/' + src:>10} {f1:>30} {f2:>30} {pp:>15}")
    print("\n  A cosine whose CI excludes 0 is a real effect at this sample")
    print("  size; one whose CI spans 0 is episode-sampling noise.")

    blob = dict(
        probe=f"{TAG}_bootstrap",
        status="OFFLINE DIAGNOSTIC. No training, no production modification.",
        role="third estimator-noise measurement pre-registered in "
             "SPRINT_7_RUNG3_PREREGISTRATION.md S9. Does NOT gate the verdict "
             "and introduces no new success metric -- D1-D5 are scored from "
             "the point estimates by _diag_rung3_score.py.",
        preregistration=dict(file="SPRINT_7_RUNG3_PREREGISTRATION.md",
                             md5="972dc323bc4a8d98ca0c5ad3273540ad"),
        method="cluster bootstrap over episodes, exact by linearity of the "
               "actor PG in the advantage: a replicate's gradient is "
               "SUM_e m_e (per-episode contribution), so every reported cosine "
               "and norm ratio is a quadratic form in the multiplicity vector "
               "m via six n_eps x n_eps Gram matrices. No replicate gradient "
               "vector is materialised. Reassembly asserted to 1e-5 per cell.",
        invariance="all reported quantities are cosines or ratios of norms, "
                   "both invariant to the replicate's 1/D_m normaliser, so the "
                   "changing denominator cannot bias them",
        disclosed_approximation="the advantage standardisation is held FIXED at "
                                "its full-sample value across replicates; "
                                "re-standardising per replicate would make the "
                                "contributions non-linear in the resample and "
                                "break the exact Gram identity. GAE needs no "
                                "such caveat, being computed within an episode.",
        measurement=dict(episodes=args.episodes, boot=args.boot,
                         start_seed=args.start_seed,
                         random_seed=args.random_seed,
                         boot_seed=args.boot_seed, device=args.device),
        per_arm=out,
    )
    p = OUT_DIR / f"{TAG}_bootstrap_{args.tag}.json"
    p.write_text(json.dumps(blob, indent=2, default=str))
    print(f"\n  wrote {p}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
