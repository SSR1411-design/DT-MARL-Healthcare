# SPRINT 7 RUNG 3 (DILUTION) — DIAGNOSTIC REPORT

**Offline diagnostic only.** No training run of any kind was performed. No production
module was modified. No weighting was applied to any real update. No R4 was designed.

**Pre-registration:** `SPRINT_7_RUNG3_PREREGISTRATION.md`, md5
`972dc323bc4a8d98ca0c5ad3273540ad` — written to disk before the probe was written or
run, and **verified unchanged** after all measurements (§2).

**Verdict, produced mechanically by `marl/_diag_rung3_score.py` from the JSON with the
thresholds hard-coded:**

> ## SUPPORTED (simple dilution) — DOES NOT EXPLAIN R3
> ## ⇒ NO-GO for a training rung
>
> `GO = D1 ∧ D2 ∧ D4` → **D1 PASS, D2 PASS, D4 FAIL**.

Hypothesis **(A)** — *the high-risk gradient contains a useful directional signal* —
passes on the pre-registered point estimates. Hypothesis **(B)** — *the high-risk
gradient is large enough to materially control the full PPO update* — **fails**: the
direction PPO follows is set by the ~95% low-risk remainder. That is the dilution
picture. But **D4, the honesty check, fails**: no measured quantity reproduces the
behavioural ordering R2 > R3 > A0, so this mechanism is present in all arms roughly
alike and **does not account for the R3 degradation it was invoked to explain**.

Two supplementary measurements pre-registered in §9 (the 8-vs-32-episode stability
check and the cluster bootstrap) additionally show that **the D1 half of that verdict is
not robust**, while the D2 half is. See §8.2. This is reported as a qualification of the
pre-registered result, not as a change to it.

---

## 1. What was measured, and the exact identity it rests on

The production actor policy gradient at a checkpoint's own parameters is a sum over
decision entries (entropy term excluded, being advantage-independent):

```
g_full = -(1/D) · Σ_{j ∈ decision}  A_j · ∇ log π(a_j | s_j)
```

Partitioning the decision set by the production high-risk rule (`risk > 0.50`) gives,
because `g` is **linear** in the per-entry contributions:

```
g_full = g_hi + g_lo        EXACTLY, with the SAME 1/D normaliser in both terms
```

This holds regardless of the PPO ratio. It was asserted numerically in every cell:

| check | result, all 9 cells |
|---|---|
| `‖g_full − g_ref‖/‖g_ref‖` vs the existing production-convention `flat_pg_grad` | **0.00e+00** (bit-identical) |
| additivity `‖g_full − (g_hi + g_lo)‖/‖g_full‖` | ≤ **2.68e-07** (tolerance 1e-5) |

The flattened actor gradient has dimension 10 × 23,300 = **233,000**.

**Reference direction** `synth` (unchanged from Rung 2.75): `+1` on MIGRATE_EDGE, `−1`
on STAY, at high-risk decision entries, zero elsewhere, rescaled to `g_full`'s L2
advantage mass over decision entries. It is a reference *direction*, not a target — it
presupposes that MIGRATE_EDGE is the correct high-risk action, which a training run
cannot know. §8.4 shows this presupposition is doing less work than intended.

**Scaling convention.** `g_full`, `g_hi`, `g_lo` are left on their **natural** scale
because *mass is the quantity under test*; only `synth` is mass-matched, per the
Rung 2.75 convention. This does not break comparability: Rung 2.75's `g_hi` came from a
mass-matched `adv_hi_m = k_hi · adv_hi` with scalar `k_hi > 0`, and
`cos(k·g_hi, g_synth) ≡ cos(g_hi, g_synth)`. Only the norm differs, and Rung 2.75 never
reported the natural-scale norm.

**Reuse, not reimplementation.** State sets come from `_diag_rung2_75_matched_states`'
own `eval_starts` / `random_trajectory` / `trajectory` / `union_source`; the buffer from
`_diag_rung2_5_actor_stall.build_buffer_r2`; the gradient and mass-matching from
`_diag_rung2_75_coherence`. All imported unmodified. Every recorded rollout was asserted
**bit-identical** to the existing constructor on `obs`, `mask` and `risk`:

```
       RANDOM: obs IDENTICAL  mask IDENTICAL  risk IDENTICAL
    A0-greedy: obs IDENTICAL  mask IDENTICAL  risk IDENTICAL
    R2-greedy: obs IDENTICAL  mask IDENTICAL  risk IDENTICAL
        UNION: obs IDENTICAL  mask IDENTICAL  risk IDENTICAL   (32 + 32 episodes)
```

No evaluation definition was changed. The two coexisting high-risk rules in the existing
machinery — `risk > 0.50` in the gradient probe, `min(int(risk·5),4) ≥ 3` for
state-set scoring — were **not reconciled**; each is used exactly where it already was,
and the report records what the other rule would have given (`n_highrisk_bucket_rule`:
848 vs 850 on A0/OWN, i.e. immaterial).

### Independent validation of the whole probe

The 32-episode OWN cosines reproduce the values disclosed in advance in prereg §6 — from
a separately written probe with a different scaling convention:

| quantity, OWN 32 ep | A0 | R2 | R3 |
|---|---|---|---|
| `cos(g_hi, synth)` measured here | +0.91298 | +0.53639 | +0.45051 |
| prereg §6 disclosed prior (Rung 2.75 probe) | +0.913 | +0.536 | +0.451 |
| `cos(g_full, synth)` measured here | +0.40228 | −0.23723 | −0.24371 |
| prereg §6 disclosed prior | +0.402 | −0.237 | −0.244 |

RANDOM and UNION cells were additionally **bit-identical** between the smoke run and the
main run, as they must be, since neither depends on `--episodes`.

### The three state sources, and the disclosed limitation

| source | actions from | ratio at own params | window | role |
|---|---|---|---|---|
| OWN | the arm's own stochastic samples | **exactly 1** | train | production-faithful; the only source where the vanilla-PG identity is exact |
| RANDOM | uniform-legal, seed 31337 | not 1 | eval | policy-independent; isolates state selection |
| UNION | A0-greedy pooled with R2-greedy | not 1 | eval | on-manifold, symmetric, contains neither arm's R3-specific selection |

**Counterfactual caveat, declared in advance (prereg §5):** on RANDOM and UNION the
actions come from a foreign behaviour policy, so the PPO ratio is not 1 and the quantity
computed there is a *counterfactual* gradient — "what the actor gradient would be if
these were the sampled entries" — **not the production update**. OWN is the source that
speaks to production. No conclusion is drawn from RANDOM/UNION alone.

The two windows are deliberate and were flagged, not silently mixed: OWN uses the TRAIN
window and `training_start_ticks` (the coherence-probe convention, and where training
actually happened); RANDOM/UNION use the EVAL window and `eval_starts` (the
matched-states convention, where P1/P2 were scored). `load_agent_and_cfg`'s `window`
argument changes only `cfg.env.start_frac_lo/hi`, so each arm was loaded twice rather
than having its config hand-mutated.

---

## 2. Integrity verification (deliverable 6)

Snapshots were taken **before** any Rung 3 measurement existed, in
`saved_models/marl/_RUNG3_integrity/`. Re-verified after all runs completed:

| manifest | scope | verified from | result |
|---|---|---|---|
| `SPRINT_7_RUNG3_code_before.md5` | all 38 pre-existing `marl/*.py` | `python-ai/marl/` | **38/38 OK** |
| `SPRINT_7_RUNG3_artifacts_before.md5` | all checkpoints, configs, reports, JSONs | `python-ai/saved_models/marl/` | **122/122 OK** |
| `SPRINT_7_RUNG3_R3outputs_before.md5` | R3 training + probe logs | `python-ai/` | **11/11 OK** |
| `SPRINT_7_RUNG3_prior_manifests.md5` | the earlier rungs' own manifests | `python-ai/saved_models/marl/` | **11/11 OK** |
| `SPRINT_7_RUNG3_inputs_before.md5` | failure history / log / OOF predictions | repo root | **3/3 OK** |
| `SPRINT_7_RUNG3_prereg.md5` | the pre-registration itself | `python-ai/` | **OK** — unchanged |

Manifests are verified from the cwd they were captured in; their paths are relative to
that directory. (An earlier rung produced a spurious "0/38 failed — No such file" by
checking from the wrong cwd; that is a path artefact, not a change.)

**No production module was touched.** All three checkpoints under test are
byte-identical to their pre-Rung-3 hashes, being members of the 122/122 artifact
manifest:

```
d9f8a3dfbe1caf3b57402b9ef05313e3 *R3_batch32.pth
```

**Three new files, all new, none a modification of existing machinery:**

```
a81f6abe036facfd3414803c4bc31c07 *marl/_diag_rung3_dilution.py     (the probe)
69b31db75b303420792ec51e331d18ee *marl/_diag_rung3_score.py        (the scorer)
1cc3e5eea8201822aee3ce0cad208066 *marl/_diag_rung3_bootstrap.py    (estimator noise)
```

Git testifies separately for the tracked tree: the set of tracked modifications is
**byte-for-byte the same 19 files** as at the start of this rung, all of them
pre-existing Sprint-5-era simulation/data changes unrelated to Rung 3. `python-ai/marl/`
is itself untracked in git (`?? python-ai/marl/`), so for that tree the md5 manifests —
not git — are the authority, which is why they were captured.

After-manifests written for the record: `SPRINT_7_RUNG3_code_after.md5` (41 files = 38
pre-existing + 3 new), `SPRINT_7_RUNG3_artifacts_after.md5` (95),
`SPRINT_7_RUNG3_newcode.md5`, `SPRINT_7_RUNG3_newartifacts.md5`.

### Runs performed, and which are results

| artifact | `--episodes` | `--perms` | status |
|---|---|---|---|
| `SPRINT_7_RUNG3_dilution_main.json` / `_verdict_main.json` | 32 | 200 | **THE pre-registered result** (§4 declared 32 in advance) |
| `SPRINT_7_RUNG3_dilution_ep8.json` / `_verdict_ep8.json` | 8 | 200 | estimator-noise control (§8.2) |
| `SPRINT_7_RUNG3_bootstrap_main.json` | 32 | 10,000 resamples | estimator-noise control (§8.2) |
| `SPRINT_7_RUNG3_dilution_SMOKE.json` / `_verdict_SMOKE.json` | 4 | 3 | **SMOKE TEST — not a result.** Do not cite. |
| `SPRINT_7_RUNG3_bootstrap_SMOKE.json` | 4 | 200 | **SMOKE TEST — not a result.** Do not cite. |

All runs `--device cpu`, `--start-seed 20260825`, `--random-seed 31337`. All exited 0.

---

## 3. The two hypotheses are different, and are scored separately

The brief requires distinguishing:

- **(A)** the high-risk gradient *contains* a useful directional signal → criterion **D1**
- **(B)** the high-risk gradient is *large enough to materially control* the full PPO
  update → criterion **D2** (where **D2 pass = (B) is FALSE**)

These are not the same claim and the results diverge: **(A) holds on the point estimates,
(B) does not.** A subspace can point exactly the right way and still be outvoted.

---

## 4. Exact measurements (deliverable 1)

### 4.1 Counts and high-risk share

| cell | T | n_decision | n_high | n_low | high share |
|---|---|---|---|---|---|
| A0/OWN | 12,523 | 13,144 | 850 | 12,294 | 0.0647 |
| A0/RANDOM | 12,715 | 4,975 | 216 | 4,759 | 0.0434 |
| A0/UNION | 24,521 | 74,237 | 3,597 | 70,640 | 0.0485 |
| R2/OWN | 12,684 | 22,098 | 1,071 | 21,027 | 0.0485 |
| R2/RANDOM | 12,715 | 4,975 | 216 | 4,759 | 0.0434 |
| R2/UNION | 24,521 | 74,237 | 3,597 | 70,640 | 0.0485 |
| R3/OWN | 12,554 | 19,301 | 849 | 18,452 | 0.0440 |
| R3/RANDOM | 12,715 | 4,975 | 216 | 4,759 | 0.0434 |
| R3/UNION | 24,521 | 74,237 | 3,597 | 70,640 | 0.0485 |

RANDOM and UNION counts are identical across arms by construction — the states and
actions are fixed and the decision rule `mask.sum(-1) > 1` does not depend on the policy.
The gradients still differ, because the actor does.

High-risk entries are **4.3–6.5%** of decision entries, confirming the 3.3–6.5% figure
that motivated the hypothesis.

### 4.2 Gradient contribution, high-risk vs low-risk, separately (brief item 3)

Natural scale (see §1); these are the numbers that hypothesis (B) turns on.

| cell | ‖g_full‖ | ‖g_hi‖ | ‖g_lo‖ | ‖g_hi‖/‖g_full‖ | ‖g_hi‖/‖g_lo‖ |
|---|---|---|---|---|---|
| A0/OWN | 2.63e-02 | 1.10e-02 | 2.24e-02 | 0.4190 | 0.4922 |
| A0/RANDOM | 4.68e-01 | 5.57e-02 | 4.53e-01 | 0.1189 | 0.1229 |
| A0/UNION | 3.23e-01 | 2.55e-02 | 3.05e-01 | 0.0788 | 0.0835 |
| R2/OWN | 1.82e-02 | 3.16e-03 | 1.77e-02 | 0.1737 | 0.1785 |
| R2/RANDOM | 9.44e-01 | 3.89e-02 | 9.36e-01 | 0.0412 | 0.0415 |
| R2/UNION | 7.95e-02 | 2.76e-02 | 8.37e-02 | 0.3479 | 0.3303 |
| R3/OWN | 4.30e-02 | 1.58e-02 | 4.09e-02 | 0.3670 | 0.3865 |
| R3/RANDOM | 8.26e-01 | 3.17e-02 | 8.23e-01 | 0.0384 | 0.0385 |
| R3/UNION | 3.12e-01 | 2.35e-02 | 2.98e-01 | 0.0753 | 0.0790 |

### 4.3 All four cosines (brief item 4)

| cell | cos(g_hi,synth) | cos(g_full,synth) | cos(g_lo,synth) | cos(g_hi,g_lo) | cos(g_full,g_lo) | cos(g_full,g_hi) |
|---|---|---|---|---|---|---|
| A0/OWN | **+0.9130** | +0.4023 | +0.0231 | +0.1397 | +0.9099 | +0.5380 |
| A0/RANDOM | **+0.7830** | +0.1501 | +0.0589 | +0.2142 | +0.9933 | +0.3262 |
| A0/UNION | **+0.1950** | +0.0723 | +0.0603 | +0.6860 | +0.9984 | +0.7266 |
| R2/OWN | **+0.5364** | −0.2372 | −0.3394 | +0.0650 | +0.9849 | +0.2370 |
| R2/RANDOM | **+0.8112** | −0.0778 | −0.1122 | +0.1773 | +0.9992 | +0.2171 |
| R2/UNION | **+0.9082** | −0.1439 | −0.4366 | −0.3149 | +0.9439 | +0.0162 |
| R3/OWN | **+0.4505** | −0.2437 | −0.4308 | −0.0520 | +0.9305 | +0.3177 |
| R3/RANDOM | **+0.5652** | −0.0080 | −0.0298 | +0.0793 | +0.9994 | +0.1174 |
| R3/UNION | **+0.2073** | −0.1379 | −0.1609 | +0.5841 | +0.9982 | +0.6327 |

### 4.4 Does the high-risk gradient point toward EDGE while the full gradient does not? (brief item 5)

Yes, and this is the cleanest single result in the rung.

| cell | cos(g_hi,synth) | cos(g_full,synth) | high-risk points to EDGE, full does not? |
|---|---|---|---|
| A0/OWN | +0.9130 | +0.4023 | both positive (full weaker) |
| A0/RANDOM | +0.7830 | +0.1501 | both positive (full much weaker) |
| A0/UNION | +0.1950 | +0.0723 | both weakly positive |
| R2/OWN | +0.5364 | **−0.2372** | **YES — sign disagrees** |
| R2/RANDOM | +0.8112 | **−0.0778** | **YES** |
| R2/UNION | +0.9082 | **−0.1439** | **YES** |
| R3/OWN | +0.4505 | **−0.2437** | **YES** |
| R3/RANDOM | +0.5652 | **−0.0080** | **YES** |
| R3/UNION | +0.2073 | **−0.1379** | **YES** |

In **6 of 9 cells — every R2 and R3 cell — the restricted high-risk gradient points
toward the desired EDGE response while the full gradient points away from it.** In the
remaining three (all A0) both are positive and the full gradient is merely weaker.
`cos(g_lo, synth)` is the reason: it is ≈0 for A0 (+0.023 to +0.060) and **negative for
every R2 and R3 cell** (−0.030 to −0.437). The low-risk bulk is what drags the full
gradient's sign negative in the trained arms.

### 4.5 Criteria D1 and D2 as scored

```
D1a  cos(g_hi,g_synth) > 0                     9/9 cells   PASS  (need 9/9)
D1b  beats cardinality-matched null p < 0.05   9/9 cells   PASS  (need 7/9)
                                          ==>  D1 PASS   → hypothesis (A) holds

D2a  n_hi/n_dec < 0.10                         9/9 cells   PASS  (need 9/9)
D2b  ‖g_hi‖/‖g_full‖ < 0.50                    9/9 cells   PASS  (need 7/9)
D2c  cos(g_full,g_lo) > 0.90  <- OPERATIVE     9/9 cells   PASS  (need 7/9)
                                          ==>  D2 PASS   → hypothesis (B) FAILS
```

D2c is the operative condition and it passes in **all nine** cells, range **+0.9099 to
+0.9994**. Deleting the entire high-risk gradient moves the update direction by at most
~24° and typically by ~2°. **The direction PPO actually follows is set by the low-risk
bulk.**

---

## 5. Null / permutation results (deliverable 2, brief item 7)

The pre-registered null for D1b draws `n_hi` decision entries **uniformly at random from
all decision entries**, keeps the arm's own advantages on the drawn support, and takes
`cos(g_subset, synth)`. 200 draws per cell. It asks: *is the high-risk set special
versus an arbitrary set of the same cardinality?*

This is a **different null** from Rung 2.75's, which permuted advantage *values* while
keeping the support fixed. Both are reported in the JSON; D1b uses this one. They are
not interchangeable and are not conflated.

| cell | n_hi | null mean | null sd | null p95 | observed | z | one-sided p |
|---|---|---|---|---|---|---|---|
| A0/OWN | 850 | +0.1293 | 0.3378 | +0.6205 | **+0.9130** | 2.32 | **0.0000** |
| A0/RANDOM | 216 | +0.1070 | 0.2244 | +0.4719 | **+0.7830** | 3.01 | **0.0000** |
| A0/UNION | 3,597 | +0.0746 | 0.0297 | +0.1184 | **+0.1950** | 4.05 | **0.0000** |
| R2/OWN | 1,071 | −0.0650 | 0.2401 | +0.3276 | **+0.5364** | 2.50 | **0.0050** |
| R2/RANDOM | 216 | −0.0647 | 0.0761 | +0.0574 | **+0.8112** | 11.50 | **0.0000** |
| R2/UNION | 3,597 | −0.1224 | 0.1156 | +0.0624 | **+0.9082** | 8.91 | **0.0000** |
| R3/OWN | 849 | −0.0968 | 0.3094 | +0.3937 | **+0.4505** | 1.77 | **0.0250** |
| R3/RANDOM | 216 | −0.0140 | 0.0985 | +0.1294 | **+0.5652** | 5.88 | **0.0000** |
| R3/UNION | 3,597 | −0.1351 | 0.0261 | −0.0975 | **+0.2073** | 13.10 | **0.0000** |

**9/9 exceed the null.** So the high-risk *set* is genuinely special — the alignment is
not an artefact of taking any ~5% slice of the decision entries.

Two honest observations about this table:

1. **R3/OWN is the weakest cell in the rung** (z = 1.77, p = 0.0250, 5 of 200 draws
   exceeded). The arm whose degradation this rung set out to explain has the least
   convincing high-risk signal on the only production-faithful source. §8.2 shows this
   cell does not survive either robustness check.
2. **The null's spread is enormous on OWN** — sd 0.24–0.34 for a cosine, versus
   0.026–0.116 on UNION where n_hi is 3,597. A 216-to-1,071-entry subsample cannot
   determine a direction in 233,000 dimensions with any precision. This is itself an
   estimator-noise finding and is carried into §8.2.

### The same null run on the NORM, not the direction — a direct test of (B)

The probe also asks whether `‖g_hi‖` exceeds a random same-size subset's gradient norm:

| cell | ‖g_hi‖ | null norm mean | one-sided p |
|---|---|---|---|
| A0/OWN | 1.10e-02 | 6.16e-03 | 0.035 |
| A0/RANDOM | 5.57e-02 | 3.04e-02 | 0.005 |
| A0/UNION | 2.55e-02 | 1.57e-02 | 0.000 |
| R2/OWN | 3.16e-03 | 3.31e-03 | **0.545** |
| R2/RANDOM | 3.89e-02 | 4.84e-02 | **0.805** |
| R2/UNION | 2.76e-02 | 3.98e-03 | 0.000 |
| R3/OWN | 1.58e-02 | 4.53e-03 | 0.000 |
| R3/RANDOM | 3.17e-02 | 4.33e-02 | **0.815** |
| R3/UNION | 2.35e-02 | 1.52e-02 | 0.000 |

In three cells the high-risk subset's gradient is **not even larger than an arbitrary
subset of the same size** (p = 0.545, 0.805, 0.815). Directional specialness without
magnitude specialness is precisely the (A)-yes / (B)-no split.

---

## 6. The decomposition A0 vs R2 vs R3 — criterion D4 (deliverable 3)

The behavioural fact to be explained (UNION risk response Δ, `SPRINT_7_R3_REPORT.md` §6):

```
R2 (+0.1579)  >  R3 (+0.1242)  >  A0 (+0.0191)
```

D4 asks whether at least one of `‖g_hi‖/‖g_lo‖`, `‖g_hi‖/‖g_full‖`, or
`cos(g_full, synth)` reproduces **R2 > R3 > A0** on OWN.

| quantity, OWN source | A0 | R2 | R3 | observed order | reproduces? |
|---|---|---|---|---|---|
| ‖g_hi‖/‖g_lo‖ | +0.4922 | +0.1785 | +0.3865 | A0 > R3 > R2 | **no** |
| ‖g_hi‖/‖g_full‖ | +0.4190 | +0.1737 | +0.3670 | A0 > R3 > R2 | **no** |
| cos(g_full,synth) | +0.4023 | −0.2372 | −0.2437 | A0 > R2 > R3 | **no** |

**D4 FAIL.** Worse than merely failing to match: the two mass ratios put the arms in
almost exactly the *reverse* behavioural order — A0, the arm with essentially no risk
response (Δ +0.0191), has the **largest** high-risk gradient share (0.4190), and R2, the
best arm (Δ +0.1579), has the **smallest** (0.1737).

So a *larger* high-risk gradient share is not what makes an arm respond to risk. The
mechanism is present in all three arms and its cross-arm variation runs the wrong way.
**It cannot explain the R3 degradation.** Per prereg §8 this appends "DOES NOT EXPLAIN
R3" to the verdict and stops.

The ep8 control fails D4 too, and with *different* orderings (`‖g_hi‖/‖g_full‖`:
R2 > A0 > R3; `cos(g_full,synth)`: A0 > R2 > R3) — see §8.2. Failing under two
independent samplings with inconsistent orderings is stronger evidence than failing once.

---

## 7. Analytical sensitivity to hypothetical high-risk weighting (deliverable 4)

**Offline and analytical only. No weighting was applied to any training update.** For
`w ∈ {1,2,4,8,16}` form `g(w) = w·g_hi + g_lo`. Because the decomposition is exact, this
is arithmetic on two already-computed vectors, not a simulation.

Pre-defined in prereg §7 D5: "materially change" = `cos(g(w), g(1)) < 0.90`; "align
better" = `cos(g(w), synth) > cos(g(1), synth) + 0.10`. **D5 is descriptive and does not
gate the verdict.**

### 7.1 cos(g(w), synth) — would the direction align better?

`w*` is the exact analytic sign-crossing `w* = −⟨g_lo,synth⟩ / ⟨g_hi,synth⟩`.

| cell | w* | w=1 | w=2 | w=4 | w=8 | w=16 |
|---|---|---|---|---|---|---|
| A0/OWN | −0.051 | +0.4023 | +0.6154 | +0.7816 | +0.8623 | +0.8932 |
| A0/RANDOM | −0.612 | +0.1501 | +0.2328 | +0.3682 | +0.5362 | +0.6690 |
| A0/UNION | −3.705 | +0.0723 | +0.0828 | +0.1001 | +0.1240 | +0.1493 |
| R2/OWN | **+3.545** | −0.2372 | −0.1365 | +0.0344 | +0.2375 | +0.3863 |
| R2/RANDOM | **+3.329** | −0.0778 | −0.0440 | +0.0217 | +0.1420 | +0.3297 |
| R2/UNION | **+1.456** | −0.1439 | +0.1617 | +0.5518 | +0.7810 | +0.8623 |
| R3/OWN | **+2.474** | −0.2437 | −0.0670 | +0.1479 | +0.3007 | +0.3791 |
| R3/RANDOM | **+1.367** | −0.0080 | +0.0137 | +0.0560 | +0.1350 | +0.2621 |
| R3/UNION | **+9.832** | −0.1379 | −0.1165 | −0.0787 | −0.0205 | +0.0500 |

- **8 of 9 cells** align better by > 0.10 at w = 16 (all but A0/UNION, +0.0723 → +0.1493).
- **6 of 9 cells** have the alignment numerator crossing zero at some `w ∈ (0,16]`. The
  three negative `w*` are all A0, which is already positively aligned at w = 1 and so has
  no crossing ahead of it.

### 7.2 cos(g(w), g(1)) — would the direction materially change?

| cell | w=2 | w=4 | w=8 | w=16 | first material (<0.90) |
|---|---|---|---|---|---|
| A0/OWN | +0.9609 | +0.8453 | +0.7217 | +0.6373 | w=4 |
| A0/RANDOM | +0.9943 | +0.9573 | +0.8504 | +0.6842 | w=8 |
| A0/UNION | +0.9987 | +0.9906 | +0.9653 | +0.9164 | **never (≤16)** |
| R2/OWN | +0.9872 | +0.9117 | +0.7370 | +0.5384 | w=8 |
| R2/RANDOM | +0.9993 | +0.9932 | +0.9667 | +0.8829 | w=16 |
| R2/UNION | +0.9451 | +0.6979 | +0.3926 | +0.2035 | w=4 |
| R3/OWN | +0.9548 | +0.7911 | +0.5978 | +0.4660 | w=4 |
| R3/RANDOM | +0.9994 | +0.9938 | +0.9682 | +0.8816 | w=16 |
| R3/UNION | +0.9985 | +0.9885 | +0.9562 | +0.8908 | w=16 |

At w = 2 the update is essentially unchanged everywhere (≥ +0.945). Nothing below w = 4
would matter at all.

### 7.3 ‖g(w)‖/‖g(1)‖ — the cost side

| cell | w=2 | w=4 | w=8 | w=16 |
|---|---|---|---|---|
| A0/OWN | 1.275 | 1.983 | 3.572 | 6.876 |
| R2/OWN | 1.055 | 1.232 | 1.748 | 3.005 |
| R2/UNION | 1.064 | 1.457 | 2.648 | 5.330 |
| R3/OWN | 1.170 | 1.706 | 3.039 | 5.900 |
| R3/UNION | 1.049 | 1.156 | 1.395 | 1.925 |

Gradient norm grows 1.9× to 6.9× at w = 16. A re-weighting of this size is not a free
reparameterisation: it changes the effective step size and would interact with the PPO
trust region and the (still-unfixed) minibatch tail.

### 7.4 The honest reading of D5 — it argues *against* the intervention it quantifies

The purpose of D5 was to stop a future rung being designed on an unquantified guess. It
does that, and what it says is unfavourable:

- **The arm needing help benefits least.** R3/UNION — the treatment arm on the
  symmetric on-manifold set — has `w* = +9.832` and reaches only **+0.0500** at w = 16.
  Up-weighting 16× barely moves R3 toward the reference direction. R2/UNION, the arm that
  *already* works best, gains most (−0.1439 → **+0.8623**). A lever that helps the good
  arm and not the bad arm is not a fix for the bad arm.
- **A0, the least risk-responsive arm, is already the best-aligned at w = 1** (+0.4023 on
  OWN) and has the largest high-risk share. Alignment with `synth` is therefore not
  tracking the behaviour it is meant to predict.
- The w = 16 gains are gains in cosine against a reference direction that §8.4 shows
  **does not discriminate the action channel**. Aligning better with a non-discriminating
  reference is not evidence of a better policy.

Per prereg §7, **D5 magnitudes cannot create a GO.** They do not, and on inspection they
would not have argued for one anyway.

---

## 8. Mechanism separation (brief item 9)

Four mechanisms, assigned to measurements **in advance** in prereg §9. Each gets its own
numbers. No number is used as evidence for more than one. They are not conflated.

### 8.1 Signal dilution

*Requires: `g_hi` aligned **and** small **and** `g_full ≈ g_lo`.* Measurements:
`n_hi/n_dec`, `‖g_hi‖/‖g_full‖`, `cos(g_full, g_lo)`.

All three hold. High-risk entries are 4.3–6.5% of decision entries; `‖g_hi‖/‖g_full‖` is
0.038–0.419, under 0.50 in all nine cells; and `cos(g_full, g_lo)` is +0.9099 to +0.9994
in all nine. Removing the high-risk gradient entirely leaves the update direction
essentially intact.

**Dilution is real and is the best-supported mechanism in this rung.** It is also the
one part of the rung that survives every robustness check (§8.2). What it does **not**
do is explain the arm differences (§6).

### 8.2 Estimator noise

*Rule fixed in advance (prereg §9): "a quantity that flips sign between buffers is noise,
not mechanism."* Three measurements were pre-registered: 8ep-vs-32ep stability, the
cardinality-matched null's spread, and a cluster bootstrap over the 32 episodes. All
three were performed.

**(i) 8 vs 32 episodes.** RANDOM and UNION are **bit-identical** between the two runs, as
they must be (neither depends on `--episodes`) — which validates the comparison. On OWN,
where the buffer actually changes:

| OWN quantity | A0 8ep → 32ep | R2 8ep → 32ep | R3 8ep → 32ep |
|---|---|---|---|
| cos(g_hi, synth) | +0.9450 → +0.9130 | +0.8734 → +0.5364 | **−0.2450 → +0.4505** ⚠ |
| cos(g_full, synth) | +0.4993 → +0.4023 | **+0.3013 → −0.2372** ⚠ | −0.6571 → −0.2437 |
| cos(g_hi, g_lo) | +0.3049 → +0.1397 | **−0.0013 → +0.0650** ⚠ | −0.2333 → −0.0520 |
| cos(g_hi, synth_CLOUD) | +0.9623 → +0.9133 | +0.7328 → +0.5431 | **−0.4773 → +0.7598** ⚠ |
| ‖g_hi‖/‖g_full‖ | 0.2129 → 0.4190 | 0.3529 → 0.1737 | 0.1673 → 0.3670 |
| D1b one-sided p | 0.0000 → 0.0000 | 0.0000 → 0.0050 | **0.6300 → 0.0250** ⚠ |

**Four sign flips (⚠), all on OWN, and R3 is in three of them.** `‖g_hi‖/‖g_full‖`
reshuffles the arm order completely: R2 > A0 > R3 at 8 episodes, A0 > R3 > R2 at 32.
R3/OWN's high-risk alignment is **p = 0.63 — indistinguishable from a random same-size
subset — at 8 episodes**, and only marginally significant (p = 0.0250) at 32.

Consequence for the verdict: **the ep8 run scores D1a 8/9 → D1 FAIL → verdict
"FALSIFIED — DOES NOT EXPLAIN R3"**, while the 32-episode run scores "SUPPORTED (simple
dilution) — DOES NOT EXPLAIN R3". The pre-registered result is the 32-episode one — §4
declared 32 in advance, for documented reasons — and it stands. But the verdict *label*
is batch-size fragile. **Both batch sizes agree on NO-GO and on "DOES NOT EXPLAIN R3";
only the D1 half moves.**

**(ii) The null's spread.** Cosine sd 0.24–0.34 on OWN (n_hi = 850–1,071) versus
0.026–0.116 on UNION (n_hi = 3,597). The OWN high-risk subsample is far too small to
determine a direction in 233,000 dimensions.

**(iii) Cluster bootstrap, 10,000 episode resamples.** Exact by linearity: a replicate's
gradient is `Σ_e m_e h_e`, so every reported cosine and norm ratio is a quadratic form
`mᵀ(ABᵀ)m` in the multiplicity vector, computed from six `n_eps × n_eps` Gram matrices.
No replicate vector is materialised. Reassembly `‖Σ_e h_e/D − g_hi‖/‖g_hi‖` was asserted
per cell: **7.9e-08 to 2.8e-07**. Cosines and norm ratios are invariant to the
replicate's `1/D_m`, so the varying denominator cannot bias them.

| cell | cos(g_hi, synth) 95% CI | P(>0) | cos(g_full, g_lo) 95% CI |
|---|---|---|---|
| A0/OWN | +0.8088 [+0.5087, +0.9548] | 0.9999 | +0.9381 [+0.8596, +0.9806] |
| A0/RANDOM | +0.7270 [+0.3266, +0.9333] | 0.9985 | +0.9931 [+0.9832, +0.9983] |
| A0/UNION | +0.1947 **[−0.1016, +0.4781]** | 0.8979 | +0.9982 [+0.9971, +0.9990] |
| R2/OWN | +0.3478 **[−0.4392, +0.8449]** | 0.8169 | +0.9730 [+0.9267, +0.9938] |
| R2/RANDOM | +0.7453 [+0.4588, +0.9136] | 1.0000 | +0.9991 [+0.9980, +0.9997] |
| R2/UNION | +0.8951 [+0.8338, +0.9316] | 1.0000 | +0.9400 [+0.8568, +0.9807] |
| R3/OWN | +0.4094 **[−0.0276, +0.6584]** | 0.9701 | +0.9315 [+0.7893, +0.9944] |
| R3/RANDOM | +0.5235 **[−0.1052, +0.8931]** | 0.9497 | +0.9990 [+0.9974, +0.9997] |
| R3/UNION | +0.2016 **[−0.0743, +0.4756]** | 0.9191 | +0.9980 [+0.9964, +0.9990] |

**`cos(g_hi, synth)`'s CI spans zero in 5 of 9 cells — including all three R3 cells.**
Only 4 of 9 (A0/OWN, A0/RANDOM, R2/RANDOM, R2/UNION) are separable from zero at 32
episodes. Meanwhile **`cos(g_full, g_lo)`'s CIs are tight and high everywhere** (means
+0.93 to +0.999, lowest bound +0.7893).

The bootstrap means sit below the point estimates (A0/OWN +0.8088 vs +0.9130; R2/OWN
+0.3478 vs +0.5364). That is expected for a nonlinear functional of a resample and is
**not** a bias correction — the point estimate remains the estimate; the CI is the
precision statement.

**Conclusion of this subsection.** The three pre-registered estimator-noise measurements
agree: **D2's finding is robust, D1's is not.** "The low-risk bulk sets the update
direction" is stable across batch size, identical on fixed state sets, and tightly
bounded by the bootstrap. "The high-risk gradient contains useful signal" holds on the
32-episode point estimates, but flips sign at 8 episodes for R3 and is not separable from
zero for any R3 cell. This weakens the "SUPPORTED" half of the verdict and **strengthens
the "DOES NOT EXPLAIN R3" half.** It is a pre-registered analysis, not a new metric.

### 8.3 Policy-state selection

*Rule fixed in advance: "if a conclusion holds only on OWN it is a selection artefact."*
Measurement: OWN vs RANDOM vs UNION disagreement.

**The D1/D2 conclusions do not depend on the source** — D1a, D2a, D2b and D2c each pass
in all nine cells, on all three sources. Neither conclusion is a selection artefact.

**The arm ordering is entirely source-dependent.** `cos(g_hi, synth)`:

| source | ordering |
|---|---|
| OWN | A0 (+0.9130) > R2 (+0.5364) > R3 (+0.4505) |
| RANDOM | R2 (+0.8112) > A0 (+0.7830) > R3 (+0.5652) |
| UNION | R2 (+0.9082) > R3 (+0.2073) > A0 (+0.1950) |

A0 goes from first to last between OWN and UNION. This is the on-policy selection
confound documented in earlier rungs: risk is high at a host *because* tasks remain on
it, so conditioning on "currently high-risk" on an arm's own trajectories selects for
"just chose STAY", and each arm's OWN states are the states its own policy produced. The
UNION set exists precisely to remove that.

This is also the mechanism-level reason **D4 could not have been rescued by choosing a
different source**: the quantity whose ordering one would want to read off is the
quantity most contaminated by which policy generated the states.

### 8.4 Action-channel effects

*Measurement fixed in advance: `cos(g_hi, synth_CLOUD)` alongside `cos(g_hi, synth)`,
where `synth_CLOUD` is +1 on MIGRATE_CLOUD / −1 on STAY at high risk. Purpose: test
whether the reference direction, not the gradient, is misspecified.*

| cell | cos(g_hi, synth_EDGE) | cos(g_hi, synth_CLOUD) | difference |
|---|---|---|---|
| A0/OWN | +0.9130 | **+0.9133** | +0.0003 |
| A0/RANDOM | +0.7830 | +0.5282 | −0.2548 |
| A0/UNION | +0.1950 | +0.2715 | +0.0765 |
| R2/OWN | +0.5364 | **+0.5431** | +0.0067 |
| R2/RANDOM | +0.8112 | +0.7035 | −0.1077 |
| R2/UNION | +0.9082 | **+0.9248** | +0.0166 |
| R3/OWN | +0.4505 | **+0.7598** | **+0.3093** |
| R3/RANDOM | +0.5652 | +0.8224 | **+0.2572** |
| R3/UNION | +0.2073 | +0.2031 | −0.0042 |

**The reference direction barely discriminates the action channel.** In A0/OWN the two
agree to within 0.0003; in R2/OWN and R2/UNION to within 0.017. And in **both R3
cells where they differ materially, R3's high-risk gradient aligns *better* with the
CLOUD direction than the EDGE direction** (+0.7598 vs +0.4505; +0.8224 vs +0.5652).

The reason is in the synth support counts:

| cell | n(MIGRATE_EDGE, +1) | n(MIGRATE_CLOUD, +1) | n(STAY, −1) |
|---|---|---|---|
| A0/OWN | 70 | 79 | **701** |
| R2/OWN | 69 | 91 | **911** |
| R3/OWN | 74 | 68 | **706** |
| */UNION | 182 | 278 | **3,137** |

Both reference directions are dominated by the **shared `−1` on STAY** term, which is 8×
to 17× more populous than either positive channel. So `cos(g_hi, synth)` is largely
measuring **"push away from STAY at high risk"** — which is channel-agnostic — rather
than "push toward EDGE specifically."

**This is a limitation of the reference direction, and it qualifies D1's meaning
directly.** D1 passing means the high-risk gradient wants to *move off STAY*. It is much
weaker evidence that the gradient wants the *specific* EDGE action that P1/P2 score. It
does not change the pre-registered verdict — `synth` was fixed in advance and is
unchanged from Rung 2.75, and redefining it now would be exactly the post-hoc metric
change the brief forbids. It does mean **D1 should not be read as "the gradient already
knows the right action."**

Consistency note, no conflation: this is a statement about the *gradient's* channel
alignment. `SPRINT_7_R3_REPORT.md` §10.7 separately established that R3's *behavioural*
Δ_CLOUD is negative on both neutral sets. A gradient that aligns with the CLOUD
direction and a behaviour that does not move toward CLOUD are different quantities
measured on different objects; both are reported, neither is used to support the other.

---

## 9. Verdict (deliverable 5)

Produced by `marl/_diag_rung3_score.py` from
`SPRINT_7_RUNG3_dilution_main.json`, with every threshold transcribed from the
pre-registration as a module constant, so applying the rule could not drift toward the
result:

```
D1a  9/9 PASS   D1b  9/9 PASS                 ==>  D1 PASS
D2a  9/9 PASS   D2b  9/9 PASS   D2c 9/9 PASS  ==>  D2 PASS
D3   1/9        (need 5/9)                    ==>  D3 FAIL
D4                                            ==>  D4 FAIL

VERDICT: SUPPORTED (simple dilution) -- DOES NOT EXPLAIN R3

GO for a future TRAINING rung requires D1 AND D2 AND D4:
  D1 PASS  D2 PASS  D4 FAIL   ==>  NO-GO for a training rung
```

**D3 (competing-gradient interference) FAILS, and that matters for classification.**
`cos(g_hi, g_lo)` is **positive in 7 of 9 cells** (+0.0650 to +0.6860). Two are negative —
R3/OWN (−0.0520) and R2/UNION (−0.3149) — and only R2/UNION clears the pre-registered
−0.10 threshold, giving D3 1/9 against a required 5/9. The low-risk gradient does **not**
actively oppose the high-risk one — it merely outweighs it. So this is **genuine dilution,
not interference**, and the two are reported as the distinct mechanisms the
pre-registration required.

Answering the brief's two hypotheses explicitly:

> **(A) "the high-risk gradient contains a useful directional signal"** — **YES** on the
> pre-registered 32-episode point estimates: positive in 9/9 cells and beating a
> cardinality-matched null in 9/9. **Qualified twice**: it is not robust to batch size
> (R3/OWN flips sign at 8 episodes, p 0.63 → 0.0250) and its bootstrap CI spans zero in
> 5/9 cells including all three R3 cells (§8.2); and "useful direction" here largely
> means "off STAY" rather than "toward EDGE", because the reference direction is dominated
> by its shared STAY term (§8.4).

> **(B) "the high-risk gradient is large enough to materially control the full PPO
> update"** — **NO**, robustly. `cos(g_full, g_lo) ∈ [+0.9099, +0.9994]` in 9/9 cells,
> with tight bootstrap CIs. Deleting the high-risk gradient entirely leaves the update
> direction essentially unchanged. `‖g_hi‖/‖g_full‖ ≤ 0.419` everywhere. In three cells
> `‖g_hi‖` is not even larger than that of a random subset of the same size.

**Per the pre-registration, this rung stops here.** D4 failed, so the compositional
hypothesis does not account for the degradation it was invoked to explain. No training
rung is proposed. No new success metric was introduced after seeing the results. No
weighting was applied to training.

---

## 10. Limitations and falsification

1. **The verdict label is batch-size fragile.** At 8 episodes the same scorer returns
   FALSIFIED rather than SUPPORTED. The 32-episode run is the pre-registered one (§4,
   declared in advance) and stands, but the D1 half should be treated as unresolved
   rather than established. Both batch sizes agree on **NO-GO** and on **DOES NOT EXPLAIN
   R3** — those are the robust conclusions.
2. **The reference direction is only weakly channel-discriminating** (§8.4). Any future
   use of `cos(·, synth)` as a criterion should first fix this, e.g. by scoring the EDGE
   and STAY components separately. That is a methodology observation for a later rung, not
   a change applied here.
3. **RANDOM and UNION gradients are counterfactual** (prereg §5, disclosed in advance):
   the PPO ratio is not 1 there. OWN is the only production-faithful source, and OWN is
   also the noisiest and the most selection-contaminated. This tension is intrinsic, not
   an oversight.
4. **The bootstrap holds advantage standardisation fixed** at its full-sample value
   across replicates. Re-standardising per replicate would make the contributions
   non-linear in the resample and destroy the exact Gram identity the method rests on.
   GAE needs no such caveat, being computed within an episode.
5. **`w*` and the sensitivity are first-order in the gradient only.** They describe the
   direction of a single update, not the trajectory a weighted training run would take.
   That is why they cannot create a GO.
6. **Single seed (20260818 lineage checkpoints, start-seed 20260825).** Cross-arm
   comparisons inherit the single-seed limitation of the arms themselves. Nothing here
   was cherry-picked across seeds; only one seed exists.
7. **Not measured:** whether a *learned* (rather than fixed) high-risk weighting would
   behave differently; anything requiring a training run. Out of scope by instruction.
8. **The minibatch tail remains unfixed** and `explained_var` remains target-relative;
   neither is used in this rung, but both remain open from earlier rungs.

---

## 11. What this rung establishes, and what it forecloses

**Established:**
- The exact linear decomposition `g_full = g_hi + g_lo` is verified numerically
  (additivity ≤ 2.7e-07) and the probe's gradient is **bit-identical** to the existing
  production-convention implementation.
- High-risk decision entries are **4.3–6.5%** of decision entries, confirming the figure
  that motivated the hypothesis.
- **The low-risk bulk sets the update direction**, robustly: `cos(g_full, g_lo) ≥ 0.9099`
  in 9/9 cells, tight bootstrap CIs, stable across batch size. Hypothesis (B) is false.
- **This is dilution, not interference** — `cos(g_hi, g_lo)` is positive in 7 of 9 cells,
  and only 1 of 9 reaches the pre-registered interference threshold of −0.10.
- **The mechanism does not explain the arm differences.** The two mass ratios order the
  arms nearly *opposite* to the behavioural ordering: A0 (Δ +0.0191) has the largest
  high-risk share, R2 (Δ +0.1579) the smallest.

**Foreclosed (do not re-test):**
- Do not propose a high-risk-weighted training rung on the strength of this rung. It is
  NO-GO by the pre-registered rule, and D5 shows the leverage is smallest exactly where it
  would be needed (R3/UNION: `w* = +9.832`, +0.0500 at w = 16) and largest where it is not
  (R2/UNION: +0.8623 at w = 16).
- Do not treat `cos(g_hi, synth) > 0` as evidence that the gradient knows the right
  action; it is dominated by the shared STAY term.
- Do not compare arm orderings of any high-risk mass ratio on the OWN source; the
  8-vs-32 comparison reshuffles them and the source comparison reverses them.

**Not done, by instruction:** no training run, no production modification, no R4 design,
no weighting applied to a real update, no new success metric.
