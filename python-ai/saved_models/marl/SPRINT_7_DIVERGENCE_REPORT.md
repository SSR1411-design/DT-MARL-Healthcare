# SPRINT 7 — R2 vs R3 DIVERGENCE DIAGNOSTIC

**Offline diagnostic phase. No training was run. No production code was modified.
No new success criterion is proposed. No R4 is designed.**

The question this report answers, and the only question it answers:

> **WHAT DIFFERENCE BETWEEN R2 AND R3 IS ACTUALLY CAPABLE OF EXPLAINING THE
> BEHAVIOURAL DIFFERENCE?**

The Rung 3 verdict — *SUPPORTED (simple dilution) but DOES NOT EXPLAIN R3,
therefore NO-GO* — is treated as locked throughout. Nothing here reinterprets it.
The fixed RANDOM and UNION state sets are used exactly as previously defined;
R3 is scored on them and is never a generator.

**Headline.** The answer is largely negative and it is worth stating up front,
because four of the five candidate mechanisms named in the brief are ruled out
by direct measurement rather than left open:

- Increasing `rollout_episodes` from 8 to 32 is a **pure variance intervention on
  the update content**. It shifts the expected content by ≤ 0.105 σ₈ on every
  statistic measured, while the standard deviation tracks the finite-population
  sampling law to within 5%. It did **not** change *what* the actor was trained on.
- The one previously-reported result that looked like a batch-size mechanism —
  the high-risk EDGE-vs-STAY drive collapsing 5.1× and flipping sign between an
  8- and a 32-episode buffer — **is an outlier artifact of the 8-episode
  instrument, not a mechanism.** This is an explicit self-correction and is
  documented in §4.3.
- R3's high-risk critic deficit is a property of **the buffer, not the critic**:
  in a matched 2×2, the two critics differ by 0.009 on the same states.
- R3's `frac_mb_zero_hi_EDGE = 0.20` is **the four degenerate `T mod 4 = 2` tail
  chunks**, a shared arithmetic confound — not dilution.

What survives is a single, regime-selective asymmetry, replicated on two
statistically independent instruments (the actor's function-space displacement
and the critic's explained variance). It is a **"what differs" finding, not a
causal one**, and §9 says so plainly rather than dressing it up.

---

## 1. Integrity

### 1.1 Manifests

Before-manifests were taken at 21:04 on 2026-08-27, prior to any diagnostic in
this phase. After-manifests were taken on completion, over **identical scopes**.

| scope | root | glob | before | after | changed or removed | added |
|---|---|---|---|---|---|---|
| artifacts | `python-ai/saved_models/marl/` | `*.pth *.json *.md` | 96 | 108 | **0** | 12 |
| code | `python-ai/marl/` | `*.py` | 41 | 47 | **0** | 6 |
| logs | `python-ai/` | `SPRINT_7*.log run_*.log` | 58 | 67 | **0** | 9 |
| inputs | repo root | 3 named files | 3 | 3 | **0** | 0 |

`md5sum -c` against each before-manifest returns OK on **every** pre-existing
line: 96/96 artifacts, 41/41 code files, 58/58 logs, 3/3 inputs. The
before→after delta is **purely additive**, and every added line is one of this
phase's own outputs:

- **code (6):** `_diag_div_logs.py`, `_diag_div_geometry.py`, `_diag_div_shape.py`,
  `_diag_div_content.py`, `_diag_div_variance.py`, `_diag_div_critic.py`
- **artifacts (12):** eight `SPRINT_7_DIV_*.json`, four `diag_S7_D?_*_b32.json`
- **logs (9):** nine `SPRINT_7_DIV_*.log`

Written this phase: `SPRINT_7_DIV_artifacts_after.md5`,
`SPRINT_7_DIV_code_after.md5`, `SPRINT_7_DIV_logs_after.md5`,
`SPRINT_7_DIV_inputs_after.md5`, plus additive-only manifests
`SPRINT_7_DIV_newcode.md5`, `SPRINT_7_DIV_newartifacts.md5`,
`SPRINT_7_DIV_newlogs.md5`. All in
`python-ai/saved_models/marl/_DIVERGENCE_integrity/`.

### 1.2 Non-interference

`mappo.py`, `train.py`, `env.py` and every other production module are
byte-identical to their pre-phase checksums. All six probes are read-only:
they load checkpoints, replay episodes through existing machinery, and evaluate
under `torch.no_grad` or via `compute_gae`/`compute_mc_returns`, which are pure
functions of a buffer. No optimiser was constructed for training; no parameter
tensor was written.

### 1.3 The shared initial policy is exactly reconstructable

`MAPPO.__init__` calls `torch.manual_seed(seed)` at `mappo.py:218` immediately
before constructing the actor and critic. All arms carry `train.seed = 20260818`
and identical architecture fields (`actor_hidden`, `critic_hidden`,
`separate_actors` — asserted in the probe). Two independent constructions give
`max|diff| = 0.0e+00` for both actor and critic. **All arms therefore share one
θ₀ and displacements from it are directly comparable.** d_actor = 233,000,
d_critic = 194,049; chance cosine 1/√d = 0.00207 (actor), 0.00227 (critic).

---

## 2. R2 vs R3 experimental invariants

### 2.1 The manipulation is clean

Of **99 config fields**, five differ, and only one is substantive:

| field | R2 | R3 | status |
|---|---|---|---|
| `train.rollout_episodes` | 8 | 32 | **the manipulation** |
| `train.episodes` | 600 | 2400 | consequence: holds `n_updates` fixed |
| `train.tag` | `mappo_R2_mc_target` | `R3_batch32` | provenance |
| `episodes` | 600 | 2400 | provenance |
| `wall_time_s` | 1585.6 | 5806.0 | provenance |

`n_updates = episodes // rollout_episodes` (`train.py:168`) = **75 for both**.
Learning-rate schedules are **bit-identical**: `max_abs_lr_diff = 0.0`,
`lr_scale` running 1.0000 → 0.0133 over the same 75 updates. Same seed, same
architecture, same γ = 0.999, same λ = 0.995, same critic target (`mc`).

### 2.2 Episode matching is exact and bit-identical

`training_start_ticks` builds start ticks as iid draws from a single re-seeded
`np.random.default_rng(TRAIN_SEED)` stream, and `build` calls
`torch.manual_seed(TRAIN_SEED + j)` per episode *j*. Consequences, both verified:

- the streams are **nested**: identical prefix of 8 episodes, **first divergent
  episode = 9**, exactly as structurally predicted (`prediction_holds: true`);
- **episode *j* is byte-identical** whether it sits in an 8-, 32-, or
  128-episode build.

This is what licenses the subsampling design in §4.2, and it is a stronger
guarantee than the GAE-boundary argument alone. It also means the first 8
episodes of a 128-episode population **are** R2's actual buffer and the first 32
**are** R3's.

### 2.3 The behavioural difference being explained

Reproduced Δ = P(MIGRATE_EDGE | high risk) − P(MIGRATE_EDGE | low risk) on the
unchanged fixed sets (this is the pre-registered P1/P2 quantity, recomputed here
purely as a population check before anything else measured on those sets is
believed):

| set | θ₀ | A0 | R2 | R3_best | R3 |
|---|---|---|---|---|---|
| RANDOM | +0.0165 | −0.0264 | **+0.2024** | +0.1467 | **+0.1576** |
| UNION | +0.0007 | +0.0191 | **+0.1579** | +0.0946 | **+0.1242** |

R3's risk response is real and well above A0's — it is *worse than R2's*, which
is the gap under investigation. `R2_best` is **bit-identical to `R2`** in every
measured quantity (same ‖Δθ‖ 6.1710, same Adam state, same Δ), so R2 has only
one distinct policy snapshot. `R3_best` is a genuine **update-45** checkpoint:
its saved `lr = 2.8933e-04` against a base implying `lr_scale = 0.41333`, and it
has 872 Adam steps versus 1440 at the end.

### 2.4 Correction to an earlier provisional reading

"R3's reward peaked at update 45 then declined" is **not supported**. That
reading came from `_best` selection on a single noisy 32-episode mean. Block
means show R3 improving monotonically across all four quartiles
(−16.11 → −12.16 → −10.59 → −8.51). Only A0 shows a genuine peak-then-drop. The
E2 channel-collapse fact in §7.3 must therefore be framed as
**"R3_best vs R3_final"**, not as post-peak decay.

---

## 3. Update-level divergence (candidate mechanism 5)

Both arms ran exactly 75 updates on an identical LR schedule, so update index is
a legitimate alignment axis. Onset = first update after which the R2−R3 gap
stays beyond 1 ref-sd (ref sd = sd of the first 10 updates) for the remainder of
training.

### 3.1 Update-aligned quartile means

| metric | R2 Q1 | Q2 | Q3 | Q4 | R3 Q1 | Q2 | Q3 | Q4 | onset |
|---|---|---|---|---|---|---|---|---|---|
| mean_reward | −16.76 | −7.70 | +0.08 | +1.11 | −16.11 | −12.16 | −10.59 | −8.51 | **47** |
| explained_var | 0.3001 | 0.5784 | 0.6754 | 0.7020 | 0.2969 | 0.5707 | 0.6344 | 0.6564 | **34** |
| decision_frac | 0.0601 | 0.1342 | 0.1795 | 0.1787 | 0.0687 | 0.0829 | 0.0905 | 0.1345 | **25** |
| value_mean | −0.517 | +1.899 | +1.094 | +1.301 | +0.547 | −0.307 | +1.253 | +1.290 | **10** |
| entropy | 0.5751 | 0.2784 | 0.1813 | 0.1611 | 0.5040 | 0.3714 | 0.3526 | 0.2923 | None |
| critic_loss | 16.31 | 12.83 | 10.67 | 8.72 | 19.47 | 16.26 | 12.47 | 10.00 | None |
| adv_std | 3.660 | 3.360 | 3.084 | 2.862 | 3.982 | 3.621 | 3.266 | 3.000 | None |
| approx_kl (k1) | 0.00400 | 0.00207 | −0.00031 | −0.00033 | 0.00367 | 0.00140 | 0.00072 | 0.00030 | None |
| clip_frac | 0.0426 | 0.0189 | 0.0027 | 0.00004 | 0.0355 | 0.0136 | 0.0053 | 0.0008 | None |
| lr_scale | 0.8867 | 0.6400 | 0.3867 | 0.1333 | 0.8867 | 0.6400 | 0.3867 | 0.1333 | — |

Finals: reward +12.654 / −4.484; entropy 0.1380 / 0.2659; ev 0.7409 / 0.6822;
critic_loss 5.771 / 9.421; decision_frac 0.1767 / 0.1493.

**The ordering of onsets is the substantive result of this section.** The
critic-quality gap (`explained_var`, onset 34, opening in Q3 = updates 31–45)
**precedes** the reward gap (onset 47). `decision_frac` moves earlier still
(onset 25). `value_mean`'s onset of 10 is not interpreted: its ref_sd (0.8148)
is three times its mean difference (0.2658), so the rule fires on a quantity
whose gap is well inside its own early-training noise.

### 3.2 Movement is not the difference

| proxy | R2 | R3 |
|---|---|---|
| Σ k1 | 0.099195 | 0.112050 |
| Σ \|k1\| | 0.153265 | **0.152772** |
| updates with k1 < 0 | 26 | 21 |
| Σ clip_frac | 1.1763 | 1.0121 |
| updates clip-engaged | 53 (last 61) | 58 (last 60) |
| Adam actor steps | 1392 | 1440 |

Cumulative within-update policy movement is **near-identical** (Σ|k1| agrees to
0.3%), and R3 took *more* optimiser steps. Yet net displacement is smaller:
‖Δθ_actor‖ = 6.1710 (R2) vs 4.5471 (R3). Equal per-update movement with less net
displacement means **R3's steps cancel more**. That is a description of the
geometry, not yet an explanation of it.

### 3.3 The progress clock

Locating R3's endpoint on R2's trajectory by nearest value:

| metric | R3 final | closest R2 update | R2 value there |
|---|---|---|---|
| entropy | 0.2659 | **25** | 0.2746 |
| decision_frac | 0.1493 | **27** | 0.1504 |
| explained_var | 0.6822 | 44 | 0.6828 |
| mean_reward | −4.4836 | 56 | −4.6479 |
| adv_std | 2.8730 | 63 | 2.8852 |
| value_mean | 1.0388 | 67 | 1.0261 |
| critic_loss | 9.4207 | 70 | 9.4464 |

Median 56. **Actor-side metrics place R3 at R2's update 25–27; critic-side
metrics place it at 44–70.** R3 is not uniformly "an earlier R2" — the actor is
much further behind than the critic. §7 returns to this.

### 3.4 Experience-aligned view (18 blocks of 4 R2 updates per R3 update)

Per unit of environment experience R2 is strictly ahead on everything: by
episode 600 R2 has completed 75 updates and R3 only 18 (4.17×). Block means
R2/R3: reward −6.14/−16.11; entropy 0.3014/0.5040; ev 0.5612/0.2969;
critic_loss 12.28/19.47; decision_frac 0.1377/0.0687; adv_std 3.259/3.982. This
is arithmetically forced by the design and carries no mechanistic information;
it is recorded so the update-aligned comparison is not mistaken for a
sample-efficiency claim.

### 3.5 Variability, against the pure-noise prediction

If a 4× larger buffer only reduced estimator noise, detrended sd ratios R3/R2
should be ≈ 0.5:

| metric | detrended sd R2 | R3 | ratio | vs 0.5 |
|---|---|---|---|---|
| decision_frac | 0.00852 | 0.00408 | 0.479 | ≈ |
| adv_std | 0.18319 | 0.09983 | 0.545 | ≈ |
| clip_frac | 0.01595 | 0.01272 | 0.797 | above |
| explained_var | 0.03158 | 0.02544 | 0.806 | above |
| critic_loss | 1.70257 | 1.37757 | 0.809 | above |
| approx_kl | 0.00317 | 0.00299 | 0.944 | above |
| **entropy** | 0.03410 | 0.03880 | **1.138** | **R3 MORE variable** |

The buffer did buy noise reduction where the estimator is the dominant term
(`decision_frac`, `adv_std`), but **entropy is more variable in R3 despite 4×
the data**. That is not a variance-reduction signature.

### 3.6 Matched-pair episode divergence (EXPLORATORY)

600 exactly-paired episodes (§2.2 guarantees identity):

| metric | R2 | R3 | diff | z paired | z sign | onset |
|---|---|---|---|---|---|---|
| `infeasible` | 14.560 | 1.698 | −12.862 | **−34.51** | −22.04 | **130** |
| `preemptive` | 74.078 | 80.173 | +6.095 | +22.43 | +16.11 | 454 |
| `relocations` | 74.983 | 81.048 | +6.065 | +22.10 | +15.64 | 454 |
| reward | −5.673 | −15.681 | −10.009 | −12.59 | −9.62 | None |
| success_rate | 0.7229 | 0.6931 | −0.0298 | −11.48 | −9.33 | None |
| lost | 10.127 | 11.160 | +1.033 | +8.95 | +7.81 | None |
| energy | 2118.3 | 2080.6 | −37.76 | −7.12 | −5.34 | None |
| critical_lost | 4.105 | 4.367 | +0.262 | +2.82 | +1.71 | None |
| sla | 1.468 | 1.588 | +0.120 | +1.78 | +2.46 | None |
| protected | 8.743 | 8.442 | −0.302 | −1.68 | −1.86 | None |

**`infeasible` is a contention counter, not an illegal-action counter**
(`env.py:405-422`, increments at 565/630, rotating apply order 620-626, reward
penalty at 934). The actor is masked (`masked_dist`, `mappo.py:77-79`), so
illegal actions cannot be sampled. R2 generates 8.6× more contention than R3 —
that is R2's agents *competing to act on the same host*, which is what a policy
that actually migrates does. R3 is quiet. This is the earliest and by far the
strongest environment-side signal (|z| = 34.5, onset episode 130), and it is
consistent with, not independent of, the actor being behind.

---

## 4. Rollout / data-distribution comparison (candidates 1 and 2)

### 4.1 On-own-distribution occupancy and action frequencies (C1)

**Labelled ON-OWN-DISTRIBUTION and NOT comparable to P1/P2**, which are measured
on the fixed sets.

| | A0 | R2 | R2_b32 | R3_best | R3 |
|---|---|---|---|---|---|
| T | 3100 | 3200 | 12684 | 12552 | 12554 |
| episodes | 8 | 8 | 32 | 32 | 32 |
| decision entries | 3452 | 4967 | 22098 | 10817 | 19301 |
| decision_frac | 0.1114 | 0.1552 | 0.1742 | 0.0862 | 0.1537 |
| hi (≥0.6) decision | 212 | 269 | 1069 | 584 | 847 |
| hi share of decision | 0.0614 | 0.0542 | 0.0484 | 0.0540 | 0.0439 |
| STAY selfrac | 0.7998 | 0.8581 | 0.8697 | 0.7612 | 0.8519 |
| EDGE selfrac | 0.1228 | 0.0729 | 0.0647 | 0.1569 | 0.0832 |
| EDGE at hi | 0.0660 | 0.0372 | 0.0645 | 0.1284 | 0.0874 |
| EDGE at lo | 0.1265 | 0.0751 | 0.0648 | 0.1590 | 0.0831 |
| hi − lo (own dist) | −0.0605 | −0.0379 | −0.0002 | −0.0306 | +0.0043 |

The high-risk *share* of decision entries is similar across all cells
(0.044–0.061), so R3's buffer is not structurally starved of high-risk states.

### 4.2 The identifying measurement: sampling law of the high-risk signal

`SPRINT_7_DIV_variance_main.json` — policy held **fixed**, 128-episode
population, 4000 subsets per size drawn **without replacement**, normalisation
pool recomputed per subset exactly as `mappo.py:360-366` does.

The statistic that matters is the per-sample coefficient the actor loss puts on
`log π(a|s)`: `z` = the normalised advantage, and the actual channel push
`channel_drive(hi,EDGE) = (n_hi_EDGE / n_decision) × mean z(hi,EDGE)`.

**Means barely move; only the spread does.** Shift from n=8 to the full
population, expressed in units of the n=8 sampling sd:

| quantity | R2 m@8 | m@32 | pop | shift/σ₈ | R3 m@8 | m@32 | pop | shift/σ₈ |
|---|---|---|---|---|---|---|---|---|
| z_hi_EDGE | −0.0951 | −0.0894 | −0.0846 | +0.022 | +0.0023 | +0.0037 | +0.0044 | +0.005 |
| z_hi_STAY | −0.2862 | −0.3106 | −0.3157 | −0.089 | −0.3340 | −0.3620 | −0.3679 | −0.094 |
| **contrast** | **+0.1912** | +0.2212 | **+0.2312** | **+0.088** | **+0.3363** | +0.3657 | **+0.3723** | **+0.081** |
| channel_drive | −0.0003 | −0.0003 | −0.0002 | 0.000 | 0.0000 | 0.0000 | 0.0000 | −0.019 |
| z_lo_EDGE | −0.2859 | −0.2836 | −0.2810 | +0.017 | −0.1807 | −0.1797 | −0.1793 | +0.009 |
| pool_std_raw | 3.4964 | 3.5203 | 3.5280 | +0.075 | 3.4751 | 3.5017 | 3.5060 | +0.080 |
| raw_mean_hi | −0.7020 | −0.7965 | −0.8190 | −0.105 | −0.8348 | −0.9094 | −0.9184 | −0.072 |

**Every shift is ≤ 0.105 σ₈.** Meanwhile the sd changes by *exactly* the
finite-population prediction. For subsets drawn without replacement from N=128,
`sd(n) ∝ √((N−n)/(N−1))/√n`, so the correct prediction is
**sd(8)/sd(32) = 2.236**, not 2.000:

| quantity | R2 observed ratio | obs/FPC | R3 observed | obs/FPC |
|---|---|---|---|---|
| z_hi_EDGE | 2.364 | 1.057 | 2.313 | 1.034 |
| z_hi_STAY | 2.350 | 1.051 | 2.289 | 1.024 |
| contrast | 2.356 | 1.053 | 2.295 | 1.027 |
| channel_drive | 2.293 | 1.025 | 2.282 | 1.020 |
| z_lo_EDGE | 2.268 | 1.014 | 2.247 | 1.005 |
| pool_std_raw | 2.162 | 0.967 | 2.191 | 0.980 |
| raw_mean_hi | 2.289 | 1.024 | 2.248 | 1.005 |

(`n_hi_EDGE` is excluded: it is a *count*, so its sd grows as √n. Observed
0.557 / 0.545 — correct for a count, and the reason it is not evidence of
anything.)

> **Candidate mechanism 1's literal question — "did increasing
> `rollout_episodes` from 8→32 change WHAT the actor was trained on?" — is
> answered: NO, not in expectation. Only the noise.** Across seven update-content
> statistics the expected value is invariant to within a tenth of a standard
> deviation while the spread halves exactly as the finite-population law
> requires. `rollout_episodes` is a **pure variance intervention** on the update
> content.

### 4.3 SELF-CORRECTION: the "5.1× collapse" is an outlier, not a mechanism

An earlier provisional reading held that holding R2's parameters fixed and
changing only the buffer from 8 to 32 episodes collapsed the high-risk drive
5.1× and flipped `z(hi,EDGE)`'s sign (+0.9080 → +0.1769; +0.2988 → −0.1461) —
"a difference that exists in the DATA, before any parameter moves."

**That reading is superseded.** Where each arm's realized buffer sits in its own
sampling law:

| arm | statistic | realized | law mean | law sd | **z** |
|---|---|---|---|---|---|
| R2 @ 8 | contrast | **+0.9080** | +0.1912 | 0.3429 | **+2.09** |
| R2 @ 8 | z_hi_EDGE | +0.2988 | −0.0951 | 0.2626 | +1.50 |
| R2 @ 8 | channel_drive | — | — | — | +1.17 |
| R2 @ 8 | pool_std_raw | 3.1041 | 3.4964 | 0.3202 | −1.23 |
| R2 @ 8 | n_hi_EDGE | 10 | 16.84 | 4.16 | −1.64 |
| R3 @ 32 | contrast | +0.4701 | +0.3657 | 0.1589 | +0.66 |
| R3 @ 32 | z_hi_EDGE | — | +0.0037 | 0.1184 | +0.85 |
| R3 @ 32 | z_hi_STAY | — | −0.3620 | 0.1298 | −0.02 |
| R3 @ 32 | raw_mean_hi | — | −0.9094 | 0.4587 | +0.10 |
| R3 @ 32 | channel_drive | — | — | — | +0.73 |

R2's native contrast sits at the **97.9th percentile** of its own n=8 law (the
97.5% quantile is +0.8501). **R2's single realized 8-episode diagnostic buffer
is a favourable outlier on three correlated high-risk statistics at once, from
an unusually small high-risk sample (10 hi&EDGE entries) with an unusually
narrow normalisation pool. R3's realized buffer is unremarkable on all of them.**
The "collapse" is regression to the mean of an unstable instrument.

This independently confirms and quantifies **limitation #2 of the R3 report**
(the 8-episode measurement buffer is an unstable instrument) and it retracts the
provisional batch-size mechanism.

### 4.4 The endpoint update content orders the arms OPPOSITE to their behaviour

R3's population contrast (+0.3723) is **larger** than R2's (+0.2312); frac > 0
among n=32 draws is 0.9865 (R3) vs 0.9315 (R2). Honest significance: the
superpopulation sd at n=128 is sd@32 × 0.5 → 0.0728 (R2) and 0.0795 (R3),
difference sd 0.1078, gap 0.1411 → **1.31 σ. Not significant.**

Calibrated claim: **the update content is not measurably worse for R3, and
certainly not worse by enough to explain a behavioural gap whose CIs do not
overlap.** This replicates the Rung 3 pattern — gradient-mass ratios ordering the
arms opposite to their behaviour — on an independent statistic. Caveat: measured
on each arm's own final policy and own state distribution, so it is not a matched
comparison.

### 4.5 Within-update mixing (C4) — candidate 2

| | A0 | R2 | R2_b32 | R3_best | R3 |
|---|---|---|---|---|---|
| episodes | 8 | 8 | 32 | 32 | 32 |
| start-tick span | 311 | 311 | 465 | 465 | 465 |
| hi&EDGE per episode, mean | 1.75 | 1.25 | 2.156 | 2.344 | 2.313 |
| … sd | 2.11 | 1.56 | 1.72 | 1.91 | 2.10 |
| frac episodes with 0 hi&EDGE | 0.250 | 0.375 | 0.1875 | 0.1875 | 0.219 |
| **hi&EDGE mass participation ratio** | 3.63 | 2.15 | 14.42 | 14.64 | **14.86** |
| PR / n | 0.454 | 0.269 | 0.451 | 0.458 | **0.464** |
| between-episode sd of mean z | 0.2814 | 0.2000 | 0.2206 | 0.2820 | 0.2260 |

The participation ratio `PR = (Σmₑ)²/Σmₑ²` measures how many episodes
effectively carry the high-risk gradient mass. **R3's is the highest of any cell
both absolutely (14.86) and normalised (0.464).** R3's high-risk signal is spread
across *more* episodes, not concentrated into fewer. There is no
temporal-mixing pathology in R3; if anything R3's mixing is the healthiest
measured. **Candidate mechanism 2 finds nothing that favours R2.**

---

## 5. PPO minibatch / reuse analysis (candidate 3)

### 5.1 Exact arithmetic

`mappo.py:368-380`: `n_mb = max(1, cfg.minibatches)`; `mb_size = max(1, T//n_mb)`;
`rng = np.random.default_rng(0)` **re-seeded every `update()`**; per epoch
`order = rng.permutation(T)`; `for start in range(0, T, mb_size)`.

With `minibatches = 4`, `mb_size = T//4`, so `range(0, T, T//4)` yields **5
chunks whenever T mod 4 ≠ 0**, and the fifth chunk has exactly **T mod 4** rows.

| | A0 | R2 | R2_b32 | R3_best | R3 |
|---|---|---|---|---|---|
| T | 3100 | 3200 | 12684 | 12552 | 12554 |
| mb_size | 775 | 800 | 3171 | 3138 | 3138 |
| **T mod 4** | 0 | 0 | 0 | 0 | **2** |
| chunks per epoch | 4 | 4 | 4 | 4 | **5** |
| minibatches per update | 16 | 16 | 16 | 16 | **20** |
| degenerate tail | no | no | no | no | **yes, 2 rows** |
| timestep rows per update | 12400 | 12800 | 50736 | 50208 | 50216 |
| **grad passes per env decision** | **4** | **4** | **4** | **4** | **4** |
| hi&EDGE per minibatch, mean | 3.5 | 2.5 | 17.25 | 18.75 | 14.8 |
| frac mb with 0 hi&EDGE | 0.0625 | 0.0625 | 0 | 0 | **0.20** |
| \|adv\| share hi&EDGE per mb | 0.0036 | 0.0019 | 0.0028 | 0.0061 | 0.0028 |

**Gradient passes per environment decision is 4 for every arm.** Sample reuse is
identical; the larger buffer buys no extra reuse.

### 5.2 R3's `frac_mb_zero_hi_EDGE = 0.20` is a tail artifact, not dilution

The per-minibatch dump settles this: the four minibatches with zero high-risk
EDGE entries are **exactly** the four `size_t = 2` chunks (indices 4, 9, 14, 19 —
one per epoch). The 16 real minibatches carry 14–23 hi&EDGE entries each and
**never zero**. Two rows out of 12,554 cannot contain a high-risk EDGE decision
by arithmetic, not by dilution.

I also checked for a 0/0 NaN bug and found none: minibatch 19 has `decision = 0`,
but `denom = d_mb.sum().clamp(min=1.0)` (`mappo.py:384`) guards it. Adam does
still take a momentum-carried step on the resulting zero gradient. That is a
real perturbation, but it is **shared and non-differential** — it is a property
of `T mod 4`, which is an accident of episode lengths, not of
`rollout_episodes`. It is recorded as a confound, not offered as a mechanism.

### 5.3 Correction to my own earlier tail reading

I earlier reported tail sizes 775/800/3171/3138/2 with tail/T = 0.25 for four of
five cells. **The 0.25 values are the last FULL chunk, not a tail.** A degenerate
tail exists only when T mod 4 ≠ 0 and is then exactly `T mod 4` rows. Of the five
diagnostic buffers only R3's had one.

### 5.4 Trajectory comparison, not endpoints

`clip_frac`, `approx_kl` (k1), entropy, policy loss, value loss and explained
variance are compared as complete 75-update trajectories in §3.1, not as
endpoints. `approx_kl` is reported throughout as **k1 = mean(old_lp − new_lp)**,
which is why it goes negative (26 updates for R2, 21 for R3): k1 is an unbiased
but signed estimator. k3 is unavailable as a production trajectory (R3 report
limitation #7 is accurate about the logs); it *is* available offline at the three
D4 snapshots per arm, where R2 has k1 −0.000073 with 11/20 minibatches negative
against k3 0.000000, and R3 has k1 −0.000025 with 8/20 negative against k3
0.000001. That is a wording clarification of limitation #7, not a contradiction
of it.

**Candidate mechanism 3 yields no differential mechanism.** Minibatch sizes,
sample reuse, tail behaviour and within-update movement are either identical or
differ by a shared arithmetic accident.

---

## 6. Critic / GAE analysis (candidate 4)

### 6.1 The confound in the own-rollout numbers

Explained variance on each arm's **own** rollout (C2):

| | A0 | R2 | R2_b32 | R3_best | R3 |
|---|---|---|---|---|---|
| all entries | 0.8589 | 0.7808 | 0.7497 | 0.7214 | 0.7056 |
| hi (≥0.6) | 0.9003 | 0.8155 | 0.7924 | 0.7175 | **0.6083** |
| lo (<0.2) | 0.8543 | 0.7814 | 0.7467 | 0.7211 | 0.7113 |
| hi_EDGE | 0.8928 | 0.1817 | 0.5005 | 0.4108 | 0.5049 |
| mean(ret − V) at hi | +0.4264 | −1.3737 | −0.2572 | +2.1958 | −0.9295 |
| mean\|ret − V\| at hi | 3.4830 | 4.3820 | 4.1874 | 5.4705 | 5.4307 |
| mean\|TD residual\| at hi | 0.3097 | 0.3445 | 0.3436 | 0.3743 | 0.4035 |

R3 is the only arm whose high-risk ev is *below* its low-risk ev. But each arm
was scored on its own rollout, so **critic quality and state distribution are
confounded**. The `R2_b32` cell shows the inversion is not created by buffer
*size*; it does not show it is not created by buffer *content*.

### 6.2 The identifying 2×2 (EXPLORATORY)

`SPRINT_7_DIV_critic_main.json`. Both critics scored against the **same**
`compute_mc_returns` target on the **same** states, for both buffers, with
**buffer size held at 32 for both cells** so the estimator's sampling variance
(§4.2) is matched and cannot masquerade as a critic effect.

The target is legitimately shared because `compute_mc_returns`
(`mappo.py:310-338`) is a function of `buf.rew`, `buf.cont` and `buf.trunc` only,
plus `buf.boot` at genuine time-limit truncations — and `boot` enters through the
*collecting* agent, so it is identical for both critics within a column.
(Truncations: 26/32 in R2's column, 22/32 in R3's.)

**High-risk explained variance:**

| critic | buffer R2@32 | buffer R3@32 |
|---|---|---|
| R2 | 0.7924 | 0.6171 |
| R3 | 0.7834 | 0.6083 |

**Down each column the two critics differ by 0.009.** R3's low high-risk ev
(0.6083) is because R3's high-risk *returns are harder to predict* — R2's critic
does essentially the same on them (0.6171), and R3's critic does essentially as
well as R2's on R2's buffer (0.7834 vs 0.7924).

Column detail — **buffer R2** (T=12684, decision 22098, hi 1069, lo 20997,
hi_EDGE 69): ev all_entries 0.7078/0.5785, decision 0.7497/0.6980, lo
0.7467/0.6920, hi_EDGE 0.5005/0.3563, lo_EDGE 0.4478/0.2106; mean(target−V) at
hi −0.2572 (R2) vs +1.4116 (R3); target hi−lo = −0.0879, V hi−lo +0.8482 (R2) /
+1.1041 (R3). **Buffer R3** (T=12554, decision 19301, hi 847, lo 18420, hi_EDGE
74): ev all 0.4884/0.6575, decision 0.5372/0.7056, lo 0.5325/0.7113, hi_EDGE
0.4212/0.5049, lo_EDGE 0.3940/0.5411; mean(target−V) at hi −1.2666 (R2) /
−0.9295 (R3); target hi−lo = **+1.4983**, V hi−lo +3.4815 (R2) / +2.6240 (R3).

> **Candidate mechanism 4's specific hypothesis — "the critic systematically
> misvalues high-risk states" — is ruled out as an R2-vs-R3 differential.
> High-risk explained variance is a property of the buffer, not of the critic.**

Note the target itself differs sharply between columns (hi−lo = −0.0879 on R2's
buffer vs +1.4983 on R3's). The two arms' policies place them in genuinely
different high-risk return regimes; that is a real difference, but it is
downstream of the policy, not a critic defect.

### 6.3 The finding that survives: home advantage is concentrated in the low-risk bulk

Cluster bootstrap, 10,000 replicates, **clusters = episodes**, both critics
paired within each replicate. Explained variance is a ratio of sums of squares,
so per-episode `(n, Σe, Σe², Σt, Σt²)` are sufficient statistics and a replicate
is a resample of the 5-tuples — exact, with no replicate array materialised.

Home advantage = ev(own critic) − ev(other critic), on the arm's own buffer:

| owner | Δev at hi | 95% CI | frac>0 | Δev at lo | 95% CI | frac>0 | **lo − hi** | 95% CI | frac>0 |
|---|---|---|---|---|---|---|---|---|---|
| R2 | **+0.0090** | [−0.0107, +0.0335] | 0.797 | **+0.0547** | [+0.0231, +0.0933] | 0.9997 | **+0.0457** | [+0.0131, +0.0823] | 0.9963 |
| R3 | **−0.0089** | [−0.0576, +0.0394] | 0.344 | **+0.1788** | [+0.1192, +0.2441] | 1.0000 | **+0.1876** | [+0.1304, +0.2541] | 1.0000 |

Also: `decision` +0.0517 [+0.0216,+0.0880] (R2) and +0.1684 [+0.1111,+0.2312]
(R3); `hi_EDGE` +0.1443 [+0.0140,+0.3413] over 26 clusters (R2) and +0.0837
[−0.0289,+0.2523] over 25 clusters (R3); `lo_EDGE` +0.2372 (R2) / +0.1471 (R3).

**R3's critic gained a large, unambiguous advantage over R2's in its own
low-risk bulk (+0.179, CI excludes zero, 100% of replicates) and gained nothing
whatsoever at high risk (−0.009, CI straddles zero, only 34% of replicates
positive).** R2's critic at least has a small positive high-risk home advantage.
The lo−hi difference-of-differences is **4.1× larger for R3**, and its CI excludes
zero for both arms.

---

## 7. Candidate mechanism ranking

Ordered by what the evidence supports, not by prior plausibility.

### 7.1 RULED OUT — candidate 1 (update content / data distribution)

`rollout_episodes` 8→32 shifts the expected value of every measured update
statistic by **≤ 0.105 σ₈** while the sd tracks the finite-population null at
**observed/predicted 1.005–1.057**. It is a pure variance intervention. The one
result that looked like a content mechanism was a **+2.09 σ outlier** of R2's
own 8-episode instrument (§4.3). Where the endpoint content does differ, it
orders the arms **opposite** to their behaviour, and only by 1.31 σ (§4.4).

### 7.2 RULED OUT — candidate 2 (within-update temporal mixing)

R3's high-risk gradient mass is spread over **more** episodes than any other
cell (participation ratio 14.86, PR/n 0.464 — the highest measured). Between-
episode sd of mean z is comparable across arms. There is no dilution beyond the
simple ‖g_hi‖/‖g_full‖ decomposition, and what mixing difference exists favours
R3.

### 7.3 RULED OUT — candidate 3 (PPO epoch / minibatch reuse)

Gradient passes per environment decision = **4 for every arm**. R3's
`frac_mb_zero_hi_EDGE = 0.20` is exactly its four `size_t = 2` degenerate tail
chunks, a shared `T mod 4` confound with `denom` clamped so no NaN occurs (§5.2).
Cumulative within-update movement is near-identical (Σ|k1| 0.1533 vs 0.1528).

### 7.4 RULED OUT AS A DIFFERENTIAL — candidate 4 (critic level)

Matched 2×2: the two critics differ by **0.009** in high-risk explained variance
down each column. R3's low high-risk ev is a property of its buffer's returns.

### 7.5 PARTIALLY LOCALIZED — candidate 5 (update-level causal localization)

The critic-quality divergence (`explained_var`, onset **34**, opening across
updates 31–45) **precedes** the reward divergence (onset **47**). `decision_frac`
diverges earlier (onset **25**). R3's endpoint sits at R2's **update 25–27** on
actor-side metrics and **44–70** on critic-side metrics. So R3 is not a uniformly
earlier R2: **its actor is far more behind than its critic.** This localizes
*when* and *where*, but by itself it does not identify a mechanism.

### 7.6 WHAT SURVIVES: a regime-selective learning asymmetry

The one difference that survives every ruling-out is not in the update content,
not in the estimator, and not in critic *level*. It is that:

> **R3's learning was better in the low-risk bulk and no better at all in the
> high-risk minority — in the actor AND in the critic — while the optimiser ran
> the same step budget under the same LR schedule with strictly less gradient
> noise.**

Two statistically independent instruments say the same thing.

**Instrument 1 — the actor, in function space.** Displacement from the shared θ₀
projected onto R2's displacement direction, on the unchanged fixed sets. Cluster
bootstrap, 10,000 replicates, clusters = episodes:

| set | arm | proj_frac hi | proj_frac lo | hi − lo | 95% CI | frac < 0 |
|---|---|---|---|---|---|---|
| UNION | R3 | 0.2147 | 0.7124 | **−0.4977** | [−0.5152, −0.4811] | **1.0000** |
| UNION | R3_best | 0.0646 | 0.4888 | −0.4241 | [−0.4390, −0.4102] | 1.0000 |
| UNION | A0 | 0.3458 | 0.5965 | −0.2507 | [−0.2953, −0.2070] | 1.0000 |
| RANDOM | R3 | 0.1599 | 0.6858 | **−0.5260** | [−0.6822, −0.4209] | **1.0000** |
| RANDOM | R3_best | 0.1677 | 0.5617 | −0.3940 | [−0.5143, −0.2812] | 1.0000 |
| RANDOM | A0 | 0.4542 | 0.6992 | −0.2450 | [−0.5138, −0.0386] | 0.9922 |

**The ordering inverts between regimes.** R3 is *closer* to R2 than A0 is in the
low-risk bulk (0.7124 vs 0.5965 on UNION) and *farther* from R2 than A0 is at
high risk (0.2147 vs 0.3458). Same sign, same inversion, on both sets. This is
not "R3 is an earlier R2 that walked the same path less far" — that would give an
equal proj_frac in both regimes. R3 tracked the majority direction faithfully and
under-moved specifically where the minority signal lives.

**Instrument 2 — the critic, in return space.** §6.3: home advantage +0.1788 at
low risk (CI [+0.1192,+0.2441], 100% of replicates) versus −0.0089 at high risk
(CI [−0.0576,+0.0394], 34%); lo−hi = +0.1876 [+0.1304, +0.2541].

These two measurements share no statistics, no state set, and no estimator. The
actor measurement uses policy probabilities on RANDOM/UNION; the critic
measurement uses MC returns on on-policy buffers. They agree.

**Supporting, EXPLORATORY (E2) — the shape of the actor's high-risk
displacement.** Decomposing high-risk displacement into a uniform shift plus a
state-varying residual, `uniform_share = ‖u‖√n/‖Δp‖`:

| set | quantity | θ₀ | A0 | R2 | R3_best | R3 |
|---|---|---|---|---|---|---|
| UNION | arg-max → EDGE at hi (/3592) | 830 | 628 | **1136** | 709 | **0** |
| UNION | uniform_share at hi | — | 0.7213 | 0.5948 | 0.5675 | **0.7562** |
| RANDOM | arg-max → EDGE at hi (/216) | 18 | 18 | **100** | 72 | **0** |
| RANDOM | uniform_share at hi | — | 0.7035 | 0.3815 | 0.4218 | **0.6799** |

R2 **re-ranks** actions at high-risk states — it makes MIGRATE_EDGE the greedy
choice at 1136/3592 UNION high-risk states (31.6%), with a low uniform_share
meaning the change is state-specific. **R3's final policy makes EDGE the arg-max
at zero high-risk states on either set**, with the highest uniform_share of any
arm: R3 raised P(EDGE) roughly uniformly without ever re-ranking. R3_best (update
45) had 709 UNION flips and uniform_share 0.5675 — so **this channel collapsed
between update 45 and update 75**, which is bracketed but not localized (§9).
Note this must be read as *R3_best vs R3_final*, not as post-peak decay (§2.4).

**Parameter-space context (Part A):** ‖Δθ_actor‖ A0 5.5321 / R2 6.1710 /
R2_best 6.1710 (bit-identical) / R3 4.5471 / R3_best 3.9510; ‖Δθ_critic‖ 9.7844 /
9.9936 / 9.9936 / 10.3278 / 9.1554. Projection onto R2's actor direction: A0
proj_frac 0.1672 (cos 0.1865), R3 0.1244 (cos 0.1689), R3_best 0.0990 (cos
0.1546). cos(R3, R3_best) = 0.9300. All cross-arm actor cosines are 0.155–0.190
against a chance cosine of 0.00207 — small in absolute terms but 75–92× chance,
so the arms are not moving in unrelated directions. Adam actor state at the
final checkpoint: steps 1428/1392/1440/872, ‖m‖ 0.0756/0.0187/0.0637/0.0446,
‖update‖ 0.000746/0.000395/0.001199/0.013268.

**D4 final-snapshot gradient geometry** (actor-side quantities exact; see §9 for
the critic-loss replica caveat): `cos(g_full, g_hiEDGE)` A0 −0.2432, R2 +0.1964,
R3_best +0.4579, R3 **−0.5469**; `cos(g_hiEDGE, g_loEDGE)` R3_best +0.4330 → R3
−0.1740; `cos(g_full, g_loEDGE)` R3_best 0.0675 → R3 0.6683; hi_EDGE share of
aggregate gradient norm R3_best 0.5657 → R3 0.2141 (R2 0.1386, A0 0.3296);
‖g_all‖ 0.0556/0.0345/0.0169/0.0256. At its endpoint R3's aggregate gradient is
strongly aligned with the low-risk EDGE direction and strongly *anti*-aligned
with the high-risk EDGE direction. This is consistent with the surviving finding
and is its most direct gradient-level expression — but it is a single snapshot,
so it describes the endpoint rather than the path.

**This is a "what differs" statement, supported by two independent instruments.
It is not a causal claim and cannot be made one offline.** Establishing causality
would require training, which is out of scope.

---

## 8. What is ruled out

1. **Update content / data distribution as a mean effect.** `rollout_episodes`
   does not shift what the actor is trained on: ≤ 0.105 σ₈ on seven statistics
   (§4.2). *Decisive.*
2. **The "5.1× collapse / sign flip" as a batch-size mechanism.** It is a +2.09 σ
   draw from R2's own n=8 sampling law (§4.3). *Retracted, decisively.*
3. **Update content as an explanation in the direction required.** Where the
   endpoint content differs it favours **R3** (+0.3723 vs +0.2312), i.e. opposite
   to behaviour — and only at 1.31 σ (§4.4).
4. **Within-update temporal mixing / extra dilution.** R3's participation ratio
   is the highest measured (§4.5).
5. **Sample reuse.** 4 gradient passes per environment decision for every arm
   (§5.1).
6. **The degenerate minibatch tail as an R3-specific pathology.** It is `T mod 4`
   = 2 rows, a shared confound, and it fully accounts for R3's
   `frac_mb_zero_hi_EDGE = 0.20` (§5.2). No NaN occurs (`denom` clamp).
7. **"The critic systematically misvalues high-risk states" as a differential.**
   Matched 2×2: 0.009 apart down each column (§6.2). *Decisive.*
8. **A NaN or divide-by-zero fault in the update path.** Checked; guarded.
9. **Total policy movement as the difference.** Σ|k1| agrees to 0.3% and R3 took
   *more* Adam steps (§3.2).
10. **Gradient noise as R3's handicap.** R3 had strictly *less* estimator noise
    (sd halved per the FPC law) and still did worse. Any mechanism proposed in
    future must be compatible with that inversion.
11. **A configuration or seeding confound.** 1 substantive field of 99;
    bit-identical LR schedules; nested, byte-identical episode streams (§2.1–2.2).

---

## 9. What remains unidentified

Stated plainly, per the brief's instruction to say so rather than force a
conclusion.

**The mechanism is not identified.** §7.6 establishes *that* R3's learning was
regime-selective — good in the low-risk bulk, absent in the high-risk minority,
on two independent instruments. It does not establish *why*. Specifically, none
of the following can be settled from the existing artifacts:

1. **Direction of causation between actor and critic.** `explained_var` diverges
   at update 34 and reward at 47, which is *consistent* with a critic-first
   story, but a 13-update lag between two correlated training metrics on one seed
   is not identification. The critic's regime-selective home advantage could be a
   cause of, a consequence of, or a co-symptom with the actor's.
2. **Why the low-risk direction dominated in R3 but not R2**, given that the
   high-risk *content* was statistically indistinguishable and R3's high-risk
   signal was spread over *more* episodes. The obvious candidate — that a
   lower-noise estimator lets the majority direction be followed more precisely,
   and the majority direction is the wrong one for this channel — is a
   **hypothesis that this diagnostic did not test and cannot test offline.**
3. **When the arg-max channel collapsed.** It is bracketed to updates 45–75
   (R3_best has 709 UNION high-risk flips, R3_final has 0) but not localized,
   because **no intermediate per-update checkpoints exist** — only `final` and
   `_best`.
4. **Whether R2 would also degrade past update 75.** `R2_best` is bit-identical
   to `R2`, so R2 has exactly one distinct policy snapshot and cannot be examined
   at update 45. It is entirely possible R2's channel would collapse too if run
   longer. Settling this requires training and is out of scope.
5. **Whether any of this generalizes beyond one seed.** Everything here is n = 1
   run per arm.

### Limitations of this diagnostic

- **EXPLORATORY vs pre-registered.** The following are **exploratory**, designed
  after seeing earlier results, and set no threshold and change no locked
  verdict: E2 (§7.6 shape decomposition), the variance probe (§4.2–4.4), the
  matched-buffer critic probe (§6.2–6.3), the `infeasible` contention proxy
  (§3.6), and the sustained-onset rule (§3). The E1 bootstrap CIs (§7.6
  instrument 1) are reported so the part-2 finding can be judged, not to license
  a new threshold. Everything in §2 (invariants) is verification of locked facts.
- **`--episodes` is IGNORED by the D3/D4 census.** Independently reproduced:
  `diag_S7_D4_ppo_update_R2_b32.json` is **bit-identical** to
  `diag_S7_D4_ppo_update_R2_mc_target.json` in every field (n_dec 5770, n_hiE 8,
  ‖g_all‖ 0.0345, all cosines, k1_mean −0.000073, fidelity 3.905). Root cause:
  those probes read `rollout_episodes` from the checkpoint config. The
  matched-batch control therefore lives in `_diag_div_content.py`, which passes
  `n_eps` straight to `build`. **The four `*_b32.json` census artifacts produced
  this phase are duplicates and must not be read as a batch-size control.**
- **D4 replica fidelity is critic-side only, and only for the mc-target arms.**
  Per-key breakdown confirms the mismatch is entirely `critic_loss` (7.459/3.905
  R2; 9.988/5.599/5.280 R3) and `explained_var` (≤0.153). A0 (lambda target) is
  `ok=True, max 0.0000` at all three snapshots. **All actor-side quantities in D4
  are exact.**
- **Own-distribution quantities** (§4.1, §5.1, §6.1) carry estimator noise and
  confound policy with state distribution. They are labelled as such and are not
  compared against P1/P2.
- **The matched-batch content control exists only for R2.** There is no matched
  *8-episode* cell for R3's parameters.
- **The shared MC target retains one critic dependence** — `boot` at genuine
  time-limit truncations — which enters through the collecting agent and is
  therefore identical for both critics within a column. Truncation counts (26/32
  and 22/32) are reported so its weight can be judged.
- **One seed throughout.**

---

## 10. Exact evidence required before any future training rung

**No rung is proposed here.** This section states what would have to be measured
for the surviving finding to become causal, so that the standard is fixed in
advance rather than after seeing results. It is not a recommendation to train.

For the §7.6 asymmetry to be promoted from "what differs" to "what explains",
the following would be needed:

1. **Per-update checkpoints, not just `final` and `_best`.** The single largest
   gap in this diagnostic. Every question in §9 items 1 and 3 is unanswerable
   without a policy snapshot at each update (or at least every 5). Cost is small;
   information gain is decisive.
2. **The regime-selective measurement computed *during* training, per update,**
   on the frozen RANDOM/UNION sets: proj_frac at hi and at lo, arg-max→EDGE
   counts at hi, and the critic's home advantage by regime. The surviving finding
   is currently an endpoint measurement; as a trajectory it would show whether
   the asymmetry is present from the start, emerges gradually, or appears at a
   specific update.
3. **A per-update k3 KL trajectory in the production logs.** k1 is signed and
   goes negative; k3 exists offline at snapshots only. This is a cheap logging
   change that removes a known ambiguity.
4. **≥ 3 seeds per arm.** Every quantitative claim in this report is n = 1 per
   arm. The bootstrap CIs quantify *within-run* episode sampling, not *between-
   run* variability. A 1.31 σ gap (§4.4) and a 13-update onset lag (§3.1) are
   exactly the kind of result that a second seed can erase.
5. **An R2 arm trained past update 75.** Required to distinguish "R3 collapsed
   the high-risk channel" from "both arms collapse it eventually and R3 got there
   sooner in wall-clock". Without this, §9 item 4 stands.
6. **A control that separates estimator noise from buffer size.** The finding to
   be explained is that R3 had *less* gradient noise and did *worse*. Any
   intervention proposed in future must be able to move noise without moving
   anything else, or the same confound recurs.
7. **The 8-episode instrument must not be used for high-risk measurement.**
   §4.3 shows it produces +2 σ excursions on the statistics that matter, from
   ~10 high-risk EDGE entries. Any future high-risk quantity should be measured
   on ≥ 32 episodes, with its sampling law reported alongside the point estimate.

---

## Appendix — artifacts produced by this phase

**Probes** (`python-ai/marl/`, all additive, all offline, all independently
re-runnable):

| file | part | what it establishes |
|---|---|---|
| `_diag_div_logs.py` | 1 | update-aligned / experience-aligned / matched-pair divergence, onsets, progress clock |
| `_diag_div_geometry.py` | 2 | parameter-space displacement from shared θ₀; function-space displacement on the fixed sets |
| `_diag_div_shape.py` | 3 | E1 cluster-bootstrap CIs on proj_frac; E2 uniform-shift vs re-ranking (exploratory) |
| `_diag_div_content.py` | 4 | C1–C5 update content, including the matched-batch control the census cannot run |
| `_diag_div_variance.py` | 5 | sampling law of the high-risk signal at 8 vs 32 episodes (exploratory) |
| `_diag_div_critic.py` | 6 | matched-buffer 2×2 critic comparison + home advantage by regime (exploratory) |

**Data** (`python-ai/saved_models/marl/`): `SPRINT_7_DIV_logs_R2vsR3.json`,
`SPRINT_7_DIV_geometry_partA.json`, `SPRINT_7_DIV_geometry_eval.json`,
`SPRINT_7_DIV_geometry_full.json`, `SPRINT_7_DIV_shape_main.json`,
`SPRINT_7_DIV_content_main.json`, `SPRINT_7_DIV_variance_main.json`,
`SPRINT_7_DIV_critic_main.json`, plus four `diag_S7_D?_*_b32.json` census files
(**duplicates — see §9**).

**Logs** (`python-ai/`): nine `SPRINT_7_DIV_*.log`.

**Manifests** (`python-ai/saved_models/marl/_DIVERGENCE_integrity/`): four
`*_before.md5` (pre-phase), four `*_after.md5` (post-phase), three `*_new*.md5`
(additive only), one `*_prior_manifests.md5`.
