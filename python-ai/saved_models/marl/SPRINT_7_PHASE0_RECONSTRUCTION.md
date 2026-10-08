# Sprint 7 — Phase 0 (Reconstruct) + Phase 1 (Verify) + Phase 2 (Next causal question)

**Date:** 2026-08-30 · **Author:** reconstruction audit · **Status:** no training, no
production-code change, no new checkpoint, no R4 created, RANDOM/UNION not redefined.

New files produced by this phase (both additive):

| file | role |
|---|---|
| `marl/diag/_phase0_calibration.py` | read-only probe, recomputes everything below |
| `saved_models/marl/SPRINT_7_PHASE0_calibration.json` | machine-readable results |
| `saved_models/marl/SPRINT_7_PHASE0_RECONSTRUCTION.md` | this report |

---

## 1. Phase 0 — state table

### 1.1 Arms

| arm | checkpoint | manipulation vs its control | updates | final Δ RANDOM | final Δ UNION |
|---|---|---|---|---|---|
| A0 | `mappo_A0_cpu_repro.pth` | baseline, `critic_target=lambda` | 75 | −0.0264 | +0.0191 |
| A1 | `mappo_A1_cpu_bugfix.pth` | attribution bugfix | 75 | +0.1375 | +0.0628 |
| A2 | `mappo_A2_crit_sign.pth` | critic sign | 75 | +0.0398 | −0.0248 |
| A3 | `mappo_A3_entropy.pth` | `entropy_coef` 0.01→0.02 | 75 | +0.1322 | +0.0933 |
| **R2** | `mappo_R2_mc_target.pth` | **`critic_target` lambda→mc** | 75 | **+0.2024** | **+0.1579** |
| R3 | `R3_batch32.pth` | **`rollout_episodes` 8→32** | 75 | +0.1575 | +0.1242 |
| R3_best | `R3_batch32_best.pth` | update-45 snapshot of R3 | 45 | +0.1467 | +0.0946 |

`R2_best` is **bit-identical** to `R2` — R2 has exactly one distinct policy snapshot.
`R3_best` is the only genuine mid-training checkpoint in the repository.
Behavioural order on both fixed sets: **R2 > R3 > A0**.

### 1.2 Rung history and verdicts

| rung | date | question | verdict |
|---|---|---|---|
| Phase 1 diag | 08-19 | why does MARL lose to a risk threshold rule? | scoped Sprint 7 |
| Rung 0 | 08-20 | is the high-risk migration signal real, and does GAE see it? | signal **real** (+3.363 forced-EDGE true adv); GAE **anti-correlated** (29.0% sign agreement at risk>0.50) |
| Rung 1 | 08-25 02:36 | is the self-referential λ-return critic target causal? | **SUPPORTED for A0** (26.3%→60.7% sign agreement offline) |
| Rung 2 | 08-25 04:02 | production change `critic_target="mc"` | R2 trained; best arm to date |
| Rung 2.5 | 08-25 06:45 | is the *remaining* target dependence causal? | **NO-GO for a target-focused rung**; separated **three** mechanisms |
| Rung 2.75 | 08-25 13:08 | is the headline metric identified? what is the actor stall? | headline metric **not identified**; stall attributed to **signal variance** from the per-state offset |
| R3 | 08-25 18:22 | H1: is gradient variance / low SNR the binding constraint? | **NO-GO** — SNR rose (p=0.0000), behaviour got worse |
| Rung 3 | 08-27 13:16 | does low-risk dilution explain R3? | **SUPPORTED but DOES NOT EXPLAIN R3**; `go_for_training_rung = False` |
| Divergence | 08-30 11:02 | what R2-vs-R3 difference *could* explain the behaviour gap? | regime-selective learning asymmetry — explicitly **"what differs", not causal** |

### 1.3 Frozen production values (verified unchanged in source)

`gamma 0.999` · `gae_lambda 0.995` (horizon 1/(1−γλ) = 166.8 steps) · `clip_eps 0.2` ·
`value_clip_eps 0.2` · `entropy_coef 0.02` · `value_coef 0.5` · `max_grad_norm 0.5` ·
`ppo_epochs 4` · `minibatches 4` · `normalise_advantages True` ·
`critic_target "mc"` · `anneal_lr True` · `episodes 600` · `rollout_episodes 8` ·
`seed 20260818`. Production source last changed **08-25 04:02** (the Rung 2 target
change) and is byte-identical in all eight subsequent manifests and at HEAD.

---

## 2. Phase 1 — integrity

Git HEAD `a8df6d9` (2026-08-30 10:44:38 +0530) tracks every artifact, checkpoint, log and
diagnostic script; `git diff --stat HEAD` was empty at the start of this phase. This is a
**stronger baseline than MD5 alone** and supersedes the brief's assumption that the
repository might not be backed by a clean git baseline.

All divergence-phase before-manifests re-verified: **96/96, 41/41, 58/58, 3/3 OK**.
Across the full 14-manifest prior chain, 11 verified clean and **3 failed**. All three were
investigated rather than accepted, per the brief's instruction not to silently "fix" a report:

1. **`_rung0_integrity/code_before.md5`** — `config.py`/`mappo.py`/`train.py` fail.
   Hashes moved exactly once, between the Rung 0 manifest (08-19 21:35) and
   `rung2_5_code_before` (08-25 04:02) — i.e. *the Rung 2 MC-target production change* —
   and are byte-identical in every later manifest. **Documented and intended.**
2. **`_rung2_75_integrity/artifacts_after.md5`** — `SPRINT_7_RUNG2_75_REPORT.md` fails.
   One post-manifest edit to a *report*, captured by the next phase's before-manifest.
   A minor RULE-12 process slip; **no data artifact affected.**
3. **`_rung2_75_integrity/code_after.md5`** — `_diag_rung2_75_coherence.py` and
   `_diag_rung2_75_matched_states.py` fail. Both changed inside the R3 evaluation window.
   Because `_diag_rung2_75_matched_states.py` **constructs RANDOM and UNION**, this was not
   accepted on narrative grounds but proved empirically: all 10 shared arm×source values and
   all four state-set sizes are bit-identical across the two script versions,
   **MAX |diff| = 0.000e+00**. The edit was purely additive (it added the `R3` column).

**One source discrepancy found and deliberately NOT fixed** (production code is frozen):
`mappo.py:447` comments that "GAE's own averaging horizon is only 1/(1 − gamma*lambda) ~ 20
steps". That is correct for the retired λ=0.95 (19.6) and wrong at the current λ=0.995
(166.8). `config.py:434` records the correct value. Comment-only; no code effect.

Reproduced from artifacts: the Rung 3 verdict and all five R3 pre-registered criteria; Rung
2.5's 130/130-zero-censored and 58.4%→7.8% (87%) target-contamination result; the
R2-vs-R3 config diff (99 fields, 5 differing, only `rollout_episodes` substantive, with
`episodes` 600→2400 the dependent change holding updates at 75 = 600/8 = 2400/32, verified
as 75 rows in both `_updates.csv`).

---

## 3. The central Phase 1 finding — every mechanism instrument orders the arms backwards

RULE 9: *"If a metric fails to correlate with known behaviour across existing arms, demote
it."* No Sprint 7 report applies this test to its own mechanism instruments. Applying it to
all of them, against the pre-registered primary metric
`Δ = π(EDGE|risk≥0.6) − π(EDGE|risk<0.2)` on the fixed policy-independent sets:

| instrument | rung | measured on | arms | ρ vs Δ(RANDOM) | ρ vs Δ(UNION) |
|---|---|---|---|---|---|
| E1 proj_frac hi−lo | Divergence | RANDOM | 2 | **−1.000** | **−1.000** |
| E1 proj_frac at hi | Divergence | RANDOM | 2 | **−1.000** | **−1.000** |
| cos(g_full, g_synth) | Rung 3 | RANDOM | 3 | **−1.000** | **−1.000** |
| cos(g_hi, g_lo) | Rung 3 | RANDOM | 3 | −0.500 | −0.500 |
| E1 proj_frac hi−lo | Divergence | UNION | 2 | **−1.000** | **−1.000** |
| E1 proj_frac at hi | Divergence | UNION | 2 | **−1.000** | **−1.000** |
| cos(g_full, g_synth) | Rung 3 | UNION | 3 | **−1.000** | **−1.000** |
| cos(g_hi, g_lo) | Rung 3 | UNION | 3 | **−1.000** | **−1.000** |
| ‖g_hi‖/‖g_lo‖ | Rung 3 D4 | OWN | 3 | **−1.000** | **−1.000** |
| cos(g_full, g_synth) | Rung 3 D4 | OWN | 3 | −0.500 | −0.500 |

**20 readings. 20 negative. 0 positive. Mean ρ = −0.900.**

Rung 3 noticed two instances of this ("`reproduces = False`", "the mass ratios order the arms
opposite to behaviour") and correctly returned NO-GO. What it did not do — and what nothing
in the corpus does — is observe that **the property is universal**, and therefore diagnostic
of the measurement apparatus rather than of any one hypothesis.

### 3.1 Why they all invert — one structural cause

Every one of these instruments is evaluated at a **single terminal checkpoint**. A terminal
gradient (or a terminal displacement projected on another arm's direction) measures the
*signal still left uncorrected* at that point — i.e. **remaining headroom**. Headroom is
anti-correlated with achieved behaviour by construction:

- **A0 never learned risk-awareness**, so its endpoint gradient still points strongly at the
  risk-aware direction: `cos(g_full, g_synth) = +0.4023` (OWN), the best of any arm — and its
  behaviour is the worst (Δ = −0.0264 / +0.0191).
- **R2 learned the most**, so its endpoint gradient points *away*: **−0.2372** — and its
  behaviour is the best (+0.2024 / +0.1579).

The same inversion drives the divergence phase's E1 result, where A0 has the *smallest*
regime inversion (−0.2507) and the *worst* risk response. **Under RULE 9 the divergence
phase's surviving regime-inversion finding must be demoted**, and so must Rung 3's cosine and
mass-ratio channels, as instruments for choosing between mechanisms.

**Honest limits on this claim.** Each instrument has only 2 or 3 arms, so no single ρ is
significant (permutation p ≥ 1/6 at n=3). The 20 readings share arms, share the terminal-checkpoint
construction, and reuse two state sets, so they are **not independent** and the sign count must
**not** be read as p = 2⁻²⁰. R2 is structurally unscorable on E1 because it *is* the reference
direction — the instrument can never be calibrated on its own best arm. And A0 differs from R2 in
critic target, not batch size, so part of its misalignment is expected. The claim is therefore:
*no instrument in the corpus has been shown to track behaviour, and every one that can be checked
points the wrong way.* That is enough to block their use for mechanism selection; it is not a
quantified effect size.

---

## 4. Mechanism 2 (the per-state offset) is closed, and my own successor hypothesis fails

Rung 2.5 separated three mechanisms and set a redirect priority. Tracing what happened to each:

| mech | name | status |
|---|---|---|
| 1 | critic target formulation | **CLOSED.** Real and severe for A0; 87% fixed by R2; remaining headroom ≤3.5pp, at chance. |
| 2 | per-state critic value calibration | **CLOSED — see below.** |
| 3 | actor softmax saturation | **DOWNGRADED to a symptom** by R3. |

Mechanism 2's route into an experiment ran through Rung 2.75, which decomposed
`gae(s,a) = c(s) + paired(s,a)` with `c(s) := gae(s, a_ref)` and `a_true(s,a_ref) ≡ 0`, found
`Var(raw)/Var(paired) = 3.72×`, argued that *a per-state constant contributes zero in
expectation so the offset is variance and not bias*, and concluded that **3.72× is "the exact
data multiplier needed"**. R3 supplied 4×, measurably raised gradient SNR
(real/shuffled 2.03, p = 0.0000), and made behaviour **worse**. The divergence phase then
showed `rollout_episodes` shifts expected update content by ≤0.105 σ₈ — a pure variance lever.

**So R3 falsifies mechanism 2, not merely "H1 variance".** Mechanism 2's only causal pathway
was variance; that pathway was supplied in excess and the outcome moved the wrong way. No
report states this, because R3 was framed as testing variance rather than the mechanism whose
decomposition produced it.

### 4.1 A successor hypothesis, tested offline and NOT supported

The load-bearing premise above ("zero in expectation ⇒ pure variance") holds for the vanilla
policy gradient but **not** for PPO's clipped surrogate, whose branch is selected by
`sign(A)`. If the clip were routinely active, a sign-flipping offset would become a genuine
bias — and because `c(s)` is risk-dependent, a **regime-selective** bias, which would explain
why more samples could not fix it. I tested the components before proposing any experiment.

The offset is unambiguously large enough to flip signs. On R2's own native rows (547 states,
716 forced replays, `replay_mismatches = 0`):

| bucket | n | mean c(s) | sd | frac \|c\|>1 | sign(raw) ≠ sign(paired) | median \|c\|/\|paired\| |
|---|---|---|---|---|---|---|
| lo | 290 | −0.8814 | 4.106 | 0.790 | **45.9%** | 4.14 |
| mid | 27 | +2.4009 | 4.057 | 0.852 | 46.7% | 3.50 |
| hi | 230 | −0.8695 | 5.190 | 0.883 | **56.7%** | 2.96 |

A single illustrative row: raw GAE for MIGRATE_EDGE is **−9.39** against a true advantage of
**−0.80**, because `c(s) = −8.59`; the paired value is −0.797, i.e. near-exact.

**But the clip is not active enough to convert that into a bias.** R2's `clip_frac` runs
0.1625 at update 1 and decays to zero: quartile means **0.0426 / 0.0196 / 0.0031 / 0.0000**,
with 22 of 75 updates at exactly zero. With ~4% of samples clipped in the first quartile and
none in the last, clipping cannot carry a mechanism that must operate throughout training.
The always-active term — pooled normalisation dividing by an offset-inflated SD,
√3.72 = **1.93×** attenuation of the genuine signal — is a *magnitude* effect, not a
direction effect.

**And the offset has no implementable online remover.** The paired estimator needs
`gae(s, a_ref)` — a counterfactual replay — so it cannot run inside a PPO update; Rung 2.75
said so. The natural online substitute is centring across agents at a tick (the critic is
centralised, `489+10 → 1`). That is now ruled out: on the 27 ticks where two selected agents
co-occur, the **within-tick spread of c(s) averages 6.65 (max 18.61) against an overall
SD(c) of 4.65** — two agents at the same tick differ *more* than two random states do, because
`V` is conditioned on the agent one-hot. Per-tick centring would add noise, not remove it.
(n = 27 groups is small; the effect size is what makes it decisive, and it is directional only.)

**Conclusion, reported as a negative result per RULE 10:** the clip-mediated
baseline-invariance hypothesis is **not supported**, and mechanism 2 has no surviving causal
pathway and no implementable intervention. It is closed. No training was spent to learn this.

---

## 5. What the corpus now establishes, and the one thing it cannot

**Established and not in doubt.** The premise of Sprint 7 holds under two independent
attributions and both risk conventions: at high risk, migrating is strongly better than
staying. Forced-replay counterfactual truth on R2's own states gives
STAY **−0.8822** (t = −3.99) vs MIGRATE_EDGE **+0.8675** (t = +3.71) own-attribution, and
+1.6227 (t = +3.78) team; Rung 0's A0 pool gives forced-EDGE true discounted advantage
**+3.363** at risk>0.50, and `frac_positive(EDGE > STAY)` of **0.7012** at risk>0.50 against
**0.3931** at risk<0.10. The learner's signal, once the per-state offset is removed, agrees
with that truth almost perfectly (paired sign agreement **0.9463** own, r = +0.9255). The
aggregated high-risk gradient points the right way in every arm
(`cos(g_hi, g_synth)` = +0.913 A0 / +0.536 R2 / +0.451 R3).

**Closed by direct measurement:** critic target formulation (mech 1); the per-state offset
(mech 2, §4); actor saturation as a cause (mech 3); gradient variance and `rollout_episodes`
(R3); high-risk dilution as an explanation of R3 (Rung 3); the critic as the R2/R3
differential (0.009 down each column of the matched 2×2).

**The one thing the corpus cannot do.** Every remaining hypothesis is a claim about the
**learning process** — about *when* and *why* the high-risk channel stopped moving. Every
instrument available is a **single-endpoint** measurement, and §3 shows endpoint quantities
measure headroom and order the arms backwards. The corpus therefore cannot distinguish
"never learned it" from "learned it then lost it", and the divergence phase said so
explicitly: R3's arg-max channel **collapsed somewhere between update 45 and 75**, and this
is *"bracketed, not localized, because no intermediate checkpoints exist"*. `R2_best` is
bit-identical to `R2`, so the best arm has **one** snapshot. The divergence report's own §10
names per-update checkpoints as **the single largest gap**.

This is not a missing hypothesis. It is a **missing observable**, and it is the reason three
consecutive rungs returned NO-GO.

---

## 6. Phase 2 — the next causal question

Evidence **does not support** any surviving single-mechanism hypothesis, and — more
importantly — evidence shows that the instruments used to choose between them are
**uncalibrated and invert against behaviour on all 20 available readings**. Proposing a
fourth mechanism now would be scored by the same broken apparatus.

Therefore the next controlled experiment is a **zero-variable, checkpoint-instrumented
replication of R2**, because it isolates the variable **observability of the learning
trajectory** while manipulating **no** production component:

- **Manipulated variable:** none. Same `seed 20260818`, same 600 episodes, same
  `rollout_episodes 8`, same `critic_target "mc"`, same `--device cpu`, same everything.
  The only difference is that `agent.save` is invoked after each of the 75 `agent.update`
  calls. Under RULE 1 this is the cleanest possible rung: it changes nothing.
- **Control:** the frozen `mappo_R2_mc_target.pth` and `mappo_R2_mc_target_updates.csv`.
- **Falsifiable integrity check that makes it worth doing:** if the replication's 75 update
  rows are byte-identical to R2's `_updates.csv` and the final `.pth` matches
  `mappo_R2_mc_target.pth`, then the intermediate checkpoints are *provably* the trajectory
  that produced R2, and every historical R2 finding transfers to them. If they are not, we
  have found a reproducibility defect, which is itself a result. The project has already
  demonstrated arm reproducibility once (`mappo_A0_cpu_repro.pth`).
- **Why training is justified despite RULE 4:** RULE 4 forbids training to answer what
  frozen artifacts can answer. A learning *trajectory* is definitionally not recoverable
  from terminal checkpoints — there is exactly one snapshot per arm, `R3_best` excepted.
  No offline diagnostic can produce it.
- **Implementation note (feasibility, not yet a plan):** `train.py` saves only `_best`
  (line 232, gated on `mean_r > best_mean`) and the final (line 252). Per-update saving must
  therefore come from an **additive driver** that wraps `MAPPOAgent.update`; `torch.save`
  consumes no RNG, so bit-identity is preserved and verifiable. **`train.py` must not be
  edited.**
- **What it unlocks:** every endpoint instrument becomes a learning *curve*, which can be
  calibrated against behaviour — the test §3 shows no current instrument passes. It also
  localizes the arg-max collapse that is currently only bracketed to updates 45–75.

**This report ends Phase 0–2. Nothing has been trained and no R4 exists.** The next action is
**Phase 4 pre-registration** of the run above — hypothesis, single variable, control, success
metric, failure criterion, and hash manifests — and only then Phase 5.

**Replication status:** one seed throughout Sprint 7. Per RULE 11 every causal claim in the
corpus remains exploratory until ≥3 seeds.
