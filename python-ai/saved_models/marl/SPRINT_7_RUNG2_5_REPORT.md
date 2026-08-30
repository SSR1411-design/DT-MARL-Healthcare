# SPRINT 7 — RUNG 2.5: TRUNCATION TARGET ISOLATION

**Status:** diagnostic complete. **No production training was run. No reward, actor, GAE,
value-clipping, minibatch, horizon, action-space, observation-space or risk-predictor
change was made.** Rung 3 has **not** been launched.

**Question this rung was asked:** how much of R2's "MC" target was still controlled by the
critic at truncation, and is a genuinely critic-free continuation return feasible without
touching the online horizon or actor/GAE?

**Both questions are answered with exact measurements.** The answers do not flatter the
Rung 1 hypothesis, and Section F says so explicitly.

---

## HEADLINE — read this first

1. **Continuation-MC is exactly obtainable.** 130/130 truncated episodes reached genuine
   terminal completion inside the trace, **zero censored**. No bounds or approximations
   were needed.

2. **The brief's premise is quantitatively wrong at episode start.** γ⁴⁰⁰ ≈ 0.67 does *not*
   imply the target is mostly critic-controlled. At t=0 the critic supplies **9.4%** of
   R2's target magnitude. It dominates (96.7%) only in the last handful of steps.

3. **R2's MC target largely worked.** Measured against the critic-free ideal, A0's λ-target
   was off by **58.4%** of target SD; R2's MC target is off by **7.8%**. R2 closed **~87%**
   of the target-fidelity gap. Remaining headroom is ~3–8% of SD.

4. **But fixing the remaining bootstrap does essentially nothing to the advantage sign.**
   High-risk sign agreement: C0 0.5285, M_R2 0.4980, **M_cont 0.5171**, M_rew 0.5133 — all
   at chance, spread ≤ 3.5pp, and M_cont is *not* better than R2's own checkpoint critic.
   On R2's **own** states (Section E.7) the deployed critic is **below** chance at high risk:
   **0.4396** team / 0.4597 own, with Spearman **−0.104**.

5. **The real error is a per-state offset, not a target defect.** Switching from the raw to
   the *paired* estimator lifts sign agreement from ~0.51 to **0.848 (team) / 0.966 (own)** on
   Rung 0's rows and from below chance to **0.711 / 0.946** on R2-native rows — and does so
   **identically for all four arms** (spread ≤ 0.4pp). The sign information is
   present; it is buried under a per-state V(s) offset that cancels on differencing.

6. **Where the high-risk miscalibration actually comes from: 84.5% refitting, 15.5% target.**
   High-risk residual C0 −0.7424 → M_R2 (refit on R2's *own* target) −0.2263 → M_cont −0.1318.

7. **The actor stall is a separate mechanism: softmax saturation.** R2 has 50.5% of decision
   entries at max-prob > 0.99 and 24.2% at per-entry entropy < 0.01; **A0 has 0.0000 of both.**
   Gradients have *not* vanished (‖g‖ = 6.08e-02, params move L2 5.43e-03).

8. **`clip_frac → 0` is NOT the anomaly I reported in Rung 2.** A0 does it too (12/75 zero
   updates, last nonzero at 68). Both runs' PPO ratios sit far inside the clip band. This
   corrects my Rung 2 write-up.

9. **The high-risk-EDGE-sparsity hypothesis fails.** 0/16 minibatches lacked high-risk
   decision entries; only 1/16 lacked a high-risk EDGE entry (mean 2.50/minibatch). The
   "30% zero-high-risk minibatch" figure does not reproduce.

---

## I. INTEGRITY (reported first, because it gates everything else)

Hashes were recorded **before** any Rung 2.5 work and re-verified **after** the last probe
completed:

- `_rung2_5_integrity/SPRINT_7_RUNG2_5_artifacts_before.md5` — **23/23 OK, 0 FAILED**,
  including `mappo_A0_cpu_repro.pth`, `mappo_R2_mc_target.pth`, both `_best.pth`, all
  `_updates.csv`/`_history.csv`/`_eval.*`, and the Rung 0/Rung 1 reports and JSON.
- `_rung2_5_integrity/SPRINT_7_RUNG2_5_code_before.md5` — **23/23 OK, 0 FAILED**
  (`mappo.py`, `env.py`, `config.py`, `train.py`, `_diag_rung0.py`,
  `_diag_rung1_critic.py`, `tests_env.py` all individually confirmed).

**A0 and R2 checkpoints are byte-identical to their pre-Rung-2.5 state.** Every Rung 0,
Rung 1, Rung 2, Sprint 6 and Sprint 6.5 artifact is untouched.

**Disclosed deviation from the brief's naming rule.** The brief said all new files should
carry the `SPRINT_7_RUNG2_5_` prefix. All new **artifacts and logs** do. The four new
**scripts** follow the repository's existing `_diag_rungN_` convention
(`_diag_rung2_5_targets.py`, `_diag_rung2_5_signtest.py`, `_diag_rung2_5_actor_stall.py`,
`_diag_rung2_5_native_dev.py`) so they sit alongside `_diag_rung0.py` and
`_diag_rung1_critic.py`. Flagging rather than silently diverging.

---

## Method, and what it is not

Continuation is implemented by **ignoring the `done` flag**, not by extending the horizon.
`env.py:594` computes `done` as a returned flag only; `step` never refuses to run past the
limit. Critically, `cfg.episode_steps` is **deliberately not mutated**, because
`prog = step_idx / episode_steps` feeds the per-agent observation (`env.py:445`) and the
global state (`env.py:524`) — changing it would change observations and therefore the
trajectory. (An earlier feasibility probe, `_diag_rung2_5_feasibility.py`, did mutate it;
that mechanism is **known invalid** for building the target and served only to expose the
`prog` leak.)

**Determinism.** An episode is fully determined by `episode_start_tick`: `self._rng` appears
only at `env.py:192/219/224` and solely selects the start; `build_patient_tasks`
(`criticality.py:109`) has no RNG. Verified: repeat replays are bit-identical, forced-action
replays reproduce rewards exactly, and an intervening episode does not perturb a replay.

**Three honest limits.**
1. The **final frozen** R2 policy is replayed over the training start distribution. This is
   **not** a reconstruction of the actual training rollouts — per-episode torch RNG states
   were never saved, so the real rollouts are unrecoverable.
2. Past step 400, `prog > 1.0`, so continuation **actions** are chosen on out-of-distribution
   observations. Continuation **rewards** are true environment rewards.
3. The per-episode truncation *trend over training* is **not recoverable**: `history.csv` has
   no length column and `train.py:271` prints only a single end-of-run aggregate (377/600).

**OOD sensitivity is empirically negligible.** Bracketing the continuation policy between
greedy and all-STAY: true remaining tail +0.3536 vs +0.3550; MAD vs the R2 target 0.5583 vs
0.5596. The bracket is tight, so the continuation target is well-determined — bounds were
not required.

---

## A. TARGET DECOMPOSITION

200 stochastic episodes, R2, training start window, greedy continuation.

**End classification.** `term` (all tasks terminal before 400) 64 · `both` (terminal exactly
at 400) 6 · `trunc` (alive at 400) 130. `train.py`'s own rule
(`truncated = step_idx >= episode_steps`) flags **136/200 = 68.0%**, consistent with
training's 377/600 = 62.8%. Note the `both` class is swept into "truncated" by that rule.

### Bootstrap contribution by depth (bootstrapped episodes only)

Signed means are near zero because rewards are two-sided; **magnitudes are what matter.**

| depth | mean&#124;boot&#124; | mean&#124;reward-only&#124; | mean&#124;target&#124; | &#124;boot&#124; share: mean / p50 / p95 |
|---|---|---|---|---|
| t = 0 | 0.407 | 6.289 | 6.620 | **0.094** / 0.058 / 0.320 |
| t = T/2 | 0.497 | 5.188 | 5.548 | **0.135** / 0.086 / 0.475 |
| t = T−1 | 0.607 | 0.045 | 0.632 | **0.967** / 0.999 / 1.000 |

**This refutes the brief's framing.** γ⁴⁰⁰ ≈ 0.67 describes the *discount survival*, not the
*share of target magnitude*. Because 400 steps of two-sided reward accumulate to
mean&#124;·&#124; ≈ 6.3 while the bootstrap is ≈ 0.4, the critic controls **~9%** of the
target at episode start. It only dominates in the final few steps, where the reward-only
term has almost nothing left (0.045).

### V(s_T) versus the true remaining return at the boundary

| | `trunc` (n=130 eps / 1300 entries) | `both` (n=6 / 60) |
|---|---|---|
| V(s_T) | +0.2760 (sd 0.8589) | +0.0870 (sd 0.7671) |
| true remaining return | +0.3536 (sd 0.9729) | **0.0000 exactly** |
| mean &#124;V − true&#124; | **0.6778** | **0.5668** |
| fraction of true tail explained by V | **0.6589** | 0.0000 |

Two things worth stating plainly. First, on `trunc` the per-episode error (0.678) is roughly
**twice the magnitude of the quantity being estimated** (0.354) — V(s_T) is the right order
of magnitude on average but badly wrong episode by episode. Second, on `both` the true tail
is **exactly zero**, so that bootstrap is **pure injected bias**. I did not assume this: I
forced 120 extra steps across the 6 episodes with the all-terminal break disabled and
measured **sum&#124;reward&#124; = 0.000000**.

Full distributions (n / mean / sd / min / p5 / p25 / p50 / p75 / p95 / max) for every
quantity above are in `SPRINT_7_RUNG2_5_targets_R2_contGREEDY.json`.

---

## B. TRUNCATION STRATIFICATION

| | `term` (64) | `both` (6) | `trunc` (130) |
|---|---|---|---|
| length mean | 377.75 (sd 16.01) | 400.0 | 400.0 |
| **episode return mean** | **−10.5763** (sd 16.92) | +7.4624 | **+7.3281** (sd 14.10) |
| target at t=0 | −1.2854 | +0.0538 | +0.2986 |
| reward-only at t=0 | −1.2854 | −0.0045 | +0.1136 |
| V(s_T) | +0.1205 | +0.0870 | +0.2760 |
| bootstrap contribution at t=0 | **0.0000** (correctly none) | +0.0583 | +0.1850 |
| risk mean | 0.0823 | 0.0853 | 0.0852 |
| decision-entry fraction | 0.1922 | 0.1644 | 0.1842 |
| high-risk decision entries | 1598 | 240 | 3838 |
| **high-risk EDGE share** | **0.1014** | 0.0542 | 0.0724 |
| high-risk STAY share | 0.7985 | 0.8917 | 0.8390 |
| continuation steps needed | 0 | 0 | **27.12** (sd 19.76, p95 61.1) |

**The dominant finding here is a selection effect the brief did not anticipate.** Truncated
episodes are the **good** ones (mean return **+7.33**); naturally-terminating episodes are
the **bad** ones (**−10.58**) — an ~18-unit gap. Early termination means tasks left the
system, which in this environment correlates with loss. So the truncation bootstrap is
applied **selectively to the high-return episodes**, making any V(s_T) bias a
non-random perturbation correlated with outcome, not white noise.

Risk distributions are near-identical across classes (0.082–0.085), so **risk does not drive
truncation**. High-risk EDGE is actually *most* frequent in the naturally-terminating class
(0.1014) and *less* frequent in truncated episodes (0.0724).

Continuation needed only **27 steps on average** (p95 = 61) against a mean headroom of 225
steps — which is why the next section comes out clean.

---

## C. CRITIC DEPENDENCE — is a critic-free target obtainable?

**Yes, exactly.** Feasibility was settled empirically, not assumed:

| question | answer |
|---|---|
| Reproducible from start tick + action history? | **Yes** — bit-identical replays; forced-action replay reproduces rewards |
| Can `step_idx` continue past `episode_steps`? | **Yes** — `done` is an advisory flag; `step` never blocks |
| Hidden state making observation-only replay insufficient? | **No** — but `prog` leaks the horizon into obs/state, so `episode_steps` must not be mutated |
| Deterministic replay already in Rung 0/1 machinery? | **Yes** — `_diag_rung0._replay`, greedy by default |

**Result: 130 truncated episodes → 130 continuations attempted → 130 reached genuine terminal
→ 0 censored by trace end.** Trace geometry makes this comfortable: 1500 ticks,
`ticks_per_step=2`, headroom 345 steps at t₀=9 falling to 104 at t₀=491 (mean 225), against a
27-step mean requirement. `obs_channel` indexes numpy directly, so an out-of-range read
would raise `IndexError` rather than silently corrupt — it never fired.

---

## D. THREE TARGETS COMPARED

520,000 entries, 130 exactly-continued episodes.

| | mean | sd | p50 | p95 |
|---|---|---|---|---|
| (1) R2 target (reward-to-end + V(s_T) bootstrap) | +1.7756 | 7.4481 | +0.4696 | +15.1043 |
| (2) reward-only to the 400-step boundary | +1.5483 | 6.9905 | +0.2313 | +13.5608 |
| (3) continuation-MC through genuine terminal | +1.8396 | 7.1970 | +0.3333 | +14.4201 |

| comparison | value |
|---|---|
| MAD(R2 target, continuation-MC) | **0.5583** = **7.8%** of target SD |
| MAD(reward-only, continuation-MC) | **0.3307** = **4.6%** of target SD |
| mean signed (R2 − continuation) | −0.0639 |
| corr(R2, continuation) | 0.9941 |
| corr(reward-only, continuation) | 0.9939 |

**The critic bootstrap is ~1.7× worse than simply ignoring the tail.** Dropping the bootstrap
entirely (target 2) is *closer* to the truth than R2's actual target. Correlations are
near-identical (0.9941 vs 0.9939), so the bootstrap adds error without adding rank
information.

### The A0 control — the single most important table in this rung

Same probe, A0 checkpoint (λ-target), 40 stochastic episodes:

| | A0 (λ-target) | R2 (MC target) |
|---|---|---|
| bootstrapped episodes | 22/40 = 55.0% | 136/200 = 68.0% |
| &#124;boot&#124; share at t=0 | **0.4415** | **0.0938** |
| mean&#124;boot&#124; at t=0 | 3.0261 | 0.4071 |
| V(s_T) mean | **+4.5651** | +0.2760 |
| true remaining tail mean | +0.2755 | +0.3536 |
| mean &#124;V − true tail&#124; | **4.2961** | **0.6778** |
| fraction of true tail explained | **0.0938** | **0.6589** |
| MAD(actual target, continuation-MC) | **3.5387 = 58.4% of SD** | **0.5583 = 7.8% of SD** |
| mean signed (actual − continuation) | **+3.5334** | −0.0639 |
| corr(actual, continuation) | 0.9634 | 0.9941 |
| MAD(reward-only, continuation-MC) | 0.3528 = 5.8% | 0.3307 = 4.6% |

**A0's λ-trained critic explained only 9.4% of the true boundary tail and inflated it ~16×**
(V = 4.57 against a truth of 0.28), contaminating its target by 58.4% of target SD. **R2's MC
target cut that to 7.8% — an 87% reduction.** So the Rung 1 hypothesis was *correct as a
description of A0's defect*, and **R2's intervention substantially fixed it.** The residual
headroom for any further target surgery is ~3–8% of SD, i.e. roughly one-seventh of the
defect already corrected.

---

## E. ADVANTAGE / SIGN TEST (offline causal diagnostic — **not** an online result)

### E.0 A correction to my own Rung 2.5 framing, and a design constraint

Rung 1's artifact records `truncated_episodes = [False]×16`. I earlier attributed this to
Rung 1 using greedy replays while training sampled. **That was incomplete.** The artifact's
`model` field is `mappo_A0_cpu_repro.pth` — **Rung 1 ran on A0, not R2** — and the truncation
profile is *policy-dependent*:

| greedy replay, same 16 start ticks | truncated | mean length |
|---|---|---|
| **A0** | **0/16** | 367.2 |
| **R2** | **10/16** | 393.2 |

So Rung 1's zero-truncation regime was real and correctly measured **for A0**, and it arose
from *both* greedy replay *and* A0's policy. The 2×2: A0 greedy 0%, A0 stochastic 55%, R2
greedy 62.5%, R2 stochastic 68%.

**This is a feedback mechanism worth naming: the MC target is critic-free only while episodes
terminate naturally, and R2's own training moved the policy into the regime where they do
not** (greedy mean length 367 → 393, keeping tasks alive 26 steps longer). The target's
critic-freeness is a property of the *policy*, not of the target — and it self-erodes.

**Consequence for design:** the three targets are identical wherever nothing truncates, so
the arms must differ in the **fit dataset** (stochastic, truncation-bearing: 40 episodes,
15,823 timesteps, classes trunc 31 / both 1 / term 8, **0 dropped**), while **evaluation**
stays on Rung 0's exact 583 states / 745 forced replays. Deviation replays stay greedy so
Rung 0's noise-floor identity (A(s, a_baseline) ≡ 0) holds.

*(A superseded claim: `SPRINT_7_RUNG2_5_signtest.json` contains a `structural_null` block
asserting the three targets "coincide exactly" on Rung 0's pairs. That was written assuming
A0's 0/16 profile and is **false for R2**, where 10/16 baselines truncate. The arms in phase D
still differ only through their fitted critics, which is what the phase measures. Corrected
here rather than by editing the artifact.)*

### E.1 Arms

| arm | what |
|---|---|
| **C0** | R2's critic exactly as loaded from the checkpoint (reference) |
| **M_R2** | fresh critic, R2's **actual** target: reward-to-end + γ^(T−t)·V(s_T) on truncated episodes |
| **M_cont** | fresh critic, **continuation-MC** (critic-free ideal) |
| **M_rew** | fresh critic, reward-only to the boundary (Rung 1's C1 as literally written) |

All three fresh arms: plain MSE, identical optimiser/schedule/seed, **early-stopped on a
common held-out continuation-MC yardstick** so no arm is scored on its own target.

### E.2 Critic fit and residual-vs-risk

| arm | best val (cont-MC) MSE | EV train | EV val | hi-risk residual | lo→hi swing | hi-only OLS slope |
|---|---|---|---|---|---|---|
| C0 | — (not fitted) | 0.6870 | 0.6931 | **−0.7424** | −0.7756 | −0.8808 |
| M_R2 | 13.1789 @ ep 42 | 0.8211 | 0.7475 | −0.2263 | +0.0532 | −0.1046 |
| M_cont | 13.0520 @ ep 42 | 0.8395 | **0.7487** | **−0.1318** | **+0.0927** | −0.7097 |
| M_rew | 13.0335 @ ep 32 | 0.8076 | 0.7486 | −0.3636 | −0.2157 | −0.4764 |

Two readings. (i) **C0 is specifically miscalibrated at high risk**: residual −0.7424 at
hi versus +0.0331 at lo — R2's deployed critic believes high-risk states are ~0.74 better
than they are. That is the Rung 1 story, confirmed. (ii) **The three targets are nearly
indistinguishable**: val MSE 13.18/13.05/13.03 (1.1% spread), EV_val 0.7475/0.7487/0.7486
(0.0012 spread).

**Attribution of the high-risk fix — the decisive decomposition:**

| step | hi-risk residual | improvement | share |
|---|---|---|---|
| C0 (deployed critic) | −0.7424 | — | — |
| → M_R2 (refit on R2's **own** target) | −0.2263 | 0.5161 | **84.5%** |
| → M_cont (refit on critic-free target) | −0.1318 | 0.0945 | **15.5%** |

**84.5% of the available high-risk calibration gain comes from merely fitting the critic to
convergence on on-policy data — not from changing the target.** C0's miscalibration is
therefore mostly an *under-fitting / distribution-shift* artifact (75 online updates × 4
epochs on a moving distribution) rather than a target-formulation artifact. **That is a third
mechanism, distinct from both the Rung 1 target hypothesis and the actor stall.**

### E.3 Sign agreement on Rung 0's exact 583 states / 745 forced replays

583/583 usable, **745 forced replays, 0 mismatches**, GAE replica vs `MAPPO.compute_gae`
max&#124;Δadv&#124; = 0.00e+00.

| arm | team hi (n=263) | team all (n=559) | own hi | own all | spearman hi (team) |
|---|---|---|---|---|---|
| C0 | **0.5285** | 0.5349 | **0.5627** | 0.5474 | 0.056 |
| M_R2 | 0.4981 | 0.5134 | 0.4943 | 0.4937 | 0.074 |
| M_cont | 0.5171 | 0.5420 | 0.5133 | 0.5152 | 0.081 |
| M_rew | 0.5133 | 0.5152 | 0.5019 | 0.4919 | 0.089 |

**Every arm sits at chance.** The critic-free ideal (M_cont, 0.5171 team / 0.5133 own) is
**not better** than R2's deployed critic (C0, 0.5285 / 0.5627). On own-attribution high risk,
C0 is the *best* arm. Spearman 0.056–0.089 everywhere.

### E.4 Per-action means at high risk — where the real inversion lives

Team attribution, high-risk bucket, with standard errors:

| arm | STAY (n=101) | MIGRATE_TO_NEIGHBOR_EDGE (n=230) | ordering |
|---|---|---|---|
| **truth** | −0.0298 ± 0.0707 (t=−0.42) | **+1.1772 ± 0.2857 (t=+4.12)** | **[EDGE, STAY]** |
| C0 | −0.9037 ± 0.4914 (t=−1.84) | **−1.1089 ± 0.3762 (t=−2.95)** | [STAY, EDGE] ✗ |
| M_R2 | +0.1041 ± 0.4308 (t=+0.24) | +0.0213 ± 0.3014 (t=+0.07) | [STAY, EDGE] ✗ |
| M_cont | +0.0719 ± 0.3972 (t=+0.18) | +0.1106 ± 0.2893 (t=+0.38) | **[EDGE, STAY]** ✓ |
| M_rew | +0.0756 ± 0.4354 (t=+0.17) | −0.0687 ± 0.3085 (t=−0.22) | [STAY, EDGE] ✗ |

**The truth is unambiguous and significant: migrating at high risk is genuinely good
(+1.1772, t = +4.12).** R2's deployed critic assigns it a **significantly negative** advantage
(−1.1089, t = −2.95) — a confirmed sign inversion, and exactly what
`critic-bias-inverts-advantage-sign` predicted.

> ⚠️ **RETRACTED in E.7.** The C0 row of this table does **not** survive on R2-native rows.
> Rung 0's D1 rows force an action R2's mask marks illegal in 53% of cases (see E.6), and that
> contamination pushes C0's EDGE estimate down. Measured on R2's own states, C0 gives high-risk
> EDGE **+0.1655 (t = +0.51)** — correctly signed and correctly ranked above STAY. **The
> "significantly inverted EDGE" claim is withdrawn.** The *truth* column is unaffected in
> direction and is confirmed larger on native rows (+1.6227, t = +3.78). What survives as
> evidence of inversion is in E.7: below-chance high-risk sign agreement, negative high-risk
> Spearman, a ~10× understatement of EDGE, and a significant inversion on MIGRATE_TO_CLOUD.

**But the honest caveat: M_cont is the only arm that recovers the correct ordering, and that
ordering is not statistically significant** (+0.1106 ± 0.2893 vs +0.0719 ± 0.3972; the two
overlap heavily and neither differs from zero). It could be noise. None of the three refit
arms recovers the *magnitude* — the truth is +1.18, the best arm gives +0.11, a 10×
understatement.

### E.5 The paired estimator — the finding that reframes the rung

C0's noise floor at high risk is **−1.3093** (sd 5.66) where the true advantage of the
baseline action is *identically zero*. That is a near-constant per-state offset. Since the
truth is defined as a **difference** at one state, A_true(a) = Q(a) − Q(a_ref), the matched
estimator is the **paired** GAE difference, in which any per-state offset cancels:

| truth | arm | raw | **paired** | Δ |
|---|---|---|---|---|
| team, hi (n=263) | C0 | 0.5285 | **0.8441** | +0.3156 |
| | M_R2 | 0.4981 | **0.8479** | +0.3498 |
| | M_cont | 0.5171 | **0.8479** | +0.3308 |
| | M_rew | 0.5133 | **0.8479** | +0.3346 |
| own, hi (n=263) | C0 | 0.5627 | **0.9620** | +0.3992 |
| | M_R2 | 0.4943 | **0.9658** | +0.4715 |
| | M_cont | 0.5133 | **0.9658** | +0.4525 |
| | M_rew | 0.5019 | **0.9658** | +0.4639 |

**Sign agreement jumps from chance (~0.51) to 0.848 (team) / 0.966 (own) — and it does so
essentially identically for every arm, C0 included (spread ≤ 0.4pp).**

The interpretation is direct: **the advantage-sign information is already present in R2's
deployed critic.** It is buried under a per-state offset error in V(s). The truncation-target
choice moves this by ≤ 0.4pp; differencing away the offset moves it by +30 to +45pp — a
**~100× larger effect**. PPO's surrogate multiplies log π(a|s) by the **raw** advantage, so
it inherits the full offset error rather than the clean paired signal.

Noise floors (mean GAE at the baseline action, where truth ≡ 0):

| arm | hi (n=251) | lo (n=290) | all (n=583) |
|---|---|---|---|
| C0 | **−1.3093** (sd 5.66) | −0.0295 (sd 4.65) | −0.5473 |
| M_R2 | −0.1603 (sd 4.80) | −0.0754 (sd 4.70) | −0.0562 |
| M_cont | −0.1453 (sd 4.54) | −0.4338 (sd 4.90) | −0.2398 |
| M_rew | −0.2474 (sd 4.82) | +0.0038 (sd 4.70) | −0.0766 |

Refitting removes C0's large negative high-risk *bias* (−1.31 → −0.15) but leaves the
*spread* at ~4.5–4.8 in every arm, against a true high-risk EDGE effect of +1.18. **After the
bias is fixed, estimator variance is the binding constraint, not target bias.**

### E.6 A validity limit on Rung 0's rows that I must disclose

The brief asked for "the same 745 deviation pairs from Rung 0 where possible." Reusing them
turned out to be only **partly** possible, and here is the measurement:

| at Rung 0's 583 (start, step, agent) rows | A0 | R2 |
|---|---|---|
| `ref_action` still equals the baseline action | **100.0%** | **67.6%** |
| every D1-recorded legal action still legal | **100.0%** | **47.0%** |

Rung 0 and Rung 1 both ran on A0, so their rows were self-consistent by construction. R2 is
a different policy that has usually walked elsewhere by that step, so 32.4% of rows carry a
stale reference action and 53.0% force at least one action R2's own mask marks illegal.

**What this does and does not damage.** It does **not** affect the arm-vs-arm or raw-vs-paired
contrasts — every arm was scored on identical rows, and `agreement`/`per_action` skip the
reference entry while `noise_floor` and my paired statistic read the baseline GAE
*positionally*, so all remain valid. It **does** make the absolute levels and per-action
means on D1 rows provisional for R2. Section E.7 therefore rebuilds the set natively.

### E.7 R2-native deviation set — clean absolute numbers, and a correction to E.4

Rung 0's selection rule was applied **verbatim** to R2's own greedy baselines (same 16 start
ticks, same `has_task`/≥2-legal filter, same pools keyed by the action greedy actually chose,
same risk buckets, same caps A=150 / B=80 / C=60, same evenly-spaced `np.linspace` take —
no RNG, so the selection cannot be cherry-picked). Every forced action is legal under R2's
*own* mask, and every `ref_action` is R2's *own* baseline action.

| | value |
|---|---|
| greedy baseline lengths | [369, 372, 376, 378, 398, 399, **400 ×10**] — 10/16 truncated |
| pools available | A_hi 659 · A_lo 13900 · A_mid 24 · B_hi 62 · B_lo 919 · B_mid 3 · C_hi 18 · C_lo 512 · C_mid 0 |
| rows built = usable states | **547** |
| forced replays | **716**, replay mismatches **0** |
| GAE replica vs `MAPPO.compute_gae` | max&#124;Δadv&#124; = max&#124;Δret&#124; = **0.0** |

Only arm C0 (R2's deployed critic) is evaluated here — the point is a clean *absolute*
measurement, not another arm race.

**Sign agreement, R2's deployed critic, R2-native rows:**

| bucket | n | team sign | team spearman | own sign | own spearman |
|---|---|---|---|---|---|
| lo | 388 | 0.5155 | +0.057 | 0.5464 | +0.087 |
| mid | 30 | 0.4667 | +0.043 | 0.5000 | +0.163 |
| **hi** | **298** | **0.4396** | **−0.104** | **0.4597** | **−0.051** |
| all | 716 | 0.4818 | −0.018 | 0.5084 | +0.021 |

**On its own states, R2's critic is *below chance* at high risk (0.4396 / 0.4597) with a
negative rank correlation (Spearman −0.104 team).** The D1-row estimate (0.5285 / 0.5627) was
**flattering** R2 — the stale rows were the optimistic measurement, not the pessimistic one.
Sign fidelity degrades monotonically with risk: 0.5155 (lo) → 0.4667 (mid) → 0.4396 (hi).

**Per-action means at high risk (R2-native):**

| action | n | true (team) | true (own) | C0 gae |
|---|---|---|---|---|
| STAY | 80 | −0.2630 ± 0.2405 (t=−1.09) | **−0.8822 ± 0.2208 (t=−3.99)** | −0.7668 ± 0.4471 (t=−1.72) |
| MIGRATE_TO_NEIGHBOR_EDGE | 212 | **+1.6227 ± 0.4295 (t=+3.78)** | **+0.8675 ± 0.2341 (t=+3.71)** | +0.1655 ± 0.3266 (t=+0.51) |
| MIGRATE_TO_CLOUD | 6 | **+5.2115 ± 0.5998 (t=+8.69)** | **+4.5311 ± 0.5578 (t=+8.12)** | **−5.4068 ± 1.6960 (t=−3.19)** |
| ordering by **truth** | | **[CLOUD, EDGE, STAY]** | [CLOUD, EDGE, STAY] | |
| ordering by **C0 gae** | | | | **[EDGE, STAY, CLOUD]** |

**The ground truth is now unambiguous and significant under both attributions: at high risk,
migrating is strongly better than staying.** Own-attribution makes it sharpest — STAY
−0.8822 (t = −3.99) versus EDGE +0.8675 (t = +3.71), two significant effects of opposite
sign. The Sprint 7 premise that the policy *should* be using `predicted_failure_risk` is
therefore **independently confirmed on R2's own states**: the signal is real and large.

**Two corrections to E.4, which used the stale D1 rows:**

1. **The specific claim "R2's critic assigns high-risk EDGE a significantly negative
   advantage (−1.1089, t = −2.95)" does not survive.** On R2-native rows C0 gives EDGE
   **+0.1655** (t = +0.51) — correctly signed, and correctly ranked above STAY (−0.7668).
   The D1 rows' 53% illegal-action contamination was pushing EDGE's estimate down. I am
   retracting the "significantly inverted EDGE" statement.
2. **The inversion is real but lives elsewhere.** It shows up as (a) below-chance high-risk
   sign agreement, (b) *negative* high-risk Spearman, (c) a ~10× magnitude understatement of
   EDGE (+0.166 estimated against +1.623 true), and (d) a **highly significant inversion on
   MIGRATE_TO_CLOUD**: true **+5.2115** (t = +8.69) versus gae **−5.4068** (t = −3.19). Both
   sides are individually significant and they have opposite signs. **Caveat: n = 6** (pool
   B_hi had 62 candidates but only 6 survived as high-risk deviations), so this is suggestive,
   not established. Worth noting the direction: the critic is *most* wrong about the action
   that is *most* valuable.

**Paired vs raw on native rows — the E.5 result reproduces:**

| truth | bucket | n | raw | **paired** | Δ |
|---|---|---|---|---|---|
| team | hi | 298 | 0.4396 | **0.7114** | **+0.2718** |
| team | lo | 388 | 0.5155 | 0.7423 | +0.2268 |
| team | all | 716 | 0.4818 | 0.7235 | +0.2416 |
| own | hi | 298 | 0.4597 | **0.9463** | **+0.4866** |
| own | lo | 388 | 0.5464 | 0.9278 | +0.3814 |
| own | all | 716 | 0.5084 | 0.9372 | +0.4288 |

The paired estimator recovers **0.7114 (team) / 0.9463 (own)** at high risk from a raw
baseline *below chance* — Δ = +0.27 / +0.49. Absolute levels are a little lower than on D1
rows (0.8441 / 0.9620) because native rows include the hard `mid` and `CLOUD` cases, but the
conclusion is identical and now rests on a self-consistent set: **the sign information is
present in R2's deployed critic and is destroyed by a per-state offset in V(s).**

Native noise floor (mean GAE at the baseline action, where the true advantage is identically
zero): lo −0.8814 (sd 4.11, n=290) · mid **+2.4009** (sd 4.06, n=27) · hi −0.8695 (sd 5.19,
n=230) · all −0.7144 (sd 4.65, n=547). The offset is large, risk-dependent, and **changes
sign across buckets** (+2.40 at mid versus −0.87 at hi) — which is precisely why it cannot be
absorbed as a constant baseline and why differencing is what rescues the sign.

---

## G. ACTOR STALL ANALYSIS (attribution only — nothing fixed)

### G.1 The gradient did **not** vanish

At R2's final policy, replica update (16 minibatches = 4 epochs × 4; T = 8×400 = 3200 is
divisible by 4, so the known minibatch-tail quirk **did not manifest** and was not fixed):

| quantity | R2 | A0 |
|---|---|---|
| actor grad norm (pre-clip, total) | 6.084e-02 | 1.238e-01 |
| — policy-gradient only | 6.113e-02 | 1.197e-01 |
| — entropy term only (incl. coef 0.02) | 4.786e-03 | 1.282e-02 |
| actor param delta L2 (one update) | 5.430e-03 | 1.060e-02 |
| actor param delta max&#124;Δ&#124; / rms | 1.433e-04 / 1.125e-05 | 1.439e-04 / 2.195e-05 |
| k1 KL | +6.98e-06 | +1.45e-05 |
| k3 KL | 8.67e-08 | 9.44e-07 |
| clip_frac | **0.000000** | **0.000000** |
| adv mean / sd | −0.0011 / 2.652 | −0.0124 / 2.935 |
| critic_target_used | **mc** ✓ | lambda ✓ |

**Parameters do move.** So this is not a dead-gradient stall. R2 simply gets **half** A0's
gradient magnitude and moves half as far per update.

### G.2 Why clip_frac and k3 are exactly 0 — and a correction to my Rung 2 report

The PPO ratio across all 16 minibatches spans **[0.9920911, 1.0059656]** for R2 and
**[0.9949841, 1.0184942]** for A0. Clipping requires the ratio to leave **[0.8, 1.2]**. Both
runs are ~30–400× short of that. `clip_frac = 0` is the arithmetically inevitable
consequence, and `k3 = (r−1) − log r` is second-order in (r−1), hence ~1e-8.

**Corrections to what I reported in Rung 2:**
1. I wrote that clip_frac was "exactly 0 from ~update 50". Precisely: it first touches 0 at
   update **36**, is nonzero again through update **61**, and is permanently 0 only from
   update **62**. 22 of 75 updates are exactly zero.
2. More importantly, **I flagged clip_frac → 0 as an R2 anomaly. It is not.** A0 does the
   same thing — 12/75 zero updates, last nonzero at update **67**. `clip_frac → 0` at
   convergence is **normal behaviour for this setup**, not evidence of pathology, and it is
   not the discriminating signal between the two runs.
3. `adv_std` stays in 2.47–3.93 across all 75 R2 updates — the advantage signal never
   vanished.

### G.3 What *is* different: softmax saturation

Saturation census at decision entries (8 stochastic episodes each):

| | R2 | A0 |
|---|---|---|
| decision entries | 4967 | 3452 |
| max prob mean / p50 / p95 | 0.9107 / 0.9948 / 0.9991 | 0.8767 / 0.9382 / 0.9651 |
| **frac max prob > 0.99** | **0.5051** | **0.0000** |
| **frac max prob > 0.999** | **0.2203** | **0.0000** |
| **frac per-entry entropy < 0.01** | **0.2424** | **0.0000** |
| mean per-entry entropy | **0.1841** (13.3% of ln4) | 0.3176 (22.9%) |
| episode lengths | 400.0 all 8 (8/8 truncated) | — |

**A0 is the decisive control: 0.0000 of A0's entries exceed max-prob 0.99, against 50.5% of
R2's.** The collapse is specific to R2 — not a property of the architecture, the task, or the
entropy coefficient (identical at 0.02 in both).

The mechanism is standard: as π(a|s) → 1, ∇log π(a) → 0, so the ratio stays pinned at 1 and
the surrogate becomes locally flat. The entropy bonus cannot reverse it — the entropy-only
gradient is 4.79e-03, just **7.8%** of R2's total actor gradient, already including
`entropy_coef = 0.02`.

Training trajectories (from the untouched `_updates.csv` files):

| update | A0 entropy / clip_frac | R2 entropy / clip_frac |
|---|---|---|
| 1 | 0.7165 / 0.1081 | 0.8599 / 0.1625 |
| 10 | 0.7145 / 0.0447 | 0.5515 / 0.0384 |
| 30 | 0.5322 / 0.0446 | 0.2496 / 0.0129 |
| 50 | 0.3457 / 0.0045 | 0.1474 / 0.0005 |
| 75 | **0.3329** / 0.0000 | **0.1380** / 0.0000 |

A0 settles near 0.333 and *recovers* late (0.278 at update 60 → 0.333 at 75). R2 decays
monotonically to 0.1380 — **9.95% of ln 4 = 1.3863** — and is still sagging at the end.
**A0 = converged but still stochastic; R2 = collapsed onto a vertex of the simplex.**

**Diagnosis: this is neither normal convergence nor insufficient gradient signal. It is
policy collapse via softmax saturation, and it is specific to R2.**

### G.4 The high-risk-EDGE-sparsity hypothesis **fails**

| | R2 | A0 |
|---|---|---|
| minibatches | 16 | 16 |
| **with ZERO high-risk decision entries** | **0** | **0** |
| with zero high-risk **EDGE** entries | **1** | 1 |
| high-risk entries per minibatch (mean) | 67.25 (min 57) | 53.00 |
| high-risk EDGE per minibatch (mean) | **2.50** (min 0, max 6) | 3.50 |

**The proposed explanation does not survive.** Every minibatch contains 57–79 high-risk
decision entries; only one of sixteen lacks a high-risk EDGE entry. The "37 → 8 usable
high-risk EDGE samples and 30% zero-high-risk minibatch rate" figures **do not reproduce** at
the final policy. High-risk EDGE data is sparse (2.5 per minibatch of 800) but **present**,
and A0 — which did *not* collapse — had a comparable 3.5.

### G.5 A discrepancy I am reporting rather than resolving favourably

Rung 2's headline was high-risk EDGE share **2.7% → 8.1%** (A0 → R2) at risk > 0.6, greedy.
Rung 2.5's saturation census at risk > 0.50 over 8 stochastic episodes gives **R2 3.7%
(10/269)** and **A0 6.6% (14/212)** — i.e. **R2 worse, the opposite direction.**

Differences: threshold (0.6 vs 0.50), episode sets, greedy vs stochastic, and very small
counts (10 and 14 EDGE events). Section B's 200-episode stratification gives R2 high-risk
EDGE shares of 0.1014 / 0.0542 / 0.0724 by end-class, spanning both figures. **I cannot
currently reproduce Rung 2's direction of change, and I am not going to pick the flattering
measurement.** Treat "R2 migrates more at high risk" as **unresolved**.

### H. R2 REPLICA FIX

The old Rung 0 replica was not modified. Instead a new R2-aware replica was written, after
finding **two** defects in the old one:

1. **`_diag_rung0.py:1009`** hardcodes `agent.compute_gae(buf)`, so the critic is regressed
   on the **λ-return** even when the checkpoint was trained with `critic_target="mc"`. This
   is the known cause of the D4 replica failure.
2. **`_diag_rung0.py:276`** calls `buf.set_bootstrap(buf.ptr - 1, boot)` with no `truncated`
   argument, which now defaults to `True` — flagging **every** replica episode as a
   time-limit truncation. Harmless for `compute_gae` (ignores the flag) but systematically
   wrong for `compute_mc_returns` (reads it), so an MC replica built on the old buffer would
   bootstrap on every episode.

The new replica passes the real flag (`set_bootstrap(buf.ptr - 1, boot, bool(tr))`) and
honours the checkpoint's target, reporting **`critic_target_used = "mc"`** for R2 and
`"lambda"` for A0. `_diag_rung0.py` was deliberately left untouched so Rung 0's artifacts
remain reproducible.

**Independent confirmation of the freeze:** with the correct MC target, R2's actor still shows
k1 = 6.98e-06, k3 = 8.67e-08, clip_frac = 0.000000, actor loss 9.81e-05, and a param delta of
L2 5.43e-03. **The R2 actor really did stop making meaningful policy updates, and this no
longer depends on the stale λ-target replica.**

---

## F. CRITICAL DECISION TEST

**1. How much of R2's target was actually critic-dependent?**
Far less than the brief assumed. **9.4%** of target magnitude at episode start, 13.5% at
midpoint, 96.7% in the final steps; 68% of episodes carried any bootstrap at all. Measured
against the exact critic-free ideal, R2's target deviates by **7.8% of target SD** — versus
**58.4%** for A0's λ-target. The γ⁴⁰⁰ ≈ 0.67 framing conflates discount survival with share
of target magnitude.

**2. Does continuation-MC materially change the target?**
**No, not for R2.** MAD 0.5583 = 7.8% of SD, correlation 0.9941, mean signed bias −0.0639.
And reward-only (no bootstrap at all) is *closer* to the ideal (0.3307 = 4.6%) than R2's
actual target. It **did** matter enormously for A0 (58.4% of SD) — but that gap is already
closed.

**3. Does it materially improve high-risk advantage sign agreement offline?**
**No.** High-risk sign agreement: C0 0.5285, M_R2 0.4980, M_cont **0.5171**, M_rew 0.5133 —
all at chance, spread ≤ 3.5pp, and M_cont is not better than R2's deployed critic. M_cont is
the only arm to recover the correct EDGE-over-STAY *ordering*, but not significantly
(+0.1106 ± 0.2893). Meanwhile the **paired** estimator lifts every arm to 0.848/0.966,
identically — a ~100× larger effect than the target choice. The R2-native set (E.7) reaches
the same conclusion from a cleaner baseline: raw high-risk agreement **0.4396** team /
0.4597 own, paired **0.7114 / 0.9463**.

**4. Is the Rung 1 hypothesis still supported?**
**Partly — and it is no longer the leading explanation.**
- **Supported:** the self-referential λ-target genuinely wrecked A0's critic (explained 9.4%
  of the true boundary tail, inflated it ~16×, contaminated the target by 58.4% of SD). And
  R2's critic still gets the high-risk ranking wrong on its own states: **below-chance** sign
  agreement (0.4396 team / 0.4597 own), **negative** high-risk Spearman (−0.104), a ~10×
  understatement of EDGE's true value (+0.166 estimated vs **+1.6227**, t = +3.78), and a
  significant inversion on MIGRATE_TO_CLOUD (true **+5.21**, t = +8.69; gae **−5.41**,
  t = −3.19; n = 6, so suggestive only). The ground-truth risk signal is real and large under
  both attributions — the Sprint 7 premise holds.
- **Not supported:** fixing the *remaining* target dependence does not fix the advantage
  sign (≤3.5pp, at chance). And **84.5%** of the achievable high-risk calibration gain comes
  from merely refitting the critic to convergence on R2's *own* target — only **15.5%** from
  changing the target. The dominant defect is **critic under-fitting / distribution shift**,
  plus a **per-state V(s) offset** that pairing removes entirely. I also **retracted** the
  strongest single piece of Rung-1-style evidence I had reported (E.4's "significantly
  inverted EDGE, t = −2.95") once measured on self-consistent rows.
- **Independent of all of it:** the actor stalled by **softmax saturation** (50.5% of entries
  at max-prob > 0.99 versus A0's 0.0000), which no target change addresses.

**5. Is a clean online Rung 3 justified?**

**Go/no-go rule, stated explicitly and applied:**

> Launch an online Rung 3 whose *primary* intervention is the truncation target **only if**
> the offline diagnostic shows continuation-MC improving high-risk sign agreement by **≥ 5
> percentage points** over R2's deployed critic under **both** attributions, **or** reducing
> MAD-to-ideal by **≥ 25% of target SD**.

**Measured: +(−1.1)pp team / (−4.9)pp own, and 3.2% of SD. Both criteria fail decisively.
Verdict: NO-GO for a target-focused Rung 3.**

The evidence redirects Rung 3 to two mechanisms this rung *did* isolate, in priority order:
1. **Actor-side plasticity** — R2's entropy collapsed to 9.95% of ln 4 with half A0's
   gradient magnitude, while A0 under an identical `entropy_coef` never saturated at all.
2. **Critic value-level calibration** — the per-state offset that pairing removes
   (+30 to +45pp of sign agreement), and the 84.5% of high-risk calibration recoverable by
   fitting to convergence rather than by changing the target.

Designing that intervention is **out of scope for this rung**, and per instruction **Rung 3
has not been launched.**

---

## Artifacts

New (all `SPRINT_7_RUNG2_5_`-prefixed, in `python-ai/saved_models/marl/`):
**`SPRINT_7_RUNG2_5_VERDICT.json`** (consolidated machine-readable verdict — every number in
this report, plus the go/no-go evaluation, the retraction, the superseded-claim record and the
constraint checklist), `targets_R2_contGREEDY.json`, `targets_R2_contSTAY.json`,
`targets_A0_SAMPLE_CONTROL.json`, `targets_SMOKE.json`, `signtest_data.json`, `signtest.json`,
`paired_signtest.json`, `native_dev_R2.json`, `actor_stall.json`; integrity manifests under
`_rung2_5_integrity/`. Logs in `python-ai/`. Scripts: `marl/_diag_rung2_5_{targets,
signtest, actor_stall, native_dev, feasibility}.py`.

**Nothing was overwritten. No Sprint 6 / 6.5 / Rung 0 / Rung 1 / Rung 2 artifact changed.**
