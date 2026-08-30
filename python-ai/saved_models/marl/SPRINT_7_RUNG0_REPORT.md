# Sprint 7 — Rung 0: Measurement Report

**Status:** measurement only. No training was run. No environment, reward, action space,
observation space, cloud capacity, risk threshold, or MAPPO architecture was modified.
All 67 pre-existing artifacts and all 20 production `marl/*.py` files verified
bit-identical after the work (md5, §7).

**Baseline under measurement:** `mappo_A0_cpu_repro.pth` (Sprint 6.5 arm A0), final
checkpoint, episode 600, seed 20260818, device cpu, γ=0.999, λ=0.995, `cloud_slots=8`.

**Method note.** D1/D2 use *greedy* (argmax over legal) replay. This is deliberate: under a
deterministic policy `A(s,a) = Q(s,a) − Q(s,a_greedy)` **exactly** — no critic, no bootstrap,
no expectation to approximate, and a single replay yields the exact `Q`. It also gives a free
calibration constant: the true advantage of the action actually taken is *identically zero*,
so any non-zero GAE there is measured estimator error, not an inference. D3/D4 use *sampling*,
because they must reproduce the training distribution. Both are labelled throughout.

Replay fidelity: 745/745 forced replays matched their baseline bit-for-bit up to the deviation
step on trail, team reward, and observations (0 mismatches). The GAE replica reproduces
`MAPPO.compute_gae` to `max|Δadv| = 0.00e+00`. The instrumented PPO update reproduces the real
`MAPPO.update` to `max|Δ| = 0.00e+00` across all 10 reported statistics on all three snapshots.

---

## 1. D1 — True counterfactual advantage vs MAPPO's GAE estimate

Sample counts for every category (states available → measured). Three deviation pools keyed on
the action greedy actually chose, so STAY/EDGE/CLOUD each appear as both reference and
deviation — this fixes the flaw in Sprint 6.5's p9, which deviated only from STAY-chosen states
and therefore made `A(STAY)` zero by construction.

| pool | reference action | risk<0.10 | mid | risk>0.50 |
|---|---|---|---|---|
| A | STAY | 16760 → 150 | 38 → **38** | 1007 → 150 |
| B | CLOUD | 1422 → 80 | 2 → **2** | 91 → 80 |
| C | EDGE | 195 → 60 | 2 → **2** | 21 → **21** |

Middle-risk states are genuinely scarce (42 total). Per instruction, **no synthetic middle-risk
states were created**; the counts are reported as-is and mid-risk numbers are indicative only.
Deviation selection is evenly spaced over the sorted pool with no RNG, so it cannot be
cherry-picked. Total deviation pairs: **745** (vs n=4/n=9 in Sprint 6.5).

### 1a. True action ordering (exact, greedy policy)

`Q(a) − Q(STAY)`, discounted, mean ± s.e.:

| stratum | n | STAY | MIGRATE_EDGE | MIGRATE_CLOUD | true ordering |
|---|---|---|---|---|---|
| risk<0.10 | 290 | 0 | **+0.458 ± 0.259** (39% >0, med −0.006) | −0.198 ± 0.264 (n=80) | EDGE > STAY > CLOUD |
| mid | 42 | 0 | **+2.398 ± 0.809** (69% >0, med +1.058) | +0.008 (n=2) | EDGE > CLOUD > STAY |
| risk>0.50 | 251 | 0 | **+3.183 ± 0.325** (70% >0, med +2.504) | +0.085 ± 0.203 (n=80) | EDGE > CLOUD > STAY |

**The true EDGE advantage rises monotonically with risk: +0.46 → +2.40 → +3.18** (t ≈ 9.8 at
high risk). The environment and reward already encode exactly the risk-conditioned behaviour the
project wants. Continuity with Sprint 6.5's legacy 0.18 split, now at n=163 instead of n=9:
true undiscounted EDGE advantage **+4.118 ± 0.396** (median +6.016, 75% positive) — Sprint 6.5's
+2.671 is confirmed and was an *underestimate*.

### 1b. Does the learned GAE ordering disagree with the truth? — **Yes, it is anti-correlated**

GAE computed at the same entry on the trajectory in which that action was actually taken
(i.e. what PPO would have used had it sampled that action), with normalisation constants from
the corresponding 8-episode block:

| stratum | n pairs | sign agreement | argmax agreement | Pearson | Spearman |
|---|---|---|---|---|---|
| risk<0.10 | 370 | 50.3% | 66.9% | −0.053 | −0.024 |
| mid | 44 | 36.4% | 66.7% | −0.203 | −0.250 |
| **risk>0.50** | **331** | **29.0%** | 72.5% | **−0.149** | **−0.234** |
| all | 745 | 40.0% | 69.3% | −0.169 | −0.209 |

At high risk the GAE sign is **wrong 71% of the time** — worse than a coin flip. And the
disagreement is a *systematic bias*, not variance. Forcing EDGE at risk>0.50 (n=230):

|  | mean | frac > 0 |
|---|---|---|
| true discounted advantage | **+3.363** | 0.70 |
| GAE for those same samples | **−2.867** | 0.32 |

The reward says +3.36; PPO's target says −2.87. The learner is being pushed *away* from the
action that is in fact correct. (argmax agreement of ~70% is above the ~40% chance rate for
2–3 measured actions, so GAE retains weak top-1 information while getting signs and magnitudes
backwards; it is not uniformly random, it is biased.)

### 1c. Where the bias comes from — the measured noise floor

On the reference action, whose true advantage is *identically zero by construction*:

| stratum | GAE mean (should be 0) | GAE sd |
|---|---|---|
| risk<0.10 | **+0.671** | 5.10 |
| mid | **−2.628** | 8.94 |
| risk>0.50 | **−4.570** | 7.45 |

The critic's error is a **monotone function of risk** — the exact feature the policy is supposed
to learn from. At high risk the offset (−4.57) is larger in magnitude than the true effect
(+3.18) and opposite in sign, which fully accounts for the observed sign inversion
(+3.36 − 4.57 ≈ −1.2, same sign and order as the measured −2.87).

**Why advantage normalisation does not save it:** `MAPPO.update` normalises by subtracting a
single *global* batch mean over decision entries (+0.41). Subtracting a scalar cannot remove a
*state-dependent* offset. A risk-correlated bias survives normalisation essentially intact.

---

## 2. D2 — How much of the delayed payoff is inside the GAE horizon

| quantity | value |
|---|---|
| γ, λ, γλ | 0.999, 0.995, 0.994005 |
| effective GAE horizon 1/(1−γλ) | **166.8 steps** |
| 95%-mass horizon (γλ) | 498.2 steps |
| 95%-mass horizon (γ alone) | 2994.2 steps |
| mean episode length | ≈ 365 steps (not fixed at 400 — episodes end when all tasks resolve) |

A reward difference at offset K enters the true return with weight γ^K but enters GAE's
observed-reward path with weight (γλ)^K; the ratio is λ^K. Measured empirical payoff delay for
EDGE at risk>0.50 (n=230), medians:

| k50 | k90 | k95 | argmax\|Δ\| offset |
|---|---|---|---|
| **7 steps** | 110.5 | 113.5 | 14 |

**The payoff arrives early, well inside the horizon.** λ^7 = 0.965, λ^110 = 0.577.
Median `gae_capture` (fraction of the true discounted payoff carried by the observed-reward
path) = **0.802**; median `frac_within_effective_horizon` = **1.0** (91.7% of samples ≥ 0).
*(The `gae_capture` **mean** of −3.43 is a ratio artifact — heavy-tailed division by
near-zero true payoffs, sd 69. The median is the valid statistic; the mean is not reportable.)*

**This refutes my own Phase 1 mechanism #1.** The `horizon: 363–373` field in Sprint 6.5's p9
was episode-length-remaining, not the empirical decision-to-payoff delay. Horizon truncation is
**not** the problem.

### The decomposition that localises the fault

GAE = (observed-reward path) + (V-term contributions). For high-risk EDGE:

| component | value |
|---|---|
| true discounted advantage | +3.363 |
| **reward path delivers** | **+2.946** (88% of truth, correct sign, 77% positive) |
| total GAE actually delivered | **−2.867** |
| ⟹ V terms contribute | **≈ −5.81** |

The reward signal arrives correctly and on time. The critic then inverts it.

### Critic residual, stratified

| stratum | n | explained var | residual sd | adv mean | return sd |
|---|---|---|---|---|---|
| all entries | 58750 | 0.798 | 4.26 | +0.130 | 9.48 |
| decision entries | 19539 | 0.766 | 5.95 | +0.415 | 12.31 |
| risk < 0.10 | 18378 | 0.778 | 5.69 | **+0.764** | 12.07 |
| risk 0.10–0.50 | 42 | 0.720 | 8.94 | **−2.628** | 16.88 |
| risk > 0.50 | 1119 | **0.720** | **7.10** | **−5.212** | 13.42 |
| risk > 0.18 | 1133 | 0.722 | 7.20 | −5.129 | 13.66 |

High-risk states are the worst-fit stratum (EV 0.720 vs 0.778) **and** the highest-variance
one, and they are only 5% of entries (1593/31660). This is textbook under-fitting of a rare,
high-variance stratum: the critic regresses toward the global mean, over-valuing high-risk
states by ≈ 5.2.

**Not a missing-feature problem.** The critic's 489-dim global state is 10×48 observations
(each containing that node's risk at index 12) plus 9 extras including `risks.mean()` and
`risks.max()` ([env.py:506](python-ai/marl/env.py:506)). The critic *has* the risk signal; it
does not fit it.

**Quantified answer to "how much of the +2.671 is lost or distorted":** none of it is lost to
the horizon (~20% attenuation from λ only). All of it is destroyed by the critic — the V terms
contribute −5.81 against a +3.36 true effect, flipping the sign.

---

## 3. D3 — Sample census per PPO update (no retraining)

Three snapshots. **Update 1 is bit-exact and verified**: reproduced episode start ticks
`[158, 202, 32, 228, 343, 164, 277, 205]` match the recorded `_history.csv` exactly, as do the
per-episode rewards, and `decision_frac` 0.0392 matches the logged 0.0392. Updates 57/75 are
labelled **APPROXIMATE** in the artifact — they are post-update weights re-rolled on the
historical start ticks (one PPO update later, fresh torch RNG stream). Retraining was out of
scope. The real logged trace of all 75 updates is also ingested.

| metric | update 1 (exact) | update 57 (best, approx) | update 75 (final, approx) |
|---|---|---|---|
| total (agent,step) entries | 31660 | 38290 | 31290 |
| decision entries | 1240 (**3.92%**) | 3829 (12.07%) | 3333 (10.65%) |
| risk>0.50, all entries | 1593 | 1540 | 1636 |
| risk>0.50, **decision** entries | **67** | 352 | 202 |
| risk>0.18, decision | 70 | — | — |
| risk<0.10, decision | 1168 | — | — |
| STAY selected (all / at risk>0.50) | 549 / 20 | — / **290** | — / **157** |
| EDGE selected (all / at risk>0.50) | 517 / **35** | — / **25** | — / **22** |
| CLOUD selected (all / at risk>0.50) | 174 / 12 | — / 37 | — / 23 |
| PREEMPTIVE_REROUTE legal / selected | 8 / **0** | — / 0 | — / 0 |
| useful high-risk EDGE samples | **35** | 25 | 22 |
| per substantive minibatch (mean/min/max) | 8.75 / 6 / 12 | 6.25 / 4 / 9 | 5.50 / 2 / 10 |
| **gradient weight share of high-risk EDGE** | **2.26%** (max 4.20%) | **0.65%** | **0.53%** |

Three findings:

1. **96% of high-risk states are not decision states** (67 of 1593 at update 1) — the node has
   no task to move. The usable high-risk population is far smaller than the risk distribution
   suggests.

2. **The policy moved *away* from EDGE at high risk as training progressed.** Share of high-risk
   decisions taking EDGE: **52% → 7.1% → 10.9%**; STAY: 30% → 82% → 78%. This is not a policy
   that failed to learn. It learned successfully, in the wrong direction — exactly what D1
   predicts when the target's sign is inverted.

3. **Starvation is not the mechanism.** No *substantive* minibatch is ever empty of high-risk
   EDGE samples (min 2–6). But those samples carry only **0.53%** of the actor gradient weight
   at the end, because the actor loss divides by `denom` = decision entries in the minibatch
   (~248 → ~830).

**Production quirk found (real, not a replica artifact):** `mappo.py:325-334` computes
`mb_size = T // n_mb` then iterates `range(0, T, mb_size)`, which yields **n_mb + 1** chunks
whenever `n_mb` does not divide `T`. At update 75 (T=3129, mb_size=782) the starts are
0/782/1564/2346/**3128** — a 5th minibatch of **1 timestep**. So 4 of every 20 actor optimiser
steps are driven by a 1–2 timestep sample. See §4 for its gradient.

---

## 4. D4 — PPO update instrumentation (dry-run, on deepcopies)

The k1 KL concern is correct and is now quantified. All numbers below come from a replica
verified byte-identical to production (`max|Δ| = 0.00e+00`); the only differences are additive
measurements. **The training algorithm was not changed.**

| statistic | update 1 | update 57 (best) | update 75 (final) |
|---|---|---|---|
| **k1 KL** (production's metric) | +3.54e−03 | +5.07e−04 | +3.59e−04 |
| — minibatches with k1 **negative** | 2/20 | 1/16 | **11/20** |
| **k3 KL** (new diagnostic) | 7.09e−03 | 8.73e−05 | **1.88e−06** |
| — minibatches with k3 negative | 1 (−7e−10, float noise) | 1 (−2e−9) | 1 (−1e−9) |
| clip fraction | 0.108 | 0.0013 | **0.0000** |
| entropy | 0.717 | — | 0.307 |
| explained variance | 0.018 | — | 0.836 |
| adv std | 3.73 | — | 3.03 |
| actor grad norm (substantive mbs) | 0.208 | 0.099 | 0.096 |
| critic grad norm | ~6.2 | — | 12.1 |
| **policy ratio span** | [0.704, 1.286] | [0.916, 1.361] | **[0.990, 1.011]** |

**k1 is unfit for purpose here** — at update 75 it is negative in 11 of 20 minibatches, so its
mean (+3.59e−04) is a cancellation of noise. k3 gives the honest answer: **1.9e−06 nats**, about
four orders of magnitude below a typical PPO target of 1e−02. Together with `clip_frac` exactly
0.0000 for the final 12 recorded updates and a ratio span of ±1.1%, **the policy is numerically
frozen.** The stall is real and now has a clean metric.

### Is the gradient small, or large but cancelling? — **large and cancelling**

Full-batch policy gradient at ratio==1, each restricted to a subset but divided by the *same*
`denom` production uses, so the norms are comparable and additive (update 75):

| subset | entries | ‖grad‖ |
|---|---|---|
| all decision entries (the actual update) | 3333 | **0.0556** |
| low-risk, not EDGE | 2723 | 0.1059 |
| low-risk, EDGE | 398 | 0.1165 |
| high-risk, not EDGE | 180 | 0.0198 |
| high-risk, EDGE | 22 | 0.0183 |
| entropy term | 3333 | 0.0128 |

Σ‖subsets‖ = 0.2605 but ‖Σ‖ = 0.0556 → **79% of the gradient magnitude cancels.** Cosines:

| pair | cosine |
|---|---|
| low-risk EDGE ↔ low-risk not-EDGE | **−0.828** |
| high-risk EDGE ↔ low-risk not-EDGE | **−0.721** |
| high-risk EDGE ↔ high-risk not-EDGE | −0.551 |
| **all-decision ↔ high-risk EDGE** | **−0.243** |
| all-decision ↔ entropy term | −0.700 |

The net update is *negatively* aligned with the high-risk EDGE direction: whatever the high-risk
samples ask for, the aggregate update does not deliver — and per D1 their ask is wrong-signed
anyway. High-risk EDGE is 33.0% of the resulting aggregate norm from 0.66% of the samples, so
per-sample its pull is not weak; it is drowned and opposed.

**Degenerate-minibatch damage (new finding).** The 1-timestep 5th minibatch at update 75
produces `actor_grad_norm_preclip` = **4.42** — 46× the substantive minibatches' 0.096 — and is
**clipped** to 0.5, with `policy_loss` = 0.707. At update 1 a 2-timestep chunk likewise hit
0.623 and was clipped. So roughly 1 in 5 actor steps is a clipped, single-sample gradient. This
does not cause the D1 sign inversion, but it injects noise at 20% of update steps.

---

## 5. Verdict on the original (Phase 1) diagnosis: **PARTIALLY CONFIRMED**

| Phase 1 mechanism | verdict | evidence |
|---|---|---|
| **#1 Payoff lands at/beyond the GAE horizon** | **REFUTED** | k50 = 7 steps, k90 = 110 steps, both inside the 167-step horizon; median `gae_capture` 0.802; the reward path delivers +2.95 of +3.36. The p9 `horizon: 363–373` field was episode-length-remaining, which I misread as payoff delay. |
| **#2 Critic residual is as large as or larger than the effect** | **CONFIRMED, and stronger than stated** | Not merely large noise — a *risk-correlated systematic bias*: residual +0.76 (lo) → −2.63 (mid) → −5.21 (hi), monotone in risk, exceeding the +3.18 true effect and inverting its sign. It survives batch-level advantage normalisation, which subtracts only a scalar. |
| **#3 High-risk samples are diluted in the gradient** | **CONFIRMED, but secondary** | 0.53–2.26% of gradient weight; 79% overall cancellation; cos(net, high-risk EDGE) = −0.243. Real, but it throttles a signal that already has the wrong sign, so fixing dilution alone would only make the policy learn the wrong thing faster. |

**Revised diagnosis.** This is not a learning stall in the sense of "no gradient reaches the
policy". The policy *did* learn, converge, and freeze (k3 KL 1.9e−06, clip_frac 0.0000, ratio
span ±1.1%) — it learned to STAY at high risk (EDGE 52% → 11%) because the advantage target it
was given says STAY is better at high risk, by −2.87. The root cause is a **mis-specified
learning target produced by a critic that under-fits a rare, high-variance stratum**, not the
reward, not the horizon, and not the observation.

**Directly relevant to the Sprint 7 constraint:** the reward formulation is now *proven
sufficient* by a concrete test. The true, exact, reward-derived advantage of migrating to a
neighbour edge under high predicted failure risk is **+3.18 ± 0.33** (n=251) and rises
monotonically with risk. There is no case for redesigning the reward, and no case for a
`if risk > threshold → migrate` rule: the reward already contains that policy as its optimum.

---

## 6. Recommended first intervention for Rung 1

**Target the critic's risk-correlated residual. Do not touch the reward, the actor, or the
action space; do not add high-risk sampling or BC initialisation yet.**

Rationale, straight from the measurements: the reward path already delivers 88% of the true
signal with the correct sign (D2), so the reward and horizon are adequate. The V terms
contribute −5.81 and flip it (D2). The residual is monotone in risk (D2) and survives
normalisation (D1c). Every other candidate intervention — high-risk oversampling, BC init,
entropy schedules, `cloud_slots` — either amplifies a wrong-signed target or is irrelevant to
it. Fixing dilution (#3) without fixing the sign (#2) would make the policy learn the wrong
behaviour *faster*.

**The cheapest decisive test, and it needs no new training run.** Refit only the critic offline
on the already-collected returns from the D3 buffers (more critic epochs / higher critic LR,
actor untouched), then recompute the D1 agreement statistics with the refitted critic:

- **Success:** high-risk sign agreement rises from 29.0% toward >50%, high-risk EDGE GAE mean
  turns positive, and the residual-vs-risk slope flattens toward 0.
- **Failure:** the residual stays monotone in risk even at convergence, which would mean the
  critic *cannot* represent it from the 489-dim global state — a capacity/representation problem
  requiring a different fix.

Either outcome is informative, it costs one offline fit rather than a 600-episode campaign, and
it keeps the ladder small as instructed. Only if that test succeeds should a real training arm
be launched — and the primary metric there must remain the risk-conditioned behaviour
(P(MIGRATE|risk) sweep, sensitivity span, zero-risk ablation), not reward.

Secondary items to fold in when a training arm is eventually run, both cheap and both defects
rather than tuning choices:
- `mb_size = T // n_mb` with `range(0, T, mb_size)` creating a 1–2 timestep 5th minibatch whose
  clipped gradient drives 20% of actor steps (§3, §4).
- Log **k3** KL alongside k1; k1 is negative in 11/20 minibatches at update 75 and cannot
  detect the freeze (§4).

---

## 7. Artifacts

**New (all created by Rung 0, `_diag_`/`diag_S7_` prefixed):**

| file | bytes | contents |
|---|---|---|
| [marl/_diag_rung0.py](python-ai/marl/_diag_rung0.py) | 62214 | the whole Rung 0 probe; imports production code, reads checkpoints, never writes one, never steps an optimiser on a loaded model (every PPO probe runs on `copy.deepcopy`) |
| [diag_S7_D1_advantage_fidelity.json](python-ai/saved_models/marl/diag_S7_D1_advantage_fidelity.json) | 873419 | D1: all 745 pairs, 9 cells, agreement block, legacy 0.18 split |
| [diag_S7_D2_horizon_residual.json](python-ai/saved_models/marl/diag_S7_D2_horizon_residual.json) | 16597 | D2: horizons, payoff delay, λ^K weights, stratified residual, noise floor |
| [diag_S7_D3_sample_census.json](python-ai/saved_models/marl/diag_S7_D3_sample_census.json) | 35985 | D3: 3 snapshots, exact per-minibatch counts, reproduction check, 75 recorded updates |
| [diag_S7_D4_ppo_update.json](python-ai/saved_models/marl/diag_S7_D4_ppo_update.json) | 52443 | D4: k1-vs-k3, grad norms, per-minibatch trace, gradient decomposition, replica fidelity |
| `run_S7_rung0_d1d2_train.log`, `run_S7_rung0_d3d4.log` | 2333, 2922 | run logs |
| this report | — | `SPRINT_7_RUNG0_REPORT.md` |

**Preservation verified (md5, after all work):** 67/67 pre-existing artifacts in
`saved_models/marl/` unchanged; 20/20 production `marl/*.py` files unchanged. Manifests in
`saved_models/marl/_rung0_integrity/`. Note `git status` shows both `marl/` and
`saved_models/marl/` are untracked, so git provides no protection here — hence the manifests.

---

## 8. Limitations

1. **D1/D2 use a greedy policy**, chosen because it makes the true advantage exact. The critic
   was trained on sampled trajectories, so part of the high-risk residual may be distribution
   shift. This weakens the *attribution* but not the *conclusion*: greedy replay earns higher
   returns than training-time sampling (mean +13.07), which would push the residual *positive*,
   yet it is −5.21 at high risk. The bias is if anything understated.
2. **D3/D4 updates 57 and 75 are approximate** — post-update weights re-rolled on historical
   start ticks, one PPO update later, fresh torch RNG. Update 1 is bit-exact and verified.
   The direction of every trend is corroborated by the 75 recorded updates.
3. **Middle-risk n is small** (42 states; 2 each in pools B and C). Per instruction no synthetic
   states were created, so mid-risk rows are indicative, not inferential. The lo/hi contrast
   carries the argument.
4. **Single seed (20260818), single arm (A0)**, as required for comparability with Sprint 6.5.
   The sign inversion is ~6 s.e. from zero at n=331, but seed-generality is untested.
5. **`gae_capture` means are not reportable** (heavy-tailed ratio); medians are used throughout.
6. **One checkpoint's critic.** The residual-vs-risk monotonicity is measured at the final
   checkpoint; whether it is present from initialisation or develops during training is not
   established, and the Rung 1 test above would partly answer it.
7. `PREEMPTIVE_REROUTE` is legal in only 8–16 decision entries and was **never selected** —
   effectively a dead action, corroborating Phase 1. Not investigated further here.

**Rung 0 is complete. No training was run; Rung 1 is not implemented.**
