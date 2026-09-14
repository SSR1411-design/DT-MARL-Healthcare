# SPRINT 7 — RUNG 1 REPORT

**Offline critic diagnostic. No PPO training was run. No production file was modified.**

Probe code: `python-ai/marl/_diag_rung1_critic.py` (new, self-contained)
Artifacts: `SPRINT_7_RUNG1_critic_fit_and_residual.json`,
`SPRINT_7_RUNG1_deviation_agreement.json`,
`SPRINT_7_RUNG1_minibatch_tail_issue.json`
Logs: `run_S7_rung1_fit.log`, `run_S7_rung1_deviations.log`

**Verdict up front: CONFIRMED.** The critic is the source of the wrong-signed
learning target, and the implicated variable is **target construction**, not
value clipping. Refitting only the critic — changing nothing else — moves
high-risk sign agreement from **26.3% to 60.7%** and flips the high-risk
MIGRATE_EDGE advantage from **−2.867 to +0.840** against a truth of **+3.363**.

---

## 1. Data used

No new PPO training run was launched. No new environment interaction was
performed beyond the forced counterfactual replays Rung 0 already specified.

| item | value |
|---|---|
| model | `mappo_A0_cpu_repro.pth` (Rung 0's arm A0, seed 20260818) |
| device | cpu |
| window | `train`, ticks [9, 491] |
| episode starts | `[9, 41, 73, 105, 138, 170, 202, 234, 266, 298, 330, 362, 395, 427, 459, 491]` — **identical to Rung 0** |
| baseline mean reward | **+13.0725** — identical to Rung 0, confirming apples-to-apples |
| GAE replica vs `MAPPO.compute_gae` | `ok=True`, `max|dadv| = 0.0`, `max|dret| = 0.0` |
| critic-fitting dataset | 5875 timesteps × 10 agents = 58 750 state–return pairs, `state_dim=489` |
| train / val split | 4459 / 1416 timesteps; val = whole episodes [12, 13, 14, 15] |
| episodes hitting the 400-step limit | **0 / 16 → the MC targets are EXACT** |
| deviation set | Rung 0's exact **583 states / 745 pairs**, read out of `diag_S7_D1_advantage_fidelity.json` |
| replay mismatches | **0 / 745** (three-way identity check on trail, `team_r`, `obs`) |
| forced replays | 745 |

Two design points that make this reusable rather than a fresh experiment:

- The deviation pairs are **read from the Rung 0 artifact**, not re-derived. The
  same states, same agents, same reference actions, same seeds. Only the critic
  differs between arms.
- Because no episode was truncated, the Monte-Carlo discounted return is the
  **exact** discounted return, not an approximation. This matters: it gives a
  target that contains **no critic term at all**, which is the only way to
  measure the critic's bias without the measurement depending on the critic.

### Why the residual is measured differently from Rung 0

Rung 0 reported the residual as `ret − V`, and in production
`ret = adv + val` (`mappo.py:298`), so `ret − V` **is** GAE — a quantity that
already contains the critic. Rung 1 measures the residual as
**`MC_return − V`**, which is critic-free. Both are reported below; they agree
in direction and differ in magnitude, and the difference is itself informative.

---

## 2. Offline critic methodology

Four arms, strictly one variable apart. Every arm gets a **fresh**
`CentralisedCritic` with production's architecture (`[256, 256]`, input
`state_dim + n_agents = 499`); evaluation uses a `copy.deepcopy` of the agent
with `.critic` swapped, so the loaded production critic is never mutated. No
actor gradient is taken anywhere in this rung. No production optimiser is
stepped. No production checkpoint is written.

| arm | target | loss | isolates |
|---|---|---|---|
| **C0** | — (production critic as loaded) | — | reference; reproduces Rung 0 |
| **C1** | MC discounted return | plain MSE | *can* the critic fit the true return at all? |
| **C2** | λ-return recomputed from the critic being trained (production's construction) | plain MSE | **target construction** |
| **C3** | MC discounted return | production's **clipped** value loss at `value_clip_eps=0.2` | **value clipping** |

Shared fitting protocol: `lr = 1e-3` (= production `lr_critic`), batch 256,
max 600 epochs, early stopping patience 40, `torch.manual_seed` fixed.
**Every arm is early-stopped on held-out MC MSE** — the same yardstick for all
four, so C2 is not judged on its own self-referential objective.

Determinism was verified rather than assumed: the fit phase was run twice (once
standalone, once again inside the deviation phase) and all four arms reproduced
their best-epoch and best-loss values **bit-identically**.

Both suspects identified by reading the production code were tested:

- **Suspect (a) — self-referential target.** `ret = adv + buf.val[:T]` is a
  λ-return bootstrapped off the critic's own predictions. → arm C2.
- **Suspect (b) — value clipping.** `value_clip_eps = 0.2` is in absolute
  reward units against a return sd of 10.06, so it rate-limits the critic to
  ±0.2 of movement per update per state. → arm C3.

---

## 3. Before vs after critic metrics

| arm | MSE train | MSE val | MAE val | EV train | EV val | best epoch | mean V |
|---|---|---|---|---|---|---|---|
| **C0** prod | 43.177 | 52.398 | 5.488 | +0.6545 | +0.6815 | — | **+4.976** |
| **C1** MC | **5.318** | **22.221** | **3.308** | **+0.9438** | **+0.8208** | 93 / 134 run | +1.756 |
| **C2** λ-ret | 58.583 | **87.305** | 6.771 | +0.4886 | **+0.3101** | **4** / 45 run | −1.272 |
| **C3** MC+clip | 5.537 | 24.165 | 3.448 | +0.9420 | +0.8040 | 125 / 166 run | +1.616 |

True return: mean **+1.622**, sd **10.058**.

Three things to read off this table:

1. **C1 more than halves val MSE (52.4 → 22.2) and lifts val EV 0.681 → 0.821.**
   The critic architecture is fully capable of fitting these returns. Capacity
   was never the constraint.
2. **C3 ≈ C1** (24.2 vs 22.2; EV 0.804 vs 0.821). Production's clipped value
   loss costs about **9% of MSE and nothing structural**. Suspect (b) is
   **refuted** — clipping is not the cause.
3. **C2 is catastrophic and early-stops at epoch 4.** Trained with production's
   own target construction, the critic's ability to predict the true return gets
   *worse from epoch 4 onward* and settles at val MSE 87.3 — **1.7× worse than
   the production critic** and 3.9× worse than C1. Suspect (a) is implicated.

Per the instruction not to stop at a lower MSE: the MSE is not the finding.
Sections 4–6 are the finding.

---

## 4. Residual-vs-risk analysis

Residual = `MC_return − V`. Negative means the critic **over-predicts**.
Buckets: `lo` risk < 0.10 (n = 55 710), `mid` (n = 262), `hi` risk > 0.50
(n = 2778).

### 4a. The core defect, in one row

| arm | V at hi-risk | true MC return at hi-risk | over-prediction |
|---|---|---|---|
| **C0** | **+3.657** | **−0.723** | **+4.380** |
| C1 | −0.396 | −0.723 | +0.327 |
| C2 | −3.599 | −0.723 | −2.875 |
| C3 | −0.517 | −0.723 | +0.206 |

**The production critic values high-risk states at +3.66 when the truth is
−0.72.** High-risk states are genuinely bad — their true return is negative —
and the critic reports them as better than average. C1 and C3 get them right.

### 4b. Slope, correlation, stratified means

| arm | slope (all) | intercept | Pearson | Spearman | slope (**hi-only**) | lo→hi swing |
|---|---|---|---|---|---|---|
| **C0** | **−1.5916** | −3.2202 | −0.0434 | −0.0123 | **−5.5960** | **−1.0743** |
| **C1** | **−0.2869** | −0.1102 | −0.0149 | −0.0017 | **−0.2542** | −0.2053 |
| **C2** | −0.2019 | **+2.9111** | −0.0043 | −0.0014 | **−5.6920** | −0.0158 |
| **C3** | −0.3144 | +0.0324 | −0.0158 | −0.0029 | −0.3405 | −0.2258 |

| arm | resid lo | resid mid | resid hi | EV lo | EV mid | EV hi |
|---|---|---|---|---|---|---|
| **C0** | −3.3057 | −2.7307 | **−4.3800** | 0.6636 | 0.7193 | **0.6328** ← worst |
| **C1** | −0.1217 | −0.7712 | −0.3270 | 0.9072 | 0.9216 | 0.9045 |
| **C2** | +2.8911 | +3.7412 | +2.8753 | 0.4301 | 0.4557 | 0.4860 |
| **C3** | +0.0196 | −0.6448 | −0.2062 | 0.9015 | 0.9217 | 0.8862 |

- **Slope: −1.5916 → −0.2869, an 82% reduction.** Within the high-risk stratum
  it falls from **−5.5960 to −0.2542, a 95% reduction.**
- **Rung 0's "high-risk is the worst-fit stratum" is fixed.** C0's EV is lowest
  at high risk (0.633 vs 0.664/0.719); after refit EV is flat across buckets
  (0.907 / 0.922 / 0.905). The rare stratum is no longer under-fitted.
- **C2 keeps the steep within-high-risk slope: −5.6920, statistically
  indistinguishable from C0's −5.5960.** This is the single-variable
  attribution. The λ-return target does not remove the risk-correlated bias;
  the MC target removes 95% of it.

**Honest note on C2.** C2 is not a faithful reproduction of C0's failure *in
sign*: C0 over-predicts (residual −3.3/−4.4), C2 under-predicts (+2.9). What C2
reproduces is the **steep within-high-risk slope** and the **EV collapse**. So
target construction is implicated by the slope and by C2's inability to track
the true return, but C2 is a cold-start refit with a self-referential target
and may diverge in ways a warm production run does not. Listed in §7 limits.

### 4c. Noise floor — the cleanest statistic in this rung

Under a deterministic policy the true advantage of the action actually taken is
**identically zero**, so any non-zero GAE there is pure estimator error. Same
calibration constant Rung 0 used.

| arm | lo (n=290) | mid (n=42) | hi (n=251) | **lo→hi swing** | sd at hi |
|---|---|---|---|---|---|
| **C0** | +0.671 | −2.628 | **−4.570** | **−5.241** | 7.45 |
| **C1** | −0.248 | −1.784 | −1.195 | **−0.947** (−82%) | **3.41** |
| **C2** | +1.554 | −2.115 | −3.957 | **−5.511** (worse) | 8.84 |
| **C3** | −0.335 | −2.183 | −1.509 | −1.175 (−78%) | 3.85 |

C0 reproduces Rung 0 exactly (+0.671 / −2.628 / −4.570). C1 removes 82% of the
risk-correlated error **and halves its spread**. C2 removes none of it.

Note the magnitude gap between §4b's per-state differential (−1.07) and this
−5.24. GAE accumulates the critic's errors along the remaining trajectory with
weight (γλ)^k, so a per-state differential appears amplified in the learning
signal. I did not derive the exact multiplier; I report the two numbers and the
qualitative reason, not a fitted relationship.

---

## 5. Advantage sign agreement

745 deviation pairs; high-risk subset n = 331. Truth is reported under **both**
attributions (see §7 for why this matters).

### High risk (risk > 0.50), n = 331

| arm | sign agr. (team) | Pearson | Spearman | sign agr. (**own**) | Pearson | Spearman | GAE mean |
|---|---|---|---|---|---|---|---|
| **C0** | **0.263** | −0.1529 | −0.2725 | **0.272** | −0.2376 | −0.3483 | **−3.135** |
| **C1** | **0.607** ✅ | **+0.4040** | **+0.3716** | **0.647** ✅ | **+0.5206** | **+0.5148** | **+0.169** |
| **C2** | 0.281 ❌ | −0.1492 | −0.2673 | 0.290 ❌ | −0.2769 | −0.3507 | −2.339 |
| **C3** | **0.592** ✅ | +0.3603 | +0.3533 | **0.595** ✅ | +0.4260 | +0.4236 | −0.082 |

True mean: **+2.336** (team) / **+1.422** (own). True fraction positive 0.659 /
0.680. GAE fraction positive: C0 **0.356** → C1 **0.508**.

**Every correlation flips sign.** C0's GAE is *anti*-correlated with truth
(Spearman −0.27 team, −0.35 own); C1's is positively correlated (+0.37, +0.51).
C2 stays anti-correlated. This is the primary success test and C1/C3 pass it.

### All buckets pooled, n = 745

| arm | sign agr. (team) | Spearman | sign agr. (own) | Spearman |
|---|---|---|---|---|
| C0 | 0.381 | −0.2414 | 0.358 | −0.3352 |
| C1 | **0.583** | **+0.2547** | **0.640** | **+0.4576** |
| C2 | 0.365 | −0.2588 | 0.361 | −0.3697 |
| C3 | 0.557 | +0.2449 | 0.599 | +0.3883 |

### Per bucket (sign agreement, team / own)

| arm | lo | mid | hi |
|---|---|---|---|
| C0 | 0.486 / 0.441 | 0.386 / 0.318 | 0.263 / 0.272 |
| C1 | 0.573 / 0.646 | 0.477 / 0.545 | 0.607 / 0.647 |
| C2 | 0.435 / 0.416 | 0.409 / 0.432 | 0.281 / 0.290 |
| C3 | 0.524 / 0.597 | 0.568 / 0.636 | 0.592 / 0.595 |

C0 degrades monotonically as risk rises (0.486 → 0.386 → 0.263) — the signature
of a risk-correlated fault. C1 is roughly flat and above 50% in every bucket
under own-attribution.

---

## 6. EDGE / STAY / CLOUD ordering

High risk, team attribution. Values are mean GAE ± s.e.

| arm | STAY (n=101) | MIGRATE_EDGE (n=230) | MIGRATE_CLOUD |
|---|---|---|---|
| **TRUE** | −0.001 ± 0.16 | **+3.363 ± 0.31** | not estimable |
| **C0** | −3.748 ± 0.76 | **−2.867 ± 0.41** ❌ wrong sign | not estimable |
| **C1** | −1.358 ± 0.30 | **+0.840 ± 0.27** ✅ | not estimable |
| **C2** | −2.995 ± 0.89 | −2.051 ± 0.52 ❌ | not estimable |
| **C3** | −1.447 ± 0.34 | +0.517 ± 0.29 ✅ | not estimable |

Own attribution, same states: TRUE STAY −0.238 ± 0.11, TRUE EDGE
**+2.152 ± 0.21**. Same conclusion.

Low risk, team attribution: TRUE STAY +0.121 ± 0.15, TRUE EDGE +0.618 ± 0.29;
C0 STAY +0.484, EDGE +0.359; C1 STAY −0.369, EDGE −1.016.

**Two honest qualifications to this section.**

1. **The relative ordering at high risk was never inverted.** EDGE > STAY holds
   in *all four* arms, C0 included. What was wrong in C0 is the **level**: both
   actions sit at −2.9 and −3.7, so PPO — which subtracts a single global batch
   mean, not a per-state baseline — sees "everything I do in high-risk states is
   bad" and learns to avoid the states rather than to migrate within them. The
   success criterion phrased as "ordering approaches EDGE > STAY > CLOUD" is
   therefore satisfied but was not the discriminating test; the discriminating
   tests are the sign (§5) and the level (this table). At low risk C1 does
   invert the true EDGE > STAY ordering (−1.016 vs −0.369), which is a genuine
   remaining defect of the refit, not a success.
2. **CLOUD is structurally unmeasurable in this design.** There are 162
   cloud-legal states in the set, and in **all 162** the greedy policy *chooses*
   CLOUD. CLOUD therefore never appears as a deviation candidate, so its
   advantage cannot be estimated by forced deviation from a greedy baseline. The
   required three-way EDGE > STAY > CLOUD ordering is **not testable from Rung
   0's pair set**; only EDGE > STAY is. Measuring CLOUD needs a deviation pool
   built from states where CLOUD is legal but *not* chosen, which this set does
   not contain. Reported as unavailable rather than approximated.

---

## 7. Verdict: **CONFIRMED**

The critic is the source of the wrong-signed learning target, and the
implicated variable is **target construction**.

| success criterion (pre-registered) | before | after (C1) | result |
|---|---|---|---|
| high-risk sign agreement > 50% | 26.3% / 27.2% | **60.7% / 64.7%** | ✅ PASS |
| residual-vs-risk slope substantially reduced | −1.5916 (hi-only −5.5960) | **−0.2869 (hi-only −0.2542)** | ✅ PASS (−82% / −95%) |
| EDGE advantage correctly positive at high risk | −2.867 | **+0.840** | ✅ PASS |
| ordering approaches EDGE > STAY > CLOUD | EDGE > STAY | EDGE > STAY | ⚠️ PARTIAL — already held in C0; CLOUD not estimable |

Single-variable attribution:

- **C3 passes while keeping production's clipped value loss** → value clipping
  is **not** the cause. One of my two suspects, refuted by its own arm.
- **C2 fails while changing only the target back to production's construction**
  → **target construction is the cause.** Its within-high-risk residual slope
  (−5.692) matches the unmodified production critic's (−5.596).

### The Rung 0 confound I raised against my own analysis is refuted

Rung 0's "true" advantage was **team-level** (`team_r.sum()`) while `gae_raw` is
**per-agent**. Since migration moves a task and its reward to another node, a
team gain can be a per-agent loss — which would explain the sign disagreement
with **no critic fault at all**. Rung 1 therefore computed every true advantage
under both attributions. The result:

| | team truth | own truth |
|---|---|---|
| C0 high-risk sign agreement | 0.263 | 0.272 |
| C1 high-risk sign agreement | 0.607 | 0.647 |
| C0 high-risk Spearman | −0.2725 | −0.3483 |
| C1 high-risk Spearman | +0.3716 | +0.5148 |

The own-agent attribution gives the **same conclusion, slightly more strongly**.
Credit attribution is not the explanation. Rung 0's conclusion survives, and the
per-agent truth confirms rather than undercuts it.

### Where Rung 0 overstated the magnitude

Rung 0 reported the critic residual as +0.67 → −2.63 → −4.57, a swing of −5.24,
using `ret − V` (= GAE, which contains the critic). The critic-free measure
(`MC − V`) gives −3.31 → −2.73 → −4.38, a swing of only **−1.07**, and shows the
critic over-predicting *everywhere*, not flipping sign across buckets. The
direction and the conclusion are unchanged; the **per-state** bias is about 1
reward unit, and it is GAE's accumulation along the trajectory that turns it
into the ~5-unit swing in the learning signal. Rung 0's numbers were correctly
computed but were a property of GAE, not of the critic alone.

### Limitations

1. **This is off-policy evaluation of a counterfactual.** The refit is measured
   on the frozen A0 policy's own state distribution. A better critic changes the
   policy, which changes the distribution. Nothing here proves the improvement
   survives online training — that is precisely why §9 is a training experiment
   and not a victory claim.
2. **Sign agreement is 60.7%, not ~100%.** The residual is *largely* removed,
   not eliminated: high-risk noise floor is still −1.195 vs −0.248 at low risk,
   and C1's high-risk GAE mean (+0.169) is correctly signed but a large
   *underestimate* of the +2.336 truth. C3 is weaker still, at −0.082.
3. **C2 is a suggestive, not clean, reproduction** of C0's failure (§4b note).
4. **C1 inverts the true low-risk EDGE > STAY ordering** (§6 note 1).
5. **CLOUD advantage is not measurable** from this pair set (§6 note 2).
6. **Single seed, single checkpoint**, 16 episodes / 5875 timesteps, val = 4
   episodes. The mid-risk bucket has only 262 timesteps and 42 deviation states;
   its numbers are thin and should not be leaned on.
7. **C1 overfits somewhat** — train EV 0.944 vs val 0.821. Early stopping was
   used but the gap is real, and 58 750 pairs from 16 episodes are highly
   temporally correlated, so val is not fully independent.
8. `lr` and schedule were not tuned per arm; C2 might do better with a different
   lr, though its failure mode (immediate divergence from the true return) is
   not obviously an lr artifact.

---

## 8. Exact implication for PPO

Per instruction, **nothing was implemented**. This is the change that the
experiment says is required, and its exact location.

**The one line at issue** — `python-ai/marl/mappo.py:298`:

```python
ret = adv + buf.val[:T]     # λ-return, bootstrapped off the critic's own V
```

The critic is regressed onto a target computed from its own current predictions.
C2 shows that this target, and not the loss, the clipping, the architecture, or
the feature set, is what leaves the risk-correlated bias in place.

**Smallest change that the evidence supports:** compute the **critic's
regression target** as the within-episode Monte-Carlo discounted return,
bootstrapping *only* on genuine time-limit truncation. Roughly 5 lines at
`mappo.py:298`, behind a config flag that defaults to current behaviour so no
existing artifact or arm changes.

**Constraint compliance — read this before objecting that it changes λ:** the
actor's GAE stays *exactly* as it is, γ = 0.999 and λ = 0.995 untouched. What
changes is only the target the **critic** regresses onto. That distinction is
the whole point of the C1-vs-C2 contrast, and honouring it keeps the "do not
modify GAE gamma/lambda" constraint intact.

**What must NOT change at the same time:**

- **Do not touch `value_clip_eps`.** C3 proves clipping is not the culprit and
  it costs only ~9% of MSE. Changing it would introduce a second simultaneous
  variable. My own suspicion about it was wrong; leave it alone.
- Do not touch the reward, environment, action space, observation space, risk
  predictor, `cloud_slots`, actor architecture, γ, λ, or any PPO
  hyperparameter.
- **Do not fix the minibatch tail issue in the same run** (§ below).

**Cost of the change:** MC targets have higher variance than λ-returns. C1's
val EV of 0.821 shows they remain fittable at this scale, but variance is the
real trade-off being accepted, and it is the main way the next experiment could
fail.

### Minibatch tail issue — measured and documented only, NOT fixed

Kept deliberately isolated from the critic experiment so it cannot confound the
diagnosis. Artifact: `SPRINT_7_RUNG1_minibatch_tail_issue.json`.

`mb_size = max(1, T // n_mb)` followed by `for start in range(0, T, mb_size)`
yields `n_mb + 1` chunks whenever `n_mb ∤ T`. Scanning T ∈ [3000, 3299] with
`minibatches=4`: **75% of T values produce an extra chunk**, of size 1–3
(median 2).

Observed in the Rung 0 D4 traces:

| update | steps | chunk sizes | degenerate | degenerate grad norms | substantive mean | ratio |
|---|---|---|---|---|---|---|
| 01 init | 20 | {2, 791} | 4 | [0.0, 0.623, 0.0, 0.0] | 0.2075 | 3.0× |
| 57 best | 16 | {793} | 0 | — | 0.0994 | — |
| 75 final | 20 | {1, 782} | 4 | [0.125, 0.0, 0.0, **4.423**] | 0.0961 | **46.0×** |

So **4 of 20 actor update steps (20%)** come from a 1–2 timestep chunk. Most
produce a *zero* gradient (such a chunk often contains no decision entry at all,
and `denom` is clamped to 1), and occasionally one produces a gradient 46× the
substantive norm that is then clipped to 0.5. Notably, update 57 — the
**best** checkpoint — is the one where T happened to divide evenly and no
degenerate chunk existed. That is suggestive, not causal, on one observation.

Fix this in its own rung, after the critic change is evaluated.

### Preservation verified

- `_rung0_integrity/artifacts_before.md5`: **67 / 67 OK, 0 FAILED**
- `_rung0_integrity/code_before.md5`: **20 / 20 OK, 0 FAILED**
- Sprint 6, Sprint 6.5 and Rung 0 artifacts untouched; all new files carry the
  `SPRINT_7_RUNG1_` prefix. No commit, no push, no PR.

---

## 9. Smallest next controlled experiment

**Rung 2 — one variable, two arms, go/no-go.**

| arm | definition |
|---|---|
| **R2-base** | *No new run needed.* Reuse `mappo_A0_cpu_repro.pth` as the control. |
| **R2-mc** | Identical to A0 in every respect — seed 20260818, `--device cpu`, 600 episodes, same reward/env/actor/γ/λ/PPO hyperparameters, `value_clip_eps` unchanged — **except** the critic's regression target is the within-episode MC discounted return. |

One training run. Checkpoint name `mappo_R2_mc_target.pth`; nothing overwritten.

**Pre-registered success metrics. Reward is explicitly NOT the primary metric.**

1. **High-risk EDGE share over training.** Rung 0 measured A0 collapsing
   52% → 7.1% → 10.9%. Success = this does **not** collapse. This is the
   headline: it is the direct test of "does the policy use
   `predicted_failure_risk` to change its decision?"
2. **Re-run the Rung 0 D1 probe on the new checkpoint** (~13 min) → high-risk
   sign agreement and noise-floor lo→hi swing. Success = sign agreement > 50%
   *in a trained-online policy*, swing well below C0's −5.241. This closes the
   off-policy gap in §7 limit 1.
3. **Risk-sweep P(relocate) span.** Sprint 6.5 baseline 0.002–0.016; the BC
   ceiling on the same actor is 0.3935. Success = span moves decisively off the
   floor toward the BC ceiling.
4. **`clip_frac` and k3 KL** — confirm the gradient does not die as it did in
   all four Sprint 6.5 arms (`clip_frac` → 0.0000). Log **k3**, not k1: Rung 0
   found k1 negative in 11/20 minibatches, so it cannot detect the freeze.

**Decision rule, stated in advance so it cannot be moved afterwards:**

- Metrics 1 **and** 2 both pass → the critic fix is real; *then* look at reward,
  and only then consider high-risk oversampling or BC init (which Rung 0 showed
  would amplify a wrong-signed target if applied first).
- Metric 1 passes but 2 fails, or vice versa → do not proceed; investigate the
  disagreement.
- Both fail → the critic is not sufficient. Next suspects, one at a time, in
  this order: critic **input representation** (does a 489-dim flat global state
  let a 5%-frequency stratum be learned at all?), then **return
  normalisation**, then **temporal/state aggregation**.

**Replication before any claim.** One seed is a go/no-go, not a result. Nothing
about arm ordering gets asserted until ≥ 3 seeds agree — the explicit lesson of
Sprint 6.5. A single run that merely raises reward will **not** be reported as
success.

**Not in scope for Rung 2:** the minibatch tail fix, any reward change, any
`if risk > threshold → migrate` rule, and any change to the eight frozen
components listed in §8.

---

**Status: STOP.** Per instruction, the offline refit succeeded, so the
production critic change is *reported* (§8) and the next experiment is
*proposed* (§9) rather than implemented. No PPO training was run in this rung.
No production MAPPO file was modified. The minibatch tail issue is documented
and left unfixed. No commit, no push, no PR.
