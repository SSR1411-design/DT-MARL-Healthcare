# Sprint 7 — Rung 2.75: diagnostic-only rung

**Status: COMPLETE (diagnosis). No training was run. No production code was modified.**

Scope set by the Phase-2 brief: items **A–D** are diagnosis; item **E** is a
pre-registered recommendation for exactly **one** next rung, submitted for
approval *before* anything is trained.

| brief constraint | compliance |
|---|---|
| only offline analysis/replay on existing checkpoints and data | held — 9 probes, all forward-pass / CSV / replay only |
| do not modify production code | held — `mappo.py`, `train.py`, `env.py`, `rollout.py` md5-unchanged |
| do not train anything | held |
| do not fix multiple variables simultaneously | held — item E changes one config value |
| record useful interventions as hypotheses, do not implement | held — item E is a proposal, not a change |
| preserve all existing artifacts | held — 98/98 verify |
| hash/verify before and after | held — see *Integrity* below |
| prefix new artifacts with the rung name | held — every new file is `*RUNG2_75*` |
| reuse existing replay/diagnostic machinery | held — all probes import `marl._diag_rung0` |
| if an existing diagnostic is flawed, document the flaw and build the smallest correction | held — 9 flaws documented in §6, 3 of them mine, 2 of them reverse a prior conclusion |

## Integrity

Manifests taken before any diagnostic work, re-verified after:

```
saved_models/marl/_rung2_75_integrity/SPRINT_7_RUNG2_75_artifacts_before.md5   98/98 OK
saved_models/marl/_rung2_75_integrity/SPRINT_7_RUNG2_75_code_before.md5        28/28 OK
```

The code manifest covers all production modules (`mappo.py`, `train.py`,
`env.py`, `rollout.py`, `config.py`, `evaluate.py`, `risk_provider.py`, …) and
every pre-existing diagnostic from Rungs 0–2.5. All 28 unchanged. The nine
`_diag_rung2_75_*.py` files are additions and are correctly absent from the
"before" manifest.

"After" manifests were also written, so the next rung can verify against this
rung's end state:

```
SPRINT_7_RUNG2_75_artifacts_after.md5   111 entries  (98 pre-existing + 12 new JSON + this report)
SPRINT_7_RUNG2_75_code_after.md5         37 entries  (28 pre-existing + 9 new probes)
```

---

## 0. The one-paragraph answer

The actor is **not** frozen, **not** starved of high-risk samples, and **not**
suffering vanishing gradients. At the learning rate training actually used, a
*coherent* advantage signal moves π(MIGRATE_EDGE | high risk) from 0.08 to 0.53
and flips the argmax on 68% of high-risk entries **within one to two updates**.
The problem is upstream: the advantage vector the actor is fed carries, in
aggregate, **no more usable state–action information than a random permutation
of its own values** (real/shuffled gradient-norm ratio 1.30 for A0 with
permutation p = 0.105; 0.80 for R2 with p = 0.805 — both inside their own
nulls). PPO nevertheless sharpens on it, and sharpening — not the learning-rate
schedule — accounts for the trust-region collapse in **all five** existing arms
(schedule 6.7× uniformly; sharpening 15.8× A0, 152.0× R2, 30.7× A1, 1547.9× A2,
22.4× A3). The dominant term in the advantage's variance is a **per-state
offset** that contributes exactly zero in expectation: Var(raw)/Var(paired) =
21.522/5.785 = **3.72×**. So the stall is a **signal-variance** problem, and
the smallest test of that is to give the estimator more samples per update
without touching the estimator.

Separately, item A found that the metric this project has used as its headline
since Sprint 6 — high-risk EDGE share on each arm's own trajectory — is **not
identified**. Its sign reverses depending on whose states you evaluate on. Both
the Rung 2 "win" and the Rung 2.5 "loss" fail as policy-quality claims. On
state sets that privilege neither arm, R2 does beat A0, and R2 has a real risk
response where A0 has none — so the confound invalidated the *metric*, not the
*direction*.

---

## 1. Item A — the Rung 2 EDGE-share discrepancy

Brief: *"Find exactly how the original R2 training metric was calculated. Find
exactly how the later replay metric was calculated. Determine whether they
measure the same quantity … Do not invent a correction if the artifacts do not
support one … DO NOT select whichever result is flattering."*

### A.1 Both on-record figures are arithmetically correct

`_diag_rung2_75_edgeshare.py` rebuilds the metric from the frozen eval
artifacts and reproduces the Rung 2 headline **exactly**, entry for entry:

```
D0 parity check against the frozen eval artifacts:
  A0: replica n=546 EDGE=15  artifact n=546 EDGE=15  MATCH
  R2: replica n=384 EDGE=31  artifact n=384 EDGE=31  MATCH
```

Neither figure is an arithmetic error. **No correction is warranted and none
was invented.**

### A.2 They measure different quantities — a five-step definitional ladder

| rung | definition | A0 | R2 | gap |
|---|---|---|---|---|
| **D0** | **eval** window, **greedy**, mask-only, bucket ≥ 0.6 — *the Rung 2 headline* | 0.0275 (n=546) | 0.0807 (n=384) | **+0.0532** |
| D1 | **train** window, greedy, mask-only, bucket ≥ 0.6 | 0.0173 (n=579) | 0.0243 (n=371) | +0.0070 |
| D2 | train, greedy, mask-only, `risk > 0.50` | 0.0172 (n=581) | 0.0241 (n=373) | +0.0069 |
| D3 | train, greedy, **has_task**, `risk > 0.50` | 0.0172 (n=581) | 0.0241 (n=373) | +0.0069 |
| **D4** | train, **stochastic**, has_task, `risk > 0.50` — *what the Rung 2.5 census measures* | **0.0863** (n=705) | **0.0852** (n=760) | **−0.0011** |
| X | eval, greedy, has_task, `risk > 0.50` | 0.0274 (n=547) | 0.0805 (n=385) | +0.0530 |

Six definitional differences separate the two on-record numbers:

1. **evaluation window** — eval `[491, 698]` vs train `[9, 491]` (disjoint);
2. **policy** — greedy/deterministic vs stochastic sampling;
3. `has_task` filter present vs absent (the Rung 2.5 census is **mask-only**);
4. threshold form — production bucket `min(int(risk*5),4) >= 3` vs `risk > 0.50`;
5. number of episodes — 8 vs 32/40;
6. **start-tick source** — the Rung 2.5 census draws `training_start_ticks`
   (train.py's own RNG), not `rollout.episode_starts`.

Differences 3 and 4 are inert here (D1 → D2 → D3 moves the share by 0.0001).
Differences 1 and 2 carry everything:

* moving the **window** (D0 → D1) shrinks the gap from +0.0532 to +0.0070;
* switching to **stochastic** evaluation (D1 → D4) removes it entirely
  (−0.0011), raising *both* arms to ≈0.086.

**Answer to the brief's question:** the discrepancy is a *measurement
definition* difference — specifically window and greedy-vs-stochastic — not a
reproducibility failure, not a training-vs-final-checkpoint effect, and not an
arithmetic error. Under the policy training actually samples from (stochastic),
in the window training actually uses, **the two arms are indistinguishable on
this metric.**

### A.3 Episode-level inference on the surviving D0 gap

`_diag_rung2_75_edgeshare_cluster.py`, 32 clusters, cluster bootstrap over
episodes:

* **Rung 2 headline (D0-style, eval window):** A0 60/2174 = 0.0276 vs R2
  122/1418 = 0.0860; difference +0.0584, cluster SE 0.00901, **z = +6.48**,
  CI95 [+0.0417, +0.0770]; exact sign test **31/32 episodes, p < 0.0001**;
  design effect 1.45. On the on-record 8-episode sample the same contrast gives
  z = +2.86, p = 0.0042. **Survives episode-level inference.**
* **Rung 2.5 census contrast:** on 32 matched train starts, A0 77/803 vs R2
  83/852 — difference **+0.0015**, cluster **z = +0.10**, design effect 1.05.
  The on-record 8-episode figure's own z is **−0.64, p = 0.5206** — not the
  −1.44 previously quoted. **The Rung 2.5 "R2 is worse" contrast is not real at
  any sample size we have.**

### A.4 …but the metric is not identified, in either direction

`_diag_rung2_75_matched_states.py` cross-evaluates both actors at **identical
recorded states**, paired within state, clustered by episode. The diagonal
reproduces the headline exactly (`A0 60/2174 MATCH`, `R2 122/1418 MATCH`),
confirming it is the same measurement.

| states drawn from | π_A0 | π_R2 | paired R2−A0 | z | note |
|---|---|---|---|---|---|
| A0's own greedy trajectory | 0.1494 | 0.3837 | **+0.2343** | +33.96 | confounded: selected by A0 |
| R2's own greedy trajectory | 0.3366 | 0.1721 | **−0.1645** | −16.44 | confounded: selected by R2 |
| RANDOM (uniform over legal actions) | 0.1261 | 0.3449 | +0.2188 | +6.29 | policy-independent |
| UNION (A0's states pooled with R2's) | 0.2233 | 0.3002 | +0.0769 | +3.13 | symmetric |

Each arm scores **low on its own states and high on the other's**, and the sign
**reverses**. The mechanism is structural, not statistical: risk is high at a
host *because* tasks remain on it. STAY keeps an agent in a high-risk state;
MIGRATE_EDGE removes it from one. Conditioning on "currently in a high-risk
state" therefore selects for "just chose STAY" — **the metric has its own
outcome in its denominator.**

Consequently **both** the Rung 2 "win" and the Rung 2.5 "loss" fail as
policy-quality claims. This is stated in the direction that costs us the
headline, per the brief.

### A.5 The direction does survive, on neutral state sets

On the two state sets that privilege neither arm the sign is **consistent and
positive**: RANDOM +0.2188 (z = +6.29, R2 higher on 82.9% of entries) and UNION
+0.0769 (z = +3.13, 69.7%). The confound invalidated the metric, not the
direction of the underlying policy difference.

### A.6 The unconfounded replacement metric

The project's primary question is *"does the learned policy actually use
`predicted_failure_risk` to change its decision?"* That is a **risk response**,
not a share:

> Δ = π(MIGRATE_EDGE | risk ≥ 0.6) − π(MIGRATE_EDGE | risk < 0.2), over all
> decision entries of a **fixed** state set, plus Spearman(risk, π(EDGE)).

No policy appears in the state distribution, so the quantity is identified.

| state set | arm | π(EDGE) risk<0.2 | risk≥0.6 | **Δ** | z | Spearman |
|---|---|---|---|---|---|---|
| RANDOM | A0 | 0.1525 | 0.1261 | **−0.0264** | −1.86 | −0.0286 |
| RANDOM | R2 | 0.1425 | 0.3449 | **+0.2024** | +7.72 | +0.0777 |
| UNION | A0 | 0.2042 | 0.2233 | **+0.0191** | +1.62 | +0.0232 |
| UNION | R2 | 0.1423 | 0.3002 | **+0.1579** | +23.25 | +0.0860 |

On the policy-independent set, **R2 responds to risk and A0 does not.** On the
symmetric set both respond, R2 8× more strongly. The two neutral sets agree
(Spearman across arms +0.80).

**Honest caveats on this metric.** RANDOM is policy-independent but visits
states off-manifold for *every* policy (4,975 decision entries, 3.9% of
timesteps, vs A0's 43,519 at 36%); UNION is on-manifold but contains *both*
arms' selection effects rather than none. Requiring a result on **both** guards
against either weakness. Composition also differs: within the high-risk set,
R2's exposure is both rarer (4.6% vs 5.0% of decision entries) and less severe
(median risk 0.6799 vs A0's 0.8450), which is itself a consequence of R2
migrating more.

---

## 2. Item B — actor saturation and plasticity

### B.1 Static census (`_diag_rung2_75_edgeshare.py`)

| arm | high-risk n | saturated (max-prob > 0.99) | saturated on | `frac_legal_edge` |
|---|---|---|---|---|
| A0 train/greedy | 581 | **0.0000** (0) | — | 1.0000 |
| A0 train/stoch | 241 | 0.0000 (0) | — | 1.0000 |
| R2 train/greedy | 373 | 0.2869 (107) | **STAY, 107/107** | 1.0000 |
| R2 train/stoch | 247 | 0.4656 (115) | **STAY, 115/115** | 1.0000 |

Full-buffer: R2 max-prob > 0.99 on **0.5051** of decision entries, entropy
0.1841; A0 **0.0000**, entropy 0.3176. `frac_legal_edge = 1.0000` in every
bucket for both arms — **legality never constrains MIGRATE_EDGE**, so
saturation is a learned commitment, not a masking artifact.

### B.2 Dynamic plasticity (`_diag_rung2_75_plasticity.py`) — **this reverses its own first conclusion**

The probe applies a *known-favourable* synthetic advantage (+1 on
MIGRATE_EDGE / −1 on STAY at risk > 0.50) to a `deepcopy` and measures how far
the policy moves. Its first run reported near-zero movement and I read that as
"the actor is frozen". **That run was flawed:** `load_agent_and_cfg(..., "train")`
restores optimiser state, so it ran at the checkpoint's **fully annealed**
learning rate `9.333333333333316e-06 = 7e-4 × 1/75`, not at the rate training
used while the trust region was healthy. The flawed condition is retained as a
named cell rather than silently replaced.

M = 1, 10 updates, measured at the same buffer states:

| cell | arm | lr | π(EDGE\|hi) before → after | Δ | argmax==EDGE |
|---|---|---|---|---|---|
| `anneal_end` | A0 | 9.33e-06 | 0.0803 → 0.1000 | +0.0196 | 0.0000 → 0.0000 |
| `anneal_end` | R2 | 9.33e-06 | 0.0503 → 0.0584 | +0.0081 | 0.0223 → 0.0223 |
| **`lr_full`** | **A0** | **7e-04** | **0.0803 → 0.5326** | **+0.4523** | **0.0000 → 0.6840** |
| **`lr_full`** | **R2** | **7e-04** | **0.0503 → 0.2630** | **+0.2127** | **0.0223 → 0.4387** |
| `lr_full_fresh` | A0 | 7e-04 | 0.0803 → 0.5152 | +0.4348 | 0.0000 → 0.8349 |
| `lr_full_fresh` | R2 | 7e-04 | 0.0503 → **0.5046** | +0.4543 | 0.0223 → 0.6394 |

At the training learning rate both arms move π(EDGE) from ~5–8% to ~50% and
flip the argmax to MIGRATE_EDGE on the **majority** of high-risk entries within
one to two updates — **23.0× (A0) and 26.3× (R2)** the annealed cell. R2's
saturation only *halves* the response, and clearing Adam's stale second moment
removes even that gap (R2 reaches 0.5046, matching A0). R2's saturated fraction
*falls* 0.4721 → 0.4052 under `lr_full`.

**Conclusion: the actor is fully plastic. Saturation attenuates but does not
prevent movement. The stall is a signal problem, not a plasticity problem.**

This is the diagnostic *upper bound*, not a proposed fix: the synthetic
advantage presupposes that MIGRATE_EDGE is correct at high risk, which no
training run can know. Nothing was written to any checkpoint.

### B.3 Training-time collapse (`_diag_rung2_75_stepcollapse.py`) — the half no prior diagnostic covered

**Every** actor-stall measurement in this project (Rung 2.5's replica, and this
rung's plasticity, coherence and mbtail probes) was taken at the **final**
checkpoint. But the per-update CSVs say the trust region *started healthy*:

```
clip_frac   A0  0.1081 @ update 1  ->  0.0000 @ update 75
            R2  0.1625 @ update 1  ->  0.0000 @ update 75
```

So every prior probe measured the **bottom of a collapse** and could not
distinguish "the actor cannot move" from "the actor was annealed and sharpened
until it stopped moving". This probe measures the collapse itself, from CSVs
only — no model loaded, no rollout run.

Because Adam's step is ≈lr-sized regardless of gradient magnitude,
|ratio − 1| is first order in `lr_scale`, so `clip_frac / lr_scale` is **lr-free
by construction**. Using the ratio of quartile means the attribution factorises
*identically*: total = schedule × sharpening.

| arm | clip_frac Q1→Q4 | total | schedule | **sharpening** | dominant | product check |
|---|---|---|---|---|---|---|
| A0 | 0.06102 → 0.00058 | 105.4× | 6.7× | **15.8×** | sharpening | ✔ matches |
| R2 | 0.04256 → 0.000042 | 1010.8× | 6.7× | **152.0×** | sharpening | ✔ matches |
| A1 | — | 204.0× | 6.7× | **30.7×** | sharpening | ✔ matches |
| A2 | — | 10293.8× | 6.7× | **1547.9×** | sharpening | ✔ matches |
| A3 | — | 149.2× | 6.7× | **22.4×** | sharpening | ✔ matches |

Trust region engaged (`clip_frac > 0.01`): A0 38/75 updates (51%, last at
update 52); R2 28/75 (37%, last at 40); A3 34/75 (45%, last at 43). The lr-free
utilisation tracks the policy's own entropy (Spearman +0.7896 A0, +0.8681 R2,
+0.6938 A3), which is what `‖e_a − π‖ → 0` predicts.

**Ceiling on an lr-only intervention:** it recovers the 6.7× schedule factor and
nothing else, leaving 15.8×–1547.9× untouched. **An lr-only rung addresses the
smaller of the two causes and is therefore not the right next experiment.**

### B.4 Sharpening does *not* predict the outcome — so it cannot be a success criterion

Scoring all five existing arms on the item-A replacement metric and joining to
their training-time collapse statistics:

| arm | sharpening | final entropy | updates engaged | Δ RANDOM | Δ UNION |
|---|---|---|---|---|---|
| A0 | 15.8× | 0.3329 | 38/75 | −0.0264 | +0.0191 |
| A1 | 30.7× | 0.3561 | 29/75 | +0.1375 | +0.0628 |
| A2 | 1547.9× | 0.2965 | 30/75 | +0.0398 | −0.0248 |
| A3 | 22.4× | 0.3745 | 34/75 | +0.1322 | +0.0933 |
| R2 | 152.0× | 0.1380 | 28/75 | +0.2024 | +0.1579 |

Rank correlations across the five arms (n = 5, nothing significant, direction
only):

```
spearman(sharpening,      Δ RANDOM) = +0.40    (Δ UNION) = -0.10
spearman(final entropy,   Δ RANDOM) = -0.20    (Δ UNION) =  0.00
spearman(updates engaged, Δ RANDOM) = -0.90    (Δ UNION) = -0.50
```

The arm with the **least** sharpening (A0, 15.8×) has the **worst** risk
response; the arm with the **most** (A2, 1547.9×) is second-worst; the best
(R2) sits in the middle. Trust-region engagement is *negatively* ranked against
the outcome. **Therefore the trust-region / sharpening statistics are recorded
in item E as mechanism checks only, explicitly NOT as success criteria.** Had I
not run this join I would have pre-registered a metric that the existing
evidence does not support — this is exactly the class of error the brief warned
about.

---

## 3. Item B/C bridge — is the advantage signal *weak* or *incoherent*?

`_diag_rung2_75_coherence.py`. At a checkpoint's own parameters `new_lp ==
old_lp`, so the PPO ratio is exactly 1, `min(s1,s2) == A`, and the clipped
surrogate reduces **exactly** to the vanilla policy gradient, which is *linear*
in A. A matched-L2-mass comparison across advantage vectors is therefore
assumption-free. Four conditions, entropy term deliberately excluded (it is
advantage-independent and would inflate every cosine):

| ratio / cosine | A0 | R2 | reading |
|---|---|---|---|
| ‖g_real‖ / ‖g_synth‖ | 0.4077 | 0.2866 | 2.5–3.5× of the available magnitude is lost to cancellation |
| **‖g_real‖ / ‖g_shuffled‖** | **1.2989** (z +1.35, perm p **0.1050**) | **0.7987** (z −0.88, perm p **0.8050**) | **inside its own null — SNR ≈ 1** |
| cos(g_real, g_synth) | +0.49925 (perm p 0.1100) | +0.30127 (perm p 0.2450) | positive but inside the null (shuffled \|cos\| p95 = 0.5902 for A0, 0.5028 for R2) |
| **cos(g_real_hi, g_synth)** | **+0.94495** | **+0.87342** | the high-risk-restricted signal *does* point the right way |

The shuffled condition permutes real's own advantage values among decision
entries: it preserves the marginal distribution of A exactly and destroys only
the state–action correspondence. **The aggregated gradient carries no more
usable information than a permutation of its own advantages.** No existing arm
passes this test.

The signal is not absent — restricted to high-risk entries it is almost
perfectly aligned with the coherent reference (cos +0.945 / +0.873) — but it
contributes only ‖g‖ 1.50e-02 of A0's 7.06e-02 (~21% from 6.1% of entries) and
9.73e-03 of R2's 2.755e-02 (~35% from 5.4%). The remaining ~65–79% of gradient
mass is noise that cancels.

---

## 4. Item C — do raw per-state advantage offsets suppress PPO learning?

`_diag_rung2_75_offset.py`, on the **547 R2-native deviation rows only** (Rung
2.5's set, already on disk; no re-derivation, no new deviation set). High-risk
subset n = 298.

Under a greedy baseline `A_true(s, a) = Q(s,a) − Q(s,a_ref)`, so
`A_true(s, a_ref) ≡ 0` **exactly** and whatever GAE reports there is pure
estimator error. Define the per-state offset `c(s) := gae(s, a_ref)`; then
`gae(s,a) = c(s) + paired(s,a)`. **PPO's actor loss consumes the raw
left-hand side.**

| estimator | MAE | r | sign agreement |
|---|---|---|---|
| **team truth** — raw GAE | 5.8418 | −0.0795 | **0.4396** (below chance) |
| **team truth** — paired | 3.2132 | +0.3391 | **0.7114** |
| **own-agent truth** — raw GAE | 4.5878 | −0.0137 | **0.4597** (below chance) |
| **own-agent truth** — paired | 0.7257 | **+0.9255** | **0.9463** |

Offset: mean −0.8640, sd 5.2643; **SD(offset)/SD(paired) = 2.19×**.

Variance decomposition:

```
Var(raw) 21.522  =  Var(offset) 27.713 (128.8%)
                 +  Var(paired)  5.785 ( 26.9%)
                 +  2·Cov      -11.976
Var(raw) / Var(paired) = 3.72x
```

**Two facts that together settle how to treat this.** A per-state constant in
the advantage contributes **zero in expectation** — the textbook baseline
property — so the offset is not a *bias*; it is **pure variance**. And PPO's
advantage normalisation (`mappo.py:360–366`) subtracts **one batch mean** over
decision entries, so it removes `E[c]` and leaves the within-state dispersion
`SD(c)` entirely intact.

**Therefore: 3.72× is the exact data multiplier needed to give the raw
estimator paired-quality gradient variance by sample size alone.** Since
gradient SNR ∝ √N, a 4× batch buys ≈2× SNR per step and ≥ the 3.72×
variance-equivalent in total mass.

Per the brief, **no centred or matched advantage was implemented.** Note also
that the paired estimator is *not* available inside a normal PPO update: it
requires `gae(s, a_ref)` at the same state, i.e. counterfactual environment
replay (offline machinery) or an action-conditional critic (an architecture
change). Implementing it is a multi-variable change and is **not** what item E
proposes.

---

## 5. Item D — the minibatch tail, as a candidate only

`_diag_rung2_75_mbtail.py`. `mb_size = max(1, T//4)` (`mappo.py:369`) then
`for start in range(0, T, mb_size)` (`mappo.py:376`) ⇒ **5** chunks whenever
`T mod 4 ≠ 0`, with a tail of exactly `T mod 4` timesteps. Because
`pg = -(min(s1,s2)*d_mb).sum()/denom` with `denom = d_mb.sum().clamp(min=1.0)`
is a **mean over the chunk's own decision entries**, a 1-timestep tail produces
a *full-magnitude* gradient, is clipped to `max_grad_norm = 0.5`, and receives a
full Adam step. That is 4 of 20 optimiser steps per update — **20% of steps on
0.13% of the data**.

| arm | blocks with a tail | early-terminating episodes | tail/substantive ‖g‖ | tail/substantive ‖Δθ‖ |
|---|---|---|---|---|
| **A0** | **80%** | 35% | **7.9× – 9.2×** | 1.09× – 1.13× |
| R2 | 60% | 20% | 1.7× – 4.2× | 1.06× – 1.64× |

R2's substantive ‖g‖ = 6.0839e-02 reproduces the on-record 6.08e-02 exactly.

**Can it plausibly cause the observed R2 saturation? No — the evidence points
the wrong way.** A0 is the arm with **zero** saturation, yet it has *more*
exposure (80% vs 60% of blocks) and *larger* inflation (7.9–9.2× vs 1.7–4.2×).
A defect that is stronger in the unsaturated arm cannot explain the saturated
arm's asymmetry. And because Adam normalises by the second moment, the ‖g‖
inflation translates into only 1.06–1.64× extra *parameter movement*: the tail
adds ~9% extra Δθ per block for A0. It adds **variance, not bias**, shared by
both arms.

**Kept as a candidate, not fixed, and not proposed as the next rung.** It
remains a legitimate independent hygiene rung later, on its own.

---

## 6. Methodological corrections — including my own errors

Recorded in full, because several were load-bearing.

**Errors in prior artifacts, now corrected:**

1. **The Rung 2.5 census has no `has_task` filter** — it is mask-only, so it
   includes entries where the agent holds no task. Inert here (D2 → D3 moves the
   share by 0.0001) but it is a real definitional difference and was undocumented.
2. **The Rung 2.5 census draws `training_start_ticks`, not
   `rollout.episode_starts`** — a *sixth* definitional difference from the
   headline, previously uncounted.
3. **The on-record Rung 2.5 census z of −1.44 does not reproduce.** The same
   8-episode contrast gives **z = −0.64, p = 0.5206**, and on 32 matched starts
   **z = +0.10**. The "R2 is worse" contrast was never significant.
4. **`explained_var` in `*_updates.csv` is target-relative**, so cross-arm
   comparison of it is invalid. A0 uses a λ-return target and R2 a
   continuation-MC target; their explained-variance columns are not on the same
   scale.
5. **Every prior actor-stall diagnostic measured only the final checkpoint** —
   i.e. the bottom of a collapse that starts healthy (§B.3). This is the single
   most consequential methodological gap found this rung.

**My own errors this rung, self-caught:**

6. **The plasticity probe's first run silently used the annealed lr
   9.333333333333316e-06**, because `load_agent_and_cfg(..., "train")` restores
   optimiser state including the fully-annealed rate. I originally concluded
   "the actor is frozen". **The corrected run reverses that conclusion** (§B.2).
   The flawed condition is preserved as the named cell `anneal_end` rather than
   silently replaced. The same flaw touches the mbtail probe's Δθ figures, which
   are therefore annealed-lr magnitudes and must be read as ratios, not levels.
7. **My step-collapse factorisation was wrong and its own product check caught
   it.** I first defined the sharpening factor as the Q1/Q4 ratio of the
   per-update series `clip_frac/lr_scale`, but `mean(cf/ls) ≠ mean(cf)/mean(ls)`,
   so `schedule × sharpening ≠ total` (A0: 6.7 × 21.1 = 140.6 against a total of
   105.4; `product_matches_total: False`). Replaced with the exactly-factorising
   ratio of quartile means; the honest average of the lr-free series is retained
   alongside with a caveat. All five arms now report
   `product_matches_total: True` within 2%. The pre-fix numbers (A0 21.1, R2
   230.3, A1 58.4, A2 2849.3, A3 32.2) are **superseded**.
8. **My clustering premise was largely refuted by my own probe.** I expected
   episode clustering to explain the discrepancy; the design effect for the
   Rung 2.5 cell is **1.05** (essentially none) and for the headline cell 1.45.
   Clustering was not the story. The real weakness in the Rung 2.5 figure is
   **seed replication on a fixed 8-episode sample**, which cannot reduce
   between-episode variance.
9. **Item C's within-state action-ordering result (C3) is an identity, not
   evidence.** `paired(s,a) = gae(s,a) − c(s)` differs from `gae(s,a)` by a
   per-state constant, so it induces the *same* within-state ranking by
   construction. Any Kendall tau of 1.0 there is arithmetic, and it has been
   removed from the evidence base.
10. **`push_if_matched` needed restating.** It measures mean|paired| ×
    attenuation, so the 2.12× figure means *"the spurious push applied to the
    taken action is 2.12× the size of the genuine per-state signal"* — not
    "2.12× more gradient".
11. **My earlier "tiny-step regime" framing was wrong.** `clip_frac` starts
    healthy and *collapses*; it is not small throughout.
12. Two minor code errors, fixed at source: `compute_gae` returns a **tuple**
    `(adv, ret)` (fixed with `adv_raw, _ =`), and a print-with-ternary crash
    risk in the coherence probe (fixed with pre-computed strings).

**The finding that costs us the most:** §A.4. The high-risk EDGE share, this
project's headline metric since Sprint 6, is not identified. Both the Rung 2
"win" and the Rung 2.5 "loss" must be withdrawn as policy-quality claims. The
replacement metric in §A.6 is reported *with* its own caveats rather than
presented as clean.

---

## 7. Hypotheses recorded, not implemented

Per the stop condition, every candidate intervention found during the
diagnostic is recorded here. **None was implemented or tested.**

| # | hypothesis | status |
|---|---|---|
| H1 | Advantage variance is the binding constraint; more samples per update raises gradient SNR | **proposed as the next rung (item E)** |
| H2 | Remove the per-state offset directly (centred / matched advantage, COMA-style counterfactual baseline) | recorded. Requires counterfactual replay or an action-conditional critic ⇒ multiple variables. Deferred; H1 is the same variance reduction as a single config change. |
| H3 | Raise or floor the actor learning rate / change the anneal schedule | recorded and **argued against**: ceiling is the 6.7× schedule factor, vs 15.8–1547.9× sharpening (§B.3) |
| H4 | Raise `entropy_coef` to keep the policy from committing to noise | recorded and **closed**: A3 *is* an `entropy_coef = 0.05` arm (2.5× A0), config otherwise identical, and it still sharpened 22.4× with Δ RANDOM +0.1322 < R2's +0.2024 |
| H5 | Fix the minibatch tail (`range(0, T, mb_size)` ⇒ 5 chunks) | recorded as a **candidate only** (§5); points the wrong way for the A0-vs-R2 asymmetry; legitimate later hygiene rung |
| H6 | Select or reweight training entries by risk | recorded and **rejected on grounds of the brief**: that is a hard-coded `risk > threshold` rule, explicitly forbidden |
| H7 | Continuation-MC critic target variants | **forbidden by the brief** ("do not implement continuation-MC"); also Rung 2.5 already closed ~87% of the target gap |

### 7.1 Hypotheses the brief declared closed — status after this rung

Reported for completeness; none was re-opened.

| closed hypothesis | this rung's bearing on it |
|---|---|
| critic target | left closed. Rung 2.5's ~87% gap closure stands; A0 λ boundary-tail error 58.4% of target SD vs R2 MC 7.8%. |
| value clipping | left closed (exonerated in Rung 2.5). Untouched. |
| minibatch tail | still **not fixed**, and deliberately kept isolated (§5). New evidence points the wrong way for the A0-vs-R2 asymmetry. |
| high-risk sparsity | **reconfirmed closed.** `SPRINT_7_RUNG2_5_actor_stall.json`: R2's replica update has `minibatches 16, zero high-risk: 0`, mean 2.50 high-risk EDGE entries per minibatch (min 0, max 6). The old "30% of minibatches lack high-risk entries" premise does not reproduce. |
| vanishing gradients | **reconfirmed false.** R2 actor grad norm 6.083908e-02 (pg-only 6.112536e-02), actor Δθ L2 5.429798e-03 — reproduced exactly by this rung's mbtail probe (substantive ‖g‖ 6.0839e-02). The problem is gradient *direction*, not magnitude (§3). |

---

## 8. Item E — PRE-REGISTERED recommendation for exactly ONE next rung

> **Nothing below has been run. This is submitted for approval.**

### E.1 Hypothesis (falsifiable, single mechanism)

> **H1.** The MAPPO actor's learning stall is caused by **advantage-estimator
> variance**, not by policy incapacity, sample sparsity, or critic-target
> misspecification. The dominant variance term is a per-state offset `c(s)` that
> contributes zero in expectation and that PPO's single batch-mean
> normalisation cannot remove (`Var(raw)/Var(paired) = 3.72×`). At the current
> batch size the aggregated policy gradient is statistically indistinguishable
> from a permutation of its own advantages (real/shuffled 1.30 / 0.80, both
> p > 0.10). **Increasing the number of samples the estimator is averaged over
> — with nothing else changed — should make the aggregated gradient carry
> genuine state–action information and produce a measurable risk response.**

### E.2 The single variable

**`rollout_episodes`: 8 → 32** (4× decision entries per PPO update ⇒ ≈2× gradient
SNR per step, ≥ the 3.72× variance-equivalent in total advantage mass).

**One derived control, not a second manipulation:** `episodes` 600 → 2400,
chosen *solely* to hold `n_updates` at exactly 75.
`train.py:168` computes `n_updates = episodes // rollout_episodes`, so
2400 // 32 = 75 = 600 // 8. The lr-anneal schedule (`train.py:211`,
`frac = 1 − (update_id−1)/n_updates`) is then **bit-identical** to A0/R2:
1.0 → 0.01333. Stated plainly: total environment samples rise 4×. That is
inherent to the manipulation — batch size cannot be raised at a fixed update
count without more data. The alternative (hold samples fixed, accept 18
updates) would confound with a 4× cut in optimiser steps, which is worse given
that the trust region collapses *over updates* (§B.3).

**Code location:** no code change. `python-ai/marl/train.py:46` already exposes
`--rollout-episodes` and `train.py:45` `--episodes`; `train.py:139` sizes the
buffer as `cap = rollout_episodes × episode_steps` (3200 → 12800) and
`train.py:168`/`:211` derive updates and the lr schedule from the same two
values. **Nothing in `mappo.py` is touched.**

**Control arm:** **R2** (`mappo_R2_mc_target.pth`), already on disk. No second
training run. R2 is the correct incumbent: Rung 2 accepted the continuation-MC
target as standing configuration, and R2 is the best of all five existing arms
on the pre-registered primary metric.

### E.3 What remains frozen

Everything else, explicitly: `critic_target=mc`; `gamma=0.999`;
`gae_lambda=0.995`; `clip_eps=0.2`; `value_clip_eps=0.2`; `entropy_coef=0.02`;
`value_coef=0.5`; `max_grad_norm=0.5`; `ppo_epochs=4`; `minibatches=4`
(**the tail is deliberately left unfixed** — item D stays isolated);
`normalise_advantages=True`; `anneal_lr=True`; `lr_actor=7e-4`;
`lr_critic=1e-3`; `episode_steps=400`; reward weights; environment; topology;
risk predictor (`risk_source=oof`, uncalibrated); actor/critic architecture;
`seed=20260818`; `device=cpu`.

### E.4 Exact training command

```bash
cd python-ai && python -m marl.train --critic-target mc --rollout-episodes 32 --episodes 2400 --seed 20260818 --device cpu --tag R3_batch32 2>&1 | tee run_R3_batch32_train.log
```

This is R2's exact command plus `--rollout-episodes 32 --episodes 2400`.
Expected cost ≈4× A0's 1450 s ⇒ **≈1.6 h** on CPU. CPU is mandatory
(`--device cuda` breaks seed reproducibility and is 2.8× slower on these tiny
actors).

### E.5 Primary success metrics (exactly 5) and GO/NO-GO thresholds

Thresholds are calibrated against the **observed spread across the five arms
that already exist**, scored on the same fixed state sets
(`SPRINT_7_RUNG2_75_matched_states_main.json` →
`calibration_of_primary_metric`), not invented:

```
RANDOM: A0 -0.0264  A2 +0.0398  A3 +0.1322  A1 +0.1375  R2 +0.2024
        range 0.2289  sd 0.0901  best R2  median within-arm CI95 half-width 0.0419
UNION : A2 -0.0248  A0 +0.0191  A1 +0.0628  A3 +0.0933  R2 +0.1579
        range 0.1826  sd 0.0699  best R2  median within-arm CI95 half-width 0.0109
```

| # | metric | R2 (incumbent) | **GO** | NO-GO |
|---|---|---|---|---|
| **P1** | Δ = π(EDGE\|risk≥0.6) − π(EDGE\|risk<0.2) on the **RANDOM** set | +0.2024, CI95 [+0.1534, **+0.2554**] | **Δ ≥ +0.26** *and* own CI95 lower bound > 0 *and* paired cluster-bootstrap (Δ_R3 − Δ_R2) CI95 excludes 0 | Δ < +0.26, or CI includes 0 |
| **P2** | same Δ on the **UNION** set | +0.1579, CI95 [+0.1445, **+0.1710**] | **Δ ≥ +0.18** *and* CI95 lower bound > 0 | Δ < +0.18 |
| **P3** | Spearman(risk, π(EDGE)) over decision entries, RANDOM set | +0.0777 (best existing: A1 +0.0831) | **> +0.0831** (beats *every* existing arm) | ≤ +0.0831 |
| **P4** | coherence ‖g_real‖/‖g_shuffled‖ with its permutation p (200 perms), on R3's own buffer | 0.7987, p = 0.8050 (A0: 1.2989, p = 0.1050) | **ratio > 1.0 with one-sided permutation p < 0.05** — i.e. the aggregated gradient beats a permutation of its own advantages. **No existing arm passes.** | ratio ≤ 1.0, or p ≥ 0.05 |
| **P5** | cos(g_real, g_synth) against the shuffled \|cos\| null on the same buffer | +0.30127, p = 0.2450 (R2's shuffled \|cos\| p95 = 0.5028) | **perm p < 0.05** | p ≥ 0.05 |

**Overall GO** = P1 **and** P2 **and** (P3 **or** P4). P4/P5 are the direct
mechanism test of H1; P1/P2 are the project's primary question. Requiring both
neutral state sets guards against RANDOM's off-manifold weakness and UNION's
selection weakness independently.

**Explicitly NOT success criteria:**

* **Reward** (per the brief, in force since Rung 0).
* **`clip_frac` / sharpening factor / trust-region engagement / final entropy.**
  Recorded as **secondary mechanism diagnostics only**, because across the five
  existing arms they do not predict P1 (§B.4: Spearman +0.40 / −0.10 for
  sharpening, −0.90 / −0.50 for engagement, n = 5). A rung that "fixed
  sharpening" while P1 stood still would be a null result, and vice versa.
* **High-risk EDGE share on R3's own trajectory** — not identified (§A.4).
  It will be reported for continuity and labelled unidentified.

### E.6 What would falsify H1

* **Direct falsification:** Δ on RANDOM lands **inside** R2's existing interval
  [+0.1534, +0.2554] **and** P4's ratio stays ≤ 1.0 with p ≥ 0.05. Then 4× data
  bought only smoothness, the advantage signal's defect is **systematic**
  (estimator misspecification, not sampling noise), and the correct next move is
  H2 — attack the offset itself with an action-conditional critic or a
  counterfactual baseline — which is a multi-variable design needing its own
  brief.
* **Partial / split result:** P4 improves (the gradient becomes coherent) but
  P1/P2 do not. Then the advantage estimator was the variance bottleneck but the
  *reward* does not in fact prefer MIGRATE_EDGE at high risk. That is a
  reward-specification question and must not be relabelled as a learning fix.
* **Mechanism contradiction:** P1 clears +0.26 while P4 stays inside its null.
  Then the improvement did not come through the hypothesised channel and the
  result must be reported as unexplained, not as confirmation.

### E.7 Pre-registered evaluation protocol (traps to avoid)

1. **The state sets must not move.** RANDOM depends only on A0's config plus
   `--start-seed 20260825` and `--random-seed 31337`; UNION is A0's ∪ R2's
   recorded greedy states. **R3 must be added to `EXTRA` (scored only), never to
   `MODELS` (which generates state sources).** Adding R3 as a generator would
   silently redefine UNION and destroy comparability with the numbers in this
   report.
2. Re-run, unchanged: `_diag_rung2_75_matched_states.py --extra-arms A1,A2,A3,R3`
   (P1–P3), `_diag_rung2_75_coherence.py` on the R3 checkpoint (P4, P5),
   `_diag_rung2_75_stepcollapse.py --also R3=mappo_R3_batch32_updates.csv`
   (secondary).
3. **One seed only.** A GO result licenses a *seed-replication rung*, not a
   paper claim. Per the brief: no improvement claimed from a single noisy run
   without qualification.
4. Take fresh md5 manifests before and after. Preserve all five existing arms
   untouched; `--tag R3_batch32` writes only new filenames.

---

## 9. Artifact index (all new files, all prefixed for this rung)

**Probes** (`python-ai/marl/`): `_diag_rung2_75_edgeshare.py`,
`_diag_rung2_75_edgeshare_power.py`, `_diag_rung2_75_edgeshare_cluster.py`,
`_diag_rung2_75_matched_states.py`, `_diag_rung2_75_plasticity.py`,
`_diag_rung2_75_stepcollapse.py`, `_diag_rung2_75_coherence.py`,
`_diag_rung2_75_offset.py`, `_diag_rung2_75_mbtail.py`.

**Results** (`python-ai/saved_models/marl/`):
`SPRINT_7_RUNG2_75_edgeshare_main.json`,
`SPRINT_7_RUNG2_75_edgeshare_power_main.json`,
`SPRINT_7_RUNG2_75_edgeshare_cluster_{main,production_main,rung2_5_main}.json`,
`SPRINT_7_RUNG2_75_matched_states_main.json`,
`SPRINT_7_RUNG2_75_plasticity_main.json`,
`SPRINT_7_RUNG2_75_stepcollapse_{main,allarms}.json`,
`SPRINT_7_RUNG2_75_coherence_main.json`,
`SPRINT_7_RUNG2_75_offset_R2.json`,
`SPRINT_7_RUNG2_75_mbtail_main.json`,
`_rung2_75_integrity/SPRINT_7_RUNG2_75_{artifacts,code}_before.md5`.

**Logs** (`python-ai/`): `SPRINT_7_RUNG2_75_*.log` (one per probe).

**Not modified:** every Sprint 6 / 6.5 / Rung 0–2.5 artifact, and all 28
pre-existing code files.
