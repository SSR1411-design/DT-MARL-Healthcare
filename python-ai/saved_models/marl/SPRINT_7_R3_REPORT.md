# SPRINT 7 R3 — BATCH 32 REPORT

**Pre-registered outcome: NO-GO.** P1, P2, P3 and P4 all fail. P2 fails
*significantly in the wrong direction* — R3's confidence interval lies strictly
below R2's. H1 is not supported.

The triggered pre-registration rule is: *"If P1 AND P2 fail → H1 is not
supported, regardless of reward. Do not run another batch-size variant.
Recommend the next hypothesis based on the existing evidence."*

The mechanism H1 named was nevertheless **delivered**: the 4× batch measurably
raised actor-gradient SNR (§9, §10.6). That combination — manipulation worked,
predicted consequence did not follow — is a stronger falsification of H1 than a
failed manipulation would have been.

---

## 1. Experiment actually run

Command, run verbatim as approved, no alteration:

```bash
cd python-ai && python -m marl.train --critic-target mc --rollout-episodes 32 --episodes 2400 --seed 20260818 --device cpu --tag R3_batch32 2>&1 | tee run_R3_batch32_train.log
```

| | |
|---|---|
| wall clock | 5806 s (1.61 h); R2 was 1585.6 s |
| episodes | 2400 (1456 truncation / 944 true terminal) |
| PPO updates | **75** (R2: 75) — confirmed, §3 |
| device | cpu |
| seed | 20260818 |
| final checkpoint | `R3_batch32.pth` |

Reward trajectory: first-20 episode mean −17.95 → last-20 −2.14; best update
mean −2.40; final greedy 5-episode TRAIN-window check reward +26.02, success
0.770, reloc 29.0. **Reward is not a pre-registered criterion and is not used
below except as a reported diagnostic.**

### Naming consequence, flagged not fixed

`train.py` composes output filenames as `{tag}_*`. R2's tag was literally
`mappo_R2_mc_target`, so its artifacts carry a `mappo_` prefix; `--tag
R3_batch32` therefore produced `R3_batch32.*` with no prefix. The approved
command was **not** edited to match the older convention. The actual filename
was registered in the diagnostic harness instead.

### Five before-starting confirmations

1. **Integrity snapshot** — created and self-verified before launch (§2).
2. **No artifact collision** — no `R3_batch32*` name existed; 0 pre-existing
   artifacts were overwritten (verified after, §2).
3. **Resolved configuration confirmed** — `critic_target=mc`,
   `rollout_episodes=32`, `episodes=2400`, `seed=20260818`, `device=cpu`,
   `ppo_epochs=4`, `minibatches=4`, `gamma=0.999`, `gae_lambda=0.995`,
   `entropy_coef=0.02`, `value_clip_eps` unchanged. Confirmed twice: from the
   dataclasses before launch, and from the saved `R3_batch32_config.json` after.
4. **Minibatch tail confirmed UNFIXED** — `mappo.py` unchanged
   (`8614d016c5c60284898e932970d4ea76`). `mb_size = max(1, T // n_mb)` followed
   by `for start in range(0, T, mb_size)` still yields 5 chunks whenever
   `T mod 4 ≠ 0`, i.e. 20 optimiser steps per update of which 4 are degenerate
   tails.
5. **No other code or config change** — declared before launch: exactly two
   diagnostic-harness files, no production module. Verified after (§2).

---

## 2. Integrity / preservation

Manifests in `python-ai/saved_models/marl/_R3_integrity/`.

| manifest | entries | result |
|---|---|---|
| `SPRINT_7_R3_artifacts_before.md5` | 111 | **0 mismatches** after everything |
| `SPRINT_7_R3_code_before.md5` | 37 | 2 mismatches, both declared (below) |
| `SPRINT_7_R3_inputs_before.md5` | 3 | **identical** before vs after |
| `SPRINT_7_R3_raw_outputs.md5` | 6 | **all OK** — R3's raw outputs unchanged since capture |

After-manifests (`*_artifacts_after.md5` 121, `*_code_after.md5` 38,
`*_inputs_after.md5` 3) are written so the next rung can verify against this
rung's end state.

**No pre-existing artifact was modified or overwritten.** Diff of before vs
after over the 111 shared entries: zero changed hashes.

### Production source: all seven modules unchanged

```
mappo.py         8614d016c5c60284898e932970d4ea76
train.py         8ef424240f82d0bff6395e00a55c1129
env.py           4e5652820866b892edec7f12b1ae0787
rollout.py       3a829646a60214125109d2781606284d
config.py        3adca5da23bd65e7927b46dd5a69799c
evaluate.py      79c8e98466b3a2f785ca8e8c2a933672
risk_provider.py 02219832596a38e9800ba4f89659a555
```

### The two changed files, both diagnostic harness, both declared pre-launch

- `_diag_rung2_75_matched_states.py` — R3 added to `EXTRA` (scored) and
  deliberately **not** to `MODELS` (state-source-generating). Promoting it to
  `MODELS` would have silently redefined RANDOM and UNION, so the pre-registered
  thresholds would no longer be measured on the sets they were calibrated on.
- `_diag_rung2_75_coherence.py` — additive `--also name=file` flag, defaulting
  to empty so the Rung 2.75 `main` artifact stays reproducible; plus a
  `RUNG2_5_DTHETA.get(tag)` fix (see §13).

### New code file

- `_diag_R3_action_channels.py` (`a25034e565c139b832e399cf6fed1f7c`) — a new,
  purely exploratory probe (§13). Written as a new file rather than an edit to
  `_diag_rung2_75_matched_states.py` precisely so the script that produced the
  pre-registered P1/P2/P3 numbers still exists in the form that produced them.

### Every newly created artifact

```
d9f8a3dfbe1caf3b57402b9ef05313e3  saved_models/marl/R3_batch32.pth
c0e35a21191fe35b7f5064224d053c07  saved_models/marl/R3_batch32_best.pth
9f63a8ecde43ccaad2a75bc7e1a187f8  saved_models/marl/R3_batch32_config.json
d04b5e93ed1d9a492841744312ffedcc  saved_models/marl/R3_batch32_history.csv
7fcf40d42c3e699b0f76866bf7670b80  saved_models/marl/R3_batch32_updates.csv
023a99e05903f6f515cd99a3894c5cb5  saved_models/marl/SPRINT_7_RUNG2_75_matched_states_R3.json
0a807dc6565cb4b4c22a8c19de90ff45  saved_models/marl/SPRINT_7_RUNG2_75_coherence_R3.json
1f8e2fa25cbd0c5acbb517ea8ac7b2ea  saved_models/marl/SPRINT_7_RUNG2_75_coherence_R3_bs32probe.json
7ff7b693b59e7c93c7f02c35336663e3  saved_models/marl/SPRINT_7_RUNG2_75_stepcollapse_R3.json
53955350c3d84c10c234d33464ea8d5c  saved_models/marl/SPRINT_7_R3_action_channels_R3.json
dc84cccbbbbb93cfcb9ab88a84b429bc  run_R3_batch32_train.log
d27e894ac7c42d38ca30a0e3b94fb66e  SPRINT_7_R3_matched_states.log
80d5107badaad67436ba61ba78cc8fed  SPRINT_7_R3_coherence.log
d7deae1ae4ac669a06ba5b6a0d6c11df  SPRINT_7_R3_coherence_bs32.log
8743c271e4d3d52b374827bb23a0ac32  SPRINT_7_R3_stepcollapse.log
69d12b1dab39e715d5e2b6e88a43d575  SPRINT_7_R3_action_channels.log
a25034e565c139b832e399cf6fed1f7c  marl/_diag_R3_action_channels.py
```

### Environment provenance

Trace inputs (`failure_history.csv`, `failure_log.csv`,
`failure_predictor_oof.npz`) dated 2026-08-18, R2's checkpoint 2026-08-25.
`train_start_window = [9, 491]` identical for R2 and R3. Inputs manifest
verified bit-identical after the run, so R2 and R3 saw the same environment.
Note: no prior rung pinned the trace CSVs — `SPRINT_7_R3_inputs_before.md5` is
an **addition** by this rung, recorded as such.

### Harness validation — adding R3 perturbed nothing

- Diagonal parity reproduced exactly: A0 60/2174 `match: true`, R2 122/1418
  `match: true`.
- Coherence reproduced A0 (real/synth 0.4077, real/shuffled 1.2989, cos
  +0.49925, p 0.1100) and R2 (0.2866, 0.7987, +0.30127, 0.2450) to five
  decimals.
- The exploratory action-channel probe independently reproduces P1 and P2 to
  four decimals (§10.7), from separately-constructed state sets.

---

## 3. R2 vs R3 configuration

91 config fields compared. **Exactly 3 differ:**

| field | R2 | R3 |
|---|---|---|
| `train.episodes` | 600 | 2400 |
| `train.rollout_episodes` | 8 | 32 |
| `train.tag` | `mappo_R2_mc_target` | `R3_batch32` (label only) |

The only substantive experimental variable is rollout batch size. Everything
else — critic target, reward, environment, γ, λ, entropy coefficient, PPO
epochs, minibatch count, clip epsilons, seed, device — is identical.

**PPO update count: exactly 75 in both.** `n_updates = max(1, episodes //
rollout_episodes)`: 600//8 = 75, 2400//32 = 75. Both `updates.csv` files have
75 rows; R2's final row is at episode 600, R3's at episode 2400.

**LR schedule: bit-identical.** `max |lr_scale_R2 − lr_scale_R3| = 0.00e+00`
across all 75 updates; both run 1.0000 → 0.0133. The schedule depends only on
`update_id / n_updates`, and `n_updates` matched, so the control held exactly.

**Effective rollout sample size — 2.71×, not 4×:**

| | rollout cap | × 10 agents | mean `decision_frac` | decision entries/update |
|---|---|---|---|---|
| R2 | 3200 steps | 32,000 | 0.1392 | ~4,454 |
| R3 | 12,800 steps | 128,000 | 0.0945 | ~12,092 |

The raw step budget rose exactly 4×, but R3's decision fraction fell from 0.1392
to 0.0945, so the quantity that actually enters the actor loss — decision
entries — rose only **2.71×**. Expected SNR gain is therefore √2.71 ≈ 1.65×, not
2×. This matters for reading §9: the delivered gain (1.64×) matches the 2.71×
prediction almost exactly, not the 4× one.

---

## 4. Training trajectory

Per-update quartile means (75 updates each), R2 above R3:

| metric | Q1 | Q2 | Q3 | Q4 | first → last |
|---|---|---|---|---|---|
| `mean_reward` R2 | −16.760 | −7.704 | +0.083 | +1.107 | −12.07 → +12.65 |
| `mean_reward` R3 | −16.113 | −12.163 | −10.591 | −8.511 | −17.39 → −4.48 |
| `entropy` R2 | 0.5751 | 0.2784 | 0.1813 | 0.1611 | 0.8599 → 0.1380 |
| `entropy` R3 | 0.5040 | 0.3714 | 0.3526 | 0.2923 | 0.7877 → 0.2659 |
| `clip_frac` R2 | 0.0426 | 0.0189 | 0.0027 | 0.0000 | 0.1625 → 0.0000 |
| `clip_frac` R3 | 0.0355 | 0.0136 | 0.0053 | 0.0008 | 0.0766 → 0.0000 |
| `approx_kl` R2 | 0.0040 | 0.0021 | −0.0003 | −0.0003 | 0.0160 → 0.0000 |
| `approx_kl` R3 | 0.0037 | 0.0014 | 0.0007 | 0.0003 | 0.0010 → −0.0001 |
| `adv_std` R2 | 3.6595 | 3.3603 | 3.0839 | 2.8615 | 3.4471 → 2.4680 |
| `adv_std` R3 | 3.9825 | 3.6210 | 3.2663 | 3.0000 | 3.5308 → 2.8730 |
| `critic_loss` R2 | 16.31 | 12.83 | 10.67 | 8.73 | 14.67 → 5.77 |
| `critic_loss` R3 | 19.47 | 16.26 | 12.47 | 10.00 | 17.29 → 9.42 |
| `explained_var` R2 | 0.300 | 0.578 | 0.675 | 0.702 | 0.027 → 0.741 |
| `explained_var` R3 | 0.297 | 0.571 | 0.634 | 0.656 | 0.017 → 0.682 |

Trajectory comparison, not just the endpoint:

- **R3's trajectory is uniformly flatter.** Entropy ends at 0.2923 vs R2's
  0.1611; `clip_frac` stays nonzero into Q4 (0.0008) where R2 is at exactly
  0.0000 by Q4; `approx_kl` stays positive throughout where R2 goes slightly
  negative from Q3 on.
- **Trust-region engagement is comparable, marginally later in R3:** R2 engaged
  on 28/75 updates (37%, last at update 40), R3 on 25/75 (33%, last at update
  47). So R3 was not simply frozen — it kept moving slightly longer.
- **`adv_std` (pre-normalisation) is HIGHER in R3 at every quartile.** This is
  the first direct contradiction of H1's premise: the per-entry advantage
  standard deviation did not shrink with the larger batch, it grew slightly
  (mean 3.4606 sd 0.3951 vs R2's 3.2357 sd 0.3533). H1 was about the *gradient*
  estimator's variance, not the per-entry advantage spread, so this is not
  itself a refutation — but it removes one route by which the batch could have
  helped.
- **`mean_reward` is much worse in R3** (Q4 −8.51 vs +1.11). Not a criterion.
  Recorded because ignoring it would be selective reporting, and because it
  independently rules out "R3 traded risk response for reward."
- `explained_var` is target-relative, so cross-arm comparison is not
  strictly valid; both arms use the `mc` target here, which makes the
  comparison closer to like-for-like than λ-vs-mc would be, but I do not draw
  any conclusion from it.

---

## 5. P1 — RANDOM risk response

**Definition** (unchanged from Rung 2.75): Δ = P(EDGE | risk ≥ 0.6) − P(EDGE |
risk < 0.2) on the RANDOM state set (uniform-legal actions, policy-independent;
4,975 decision entries, 216 at risk ≥ 0.6, 4,757 at risk < 0.2). Greedy
evaluation window, production decision rule `mask.sum(-1) > 1`, cluster
bootstrap over 32 episodes.

| arm | Δ | CI95 | z |
|---|---|---|---|
| A0 | −0.0264 | [−0.0543, +0.0008] | −1.86 |
| **R2** | **+0.2024** | [+0.1534, +0.2554] | +7.72 |
| A1 | +0.1375 | [+0.1052, +0.1706] | +8.15 |
| A2 | +0.0398 | [−0.0011, +0.0828] | +1.87 |
| A3 | +0.1322 | [+0.0918, +0.1764] | +6.19 |
| **R3** | **+0.1575** | [+0.1184, +0.1931] | +8.14 |

**GO threshold ≥ +0.26. R3 = +0.1575. FAIL.**

The CI **excludes zero** ([+0.1184, +0.1931], z = +8.14), so R3 does have a
statistically solid positive risk response — it is simply well below both the
threshold and R2's +0.2024. R3 ranks third of six arms.

---

## 6. P2 — UNION risk response

Same Δ on the UNION set (A0's states pooled with R2's, episode ids disjoint;
74,237 decision entries, 3,592 at risk ≥ 0.6, 70,621 at risk < 0.2).

| arm | Δ | CI95 | z |
|---|---|---|---|
| A0 | +0.0191 | [−0.0020, +0.0434] | +1.62 |
| **R2** | **+0.1579** | [+0.1445, +0.1710] | +23.25 |
| A1 | +0.0628 | [+0.0536, +0.0710] | +14.14 |
| A2 | −0.0248 | [−0.0362, −0.0143] | −4.42 |
| A3 | +0.0933 | [+0.0861, +0.1008] | +24.90 |
| **R3** | **+0.1242** | [+0.1102, +0.1398] | +16.35 |

**GO threshold ≥ +0.18. R3 = +0.1242. FAIL.**

The CI excludes zero (z = +16.35). But the important fact is stronger than a
threshold miss: **R3's CI upper bound (+0.1398) lies strictly below R2's CI
lower bound (+0.1445).** On the larger and fully on-manifold state set, R3 is
*significantly worse than its own control*. This is the single most decisive
number in the experiment — it is not a null result, it is a result in the
opposite direction from the hypothesis.

---

## 7. P3 — RANDOM Spearman

Spearman(risk, P(EDGE)) over all RANDOM decision entries.

| arm | ρ |
|---|---|
| A0 | −0.0286 |
| R2 | +0.0777 |
| **A1** | **+0.0831** ← the threshold was set at the best existing arm |
| A2 | +0.0100 |
| A3 | +0.0349 |
| **R3** | **+0.0656** |

**GO threshold > +0.0831. R3 = +0.0656. FAIL.** R3 also falls below R2's
+0.0777, i.e. below its own control on the rank-correlation measure as well as
the difference-of-means measure.

---

## 8. P4 — gradient coherence

Established Rung 2.75 methodology, at the probe's own default 8-episode
measurement buffer: full-batch actor policy gradient at the checkpoint's own
parameters (where the PPO ratio is exactly 1, so the loss reduces exactly to the
vanilla PG and is linear in A); all advantage vectors rescaled to matched L2
mass over decision entries; null = permutation of the arm's own advantages among
decision entries; 200 permutation draws.

| arm | ‖g_real‖ | ‖g_shuf‖ | real/shuffled | z | perm p |
|---|---|---|---|---|---|
| A0 | 7.063e-02 | 5.437e-02 | 1.2989 | +1.35 | 0.1050 |
| R2 | 2.755e-02 | 3.450e-02 | 0.7987 | −0.88 | 0.8050 |
| **R3** | 5.557e-02 | 4.486e-02 | **1.2387** | +1.00 | **0.1450** |

**GO threshold: ratio > 1.0 AND one-sided permutation p < 0.05.
R3 = 1.2387 at p = 0.1450. FAIL on the p condition.**

The ratio condition passes (1.2387 > 1.0) and R3 beats R2's 0.7987, but the
permutation test does not reach significance at the pre-registered 8-episode
measurement buffer. Per the pre-registration, both conditions were required, so
P4 fails.

This is the metric where the pre-registered measurement turns out to be
underpowered — see §9 and §13, and the 32-episode re-measurement in §10.6 which
reaches p = 0.0000. **That re-measurement is not substituted for P4.** P4 is
scored as specified, on the specified buffer, and it fails.

---

## 9. P5 — gradient alignment

cos(g_real, g_synth), where `synth` is the perfectly-coherent reference
direction (+1 on MIGRATE_EDGE, −1 on STAY, at risk > 0.50, zero elsewhere),
two-sided permutation p from the same 200 draws.

| arm | cos(g_real, g_synth) | perm p | cos(g_real_hi, g_synth) | null \|cos\| p95 |
|---|---|---|---|---|
| A0 | +0.49925 | 0.1100 | +0.94495 | 0.59022 |
| R2 | +0.30127 | 0.2450 | +0.87342 | 0.50276 |
| **R3** | **−0.65706** | **0.0100** | −0.24495 | 0.49876 |

**The p-threshold is met (0.0100 < 0.05). The sign is negative.**

**This is a defect in my pre-registration, and I am reporting it as a
directional failure rather than a pass.** I specified P5 as "cos(g_real,
g_synth) — GO threshold: permutation p < 0.05" with **no sign condition**. Read
literally, R3 passes P5. Read as intended — "does the actor gradient point the
way that would raise MIGRATE_EDGE at high risk?" — R3 is the only arm whose
gradient points *significantly the wrong way*. A metric that scores "pushes
strongly toward the opposite of the hypothesis" as a success is measuring the
wrong thing, and I will not claim the pass.

P5 was not part of the GO formula (`P1 ∧ P2 ∧ (P3 ∨ P4)`), so this does not
change the decision either way.

**Additional caveat that weakens P5 in BOTH directions.** Re-measured at a
32-episode buffer, R3's cosine moves from −0.65706 (p = 0.0100) to −0.24368
(p = 0.4250), and R2's flips sign from +0.30127 to −0.23720. The 8-episode
cosine estimate is not stable under a 4× measurement buffer. So the significant
negative cosine should not be over-read as evidence of active
counter-productive learning either — it is a noisy estimate that happened to
land far from zero. See §13.

---

## 10. Secondary mechanism diagnostics

Everything in this section is **mechanism evidence only** and is explicitly not
substituted for any GO criterion.

### 10.1 Saturation and entropy — the batch DID prevent softmax collapse

| set | arm | `frac_maxp_gt_099` | `maxp_mean` | `entropy_mean` |
|---|---|---|---|---|
| RANDOM | A0 | 0.0000 | 0.8573 | 0.3664 |
| RANDOM | R2 | **0.0185** | 0.6485 | 0.6379 |
| RANDOM | **R3** | **0.0000** | 0.6329 | 0.6603 |
| UNION | A0 | 0.0000 | 0.7866 | 0.4402 |
| UNION | R2 | **0.0690** | 0.7344 | 0.5062 |
| UNION | **R3** | **0.0000** | 0.7114 | 0.5254 |

R3 has **zero** saturated decision entries on both neutral sets, where R2 had
1.85% and 6.90%, and R3's action entropy is higher on both. The intervention
did what an anti-collapse intervention should do.

### 10.2 Step-collapse attribution

`SPRINT_7_RUNG2_75_stepcollapse_R3.json`, all products matching total, all
dominant factor = sharpening:

```
arm     total  schedule   sharpen      dominant  lr-only ceiling
 A0    105.4x      6.7x     15.8x    sharpening            15.8x
 R2   1010.8x      6.7x    152.0x    sharpening           152.0x
 R3     46.5x      6.7x      7.0x    sharpening             7.0x
```

R3's sharpening collapse is **7.0×** against R2's **152.0×** — by far the
mildest of the three arms, and consistent with §10.1's zero saturation. The LR
schedule contributes an identical 6.7× in all arms, as it must.

### 10.3 Trust-region engagement and KL

R2 engaged the clip on 28/75 updates (37%, last at update 40); R3 on 25/75
(33%, last at update 47). `approx_kl` at `mappo.py:426` is the **k1** estimator
`((old_lp − new_lp) · d).sum() / denom`, **not k3**; k3 is not logged anywhere in
production, so the requested k3 KL is **unavailable** and the k1 values in §4
are reported in its place, labelled.

### 10.4 Advantage variance

`adv_std` is logged pre-normalisation. R2 mean 3.2357 (sd 0.3533), R3 mean
3.4606 (sd 0.3951) — R3's per-entry advantage spread is *higher*. See §4.

### 10.5 Effective rollout sample size

2.71× (not 4×), because `decision_frac` fell from 0.1392 to 0.0945. Full
derivation in §3.

### 10.6 Does the larger batch actually increase actor-gradient SNR? — YES

This is the mechanism question H1 rests on. Re-running the *same* coherence
probe at a 32-episode measurement buffer — matching the batch R3 was actually
trained at — with 4× the entries, so pure noise should scale by exactly 1/√4 =
0.500:

| arm | ‖g_shuf‖ 8ep | ‖g_shuf‖ 32ep | ratio | expected |
|---|---|---|---|---|
| A0 | 5.437e-02 | 2.839e-02 | **0.522** | 0.500 |
| R2 | 3.450e-02 | 1.594e-02 | **0.462** | 0.500 |
| R3 | 4.486e-02 | 2.123e-02 | **0.473** | 0.500 |

The shuffled null scales as 1/√N for all three arms to within a few percent.
That validates the null as pure noise and validates the measurement itself.

Against that validated null, the real gradient:

| arm | ‖g_real‖ 8ep→32ep | decay | real/shuffled 8ep | 32ep | growth | z | perm p |
|---|---|---|---|---|---|---|---|
| A0 | 7.063e-02 → 2.626e-02 | 0.372 | 1.2989 | 0.9250 | 0.712 | −0.31 | 0.5900 |
| R2 | 2.755e-02 → 1.818e-02 | 0.660 | 0.7987 | 1.1400 | 1.427 | +0.63 | 0.2250 |
| **R3** | 5.557e-02 → 4.304e-02 | **0.775** | 1.2387 | **2.0275** | **1.637** | **+4.37** | **0.0000** |

Reading this: a gradient that is pure noise decays at 0.500; a gradient with an
N-independent coherent component decays more slowly. R3 decays slowest (0.775),
retains the largest coherent component, and reaches **real/shuffled = 2.0275 at
z = +4.37, p = 0.0000** — unambiguously significant, where A0 (0.9250, p 0.59)
and R2 (1.1400, p 0.2250) both sit at their nulls. R3's growth factor 1.637
also matches the 2.71× effective-sample prediction (√2.71 = 1.65) almost
exactly.

**Answer: yes, decisively. The 4× batch delivered the SNR increase H1 predicted,
at close to the magnitude the effective sample size implies.** And the
risk-conditioned policy still got worse. This is why §12 treats H1 as falsified
at the mechanism level rather than as an intervention that failed to fire.

Where does R3's newly-coherent gradient point? Not at the hypothesis: at 32
episodes cos(g_real, g_synth) = −0.24368 while cos(g_real_hi, g_synth) =
+0.45051. The high-risk-restricted component points the right way for all three
arms (+0.913 / +0.536 / +0.451), but the full gradient's direction is dominated
by the ~95% of decision entries that are *not* high-risk.

### 10.7 EXPLORATORY — which action channel carries the risk response?

Every pre-registered metric is defined on MIGRATE_EDGE alone: P1, P2 and P3
directly, and P4/P5 through the `synth` reference direction. A policy answering
risk with MIGRATE_TO_CLOUD would score as unresponsive on all five. That was a
live possibility for R3 specifically, because R3's greedy argmax selects
MIGRATE_EDGE **zero** times at high risk on both neutral sets while selecting
MIGRATE_CLOUD 56/216 (RANDOM) and 288/3592 (UNION).

`_diag_R3_action_channels.py` measures the same Δ on all four actions, on the
same state sets, with the same cut points, clustering and bootstrap. Σ over the
four Δ is 0 by construction and is asserted at runtime.

| set | arm | Δ STAY | Δ EDGE | Δ CLOUD | Δ PREEMPT |
|---|---|---|---|---|---|
| RANDOM | A0 | +0.1151 | −0.0264 | −0.0886 | −0.0000 |
| RANDOM | R2 | −0.0692 | +0.2024 | −0.1331 | −0.0002 |
| RANDOM | **R3** | −0.0503 | **+0.1575** | **−0.1054** | −0.0018 |
| UNION | A0 | −0.0126 | +0.0191 | −0.0066 | −0.0000 |
| UNION | R2 | −0.1434 | +0.1579 | −0.0145 | −0.0000 |
| UNION | **R3** | −0.1061 | **+0.1242** | **−0.0178** | −0.0002 |

**The alternative is refuted.** R3's CLOUD Δ is *negative* on both sets
(−0.1054, −0.0178): at high risk R3 uses CLOUD **less**, not more. The risk
response is not hiding in an unmeasured channel — MIGRATE_EDGE genuinely is the
channel that absorbs mass leaving STAY, and P1/P2 measure the right quantity.
R3 simply does less of it than R2.

Two useful by-products. First, the EDGE column reproduces P1 and P2 to four
decimals from independently-constructed state sets, which is an extra check on
the pre-registered numbers. Second, A0's Δ STAY is **+0.1151** on RANDOM — the
untreated baseline actively *increases* STAY as risk rises, which is worse than
indifference and gives a sharper picture of what the treated arms are fixing.

### 10.8 Exploratory — R3's positive Δ never reaches a greedy decision

R3's greedy argmax counts at high risk: RANDOM `{STAY 160, MIGRATE_EDGE 0,
MIGRATE_CLOUD 56, PREEMPT 0}`; UNION `{STAY 3304, MIGRATE_EDGE 0, MIGRATE_CLOUD
288, PREEMPT 0}`. So R3's entire +0.1575 / +0.1242 response lives in the
stochastic policy's probability mass and **never once flips the arg-max**. R2 by
contrast flips it on 46.3% of high-risk RANDOM entries. A deployed greedy policy
derived from R3 would show no risk-driven edge migration at all.

### 10.9 Not computed, and why

The Var(raw) = Var(offset) + Var(paired) + 2Cov decomposition was **not**
recomputed for R3. `_diag_rung2_75_offset.py` requires a Rung 2.5
counterfactual native-deviation set (`SPRINT_7_RUNG2_5_native_dev_R2.json`)
that exists only for R2; there is no R3 equivalent and manufacturing one would
have meant a new training-time intervention during a frozen run. Stated as a
gap rather than omitted.

---

## 11. Pre-registered GO/NO-GO table

| # | metric | threshold | R2 (control) | R3 | verdict |
|---|---|---|---|---|---|
| P1 | RANDOM risk response Δ | ≥ +0.26 | +0.2024 | +0.1575, CI [+0.1184, +0.1931], z +8.14 | **FAIL** |
| P2 | UNION risk response Δ | ≥ +0.18 | +0.1579 | +0.1242, CI [+0.1102, +0.1398], z +16.35 | **FAIL** (CI strictly below R2's) |
| P3 | RANDOM Spearman(risk, P(EDGE)) | > +0.0831 | +0.0777 | +0.0656 | **FAIL** |
| P4 | real/shuffled gradient coherence | > 1.0 AND p < 0.05 | 0.7987, p 0.805 | 1.2387, z +1.00, p 0.1450 | **FAIL** (p) |
| P5 | cos(g_real, g_synth) | p < 0.05 | +0.30127, p 0.2450 | **−0.65706**, p 0.0100 | **directional FAIL** (§9) |

**Decision rule: GO iff P1 ∧ P2 ∧ (P3 ∨ P4).**

`FAIL ∧ FAIL ∧ (FAIL ∨ FAIL)` = **NO-GO.**

Not used in this decision, as pre-registered: reward, `clip_frac`, entropy,
sharpening, update engagement, final EDGE share, explained variance, critic
loss. All are reported in §4 and §10 as mechanism diagnostics only. I note
explicitly that several of them moved in R3's favour (zero saturation, higher
entropy, mildest sharpening collapse, coherence ratio above 1.0, and a
significant SNR gain at the training batch size) and that **none of that changes
the NO-GO**, because none of it is a criterion and none of it produced better
risk-conditioned behaviour.

---

## 12. Interpretation

**Per the pre-registration: "If P1 AND P2 fail → H1 is not supported, regardless
of reward."** Both failed. H1 is not supported.

The rule for a worse result also applies — P2's CI lies strictly below R2's, so
R3 is significantly worse than its control on the primary on-manifold metric:
**"do not retune batch size; treat the result as evidence against the current H1
formulation."**

What makes this more informative than a plain null:

1. **The manipulation was faithful.** 91 config fields, 3 differ, one of them a
   label. 75 updates in both. LR schedule identical to 0.00e+00. The only
   substantive change was rollout batch size.
2. **The mechanism H1 named was delivered.** At the batch R3 was trained at, its
   actor gradient carries a genuine coherent component that R2's and A0's do not:
   real/shuffled 2.0275, z +4.37, p = 0.0000, against a null verified to scale as
   1/√N. The SNR gain (1.64×) matches what the 2.71× effective sample size
   predicts.
3. **The predicted consequence did not follow.** Risk response got *worse* on
   both neutral state sets, and the rank correlation fell too.

So the inference is not "the batch was too small to help." It is: **variance /
low SNR in the raw GAE actor signal is not the binding constraint on
risk-conditioned learning in this system.** Raising SNR while holding everything
else fixed moved the mechanism metrics in the intended direction and moved the
behaviour in the opposite one.

The secondary diagnostics sharpen where the constraint actually lies. R3 has
zero softmax saturation, the highest entropy, and the mildest sharpening
collapse (7.0× vs R2's 152.0×) — so the collapse metrics that Rung 2.5 and 2.75
identified as symptoms are **not** on the causal path to risk response either.
An arm can be maximally healthy on every plasticity and saturation measure and
still be worse at the actual research question. Confirmed prospectively here,
not just cross-sectionally.

The most likely remaining explanation is visible in §10.6 and §10.8. The
high-risk-restricted gradient points the right way for every arm (cos(g_hi,
synth) = +0.913 / +0.536 / +0.451), but high-risk entries are only 3.3–6.5% of
decision entries, and the full gradient's direction is set by the other ~95%.
Raising SNR raises the fidelity of the *whole* gradient, which is dominated by
the low-risk bulk — so a better-estimated gradient is a better-estimated
*mostly-low-risk* gradient. R3's response never once flips a greedy arg-max
(§10.8) while R2's flips 46.3% of them, which is what "correct direction,
insufficient magnitude where it matters" looks like at the policy level.

**Classification, using the brief's own categories:** P4 improved on the ratio
and improved dramatically at the training batch size, while P1 and P2 did not.
The rule for that case is explicit — **do NOT label this a learning success;
classify it as mechanistic evidence only.** That is the classification I am
applying. R3 is a successful mechanism measurement and a failed intervention.

**Reproducibility, stated plainly:** this is one seed. A failed R3 does not
justify stacking multiple fixes, and it does not by itself prove H1 false in
general — it falsifies H1 for this configuration at this seed, with the unusual
strength that comes from the mechanism having demonstrably fired. No paper-level
claim is made from R3 in either direction.

---

## 13. Falsification / limitations

**1. P5 was under-specified by me.** I wrote a magnitude-free significance
threshold for a directional quantity. R3 met it with the sign reversed. I am
reporting it as a directional failure and not claiming the pass (§9). Any future
pre-registration of a cosine must state the sign.

**2. The pre-registered P4/P5 measurement buffer is underpowered.** At 8
episodes, `real/shuffled` moves A0 1.2989 → 0.9250, R2 0.7987 → 1.1400 (rank
order between A0 and R2 *reverses*) and R3 1.2387 → 2.0275 when the buffer goes
to 32. R3's cosine moves −0.65706 (p 0.0100) → −0.24368 (p 0.4250); R2's flips
sign. **The 8-episode coherence numbers quoted in Rung 2.75 for A0 and R2 are
therefore not stable estimates, and neither is R3's P4 or P5.** Following the
standing instruction, I have documented the flaw and produced the smallest
corrected measurement — the same probe at `--episodes 32`, written to a separate
artifact (`..._coherence_R3_bs32probe.json`) — rather than silently changing the
methodology. The pre-registered P4/P5 are still scored on the 8-episode buffer
as specified. **Future rungs should use the 32-episode buffer,** where the null
is verified to scale as 1/√N and the z-statistics are interpretable.

**3. P4's failure is a power failure, not a mechanism failure.** Scored as
specified, P4 fails on p. Measured with adequate power, the same quantity is
p = 0.0000. Both facts are reported; only the first enters the decision. A future
GO threshold on coherence should be set on the 32-episode buffer.

**4. One seed.** Every conclusion is seed-20260818-specific. R3's advantage on
the SNR mechanism is large and unlikely to be noise (z +4.37); its *deficit*
against R2 on P1/P3 is one run against one run and could plausibly be seed
variation. P2's deficit is the exception — non-overlapping CIs on 74,237
decision entries — but non-overlapping CIs *within* one seed pair still do not
establish a population difference across seeds.

**5. RANDOM is off-manifold for every arm.** It is neutral by construction, not
realistic; it visits states no trained policy would reach. UNION is on-manifold
but contains A0's and R2's selection effects rather than none, and contains
*neither* R3's. R3 is scored on state sets it did not help generate, which is
what makes the comparison identified but also means R3's own state distribution
is unexamined — deliberately, since the Sprint-6-era own-trajectory metric is
withdrawn in both directions.

**6. The minibatch tail is still unfixed**, in both arms equally. 4 of every 20
optimiser steps per update operate on a degenerate tail chunk. It is a shared
confound, not a differential one, but it means neither arm ran clean PPO.

**7. k3 KL is unavailable.** Production logs k1 (`mappo.py:426`). The requested
k3 comparison could not be made without modifying production code, which was
forbidden.

**8. The Var(offset)/Var(paired) decomposition is missing for R3** (§10.9).

**9. `explained_var` is target-relative.** Both arms use `mc` here, so §4's
comparison is nearer to like-for-like than a λ-vs-mc one, but I draw no
conclusion from it.

**10. `| tee` masks nonzero exit codes.** The coherence probe crashed with
`KeyError: 'R3'` at `_diag_rung2_75_coherence.py:318` after printing all per-arm
numbers but before writing its JSON, and the piped exit code was **0**. I caught
it only because the output file was absent. Fixed (`RUNG2_5_DTHETA.get(tag)`,
returning `None` rather than a fabricated number for arms with no Rung 2.5
replica) and re-run with `set -o pipefail` and no `tee`, reproducing A0's and
R2's values exactly. **Recorded as a methodological hazard for this project: any
probe piped through `tee` can appear to succeed while having crashed.** The
approved training command used `tee` as specified and was not altered; the
training run's own completion was verified from its artifacts, not its exit code.

**11. What would falsify §12's proposed explanation.** If the binding constraint
really is that high-risk entries are a 3–6% minority whose gradient contribution
is swamped, then re-weighting or re-sampling the actor loss toward high-risk
decision entries — holding batch size, updates and LR schedule fixed — should
raise P1/P2. If it does not, the constraint is elsewhere (candidates: the reward
does not actually reward correct migration enough at high risk; or the critic's
per-state offset structure, which Rung 2.5 showed pairing removes, still
corrupts the *relative* ordering within high-risk states). This is a
falsification condition, not a proposed R4 design.

---

## 14. Exact recommendation for the next rung

**Do not run another batch-size variant.** Pre-registered rule, triggered by P1
∧ P2 failing. Batch size is settled: the manipulation delivered its mechanism
and the behaviour got worse.

**Do not stack fixes.** A failed R3 does not license combining interventions.

**Recommended next hypothesis (H2), stated as a hypothesis only — no R4 design,
no parameters chosen, no command written:**

> The binding constraint on risk-conditioned learning is not the variance of the
> actor gradient but its **composition**: high-risk decision entries are 3.3–6.5%
> of all decision entries, the gradient restricted to them points the correct way
> in every arm measured (cos(g_hi, synth) = +0.913 / +0.536 / +0.451), and the
> full gradient's direction is set by the ~95% low-risk remainder. Improving the
> estimator therefore improves a predominantly low-risk gradient. Correcting the
> *composition* of the actor objective with respect to risk — rather than its
> variance — should raise P1 and P2.

Evidence this rests on, all from artifacts already on disk: §10.6 (cos(g_hi,
synth) positive for every arm while cos(g_real, synth) is not), the 3.3–6.5%
high-risk fractions, §10.8 (R3's response never flips a greedy arg-max), §10.7
(the response is in the EDGE channel, so the objective is aimed at the right
action), and §10.1–10.2 (R3 is the healthiest arm on every saturation and
plasticity measure and still the worse performer, so those are off the causal
path).

**Two things the next rung should settle before it commits to H2**, both cheap
and both requiring no new training run:

1. **Re-baseline P4/P5 on the 32-episode buffer** for A0, R2, A1, A2, A3, R3.
   The 8-episode numbers are unstable (§13.2) and one published rank order
   (A0 vs R2) reverses. Any coherence threshold for a future GO must be
   calibrated on the buffer where the null is verified to scale as 1/√N.
2. **Add a sign condition to every directional pre-registered metric**, so the
   P5 defect cannot recur.

**Also recommended, independent of H2:** because R3's mechanism gain was real and
its behavioural deficit is one seed against one seed, the R2-vs-R3 behavioural
comparison would benefit from replication before it is treated as established.
That is a lower priority than H2 — the pre-registration explicitly reserves
replication rungs for *successful* results, and R3 was not one.

**Explicitly not done here, per instruction:** R4 is not designed. No
hyperparameters, thresholds, commands or arm definitions for the next rung are
proposed in this report beyond the hypothesis statement and the two
methodological corrections above.
