# Sprint 7 — Phase 4 report: the R2 update-by-update trajectory

**Pre-registration:** `SPRINT_7_PHASE4_PREREG.md` (§1–§20.1), written and frozen before the run.
**Driver:** `marl/diag/_phase4_r2_trajectory.py` (md5 `504cff2d5eb563cc046abf01ee2ca0aa`, unmodified).
**Verifiers:** `marl/diag/_phase4_verify.py`, `marl/diag/_phase4_equiv.py`.
**Status:** run complete; **28 of 29 pre-registered criteria PASS, B6 FAILS**. The failure was not
repaired and the run was not repeated (per the execution brief). Evidence in §C.

This report contains no new experiment, no configuration change, no R4 proposal, and no
behavioural evaluation of the 76 checkpoints — that last is Phase 6 and is deliberately deferred
by §15.3.

---

## A. Execution record

```
python -m marl.diag._phase4_r2_trajectory --tag R2_traj_repro --critic-target mc \
    --rollout-episodes 8 --episodes 600 --seed 20260818 --device cpu
```

| quantity | value |
|---|---|
| wall time | 1604 s |
| updates observed | 75 / 75 expected |
| checkpoints written | **76 / 76** (`u000` … `u075`), all filenames unique |
| RNG-neutrality guard (B10) | **76 / 76** saves verified: neither the torch nor the numpy global generator moved |
| torch / threads | 2.7.1+cu118, 4 compute / 4 interop, `deterministic_algorithms=False` |
| driver's own final summary | first-20-episode mean reward −18.97, last-20 +4.68 (Δ +23.65); best update mean +12.65; episode ends 377 time-limit truncation / 223 true terminal; greedy TRAIN-window eval reward +8.71, success 0.740, reloc 59.4 |

New artifacts: `R2_traj_repro{.pth,_best.pth,_config.json,_history.csv,_updates.csv}`,
`R2_trajectory/R2_trajectory_u000.pth … _u075.pth`,
`R2_trajectory/SPRINT_7_P4_trajectory_manifest.jsonl` (76 rows),
`R2_trajectory/SPRINT_7_P4_trajectory_summary.json`. Nothing existing was overwritten (§J).

---

## B. Faithfulness verdict

```
python -m marl.diag._phase4_verify --self-test --tag R2_traj_repro   ->  18/19
python -m marl.diag._phase4_equiv  --self-test --tag R2_traj_repro   ->  23/23
```

| criterion | what it asserts | result |
|---|---|---|
| S1–S4 | B1 substitution harness is sound and lossless | PASS (4/4) |
| A, I6, B, C, D, D1a–d, E0, E1–E15, N1, N2, N3, RO | equivalence-comparator self-test: 14 mutations detected, 1 normalised, comparator strictly stronger than `torch.equal` in both directions, R2 checkpoint confirmed unmodified | PASS (23/23) |
| **B1a** | final `.pth` vs control, **container layer, NO normalisation** | **PASS** — 271/271 members; names identical modulo the archive stem (`mappo_R2_mc_target` ↔ `R2_traj_repro`); **265 / 265 tensor storages byte-identical**; only `data.pkl` and `.data/serialization_id` differ |
| B1b | same, after normalising `train.tag` + stem | PASS — **0 of 271** members differ |
| B1c | exact structural comparison of the unpickled graph | PASS — 476 leaves + 93 container nodes, **0 differences**, 1 leaf normalised (`extra.config.train.tag`), 0 opaque nodes, allowlist fully accounted |
| B1d | normalised whole-file md5 | PASS — `00da5284504cee8c1687866b315d6194` |
| B1e (a–d) | the same four layers on `_best.pth` | PASS — normalised md5 `4937a745120601ff79f3634b4b4b5d71` |
| B2 | `_updates.csv` raw md5 | PASS — `0181e2e93d8ae8d9b2266335de9e8156` (byte-identical) |
| B3 | `_history.csv` raw md5 | PASS — `42b44b4ad8e240d56a521d93440024be` (byte-identical) |
| B4 | `_config.json` modulo `wall_time_s` and `train.tag` | PASS |
| B5 | `u075` payload ≡ final `.pth` payload | PASS |
| **B6** | `u075` learning rates equal the pre-registered finals | **FAIL** — see §C |
| B7 | `u000` weight md5 ≡ pre-registered post-probe init | PASS — `ab714064bdf1ac56daabf5c163c92215` |
| B8 | `_best.pth` reproduces "best = last update" | PASS — episode 600, `mean_reward` 12.653897495678393, payload ≡ final |
| B9a–f | 76 checkpoints, unique, contiguous, 76 manifest rows re-verify on disk, all load through `MAPPO.load`, Adam state present in `u001`…`u075` and empty in `u000` | PASS (6/6) |
| B10 | instrumentation neutrality | PASS — 76/76 |

**The pre-flagged risk did not materialise.** The prereg warned that CPU BLAS thread scheduling
could make bit-exact replication impossible. It did not: 265 of 265 tensor storages — every weight,
every Adam first and second moment, every step counter — are **byte-identical to the control with no
normalisation applied at any point**. R2 is bit-exactly reproducible on this machine from
`seed 20260818` on `--device cpu`.

---

## C. The single failure: B6

`u075` stores `lr_critic = 1.3333333333333308e-05`. The pre-registered constant in
`_phase4_verify.py:78` is `1.3333333333333309e-05`, which Python parses to `1.333333333333331e-05`
— **exactly +1 ULP** above the stored value. `lr_actor` matched bit-for-bit.

The decisive point is that the discrepancy is demonstrable **against the frozen control alone**,
without reference to the replication:

| artifact | `repr(lr_critic)` | float64 bits |
|---|---|---|
| **CONTROL** `mappo_R2_mc_target.pth` | `1.3333333333333308e-05` | `3eebf647612f3687` |
| **CONTROL** `mappo_R2_mc_target_best.pth` | `1.3333333333333308e-05` | `3eebf647612f3687` |
| REPRO `R2_traj_repro.pth` | `1.3333333333333308e-05` | `3eebf647612f3687` |
| REPRO `R2_traj_repro_best.pth` | `1.3333333333333308e-05` | `3eebf647612f3687` |
| REPRO `R2_trajectory_u075.pth` | `1.3333333333333308e-05` | `3eebf647612f3687` |
| — pre-registered constant | `1.3333333333333309e-05` | `3eebf647612f3688` |
| — `1e-3 * (1 - 74/75)` evaluated in float64 | `1.3333333333333308e-05` | `3eebf647612f3687` |

All five artifacts agree bit-for-bit, and they agree with the exact float64 value of the schedule
expression `base_lr × (1 − 74/75)`. The pre-registered literal is therefore a **1-ULP transcription
error in the reference value**, not a difference between the control and the replication.

**No repair was made.** The constant in `_phase4_verify.py` was not edited, no rerun was attempted,
and B6 is recorded as FAIL. Editing a pre-registered reference after seeing the measurement is
exactly what RULE 3 and RULE 9 forbid, and the execution brief said explicitly not to repair or
rerun. The correction belongs in a future amendment log entry, made deliberately and visibly, not
silently inside this report.

**Consequence for the science:** none. B5 independently establishes that `u075`'s payload is
identical to the final checkpoint, and B1a establishes that both are byte-identical to the control,
including the optimizer `param_groups` that hold the learning rates. The quantity B6 was meant to
guard is verified twice over by criteria that passed.

---

## D. Manifest ↔ CSV consistency (§16)

75 updates × 9 stats shared between `SPRINT_7_P4_trajectory_manifest.jsonl` and
`R2_traj_repro_updates.csv` = **675 values compared, 0 mismatches**. `adv_mean` is the one manifest
stat not present in the CSV. `u000` correctly carries `stats: null` and `buffer_T: null`. Episode
bookkeeping is `[0] + [8i]` as pre-registered. `buffer_T` varies 3035–3200 across updates (episode
lengths differ; the rollout is 8 episodes, not a fixed step count). `lr_actor` and `lr_critic` share
one `lr_scale`, ending at 0.013333333333333308.

---

## E. WHEN: the pre-registered per-update trajectory

The admissible measurement set here is §15's: the 76 manifest rows and their per-update stats. All
ten stats are reported; none were selected after inspection.

### E.1 Five blocks of 15 updates (updates 1–75 = episodes 8–600)

| stat | u1–15 | u16–30 | u31–45 | u46–60 | u61–75 |
|---|---|---|---|---|---|
| entropy | 0.6158 | 0.3281 | 0.2007 | 0.1702 | 0.1616 |
| clip_frac | 0.0479 | 0.0226 | 0.0070 | 0.00085 | 0.00003 |
| approx_kl | +0.00442 | +0.00200 | +0.00083 | −0.00016 | −0.00047 |
| explained_var | 0.2629 | 0.5327 | 0.6566 | 0.6727 | 0.7126 |
| decision_frac | 0.0548 | 0.1117 | 0.1708 | 0.1803 | 0.1782 |
| critic_loss | 16.49 | 13.44 | 11.67 | 10.52 | 8.26 |
| adv_std | 3.678 | 3.403 | 3.228 | 3.055 | 2.814 |
| value_mean | −0.648 | 1.350 | 1.788 | 1.010 | 1.320 |
| adv_mean | 0.574 | 0.277 | 0.113 | 0.432 | 0.198 |
| actor_loss | −0.0344 | −0.0041 | −0.0039 | +0.0089 | −0.0015 |

`approx_kl` here is the **k1** estimator (`mappo.py:426`:
`((old_logp − new_logp) · mask).sum() / denom`). k1 is unbiased for the KL but is not
non-negative sample-by-sample, so the negative block means in the last two blocks are a
sampling artifact of a KL that has gone to ~0, not a negative divergence.

### E.2 Monotonicity (Spearman ρ against update index, n = 75)

| stat | ρ | t(73) |
|---|---|---|
| entropy | **−0.946** | −24.90 |
| clip_frac | **−0.926** | −20.98 |
| explained_var | **+0.902** | +17.87 |
| critic_loss | −0.871 | −15.12 |
| adv_std | −0.863 | −14.58 |
| decision_frac | +0.859 | +14.32 |
| approx_kl | −0.594 | −6.30 |
| value_mean | +0.400 | +3.73 |
| actor_loss | +0.359 | +3.29 |
| adv_mean | −0.188 | −1.63 (n.s.) |

### E.3 Timing landmarks

| landmark | update | ≈ episode |
|---|---|---|
| `explained_var` first reaches 0.50 | u20 | 160 |
| `decision_frac` first reaches 0.10 | u24 | 192 |
| `entropy` permanently below 0.30 | u25 | 200 |
| `decision_frac` first reaches 0.15 | u27 | 216 |
| `explained_var` first reaches 0.65 | u30 | 240 |
| `decision_frac` first reaches 0.17 (plateau) | u36 | 288 |
| `clip_frac` first zero reading | u36 | 288 |
| `entropy` permanently below 0.20 | u40 | 320 |
| `explained_var` first reaches 0.70 | u41 | 328 |
| **last update with `clip_frac` > 0** | **u61** | **488** |
| `clip_frac` identically 0 from here on | u62 | 496 |
| \|`approx_kl`\| < 1e-3 from here on | u65 | 520 |

### E.4 Is anything still moving in the last 30 updates?

OLS slope per update over u46–u75, normalised by the block's own standard deviation:

| stat | mean u46–75 | sd | slope/update | \|slope\|·30 / sd |
|---|---|---|---|---|
| critic_loss | 9.389 | 1.780 | −0.13614 | 2.29 |
| clip_frac | 0.00044 | 0.00092 | −0.000068 | 2.19 |
| adv_std | 2.935 | 0.222 | −0.01534 | 2.08 |
| adv_mean | 0.315 | 0.223 | −0.01322 | 1.78 |
| explained_var | 0.6927 | 0.0369 | +0.00212 | 1.72 |
| entropy | 0.1659 | 0.0148 | −0.00084 | 1.71 |
| value_mean | 1.165 | 0.478 | +0.01573 | 0.99 |
| approx_kl | −0.00032 | 0.00129 | +0.0000073 | 0.16 |
| decision_frac | 0.1793 | 0.0091 | −0.000045 | 0.15 |
| actor_loss | 0.00371 | 0.0497 | −0.00011 | 0.07 |

No stat's total drift over the final 30 updates exceeds ~2.3 within-block standard deviations. The
critic's two stats (`critic_loss`, `explained_var`) are the ones still moving fastest and in the
improving direction; the actor's trust-region stats are at the floor.

### E.5 What §E licenses, stated narrowly

1. **The actor's optimisation signal extinguishes in the first two thirds of training and does not
   return.** `entropy` is below 0.30 from u25 (ep 200) and below 0.20 from u40 (ep 320);
   `clip_frac` has its last non-zero reading at u61 (ep 488) and is identically 0 for the final
   14 updates; `approx_kl` is under 1e-3 from u65. This is a **progressive** decline over u1–u40,
   not a discrete event at any single update.
2. **The critic is still improving when the actor stops responding.** `explained_var` passes 0.50 at
   u20, 0.65 at u30, 0.70 at u41 and is the fastest-moving stat in the final block (+0.0021/update);
   `critic_loss` falls monotonically 16.5 → 8.3 and is still falling at u75. So the actor's
   trust-region activity dies **while** value estimation gets better, not because it gets worse.
   Any account in which poor critic quality is the binding constraint on the actor over the second
   half of R2 has to explain that ordering.
3. **The final ~14–19% of training carries almost no policy change.** With `clip_frac ≡ 0` and
   \|`approx_kl`\| < 1e-3 from u62–u65 onward, the policy's per-update movement from ep ~496 to
   ep 600 is below the trust-region and KL resolution of the pre-registered instruments. Whether the
   *cumulative* drift over those 14 updates is behaviourally meaningful is not answerable here — but
   the checkpoints exist, so it is answerable in Phase 6.
4. **The induced state distribution shifts on the same clock and then freezes.** `decision_frac`
   (fraction of agent-timesteps with ≥2 legal actions — choice availability, *not* risk response)
   more than triples, 0.055 → 0.178, reaching its plateau at u36 (ep 288) and going flat thereafter
   (slope/sd 0.15, the second-smallest of the ten). The window in which the policy reshapes the
   states it visits is the same u1–u36 window in which entropy collapses.
5. **`adv_mean` is the only stat with no reliable trend** (ρ = −0.188, t = −1.63). `value_mean`,
   `actor_loss` and `adv_mean` are all non-monotone; they are reported for completeness and no
   claim rests on them.

---

## F. The `history.csv` series — descriptive, and explicitly outside §15

**Boundary, stated before the numbers.** `R2_traj_repro_history.csv` is byte-identical to
`mappo_R2_mc_target_history.csv` (criterion B3). It therefore contains **nothing Phase 4 discovered**
— it has been on disk since R2 was trained. Trend analysis of it is also **not in §15's admissible
list**. It is included because it is the only per-episode record of environment outcomes that
already existed, because the analysis was run over *all ten* columns rather than a chosen subset,
and because it is purely descriptive. **No causal claim in this report rests on it.**

### F.1 Six blocks of 100 episodes

| episodes | reward | relocations | preemptive | protected | protected/reloc | lost | infeasible |
|---|---|---|---|---|---|---|---|
| 1–100 | −16.73 | 85.12 | 84.38 | 8.83 | 0.1037 | 10.94 | 1.39 |
| 101–200 | −13.64 | 79.58 | 78.62 | 8.18 | 0.1028 | 10.99 | 6.95 |
| 201–300 | −5.23 | 73.10 | 72.21 | 8.17 | 0.1118 | 10.31 | 17.47 |
| 301–400 | +0.42 | 71.49 | 70.42 | 9.43 | 0.1319 | 9.50 | 21.22 |
| 401–500 | −0.20 | 69.95 | 68.95 | 9.24 | 0.1321 | 9.68 | 20.93 |
| 501–600 | +1.34 | 70.66 | 69.89 | 8.61 | 0.1219 | 9.34 | 19.40 |

`protected/reloc` is a **post-hoc derived ratio**, not a logged column; it is shown for context only
and is not admissible evidence.

### F.2 All ten columns, Spearman ρ against episode (n = 600)

| column | ρ | t(598) | ep 1–60 → ep 541–600 | reading |
|---|---|---|---|---|
| relocations | **−0.704** | −24.22 | 86.43 → 70.65 | falls |
| infeasible | **+0.689** | +23.26 | 0.72 → 18.87 | rises |
| preemptive | **−0.686** | −23.09 | 85.82 → 69.90 | falls |
| reward | +0.381 | +10.09 | −19.55 → +3.74 | rises |
| success_rate | +0.318 | +8.21 | 0.681 → 0.745 | rises |
| lost | −0.264 | −6.69 | 11.22 → 8.97 | falls |
| critical_lost | −0.140 | −3.47 | — | flat |
| energy | +0.136 | +3.37 | — | flat |
| sla | +0.063 | +1.55 | — | flat (n.s.) |
| **protected** | **+0.053** | **+1.29** | **8.667 → 8.617** | **flat (n.s.)** |

### F.3 Where a transition is resolvable

Under 20-episode smoothing, the episode at which a series first reaches 50% and 90% of its
block-1 → block-5 change is only meaningful when that change is large relative to the noise. It is
resolvable for exactly three columns:

| column | total change | 50% reached | 90% reached |
|---|---|---|---|
| relocations | −14.375 | ep 201 | ep 295 |
| preemptive | −14.367 | ep 194 | ep 296 |
| infeasible | +17.617 | ep 196 | ep 216 |

For the other seven columns (reward, success_rate, lost, critical_lost, sla, protected, energy) the
test **degenerates** — the first smoothing window already sits within the target band, so it reports
ep 1 for both thresholds. That is a statement about the test being undefined at noise-scale changes,
not a finding that those series changed at episode 1.

### F.4 The two things worth naming

**`protected` does not move.** `protected` = `tasks_protected_before_failure`
(`env.py:955–996`) is the count of migrations whose source host actually failed inside the window
and whose task survived. It is computed post-hoc in `finalise_migration_outcomes()`, is never called
from `step()`, is never observed by a policy, and never enters the reward — it is the closest thing
in this corpus to ground-truth risk-aware-migration success. Over 600 episodes it is **statistically
flat** (8.83 → 8.61 by block, ρ = +0.053, t = 1.29), while total migration volume falls ~17% and
`reward` rises +23. Descriptively: across the episodes in which the actor's entropy collapses and its
visited-state distribution shifts, R2 learns to migrate *less*, its reward improves, and the number
of successful risk-protective migrations does not change.

**`infeasible` is a contention counter, and its rise says the targets concentrated.** Verified at
`env.py:562–567`: the actor samples from a masked distribution (`mappo.py:77–79`), so an action that
is illegal *at sample time* cannot be drawn; `infeasible` increments only when an action that was
legal at sample time has become illegal by the time that agent is applied, because an earlier agent
in the rotating apply order already consumed the destination slot. It therefore counts **agents
competing for the same destination host at the same tick**, not illegal actions. So `relocations`
falling 85 → 71 while `infeasible` rises 1.4 → 19.4 means the policy issued *fewer* migrations
aimed at a *narrower* set of destinations — a concentration of migration targets, consistent with
the entropy collapse over the same span. It is also the earliest and sharpest of the three
resolvable transitions (90% of its change by ep 216, versus ep 295–296 for the two volume series).

---

## G. Which Sprint 7 mechanisms Phase 4 can rule in or out

Phase 4 manipulated zero variables, so it cannot confirm a causal mechanism. What it can do is
(i) settle claims about reproducibility, and (ii) date things the endpoint corpus could not date.

### G.1 Ruled IN

**Bit-exact reproducibility of the R2 arm.** 265/265 tensor storages byte-identical to the control
with no normalisation. Two consequences that were previously open:

- No Sprint 7 result about R2 can be attributed to run-to-run nondeterminism. The pre-flagged
  thread/BLAS-scheduling risk is empirically absent on this machine at 4 threads.
- The 76 checkpoints are a decomposition of *the actual control run*, not of a statistically similar
  one. Anything measured on them in Phase 6 speaks about `mappo_R2_mc_target.pth` itself.

**Actor saturation, now dated.** The *softmax collapse* finding (previously an endpoint observation:
50.5% of R2's high-risk states at max-prob > 0.99 versus A0's 0.0000) is confirmed as a real
temporal process and localised: progressive from u1, `entropy` past 0.30 by u25 and 0.20 by u40,
trust-region activity gone by u62. It is not an artifact of the final checkpoint and not a discrete
late-training event.

### G.2 Ruled OUT / downgraded

**"The critic's value quality is what gates the actor in late R2."** The trajectory orders these the
wrong way for that story: `explained_var` climbs monotonically through 0.50 (u20), 0.65 (u30) and
0.70 (u41) and is still the fastest-improving stat over u46–75, while the actor's `clip_frac` and
`approx_kl` fall to the floor over exactly that span. Improving value estimates coincide with *less*
policy movement, not more. This does **not** revive or refute the *critic bias inverts the advantage
sign* mechanism as such — that concerns the sign of the advantage, which no stat here measures — but
it removes "the critic simply had not learned yet" as an explanation for the second half of R2.

A weak quantitative by-product, offered as a bound only: a per-state offset `c(s)` added to `V`
inflates `Var(return − V)`, so `explained_var = 0.71` caps the variance of any such offset at ≤0.29
of return variance by u75. That is consistent with *Mechanism 2 (per-state offset) is closed* and
adds nothing to it.

**"The endpoint checkpoint is a snapshot of a still-learning policy."** It is not. From u62 no ratio
left the trust region and per-update KL is under 1e-3, so ep ~496–600 — 14 of 75 updates — carries
policy change below the resolution of the pre-registered instruments. The prior finding that *20/20
endpoint mechanism readings order the arms backwards* (mean ρ −0.900) was interpreted as
"endpoint = headroom, not learning". Phase 4 adds a sharper reason to distrust endpoint-only
measurement: for R2 the endpoint is a near-fixed-point that had already been reached with ~19% of
training left to run, so an endpoint reading cannot distinguish "learned this" from "stopped moving
here".

### G.3 Cannot be touched by Phase 4's measurement set

None of the pre-registered per-update stats is conditioned on predicted failure risk. There is no
high-risk/low-risk split, no per-regime advantage, no migration outcome anywhere in the manifest.
So Phase 4 is silent — neither for nor against — on:

| mechanism | why Phase 4 cannot speak to it |
|---|---|
| Regime-selective learning asymmetry (R3 learned the low-risk bulk, not the high-risk minority) | every instrument for it is risk-split; none is in the manifest |
| Low-risk-bulk dilution (Rung 3) | requires risk-conditioned gradient mass ratios |
| Paired advantage estimator / high-risk agreement | requires paired per-state evaluation of saved policies |
| `rollout_episodes` as a pure variance intervention | one batch size was run; no contrast exists here |
| The sign of the advantage in high-risk states | no stat in the manifest carries a sign-quality measure |

The blocker recorded before Phase 4 was **a missing observable, not a missing hypothesis**. That
observable now exists — 76 policy+critic snapshots on the exact control trajectory. Reading it is
Phase 6. §15.3 defers it deliberately: choosing which behavioural curve to compute *after* seeing
the trajectory is the cherry-picking RULE 9 forbids, so the Phase 6 metrics must be pre-registered
before any of the 76 checkpoints is evaluated on the frozen RANDOM/UNION sets.

---

## H. Integrity (RULE 12)

`_PHASE41_integrity/SPRINT_7_P41_*_after.md5` is the before-state and was **not** overwritten.
`_PHASE4RUN_integrity/` records the after-state under the same §20.1 scopes, plus a new
`SPRINT_7_P4RUN_trajectory.md5` covering everything in `R2_trajectory/`.

| manifest | lines | scope |
|---|---|---|
| `SPRINT_7_P4RUN_code_after.md5` | 52 | `marl/**/*.py` |
| `SPRINT_7_P4RUN_artifacts_after.md5` | 156 | `saved_models/marl/*.{md,json,csv,pth,txt}` (maxdepth 1) |
| `SPRINT_7_P4RUN_checkpoints_after.md5` | 25 | `saved_models/marl/*.pth` (maxdepth 1) |
| `SPRINT_7_P4RUN_trajectory.md5` | **78** | all of `R2_trajectory/` — 76 `.pth` + manifest + summary |
| `SPRINT_7_P4RUN_inputs_after.md5` | 3 | failure history / log / OOF predictor |
| `SPRINT_7_P4RUN_prior_manifests.md5` | 60 | every earlier `*.md5`, self-reference excluded |
| `_newcode.md5` / `_newartifacts.md5` / `_changed.md5` | 0 / 6 / 0 | deltas vs the Phase 4.1 after-state |

**Verdict — Phase 4.1 after → Phase 4 run after, by manifest diff:**

| manifest | added | modified | removed |
|---|---|---|---|
| code (52 `.py`) | **none** | **none** | none |
| artifacts (150 → 156) | the 5 `R2_traj_repro*` files + this report | **none** | none |
| checkpoints (23 → 25) | `R2_traj_repro.pth`, `R2_traj_repro_best.pth` | **none** | none |
| inputs (3) | none | **none** | none |

So: 6 additions, **0 modifications, 0 removals** — the run added artifacts and changed nothing,
including changing no code (the driver rebinds `MAPPO.update` in memory and restores it, so not even
the diagnostic modules differ). `git status` reports **no** `M` or `D` line; every Phase 4 path is
new and untracked.

Corroborating the CSV criteria directly from the manifest, `R2_traj_repro_history.csv` hashes
`42b44b4ad8e240d56a521d93440024be` and `R2_traj_repro_updates.csv` hashes
`0181e2e93d8ae8d9b2266335de9e8156` — the control's own values, recorded independently of B2/B3.

All five protected R2 artifacts hash-unchanged: `mappo_R2_mc_target.pth 00da5284…`,
`_best.pth cbf53ca7…`, `_config.json cd36140b…`, `_history.csv 42b44b4a…`,
`_updates.csv 0181e2e9…`.

All seven forbidden production files hash-unchanged: `mappo.py 8614d016…`, `train.py 8ef42424…`,
`env.py 4e565282…`, `rollout.py 3a829646…`, `config.py 3adca5da…`, `evaluate.py 79c8e984…`,
`risk_provider.py 02219832…`.

No production code was modified, no configuration was changed, no driver was tuned toward
agreement, no second variable was introduced, and no R4 was designed. The `_changed.md5` delta is
empty because `SPRINT_7_PHASE4_PREREG.md` was deliberately **not** amended in this phase — see §I.

---

## I. Open item requiring a decision (not acted on)

`_phase4_verify.py:78` holds a 1-ULP-wrong `final_lr_critic`. It has been left wrong on purpose so
that B6's FAIL stands in the record. Correcting it is an amendment to a pre-registration and should
be logged in `SPRINT_7_PHASE4_PREREG.md` §19 with this report as the evidence, not edited silently.
