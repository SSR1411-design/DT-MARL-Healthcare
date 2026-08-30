# Sprint 6.5 — MAPPO policy learning / risk-sensitivity diagnosis

Status: **diagnosis complete and verified. The fix objective was NOT met.**
Nothing committed, nothing pushed, no PR. Sprint 7 not started.

All numbers below are held-out evaluation (8 episodes, eval start ticks
[491, 698], disjoint from the training start window [9, 491]), single seed
20260818, 600 episodes, `--device cpu`.

---

## 1. Root cause

MAPPO ignores predicted failure risk because of **learning dynamics, not
representation and not reward specification**:

* **H — credit assignment** on a payoff that is spread over a task's whole
  lifetime, combined with
* **G — the PPO update going to zero** long before the policy has visited
  enough high-risk / migrate states to learn from them, acting on
* **F — a sparse high-risk region** (only ~4.7% of decision states have
  risk > 0.18; the trained policy chose MIGRATE_TO_NEIGHBOR_EDGE in such
  states 15 times out of 8659 decisions).

Two genuine implementation defects (E-class) were also found and fixed, but
neither is the cause of risk insensitivity.

Refuted by measurement: **A** (risk numerically underrepresented), **B**
(criticality drowned out), **C**/**D** (reward does not reward correct
preemption / rewards migration regardless of risk), **J** (architecture
inadequate), **K** (baseline exploiting a shortcut).

## 2. Evidence

* **The environment does reward the intended behaviour.** Exact one-step
  deviation counterfactuals by deterministic replay (0 replay mismatches,
  n=30/cell): true `A(MIGRATE_EDGE | risk>0.18) = +2.671 ± 0.806 SE`
  (median +4.058, P(>0)=0.77, ΔLost −0.50) versus
  `A(MIGRATE_EDGE | risk≤0.18) = −1.337 ± 0.967`. Gap **+4.01**. So C and D
  are refuted: the reward already encodes the risk-conditioned relationship.
* **The architecture can represent the mapping.** Behaviour-cloning
  risk-threshold@0.18 into the *same* actor reaches accuracy 1.0000 with a
  risk-sweep span of **0.3935**, versus the trained policy's 0.004. So A, B
  and J are refuted — this is a learning failure, not a capacity failure.
* **The learner's own value estimate is wrongly signed** where it matters:
  −0.266 at high risk against a true +2.671; correctly signed at low risk.
* **The PPO update dies in every arm.** `clip_frac → 0.0000` and
  `approx_kl → ~0` by update 75 for A0/A1/A2/A3 alike; entropy falls to
  0.30–0.37. Only 75 updates exist (600 episodes / 8 per update).
* **Reward sizing is correct over the horizon** (this corrects an earlier
  per-step reading of mine): one step of migrating costs more than one step of
  staying (−0.6443 vs −0.0143), but exposure is charged every step for ~150
  steps while the migration charge is paid once, so over the failure horizon
  migrating wins by **+8.267** (`sanity_test.py` probes 4a/4b). Rescaling
  `P_risk_expose` would therefore attack a non-problem and double-count the
  loss; that planned arm was dropped.
* **Contention is inherent, not a bug.** 545/783 = 69.6% of relocation
  attempts are refused, all cloud, all attributable to genuine
  oversubscription of `cloud_slots=8` — essentially unchanged from Sprint 6's
  ~70%. Forced cloud migrations return exactly −0.050 = `P_infeasible ×
  reward_scale` in both risk buckets.
* **The policy concentrates on the worthless action.** A1 attempted cloud 1060
  times (~77% refused, true advantage −0.050) and edge only 10 times (true
  advantage +2.671). `PREEMPTIVE_REROUTE` is 0 in every arm.

## 3. Files modified

| file | change |
|---|---|
| `marl/env.py` | per-agent `agent_mask(i)`; rotating apply order + re-check in `step()`; dedicated `migration_severity` event field; `crit_m` reads it |
| `marl/config.py` | corrected the documented reward equation; added two default-off legacy switches |
| `marl/train.py` | `--legacy-sprint6`, `--legacy-fixed-apply-order`, `--legacy-dead-migration-criticality`, `--w-criticality-migration` |
| `marl/tests_env.py` | T6 replica corrected; T17/T18/T19 added (19 tests) |
| `marl/_diag_counterfactual.py` | contention probe moved to `step()`; `crit_m` replica corrected |
| `marl/_diag_reward_terms.py` | `crit_m` replica corrected |

`marl/evaluate.py` was **not** modified — it already restores `EnvConfig` from
the checkpoint and already implements all six Phase 6 probes. No Java or
CloudSim file was touched. The failure predictor was not touched.

## 4. Exact changes

1. **Contention was decided by agent index.** Sprint 6 applied actions in fixed
   order 0..n−1, so every contested cloud slot went to the lowest-index
   demander (measured: lowest index won 7/7). With ten separate actors that
   gives ten nominally homogeneous agents ten different effective action
   spaces. Now the order rotates by `step_idx` (measured: lowest index wins
   3/21 over 21 contended steps). **This does not reduce the refusal rate**
   (~70% before and after) — it redistributes it.
2. **`w_criticality_migration` was dead code.** `ev["severity"]` is written
   only on the loss/completion/SLA paths, never in `_relocate`, so `crit_m` was
   identically 1.0 and the charge was pinned at exactly `P_migration = 4.000`
   per unit cost. A dedicated `migration_severity` field now carries the
   severity of the task actually moved; the charge spans 5.280–6.576 for
   severity 0.320–0.644. A separate field was required because `ev["severity"]`
   also weights the completion, loss and SLA terms.

Neither change adds any `if risk > threshold` logic. No decision rule was
hard-coded; only coefficients and two bug fixes changed.

## 5. Training configuration

Identical across arms except the single stated change: seed 20260818, 600
episodes, `rollout_episodes=8` → 75 PPO updates, 400 steps/episode (dt 2.0 s =
800 s), γ=0.999, λ=0.995, clip 0.2, value clip 0.2, `entropy_coef` 0.02,
`value_coef` 0.5, 4 epochs × 4 minibatches, `lr_actor` 7e-4 / `lr_critic` 1e-3
with linear anneal, 10 separate actors [128,128] (23.3k params each), critic
[256,256] (194k), obs 48 / state 489, `cloud_slots=8`, **`--device cpu`**.

## 6. Before / after metrics

| arm | change | reward | rt@0.18 | gap | success | lost | critLo | reloc | protected |
|---|---|---|---|---|---|---|---|---|---|
| A0 | Sprint 6 (legacy switches) | 20.62 ± 6.16 | 76.50 | −55.9 | 0.719 | 11.2 | 4.4 | 35.9 | 3.25 ± 2.22 |
| A1 | both bug fixes | 15.69 ± 7.26 | 67.31 | −51.6 | 0.741 | 10.4 | 4.2 | 29.1 | 3.25 ± 2.17 |
| A2 | A1 + `w_criticality_migration=0` | 12.33 ± 19.39 | 76.54 | −64.2 | 0.719 | 10.6 | 3.9 | 67.8 | 7.25 ± 2.95 |
| A3 | A1 + `entropy_coef=0.05` | 6.82 ± 12.87 | 67.31 | −60.5 | 0.744 | 8.9 | 4.0 | 55.0 | 11.12 ± 2.15 |

**Reward is not comparable across all four arms.** Fix 2 changes the reward
function, so A0/A2 sit on one scale (risk-threshold@0.18 ≈ 76.5) and A1/A3 on a
stricter one (67.31). `static-no-migration` is an exact internal control at
−37.06 in every arm, and risk-threshold's *behaviour* is byte-identical
everywhere (reloc 36.2, prot 30.2, lost 3.4) — only its reward moves. Compare
arms on the reward-independent columns and on the within-arm gap.

**A0 reproduces Sprint 6 bitwise**: 600/600 episodes and 75/75 PPO updates
identical, held-out reward +20.621 ± 6.159 vs the recorded 20.62 ± 6.159,
protected 3.25 vs 3.25, ablation 3.25 → 3.12 vs 3.25 → 3.12, action histogram
9832/77/842/0 vs 9832/77/842/0.

## 7. Risk sensitivity, before / after

P(relocate) span over a risk sweep 0.00 → 0.99 on held-out states:

| arm | span | corr |
|---|---|---|
| A0 (= Sprint 6) | 0.004 | +0.990 |
| A1 | **0.016** | +0.999 |
| A2 | 0.004 | +1.000 |
| A3 | 0.002 | +0.974 |
| behaviour-cloned threshold (capacity ceiling) | **0.3935** | — |

A1 improved sensitivity 4× and remains **~25× short of what the same network
demonstrably can express**. A2 and A3 fall back to A0 levels. The sign is
correct everywhere; the magnitude is negligible everywhere.

## 8. Criticality sensitivity, before / after

| arm | span | corr | direction |
|---|---|---|---|
| A0 | 0.004 | −0.987 | wrong |
| A1 | 0.004 | −0.989 | wrong |
| A2 | 0.001 | −0.993 | wrong |
| A3 | 0.001 | +0.801 | nominally right, magnitude meaningless |

A2 was designed to test whether the criticality-scaled migration charge caused
the wrong direction — the only criticality term in the reward that opposes
protecting critical patients. **It did not**: the correlation stayed negative
(−0.993). That hypothesis of mine is refuted. A3 flips the sign to +0.801 but
with a span of 0.001, which is numerically meaningless.

Caveat that limits both probes: severity is derived as
`0.036 · (task_id % 10) + 0.32`, so its real support is **[0.320, 0.644]** and
the 0.00 → 1.00 sweep grid is ~3× wider than anything seen in training.

## 9. Baseline comparison

MAPPO loses to risk-threshold in **every** arm, on reward and on protection:
best gap −51.6 (A1); protected 11.12 (A3, the best arm) vs 30.2; lost 8.9 vs
3.4; success 0.744 vs 0.847. MAPPO does beat static-no-migration (−37.06),
random-legal and reactive-threshold in all arms.

**The gains in `protected` are not risk-driven.** They come from migrating
more. Zero-risk ablation, same trained policy with the risk channel forced to 0:

* A0 3.25 → 3.12 · A1 3.25 → 2.88 · A2 7.25 → 7.00 · A3 11.12 → 10.25

A3 keeps 92% of its protections with **no risk input at all**, while its
relocations rose from 29.1 to 55.0. It protects by volume, not by targeting.

## 10. Tests

`marl/tests_env.py` **19/19**, `marl/sanity_test.py` **6/6**. New tests: T17
(contention not decided by agent index), T18 (criticality reaches the migration
charge), T19 (legacy switches reproduce both Sprint 6 defects, so A0 is
trustworthy). T6 now matches the documented equation to 4.48e-07 (float32
noise). Note T2 is a **weak guard**: it samples uniformly among legal actions
and so passed all through Sprint 6 while ~70% of the trained policy's
relocations were being refused.

## 11. Remaining limitations

1. **One seed per arm**, 8 held-out episodes, reward SD ±6–19. Arm-to-arm
   reward differences are largely within noise; only the sweep spans, the
   ablations and the bitwise A0 reproduction are tight. No seed was selected
   for favourability — 20260818 is Sprint 6's seed, used unchanged everywhere.
2. Risk sensitivity was never established. The central objective is unmet.
3. `PREEMPTIVE_REROUTE` is never used in any arm (its mask requires a task that
   has not begun computing).
4. ~70% of cloud relocation attempts are refused by genuine oversubscription;
   the action is close to worthless as offered (true advantage −0.050).
5. Severity support is [0.320, 0.644], narrower than the probe grid.
6. Integration remains **trace-driven replay**, not closed-loop co-simulation.
7. Risk scores are uncalibrated out-of-fold sigmoid outputs — the term
   `predicted_failure_risk` is used deliberately.
8. Two arms were run on CUDA before the device issue was found; they are
   preserved as `mappo_A0_CUDA_device_mismatch*` and
   `mappo_A1_CUDA_killed_at_90ep*` and are **not** comparable to the CPU ladder.

## 12. Is Sprint 6.5 complete?

**As a diagnosis, yes.** The cause is identified, localised and supported by
counterfactual replay, a capacity probe, per-term reward decomposition, PPO
update traces and a bitwise-reproducible baseline. Two real defects were fixed
and unit-tested.

**As a fix, no.** MAPPO is still risk-insensitive and still loses to a
one-line threshold rule. Reporting that plainly is the result.

## 13. Sprint 7

1. **Attack the learning stall, not the reward.** The gradient is dead by
   update 75 in all four arms. Try many more updates with a smaller
   `rollout_episodes`, KL-targeted early stopping, or a learning-rate schedule
   that does not anneal to 1.3% while `clip_frac` is already 0.
2. **Fix the sample problem directly.** High-risk states are ~4.7% of
   decisions and the profitable action is chosen in ~0.17% of them. Prioritised
   replay of high-risk decision states, or start-state biasing toward
   pre-failure windows, targets F without touching the reward.
3. **Make the cloud action honest.** ~70% refusal means the learner is mostly
   sampling a no-op. Either expose occupancy so the mask is predictive, or
   raise `cloud_slots`, then re-measure.
4. **Use the capacity probe as a warm start.** BC to risk-threshold reaches
   span 0.3935 and accuracy 1.0; initialising from it and fine-tuning with PPO
   separates "cannot learn" from "cannot learn *from this signal*".
5. **Replicate over ≥5 seeds** before believing any arm ordering.
6. Only after the above: revisit the criticality direction, which no
   intervention here corrected.
