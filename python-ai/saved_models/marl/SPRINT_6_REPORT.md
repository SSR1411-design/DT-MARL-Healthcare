# Sprint 6 — MAPPO Baseline

Status: **empirical deliverables complete.** This is an additive reconstruction
from surviving repository evidence. No training or evaluation was rerun, and no
historical artifact was changed to create this report.

## 1. Objective

Sprint 6 established the first working MAPPO baseline for proactive workload
migration in the trace-driven DT-MARL environment. Its practical question was
whether a learned multi-agent policy using the predicted-failure-risk channel
could outperform simple scheduling baselines, especially the one-line
`risk-threshold` policy operating on that same risk channel.

This report records what the surviving Sprint 6 artifacts establish. It does
not attribute the later Sprint 7 advantage-estimator mechanism to Sprint 6
alone.

## 2. Evidence sources and execution chain

The reconstructed chain is:

```text
Java Digital Twin simulation
  -> simulation/failure_history.csv and simulation/failure_log.csv
  -> leakage-safe failure_predictor_oof.npz risk signal
  -> MAPPO training
  -> mappo checkpoint/config/history/update artifacts
  -> held-out evaluation
  -> sensitivity and counterfactual diagnostics
```

`simulation/src/main/java/com/dtmarl/Main.java` instantiates
`SimulationManager`, whose source exports `failure_log.csv` and
`failure_history.csv`. `simulation/failure_history.csv` contains 15,000 rows
(10 nodes x 1,500 ticks); `simulation/failure_log.csv` contains 174 event rows.
`SimulationManager.SIM_SEED` is `20260817L`. The exact historical Maven/shell
invocation that produced these files is not retained.

For MAPPO, the captured configuration names
`python-ai/saved_models/failure_predictor_oof.npz` as the risk input. The
runtime banner and configuration describe it as leakage-safe, out-of-fold,
uncalibrated sigmoid risk. `simulation/predicted_risk.csv` is an existing
Java/Python seam artifact, but is not the direct risk file named by the MAPPO
OOF configuration.

The surviving drivers are `marl/train.py`, `marl/evaluate.py`,
`marl/_diag_sensitivity.py`, and `marl/_diag_counterfactual.py`, with the
environment and policy support in `marl/{config,env,mappo,rollout,baseline,
risk_provider,trace,topology,criticality,destination}.py`. The current source
tree is not a separately retained pre-Sprint-6.5 source snapshot; later source
changes are therefore not treated as an original Sprint 6 snapshot here.

## 3. Experiment configuration

The following values are verified by `mappo_config.json`,
`run_mappo_train.log`, and the recorded evaluation:

| Setting | Verified value |
|---|---|
| Training seed | `20260818` |
| Device | `cpu` |
| Training episodes | 600 |
| Rollout episodes per PPO update | 8 |
| PPO updates | 75 |
| Episode length | 400 steps; 2 recorded ticks/step (800 s cluster time) |
| Discount factor | `gamma = 0.999` |
| GAE parameter | `lambda = 0.995` |
| Risk source | `oof`, uncalibrated |
| Training start window | ticks `[9, 491]` |
| Evaluation start window | ticks `[491, 698]` |
| Held-out starts | `[491, 521, 550, 580, 609, 639, 668, 698]` |
| Evaluation policy mode | greedy; exploration disabled |

The configuration also records 10 separate actors, four actions,
`cloud_slots=8`, actor hidden layers `[128, 128]`, critic hidden layers
`[256, 256]`, actor/critic learning rates `7e-4` / `1e-3`, four PPO epochs,
and four minibatches.

## 4. Artifact inventory

### Primary Sprint 6 run and evaluation

| Artifact | Contents / role |
|---|---|
| `python-ai/run_mappo_train.log` | Training banner, progress, saved-output list, and summary. |
| `python-ai/saved_models/marl/mappo.pth` | Final Sprint 6 MAPPO checkpoint. |
| `python-ai/saved_models/marl/mappo_best.pth` | Best-update checkpoint recorded during training. |
| `python-ai/saved_models/marl/mappo_config.json` | Full captured environment, reward, MAPPO, and training configuration. |
| `python-ai/saved_models/marl/mappo_history.csv` | 600 per-episode records. |
| `python-ai/saved_models/marl/mappo_updates.csv` | 75 per-update PPO records. |
| `python-ai/run_mappo_eval.log` | Held-out policy comparison, probes, ablation, and verdict. |
| `python-ai/saved_models/marl/mappo_eval.json` | Machine-readable held-out evaluation and probes. |
| `python-ai/saved_models/marl/mappo_eval.csv` | Tabular policy comparison. |

### Related Sprint 6 provenance and diagnostics

| Artifact group | Contents / status |
|---|---|
| `run_mappo_train_attrbug_gamma99.log` and `mappo_attrbug_gamma99.*` | Completed exploratory pre-arm run. |
| `run_mappo_train_attrbug_gamma95_killed.log`, `run_mappo_train_attrfix_gamma95_killed.log`, `run_mappo_train_g999_lowEV_killed.log`, `run_mappo_lam95_killed.log`, and their corresponding `mappo_*_killed*` artifacts | Preserved partial/killed exploratory runs; not the production baseline. |
| `diag_sensitivity.json` | Sensitivity diagnostic for `mappo.pth`; records six starts, 6,000 decision states, and 329 high-risk states. |
| `run_diag_counterfactual.log` and `diag_counterfactual.json` | Counterfactual replay evidence at the Sprint 6/Sprint 6.5 boundary. The project inventory catalogues it with the Stage 3 outputs, while the chronology labels the run as the first Sprint 6.5 probe; this report preserves that additive classification rather than resolving it. |

No standalone `SPRINT_6_REPORT.md` was found in the pre-existing working tree
or reachable Git history. `SPRINT_6_5_REPORT.md` is a later report and remains
unchanged.

## 5. Training result

`mappo_history.csv` has 600 records and `mappo_updates.csv` has 75 records.
The recorded first-20 mean reward is approximately `+0.02`; the recorded
last-20 mean is approximately `+14.55`. The training log reports best-update
mean reward `+24.44` and wall time `1460 s`.

### Historical inconsistency retained without reconciliation

Two surviving records disagree at episode 600:

| Record | Episode-600 reward |
|---|---:|
| `run_mappo_train.log` | `+16.94` |
| `mappo_history.csv` | `7.2551` |

The last-20 mean agrees at approximately `+14.55`. No artifact establishes
which episode-600 record should supersede the other. This is a historical
inconsistency, not a correction target for this report.

## 6. Held-out evaluation

The following means are from the eight fixed held-out starts in
`mappo_eval.json`. Field names follow the evaluation artifact: `protected`
means `tasks_protected_before_failure`, and `success rate` means
`task_success_rate`.

| Policy | Reward | Protected | Relocations | Lost | Success rate |
|---|---:|---:|---:|---:|---:|
| `mappo-greedy` | 20.6205 | 3.25 | 35.875 | 11.25 | 0.71875 |
| `risk-threshold@0.18` | 76.4963 | 30.25 | 36.25 | 3.375 | 0.846875 |

The MAPPO greedy action histogram is:

| Action | Count |
|---|---:|
| `STAY` | 9832 |
| `MIGRATE_TO_NEIGHBOR_EDGE` | 77 |
| `MIGRATE_TO_CLOUD` | 842 |
| `PREEMPTIVE_REROUTE` | 0 |

### Zero-risk ablation

With the same trained policy evaluated with its risk channel forced to zero,
the recorded reward is `23.1316` and protected count is `3.125`, compared with
risk-enabled `20.6205` and `3.25`. The evaluation log cautions that this
ablation changes the policy input, exposure reward term, and destination
selector simultaneously; reward is therefore not a directly isolating
comparison. The physical outcomes recorded here did not materially improve.

## 7. Interpretation and transition to Sprint 6.5

The narrow Sprint 6 conclusion is that MAPPO was reproducibly worse than the
`risk-threshold@0.18` baseline despite broadly comparable migration volume:
35.875 versus 36.25 relocations. The threshold rule protected 30.25 tasks
before failure versus MAPPO's 3.25 and had fewer lost tasks (3.375 versus
11.25).

This justified Sprint 6.5 because:

1. the threshold baseline substantially outperformed MAPPO;
2. migration volume alone was insufficient to explain the protection gap;
3. the zero-risk ablation did not materially improve the physical outcome; and
4. the learning/risk mechanism therefore required diagnosis.

Sprint 6 did not by itself establish the later Sprint 7 estimator mechanism.
Sprint 6.5 investigated the policy's risk sensitivity and related learning
dynamics; Sprint 7 subsequently tested later mechanistic hypotheses. Those
later reports are not revised or duplicated here.

## 8. Reproducibility and forensic status

The Sprint 6.5 A0 control later reproduced the retained Sprint 6 result: its
evaluation action histogram is the same `9832/77/842/0`, and its held-out
reward, protected count, and zero-risk-ablation protected count match the
original values recorded above. The original and A0 `mappo_history.csv` files
are byte-identical; the original and A0 `mappo_updates.csv` files are also
byte-identical.

No Sprint-6-specific integrity manifest exists. The later integrity directories
begin with Sprint 7 material. The following MD5 values are current observations
of retained artifacts, not historical preregistered digests and not a
standalone proof of historical provenance:

| Artifact | Current MD5 observation |
|---|---|
| `mappo.pth` | `F8C3A9C0DFCAA2B97C85B33D594A903F` |
| `mappo_best.pth` | `BE290F376378F519875996CF438BEE81` |
| `mappo_config.json` | `9702DB4D9C194CA4CD01254C2D2A91BC` |
| `mappo_history.csv` | `F378CFF4C9390365C21A3EF76D4D6CEC` |
| `mappo_updates.csv` | `5C33137CC961F2953E337C1F32A66D85` |
| `mappo_eval.json` | `AAA6FE586B0911141A997C3F90912CBF` |
| `mappo_eval.csv` | `8620A93384CB15F193A40765B0005F84` |

The exact historical shell command and Maven invocation are not retained. The
single retained seed does not justify a multi-seed claim.

## 9. Limitations and open historical items

- No original standalone Sprint 6 report was found.
- The exact original shell/Maven invocation is not retained.
- A dedicated historical source snapshot from before Sprint 6.5 changes is not retained.
- No Sprint-6-specific integrity manifest exists.
- The episode-600 training-log/CSV reward discrepancy is unresolved.
- Multi-seed behavior cannot be claimed from the retained single-seed result.

## 10. Relation to later sprints

Sprint 6 established the baseline failure against the risk-threshold rule.
Sprint 6.5 used A0 as its Sprint 6 reproduction control and investigated why
the policy was not meaningfully risk-sensitive. Sprint 7 then pursued the
later diagnostic ladder. This document records only the Sprint 6 baseline and
the reason for that transition; the later reports remain the authoritative
records for their own findings.

## 11. Final status

**Sprint 6 empirical deliverables are COMPLETE.** This report is an additive
reconstruction created from surviving evidence. No experiment was rerun to
create this report.
