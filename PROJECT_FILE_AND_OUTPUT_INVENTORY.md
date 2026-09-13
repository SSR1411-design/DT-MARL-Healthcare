# PROJECT FILE AND OUTPUT INVENTORY

**Single reference document for the DT-MARL-Healthcare repository.**
Purpose: let a future session understand the entire project — every sprint, rung, phase, arm,
diagnostic, artifact and checkpoint — without repeating the archaeology.

Built by read-only inspection of the repository on **2026-09-04**. Every path, filename, command
and count below was verified against the working tree. Anything that could not be verified is
marked **NOT VERIFIED FROM REPOSITORY** and is listed in §14.

Companion document: [PROJECT_RESEARCH_TUTORIAL.md](PROJECT_RESEARCH_TUTORIAL.md) — the 21-part
narrative handbook (~3,660 lines). This file is the *index*; that file is the *explanation*.
Where this file says "see Tutorial Part N", the detail is there and is deliberately not repeated.

Repository root: `D:\MY_GURL\RESEARCH_PAPER\DT-MARL-Healthcare`
Branch: `host-predictor-finalization` (7 commits; **no new branch is required**)

---

## Contents

| § | Section | What it gives you |
|---|---------|-------------------|
| 1 | Project overview | What this is, in one screen |
| 2 | Complete sprint / rung / phase history | Every stage that actually exists |
| 3 | Production code inventory | What each source file is for |
| 4 | Complete file / artifact inventory | **The main index** |
| 5 | Checkpoint inventory | Every `.pth`, its role, its valid comparison |
| 6 | Experiment configuration matrix | Arm × hyperparameter |
| 7 | Command cheat sheet | Verified commands, risk-labelled |
| 8 | "What do I open?" lookup table | Question → file |
| 9 | Experiment dependency map | What feeds what |
| 10 | Hashes / integrity | Protected vs regenerable |
| 11 | Scientific findings | Claim, evidence, type, status |
| 12 | Known gotchas / corrections | Wrong assumption → correct fact |
| 13 | Final project status | Done / not done / do not rerun |
| 14 | Future Claude quick start | Read these files, in this order |

---

# 1. PROJECT OVERVIEW

**What it is.** A two-part research system for *proactive* workload management in an
edge-cloud healthcare infrastructure:

1. **Java digital twin** (`simulation/`, CloudSim Plus 8.5.7, 44 `.java` under `com.dtmarl.*`) —
   simulates 10 edge hosts serving medical devices, injects host failures, and exports per-tick
   telemetry plus a predicted-risk CSV.
2. **Python AI stack** (`python-ai/`) — a BiLSTM **host failure predictor** (leakage-safe,
   26 features, `threshold=0.18`) and a **MAPPO multi-agent policy** (10 edge agents + cloud)
   trained by *trace-driven replay* of the exported telemetry.

**The research problem.** Can a multi-agent RL policy learn to use a failure-risk signal —
i.e. migrate critical workloads *before* a host fails — better than a trivial risk threshold
rule?

**The answer the repository actually reached: no, and the project spent Sprint 7 finding out
why.** Every MARL arm loses to a one-line risk-threshold baseline on held-out task performance,
and the final diagnosis is that the learned improvement in the primary metric is **differential
suppression of low-risk migration, not acquisition of high-risk migration**.

**Major experimental progression** (derived from the repository, not assumed):

```
Java sim + failure injection  →  host failure predictor (leakage-safe rebuild)
   →  Sprint 6 MAPPO baseline (loses to threshold rule)
   →  Sprint 6.5 four-arm ladder A0/A1/A2/A3 (isolate the cause)
   →  Sprint 7 Phase 1 diagnosis (hypothesis: converged, pointed the wrong way)
   →  Rung 0 → 1 → 2 (R2) → 2.5 → 2.75 → R3 → Rung 3   (mechanism ladder)
   →  Divergence study → Phase 0 recalibration → Phase 4 (reproducibility)
   →  Phase 5 (trajectory of the metric itself)  →  Final Synthesis Audit
```

**Final state.** Sprint 7 is **formally closed**.
`python-ai/saved_models/marl/SPRINT_7_FINAL_SYNTHESIS_AUDIT.md` (2026-08-31) closed it
**COMPLETE** with the verdict *"no further training justified"*, and it supersedes the
11-report chain. Nothing is mid-flight. No training is pending.

---

# 2. COMPLETE SPRINT / RUNG / PHASE HISTORY

Chronology derived from artifact mtimes, run-log ordering and report contents — **not** from
sprint numbers, which are inconsistent across the codebase (Java doc-comments name Sprints
1, 3, 4, 5, 6, 7, 8; Python names 3, 3.5, 3.75, 4, 5, 6, 6.5, 7; Sprints 8/9/11 are
unimplemented forward references). **Sprints 0, 1 and 2 do not exist as discrete named stages
in the Python stack.**

> **Chronology fact worth knowing:** commits jump from `208e5a2` (2026-07-29) to `a8df6d9`
> (2026-08-30). **All of Sprints 6, 6.5 and 7 happened inside that month-long gap, uncommitted.**
> That is the entire justification for the MD5 integrity methodology in §10.

---

## Stage 1 — Java digital twin + failure simulation

**Purpose.** Build the simulated infrastructure and generate the telemetry every later stage
consumes.

**Runs / configuration.** `SIM_SEED = 20260817L`, `MAX_SIMULATION_SECONDS = 1500.0`,
`DEVICE_COUNT = 10`. Exports at `SimulationManager.java:413-415`.

**Important commands.** Maven build/run of `simulation` (artifactId `simulation`, v1.0-SNAPSHOT).
*Exact Maven invocation:* **NOT VERIFIED FROM REPOSITORY** — no build script or documented
command exists.

**Files created.** 44 `.java` under `simulation/src/main/java/com/dtmarl/**`
(earliest `BrokerManager.java` 2026-07-20 08:29, latest `MigrationDemo.java` 2026-08-18 17:21).

**Outputs.** `simulation/predicted_risk.csv` (2026-08-18 16:02),
`simulation/device_failure_history.csv`, `simulation/failure_history.csv`,
`simulation/failure_log.csv` (all 2026-08-18 17:03).

**Main findings.**
- `DigitalTwinManager.mirrorNetworkLinks(int)` creates one *access link per node* — it is **not**
  a node-to-node graph. The MARL ring topology is a Python-side construct.
- `PredictionGateways.DEFAULT_RISK_CSV = simulation/predicted_risk.csv` is the hand-off point.

**Status.** COMPLETE. Frozen.
**Dependencies.** None (root of the chain).

---

## Stage 2 — Host failure predictor (leakage-safe rebuild)

**Purpose.** Produce a *leakage-free* risk signal. An earlier predictor was found degenerate;
the dataset was regenerated and the feature set restricted to genuinely observable columns.

**Runs.** Observable-feature run, legacy run (to reproduce the old defect), leave-one-node-out
(LONO) cross-validation, and an OLDDATA control.

**Important commands** *(reconstructed from scripts in `python-ai/training/`; not quoted
verbatim from any report)*:
- `python -m training.train_failure_predictor` → `run_host_observable.log`
- `python -m training.host_cv` → `run_lono_observable.log`
- `python -m training.evaluate_lead_time`, `python -m training.eval_leakage_audit`

**Files created.** `data/failure_dataset.py` (the leakage firewall: `OBSERVABLE_COLUMNS` 12,
`AUDIT_COLUMNS` 6, `FORBIDDEN_COLUMNS = AUDIT + willFailSoon, time, nodeId`,
`assert_no_leakage()`, `LEGACY_RAW_FEATURES` 7), `models/bilstm.py`,
`models/failure_predictor.py`.

**Outputs.** 15 artifacts in `saved_models/` (§4.2) — three predictor triplets
(`failure_predictor*`, `*_legacy*`, `*_OLDDATA*`), `host_lono.json`, two lead-time CSVs;
run logs 2026-08-17 17:33 → 18:32.

**Main findings.**
- BiLSTM, `sequence_length=10`, **26** features (12 observable + 12 `d_*` +
  `time_since_active` + `time_since_linkup`), `threshold=0.18`, 80 epochs, 5 temporal folds.
- The out-of-fold (OOF) risk used by MARL is **leakage-safe but UNCALIBRATED and near-bimodal**.
  Banner: `min=0.0145 max=0.9585 mean=0.0785 distinct=203 first_valid_tick=9`.
- Only **~4.7%** of MARL decision states carry `risk > 0.18`. This scarcity shapes every later
  result.

**Status.** COMPLETE. Frozen (production source last changed 2026-08-25 04:02).
**Dependencies.** Stage 1 telemetry.

---

## Stage 3 — Sprint 6 MAPPO baseline (+ exploratory pre-arm runs)

**Purpose.** First working MAPPO policy on the replay environment.

**Runs** (in log order, 2026-08-18):
| Log | Tag | Fate |
|---|---|---|
| `run_mappo_train_attrbug_gamma99.log` 15:05 | `attrbug_gamma99` | exploratory, kept |
| `..._attrbug_gamma95_killed.log` 15:14 | `attrbug_gamma95_killed` | killed |
| `..._attrfix_gamma95_killed.log` 15:50 | `attrfix_gamma95_killed` | killed |
| `..._g999_lowEV_killed.log` 16:04 | `g999_lowEV_killed` | killed |
| `run_mappo_lam95_killed.log` 16:27 | `lam95_killed` | killed |
| `run_mappo_train.log` 16:57 | *(untagged)* `mappo` | **the Sprint 6 production run** |
| `run_mappo_eval.log` 17:01 | — | its held-out evaluation |
| `run_diag_counterfactual.log` 23:31 | — | first counterfactual probe |

**Important commands.** `python -m marl.train --device cpu` and
`python -m marl.evaluate --model saved_models/marl/mappo_best.pth`
*(reconstructed from the CLI in `marl/train.py` / `marl/evaluate.py`)*.

**Files created.** `marl/` core production stack — `config.py` (576), `env.py` (1037),
`mappo.py` (593), `train.py` (294), `rollout.py` (114), `evaluate.py` (513), `baseline.py` (153),
`risk_provider.py` (198), `trace.py` (160), `topology.py` (58), `criticality.py` (185),
`destination.py` (88), `export_risk_csv.py` (137), `sanity_test.py`, `tests_env.py`.

**Outputs.** `mappo.pth`, `mappo_best.pth`, `mappo_config.json`, `mappo_history.csv`,
`mappo_updates.csv`, `mappo_eval.json`, `mappo_eval.csv`, plus the five killed runs' partial
CSVs/checkpoints; `diag_counterfactual.json`, `diag_sensitivity.json`.

**Main findings.**
- The trained policy is **beaten by a one-line risk-threshold rule** on held-out task
  performance. This is the finding that launched everything after it.
- The environment is a **trace replay**, not closed-loop co-simulation — the trajectory is
  exogenous (`env.py` docstring).

**Status.** SUPERSEDED as a result; PROTECTED as an artifact (it is A0's ancestor).
**Dependencies.** Stages 1–2.

---

## Stage 4 — Sprint 6.5: the four-arm ladder (A0 → A3)

**Purpose.** Isolate which of four suspected defects caused the Sprint 6 failure, by changing
**one thing per arm** from a shared initialisation (`torch.manual_seed(seed)` at `mappo.py:218`
means all arms share θ₀).

**Runs.**
| Arm | Manipulation | Logs (2026-08-19) |
|---|---|---|
| `A0_sprint6_repro` | none — reproduce Sprint 6 with legacy switches ON | 06:44 |
| `A0_CUDA_device_mismatch` | same on cuda | 06:46 (eval only) |
| `A1_CUDA_killed_at_90ep` | bugfix on cuda | 06:54 — killed at 90 ep |
| `A0_cpu_repro` | legacy ON, cpu | 07:23 / eval 07:27 |
| `A1_cpu_bugfix` | `legacy_*` switches OFF | 07:47 / eval 07:49 |
| `A2_crit_sign` | `w_criticality_migration 1.0 → 0.0` | 08:17 / eval 08:18 |
| `A3_entropy` | `entropy_coef 0.02 → 0.05` | 08:42 / eval 08:43 |

Chain drivers `chain_cpu.log` (06:58) and `chain_A2_A3.log` (07:52) are **0 bytes**.

**Important commands** *(reconstructed from `marl/train.py`'s CLI)*:
`python -m marl.train --device cpu --tag A0_cpu_repro --legacy-sprint6` and
`... --tag A2_crit_sign --w-criticality-migration 0.0`,
`... --tag A3_entropy --entropy-coef 0.05`.

**Files created.** `SPRINT_6_5_REPORT.md` (2026-08-19 08:48).

**Outputs.** Per arm: `mappo_<arm>.pth`, `_best.pth`, `_config.json`, `_history.csv`,
`_updates.csv`, `_eval.json`, `_eval.csv`.

**Main findings.**
- No single arm fixes the problem; **held-out task performance monotonically declines**
  A0 20.62 → A1 15.69 → A2 12.33 → A3 6.82.
- §13 recommended a **smaller** `rollout_episodes`. (R3 later tested a **larger** one — see §11.)
- The arm labels are misleading: **A2 "crit_sign" is a criticality-weight change, not a critic
  sign change**; A3's endpoints are 0.05 vs 0.02, not 0.01 vs 0.02 (see §12).

**Status.** COMPLETE; its causal question was answered later, at Sprint 7 Rung 0.
**Dependencies.** Stage 3.

---

## Stage 5 — Sprint 7 Phase 1: diagnosis

**Purpose.** Form a mechanistic hypothesis before spending any more compute. *"Inspection only.
No code modified, no training run, no artifact overwritten."*

**Runs.** None (static analysis of A0's rollouts).

**Files created.** `SPRINT_7_PHASE1_DIAGNOSIS.md` (2026-08-19 21:31).

**Main findings.**
- Verbatim hypothesis: *"PPO is converging correctly onto an advantage estimate whose action
  ordering at high risk is inverted relative to the truth. The dead gradient at update 60+ is
  what convergence looks like. Not a stalled optimiser — a converged one, pointed the wrong way."*
- At `risk > 0.18`: truth orders `EDGE > STAY > CLOUD` (EDGE `+2.671 ± 0.806`, n=4); the
  estimator orders `CLOUD > STAY > EDGE` (EDGE **−0.266**, CLOUD **+0.080**, n=9).
- Greedy A0 plays MIGRATE_TO_CLOUD in **842 / 855** states where cloud is legal (98.5%) and
  MIGRATE_TO_NEIGHBOR_EDGE in **77 / 10,751** (0.7%) — *"cloud whenever legal, else stay"*, a
  rule with no risk term in it.
- Self-caveated: n=4 / n=9. *"Establishing or refuting it is rung 0 of the ladder."*

**Status.** SUPERSEDED (the hypothesis was tested and partly falsified downstream).
**Dependencies.** Stage 4.
**Note.** The B1–B3 / C1–C4 / E1–E3 ladder proposed here **has no artifacts**; only "Rung 0"
survived by name (§12).

---

## Stage 6 — Sprint 7 Rung 0: diagnostics D1–D4

**Purpose.** Establish or refute Phase 1's inverted-ordering claim on a real sample.

**Runs.** `run_S7_rung0_d1d2_train.log` (2026-08-20 06:12), `run_S7_rung0_d3d4.log` (06:19).

**Important commands** *(reconstructed)*: `python -m marl._diag_S7_D1_advantage_fidelity`,
`..._D2_horizon_residual`, `..._D3_sample_census`, `..._D4_ppo_update`.

**Files created.** `SPRINT_7_RUNG0_REPORT.md` (2026-08-20 06:40), `_rung0_integrity/`.

**Outputs.** `diag_S7_D1_advantage_fidelity.json`, `diag_S7_D2_horizon_residual.json`,
`diag_S7_D3_sample_census.json`, `diag_S7_D4_ppo_update.json`.

**Main findings.**
- The advantage-sign inversion is **real for A0**, and it is traced to the λ-return critic
  target at `mappo.py:298`.
- D3/D4 **silently ignore `--episodes`** (§12).

**Status.** COMPLETE.
**Dependencies.** Stage 5.

---

## Stage 7 — Sprint 7 Rung 1: critic fit and residual geometry

**Purpose.** Characterise the critic's error structure rather than its magnitude.

**Runs.** `run_S7_rung1_fit.log` (2026-08-21 05:07), `run_S7_rung1_deviations.log` (05:31).

**Files created.** `SPRINT_7_RUNG1_REPORT.md` (2026-08-25 02:36), plus code
`_diag_rung1_*` scripts.

**Outputs.** `SPRINT_7_RUNG1_critic_fit_and_residual.json`,
`SPRINT_7_RUNG1_deviation_agreement.json`, `SPRINT_7_RUNG1_minibatch_tail_issue.json`.

**Main findings.**
- The sign error is a **per-state offset `c(s)`**, not a target defect. Decomposition:
  `gae(s,a) = c(s) + paired(s,a)` with `c(s) := gae(s, a_ref)`.
- The **paired advantage estimator** lifts high-risk action agreement from *below chance* to
  **0.71 / 0.95 for every arm**. `Var(raw)/Var(paired) = 3.72×` ⇒ `√3.72 = 1.93×` attenuation.
- Minibatch tail issue: `T mod 4 ≠ 0` produces **5** chunks, not 4 (§12).

**Status.** COMPLETE. This is the methodological high point of the project.
**Dependencies.** Stage 6.

---

## Stage 8 — Sprint 7 Rung 2 (arm `R2_mc_target`)

**Purpose.** Test the fix implied by Rung 0/1 — replace the λ-return critic target with a
Monte-Carlo target (`--critic-target mc`).

**Runs.** `run_R2_mc_target_train.log` (2026-08-25 03:37), `_eval.log` 03:45, `_d3_k3.log` 03:47,
`_best_eval.log` 03:49, `_d4_k3.log` 03:51.

**Important commands** *(reconstructed)*:
`python -m marl.train --critic-target mc --device cpu --tag R2_mc_target`.

**Files created.** No standalone `SPRINT_7_RUNG2_REPORT.md` exists (§12/§14) — Rung 2 is
reported inside `SPRINT_7_RUNG2_5_REPORT.md`.

**Outputs.** `mappo_R2_mc_target.pth`, `_best.pth`, `_config.json`, `_history.csv`,
`_updates.csv`, `_eval.json`, `_best_eval.json`, `_eval.csv`, `_best_eval.csv`;
`diag_S7_D3_sample_census_R2_mc_target.json`, `diag_S7_D4_ppo_update_R2_mc_target.json`.

**Main findings.**
- The critic-bias problem is **~87% fixed** — the intervention works on its own terms.
- Held-out task performance nevertheless **falls to −0.46**, the worst of any arm.
- `R2_best` ≡ `R2` (the `_best` gate at `train.py:232` never fired after the final update) (§12).

**Status.** COMPLETE; becomes the **control** for every later comparison.
**Dependencies.** Stage 7.

---

## Stage 9 — Sprint 7 Rung 2.5: the actor stall

**Purpose.** Explain why R2 fixed the critic but not the behaviour.

**Runs.** Six `SPRINT_7_RUNG2_5_*.log` (2026-08-25 05:54 → 06:39).

**Files created.** `SPRINT_7_RUNG2_5_REPORT.md` (2026-08-25 06:45), `_rung2_5_integrity/`.

**Outputs.** 7 JSONs — `_VERDICT`, `_actor_stall`, `_native_dev_R2`, `_paired_signtest`,
`_signtest`, `_signtest_data`, and 4 `_targets_*` files
(`A0_SAMPLE_CONTROL`, `R2_contGREEDY`, `R2_contSTAY`, `SMOKE`).

**Main findings.**
- The stall is **softmax saturation**, not a vanishing gradient: **50.5%** of R2's decision states
  sit at `max-prob > 0.99` against A0's **0.0000**. `clip_frac = 0` is *normal* here (§12).
- Rung 2.5 **partly downgraded** the critic-bias finding from cause to contributor.
- §G.5 leaves Rung 2's high-risk EDGE-share **direction formally UNRESOLVED**.

**Status.** COMPLETE, but its headline metric was withdrawn at Rung 2.75 §A.4 (§11).
**Dependencies.** Stage 8.

---

## Stage 10 — Sprint 7 Rung 2.75: the frozen state sets and Δ_EDGE

**Purpose.** Build a metric that can be measured identically on every arm, and stop the
metric-churn.

**Runs.** Eight `SPRINT_7_RUNG2_75_*.log` (2026-08-25 10:25 → 12:07, ending with
`_matched_states.log`).

**Files created.** `SPRINT_7_RUNG2_75_REPORT.md` (2026-08-25 13:08),
`_diag_rung2_75_matched_states.py`, `_diag_rung2_75_coherence.py`, `_rung2_75_integrity/`.

**Outputs.** 16 JSONs — `_matched_states_main` / `_R3`, `_coherence_main` / `_R3` /
`_R3_bs32probe`, `_edgeshare_main`, `_edgeshare_cluster_main` / `_production_main` /
`_rung2_5_main`, `_edgeshare_power_main`, `_stepcollapse_main` / `_R3` / `_allarms`,
`_plasticity_main`, `_offset_R2`, `_mbtail_main`.

**Main findings.**
- Defines the project's **primary metric**:
  `Δ_EDGE = π(EDGE | risk ≥ 0.6) − π(EDGE | risk < 0.2)`, evaluated over all decision entries
  of two **frozen** state sets — **RANDOM** (4,975 entries / 216 high-risk) and **UNION**
  (74,237 / ~3,592 high-risk) — plus `Spearman(risk, π(EDGE))`.
- §A.4 **withdraws** the Rung 2.5 edge-share metric.
- §7 H4 records the *correct* A3 manipulation (0.05 vs 0.02), contradicting Phase 0 §1.1 (§12).

**Status.** COMPLETE. The frozen sets are PROTECTED — every later number depends on them.
**Dependencies.** Stage 9.

---

## Stage 11 — Sprint 7 R3 (arm `R3_batch32`)

**Purpose.** Test the variance hypothesis: if the update is too noisy, a 4× batch should help.

**Runs.** `run_R3_batch32_train.log` (2026-08-25 17:38), then five `SPRINT_7_R3_*.log`
17:43 → 18:10 (ending `_action_channels.log`).

**Important command — the ONE fully verbatim command in the entire `.md` corpus:**
```bash
python -m marl.train --critic-target mc --rollout-episodes 32 --episodes 2400 --seed 20260818 --device cpu --tag R3_batch32
```

**Files created.** `SPRINT_7_R3_REPORT.md` (2026-08-25 18:22), `_diag_R3_action_channels.py`
(md5 `a25034e565c139b832e399cf6fed1f7c`), `_R3_integrity/`.

**Outputs.** `R3_batch32.pth`, `_best.pth`, `_config.json`, `_history.csv`, `_updates.csv`;
`SPRINT_7_R3_action_channels_R3.json`;
`diag_S7_D3_sample_census_R3_b32.json`, `diag_S7_D4_ppo_update_R3_b32.json`
(plus `*_R2_b32.json` probe counterparts).
**`R3` has NO held-out evaluation** — no `R3*eval*.json` and no `*eval*.log` exists.

**Main findings.**
- **Pre-registered outcome: NO-GO.** P1, P2, P3, P4 all fail.
- The 4× batch **raised gradient SNR (p = 0.0000)** yet made risk response **worse** — the
  variance hypothesis is **falsified**.
- Reward is explicitly *"not a pre-registered criterion"*.

**Status.** COMPLETE (NO-GO).
**Dependencies.** Stage 10 (uses its frozen sets).

---

## Stage 12 — Sprint 7 Rung 3: the dilution hypothesis

**Purpose.** Test R3's successor guess — that the low-risk bulk *dilutes* the high-risk signal.

**Runs.** Seven `SPRINT_7_RUNG3_*.log` (2026-08-27 12:27 → 13:01).

**Files created.** `SPRINT_7_RUNG3_PREREGISTRATION.md` (2026-08-27 12:07) — written **before**
the runs — and `SPRINT_7_RUNG3_REPORT.md` (13:16), `_RUNG3_integrity/`.

**Outputs.** `_dilution_main` / `_ep8` / `_SMOKE`, `_verdict_main` / `_ep8` / `_SMOKE`,
`_bootstrap_main` / `_SMOKE`.

**Main findings.**
- Dilution is **real** — the low-risk bulk sets the update direction in **9/9 cells** — but the
  mass ratios order the arms **opposite** to behaviour ⇒ **NO-GO**. Dilution does not explain R3.
- The reference direction **`synth` does not discriminate the action channel** — a limitation of
  the instrument, recorded rather than patched.
- `rollout_episodes` is a **pure variance intervention**: expected update content is invariant
  to batch size (≤ 0.105σ₈) and `sd` follows the finite-population law `sd(8)/sd(32) = 2.236`.
  The earlier *"5.1× collapse"* was a **+2.09σ outlier of the 8-episode instrument**.

**Status.** COMPLETE (NO-GO).
**Dependencies.** Stage 11.

---

## Stage 13 — Sprint 7 Divergence study

**Purpose.** Ask what *actually* differs between R2 and R3, having run out of hypotheses.

**Runs.** Nine `SPRINT_7_DIV_*.log` (2026-08-27 21:32 → 23:19).

**Files created.** `SPRINT_7_DIVERGENCE_REPORT.md` (2026-08-30 11:02), `_diag_div_critic.py`
(2026-08-27 23:15, the newest `marl/*.py`), `_DIVERGENCE_integrity/`.

**Outputs.** `SPRINT_7_DIV_content_main.json`, `_critic_main`, `_geometry_eval`,
`_geometry_full`, `_geometry_partA`, `_logs_R2vsR3`, `_shape_main`, `_variance_main`.

**Main findings.**
- **Regime-selective learning asymmetry** — the only difference surviving every ruling-out:
  R3 learned the **low-risk bulk** and *not* the high-risk minority, on two independent
  instruments. This is *"what differs"*, **not causal**.
- **Endpoint instruments invert against behaviour**: **20/20** readings of every Sprint 7
  mechanism metric order the arms **backwards** (mean ρ = −0.900). Endpoint measures headroom,
  not learning. **The blocker is a missing observable, not a missing hypothesis.**
- Mechanism 2 (per-state offset) is **closed**: R3 falsified it, the clip is too rarely active to
  make it a bias, and `c(s)` is agent-idiosyncratic so no online centring exists.

**Status.** COMPLETE. This is the stage that redirected Sprint 7 from endpoints to trajectories.
**Dependencies.** Stages 8, 11, 12.

---

## Stage 14 — Sprint 7 Phase 0: reconstruction and recalibration

**Purpose.** Rebuild the arm ledger from first principles before trusting any trajectory work.

**Files created.** `SPRINT_7_PHASE0_RECONSTRUCTION.md` (2026-08-30 11:59),
`marl/diag/_phase0_calibration.py` (11:58), `_PHASE0_integrity/`.

**Outputs.** `SPRINT_7_PHASE0_calibration.json`.

**Main findings.**
- Recalibrates every arm's Δ_EDGE on the frozen sets.
- §1.1 contains **two label errors** (A3 endpoints, A2 naming) — see §12. They were corrected in
  the Final Synthesis Audit, **not** by editing this report.
- §281 notes the trajectory driver must be **additive**, wrapping `MAPPOAgent.update` and
  `torch.save` — which is what Phase 4 then did.

**Status.** COMPLETE.
**Dependencies.** Stage 13.

---

## Stage 15 — Sprint 7 Phase 4 (+ 4.1, 4RUN): reproducibility and the R2 trajectory

**Purpose.** Prove R2 is bit-exactly reproducible, then capture its *entire* learning trajectory
so the metric can be watched over time rather than only at the endpoint.

**Runs.** `R2_traj_repro` — a storage-for-storage replication of R2 on CPU.

**Files created.** `SPRINT_7_PHASE4_PREREG.md` (2026-08-30 15:08) — registered **before** the run,
`SPRINT_7_P4_EQUIV_SELFTEST.txt` (15:05), `SPRINT_7_PHASE4_REPORT.md` (16:37);
`marl/diag/_phase4_r2_trajectory.py` (14:21), `_phase4_smoke.py` (14:36), `_phase4_verify.py`
(15:01), `_phase4_equiv.py` (15:05); `_PHASE4_integrity/`, `_PHASE41_integrity/`,
`_PHASE4RUN_integrity/`.

**Outputs.** `R2_traj_repro.pth`, `_best.pth`, `_config.json`, `_history.csv`, `_updates.csv`;
`SPRINT_7_P4_SMOKE_REPORT.json`; **`R2_trajectory/` — 76 checkpoints `u000`–`u075` plus
`SPRINT_7_P4_trajectory_manifest.jsonl`**.

**Main findings.**
- **R2 is bit-exactly reproducible.** B1a: 271/271 members, **265/265 tensor storages
  byte-identical with no normalisation**; only `data.pkl` and `.data/serialization_id` differ,
  and those differ *only* because of the tag string and the archive stem. Nondeterminism
  explains nothing; the 76 checkpoints belong to the real control run.
- `torch.save` byte-stream depends on the **file's stem** (member *names* only, 271 × 5 = 1,355
  bytes) and on the **tag string** (2 container members, 0 of 265 storages) — measured, not
  assumed (§12).
- §G.3 defers to *"Phase 6"*; the report that actually did the work is **Phase 5** (§12).
- **Phase 4.1 has no report** — it exists only as manifests in `_PHASE41_integrity/` (§14).

**Status.** COMPLETE.
**Dependencies.** Stage 14.

---

## Stage 16 — Sprint 7 Phase 5: the risk trajectory

**Purpose.** Use the 76 checkpoints to ask *when* and *how* Δ_EDGE grew.

**Runs.** `SPRINT_7_PHASE5_run.log` (2026-08-31 02:51) — the only `.log` inside
`saved_models/marl/`.

**Files created.** `SPRINT_7_PHASE5_REPORT.md` (2026-08-31 03:01),
`marl/diag/_phase5_risk_trajectory.py` (02:45), `_PHASE5_integrity/`.

**Outputs.** `SPRINT_7_PHASE5_risk_trajectory_main.json`, `..._parity.json`.

**Main findings — the terminal finding of the project.**
- **Δ_EDGE is differential suppression, not risk acquisition.** `π(EDGE | high risk)` ends
  **BELOW random initialisation** and **never exceeds it in 76/76 checkpoints**. **159–218%** of
  the metric's growth is *low-risk suppression*.
- **R2's policy freezes at update 62** — the actor stops moving with ~19% of training left while
  the critic is still improving fastest. This rules out *"the critic gated the actor"* and dates
  the softmax collapse to **u1–u40**.
- §4.4 claims `SPRINT_7_R3_action_channels.json` does not exist — **wrong**, the file is
  `SPRINT_7_R3_action_channels_R3.json` (§12).

**Status.** COMPLETE. **Sprint 7 stopped here.**
**Dependencies.** Stage 15.

---

## Stage 17 — Sprint 7 Final Synthesis Audit

**Purpose.** Audit the whole 11-report chain, record every internal contradiction, and close the
sprint.

**Files created.** `SPRINT_7_FINAL_SYNTHESIS_AUDIT.md` (2026-08-31 04:39), `_AUDIT_integrity/`.

**Main findings.**
- Closes Sprint 7 **COMPLETE**, verdict **"no further training justified"**.
- **Supersedes the 11-report chain.** Read this first for anything Sprint 7.
- §8 is the corrections table — e.g. row 17 fixes Phase 5 §4.4's missing-file claim **without
  editing the report**. This is the project's convention: *corrections are additive*.

**Status.** COMPLETE. **This is the final source of truth.**
**Dependencies.** Stages 5–16.

---

# 3. PRODUCTION CODE INVENTORY

Four categories, kept separate as required: **production**, **diagnostic**, **experiment
harness**, **analysis/self-test**. No source is copied; functions are described.

## 3.1 Production code — `python-ai/marl/` (the frozen experimental apparatus)

| File | Lines | Purpose | Important classes/functions | Used by | Experimental significance | Protected? |
|---|---|---|---|---|---|---|
| `config.py` | 576 | Single source of every hyperparameter; dataclass tree `env / reward / mappo / train` | `Config`, the `mappo` block (frozen values below) | everything | **Any edit invalidates every arm comparison.** `:434` carries the *correct* GAE-horizon note | **YES** |
| `env.py` | 1037 | `DTMarlEnv` — trace-driven replay environment | `DTMarlEnv`, `describe()`, masking/legality at `:405-422`, `:565`, `:620-626`, `:630`, `:934` | `train.py`, `evaluate.py`, all diagnostics | Defines obs 48/agent, state 489 (→499 with agent one-hot), 4 actions, 400-step episodes. Docstring states it is **not** closed-loop co-simulation | **YES** |
| `mappo.py` | 593 | MAPPO agent: 10 separate actors + 1 centralised critic (CTDE) | `MAPPOAgent`, `update()`, `masked_dist` `:77-79`, `torch.manual_seed` `:218`, λ-return target `:298`, `save` `:475-489`, `assert_actors_independent` `:514` | `train.py`, all diagnostics | `:298` is the object of Rungs 0–2. `:218` is why all arms share θ₀. `:447` has a **deliberately unfixed stale comment** | **YES** |
| `train.py` | 294 | Training driver + CLI | `main()`; `_best` save gate `:232`; final save `:252`; independence print `:129` | every arm | Its CLI *is* the experiment interface (§7) | **YES** |
| `rollout.py` | 114 | Episode collection into the PPO buffer | rollout loop | `train.py` | Buffer `T` determines the minibatch tail (§12) | **YES** |
| `evaluate.py` | 513 | Held-out evaluation + ablations | `main()`, ablation runners | post-hoc | Produces the 8-start held-out suite used for the task-performance numbers | **YES** |
| `baseline.py` | 153 | Non-learned policies incl. the risk-threshold rule | baseline policies | `evaluate.py` | **The rule that beats every MARL arm** | **YES** |
| `risk_provider.py` | 198 | Supplies risk per (node, tick); `oof` / `model` / `zero` | risk lookup, banner printer | `env.py` | md5 `02219832596a38e9800ba4f89659a555` recorded in R3's manifest | **YES** |
| `trace.py` | 160 | Loads/indexes the exported telemetry trace | trace loader | `env.py` | Fixes the 10-node × 1500-tick window and the train/eval split | **YES** |
| `topology.py` | 58 | Ring neighbour graph, `neighbour_offsets=[-2,-1,1,2]` | neighbour lookup | `env.py` | Degree-4 ring is a Python construct, **not** the Java link graph | **YES** |
| `criticality.py` | 185 | Workload criticality weighting | criticality scoring | `env.py`, reward | The channel A2 zeroed (`w_criticality_migration`) | **YES** |
| `destination.py` | 88 | Chooses a migration target host | destination selection | `env.py` | Makes MIGRATE_EDGE a single action rather than per-neighbour | **YES** |
| `export_risk_csv.py` | 137 | Writes the OOF risk CSV for the Java side | exporter | Java hand-off | Bridges Stage 2 → Stage 1's `predicted_risk.csv` | YES |
| `__init__.py` | — | package marker | — | — | — | YES |

**Frozen production hyperparameters** — all live at `config.mappo`, **not** `config.train` (§12):
`gamma 0.999` · `gae_lambda 0.995` (horizon 166.8) · `clip_eps 0.2` · `value_clip_eps 0.2` ·
`entropy_coef 0.02` · `value_coef 0.5` · `max_grad_norm 0.5` · `ppo_epochs 4` · `minibatches 4` ·
`normalise_advantages True` · `critic_target "mc"` · `anneal_lr True` · `separate_actors True` ·
`episodes 600` · `rollout_episodes 8` · `seed 20260818` · `lr_actor 7e-4` · `lr_critic 1e-3`.
Derived: `n_updates = 600 // 8 = 75`; `lr_scale 1.0000 → 0.0133`; final `lr_actor 9.3333e-06`,
`lr_critic 1.3333333333333308e-05`; 1,392 Adam steps.
Network sizes: actor MLP(48 → [128,128] → 4) = **23,300** params each; critic
MLP(499 → [256,256] → 1) = **194,049**.

## 3.2 Analysis / self-test code

| File | Purpose | Important functions | Significance | Protected? |
|---|---|---|---|---|
| `marl/tests_env.py` | 19-test environment/algorithm suite `t1`–`t19` | `t7 no_future_information` `:237`, `t8 no_forbidden_columns` `:293`, `t15 discount_preserves_policy_ranking` `:433`, `t16 gae_horizon_covers_task_lifetime` `:504`, `t17 contention_is_not_decided_by_agent_index` `:558`, `t19 legacy_switches_reproduce_sprint6_defects` `:680` | The leakage and legacy-reproduction guarantees rest on t7/t8/t19 | **YES** |
| `marl/sanity_test.py` | 6 fast probes | probe functions | Quick "is the repo intact" check | **YES** |
| `data/failure_dataset.py` | Predictor dataset + **leakage firewall** | `OBSERVABLE_COLUMNS` (12), `AUDIT_COLUMNS` (6), `FORBIDDEN_COLUMNS`, `assert_no_leakage()`, `LEGACY_RAW_FEATURES` (7) | The one file that makes the risk signal defensible | **YES** |

## 3.3 Diagnostic code

- **32** `_diag_*` / `_smoke_*` scripts at `python-ai/marl/` top level
  (oldest context `criticality.py` 2026-08-18 13:56; newest diagnostic `_diag_div_critic.py`
  2026-08-27 23:15).
- **6** scripts in `python-ai/marl/diag/`:
  `_phase0_calibration.py` (08-30 11:58), `_phase4_r2_trajectory.py` (14:21),
  `_phase4_smoke.py` (14:36), `_phase4_verify.py` (15:01), `_phase4_equiv.py` (15:05),
  `_phase5_risk_trajectory.py` (08-31 02:45).
- Key ones by stage: `_diag_S7_D1..D4` (Rung 0) · `_diag_rung1_*` (Rung 1) ·
  `_diag_rung2_75_matched_states.py` + `_diag_rung2_75_coherence.py` (Rung 2.75 — **these define
  the frozen RANDOM/UNION sets**) · `_diag_R3_action_channels.py` (R3) ·
  `_diag_rung3_*` (Rung 3) · `_diag_div_*` (Divergence) · `diag/_phase*` (Phases 0/4/5).
- **Diagnostic code is additive by convention.** R3 wrote a *new* file rather than editing
  `_diag_rung2_75_matched_states.py` "precisely so the script that produced the pre-registered
  P1/P2/P3 numbers still exists in the form that produced them."

## 3.4 Experiment harnesses / drivers

| File | Purpose | Note |
|---|---|---|
| `chain_cpu.log`, `chain_A2_A3.log` | shell chain drivers for Sprint 6.5 | **0 bytes** — the chaining shell script itself is **NOT VERIFIED FROM REPOSITORY** |
| `python-ai/scratch_*.py` (5 files) | ad-hoc scratch | not part of any result |
| `python-ai/config.py`, `python-ai/main.py` | **0 bytes** | dead placeholders |
| `temp.py`, `da` (2,658 B) at repo root | stray | not referenced by anything |

## 3.5 Predictor pipeline code

`data/`: `dataset.py`, `device_failure_dataset.py`, `explore.py`, `failure_dataset.py`,
`preprocess.py`, `preprocessing/`.
`models/`: `bilstm.py`, `failure_predictor.py`, `htcf.py`, `transformer.py`.
`training/`: `train.py`, `train_failure_predictor.py`, `train_device_failure_predictor.py`,
`host_cv.py`, `evaluate_lead_time.py`, `evaluate_device_lead_time.py`, `eval_compare.py`,
`eval_leakage_audit.py`.
**Empty directories:** `docs/`, `results/`, `datasets/cluster/`, `python-ai/inference/`,
`python-ai/utils/`. **There is no `README.md` anywhere in the repository.**

---

# 4. COMPLETE FILE / ARTIFACT INVENTORY

Grouped by stage. All paths relative to the repository root. Cache/build files omitted.

**Totals in `python-ai/saved_models/marl/`:** 25 root `.pth` · 76 `R2_trajectory/*.pth` ·
1 `.jsonl` · 38 `.csv` · 81 `.json` · 15 `.md` + 1 `.txt` · 12 integrity directories.

## 4.1 Stage 1 — Java / simulation outputs

| Path | Type | Stage | Purpose | I/O | Important contents | Regenerable? | Protected? |
|---|---|---|---|---|---|---|---|
| `simulation/src/main/java/com/dtmarl/**` (44 files) | source | 1 | digital twin | — | `SIM_SEED 20260817L`, exports `SimulationManager.java:413-415` | — | YES |
| `simulation/predicted_risk.csv` | data | 1↔2 | risk hand-off to Java | output of `export_risk_csv.py` | `PredictionGateways.DEFAULT_RISK_CSV` | yes | YES |
| `simulation/device_failure_history.csv` | data | 1 | device failure ledger | sim output | 2026-08-18 17:03 | yes (re-sim) | YES |
| `simulation/failure_history.csv` | data | 1 | host failure ledger | sim output | 2026-08-18 17:03 | yes (re-sim) | YES |
| `simulation/failure_log.csv` | data | 1 | raw failure events | sim output | 2026-08-18 17:03 | yes (re-sim) | YES |

## 4.2 Stage 2 — predictor artifacts (`python-ai/saved_models/`)

| Path | Type | Purpose | Important contents | mtime | Regenerable? | Protected? |
|---|---|---|---|---|---|---|
| `failure_predictor.pth` | weights | **production** host predictor | BiLSTM, 26 feat, seq 10 | 08-17 17:33 | yes (expensive) | **YES** |
| `failure_predictor_meta.json` | metadata | thresholds/features | `threshold=0.18` | 08-17 17:33 | yes | **YES** |
| `host_metrics.json` | metrics | production fold metrics | 5 temporal folds | 08-17 17:33 | yes | **YES** |
| `failure_predictor_legacy.pth` / `_legacy_meta.json` / `host_metrics_legacy.json` | weights+meta | legacy 7-feature control (reproduces the old defect) | `LEGACY_RAW_FEATURES` | 08-17 17:43 | yes | YES |
| `failure_predictor_OLDDATA.pth` / `_OLDDATA_meta.json` / `host_metrics_OLDDATA.json` | weights+meta | pre-regeneration dataset control | — | 08-17 18:26 | no (old dataset) | **YES** |
| `host_lono.json` | metrics | leave-one-node-out CV | generalisation across hosts | 08-17 18:19 | yes | YES |
| `host_lead_time.csv` / `host_lead_time_OLDDATA.csv` | metrics | warning lead time | — | 08-17 18:32 | yes | YES |
| `host_lead_time_legacy.csv` | metrics | legacy lead time | — | 08-17 17:53 | yes | YES |
| `device_failure_predictor.pth` | weights | device-level predictor (unused by MARL) | — | 07-26 08:32 | yes | YES |
| `htcf_model.pth` | weights | HTCF baseline | — | 07-21 03:19 | yes | YES |
| `_backup_pre_finalization/` | dir | pre-finalization snapshot | — | — | no | **YES** |

## 4.3 Stage 3 — Sprint 6 baseline (`python-ai/saved_models/marl/`)

| Path | Type | Purpose | Notes | Regenerable? | Protected? |
|---|---|---|---|---|---|
| `mappo.pth`, `mappo_best.pth` | ckpt | Sprint 6 production policy | A0's ancestor | in principle | **YES** |
| `mappo_config.json` | config | its full config snapshot | `legacy_*` keys **absent** | no | **YES** |
| `mappo_history.csv`, `mappo_updates.csv` | log | per-episode / per-update | schemas in §4.10 | no | **YES** |
| `mappo_eval.json`, `mappo_eval.csv` | eval | held-out suite | 8 fixed starts | yes | **YES** |
| `mappo_attrbug_gamma99.pth`, `_best.pth`, `_config.json`, `_history.csv`, `_updates.csv` | run | exploratory pre-arm run | `gamma 0.99`, `gae_lambda 0.95`, `lr_actor 3e-4`, `entropy_coef 0.01`, `rollout_episodes 4`, `episodes 400`, wall 938 s | no | YES |
| `mappo_attrbug_gamma95_killed_best.pth`, `..._history.csv`, `..._updates.csv` | run | killed run | partial | no | YES |
| `mappo_attrfix_gamma95_killed_best.pth`, `..._history.csv`, `..._updates.csv` | run | killed run | partial | no | YES |
| `mappo_g999_lowEV_killed_best.pth`, `..._history.csv`, `..._updates.csv` | run | killed run | partial | no | YES |
| `mappo_best_lam95_killed.pth`, `mappo_history_lam95_killed.csv`, `mappo_updates_lam95_killed.csv` | run | killed run (note the **suffix-after-stem** naming) | partial | no | YES |
| `diag_counterfactual.json`, `diag_counterfactual_A1.json`, `diag_sensitivity.json` | diag | first counterfactual/sensitivity probes | — | yes | additive |

## 4.4 Stage 4 — Sprint 6.5 arms A0–A3

| Path pattern | Type | Purpose | Regenerable? | Protected? |
|---|---|---|---|---|
| `mappo_A0_cpu_repro{,_best}.pth`, `_config.json`, `_history.csv`, `_updates.csv`, `_eval.json`, `_eval.csv` | arm A0 | the reproduced Sprint 6 control (legacy switches ON) | bit-exactly, on cpu | **YES** |
| `mappo_A0_CUDA_device_mismatch{,_best}.pth`, `_config.json`, `_history.csv`, `_updates.csv`, `_eval.json`, `_eval.csv` | arm | cuda counterpart — kept as **evidence that cuda breaks reproducibility** | no | **YES** |
| `mappo_A1_cpu_bugfix{,_best}.pth`, `_config.json`, `_history.csv`, `_updates.csv`, `_eval.json`, `_eval.csv` | arm A1 | `legacy_*` OFF | yes | **YES** |
| `mappo_A1_CUDA_killed_at_90ep_best.pth`, `_history.csv`, `_updates.csv` | arm | killed at 90 ep; **no `_config.json`, no eval** | no | YES |
| `mappo_A2_crit_sign{,_best}.pth`, `_config.json`, `_history.csv`, `_updates.csv`, `_eval.json`, `_eval.csv` | arm A2 | `w_criticality_migration 1.0 → 0.0` (**not** a critic-sign change) | yes | **YES** |
| `mappo_A3_entropy{,_best}.pth`, `_config.json`, `_history.csv`, `_updates.csv`, `_eval.json`, `_eval.csv` | arm A3 | `entropy_coef 0.02 → 0.05` | yes | **YES** |
| `SPRINT_6_5_REPORT.md` | report | the ladder's findings; §13 recommends a *smaller* `rollout_episodes` | no | **YES** |

## 4.5 Stage 5–7 — Phase 1, Rung 0, Rung 1

| Path | Type | Stage | Purpose | Regenerable? | Protected? |
|---|---|---|---|---|---|
| `SPRINT_7_PHASE1_DIAGNOSIS.md` | report | Phase 1 | the inverted-ordering hypothesis | no | **YES** |
| `SPRINT_7_RUNG0_REPORT.md` | report | Rung 0 | D1–D4 verdicts | no | **YES** |
| `diag_S7_D1_advantage_fidelity.json` | diag | Rung 0 | advantage truth-vs-estimate | yes | additive |
| `diag_S7_D2_horizon_residual.json` | diag | Rung 0 | horizon residual | yes | additive |
| `diag_S7_D3_sample_census.json` | diag | Rung 0 | state census (A0) — **ignores `--episodes`** | yes | additive |
| `diag_S7_D4_ppo_update.json` | diag | Rung 0 | PPO update anatomy (A0) — **ignores `--episodes`** | yes | additive |
| `_rung0_integrity/` | manifests | Rung 0 | before/after md5 | no | **YES** |
| `SPRINT_7_RUNG1_REPORT.md` | report | Rung 1 | per-state offset + paired estimator | no | **YES** |
| `SPRINT_7_RUNG1_critic_fit_and_residual.json` | diag | Rung 1 | critic residual geometry | yes | additive |
| `SPRINT_7_RUNG1_deviation_agreement.json` | diag | Rung 1 | **0.71 / 0.95** paired agreement | yes | additive |
| `SPRINT_7_RUNG1_minibatch_tail_issue.json` | diag | Rung 1 | the `T mod 4` 5-chunk finding | yes | additive |

## 4.6 Stage 8–10 — R2, Rung 2.5, Rung 2.75

| Path | Type | Stage | Purpose | Regenerable? | Protected? |
|---|---|---|---|---|---|
| `mappo_R2_mc_target{,_best}.pth` | ckpt | R2 | **the control arm** (`_best` ≡ final) | bit-exactly | **YES — highest** |
| `mappo_R2_mc_target_config.json`, `_history.csv`, `_updates.csv` | run | R2 | provenance | no | **YES** |
| `mappo_R2_mc_target_eval.json` / `_eval.csv`, `_best_eval.json` / `_best_eval.csv` | eval | R2 | held-out (task perf **−0.46**) | yes | **YES** |
| `diag_S7_D3_sample_census_R2_mc_target.json`, `diag_S7_D4_ppo_update_R2_mc_target.json` | diag | R2 | census/update for R2 | yes | additive |
| `diag_S7_D3_sample_census_R2_b32.json`, `diag_S7_D4_ppo_update_R2_b32.json` | diag | R3 probe | R2 measured *under the 32-episode instrument* — **not a second R2 run** | yes | additive |
| `SPRINT_7_RUNG2_5_REPORT.md` | report | 2.5 | actor stall; **also contains Rung 2's report** | no | **YES** |
| `SPRINT_7_RUNG2_5_VERDICT.json` | diag | 2.5 | verdict | yes | additive |
| `SPRINT_7_RUNG2_5_actor_stall.json` | diag | 2.5 | **50.5% at max-prob > 0.99** vs A0's 0.0000 | yes | additive |
| `SPRINT_7_RUNG2_5_native_dev_R2.json` | diag | 2.5 | native deviations | yes | additive |
| `SPRINT_7_RUNG2_5_signtest.json` / `_paired_signtest.json` / `_signtest_data.json` | diag | 2.5 | raw vs paired sign tests | yes | additive |
| `SPRINT_7_RUNG2_5_targets_A0_SAMPLE_CONTROL.json` | diag | 2.5 | A0 sampling control | yes | additive |
| `SPRINT_7_RUNG2_5_targets_R2_contGREEDY.json` / `_contSTAY.json` | diag | 2.5 | continuation-policy variants | yes | additive |
| `SPRINT_7_RUNG2_5_targets_SMOKE.json` | diag | 2.5 | smoke run (**not a result**) | yes | additive |
| `SPRINT_7_RUNG2_75_REPORT.md` | report | 2.75 | defines Δ_EDGE; §A.4 withdraws the 2.5 metric | no | **YES** |
| `SPRINT_7_RUNG2_75_matched_states_main.json` | diag | 2.75 | **the frozen RANDOM + UNION sets** | yes, but must not change | **YES** |
| `SPRINT_7_RUNG2_75_matched_states_R3.json` | diag | R3 | R3 scored on the frozen sets | yes | additive |
| `SPRINT_7_RUNG2_75_coherence_main.json` / `_R3.json` / `_R3_bs32probe.json` | diag | 2.75/R3 | gradient coherence | yes | additive |
| `SPRINT_7_RUNG2_75_edgeshare_main.json` | diag | 2.75 | edge-share point estimate | yes | additive |
| `SPRINT_7_RUNG2_75_edgeshare_cluster_main.json` / `_production_main.json` / `_rung2_5_main.json` | diag | 2.75 | cluster-bootstrap CIs (clusters = episodes) | yes | additive |
| `SPRINT_7_RUNG2_75_edgeshare_power_main.json` | diag | 2.75 | power analysis | yes | additive |
| `SPRINT_7_RUNG2_75_stepcollapse_main.json` / `_R3.json` / `_allarms.json` | diag | 2.75 | step-size collapse | yes | additive |
| `SPRINT_7_RUNG2_75_plasticity_main.json` | diag | 2.75 | plasticity | yes | additive |
| `SPRINT_7_RUNG2_75_offset_R2.json` | diag | 2.75 | per-state offset for R2 | yes | additive |
| `SPRINT_7_RUNG2_75_mbtail_main.json` | diag | 2.75 | minibatch-tail follow-up | yes | additive |
| `_rung2_5_integrity/`, `_rung2_75_integrity/` | manifests | 2.5/2.75 | before/after md5 | no | **YES** |

## 4.7 Stage 11–12 — R3 and Rung 3

| Path | Type | Stage | Purpose | Regenerable? | Protected? |
|---|---|---|---|---|---|
| `R3_batch32.pth`, `R3_batch32_best.pth` | ckpt | R3 | the 32-episode treatment. **`R3_best` = update 45** (not the final) | yes (verbatim command, §7) | **YES** |
| `R3_batch32_config.json`, `_history.csv`, `_updates.csv` | run | R3 | provenance | no | **YES** |
| *(no `R3*eval*`)* | — | R3 | **R3 was never evaluated on the held-out suite** | — | — |
| `SPRINT_7_R3_REPORT.md` | report | R3 | NO-GO on P1–P4 | no | **YES** |
| `SPRINT_7_R3_action_channels_R3.json` | diag | R3 | per-action-channel Δ (note the `_R3` **tag suffix**) | yes | additive |
| `diag_S7_D3_sample_census_R3_b32.json`, `diag_S7_D4_ppo_update_R3_b32.json` | diag | R3 | census/update for R3 | yes | additive |
| `_R3_integrity/` | manifests | R3 | before/after md5 | no | **YES** |
| `SPRINT_7_RUNG3_PREREGISTRATION.md` | prereg | Rung 3 | written **before** the runs (12:07 vs 12:27) | no | **YES** |
| `SPRINT_7_RUNG3_REPORT.md` | report | Rung 3 | dilution real but does not explain R3 → NO-GO | no | **YES** |
| `SPRINT_7_RUNG3_dilution_main.json` / `_ep8.json` / `_SMOKE.json` | diag | Rung 3 | **`_main` (32 ep) is THE pre-registered result**; `_ep8` is supplementary; `_SMOKE` is not a result | yes | additive |
| `SPRINT_7_RUNG3_verdict_main.json` / `_ep8.json` / `_SMOKE.json` | diag | Rung 3 | GO/NO-GO evaluation | yes | additive |
| `SPRINT_7_RUNG3_bootstrap_main.json` / `_SMOKE.json` | diag | Rung 3 | finite-population / bootstrap check | yes | additive |
| `_RUNG3_integrity/` | manifests | Rung 3 | before/after md5 | no | **YES** |

## 4.8 Stage 13–14 — Divergence and Phase 0

| Path | Type | Purpose | Regenerable? | Protected? |
|---|---|---|---|---|
| `SPRINT_7_DIVERGENCE_REPORT.md` | report | regime-selective asymmetry; the 20/20 inversion; closes Mechanism 2 | no | **YES** |
| `SPRINT_7_DIV_content_main.json` | diag | update-content decomposition `g_full = g_hi + g_lo` | yes | additive |
| `SPRINT_7_DIV_critic_main.json` | diag | critic-side differential test | yes | additive |
| `SPRINT_7_DIV_geometry_partA.json` / `_full.json` / `_eval.json` | diag | gradient geometry (the `_eval` one is the only `*_eval*` file that is **not** a policy evaluation) | yes | additive |
| `SPRINT_7_DIV_logs_R2vsR3.json` | diag | direct R2-vs-R3 training-log comparison | yes | additive |
| `SPRINT_7_DIV_shape_main.json` | diag | distribution shape | yes | additive |
| `SPRINT_7_DIV_variance_main.json` | diag | variance decomposition | yes | additive |
| `_DIVERGENCE_integrity/` | manifests | before/after md5 | no | **YES** |
| `SPRINT_7_PHASE0_RECONSTRUCTION.md` | report | arm-ledger reconstruction; **contains 2 label errors** (§12) | no | **YES** |
| `SPRINT_7_PHASE0_calibration.json` | diag | recalibrated Δ_EDGE per arm | yes | additive |
| `_PHASE0_integrity/` | manifests | before/after md5 | no | **YES** |

## 4.9 Stage 15–17 — Phase 4 / 4.1 / 4RUN, Phase 5, Audit

| Path | Type | Purpose | Regenerable? | Protected? |
|---|---|---|---|---|
| `SPRINT_7_PHASE4_PREREG.md` | prereg | B1a–B1e criteria, the N1/N2 blast-radius measurement, F1–F9 failure modes | no | **YES** |
| `SPRINT_7_P4_EQUIV_SELFTEST.txt` | self-test | equivalence self-test output (15:05) | yes | **YES** |
| `SPRINT_7_PHASE4_REPORT.md` | report | 271/271 members, 265/265 storages byte-identical | no | **YES** |
| `SPRINT_7_P4_SMOKE_REPORT.json` | diag | smoke run of the trajectory driver | yes | additive |
| `R2_traj_repro.pth`, `_best.pth` | ckpt | the bit-exact replication of R2 | yes | **YES** |
| `R2_traj_repro_config.json`, `_history.csv`, `_updates.csv` | run | provenance | no | **YES** |
| `R2_trajectory/u000.pth` … `u075.pth` (**76**) | ckpt series | R2's full learning trajectory | only by rerunning Phase 4 | **YES — do not touch** |
| `R2_trajectory/SPRINT_7_P4_trajectory_manifest.jsonl` | manifest | per-checkpoint provenance | no | **YES** |
| `_PHASE4_integrity/`, `_PHASE41_integrity/`, `_PHASE4RUN_integrity/` | manifests | before/after md5 (**4.1 has manifests but no report**) | no | **YES** |
| `SPRINT_7_PHASE5_REPORT.md` | report | Δ_EDGE = differential suppression; freeze at update 62 | no | **YES** |
| `SPRINT_7_PHASE5_run.log` | log | the only `.log` inside `saved_models/marl/` | no | **YES** |
| `SPRINT_7_PHASE5_risk_trajectory_main.json` | diag | the 76-checkpoint risk trajectory | yes | additive |
| `SPRINT_7_PHASE5_risk_trajectory_parity.json` | diag | parity/consistency check | yes | additive |
| `_PHASE5_integrity/` | manifests | 6 files: `SPRINT_7_P5_{artifacts,code,inputs,trajectory}_{before,after}.md5` + `_prior_manifests.md5` + `_own_manifests.md5` (counts 156/160, 52/53, 3/3, 78/78, 69, 10) | no | **YES** |
| `SPRINT_7_FINAL_SYNTHESIS_AUDIT.md` | report | **closes Sprint 7; the final source of truth**; §8 is the corrections table | no | **YES — read first** |
| `_AUDIT_integrity/` | manifests | before/after md5 | no | **YES** |

## 4.10 Run logs and artifact schemas

- **69 `.log` files at `python-ai/` root** — this is what dates every experiment. Chronology:
  predictor 08-17 17:33→18:32 · Sprint 6 08-18 15:05→23:31 · Sprint 6.5 08-19 06:44→08:43 ·
  Rung 0 08-20 06:12→06:19 · Rung 1 08-21 05:07→05:31 · R2 08-25 03:37→03:51 ·
  Rung 2.5 08-25 05:54→06:39 · Rung 2.75 08-25 10:25→12:07 · R3 08-25 17:38→18:10 ·
  Rung 3 08-27 12:27→13:01 · Divergence 08-27 21:32→23:19.
- `*_history.csv` header (12 cols):
  `episode,start_tick,reward,success_rate,lost,critical_lost,relocations,preemptive,sla,protected,energy,infeasible`
- `*_updates.csv` header (13 cols):
  `update,episode,mean_reward,actor_loss,critic_loss,entropy,approx_kl,clip_frac,adv_std,value_mean,decision_frac,lr_scale,explained_var`
  — **there is no `adv_mean` column** (§12).
- `*_config.json` top-level keys: `config / train_start_window / train_frac / device / episodes /
  wall_time_s / risk`; `config` contains `env / reward / mappo / train`.
- `R2_trajectory/SPRINT_7_P4_trajectory_manifest.jsonl` row keys:
  `update, file, bytes, md5, sha256, episode, lr_scale, lr_actor, lr_critic, buffer_T, stats`.
  **`u000`** = 1,736,317 B with `buffer_T: null, stats: null`; **`u001`+** = 5,207,826 B with a
  10-key `stats` dict that *does* include `adv_mean`.

---

# 5. CHECKPOINT INVENTORY

`rollout_episodes = 8`, `episodes = 600`, `seed = 20260818`, `device = cpu` unless stated.
`critic_target` is recorded in the config JSON **only** for R2 / R3 / R2_traj_repro (all `"mc"`);
for earlier arms the field is absent, meaning the then-default λ-return target.

## 5.1 Root checkpoint families in `python-ai/saved_models/marl/` (25 files)

| Family | Files | Arm/exp | Seed | Episodes | rollout_ep | critic_target | Device | Tag | Role | Purpose | Protected | Valid comparison target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Sprint 6 production | `mappo.pth`, `mappo_best.pth` | Sprint 6 | 20260818 | 600 | 8 | *(absent)* | cpu | *(none)* | original | first working policy | YES | A0 (it is the same configuration reproduced) |
| A0 cpu | `mappo_A0_cpu_repro{,_best}.pth` | A0 | 20260818 | 600 | 8 | *(absent)* | cpu | `A0_cpu_repro` | **control** | legacy switches ON | YES | A1/A2/A3 |
| A0 cuda | `mappo_A0_CUDA_device_mismatch{,_best}.pth` | A0 | 20260818 | 600 | 8 | *(absent)* | **cuda** | `A0_CUDA_device_mismatch` | negative control | evidence cuda breaks seeds | YES | **none** — not comparable |
| A1 cpu | `mappo_A1_cpu_bugfix{,_best}.pth` | A1 | 20260818 | 600 | 8 | *(absent)* | cpu | `A1_cpu_bugfix` | treatment | `legacy_*` OFF | YES | A0 cpu |
| A1 cuda (killed) | `mappo_A1_CUDA_killed_at_90ep_best.pth` | A1 | 20260818 | **90 of 600** | 8 | *(absent)* | cuda | `A1_CUDA_killed_at_90ep` | abandoned | — | YES | **none** |
| A2 | `mappo_A2_crit_sign{,_best}.pth` | A2 | 20260818 | 600 | 8 | *(absent)* | cpu | `A2_crit_sign` | treatment | `w_criticality_migration 0.0` | YES | A1 |
| A3 | `mappo_A3_entropy{,_best}.pth` | A3 | 20260818 | 600 | 8 | *(absent)* | cpu | `A3_entropy` | treatment | `entropy_coef 0.05` | YES | A1 |
| **R2** | `mappo_R2_mc_target{,_best}.pth` | R2 | 20260818 | 600 | 8 | `mc` | cpu | `R2_mc_target` | **the control for all Sprint 7** | MC critic target | **YES** | R3, R2_traj_repro |
| **R3** | `R3_batch32{,_best}.pth` | R3 | 20260818 | **2400** | **32** | `mc` | cpu | `R3_batch32` | treatment | 4× batch | **YES** | R2 (same 75 updates) |
| R2 replication | `R2_traj_repro{,_best}.pth` | R2 replica | 20260818 | 600 | 8 | `mc` | cpu | `R2_traj_repro` | **replication** | prove bit-exactness | **YES** | R2 (must be identical) |
| `attrbug_gamma99` | `mappo_attrbug_gamma99{,_best}.pth` | exploratory | — | **400** | **4** | *(absent)* | cpu | `attrbug_gamma99` | exploratory | `gamma 0.99`, `lam 0.95`, `lr_a 3e-4`, `ent 0.01` | YES | **none** — different HPs |
| killed runs (4) | `mappo_attrbug_gamma95_killed_best.pth`, `mappo_attrfix_gamma95_killed_best.pth`, `mappo_g999_lowEV_killed_best.pth`, `mappo_best_lam95_killed.pth` | exploratory | — | partial | varies | *(absent)* | cpu | *(as named)* | abandoned | kept for provenance | YES | **none** |

## 5.2 `_best` vs final

`train.py:232` saves `<tag>_best.pth` **only when `mean_r > best_mean`**; `train.py:252` always
writes the final `<tag>.pth`.
- **`R2_best` ≡ `R2`** — the best update *was* the last one, so the two files are equivalent.
- **`R3_best` = update 45**, not the final. Any claim that R3 "peaked at 45" is **FALSE** as a
  statement about learning — it is only the reward-gate's argmax, and reward is explicitly not a
  pre-registered criterion.
- Four killed runs have a `_best.pth` but **no** final `.pth`, because training never reached the
  final save.

## 5.3 The trajectory family — `R2_trajectory/` (76 files)

- Naming: **`u000.pth` … `u075.pth`**, one file per PPO update index.
- **`u000` = the initialisation, *before* any update.** It is 1,736,317 B and its manifest row has
  `buffer_T: null, stats: null` — there was no rollout buffer yet. This is the "random init"
  reference point that Phase 5's central finding is stated against.
- **`u001` … `u075` = the policy *after* update *n*.** Each is 5,207,826 B and carries a 10-key
  `stats` dict. `n_updates = 600 // 8 = 75`, so `u075` is the fully trained policy and is
  equivalent to `R2_traj_repro.pth`.
- Role: **control-run trajectory**. Because Phase 4 proved bit-exact replication, these 76
  checkpoints are legitimately R2's own history, not a similar run's.
- Valid comparison: within-family across `u`; and `u075` ↔ `mappo_R2_mc_target.pth`.
- **There is no trajectory series for any other arm.** R3, A0–A3 have endpoints only. This is the
  single biggest asymmetry in the evidence base.

## 5.4 Final / best / trajectory / replicated / control at a glance

| Role | File |
|---|---|
| Project's final trained policy of record | `mappo_R2_mc_target.pth` |
| Its bit-exact replication | `R2_traj_repro.pth` |
| Its trajectory | `R2_trajectory/u000.pth` … `u075.pth` |
| The Sprint 7 control | `mappo_R2_mc_target.pth` |
| The Sprint 6.5 control | `mappo_A0_cpu_repro.pth` |
| The treatment that failed the variance test | `R3_batch32.pth` |
| The policy that everything is compared against and that **wins** | *not a checkpoint* — the risk-threshold rule in `marl/baseline.py` |

---

# 6. EXPERIMENT CONFIGURATION MATRIX

Values read from the arms' own `*_config.json`. `—` = **field absent from the JSON**, not guessed.

| Arm | Stage | Seed | Episodes | rollout_episodes | critic_target | Device | Purpose |
|---|---|---|---|---|---|---|---|
| `mappo` (Sprint 6) | 3 | 20260818 | 600 | 8 | — | cpu | first production policy |
| `attrbug_gamma99` | 3 | — | 400 | 4 | — | cpu | exploratory (`gamma 0.99`, `gae_lambda 0.95`, `lr_actor 3e-4`, `entropy_coef 0.01`, wall 938 s) |
| `A0_cpu_repro` | 4 | 20260818 | 600 | 8 | — | cpu | reproduce Sprint 6; `legacy_dead_migration_criticality=True`, `legacy_fixed_apply_order=True` |
| `A0_CUDA_device_mismatch` | 4 | 20260818 | 600 | 8 | — | **cuda** | same, on cuda (reproducibility counter-example) |
| `A1_cpu_bugfix` | 4 | 20260818 | 600 | 8 | — | cpu | both `legacy_*` = False |
| `A1_CUDA_killed_at_90ep` | 4 | — | 90/600 | 8 | — | cuda | killed; no config JSON |
| `A2_crit_sign` | 4 | 20260818 | 600 | 8 | — | cpu | `w_criticality_migration 1.0 → **0.0**` |
| `A3_entropy` | 4 | 20260818 | 600 | 8 | — | cpu | `entropy_coef 0.02 → **0.05**` |
| `R2_mc_target` | 8 | 20260818 | 600 | 8 | **`mc`** | cpu | fix the critic target |
| `R3_batch32` | 11 | 20260818 | **2400** | **32** | **`mc`** | cpu | 4× batch — variance test |
| `R2_traj_repro` | 15 | 20260818 | 600 | 8 | **`mc`** | cpu | bit-exact replication + 76-checkpoint trajectory |

**Universal across all arms:** `w_criticality = 2.0`, `team_reward_share = 0.3`.
**`legacy_*` switches:** `True/True` for A0 and A0_CUDA · `False/False` for A1, A2, A3, R2, R3,
R2_traj_repro · **absent** for `mappo` and `attrbug_gamma99`.
**`entropy_coef`:** 0.02 everywhere except A3 (0.05) and `attrbug_gamma99` (0.01).
**`w_criticality_migration`:** 1.0 everywhere except A2 (0.0).

---

# 7. COMMAND CHEAT SHEET

Only commands verified against the repository. Risk labels:
`[READ-ONLY]` `[DIAGNOSTIC]` `[EXPENSIVE — TRAINING]` `[DESTRUCTIVE — DO NOT RUN]`.
Unless stated, run from `python-ai/`.

## 7.1 Repository inspection `[READ-ONLY]`

```bash
cd python-ai/saved_models/marl && ls *.pth *.csv *.json *.md *.txt
```

```bash
cd python-ai/saved_models/marl && ls -d _*integrity*
```

```bash
cd python-ai/saved_models/marl/R2_trajectory && ls u*.pth | wc -l
```

```bash
cd python-ai && ls -l --time-style=long-iso *.log
```

## 7.2 Integrity verification `[READ-ONLY]`

The manifests store lines like `c88a7c35edbe50a3b0b17f4107012a10 *./__init__.py`. The `*./`
prefix is binary-mode with a relative path, so **verification only works from the directory the
manifest was captured in** (§12). `set -o pipefail` is mandatory whenever piping through `tee`.

```bash
set -o pipefail && cd python-ai/marl && md5sum -c ../saved_models/marl/_PHASE5_integrity/SPRINT_7_P5_code_after.md5
```

```bash
set -o pipefail && cd python-ai/saved_models/marl && md5sum -c _PHASE5_integrity/SPRINT_7_P5_artifacts_after.md5
```

```bash
set -o pipefail && cd python-ai/saved_models/marl/R2_trajectory && md5sum -c ../_PHASE5_integrity/SPRINT_7_P5_trajectory_after.md5
```

## 7.3 Self-tests `[READ-ONLY]`

```bash
cd python-ai && python -m marl.tests_env
```

```bash
cd python-ai && python -m marl.sanity_test
```

## 7.4 Diagnostic verification `[DIAGNOSTIC]`

Reconstructed from the scripts' own CLIs, not quoted from a report. All of these **write** their
output JSON — ⚠ they will overwrite an existing artifact of the same name.

```bash
cd python-ai && python -m marl.diag._phase4_verify
```

```bash
cd python-ai && python -m marl.diag._phase4_equiv
```

```bash
cd python-ai && python -m marl.diag._phase4_smoke
```

⚠ **writes** `SPRINT_7_PHASE5_risk_trajectory_<tag>.json` (default tag `main` — i.e. it would
overwrite the published artifact; pass `--tag` to write elsewhere):

```bash
cd python-ai && python -m marl.diag._phase5_risk_trajectory --device cpu --clusters 32 --start-seed 20260825 --boot 5000 --random-seed 31337 --tag recheck
```

## 7.5 Checkpoint verification `[READ-ONLY]`

```bash
cd python-ai/saved_models/marl && md5sum mappo_R2_mc_target.pth R2_traj_repro.pth
```

```bash
cd python-ai/saved_models/marl/R2_trajectory && head -1 SPRINT_7_P4_trajectory_manifest.jsonl
```

## 7.6 Report inspection `[READ-ONLY]`

```bash
cd python-ai/saved_models/marl && ls -l --time-style=long-iso *.md *.txt
```

```bash
cd python-ai/saved_models/marl && grep -n '^#' SPRINT_7_FINAL_SYNTHESIS_AUDIT.md
```

## 7.7 CSV / JSON inspection `[READ-ONLY]`

```bash
cd python-ai/saved_models/marl && head -1 mappo_R2_mc_target_updates.csv
```

```bash
cd python-ai/saved_models/marl && python -c "import json;d=json.load(open('mappo_R2_mc_target_config.json'));print(json.dumps(d['config']['mappo'],indent=1))"
```

> Read hyperparameters from `d['config']['mappo']` — **`d['config']['train']` returns `None` for
> `entropy_coef`, `critic_target` and `gamma`** (§12).

## 7.8 Git inspection `[READ-ONLY]`

```bash
git log --pretty=format:'%h|%ad|%s' --date=iso
```

```bash
git status -sb
```

## 7.9 Training commands `[EXPENSIVE — TRAINING]`

**Standing rule: never rerun training to "see if it goes away."** Sprint 7 is closed with the
verdict *"no further training justified"*. These are recorded for provenance only.

The **only fully verbatim command in the entire `.md` corpus** (R3, `SPRINT_7_R3_REPORT.md`):

```bash
python -m marl.train --critic-target mc --rollout-episodes 32 --episodes 2400 --seed 20260818 --device cpu --tag R3_batch32
```

Reconstructed from `marl/train.py`'s CLI — ⚠ each **overwrites** `<tag>.pth`, `<tag>_best.pth`,
`<tag>_config.json`, `<tag>_history.csv`, `<tag>_updates.csv`:

```bash
python -m marl.train --critic-target mc --device cpu --tag R2_mc_target
```

```bash
python -m marl.evaluate --model saved_models/marl/mappo_R2_mc_target.pth --episodes 8 --device cpu
```

Full `marl/train.py` CLI surface:
`--episodes --rollout-episodes --episode-steps --seed --lr-actor --lr-critic --entropy-coef
--w-criticality-migration --risk-source {oof,model,zero} --critic-target {lambda,mc} --device
--tag --log-every --legacy-sprint6 --legacy-fixed-apply-order
--legacy-dead-migration-criticality`
Full `marl/evaluate.py` CLI: `--model --episodes (8) --device (cpu) --out --skip-ablation`.

**`--device cpu` is mandatory.** cuda silently breaks seed reproducibility and is 2.8× slower on
these tiny actors.

## 7.10 `[DESTRUCTIVE — DO NOT RUN]`

- `git checkout -- python-ai/saved_models/` — would discard uncommitted experiment outputs.
- Any `git rebase` / `git commit --amend` / history rewrite — reports cite commit `a8df6d9`
  **by hash**.
- Re-running any diagnostic with its **default `--tag`** while intending to keep the published
  artifact — the default writes over `*_main.json`.
- Re-running `marl.train` with an existing tag.
- Deleting or regenerating `R2_trajectory/` — it cannot be recovered without a full Phase 4 rerun.

---

# 8. "WHAT DO I OPEN?" LOOKUP TABLE

All paths relative to `python-ai/saved_models/marl/` unless a fuller path is given.

| If I want to know… | Open / check this |
|---|---|
| **Anything about Sprint 7 — start here** | `SPRINT_7_FINAL_SYNTHESIS_AUDIT.md` (it supersedes the 11-report chain) |
| The complete project narrative | `PROJECT_RESEARCH_TUTORIAL.md` (repo root), 21 parts |
| What Sprint 6 did | `SPRINT_6_5_REPORT.md` §early + `mappo_config.json` + `mappo_eval.json` |
| What Sprint 6.5 did, and why A0–A3 exist | `SPRINT_6_5_REPORT.md` |
| Why Sprint 7 started at all | `SPRINT_7_PHASE1_DIAGNOSIS.md` |
| **Why R2 exists** | `SPRINT_7_RUNG0_REPORT.md` (the sign inversion) → `SPRINT_7_RUNG1_REPORT.md` (the offset) |
| **R2's exact configuration** | `mappo_R2_mc_target_config.json`, key `config.mappo` |
| **R2's final result** | `mappo_R2_mc_target_eval.json` (task perf) + `SPRINT_7_RUNG2_5_REPORT.md` (mechanism) |
| **R2's trajectory** | `R2_trajectory/u000.pth`…`u075.pth` + `R2_trajectory/SPRINT_7_P4_trajectory_manifest.jsonl` + `SPRINT_7_PHASE5_risk_trajectory_main.json` |
| Whether R2 is trustworthy / reproducible | `SPRINT_7_PHASE4_REPORT.md` (criteria in `SPRINT_7_PHASE4_PREREG.md`) |
| **What R3 was and why it failed** | `SPRINT_7_R3_REPORT.md` (NO-GO on P1–P4) |
| R3's held-out task performance | **Does not exist.** No `R3*eval*` artifact was ever produced |
| Whether dilution explained R3 | `SPRINT_7_RUNG3_REPORT.md` + `SPRINT_7_RUNG3_dilution_main.json` (32 ep = the pre-registered one) |
| **How the frozen state sets were built** | `marl/_diag_rung2_75_matched_states.py` → `SPRINT_7_RUNG2_75_matched_states_main.json`, documented in `SPRINT_7_RUNG2_75_REPORT.md` |
| **The advantage analysis** | `SPRINT_7_RUNG1_REPORT.md` + `SPRINT_7_RUNG1_critic_fit_and_residual.json` + `_deviation_agreement.json` |
| Why the actor stopped learning | `SPRINT_7_RUNG2_5_REPORT.md` + `SPRINT_7_RUNG2_5_actor_stall.json` (softmax saturation) |
| Why every endpoint metric points the wrong way | `SPRINT_7_DIVERGENCE_REPORT.md` (20/20 inversions, mean ρ = −0.900) |
| What Δ_EDGE actually measures | `SPRINT_7_PHASE5_REPORT.md` (differential suppression, not acquisition) |
| **The integrity history** | the 12 `_*integrity/` directories; the most complete is `_PHASE5_integrity/` |
| **The preregistrations** | `SPRINT_7_RUNG3_PREREGISTRATION.md` and `SPRINT_7_PHASE4_PREREG.md` (both written before their runs) |
| The final Sprint 7 conclusion | `SPRINT_7_FINAL_SYNTHESIS_AUDIT.md` — COMPLETE, "no further training justified" |
| Which reports contradict each other | `SPRINT_7_FINAL_SYNTHESIS_AUDIT.md` §8 (corrections table) and §12 below |
| The frozen hyperparameters | `marl/config.py`, the `mappo` block |
| The masking / legality rules | `marl/env.py:405-422`, `:620-626`; `marl/mappo.py:77-79` |
| The critic target that started it all | `marl/mappo.py:298` |
| Whether the risk signal leaks the future | `data/failure_dataset.py` (`assert_no_leakage()`) + `marl/tests_env.py` t7/t8 |
| Whether the repo is intact | `python -m marl.tests_env`, `python -m marl.sanity_test`, then the `_PHASE5_integrity` manifests |
| What the baseline that wins actually is | `marl/baseline.py` |
| Why cpu and not cuda | §12 below; `mappo_A0_CUDA_device_mismatch_*` is the kept evidence |
| Every experiment's date | the 69 `.log` files in `python-ai/` (`ls -l --time-style=long-iso *.log`) |

---

# 9. EXPERIMENT DEPENDENCY MAP

Only stages that actually exist. One sentence per transition.

```
INPUTS
  Java digital twin (simulation/, SIM_SEED 20260817L, 1500 s, 10 devices)
        │  exports per-tick host telemetry + failure ledgers
        ▼
  Failure dataset (data/failure_dataset.py — leakage firewall)
        │  restricts features to the 12 observable columns; forbids audit/label columns
        ▼
  BiLSTM host failure predictor (26 feat, seq 10, thr 0.18, 5 temporal folds)
        │  emits OUT-OF-FOLD risk: leakage-safe, uncalibrated, near-bimodal, ~4.7% > 0.18
        ▼
  Replay trace + risk provider (marl/trace.py, marl/risk_provider.py)
        │  fixes the 10×1500 window, train ticks [9,491], eval [491,698], 8 fixed starts
        ▼
STAGE 3  Sprint 6 MAPPO  (mappo.pth)
        │  loses to the risk-threshold rule ⇒ something is wrong, not just undertrained
        ▼
STAGE 4  Sprint 6.5 ladder  A0 → A1 → A2 → A3
        │  one change per arm from a shared θ₀; none fixes it, performance declines monotonically
        ▼
STAGE 5  Phase 1 diagnosis
        │  static analysis of A0's rollouts yields the inverted-ordering hypothesis (n=4/n=9)
        ▼
STAGE 6  Rung 0  (D1–D4)
        │  confirms the sign inversion on a real sample and localises it to mappo.py:298
        ▼
STAGE 7  Rung 1  (critic fit, paired estimator)
        │  shows the error is a per-state offset c(s), not a target defect; pairing restores agreement
        ▼
STAGE 8  Rung 2 = arm R2_mc_target
        │  the MC target fixes ~87% of the bias — and task performance falls to its worst value
        ▼
STAGE 9  Rung 2.5  (actor stall)
        │  explains the non-response as softmax saturation (50.5% at max-prob > 0.99)
        ▼
STAGE 10 Rung 2.75  (frozen RANDOM/UNION sets, Δ_EDGE)
        │  gives every later stage ONE metric on ONE fixed sample; withdraws the 2.5 metric
        ├──────────────────────────────┐
        ▼                              │  (frozen sets are reused, unchanged, by everything below)
STAGE 11 R3 = arm R3_batch32           │
        │  4× batch raises SNR yet worsens risk response ⇒ variance hypothesis FALSIFIED
        ▼                              │
STAGE 12 Rung 3  (dilution)  ◀─────────┘
        │  dilution is real (9/9) but orders the arms backwards ⇒ NO-GO; also proves rollout_episodes
        │  is a pure variance knob and that the "5.1× collapse" was a +2.09σ instrument outlier
        ▼
STAGE 13 Divergence study
        │  out of hypotheses, asks what merely DIFFERS: regime-selective asymmetry; and finds every
        │  endpoint instrument inverted (20/20) ⇒ the blocker is a missing observable
        ▼
STAGE 14 Phase 0 reconstruction
        │  rebuilds the arm ledger and recalibrates Δ_EDGE before any trajectory work is trusted
        ▼
STAGE 15 Phase 4 (+4.1, 4RUN) = arm R2_traj_repro + R2_trajectory/u000–u075
        │  proves bit-exact replication first, so the 76 checkpoints are R2's OWN history
        ▼
STAGE 16 Phase 5  (risk trajectory)
        │  watching Δ_EDGE over the 76 checkpoints shows it is differential SUPPRESSION, and dates
        │  the policy freeze to update 62
        ▼
STAGE 17 Final Synthesis Audit
           audits the whole chain, records the contradictions, closes Sprint 7 COMPLETE
```

**Side branches that feed nothing:** the five killed Sprint 6 runs, `attrbug_gamma99`,
`A0_CUDA_device_mismatch` and `A1_CUDA_killed_at_90ep` (kept as provenance / negative controls),
`diag_counterfactual*.json` and `diag_sensitivity.json` (superseded by Rung 0's D1–D4).

---

# 10. IMPORTANT HASHES / INTEGRITY

The methodology exists because **Sprints 6, 6.5 and 7 all ran inside the uncommitted month-long
gap between commits `208e5a2` (2026-07-29) and `a8df6d9` (2026-08-30)**. Git could not attest to
those artifacts, so each stage captured **before/after MD5 manifests** instead.

## 10.1 How the manifests work

- 12 per-stage directories, with **inconsistent case** exactly as found:
  `_rung0_integrity`, `_rung2_5_integrity`, `_rung2_75_integrity`, `_R3_integrity`,
  `_RUNG3_integrity`, `_DIVERGENCE_integrity`, `_PHASE0_integrity`, `_PHASE4_integrity`,
  `_PHASE41_integrity`, `_PHASE4RUN_integrity`, `_PHASE5_integrity`, `_AUDIT_integrity`.
- Line format: `c88a7c35edbe50a3b0b17f4107012a10 *./__init__.py`. The `*` means binary mode and
  `./` means a relative path ⇒ **`md5sum -c` only works from the capture directory**.
- `_PHASE5_integrity/` is the most complete: `SPRINT_7_P5_{artifacts,code,inputs,trajectory}_{before,after}.md5`
  plus `_prior_manifests.md5` and `_own_manifests.md5`, with counts
  **156/160** (artifacts), **52/53** (code), **3/3** (inputs), **78/78** (trajectory),
  **69** (prior manifests), **10** (own manifests).
- `set -o pipefail` is mandatory when piping to `tee`, or a failing check exits 0 (§12).

## 10.2 PROTECTED / DO NOT TOUCH

| Category | Items |
|---|---|
| Production source | all 13 `marl/*.py` production modules + `sanity_test.py`, `tests_env.py`; `data/failure_dataset.py`; the 44 `.java` files. **Production source last changed 2026-08-25 04:02** and must stay that way |
| Protected checkpoints | `mappo_R2_mc_target{,_best}.pth`, `R2_traj_repro{,_best}.pth`, `R3_batch32{,_best}.pth`, all 76 `R2_trajectory/u*.pth`, `mappo{,_best}.pth`, the four A-arm pairs and the cuda/killed evidence checkpoints |
| Protected R2 artifacts | `mappo_R2_mc_target_config.json`, `_history.csv`, `_updates.csv`, `_eval.json`, `_best_eval.json`, `_eval.csv`, `_best_eval.csv`; `R2_trajectory/SPRINT_7_P4_trajectory_manifest.jsonl` (md5 + sha256 per checkpoint) |
| Protected inputs | `simulation/predicted_risk.csv`, the three failure CSVs, `saved_models/failure_predictor.pth` + `_meta.json` + `host_metrics.json` (the `_PHASE5_integrity` "inputs" manifest covers exactly **3/3** files) |
| Frozen state sets | `SPRINT_7_RUNG2_75_matched_states_main.json` — RANDOM (4,975 / 216 hi) and UNION (74,237 / ~3,592 hi). Regenerating these silently invalidates every downstream number |
| Reports (16) | the 15 `.md` + `SPRINT_7_P4_EQUIV_SELFTEST.txt` in `saved_models/marl/`. **Corrections are additive — never edit a historical report** |
| Preregistrations | `SPRINT_7_RUNG3_PREREGISTRATION.md`, `SPRINT_7_PHASE4_PREREG.md` |
| Manifests | all 12 `_*integrity/` directories |
| Recorded hashes worth knowing | `risk_provider.py` = `02219832596a38e9800ba4f89659a555`; `_diag_R3_action_channels.py` = `a25034e565c139b832e399cf6fed1f7c`; `R3_batch32.pth` = `d9f8a3dfbe1caf3b57402b9ef05313e3` (all three recorded in R3's manifest section) |

## 10.3 REGENERABLE

- Any `*_eval.json` / `*_eval.csv` — `marl.evaluate` is deterministic on cpu over the 8 fixed
  starts.
- Any diagnostic JSON, **provided** the frozen state sets and the checkpoints are unchanged.
- Predictor artifacts (expensive: 80 epochs × 5 folds).
- Whole arms, in principle — R2 and A0 were both demonstrated bit-exactly reproducible on cpu.
  **Regenerable ≠ should be regenerated.**

## 10.4 DIAGNOSTIC / ADDITIVE

- All 32 `marl/_diag_*` and 6 `marl/diag/_phase*` scripts, and all diagnostic JSON outputs.
- Convention: a new probe is a **new file**; existing probes gain **defaulted-off flags** only
  (e.g. `_diag_rung2_75_coherence.py`'s `--also name=file` defaults to empty so the `main`
  artifact stays reproducible).
- `*_SMOKE.json` files are **smoke runs, not results** — never cite them.

---

# 11. SCIENTIFIC FINDINGS

Project terminology preserved. "Type" and "Status" are stated as the repository states them; no
correlational result is promoted to causal.

| # | Finding | Evidence | Stage | Type | Status |
|---|---|---|---|---|---|
| 1 | The OOF risk signal is leakage-safe but **uncalibrated and near-bimodal**; only ~4.7% of decision states exceed the 0.18 threshold | `failure_dataset.py` firewall, `tests_env.py` t7/t8, the risk banner | 2 | methodological | **VALID** |
| 2 | The MAPPO policy **loses to a one-line risk-threshold rule** on held-out task performance, for every arm | `mappo_eval.json`, the A0–A3 and R2 eval artifacts | 3–8 | descriptive | **VALID** |
| 3 | Held-out performance **declines monotonically along the intervention ladder**: A0 20.62 → A1 15.69 → A2 12.33 → A3 6.82 → R2 −0.46, while Δ_EDGE improves | the arms' eval JSONs vs Phase 0 calibration | 4–8 | descriptive | **VALID** (and unexplained) |
| 4 | The advantage estimator's **action ordering at high risk is inverted** relative to the truth | Phase 1 (n=4/n=9, self-caveated) then Rung 0 D1 | 5–6 | causal (for A0) | **VALID but superseded in scope** — ~87% fixed by R2, partly downgraded by Rung 2.5 |
| 5 | The sign error is a **per-state offset `c(s)`**, not a defect in the target; `gae(s,a) = c(s) + paired(s,a)` | Rung 1; `Var(raw)/Var(paired) = 3.72×` ⇒ 1.93× attenuation | 7 | causal | **VALID** |
| 6 | The **paired estimator** lifts high-risk action agreement from below chance to **0.71 / 0.95 for every arm** | `SPRINT_7_RUNG1_deviation_agreement.json` | 7 | methodological | **VALID** — the project's best methodological result |
| 7 | The R2 stall is **softmax saturation**, not a vanishing gradient: 50.5% of states at max-prob > 0.99 vs A0's 0.0000 | `SPRINT_7_RUNG2_5_actor_stall.json` | 9 | descriptive | **VALID as a symptom** — downgraded from cause by Rung 3 |
| 8 | **The variance hypothesis is falsified**: a 4× batch raised gradient SNR (p = 0.0000) yet made risk response worse | R3, pre-registered P1–P4 all fail | 11 | causal (falsification) | **VALID** |
| 9 | `rollout_episodes` is a **pure variance intervention** — expected update content is invariant to batch size (≤ 0.105σ₈) and `sd` follows the finite-population law `sd(8)/sd(32) = 2.236` | Rung 3 bootstrap | 12 | causal | **VALID** |
| 10 | The earlier **"5.1× collapse" was a +2.09σ outlier** of the 8-episode instrument, not a real effect | Rung 3 | 12 | methodological correction | **VALID — supersedes the earlier claim** |
| 11 | **Dilution is real** — the low-risk bulk sets the update direction in 9/9 cells — **but does not explain R3**, because the mass ratios order the arms opposite to behaviour | `SPRINT_7_RUNG3_dilution_main.json`, `_verdict_main.json` | 12 | causal (partial) + falsification | **VALID (NO-GO)** |
| 12 | The reference direction **`synth` does not discriminate the action channel** | Rung 3 §on `synth` | 12 | methodological limitation | **VALID (open limitation)** |
| 13 | **Regime-selective learning asymmetry**: R3 learned the low-risk bulk and not the high-risk minority, on two independent instruments | Divergence study | 13 | **descriptive — explicitly NOT causal** | **VALID, UNRESOLVED as a cause** |
| 14 | **Endpoint instruments invert against behaviour** — 20/20 readings of every Sprint 7 mechanism metric order the arms backwards (mean ρ = −0.900). Endpoint measures *headroom*, not learning | Divergence study | 13 | methodological | **VALID** — *"the blocker is a missing observable, not a missing hypothesis"* |
| 15 | **Mechanism 2 (per-state offset as a bias) is closed** — R3 falsified it, the clip is too rarely active, and `c(s)` is agent-idiosyncratic so no online centring exists | Divergence §on mechanism 2 | 13 | causal (closure) | **VALID — closed** |
| 16 | **R2 is bit-exactly reproducible**: 271/271 members, **265/265 tensor storages byte-identical with no normalisation**; only `data.pkl` and `.data/serialization_id` differ, and only because of the tag string and archive stem | Phase 4, criteria B1a–B1e pre-registered | 15 | reproducibility | **VALID** |
| 17 | **R2's policy freezes at update 62** — the actor stops moving with ~19% of training left while the critic is still improving fastest. Rules out "the critic gated the actor" and dates the softmax collapse to u1–u40 | Phase 5 over the 76 checkpoints | 16 | descriptive | **VALID** |
| 18 | **Δ_EDGE is differential suppression, not risk acquisition** — π(EDGE given high risk) ends *below* random init and never exceeds it in 76/76 checkpoints; **159–218%** of the metric's growth is low-risk suppression | Phase 5 | 16 | causal (for the metric) | **VALID — the terminal finding** |
| 19 | Sprint 7 is closed **COMPLETE** with the verdict **"no further training justified"** | Final Synthesis Audit | 17 | methodological | **VALID — final** |
| 20 | Rung 2's high-risk EDGE-share **direction** was never resolved; the metric that would have decided it was withdrawn | Rung 2.5 §G.5; Rung 2.75 §A.4 | 9–10 | — | **UNRESOLVED (metric withdrawn)** |
| 21 | Phase 5's expectation-vs-greedy readings conflict: 0/76 and 1/76 under expectation vs greedy argmax-EDGE rising 0.083 → 0.463 | Phase 5 | 16 | descriptive | **UNRESOLVED** |

## 11.1 Recorded disagreements between documents (not reconciled)

Per the project's own convention, contradictions are **recorded**, and the later document is
identified as superseding — the earlier report is never edited.

| # | Disagreement | Documents | Which supersedes |
|---|---|---|---|
| 1 | R2-vs-R3 config field count: **91 fields / 3 differ** vs **99 fields / 5 differ / 1 substantive** | `SPRINT_7_R3_REPORT.md` §3 vs `SPRINT_7_DIVERGENCE_REPORT.md` §2.1 | Divergence (later) — **not formally reconciled** |
| 2 | R3's Δ on RANDOM: **+0.1575** vs **+0.1576** | R3 report / Phase 0 vs Divergence §2.3 | immaterial rounding; recorded, not resolved |
| 3 | Phase 4 §G.3 defers to **"Phase 6"**; the report that did the work is **Phase 5** | `SPRINT_7_PHASE4_REPORT.md` vs `SPRINT_7_PHASE5_REPORT.md` | Phase 5 (the numbering in Phase 4 is simply wrong) |
| 4 | Phase 5 §4.4 claims `SPRINT_7_R3_action_channels.json` does not exist — the file is `SPRINT_7_R3_action_channels_R3.json` | `SPRINT_7_PHASE5_REPORT.md` §4.4 | `SPRINT_7_FINAL_SYNTHESIS_AUDIT.md` §8 row 17 corrects it **additively** |
| 5 | Rung 2's EDGE-share direction UNRESOLVED, then the metric withdrawn | Rung 2.5 §G.5 vs Rung 2.75 §A.4 | Rung 2.75 (withdrawal stands) |
| 6 | Sprint 6.5 §13 recommended a **smaller** `rollout_episodes`; R3 tested a **larger** one | `SPRINT_6_5_REPORT.md` §13 vs R3 | R3's direction was chosen; the recommendation was never tested |
| 7 | Phase 0 §1.1 states A3's manipulation as `entropy_coef 0.01 → 0.02` — **both endpoints wrong** (actual 0.05 vs 0.02) | Phase 0 §1.1 vs Rung 2.75 §7 H4 | Rung 2.75 §7 H4 is correct; Phase 0 not edited |
| 8 | A2 is labelled "critic sign" but the manipulation is **criticality** (`w_criticality_migration 1.0 → 0.0`) | arm tag / Phase 0 §1.1 vs `mappo_A2_crit_sign_config.json` | the config JSON is authoritative |
| 9 | Phase 1's proposed **B1–B3 / C1–C4 / E1–E3 ladder has no artifacts**; only "Rung 0" survived by name | Phase 1 vs the artifact tree | the artifact tree |
| 10 | Phase 5's expectation (0/76, 1/76) vs greedy (0.083 → 0.463) readings | within Phase 5 | unresolved, recorded |
| 11 | **"RULE 1–12" is cited throughout the corpus and enumerated in no file** | corpus-wide | **NOT VERIFIED FROM REPOSITORY** |
| 12 | No standalone `SPRINT_7_RUNG2_REPORT.md`, and no `SPRINT_7_PHASE4_1*.md` (Phase 4.1 exists only as `_PHASE41_integrity/` manifests) | expected vs actual file list | the actual file list |

---

# 12. Known Gotchas

Every item below was verified in the repository. Nothing is included on suspicion.

| # | Wrong assumption | Correct fact | Where verified |
|---|---|---|---|
| 1 | Hyperparameters live under `config['train']` | They live under **`config['mappo']`**. Reading `entropy_coef`, `critic_target` or `gamma` from `d['config']['train']` silently returns **`None`** | any `*_config.json`; `marl/config.py` |
| 2 | `torch.save` output depends only on the payload | It also depends on the **archive stem** (member *names* only — 271 members × 5-char delta = 1,355 bytes; content, including `data.pkl` and `serialization_id`, is byte-identical under an equal-length stem) and on the **tag string** (changes exactly 2 container members, **0 of 265 tensor storages**) | `SPRINT_7_PHASE4_PREREG.md` §13, N1/N2 table; `SPRINT_7_PHASE4_REPORT.md` B1a/B1b |
| 3 | A different `.pth` md5 means a different policy | Not necessarily — compare at the **container layer**. R2 vs `R2_traj_repro`: 271/271 members, 265/265 storages byte-identical, differing set ⊆ `{data.pkl, .data/serialization_id}` | Phase 4 B1a |
| 4 | `train.py`'s "independence" line is decorative | It calls **`MAPPOAgent.assert_actors_independent()`** (`mappo.py:514`), which proves the 10 actors share **no** parameter objects and raises `AssertionError` if any two do; it returns a string instead when `separate_actors=False` | `marl/train.py:129`; `marl/mappo.py:514-526` |
| 5 | `<tag>_best.pth` is the "good" checkpoint | It is only the **reward-gate argmax** (`train.py:232`, `mean_r > best_mean`). **`R2_best` ≡ `R2`**; **`R3_best` = update 45**, and "R3 peaked at 45" is FALSE as a learning claim because reward is explicitly not a pre-registered criterion | `marl/train.py:232`, `:252`; `SPRINT_7_R3_REPORT.md` §37 |
| 6 | Adding a model to a diagnostic's model list is harmless | In `_diag_rung2_75_matched_states.py`, `MODELS` is **state-source-generating** and `EXTRA` is scored-only. R3 was deliberately added to `EXTRA` and **not** to `MODELS`: promoting it "would have silently redefined RANDOM and UNION, so the pre-registered thresholds would no longer be measured on the sets they were calibrated on" | `SPRINT_7_R3_REPORT.md` §"The two changed files" |
| 7 | `md5sum -c` can be run from anywhere | Manifest lines are `<hash> *./path` — binary mode, relative path. **Verify only from the directory the manifest was captured in**, or every entry fails | any `_*integrity/*.md5` |
| 8 | A single failing manifest entry means corruption | A lone failure is usually the manifest's **own self-entry** (circular hashing — a file cannot contain its own hash) | `_PHASE5_integrity/_own_manifests.md5` |
| 9 | `cmd 2>&1 \| tee log` reports failures | The pipeline's exit status is **`tee`'s**, so a non-zero exit is masked. **`set -o pipefail` is mandatory** | integrity procedure in the Phase reports |
| 10 | `approx_kl` should be non-negative | The column is **signed k1** (`log ratio`), not k3 (`ratio − 1 − log ratio`). Negative values are expected and are not a bug | `*_updates.csv`; k1-vs-k3 note in the reports |
| 11 | `clip_frac = 0` means PPO is broken | With a saturated softmax the ratio rarely leaves the clip band. `clip_frac = 0` is **normal here** | `SPRINT_7_RUNG2_5_actor_stall.json` |
| 12 | `*_updates.csv` has an `adv_mean` column | It does **not** — the 13 columns are listed in §4.10. `adv_mean` exists only inside the trajectory manifest's per-update `stats` dict | header of any `*_updates.csv`; `SPRINT_7_P4_trajectory_manifest.jsonl` |
| 13 | `--episodes` controls a diagnostic's sample | **D3 and D4 silently ignore `--episodes`** | Rung 0 diagnostics |
| 14 | Passing `--tag foo` leaves the default artifact alone but writes `<name>.json` | Tagged probes write **`<name>_<tag>.json`**, so suffixed artifacts *look* missing (e.g. `SPRINT_7_R3_action_channels_R3.json`, `..._matched_states_R3.json`, `..._offset_R2.json`). Conversely the **default tag is `main`**, which overwrites the published artifact | `saved_models/marl/` filenames; `_phase5_risk_trajectory.py` `--tag` default |
| 15 | 4 minibatches means 4 chunks | When `T mod 4 ≠ 0` there is a **tail chunk**, giving **5** | `SPRINT_7_RUNG1_minibatch_tail_issue.json`, `..._RUNG2_75_mbtail_main.json` |
| 16 | `_phase4_verify.py` B6 failing means something is wrong | **B6 fails by design** on a 1-ULP reference constant (`…3688` vs `…3687`) at `_phase4_verify.py:78` | `marl/diag/_phase4_verify.py` |
| 17 | `infeasible` counts illegal actions | It counts **contention** | `*_history.csv` semantics; `env.py` |
| 18 | The agents can use PREEMPT_REROUTE at high risk | **`PREEMPT_REROUTE` is legal in 0.00% of high-risk states** | Rung 0/2.75 diagnostics |
| 19 | `by_state_source` percentages share a denominator | They use **mixed denominators** — read the key before comparing | Rung 2.75 / Rung 3 JSONs |
| 20 | Rewards are comparable across arms | They are not, when the reward shaping differs (A2 zeroes `w_criticality_migration`). And reward is **not a pre-registered criterion** in R3 | `SPRINT_7_R3_REPORT.md` §37 |
| 21 | Training on cuda is equivalent and faster | cuda **silently breaks seed reproducibility** and is **2.8× slower** on these tiny actors. `--device cpu` is mandatory; `mappo_A0_CUDA_device_mismatch_*` is the retained evidence | Sprint 6.5 arms; the two cuda runs |
| 22 | D4's `replica_fidelity` covers the whole agent | It is **critic-side only**; `ok=False` does not imply an actor mismatch | Rung 0 D4 output |
| 23 | The `mappo.py` horizon comment is a bug to fix | `mappo.py:447` says ~20 steps, correct for the **retired** λ=0.95 and wrong at λ=0.995 (horizon **166.8**). `config.py:434` has it right. It is **deliberately NOT fixed** so production source stays frozen | `marl/mappo.py:447`, `marl/config.py:434` |
| 24 | The MARL ring topology mirrors the Java link graph | `DigitalTwinManager.mirrorNetworkLinks(int)` makes **one access link per node**, not a node-to-node graph. The degree-4 ring (`neighbour_offsets=[-2,-1,1,2]`) is a Python-side construct | `simulation/.../DigitalTwinManager.java`; `marl/topology.py` |
| 25 | The environment co-simulates with CloudSim | It is a **trace replay**; *"the trajectory is exogenous"* | `marl/env.py` docstring |
| 26 | `*_SMOKE.json` and `*_ep8.json` are results | `_SMOKE` files are smoke runs. For Rung 3, the **`_main` (32-episode) artifact is THE pre-registered result**; `_ep8` is supplementary | `SPRINT_7_RUNG3_REPORT.md` §176, §517 |
| 27 | `SPRINT_7_DIV_geometry_eval.json` is a policy evaluation | It is a **gradient-geometry** artifact — the only `*_eval*` file in the tree that is not a held-out policy evaluation | `saved_models/marl/` |
| 28 | Every arm has an evaluation | **R3 has none.** No `R3*eval*.json` and no `R3*eval*.log` exists | exhaustive `ls` of `*eval*` |
| 29 | Corrections are applied by editing the report | The project convention is **additive correction** — e.g. Phase 5 §4.4's error is fixed in `SPRINT_7_FINAL_SYNTHESIS_AUDIT.md` §8 row 17, with the report left as written | Audit §8 |

---

# 13. PROJECT STATUS

| Stage | Status | Final output | Remaining work |
|---|---|---|---|
| 1 — Java digital twin | **COMPLETE** (frozen) | `simulation/predicted_risk.csv` + 3 failure CSVs | none |
| 2 — Host failure predictor | **COMPLETE** (frozen) | `failure_predictor.pth` + `_meta.json` + `host_metrics.json` | none |
| 3 — Sprint 6 MAPPO | **SUPERSEDED** as a result, protected as an artifact | `mappo{,_best}.pth`, `mappo_eval.json` | none |
| 4 — Sprint 6.5 ladder A0–A3 | **COMPLETE** | `SPRINT_6_5_REPORT.md` + 4 arm families | its §13 recommendation (a *smaller* `rollout_episodes`) was never tested — deliberately |
| 5 — Phase 1 diagnosis | **SUPERSEDED** | `SPRINT_7_PHASE1_DIAGNOSIS.md` | its B1–B3 / C1–C4 / E1–E3 ladder was never built |
| 6 — Rung 0 (D1–D4) | **COMPLETE** | `SPRINT_7_RUNG0_REPORT.md` + 4 `diag_S7_D*.json` | none |
| 7 — Rung 1 | **COMPLETE** | `SPRINT_7_RUNG1_REPORT.md` + 3 JSONs | none |
| 8 — Rung 2 (R2) | **COMPLETE** | `mappo_R2_mc_target*` — the control arm | none |
| 9 — Rung 2.5 | **COMPLETE**, headline metric later withdrawn | `SPRINT_7_RUNG2_5_REPORT.md` + 7 JSONs | Rung 2's high-risk EDGE-share **direction is UNRESOLVED** |
| 10 — Rung 2.75 | **COMPLETE** | frozen RANDOM/UNION sets + Δ_EDGE + 16 JSONs | none |
| 11 — R3 | **COMPLETE (NO-GO)** | `R3_batch32*`, `SPRINT_7_R3_REPORT.md` | R3 was **never evaluated** on the held-out suite — intentionally not done |
| 12 — Rung 3 | **COMPLETE (NO-GO)** | `SPRINT_7_RUNG3_REPORT.md` + prereg + 8 JSONs | `synth`'s inability to discriminate the action channel is an **open limitation** |
| 13 — Divergence study | **COMPLETE** | `SPRINT_7_DIVERGENCE_REPORT.md` + 8 JSONs | the missing observable it identifies is **OPEN** |
| 14 — Phase 0 | **COMPLETE** | `SPRINT_7_PHASE0_RECONSTRUCTION.md` + calibration JSON | contains 2 label errors, corrected additively elsewhere |
| 15 — Phase 4 / 4.1 / 4RUN | **COMPLETE** | prereg + report + `R2_traj_repro*` + `R2_trajectory/` (76) | **Phase 4.1 has manifests but no report** |
| 16 — Phase 5 | **COMPLETE** | `SPRINT_7_PHASE5_REPORT.md` + 2 JSONs | its expectation-vs-greedy conflict is **UNRESOLVED** |
| 17 — Final Synthesis Audit | **COMPLETE — closes Sprint 7** | `SPRINT_7_FINAL_SYNTHESIS_AUDIT.md` | **none. Verdict: "no further training justified."** |
| Documentation | **COMPLETE** | `PROJECT_RESEARCH_TUTORIAL.md`, this file | none |

### DONE

- The full Java → predictor → MARL pipeline, end to end.
- 11 trained MAPPO arms (7 complete + 1 exploratory + 3 killed/partial), all retained.
- A 12-stage Sprint 7 mechanism ladder, with preregistration at the two points where it mattered.
- Bit-exact reproducibility of R2, proven at the tensor-storage level.
- A 76-checkpoint trajectory for the control arm.
- Per-stage MD5 integrity manifests for every stage that ran uncommitted.
- A terminal, negative-but-clean answer: **Δ_EDGE was differential suppression, not risk
  acquisition** — and the honest statement that the blocker is a **missing observable**.
- Two documentation artifacts: the 21-part tutorial and this inventory.

### INTENTIONALLY NOT DONE

- **No Sprint 8, no R4.** The Audit's verdict is *"no further training justified"*.
- **No held-out evaluation of R3** — reward was not a pre-registered criterion and the arm was
  already NO-GO on P1–P4.
- **No trajectory series for any arm other than R2.**
- **No fix to `mappo.py:447`'s stale horizon comment** — production source is frozen at
  2026-08-25 04:02 and a cosmetic edit would break every code manifest.
- **No edits to historical reports.** Corrections are additive (Audit §8).
- **No test of Sprint 6.5 §13's smaller-batch recommendation.**
- **No online centring of `c(s)`** — Mechanism 2 is closed; no such estimator exists.
- **No calibration of the risk signal** — it remains uncalibrated and near-bimodal by design of
  the leakage-safe OOF procedure.
- **No new branch, no commit of the artifacts.** The work belongs on `host-predictor-finalization`.

### DO NOT RERUN

- Any `marl.train` invocation. Every one overwrites `<tag>.pth`, `<tag>_best.pth`,
  `<tag>_config.json`, `<tag>_history.csv`, `<tag>_updates.csv`.
- Any diagnostic under its **default `--tag`** — the default is `main`, which is exactly the
  published artifact name.
- `_diag_rung2_75_matched_states.py` in any way that would regenerate the frozen RANDOM/UNION sets.
- Predictor training (`training/train_failure_predictor.py`, `training/host_cv.py`) — expensive
  and would replace the frozen risk signal.
- The Java simulation, if it would rewrite `predicted_risk.csv`.
- **Never rerun training "to see if it goes away."**

### SAFE TO INSPECT

- Every `.md`, `.txt`, `.json`, `.jsonl`, `.csv`, `.log` in the tree — all read-only.
- `python -m marl.tests_env` and `python -m marl.sanity_test` — no artifacts written.
- `md5sum -c` against any manifest, **from the manifest's capture directory**, with
  `set -o pipefail`.
- `git log`, `git status -sb`, `git diff` — all read-only.
- Loading any checkpoint with `torch.load` for inspection.

### FINAL SOURCES OF TRUTH

| For | The authority |
|---|---|
| Sprint 7 as a whole | `python-ai/saved_models/marl/SPRINT_7_FINAL_SYNTHESIS_AUDIT.md` |
| Any contradiction between reports | that Audit's §8 corrections table |
| The narrative, the reasoning, the how-to | `PROJECT_RESEARCH_TUTORIAL.md` |
| Where a file is and what it is | this document |
| Any arm's actual configuration | that arm's own `*_config.json`, key `config.mappo` |
| Any hyperparameter's frozen value | `python-ai/marl/config.py` |
| Whether an artifact is unmodified | the `_*integrity/` manifests |
| Checkpoint provenance in the trajectory | `R2_trajectory/SPRINT_7_P4_trajectory_manifest.jsonl` |

---

# 14. FUTURE CLAUDE QUICK START

Read these, in this order. The order is derived from the repository's own supersession chain:
the Audit supersedes the reports, and the reports supersede each other forward in time.

**Minimum to be oriented (stop here if the task is narrow):**

1. **This file** — `PROJECT_FILE_AND_OUTPUT_INVENTORY.md`. §1, §2, §12, §13.
2. **`python-ai/saved_models/marl/SPRINT_7_FINAL_SYNTHESIS_AUDIT.md`** — the closing verdict and
   the corrections table. **It supersedes the 11-report chain; do not reconstruct Sprint 7 from the
   individual reports.**
3. **`python-ai/saved_models/marl/SPRINT_7_PHASE5_REPORT.md`** — the terminal finding
   (Δ_EDGE = differential suppression; policy freeze at update 62).

**If the task touches mechanism or interpretation, add:**

4. `SPRINT_7_DIVERGENCE_REPORT.md` — why every endpoint instrument inverts, and why the blocker is
   a missing observable. This is the single most important methodological caution in the project.
5. `SPRINT_7_RUNG2_75_REPORT.md` — the definition of Δ_EDGE and the frozen RANDOM/UNION sets.
   Any new number must be measured on those sets or it is not comparable.
6. `SPRINT_7_RUNG1_REPORT.md` — the paired advantage estimator, the project's best result.

**If the task touches code, add:**

7. `python-ai/marl/config.py` (the `mappo` block) and `python-ai/marl/env.py` docstring.
8. `PROJECT_RESEARCH_TUTORIAL.md` Parts 3–7 for the apparatus, Part 18 for troubleshooting.

**If the task touches reproducibility or integrity, add:**

9. `SPRINT_7_PHASE4_PREREG.md` §13 (the N1/N2 `torch.save` blast-radius measurement) and
   `SPRINT_7_PHASE4_REPORT.md` (B1a–B1e).
10. `python-ai/saved_models/marl/_PHASE5_integrity/` — the most complete manifest set.

**Before doing anything, know these five things:**

- Sprint 7 is **closed**, verdict *"no further training justified."* Do not propose Sprint 8 or R4.
- **`--device cpu`** is mandatory; cuda breaks seed reproducibility.
- Hyperparameters are at **`config['mappo']`**, not `config['train']`.
- Diagnostics default to **`--tag main`**, which overwrites published artifacts.
- Corrections are **additive** — never edit a historical report.

### Unverified references (do not treat as facts)

| Item | Why it is unverified |
|---|---|
| The exact Maven command used to run the Java simulation | no build script, run script or documented invocation exists in the repository |
| The shell script that chained the Sprint 6.5 arms | `chain_cpu.log` and `chain_A2_A3.log` are **0 bytes**; the driver itself is absent |
| **"RULE 1–12"** | cited throughout the report corpus and **enumerated in no file** |

---

```
FILES INVENTORIED: 448
SPRINTS/RUNGS/PHASES INVENTORIED: 17
CHECKPOINT FAMILIES: 13
VERIFIED COMMANDS: 26
IMPORTANT REPORTS: 16
IMPORTANT PRODUCTION FILES: 16
UNVERIFIED REFERENCES: 3
```

*Counts: files = 44 Java + 4 simulation outputs + 15 predictor artifacts + 161 `saved_models/marl/`
root files + 77 `R2_trajectory/` files + 69 run logs + 16 production Python modules +
38 diagnostic scripts + 17 predictor-pipeline modules + 7 scratch/empty, across 12 additional
integrity-manifest directories. Checkpoint families = the 12 root families in §5.1 plus the
`R2_trajectory/` series. Verified commands = the 26 fenced commands in §7.
Important production files = 13 `marl/` modules + `sanity_test.py` + `tests_env.py` +
`data/failure_dataset.py`.*
