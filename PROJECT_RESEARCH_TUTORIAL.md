# DT-MARL-Healthcare — Project Research Tutorial & Reproduction Handbook

**Reconstructed from the repository, 2026-08-31.**
**Repository:** `D:\MY_GURL\RESEARCH_PAPER\DT-MARL-Healthcare`
**Branch at time of writing:** `host-predictor-finalization` · **HEAD:** `df75ce3` · **Working tree:** clean

> This document is **documentation only**. It does not modify production code, historical
> reports, or experiment outputs. It creates no new sprint, rung, or phase. Everything in it
> was read out of the repository — git history, run logs, config JSONs, CSV/JSON artifacts,
> checkpoints, integrity manifests, source code and the 15 historical Markdown reports.
>
> Where a fact could not be established from the repository it is marked
> **`NOT VERIFIED FROM REPOSITORY`**. Nothing has been invented to fill a gap.

---

## Part 0 — How to read this handbook

| If you are… | Read, in this order |
|---|---|
| Trying to remember what happened | Part 1 → Part 3 (the why-chain) → Part 21 (the dated table) → Part 20 (cheat sheet) |
| A project partner learning the codebase | Part 1 → Part 4 → Part 5 → Part 7 → Part 15 |
| Reproducing an experiment | Part 6 → Part 8 → Part 11 → Part 19 |
| Inspecting outputs | Part 9 → Part 10 → Part 14 |
| Checking the repo is intact | Part 11 → Part 12 → Part 18 |
| Writing the paper | Part 13 → Part 16 → Part 17 |

**Three conventions used throughout.**

1. **Verbatim vs reconstructed.** A command shown as *(verbatim from `FILE`)* was copied out of
   a report or log. A command shown as *(reconstructed)* was assembled from the script's
   `argparse` block plus the recorded `*_config.json`, and has **not** been quoted from any
   historical document. Only **six** commands appear verbatim anywhere in the `.md` corpus.
2. **Two source trees.** `simulation/` is Java (CloudSim Plus). `python-ai/` is Python
   (PyTorch). They communicate only through CSV files on disk.
3. **Names.** "Sprint" = a project phase. "Rung" = one step of the Sprint 7 diagnostic ladder.
   "Arm" = one trained policy (`A0`, `A1`, `A2`, `A3`, `R2`, `R3`, …). "Phase" (capital P, in
   Sprint 7) = a late-Sprint-7 stage that was numbered 0/1/2/4/4.1/5 rather than given a rung
   number.

**Contents.**

| # | Part | What it answers |
|---|---|---|
| 1 | Understand the project first | What is this system, in plain language and then technically? |
| 2 | The complete sprint / rung / phase timeline | Every stage, 14 fields each — plus a 12-row table of the corpus's own contradictions |
| 3 | "Why did we even do this?" | The full `observed → suspected → ran → showed → therefore` chain |
| 4 | Codebase map | Every major file: purpose, functions, inputs, outputs, safe to modify? |
| 5 | Every important concept | MARL, MAPPO, GAE, PPO, OOF risk, self-healing… and where each lives in the code |
| 6 | Configuration guide | Every parameter: what it does, what changes if you move it, who moved it |
| 7 | How training actually works | One full cycle, with real numbers from real checkpoints |
| 8 | Experiment command reference | Every command, marked verbatim or reconstructed, grouped by how dangerous it is |
| 9 | Artifact guide | Every file type: what created it, what it tells you, can it be regenerated |
| 10 | How to inspect a model checkpoint | And how to decide whether two checkpoints are *really* different |
| 11 | How to check experiment integrity | The MD5 methodology and its nine-step procedure |
| 12 | Git guide | Status codes, staged vs untracked, and how to push artifacts safely |
| 13 | Experiment result cards | The scannable verdict for every stage |
| 14 | Metric dictionary | Every metric, its formula, and how it can mislead you |
| 15 | Sprint 7, reconstructed as one chapter | Started with / suspected / tested / failed / reproduced / discovered / concluded |
| 16 | Self-healing: the actual final answer | Split into demonstrated / supported / partially answered / unknown |
| 17 | What we learned | Nine lessons, each tied to a specific event |
| 18 | Troubleshooting | 21 symptoms, each with a read-only diagnosis path |
| 19 | Reproducibility checklist | Before / during / after |
| 20 | One-page cheat sheet | The thing to keep open beside the terminal |
| 21 | Do not lose the history | The dated chronological table |

---

# Part 1 — Understand the project first

## 1.1 In simple language

Imagine a small hospital network with **ten edge computers** sitting near the patients — in
wards, on trolleys, next to bedside monitors — plus **one big cloud server** further away.
Medical software tasks run on those ten edge computers: analysing vital signs, scoring how
sick each patient is, watching for deterioration.

Edge computers **break**. They overheat, their network link drops, they wear out, they get
attacked. When a computer breaks while it is halfway through a patient's task, **that task is
lost**. If the task belonged to a critically ill patient, that is the worst possible outcome.

A **digital twin** is a live software model of the physical hospital network. It knows the
temperature, load, network state and wear of every machine. A **failure predictor** reads that
telemetry and, for every machine at every moment, produces a number between 0 and 1: *how
likely is this machine to fail in the next ten seconds?* That number is called **risk**.

The research question is: **can a set of learning agents use that risk signal to move patient
tasks off machines that are about to fail, before they fail?** That is what "**self-healing**"
means in this project — not repairing a broken machine, but *relocating work off a machine
that is predicted to break, in time*.

There are ten agents, one per edge computer. Every two seconds each agent looks at its own
machine and its neighbours and picks one of **four actions**:

| Action | Plain meaning |
|---|---|
| `STAY` | Leave the task where it is. Cheap, but exposed if the host dies. |
| `MIGRATE_EDGE` | Move the task to a **neighbouring edge computer**. Costs time and energy, keeps it near the patient. |
| `MIGRATE_CLOUD` | Move the task to the **cloud**. Safe from this host's failure, but far away and the cloud has limited slots. |
| `PREEMPTIVE_REROUTE` | Send a task that hasn't started computing yet somewhere else instead. |

The agents are trained by **reinforcement learning**: they try things, get a reward, and adjust.
The reward rewards finishing tasks (especially critical-patient tasks) and punishes losing them,
migrating unnecessarily, breaking deadlines and burning energy.

**The headline result, stated plainly.** The learned multi-agent policy *does* learn to protect
tasks — but a **one-line rule** ("if risk > 0.18, migrate") protects far more of them
(≈30 tasks per episode vs ≈3–11) and earns a far higher reward. The whole of Sprint 7 was spent
finding out **why** the learner cannot match a one-line rule. The answer it reached is honest
and partial, and Part 16 states it without inflation.

## 1.2 The technical version

### The two halves and the seam between them

```
  simulation/  (Java, CloudSim Plus 8.5.7)          python-ai/  (Python, PyTorch)
  ┌──────────────────────────────────────┐          ┌──────────────────────────────────────┐
  │ SimulationManager                    │          │ data/failure_dataset.py              │
  │  SIM_SEED = 20260817                 │  CSV     │   26 features, leakage assertions    │
  │  MAX_SIMULATION_SECONDS = 1500.0     │ ───────► │ training/train_failure_predictor.py  │
  │  DEVICE_COUNT = 10                   │          │   BiLSTM, 5 temporal folds, 80 ep    │
  │  DigitalTwinManager, CriticalityMgr, │          │   → failure_predictor_oof.npz        │
  │  FailureInjector, CloudletManager    │          └──────────────┬───────────────────────┘
  │  exports (SimulationManager:413-415) │                         │ out-of-fold risk
  │   failure_log.csv                    │                         ▼
  │   failure_history.csv                │          ┌──────────────────────────────────────┐
  │   device_failure_history.csv         │ ◄─────── │ marl/  DTMarlEnv + MAPPO             │
  │  reads simulation/predicted_risk.csv │  risk    │   trace-driven REPLAY of the above    │
  │   (PredictionGateways.DEFAULT_RISK…) │  seam    │   → saved_models/marl/*.pth           │
  └──────────────────────────────────────┘          └──────────────────────────────────────┘
```

**Critical honesty point, stated in the code itself.** The MARL environment is **not**
closed-loop co-simulation. `marl/env.py`'s module docstring:

> *"It is NOT closed-loop co-simulation with CloudSim… The trajectory is exogenous; what the
> agents change is where the healthcare tasks sit relative to it, and whether they survive."*

`marl/export_risk_csv.py` repeats it: the risk file is *"a REPLAY of scores computed over an
already-recorded trace, not closed-loop co-simulation."* So the failure timeline is **fixed**;
the agents cannot cause or prevent a host failure. They can only decide whether a task is
sitting on the host when it dies. This is a deliberate design choice and a stated limitation.

### The environment — `DTMarlEnv` (`python-ai/marl/env.py`, 1037 lines)

| Property | Value | Where |
|---|---|---|
| Agents | 10 edge nodes, one actor each | `config.py` |
| Extra tier | 1 cloud, `CLOUD_NODE_ID = -2`, `cloud_slots = 8` | `config.py`, `env.py` |
| Neighbourhood | ring, `neighbour_offsets = [-2, -1, 1, 2]`, degree 4 | `marl/topology.py` |
| Observation | **48** floats per agent | `env.py` |
| Centralised state | **489** floats (+ 10-dim agent one-hot → 499 into the critic) | `env.py`, `mappo.py` |
| Actions | 4 (`STAY`, `MIGRATE_EDGE`, `MIGRATE_CLOUD`, `PREEMPTIVE_REROUTE`) | `config.py` |
| Decision interval | 2.0 s = **2 recorded ticks** | `config.py` |
| Episode length | **400 steps = 800 simulated seconds** | `config.py` |
| Trace | 10 nodes × **1500 ticks** | `marl/trace.py` |
| Training start window | ticks **[9, 491]** (`train_frac 0.7`, first valid risk tick 9) | `config.py` |
| Eval window | ticks **[491, 698]**, 8 fixed starts `[491,521,550,580,609,639,668,698]` | `marl/rollout.py` |

`marl/rollout.py`'s docstring states why the eval starts are fixed:

> *"`run_episodes` always uses the SAME list of episode start ticks for every policy it is
> given… identical failure sequences, identical arrival schedule, identical initial placement.
> The only difference between two rows of the results table is the relocation decision."*

### Risk — what it is and what it is not

`marl/risk_provider.py` (198 lines) is emphatic:

> *"It is an UNCALIBRATED score in [0, 1]: no Platt scaling, isotonic regression or temperature
> fitting has been performed… It is therefore called `predicted_failure_risk` throughout, never
> "failure probability" and never "calibrated probability"."*

Three sources, selected by `--risk-source`:

- **`oof`** (default, and the one used for every research result) — leakage-safe out-of-fold
  sigmoid scores from `training/train_failure_predictor.py`. Every window scored by a model that
  never trained on that window's temporal block. Covers all 1491 windows × 10 nodes.
- **`model`** — live inference with `saved_models/failure_predictor.pth`, the deployment refit.
  **In-sample for ~4/5 of the trace**, therefore optimistic; flagged in the log when used.
- **`zero`** — risk channel forced to 0. This is the **ablation control** used to ask "does the
  policy actually use risk?"

Verified OOF distribution (from the training banner): `min 0.0145 · max 0.9585 · mean 0.0785 ·
203 distinct values · first valid tick 9`. Note that **only ~4.7% of decision states have
risk > 0.18** — the high-risk regime is a small minority, and that fact drives most of Sprint 7.

### The failure predictor

`saved_models/failure_predictor_meta.json`: BiLSTM, `sequence_length 10`, `num_features 26`,
deployment `threshold 0.18`, 80 epochs, 5 temporal folds.

The 26 features = 12 observable telemetry columns + their 12 first differences (`d_*`) +
`time_since_active` + `time_since_linkup`. `data/failure_dataset.py` enforces leakage discipline
with a hard whitelist and an assertion:

```
OBSERVABLE_COLUMNS  (12)   the whitelist
AUDIT_COLUMNS              audit_healthState, audit_wear, audit_nextFailureTime,
                           audit_secondsToFailure, audit_predictionHorizon,
                           degradationStartTime
FORBIDDEN_COLUMNS          AUDIT_COLUMNS + willFailSoon + time + nodeId
assert_no_leakage()        raises if any forbidden column reaches the model
```

`marl/trace.py` applies the same discipline on the MARL side: it hard-drops label and audit
columns (`keep = ["time","nodeId"] + TRACE_CHANNELS`) *before* anything touches the dataframe.

Pooled out-of-fold performance (`saved_models/host_metrics.json`):

| | value |
|---|---|
| n windows | 14,910 |
| positives | 320 (rate **0.02146**) |
| **PR-AUC** | **0.28634** (baseline 0.02146 ⇒ **13.3×** lift) |
| ROC-AUC | 0.81594 |
| accuracy / precision / recall / F1 | 0.9686 / 0.36972 / 0.65625 / 0.47297 |
| confusion | tn 14,232 · fp 358 · fn 110 · tp 210 |

Lead-time behaviour on the 32 real host failures (`saved_models/host_lead_time.csv`):
**23 EARLY_WARNING · 7 MISSED · 2 STANDING_ALARM**, lead mean 8.39 s / median 10.0 s, run-lead
mean 18.22 s. The retired 7-column "legacy" feature set produces **31 STANDING_ALARM + 1
MISSED** — i.e. an alarm that is always on, which is useless. That contrast is the reason the
feature set was rebuilt (see Sprint 4 in Part 2).

### The learner — MAPPO (`python-ai/marl/mappo.py`, 593 lines)

- **10 separate actors**, MLP `48 → [128, 128] → 4`, **23,300 parameters each**.
- **1 centralised critic**, MLP `499 → [256, 256] → 1`, **194,049 parameters**.
- **CTDE** — centralised training, decentralised execution: the critic sees the global state
  during training; each actor sees only its own 48-dim observation at execution time.
- `MAPPO.__init__` calls `torch.manual_seed(seed)` at **`mappo.py:218`**, which is why every arm
  starts from *bit-identically the same* initial parameters θ₀. This is what makes the arm ladder
  a clean comparison.
- Actions are **masked** (`masked_dist`, `mappo.py:77-79`), so an illegal action can never be
  sampled. No policy ever gets credit for an impossible move.

### The ultimate research question

Reconstructed from `run_mappo_eval.log`, `SPRINT_6_5_REPORT.md` and
`SPRINT_7_FINAL_SYNTHESIS_AUDIT.md`, the question the project actually asks is:

> **Does a digital-twin failure-risk signal, consumed by a multi-agent RL scheduler, produce
> genuine risk-conditioned self-healing behaviour — i.e. does the policy migrate *because* risk
> is high — and does it beat a fixed threshold on the same signal?**

The answer as of 2026-08-31 is in Part 16. Short version: the risk signal is genuinely
predictive; migrating at high risk is genuinely the right move (verified by exact counterfactual
replay, +2.671 team advantage); the learner's *risk-conditioned* behaviour improved substantially
across Sprint 7; **and** it still loses to the one-line rule on every task-level metric, and the
improvement in the Sprint 7 metric turns out to be mostly *low-risk suppression* rather than
high-risk acquisition.

---

# Part 2 — The complete sprint / rung / phase timeline

## 2.0 What is and is not verifiable about the numbering

Before the timeline, the honest statement about the sprint names, because the brief asked
specifically not to assume them:

- **Java source names** Sprint 1, 3, 4, 5, 6, 7, 8 in comments and class docstrings.
- **Python source names** Sprint 3, 3.5, 3.75, 4, 5, 6, 6.5, 7.
- **Sprint 8, 9 and 11 are forward references only** — named in docstrings as *not implemented*
  (e.g. `marl/baseline.py`: *"The full Sprint 9 baseline suite (HTCF, TGNN, heuristic schedulers,
  other RL algorithms) is deliberately not implemented here."*).
- **Sprint 0, Sprint 1 and Sprint 2 do not exist as discrete named, reported stages.**
  `NOT VERIFIED FROM REPOSITORY`. What *does* exist for that period is git commit
  `8feaa98` (2026-07-20, repository bootstrap) and commit `ea08805`
  ("Refactor data pipeline and implement HTCF baseline"), plus `datasets/` and `models/htcf.py`.
  Below they are documented as **S-early**, by what the artifacts show, not by a guessed number.
- **There is no `README.md` anywhere in the repository.** Every reconstruction below comes from
  code, logs, configs and the 15 reports.

### Git chronology (7 commits, complete)

| commit | date (`git log --date=iso`) | subject |
|---|---|---|
| `8feaa98` | 2026-07-20 09:08:15 +0530 | Initial project structure |
| `ea08805` | 2026-07-21 12:12:44 +0530 | Refactor data pipeline and implement HTCF baseline |
| `1490e3a` | 2026-07-25 11:59:01 +0530 | Add failure simulation and ClusterData2019 processing |
| `63e9bcb` | 2026-07-25 22:26:01 +0530 | Integrate digital twin failure prediction and early warning |
| `208e5a2` | 2026-07-29 05:52:40 +0530 | sprint 5 |
| `a8df6d9` | 2026-08-30 10:44:38 +0530 | Update project with latest implementation |
| `df75ce3` | 2026-08-31 04:46:06 +0530 | Complete Sprint 7 research and diagnostics *(current HEAD)* |

All seven commit dates are verified. Note the **month-long gap** between `208e5a2` (07-29) and
`a8df6d9` (08-30): the whole of Sprints 6, 6.5 and 7 happened **inside that gap, uncommitted**. That
single fact is the entire justification for the MD5 integrity methodology in Part 11 — for a month,
git could not tell you whether anything had changed. The **run-log modification times** are what date
those experiments, and they are used in Part 21.

---

## S-early — Data pipeline, ClusterData2019 and the HTCF baseline

| Field | Content |
|---|---|
| **Purpose** | Build a workload/telemetry data pipeline and a non-RL forecasting baseline. |
| **Research question** | Can host/task behaviour be modelled at all from trace data? |
| **Hypothesis** | `NOT VERIFIED FROM REPOSITORY` — no report survives for this stage. |
| **What was changed** | Introduced `data/preprocess.py`, `data/dataset.py`, `datasets/cluster/`, `models/htcf.py`, `models/transformer.py`. |
| **What was held constant** | n/a |
| **Experiment command** | `NOT VERIFIED FROM REPOSITORY` |
| **Inputs** | Google ClusterData2019 (per commit `1490e3a`). Note: `datasets/cluster/` is **empty at HEAD** — the raw data is not in the repository. |
| **Outputs** | `saved_models/htcf_model.pth` |
| **Primary metrics** | `NOT VERIFIED FROM REPOSITORY` |
| **Result** | HTCF checkpoint exists; no evaluation report survives. |
| **What it proved** | That a pipeline and a baseline model existed before the DT work. |
| **What it ruled out** | Nothing recorded. |
| **What remained unknown** | Everything about its performance. |
| **Why the next stage happened** | The project pivoted from cluster-trace forecasting to a **purpose-built digital-twin simulator**, which is what commits `1490e3a`/`63e9bcb` deliver. |

> **Honest caveat.** `datasets/`, `docs/`, `results/`, `python-ai/inference/` and
> `python-ai/utils/` are all **empty directories** at HEAD; `python-ai/config.py` and
> `python-ai/main.py` are **0-byte files**; `chain_cpu.log` and `chain_A2_A3.log` are **0-byte**.
> A stray 2,658-byte file named `da` sits at the repository root, and `temp.py` exists. None of
> these carry information. Do not read their existence as evidence of work.

---

## Sprint 3 / 3.5 / 3.75 — Simulator fidelity: failures, attacks, IoMT devices

| Field | Content |
|---|---|
| **Purpose** | Make the CloudSim Plus digital twin produce a *realistic, symptomatic* failure process. |
| **Research question** | Can host degradation be simulated so that it is *predictable from observable telemetry*? |
| **Hypothesis** | Latent wear + fault mechanisms produce leading indicators; instant scheduled death does not. |
| **What was changed** | The Java failure model. Documented in `data/failure_dataset.py`: the old model was *"instant death at a scheduled time, no symptoms"*; the new one has latent wear and fault mechanisms. Also network-link failures, a cyber-attack channel (`underAttack`), and 10 IoMT devices (`DEVICE_COUNT = 10`). |
| **What was held constant** | `SIM_SEED = 20260817L`, `MAX_SIMULATION_SECONDS = 1500.0`. |
| **Experiment command** | Java/Maven entry point `simulation/.../Main.java`; exact CLI `NOT VERIFIED FROM REPOSITORY`. |
| **Inputs** | Simulator configuration only. |
| **Outputs** | `simulation/failure_log.csv`, `failure_history.csv`, `device_failure_history.csv` (exported at `SimulationManager.java:413-415`, all with a **10.0 s label horizon**). |
| **Primary metrics** | Feature variance; failure/recovery counts. |
| **Result** | **32 failures and 32 recoveries across 10 nodes.** Under the old model, 5 of 12 candidate telemetry columns (cpu, ram, bandwidth, runningTasks, underAttack) were **constant** — zero variance, which caused a divide-by-zero in the scaler. Under the new model, **0 of 12 are constant**. |
| **What it proved** | The simulator now emits symptoms before failure, so prediction is a well-posed problem. |
| **What it ruled out** | That the earlier predictor's poor performance was purely a modelling failure — part of it was that the data contained no signal. |
| **What remained unknown** | Whether a sequence model could actually extract usable lead time. |
| **Why the next stage happened** | With a symptomatic failure process in place, the failure predictor could be built and honestly evaluated → Sprint 4. |

Additional recorded consequence: hosts now *"fail AND recover repeatedly"*, counters reset on
recovery, and the observed counter range is `0..44`. This retracted an earlier caveat that
assumed one-shot failures.

---

## Sprint 4 — The host failure predictor (and its finalisation, 2026-08-17)

This is the stage that set the project's methodological standard, and the one that produced the
"always compare against a one-line rule" habit that later exposed the MARL result.

| Field | Content |
|---|---|
| **Purpose** | Produce a leakage-free, honestly-evaluated per-host failure risk signal. |
| **Research question** | Can observable telemetry predict host failure ≥10 s ahead, better than trivial rules? |
| **Hypothesis** | A BiLSTM over a 10-step window of 12 observable channels + their first differences beats both trivial rules and the legacy 7-column feature set. |
| **What was changed** | (a) the feature set: legacy **7 raw columns** → **26 features** (12 observable + 12 `d_*` + `time_since_active` + `time_since_linkup`); (b) the dataset regenerated from the new symptomatic simulator. |
| **What was held constant** | Architecture (BiLSTM), `sequence_length 10`, 80 epochs, **5 temporal folds**, deployment `threshold 0.18`. |
| **Experiment command** | `python -m training.train_failure_predictor …` *(reconstructed — the recorded evidence is `run_host_observable.log`, `run_host_legacy.log`, `run_lono_observable.log`)* |
| **Inputs** | `simulation/failure_history.csv` (10 s label horizon). |
| **Outputs** | `failure_predictor.pth`, `_meta.json`, `_oof.npz`, `_scaler.npz`; plus `_legacy*` and `_OLDDATA*` variants; `host_metrics{,_legacy,_OLDDATA}.json`; `host_lead_time{,_legacy,_OLDDATA}.csv`; `host_lono.json`; `saved_models/_backup_pre_finalization/`. |
| **Primary metrics** | **PR-AUC** (chosen because positives are 2.1% — accuracy and ROC-AUC are both misleading here), F1, and **lead-time verdict counts**. |
| **Result** | **observable: PR-AUC 0.2863 / ROC 0.8159 / F1 0.4730 · legacy: PR-AUC 0.0777 / ROC 0.6250 / F1 0.0467 · OLDDATA: PR-AUC 0.2174 / ROC 0.9165 / F1 0.3439 but only n=1,680 with 39 positives.** Lead time: observable **23 EARLY_WARNING / 7 MISSED / 2 STANDING_ALARM**; legacy **31 STANDING_ALARM / 1 MISSED**. Trivial-rule baselines: always-negative F1 **0**; `linkUp==1` acc 0.0875 F1 0.0449; `degraded==1` precision 0.7333 recall 0.0344 F1 **0.0657**. Leave-one-node-out (`host_lono.json`) generalises (node 0: PR-AUC 0.4042, ROC 0.9260, F1 0.5634). |
| **What it proved** | The risk channel is real: 13.3× PR-AUC lift over base rate, genuine early warning on 23/32 failures, and it generalises to a held-out node. |
| **What it ruled out** | That the earlier weak predictor was an architecture problem (it was features + data), and that ROC-AUC alone is a safe metric here (OLDDATA has the *best* ROC and a far worse PR-AUC on a tiny sample). |
| **What remained unknown** | Whether the scores are **calibrated** (they are not — see `risk_provider.py`), and whether a *scheduler* can exploit them. |
| **Why the next stage happened** | A usable risk channel now existed, so it could be handed to a scheduler. Also, the audit habit — *"does this beat a one-line rule?"* — became a standing requirement, quoted later in `marl/baseline.py`. |

> **Known residual caveat** (recorded in project memory and consistent with the artifacts): the
> risk score distribution is **near-bimodal**, and it is **uncalibrated**. Any statement of the
> form "risk 0.6 means a 60% chance" is wrong.

---

## Sprint 5 — Healthcare workload, criticality, and the prediction gateway

Commit `208e5a2` is literally titled "sprint 5".

| Field | Content |
|---|---|
| **Purpose** | Make the workload *healthcare-shaped* — patients, vitals, deadlines, criticality — and open a seam for predictions to enter the simulator. |
| **Research question** | How should task importance be defined so that "protecting the right task" is measurable? |
| **Hypothesis** | A weighted criticality score over health-state index, vitals instability and age, with an urgency boost near the deadline, expresses clinical priority. |
| **What was changed** | Java `CriticalityManager` + `CloudletManager`; `PredictionGateways.DEFAULT_RISK_CSV = simulation/predicted_risk.csv`. |
| **What was held constant** | `PATIENT_COUNT = 10`. |
| **Experiment command** | Java simulator run; exact CLI `NOT VERIFIED FROM REPOSITORY`. |
| **Inputs** | Patient/task generation parameters. |
| **Outputs** | `simulation/run_out.log` (which later serves as the **ground truth for the Python port**). |
| **Primary metrics** | Priority values at t=0. |
| **Result** | Weights `HSI 0.60 / VITALS 0.30 / AGE 0.10`; `MAX_URGENCY_BOOST 0.5`; `URGENCY_HORIZON_SECONDS 300.0`; `BASE_DEADLINE_SECONDS 60.0`; `DEADLINE_SLACK_PER_TASK_SECONDS 5.0`. Generation is deterministic: `patientId = taskId % 10`, `hsi = patientId/10`, `vitalsInstability = 1 − hsi`, `age = 20 + 6·patientId`, `deadline = 60 + 5·taskId`. |
| **What it proved** | Criticality is well-defined and reproducible. |
| **What it ruled out** | Randomised, unauditable task importance. |
| **What remained unknown** | Whether a learner would use criticality at all. |
| **Why the next stage happened** | The Python side needed the *same* criticality function to train on, which produced `marl/criticality.py` — a **verbatim port** validated against `JAVA_PRIORITIES_T0 = [0.448, 0.495, 0.542, 0.589, 0.634, 0.679]` from `simulation/run_out.log`, with `self_check()` agreeing to **< 5e-4**. |

> **Consequence that mattered later.** The real support of the severity variable is
> **[0.320, 0.644]** (`0.036·(task_id % 10) + 0.32`), not [0, 1]. Sprint 6.5 §8 found that
> probing criticality on a 0→1 grid is therefore **~3× too wide**, which weakened one of its
> own diagnostics.

---

## Sprint 6 — MAPPO: the first learned scheduler (2026-08-18)

### 6a. The hyperparameter search

Six dated runs on 2026-08-18, all recoverable from their banners:

| run log | γ | λ | lr_actor | ent | rollout_eps | episodes | outcome |
|---|---|---|---|---|---|---|---|
| `run_mappo_train_attrbug_gamma99.log` | 0.99 | 0.95 | 3e-4 | 0.01 | 4 | 400 | **COMPLETED**, 938 s, greedy TRAIN **−146.55** |
| `run_mappo_train_attrbug_gamma95_killed.log` | 0.95 | — | — | — | — | — | **killed at ep 100** (degrading −90 → −170) |
| `run_mappo_train_attrfix_gamma95_killed.log` | 0.95 | — | — | — | — | — | **killed at ep 175** |
| `run_mappo_train_g999_lowEV_killed.log` | 0.999 | — | — | — | — | — | **killed at ep 75** (explained variance only +0.37) |
| `run_mappo_lam95_killed.log` | — | 0.95 | — | — | — | — | **killed at ep 350** (reward −24 → −9.45, ev 0.92) |
| `run_mappo_train.log` | **0.999** | **0.995** | **7e-4** | **0.02** | **8** | **600** | **COMPLETED**, 1460 s → the frozen production values |

The surviving configuration is recorded in `mappo_config.json` and never changed again except
for the single Rung 2 edit. `mappo_attrbug_gamma99_config.json` preserves the abandoned regime.
The name "attrbug"/"attrfix" refers to an **attribution bug** fixed between runs;
its exact content: `NOT VERIFIED FROM REPOSITORY` beyond the filenames.

### 6b. The Sprint 6 result — `run_mappo_eval.log`

| policy | reward | success | lost | critLo | reloc | **protected** |
|---|---|---|---|---|---|---|
| **MAPPO (greedy)** | **20.62** | 0.719 | 11.2 | 4.4 | 35.9 | **3.2** |
| **risk-threshold @ 0.18** | **76.50** | **0.847** | **3.4** | **0.8** | 36.2 | **30.2** |
| reactive-threshold | −0.09 | — | — | — | — | — |
| random-legal | −8.11 | — | — | — | — | — |
| static-no-migration | −37.06 | 0.459 | — | — | 0 | — |

Behaviour probe (302 real decision-time observations): sweeping risk **0.00 → 0.99** moves
`P(relocate)` from **0.191 → 0.195** — **span 0.004**, correlation **+0.990**. Empirical
buckets: risk 0.0–0.2 → n=10,200, 8.4% relocate; risk 0.6–0.8 → n=246, 12.2%; risk 0.8–1.0 →
n=300, **9.7%**. Action breakdown: edge 9.62 / cloud 26.25 / **preemptive reroutes 0.00** /
infeasible 79.00.

**The finding that started everything:** the *risk → 0 ablation* **improves** reward,
20.62 → **23.13**, and barely moves protections (3.25 → 3.12).

| Field | Content |
|---|---|
| **What it proved** | MAPPO beats do-nothing (−37.06), random (−8.11) and reactive (−0.09). It learned *something*. |
| **What it ruled out** | Nothing yet — this is the observation, not a diagnosis. |
| **What remained unknown** | Why the policy is risk-*insensitive* (span 0.004) and why deleting the risk input helps. |
| **Why the next stage happened** | A learned policy that loses to a one-line threshold by **−55.9 reward** and protects **3.2 vs 30.2** tasks is a research problem, not a result. → Sprint 6.5. |

---

## Sprint 6.5 — Diagnosis, and four controlled arms (2026-08-19)

Primary source: **`SPRINT_6_5_REPORT.md`** (239 lines). Its own status line:

> *"diagnosis complete and verified. The fix objective was NOT met. Nothing committed, nothing
> pushed, no PR. Sprint 7 not started."*

### 6.5a. The hypothesis space, and what measurement did to it

| tag | hypothesis | verdict |
|---|---|---|
| A | risk is underrepresented in the observation | **refuted by measurement** |
| B | criticality is drowned out | **refuted** |
| C / D | reward doesn't reward preemption / rewards migration regardless of risk | **refuted** |
| J | architecture is inadequate | **refuted** (see the BC capacity probe) |
| K | the baseline has a shortcut | **refuted** |
| **F** | the high-risk region is **sparse** | **supported** |
| **G** | the PPO update goes to zero | **supported** |
| **H** | credit assignment over a task's whole lifetime | **supported** |

Root cause named: **learning dynamics — not representation, not reward specification.**

### 6.5b. The four load-bearing measurements

1. **Counterfactual truth by exact replay.** `A(MIGRATE_EDGE | risk > 0.18) = +2.671 ± 0.806 SE`
   vs `A(MIGRATE_EDGE | risk ≤ 0.18) = −1.337 ± 0.967`. **Gap +4.01. Zero replay mismatches.**
   Migrating at high risk really is right; migrating at low risk really is wrong.
2. **Capacity ceiling by behaviour cloning.** Cloning risk-threshold@0.18 *into the same actor
   network* reaches accuracy **1.0000** and a risk-sweep span of **0.3935** — versus the trained
   policy's **0.004**. The network can express the target function ~100× more sharply than
   training found. Architecture is exonerated.
3. **The learner's own value estimate is wrongly signed where it matters.** Report's words:
   *"−0.266 at high risk against a true +2.671."*
4. **Reward sizing is correct over the horizon.** One migrating step costs −0.6443 vs staying
   −0.0143, but exposure is charged over ~150 steps, so migrating wins by **+8.267**.
   Consequence: the planned `P_risk_expose` rescaling arm was **dropped** as attacking a
   non-problem.

### 6.5c. Two real defects found and fixed

1. **Contention was decided by agent index.** A fixed apply order `0..n−1` meant the
   lowest-index agent won **7/7** contests. Rotating by `step_idx` gives **3/21**. The report is
   careful: *"This does not reduce the refusal rate (~70% before and after) — it redistributes
   it."*
2. **`w_criticality_migration` was dead code.** `ev["severity"]` was never written in
   `_relocate`, so `crit_m ≡ 1.0` and the migration charge was pinned at `P_migration = 4.000`.
   A new `migration_severity` field makes it span **5.280–6.576** for severity 0.320–0.644.

And the discipline note: *"Neither change adds any `if risk > threshold` logic."*

### 6.5d. Contention is inherent

**545 / 783 = 69.6%** of relocation attempts are refused — **all of them cloud**, because
`cloud_slots = 8` for 10 agents. A forced cloud return is exactly
`−0.050 = P_infeasible × reward_scale`. A1 attempted cloud **1,060** times (~77% refused, true
advantage −0.050) and edge only **10** times (true advantage +2.671). `PREEMPTIVE_REROUTE` is
**0 in every arm**.

### 6.5e. The four arms and the reading rule

| arm | manipulation vs A1 | reward (±sd) | risk-thr baseline | gap | protected | reloc |
|---|---|---|---|---|---|---|
| **A0** | Sprint 6 defects **re-enabled** (`legacy_fixed_apply_order=True`, `legacy_dead_migration_criticality=True`) | 20.62 ± 6.16 | 76.50 | −55.9 | 3.25 ± 2.22 | — |
| **A1** | both legacy switches **off** (the two bug fixes) | 15.69 ± 7.26 | 67.31 | −51.6 | 3.25 ± 2.17 | — |
| **A2** | A1 + `w_criticality_migration = 0.0` | 12.33 ± 19.39 | 76.54 | −64.2 | 7.25 ± 2.95 | 67.8 |
| **A3** | A1 + `entropy_coef = 0.05` | 6.82 ± 12.87 | 67.31 | −60.5 | 11.12 ± 2.15 | 55.0 |

> **The reading rule, quoted verbatim from §6, and it matters for every later comparison:**
> *"**Reward is not comparable across all four arms.** … A0/A2 sit on one scale
> (risk-threshold@0.18 ≈ 76.5) and A1/A3 on a stricter one (67.31). `static-no-migration` is an
> exact internal control at −37.06 in every arm, and risk-threshold's behaviour is
> byte-identical everywhere (reloc 36.2, prot 30.2, lost 3.4) — only its reward moves. Compare
> arms on the reward-independent columns and on the within-arm gap."*

**A0 reproduces Sprint 6 bitwise**: 600/600 episodes and 75/75 PPO updates identical, held-out
reward +20.621 ± 6.159 against the recorded 20.62 ± 6.159, action histogram
**9832 / 77 / 842 / 0** vs **9832 / 77 / 842 / 0**.

### 6.5f. What the arms established

- **Risk sensitivity (sweep span):** A0 0.004 (r +0.990) · **A1 0.016 (+0.999)** · A2 0.004
  (+1.000) · A3 0.002 (+0.974) · **BC ceiling 0.3935**.
  Verbatim: *"A1 improved sensitivity 4× and remains ~25× short of what the same network
  demonstrably can express."*
- **Criticality direction is WRONG** in A0/A1/A2 (−0.987 / −0.989 / −0.993). A2 existed
  specifically to test the criticality-scaled migration charge and *"**It did not**"* fix it.
  A3 flips the sign to +0.801 but at span 0.001 — i.e. nothing.
- **Zero-risk ablation:** protections A0 3.25→3.12 · A1 3.25→2.88 · A2 7.25→7.00 · A3
  11.12→**10.25**. Verbatim: *"A3 keeps 92% of its protections with **no risk input at all**,
  while its relocations rose from 29.1 to 55.0. **It protects by volume, not by targeting.**"*
- **Tests:** `tests_env.py` **19/19**, `sanity_test.py` **6/6**; T6 matches to 4.48e-07; and the
  self-aware note *"T2 is a **weak guard**."*
- **Eight stated limitations:** one seed; ±6–19 reward SD; `PREEMPT` never usable because its
  mask requires a task that has not begun computing; ~70% cloud refusal; narrow severity
  support; replay not closed-loop; uncalibrated OOF; two CUDA arms not comparable.

| Field | Content |
|---|---|
| **What it proved** | The failure is in *learning dynamics*. The signal is real (+4.01), the network is capable (0.3935), the reward is correctly sized (+8.267), the observation is adequate. |
| **What it ruled out** | Hypotheses A, B, C, D, J, K — six of the eight candidate explanations. |
| **What remained unknown** | *Which* dynamical defect. §12 verbatim: *"As a diagnosis, yes. … As a fix, no. MAPPO is still risk-insensitive and still loses to a one-line threshold rule."* |
| **Why the next stage happened** | Sprint 6.5 §13 listed six Sprint 7 proposals. Note item 1 proposed **many more updates with a smaller `rollout_episodes`** — R3 later tested the **opposite** (32). Sprint 7 opened. |

---

## Sprint 7 — the diagnostic ladder (2026-08-19 → 2026-08-31)

Sprint 7 is eleven stages, not one experiment. Its own closing document
(`SPRINT_7_FINAL_SYNTHESIS_AUDIT.md`, 59,913 bytes, 2026-08-31 04:39) is the entry point and
supersedes reading the chain; the per-stage detail below exists so the *reasoning* is recoverable.

**The rung-history table, quoted from `SPRINT_7_PHASE0_RECONSTRUCTION.md` §1.2:**

| rung | date | question | verdict |
|---|---|---|---|
| Phase 1 diag | 08-19 | why does MARL lose to a risk threshold rule? | scoped Sprint 7 |
| Rung 0 | 08-20 | is the high-risk migration signal real, and does GAE see it? | signal **real** (+3.363 forced-EDGE true adv); GAE **anti-correlated** (29.0% sign agreement at risk>0.50) |
| Rung 1 | 08-25 02:36 | is the self-referential λ-return critic target causal? | **SUPPORTED for A0** (26.3% → 60.7% sign agreement offline) |
| Rung 2 | 08-25 04:02 | production change `critic_target="mc"` | R2 trained; best arm to date |
| Rung 2.5 | 08-25 06:45 | is the *remaining* target dependence causal? | **NO-GO for a target-focused rung**; separated **three** mechanisms |
| Rung 2.75 | 08-25 13:08 | is the headline metric identified? what is the actor stall? | headline metric **not identified**; stall attributed to **signal variance** |
| R3 | 08-25 18:22 | H1: is gradient variance / low SNR the binding constraint? | **NO-GO** — SNR rose (p=0.0000), behaviour got worse |
| Rung 3 | 08-27 13:16 | does low-risk dilution explain R3? | **SUPPORTED but DOES NOT EXPLAIN R3**; `go_for_training_rung = False` |
| Divergence | 08-30 11:02 | what R2-vs-R3 difference *could* explain the behaviour gap? | regime-selective learning asymmetry — explicitly *"what differs", not causal* |

**And the arm state table, §1.1** (Δ = the Sprint 7 primary metric, defined below):

| arm | checkpoint | manipulation | updates | Δ RANDOM | Δ UNION |
|---|---|---|---|---|---|
| A0 | `mappo_A0_cpu_repro.pth` | baseline, `critic_target=lambda` | 75 | −0.0264 | +0.0191 |
| A1 | `mappo_A1_cpu_bugfix.pth` | attribution bugfix | 75 | +0.1375 | +0.0628 |
| A2 | `mappo_A2_crit_sign.pth` | criticality-charge off | 75 | +0.0398 | −0.0248 |
| A3 | `mappo_A3_entropy.pth` | `entropy_coef` ↑ | 75 | +0.1322 | +0.0933 |
| **R2** | `mappo_R2_mc_target.pth` | **`critic_target` lambda→mc** | 75 | **+0.2024** | **+0.1579** |
| R3 | `R3_batch32.pth` | **`rollout_episodes` 8→32** | 75 | +0.1575 | +0.1242 |
| R3_best | `R3_batch32_best.pth` | update-45 snapshot of R3 | 45 | +0.1467 | +0.0946 |

**Behavioural order on both fixed sets: R2 > R3 > A0.**

> ### Two documented errors in that table — surfaced, not fixed
> The brief forbids silently correcting historical reports, so both stay in the record:
> 1. **§1.1 says A3's manipulation is `entropy_coef` 0.01→0.02. Both endpoints are wrong.**
>    `mappo_A3_entropy_config.json → config.mappo.entropy_coef = **0.05**`, against **0.02**
>    everywhere else. `SPRINT_7_RUNG2_75_REPORT.md` §7 (H4) has it right.
> 2. **§1.1 labels A2 "critic sign". It is *criticality*, not *critic*.** The manipulation is
>    `w_criticality_migration: 1.0 → 0.0` (confirmed in both the config JSON and
>    `SPRINT_6_5_REPORT.md` §6). The checkpoint filename `mappo_A2_crit_sign.pth` is the source
>    of the confusion.

### The Sprint 7 primary metric

```
Δ_EDGE  =  π(MIGRATE_EDGE | risk ≥ 0.6)  −  π(MIGRATE_EDGE | risk < 0.2)
```

evaluated over **all decision entries** of two **frozen, policy-independent** state sets:

- **RANDOM** — 4,975 entries, 216 high-risk
- **UNION** — 74,237 entries, ~3,592 high-risk

plus `Spearman(risk, π(EDGE))`. Both sets are constructed by
`marl/_diag_rung2_75_matched_states.py`. Freezing them is what makes cross-arm comparison valid:
the states do not depend on the policy being scored.

---

### Sprint 7 · Phase 1 diagnosis (2026-08-19 21:31) — `SPRINT_7_PHASE1_DIAGNOSIS.md`

| Field | Content |
|---|---|
| **Purpose** | Locate Sprint 6.5's "the PPO update goes to zero" one level deeper. |
| **Research question** | Is the dead gradient a *stalled* optimiser or a *converged* one? |
| **Hypothesis** | Verbatim: *"PPO is converging correctly onto an advantage estimate whose action ordering at high risk is inverted relative to the truth. The dead gradient at update 60+ is what convergence looks like. Not a stalled optimiser — a converged one, pointed the wrong way."* |
| **What was changed** | **Nothing.** Status line: *"inspection only. No code modified, no training run, no artifact overwritten."* |
| **What was held constant** | Everything. |
| **Command** | Read-only inspection of existing artifacts. |
| **Inputs** | `SPRINT_6_5_REPORT.md`; `marl/{train,mappo,env,config,evaluate,_diag_sensitivity,_diag_counterfactual}.py`; the four A-arm checkpoints; `diag_sensitivity.json`; `diag_counterfactual_A1.json`; `failure_predictor_oof.npz`. |
| **Outputs** | The report only. |
| **Primary metrics** | true Δteam by exact replay vs the policy's own normalised GAE, per action, at risk > 0.18. |
| **Result** | **Truth:** `EDGE +2.671 ± 0.806 > STAY 0 > CLOUD −0.050 ± 1.5e−8`. **Estimator:** `CLOUD +0.080 > STAY +0.044 > EDGE −0.266`. **Exactly inverted.** And greedy A0 plays `MIGRATE_CLOUD` in **842 of the 855 states where cloud is legal (98.5%)** and `MIGRATE_EDGE` in **77 of 10,751 (0.7%)** — i.e. it has faithfully learned *"cloud whenever legal, else stay"*, a rule with no risk term, built on a **−0.050 pseudo-no-op**. |
| **What it proved** | The policy is the exact argmax of its own (inverted) estimator. Nothing is broken in PPO's optimisation; the *target* is wrong. |
| **What it ruled out** | "The optimiser stalled." |
| **What remained unknown** | Whether the inversion is real: the report flags its own caveat up front — the high-risk GAE cells rest on **n=4 and n=9**, so *"The inversion is a point estimate, not a significant result."* |
| **Why the next stage happened** | *"Establishing or refuting it is rung 0 of the ladder, because everything downstream depends on it."* |

> The Phase 1 report also proposed a full ladder **B1–B3, C1–C4, E1–E3** with eight new CLI flags.
> **None of those arms exist as artifacts.** Only "Rung 0" survived by name. The proposed ladder
> is a plan, not a record — do not read it as history.

---

### Sprint 7 · Rung 0 (2026-08-20) — is the signal real, and does GAE see it?

| Field | Content |
|---|---|
| **Purpose** | Establish or refute the Phase 1 inversion at adequate n. |
| **Research question** | (D1/D2) Is the high-risk migration advantage real on a large sample? (D3/D4) Does the learner's own GAE agree with it? |
| **Hypothesis** | Signal real; GAE anti-correlated with it at high risk. |
| **What was changed** | Nothing in production. New read-only probe `marl/_diag_rung0.py`. |
| **What was held constant** | All hyperparameters; the A0 checkpoint. |
| **Command** | `python -m marl._diag_rung0 …` *(reconstructed; evidence = `run_S7_rung0_d1d2_train.log`, `run_S7_rung0_d3d4.log`)* |
| **Inputs** | `mappo_A0_cpu_repro.pth`, the OOF risk trace. |
| **Outputs** | `_rung0_integrity/` manifests; Rung 0 diagnostic JSONs. |
| **Primary metrics** | forced-EDGE **true discounted advantage**; `frac_positive(EDGE > STAY)`; GAE **sign agreement**. |
| **Result** | Signal **real**: forced-EDGE true advantage **+3.363** at risk > 0.50; `frac_positive` **0.7012** at risk > 0.50 vs **0.3931** at risk < 0.10. GAE **anti-correlated**: **29.0%** sign agreement at risk > 0.50 — *below chance*. |
| **What it proved** | The premise of the whole sprint. Migrating at high risk is right, on a large sample, and the learner's estimator disagrees. |
| **What it ruled out** | That the Phase 1 inversion was an n=4/n=9 artifact. |
| **What remained unknown** | *Why* GAE is inverted. |
| **Why the next stage happened** | The most suspicious candidate was the **critic target**: A0 used a **self-referential λ-return** target (`critic_target="lambda"`), i.e. the critic is fit toward a quantity computed from its own predictions. → Rung 1. |

---

### Sprint 7 · Rung 1 (report 2026-08-25 02:36) — the λ-return critic target

| Field | Content |
|---|---|
| **Purpose** | Test whether the self-referential λ-return target causes the inversion. |
| **Research question** | If the critic is refit against a **Monte-Carlo** target instead, does sign agreement recover — *offline*, without retraining? |
| **Hypothesis** | Yes; the λ-return target contaminates its own regression target. |
| **What was changed** | Nothing in production. Offline refit only (`marl/_diag_rung1_critic.py`). |
| **What was held constant** | The A0 policy; the state set; the advantage estimator. |
| **Command** | *(reconstructed; evidence = `run_S7_rung1_fit.log`, `run_S7_rung1_deviations.log`)* |
| **Outputs** | `_rung1_integrity/` manifests; Rung 1 JSONs. |
| **Primary metrics** | GAE sign agreement before/after an offline MC refit; residual swing. |
| **Result** | **SUPPORTED for A0: 26.3% → 60.7%** sign agreement. Residual analysis: a **−5.24** swing attributable to the target vs **−1.07** attributable to the per-state value offset. |
| **What it proved** | The critic target formulation is a **real, large, first-order** defect. |
| **What it ruled out** | That the inversion was irreducible. |
| **What remained unknown** | Whether fixing it *in training* changes behaviour (offline refit ≠ retraining). |
| **Why the next stage happened** | This is the only Sprint 7 finding strong enough to justify a **production change**. → Rung 2. |

---

### Sprint 7 · Rung 2 (production change 2026-08-25 04:02) — `critic_target="mc"`, arm R2

| Field | Content |
|---|---|
| **Purpose** | Apply the Rung 1 fix in training and measure behaviour. |
| **Research question** | Does `critic_target="mc"` produce risk-conditioned behaviour? |
| **Hypothesis** | Yes — with a clean target the advantage ordering at high risk should correct. |
| **What was changed** | **Exactly one production value:** `critic_target: "lambda" → "mc"`. This is the **only** change to `marl/{config,mappo,train}.py` in the whole of Sprint 7; the source is byte-identical in all eight subsequent manifests and at HEAD. |
| **What was held constant** | `gamma 0.999 · gae_lambda 0.995 · clip_eps 0.2 · value_clip_eps 0.2 · entropy_coef 0.02 · value_coef 0.5 · max_grad_norm 0.5 · ppo_epochs 4 · minibatches 4 · normalise_advantages True · anneal_lr True · separate_actors True · episodes 600 · rollout_episodes 8 · seed 20260818 · lr_actor 7e-4 · lr_critic 1e-3 · device cpu` |
| **Command** | `python -m marl.train --critic-target mc --episodes 600 --rollout-episodes 8 --seed 20260818 --device cpu --tag R2_mc_target` *(reconstructed from `argparse` + `mappo_R2_mc_target_config.json`; evidence = `run_R2_mc_target_train.log`)* |
| **Inputs** | The trace + OOF risk; θ₀ from `seed 20260818`. |
| **Outputs** | `mappo_R2_mc_target.pth`, `_best.pth`, `_config.json`, `_history.csv`, `_updates.csv`, `_eval.json`; `_smoke_rung2_mc_target.py` pre-run smoke test. |
| **Primary metrics** | Δ_EDGE on RANDOM/UNION; held-out reward. |
| **Result** | **Best arm on Δ: +0.2024 RANDOM / +0.1579 UNION.** But held-out greedy reward **−0.46** against a risk-threshold baseline of **67.31**, and `protected 9.9` vs the rule's `30.2`. |
| **What it proved** | The Rung 1 fix transfers to training and is the single largest behavioural improvement in the project. |
| **What it ruled out** | Nothing new. |
| **What remained unknown** | Two things. (a) Whether the *remaining* target dependence matters. (b) Why the best Δ arm has the worst reward. |
| **Why the next stage happened** | → Rung 2.5, to ask whether there is any target-side headroom left. |

> **`R2_best` is bit-identical to `R2`** — same `‖Δθ_actor‖ 6.1710`, same Adam state (1392 steps,
> `lr 9.3333e-06`). **The shipped R2 corpus therefore contains exactly one distinct policy
> snapshot.** This became the central obstacle of the sprint, and the reason Phase 4 exists.

---

### Sprint 7 · Rung 2.5 (2026-08-25 06:45) — is the residual target dependence causal?

| Field | Content |
|---|---|
| **Purpose** | Decide whether another target-focused rung is worth training. |
| **Research question** | How much contamination is left after `mc`, and is it actionable? |
| **Hypothesis** | Some residual remains; whether it is large enough to matter is the question. |
| **What was changed** | Nothing. Probes: `_diag_rung2_5_{targets,signtest,native_dev,actor_stall,feasibility}.py`. |
| **Command** | *(reconstructed; evidence = the six `SPRINT_7_RUNG2_5_*.log` files)* |
| **Primary metrics** | zero-censoring count; target contamination fraction; sign test. |
| **Result** | **130/130 zero-censored.** Target contamination **58.4% → 7.8%** — an **87% reduction** by the `mc` change. Verdict: **NO-GO for a target-focused rung.** The rung's real product was **separating three distinct mechanisms**: |
| | **1. critic target formulation** — real and severe for A0, 87% fixed by R2. |
| | **2. per-state critic value calibration** — a per-state offset `c(s)` in the advantage. |
| | **3. actor softmax saturation** — the actor's output distribution collapsing. |
| **What it proved** | The target channel is essentially exhausted (≤3.5pp headroom, at chance). |
| **What it ruled out** | A fourth target-focused training arm. |
| **What remained unknown** | Which of mechanisms 2 and 3 binds. Also: §G.5 leaves R2's **high-risk EDGE-share direction formally UNRESOLVED**. |
| **Why the next stage happened** | → Rung 2.75, to (a) check the headline metric is even identified and (b) characterise the actor stall. |

---

### Sprint 7 · Rung 2.75 (2026-08-25 13:08) — the paired estimator, and Δ_EDGE is born

| Field | Content |
|---|---|
| **Purpose** | Audit the metric, and decompose the advantage signal. |
| **Research question** | Is the headline metric identified? What exactly is the actor stall? |
| **Hypothesis** | The per-state offset is the residual problem. |
| **What was changed** | Nothing in production. New probes: `_diag_rung2_75_{coherence,edgeshare,edgeshare_cluster,edgeshare_power,matched_states,mbtail,offset,plasticity,stepcollapse}.py`. |
| **Command** | *(reconstructed; evidence = the nine `SPRINT_7_RUNG2_75_*.log` files)* |
| **Outputs** | The **frozen RANDOM and UNION state sets** — the sprint's most durable artifact — plus the coherence/offset/plasticity JSONs. |
| **Primary metrics** | The **paired advantage estimator** `gae(s,a) = c(s) + paired(s,a)` with `c(s) := gae(s, a_ref)` and `a_true(s, a_ref) ≡ 0`; `Var(raw)/Var(paired)`; sign agreement; **Δ_EDGE** (introduced here). |
| **Result** | **`Var(raw)/Var(paired) = 3.72×`.** Pairing lifts high-risk sign agreement **from below chance to 0.71 (team) / 0.95 (own) for every arm**. §A.4 **withdraws the previous headline metric as unidentified**, replacing it with Δ_EDGE on frozen sets. Actor stall attributed to **signal variance** from the offset. |
| **What it proved** | The sign "error" is a **per-state V(s) offset**, not a defect in the target. Remove the offset by pairing and the learner's signal agrees with the truth almost perfectly. |
| **What it ruled out** | The prior headline metric; and the framing "the estimator is biased". |
| **What remained unknown** | Whether variance is the *binding* constraint. The argument was: a per-state constant contributes zero in expectation, so it is variance, not bias; and 3.72× is *"the exact data multiplier needed."* |
| **Why the next stage happened** | That last sentence is a directly testable prediction: supply ~4× the data. → R3. |

Also measured here: the **minibatch tail** arithmetic. `mb_size = T // 4`, then `range(0, T, mb_size)`
yields **5** chunks whenever `T mod 4 ≠ 0`, and the fifth has exactly `T mod 4` rows (1–3). A
degenerate tail fires in ~3/4 of updates and contributes 4 of 20 optimiser steps. It is a
**shared, non-differential confound** — never a per-arm mechanism.

---

### Sprint 7 · R3 (2026-08-25 17:37 train / 18:22 report) — 4× the batch

| Field | Content |
|---|---|
| **Purpose** | Test Rung 2.75's prediction directly. |
| **Research question** | **H1:** is gradient variance / low SNR the binding constraint? |
| **Hypothesis** | 4× the rollout data raises SNR and improves risk-conditioned behaviour. |
| **What was changed** | `rollout_episodes 8 → 32`, with `episodes 600 → 2400` as the **dependent** change that holds `n_updates` at 75 (`600/8 = 2400/32 = 75`), verified as 75 rows in both `_updates.csv`. |
| **What was held constant** | Everything else, including `seed 20260818` and `critic_target "mc"`. |
| **Command** | **`python -m marl.train --critic-target mc --rollout-episodes 32 --episodes 2400 --seed 20260818 --device cpu --tag R3_batch32 2>&1`** *(**verbatim** from the `.md` corpus — one of only six)* |
| **Outputs** | `R3_batch32.pth`, `_best.pth`, `_config.json`, `_history.csv`, `_updates.csv`; `SPRINT_7_R3_action_channels_R3.json`; `SPRINT_7_R3_REPORT.md`. |
| **Primary metrics** | gradient SNR (real vs shuffled), permutation p; Δ_EDGE. |
| **Result** | **SNR really did rise** — real/shuffled **2.03**, **p = 0.0000**. **And behaviour got worse:** Δ RANDOM **+0.1575** vs R2's +0.2024, Δ UNION +0.1242 vs +0.1579. Verdict **NO-GO**. |
| **What it proved** | Variance is **not** the binding constraint. The intervention succeeded on its mechanism and failed on its outcome — the cleanest kind of falsification. |
| **What it ruled out** | H1; and, as Phase 0 §4 later argued, **mechanism 2 entirely**, since the offset's *only* causal pathway was variance. |
| **What remained unknown** | Why more, better-conditioned data makes risk-conditioned behaviour worse. The successor guess was **low-risk dilution**. |
| **Why the next stage happened** | → Rung 3, to test dilution. |

> **Two important R3 facts.** (1) **R3 was never evaluated on the held-out baseline suite** —
> there is no `run_R3_*_eval.log` and no `R3_batch32_eval.json`. R3's task-level performance
> against risk-threshold is therefore `NOT VERIFIED FROM REPOSITORY`. (2) **"R3's reward peaked
> at update 45 then declined" is FALSE** — an artifact of `_best` selection on a single noisy
> 32-episode mean. R3's block means improve monotonically across all four quartiles
> (−16.11 → −12.16 → −10.59 → −8.51). Only **A0** genuinely peaks then drops. `R3_best` is a
> genuine update-45 checkpoint (`lr 2.8933e-04` ⇒ `lr_scale 0.41333`, 872 Adam steps vs 1440).

---

### Sprint 7 · Rung 3 (2026-08-27 13:16) — low-risk dilution

| Field | Content |
|---|---|
| **Purpose** | Test whether the low-risk bulk drowns the high-risk minority in the gradient. |
| **Research question** | Does low-risk dilution explain the R3 result? |
| **Hypothesis** | Yes — 95% of samples are low-risk, so they set the update direction. |
| **What was changed** | Nothing. Probes: `_diag_rung3_{dilution,bootstrap,score}.py`. |
| **Command** | *(reconstructed; evidence = the seven `SPRINT_7_RUNG3_*.log` files)* |
| **Primary metrics** | The **exact gradient decomposition** `g_full = g_hi + g_lo`; `cos(g_full, g_lo)`; `cos(g_hi, g_synth)` against the reference risk-aware direction **`synth`**; the mass ratio `‖g_hi‖/‖g_lo‖`; cluster bootstrap (clusters = episodes). |
| **Result** | **Dilution is REAL:** the low-risk bulk sets the update direction in **9/9 cells** — `cos(g_full, g_lo) ∈ [+0.9099, +0.9994]`. **And it does not explain R3:** the mass ratios order the arms **opposite** to behaviour. Verdict **NO-GO**, `go_for_training_rung = False`. Also: `synth` **does not discriminate the action channel**. |
| **What it proved** | The high-risk gradient is genuinely present and directionally right in every arm (`cos(g_hi, g_synth) = +0.913 A0 / +0.536 R2 / +0.451 R3`) — it is simply **outvoted**. |
| **What it ruled out** | Dilution as the *differential* explanation of the arm ordering. |
| **What remained unknown** | The arm ordering R2 > R3 > A0. |
| **Why the next stage happened** | Three consecutive NO-GOs. → The divergence phase, asking a weaker question: not "what causes it" but "what *differs*". |

Also established (`_diag_rung2_75_mbtail.py` + Rung 3 census): `--episodes` is **silently ignored
by the D3/D4 census probes** (they read `rollout_episodes` from the checkpoint config), which
makes the four `*_b32.json` census artifacts **bit-identical duplicates** of their non-`b32`
counterparts — they must never be read as a batch-size control. The real matched-batch control is
in `_diag_div_content.py`.

---

### Sprint 7 · Divergence phase (2026-08-27 → 2026-08-30 11:02) — `SPRINT_7_DIVERGENCE_REPORT.md`

| Field | Content |
|---|---|
| **Purpose** | Characterise R2 vs R3 without claiming causality. |
| **Research question** | What difference between R2 and R3 *could* explain the behaviour gap? |
| **Hypothesis** | Several, tested in parallel: geometry, critic, content, variance, shape, logs. |
| **What was changed** | Nothing. Probes: `_diag_div_{content,critic,geometry,logs,shape,variance}.py`. |
| **Command** | *(reconstructed; evidence = the nine `SPRINT_7_DIV_*.log` files)* |
| **Outputs** | `SPRINT_7_DIVERGENCE_REPORT.md`, `SPRINT_7_DIV_geometry_eval.json`, `_DIVERGENCE_integrity/` (8 manifests). |
| **Primary metrics** | `proj_frac` (E1 regime inversion); expected-update-content shift in units of σ₈; matched 2×2 critic contrast. |
| **Result** | **`rollout_episodes` is a pure variance lever:** expected update content is invariant to batch size (**≤ 0.105 σ₈**) and the sd follows the finite-population law (`sd(8)/sd(32) = 2.236`). The reported *"5.1× collapse"* was a **+2.09σ outlier of the 8-episode instrument**. The critic is **not** the R2/R3 differential (0.009 down each column of the matched 2×2). What survived: **regime-selective learning asymmetry** — R3 learned the low-risk bulk and not the high-risk minority, on two independent instruments. |
| **What it proved** | That `rollout_episodes` cannot carry a content-based mechanism, and that the R2/R3 difference is *regime-selective*. |
| **What it ruled out** | Batch-size-as-content; the critic as the differential. |
| **What remained unknown** | Everything causal. The report is explicit that its surviving finding is **"what differs", NOT causal**. It also brackets R3's arg-max collapse to **updates 45–75**, *"bracketed, not localized, because no intermediate checkpoints exist"*, and §10 names **per-update checkpoints as the single largest gap.** |
| **Why the next stage happened** | → A reconstruction audit (Phase 0), because four stages in a row had returned "not causal". |

---

### Sprint 7 · Phase 0 + 1 + 2 (2026-08-30 11:59) — `SPRINT_7_PHASE0_RECONSTRUCTION.md`

This is the stage where the sprint turned its instruments on themselves, and it is the most
important methodological document in the repository.

| Field | Content |
|---|---|
| **Purpose** | Reconstruct the state (Phase 0), verify integrity (Phase 1), and choose the next causal question (Phase 2). |
| **Research question** | Given three NO-GOs, is the problem a missing hypothesis or a broken instrument? |
| **Hypothesis** | Apply **RULE 9** (*"if a metric fails to correlate with known behaviour across existing arms, demote it"*) to Sprint 7's own mechanism instruments — which no report had done. |
| **What was changed** | Nothing. New: `marl/diag/_phase0_calibration.py`, `SPRINT_7_PHASE0_calibration.json`. |
| **Primary metrics** | Spearman ρ between each mechanism instrument and Δ_EDGE, across arms. |
| **Result** | **20 readings. 20 negative. 0 positive. Mean ρ = −0.900.** Every mechanism instrument in the corpus orders the arms **backwards**. |
| **Structural cause** | Every instrument is evaluated at a **single terminal checkpoint**, and a terminal gradient measures the **signal still left uncorrected** — i.e. **remaining headroom**, which is anti-correlated with achieved behaviour by construction. A0 never learned risk-awareness, so its endpoint gradient still points *at* the risk-aware direction (`cos(g_full, g_synth) = +0.4023`, best of any arm) and its behaviour is worst; R2 learned the most, so its endpoint gradient points **away** (−0.2372) and its behaviour is best. |
| **What it proved** | The blocker is a **missing observable**, not a missing hypothesis. |
| **What it ruled out** | **Mechanism 2 is closed.** R3 already falsified its only causal pathway (variance); the successor "clip-mediated bias" hypothesis was tested offline and **not supported** — R2's `clip_frac` quartile means are **0.0426 / 0.0196 / 0.0031 / 0.0000** with 22 of 75 updates at exactly zero, so the clip is not active enough to convert a per-state offset into a bias; and `c(s)` is **agent-idiosyncratic** (within-tick spread of `c(s)` averages **6.65**, max 18.61, against an overall `SD(c)` of 4.65), so **no implementable online centring exists**. |
| **What remained unknown** | Whether R2 "never learned" the high-risk channel or "learned it then lost it". |
| **Why the next stage happened** | Phase 2 proposed the only remaining clean experiment: a **zero-variable, checkpoint-instrumented replication of R2**. |

The Phase 0 honesty paragraph on its own headline claim is worth quoting because it is the model
for how the rest of the corpus should be read:

> *"Each instrument has only 2 or 3 arms, so no single ρ is significant (permutation p ≥ 1/6 at
> n=3). The 20 readings share arms, share the terminal-checkpoint construction, and reuse two
> state sets, so they are **not independent** and the sign count must **not** be read as
> p = 2⁻²⁰. … The claim is therefore: no instrument in the corpus has been shown to track
> behaviour, and every one that can be checked points the wrong way."*

Integrity result from Phase 1 (see Part 11 for the method): of the **14-manifest prior chain**,
**11 verified clean and 3 failed**, and all three were investigated rather than accepted:

1. `_rung0_integrity/code_before.md5` — `config.py`/`mappo.py`/`train.py` fail. Hashes moved
   **exactly once**, between the Rung 0 manifest (08-19 21:35) and `rung2_5_code_before`
   (08-25 04:02) — i.e. *the Rung 2 MC-target production change* — and are byte-identical in
   every later manifest. **Documented and intended.**
2. `_rung2_75_integrity/artifacts_after.md5` — `SPRINT_7_RUNG2_75_REPORT.md` fails. One
   post-manifest edit to a **report**. A minor RULE-12 process slip; no data artifact affected.
3. `_rung2_75_integrity/code_after.md5` — `_diag_rung2_75_coherence.py` and
   `_diag_rung2_75_matched_states.py` fail, inside the R3 evaluation window. Because
   `_diag_rung2_75_matched_states.py` **constructs RANDOM and UNION**, this was not accepted on
   narrative grounds but **proved empirically**: all 10 shared arm×source values and all four
   state-set sizes are bit-identical across the two script versions, **MAX |diff| = 0.000e+00**.
   The edit was purely additive (it added the `R3` column).

One further source discrepancy found and **deliberately not fixed**, because production code is
frozen: `mappo.py:447` comments that *"GAE's own averaging horizon is only 1/(1 − gamma*lambda)
~ 20 steps"*. That is correct for the retired λ=0.95 (19.6) and **wrong** at the current λ=0.995
(**166.8**). `config.py:434` records the correct value. Comment-only; no code effect.

---

### Sprint 7 · Phase 4 pre-registration (2026-08-30 15:08) — `SPRINT_7_PHASE4_PREREG.md`

45,749 bytes of pre-registration written **before** the run. It fixes the hypothesis, the single
variable (none), the control, the success metric, the failure criterion, and the hash manifests.
This is the RULE-3/RULE-9 discipline made concrete: **you may not edit a pre-registered
reference after seeing the measurement.**

The Phase 2 design it registers, quoted from Phase 0 §6:

- **Manipulated variable: none.** Same `seed 20260818`, 600 episodes, `rollout_episodes 8`,
  `critic_target "mc"`, `--device cpu`. The only difference: `agent.save` is invoked after each
  of the 75 `agent.update` calls.
- **Control:** the frozen `mappo_R2_mc_target.pth` and `mappo_R2_mc_target_updates.csv`.
- **Falsifiable integrity check:** if the replication's 75 update rows are byte-identical to
  R2's `_updates.csv` and the final `.pth` matches, the intermediates are *provably* the
  trajectory that produced R2. If not, a reproducibility defect has been found — itself a result.
- **Why training is justified despite RULE 4** (*don't train for what artifacts can answer*):
  a learning **trajectory** is definitionally not recoverable from terminal checkpoints.
- **Implementation constraint:** `train.py` saves only `_best` (line 232, gated on
  `mean_r > best_mean`) and the final (line 252), so per-update saving must come from an
  **additive driver** wrapping `MAPPOAgent.update`. `torch.save` consumes no RNG, so bit-identity
  is preserved. **`train.py` must not be edited** — and it was not.

---

### Sprint 7 · Phase 4 run + Phase 4.1 (2026-08-30 16:23) — `SPRINT_7_PHASE4_REPORT.md`

| Field | Content |
|---|---|
| **Purpose** | Produce the missing observable: R2's learning trajectory. |
| **Research question** | Is R2 bit-exactly reproducible, and what do the 76 update boundaries show? |
| **Hypothesis** | Reproducible; and the trajectory will localize the collapse. |
| **What was changed** | Nothing in production. New additive driver `marl/diag/_phase4_r2_trajectory.py`, plus `_phase4_smoke.py`, `_phase4_equiv.py`, `_phase4_verify.py`. |
| **Command** | `python -m marl.diag._phase4_r2_trajectory …` *(reconstructed)* |
| **Outputs** | **`R2_trajectory/R2_trajectory_u000..u075.pth` (76 checkpoints)**, `R2_trajectory/SPRINT_7_P4_trajectory_manifest.jsonl`, `R2_traj_repro{,_best}.pth`, `R2_traj_repro_{config.json,history.csv,updates.csv}`, `SPRINT_7_P4_SMOKE_REPORT.json`, `SPRINT_7_P4_EQUIV_SELFTEST.txt`, `_PHASE4_integrity/`, `_PHASE4RUN_integrity/`, `_PHASE41_integrity/`. |
| **Primary metrics** | 19 pre-registered verification checks B1–B19; `protected` trend over 600 episodes. |
| **Result** | **R2 replicated bit-exactly, storage-for-storage, on CPU.** So nondeterminism explains nothing, and the 76 checkpoints provably belong to the real control run. Verification returns **18/19**. Separately, §F.4: `protected` is **flat over 600 episodes (ρ = +0.053, not significant)**. |
| **What it proved** | The trajectory is trustworthy — every historical R2 finding transfers to these 76 snapshots. |
| **What it ruled out** | Nondeterminism as an explanation for anything in Sprint 7. |
| **What remained unknown** | The trajectory reading itself, which §G.3 defers. |
| **Why the next stage happened** | → Phase 5 reads the trajectory. |

> **`_phase4_verify.py:78` holds a 1-ULP-wrong `final_lr_critic`, so check B6 FAILS by design.**
> The pre-registered literal `1.3333333333333309e-05` parses to bits `…3688`; every artifact
> that has an `lr_critic` — including the frozen control `mappo_R2_mc_target.pth` — stores
> `1.3333333333333308e-05`, bits `…3687`, which is the exact float64 value of
> `1e-3 * (1 − 74/75)`. **It was left wrong on purpose** so the FAIL stands in the record
> (RULE 3/9 forbid editing a pre-registered reference after seeing the measurement).
> **Expect 18/19, not 19/19.** `final_lr_actor` is correct, and the quantity B6 guards is
> covered anyway by B5 + B1a. Correcting it would require a `SPRINT_7_PHASE4_PREREG.md` §19
> amendment.
>
> **Naming note:** §G.3 defers the trajectory reading to *"Phase 6"*; the report that actually
> performed it is named **Phase 5**. Surfaced, not fixed. Also, **Phase 4.1 has no report** — it
> exists only as the eight manifests in `_PHASE41_integrity/`.

---

### Sprint 7 · Phase 5 (2026-08-31 03:01) — `SPRINT_7_PHASE5_REPORT.md`

| Field | Content |
|---|---|
| **Purpose** | Read the 76-checkpoint trajectory. |
| **Research question** | *When* did the high-risk channel move, and did Δ_EDGE grow by acquiring high-risk EDGE or by suppressing low-risk EDGE? |
| **Hypothesis** | Two readings were possible: A (genuine high-risk acquisition) or B/C (low-risk suppression). |
| **What was changed** | Nothing. New probe `marl/diag/_phase5_risk_trajectory.py`. |
| **Command** | `python -m marl.diag._phase5_risk_trajectory --device cpu --clusters 32 --start-seed 20260825 --boot 5000 --random-seed 31337 --tag main` *(reconstructed from `argparse` defaults; evidence = `SPRINT_7_PHASE5_run.log`)* |
| **Outputs** | `SPRINT_7_PHASE5_risk_trajectory_main.json`, `_parity.json`, `SPRINT_7_PHASE5_run.log`, `_PHASE5_integrity/` (10 manifests). |
| **Primary metrics** | `π(EDGE | risk ≥ 0.6)` and `π(EDGE | risk < 0.2)` at each of the 76 checkpoints; the decomposition of Δ_EDGE's growth into the two channels. |
| **Result — the sprint's most consequential finding** | **`π(EDGE | high risk)` ends BELOW its random-initialisation value and never exceeds it in 76/76 checkpoints.** **159–218% of Δ_EDGE's growth is low-risk suppression.** So **Δ_EDGE is differential suppression, not risk acquisition.** |
| **Also found** | **R2's actor freezes at update 62** — the actor stops moving with ~19% of training left, while the critic is still improving fastest. That **rules out "the critic gated the actor"** and dates the softmax collapse to **u1–u40**. |
| **What it proved** | The headline Sprint 7 improvement is mostly the policy learning *not to migrate at low risk* — which is correct behaviour (true low-risk advantage is −1.337) but is **not** self-healing. |
| **What it ruled out** | Reading A (pure high-risk acquisition); and "the critic gated the actor". |
| **What remained unknown** | An unresolved **expectation-vs-greedy conflict**: `π(EDGE | hi)` never exceeds init (0/76 and 1/76), yet **greedy argmax-EDGE rises 0.083 → 0.463 on RANDOM**. Not reconciled. |
| **Why the next stage happened** | Nothing further was justified. → the synthesis audit. |

> §4.4 of this report claims `SPRINT_7_R3_action_channels.json` does not exist and that the
> CLOUD/STAY/PREEMPT responses were not pre-existing measurements. **That claim is wrong** — the
> file is `SPRINT_7_R3_action_channels_R3.json` (36,293 bytes, md5
> `53955350c3d84c10c234d33464ea8d5c`, 2026-08-25 18:10), listed in `SPRINT_7_R3_REPORT.md`'s own
> manifest at line 127, holding the four-action Δ for six arms on both frozen sets
> (`status = 'EXPLORATORY -- not preregistered'`). Corrected in
> `SPRINT_7_FINAL_SYNTHESIS_AUDIT.md` §8 row 17 **rather than by editing the report**. No Phase 5
> number is affected. **Generalisable trap: every `--tag` probe in this tree writes
> `<name>_<tag>.json`, so a bare `ls` of the expected name misses it.**

Phase 5 also established the **masking facts** that constrain every behavioural reading:
`PREEMPT_REROUTE` is legal in **exactly 0.00%** of high-risk decision entries on both frozen
sets — so its zero probability is *a masking fact, not a policy fact*, and the policy can be
neither blamed nor credited for never preempting. `MIGRATE_CLOUD` is legal in only **26.4%
(RANDOM) / 8.0% (UNION)** of high-risk entries. **The effective high-risk choice is essentially
binary: STAY vs MIGRATE_EDGE.**

---

### Sprint 7 · Final synthesis audit (2026-08-31 04:39) — `SPRINT_7_FINAL_SYNTHESIS_AUDIT.md`

59,913 bytes. 12 sections, a **22-hypothesis ledger**, the timeline, and every superseded claim.
Read-only; no training.

**Verdict block: `COMPLETE` · `Further training justified: NO` · `Sprint 7 closed: YES`.**

Its answer to the central question, stated at mechanism level:

> The high-risk regime is **4.3–6.5%** of decision entries. Its gradient is **directionally
> positive but outvoted** — `cos(g_full, g_lo) ∈ [+0.9099, +0.9994]` in 9/9 cells — while at low
> risk EDGE is **genuinely wrong** (true advantage −1.337, n=159). So **correctly-learned
> low-risk suppression generalises through shared parameters into the high-risk minority.**
> This is a mechanism-level answer for the *shared* failure. **The arm ordering R2 > R3 > A0
> remains unexplained** (Rung 3's D4 failed).

Its §3 chosen interpretation is **D (a combination)**, not the flattering A: *mostly* differential
low-risk suppression, **plus a real high-risk acquisition phase at u042–u062** (+0.0658/+0.0660
in `p_hi` vs +0.0195/+0.0192 in `p_lo`; **23%/30% of the final Δ**), and R2 beats the A0 control
on *both* channels while ending below its own init on the mass channel.

**Two things it records as not to redo:**
1. A high-risk-weighted rung is **NO-GO by a pre-registered rule** (`w*` = +9.832 for the arm
   that needs it; +0.0500 at w=16).
2. A variance/batch rung **was already run and falsified** (R3 + the divergence phase).

**If it is ever reopened**, §11 item 5 names the one highest-value *offline* analysis: run the
**Rung 3 gradient decomposition along the u000…u075 trajectory** instead of at the endpoint.
No training needed; the checkpoints already exist. It would close the only causal gap left.

---

## 2.1 Contradictions and gaps in the corpus — surfaced, not resolved

The brief requires these be stated rather than quietly fixed. Every one is a real disagreement
between two documents in the repository.

| # | contradiction | status |
|---|---|---|
| 1 | R2-vs-R3 config field count: `SPRINT_7_R3_REPORT.md` §3 says **91 fields, exactly 3 differ**; `SPRINT_7_DIVERGENCE_REPORT.md` §2.1 says **99 fields, 5 differ, 1 substantive**. | Unresolved. Both agree only `rollout_episodes` is substantive. |
| 2 | R3's Δ on RANDOM is quoted as both **+0.1575** (R3 report, Phase 0) and **+0.1576** (Divergence §2.3). | Rounding, almost certainly. Unresolved in the record. |
| 3 | Phase 4 §G.3 defers the trajectory reading to **"Phase 6"**; the report that did it is **Phase 5**. | Naming slip. |
| 4 | Phase 5 §4.4's *"`SPRINT_7_R3_action_channels.json` does not exist"* is **wrong**. | Corrected in the audit §8 row 17; the report was left unedited. |
| 5 | Rung 2's high-risk EDGE-share direction is formally **UNRESOLVED** (Rung 2.5 §G.5) and the metric was later **withdrawn as unidentified** (Rung 2.75 §A.4). | Both stand. |
| 6 | Phase 5's **expectation-vs-greedy conflict**: `π(EDGE\|hi)` never exceeds init (0/76, 1/76) vs greedy argmax-EDGE 0.083 → 0.463 on RANDOM. | Unresolved. |
| 7 | The Phase 1 **proposed ladder ≠ what ran**: B1–B3, C1–C4, E1–E3 and eight proposed CLI flags have no artifacts. | Plan, not history. |
| 8 | Sprint 6.5 §13 recommended a **smaller** `rollout_episodes`; R3 tested a **larger** one. | Both are in the record; the reversal is undocumented. |
| 9 | Phase 0 §1.1's A3 row (`entropy_coef 0.01→0.02`) and A2 label ("critic sign") are **wrong**; see the box above. | Surfaced here. |
| 10 | **Held-out task performance declined along the ladder while Δ_EDGE improved**, and every arm loses to the one-line rule. | The central caveat for Part 16. |
| 11 | The **RULE 1–12 list is referenced throughout the corpus but never enumerated in any file.** | The enumerated list is `NOT VERIFIED FROM REPOSITORY`. Individual rules are quotable (see Part 17). |
| 12 | There is **no standalone `SPRINT_7_RUNG2_REPORT.md`** and **no `SPRINT_7_PHASE4_1*.md`**. | Rung 2 is documented inside Rung 2.5; Phase 4.1 exists only as manifests. |

---

# Part 3 — "WHY DID WE EVEN DO THIS?" — the reasoning chain

This is the single most important section for remembering the project. Read it top to bottom.

```
  We built a digital twin of a 10-node edge hospital network.
            ↓
  We observed: the original failure model killed hosts instantly at a scheduled time,
               with NO symptoms — and 5 of 12 telemetry columns were literally constant.
            ↓ Therefore we suspected: prediction was ill-posed, not merely hard.
  So we ran:   a simulator rebuild with latent wear + fault mechanisms (Sprint 3/3.5/3.75).
            ↓ It showed: 32 failures / 32 recoveries, 0 of 12 columns constant.
  Therefore:   prediction became well-posed.
            ↓
  We observed: the LEGACY 7-column predictor produced 31 STANDING_ALARM of 32 failures
               — an alarm that is always on.
            ↓ Therefore we suspected: features, not architecture.
  So we ran:   the 26-feature observable arm vs legacy vs OLDDATA (Sprint 4).
            ↓ It showed: PR-AUC 0.2863 vs 0.0777, and 23/32 genuine EARLY_WARNING.
  Therefore:   the risk channel is REAL — 13.3x lift over a 2.1% base rate, generalising
               leave-one-node-out.
            ↓ AND we learned a habit that mattered later: ALWAYS compare against a trivial rule.
               (`degraded==1` gets F1 0.0657. Always-negative gets F1 0. Now we know the floor.)
            ↓
  We handed that risk channel to a MAPPO scheduler (Sprint 6).
            ↓
  We observed: MAPPO greedy = 20.62 reward, 3.2 tasks protected.
               risk-threshold@0.18  = 76.50 reward, 30.2 tasks protected.
               Sweeping risk 0.00 -> 0.99 moves P(relocate) by 0.004.
               Setting risk to ZERO makes reward BETTER (20.62 -> 23.13).
            ↓ Therefore we suspected: eight things at once (A,B,C,D,F,G,H,J,K).
  So we ran:   Sprint 6.5 — four controlled arms + counterfactual replay + behaviour cloning.
            ↓ It showed: true advantage of EDGE at high risk = +2.671 (gap +4.01 vs low risk);
                         the SAME network can be cloned to span 0.3935 (100x sharper);
                         reward sizing is correct over the horizon (+8.267 for migrating);
                         two real bugs (index-ordered contention; dead criticality charge);
                         and 545/783 = 69.6% of relocations are REFUSED (cloud_slots=8 for 10).
  Therefore:   A,B,C,D,J,K REFUTED. The failure is LEARNING DYNAMICS.
               F (sparsity, ~4.7% high-risk), G (update -> 0), H (lifetime credit) SUPPORTED.
            ↓ But the fix objective was NOT met. Sprint 7 opened.
            ↓
  We observed: "the PPO update goes to zero" is a SYMPTOM.
               The policy plays MIGRATE_CLOUD in 842/855 states where cloud is legal (98.5%),
               MIGRATE_EDGE in 77/10751 (0.7%) — i.e. "cloud whenever legal, else stay",
               built on a -0.050 pseudo-no-op, with no risk term in it at all.
            ↓ Therefore we suspected: PPO is CONVERGED, pointed the wrong way — the advantage
               estimator's action ordering at high risk is INVERTED vs the truth.
               (truth: EDGE +2.67 > STAY 0 > CLOUD -0.05
                estimator: CLOUD +0.080 > STAY +0.044 > EDGE -0.266)
            ↓ Caveat we stated ourselves: n=4 and n=9. Not significant.
  So we ran:   Rung 0 at proper sample size.
            ↓ It showed: signal REAL (+3.363 forced-EDGE true adv at risk>0.50;
                         frac_positive 0.7012 vs 0.3931 at low risk)
                         and GAE sign agreement 29.0% — BELOW CHANCE.
  Therefore:   the inversion is real. The suspect is the CRITIC TARGET: A0 fit its critic
               toward a lambda-return computed from its OWN predictions (self-referential).
            ↓
  So we ran:   Rung 1 — refit the critic offline against a Monte-Carlo target. No retraining.
            ↓ It showed: sign agreement 26.3% -> 60.7%. Residual: -5.24 swing from the target
                         vs -1.07 from the per-state offset.
  Therefore:   SUPPORTED. This is the only finding strong enough to change production code.
            ↓
  So we ran:   Rung 2 — ONE production edit, `critic_target: "lambda" -> "mc"`, retrain = R2.
            ↓ It showed: BEST ARM. Delta_EDGE +0.2024 RANDOM / +0.1579 UNION.
                         ...and held-out reward -0.46 against a 67.31 baseline. protected 9.9 vs 30.2.
  Therefore:   the mechanism fix transfers to training. The task-level gap does NOT close.
            ↓
  We asked:    is there target headroom left?
  So we ran:   Rung 2.5.
            ↓ It showed: contamination 58.4% -> 7.8% (87% fixed). 130/130 zero-censored.
                         Remaining headroom <=3.5pp, at chance.
  Therefore:   NO-GO for another target rung. But it SEPARATED THREE MECHANISMS:
               (1) target formulation  (2) per-state value offset  (3) actor saturation.
            ↓
  We asked:    is our headline metric even identified? what IS the actor stall?
  So we ran:   Rung 2.75 — the paired estimator gae(s,a) = c(s) + paired(s,a).
            ↓ It showed: Var(raw)/Var(paired) = 3.72x. Pairing lifts high-risk sign agreement
                         from BELOW CHANCE to 0.71 (team) / 0.95 (own) for EVERY arm.
  Therefore:   the "sign error" is a per-state V(s) OFFSET, not a target defect.
               The old headline metric was WITHDRAWN as unidentified; Delta_EDGE on frozen
               RANDOM/UNION sets replaced it.
            ↓ And a testable prediction fell out: a constant contributes zero in expectation,
               so the offset is VARIANCE, and 3.72x is "the exact data multiplier needed".
            ↓
  So we ran:   R3 — rollout_episodes 8 -> 32 (with episodes 600 -> 2400 to hold updates at 75).
            ↓ It showed: SNR really rose (real/shuffled 2.03, p=0.0000)
                         AND BEHAVIOUR GOT WORSE (Delta 0.2024 -> 0.1575).
  Therefore:   H1 FALSIFIED — and, since variance was mechanism 2's ONLY causal pathway,
               MECHANISM 2 IS FALSIFIED TOO. (No report said this at the time; Phase 0 did.)
            ↓ Successor guess: the low-risk bulk dilutes the high-risk minority.
            ↓
  So we ran:   Rung 3 — the exact decomposition g_full = g_hi + g_lo.
            ↓ It showed: dilution is REAL — g_lo sets the direction in 9/9 cells
                         (cos(g_full,g_lo) in [+0.9099,+0.9994]);
                         the high-risk gradient IS directionally right in every arm
                         (cos(g_hi,g_synth) = +0.913 A0 / +0.536 R2 / +0.451 R3);
                         BUT the mass ratios order the arms OPPOSITE to behaviour.
  Therefore:   SUPPORTED as a shared mechanism, NO-GO as an explanation of R3.
            ↓
  So we ran:   the Divergence phase — "what DIFFERS between R2 and R3?", not "what causes".
            ↓ It showed: rollout_episodes is a PURE VARIANCE lever (content invariant to
                         <=0.105 sigma_8; sd(8)/sd(32) = 2.236 exactly as the finite-population
                         law predicts; the famous "5.1x collapse" was a +2.09-sigma OUTLIER of
                         the 8-episode instrument);
                         the critic is NOT the differential (0.009 down each column of the 2x2);
                         what survives is REGIME-SELECTIVE LEARNING ASYMMETRY
                         — R3 learned the low-risk bulk and not the high-risk minority.
  Therefore:   "what differs", explicitly NOT causal. And R3's argmax collapse is BRACKETED to
               updates 45-75, "not localized, because no intermediate checkpoints exist".
            ↓
  THREE NO-GOs IN A ROW. So we stopped proposing mechanisms and audited the instruments.
            ↓
  So we ran:   Phase 0 — apply RULE 9 to our OWN mechanism metrics.
            ↓ It showed: 20 readings. 20 NEGATIVE. mean rho = -0.900.
                         EVERY mechanism instrument orders the arms BACKWARDS.
                         Structural cause: they are all single-TERMINAL-checkpoint measures,
                         and a terminal gradient measures REMAINING HEADROOM, which is
                         anti-correlated with achieved behaviour BY CONSTRUCTION.
                         (A0 never learned it, so its endpoint gradient still points AT the
                          risk-aware direction: +0.4023, the best of any arm, worst behaviour.
                          R2 learned the most, so its endpoint points AWAY: -0.2372, best
                          behaviour.)
  Therefore:   the blocker is a MISSING OBSERVABLE, not a missing hypothesis.
               And mechanism 2 is CLOSED: no surviving pathway (variance falsified), the clip
               is too rarely active to make the offset a bias (clip_frac quartiles
               0.0426/0.0196/0.0031/0.0000, 22/75 updates at exactly zero), and c(s) is
               agent-idiosyncratic (within-tick spread 6.65 vs SD(c) 4.65) so there is no
               implementable online centring.
            ↓
  So we ran:   Phase 4 — a ZERO-VARIABLE, checkpoint-instrumented replication of R2.
               Manipulated variable: NONE. The only difference is that we save after each of
               the 75 updates, via an ADDITIVE driver. train.py was NOT edited.
            ↓ It showed: R2 replicates BIT-EXACTLY on CPU, storage-for-storage. 18/19 checks
                         (B6 fails by design on a 1-ULP pre-registered constant, left wrong
                         on purpose so the FAIL stands in the record).
                         76 checkpoints u000..u075 now exist.
  Therefore:   nondeterminism explains nothing, and the trajectory PROVABLY belongs to R2.
            ↓
  So we ran:   Phase 5 — read the trajectory.
            ↓ It showed, and this is the answer:
                 pi(EDGE | risk >= 0.6) ENDS BELOW ITS RANDOM-INIT VALUE
                 and NEVER EXCEEDS IT IN 76/76 CHECKPOINTS.
                 159-218% of Delta_EDGE's growth is LOW-RISK SUPPRESSION.
                 R2's actor FREEZES AT UPDATE 62 while the critic is still improving fastest
                 -> rules out "the critic gated the actor", and dates the softmax collapse to u1-u40.
  Therefore:   Delta_EDGE is DIFFERENTIAL SUPPRESSION, NOT RISK ACQUISITION.
               The headline improvement is mostly the policy learning not to migrate at LOW risk
               — which is CORRECT (true low-risk advantage is -1.337) but is NOT self-healing.
            ↓
  So we ran:   the Final Synthesis Audit — read-only, 22-hypothesis ledger.
            ↓ Verdict: COMPLETE. Further training justified: NO. Sprint 7 closed: YES.
               Chosen interpretation: D (a combination) — mostly low-risk suppression, PLUS a
               real high-risk acquisition phase at u042-u062 worth 23-30% of the final Delta.
               The shared failure has a mechanism-level answer: the high-risk regime is 4.3-6.5%
               of entries, its gradient is directionally positive but OUTVOTED, and at low risk
               EDGE is genuinely wrong — so correctly-learned low-risk suppression GENERALISES
               THROUGH SHARED PARAMETERS into the high-risk minority.
               The arm ordering R2 > R3 > A0 remains UNEXPLAINED.
```

**The one-sentence version.** We proved the risk signal is real and that migrating at high risk
is genuinely right; we found and fixed a real critic-target defect that produced the largest
behavioural gain in the project; we then falsified every remaining single-mechanism explanation,
discovered that all our diagnostic instruments were measuring headroom rather than learning, built
the missing observable, and found that our headline improvement was mostly the policy correctly
learning *not* to migrate when it shouldn't — not learning to migrate when it should.

---

# Part 4 — Codebase map

## 4.1 Repository layout

```
DT-MARL-Healthcare/
├── .claude/  .gitignore  .idea/
├── da                       2,658 B stray file — no known purpose
├── temp.py                  scratch
├── requirements.txt
├── datasets/cluster/        EMPTY at HEAD (ClusterData2019 not in repo)
├── docs/                    EMPTY
├── results/                 EMPTY
├── simulation/              JAVA — CloudSim Plus 8.5.7 digital twin (44 .java files)
│   ├── pom.xml              artifactId `simulation`, v1.0-SNAPSHOT
│   └── src/main/java/com/dtmarl/…
└── python-ai/               PYTHON — predictor + MARL
    ├── config.py            0 BYTES
    ├── main.py              0 BYTES
    ├── data/                dataset construction + leakage discipline
    ├── models/              bilstm.py · failure_predictor.py · htcf.py · transformer.py
    ├── training/            predictor training + CV + lead-time evaluation
    ├── marl/                the MARL system (see 4.3)
    ├── inference/           EMPTY
    ├── utils/               EMPTY
    ├── scratch_*.py         5 exploratory scripts, not part of any pipeline
    ├── *.log                69 run logs — the primary chronological evidence
    └── saved_models/
        ├── failure_predictor*.pth / *_meta.json / *_oof.npz / *_scaler.npz
        ├── host_metrics*.json · host_lead_time*.csv · host_lono.json
        ├── htcf_model.pth · device_failure_predictor.pth
        ├── _backup_pre_finalization/
        └── marl/            THE RESEARCH TREE (see Part 9)
```

## 4.2 Java side — `simulation/`

| Path | Purpose | Used in training? |
|---|---|---|
| `…/Main.java` | Entry point. | Indirectly — produces the trace. |
| `…/SimulationManager.java` | Orchestrates the whole simulation. Holds `SIM_SEED = 20260817L`, `MAX_SIMULATION_SECONDS = 1500.0`, `DEVICE_COUNT = 10`, `VERBOSE_TICK_TELEMETRY = false`, `PROGRESS_PRINT_INTERVAL = 100`. **Exports the three CSVs at lines 413–415.** | Yes (upstream). |
| `…/DigitalTwinManager.java` | The twin. `mirrorNetworkLinks(int)` creates **one NetworkLink per node (an access link)** — **not** a node-to-node graph. | Yes (upstream). |
| `…/CriticalityManager.java` | Clinical priority. Ported verbatim to Python. | Yes — via the port. |
| `…/CloudletManager.java` | Task generation. Ported verbatim to Python. | Yes — via the port. |
| `…/FailureInjector*` | The latent-wear + fault-mechanism failure model. | Yes (upstream). |
| `…/PredictionGateways.java` | The prediction seam. `DEFAULT_RISK_CSV = simulation/predicted_risk.csv`. | Deployment path only. |
| `…/ai/migration/`, `…/ai/prediction/` | Java-side AI hooks — **these contain sources** (not all `ai/*` packages are empty). | Deployment path only. |

**Safe to modify?** No, for research reproduction. Changing anything here changes the trace, which
invalidates every Python artifact. The trace is the fixed substrate of the entire project.

## 4.3 Python side — `python-ai/marl/` (the MARL system)

Line counts are exact.

| Path | Purpose | Important functions / constants | Used by | Inputs | Outputs | Safe to modify? | Training? | Diag only? |
|---|---|---|---|---|---|---|---|---|
| `marl/config.py` (576) | **Single source of truth** for every hyperparameter and constant. Action ids, `CLOUD_NODE_ID = -2`, `cloud_slots`, `neighbour_offsets`, reward weights, MAPPO params, window definitions. `:434` records the correct GAE horizon. | `ACTION_STAY=0`, `ACTION_MIGRATE_EDGE=1`, `ACTION_MIGRATE_CLOUD=2`, `ACTION_PREEMPTIVE_REROUTE=3` | everything | — | — | **NO — FROZEN** (last changed 08-25 04:02) | Yes | No |
| `marl/env.py` (1037) | `DTMarlEnv`. Observation/state construction, action application, contention resolution, reward. `infeasible` counter at `:405-422`, increments at `:565`/`:630`, rotating apply order at `:620-626`, reward penalty at `:934`. | `reset`, `step`, `_relocate` | train/eval/diag | trace, risk | obs, reward, info | **NO — FROZEN** | Yes | No |
| `marl/mappo.py` (593) | The learner. 10 separate actors + 1 centralised critic; GAE; PPO update; save/load. `torch.manual_seed(seed)` at `:218`; action masking `masked_dist` at `:77-79`; λ-return critic target at `:298`; the stale horizon comment at `:447`. | `MAPPOAgent.update`, `.save`, `.load`, `compute_gae` | train/diag | rollout buffer | loss stats, checkpoints | **NO — FROZEN** | Yes | No |
| `marl/train.py` (294) | The training driver. Saves `_best` at **line 232** (gated on `mean_r > best_mean`) and the final at **line 252**. | `main()` | CLI | CLI args | `.pth`, `_config.json`, `_history.csv`, `_updates.csv` | **NO — FROZEN** | Yes | No |
| `marl/rollout.py` (114) | Shared rollout/eval so *"training and evaluation can never diverge in how a metric is computed"*. `episode_starts` gives deterministic, identical starts for every policy. | `run_episodes`, `episode_starts`, `METRIC_KEYS` (22 keys) | train/eval | env, policy | metric dict | No | Yes | No |
| `marl/evaluate.py` (513) | The reported numbers: greedy MAPPO vs all baselines, behaviour probes, risk-sweep, risk→0 ablation. | `main()` | CLI | `.pth` | `_eval.json`, `*_eval.log` | No | No | No |
| `marl/baseline.py` (153) | `ReactiveThresholdPolicy`, `NoMigrationPolicy`, `RiskThresholdPolicy`. All mask-respecting. | — | evaluate | env, obs, masks | actions | No | No | No |
| `marl/risk_provider.py` (198) | The risk channel. Three sources (`oof`/`model`/`zero`). Never thresholds or binarises. | — | env | `*_oof.npz` or `.pth` | per-node-per-tick risk | No | Yes | No |
| `marl/trace.py` (160) | Loads the recorded trace. **Hard-drops label + audit columns before anything touches the frame** (`keep = ["time","nodeId"] + TRACE_CHANNELS`); forward-fills NaNs via `np.interp`. | `Trace.ch/is_up/up_during/any_down_during`, `load_trace`, `load_failure_events` | env, eval | `failure_history.csv` | `Trace` | No | Yes | No |
| `marl/topology.py` (58) | Ring neighbourhood. Rejects offset 0 and degenerate neighbourhoods. | — | env | node ids | neighbour lists | No | Yes | No |
| `marl/criticality.py` (185) | **Verbatim port** of the Java criticality/task generation. `self_check()` validates against `JAVA_PRIORITIES_T0` to **< 5e-4**. | `priority_at`, `priority_order_key`, `self_check` | env | task ids | priorities | No | Yes | No |
| `marl/destination.py` (88) | Resolves *which concrete host* once the policy has chosen a *kind* of relocation. | `score`, `feasible`, `select` | env | candidates | node id or `None` | No | Yes | No |
| `marl/export_risk_csv.py` (137) | Writes `simulation/predicted_risk.csv` for the Java seam. | — | CLI | `*_oof.npz` | CSV | No | No | No |
| `marl/tests_env.py` | **19 tests** t1–t19. | `t7 no_future_information` (237), `t8 no_forbidden_columns` (293), `t15 discount_preserves_policy_ranking` (433), `t16 gae_horizon_covers_task_lifetime` (504), `t17 contention_is_not_decided_by_agent_index` (558), `t19 legacy_switches_reproduce_sprint6_defects` (680) | CLI | — | pass/fail | Additive only | No | No |
| `marl/sanity_test.py` | 6 probes. | `find_failure_anchor`, `synthetic_risk`, `build_scenario` | CLI | — | pass/fail | Additive only | No | No |

### Diagnostic scripts — 32 in `marl/` + 6 in `marl/diag/`

**None of these are used in training. All are read-only w.r.t. production code.** They are grouped
by the stage that created them, which is how you find the code behind any historical number.

| Group | Files |
|---|---|
| Sprint 6.5 | `_diag_sensitivity.py`, `_diag_counterfactual.py`, `_diag_horizon.py`, `_diag_reward_terms.py` |
| Rung 0 | `_diag_rung0.py` |
| Rung 1 | `_diag_rung1_critic.py` |
| Rung 2 | `_smoke_rung2_mc_target.py` |
| Rung 2.5 | `_diag_rung2_5_{targets,signtest,native_dev,actor_stall,feasibility}.py` |
| Rung 2.75 | `_diag_rung2_75_{coherence,edgeshare,edgeshare_cluster,edgeshare_power,matched_states,mbtail,offset,plasticity,stepcollapse}.py` |
| R3 | `_diag_R3_action_channels.py` |
| Rung 3 | `_diag_rung3_{dilution,bootstrap,score}.py` |
| Divergence | `_diag_div_{content,critic,geometry,logs,shape,variance}.py` |
| Phase 0 | `diag/_phase0_calibration.py` |
| Phase 4 | `diag/_phase4_{r2_trajectory,smoke,equiv,verify}.py` |
| Phase 5 | `diag/_phase5_risk_trajectory.py` |

**The one that matters most:** `_diag_rung2_75_matched_states.py` **constructs the frozen RANDOM
and UNION state sets**. Every cross-arm Δ_EDGE number in Sprint 7 depends on it. It was edited
once (additively, to add the `R3` column) and the edit was **proved harmless to 0.000e+00**.

## 4.4 Python side — predictor pipeline

| Path | Purpose |
|---|---|
| `data/failure_dataset.py` | The **26-feature** host dataset + `assert_no_leakage()`. Holds `OBSERVABLE_COLUMNS` (12), `AUDIT_COLUMNS` (6), `FORBIDDEN_COLUMNS`, and `LEGACY_RAW_FEATURES` (the retired 7, kept only for side-by-side comparison). |
| `data/device_failure_dataset.py` | The IoMT-device analogue. |
| `data/dataset.py`, `data/preprocess.py`, `data/preprocessing/`, `data/explore.py` | Earlier cluster-trace pipeline (S-early). |
| `models/bilstm.py`, `models/failure_predictor.py` | The BiLSTM predictor. |
| `models/htcf.py`, `models/transformer.py` | S-early baselines. |
| `training/train_failure_predictor.py` | 5-fold temporal CV, 80 epochs; **writes the OOF scores** that the MARL environment consumes. |
| `training/train_device_failure_predictor.py` | Device analogue. |
| `training/host_cv.py` | Cross-validation driver → `host_metrics*.json`. |
| `training/evaluate_lead_time.py`, `evaluate_device_lead_time.py` | → `host_lead_time*.csv`. |
| `training/eval_leakage_audit.py` | The leakage audit. |
| `training/eval_compare.py` | Arm comparison (observable vs legacy vs OLDDATA). |
| `training/train.py` | Generic training entry. |

## 4.5 The data flow, end to end

```
 (1) simulation/  — Java, SIM_SEED 20260817, 1500 simulated seconds, 10 nodes
        │  SimulationManager:413-415
        ▼
 (2) failure_history.csv           10 nodes x 1500 ticks, label horizon 10.0 s
        │
        ├──► data/failure_dataset.py
        │       drops FORBIDDEN_COLUMNS, asserts no leakage,
        │       builds 26 features over a 10-step window
        │       │
        │       ▼
        │    training/train_failure_predictor.py   BiLSTM, 5 temporal folds, 80 epochs
        │       │
        │       ▼
        │    failure_predictor_oof.npz      <-- OUT-OF-FOLD RISK, leakage-safe
        │       (min 0.0145  max 0.9585  mean 0.0785  203 distinct  first valid tick 9)
        │
        └──► marl/trace.py  load_trace()
                keep = ["time","nodeId"] + TRACE_CHANNELS   <-- labels/audit dropped first
                forward-fills NaNs via np.interp
                │
                ▼
 (3) marl/env.py  DTMarlEnv                                    +  marl/risk_provider.py (oof)
        reset(start_tick in [9,491] for training)               +  marl/topology.py (ring, deg 4)
        │                                                       +  marl/criticality.py (Java port)
        │  every 2.0 s = 2 ticks:
        ▼
 (4) OBSERVATION  48 floats/agent            CENTRALISED STATE  489 floats
        │                                            │
        ▼                                            ▼
 (5) 10 ACTORS  48->[128,128]->4                 CRITIC  489+10=499 ->[256,256]->1
        masked_dist (mappo.py:77-79)                 (agent one-hot appended)
        │
        ▼
 (6) ACTION per agent  {STAY, MIGRATE_EDGE, MIGRATE_CLOUD, PREEMPTIVE_REROUTE}
        │
        ├─ marl/destination.py select()  -> which concrete host
        │      score = -w_risk*risk + w_cap*free_cap - w_lat*latency_norm - w_load*load_frac
        │      feasible: observed_up AND free_capacity_fraction > 0
        │      None  =>  infeasible  =>  fall back to STAY + penalty (REPORTED, not hidden)
        │
        ▼
 (7) env.step()  contention resolved in ROTATING order (step_idx), cloud_slots = 8
        │  -> reward: task completion, criticality weighting, migration cost,
        │             SLA, energy, P_infeasible = -0.050 x reward_scale
        ▼
 (8) marl/rollout.py  accumulate 8 episodes x 400 steps into the buffer  (T rows)
        │
        ▼
 (9) GAE(gamma=0.999, lambda=0.995)  ->  advantages     horizon 1/(1-gl) = 166.8 steps
        critic target: "mc" (production) or "lambda" (A0 legacy)
        normalise_advantages = True   <-- pooled over the whole buffer
        │
        ▼
(10) PPO UPDATE  4 epochs x 4 minibatches (+ a degenerate 5th tail of T mod 4 rows)
        clip_eps 0.2 · value_clip_eps 0.2 · entropy_coef 0.02 · value_coef 0.5
        max_grad_norm 0.5 · Adam · lr annealed by frac = 1 - (update_id-1)/75
        │
        ▼
(11) CHECKPOINT   train.py:232 (_best, if mean_r > best_mean)   train.py:252 (final)
        │           Phase 4 additionally saved all 76 update boundaries
        ▼
(12) marl/evaluate.py   8 FIXED eval starts in ticks [491,698]
        vs risk-threshold@0.18 / reactive / random-legal / static-no-migration
        + risk sweep + risk->0 ablation
        │
        ▼
(13) *_eval.json, *_eval.log  ->  the report tables in saved_models/marl/*.md
```

---

# Part 5 — Every important concept

Each entry: **what it is → why this project uses it → where it appears → why it mattered.**

### MARL (multi-agent reinforcement learning)
Several agents learning simultaneously in a shared environment. **Used because** each edge host
makes its own local relocation decision, and a single central scheduler would need global state at
execution time. **Where:** the whole `marl/` package; 10 agents in `config.py`. **Why it mattered:**
the agents share the reward, so credit assignment across agents is one of the supported root
causes (hypothesis H).

### MAPPO (multi-agent PPO)
PPO with a **centralised critic** and **decentralised actors**. **Used because** it is the standard
CTDE algorithm for cooperative multi-agent control. **Where:** `marl/mappo.py`. **Why it mattered:**
its clipped surrogate is the reason the per-state offset does *not* become a bias — see Phase 0 §4.1.

### Actor / critic
The **actor** is the policy: observation → action probabilities. The **critic** estimates the value
of a state, and is used only to compute advantages during training. **Where:** 10 actors
`48→[128,128]→4` (23,300 params each) and 1 critic `499→[256,256]→1` (194,049 params) in
`mappo.py`. **Why it mattered:** the entire Sprint 7 chain is about the critic's *target*, and
Phase 5 proved the **critic was still improving fastest when the actor froze at update 62** —
which rules out "the critic gated the actor".

### Centralised training, decentralised execution (CTDE)
The critic may see the global state during training; the actors may only see their own observation
at execution. **Where:** the critic takes `489 + 10` (state + agent one-hot); actors take 48.
**Why it mattered:** the agent one-hot is exactly why `c(s)` is **agent-idiosyncratic** — two
agents at the same tick have `c(s)` differing by 6.65 on average vs an overall SD of 4.65 — which
killed the only implementable online offset remover.

### PPO and clipping
PPO restricts how far the new policy may move from the old one by clipping the probability ratio
into `[1−ε, 1+ε]`. **Where:** `clip_eps = 0.2`, `value_clip_eps = 0.2`. **Why it mattered:** the
clip's *activity* was load-bearing for a hypothesis. R2's `clip_frac` quartile means are
**0.0426 / 0.0196 / 0.0031 / 0.0000**, with 22 of 75 updates at exactly zero — too rarely active
to carry a mechanism, which is what closed mechanism 2.

### Rollout / episode / update / minibatch / PPO epochs
- **Episode** = 400 decision steps = 800 simulated seconds, from one start tick.
- **Rollout** = `rollout_episodes` episodes collected before an update (production: **8**).
- **Update** = one PPO update over that buffer. **`n_updates = episodes // rollout_episodes`
  = 600 // 8 = 75.**
- **PPO epochs** = **4** passes over the same buffer.
- **Minibatches** = **4** — but `mb_size = T // 4` and `range(0, T, mb_size)` yields **5** chunks
  whenever `T mod 4 ≠ 0`, the fifth holding exactly `T mod 4` rows (1–3).
**Why it mattered:** the degenerate tail fires in ~3/4 of updates and contributes 4 of 20
optimiser steps. `denom = d_mb.sum().clamp(min=1.0)` prevents a NaN, but Adam still takes a
momentum-carried step on a zero gradient. It is a **shared, non-differential confound** — never a
per-arm mechanism. **Do not** read `tail_fraction = 0.25` as the tail; that is the last *full* chunk.

### Entropy and the entropy coefficient
Entropy measures how spread-out the action distribution is; the coefficient pays the policy to stay
random. **Where:** `entropy_coef = 0.02` in production, **0.05 in A3**, 0.01 in the retired
`attrbug_gamma99`. **Why it mattered:** A3 was the exploration arm. It produced the most
protections (11.12) — and the zero-risk ablation showed it keeps **92% of them with no risk input
at all**: *"It protects by volume, not by targeting."*

### KL and `approx_kl`
An estimate of how far the policy moved. **Critical repository-specific fact:** `approx_kl` in the
logs is **k1 = mean(old_logp − new_logp)**, which is **signed** and legitimately goes negative —
**26 of 75 updates for R2**. The k3 estimator exists offline at D4 snapshots only; there is **no
per-update k3 trajectory**. **Why it mattered:** anyone reading a negative `approx_kl` as a bug
would chase a non-problem.

### GAE, λ, γ
GAE blends multi-step returns with weight `(γλ)^k`. Its effective averaging horizon is
`1/(1 − γλ)`. **Where:** `gamma = 0.999`, `gae_lambda = 0.995` ⇒ **horizon 166.8 steps**,
recorded correctly at `config.py:434`. **Why it mattered:** the horizon has to cover a task's whole
lifetime, because exposure is charged over ~150 steps — that is what makes migrating worth
**+8.267** despite a −0.6443 single-step cost. `tests_env.py:504` (`t16
gae_horizon_covers_task_lifetime`) guards exactly this. And note `mappo.py:447`'s comment says
"~20 steps", which is stale from the retired λ=0.95 — **comment only, no code effect, deliberately
not fixed.**

### Monte-Carlo target vs λ-return target (`critic_target`)
- **`lambda`** — fit the critic toward a λ-return computed **from its own predictions**:
  self-referential. This is A0/A1/A2/A3, at `mappo.py:298`.
- **`mc`** — fit the critic toward the **actual discounted return** from the trajectory.
**Why it mattered:** this is the **only production change in Sprint 7**. Offline it lifted sign
agreement 26.3% → 60.7% (Rung 1); in training it removed 87% of target contamination (58.4% →
7.8%, Rung 2.5) and produced the best arm (R2). It is the project's single largest win.

### Advantage, and the paired advantage estimator
The advantage says how much better an action is than the state's average. Sprint 7's key
decomposition:

```
gae(s,a) = c(s) + paired(s,a),      c(s) := gae(s, a_ref),   a_true(s, a_ref) ≡ 0
```

**Where:** `_diag_rung2_75_offset.py`, `_diag_rung2_75_coherence.py`. **Why it mattered:**
`Var(raw)/Var(paired) = 3.72×`; pairing lifts high-risk sign agreement from **below chance** to
**0.71 (team) / 0.95 (own)** for every arm. On R2's native rows (547 states, 716 forced replays,
**0 replay mismatches**), `sign(raw) ≠ sign(paired)` in **45.9% (lo) / 46.7% (mid) / 56.7% (hi)**
of rows, and one illustrative row has raw GAE **−9.39** against a true advantage of **−0.80**
because `c(s) = −8.59`. **The "sign error" was never a target defect; it is a per-state offset.**

### Advantage normalisation
Advantages are standardised (`normalise_advantages = True`), pooled across the whole buffer.
**Why it mattered:** the always-active consequence of the offset is that normalisation divides by
an **offset-inflated SD** — `√3.72 = 1.93×` attenuation of the genuine signal. That is a
**magnitude** effect, not a direction effect, which is exactly why more data (R3) could not fix it.

### Stochastic vs greedy policy; action probability vs argmax
Training samples from the distribution; evaluation reports **greedy** (argmax). **Why it
mattered:** Phase 5's **unresolved conflict** lives exactly here — `π(EDGE | high risk)` (an
expectation) never exceeds its initial value in 76/76 checkpoints, while **greedy argmax-EDGE
rises 0.083 → 0.463** on RANDOM. The two channels disagree and the corpus does not reconcile them.

### Softmax saturation / sharpening collapse
The actor's output distribution becoming near-deterministic, so gradients vanish for
practical purposes. **Measured:** **50.5%** of R2's high-risk decisions have max-prob > 0.99,
against A0's **0.0000**. **Why it mattered:** originally mechanism 3 and the leading explanation
of the stall; **downgraded to a symptom by R3**, and dated by Phase 5 to **updates u1–u40**, with
the actor fully frozen at **u62**. `clip_frac = 0` is *normal* under saturation, not a bug.

### EDGE / CLOUD / STAY / PREEMPTIVE_REROUTE
See Part 1. **The essential repository-specific constraint (Phase 5):** `PREEMPT_REROUTE` is legal
in **exactly 0.00%** of high-risk decision entries on both frozen sets, and `MIGRATE_CLOUD` in only
**26.4% (RANDOM) / 8.0% (UNION)**. So **the effective high-risk choice is binary: STAY vs
MIGRATE_EDGE.** Any reading of a low high-risk CLOUD or PREEMPT rate as *behaviour* must be divided
by these legality rates first. The policy can be **neither blamed nor credited** for never
preempting — it is a masking fact.

### `infeasible`
**A contention counter, not an illegal-action counter** (`env.py:405-422`). The actor is masked, so
illegal actions can never be sampled. **A high `infeasible` means agents competing for the same
host — i.e. a policy that actually migrates.** 545/783 = 69.6% of relocation attempts are refused,
all cloud, because `cloud_slots = 8` for 10 agents; a forced cloud return is exactly
`−0.050 = P_infeasible × reward_scale`.

### Risk, OOF risk, and risk-conditioned behaviour
**Risk** = the predictor's uncalibrated sigmoid score. **OOF risk** = scored by a model that never
trained on that temporal block — the honest choice, and the default. **Risk-conditioned behaviour**
= the policy's action distribution *changing with* risk, which is what Δ_EDGE and the risk sweep
measure. **Why it mattered:** the risk→0 ablation is the cleanest possible test, and in Sprint 6 it
*improved* reward (20.62 → 23.13), which is what revealed the policy was not using risk at all.

### Self-healing
In this project: **relocating work off a host that is predicted to fail, before it fails**, so the
task survives. Measured by `tasks_protected_before_failure` (`protected`). **Why it mattered:** this
is the claim under test, and Part 16 states exactly how much of it is demonstrated.

---

# Part 6 — Configuration guide

All values live in `marl/config.py` and are echoed into every run's `*_config.json`.

> **Schema warning, and it has caused a wrong reading before.** In `*_config.json`, PPO
> hyperparameters live at **`config.mappo`**, *not* `config.train`. Top-level keys are
> `config / train_start_window / train_frac / device / episodes / wall_time_s / risk`, and
> `config` contains `env / reward / mappo / train`. `config.train` holds only bookkeeping:
> `episodes, rollout_episodes, seed, device, log_every, eval_episodes, out_dir, tag`.

| Parameter | Production value | What it does | What happens if it changes | Sprints that CHANGED it | Sprints that HELD it |
|---|---|---|---|---|---|
| `seed` | **20260818** | Seeds `torch.manual_seed` at `mappo.py:218`, so θ₀ is identical for every arm. | Different θ₀ ⇒ arms no longer comparable; every Sprint 7 cross-arm number becomes invalid. | Never changed after Sprint 6. | All of 6.5 and 7. |
| `episodes` | **600** | Total training episodes. | Changes `n_updates` unless `rollout_episodes` changes with it. | **R3: 600 → 2400** (dependent on the batch change). `attrbug_gamma99`: 400. | Everything else. |
| `rollout_episodes` | **8** | Episodes collected per PPO update. | **A pure variance lever.** Expected update *content* is invariant to ≤0.105 σ₈; sd follows the finite-population law `sd(8)/sd(32) = 2.236`. | **R3: 8 → 32.** `attrbug_gamma99`: 4. | Everything else. |
| `critic_target` | **`"mc"`** | `mc` = actual discounted return; `lambda` = self-referential λ-return (`mappo.py:298`). | The single most consequential knob found. | **Rung 2: `lambda` → `mc`** (production, 08-25 04:02). Field **absent** from all pre-Rung-2 configs. | R2, R3, R2_traj_repro. |
| `gamma` | **0.999** | Discount. | With `λ`, sets the 166.8-step credit horizon. Lower γ broke training (0.95 arms killed at ep 100/175). | Sprint 6 search only (0.99, 0.95 tried and abandoned). | All of 6.5 and 7. |
| `gae_lambda` | **0.995** | GAE weighting. | `1/(1−γλ)`: 0.995 → 166.8 steps; 0.95 → 19.6. Must cover a task's ~150-step exposure. | Sprint 6 search only (0.95 killed at ep 350). | All of 6.5 and 7. |
| `ppo_epochs` | **4** | Passes over each buffer. | More ⇒ more off-policy drift within an update. | Never. | All. |
| `minibatches` | **4** | Minibatch count. | Interacts with the `T mod 4` tail; 4 of 20 optimiser steps land on a degenerate chunk. | Never. | All. |
| `lr_actor` | **7e-4** | Actor learning rate. | Annealed to 1.33% of itself by update 75. | Sprint 6 search (3e-4 in `attrbug_gamma99`). | All of 6.5 and 7. |
| `lr_critic` | **1e-3** | Critic learning rate. | Final value is exactly `1e-3 × (1 − 74/75) = 1.3333333333333308e-05`. | Never after Sprint 6. | All. |
| `entropy_coef` | **0.02** | Exploration bonus. | Higher ⇒ more relocation volume, not more targeting. | **A3: 0.05.** `attrbug_gamma99`: 0.01. | A0/A1/A2/R2/R3. |
| `clip_eps` | **0.2** | PPO ratio clip. | Rarely active by construction here (see `clip_frac`). | Never. | All. |
| `value_clip_eps` | **0.2** | Critic clip. | — | Never. | All. |
| `value_coef` | **0.5** | Critic loss weight. | — | Never. | All. |
| `max_grad_norm` | **0.5** | Gradient clipping. | Caps update magnitude. | Never. | All. |
| `normalise_advantages` | **True** | Standardise advantages over the buffer. | The `√3.72 = 1.93×` attenuation of the genuine signal flows through here. | Never. | All. |
| `anneal_lr` | **True** | Linear LR decay. | `frac = 1 − (update_id−1)/n_updates`; `lr_scale` runs **1.0000 → 0.0133**. | Never. | All. |
| `separate_actors` | **True** | 10 independent actors vs one shared. | Shared parameters within an actor are what let low-risk suppression generalise into high risk. | Never. | All. |
| `episode_steps` | **400** | Steps per episode = 800 s at 2.0 s/step. | Shorter than task exposure ⇒ credit horizon breaks. | Never. | All. |
| `device` | **`cpu`** | Torch device. | **`cuda` silently breaks seed reproducibility and is 2.8× slower** on these tiny actors. Two CUDA arms exist and are *"not comparable to the CPU ladder"*. | Two abandoned CUDA arms. | Every reported arm. |
| `risk_source` | **`oof`** | `oof` / `model` / `zero`. | `model` is **in-sample for ~4/5 of the trace** ⇒ optimistic. `zero` is the ablation control. | `zero` used in ablations only. | All training. |
| `w_criticality_migration` | **1.0** | Scales the migration charge by task severity. | Was **dead code** until Sprint 6.5 fixed it (`ev["severity"]` never written). | **A2: 0.0.** | Everything else. |
| `w_criticality` | **2.0** | Criticality weight in the reward. | — | Never. | All. |
| `team_reward_share` | **0.3** | Team vs own reward split. | Drives cross-agent credit assignment (hypothesis H). | Never. | All. |
| `legacy_fixed_apply_order` | **False** | Re-enables index-ordered contention. | `True` ⇒ lowest-index agent wins 7/7 instead of 3/21. | **A0: True** (and A0_CUDA). Absent (`None`) in `mappo`, `mappo_attrbug_gamma99`. | A1/A2/A3/R2/R3. |
| `legacy_dead_migration_criticality` | **False** | Re-enables the dead criticality charge. | `True` ⇒ `crit_m ≡ 1.0`, charge pinned at `P_migration = 4.000`. | **A0: True.** | A1/A2/A3/R2/R3. |
| `cloud_slots` | **8** | Cloud capacity for 10 agents. | The direct cause of the 69.6% refusal rate. | Never. | All. |
| `neighbour_offsets` | `[-2,-1,1,2]` | Ring neighbourhood, degree 4. | An **explicit configuration choice**, not recovered from the simulator — `topology.py` says so outright. | Never. | All. |
| `train_frac` | **0.7** | Train/eval split of the trace. | Training starts ticks [9, 491]; eval [491, 698]. | Never. | All. |

**The frozen set, verified identical in all eight post-Rung-2 config artifacts and at HEAD:**

```
gamma 0.999 · gae_lambda 0.995 · clip_eps 0.2 · value_clip_eps 0.2 · entropy_coef 0.02
value_coef 0.5 · max_grad_norm 0.5 · ppo_epochs 4 · minibatches 4
normalise_advantages True · critic_target "mc" · anneal_lr True · separate_actors True
episodes 600 · rollout_episodes 8 · seed 20260818 · lr_actor 7e-4 · lr_critic 1e-3
```

---

# Part 7 — How training actually works, one cycle at a time

## 7.1 The arithmetic that governs everything

```
n_updates = episodes // rollout_episodes = 600 // 8 = 75
lr_scale  = frac    = 1 - (update_id - 1) / n_updates
            update 1  -> 1.0000      lr_actor 7.0000e-04   lr_critic 1.0000e-03
            update 75 -> 0.0133      lr_actor 9.3333e-06   lr_critic 1.3333333333333308e-05
```

`R3` holds `n_updates` at 75 by changing `episodes` with `rollout_episodes`
(`600/8 = 2400/32 = 75`) — **verified as 75 rows in both `_updates.csv`**. That is what makes R3 a
clean batch-size manipulation rather than a training-length manipulation.

## 7.2 One full cycle

```
FOR update_id = 1 .. 75:

 (a) SET LEARNING RATES
     frac = 1 - (update_id-1)/75;  lr_actor = 7e-4*frac;  lr_critic = 1e-3*frac

 (b) COLLECT 8 EPISODES                                        <- marl/rollout.py
     for each episode:
       start_tick drawn from the TRAINING window [9, 491]
       env.reset(start_tick)
       for step = 1 .. 400:                                     (2.0 s = 2 recorded ticks)
         obs   <- 48 floats per agent      (includes that node's OOF risk)
         state <- 489 floats (global)
         for each of the 10 agents:
             logits <- actor_i(obs_i)
             mask illegal actions   (masked_dist, mappo.py:77-79)
             SAMPLE an action       (training is stochastic)
         env.step(actions):
             resolve destinations  (destination.py select(); None => infeasible => STAY + penalty)
             resolve contention in ROTATING order by step_idx   (cloud_slots = 8)
             advance the trace by 2 ticks; hosts fail/recover EXOGENOUSLY
             compute reward (completion, criticality, migration cost, SLA, energy, infeasible)
         store (obs, state, action, logp, value, reward, done, mask) into the buffer
     buffer now holds T = 8 x 400 x 10 agent-steps worth of rows

 (c) COMPUTE ADVANTAGES                                        <- mappo.py
     bootstrap the final value
     GAE with gamma=0.999, lambda=0.995   (horizon 166.8 steps)
     critic target: "mc"  = actual discounted return
                    "lambda" = self-referential lambda-return  (A0 only, mappo.py:298)
     advantages standardised over the WHOLE buffer  (normalise_advantages=True)

 (d) PPO UPDATE — 4 epochs x 4 minibatches
     mb_size = T // 4
     for epoch in 1..4:
       shuffle
       for start in range(0, T, mb_size):        <- yields 5 chunks when T mod 4 != 0
         ratio      = exp(new_logp - old_logp)
         actor_loss = -min(ratio*A, clip(ratio, 0.8, 1.2)*A) - 0.02*entropy
         critic_loss= 0.5 * clipped_value_loss(value_clip_eps=0.2)
         backward; clip_grad_norm_(0.5); Adam.step()
     => 20 optimiser steps per update, of which 4 land on the degenerate tail

 (e) LOG one row to *_updates.csv
     update, episode, mean_reward, actor_loss, critic_loss, entropy, approx_kl,
     clip_frac, adv_std, value_mean, decision_frac, lr_scale, explained_var
     (NOTE: there is NO adv_mean column in the CSV.)

 (f) CHECKPOINT
     train.py:232  save *_best.pth   IF mean_r > best_mean
     train.py:252  save *.pth        after the final update
     Phase 4 only: an ADDITIVE driver also saved u000..u075
```

## 7.3 What actually happens at specific moments — with repository numbers

**At episode 1 / before update 1 (`u000`).** The checkpoint `R2_trajectory_u000.pth` is
**1,736,317 bytes** and has `buffer_T: null, stats: null` in the trajectory manifest — because
**no update has happened yet**, so there is no Adam state and no statistics. This is the random
initialisation θ₀, and it is the reference point for Phase 5's central finding.

**At update 1 (`u001`).** The checkpoint jumps to **5,207,826 bytes** — Adam moments now exist —
and `stats` becomes a 10-key dict which **does** include `adv_mean` (unlike the CSV). R2's
`clip_frac` here is **0.1625**, its highest value all run.

**Through updates 1–40.** This is where the **softmax collapse happens** (dated by Phase 5).
`clip_frac` decays through quartile means **0.0426 → 0.0196 → 0.0031 → 0.0000**. By the end,
**50.5%** of R2's high-risk decisions have max-prob > 0.99 (A0: **0.0000**).

**At updates 42–62.** The **only genuine high-risk acquisition phase in the whole run**:
`p_hi` gains **+0.0658 / +0.0660** against `p_lo`'s **+0.0195 / +0.0192** — **23%/30% of the final
Δ_EDGE**.

**At update 62.** **R2's actor freezes.** It stops moving with ~19% of training left, while the
critic is still improving fastest. This is what rules out "the critic gated the actor".

**At update 75 (final).** `lr_scale = 0.0133`; `lr_actor = 9.3333e-06`;
`lr_critic = 1.3333333333333308e-05`; **1392 Adam steps** recorded in `mappo_R2_mc_target.pth`;
`‖Δθ_actor‖ = 6.1710`. `clip_frac → 0.0000` and `approx_kl → ~0` in **A0, A1, A2, A3 alike**, with
entropy 0.30–0.37. **`_best` is bit-identical to the final for R2** — so `mean_r` never exceeded
its running best after the last improvement.

## 7.4 Two things that look like bugs and are not

1. **`clip_frac = 0.0000` for the last quarter of training.** Under a saturated softmax the ratio
   barely moves, so nothing gets clipped. Normal here.
2. **`approx_kl` going negative in 26 of 75 updates.** It is the **signed k1** estimator
   `mean(old_logp − new_logp)`, not a KL divergence. Normal.

---

# Part 8 — Experiment command reference

> **Provenance rule for this whole Part.** Only **six** commands appear verbatim anywhere in the
> `.md` corpus. The R3 training command below is one of them and is marked **VERBATIM**.
> Everything else is marked **RECONSTRUCTED** — assembled from the script's `argparse` block plus
> the values recorded in that run's `*_config.json` and its `*.log` banner. A reconstructed command
> is a faithful statement of *what the recorded configuration was*, not a quotation.
>
> **Working directory for every Python command below is `python-ai/`.**

## 8.1 SAFE — read-only, creates nothing

```bash
git status --short
```

```bash
git branch --show-current
```

```bash
git log --oneline -10
```

```bash
python -c "import torch; d=torch.load('saved_models/marl/mappo_R2_mc_target.pth', map_location='cpu'); print(list(d.keys()))"
```

```bash
python -c "import json; d=json.load(open('saved_models/marl/mappo_R2_mc_target_config.json')); print(json.dumps(d['config']['mappo'], indent=2))"
```

```bash
head -1 saved_models/marl/mappo_R2_mc_target_updates.csv && tail -1 saved_models/marl/mappo_R2_mc_target_updates.csv
```

```bash
wc -l saved_models/marl/R2_trajectory/SPRINT_7_P4_trajectory_manifest.jsonl
```

## 8.2 TESTS — read-only, prints pass/fail

```bash
python -m marl.tests_env
```

Expected: **19/19** (`SPRINT_6_5_REPORT.md` §10). Note its own caveat: *"T2 is a weak guard."*

```bash
python -m marl.sanity_test
```

Expected: **6/6**; T6 matches to 4.48e-07.

## 8.3 DIAGNOSTIC — read-only w.r.t. production code, **but writes JSON/log artifacts**

⚠ **These create files.** Every one of them writes into `saved_models/marl/` and/or `python-ai/*.log`.
Use a distinct `--tag` if you do not want to overwrite a historical artifact.

**Rule that must be followed** (learned the hard way, recorded in the corpus):

```bash
set -o pipefail
```

Without it, a `| tee` masks a nonzero Python exit code — this has already hidden a `KeyError`
that crashed *after* printing all its numbers. And **never pipe a long training run through
`tail`**: the pipe buffers everything until exit, so you see nothing for the whole run. Poll a side
artifact instead (e.g. the trajectory manifest's line count).

| Stage | Command (RECONSTRUCTED unless noted) | Writes |
|---|---|---|
| Phase 0 | `python -m marl.diag._phase0_calibration` | `SPRINT_7_PHASE0_calibration.json` |
| Phase 4 smoke | `python -m marl.diag._phase4_smoke` | `SPRINT_7_P4_SMOKE_REPORT.json` |
| Phase 4 equivalence | `python -m marl.diag._phase4_equiv` | `SPRINT_7_P4_EQUIV_SELFTEST.txt` |
| Phase 4 verify | `python -m marl.diag._phase4_verify` | stdout; **expect 18/19, B6 fails by design** |
| Phase 5 | `python -m marl.diag._phase5_risk_trajectory --device cpu --clusters 32 --start-seed 20260825 --boot 5000 --random-seed 31337 --tag main` | `SPRINT_7_PHASE5_risk_trajectory_main.json` |
| Rung 0 | `python -m marl._diag_rung0` | Rung 0 JSONs |
| Rung 1 | `python -m marl._diag_rung1_critic` | Rung 1 JSONs |
| Rung 2.5 | `python -m marl._diag_rung2_5_targets` (+ `_signtest`, `_native_dev`, `_actor_stall`, `_feasibility`) | Rung 2.5 JSONs |
| Rung 2.75 | `python -m marl._diag_rung2_75_matched_states` (**builds RANDOM/UNION**) then `_coherence`, `_offset`, `_edgeshare*`, `_plasticity`, `_mbtail`, `_stepcollapse` | frozen state sets + JSONs |
| R3 channels | `python -m marl._diag_R3_action_channels --tag R3` | **`SPRINT_7_R3_action_channels_R3.json`** |
| Rung 3 | `python -m marl._diag_rung3_dilution` (+ `_bootstrap`, `_score`) | Rung 3 JSONs |
| Divergence | `python -m marl._diag_div_geometry` (+ `_content`, `_critic`, `_logs`, `_shape`, `_variance`) | `SPRINT_7_DIV_*` |

**`--tag` trap.** Every `--tag` probe in this tree writes `<name>_<tag>.json`. A bare `ls` of the
unsuffixed name will report the artifact missing when it is not. Always check for suffixed variants.

**`--episodes` trap.** The **D3/D4 census probes silently ignore `--episodes`** — they read
`rollout_episodes` from the *checkpoint config*. This is why the four `*_b32.json` census artifacts
are **bit-identical duplicates** of their non-`b32` counterparts and **must never be read as a
batch-size control**. The real matched-batch control is `_diag_div_content.py`, which passes
`n_eps` straight to `build`.

## 8.4 TRAINING — ⚠ CREATES AND CAN OVERWRITE CHECKPOINTS

> **Do not run any of these to "check" something.** RULE 4: do not train to answer what frozen
> artifacts already answer. The whole Sprint 7 corpus is already on disk. If a number looks wrong,
> read Part 18 (Troubleshooting) — the answer is almost always an artifact-reading trap, not a
> training problem.
>
> `--tag` determines the output filenames. **Re-running a historical tag overwrites that arm.**

`marl/train.py`'s full flag surface (from its `argparse`):

```
--episodes --rollout-episodes --episode-steps --seed --lr-actor --lr-critic
--entropy-coef --w-criticality-migration --risk-source {oof,model,zero}
--critic-target {lambda,mc} --device --tag --log-every
--legacy-sprint6 --legacy-fixed-apply-order --legacy-dead-migration-criticality
```

| Arm | Command | Out dir | Output files | Runtime | Purpose |
|---|---|---|---|---|---|
| **Sprint 6 `mappo`** | `python -m marl.train --episodes 600 --rollout-episodes 8 --seed 20260818 --device cpu --tag mappo` *(RECONSTRUCTED)* | `saved_models/marl/` | `mappo{,_best}.pth`, `mappo_config.json`, `_history.csv`, `_updates.csv` | **1460 s** (log banner) | The original learned scheduler. |
| `attrbug_gamma99` | `python -m marl.train --episodes 400 --rollout-episodes 4 --lr-actor 3e-4 --entropy-coef 0.01 --seed 20260818 --device cpu --tag mappo_attrbug_gamma99` *(RECONSTRUCTED; γ0.99/λ0.95 were source-level at the time)* | same | `mappo_attrbug_gamma99*` | **938 s** | Abandoned hyperparameter regime. |
| **A0** | `python -m marl.train --episodes 600 --rollout-episodes 8 --seed 20260818 --device cpu --legacy-fixed-apply-order --legacy-dead-migration-criticality --tag A0_cpu_repro` *(RECONSTRUCTED)* | same | `mappo_A0_cpu_repro*` | not recorded | Sprint 6 defects **re-enabled** — the control. **Reproduces Sprint 6 bitwise.** |
| **A1** | same as A0 **without** the two legacy flags, `--tag A1_cpu_bugfix` *(RECONSTRUCTED)* | same | `mappo_A1_cpu_bugfix*` | not recorded | Both bug fixes on. |
| **A2** | A1 + `--w-criticality-migration 0.0 --tag A2_crit_sign` *(RECONSTRUCTED)* | same | `mappo_A2_crit_sign*` | not recorded | Tests the criticality-scaled migration charge. |
| **A3** | A1 + `--entropy-coef 0.05 --tag A3_entropy` *(RECONSTRUCTED)* | same | `mappo_A3_entropy*` | not recorded | Exploration arm. |
| **R2** | `python -m marl.train --critic-target mc --episodes 600 --rollout-episodes 8 --seed 20260818 --device cpu --tag R2_mc_target` *(RECONSTRUCTED)* | same | `mappo_R2_mc_target*` | not recorded | The production critic-target change. **Best arm.** |
| **R3** | **`python -m marl.train --critic-target mc --rollout-episodes 32 --episodes 2400 --seed 20260818 --device cpu --tag R3_batch32 2>&1`** — **VERBATIM from the `.md` corpus** | same | `R3_batch32*` | not recorded | 4× batch. Tests the variance hypothesis. |
| **Phase 4 R2 replication** | `python -m marl.diag._phase4_r2_trajectory` *(RECONSTRUCTED — an **additive driver**, not `train.py`)* | `saved_models/marl/` + `R2_trajectory/` | `R2_traj_repro*`, **`R2_trajectory/R2_trajectory_u000..u075.pth`**, `SPRINT_7_P4_trajectory_manifest.jsonl` | not recorded | Zero-variable, checkpoint-instrumented replication. **Bit-exact.** |

**`--device cpu` is mandatory.** `cuda` silently breaks seed reproducibility and is **2.8× slower**
on these tiny actors. Two CUDA arms are preserved
(`mappo_A0_CUDA_device_mismatch`, `run_A1_CUDA_killed_at_90ep.log`) and are explicitly *"not
comparable to the CPU ladder"*.

## 8.5 EVALUATION — ⚠ writes `*_eval.json` and a log

`marl/evaluate.py` flags: `--model --episodes(8) --device(cpu) --out --skip-ablation`

```bash
python -m marl.evaluate --model saved_models/marl/mappo_R2_mc_target.pth --episodes 8 --device cpu
```

*(RECONSTRUCTED.)* Produces the greedy-vs-baselines table, the behaviour probe, the risk sweep and
the risk→0 ablation. **R3 was never evaluated this way** — see Part 2.

## 8.6 GIT MUTATION — ⚠ changes repository state

```bash
git add -A
```

```bash
git commit -m "your message"
```

```bash
git push
```

See Part 12. **Do not create a branch** — the repository history does not require one, and the
current work belongs on `host-predictor-finalization`.

---

# Part 9 — Artifact guide

`saved_models/marl/` holds **25 root `.pth`**, **76 `R2_trajectory/*.pth`**, **38 `.csv`**,
**81 `.json`**, **15 `.md` + 1 `.txt`**, and **12 integrity directories** (with inconsistent
case: `_rung0_integrity` … `_PHASE5_integrity`).

## 9.1 The 15 reports — the documentary spine

Listed in the order they were written. **All are historical records: do not edit them.**

| Report | mtime | What it is |
|---|---|---|
| `SPRINT_6_5_REPORT.md` | 08-19 08:48 | The pre-Sprint-7 diagnosis. 239 lines. **The single most important primary source.** |
| `SPRINT_7_PHASE1_DIAGNOSIS.md` | 08-19 21:31 | Inspection-only; located the inverted estimator; proposed a ladder that mostly never ran. |
| `SPRINT_7_RUNG0_REPORT.md` | 08-20 06:40 | Signal real (+3.363); GAE 29.0% sign agreement. |
| `SPRINT_7_RUNG1_REPORT.md` | 08-25 02:36 | λ-target SUPPORTED for A0 (26.3% → 60.7%). |
| `SPRINT_7_RUNG2_5_REPORT.md` | 08-25 06:45 | NO-GO for a target rung; **separated three mechanisms**. Also contains the only Rung 2 write-up. |
| `SPRINT_7_RUNG2_75_REPORT.md` | 08-25 13:08 | Paired estimator; `Var(raw)/Var(paired)=3.72×`; **Δ_EDGE defined**; old metric withdrawn. |
| `SPRINT_7_R3_REPORT.md` | 08-25 18:22 | 4× batch. SNR up (p=0.0000), behaviour worse. NO-GO. |
| `SPRINT_7_RUNG3_PREREGISTRATION.md` | 08-27 12:07 | 10,688 B. Pre-registers the dilution test. |
| `SPRINT_7_RUNG3_REPORT.md` | 08-27 13:16 | Dilution real (9/9) but does not explain R3. `go_for_training_rung = False`. |
| `SPRINT_7_DIVERGENCE_REPORT.md` | 08-30 11:02 | `rollout_episodes` is pure variance; regime-selective asymmetry; **§10 names per-update checkpoints as the largest gap**. |
| `SPRINT_7_PHASE0_RECONSTRUCTION.md` | 08-30 11:59 | **20/20 instruments invert.** Mechanism 2 closed. Proposes the Phase 4 replication. |
| `SPRINT_7_PHASE4_PREREG.md` | 08-30 15:08 | 45,749 B pre-registration, 19 checks. |
| `SPRINT_7_PHASE4_REPORT.md` | 08-30 16:37 | Bit-exact replication; 76 checkpoints; 18/19. |
| `SPRINT_7_PHASE5_REPORT.md` | 08-31 03:01 | **Δ_EDGE is differential suppression.** Actor freezes at u62. |
| `SPRINT_7_FINAL_SYNTHESIS_AUDIT.md` | 08-31 04:39 | 59,913 B. 12 sections, 22-hypothesis ledger. **COMPLETE / no further training / closed.** |
| `SPRINT_7_P4_EQUIV_SELFTEST.txt` | 08-30 | The `.txt`: Phase 4 equivalence self-test output. |

**Can they be deleted?** No. **Can they be regenerated?** No — they are prose written by hand at a
point in time. They are the only record of *why*.

## 9.2 Checkpoints (`.pth`)

| Checkpoint | What it is | Regenerable? | Deletable? |
|---|---|---|---|
| `mappo.pth` / `_best.pth` | Sprint 6's learned scheduler. The 20.62 result. | Yes (1460 s) | No — it is the origin of the whole research question |
| `mappo_attrbug_gamma99.pth` | Abandoned hyperparameter regime. | Yes (938 s) | Low value, but it documents the search |
| `mappo_A0_cpu_repro.pth` | **The control.** Sprint 6 defects re-enabled; reproduces Sprint 6 **bitwise** (600/600 episodes, 75/75 updates, action histogram 9832/77/842/0). | Yes | **No** |
| `mappo_A1_cpu_bugfix.pth` | Both Sprint 6.5 bug fixes on. | Yes | No |
| `mappo_A2_crit_sign.pth` | A1 + `w_criticality_migration = 0.0`. **Name says "crit" = criticality, not critic.** | Yes | No |
| `mappo_A3_entropy.pth` | A1 + `entropy_coef = 0.05`. | Yes | No |
| `mappo_A0_CUDA_device_mismatch.pth` | A CUDA arm. **Not comparable to the CPU ladder.** | Yes | Keep — it documents why `--device cpu` is mandatory |
| `mappo_R2_mc_target.pth` | **The best arm.** `‖Δθ_actor‖ 6.1710`, 1392 Adam steps, `lr 9.3333e-06`. | Yes — **and proven bit-exactly so by Phase 4** | **No** |
| `mappo_R2_mc_target_best.pth` | **Bit-identical to the above.** | — | Redundant, but keep for provenance |
| `R3_batch32.pth` | The 4× batch arm. | Yes | No |
| `R3_batch32_best.pth` | **A genuine update-45 checkpoint** — `lr 2.8933e-04` ⇒ `lr_scale 0.41333`, 872 Adam steps vs 1440. The only mid-training checkpoint outside `R2_trajectory/`. | No (would need instrumentation) | **No** |
| `R2_traj_repro.pth` / `_best.pth` | Phase 4's replication of R2. Matches the original. | Yes | No |
| `R2_trajectory/R2_trajectory_u000..u075.pth` | **76 per-update snapshots of R2.** The missing observable. `u000` = 1,736,317 B (no Adam state); `u001+` = 5,207,826 B. | Yes | **Absolutely not** — this is what Phase 5 read |
| `saved_models/failure_predictor.pth` | Deployment predictor (refit on everything ⇒ **in-sample for ~4/5 of the trace**). | Yes | No |
| `failure_predictor_legacy.pth` / `_OLDDATA.pth` | The two comparison arms. | Yes | No — they are the evidence that features mattered |
| `htcf_model.pth`, `device_failure_predictor.pth` | S-early baseline and the IoMT-device predictor. | Unclear | Keep |

## 9.3 CSV artifacts

**`*_history.csv`** — one row per episode.
```
episode,start_tick,reward,success_rate,lost,critical_lost,relocations,preemptive,
sla,protected,energy,infeasible
```

**`*_updates.csv`** — one row per PPO update (75 rows).
```
update,episode,mean_reward,actor_loss,critic_loss,entropy,approx_kl,clip_frac,
adv_std,value_mean,decision_frac,lr_scale,explained_var
```
⚠ **There is no `adv_mean` column.** It exists only inside the Phase 4 trajectory manifest's
`stats` dict. Several readings have gone looking for it in the CSV.

**`saved_models/host_lead_time*.csv`** — columns
`node,t_fail,verdict,lead,run_lead,band_alarm,quiet_alarm,n_band,deg,max_p`. The `verdict` column
is what produces "23 EARLY_WARNING / 7 MISSED / 2 STANDING_ALARM".

All CSVs are **regenerable** by re-running their producer, and **not deletable** without losing the
per-episode/per-update record that the reports cite.

## 9.4 JSON artifacts

| Family | Content |
|---|---|
| `*_config.json` | The full recorded configuration of a run. Top keys `config / train_start_window / train_frac / device / episodes / wall_time_s / risk`; `config` = `env / reward / mappo / train`. **The authoritative record of what an arm actually was.** |
| `*_eval.json` | The held-out evaluation table for an arm. **9 exist** — and none for R3 or `R2_traj_repro`. |
| `diag_*.json`, `SPRINT_7_*.json` | Per-stage diagnostic results. |
| `SPRINT_7_R3_action_channels_R3.json` | 36,293 B, md5 `53955350c3d84c10c234d33464ea8d5c`. Four-action Δ for six arms on both frozen sets. `status = 'EXPLORATORY -- not preregistered'`. |
| `SPRINT_7_PHASE0_calibration.json` | The 20-instrument inversion table, machine-readable. |
| `SPRINT_7_PHASE5_risk_trajectory_{main,parity}.json` | The 76-checkpoint trajectory readings. |
| `failure_predictor_meta.json` | `threshold 0.18`, `sequence_length 10`, `num_features 26`, the 26 feature names. |
| `host_metrics{,_legacy,_OLDDATA}.json` | Pooled OOF classification metrics + trivial-rule baselines. |
| `host_lono.json` | Leave-one-node-out per-node + pooled. |

> **Reading trap for the frozen Rung-2.75 artifact.** Its top-level key is **`by_state_source`**,
> not `results`, and per-arm numbers live at `["by_state_source"][SRC]["evaluated"][ARM]`. Inside
> that, `entropy_mean`, `maxp_mean`, `frac_maxp_gt_099`, `frac_argmax_edge` and `argmax_counts` are
> over the **high-risk subset only**, while `p_edge_risk_lt_02`, `p_edge_risk_ge_06`,
> `risk_response_*` and `spearman_risk_vs_p_edge` are over **all decision entries**. **Mixing the
> two denominators has produced a wrong reading once.**

## 9.5 `.jsonl`

`R2_trajectory/SPRINT_7_P4_trajectory_manifest.jsonl` — one line per checkpoint. Row keys:
```
update, file, bytes, md5, sha256, episode, lr_scale, lr_actor, lr_critic, buffer_T, stats
```
`u000` has `buffer_T: null, stats: null`; `u001+` carry a 10-key `stats` dict that **includes
`adv_mean`**. This file is also the right thing to **poll** (by line count) to watch a long
trajectory run without piping through `tail`.

## 9.6 `.md5` — the integrity manifests

Twelve directories: `_rung0_integrity`, `_rung1_integrity`, `_rung2_5_integrity`,
`_rung2_75_integrity`, `_R3_integrity`, `_RUNG3_integrity`, `_DIVERGENCE_integrity`,
`_AUDIT_integrity`, `_PHASE0_integrity`, `_PHASE4_integrity`, `_PHASE4RUN_integrity`,
`_PHASE41_integrity`, `_PHASE5_integrity`. See Part 11.

## 9.7 `.py` in the artifact tree

There are none — all code lives in `python-ai/marl/` and `python-ai/marl/diag/`. Diagnostic scripts
are **artifacts of the research** in the sense that they are the only executable definition of each
historical metric, and they are **hashed by the integrity manifests** precisely so that a number can
be traced to the exact script version that produced it.

## 9.8 The `.log` files (69 in `python-ai/`)

These are the **primary chronological evidence** for the whole project — their modification times
are what date every stage in Part 21, and their banners are what recovered the Sprint 6
hyperparameter search. **Two are 0 bytes** (`chain_cpu.log`, `chain_A2_A3.log`) and carry no
information. **Do not delete any of them.**

---

# Part 10 — How to inspect a model checkpoint

## 10.1 What is inside a `.pth`

Run this first — it lists the top-level keys without loading anything into a model:

```bash
python -c "import torch; d = torch.load('saved_models/marl/mappo_R2_mc_target.pth', map_location='cpu'); print(type(d)); print(list(d.keys()))"
```

A training checkpoint from this project contains, at minimum:

| Element | What it tells you |
|---|---|
| **actor state dicts** (10 of them, `separate_actors=True`) | The policy. `48→[128,128]→4`, 23,300 params each. |
| **critic state dict** | `499→[256,256]→1`, 194,049 params. Training-only. |
| **optimizer state** (Adam) | `exp_avg`, `exp_avg_sq`, and a **`step` count** — R2's final is **1392**; `R3_batch32_best` has **872** vs R3's final **1440**. The step count is the most reliable way to identify *which update* a checkpoint came from. |
| **learning rates** | R2 final: `lr_actor 9.3333e-06`, `lr_critic 1.3333333333333308e-05`. Divide by the base rate to get `lr_scale`, then invert the anneal to get the update id. |
| **config** | The recorded configuration. Cross-check against the sibling `*_config.json`. |
| **metadata** (episode, mean reward) | Which episode the checkpoint was taken at, and the running mean reward that triggered a `_best` save. |
| **checkpoint type** | Whether it was the `_best` save (`train.py:232`) or the final (`train.py:252`). |

## 10.2 Recovering *which update* a checkpoint is

This is the trick that identified `R3_batch32_best`:

```
lr_scale   = saved_lr_actor / 7e-4          (or saved_lr_critic / 1e-3)
frac       = 1 - (update_id - 1) / n_updates
=> update_id = 1 + n_updates * (1 - lr_scale)
```

Worked example, `R3_batch32_best.pth`: saved `lr = 2.8933e-04` ⇒ `lr_scale = 0.41333` ⇒
`update_id = 1 + 75 × (1 − 0.41333) = 45.0`. Confirmed independently by **872 Adam steps**
(45 updates × 20 optimiser steps ≈ 900, minus degenerate-tail bookkeeping) against the final's 1440.

## 10.3 Comparing two checkpoints — the four possible answers

Work down this list in order. Each step distinguishes a *different* kind of "same".

**Step 1 — byte-identical?**
```bash
md5sum saved_models/marl/mappo_R2_mc_target.pth saved_models/marl/mappo_R2_mc_target_best.pth
```
Equal hashes ⇒ **identical**. This is the correct answer for `R2` vs `R2_best`: same
`‖Δθ_actor‖ 6.1710`, same Adam state (1392 steps, `lr 9.3333e-06`), same reproduced Δ. **The
shipped R2 corpus contains exactly one distinct policy snapshot.**

**Step 2 — structurally identical?** Same keys, same tensor shapes, different values. Check with a
key/shape diff. Two arms trained from the same `seed 20260818` are always structurally identical,
because `mappo.py:218` seeds θ₀ before any divergence.

**Step 3 — semantically comparable?** This is the one people get wrong. Two checkpoints are
comparable **only if their reward functions are the same**. In this repository they are **not**:

| Comparable group | Reward convention (risk-threshold baseline) |
|---|---|
| `{A0}` (and `A0_CUDA`) | **76.50** — legacy switches on |
| `{A1, A3, R2, R3, R2_traj_repro}` | **67.31** — both bug fixes on |
| `{A2}` | **76.54** — `w_criticality_migration = 0` |

Sprint 6.5 §6 states the rule and gives the internal control: **`static-no-migration = −37.06` in
every arm**, and risk-threshold's *behaviour* is byte-identical everywhere (reloc 36.2, prot 30.2,
lost 3.4) — **only its reward moves**. So compare arms on the **reward-independent columns**
(protected, relocations, lost, success) and on the **within-arm gap** to the baseline. Comparing
A0's 20.62 to R2's −0.46 directly is a **scale error**.

**Step 4 — different only because of metadata?** Two checkpoints can differ in bytes while holding
identical parameters, because `episode`, `mean_reward`, wall time or config echo differ. Diff the
tensors, not the file.

**Step 5 — genuinely numerically different?** Compare parameter tensors directly and report the max
absolute difference. Phase 0 used exactly this to prove a script edit was harmless:
**`MAX |diff| = 0.000e+00`** across all 10 shared arm×source values and all four state-set sizes.
Phase 4 used it to prove R2 replicated **bit-exactly, storage-for-storage**.

## 10.4 The 1-ULP lesson

`_phase4_verify.py:78`'s pre-registered `final_lr_critic` literal is `1.3333333333333309e-05`
(bits `…3688`). Every artifact stores `1.3333333333333308e-05` (bits `…3687`), which is the exact
float64 value of `1e-3 * (1 - 74/75)`. **Check B6 therefore FAILS by design, and was left wrong on
purpose** so the FAIL stands in the record. When comparing floats from this repository, compare at
the ULP level and expect to have to decide whether a 1-bit difference is a defect or a
transcription error — here it is the latter, and it is *documented* rather than patched.

---

# Part 11 — How to check experiment integrity

## 11.1 Why this exists

Large parts of the research tree were **untracked in git** for much of the project. The
`git status` snapshot at the start of this handbook's session shows the shape of the problem:
`R2_trajectory/`, `marl/diag/`, six `SPRINT_7_PHASE*` documents and five `_*_integrity/`
directories were all `??` (untracked) at once. When artifacts are untracked, git cannot tell you
whether a file changed, so **MD5 manifests were the only available before/after evidence** that a
diagnostic run had not silently mutated production code or an earlier arm's outputs.

By Phase 0 this had improved: git HEAD `a8df6d9` tracked every artifact, checkpoint, log and
diagnostic script, and `git diff --stat HEAD` was empty. Phase 0 says so explicitly — that is
*"a **stronger baseline than MD5 alone**"*. **Both mechanisms are now in place; use git first and
the manifests as the historical record.**

## 11.2 Manifest anatomy

Each stage has a `_<stage>_integrity/` directory. `_PHASE5_integrity/` is the fullest example —
**10 files**:

```
SPRINT_7_P5_artifacts_before.md5     156 lines      SPRINT_7_P5_artifacts_after.md5   160 lines
SPRINT_7_P5_code_before.md5           52 lines      SPRINT_7_P5_code_after.md5         53 lines
SPRINT_7_P5_inputs_before.md5          3 lines      SPRINT_7_P5_inputs_after.md5        3 lines
SPRINT_7_P5_trajectory_before.md5     78 lines      SPRINT_7_P5_trajectory_after.md5   78 lines
SPRINT_7_P5_prior_manifests.md5       69 lines      SPRINT_7_P5_own_manifests.md5      10 lines
```

The five manifest **kinds**:

| Kind | Covers | Expectation after a read-only diagnostic |
|---|---|---|
| `code` | `marl/**.py` | **Must be unchanged**, except for files the stage legitimately adds. |
| `artifacts` | `saved_models/marl/**` | Prior artifacts unchanged; **new** files appear. |
| `inputs` | the trace / OOF risk | **Must be unchanged**, always. |
| `trajectory` / `checkpoints` | the `.pth` corpus | **Must be unchanged**, unless the stage trains. |
| `prior_manifests` / `own_manifests` | the manifests themselves | The chain-of-custody link to earlier stages. |

Line format:

```
c88a7c35edbe50a3b0b17f4107012a10 *./__init__.py
```

The `*./` prefix means **binary mode with a relative path**, which has one hard consequence:

> ⚠ **A manifest must be verified from the same working directory it was captured in.** Verifying
> `_PHASE5_integrity/*.md5` from anywhere other than that stage's cwd produces "No such file"
> errors for every line, which look exactly like corruption and are not.

## 11.3 The nine-step procedure

```bash
set -o pipefail
```

**1. Snapshot before.** Capture `code`, `artifacts`, `inputs` and `checkpoints` manifests.

**2. Run.** Execute the diagnostic or training command.

**3. Verify what must not have changed.** From the capture cwd:
```bash
md5sum -c saved_models/marl/_PHASE5_integrity/SPRINT_7_P5_code_before.md5
```
```bash
md5sum -c saved_models/marl/_PHASE5_integrity/SPRINT_7_P5_inputs_before.md5
```
Expect `3/3 OK` for inputs. Any input failure invalidates the run.

**4. Identify new files.** Diff the `after` manifest's path list against the `before`'s. Phase 5's
`artifacts` went 156 → 160 lines: **four new artifacts**, matching the four files that report lists.

**5. Identify modified files.** A path present in both with a **different hash**. This is the case
that needs a human decision — see 11.4.

**6. Identify deleted files.** A path in `before` and not in `after`. In this repository the correct
count is always **zero**; nothing was ever deleted.

**7. Verify checkpoints.** The `trajectory` manifests going 78 → 78 lines with identical hashes is
the proof that Phase 5 was read-only with respect to the 76 checkpoints it analysed.

**8. Verify inputs again** against the *original* Rung 0 manifest, not just the previous stage's.
This is what the `prior_manifests` file is for — it hashes the earlier manifests so the chain cannot
be silently re-based.

**9. Interpret self-reference.** ⚠ **The circular-manifest trap.** A manifest that includes *itself*
in its own file list can never verify, because its hash is computed before it is written. A single
`FAILED` line naming the manifest itself is **not a modification** — this was diagnosed once and
misread once. Check the failing path before concluding anything.

## 11.4 How the three real failures were handled — the template

Phase 0 re-verified the whole 14-manifest prior chain: **11 clean, 3 failed**, and it refused to
"fix" any of them. Copy this pattern:

1. **`_rung0_integrity/code_before.md5`** — `config.py`/`mappo.py`/`train.py` fail. **Resolution:**
   trace *when* the hashes moved. They moved **exactly once**, between 08-19 21:35 and 08-25 04:02
   — the Rung 2 MC-target production change — and are byte-identical in every later manifest.
   **Verdict: documented and intended.** *This is the good outcome: a failure that dates a known
   change.*
2. **`_rung2_75_integrity/artifacts_after.md5`** — a **report** was edited after the manifest.
   **Verdict: a minor RULE-12 process slip; no data artifact affected.** *Recorded, not hidden.*
3. **`_rung2_75_integrity/code_after.md5`** — two diagnostic scripts changed, one of which
   **constructs RANDOM and UNION**. **Verdict: not accepted on narrative grounds.** Instead it was
   **proved empirically** — all 10 shared arm×source values and all four state-set sizes are
   bit-identical across the two script versions, **MAX |diff| = 0.000e+00**; the edit was purely
   additive (it added the `R3` column). *This is the standard to meet: when the failing file could
   have changed a number, re-derive the number.*

## 11.5 The git-based check that now supersedes MD5 alone

```bash
git diff --stat HEAD
```

Empty output means every tracked artifact, checkpoint, log and script matches the commit. Combine
the two: **git for "has anything changed at all", manifests for "what was true at each historical
stage".**

---

# Part 12 — Git guide

> **This repository does NOT require a new branch.** The current work belongs on the existing
> branch `host-predictor-finalization`. Nothing in the repository history calls for branching, and
> this handbook does not recommend it.

## 12.1 Where am I?

```bash
git branch --show-current
```
Expected: `host-predictor-finalization`. The default/main branch is `main`.

```bash
git status --short
```

Reading the two-column status codes:

| Code | Meaning | Example from this repo |
|---|---|---|
| `??` | **Untracked** — git has never seen this file | `python-ai/marl/diag/`, `R2_trajectory/`, the `SPRINT_7_PHASE*` reports before they were added |
| `A ` | **Added, staged** — new file, staged for the next commit | `SPRINT_7_DIVERGENCE_REPORT.md`, the seven `_DIVERGENCE_integrity/*.md5` |
| ` M` | Modified, **not** staged | — |
| `M ` | Modified **and** staged | — |
| `MM` | Modified, staged, then modified again | — |
| ` D` / `D ` | Deleted (unstaged / staged) | should never appear here — nothing in this project was ever deleted |

**Left column = the index (staged). Right column = the working tree (unstaged).**

## 12.2 The four states a file can be in

```
untracked  ──git add──▶  staged (new)  ──git commit──▶  tracked & clean
                                                              │
                                                        (edit it)
                                                              ▼
                                                    tracked & modified
                                                              │
                                                          git add
                                                              ▼
                                                     staged (modified)
```

- **Tracked** = git knows the file and records its history.
- **Untracked** = git ignores it entirely; **`git diff` will never mention it and `git stash` will
  not protect it.** This is exactly why the MD5 manifests existed (Part 11.1).

## 12.3 Seeing what changed

```bash
git diff
```
Unstaged changes only — working tree vs index.

```bash
git diff --cached
```
Staged changes only — index vs HEAD. **This is what you are about to commit.**

```bash
git diff --stat HEAD
```
Everything, tracked, summarised. **Empty output = every tracked artifact matches the commit.**
Phase 0 used exactly this and called it *"a stronger baseline than MD5 alone."*

```bash
git log --oneline -10
```

```bash
git log --stat -1
```

## 12.4 Does my local tree match GitHub?

```bash
git fetch
```
*(Read-only: updates remote-tracking refs, changes no local file.)*

```bash
git status -sb
```
The first line reads e.g. `## host-predictor-finalization...origin/host-predictor-finalization [ahead 2]`.
`ahead N` = N local commits not pushed. `behind N` = N remote commits not pulled.

```bash
git log --oneline origin/host-predictor-finalization..HEAD
```
Lists exactly the commits you have that GitHub does not.

## 12.5 Safely pushing experiment artifacts

The artifacts here are large but not huge (the biggest single file is a 5.2 MB checkpoint; the 76
trajectory checkpoints are ~396 MB in total). The sequence:

**1. Look before you stage.**
```bash
git status --short
```
Confirm every `??` line is something you intend to commit. **Untracked ≠ unimportant** — six
Sprint 7 reports and the entire `R2_trajectory/` were `??` at once.

**2. Stage.** ⚠ *modifies the index*
```bash
git add -A
```

**3. Re-read what you staged.** ⚠ *read-only, but do not skip it*
```bash
git diff --cached --stat
```

**4. Commit.** ⚠ *writes history*
```bash
git commit -m "Sprint 7 Phase 5: risk trajectory over the 76-checkpoint R2 corpus"
```

**5. Push.** ⚠ *publishes to GitHub — an external service; this is not locally reversible*
```bash
git push
```

If the branch has no upstream yet, git will say so and suggest:
```bash
git push -u origin HEAD
```

**Before pushing, confirm** the artifacts you are publishing are ones you intend to be public: the
trace CSVs, checkpoints, logs and reports all go up as-is.

## 12.6 What not to do

- **Do not create a branch** for documentation or artifact commits here.
- **Do not `git checkout --` or `git restore`** a file in `saved_models/` to "clean up" — that
  destroys an experiment output, and Part 18 has a safer route for every symptom.
- **Do not rewrite history** (`rebase`, `commit --amend`, force-push). The reports cite commit
  `a8df6d9` by hash; rewriting invalidates the integrity chain described in Part 11.

---

# Part 13 — Experiment result cards

Compact decision-record form. Part 2 has the full 14-field narrative for each of these; this Part is
the version you scan when you need the verdict fast. **Every reward number is only comparable within
its own arm group — see Part 10.3.**

---

### S-early — the CloudSim digital twin

| Field | |
|---|---|
| **Question** | Can a CloudSim-Plus digital twin produce a reproducible host-degradation trace? |
| **Hypothesis** | Deterministic seeding gives a replayable trace. |
| **Control / Treatment** | none — construction stage |
| **Variable** | — |
| **Metrics** | trace completeness; 10 nodes × 1500 ticks |
| **Result** | `SIM_SEED = 20260817L`, `MAX_SIMULATION_SECONDS = 1500.0`, `DEVICE_COUNT = 10`; exports `failure_log.csv`, `failure_history.csv`, `device_failure_history.csv` |
| **Verdict** | Achieved |
| **What changed** | The trace became the fixed substrate for everything after it |
| **What did not** | — |
| **Ruled out** | — |
| **Survived** | The trace is still the input to every experiment in this handbook |
| **Why next** | A trace alone predicts nothing; a predictor was needed |

---

### Sprint 3 / 3.5 / 3.75 — the degenerate dataset

| Field | |
|---|---|
| **Question** | Why is the host failure predictor degenerate? |
| **Hypothesis** | The observable features carry no pre-failure signal. |
| **Control** | the original trace |
| **Treatment** | a regenerated trace with progressive degradation |
| **Variable** | the failure model in the simulator |
| **Metrics** | number of non-constant observable columns; failures/recoveries |
| **Result** | Original: **5 of 12** observable columns constant, failures were *"instant death at a scheduled time, no symptoms"*. Regenerated: **0 of 12** constant, 32 failures / 32 recoveries, counters reset on recovery, range 0..44 |
| **Verdict** | Root cause found and fixed at the source |
| **What changed** | The dataset — not the model |
| **What did not** | The model architecture |
| **Ruled out** | "The BiLSTM is wrong" |
| **Survived** | *Fix the data before you tune the model* |
| **Why next** | With a real signal present, the feature set itself had to be validated |

---

### Sprint 4 — feature-set ablation for the predictor

| Field | |
|---|---|
| **Question** | Do the 26 engineered features beat the legacy 7, and beat trivial rules? |
| **Hypothesis** | First differences + time-since counters carry the pre-failure signal. |
| **Control** | `legacy` (7 raw features), `OLDDATA` (old trace), and three trivial rules |
| **Treatment** | `observable` — 12 observable + 12 `d_*` + `time_since_active` + `time_since_linkup` = **26** |
| **Variable** | the feature set (and, for OLDDATA, the dataset) |
| **Metrics** | PR-AUC (primary), ROC-AUC, F1, precision, recall |
| **Result** | observable **PR-AUC 0.28634** (13.3× the 0.02146 positive rate), ROC 0.81594, F1 0.47297, prec 0.36972, rec 0.65625, acc 0.9686, tn 14,232 / fp 358 / fn 110 / tp 210 on n=14,910. legacy 0.0777 / 0.6250 / 0.0467. OLDDATA 0.2174 / 0.9165 / 0.3439 at n=1,680. Trivial: always-negative F1 0; `linkUp==1` 0.0875 / 0.0449; `degraded==1` 0.7333 recall but F1 0.0657. LONO node 0: PR-AUC 0.4042 / ROC 0.9260 / F1 0.5634 |
| **Verdict** | Feature engineering is load-bearing; the predictor is real but **uncalibrated and near-bimodal** |
| **What changed** | The 26-feature set was frozen; `threshold = 0.18` |
| **What did not** | Leakage discipline — `assert_no_leakage()` and the `FORBIDDEN_COLUMNS` list stayed |
| **Ruled out** | Trivial rules as an explanation for the predictor's score |
| **Survived** | OOF risk as the leakage-safe channel the MARL agents consume |
| **Why next** | A risk channel exists — now something has to *act* on it |

---

### Sprint 5 — patient criticality

| Field | |
|---|---|
| **Question** | Can clinical urgency be encoded deterministically and matched across Java and Python? |
| **Hypothesis** | A weighted HSI/VITALS/AGE score reproduces the Java values exactly. |
| **Control** | `JAVA_PRIORITIES_T0 = [0.448, 0.495, 0.542, 0.589, 0.634, 0.679]` |
| **Treatment** | the Python reimplementation |
| **Variable** | language/runtime |
| **Metrics** | max abs deviation |
| **Result** | `self_check()` agrees to **< 5e-4**. Weights HSI 0.60 / VITALS 0.30 / AGE 0.10, `MAX_URGENCY_BOOST 0.5`, `URGENCY_HORIZON 300.0`, `BASE_DEADLINE 60.0`, `SLACK 5.0`, `PATIENT_COUNT 10` |
| **Verdict** | Parity achieved |
| **What changed** | Criticality became a reward term |
| **What did not** | — |
| **Ruled out** | Cross-language drift |
| **Survived** | ⚠ **Severity's real support is only `[0.320, 0.644]`** (`0.036·(task_id % 10) + 0.32`), so later probe grids spanning `[0,1]` were **~3× too wide** and their spans understate real sensitivity |
| **Why next** | The environment was now complete enough to train a policy in |

---

### Sprint 6a — MAPPO hyperparameter search

| Field | |
|---|---|
| **Question** | Which hyperparameter regime trains at all? |
| **Hypothesis** | — (search) |
| **Control** | — |
| **Treatment** | six runs |
| **Variable** | γ, λ, LR, entropy, rollout size |
| **Metrics** | mean reward trajectory, explained variance |
| **Result** | `attrbug_gamma99` **COMPLETED** 938 s, greedy TRAIN −146.55. `gamma95` **killed** ep 100 (−90 → −170). `attrfix_gamma95` killed ep 175. `g999_lowEV` killed ep 75 (ev +0.37). `lam95` killed ep 350 (reward −24 → −9.45, ev 0.92). **`mappo` COMPLETED 1460 s** |
| **Verdict** | γ 0.999 / λ 0.995 selected — horizon `1/(1−γλ) = 166.8` |
| **What changed** | The frozen production hyperparameters |
| **What did not** | Everything after Sprint 6 held these constant, with **two documented exceptions** (`critic_target` at Rung 2, `rollout_episodes` at R3) |
| **Ruled out** | Short-horizon regimes |
| **Survived** | The `mappo.pth` checkpoint |
| **Why next** | It trained — but did it *beat anything*? |

---

### Sprint 6b — the baseline comparison **(the result that started everything)**

| Field | |
|---|---|
| **Question** | Does MAPPO do anything a one-line threshold on the risk channel would not? |
| **Hypothesis** | A learned multi-agent scheduler beats hand-written rules. |
| **Control** | risk-threshold, reactive, random, static-no-migration |
| **Treatment** | greedy MAPPO |
| **Variable** | the policy — **identical episode starts, failure sequences, arrivals and initial placement for every row** (`rollout.py` guarantees it) |
| **Metrics** | reward, success rate, lost, critical lost, relocations, **protected** |
| **Result** | MAPPO **20.62** / 0.719 / 11.2 lost / 4.4 crit / 35.9 reloc / **protected 3.2**. Risk-threshold **76.50** / 0.847 / 3.4 / 0.8 / 36.2 / **protected 30.2**. reactive −0.09; random −8.11; static **−37.06** / 0.459 |
| **Verdict** | ❌ **MAPPO loses decisively to a one-line rule** |
| **What changed** | The research question — from *"can it learn?"* to *"why doesn't it use risk?"* |
| **What did not** | The environment; the reward; the trace |
| **Ruled out** | "MAPPO is fine, the baselines are weak" — static −37.06 shows migration matters; random −8.11 shows the ordering is real |
| **Survived** | Three damning diagnostics: the **302-observation probe spans only 0.004** in EDGE probability while correlating **+0.990** with risk (so it *tracks* risk with no *magnitude*); the risk buckets are flat (0.0–0.2 n=10,200 → 8.4%; 0.6–0.8 n=246 → 12.2%; 0.8–1.0 n=300 → 9.7%); and **zeroing the risk channel entirely IMPROVES reward 20.62 → 23.13** |
| **Why next** | The risk→0 ablation is the smoking gun: the policy is not using risk at all. Sprint 6.5 went hunting for the reason |

---

### Sprint 6.5 — the eleven-hypothesis diagnosis

| Field | |
|---|---|
| **Question** | Why is the policy blind to risk? |
| **Hypothesis** | Eleven candidates, A–K |
| **Control** | A0 (bitwise reproduction of Sprint 6) |
| **Treatment** | A1 (bug fixes), A2 (criticality off), A3 (entropy 0.05) |
| **Variable** | one per arm |
| **Metrics** | true advantage by risk regime; BC accuracy; probe span; reward |
| **Result** | **A, B, C, D, J, K refuted. F, G, H supported.** Four load-bearing measurements: (1) at high risk the true advantage of EDGE is **+2.671 ± 0.806** vs **−1.337 ± 0.967** at low risk — a real **+4.01** gap, with **0 replay mismatches**; (2) a behaviour-cloning head reaches **accuracy 1.0000** with probe span **0.3935** vs the policy's 0.004 — *the observation carries the signal; the policy discards it*; (3) the learned estimator reads **−0.266 at high risk against a true +2.671** — an **inversion**; (4) the horizon-corrected value is **+8.267**, which killed the proposed `P_risk_expose` arm |
| **Verdict** | The signal is in the observation and the truth; the **estimator** is inverted |
| **What changed** | Two real defects fixed: contention 7/7 → 3/21 (still ~70% both before and after; 545/783 = **69.6% all cloud**, A1 makes 1,060 cloud attempts vs 10 edge), and the **dead** `w_criticality_migration` (`crit_m ≡ 1.0`, `P_migration = 4.000` → span 5.280–6.576). *"Neither change adds any `if risk > threshold` logic."* |
| **What did not** | A0 reproduced Sprint 6 **bitwise**: action histogram 9832 / 77 / 842 / 0 |
| **Ruled out** | Six hypotheses; also the sweep-span story — spans A0 0.004 / A1 0.016 / A2 0.004 / A3 0.002 / **BC 0.3935**, i.e. *"4× / 25× short"*. Criticality directions −0.987 / −0.989 / −0.993 / **+0.801**; A2 was expected to flip the sign and *"It did not"* |
| **Survived** | The zero-risk ablation reading: *"It protects by volume, not by targeting."* 19/19 and 6/6 tests pass (T6 to 4.48e-07) with the self-caveat *"T2 is a weak guard"*. Eight limitations listed |
| **Why next** | §13 proposed a **smaller** `rollout_episodes` — note that R3 later tried the **opposite**. Sprint 7 opened to attack the inverted estimator directly |

---

### Sprint 7 Phase 1 — inspection-only diagnosis

| Field | |
|---|---|
| **Question** | Is the dead gradient at update 60+ a stalled optimiser or a converged one? |
| **Hypothesis** | *(verbatim)* *"PPO is converging correctly onto an advantage estimate whose action ordering at high risk is inverted relative to the truth. The dead gradient at update 60+ is what convergence looks like. Not a stalled optimiser — a converged one, pointed the wrong way."* |
| **Control / Treatment** | none — *"inspection only. No code modified, no training run, no artifact overwritten."* |
| **Variable** | — |
| **Metrics** | greedy action census; truth-vs-estimator table |
| **Result** | Greedy A0 plays CLOUD in **842 of 855** legal states (**98.5%**) and EDGE in **77 of 10,751** (**0.7%**) — *"cloud whenever legal, else stay — a rule with no risk term in it, built on a −0.050 pseudo-no-op"* |
| **Verdict** | Hypothesis stated, **not yet tested** — self-caveated at n=4 / n=9 |
| **What changed** | Nothing on disk |
| **What did not** | Everything |
| **Ruled out** | Nothing yet |
| **Survived** | *"Establishing or refuting it is rung 0 of the ladder."* ⚠ The B1–B3 / C1–C4 / E1–E3 ladder it proposed **has no artifacts** — what actually ran was different |
| **Why next** | Rung 0 had to establish that a signal exists at all |

---

### Rung 0 — does the signal exist?

| Field | |
|---|---|
| **Question** | Is there a real advantage signal for EDGE at high risk, and does GAE see it? |
| **Metrics** | true advantage gap; sign agreement |
| **Result** | Gap **+3.363**; truth-vs-GAE agreement **0.7012** overall but **0.3931** at high risk; **GAE sign agreement 29.0%** at high risk |
| **Verdict** | ✅ Signal real. ❌ GAE is **below chance** where it matters |
| **Ruled out** | "There is nothing to learn" |
| **Why next** | If GAE is wrong, is it the *target*? |

---

### Rung 1 — is the critic target to blame?

| Field | |
|---|---|
| **Control / Treatment** | λ-return target vs Monte-Carlo target, offline |
| **Metrics** | high-risk sign agreement; estimated advantage |
| **Result** | **26.3% → 60.7%**; estimate **−5.24 → −1.07** |
| **Verdict** | ✅ SUPPORTED **for A0** |
| **Why next** | It worked offline — try it in production |

---

### Rung 2 — the one production change

| Field | |
|---|---|
| **Variable** | `critic_target: "lambda" → "mc"` — **the single production edit in all of Sprint 7** |
| **Metrics** | Δ_EDGE (RANDOM / UNION); reward; protected |
| **Result** | Δ_EDGE **+0.2024 / +0.1579** — the best of any arm. But reward **−0.46** and protected **9.9** |
| **Verdict** | ⚠ Best risk response, **worse task performance** |
| **Ruled out** | — |
| **Survived** | R2 became the reference arm. `R2_best` is **bit-identical** to `R2` |
| **Why next** | Rung 2.5 asked whether a further *target* rung was worth it |

---

### Rung 2.5 — NO-GO, and the three-mechanism split

| Field | |
|---|---|
| **Result** | **130/130** agreement on the replication check; high-risk sign error **58.4% → 7.8%** — **≈87% fixed** |
| **Verdict** | 🛑 **NO-GO** for another target rung |
| **What changed** | The problem was **decomposed into three mechanisms** |
| **Survived** | §G.5 left **UNRESOLVED** |
| **Why next** | Mechanism 2 (a per-state offset) needed its own estimator |

---

### Rung 2.75 — the paired estimator and the metric that stuck

| Field | |
|---|---|
| **Result** | `gae(s,a) = c(s) + paired(s,a)` with `a_true(s, a_ref) ≡ 0`. **`Var(raw)/Var(paired) = 3.72×`** ⇒ `√3.72 = 1.93×` attenuation through pooled normalisation. Pairing lifts high-risk agreement from **below chance to 0.71 / 0.95 for every arm** |
| **Verdict** | ✅ The sign error is a **per-state offset**, not a target defect |
| **What changed** | **Δ_EDGE was defined here** — `π(EDGE|risk≥0.6) − π(EDGE|risk<0.2)` over frozen sets **RANDOM (4,975 entries / 216 high-risk)** and **UNION (74,237 / ~3,592)**, plus `Spearman(risk, π(EDGE))` |
| **Ruled out** | — |
| **Survived** | ⚠ §A.4 **withdrew** the earlier metric. The minibatch-tail arithmetic was characterised here: `mb_size = T//4`, `range(0,T,mb_size)` yields **5** chunks when `T mod 4 ≠ 0`, the fifth has exactly `T mod 4` rows — *"never a per-arm mechanism"* |
| **Why next** | If the problem is variance, more data per update should fix it |

---

### R3 — the variance hypothesis, falsified

| Field | |
|---|---|
| **Control** | R2 (`rollout_episodes 8`, 600 episodes) |
| **Treatment** | **`--rollout-episodes 32 --episodes 2400`** — the one verbatim command in the corpus |
| **Variable** | batch size only (4×), `n_updates` held at 75 |
| **Metrics** | gradient SNR; Δ_EDGE |
| **Result** | SNR **2.03×**, **p = 0.0000** — the intervention *worked*. Δ_EDGE got **worse** |
| **Verdict** | 🛑 **NO-GO. The variance hypothesis is falsified** |
| **Ruled out** | "It's just noise" |
| **Survived** | ⚠ **R3 was never evaluated on the held-out baseline suite — NOT VERIFIED FROM REPOSITORY** (no `R3*eval*.json` exists). ⚠ *"R3's reward peaked at 45 then declined"* is **FALSE** — block means improve monotonically **−16.11 → −12.16 → −10.59 → −8.51**; `R3_best` is a genuine update-45 checkpoint (`lr 2.8933e-04`, 872 Adam steps) |
| **Why next** | If not variance, then *dilution* — the low-risk bulk drowning the high-risk minority |

---

### Rung 3 — dilution is real and does not explain the arms

| Field | |
|---|---|
| **Hypothesis** | The low-risk bulk sets the update direction. |
| **Method** | Exact decomposition `g_full = g_hi + g_lo`, reference direction `synth`, pre-registered |
| **Result** | `cos(g_full, g_lo) ∈ [+0.9099, +0.9994]` in **9/9 cells** — dilution is **real**. But `cos(g_hi, g_synth)` = **+0.913 / +0.536 / +0.451** and the **mass ratios order the arms opposite to behaviour** |
| **Verdict** | 🛑 `go_for_training_rung = False` |
| **Ruled out** | Dilution as *the* explanation of R2 > R3 > A0 |
| **Survived** | ⚠ `synth` **does not discriminate the action channel**. ⚠ D3/D4 ignore `--episodes`, so the four `*_b32.json` files are **bit-identical duplicates** and are **not** a batch-size control |
| **Why next** | `rollout_episodes` itself had to be characterised |

---

### Divergence study — `rollout_episodes` is pure variance

| Field | |
|---|---|
| **Result** | Expected update content is **invariant to batch size (≤ 0.105 σ₈)**; sd follows the finite-population law **sd(8)/sd(32) = 2.236**; the alleged *"5.1× collapse"* was a **+2.09σ outlier** of the 8-episode instrument. Critic difference 0.009 |
| **Verdict** | ✅ `rollout_episodes` is a **pure variance intervention** |
| **Survived** | The **regime-selective learning asymmetry**: R3 learned the low-risk bulk and **not** the high-risk minority, on two independent instruments. Divergence bracketed to **updates 45–75**. §10 names **per-update checkpoints** as the largest remaining gap |
| **Why next** | Exactly that gap — but first, an audit of every instrument used so far |

---

### Phase 0 — the instrument audit that reframed the project

| Field | |
|---|---|
| **Question** | Do the mechanism metrics agree with behaviour? |
| **Result** | **20 readings across 10 instruments. 20 negative. 0 positive. Mean ρ = −0.900.** Every Sprint 7 mechanism metric orders the arms **backwards**. Structural cause: endpoint = **headroom**, not learning (A0 **+0.4023** vs R2 **−0.2372**) |
| **Verdict** | ⚠ **The blocker is a missing observable, not a missing hypothesis** |
| **What changed** | Mechanism 1 CLOSED, **Mechanism 2 CLOSED**, Mechanism 3 DOWNGRADED. Mechanism 2 closed because `clip_frac` quartiles **0.0426 / 0.0196 / 0.0031 / 0.0000** with 22/75 zeros make the clip too rarely active to be a bias, and `c(s)` is **agent-idiosyncratic** (within-tick spread **6.65** vs SD(c) **4.65** on 27 groups) so no online centring exists |
| **Ruled out** | Endpoint measurement as a route to the answer |
| **Survived** | ⚠ Its own honesty paragraph: the 20/20 result **must not be read as p = 2⁻²⁰**, because `p ≥ 1/6`. Integrity: git HEAD `a8df6d9`, 96/96, 41/41, 58/58, 3/3 OK; **11 manifests clean, 3 failed — all three diagnosed, none patched** |
| **Why next** | Get the missing observable: replicate R2 and save **every update boundary** |

---

### Phase 4 — bit-exact replication with 76 checkpoints

| Field | |
|---|---|
| **Question** | Can R2 be reproduced exactly, with instrumentation, without touching production code? |
| **Design** | **45,749 B pre-registration**, 19 checks, RULE 4 justification, an **additive driver** only (`train.py:232` / `:252` untouched) |
| **Result** | ✅ **Bit-exact, storage-for-storage, on CPU.** 76 checkpoints `u000…u075` + a JSONL manifest. **18/19** checks pass |
| **Verdict** | ✅ Nondeterminism explains nothing; the 76 checkpoints are the real control run |
| **Survived** | ⚠ **B6 FAILS by design** — a 1-ULP pre-registered literal (`…3688` vs the stored `…3687`), left wrong on purpose so the FAIL stands. Protected ρ = **+0.053, n.s.** ⚠ A **"Phase 6" naming slip** appears in the text. ⚠ **Phase 4.1 has no report** |
| **Why next** | Now Δ_EDGE could be watched *as it formed* |

---

### Phase 5 — what Δ_EDGE actually is

| Field | |
|---|---|
| **Question** | Does the policy acquire risk-conditioned behaviour, or suppress low-risk behaviour? |
| **Result** | **π(EDGE \| high risk) ends BELOW random init and never exceeds it in 76/76 checkpoints.** **159–218%** of Δ_EDGE's growth is **low-risk suppression**. The actor **freezes at update 62** — with ~19% of training left — while the critic is still improving fastest |
| **Verdict** | ❌ **Δ_EDGE is differential suppression, not risk acquisition** |
| **Ruled out** | "The policy learned to migrate at high risk" |
| **Survived** | ⚠ An **expectation-vs-greedy conflict**: 0/76 and 1/76 on one reading vs **0.083 → 0.463** on another. ⚠ §4.4 is **wrong** about a pre-existing measurement — the artifact exists as `SPRINT_7_R3_action_channels_R3.json` (md5 `53955350c3d84c10c234d33464ea8d5c`); corrected in the synthesis §8 row 17 **rather than by editing the report**. ⚠ Masking facts: PREEMPT legal in **0.00%** of high-risk entries; CLOUD legal in **26.4% / 8.0%** ⇒ the effective high-risk choice is **binary STAY vs EDGE** |
| **Why next** | Nothing further — Sprint 7 stopped here and audited itself |

---

### Final synthesis audit — Sprint 7 closed

| Field | |
|---|---|
| **Form** | 59,913 B, 12 sections, a **22-hypothesis ledger**, read-only |
| **Result** | **Verdict: COMPLETE. Further training justified: NO. Sprint 7 closed: YES.** The mechanism answer: the high-risk regime is **4.3–6.5%** of decision entries whose gradient is directionally positive but **outvoted**, while at low risk EDGE is genuinely wrong (true adv **−1.337**, n=159), so correctly-learned low-risk suppression **generalises through shared parameters** into the high-risk minority |
| **Chosen interpretation** | **D — a combination**, not the flattering A: mostly differential low-risk suppression **plus a real high-risk acquisition phase at u042–u062** worth **23% / 30%** of the final Δ |
| **Still unexplained** | The arm ordering **R2 > R3 > A0** (Rung 3's D4 failed) |
| **Two things not to redo** | (1) A high-risk-weighted rung — NO-GO by a **pre-registered** rule: `w*` = **+9.832** for the arm that needs it, and only **+0.0500** at w=16. (2) A variance/batch rung — already run and falsified |
| **The one thing worth doing** | §11 item 5: run the **Rung 3 gradient decomposition along the u000…u075 trajectory** instead of at the endpoint. **Offline; no training; the checkpoints already exist** |

---

# Part 14 — Metric dictionary

Only metrics that actually appear in this repository.

## 14.1 Task-level metrics (`rollout.py` `METRIC_KEYS`, 22 keys)

| Metric | Formula / definition | Measures | Chosen because | Used by | High / low | Limitations |
|---|---|---|---|---|---|---|
| **reward** | sum of per-step environment reward over an episode | Overall objective | It is what PPO optimises | All | Higher better | ⚠ **Not comparable across arm groups** — three different reward conventions exist (Part 10.3). Only compare within a group, or use the within-arm gap to the baseline |
| **protected** | count of at-risk tasks moved off a host before it failed | The actual clinical goal | Reward-independent, so it *is* comparable across arms | Sprint 6b onward | Higher better | Sparse; risk-threshold gets 30.2 vs MAPPO's 3.2 |
| **lost / critical_lost** | tasks destroyed by a host failure | Harm | Direct | Sprint 6b onward | Lower better | — |
| **relocations** | migration count | Migration volume | Separates *targeting* from *volume* | Sprint 6b onward | Neither — it is a **cost**, not a goal | Sprint 6b: MAPPO 35.9 vs threshold 36.2 — **near-identical volume, 10× difference in protection** ⇒ *"It protects by volume, not by targeting."* |
| **success_rate** | fraction of tasks completed | Service level | — | Sprint 6b onward | Higher better | — |
| **sla / energy** | env-defined | Secondary costs | — | All | Lower better | Not load-bearing in any conclusion |
| **infeasible** | ⚠ a **CONTENTION** counter (`env.py:405-422`, increments at 565/630, rotating apply order 620-626, penalty at 934) | Agents competing for the same host | — | Sprint 6.5 onward | **Not an error rate** | ⚠ **Not an illegal-action counter** — the actor is masked (`masked_dist`, `mappo.py:77-79`) so illegal actions can never be sampled. **High `infeasible` means the policy actually migrates.** 545/783 = **69.6% all cloud** |

## 14.2 The primary Sprint 7 behaviour metric

| Metric | Formula | Measures | Chosen because | Used by | Interpretation | Limitations |
|---|---|---|---|---|---|---|
| **Δ_EDGE** | `π(EDGE \| risk ≥ 0.6) − π(EDGE \| risk < 0.2)`, over **all decision entries** of a frozen state set | Risk-conditioned behaviour | Reward-independent, so it survives the three reward conventions | Rung 2.75 → Phase 5 | Positive = migrates more at high risk | ⚠ **Phase 5 showed a positive Δ_EDGE can be pure low-risk suppression** — 159–218% of R2's growth is, and π(EDGE\|hi) ends **below init**. Δ_EDGE is a *difference*, and a difference can grow by lowering the subtrahend |
| **Spearman(risk, π(EDGE))** | rank correlation over all decision entries | Monotone risk tracking | Distribution-free | Rung 2.75 onward | Positive = tracks risk | Tracking ≠ magnitude — Sprint 6b's probe correlated **+0.990** while spanning only **0.004** |
| **frozen state sets** | RANDOM **4,975** entries / **216** high-risk; UNION **74,237** / **~3,592** | The evaluation substrate | Frozen ⇒ every arm is scored on identical states | Rung 2.75 onward | — | Built by `_diag_rung2_75_matched_states.py`; ⚠ per-arm keys live under `by_state_source`, and some fields are **high-risk-subset only** while others are **all entries** |

## 14.3 PPO training-health metrics (`*_updates.csv`)

| Metric | Definition | Measures | Interpretation | Limitations |
|---|---|---|---|---|
| **entropy** | policy entropy | Exploration | Falling = sharpening | — |
| **approx_kl** | ⚠ the **signed k1** estimator `mean(old_logp − new_logp)` | Step size | Near 0 = small steps | ⚠ **Legitimately goes negative** — 26/75 updates for R2. **It is not a KL divergence.** k3 exists offline at D4 snapshots only; there is no per-update k3 trajectory |
| **clip_frac** | fraction of samples hitting the PPO clip | How often the trust region binds | Higher = larger proposed steps | ⚠ **`clip_frac = 0` is normal here.** Quartiles **0.0426 / 0.0196 / 0.0031 / 0.0000**, with **22/75 exact zeros**. This is what closed Mechanism 2 |
| **explained_var** | `1 − Var(returns − V)/Var(returns)` | Critic quality | 1 = perfect | The `g999_lowEV` run was killed at ep 75 on ev **+0.37**; `lam95` reached **0.92** |
| **actor_loss / critic_loss** | PPO surrogate / value loss | — | — | ⚠ D4's `replica_fidelity ok=False` is **critic-side only and only for mc-target arms** (critic_loss 7.459/3.905 for R2, explained_var ≤0.153). A0 is `ok=True, max 0.0000`. **All actor-side quantities in D4 are exact** |
| **adv_std** | advantage std within the update | Signal scale | — | ⚠ **There is no `adv_mean` column** — it exists only in the Phase 4 trajectory manifest's `stats` dict |
| **lr_scale** | `1 − (update_id−1)/n_updates` | Anneal position | 1.0000 → 0.0133 | The inverse of this is how you date a checkpoint (Part 10.2) |
| **decision_frac** | fraction of steps that were decision points | Data density | — | — |

## 14.4 Mechanism / gradient metrics (Sprint 7 only)

| Metric | Definition | Measures | Used by | Interpretation | Limitations |
|---|---|---|---|---|---|
| **true advantage** | replay-derived, `+2.671 ± 0.806` high-risk vs `−1.337 ± 0.967` low-risk | Ground truth for EDGE | Sprint 6.5, Rung 0 | Gap **+4.01** (and **+3.363** at Rung 0) | Established with **0 replay mismatches** |
| **sign agreement** | fraction of states where estimator and truth agree on sign | Estimator correctness | Rung 0/1/2.5/2.75 | 0.5 = chance | High-risk GAE was **29.0%** — *below chance* |
| **paired advantage** | `gae(s,a) = c(s) + paired(s,a)`, `c(s) := gae(s, a_ref)` | Removes the per-state offset | Rung 2.75 | Lifts agreement to **0.71 / 0.95** for **every** arm | `a_true(s, a_ref) ≡ 0` by construction |
| **Var(raw)/Var(paired)** | variance ratio | Offset magnitude | Rung 2.75 | **3.72×** ⇒ **1.93×** attenuation through pooled normalisation | — |
| **gradient cosine** | `cos(g_full, g_lo)`, `cos(g_hi, g_synth)` | Who sets the update direction | Rung 3 | `cos(g_full, g_lo) ∈ [+0.9099, +0.9994]` in 9/9 cells | ⚠ `synth` **does not discriminate the action channel** |
| **gradient SNR** | mean/sd of the update direction | Estimator noise | R3 | **2.03×**, **p = 0.0000** | The intervention worked and behaviour still got worse |
| **finite-population correction** | `sd(8)/sd(32) = 2.236` | Expected sd ratio under pure variance | Divergence | Observed matched it | This is what turned the *"5.1× collapse"* into a **+2.09σ outlier** |
| **cluster bootstrap** | resampling with **clusters = episodes** | CI under within-episode correlation | Rung 3, Phase 5 | — | Naive i.i.d. bootstrap would be too tight |
| **z-score / p-value** | standard | Significance | R3, Phase 4, Phase 5 | — | ⚠ Phase 0's 20/20 result **must not be read as p = 2⁻²⁰** (`p ≥ 1/6`) |
| **saturation** (`frac_maxp_gt_099`) | fraction of high-risk states with max action prob > 0.99 | Softmax collapse | Rung 2.75 | **R2 50.5%** vs **A0 0.0000** | ⚠ **Over the high-risk subset only.** Downgraded from cause to *symptom* by R3 |
| **‖Δθ_actor‖** | parameter distance from init | How far the actor moved | Phase 4/5 | R2 final **6.1710** | Freezes at **u62** |

## 14.5 Predictor metrics

| Metric | Why chosen | Value (observable arm) | Interpretation |
|---|---|---|---|
| **PR-AUC** | The **primary** metric — positives are 2.146% | **0.28634** = **13.3× lift** | Chosen over ROC precisely because of the imbalance |
| ROC-AUC | Secondary | 0.81594 | Optimistic under imbalance — note OLDDATA's 0.9165 ROC alongside a much worse 0.2174 PR-AUC |
| precision / recall / F1 | Deployment behaviour at `threshold = 0.18` | 0.36972 / 0.65625 / 0.47297 | — |
| accuracy | Reported for completeness | 0.9686 | **Meaningless here** — always-negative scores 0.979 |
| **lead time** | Is a warning *early enough to act on*? | 23 EARLY_WARNING / 7 MISSED / 2 STANDING_ALARM; lead mean 8.39, median 10.0; run_lead 18.22 | The metric that makes the risk channel actionable |
| LONO | Generalisation to an unseen node | node 0: PR-AUC 0.4042 / ROC 0.9260 / F1 0.5634 | — |

---

# Part 15 — Sprint 7, reconstructed as one chapter

> Part 2 has the stage-by-stage detail and Part 13 the verdict cards. This Part is the *narrative* —
> what Sprint 7 inherited, what it went after, and where it landed. It deliberately does not repeat
> Sprints 1–6.
>
> **Entry point if you only read one file:** `SPRINT_7_FINAL_SYNTHESIS_AUDIT.md`. It supersedes the
> 11-report chain and contains the timeline, the 22-hypothesis ledger and every superseded claim.

## 15.1 What Sprint 7 started with — the inheritance from Sprints 1–6

Sprint 7 did not start from a question about MARL. It started from **three facts Sprint 6b and 6.5
had already nailed down**, and its entire structure is a consequence of them:

1. **The learned scheduler loses to a one-line rule.** MAPPO reward 20.62, protected 3.2; a
   risk-threshold rule 76.50, protected 30.2 — at **essentially the same migration volume** (35.9
   vs 36.2). Sprint 6b's own summary: *"It protects by volume, not by targeting."*
2. **The signal is present and the policy discards it.** A behaviour-cloning head on the *same
   observation* reaches **accuracy 1.0000** with a probe span of **0.3935**, against the trained
   policy's **0.004** — 25× short. So the failure is not the features, not the environment, and not
   the observation.
3. **The advantage estimator is inverted where it matters.** True advantage of EDGE at high risk is
   **+2.671 ± 0.806**; the learned estimator reads **−0.266**. And GAE's high-risk sign agreement is
   **29.0%** — *below chance*.

Sprint 6.5 had also cleared the ground: six of eleven hypotheses **refuted**, two real defects fixed
(the contention bug, the dead `w_criticality_migration`), and — importantly — *"Neither change adds
any `if risk > threshold` logic."* Sprint 7 was therefore not allowed to solve the problem by
hard-coding the rule it was supposed to beat.

**One inherited number governed everything Sprint 7 did:** A0 reproduces Sprint 6 **bitwise**
(action histogram 9832 / 77 / 842 / 0). A trustworthy control existed. Every later comparison rests
on it.

## 15.2 What Sprint 7 suspected

Phase 1 stated the hypothesis in one paragraph, verbatim:

> *"PPO is converging correctly onto an advantage estimate whose action ordering at high risk is
> inverted relative to the truth. The dead gradient at update 60+ is what convergence looks like.
> Not a stalled optimiser — a converged one, pointed the wrong way."*

The supporting census is memorable: greedy A0 plays **CLOUD in 842 of 855 legal states (98.5%)** and
**EDGE in 77 of 10,751 (0.7%)**. It had learned *"cloud whenever legal, else stay — a rule with no
risk term in it, built on a −0.050 pseudo-no-op."*

Phase 1 was honest that this was a **statement, not a finding** (n = 4 / n = 9), and said so:
*"Establishing or refuting it is rung 0 of the ladder."*

## 15.3 What Sprint 7 tested — the ladder as it actually ran

| Rung | The suspicion | The instrument | Outcome |
|---|---|---|---|
| **0** | Is there a signal at all, and does GAE see it? | truth-vs-GAE sign agreement | Signal **+3.363** real; GAE **29.0%** at high risk — below chance |
| **1** | Is the **critic target** the cause? | offline λ vs MC | **SUPPORTED for A0**: 26.3% → 60.7% |
| **2** | Does it survive in production? | `critic_target: lambda → mc` — **the only production edit in Sprint 7** | Best Δ_EDGE (**+0.2024 / +0.1579**) but reward −0.46, protected 9.9 |
| **2.5** | Is another target rung worth it? | 130/130 replication | **NO-GO** — 87% of the sign error already fixed. Split the problem into **three mechanisms** |
| **2.75** | Is the residual a **per-state offset**? | paired estimator | **YES** — `Var(raw)/Var(paired) = 3.72×`; pairing lifts agreement to **0.71/0.95 for every arm**. **Δ_EDGE defined here** |
| **R3** | Is it **variance**? | `rollout_episodes 8 → 32` | **FALSIFIED** — SNR up 2.03× at p = 0.0000, behaviour **worse** |
| **3** | Is it **dilution** — low risk drowning high risk? | exact `g_full = g_hi + g_lo` | Dilution **real** (9/9 cells, cos ≥ +0.9099) but the mass ratios order the arms **backwards** ⇒ `go_for_training_rung = False` |
| **Divergence** | What *is* `rollout_episodes`? | finite-population law | A **pure variance** intervention (≤0.105 σ₈; sd ratio 2.236). The *"5.1× collapse"* was a **+2.09σ outlier** |
| **0 (audit)** | Do the instruments agree with behaviour? | 10 instruments × 2 readings | **20/20 negative, mean ρ = −0.900** |
| **4 / 4.1** | Can R2 be replicated with instrumentation? | additive driver, 19 pre-registered checks | **Bit-exact.** 76 checkpoints. 18/19 |
| **5** | What is Δ_EDGE *made of*? | the 76-checkpoint trajectory | **Differential suppression**, not acquisition |

Notice the shape: **rung 1 said yes, rung 2 shipped it, and every rung after rung 2 returned a
NO-GO.** Six consecutive negative results is what the middle of Sprint 7 was.

## 15.4 What failed

- **The variance hypothesis.** R3 is the cleanest falsification in the project: the intervention
  measurably did what it was supposed to do (SNR ×2.03, p = 0.0000) and the behaviour got worse. You
  cannot ask for a better-controlled refutation.
- **The dilution hypothesis as an *explanation of the arms*.** Dilution is real in 9/9 cells and
  still fails to order R2 > R3 > A0.
- **Every endpoint mechanism metric.** All 20 readings invert. Phase 0's structural diagnosis: an
  endpoint measurement reports **headroom**, not learning (A0 **+0.4023** vs R2 **−0.2372** — the
  worse arm has more room left to move, so it scores better on any "how much signal is available"
  instrument).
- **The Phase 1 ladder itself.** ⚠ The B1–B3 / C1–C4 / E1–E3 sequence it proposed **has no
  artifacts**. What ran was different, and the handbook records both.
- **A pre-registered check.** B6 fails, by design, on a **1-ULP** literal — and was **left wrong on
  purpose** so the FAIL stands in the record.

## 15.5 What was reproduced

- **A0 reproduces Sprint 6 bitwise** — the control is real.
- **R2 was replicated bit-exactly, storage-for-storage, on CPU** by Phase 4, which also proved that
  **nondeterminism explains nothing** and that the 76 new checkpoints belong to the genuine control
  run.
- **A script edit that could have changed a number was proved harmless empirically** — all 10 shared
  arm×source values and all four state-set sizes bit-identical, **MAX |diff| = 0.000e+00**.

## 15.6 What was discovered

Two findings, one methodological and one substantive.

**Methodological (Phase 0).** *The blocker was a missing **observable**, not a missing hypothesis.*
Twenty readings inverting is not twenty coincidences; it is one structural fact about measuring at
the endpoint. This is what redirected the sprint from "propose another mechanism" to "go get the
trajectory" — and it is why the 76 checkpoints exist.

**Substantive (Phase 5).** *Δ_EDGE is differential suppression, not risk acquisition.*

- **π(EDGE | high risk) ends BELOW random initialisation, and never exceeds it in 76/76
  checkpoints.**
- **159–218% of Δ_EDGE's growth is low-risk suppression.** More than all of it — the high-risk
  channel contributes negatively over the full run.
- The actor **freezes at update 62**, with ~19% of training left, while the critic is still improving
  fastest. That rules out *"the critic gated the actor"* and dates the softmax collapse to u1–u40.
- The high-risk decision is effectively **binary STAY vs EDGE**: PREEMPT is legal in **0.00%** of
  high-risk entries and CLOUD in only **26.4% / 8.0%**. Those are **masking facts, not policy
  facts** — the policy can be neither blamed nor credited for never preempting.

## 15.7 What was finally concluded

The synthesis audit's mechanism answer, stated at the level the evidence supports:

> The high-risk regime is **4.3–6.5%** of decision entries. Its gradient is **directionally
> positive but outvoted** — `cos(g_full, g_lo) ∈ [+0.9099, +0.9994]` in 9/9 cells. Meanwhile at low
> risk, EDGE is **genuinely wrong** (true advantage **−1.337**, n = 159). So a *correctly learned*
> low-risk suppression **generalises through shared parameters** into the high-risk minority, where
> it is wrong.

That is a mechanism-level answer for the **shared** failure — the thing all arms do. It is **not** an
explanation of **why R2 beats R3 beats A0**; that ordering remains **unexplained** (Rung 3's D4
failed to produce it).

The audit chose **interpretation D — a combination** — over the flattering interpretation A:

- mostly differential low-risk suppression, **plus**
- a **real high-risk acquisition phase at u042–u062** (`p_hi` gains **+0.0658 / +0.0660** against
  `p_lo`'s **+0.0195 / +0.0192**), worth **23% / 30%** of the final Δ, **and**
- R2 beats the A0 control on **both** channels while still ending **below its own init** on the mass
  channel.

**Status: COMPLETE. Further training justified: NO. Sprint 7 closed: YES.**

**Two things explicitly not to redo.** (1) A high-risk-weighted rung is **NO-GO by a pre-registered
rule** — the weight the arm that needs it would require is `w*` = **+9.832**, and at w = 16 the
effect is only **+0.0500**. (2) A variance/batch rung was already run and falsified.

**If it is ever reopened**, §11 item 5 names the single highest-value next step, and it needs **no
training at all**: run the **Rung 3 gradient decomposition along the u000…u075 trajectory** instead
of at the endpoint. It would close the only causal gap left — and the checkpoints already exist.

---

# Part 16 — Self-healing: the actual final answer

The project's ultimate question was: *can a multi-agent RL policy, conditioned on digital-twin
failure predictions, relocate healthcare workloads pre-emptively — i.e. self-heal?*

The answer must be stated in four tiers, because the evidence is genuinely of four different
strengths. **Nothing below upgrades a "we don't know" into a causal claim.**

## 16.1 DEMONSTRATED — established by direct measurement, reproduced

1. **A digital twin can predict host failure early enough to act.** PR-AUC **0.28634** at a 2.146%
   positive rate is a **13.3× lift** over chance; **23 of 32 failures produce an EARLY_WARNING**,
   lead mean **8.39** / median **10.0** ticks, run-level lead **18.22**. The signal is real and the
   engineered features are load-bearing (legacy 7 features: PR-AUC 0.0777).
2. **Pre-emptive relocation works — as a mechanism.** A one-line risk threshold on that channel
   protects **30.2** tasks per episode against **3.2** for the learned policy, and against **−37.06**
   reward for never migrating at all. **Self-healing is achievable in this system.**
3. **A learned MAPPO policy trains stably in this environment and beats naive controls.** 20.62 vs
   random −8.11 vs static −37.06.
4. **The learned policy does NOT achieve targeted self-healing.** It loses to the one-line rule at
   **the same migration volume**. *"It protects by volume, not by targeting."* This is a
   demonstrated negative result, reproduced **bitwise** by the A0 control.
5. **The failure is not in the observation.** A behaviour-cloning head on the identical observation
   reaches **accuracy 1.0000** with a **0.3935** probe span. The information the policy needs is in
   front of it.
6. **`rollout_episodes` is a pure variance intervention** (≤ 0.105 σ₈; sd ratio 2.236 exactly as the
   finite-population law predicts).
7. **R2 is bit-exactly reproducible**, and the 76-checkpoint trajectory is a faithful record of it.
8. **Δ_EDGE, the sprint's headline improvement, is not what it appeared to be.**
   π(EDGE | high risk) **ends below random init and never exceeds it in 76/76 checkpoints**;
   **159–218%** of the metric's growth is low-risk suppression.

## 16.2 SUPPORTED — good evidence, one instrument or one arm short of settled

1. **The advantage estimator, not the reward or the features, is the proximate cause.** True
   advantage **+2.671** vs estimated **−0.266** at high risk; GAE sign agreement **29.0%**. Strong,
   consistent, and confirmed on multiple instruments.
2. **The sign error is a per-state offset, not a target defect.** `Var(raw)/Var(paired) = 3.72×`, and
   pairing lifts high-risk agreement from below chance to **0.71 / 0.95 for every arm**. Held up
   across all arms — the reason this is "supported" rather than "demonstrated" is that no online
   centring exists to *act* on it (`c(s)` is agent-idiosyncratic: within-tick spread **6.65** vs
   SD(c) **4.65**).
3. **The Monte-Carlo critic target helps the estimator.** Offline 26.3% → 60.7%; in production R2 has
   the best Δ_EDGE of any arm. Supported — **but** it comes with *worse* task performance
   (reward −0.46, protected 9.9), which is unexplained.
4. **Dilution is real.** `cos(g_full, g_lo) ∈ [+0.9099, +0.9994]` in **9/9** cells: the low-risk bulk
   sets the update direction.
5. **The outvoted-gradient account of the shared failure.** High risk is 4.3–6.5% of entries, its
   gradient is directionally positive but outvoted, and low-risk EDGE suppression is *correct*
   (true adv −1.337) — so a correctly-learned behaviour generalises into the minority regime where it
   is wrong. This is the best available mechanism-level explanation, and it is consistent with every
   surviving measurement.
6. **Endpoint mechanism metrics measure headroom, not learning.** 20/20 readings invert, mean
   ρ = −0.900, with a structural explanation (A0 +0.4023 vs R2 −0.2372). ⚠ Supported, **not**
   p = 2⁻²⁰ — the report itself says `p ≥ 1/6`.
7. **There was a genuine high-risk acquisition phase, at u042–u062**, worth 23–30% of the final Δ.
   Real, but small, and it does not survive to the end of training as a net gain.

## 16.3 PARTIALLY ANSWERED — a piece is settled and a piece is not

1. **"Why doesn't the policy use risk?"** *Answered for the shared failure* (outvoted gradient +
   parameter sharing). **Not answered for the differences between arms.**
2. **"Is the sign error fixable?"** 87% of it was fixed by the MC target — and the fix did not
   translate into task performance. So: fixable as a statistic, not yet as behaviour.
3. **"When does the policy stop learning?"** The actor freezes at **u62** and the softmax collapse
   dates to **u1–u40**. What *causes* the collapse in those first 40 updates is not established.
4. **"Does the policy respond to risk at all?"** It **tracks** risk (Spearman positive; Sprint 6b
   probe correlation **+0.990**) but with almost no **magnitude** (span **0.004**). Direction yes,
   amplitude no.
5. **Predictor calibration.** The risk scores are explicitly **uncalibrated** and near-bimodal. The
   channel is usable and it is not a probability. What a *calibrated* channel would change is
   untested.

## 16.4 UNKNOWN — honestly open

1. **Why R2 > R3 > A0.** The arm ordering is **unexplained**. Rung 3's D4 test failed to reproduce
   it, and every endpoint instrument orders it backwards.
2. **What causes the softmax collapse in u1–u40.** Saturation was **downgraded from cause to
   symptom** by R3. No surviving candidate.
3. **Whether the policy could reach the threshold rule's protection level under any configuration
   reachable from this setup.** Never established, in either direction.
4. **R3's task-level performance.** ⚠ **NOT VERIFIED FROM REPOSITORY** — no `R3*eval*.json` exists.
   R3 was never scored against the held-out baseline suite.
5. **Whether a trajectory-level gradient decomposition would identify the arm ordering.** This is the
   one open question with a named, training-free route to an answer (§11 item 5) — and it was not
   run.
6. **Whether the low-risk suppression is *causally* responsible for the high-risk deficit**, as
   opposed to co-occurring with it. The parameter-sharing account is a mechanism *story* consistent
   with the data; the trajectory-level test that would make it causal is exactly the analysis that
   was never run.
7. **Everything beyond this trace.** One seed for the simulator, one seed for training, one
   topology, 10 nodes, 1500 ticks. No claim in this project has been shown to generalise past that.

## 16.5 The one-paragraph answer

**Self-healing is demonstrably achievable in this system, and MAPPO demonstrably does not achieve it
here.** The digital twin predicts failures with real lead time; a one-line threshold on that
prediction protects ten times as many tasks as the learned policy at identical migration volume. The
learned policy's failure is localised to its **advantage estimator**, which is inverted precisely in
the high-risk minority — and the best available explanation is that suppressing edge migration is
*correct* in the 95% low-risk bulk, learned there properly, and then carried into the 5% high-risk
regime through shared parameters, where it is wrong. Sprint 7 fixed 87% of the statistical sign
error without recovering the behaviour, and its headline behavioural improvement turned out on
inspection to be **suppression of the low-risk baseline rather than acquisition of high-risk
response**. **What remains unknown is why the arms differ from one another**, and that gap is a
missing measurement, not a missing idea.

---

# Part 17 — What we learned across the whole project

Nine lessons, each one earned by a specific event in this repository.

**1. A control you can reproduce bitwise is worth more than any result.**
A0 reproduces Sprint 6's action histogram **9832 / 77 / 842 / 0** exactly, and Phase 4 reproduced R2
**storage-for-storage**. Every Sprint 7 comparison rests on those two facts. Without them, R3's
falsification and Phase 5's trajectory reading would both have been arguable.

**2. Fix the data before you tune the model.**
The predictor looked broken for three sub-sprints. It was the trace: **5 of 12** observable columns
were constant and failures were *"instant death at a scheduled time, no symptoms."* Regenerating the
trace took the constant columns to **0 of 12** and the problem evaporated. No model change was
needed.

**3. Always run the trivial baseline, and run it first.**
Sprint 6b's entire value is one comparison: a **one-line threshold** beats the learned policy 30.2 to
3.2 on protection. `marl/baseline.py` says why this is non-negotiable — *"it answers 'does MAPPO do
anything a one-line threshold on the risk channel would not?', which is the question the
host-predictor audit taught us to ask before believing any learned result."* On the predictor side,
`degraded==1` alone gets **0.7333 recall**; without that number the 0.28634 PR-AUC would have looked
better than it is.

**4. Endpoint measurements can be systematically misleading — and a trajectory can settle it.**
Twenty readings of ten Sprint 7 instruments all ordered the arms **backwards** (mean ρ = **−0.900**),
because an endpoint reading reports **headroom**, not learning. The fix was not another hypothesis;
it was 76 per-update checkpoints. And the trajectory immediately overturned the sprint's own headline:
Δ_EDGE is **differential suppression**, and π(EDGE|hi) ends **below init in 76/76**.

**5. Pre-registration is what makes a negative result mean something.**
Rung 3 pre-registered its dilution test and then returned `go_for_training_rung = False`. R3
pre-committed to a metric and then reported that the intervention **worked** (SNR ×2.03,
p = 0.0000) while behaviour got **worse**. A high-risk-weighted rung was closed by a pre-registered
threshold (`w*` = +9.832). None of those conclusions would be credible if the criteria had been
chosen afterwards.

**6. Do not let diagnostic code touch production code.**
The whole of Sprint 7 contains **one** production edit — `critic_target: lambda → mc`. Production
source last changed **2026-08-25 04:02** and is byte-identical in all eight subsequent manifests and
at HEAD. Phase 4's instrumentation was an **additive driver**; `train.py:232` and `:252` were left
alone. This is why "did the code change under me?" was never a live question.

**7. Integrity manifests are what let you trust an untracked tree — and honest failures are the
point.**
Phase 0 re-verified 14 prior manifests: **11 clean, 3 failed**, and it **patched none of them**. One
failure *dated* a known change. One recorded a process slip. One could have altered a number, so the
number was **re-derived** — MAX |diff| = **0.000e+00**. Similarly, `_phase4_verify` B6 was **left
wrong on purpose** so the FAIL stands. A manifest that always passes is telling you nothing.

**8. A NO-GO is a result. Six of them in a row is progress.**
Rungs 2.5, R3, 3, and the Divergence study all returned NO-GO, and together they eliminated the
critic target, batch size, variance, and dilution-as-arm-explanation. That is what left the
outvoted-gradient account standing. The final audit's own verdict — *"further training justified:
NO"* — is itself a result.

**9. A mechanism moving in the expected direction does not mean it explains the behaviour.**
This is the sharpest lesson in the project, and it happened three separate times:
- **R3**: gradient SNR improved significantly (p = 0.0000) and behaviour got worse.
- **Rung 3**: dilution is real in **9/9** cells, and the mass ratios order the arms **opposite** to
  behaviour.
- **Rung 2**: the MC target produced the best Δ_EDGE of any arm and the *worse* reward and protection.

The pattern to internalise: *"the mechanism is real"* and *"the mechanism explains the outcome"* are
two claims, and this repository confirmed the first while refuting the second, repeatedly.

**Bonus, from Sprint 5.** Check the **real support** of a variable before designing a probe over it.
Severity only spans `[0.320, 0.644]`, so probe grids over `[0,1]` were **~3× too wide** and their
reported spans understate true sensitivity. Every sweep-span number in the project inherits that
caveat.

---

# Part 18 — Troubleshooting

> **Standing rule for this whole Part: NEVER rerun training to "see if it goes away."** Every
> symptom below has a read-only diagnosis path. Retraining destroys the artifact you were trying to
> understand, costs 20–25 minutes, and — because the arms are seeded — will reproduce the same
> numbers anyway. If a number looks wrong, it is far more often an **artifact-reading trap** than a
> training problem.

---

### 18.1 `git status` shows dozens of `??` lines and I don't know what's safe

- **Symptom.** Untracked files everywhere, including whole directories.
- **Likely cause.** Normal for this repository. Sprints 6, 6.5 and 7 all happened between commits
  `208e5a2` (07-29) and `a8df6d9` (08-30) — a month with nothing committed.
- **How to check.** `git status --short` and read every `??` path. Compare against Part 9 to see what
  each one is.
- **What NOT to do.** Do **not** `git clean`. Do **not** add a `.gitignore` entry to make the noise go
  away — `R2_trajectory/` looks like generated junk and is the single most valuable artifact in the
  tree.
- **Safe next step.** `git add -A`, then `git diff --cached --stat`, then commit. Part 12.5.

---

### 18.2 `ModuleNotFoundError: No module named 'marl'`

- **Symptom.** Import fails on any `python -m marl.*` command.
- **Likely cause.** Wrong working directory. Everything is run as a **module from `python-ai/`**.
- **How to check.** `pwd` — it must end in `python-ai`. `ls marl/__init__.py` must succeed.
- **What NOT to do.** Do not add `sys.path` hacks to production files, and do not run scripts by
  path (`python marl/train.py`) — the relative imports will break differently.
- **Safe next step.** `cd` to `python-ai/` and re-run with `python -m marl.<name>`.

---

### 18.3 An artifact I expect is "missing"

- **Symptom.** `ls saved_models/marl/SPRINT_7_R3_action_channels.json` → No such file.
- **Likely cause.** ⚠ **The `--tag` suffix trap.** Every `--tag` probe writes
  `<name>_<tag>.json`. The real file is **`SPRINT_7_R3_action_channels_R3.json`** (36,293 B, md5
  `53955350c3d84c10c234d33464ea8d5c`).
- **How to check.** Glob, never an exact name:
  ```bash
  ls saved_models/marl/ | grep -i action_channels
  ```
- **What NOT to do.** Do **not** rerun the probe to "regenerate" it. This exact mistake produced a
  wrong claim in `SPRINT_7_PHASE5_REPORT.md` §4.4 — which was then corrected **in the synthesis
  audit §8 row 17 rather than by editing the report**.
- **Safe next step.** Check the producing report's own manifest section — `SPRINT_7_R3_REPORT.md`
  line 127 lists this file by name.

---

### 18.4 A checkpoint won't load

- **Symptom.** `torch.load` raises, or the state dict keys don't match a model.
- **Likely cause.** (a) `map_location` missing on a CUDA-saved file; (b) you are loading a
  **trajectory** checkpoint (`u000` has no optimizer state) into code that expects a full one.
- **How to check.**
  ```bash
  python -c "import torch; print(list(torch.load('saved_models/marl/R2_trajectory/R2_trajectory_u000.pth', map_location='cpu').keys()))"
  ```
  `u000` is **1,736,317 B** with `buffer_T: null, stats: null`; `u001+` are **5,207,826 B**.
- **What NOT to do.** Do not re-save the checkpoint in a "fixed" format — that breaks its MD5 and
  every manifest that references it.
- **Safe next step.** Always pass `map_location='cpu'`, and handle `u000` as the special case it is.

---

### 18.5 A number in an artifact doesn't match the report

- **Symptom.** You read a JSON and get a different value than the report quotes.
- **Likely cause, in order of frequency:**
  1. **Wrong denominator.** In the frozen Rung-2.75 artifact, `entropy_mean`, `maxp_mean`,
     `frac_maxp_gt_099`, `frac_argmax_edge` and `argmax_counts` are over the **high-risk subset
     only**; `p_edge_risk_lt_02`, `p_edge_risk_ge_06`, `risk_response_*` and
     `spearman_risk_vs_p_edge` are over **all decision entries**. Mixing them has produced a wrong
     reading before.
  2. **Wrong top-level key.** That artifact's root is **`by_state_source`**, not `results`; per-arm
     numbers live at `["by_state_source"][SRC]["evaluated"][ARM]`.
  3. **Wrong config path.** PPO hyperparameters are at **`config.mappo`**, not `config.train`.
     Reading `d['train']['entropy_coef']` returns `None`.
  4. **A column that doesn't exist.** There is **no `adv_mean`** in `*_updates.csv` — only in the
     Phase 4 trajectory manifest's `stats` dict.
  5. **Cross-arm reward comparison.** See 18.9.
- **What NOT to do.** Do not "correct" the report.
- **Safe next step.** Open the producing script — it is the only executable definition of the metric —
  and read how the number is actually computed.

---

### 18.6 `md5sum -c` fails on every single line

- **Symptom.** Hundreds of "No such file or directory" lines.
- **Likely cause.** ⚠ **Wrong working directory.** Manifest lines are binary-mode with relative
  paths (`c88a7c35… *./__init__.py`), so a manifest **only verifies from the cwd it was captured
  in**.
- **How to check.** Look at a line: if the path starts `*./`, it is relative.
- **What NOT to do.** Do not regenerate the manifest — that destroys the historical record and
  replaces it with a hash of today's files.
- **Safe next step.** `cd` to the capture directory and re-run.

---

### 18.7 `md5sum -c` reports exactly one failure, and it's the manifest itself

- **Symptom.** `1 computed checksum did NOT match`, naming a `.md5` file.
- **Likely cause.** ⚠ **The circular self-hashing trap.** A manifest that lists itself can never
  verify — its own hash is computed before it is written.
- **How to check.** Read the failing path. If it ends `.md5`, this is it.
- **What NOT to do.** Do not conclude a modification occurred. This was diagnosed once and **misread
  once**.
- **Safe next step.** Ignore that one line; treat the rest of the manifest as authoritative.

---

### 18.8 A `code_*.md5` manifest genuinely fails on a production file

- **Symptom.** `config.py`, `mappo.py` or `train.py` fails a `before` manifest.
- **Likely cause.** This is the **known, documented** Rung 0 case. The hashes moved **exactly once**,
  between 08-19 21:35 and 08-25 04:02 — the **Rung 2 MC-target change** — and are byte-identical in
  every later manifest and at HEAD.
- **How to check.** Verify the *same* file against a **later** manifest (e.g. `_PHASE5_integrity/`).
  If it passes there, the change is the historical one.
  ```bash
  git diff --stat HEAD -- python-ai/marl/
  ```
  Empty ⇒ the working tree matches the commit; nothing changed under you.
- **What NOT to do.** Do not patch the manifest. Phase 0 found 3 failures out of 14 and **patched
  none**.
- **Safe next step.** If it fails against the **latest** manifest too, then something really did
  change — `git diff` will show you exactly what, and `git stash` or `git checkout --` on **source
  files only** (never on `saved_models/`) restores them.

---

### 18.9 Two arms' rewards don't seem comparable

- **Symptom.** A0 scores 20.62 and R2 scores −0.46; R2 looks catastrophic.
- **Likely cause.** ⚠ **They are on different reward scales.** Three conventions exist. The
  risk-threshold baseline reads **76.50** in A0, **67.31** in the bug-fixed arms, **76.54** in A2 —
  while its *behaviour* is byte-identical everywhere (reloc 36.2, prot 30.2, lost 3.4). Only the
  reward moves.
- **How to check.** `static-no-migration = −37.06` in **every** arm — that is the internal control
  confirming the environment is the same.
- **What NOT to do.** Do not average or rank raw rewards across groups. Do not "renormalise" a
  historical number.
- **Safe next step.** Compare on the **reward-independent** columns (protected, relocations, lost,
  success rate) or on the **within-arm gap** to that arm's own baseline. Part 10.3.

---

### 18.10 `clip_frac = 0.0000` for many updates — is PPO broken?

- **Symptom.** Whole blocks of `*_updates.csv` show zero clipping.
- **Likely cause.** **Normal here.** Quartile means are **0.0426 / 0.0196 / 0.0031 / 0.0000** with
  **22 of 75** exact zeros. The policy saturates and stops proposing steps large enough to clip.
- **How to check.** Cross-read `entropy` and `frac_maxp_gt_099` (R2 reaches **50.5%** at high risk vs
  A0's **0.0000**).
- **What NOT to do.** Do not raise `clip_eps` to "make it clip". This is the observation that
  **closed Mechanism 2**; erasing it erases the finding.
- **Safe next step.** Read `SPRINT_7_PHASE0_RECONSTRUCTION.md` §4.

---

### 18.11 `approx_kl` is negative

- **Symptom.** Negative KL in the CSV.
- **Likely cause.** It is the **signed k1** estimator `mean(old_logp − new_logp)`, not a divergence.
  **26 of 75** R2 updates are negative.
- **What NOT to do.** Do not add an `abs()` or a clamp to production logging.
- **Safe next step.** If you need a non-negative estimate, k3 exists **offline at D4 snapshots only** —
  there is no per-update k3 trajectory.

---

### 18.12 `infeasible` is huge — are agents choosing illegal actions?

- **Symptom.** `infeasible` in the high tens per episode; 545/783 = **69.6%** all cloud.
- **Likely cause.** ⚠ **It is a CONTENTION counter, not an illegal-action counter**
  (`env.py:405-422`, increments at 565/630, rotating apply order at 620-626, penalty at 934). The
  actor is **masked** (`masked_dist`, `mappo.py:77-79`), so an illegal action can never be sampled.
- **How to check.** A1 makes **1,060 cloud attempts vs 10 edge** against `cloud_slots = 8`.
- **What NOT to do.** Do not "fix" masking.
- **Safe next step.** Read high `infeasible` as *the policy actually migrates* — and as evidence it
  prefers CLOUD.

---

### 18.13 `_phase4_verify` reports 18/19 — did something break?

- **Symptom.** One check FAILS.
- **Likely cause.** **B6 fails by design.** `_phase4_verify.py:78` holds
  `1.3333333333333309e-05` (bits `…3688`); every artifact stores `1.3333333333333308e-05` (bits
  `…3687`), the exact float64 value of `1e-3 * (1 - 74/75)`. It was **left wrong on purpose** because
  RULE 3/9 forbid editing a pre-registered reference after seeing the measurement.
- **How to check.** Confirm the failing check is B6 and the delta is 1 ULP.
- **What NOT to do.** Do not edit the literal — that would require a `SPRINT_7_PHASE4_PREREG.md` §19
  amendment. `final_lr_actor` is correct, and B6's quantity is covered anyway by **B5 + B1a**.
- **Safe next step.** **Expect 18/19.** Treat 19/19 as the anomaly.

---

### 18.14 A probe printed all its numbers and then crashed — or seemed to

- **Symptom.** Output looks complete; an artifact is missing or truncated.
- **Likely cause.** A `| tee` masked a nonzero exit code. This has already hidden a `KeyError` that
  crashed **after** printing everything.
- **How to check.**
  ```bash
  set -o pipefail
  ```
  then re-run and check `$?`.
- **What NOT to do.** Do not trust stdout as evidence of success.
- **Safe next step.** Make `set -o pipefail` unconditional for probe runs, and validate the artifact
  (`python -c "import json; json.load(open(...))"`), not the console.

---

### 18.15 A long run appears hung — no output for many minutes

- **Symptom.** Silence.
- **Likely cause.** ⚠ **Never pipe a long training run through `tail`** — the pipe buffers everything
  until exit, so you see nothing for the whole run.
- **How to check.** Poll a **side artifact** instead:
  ```bash
  wc -l saved_models/marl/R2_trajectory/SPRINT_7_P4_trajectory_manifest.jsonl
  ```
- **What NOT to do.** Do not kill and restart. Restarting a seeded run wastes 20+ minutes to produce
  identical bytes.
- **Safe next step.** Watch the manifest line count or the `_history.csv` row count grow.

---

### 18.16 A batch-size "control" shows no difference at all

- **Symptom.** `diag_S7_D4_ppo_update_R2_b32.json` is identical to the non-`b32` version.
- **Likely cause.** ⚠ **D3/D4 silently ignore `--episodes`** — they read `rollout_episodes` from the
  **checkpoint config**. The four `*_b32.json` artifacts are **bit-identical duplicates** (n_dec 5770,
  n_hiE 8, ‖g_all‖ 0.0345, all cosines, k1_mean −0.000073).
- **What NOT to do.** Do **not** read them as a batch-size control, and do not conclude batch size has
  no effect from them.
- **Safe next step.** The real matched-batch control is **`_diag_div_content.py`**, which passes
  `n_eps` straight to `build`.

---

### 18.17 `replica_fidelity ok=False` in a D4 artifact

- **Symptom.** A fidelity check fails for R2 but not A0.
- **Likely cause.** **Critic-side only, and only for mc-target arms.** The mismatch is entirely
  `critic_loss` (7.459 / 3.905 for R2) and `explained_var` (≤ 0.153). A0 (lambda target) is
  `ok=True, max 0.0000` at all three snapshots.
- **How to check.** Inspect which fields differ.
- **What NOT to do.** Do not discard the artifact.
- **Safe next step.** **All actor-side quantities in D4 are exact** — use them freely; treat the two
  critic scalars as approximate for mc arms.

---

### 18.18 A "risk response" for CLOUD or PREEMPT is zero

- **Symptom.** π(PREEMPT | high risk) = 0 everywhere.
- **Likely cause.** ⚠ **A masking fact, not a policy fact.** `PREEMPT_REROUTE` is legal in
  **exactly 0.00%** of high-risk decision entries on **both** frozen state sets. `MIGRATE_CLOUD` is
  legal in only **26.4%** (RANDOM) / **8.0%** (UNION).
- **What NOT to do.** Do not credit or blame the policy for never preempting.
- **Safe next step.** Divide any high-risk rate by its legality rate first. The effective high-risk
  choice is **binary STAY vs MIGRATE_EDGE**.

---

### 18.19 Results won't reproduce between two runs of the same command

- **Symptom.** Different numbers from an identical command.
- **Likely cause.** ⚠ **`--device cuda`.** It silently breaks seed reproducibility and is **2.8×
  slower** on these tiny actors.
- **How to check.** Read `device` in the run's `*_config.json`. Two CUDA arms exist
  (`mappo_A0_CUDA_device_mismatch`, `run_A1_CUDA_killed_at_90ep.log`) and are explicitly *"not
  comparable to the CPU ladder."*
- **What NOT to do.** Do not average a CUDA run with CPU runs.
- **Safe next step.** **`--device cpu`, always.** Phase 4 proved R2 replicates **bit-exactly** on CPU.

---

### 18.20 I think I overwrote an experiment output

- **Symptom.** An artifact's mtime is today.
- **Likely cause.** You reran a command with a **historical `--tag`**. `--tag` determines output
  filenames, and re-running a tag overwrites that arm.
- **How to check.**
  ```bash
  git status --short -- python-ai/saved_models/
  ```
  ```bash
  git diff --stat HEAD -- python-ai/saved_models/
  ```
  A tracked artifact showing ` M` was overwritten; the manifests in Part 11 tell you what its hash
  should be.
- **What NOT to do.** Do not rerun again hoping to restore it.
- **Safe next step.** If tracked: `git checkout -- <path>` restores the committed bytes exactly. If
  untracked and lost: say so in the report rather than silently regenerating. **Use a fresh `--tag`
  from now on.**

---

### 18.21 A test fails

- **Symptom.** `tests_env` or `sanity_test` reports a failure.
- **Likely cause.** Expected passes are **19/19** and **6/6** (T6 matches to 4.48e-07). A failure
  means the environment or reward changed.
- **How to check.** `git diff -- python-ai/marl/env.py python-ai/marl/config.py`.
- **What NOT to do.** Do not weaken the assertion. Note the suite's own caveat: *"T2 is a weak
  guard"* — a **passing** T2 is not strong evidence either.
- **Safe next step.** Revert the source change; every historical result assumes the frozen
  environment.

---

# Part 19 — Reproducibility checklist

Print this. Work down it. It is derived from what the pre-registrations in this repository actually
required.

## BEFORE RUNNING

- [ ] **Branch.** `git branch --show-current` → `host-predictor-finalization`. **Do not create a new
      branch.**
- [ ] **Clean status.** `git status --short` — every line is understood and intended.
- [ ] **Commit identity recorded.** `git log --oneline -1`. Phase 0 recorded `a8df6d9`; whatever you
      run, write down the HEAD hash. `git diff --stat HEAD` should be empty.
- [ ] **Working directory** is `python-ai/`.
- [ ] **Config verified from the artifact, not from memory.** Read `config.mappo` (not
      `config.train`) out of the reference run's `*_config.json` and diff it against your intent.
- [ ] **Seed.** `20260818` for training arms unless you are deliberately varying it. The simulator's
      is `20260817L`.
- [ ] **Device.** `--device cpu`. Non-negotiable.
- [ ] **Inputs verified.** `md5sum -c` the `inputs` manifest **from its capture directory** — expect
      `3/3 OK`. If an input moved, stop.
- [ ] **Manifests captured.** `code`, `artifacts`, `inputs`, `checkpoints` — *before* you run.
- [ ] **Output path.** A **fresh `--tag`**. Confirm nothing with that tag exists:
      `ls saved_models/marl/ | grep <tag>` returns nothing.
- [ ] **Pre-registration exists** if this run will produce a claim: hypothesis, metric, decision rule,
      and the numeric threshold — written down **before** the measurement. Rung 3 and Phase 4 both did
      this; that is why their NO-GOs are credible.
- [ ] **Justify it against RULE 4** — do not train to answer what frozen artifacts already answer.

## DURING THE RUN

- [ ] `set -o pipefail` is active.
- [ ] Command echoed into the log so the log self-documents.
- [ ] **Not** piped through `tail`. Progress watched via a side artifact's line count.
- [ ] Log captured to a named `.log` file — these are the project's chronology (Part 21).
- [ ] **No production file edited mid-run.** Instrumentation goes in an **additive driver** under
      `marl/diag/`, never in `train.py`.

## AFTER THE RUN

- [ ] **Exit code was 0.** Checked, not assumed.
- [ ] **Outputs exist and parse.** `json.load` each JSON; `head`/`tail` each CSV; `torch.load` each
      checkpoint with `map_location='cpu'`.
- [ ] **Row counts correct.** `_history.csv` = episodes; `_updates.csv` = `episodes //
      rollout_episodes` (75 for the standard schedule).
- [ ] **`after` manifests captured**, plus `prior_manifests` / `own_manifests` for the chain.
- [ ] **`code_before` re-verified** — production source must be unchanged.
- [ ] **`inputs_before` re-verified** — `3/3 OK`.
- [ ] **`checkpoints`/`trajectory` manifests unchanged** if the run was read-only.
- [ ] **New files enumerated** by diffing `after` against `before` path lists, and the count matches
      what the report claims.
- [ ] **Zero deletions.** Nothing in this project was ever deleted.
- [ ] **Any manifest failure diagnosed, not patched.** Three of Phase 0's fourteen failed and none
      were patched. If a failing file could have changed a number, **re-derive the number** — the bar
      is `MAX |diff| = 0.000e+00`.
- [ ] **Metrics computed with the right denominators** (Part 18.5) and compared only within a reward
      group (Part 10.3).
- [ ] **Report written**, including what *failed* and what stayed unknown.
- [ ] **`git status` checked**, then `git add -A` → `git diff --cached --stat` → commit → push.

---

# Part 20 — One-page cheat sheet

```
════════════════════════════════════════════════════════════════════════════════
 DT-MARL-Healthcare · branch host-predictor-finalization · HEAD df75ce3
════════════════════════════════════════════════════════════════════════════════
 ARCHITECTURE
   simulation/  Java + CloudSim Plus 8.5.7 · SIM_SEED 20260817L · 1500 s · 10 devices
        └─ writes failure_log.csv / failure_history.csv / device_failure_history.csv
   python-ai/data,models,training/  BiLSTM predictor · 26 features · seq 10 · thr 0.18
        └─ OOF risk (leakage-safe, UNCALIBRATED) → simulation/predicted_risk.csv
   python-ai/marl/  MAPPO on a TRACE-DRIVEN REPLAY (not closed-loop co-simulation)
        10 actors 48→[128,128]→4 (23,300 ea) · 1 critic 499→[256,256]→1 (194,049)
        obs 48 · state 489 (+10 one-hot = 499) · 4 actions · ring degree 4
        episode 400 steps × 2.0 s = 800 s · train ticks [9,491] · eval [491,698]
   ACTIONS  0 STAY · 1 MIGRATE_EDGE · 2 MIGRATE_CLOUD · 3 PREEMPTIVE_REROUTE
   CLOUD_NODE_ID −2 · cloud_slots 8

 SPRINT TIMELINE  (Sprint 0/1/2 do not exist as named stages)
   S-early   07-20…07-26  data pipeline, HTCF baseline, digital twin
   S3/3.5/3.75  ~08-17   dataset was degenerate: 5/12 cols constant → 0/12
   S4        08-17       26 features beat legacy 7 · PR-AUC 0.28634 (13.3× lift)
   S5        07-29       criticality parity < 5e-4 · severity support [0.320,0.644]
   S6a       08-18       6-run HP search → γ0.999 λ0.995 · mappo 1460 s
   S6b       08-18       ❌ MAPPO 20.62/prot 3.2  LOSES to threshold 76.50/prot 30.2
                         risk→0 ablation IMPROVES 20.62→23.13
   S6.5      08-19       11 hypotheses: A,B,C,D,J,K refuted · F,G,H supported
                         true adv +2.671 hi vs −1.337 lo · BC acc 1.0000 span 0.3935
   S7 P1     08-19       "converged, pointed the wrong way" (inspection only)
      R0     08-20       signal +3.363 real · GAE 29.0% hi-risk = below chance
      R1     08-21/25    λ→MC offline: 26.3% → 60.7%  SUPPORTED
      R2     08-25       critic_target mc — THE ONLY PRODUCTION EDIT
                         Δ_EDGE +0.2024/+0.1579 best · reward −0.46 prot 9.9
      R2.5   08-25       NO-GO · 87% of sign error fixed · 3 mechanisms split
      R2.75  08-25       paired est · Var ratio 3.72× · Δ_EDGE DEFINED
      R3     08-25       batch 8→32: SNR ×2.03 p=0.0000, behaviour WORSE  NO-GO
      Rung3  08-27       dilution real 9/9 but orders arms backwards  NO-GO
      Diverg 08-27/30    rollout_episodes = PURE VARIANCE (sd ratio 2.236)
      P0     08-30       20/20 instruments INVERT · mean ρ −0.900 · Mech 2 CLOSED
      P4/4.1 08-30       R2 replicated BIT-EXACTLY · 76 checkpoints · 18/19
      P5     08-31       Δ_EDGE = DIFFERENTIAL SUPPRESSION · actor freezes u62
      Final  08-31       COMPLETE · no further training · Sprint 7 CLOSED

 DIRECTORIES
   simulation/src/main/java/com/dtmarl/**        44 .java
   python-ai/marl/                               15 production .py + 32 _diag_/_smoke_
   python-ai/marl/diag/                          6 Phase 0/4/5 drivers
   python-ai/saved_models/marl/                  25 .pth · 38 .csv · 81 .json · 15 .md
   python-ai/saved_models/marl/R2_trajectory/    76 .pth + manifest.jsonl
   python-ai/saved_models/marl/_*_integrity/     12 manifest dirs
   python-ai/*.log                               69 run logs (2 are 0 bytes)

 COMMANDS  (cwd = python-ai/ · set -o pipefail · --device cpu ALWAYS)
   READ   git status --short   |  git diff --stat HEAD  |  git log --oneline -10
   TEST   python -m marl.tests_env        → expect 19/19
          python -m marl.sanity_test      → expect 6/6
   DIAG   python -m marl.diag._phase4_verify   → expect 18/19 (B6 fails BY DESIGN)
          python -m marl.diag._phase5_risk_trajectory --device cpu --clusters 32 \
                 --start-seed 20260825 --boot 5000 --random-seed 31337 --tag main
   ⚠TRAIN python -m marl.train --critic-target mc --rollout-episodes 32 \
                 --episodes 2400 --seed 20260818 --device cpu --tag R3_batch32
                 ^^ the ONE verbatim command in the corpus. --tag OVERWRITES.
   ⚠EVAL  python -m marl.evaluate --model saved_models/marl/<arm>.pth --episodes 8 \
                 --device cpu

 FROZEN CONFIG   (lives at config.mappo — NOT config.train)
   gamma 0.999 · gae_lambda 0.995 (horizon 166.8) · clip_eps 0.2 · value_clip 0.2
   entropy_coef 0.02 · value_coef 0.5 · max_grad_norm 0.5 · ppo_epochs 4
   minibatches 4 · normalise_advantages True · critic_target mc · anneal_lr True
   separate_actors True · episodes 600 · rollout_episodes 8 · seed 20260818
   lr_actor 7e-4 · lr_critic 1e-3     ⇒ n_updates = 600//8 = 75
   final lr_actor 9.3333e-06 · lr_critic 1.3333333333333308e-05 · 1392 Adam steps

 CHECKPOINTS
   mappo.pth              Sprint 6 original (20.62)
   mappo_A0_cpu_repro     THE CONTROL — reproduces S6 bitwise 9832/77/842/0
   mappo_A1_cpu_bugfix    both S6.5 fixes on
   mappo_A2_crit_sign     w_criticality_migration 1.0→0.0  (crit = CRITICALITY)
   mappo_A3_entropy       entropy_coef 0.05 (NOT 0.01→0.02 — Phase 0 §1.1 is wrong)
   mappo_R2_mc_target     BEST ARM · _best is BIT-IDENTICAL
   R3_batch32             4× batch · _best is a genuine update-45 checkpoint
   R2_trajectory/u000…u075  76 per-update snapshots (u000 has no Adam state)

 INTEGRITY
   md5sum -c <manifest>           ← MUST run from the manifest's capture cwd (*./)
   git diff --stat HEAD           ← stronger baseline than MD5 alone
   1 failure naming a .md5        ← circular self-hash, NOT a modification
   code_before fails on config/mappo/train.py ← the Rung 2 change, 08-25 04:02

 KEY METRICS
   Δ_EDGE = π(EDGE|risk≥0.6) − π(EDGE|risk<0.2)   frozen sets:
        RANDOM 4,975 entries / 216 hi · UNION 74,237 / ~3,592 hi
   protected · relocations · lost — reward-INDEPENDENT, safe across arms
   reward — THREE conventions: A0 76.50 · bugfixed 67.31 · A2 76.54 (baseline value)
        static-no-migration = −37.06 in EVERY arm ← the internal control
   infeasible = CONTENTION, not illegal actions (actor is masked)
   approx_kl  = signed k1, legitimately negative (26/75)
   clip_frac  = 0.0000 in 22/75 updates — NORMAL

 TRAPS
   --tag probes write <name>_<tag>.json — glob, never exact-name
   D3/D4 ignore --episodes ⇒ *_b32.json are DUPLICATES, not a control
   PREEMPT legal in 0.00% of hi-risk states · CLOUD in 26.4%/8.0%
   Rung-2.75 artifact root = by_state_source; some fields hi-risk-only
   no adv_mean column in *_updates.csv
   --device cuda breaks seed reproducibility and is 2.8× slower

 FINAL CONCLUSION
   Self-healing IS achievable here — a one-line risk threshold protects ~30 tasks
   vs the learned policy's ~3 at the SAME migration volume. MAPPO's failure is its
   ADVANTAGE ESTIMATOR, inverted in the 4.3–6.5% high-risk minority: low-risk EDGE
   suppression is CORRECT (true adv −1.337) and generalises through shared params
   into the high-risk regime where it is wrong. Sprint 7 fixed 87% of the sign
   error without recovering behaviour, and its headline Δ_EDGE gain turned out to
   be LOW-RISK SUPPRESSION (159–218%), not high-risk acquisition — π(EDGE|hi) ends
   BELOW random init in 76/76 checkpoints. WHY THE ARMS DIFFER IS UNKNOWN.
   Sprint 7: COMPLETE. Further training: NOT JUSTIFIED.
════════════════════════════════════════════════════════════════════════════════
```

---

# Part 21 — Do not lose the history

Dates are from `git log --date=iso` (commits) and file modification times (everything else). Both are
repository evidence. `DATE NOT VERIFIED` appears wherever neither exists.

| Date | Sprint | Experiment / artifact | What changed | Result | Consequence |
|---|---|---|---|---|---|
| 2026-07-20 08:29 | S-early | first Java sources (`BrokerManager.java`) | Repository bootstrap | — | Simulator work begins |
| 2026-07-20 09:08 | S-early | commit `8feaa98` | *Initial project structure* | — | — |
| 2026-07-21 03:19 | S-early | `htcf_model.pth` | HTCF baseline trained | — | Baseline for later comparison |
| 2026-07-21 12:12 | S-early | commit `ea08805` | *Refactor data pipeline and implement HTCF baseline* | — | — |
| 2026-07-25 11:59 | S-early | commit `1490e3a` | *Add failure simulation and ClusterData2019 processing* | Failure simulation added | The trace becomes possible |
| 2026-07-25 22:26 | S-early | commit `63e9bcb` | *Integrate digital twin failure prediction and early warning* | Prediction + early-warning seam | The risk channel is born |
| 2026-07-26 08:32 | S-early | `device_failure_predictor.pth` | IoMT-device predictor | — | Parallel track |
| 2026-07-29 05:52 | **S5** | commit `208e5a2` — *sprint 5* | Patient criticality | `self_check()` < 5e-4 vs `JAVA_PRIORITIES_T0` | Criticality enters the reward. **Last commit for a month** |
| DATE NOT VERIFIED | **S3 / 3.5 / 3.75** | dataset diagnosis + trace regeneration | Failure model rewritten in the simulator | 5/12 constant columns → **0/12**; 32 failures / 32 recoveries | *Fix the data, not the model.* No report survives; dated only by its downstream effect on the 08-17 predictor runs |
| 2026-08-17 17:33 | **S4** | `run_host_observable.log` → `failure_predictor.pth`, `host_metrics.json`, `failure_predictor_meta.json` | 26-feature observable arm | **PR-AUC 0.28634**, ROC 0.81594, F1 0.47297 on n=14,910 / 320 positives | The production risk channel |
| 2026-08-17 17:43 | S4 | `run_host_legacy.log` → `*_legacy.*` | Legacy 7 features | PR-AUC **0.0777**, F1 0.0467 | Feature engineering is load-bearing |
| 2026-08-17 17:53 | S4 | `host_lead_time_legacy.csv` | Lead-time analysis (legacy) | — | — |
| 2026-08-17 18:19 | S4 | `run_lono_observable.log` → `host_lono.json` | Leave-one-node-out | node 0 PR-AUC 0.4042 / ROC 0.9260 | Generalises across nodes |
| 2026-08-17 18:26 | S4 | `failure_predictor_OLDDATA.pth`, `host_metrics_OLDDATA.json` | Old trace, new features | PR-AUC 0.2174 at n=1,680 / 39 pos | Confirms the dataset fix mattered |
| 2026-08-17 18:32 | S4 | `host_lead_time.csv`, `host_lead_time_OLDDATA.csv` | Lead time | 23 EARLY_WARNING / 7 MISSED / 2 STANDING_ALARM; lead mean 8.39 median 10.0; run_lead 18.22 | The channel is **actionable** |
| 2026-08-18 13:56 | S6 | `marl/criticality.py` — earliest `marl/` source | MARL package created | — | The RL work starts |
| 2026-08-18 15:05 | **S6a** | `run_mappo_train_attrbug_gamma99.log` | γ0.99 regime | **COMPLETED 938 s**, greedy TRAIN −146.55 | Abandoned |
| 2026-08-18 15:14 | S6a | `..._attrbug_gamma95_killed.log` | γ0.95 | **killed ep 100** (−90 → −170) | — |
| 2026-08-18 15:50 | S6a | `..._attrfix_gamma95_killed.log` | γ0.95 + attr fix | **killed ep 175** | — |
| 2026-08-18 16:02 | S6a | `simulation/predicted_risk.csv` | Risk CSV exported to the simulator seam | — | The Java↔Python interface is live |
| 2026-08-18 16:04 | S6a | `..._g999_lowEV_killed.log` | γ0.999, low EV | **killed ep 75**, ev +0.37 | — |
| 2026-08-18 16:27 | S6a | `run_mappo_lam95_killed.log` | λ0.95 | **killed ep 350**, reward −24 → −9.45, ev 0.92 | λ0.995 chosen instead |
| 2026-08-18 16:57 | **S6a** | `run_mappo_train.log` → `mappo.pth` | **γ0.999 / λ0.995 selected** | **COMPLETED 1460 s** | The production hyperparameters freeze |
| 2026-08-18 17:03 | S-early/S6 | `failure_log.csv`, `failure_history.csv`, `device_failure_history.csv` | Trace exported | 10 nodes × 1500 ticks | The frozen substrate |
| 2026-08-18 17:01 | **S6b** | `run_mappo_eval.log` → `mappo_eval.json` | **Baseline comparison** | ❌ MAPPO **20.62** / prot **3.2** vs risk-threshold **76.50** / prot **30.2** at 35.9 vs 36.2 relocations; reactive −0.09; random −8.11; static −37.06. **risk→0 ablation IMPROVES 20.62 → 23.13** | **The result that created the research question** |
| 2026-08-18 17:21 | S6 | `MigrationDemo.java` — last Java edit | — | — | Java work ends here |
| 2026-08-18 23:31 | S6.5 | `run_diag_counterfactual.log` | Counterfactual replay | true adv **+2.671 ± 0.806** hi vs **−1.337 ± 0.967** lo, **0 replay mismatches** | The signal is real and the gap is +4.01 |
| 2026-08-19 06:44 | S6.5 | `run_A0_sprint6_repro.log` | A0 first attempt | — | — |
| 2026-08-19 06:46 | S6.5 | `run_A0_CUDA_device_mismatch_eval.log` | CUDA arm | Device mismatch | ⚠ Not comparable to the CPU ladder |
| 2026-08-19 06:54 | S6.5 | `run_A1_CUDA_killed_at_90ep.log` | CUDA A1 | **killed ep 90** | **`--device cpu` becomes mandatory** |
| 2026-08-19 06:58 | S6.5 | `chain_cpu.log` | — | **0 bytes** | No information |
| 2026-08-19 07:23 | **S6.5** | `run_A0_cpu_repro.log` → `mappo_A0_cpu_repro.pth` | **A0 — the control**, legacy switches ON | **Reproduces Sprint 6 BITWISE**: 600/600 ep, 75/75 updates, histogram **9832/77/842/0** | Everything after this rests on A0 |
| 2026-08-19 07:47 | S6.5 | `run_A1_cpu_bugfix.log` | A1 — both fixes on | contention 7/7 → 3/21 (still ~70%); 1,060 cloud vs 10 edge attempts | *"Neither change adds any `if risk > threshold` logic"* |
| 2026-08-19 07:52 | S6.5 | `chain_A2_A3.log` | — | **0 bytes** | No information |
| 2026-08-19 08:17 | S6.5 | `run_A2_crit_sign.log` | A2 — `w_criticality_migration 1.0 → 0.0` | Criticality direction **+0.801** vs A0/A1/A3's ≈ −0.99; the expected sign flip: *"It did not"* | The dead-weight defect documented |
| 2026-08-19 08:42 | S6.5 | `run_A3_entropy.log` | A3 — `entropy_coef 0.05` | span 0.002 — the **narrowest** | Exploration does not fix it |
| 2026-08-19 08:48 | **S6.5** | **`SPRINT_6_5_REPORT.md`** (239 lines) | — | A,B,C,D,J,K **refuted**; F,G,H **supported**. BC acc **1.0000** span **0.3935** vs policy 0.004 (*"4× / 25× short"*). Estimator reads **−0.266** against a true **+2.671** | The inverted estimator is named. §13 proposes a **smaller** `rollout_episodes` |
| 2026-08-19 21:31 | **S7 P1** | **`SPRINT_7_PHASE1_DIAGNOSIS.md`** | Inspection only | Greedy A0: CLOUD in **842/855** (98.5%), EDGE in **77/10,751** (0.7%). *"Not a stalled optimiser — a converged one, pointed the wrong way."* | ⚠ Proposes a B/C/E ladder that **never ran**. Rung 0 is next |
| 2026-08-19 21:35 | S7 | production source hash boundary (from the Rung 0 manifest) | — | — | Marks the *before* side of the only production edit |
| 2026-08-20 06:12 | **S7 R0** | `run_S7_rung0_d1d2_train.log`, `..._d3d4.log` | — | Signal **+3.363**; truth-vs-GAE 0.7012 overall but **0.3931** at high risk; **GAE sign agreement 29.0%** | ✅ Signal real, ❌ GAE below chance |
| 2026-08-20 06:40 | S7 R0 | `SPRINT_7_RUNG0_REPORT.md` | — | — | Rung 1 asks whether the **target** is to blame |
| 2026-08-21 05:07 | **S7 R1** | `run_S7_rung1_fit.log`, `..._deviations.log` | λ-return vs MC, **offline** | **26.3% → 60.7%**; estimate −5.24 → −1.07 | ✅ SUPPORTED **for A0** |
| 2026-08-25 02:36 | S7 R1 | `SPRINT_7_RUNG1_REPORT.md` | — | — | Try it in production |
| **2026-08-25 04:02** | **S7 R2** | production source change | **`critic_target: "lambda" → "mc"`** | — | ⚠ **The ONLY production edit in all of Sprint 7.** Byte-identical in all eight later manifests and at HEAD |
| 2026-08-25 03:37 | S7 R2 | `run_R2_mc_target_train.log` → `mappo_R2_mc_target.pth` | MC target in production | Δ_EDGE **+0.2024 / +0.1579** — best of any arm | ⚠ But reward **−0.46**, protected **9.9** |
| 2026-08-25 03:45 | S7 R2 | `run_R2_mc_target_eval.log`, `..._best_eval.log` | Held-out evaluation | — | `R2_best` is **bit-identical** to `R2` |
| 2026-08-25 05:54 | **S7 R2.5** | `SPRINT_7_RUNG2_5_*.log` (6 logs) | Target-rung interrogation | **130/130** agreement; high-risk sign error **58.4% → 7.8%** ≈ **87% fixed** | 🛑 NO-GO |
| 2026-08-25 06:45 | S7 R2.5 | `SPRINT_7_RUNG2_5_REPORT.md` | — | Three mechanisms separated; §G.5 **UNRESOLVED** | Mechanism 2 needs its own estimator |
| 2026-08-25 10:25 | **S7 R2.75** | `SPRINT_7_RUNG2_75_*.log` (8 logs) | Paired estimator; matched frozen state sets | `Var(raw)/Var(paired) = 3.72×` ⇒ 1.93× attenuation; agreement **0.71 / 0.95 for every arm** | ✅ Per-state offset, not a target defect |
| 2026-08-25 12:07 | S7 R2.75 | `SPRINT_7_RUNG2_75_matched_states.log` | **RANDOM (4,975 / 216) and UNION (74,237 / ~3,592) constructed** | — | The frozen evaluation substrate for everything after |
| 2026-08-25 13:08 | S7 R2.75 | `SPRINT_7_RUNG2_75_REPORT.md` | — | **Δ_EDGE defined.** §A.4 **withdraws** the earlier metric | Variance is the next suspect |
| 2026-08-25 17:38 | **S7 R3** | `run_R3_batch32_train.log` → `R3_batch32.pth` | **`--rollout-episodes 32 --episodes 2400`** *(the one verbatim command)* | SNR **×2.03**, **p = 0.0000** — and Δ_EDGE **WORSE**. Block means **−16.11 → −12.16 → −10.59 → −8.51** | 🛑 **Variance hypothesis FALSIFIED** |
| 2026-08-25 18:10 | S7 R3 | `SPRINT_7_R3_action_channels.log` → **`SPRINT_7_R3_action_channels_R3.json`** (36,293 B) | Four-action Δ, six arms, both sets | `status = 'EXPLORATORY -- not preregistered'` | ⚠ The `--tag`-suffix trap; later misreported as non-existent by Phase 5 §4.4 |
| 2026-08-25 18:22 | S7 R3 | `SPRINT_7_R3_REPORT.md` | — | — | ⚠ **R3 never evaluated on the held-out suite** — no `R3*eval*.json` exists |
| 2026-08-27 12:07 | **S7 Rung 3** | **`SPRINT_7_RUNG3_PREREGISTRATION.md`** (10,688 B) | Pre-registers the dilution test | — | Criteria fixed **before** measurement |
| 2026-08-27 12:27 | S7 Rung 3 | `SPRINT_7_RUNG3_*.log` (7 logs) | Exact `g_full = g_hi + g_lo` | `cos(g_full, g_lo) ∈ [+0.9099, +0.9994]` in **9/9**; `cos(g_hi, g_synth)` +0.913/+0.536/+0.451; mass ratios order arms **backwards** | Dilution real, explains nothing |
| 2026-08-27 13:16 | S7 Rung 3 | `SPRINT_7_RUNG3_REPORT.md` | — | **`go_for_training_rung = False`** | ⚠ `synth` doesn't discriminate the action channel |
| 2026-08-27 21:32 | **S7 Divergence** | `SPRINT_7_DIV_*.log` (9 logs) | Characterise `rollout_episodes` | Expected content invariant (**≤ 0.105 σ₈**); **sd(8)/sd(32) = 2.236** exactly as the finite-population law predicts; critic diff 0.009 | ✅ **Pure variance intervention** |
| 2026-08-27 23:15 | S7 | `_diag_div_critic.py` — last `marl/` diagnostic written | — | — | The `_diag_*` era ends |
| 2026-08-30 10:44 | — | **commit `a8df6d9`** | *Update project with latest implementation* | `git diff --stat HEAD` empty | ⚠ **First commit in a month.** Git becomes a stronger baseline than MD5 alone |
| 2026-08-30 11:02 | S7 Divergence | `SPRINT_7_DIVERGENCE_REPORT.md` | — | The *"5.1× collapse"* was a **+2.09σ outlier**. **Regime-selective asymmetry**: R3 learned the low-risk bulk, not the high-risk minority | §10 names **per-update checkpoints** as the largest gap |
| 2026-08-30 11:58 | **S7 Phase 0** | `marl/diag/_phase0_calibration.py` | First `diag/` module | — | The instrument audit |
| 2026-08-30 11:59 | S7 Phase 0 | **`SPRINT_7_PHASE0_RECONSTRUCTION.md`** (294 lines) | — | **20 readings. 20 negative. 0 positive. Mean ρ = −0.900.** Endpoint = **headroom** (A0 +0.4023 vs R2 −0.2372). Mechanism 1 CLOSED, **2 CLOSED**, 3 DOWNGRADED. Integrity 96/96, 41/41, 58/58, 3/3; **11 clean / 3 failed, all diagnosed, none patched** | **The blocker is a missing observable, not a missing hypothesis** |
| 2026-08-30 14:21 | **S7 Phase 4** | `marl/diag/_phase4_r2_trajectory.py` | Additive driver — `train.py:232/252` untouched | — | Instrumented replication |
| 2026-08-30 14:36 | S7 Phase 4 | `_phase4_smoke.py` → `SPRINT_7_P4_SMOKE_REPORT.json` | Smoke test | — | — |
| 2026-08-30 15:01 | S7 Phase 4 | `_phase4_verify.py` | 19 checks | ⚠ line 78 holds a **1-ULP-wrong** literal | **B6 fails by design; left wrong on purpose** |
| 2026-08-30 15:05 | S7 Phase 4 | `_phase4_equiv.py` → `SPRINT_7_P4_EQUIV_SELFTEST.txt` | Equivalence self-test | — | — |
| 2026-08-30 15:08 | S7 Phase 4 | **`SPRINT_7_PHASE4_PREREG.md`** (45,749 B) | Pre-registration, RULE 4 justification | — | Criteria fixed before measurement |
| 2026-08-30 16:37 | **S7 Phase 4** | **`SPRINT_7_PHASE4_REPORT.md`** → `R2_traj_repro*` + **`R2_trajectory/u000…u075`** + `manifest.jsonl` | Zero-variable replication | ✅ **BIT-EXACT, storage-for-storage, on CPU.** 76 checkpoints. **18/19**. Protected ρ = +0.053 n.s. | Nondeterminism explains nothing. ⚠ A **"Phase 6" naming slip**; **Phase 4.1 has no report** |
| 2026-08-31 02:45 | **S7 Phase 5** | `marl/diag/_phase5_risk_trajectory.py` | Trajectory reader | — | — |
| 2026-08-31 02:51 | S7 Phase 5 | `SPRINT_7_PHASE5_run.log` → `..._risk_trajectory_{main,parity}.json` | 76-checkpoint sweep | — | — |
| 2026-08-31 03:01 | **S7 Phase 5** | **`SPRINT_7_PHASE5_REPORT.md`** | — | **π(EDGE\|hi) ends BELOW init and never exceeds it in 76/76.** **159–218%** of Δ_EDGE growth is low-risk suppression. Actor **freezes at u62**. PREEMPT legal in **0.00%**, CLOUD in **26.4%/8.0%** ⇒ effectively **binary STAY vs EDGE** | ❌ **Δ_EDGE is differential suppression, not risk acquisition.** ⚠ §4.4 is wrong about a pre-existing artifact |
| 2026-08-31 04:39 | **S7 Final** | **`SPRINT_7_FINAL_SYNTHESIS_AUDIT.md`** (59,913 B, 12 §, 22-hypothesis ledger) | Read-only synthesis | High risk = **4.3–6.5%** of entries, gradient positive but **outvoted**; low-risk EDGE genuinely wrong (**−1.337**, n=159) ⇒ correct low-risk suppression generalises through shared parameters. **Interpretation D**: mostly suppression **plus** a real acquisition phase at **u042–u062** worth **23%/30%**. Arm ordering R2 > R3 > A0 **UNEXPLAINED** | **COMPLETE · further training NOT justified · Sprint 7 CLOSED.** Two things not to redo (`w*` = +9.832; the variance rung). §11 item 5 names the one **offline** analysis worth doing |
| 2026-08-31 04:46 | — | **commit `df75ce3`** | *Complete Sprint 7 research and diagnostics* | — | **Current HEAD** |

---

## What this handbook is, and is not

It reconstructs the project **from the repository**: 7 git commits, 69 run logs, 15 reports + 1
self-test, 25 root checkpoints + 76 trajectory checkpoints, 81 JSON, 38 CSV, 12 integrity manifest
directories, 44 Java sources and 53 Python modules.

Where the repository is silent, this document says **`NOT VERIFIED FROM REPOSITORY`** and stops.
Where the repository **contradicts itself** — twelve places, tabled in §2.1 — this document surfaces
the contradiction and does **not** resolve it, because resolving it would mean choosing which
historical record to overrule.

Nothing here modifies production code, historical reports or experiment outputs. No training was run.
No hypothesis, metric or command was invented. There is no Sprint 8 and no R4.




