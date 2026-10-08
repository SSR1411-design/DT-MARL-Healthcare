# Sprint 8 — Real metrics logging pipeline

`training/sprint8_metrics.py` exports research CSVs from recorded sources and a
bounded, greedy evaluation replay of an existing MAPPO checkpoint. It never
trains a policy, writes a checkpoint, calculates an evaluation loss, or creates
random metric values.

Run from `python-ai` after selecting a validated checkpoint:

```powershell
python training/sprint8_metrics.py --model saved_models/marl/mappo_R2_mc_target_best.pth --episodes 1 --device cpu --overwrite
python training/validate_sprint8_metrics.py --out-dir data --model saved_models/marl/mappo_R2_mc_target_best.pth
```

`--episodes` controls only how many deterministic held-out trace windows are
replayed. It is evaluation, not training; keep the default one episode unless a
separate evaluation protocol authorizes a larger fixed set. The exporter
refuses to overwrite an existing Sprint 8 CSV set unless `--overwrite` is
explicit.

## CSV provenance

| CSV | Granularity and values | Source | Scope |
| --- | --- | --- | --- |
| `data/sprint8_node_ticks.csv` | One row per node and recorded trace tick: link latency (ms), power (W), interval energy (J), CPU/RAM/bandwidth (%), availability/link telemetry, current OOF risk. `energy_interval_j = power_w * tick_duration_s`. | `marl.trace.Trace` loaded from `simulation/failure_history.csv`; risk from the configured OOF provider. | Evaluation replay; trace telemetry is recorded, not synthesized. |
| `data/sprint8_step_rewards.csv` | Per-agent, per-decision-step reward, requested/executed action, outcome deltas, migration and reward-energy costs. | `DTMarlEnv.step()` `info["events"]` and reward vector. | Online evaluation. |
| `data/sprint8_task_events.csv` | Actual terminal `COMPLETED`/`LOST` transitions with latency when completed, HSI, priority and migration counts. | `DTMarlEnv._advance_compute`, `_kill`, and task state transitions observed after `step()`. | Online evaluation. |
| `data/sprint8_task_metadata.csv` | Per-task HSI, clinical severity, priority at episode start/end, final state and migration counts. | `marl.criticality.PatientTask` / `priority_at()` and final environment task state. | Evaluation summary. |
| `data/sprint8_migrations.csv` | One row per real `MigrationRecord`: source/destination, action, cost, latency, risks and post-episode survival outcome. | `DTMarlEnv._relocate()` records in `env.migrations`. | Online event plus explicitly labelled post-episode evaluation outcome. |
| `data/sprint8_recovery.csv` | Actual failure/recovery pair and duration (seconds) for each node. | `simulation/failure_log.csv`, pairing `HOST_FAILURE` with that node's next `HOST_RECOVERED`; ticks are matched to the DT trace time axis. | Offline recorded simulation event log. |
| `data/sprint8_prediction_windows.csv` | Each OOF window's score, label, correctness, node/event alignment. | `saved_models/failure_predictor_oof.npz` aligned against `data.failure_dataset.build_windows()`. | Offline, leakage-safe OOF evaluation only; labels are never exposed to the policy. |
| `data/sprint8_prediction_metrics.csv` | OOF confusion-matrix components and derived accuracy, precision, recall, F1 and PR-AUC when provided by `training.host_cv.full_metrics()`. | Same OOF artifact and labels as the preceding file. | Offline evaluation only. |
| `data/sprint8_episode_metrics.csv` | Real episode reward, success rate, latency, migrations, recovery counts/durations, node-telemetry means, and HSI/priority distribution summaries. | `DTMarlEnv.episode_metrics()` plus sums recorded by the evaluation observer. | Evaluation summary. |
| `data/sprint8_training_loss.csv` | Existing actor/critic loss and MAPPO update diagnostics, unchanged with source path retained. | Existing `saved_models/marl/*_updates.csv`. | Historical training-update data only; no evaluation loss is invented. |

No aggregate resource-utilization percentage is exported because the repository
does not define a defensible aggregate. CPU, RAM, and bandwidth remain separate
percentages. `energy_reward_cost` is a reward term; it is deliberately distinct
from physical `power_w` and derived `energy_interval_j`.

## Safety and limitations

- The exported prediction rows are marked `offline_evaluation_only`; OOF labels
  and post-episode migration outcomes must never be fed to the policy.
- The current risk provider reserves uncertainty as `0.0`; this is labelled
  `reserved_zero_not_an_estimate`, not reported as model uncertainty.
- PPO/MAPPO loss exists at training-update granularity only. Sprint 8 copies
  that real log and does not compute a fictitious per-episode evaluation loss.
- `data/LEGACY_SYNTHETIC_INVALID_NOT_FOR_RESEARCH_sprint8_metrics.csv` is the
  renamed historical output from the former random-value script. It is kept
  only for audit provenance and is not read, overwritten, or considered a
  Sprint 8 research result by this pipeline.

## Verification

`training/validate_sprint8_metrics.py` checks headers and non-empty CSVs,
every node-tick telemetry value against the trace, recovery durations against
the event log, OOF rows/metrics against the saved leakage-safe artifact,
migration row counts against the environment's `relocations`, loss rows against
the existing update CSV, and the absence of `random` metric generation in the
exporter.
