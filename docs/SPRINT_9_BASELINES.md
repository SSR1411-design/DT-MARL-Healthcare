# Sprint 9 — Baseline Implementations

Sprint 9 implements three comparison baselines without training a new model or changing the validated MAPPO checkpoint or Sprint 8 artifacts.

## Baselines

| Name | Definition | Prediction / uncertainty | Digital Twin forecasting | Training |
| --- | --- | --- | --- | --- |
| `reactive-no-digital-twin` | Current-state reactive scheduling | Disabled | Disabled | Not required |
| `single-agent-ppo` | One centralized, factored PPO controller | Configurable only in a later study | Current global state only | Architecture only |
| `detection-without-prediction` | Detection followed by immediate reactive restart | Disabled | Disabled | Not required |

### Baseline 1 — current-state reactive scheduling

`marl.baseline.ReactiveThresholdPolicy` uses only currently observable degradation, CPU, packet loss, focus tasks, masks, and current placement state. It never calls `risk_at`.

The factory enforces this exact configuration:

```text
risk_source = "zero"
dest_w_risk = 0.0
reward.P_risk_expose = 0.0
```

The baseline does not load OOF predictions, a live predictor, predicted risk, uncertainty, or future trace state. “No Digital Twin” means no predictive or forecasting controller input. The common recorded trace still supplies present-time availability and telemetry, ensuring identical physical workload conditions.

### Baseline 2 — centralized single-agent PPO

`marl.single_agent_ppo.SingleAgentPPO` is independently identifiable as a real single-agent implementation:

- It reads one global environment state (489 dimensions for validated R2).
- One actor emits a `10 x 4` factored MultiDiscrete-style distribution.
- It samples one masked action factor per node; joint log probability is the sum of per-node legal log probabilities.
- It receives the real scalar team reward, the sum of agent rewards.
- It has one scalar global-state critic and a single-agent rollout buffer.

It is not MAPPO rebranded: it has no `marl.mappo` import, cannot load a MAPPO checkpoint, and saves its own checkpoint type, `sprint9_single_agent_ppo`. `marl/train_single_agent.py` exits unless both `--allow-training` and an explicit positive `--episodes` are supplied. Sprint 9 includes no training result or performance claim for this baseline.

### Baseline 3 — detection followed by reactive recovery

`DTMarlEnv(detection_recovery=True)` is only constructible with the zero-risk configuration above. Its companion policy always chooses `STAY`, eliminating proactive relocation.

On a real `trace.any_down_during` outage involving a resident or in-flight task, the environment does exactly this:

```text
observed failure in the completed transition
  -> remove affected task from failed placement
  -> select a currently observed healthy, capacitated edge node
  -> reset remaining work to full task length
  -> requeue/restart at the detection boundary
```

The hook does not wait for `HOST_RECOVERED`, preserve partial work, call OOF/live prediction, query uncertainty/risk, or inspect a future label before the outage is observed. If no healthy edge capacity exists, the existing real task-loss behavior remains and a failed `RecoveryRecord` is emitted. Every attempt records failure window, task, source, destination, recovery tick/time, zero-latency immediate-recovery semantics, and success.

## Fair evaluation protocol

`marl.sprint9_benchmarks.build_baseline` reads only the validated R2 configuration JSON, not MAPPO weights. Every baseline shares the ten-node trace, VM configuration, 40-task criticality workload, task model, failure trace, reward definitions, held-out start window (final 30% of usable starts), `marl.rollout.episode_starts`, and fixed seed convention. Non-predictive baselines intentionally zero risk destination/reward terms. Baseline-specific provenance is repeated per episode in `sprint9_metadata.csv`.

## Real metric CSVs

Run bounded evaluations from `python-ai`:

```powershell
python training/sprint9_metrics.py --baseline reactive-no-digital-twin --episodes 1
python training/sprint9_metrics.py --baseline detection-without-prediction --episodes 1
```

Default outputs are `data/sprint9/<baseline>/`; replacement is refused unless `--overwrite` is explicitly supplied for that Sprint 9 directory.

| CSV | Granularity | Source |
| --- | --- | --- |
| `sprint9_metadata.csv` | episode | reproducibility and baseline provenance |
| `sprint9_node_ticks.csv` | node / recorded tick | trace latency, power, CPU, RAM, bandwidth, availability; energy is measured power x recorded tick duration |
| `sprint9_step_rewards.csv` | agent / decision step | actual environment events/rewards; team reward is real sum |
| `sprint9_task_events.csv` | terminal task event | live success/failure and latency |
| `sprint9_task_metadata.csv` | task / episode | HSI, criticality and priority distribution |
| `sprint9_migrations.csv` | relocation | actual `MigrationRecord` entries |
| `sprint9_recovery_events.csv` | recovery attempt | Baseline 3 `RecoveryRecord` entries |
| `sprint9_prediction_usage.csv` | episode | explicit prediction/uncertainty provenance |
| `sprint9_episode_metrics.csv` | episode | reward, success, latency, energy, utilization, migration, recovery, HSI and priority summaries |

No aggregate resource-utilization percentage is emitted because no defensible cross-resource aggregation is defined. CPU, RAM, and bandwidth remain distinct measured percentages. Heuristic PPO/MARL loss is `N/A`; no evaluation loss is fabricated. The single-agent update CSV will contain loss only after an approved genuine training update.

## Validation

```powershell
python training/validate_sprint9_baselines.py
python training/validate_sprint9_baselines.py `
  --reactive-dir data/sprint9/reactive-no-digital-twin `
  --detection-dir data/sprint9/detection-without-prediction
python marl/tests_sprint9.py
```

The validation is read-only. It checks protected MAPPO/Sprint 8 hashes, verifies zero-risk construction without OOF loading, exercises an actual held-out failure transition, validates healthy recovery destinations, checks deterministic starts, rejects random metric generation, and constructs the independent single-agent PPO.

## Limitations

- Baseline 3 is strictly the requested immediate restart/requeue mechanism, not a general recovery orchestrator.
- Recovery placement is edge-only and current-state-only. No healthy capacity means a real logged loss.
- Baseline 2 remains untrained until a separately approved manual study and must never use MAPPO weights.
- Non-predictive baselines use current trace telemetry and observed failures, but no forecast of future state.
