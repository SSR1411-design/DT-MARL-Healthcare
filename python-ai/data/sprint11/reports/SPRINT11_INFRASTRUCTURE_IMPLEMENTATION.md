# Sprint 11 infrastructure implementation

Status: infrastructure only. No Java simulator, MAPPO training, OOF
regeneration, full matrix, or existing artifact overwrite was performed.

## Files and protection

New Python infrastructure is under `training/sprint11/`, with the read-only
validator at `training/validate_sprint11.py`. New scenario data, trial,
aggregate, graph, and report roots are under `data/sprint11/`. Java changes
add an opt-in `Sprint11ScenarioConfig` and controlled constructor parameters
to the host, VM, task, and link layers. The existing no-argument Java path
retains R2 defaults.

No file under `data/sprint8/`, `data/sprint9/`, `data/sprint10/`,
`saved_models/marl/mappo_R2_mc_target_best.pth`, the R2 configuration, OOF
artifact, or existing experiment CSVs was edited. New writers refuse an
existing destination.

## Scenario, trial, and metadata design

`sprint11_scenario_matrix.json` defines independent physical scenarios with
`host_count`, `edge_node_count`, `device_count`, `task_count`, `patient_count`,
physical `bandwidth_mbps`, failure intensity, trial count, and the progressive
wear parameter block. Scenario IDs are the first 16 hex characters of a
SHA-256 hash of the canonical physical configuration. Text descriptions and
planned policy labels do not alter the ID.

The failure block preserves the active wear model: fault onset, background and
episode wear, lognormal wear noise, susceptibility and severity ranges,
Weibull-like hazard scale/shape, abrupt probability, and repair duration. The
intensity multiplier affects only onset/wear/hazard rates; it never enables
the legacy random-failure API.

For each `(scenario_id, trial_id)`, deterministic SHA-256-derived simulator
and Python seeds use distinct streams. One completed `trial_result.json` is
one independent simulator realization. Replay windows are metadata-labelled
as nested, not independent. Result records include the requested counts,
failure configuration, seeds, trace/risk/checkpoint IDs and hashes, episode
range, protocol, uncertainty status, and all required outcome metrics.

New scenario generation forces the Java task-prediction seam to its neutral
gateway. It cannot auto-discover or reuse the old `predicted_risk.csv` for a
new trace. An aligned risk result must attest to the same trace hash; otherwise
it is recorded as unavailable rather than zero-imputed.

## Aggregation and plots

Aggregation groups only independent trial summaries and reports mean, sample
standard deviation, sample count, standard error, and two-sided 95% Student-t
interval. It emits no interval for `n=1`.

Data-driven plot commands cover latency/energy/success vs device count,
success vs physical bandwidth, response time vs priority, CPU/RAM/bandwidth
utilization vs task count, reward vs episode, actor/critic loss vs update,
latency/physical-energy/HSI/reward histograms, and a Sprint 9 comparison.
They save PNG, SVG, PDF, and a provenance sidecar. The uncertainty-gate chart
is implemented as an explicit `NOT AVAILABLE` rejection until an input
attests to a genuine uncertainty estimator and gate definition.

`evaluate_sprint9_matched_mappo.py` is the missing full-MAPPO, 20-start,
inference-only evaluator. Its output carries the held-out-starts hash and is
the only intended route to an aligned Sprint 9 comparison.

## Validation and limitations

The validator checks schemas, duplicate trial-policy records, deterministic
seed provenance, scenario consistency, hash shape/file matches, valid metric
ranges, absence of zero-imputed unavailable risk, trace-specific OOF evidence,
checkpoint geometry, contamination across scenarios, and valid CIs.

The current simulator's physical network layer is the Digital Twin
`NetworkLink` model; it has no separate CloudSim packet-network topology.
Sprint 11 therefore changes its nominal link capacity, not Python's
`link_bw_norm_mbps`. Host-scale MAPPO requires a separately trained
scale-specific model; no training is automated here. Task scaling requires a
new Java trace, and MAPPO evaluation remains blocked until trace-aligned OOF
is deliberately regenerated. The reserved-zero uncertainty field is not an
estimator.

## Verification performed

`python -m unittest training.sprint11.tests.test_sprint11 -v`: 11 passed.
`python -m compileall -q training/sprint11 training/validate_sprint11.py`:
passed. `python training/validate_sprint11.py`: 11 manifests valid, 0 trial
results, 0 aggregate rows. `mvn -q -DskipTests compile` in `simulation/`:
passed. No simulation was run.
