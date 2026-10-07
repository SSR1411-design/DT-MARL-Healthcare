# Sprint 10 Ablation Study

## 1. Experimental Design

Five controlled ablations were evaluated against the retained Sprint 10
evaluation protocol:

- A1 — No Digital Twin
- A2 — No Failure Prediction
- A3 — Single-Agent PPO / No MARL
- A4 — No Criticality
- A5 — No Uncertainty

Each ablation used the same eight validated held-out episode starts: 491, 521,
550, 580, 609, 639, 668, and 698. The completed evaluations were
inference-only and used frozen models and configurations; this document derives
results solely from the completed CSV analysis.

## 2. Baseline and Comparison Protocol

The available R2 baseline artifact,
`data/sprint8_episode_metrics.csv`, contains only **one** legacy-schema
episode. It is therefore included only as descriptive reference evidence, not
as an aligned eight-episode baseline.

- No R2 episode rows were fabricated or aligned to the eight held-out starts.
- No improvement relative to R2 is calculated or claimed.
- A1-A5 values are arithmetic means across their respective eight held-out
  episodes.

Consequently, cross-ablation patterns among A1-A5 are the primary descriptive
results in this document. The R2 row provides context only.

## 3. Main Ablation Results

The values below are copied exactly from
`ablation_results_means.csv`; no additional rounding or transformation was
applied. `N/A` denotes a metric unavailable in the one-episode R2 legacy
schema, not a zero-imputed value.

| Experiment | Episodes | Episode Reward | Task Success Rate | Completed | Lost | Unfinished | Average Task Latency (s) | SLA Violations | Tasks Protected Before Failure | Critical Success Rate | Infeasible Actions | Migrations | Reroutes | Energy Reward Cost |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2 baseline (descriptive legacy reference) | 1 | -10.63804233702831 | 0.69999999999999996 | 28 | 12 | 0 | 282.61000000000001 | 1 | 8 | N/A | N/A | 59 | 0 | 1904.1063474365285 |
| A1 — No Digital Twin | 8 | 1.0381176881419378 | 0.72812500000000002 | 29.125 | 10.125 | 0.75 | 283.68945649781358 | 1.375 | 9.75 | 0.7421875 | 28.125 | 60.625 | 0 | 2043.9505408477778 |
| A2 — No Failure Prediction | 8 | -2.9247689616749994 | 0.71250000000000002 | 28.5 | 11.25 | 0.25 | 279.44608750684188 | 0.5 | 8.125 | 0.75 | 44.125 | 60.125 | 0 | 1945.084554988862 |
| A3 — Single-Agent PPO / No MARL | 8 | -12.687471207747876 | 0.67500000000000004 | 27 | 12.625 | 0.375 | 285.09319970527559 | 1 | 5.875 | 0.71875 | 7.875 | 60.625 | 1 | 2030.245613343051 |
| A4 — No Criticality | 8 | -16.537981626992405 | 0.734375 | 29.375 | 9.5 | 1.125 | 283.9576975918643 | 1 | 10.875 | 0.6796875 | 37.875 | 60.25 | 0 | 2054.790086184692 |
| A5 — No Uncertainty | 8 | -0.45835254394478397 | 0.72812500000000002 | 29.125 | 10.25 | 0.625 | 284.53722966269845 | 1.5 | 9.875 | 0.71875 | 38.375 | 60 | 0 | 2027.7386732948301 |

`mean_recovery_latency_s` is `NaN` in the Sprint 10 source analysis and is not
one of the requested paper-table metrics; it has not been recoded as zero.

## 4. A1 — No Digital Twin

A1 produced the highest mean episode reward among the five ablations
(1.0381176881419378), with a task success rate of 0.72812500000000002 and
29.125 completed tasks. Its mean latency was 283.68945649781358 s, and it
recorded 28.125 infeasible actions. Within this evaluated set, these outcomes
indicate that removing Digital Twin decision context did not uniformly depress
every reported aggregate outcome. Because no aligned eight-episode full-system
R2 control is available, this result does not establish the size or direction
of a Digital Twin contribution relative to the full system.

## 5. A2 — No Failure Prediction

A2 had a mean episode reward of -2.9247689616749994, a task success rate of
0.71250000000000002, and 28.5 completed tasks. It had the highest mean count
of infeasible actions among the ablations (44.125), while recording the lowest
mean SLA-violation count (0.5). These observations suggest a trade-off in this
evaluated setting when failure prediction was removed: the reported reward and
task outcomes differed alongside feasibility and SLA counts. They do not by
themselves isolate a universal effect of prediction.

## 6. A3 — Single-Agent PPO

A3 produced a mean episode reward of -12.687471207747876, the lowest mean task
success rate (0.67500000000000004), the fewest completed tasks (27), and the
most lost tasks (12.625) among the five ablations. It also had the smallest mean
infeasible-action count (7.875) and the only nonzero mean reroute count (1).
Within the evaluated setting, the results indicate that the single-agent PPO
variant exhibited a different operational trade-off from the multi-agent
ablations, with weaker aggregate task-outcome measures despite fewer infeasible
actions. This is descriptive evidence, not an inferential test of MARL across
other workloads or protocols.

## 7. A4 — No Criticality

A4 had the lowest mean episode reward (-16.537981626992405), while also
showing the highest mean task success rate (0.734375) and completed-task count
(29.375), as well as the fewest lost tasks (9.5). Its critical success rate was
the lowest of the five ablations (0.6796875), and it had the most unfinished
tasks (1.125). These jointly observed measures indicate that removing
criticality changed the balance among the evaluated reward, completion, and
critical-task outcomes. They do not support a component-necessity claim beyond
this setting.

## 8. A5 — No Uncertainty

A5 yielded a mean episode reward of -0.45835254394478397, a task success rate
of 0.72812500000000002, 29.125 completed tasks, and 10.25 lost tasks. Its
critical success rate was 0.71875 and it recorded 38.375 infeasible actions.
The completed Sprint 10 analysis records `uncertainty_used` as 0 for every
ablation row, including A5. Therefore, the available aggregate results suggest
an observed outcome for this interface ablation, but they cannot establish the
effect of actively used uncertainty gating within the evaluated exports.

## 9. Overall Findings

Across A1-A5, the results indicate heterogeneous trade-offs rather than a
single ordering on every metric. A1 had the highest mean reward in the ablation
set; A3 had the weakest mean task success and completion outcomes; and A4
combined the highest completion and task-success means with the lowest reward
and critical-success means. A2 paired the highest mean infeasible-action count
with the lowest SLA-violation mean. These patterns are useful controlled
ablation observations within the evaluated setting, but no inferential
statistical testing was performed and no claim of universal component necessity
or general causal effect is made.

## 10. Reproducibility / Evidence

This document was generated by reading the validated analysis artifacts only:

- `data/sprint10/ablation_analysis/ablation_results_means.csv`
- `data/sprint10/ablation_analysis/ablation_results_comparable.csv`
- `data/sprint10/ablation_analysis/ablation_results_metadata.json`

The analysis metadata identifies the underlying source CSVs:

| Experiment | Source CSV |
| --- | --- |
| R2 baseline | `data/sprint8_episode_metrics.csv` |
| A1 — No Digital Twin | `data/sprint10/a1_no_digital_twin/sprint10_a1_episode_metrics.csv` |
| A2 — No Failure Prediction | `data/sprint10/a2_no_failure_prediction_rerun/sprint10_a2_episode_metrics.csv` |
| A3 — Single-Agent PPO / No MARL | `data/sprint10/a3_single_agent/sprint10_a3_episode_metrics.csv` |
| A4 — No Criticality | `data/sprint10/a4_no_criticality/sprint10_a4_episode_metrics.csv` |
| A5 — No Uncertainty | `data/sprint10/a5_no_uncertainty/sprint10_a5_episode_metrics.csv` |

The documentation workflow was CSV-only. It did not rerun evaluations, invoke
model inference, train models, write checkpoints, or regenerate any experiment
output.
