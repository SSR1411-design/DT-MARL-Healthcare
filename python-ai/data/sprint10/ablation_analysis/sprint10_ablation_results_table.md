# Sprint 10 Ablation Results Table

Values are copied exactly from `ablation_results_means.csv`. R2 is a
one-episode legacy descriptive reference, not an aligned eight-episode
baseline. `N/A` denotes a field unavailable in the R2 source schema.

| Experiment | Component removed | Episode count | Episode reward | Task success rate | Completed | Lost | Critical success rate | SLA violations | Tasks protected before failure | Infeasible actions |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R2 baseline | None (legacy descriptive reference) | 1 | -10.63804233702831 | 0.69999999999999996 | 28 | 12 | N/A | 1 | 8 | N/A |
| A1 No Digital Twin | Digital Twin decision context | 8 | 1.0381176881419378 | 0.72812500000000002 | 29.125 | 10.125 | 0.7421875 | 1.375 | 9.75 | 28.125 |
| A2 No Failure Prediction | Failure prediction | 8 | -2.9247689616749994 | 0.71250000000000002 | 28.5 | 11.25 | 0.75 | 0.5 | 8.125 | 44.125 |
| A3 Single-Agent PPO / No MARL | Multi-agent action selection | 8 | -12.687471207747876 | 0.67500000000000004 | 27 | 12.625 | 0.71875 | 1 | 5.875 | 7.875 |
| A4 No Criticality | Clinical criticality awareness | 8 | -16.537981626992405 | 0.734375 | 29.375 | 9.5 | 0.6796875 | 1 | 10.875 | 37.875 |
| A5 No Uncertainty | Uncertainty interface | 8 | -0.45835254394478397 | 0.72812500000000002 | 29.125 | 10.25 | 0.71875 | 1.5 | 9.875 | 38.375 |
