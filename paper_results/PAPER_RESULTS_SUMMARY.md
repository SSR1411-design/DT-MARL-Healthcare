# Publication Results Summary

## 1. Failure Prediction

- The retained pooled out-of-fold predictor result has PR-AUC 0.2863, ROC-AUC 0.8159, and F1 0.4730 at threshold 0.18 (n=14,910 windows; 320 positives).
- The lead-time figure uses only recorded `EARLY_WARNING` events from `host_lead_time.csv`; it reports its sample size directly.
- The OOF risk used by MAPPO is leakage-safe but uncalibrated. This is a predictor-performance result, not a calibrated clinical probability claim.

## 2. MAPPO Baseline

- The completed Sprint 6 MAPPO held-out result is reward 20.6205, protected 3.25, lost 11.250, and success rate 0.71875.
- `risk-threshold@0.18` records reward 76.4963, protected 30.25, and lost 3.375 at comparable relocation volume (35.875 versus 36.250).
- The Sprint 6 training plot preserves all 600 recorded rewards without smoothing. The known episode-600 log/CSV discrepancy remains documented in `SPRINT_6_REPORT.md` and is not reconciled here.

## 3. Sprint 6.5

- A0-A3 risk-sweep spans are A0=0.0036, A1=0.0159, A2=0.0039, A3=0.0023; the report-recorded behaviour-cloning ceiling is 0.3935.
- The physical-protection zero-risk ablation is plotted as paired observations, not as a causal reward comparison. For A0, protection is 3.250 with risk and 3.125 with risk forced to zero.
- The arm rewards are not compared across all arms because the reward conventions differ; the tables retain this limitation.

## 4. Sprint 7

- The R2 trajectory contains all 76 stored checkpoints (u000-u075), scored on frozen RANDOM (4,975 decision entries; 216 high-risk) and UNION (74,237; 3,592 high-risk) sets.
- `Δ_EDGE` is plotted with its two components. The final audit concludes that its net growth is mostly low-risk suppression, while preserving the documented u042-u062 high-risk acquisition phase.
- The stochastic mean-probability and greedy argmax channels are shown separately; they are not interchangeable.
- The aligned training-dynamics figure is temporal/diagnostic evidence only. It does not claim that clip fraction, entropy, explained variance, or approximate KL caused the behaviour change.

## 5. Complexity

- The training log documents 23,300 parameters per actor, 233,000 across 10 separate actors, and 194,049 in the centralised critic; 427,049 is their arithmetic sum.
- Checkpoint byte sizes are reported as file observations. FLOPs and execution-complexity results are not available and are not estimated.

## 6. Strongest Paper Findings

1. The leakage-safe OOF host-failure predictor achieved PR-AUC 0.2863 on 14,910 windows.
2. On held-out Sprint 6 starts, the threshold baseline protected 30.25 tasks versus MAPPO's 3.25 at nearly equal relocation volume.
3. The A0 control reproduces the retained Sprint 6 action histogram and has byte-identical history/update artifacts.
4. Sprint 6.5 risk-sweep spans remain far below the report-recorded 0.3935 behaviour-cloning ceiling.
5. Zero-risk ablations show that observed arm-level protection was not established as risk-targeted protection.
6. Across the complete R2 trajectory, `Δ_EDGE` must be interpreted through both `p_hi` and `p_lo`; net growth is mostly low-risk suppression.
7. The final audit records low-risk dilution as supported and retains the causal explanation for R2 > R3 as unresolved.

## 7. Claims We MUST NOT Make

- Multi-seed generalization: the principal arms retain one seed each.
- A causal claim from temporal precedence in the R2 trajectory.
- That net `Δ_EDGE` growth alone demonstrates high-risk acquisition or self-healing.
- That the later Sprint 7 mechanism was established by Sprint 6 alone.
- Reward comparisons across arms with different reward conventions.
- FLOPs, runtime-complexity, or execution-cost claims not recorded in completed artifacts.
- Any result for Sprints 8-12; no completed evidence exists for those stages.
- A reconciliation of the Sprint 6 episode-600 log/CSV discrepancy.
