| finding | evidence | status |
|---|---|---|
| OOF risk signal is leakage-safe but uncalibrated | failure_dataset.py safeguards; mappo_config.json risk metadata | SUPPORTED |
| risk-threshold@0.18 outperforms Sprint 6 MAPPO on held-out protection | mappo_eval.json: 30.25 vs 3.25 protected | SUPPORTED |
| A0 reproduces Sprint 6 recorded behavior | A0 and original history/update artifacts byte-identical; matching held-out histogram | SUPPORTED |
| Sprint 6.5 improvement in protection is risk targeting | zero-risk ablations retain most protection while migration volume differs | RULED OUT |
| R2 fixes the full behavioral outcome | R2 improves diagnostic sign error but held-out task performance is -0.46 | RULED OUT |
| Net Delta_EDGE growth is high-risk acquisition | Phase 5: p_hi ends below initialization; low-risk suppression dominates net growth | RULED OUT |
| u042-u062 contains a high-risk acquisition phase | Phase 5 / final audit: p_hi rises on both fixed sets during this interval | SUPPORTED |
| Low-risk dilution sets the update direction | Rung 3: cos(g_full, g_lo) +0.9099 to +0.9994 in 9/9 cells | SUPPORTED |
| Softmax saturation is the primary causal mechanism | Saturation follows Delta_EDGE timing and lower saturation in R3 did not improve behavior | PARTIALLY SUPPORTED / DOWNGRADED |
| Why R2 exceeds R3 | Final synthesis audit explicitly retains no causal account | UNRESOLVED |
| Multi-seed generalization | Only one retained seed per arm for the central ladder | NOT MEASURED |
| Sprints 8-12 results | No completed artifacts | NOT MEASURED |
| FLOPs or execution-complexity result | No completed artifact supplies a FLOP measurement | NOT MEASURED |
