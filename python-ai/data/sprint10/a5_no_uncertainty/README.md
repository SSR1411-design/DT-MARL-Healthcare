# Sprint 10 A5 — No Uncertainty Gating

This inference-only A5 evaluation retains the frozen greedy R2 MAPPO policy, Digital Twin trace, OOF failure prediction, task/failure generation, environment physics, and criticality behavior.

R2 mechanism found: RiskProvider.uncertainty is an identically-zero reserved matrix; DTMarlEnv._observations exposes it only at local observation index 13; no reward, action-mask, destination-selection, threshold, or decision branch consumes it.

A5 change: A5 overrides uncertainty_at to return 0.0 without reading the provider; the reserved index remains in the 48-D checkpoint-compatible observation.

Frozen R2 MAPPO checkpoint SHA-256: `F7B06FF8AE3D9C3288004101A8329F203D09FC5E9A82D88957DAF696C24A8BEE`
