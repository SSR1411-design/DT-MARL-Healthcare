# Sprint 10 A1 — No Digital Twin Decision Context

This inference-only counterfactual retains the frozen greedy R2 MAPPO policy, OOF failure prediction, criticality, reserved-zero uncertainty contract, task/failure generation, and all trace-backed environment physics. Trace exports remain evidence; they are not restored to MAPPO observations or edge destination ranking.

R2 mechanism found: DTMarlEnv._observations places the current trace's 12 local telemetry fields at indices 0-11 and each neighbour's observed availability, CPU, and link-latency fields at indices 24/26/27 plus five per neighbour; DTMarlEnv._candidate also passes trace-derived link latency to DestinationSelector. Trace active remains availability physics/feasibility, trace energy remains the recorded reward outcome, and trace symptoms are migration-audit classification rather than policy input.

A1 change: A1 hard-zeros only the direct trace telemetry observation slots and supplies a constant zero edge link-latency term to destination ranking. It preserves the 48-D geometry, OOF risk, capacity/load state, and recorded active availability as an environment-physics feasibility constraint; it does not replace either path with a heuristic.

Frozen R2 MAPPO checkpoint SHA-256: `F7B06FF8AE3D9C3288004101A8329F203D09FC5E9A82D88957DAF696C24A8BEE`
