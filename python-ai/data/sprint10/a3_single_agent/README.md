# Sprint 10 A3 — Single-Agent PPO Ablation

This bundle contains real, inference-only evaluation traces for the eight fixed R2 held-out starts. It uses the validated Sprint 9 centralized single-agent PPO checkpoint for action selection; MAPPO is not loaded or called for actions.

The R2 Digital Twin trace, OOF failure prediction, destination-risk weighting, criticality-aware reward, task generation, and environment dynamics are retained. R2 uncertainty remains its documented reserved, zero-valued observation channel; there is no active uncertainty-gating branch to add or remove.

Single-agent checkpoint SHA-256: `223BF33C14CEF2F5EBA93C5F3697E8DDB880E4C56E5A348D82656AF4F1750CB5`

Frozen R2 MAPPO checkpoint SHA-256: `F7B06FF8AE3D9C3288004101A8329F203D09FC5E9A82D88957DAF696C24A8BEE`
