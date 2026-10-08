# Sprint 12 — Computational Complexity Analysis

## Scope and provenance

This is a checkpoint-based analysis of the validated R2 MAPPO policy and the independently trained Sprint 9 single-agent PPO baseline. It did **not** retrain MAPPO, run the Sprint 11 matrix, execute the simulator, alter prior-sprint data, or overwrite a checkpoint. FLOPs are analytical dense-layer counts; timing is a new isolated CPU-only neural benchmark.

- Generated: `2026-10-08T03:27:13.582577+00:00`
- MAPPO checkpoint: `python-ai/saved_models/marl/mappo_R2_mc_target_best.pth` (`f7b06ff8ae3d9c3288004101a8329f203d09fc5e9a82d88957daf696c24a8bee`, 5,208,669 B (4.967 MiB))
- Single-agent PPO checkpoint: `python-ai/saved_models/single_agent/sprint9/sprint9_single_agent_best.pth` (`223bf33c14cef2f5eba93c5f3697e8ddb880e4c56e5a348d82656af4f1750cb5`, 4,732,703 B (4.513 MiB))
- All source checkpoint hashes, sizes, and exact paths are included in `sprint12_complexity_results.json` under `checkpoint_inventory`.

## Phase 1 — read-only audit

### A. Model, evaluation, and timing source files

| Purpose | Exact file(s) |
|---|---|
| MAPPO actor/critic, rollout, save/load | `python-ai/marl/mappo.py` (`MultiAgentActor`, `CentralisedCritic`, `MAPPO`) |
| MAPPO architecture/training configuration | `python-ai/marl/config.py` (`MappoConfig`, `TrainConfig`) |
| MAPPO training | `python-ai/marl/train.py` |
| MAPPO evaluation and older reference policies | `python-ai/marl/evaluate.py` |
| Rule policies | `python-ai/marl/baseline.py` |
| Sprint 9 baseline factory/evaluation | `python-ai/marl/sprint9_benchmarks.py` |
| Independent centralized PPO | `python-ai/marl/single_agent_ppo.py` |
| Single-agent PPO training | `python-ai/marl/train_single_agent.py` |
| Sprint 9 baseline protocol | `docs/SPRINT_9_BASELINES.md` |
| Declared Python dependencies | `requirements.txt` |
| Existing MAPPO timing evidence | `python-ai/run_R2_mc_target_train.log`; `python-ai/saved_models/marl/mappo_R2_mc_target_config.json` |
| Existing single-agent timing evidence | `python-ai/saved_models/single_agent/sprint9/sprint9_single_agent_config.json` |

`python-ai/models/transformer.py`, `bilstm.py`, `failure_predictor.py`, and `htcf.py` define upstream failure-prediction models. They are not scheduling-policy baselines and are outside the MAPPO scheduling-complexity comparison.

### B. Checkpoints

The full exact inventory of every `.pth`, `.pt`, and `.ckpt` file under `python-ai/saved_models/` is in `sprint12_checkpoint_inventory.csv` (path, bytes, SHA-256), and is repeated in `sprint12_complexity_results.json`. The two scheduling checkpoints used for numerical analysis are listed above. This avoids ambiguity between the validated R2 checkpoint and other historical / trajectory checkpoints.

### C–E. What existing artifacts support, and limitations

| Checklist item | Existing-artifact status | Sprint 12 action / limitation |
|---|---|---|
| Parameter count | Directly available from source/checkpoint | Computed from loaded trainable tensors. |
| Model size | Directly available | Observed serialized bytes and derived raw float32 parameter memory are separated. |
| FLOPs | No prior artifact | Analytically derived from actual dense layers; not presented as runtime. |
| Training time | Directly available | Historical wall times come from saved R2 and single-agent configuration/log artifacts. |
| Inference time | No prior artifact | New fixed-input, CPU-only benchmark; excludes simulator/runtime physics. |
| Baseline comparison | Mixed | Learned single-agent baseline is measured; rule baselines legitimately have no parameters/checkpoints and no fabricated FLOP/timing values. |

Scientific limitations: timing is hardware, PyTorch version, CPU-thread, and implementation dependent; it is not a simulation-step or end-to-end latency. Tanh/masking/argmax/softmax operations are not included in analytical dense FLOPs. Historical MAPPO training says `device=cpu`; historical single-agent training says `device=cuda`, but neither artifact records the exact training CPU/GPU model.
No pre-existing FLOP profiler, FLOP measurement, or isolated neural-inference timing utility was found in the repository; the Sprint 12 script is therefore additive.

## MAPPO architecture and parameters

- Actors: 10 **independent** MLPs, each `48 → 128 → 128 → 4`, with Tanh after hidden layers. There is no actor parameter sharing.
- Critic: one shared centralized MLP, `(489 global-state + 10 agent-ID) → 256 → 256 → 1`, evaluated once for each agent ID; `forward(state)` returns 10 values.

| Component | Trainable parameters | Raw parameter memory | Dtype |
|---|---|---|---|
| One actor | 23,300 | 93,200 B (0.089 MiB) | float32 |
| All 10 independent actors | 233,000 | 932,000 B (0.889 MiB) | float32 |
| Centralized critic | 194,049 | 776,196 B (0.740 MiB) | float32 |
| Total trainable | 427,049 | 1,708,196 B (1.629 MiB) | float32 |

The critic’s `eye` one-hot buffer (100 float32 elements / 400 B) is explicitly excluded from the trainable count. Optimizer state is also excluded.

## Model size

| Model | Serialized checkpoint | Raw trainable parameters | Deployment actor-only raw parameters | Meaning |
|---|---|---|---|---|
| MAPPO R2 | 5,208,669 B (4.967 MiB) | 1,708,196 B (1.629 MiB) | 932,000 B (0.889 MiB) | `.pth` includes optimizer + metadata; raw is weights only |
| Single-agent PPO | 4,732,703 B (4.513 MiB) | 1,572,004 B (1.499 MiB) | 806,048 B (0.769 MiB) | `.pth` includes optimizer + metadata; raw is weights only |

All analyzed trainable tensors are float32, so raw parameter memory is exactly 4 bytes × parameter count. Serialized file size must not be interpreted as deployment-only model weight memory.

## FLOPs (analytical)

Convention: dense linear MACs are reported alongside `2 × MACs + bias additions` FLOPs. This is an analytical operation count for one batch/state and excludes Tanh, action masking, softmax/categorical operations, memory traffic, and framework overhead.

| MAPPO computation | MACs | FLOPs incl. bias | Interpretation |
|---|---|---|---|
| Actor, one agent | 23,040 | 46,340 | one local-observation policy MLP |
| Actor, 10-agent decentralized decision | 230,400 | 463,400 | deployed policy computation |
| Critic, one agent value | 193,536 | 387,585 | one `(state, agent-ID)` row |
| Critic, `forward(state)` → 10 values | 1,935,360 | 3,875,850 | centralized learning/value evaluation |
| Actor + critic, one state | 2,165,760 | 4,339,250 | meaningful only when both are run |

| Single-agent PPO computation | MACs | FLOPs incl. bias |
|---|---|---|
| Actor, factored 10 × 4 output | 200,960 | 402,472 |
| Critic, one global value | 190,976 | 382,465 |
| Actor + critic, one state | 391,936 | 784,937 |

## Training time (existing evidence)

| Run | Duration | Work represented | Recorded device | Evidence |
|---|---|---|---|---|
| Validated R2 MAPPO | 1585.6 s | 600 episodes × 400 steps = 240,000 environment steps; 75 updates | cpu | `python-ai/saved_models/marl/mappo_R2_mc_target_config.json`; log rounds to 1,586 s |
| Single-agent PPO baseline | 632.218 s | 400 episodes × 400 steps = 160,000 environment steps; 50 updates | cuda | `python-ai/saved_models/single_agent/sprint9/sprint9_single_agent_config.json` |

These are observed historical run times, not scaling extrapolations. They include their respective training-loop/environment costs and must not be compared as pure neural compute benchmarks because device contexts differ.

## Isolated inference / decision time

Protocol: 25 batches × 400 calls after 500 warmups; CPU with one PyTorch thread; fixed float32 inputs at the real checkpoint dimensions. No environment, trace, I/O, or checkpoint loading is timed.

| Callable | Mean µs | SD µs | Median µs | P95 µs | What is timed |
|---|---|---|---|---|---|
| MAPPO actor forward | 759.661 | 13.474 | 755.933 | 772.015 | all 10 actor MLPs only |
| MAPPO greedy decision | 1033.038 | 58.612 | 1009.089 | 1125.215 | production-style `act_greedy`, including masks/argmax |
| MAPPO critic forward | 169.896 | 8.673 | 167.399 | 187.755 | all 10 centralized values |
| Single-agent actor forward | 76.222 | 1.894 | 75.784 | 79.454 | centralized 10 × 4 logits |
| Single-agent greedy decision | 137.937 | 5.325 | 136.945 | 142.150 | production-style `act_greedy`, including masks/argmax |
| Single-agent critic forward | 74.849 | 4.589 | 73.758 | 78.695 | one centralized value |

## Paper-ready complexity comparison

| Algorithm | Trainable parameters | Actor decision FLOPs | Serialized size | Training time | CPU inference mean | Complexity |
|---|---|---|---|---|---|---|
| MAPPO (validated R2) | 427049 | 463400 | 5208669 | 1585.6 | 1033.038 | Actor decision O(N_a*(d_o*h_1+h_1*h_2+h_2*A)); critic O(N_a*((d_s+N_a)*g_1+g_1*g_2+g_2)) |
| Single-agent PPO | 393001 | 402472 | 4732703 | 632.218 | 137.937 | O(d_s*h_1+h_1*h_2+h_2*(N_a*A)) |
| Reactive no-digital-twin | N/A | N/A | N/A | N/A | N/A | O(N_a*A) worst case for per-agent legal-action preference scan |
| Detection without prediction | N/A | N/A | N/A | N/A | N/A | Policy O(N_a); recovery transition cost depends on environment healthy-node search |
| Static no-migration | N/A | N/A | N/A | N/A | N/A | O(N_a) to allocate/emit one STAY action per agent |
| Random legal | N/A | N/A | N/A | N/A | N/A | O(N_a*A) to find legal actions and sample |
| Risk threshold | N/A | N/A | N/A | N/A | N/A | O(N_a*A) worst case |

`N/A` means no model, checkpoint, or scientifically appropriate isolated measurement exists; it is not a zero-valued experimental result. The full table with status/provenance columns is `sprint12_complexity_comparison.csv`.

## Reproduce

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = (Resolve-Path python-ai).Path
python python-ai/outputs/sprint12/complexity_analysis.py --overwrite
```

This reloads the two immutable checkpoints and regenerates only Sprint 12 files. Structural counts, hashes, checkpoint sizes, and analytical FLOPs should reproduce exactly; timing statistics are expected to vary modestly with host load.

## Validation and preservation

- The analysis script loads checkpoints with `map_location='cpu'`, calls `eval()`, and never calls a training/update method.
- SHA-256 fingerprints were recorded for every discovered checkpoint and specifically for the two analyzed scheduling checkpoints.
- Prior sprint directories were read only. The only new artifacts are in `python-ai/outputs/sprint12/`.
- No Sprint 11 command or experiment matrix was invoked.
