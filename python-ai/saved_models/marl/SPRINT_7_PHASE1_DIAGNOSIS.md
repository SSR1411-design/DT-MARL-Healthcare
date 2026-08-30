# Sprint 7 — Phase 1: diagnosis and proposed experiment ladder

Status: **inspection only. No code modified, no training run, no artifact
overwritten.** Sprint 6/6.5 artifacts untouched.

Sources inspected: `SPRINT_6_5_REPORT.md`, `marl/{train,mappo,env,config,
evaluate,_diag_sensitivity,_diag_counterfactual}.py`,
`mappo_A0_cpu_repro_*`, `mappo_A1_cpu_bugfix_*`, `mappo_A2_crit_sign_*`,
`mappo_A3_entropy_*`, `diag_sensitivity.json`, `diag_counterfactual_A1.json`,
`failure_predictor_oof.npz`.

---

## 1. Diagnosis

Sprint 6.5's conclusion — "learning dynamics, not reward, not capacity" — holds.
But the mechanism it names ("the PPO update goes to zero") is the **symptom**.
The measurements already in the repo locate the cause one level down:

> **PPO is converging correctly onto an advantage estimate whose action
> ordering at high risk is inverted relative to the truth.** The dead gradient
> at update 60+ is what convergence looks like. Not a stalled optimiser — a
> converged one, pointed the wrong way.

Side-by-side, at `risk > 0.18`, from artifacts already on disk:

| action at high risk | true Δteam (exact replay, n=30) | policy's own normalised GAE (n) | ordering |
|---|---|---|---|
| `MIGRATE_TO_NEIGHBOR_EDGE` | **+2.671 ± 0.806** | **−0.266** (n=4) | inverted |
| `STAY` | 0 (reference) | +0.044 (n=60) | — |
| `MIGRATE_TO_CLOUD` | **−0.050 ± 1.5e−8** | **+0.080** (n=9) | inverted |

Truth: `EDGE (+2.67) > STAY (0) > CLOUD (−0.05)`.
Estimator: `CLOUD (+0.080) > STAY (+0.044) > EDGE (−0.266)`.

The learned policy is the exact argmax of the second row. Greedy A0 plays
`MIGRATE_TO_CLOUD` in **842 of the 855 decision states where cloud is legal
(98.5%)**, and `MIGRATE_TO_NEIGHBOR_EDGE` in 77 of 10,751 (0.7%). It has
faithfully learned "cloud whenever legal, else stay" — a rule with no risk term
in it, built on a −0.050 pseudo-no-op.

Caveat stated up front: the high-risk GAE cells rest on **n=4 and n=9**. The
inversion is a point estimate, not a significant result. Establishing or
refuting it is rung 0 of the ladder, because everything downstream depends on it.

### Why the estimator is wrong

Three mechanisms, each measurable, in descending order of my confidence:

1. **The payoff lands at or beyond the GAE horizon.** `p9_rows` records
   `horizon: 363–373` steps between the migration decision and its resolution.
   `γλ = 0.999 × 0.995 = 0.99400` → effective GAE window `1/(1−γλ) ≈ 167`
   steps. So **more than half the payoff reaches the advantage only through
   `V(s_{t+1}) − V(s_t)`**, i.e. entirely through the critic, never through
   observed reward.
2. **The critic's residual is the same size as the effect.** `explained_var`
   plateaus at 0.82–0.86, so 14–18% of return variance is unexplained. Against
   the observed `adv_std ≈ 3.15`, that puts the residual at order 1.2–1.4 in raw
   reward units — roughly **half of the +2.671 signal we need it to resolve**.
   Single-sample SNR ≈ 2; several clean samples per update are needed.
   (Order-of-magnitude inference from `explained_var` and `adv_std`, not a direct
   measurement. D2 measures it.)
3. **The policy supplies 3–19 such samples per update, falling.** Per update:
   3200 steps × 10 agents = 32,000 entries; `decision_frac` 0.039 → 0.108 gives
   1,250–3,500 decision entries; ×4.2% high-risk = 52–148; × p(EDGE) ≈ 0.05–0.13
   = **3–19 samples**, i.e. 0.2–1.2 per minibatch, each diluted by `1/denom` over
   ~300–860 co-entries. Over the entire 600-episode run that is order 500–1000
   noisy samples — **fewer than the 329 clean labels the BC probe needed**, and
   they are returns, not labels.

Meanwhile the opposing evidence is clean and plentiful: at low risk (95.8% of
states) `MIGRATE_EDGE` is genuinely bad (true −1.337, GAE −0.538, n=159). PPO
learns the 95.8% majority rule fast and at high SNR, drives `p(EDGE)` down
globally, and thereby destroys the sampling it would need to discover the 4.2%
exception. **Exploration starvation is downstream of a correctly-learned
majority rule.**

### Confounds that then finish the job

* **LR anneal.** `lr_scale` = 0.213 at update 60, 0.0133 at update 75. But
  `clip_frac` already hits 0.0053 at update 40 (`lr_scale` 0.48) and 0.0000 at
  update 60. Entropy collapse (0.72 → 0.40 by update 40) **leads** the anneal.
  The anneal is an accessory, not the primary cause. Item B will likely be a
  null result — worth running to say so.
* **75 updates total** (1200 actor steps, but only 75 independent batches).
* **`approx_kl` is the high-variance k1 estimator** (`mappo.py:383`,
  `mean(old_lp − new_lp)`). Valid in expectation but noisy and sample-negative
  (it logs −0.000033 at update 75). `clip_frac = 0.0000` is the trustworthy
  signal. Add the k3 form `(r−1) − log r` — nonneg, low variance — as a free
  diagnostic upgrade.

### Three facts that reshape the plan

1. **Risk is effectively binary, not continuous.** From the OOF scores
   (n=14,910): 95.6% below 0.10, 4.05% above 0.50, and only **0.37% in the whole
   band (0.10, 0.50)**. `p5` confirms it downstream: the `0.18–0.50` advantage
   bucket contains **zero** decision states. The target function is a single
   threshold on a near-binary feature — which is exactly why BC hits accuracy
   1.0. Consequence: the current sweep grid `[0, 0.1, 0.25, 0.5, 0.75, 0.9,
   0.99]` spends its middle **off-distribution**. Endpoints are fine; the span
   number survives, but a real-state conditional table must become the primary
   metric.
2. **The good action is never blocked and never refused; the decoy is.**
   `MIGRATE_EDGE` is legal in **100%** of decision states and refused **0/73**
   times. `MIGRATE_TO_CLOUD` is legal in **7.95%** and refused **545/710
   (76.8%)**, every refusal contention-caused. 92.05% of decision states have
   exactly two legal actions — `STAY` vs `MIGRATE_EDGE`, precisely the choice
   risk should drive. So Sprint 7 item F is **not** "expose feasibility": the
   mask already gates cloud 92% of the time and `cloud_free_fraction` is already
   in the observation (idx 43, gradient 0.120 — the largest in the cloud row).
   Cloud's real role is as a **decoy the estimator overrates**.
3. **Risk has almost no authority over the margin it must control.** Mean
   `|∂(logit_EDGE − logit_STAY)/∂risk| × sd(risk) = 0.00698` logits — rank
   **9/48** features, against a total available swing of 0.212 across all 48.
   Flipping `p(EDGE)` from 0.10 to 0.60 needs ≈ +2.6 logits. Neighbour-liveness
   channels (idx 39, 32, 34, 29) outrank risk. BC reaching span 0.394 means its
   risk swing is ~2 orders of magnitude larger.

Also noted: **5 of 48 observation channels are constants** (sd < 1e−4):
`uncertainty` (13, reserved-and-still-zero), `task_present` (15), `isRunning`
(21), `cloud_lat` (44), `cloud_risk0` (45). `PREEMPTIVE_REROUTE` is legal in
0.083% of decision states — a dead action, consistent with `isRunning` being
pinned. And `mappo.py:327` re-seeds `np.random.default_rng(0)` inside every
`update()`, so all 75 updates reuse one minibatch partition sequence.

### The most damning single number

A0's **zero-risk ablation scores +23.13 held-out reward against +20.62 with the
risk channel connected**. The policy is very slightly *better* with the
prediction removed. There is no risk-conditioned behaviour to preserve.

Conditional action rates on real held-out states (from `risk_action_table`) —
this should be Sprint 7's primary metric, not the synthetic sweep:

| arm | P(EDGE \| risk<0.1) | P(EDGE \| risk>0.5) | lift | P(relocate \| <0.1) | P(relocate \| >0.5) |
|---|---|---|---|---|---|
| A0 | 0.60% (61/10200) | 2.36% (15/636) | 3.9× | 8.42% | 9.28% |
| A1 | 0.056% (7/12587) | 0.40% (3/754) | 7.1× | 8.09% | **6.76%** (inverted) |

Relative lift looks encouraging and is meaningless: the absolute change is
1.8 pp (A0) and 0.34 pp (A1), and A1's total relocation rate is *lower* at high
risk than at low risk.

---

## 2. What follows for the plan

The brief's priority order needs one reordering, on evidence:

* **Items A and B (more updates, gentler anneal) are predicted null or
  harmful.** If the advantage ordering is inverted, more updates and a larger
  late-training LR let PPO converge *harder* onto "cloud whenever legal". Run
  them — cheaply, early — precisely to record that.
* **Item F is mis-specified but points at the right object.** Not "expose
  feasibility" (already exposed, already masked). The live question is whether
  the cloud decoy is *consuming the policy*, testable by removing it.
* **Items C, D, E are on the critical path**, with D (exposure) and E (BC init)
  the two highest-information arms.
* One item the brief does not list belongs here: **γ/λ vs. the 363–373-step
  payoff horizon**. Retuning `gae_lambda` is a credit-assignment change, not a
  reward change, so it is inside the brief's constraint.

---

## 3. Proposed ladder

Gates between rungs, so no campaign launches on a hunch. Seed **20260818**,
`--device cpu`, 600 episodes, everything else at A0/A1 defaults unless the arm
names the change. One knob per arm.

### Rung 0 — measurement only, no training (~10 min each)

| id | measures | kills / confirms |
|---|---|---|
| **D1** | true Δteam **and** the learner's GAE advantage on the *same* (state, action) pairs; ≥150 high-risk deviations (up from 30). Correlation + per-bucket sign agreement. | The inversion. **The load-bearing measurement of Sprint 7.** |
| **D2** | `std(ret − V)` restricted to high-risk decision states, against +2.671. | Whether the effect is inside the critic's noise floor. If yes, extra updates cannot help and rung 1 is pre-refuted. |
| **D3** | per-update census of (high-risk ∧ EDGE-sampled) entries, un-instrumented rollout. | The 3–19/update estimate. |
| **D4** | add k3 KL + actor grad-norm + per-bucket adv counts to `_updates.csv`. | Item C directly: gradient small, or large-but-cancelling? |

**Gate:** if D1 shows the ordering is inverted with n≥150 → rung 1 is a
formality; rung 2/3 carry the sprint. If D1 shows the ordering is *correct* and
merely noisy → rung 1 becomes the main event and my diagnosis is wrong. Either
way it is written down before the arms run.

### Rung 1 — the brief's items A and B (~25–40 min each)

| id | change | brief item |
|---|---|---|
| **B1** | `rollout_episodes=8 → 2` (600 eps → 300 updates, same data) | A |
| **B2** | `anneal_lr=False` | B |
| **B3** | B1 + B2 | A+B |

Pre-registered prediction: span stays < 0.05 in all three; B3 raises relocation
volume and `MIGRATE_TO_CLOUD` share without raising risk conditioning.

### Rung 2 — the arms I expect to matter (~25–35 min each)

| id | change | brief item | why |
|---|---|---|---|
| **C1** | mask out `MIGRATE_TO_CLOUD` entirely | F | Removes the overrated decoy; reduces ~100% of decision states to the binary STAY/EDGE choice risk should drive. **Diagnostic arm** — changes the action space, so reward is not comparable to A0; compare on span/corr/conditional rates and against a cloud-ablated `risk-threshold`. |
| **C2** | `cloud_slots 8 → 16` | F | The other branch: make cloud honest rather than absent. Own baselines; not reward-comparable to A0. |
| **C3** | episode starts sampled ∝ high-risk node-ticks in the window, **training window [9, 491] only** | D | Attacks sparsity directly. Must report the achieved high-risk decision fraction so the change is auditable, and evaluate on the **unbiased** held-out window. No future information reaches the policy; no eval-window leakage. |
| **C4** | BC-to-`risk-threshold@0.18` actor init → normal PPO fine-tune | E | Highest information in the ladder. Two questions: (i) does PPO **retain** span 0.394, or actively destroy it? Collapse back to ~0.01 is the strongest possible confirmation of the corrupted-advantage diagnosis. (ii) does reward improve? Reported as **warm-started**; the teacher is a threshold rule, so the result can never be presented as risk sensitivity learned from scratch. |

### Rung 3 — variance reduction, only if D1/D2 indict the advantage

| id | change | why |
|---|---|---|
| **E1** | `gae_lambda 0.995 → 1.0`, and/or `γ 0.999 → 0.9995` | Payoff lands at 363–373 steps; current GAE window is 167. Credit assignment, not reward. |
| **E2** | stratified minibatches so high-risk decision entries are not diluted ~1:20 | Attacks `1/denom` dilution. Report the reweighting explicitly. |
| **E3** | 1800 episodes, everything else A1 | The "was it just too short" control. |

### Promotion gate (all rungs)

An arm is promoted only if, on the **held-out** window, it clears **both**:

1. sweep span ≥ **0.05** (≈3× A1's 0.016; still 8× below the BC ceiling), and
2. `P(EDGE | risk>0.5) − P(EDGE | risk<0.1)` ≥ **5 pp absolute** (A0: 1.8 pp).

Reward is not a gate. Anything promoted is then replicated over **5 seeds fixed
and stated in advance** — 20260818, 20260819, 20260820, 20260821, 20260822 —
before any ordering is believed.

### Metric changes needed regardless

* Sweep grid → include the actual modes: `[0.02, 0.05, 0.09, 0.20, 0.60, 0.85,
  0.95]`; report in-distribution span separately from the legacy `[0 … 0.99]`
  grid so Sprint 6.5 comparisons survive.
* Promote the **real-state conditional action table** to primary; demote the
  synthetic sweep to secondary.
* Report the zero-risk ablation **delta** as a headline number, not a footnote.
  A0's is +2.51 in the wrong direction.

---

## 4. Code changes Phase 2 will need (none made yet)

New CLI flags on `marl/train.py` (all default-off, so A0/A1 stay bitwise
reproducible): `--no-anneal-lr`, `--cloud-slots`, `--disable-cloud-action`,
`--highrisk-start-bias`, `--bc-init`, `--gamma`, `--gae-lambda`,
`--stratify-minibatches`. `--rollout-episodes` already exists. Additional
`_updates.csv` columns (k3 KL, actor grad-norm, high-risk sample count) are
append-only.

Artifact naming: `mappo_S7_<rung><n>_<slug>_seed<seed>`, e.g.
`mappo_S7_C4_bcinit_seed20260818`. No Sprint 6/6.5 file is written.

## 5. Cost

Rung 0 ≈ 40 min total. Rung 1 ≈ 1.6 h. Rung 2 ≈ 2 h. Rung 3 ≈ 1.5 h (E3 alone
~75 min). Full ladder ≈ 6 h CPU, sequential, plus ~2 min eval per arm. The
5-seed replication of a single promoted arm adds ~2 h.

## 6. Honest position going in

The evidence says the reward is right, the network is capable, and the estimator
that connects them is not. If D1 confirms the inversion, then no amount of
optimiser tuning fixes this and rung 1 will produce a clean null — which is a
publishable result about MAPPO on sparse-payoff, long-horizon infrastructure
control, not a failure to report. If C4 shows PPO actively erasing a
behaviour-cloned risk sensitivity it was handed for free, that is the strongest
evidence available that the learning signal, not the architecture and not the
reward, is what is broken.
