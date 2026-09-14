# SPRINT 7 — FINAL SYNTHESIS AUDIT

**Date:** 2026-08-31 · **Type:** READ-ONLY audit · **Training runs performed:** 0 ·
**Production modules modified:** 0 · **Existing artifacts or checkpoints overwritten:** 0 ·
**New diagnostics written or run:** 0 · **New preregistrations:** 0 · **R4 designed:** no.

This report is the single deliverable. It reconstructs the complete scientific state of
Sprint 7 from the repository, classifies every hypothesis Sprint 7 actually investigated,
fixes the final interpretation of the self-healing finding, resolves the two remaining
questions as far as the existing evidence permits, records every superseded claim, and
returns a COMPLETE / NOT COMPLETE verdict.

Every number below was read from an artifact in this repository during this audit or in
the immediately preceding phases of the same audit. Where a number is inferred rather than
stored, it is labelled *derived*. Where the evidence does not support a claim, the claim is
not made.

---

## 1. Executive summary

**The central question** — *why does the MAPPO policy fail to learn and use risk-aware
migration behaviour, despite the environment exposing predicted failure risk and despite
offline diagnostics showing that useful high-risk migration signal exists?* — now has a
mechanism-level answer that survived every robustness check Sprint 7 applied to it:

> **The high-risk regime is a 4.3–6.5% minority of decision entries whose gradient is
> directionally positive but numerically outvoted. The direction PPO actually follows is
> set by the low-risk bulk — `cos(g_full, g_lo) ∈ [+0.9099, +0.9994]` in 9 of 9 measured
> cells (Rung 3 §4.5, §8.1), stable across batch size and with tight cluster-bootstrap CIs.
> At low risk, MIGRATE_EDGE is genuinely the wrong action (true advantage −1.337 on n=159,
> Phase 1 §3), so the network correctly learns to suppress EDGE, and that suppression
> generalises through shared parameters into the high-risk minority it was never fitted to.**

The behavioural consequence is measured directly and is the sprint's strongest finding:
**π(MIGRATE_EDGE | risk ≥ 0.6) ends *below* its randomly-initialised value and never
exceeds it in 76/76 R2 checkpoints on RANDOM (1/76 on UNION).** The headline
"self-healing" metric `Δ_EDGE = p_hi − p_lo` grows almost entirely because `p_lo` falls
faster than `p_hi` does. Sprint 7's own headline metric was measuring differential
suppression, not risk acquisition.

**The strongest falsification** is that the two most attractive competing explanations both
failed under direct intervention. Gradient variance was tested by quadrupling the batch
(R3): signal-to-noise rose (real/shuffled 2.03, p = 0.0000) and behaviour got **worse**.
The per-state critic offset — the mechanism that motivated R3 — is therefore closed
(Phase 0 §4), and `rollout_episodes` was subsequently shown to be a **pure variance
intervention** (every expected-update-content shift ≤ 0.105 σ₈, sd tracking the
finite-population law, Divergence §4.2), which removes the last route by which the
intervention could have acted through anything other than noise.

**The main unresolved limitation is instrumental, not conceptual.** Every gradient-level
mechanism instrument in Sprint 7 exists only at a terminal checkpoint, and Phase 0 §3
established that **20 out of 20 readings of every such instrument order the arms backwards
against behaviour (mean ρ = −0.900)** — because a terminal checkpoint measures remaining
headroom, which is anti-correlated with achieved behaviour by construction. Consequently
Sprint 7 can say *what the update direction is dominated by* and *when the behaviour
changed*, but it cannot causally attribute the high-risk decline to the dilution it
measured. It also cannot explain the **arm ordering** R2 > R3 > A0: Rung 3's D4 honesty
check failed, and the dilution mass ratios order the arms nearly in reverse.

**Verdict: Sprint 7 is COMPLETE**, by all five criteria in the brief. No further training
is justified. Section 11 gives the reason, and Section 7 names — without designing it — the
one offline analysis that would be next if the sprint were ever reopened.

---

## 2. Complete Sprint 7 timeline

One chronological row per substantive investigation. Ten fields as specified.
Expansions with the load-bearing numbers follow in §2.1.

| # | Phase / date | (1) Scientific question | (2) Intervention or observation | (3) Variable manipulated | (4) Control | (5) Primary measurement | (6) Result | (7) Verdict | (8) Hypothesis status | (9) Training? | (10) What it changed about the next step |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Phase 1 diagnosis** · 08-19 | Why does MARL lose to a risk-threshold rule? | Inspection of A0 + offline forced-replay | none | A0 as shipped | Truth-vs-estimator action ordering at high risk; greedy action census | Truth `EDGE +2.671 > STAY 0 > CLOUD −0.050`; estimator `CLOUD +0.080 > STAY +0.044 > EDGE −0.266`; greedy A0 plays CLOUD in 842/855 cloud-legal states (98.5%) and EDGE in 77/10,751 (0.7%) | "Converged, pointed the wrong way" — not a stalled optimiser | Framing established; three candidate mechanisms proposed | No | Scoped the whole sprint; created the rung ladder |
| 2 | **Rung 0** · 08-20 | Is the high-risk migration signal real, and does GAE see it? | Forced-replay counterfactual | Forced action at high-risk states | Unforced replay, same states | True advantage of forced EDGE; GAE sign agreement | Forced-EDGE true advantage **+3.363**; GAE sign agreement **29.0%** at risk > 0.50 (below chance) | Signal **real**; GAE **anti-correlated** | Phase 1 mech. #1 (horizon) **refuted** by D2 | No | Moved the target from "no signal" to "estimator inverts the signal" |
| 3 | **Rung 1** · 08-25 02:36 | Is the self-referential λ-return critic target causal? | Offline re-estimation with an MC target | `critic_target` λ → MC, **offline** | Same buffer, λ target | Advantage sign agreement at high risk | **26.3% → 60.7%** for A0 | **SUPPORTED for A0** | Target-formulation hypothesis supported | No | Authorised the one production change of the sprint |
| 4 | **Rung 2** · 08-25 04:02 | Does the MC target help in training? | Production change, then train | `critic_target` λ → **"mc"** | A0 (λ target), same seed | Δ_EDGE on frozen RANDOM / UNION | R2: **+0.2024 / +0.1579** vs A0 **−0.0264 / +0.0191** | Best arm to date | Supported *at the time*; the "win" was later withdrawn (§8) | **Yes** | Made R2 the reference arm for the rest of the sprint |
| 5 | **Rung 2.5** · 08-25 06:45 | Is the *remaining* target dependence causal? | Offline decomposition on R2 | none | A0 and R2 buffers | Sign agreement; structural nulls | Separated **three** distinct mechanisms; target dependence not the binding one | **NO-GO** for a target-focused rung | Target hypothesis **weakened** | No | Redirected from "fix the target" to "find the binding constraint" |
| 6 | **Rung 2.75** · 08-25 13:08 | Is the headline metric identified? What is the actor stall? | Frozen matched-state construction + coherence probe | none | 6 arms × 4 state sources | Matched-state Δ; paired advantage estimator; saturation census | Headline metric **not identified**; paired estimator lifts high-risk agreement from below chance to **0.71 / 0.95** for *every* arm; R2 saturation 50.5% at max-prob > 0.99 | Stall attributed to **signal variance** from a per-state offset | That attribution was later **falsified** (§8) | No | Defined the frozen RANDOM/UNION axis every later phase uses; produced the R3 hypothesis |
| 7 | **R3** · 08-25 18:22 | H1: is gradient variance / low SNR the binding constraint? | Train with 4× rollout | **`rollout_episodes` 8 → 32** (`episodes` 600→2400 holds `n_updates`=75) | R2, same seed, all else identical | SNR (real/shuffled) and Δ_EDGE | SNR **rose** (2.03, p = 0.0000); Δ **fell** to +0.1575 / +0.1242 | **NO-GO** | H1 **falsified**; with it, mechanism 2 | **Yes** | Killed the variance programme; created the R2-vs-R3 puzzle |
| 8 | **Rung 3** · 08-27 13:16 | Does low-risk dilution explain R3? | Exact gradient decomposition `g_full = g_hi + g_lo` | none | 3 arms × 3 state sources = 9 cells | `cos(g_full,g_lo)`, `‖g_hi‖/‖g_full‖`, `cos(g_hi,synth)` | D1 PASS, D2 PASS, **D3 FAIL** (dilution not interference), **D4 FAIL** | **SUPPORTED but DOES NOT EXPLAIN R3** ⇒ NO-GO | Dilution **supported** as a shared mechanism; **ruled out** as the arm-difference explanation | No | Forbade a high-risk-weighted training rung |
| 9 | **Divergence** · 08-30 11:02 | What R2-vs-R3 difference is *capable* of explaining the gap? | Config/stream/content diff + two independent instruments | none | R2 vs R3, full config surface | Update-content shift in σ₈; regime-split projection and critic home advantage | `rollout_episodes` is a **pure variance intervention** (all shifts ≤ 0.105 σ₈; FPC 1.005–1.057); a **regime-selective learning asymmetry** survives on 2 independent instruments | "What differs", explicitly **not causal** | 11 things ruled out; 5 left unidentified | No | Named per-update checkpoints as "the single largest gap" |
| 10 | **Phase 0 (reconstruct)** · 08-30 | What does the whole corpus actually support? | Calibration audit of every mechanism instrument | none | 20 instrument readings vs behaviour | Rank correlation of each instrument with behavioural Δ | **20 readings, 20 negative, 0 positive, mean ρ = −0.900** | Endpoint instruments measure **headroom**, not learning | Mechanism 2 (per-state offset) **CLOSED** | No | Demoted every endpoint metric under RULE 9; specified Phase 4 |
| 11 | **Phase 4** · 08-30 | Can R2 be replicated exactly and instrumented per update? | Zero-variable replication with checkpointing | none (zero-variable by design) | R2 itself, storage-for-storage | Bit-exact reproduction + 76 saved update boundaries | **Bit-exact**; actor freezes at **u062** (`clip_frac` ≡ 0 from u062, `|approx_kl|` < 1e-3 from u065) while the critic is still the fastest-improving quantity | Replication **succeeded** | Ruled out "the critic gated the actor in late R2"; ruled out nondeterminism | **Yes** (replication only, no new arm) | Supplied the missing per-update axis |
| 12 | **Phase 4.1** · 08-30 | (integrity closure only) | Manifest capture | none | — | 8 md5 manifests | — | No report exists | — | No | — |
| 13 | **Phase 5** · 08-31 | *When* does the risk-conditioned signal emerge, strengthen, plateau or disappear? | Score all 76 checkpoints on the **unchanged** frozen sets | none | u075 reproduces the frozen endpoint 32/32 fields, 20 bit-exact | `Δ_EDGE` and its two components per checkpoint | `Δ_EDGE` peaks at **u062** on both sets and decays to u075; **`p_hi` ends below init and never exceeds it in 76/76 (RANDOM)** | Δ_EDGE is **differential suppression** | Self-healing-as-acquisition **falsified** | No | Answered the "when" question; created the two questions of §5 |
| 14 | **This audit** · 08-31 | Is Sprint 7 scientifically complete? | Read-only synthesis | none | 5 integrity manifests re-verified | — | See §10 | **COMPLETE** | — | No | Closes Sprint 7 |

**Provenance notes that matter for reading the table.** There is **no standalone
`SPRINT_7_RUNG2_REPORT.md`** — Rung 2 is the production change plus the R2 training run, and
is reconstructible only from its citations in Rung 2.5 / Rung 2.75 / Phase 0 and from the
`mappo_R2_mc_target*` artifacts. There is **no `SPRINT_7_PHASE4_1*.md`** — Phase 4.1 exists
only as the eight manifests in `_PHASE41_integrity/`. Neither absence affects any
conclusion; both are recorded so the chain is not silently presented as denser than it is.

### 2.1 Expansions where the numbers carry the interpretation

**Row 1 (Phase 1) — three reshaping facts that constrain everything downstream.**
Risk is effectively **binary**: 95.6% of out-of-fold predictions are below 0.10, 4.05% are
above 0.50, and only 0.37% lie in (0.10, 0.50). MIGRATE_EDGE is legal in **100%** of
decision states and was refused 0/73 times; MIGRATE_CLOUD is legal in **7.95%** and was
refused 545/710 (76.8%) — so **92.05% of decision states are a binary STAY-vs-EDGE choice**
and cloud is a decoy the estimator overrates. Risk has almost no authority over the
decision margin: `|∂(logit_EDGE − logit_STAY)/∂risk| × sd(risk) = 0.00698` logits, rank
9 of 48 channels. And the single most damning number in the sprint: **A0's zero-risk
ablation scores +23.13 held-out reward versus +20.62 with the risk channel connected** —
at the endpoint, the risk channel is worse than useless to A0.

**Row 4/7 (the two training arms) — the full arm table on both frozen sets.**

| arm | manipulation vs its control | Δ RANDOM | Δ UNION |
|---|---|---|---|
| A0 | baseline, `critic_target=lambda` | −0.0264 | +0.0191 |
| A1 | attribution bugfix | +0.1375 | +0.0628 |
| A2 | critic sign | +0.0398 | −0.0248 |
| A3 | `entropy_coef` 0.01 → 0.02 | +0.1322 | +0.0933 |
| **R2** | **`critic_target` lambda → mc** | **+0.2024** | **+0.1579** |
| R3 | `rollout_episodes` 8 → 32 | +0.1575 | +0.1242 |
| R3_best | update-45 snapshot of R3 | +0.1467 | +0.0946 |

**Row 8 (Rung 3) — why D4 failing is decisive rather than inconvenient.** The behavioural
ordering is R2 (+0.1579) > R3 (+0.1242) > A0 (+0.0191) on UNION. The three pre-registered
mass/alignment quantities order the arms:

| quantity (OWN source) | A0 | R2 | R3 | ordering |
|---|---|---|---|---|
| `‖g_hi‖/‖g_lo‖` | +0.4922 | +0.1785 | +0.3865 | A0 > R3 > R2 |
| `‖g_hi‖/‖g_full‖` | +0.4190 | +0.1737 | +0.3670 | A0 > R3 > R2 |
| `cos(g_full, synth)` | +0.4023 | −0.2372 | −0.2437 | A0 > R2 > R3 |

A0, the arm with essentially no risk response, has the **largest** high-risk gradient share;
R2, the best arm, has the **smallest**. A larger high-risk gradient share is therefore not
what makes an arm respond to risk. Under RULE 9 this demotes the quantity, and under the
pre-registered rule it converts "dilution is real" into "dilution does not explain R3".

**Row 9 (Divergence) — the identifying measurement.** Of 99 config fields, five differ and
**one is substantive** (`rollout_episodes` 8→32; `episodes` 600→2400 exists only to hold
`n_updates` at 75). LR schedules are bit-identical (`max_abs_lr_diff = 0.0`). Episode
streams are **nested** — the first divergent episode is 9, and episode *j* is byte-identical
in an 8-, 32- or 128-episode build. Across seven update-content statistics every expected
shift is **≤ 0.105 σ₈** while the sd ratio tracks the finite-population law (obs/FPC
1.005–1.057; the correct prediction is sd(8)/sd(32) = 2.236, not 2.000). That is what
"pure variance intervention" means operationally.

**Row 11 (Phase 4) — the freeze, and what it rules out.** Over u046–u075 no recorded stat
drifts more than ~2.3 within-block sd, and the two still moving fastest are both the
critic's (`critic_loss` 16.5 → 8.3 still falling; `explained_var` +0.0021/update still
rising, 0.6624 at u040 → 0.7409 at u075). Trust-region activity dies while value estimation
keeps improving, which **rules out** "the critic's value quality gated the actor in late R2".
Spearman vs update (n = 75): entropy −0.946, clip_frac −0.926, explained_var +0.902,
critic_loss −0.871, adv_std −0.863, decision_frac +0.859, approx_kl −0.594, value_mean
+0.400, actor_loss +0.359, adv_mean −0.188 (the only one without a trend).

---

## 3. Hypothesis ledger

Every mechanism Sprint 7 actually investigated, in exactly one class, with the strongest
*direct* evidence. Descriptive correlations are labelled as such and are not promoted.

| # | Hypothesis | Class | Strongest direct evidence |
|---|---|---|---|
| H-A | **The high-risk migration signal does not exist** (nothing to learn) | **FALSIFIED** | Rung 0 forced-replay: forced-EDGE true advantage **+3.363**; forced-replay own-return STAY −0.8822 (t = −3.99) vs EDGE +0.8675 (t = +3.71); team +1.6227 (t = +3.78) |
| H-B | **The payoff lands at or beyond the GAE horizon** (Phase 1 mechanism #1) | **RULED OUT** | Rung 0 D2 refuted it directly. Horizon 1/(1−γλ) = **166.8** steps vs the quantity it was invoked to explain; the `mappo.py:447` comment saying "~20 steps" is stale for λ = 0.995 and was deliberately left unedited |
| H-C | **The self-referential λ-return critic target inverts the advantage sign** | **SUPPORTED (for A0), then PARTLY DOWNGRADED** | Rung 1: offline sign agreement **26.3% → 60.7%** for A0. Rung 2 turned it into production and produced the best arm. Rung 2.5 then showed the **remaining** target dependence is not the binding constraint (**NO-GO**), and Rung 2.75 §A.4 withdrew the Rung 2 "win" as a metric artefact |
| H-D | **The advantage error is a per-state offset c(s), removable by pairing** | **SUPPORTED as a statistical fact, CLOSED as an intervention** | Rung 2.75: pairing lifts high-risk sign agreement from below chance to **0.71 / 0.95 for every arm**. Phase 0 §4 closes it as an intervention: R3 supplied a 3.72× data multiplier, SNR rose, behaviour worsened; and no online remover exists — within-tick spread of c(s) averages **6.65** (max 18.61) against an overall SD(c) of **4.65** over 27 co-occurrence ticks |
| H-E | **Gradient variance / low SNR is the binding constraint** (H1) | **FALSIFIED** | R3, the only direct test: 4× batch, SNR real/shuffled **2.03, p = 0.0000**, and Δ_EDGE **fell** (+0.2024/+0.1579 → +0.1575/+0.1242). Rule 10 applies: the hypothesis failed and the programme stopped |
| H-F | **`rollout_episodes` acts through something other than noise** | **RULED OUT** | Divergence §4.2: all seven update-content shifts ≤ **0.105 σ₈**, sd follows the finite-population law. §4.3 self-corrects the earlier "5.1× collapse" to a **+2.09 σ outlier** of R2's own n = 8 sampling law (97.9th percentile) |
| H-G | **Actor softmax saturation causes the R2 stall** | **DOWNGRADED to a symptom** | Rung 2.75 measured it (R2 50.5% at max-prob > 0.99 vs A0 0.0000). R3 then made saturation *lower* and behaviour *worse*. Phase 5 dates it: on the frozen sets `frac_maxp>0.99` reaches 50% at u033 and 90% at u057–u061, i.e. **after** Δ_EDGE's 50% point at u020–u023. Saturation follows the behaviour change; it does not precede it |
| H-H | **Low-risk dilution sets the update direction** | **SUPPORTED** (and the only mechanism that survived every robustness check) | Rung 3: `cos(g_full, g_lo)` **+0.9099 to +0.9994 in 9/9 cells**, bootstrap CIs tight (lowest bound +0.7893), stable across batch size. `‖g_hi‖/‖g_full‖ ≤ 0.419` everywhere. Deleting the entire high-risk gradient moves the update by ≤ ~24°, typically ~2° |
| H-I | **The low-risk gradient actively *opposes* the high-risk one (interference)** | **FALSIFIED** | Rung 3 D3: `cos(g_hi, g_lo)` is **positive in 7 of 9 cells** (+0.0650 to +0.6860); only 1 of 9 clears the pre-registered −0.10 threshold, against a required 5/9. This is dilution, not interference |
| H-J | **Dilution explains the arm ordering R2 > R3 > A0** | **FALSIFIED** | Rung 3 D4 (see §2.1): the two mass ratios put the arms in nearly the reverse order, and the ep8 control fails D4 too with *different* inconsistent orderings |
| H-K | **Endpoint mechanism metrics measure learning** | **FALSIFIED** | Phase 0 §3: **20 readings, 20 negative, mean ρ = −0.900** against behaviour. An endpoint reading measures remaining headroom. Phase 4 sharpens this for R2 specifically: the endpoint is a near-fixed-point reached ~14 updates early, so it cannot separate "learned this" from "stopped moving here" |
| H-L | **The critic's value quality gates the actor in late R2** | **RULED OUT** | Phase 4: `clip_frac` ≡ 0 from u062 and `|approx_kl|` < 1e-3 from u065, while `critic_loss` and `explained_var` are the two fastest-moving quantities in exactly that window |
| H-M | **Nondeterminism explains the R2/R3 gap** | **RULED OUT** | Phase 4 replicated R2 **bit-exactly** on CPU, storage-for-storage |
| H-N | **R3's degradation is post-peak decay from update 45** | **FALSIFIED** | Divergence §2.4: R3's block means improve **monotonically** (−16.11 → −12.16 → −10.59 → −8.51). Only A0 genuinely peaks. `R3_best` is an artefact of `_best` selection on one noisy 32-episode mean |
| H-O | **High-risk `Δ_EDGE` growth is acquisition of high-risk migration** | **FALSIFIED** | Phase 5: `p_hi` ends **below** its init and exceeds it in **0/76** checkpoints on RANDOM, 1/76 on UNION. 159% (RANDOM) / 218% (UNION) of the net Δ growth is low-risk suppression |
| H-P | **A regime-selective learning asymmetry separates R2 from R3** | **UNRESOLVED — descriptive only** | Divergence §7.6, on two statistically independent instruments: actor-function-space `proj_frac` hi−lo R3 −0.4977 (UNION) / −0.5260 (RANDOM) vs A0 −0.2507 / −0.2450, ordering **inverting** between regimes; and critic home advantage R3 **+0.1788 at low risk** (CI [+0.1192, +0.2441], 100% of replicates) vs **−0.0089 at high risk** (CI straddles zero, 34%). The divergence report labels this "what differs, NOT causal" and this audit does not upgrade it |
| H-Q | **The high-risk gradient already knows the right action** | **MEASUREMENT-LIMITED** | Rung 3 §8.4: the reference direction `synth` is dominated by its shared `−1` on STAY (701–3,137 entries vs 68–278 on either positive channel), so `cos(g_hi, synth) > 0` largely means "push off STAY", not "push toward EDGE". R3's `g_hi` in fact aligns **better with CLOUD** than EDGE (+0.7598 vs +0.4505 OWN; +0.8224 vs +0.5652 RANDOM) |
| H-R | **The critic's high-risk value quality is a property of the critic** | **FALSIFIED** | Divergence §6.2, matched 2×2: the two critics differ by **0.009 down each column** (R2 buffer 0.7924/0.7834; R3 buffer 0.6171/0.6083). High-risk explained variance is a property of the **buffer**, not the critic |
| H-S | **R3 suffers a representational mixing pathology** | **RULED OUT** | Divergence §4.5: R3's participation ratio is the **highest measured** (14.86; PR/n 0.464) |
| H-T | **The degenerate minibatch tail is a per-arm mechanism** | **RULED OUT as differential** | `mb_size = max(1, T//4)` then `range(0,T,mb_size)` yields 5 chunks whenever `T mod 4 ≠ 0`, tail = exactly `T mod 4` rows; `rng = default_rng(0)` re-seeded every `update()`. R3's `frac_mb_zero_hi_EDGE = 0.20` is **exactly** its four `size_t = 2` degenerate chunks. Shared confound, present in every arm. **Deliberately left unfixed all sprint (RULE 7)** |
| H-U | **A high-risk-weighted update would fix the bad arm** | **RULED OUT (analytically, offline)** | Rung 3 §7: R3/UNION — the arm needing help — has `w* = +9.832` and reaches only **+0.0500** at w = 16, while R2/UNION, already the best, gains most (−0.1439 → **+0.8623**). Gradient norm grows 1.9×–6.9× at w = 16. A lever that helps the good arm and not the bad arm is not a fix |
| H-V | **Exploration starvation is the primary cause** | **SUPPORTED but SUBORDINATE** | Phase 1: 3–19 high-risk EDGE samples per update, falling. But Phase 1's own framing is the correct one — *"exploration starvation is downstream of a correctly-learned majority rule"*. At low risk (95.8% of states) EDGE is genuinely bad (true −1.337, GAE −0.538, n = 159). Starvation is a consequence of H-H, not an independent mechanism |

---

## 4. Final self-healing interpretation

### 4.1 The measurements that must be reconciled

All from `SPRINT_7_PHASE5_risk_trajectory_main.json`, the 76-checkpoint R2 trajectory scored
on the **unchanged** frozen state sets (RANDOM: 4,975 decision entries, 216 at risk ≥ 0.6,
4,757 at risk < 0.2, 32 clusters. UNION: 74,237 / 3,592 / 70,621, 64 clusters). u075
reproduces the frozen Rung-2.75 endpoint on **32/32 fields, 20 bit-exact**, including
bootstrap `se_cluster` and `z` — so this is the same axis every Sprint 7 comparison is
anchored to.

| landmark | RANDOM `p_hi` | `p_lo` | `Δ` | greedy hi-EDGE | UNION `p_hi` | `p_lo` | `Δ` | greedy hi-EDGE |
|---|---|---|---|---|---|---|---|---|
| u000 (init) | 0.4551 | 0.4385 | +0.0165 | 0.0833 | 0.4859 | 0.4852 | +0.0007 | 0.2311 |
| u008 | 0.3347 | 0.2775 | +0.0572 | 0.1620 | 0.3690 | 0.3307 | +0.0383 | 0.1679 |
| u023 | 0.2916 | 0.1624 | +0.1291 | 0.1157 | 0.3523 | 0.2109 | +0.1413 | 0.1545 |
| u042 | 0.2978 | 0.1338 | +0.1640 | 0.1065 | 0.2720 | 0.1380 | +0.1340 | 0.1542 |
| u050 | 0.3237 | 0.1368 | +0.1869 | **0.5278** | 0.2635 | 0.1304 | +0.1331 | 0.3104 |
| **u062 (peak)** | 0.3636 | 0.1533 | **+0.2102** | 0.4815 | 0.3380 | 0.1572 | **+0.1808** | 0.3694 |
| u075 (endpoint) | 0.3449 | 0.1425 | +0.2024 | 0.4630 | 0.3002 | 0.1423 | +0.1579 | 0.3163 |

Supporting quantities at the endpoint: entropy (high-risk subset) 0.6379 / 0.5062;
`frac_maxp>0.99` over all decision entries 0.2149 / 0.3559; `decision_frac` 0.1767;
`clip_frac` ≡ 0 from u062; `explained_var` 0.7409; high-risk argmax counts
u000 {STAY 190, EDGE 18, CLOUD 8, PREEMPT 0} → u075 {69, 100, 47, 0} on RANDOM and
u000 {2667, 830, 95, 0} → u075 {2201, 1136, 255, 0} on UNION. **PREEMPT_REROUTE argmax is 0
for every arm on every source** — and that is a masking fact, not a policy fact: PREEMPT is
legal in **exactly 0.00%** of high-risk decision entries, and MIGRATE_CLOUD in only 26.4%
(RANDOM) / 8.0% (UNION), so the effective high-risk choice is essentially binary STAY vs EDGE.

### 4.2 The answer: **D — a combination**, with a specific temporal decomposition

Not **A** ("the policy learned to migrate more at high risk"): that is contradicted by the
strongest single fact in the trajectory — `p_hi` **falls** from 0.4551 → 0.3449 (RANDOM)
and 0.4859 → 0.3002 (UNION), and exceeds its init in **0/76** checkpoints on RANDOM and
**1/76** on UNION. At u000 the policy is ~uniform over legal actions, so 0.45–0.49 is a
coin flip between STAY and EDGE. **R2 ends below a coin flip at high risk.**

Not **B** alone ("primarily suppression"), even though suppression dominates: over the whole
run the low-risk share of Δ growth is −59.2%/+159.2% (RANDOM) and −118.2%/+218.2% (UNION),
i.e. suppression more than accounts for the total. But a phase-by-phase decomposition
(*derived* this audit from the stored per-checkpoint values) shows suppression does **not**
account for all of it at all times:

| phase | RANDOM dP_hi / dP_lo / dΔ | share hi / lo | UNION dP_hi / dP_lo / dΔ | share hi / lo |
|---|---|---|---|---|
| u000→u008 | −0.1204 / −0.1610 / +0.0406 | −296% / +396% | −0.1169 / −0.1544 / +0.0376 | −311% / +411% |
| u008→u023 | −0.0431 / −0.1151 / +0.0720 | −60% / +160% | −0.0168 / −0.1198 / +0.1031 | −16% / +116% |
| u023→u042 | +0.0062 / −0.0286 / +0.0348 | +18% / +82% | −0.0803 / −0.0729 / −0.0073 | (Δ ≈ 0; shares meaningless) |
| **u042→u062** | **+0.0658 / +0.0195 / +0.0463** | **+142% / −42%** | **+0.0660 / +0.0192 / +0.0468** | **+141% / −41%** |
| u062→u075 | −0.0187 / −0.0108 / −0.0078 | (decay) | −0.0379 / −0.0149 / −0.0230 | (decay) |

**u042→u062 is a genuine high-risk acquisition phase** — `p_hi` rises by +0.0658 / +0.0660
against a low-risk rise of only +0.0195 / +0.0192, so more than 100% of Δ's growth in that
window comes from the high-risk side. It contributes 23% (RANDOM) / 30% (UNION) of the
final Δ. It is real, it is on both independent state sets, and it coincides exactly with the
greedy channel's reorganisation. Calling the whole finding "suppression" would erase it.

Not **C** alone ("greedy learned it, the expectation says otherwise"), though the greedy
channel is the strongest part of the case for genuine learning and must be reported with the
rest (RULE 9). Two facts make it more than a curiosity:

1. **Against its own init**, greedy high-risk EDGE rises 0.0833 → 0.4630 (RANDOM) and
   0.2311 → 0.3163 (UNION) — 18 → 100 states and 830 → 1,136 states.
2. **Against the control**, R2 beats A0 on *both* channels: greedy 0.4630 vs A0's 0.0833
   (RANDOM) and 0.3163 vs 0.1748 (UNION); and stochastic `p_hi` 0.3449 vs A0's **0.1261**
   and 0.3002 vs **0.2233**. So R2's high-risk EDGE mass is below its own init but well
   **above the baseline arm's**.

**Therefore: D.** Sprint 7 established that R2's risk-conditioned behaviour is
**predominantly differential low-risk suppression** — that is where most of the metric comes
from and it is what dominates u000–u023 — **plus a real but smaller high-risk acquisition
phase at u042–u062**, visible on both the stochastic and greedy channels, which raises R2
above the A0 control on both channels while leaving its high-risk EDGE mass below its own
random initialisation. The metric `Δ_EDGE` is not a measure of risk-aware migration; it is a
contrast whose growth is mostly driven by its low-risk term.

### 4.3 What this means for the central question

The policy does not fail to *see* risk — it fails to *act* on it in the direction the
environment rewards, and it fails for a reason that is visible in the gradient: at low risk
EDGE is genuinely wrong, the low-risk regime is 93.5–95.7% of decision entries, and the
update direction it produces is followed almost exactly (`cos(g_full, g_lo) ≥ 0.9099`, 9/9).
The high-risk subspace points a better way (`cos(g_hi, synth) > 0`, 9/9) and is outvoted.
"Self-healing" in the deployed greedy sense is partially real; "the policy learned to
migrate at high risk" in the mass sense is not.

---

## 5. R2 trajectory interpretation

**Phase structure**, from the 76 checkpoints:

- **u001–u007 — absent / unsigned.** Three sign flips in Δ_EDGE. Nothing is established yet.
- **u008–u023 — emergence, driven by suppression.** Both `p_hi` and `p_lo` collapse, `p_lo`
  faster. Δ rises from +0.0572 to +0.1291 (RANDOM) / +0.0383 to +0.1413 (UNION) while `p_hi`
  is *falling*. **R2 passes R3's entire final Δ_EDGE by u021 (UNION) / u042 (RANDOM)** — so
  whatever separates the two arms lives in this early-middle suppression phase, not at the
  endpoint.
- **u024–u042 — plateau with a real dip.** The greedy argmax is **frozen** at 0.1065
  (RANDOM, u030–u042) while the probabilities keep drifting. `p_hi` bottoms at 0.2289 @ u037
  (RANDOM) and 0.2635 @ u050 (UNION).
- **u043–u062 — reorganisation and genuine high-risk acquisition.** The greedy channel
  unfreezes and jumps; `p_hi` rises on both sets; Δ reaches its maximum.
- **u062 — the peak on both sets** (+0.2102 RANDOM, +0.1808 UNION). This is also the first
  update at which `clip_frac` is identically zero. That is an ordering coincidence, recorded
  as such, and **not** a causal claim.
- **u062–u075 — decay to the endpoint.** Δ falls by −0.0078 (RANDOM, −0.30 cluster SE — not
  resolvable) and −0.0230 (UNION, −3.38 cluster SE — resolvable). **The endpoint is not R2's
  best reading of its own primary metric**, and every cross-arm comparison in Sprint 7 is
  anchored to that endpoint.

**Precedence (ordering only; n = 1 arm, and 75 updates are not replicates).** `clip_frac`
is 90% done at u014 and on-policy `entropy` at u032 — *before* Δ_EDGE's 90% point (u046
RANDOM / u060 UNION). Saturation *follows*: `frac_maxp>0.99` on the fixed sets is 50% done
at u033 and 90% at u057–u061, versus Δ_EDGE's 50% at u020–u023. So the actor's trust-region
activity and entropy decline lead the behaviour change, and saturation trails it. This
ordering is consistent with H-G's demotion and with H-L's ruling-out; it does not by itself
establish direction of causation, and is not offered as doing so.

**The freeze.** From u062 the actor is not measurably moving (`clip_frac` ≡ 0,
`|approx_kl|` < 1e-3 from u065) while the critic is still the fastest-improving quantity
(`critic_loss` 12.54 @ u040 → 5.77 @ u075; `explained_var` 0.6672 → 0.7409). R2 reaches a
near-fixed-point with ~19% of training remaining. This is the single most important fact for
reading every other Sprint 7 result, because it means the endpoint that all cross-arm
instruments measured is a **stopping point**, not a **learning outcome**.

---

## 6. R2 / R3 explanation status

**Status: UNRESOLVED, and explicitly so.** Sprint 7 can state precisely what differs and can
rule out a long list of candidates, but it has no causal account of why R2 > R3.

**What is settled.**
- The manipulated variable is *only* `rollout_episodes` 8 → 32, and it is a **pure variance
  intervention** (Divergence §4.2). Episode streams are nested, LR schedules bit-identical.
- The hypothesis it was built to test (H-E, variance) is **falsified**: SNR rose and
  behaviour worsened.
- The mechanism invoked afterwards to rescue it (H-H, dilution) is **real but does not
  explain the arm difference** (Rung 3 D4). Its cross-arm variation runs the *wrong way*.
- R3's degradation is not post-peak decay (H-N), not mixing (H-S), not a critic property
  (H-R), not nondeterminism (H-M), not the minibatch tail differentially (H-T).
- R3's progress clock: its endpoint sits at R2's **update 25–27 on actor metrics** but
  **44–70 on critic metrics** (median 56) — *"its actor is far more behind than its critic."*
  Σ|k1| is essentially equal (0.1533 vs 0.1528, 0.3%) yet `‖Δθ_actor‖` is 6.1710 vs 4.5471,
  so R3's steps **cancel more**. Entropy is *more* variable in R3 despite 4× the data
  (sd ratio 1.138). `infeasible` — a **contention** counter, not an illegal-action counter —
  is 14.560 for R2 vs 1.698 for R3 (z = −34.51, onset ep 130), i.e. R2 actually migrates.

**What survives as a description, and only that.** The **regime-selective learning
asymmetry** (H-P): R3 learned the low-risk bulk and not the high-risk minority, on two
statistically independent instruments (actor function space, critic return space). The
divergence report labels this "what differs, NOT causal"; this audit does not upgrade it.

**The sharpest available characterisation** is the E2 shape decomposition, which explains the
*form* of the difference even though not its cause. On UNION, arg-max → EDGE at high risk
(out of 3,592): θ₀ 830, A0 628, R2 **1,136**, R3_best 709, **R3 0**; `uniform_share` at high
risk A0 0.7213, R2 **0.5948**, R3_best 0.5675, R3 **0.7562**. On RANDOM (out of 216):
θ₀ 18, A0 18, R2 **100**, R3_best 72, **R3 0**; `uniform_share` A0 0.7035, R2 **0.3815**,
R3 **0.6799**. Read together: **R2 re-ranks state-specifically** (lowest uniform share of any
arm, and the most argmax flips), while **R3 raised P(EDGE) roughly uniformly and never
re-ranked a single state** — zero argmax flips on either set despite Δ_EDGE of
+0.1575/+0.1242. A1 shows the same signature (Δ +0.1375/+0.0628 with greedy hi-EDGE
0.0139/0.0025). This is the clearest statement Sprint 7 can make about R2 vs R3, and it is
a statement about *shape*, not cause.

---

## 7. Remaining uncertainties

### 7.1 Question 1 — *why does high-risk EDGE probability itself decline?* — **PARTIALLY ANSWERABLE**

**What the existing artifacts do support:**

1. **It is front-loaded.** Of a net −0.1102 (RANDOM) / −0.1857 (UNION), **−0.1204 / −0.1169
   happens in u000–u008 alone** — the first 8 of 75 updates. On RANDOM the entire net decline
   is more than accounted for by the first eight updates.
2. **It is not a high-risk-specific event.** `p_hi` and `p_lo` co-move across the 76
   checkpoints with Spearman **+0.5116 (RANDOM) / +0.8821 (UNION)** (*derived*), and `p_lo`
   falls further in every early phase. The high-risk decline is the **attenuated high-risk
   share of a global EDGE→STAY shift**, not a separate process.
3. **The destination of the mass is known.** It goes predominantly to STAY (UNION low-risk
   `p_stay` 0.4854 → 0.7996), and to CLOUD where CLOUD is legal.
4. **It reverses.** `p_hi` bottoms at u037 (RANDOM) / u050 (UNION) and then rises +0.0658 /
   +0.0660 over u042–u062, i.e. the decline is not monotone and not terminal.
5. **The mechanism most consistent with all of the above is already measured**: a correctly
   learned low-risk majority rule generalising through shared parameters. Phase 1 established
   that EDGE is genuinely bad at low risk (true −1.337, n = 159) and that low risk is 95.8%
   of states; Rung 3 established that the low-risk gradient sets the update direction
   (`cos(g_full, g_lo) ≥ 0.9099`, 9/9) and that `cos(g_lo, synth)` is ≈0 for A0
   (+0.023…+0.060) but **negative for every R2 and R3 cell** (−0.030 to −0.437) — the
   low-risk bulk is precisely what drags the full gradient's sign away from EDGE in the
   trained arms.
6. **Four candidate causes are already closed**: target formulation (H-C, downgraded),
   per-state offset (H-D, closed), saturation (H-G, follows rather than precedes), and
   variance/`rollout_episodes` (H-E/H-F, falsified).

**What the existing artifacts do *not* support — and why:** the **causal attribution**. Item
5 is a coherent mechanism assembled from two instruments measured at *different* times on
*different* objects: the behavioural decline is a per-checkpoint trajectory measurement, while
every gradient decomposition in Sprint 7 exists **only at a terminal checkpoint**. Phase 0 §3
showed that endpoint instruments order the arms backwards 20/20 (mean ρ = −0.900), so an
endpoint gradient reading cannot be used to attribute a *temporal* change. Closing this would
require the Rung 3 decomposition evaluated along the u000…u075 trajectory — which is an
offline analysis, not a training run, and which **no existing artifact contains**. I am not
proposing it here; §11 explains why it is not required for closure.

**Also not answerable, and named rather than proxied:** why the decline resumes over
u062–u075 while every recorded training statistic is quiescent (`clip_frac` ≡ 0,
`|approx_kl|` < 1e-3). No recorded quantity distinguishes that window from u046–u061.

### 7.2 Question 2 — *why does stochastic π(EDGE) differ from greedy argmax EDGE?* — **LARGELY ANSWERABLE from existing artifacts**

Four independent components, all from stored values; **no new measurement is needed.**

1. **The two channels have incomparable baselines at u000.** The mean high-risk margin
   `p_edge − p_stay` at init is **−0.0020 (RANDOM) / −0.0015 (UNION)** (*derived*). A margin
   that close to zero means the greedy baseline (0.0833 / 0.2311) is decided by
   initialisation noise, not by any preference. So "rises above init" (greedy) and "falls
   below init" (mass) are not contradictory statements about learning — one of the two
   baselines is a coin flip.
2. **The endpoint high-risk population is arithmetically forced to be near-bimodal.** With
   argmax = EDGE on 100/216 (RANDOM) and 1,136/3,592 (UNION) at u075 and the recorded mean
   `p_hi`, the mean `p_edge` on the *non*-argmax-EDGE complement is at most **0.2112 /
   0.2077** (two-legal-action bound; **0.3549 / 0.2848** under the safe three-legal floor)
   (*derived*). A mean over a bimodal population is a mixture statistic, not a description of
   a typical state — so a rank statistic and a mean need not co-move.
3. **They demonstrably do not co-move, and the coupling is set-dependent.**
   Spearman(`p_edge_hi`, `argmax_edge_hi`) is **+0.5972 over all 76 / +0.8478 over u008–u075
   on RANDOM**, but only **+0.0883 / +0.1285 on UNION** (*derived*). The clearest single
   instance: between u042 and u050 the RANDOM greedy rate jumps **+42.1 pp** against a mean
   margin change of only **+2.8 pp**.
4. **The mechanism is already in the corpus.** Divergence §7.6's E2 decomposition
   distinguishes a **uniform probability shift** from a **state-specific re-ranking**. R2
   re-ranks (uniform share 0.3815 / 0.5948, the lowest of any arm; 100 / 1,136 argmax flips);
   R3 shifts uniformly (0.6799 / 0.7562, the highest; **0 argmax flips**). Only re-ranking
   moves the greedy channel. A policy can raise mean high-risk `p_edge` without flipping a
   single argmax, and can flip many argmaxes with a small change in the mean.

**What remains genuinely unresolved here, stated rather than proxied:** (a) **which channel
is the correct target** — deployment uses greedy, the PPO objective uses the expectation, and
no artifact adjudicates between them; (b) **why the coupling is strong on RANDOM and absent
on UNION**. Answering (b) requires the per-state joint distribution of (`p_edge`, `p_stay`) at
high risk, which **no artifact stores** — the artifacts store means, argmax counts, and
cluster bootstraps, and a mean plus a count cannot recover a joint. The bimodality bound in
item 2 is the tightest statement the stored aggregates permit, and it is a bound, not a
distribution.

### 7.3 Other standing uncertainties

- **Single seed.** Every arm is n = 1 (`seed 20260818` lineage). RULE 11 requires ≥ 3 seeds
  for a causal claim, and Sprint 7 accordingly makes none. Nothing was cherry-picked across
  seeds because only one exists.
- **Direction of actor/critic causation** — unidentified (Divergence §9).
- **Whether R2 would also degrade past update 75** — never tested.
- **When R3's arg-max channel collapsed** — bracketed only to updates 45–75.
- **`explained_var` remains target-relative** and the minibatch tail remains unfixed
  (RULE 7). Neither is load-bearing for any conclusion above.

---

## 8. Superseded historical claims

Each entry: what was believed, what overturned it, and its current status. Nothing here is
edited out of the original reports; this section is the record of the change.

| # | Original claim (where) | Overturned by | Current status |
|---|---|---|---|
| 1 | The high-risk payoff lands at/beyond the GAE horizon (Phase 1 §, mechanism #1) | Rung 0 D2 | **Superseded — refuted.** Horizon is 166.8 steps at λ = 0.995; `mappo.py:447`'s "~20 steps" comment is stale (correct only for the retired λ = 0.95) and was deliberately left unedited |
| 2 | Rung 0's residual magnitude −5.24 (GAE) and its team-vs-own confound | Rung 1 §7 | **Superseded.** Correct critic-free residual is −1.07; the team-vs-own confound was refuted |
| 3 | Rung 2 produced a "win"; Rung 2.5 produced a "loss" | Rung 2.75 §A.4 | **Both withdrawn** as metric artefacts |
| 4 | Rung 2.5 §E.4's −1.1089 | Rung 2.5 §E.7 (self-retracted) | **Retracted in place.** The `structural_null` block for R2 in `SPRINT_7_RUNG2_5_signtest.json` is **false and still present in the artifact** — do not read it |
| 5 | Rung 2.5 §G.5: "R2 migrates more at high risk" is unresolved | Phase 5 | **Partly resolved, in both directions**: yes vs the A0 control on both channels (0.3449 vs 0.1261; 0.3002 vs 0.2233) and yes on the greedy channel vs its own init; **no** on the mass channel vs its own init (0/76) |
| 6 | Rung 2.75 §B.2's plasticity finding | Rung 2.75 §B.2 itself | **Self-reversed within the same report** |
| 7 | The R2 actor stall is caused by signal variance from the per-state offset (Rung 2.75) | R3, then Phase 0 §4 | **Falsified.** 3.72× data multiplier, SNR up (p = 0.0000), behaviour down. Mechanism 2 **CLOSED** |
| 8 | Softmax collapse causes the stall (Rung 2.75) | R3, then Phase 5 precedence | **Downgraded to a symptom.** Saturation trails the behaviour change (50% at u033 vs Δ_EDGE's u020–u023) |
| 9 | `clip_frac → 0` is an R2 anomaly | Phase 4 | **Retracted.** It is the normal end-state of a converged actor; last non-zero at u061 |
| 10 | "R3's reward peaked at update 45 then declined" | Divergence §2.4 | **Not supported.** Block means are monotone (−16.11 → −12.16 → −10.59 → −8.51). `R3_best` must be framed "R3_best vs R3_final", never as post-peak decay |
| 11 | The "5.1× collapse / sign flip" in R3's update content | Divergence §4.3 (self-corrected) | **Retracted.** A +2.09 σ outlier of R2's own n = 8 law (97.9th percentile) |
| 12 | A 0.25 `tail_fraction` indicates a degenerate tail | Divergence §5.3 | **Corrected.** 0.25 is the last *full* chunk; the degenerate tail is exactly `T mod 4` rows |
| 13 | Endpoint mechanism metrics measure learning (implicit in Rungs 0–3) | Phase 0 §3 | **Superseded.** 20/20 readings invert, mean ρ = −0.900. Every endpoint instrument is demoted under RULE 9 |
| 14 | `Δ_EDGE` measures acquisition of high-risk migration (implicit in Rungs 2–2.75 and in the R3 comparison) | Phase 5 | **Superseded.** It is a contrast dominated by its low-risk term |
| 15 | Rung 3's D1 "SUPPORTED" half is established | Rung 3 §8.2 (its own pre-registered robustness checks) | **Qualified to unresolved.** At 8 episodes the same scorer returns **FALSIFIED**; the bootstrap CI for `cos(g_hi, synth)` spans zero in 5/9 cells including **all three R3 cells**. The **NO-GO** and **DOES NOT EXPLAIN R3** halves are robust across both batch sizes |
| 16 | `cos(g_hi, synth) > 0` means the gradient knows the right action | Rung 3 §8.4 | **Superseded.** `synth` is dominated by its shared `−1` on STAY; it largely measures "off STAY", and R3's `g_hi` aligns *better* with CLOUD |
| 17 | **`SPRINT_7_R3_action_channels.json` does not exist** (`SPRINT_7_PHASE5_REPORT.md` §4.4) | This audit | **Wrong.** The probe writes with a `--tag` suffix: `SPRINT_7_R3_action_channels_R3.json` exists (36,293 bytes, md5 `53955350c3d84c10c234d33464ea8d5c`, dated 08-25 18:10) and is listed in `SPRINT_7_R3_REPORT.md`'s own manifest at line 127. **The Phase 5 report is left unedited and this entry is the correction.** Consequence: Phase 5's four-action CLOUD/STAY/PREEMPT responses were *not* the first such measurement, contrary to that report's §4.4. This does **not** affect any Phase 5 number — the `_NEW_not_in_frozen_artifact` values were computed from the same `P` array as everything else and stand on their own |

**What "we believed at the time" vs "what the complete evidence supports now", in one line
each.** *Then:* the λ-return target inverts the advantage, fixing it wins, and the remaining
stall is a variance problem to be solved with more data. *Now:* the target change helped a
metric that turned out to measure something else; the variance hypothesis is falsified by
direct intervention; the surviving mechanism is that a correctly-learned low-risk majority
rule sets the update direction and outvotes a directionally-correct high-risk minority; and
the arm-to-arm differences remain unexplained.

---

## 9. Scientific limitations

1. **Single seed** for every arm. No causal claim is made anywhere in this synthesis, and
   RULE 11 is the reason.
2. **Endpoint-only gradient instruments.** Every mechanism decomposition (Rungs 0–3,
   divergence D4) exists at one terminal checkpoint per arm. Phase 0 §3 quantified the
   consequence: 20/20 inversions against behaviour. For R2 specifically the endpoint is a
   near-fixed-point reached ~14 updates early (Phase 4), so it cannot separate "learned this"
   from "stopped moving here".
3. **The reference direction `synth` is only weakly channel-discriminating** (Rung 3 §8.4).
   Any future criterion built on `cos(·, synth)` must score the EDGE and STAY components
   separately.
4. **RANDOM and UNION gradients are counterfactual** — the PPO ratio is not 1 there. OWN is
   the only production-faithful source, and it is simultaneously the noisiest (null cosine sd
   0.24–0.34 at n_hi = 850–1,071 in 233,000 dimensions) and the most selection-contaminated
   (risk is high at a host *because* tasks remain on it, so conditioning on "currently high
   risk" on an arm's own trajectories selects for "just chose STAY"). This tension is
   intrinsic.
5. **Both state sets are fixed and were built from the final A0/R2 greedy trajectories
   (UNION) or uniform random play (RANDOM).** They are the right axis for cross-checkpoint
   comparison precisely because they do not move, but they are not the on-policy distribution
   of any checkpoint but the two that built them.
6. **The frozen artifact's denominators differ by field** — `p_edge_mean`, `p_stay_mean`,
   `maxp_mean`, `frac_maxp_gt_099`, `entropy_mean`, `frac_argmax_edge` and `argmax_counts`
   are over the **high-risk subset only**, while `p_edge_risk_lt_02` / `_ge_06` /
   `risk_response_*` / `spearman_risk_vs_p_edge` are over **all decision entries**. Mixing
   them has produced a wrong reading at least once in this sprint.
7. **Two high-risk rules coexist and were deliberately not reconciled**: `risk > 0.50` in the
   Rung 3 gradient probe and `min(int(risk·5),4) ≥ 3` for state-set scoring (848 vs 850 on
   A0/OWN — immaterial), plus Phase 5's `risk ≥ 0.6` / `risk < 0.2` cuts. Each is used
   exactly where it already was.
8. **Action legality bounds every action-channel reading.** PREEMPT_REROUTE is legal in
   **0.00%** of high-risk decision entries; MIGRATE_CLOUD in 26.4% (RANDOM) / 8.0% (UNION).
   A low CLOUD or PREEMPT rate is a masking fact until divided by its legality rate.
9. **Known artifact traps that remain in the repository**: the four `*_b32.json` census
   artifacts are duplicates (D3/D4 read `rollout_episodes` from the checkpoint config and
   silently ignore `--episodes`) and must never be read as a batch-size control; `R2_best` is
   bit-identical to `R2`; `R3_best` is a genuine update-45 checkpoint; D4's
   `replica_fidelity ok=False` is critic-side only and all actor-side D4 quantities are exact;
   `_phase4_verify.py:78` holds a 1-ULP-wrong `final_lr_critic` so B6 fails **by design**
   (18/19 expected) and correcting it would require a prereg amendment; `approx_kl` in the
   logs is the signed k1 estimator and legitimately goes negative.
10. **The minibatch tail was never fixed** (RULE 7) and `explained_var` remains
    target-relative. Both are shared, non-differential confounds.

---

## 10. COMPLETE / NOT COMPLETE verdict

**Sprint 7 is scientifically COMPLETE.**

| criterion | assessment |
|---|---|
| 1. Self-healing behaviour characterised sufficiently to answer the central question | **Met.** §4 fixes the interpretation (answer **D**) on a 76-checkpoint trajectory scored on two frozen state sets, with both channels reported and the endpoint shown not to be the peak. The central question has a mechanism-level answer (H-H) that survived every robustness check applied to it |
| 2. Major competing mechanisms explicitly classified | **Met.** §3 classifies 22 hypotheses; 6 falsified, 6 ruled out, 3 supported, 3 downgraded/weakened, 2 measurement-limited, 2 unresolved-and-labelled-descriptive |
| 3. The R2 trajectory reconstructed | **Met.** Phase 4 replicated R2 bit-exactly and saved all 76 update boundaries; Phase 5 scored every one of them on the unchanged frozen sets, with u075 reproducing the frozen endpoint 32/32 fields (20 bit-exact) |
| 4. Remaining unknowns are measurement limitations rather than questions requiring another training intervention | **Met.** Every unknown in §7 is instrumental: endpoint-only gradient instruments, unstored per-state joint distributions, a single seed. **None requires a training run.** The two hypotheses that *would* have justified one (variance, and high-risk up-weighting) are respectively falsified by direct intervention and ruled out analytically |
| 5. No existing artifact contains a cheap, decisive measurement we have simply failed to inspect | **Met.** This audit read the Phase 0/1, Rung 0/1/2.5/2.75/3, R3, Divergence, Phase 4 and Phase 5 reports and re-inspected the underlying JSON for the Phase 5 trajectory, the frozen Rung-2.75 cross-arm table (all six arms × four sources) and the recorded per-update stats. The one *believed-missing* artifact turned out to exist (§8 row 17) and was inspected — it contains no measurement that changes a conclusion. Everything else that would sharpen §7 requires computing something that is not stored |

Consistent with this, **no new diagnostic was written or run for this audit** — the
efficiency rule's "otherwise, just analyze what already exists" branch applied.

---

## 11. Why no further experiment is justified

1. **The two experiments that a mechanism hypothesis would motivate have both already been
   pre-empted.** A variance/batch rung was run (R3) and falsified the hypothesis. A
   high-risk-weighted rung was analytically evaluated before being proposed (Rung 3 §7) and
   is **NO-GO by a pre-registered rule**: the leverage is smallest exactly where it is needed
   (R3/UNION `w* = +9.832`, reaching only +0.0500 at w = 16) and largest where it is not
   (R2/UNION −0.1439 → +0.8623), while the gradient norm grows 1.9×–6.9×. A lever that helps
   the good arm and not the bad arm is not a fix.
2. **The success metric that would score any new arm is now known to measure something
   else.** `Δ_EDGE` is a contrast dominated by its low-risk term (§4). Running a new arm
   against it would reproduce the same ambiguity at greater cost. Under RULE 9 the honest
   move is to demote the metric, not to collect more of it — and under the audit's own
   constraints, defining a replacement metric now would be exactly the post-hoc criterion
   change the sprint forbids.
3. **The binding limitation is instrumental, and instruments do not need training runs.**
   Every open question in §7 is blocked by a *measurement* that was never taken, not by an
   *intervention* that was never made. No training run would unblock any of them.
4. **A new arm could not be interpreted anyway.** With one seed, RULE 11 forbids a causal
   claim from it; and the endpoint at which it would be scored is, for R2 at least, a
   near-fixed-point reached ~14 updates before training ends (Phase 4), so a single new
   endpoint reading would inherit the exact defect Phase 0 §3 quantified at 20/20.
5. **The single highest-value *offline* analysis is named but not required.** If Sprint 7
   were ever reopened, the highest-value next analysis is the Rung 3 gradient decomposition
   evaluated **along the u000…u075 trajectory** rather than at the endpoint — it would test
   directly whether `cos(g_full, g_lo)` stays ≥ 0.9 throughout, whether the low-risk share of
   the update accounts for the u000–u008 collapse, and what changes at u042 and u062. It
   requires **no training**, uses checkpoints that already exist, and would close §7.1's
   causal gap. It is recorded here as a pointer for whoever picks this up next. It is **not**
   designed, **not** pre-registered, and **not** started, per the instruction to stop after
   this synthesis.

---

## 12. Integrity and provenance

**This audit was READ-ONLY.** Snapshots were taken before any analysis, in
`saved_models/marl/_AUDIT_integrity/`, and re-verified after all reading was complete and
immediately before the single file this audit writes.

| manifest | scope | entries | verification |
|---|---|---|---|
| `SPRINT_7_AUDIT_code_before.md5` | all `marl/**/*.py` | 53 | **53/53 OK** |
| `SPRINT_7_AUDIT_artifacts_before.md5` | all top-level files in `saved_models/marl/` | 160 | **160/160 OK** |
| `SPRINT_7_AUDIT_trajectory_before.md5` | `R2_trajectory/` (u000…u075 + manifest) | 78 | **78/78 OK** |
| `SPRINT_7_AUDIT_inputs_before.md5` | failure history / log / OOF predictions | 3 | **3/3 OK** |
| `SPRINT_7_AUDIT_manifests_before.md5` | every prior phase's own manifests | 84 | **83/84 OK** — see note |

**The single FAILED entry is the manifest's own self-entry.** Line 4 of
`SPRINT_7_AUDIT_manifests_before.md5` records a hash for
`./_AUDIT_integrity/SPRINT_7_AUDIT_manifests_before.md5`, which the shell had already created
(empty) when `find` enumerated it and which was then filled by the same redirect. It is a
circular-hashing artefact of capture, identical in kind to the one documented in earlier
rungs, and **not a modification**: the other 83 entries — every prior phase's manifests —
verify clean. Nothing in the repository changed during the reading phase.

**Git corroborates independently.** HEAD is `a8df6d9`; `git diff --stat HEAD` shows the same
8 files / 1,131 insertions as at the start (pre-existing `_DIVERGENCE_integrity` additions),
with **no modification or deletion of any tracked file**. `python-ai/marl/` is itself
untracked, which is why the md5 manifests are the authority for that tree.

**Production modules:** `config.py`, `mappo.py`, `train.py`, `env.py` are byte-identical to
their pre-audit hashes as members of the 53/53 code manifest. Production source last changed
**2026-08-25 04:02** — the Rung 2 MC-target change — and is byte-identical in all subsequent
manifests and at HEAD.

**Exactly what this audit added — one file:**

```
saved_models/marl/SPRINT_7_FINAL_SYNTHESIS_AUDIT.md      (this report)
```

plus the after-manifests written for the record
(`SPRINT_7_AUDIT_{code,artifacts,trajectory,inputs,manifests}_after.md5`). **Nothing was
modified. Nothing was deleted. No checkpoint, artifact, log, preregistration or production
module was touched.** No new diagnostic script was created; none was needed.

**Reports whose claims are corrected here rather than edited:** `SPRINT_7_PHASE5_REPORT.md`
§4.4 (see §8 row 17) is left exactly as written; this document is the correction of record,
per the instruction not to silently rewrite history.

**Materials read for this audit:** `SPRINT_7_PHASE1_DIAGNOSIS.md`,
`SPRINT_7_PHASE0_RECONSTRUCTION.md`, `SPRINT_7_RUNG0_REPORT.md`, `SPRINT_7_RUNG1_REPORT.md`,
`SPRINT_7_RUNG2_5_REPORT.md`, `SPRINT_7_RUNG2_75_REPORT.md`, `SPRINT_7_R3_REPORT.md`,
`SPRINT_7_RUNG3_REPORT.md`, `SPRINT_7_DIVERGENCE_REPORT.md`, `SPRINT_7_PHASE4_REPORT.md`,
`SPRINT_7_PHASE5_REPORT.md`, and the underlying artifacts
`SPRINT_7_PHASE5_risk_trajectory_main.json`, `SPRINT_7_RUNG2_75_matched_states_main.json`,
`SPRINT_7_RUNG2_75_matched_states_R3.json`, `SPRINT_7_R3_action_channels_R3.json`,
`SPRINT_7_PHASE0_calibration.json`, the `R2_trajectory/` manifest, and all
`_*_integrity/*.md5` chains.

---

## FINAL SPRINT 7 STATUS

- **Central question:** Why does the MAPPO policy fail to learn and use risk-aware migration
  behaviour, despite the environment exposing predicted failure risk and despite offline
  diagnostics showing that useful high-risk migration signal exists? — **Answered at
  mechanism level for the shared failure; unanswered for the arm-to-arm differences.**
- **Strongest finding:** The update direction is set by the low-risk bulk — `cos(g_full, g_lo)`
  ∈ [+0.9099, +0.9994] in 9/9 cells, robust to batch size and with tight bootstrap CIs — and
  the behavioural consequence is measured directly: **π(MIGRATE_EDGE | risk ≥ 0.6) ends below
  its randomly-initialised value and never exceeds it in 76/76 R2 checkpoints on RANDOM
  (1/76 on UNION)**. The headline "self-healing" metric is predominantly differential low-risk
  suppression, with a real but smaller genuine high-risk acquisition phase at u042–u062.
- **Strongest falsification:** The gradient-variance hypothesis. R3 quadrupled the batch,
  signal-to-noise **rose** (real/shuffled 2.03, p = 0.0000), and risk-conditioned behaviour got
  **worse** (+0.2024/+0.1579 → +0.1575/+0.1242) — and `rollout_episodes` was subsequently shown
  to be a **pure variance intervention** (all update-content shifts ≤ 0.105 σ₈), closing the
  per-state-offset mechanism with it.
- **Main unresolved limitation:** Every gradient-level mechanism instrument exists **only at a
  terminal checkpoint**, and endpoint instruments order the arms backwards against behaviour
  20/20 (mean ρ = −0.900). Sprint 7 can therefore say what the update direction is dominated by
  and when the behaviour changed, but cannot causally attribute the high-risk decline — and
  cannot explain the ordering R2 > R3 > A0, whose surviving description (a regime-selective
  learning asymmetry) is explicitly "what differs", not causal.
- **Further training justified?** **NO.**
- **Sprint 7 closed?** **YES.**
