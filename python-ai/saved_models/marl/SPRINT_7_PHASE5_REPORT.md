# SPRINT 7 — PHASE 5 REPORT
## When does R2's risk-conditioned (self-healing) policy signal emerge, strengthen, plateau or disappear?

Trajectory analysis. Read-only with respect to every pre-existing artifact.
No training was run. No production module was touched. No new success criterion is
proposed and no further experiment is designed.

---

## 1. Question

> **WHEN DURING R2 TRAINING DOES THE SELF-HEALING / RISK-CONDITIONED POLICY SIGNAL
> EMERGE, STRENGTHEN, PLATEAU, OR DISAPPEAR, AND WHICH ALREADY-RECORDED TRAINING
> QUANTITIES TEMPORALLY PRECEDE THAT CHANGE?**

Primary metric — unchanged, as pre-registered and already measured for every existing
arm in `SPRINT_7_RUNG2_75_matched_states_main.json`:

```
Delta_EDGE = pi(MIGRATE_EDGE | risk >= 0.6) - pi(MIGRATE_EDGE | risk < 0.2)
```

taken over **all** decision entries (`mask.sum(-1) > 1`) of a **fixed** state set.

---

## 2. Data and artifacts used

| Role | Artifact | Status |
|---|---|---|
| Policies scored | `saved_models/marl/R2_trajectory/R2_trajectory_u000.pth … _u075.pth` | 76 checkpoints, read-only |
| Update alignment + already-recorded stats | `R2_trajectory/SPRINT_7_P4_trajectory_manifest.jsonl` | 76 records, read-only |
| Frozen endpoint reference | `SPRINT_7_RUNG2_75_matched_states_main.json` | read-only |
| Existing arm endpoints for comparison | same file (A0, A1, A2, A3) + `…_R3.json` (R3) | read-only |
| State-set construction code | `marl/_diag_rung2_75_matched_states.py` | **imported, not modified** |
| Final models (used only to build UNION) | `mappo_A0_cpu_repro.pth`, `mappo_R2_mc_target.pth` | read-only |
| New driver | `marl/diag/_phase5_risk_trajectory.py` | **new additive file** |
| New output | `SPRINT_7_PHASE5_risk_trajectory_main.json`, `…_parity.json`, `SPRINT_7_PHASE5_run.log` | new |

**State sets are not redefined.** `eval_starts(cfg_A0, 32, 20260825)`,
`random_trajectory(cfg_A0, starts, 31337)`, and
`union_source(trajectory(A0_final), trajectory(R2_final))` were called through the
frozen artifact's own helpers with its own default parameters. UNION is built from the
**final** A0 and R2 actors and is therefore a fixed constant across all 76 checkpoints;
it was not rebuilt per checkpoint. Only the actor being scored varies.

**R2's own per-checkpoint on-policy state distribution was not used**, so a change in
`Delta_EDGE` cannot be a change in which states were visited.

Already-recorded per-update training quantities (`entropy`, `clip_frac`, `approx_kl`,
`explained_var`, `decision_frac`, `critic_loss`, `adv_mean`, `adv_std`, `value_mean`,
`actor_loss`) were **read** from the Phase 4 manifest, not recomputed.

Runtime 240 s, `--device cpu`, `--boot 5000`.

---

## 3. Integrity result

Snapshot taken **before** any Phase 5 action into `_PHASE5_integrity/` (5 before-manifests:
code 52, artifacts 156, trajectory 78, inputs 3, prior manifests 69), regenerated after and
diffed by filename and by hash.

**Verdict: 5 additions, 0 modifications, 0 removals.**

| scope | before | after | added | modified | removed |
|---|---|---|---|---|---|
| code (`marl/**/*.py`) | 52 | 53 | `diag/_phase5_risk_trajectory.py` | **0** | 0 |
| artifacts (`saved_models/marl/*`) | 156 | 160 | `SPRINT_7_PHASE5_REPORT.md`, `…_risk_trajectory_main.json`, `…_risk_trajectory_parity.json`, `SPRINT_7_PHASE5_run.log` | **0** | 0 |
| trajectory checkpoints | 78 | 78 | — | **0** | 0 |
| upstream inputs | 3 | 3 | — | **0** | 0 |

Plus 4 new `_PHASE5_integrity/*_after.md5` manifests. Nothing else was written.

All 7 production files are byte-identical: `config.py 3adca5da…`, `env.py 4e565282…`,
`evaluate.py 79c8e984…`, `mappo.py 8614d016…`, `risk_provider.py 02219832…`,
`rollout.py 3a829646…`, `train.py 8ef42424…`. The 9 R2 control artifacts are
byte-identical (`mappo_R2_mc_target.pth 00da5284…`, `…_best.pth cbf53ca7…`,
`…_config.json cd36140b…`, `…_history.csv 42b44b4a…`, `…_updates.csv 0181e2e9…`, plus the
4 eval files). All 76 trajectory checkpoints are byte-identical.

### Parity self-test (reproduction, not a new criterion)

u075 is the end of training and must therefore reproduce the frozen R2 endpoint row
field-for-field. Same helper code, same state sets, same bootstrap seeds (5/6/11), same
`--boot 5000`.

**32 / 32 fields PASS. 20 of them are bit-exact (|d| = 0.000e+00), including the
bootstrap `se_cluster` and `z`.**

| | RANDOM | UNION |
|---|---|---|
| `n_decision` / `n_highrisk` / `truncated` | 4975 / 216 / 26 ✓ | 74237 / 3592 / 17 ✓ |
| `risk_response_high_minus_low` | +0.2024195014 ✓ | +0.1578811413 ✓ |
| `p_edge_risk_lt_02` → `_ge_06` | 0.1424954282 → 0.3449149296 ✓ | 0.1422937944 → 0.3001749356 ✓ |
| `risk_response_cluster.se_cluster` / `z` | 0.0262086746 / +7.7233780241 ✓ | 0.0067909702 / +23.2486870762 ✓ |

The instrument therefore measures the same thing at u075 that Sprint 7 has been anchored
to all along, which is what licenses reading u000–u074 on the same axis.

---

## 4. R2 risk-response trajectory, u000 → u075

### 4.1 The headline shape

`Delta_EDGE` (both fixed state sets), 5 blocks of 15 updates:

| block | u000–014 | u015–029 | u030–044 | u045–059 | u060–075 |
|---|---|---|---|---|---|
| RANDOM Δ_EDGE | +0.0396 | +0.1125 | +0.1405 | +0.1913 | +0.2056 |
| UNION Δ_EDGE | +0.0194 | +0.1127 | +0.1376 | +0.1441 | +0.1668 |

Four regimes, identical on both sets:

1. **u001–u007 — no signal.** RANDOM Δ ∈ [−0.023, +0.023], mean +0.0060, 3 sign flips.
   UNION Δ ∈ [−0.032, +0.025], mean −0.0053, 2 sign flips. The response is absent and
   unsigned; it is not merely small.
2. **u008–u023 — emergence.** RANDOM +0.0572 → +0.1291; UNION +0.0383 → +0.1413. This
   is where most of the signal is built. u008 is the first update at which both sets are
   positive together and stay positive.
3. **u024–u042 — plateau with a real dip.** RANDOM oscillates 0.1071–0.1536 (a −0.047
   excursion from u028 to u037); UNION 0.1283–0.1581. No net gain across 19 updates.
4. **u043–u062 — second, smaller rise, then the peak.** Both sets peak at **u062**
   (RANDOM +0.2102, UNION +0.1808) — RANDOM's peak is +0.0078 above its endpoint,
   UNION's +0.0230.
5. **u063–u075 — monotone decay to the endpoint.** RANDOM 0.2102 → 0.2024,
   UNION 0.1808 → 0.1579.

The endpoint value that every prior Sprint 7 instrument measured is therefore **not the
maximum of R2's own trajectory** on either set. On RANDOM the post-peak decay is −0.30
cluster SE (not resolvable). On UNION it is **−3.38 cluster SE** (resolvable). So on the
tighter of the two instruments the risk response measurably *declines* over the last 13
updates.

### 4.2 The decomposition — and this is the substantive finding

`Delta_EDGE` is by definition `p_hi − p_lo`. Both terms are already-recorded fields.
Splitting the growth:

| | π(EDGE\|risk≥0.6) | π(EDGE\|risk<0.2) | Δ_EDGE |
|---|---|---|---|
| RANDOM u000 → u075 | 0.4551 → **0.3449** (**−0.1101**) | 0.4385 → 0.1425 (−0.2960) | +0.0165 → +0.2024 |
| share of Δ growth | **−59.2 %** | **+159.2 %** | +0.1859 |
| UNION u000 → u075 | 0.4859 → **0.3002** (**−0.1857**) | 0.4852 → 0.1423 (−0.3429) | +0.0007 → +0.1579 |
| share of Δ growth | **−118.2 %** | **+218.2 %** | +0.1572 |

**π(MIGRATE_EDGE | high risk) falls monotonically-in-block over training and never
exceeds its randomly-initialised value: 0 of 76 checkpoints on RANDOM, 1 of 76 on UNION
(u002, +0.0125).** Its maximum is at u000 (RANDOM) / u002 (UNION).

Every unit of `Delta_EDGE` growth comes from suppressing low-risk EDGE faster than
high-risk EDGE. Where the mass goes (four-action decomposition, u000 → u075):

| RANDOM | STAY | EDGE | CLOUD | PREEMPT |
|---|---|---|---|---|
| risk < 0.2 | 0.4387 → 0.5921 | 0.4385 → 0.1425 | 0.1211 → 0.2652 | 0.0017 → 0.0002 |
| risk ≥ 0.6 | 0.4570 → 0.5229 | 0.4551 → 0.3449 | 0.0879 → 0.1322 | 0.0000 → 0.0000 |

| UNION | STAY | EDGE | CLOUD | PREEMPT |
|---|---|---|---|---|
| risk < 0.2 | 0.4854 → 0.7996 | 0.4852 → 0.1423 | 0.0292 → 0.0581 | 0.0002 → 0.0000 |
| risk ≥ 0.6 | 0.4874 → 0.6562 | 0.4859 → 0.3002 | 0.0267 → 0.0436 | 0.0000 → 0.0000 |

At u000 the actor is untrained and π is close to uniform-over-legal, so 0.44–0.49 is
"coin flip between STAY and EDGE". **R2 ends migrating to edge at high risk less often
than a coin flip and less often than at initialisation, in expectation, on both neutral
state sets.** Migration propensity moves toward STAY at *both* risk levels; it moves
less far at high risk. The CLOUD channel (new here, see §4.4) moves the same way as
EDGE and is also suppressed more at high risk than at low risk, i.e. its risk response
is negative and growing on RANDOM (−0.0332 → −0.1331, ρ_u = −0.828).

### 4.3 The greedy channel disagrees about the high-risk level (reported, not selected)

The mean-probability and greedy/argmax channels agree that the *contrast* grows and that
low risk is suppressed. **They disagree on whether high-risk migration propensity
increases.**

| greedy argmax-EDGE rate, all decision entries | risk < 0.2 | risk ≥ 0.6 | Δ_argmax |
|---|---|---|---|
| RANDOM u000 → u075 | 0.3698 → 0.1755 | **0.0833 → 0.4630** (max 0.5509 @u047) | −0.2864 → +0.2874 |
| UNION u000 → u075 | 0.3944 → 0.1778 | **0.2311 → 0.3163** (max 0.6058 @u002) | −0.1634 → +0.1384 |

`Δ_argmax` starts **negative** on both sets (the untrained actor's argmax picks EDGE
*less* at high risk), first turns positive at u008, and is permanently positive from
u008 on both sets. The high-risk greedy rate exceeds its u000 value in 74/76 checkpoints
on RANDOM but only 39/76 on UNION.

So: in expectation R2 does not acquire high-risk migration; in the deployed greedy policy
it does, on RANDOM strongly and on UNION weakly. The reconciliation is visible in the
saturation series — mass concentrates, so EDGE becomes the argmax in a subset of
high-risk states while STAY dominates hard elsewhere.
`frac_maxp_gt_099` over all decision entries goes 0.0000 → 0.2149 (RANDOM) and
0.0000 → 0.3559 (UNION).

A third temporal landmark falls out of the greedy channel that the on-policy Phase 4
stats could not show: on RANDOM the greedy high-risk EDGE rate is **frozen at exactly
0.1065 from u030 to u042** and the low-risk rate at 0.0809 from u025 to u040, while the
underlying probabilities keep drifting. The greedy policy on neutral states is
therefore locked for ~13 updates in mid-training, then reorganises over u043–u062, then
locks again.

### 4.4 What is new here, and flagged as such

`SPRINT_7_R3_action_channels.json` **does not exist** — the four-action probe exists as
code but its artifact was never written, so the CLOUD/STAY/PREEMPT responses are *not*
pre-existing measurements for any arm. They are computed here (free: the same `P` array
already holds all four columns), stored under `_NEW_not_in_frozen_artifact`, and are
descriptive only. Same for the greedy responses by risk cut, and for
entropy/max-prob over *all* decision entries (the frozen artifact's entropy is over the
high-risk subset only).

**One disambiguation that changes how the CLOUD/PREEMPT numbers must be read.** Action
legality on the fixed sets:

| legal fraction | STAY | MIGRATE_EDGE | MIGRATE_CLOUD | PREEMPT_REROUTE |
|---|---|---|---|---|
| RANDOM, risk ≥ 0.6 | 1.0000 | 1.0000 | 0.2639 | **0.0000** |
| UNION, risk ≥ 0.6 | 1.0000 | 1.0000 | 0.0802 | **0.0000** |
| RANDOM, risk < 0.2 | 1.0000 | 1.0000 | 0.3651 | 0.0067 |
| UNION, risk < 0.2 | 1.0000 | 1.0000 | 0.0879 | 0.0009 |

`PREEMPT_REROUTE` is legal in **exactly 0.00 %** of high-risk decision entries on both
sets. Its zero probability is a **masking fact, not a policy fact** — the policy can be
neither blamed nor credited for it. The effective high-risk action set is
STAY / MIGRATE_EDGE always, plus MIGRATE_CLOUD in 26 % (RANDOM) / 8 % (UNION) of entries.
Any earlier reading of a zero high-risk PREEMPT or a low high-risk CLOUD rate as
policy behaviour must be re-read against this table.

---

## 5. Key temporal landmarks

| Landmark | RANDOM | UNION |
|---|---|---|
| response absent / unsigned | u001–u007 | u001–u007 |
| first update positive and permanently so | u008 | u008 |
| Δ_EDGE 50 % of its own excursion | **u023** | **u020** |
| Δ_EDGE 90 % of its own excursion | u046 | u060 |
| mid-training plateau (no net gain) | u024–u042 | u024–u042 |
| greedy argmax locked on neutral states | u030–u042 (hi), u025–u040 (lo) | u023–u042 |
| second rise | u043–u062 | u059–u062 |
| **peak** | **u062** (+0.2102) | **u062** (+0.1808) |
| decay to endpoint | u063–u075 (−0.30 SE) | u063–u075 (**−3.38 SE**) |

Already-recorded quantities, same axis (from the Phase 4 manifest):

| | 50 % | 90 % | ρ vs update |
|---|---|---|---|
| `clip_frac` | **u002** | **u014** | −0.926 |
| `approx_kl` | u003 | u021 | −0.594 (weak) |
| `entropy` (on-policy) | u012 | u032 | −0.946 |
| `explained_var` | u012 | u041 | +0.902 |
| `entropy` on RANDOM / UNION decision entries | u015 / u016 | u035 / u036 | −0.838 / −0.859 |
| **Δ_EDGE (RANDOM / UNION)** | **u023 / u020** | **u046 / u060** | +0.939 / +0.875 |
| `decision_frac` | u025 | u043 | +0.859 |
| `frac_maxp_gt_099` on RANDOM / UNION | u033 | u057 / u061 | +0.969 / +0.972 |

Excluded from that table as non-trending, with the reason: `p_edge_risk_ge_06`
(ρ = +0.256 / −0.550, extreme at u000/u002 — the correct statement is a level statement,
§4.2, not a crossing); `RANDOM.p_edge_risk_lt_02` (ρ = −0.495, dips to 0.117 @u039 then
recovers); `RANDOM.spearman_risk_vs_p_edge` (crossing test degenerates to u004);
`actor_loss`, `adv_mean` (ρ = +0.359, −0.188).

---

## 6. Which quantities precede and which follow the risk-response change

Ordering only. Update number is temporal position, not causal proof, and the 75 updates
are not independent replicates. `n = 1` arm.

**Precede Δ_EDGE (finish moving before Δ_EDGE is half-formed).**
`clip_frac` is 90 % done at **u014** and `approx_kl` at u021, while Δ_EDGE reaches only
50 % at u020–u023. On-policy `entropy` is 90 % done at **u032** and the critic's
`explained_var` at u041, while Δ_EDGE is 90 % done at u046 (RANDOM) / u060 (UNION).
So the trust region has effectively closed and the entropy collapse has essentially
completed **before** the risk response finishes assembling. The response is not built by
a wide-exploration phase; most of it accrues while `clip_frac` is already ≤ 0.03 and
falling.

**Coincide with Δ_EDGE.** The emergence window u008–u023 overlaps the steepest part of
the entropy collapse (0.860 → 0.397) and the fastest critic improvement
(`explained_var` 0.027 → 0.541). Nothing here separates "the policy learned to
discriminate risk" from "the policy sharpened and the sharpening happened to be
risk-correlated"; both series move together in that window.

**Follow Δ_EDGE.** Saturation. `frac_maxp_gt_099` on the fixed sets is 50 % done at
u033 and 90 % at u057–u061, both *after* Δ_EDGE's 50 % point and after the mid-training
plateau began. `decision_frac` (50 % at u025, 90 % at u043) also lags. So the policy's
collapse onto near-deterministic actions is largely **subsequent** to the risk response
appearing, not prior to it.

**The one exact coincidence, stated as coincidence.** Δ_EDGE peaks at **u062** on both
independent state sets, and u062 is the first update at which `clip_frac` is identically
zero (last non-zero u061). After u062 the trust region never re-opens and Δ_EDGE never
recovers. This is a two-instrument agreement on the *ordering*; with one arm it cannot
be a causal claim, and the RANDOM-side decay is inside its own cluster SE.

---

## 7. Comparison with existing A0 / R3 evidence, where equivalent

A0 and R3 have **no trajectory checkpoints**, so only their endpoints can enter. Those
endpoints were measured on these same two fixed sets by the same code, so the comparison
is valid without manufacturing anything.

First update at which R2 reaches, and permanently holds, each existing arm's final Δ_EDGE:

| beaten arm | endpoint Δ (RANDOM / UNION) | R2 first ≥ | R2 permanently ≥ |
|---|---|---|---|
| A0 | −0.0264 / +0.0191 | u000 / u002 | u000 / u008 |
| A2 | +0.0398 / −0.0248 | u008 / u000 | u008 / u002 |
| A1 | +0.1375 / +0.0628 | u026 / u012 | u041 / u018 |
| A3 | +0.1322 / +0.0933 | u026 / u021 | u040 / u021 |
| **R3** | **+0.1575 / +0.1242** | **u042 / u021** | **u042 / u021** |

R2 surpasses A0 within the first two updates and surpasses **R3's entire 600-episode
outcome by update 21–42, i.e. in 17–56 % of R2's own training.** R2's final margin over
R3 is +0.0449 (RANDOM) / +0.0337 (UNION).

This bears directly on the recorded regime-selective asymmetry between R2 and R3: the
R2-vs-R3 gap in Δ_EDGE is a gap R2 had already opened by roughly a fifth to a half of
the way through training, in the same window in which — per §4.2 — Δ_EDGE growth is
almost entirely low-risk suppression. Whatever separates R2 from R3 on this metric is
therefore located in the early-to-middle suppression phase, not in the endpoint.

No other cross-arm comparison is made. A1/A2/A3 have endpoints only and are listed for
scale, as in the frozen artifact.

---

## 8. What this rules out

1. **"R2's risk response emerges late, or is a terminal artifact."** Ruled out. It is
   absent and unsigned for u001–u007, half-formed by u020–u023, and past its 90 % point
   by u046–u060. It also is *not* maximal at the endpoint — u062 is.
2. **"The endpoint reading of Δ_EDGE is R2's best reading."** Ruled out. On UNION the
   endpoint is 3.38 cluster SE below R2's own peak. Endpoint readings understate R2 on
   its own primary metric.
3. **"R2 learned to migrate more at high risk."** Ruled out **in expectation**:
   π(EDGE | risk ≥ 0.6) ends 0.110 (RANDOM) / 0.186 (UNION) *below* its
   randomly-initialised value and never exceeds it (0/76 and 1/76 checkpoints). Not
   ruled out in the greedy channel, where the high-risk argmax-EDGE rate rises
   0.083 → 0.463 (RANDOM) / 0.231 → 0.316 (UNION); §4.3 reports both.
4. **"Δ_EDGE measures acquisition of high-risk migration."** Ruled out as a reading of
   the metric. 159 % (RANDOM) / 218 % (UNION) of its growth is low-risk suppression,
   partly offset by high-risk suppression. A larger Δ_EDGE is, arithmetically, mostly a
   more suppressed low-risk policy. This is an exact decomposition of the metric's own
   definition, not a new metric.
5. **"Softmax saturation precedes and therefore explains the risk response."** Ruled out
   as an ordering. Saturation on the fixed sets lags: `frac_maxp_gt_099` 50 % at u033,
   90 % at u057–u061, versus Δ_EDGE 50 % at u020–u023. Saturation follows.
6. **"A wide-exploration phase built the risk response."** Ruled out as an ordering.
   `clip_frac` is 90 % decayed by u014, before Δ_EDGE is half-formed.
7. **"PREEMPT_REROUTE's zero high-risk probability is a policy failure."** Ruled out.
   The action is legal in 0.00 % of high-risk decision entries on both sets — a masking
   fact. Any prior reading of it as behaviour is void.
8. **Nothing here disturbs** the mechanism findings that were already closed or
   untouchable: paired-advantage variance, `rollout_episodes` as a pure variance
   intervention, simple gradient dilution, the per-state-offset mechanism, or the
   critic-target sign. No quantity measured in this phase speaks to any of them.

---

## 9. What remains unknown

1. **Why π(EDGE | high risk) declines at all.** The trajectory shows *that* it declines
   monotonically-in-block from u000, and that it does so more slowly than the low-risk
   branch. It does not show why the gradient pushes high-risk EDGE down. Nothing
   measured here is risk-conditioned *and* gradient-side.
2. **Which channel is the behaviourally relevant one.** §4.3 leaves an unresolved
   direction conflict between the expectation and the greedy/argmax readings of
   high-risk EDGE. Deployment uses greedy; the objective uses the expectation. This
   phase cannot adjudicate that, and no measurement here was designed to.
3. **Whether the u062 coincidence is anything.** One arm, one run. Two state sets agree
   on the ordering; that is not replication.
4. **The mid-training plateau u024–u042.** The greedy policy is frozen on neutral states
   while the probabilities drift and the critic keeps improving. Unexplained.
5. **Everything about A0 and R3 between their endpoints.** They have no trajectory
   checkpoints. Every temporal statement in this report is about R2 alone.

The single most important missing observable, if one is ever wanted, is a
**risk-conditioned gradient-side quantity** — every Sprint 7 instrument to date, this
one included, measures the *policy*, so "the risk response is differential suppression"
is established while "what makes the update suppress high risk too" is not. Stated as
the gap; deliberately **not** designed into an experiment here.

---

## 10. Is another training run justified?

**No — not on this evidence, and this phase did not find a question that requires one.**

Reasons:

- The stated Phase 5 question is answered, on two independent fixed state sets, with a
  32/32 bit-exact parity check tying the instrument to the frozen Sprint 7 endpoint.
- The answer reframes the objective rather than extending it: R2's "self-healing signal"
  is **differential suppression of migration**, not acquisition of high-risk migration.
  That is a statement about what the pre-registered metric measures, and it is derivable
  from artifacts already paid for. A new arm optimising the same metric would move the
  same low-risk-suppression term.
- Both remaining open items in §9 that could motivate a run (why high-risk EDGE declines;
  expectation vs greedy) are **measurement** gaps, not arm-comparison gaps. Neither is
  addressed by adding a training condition.
- Rules 10 and 11 apply: a negative/reframing result is a valid stopping point, and no
  causal claim is being made that would need ≥ 3 seeds.

**Recommendation: stop Sprint 7 here.** No R4 is designed or proposed. No new success
criterion, GO/NO-GO threshold, or metric is introduced by this report; every number above
is either an already-recorded quantity, a field of the frozen artifact, or an exact
algebraic decomposition of the pre-registered metric's own definition.
