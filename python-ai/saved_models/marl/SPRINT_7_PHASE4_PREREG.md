# Sprint 7 — PHASE 4 PRE-REGISTRATION

**Rung name:** `R2-TRAJ` (checkpoint-instrumented replication of R2)
**Written:** 2026-08-30, **before** the driver was implemented and **before** any run.
**Status at time of writing:** nothing trained. No production source modified. No R4 designed.

Everything below is fixed at the moment of writing. Per RULE 3 and RULE 9, no threshold,
metric or reference value in this document may be changed after seeing a result. Where a
criterion has a numeric reference value, that value was measured from the frozen R2 artifacts
during Phase 4 preparation and is quoted here so that it cannot be re-derived post hoc.

---

## 1. Scientific question

> **WHEN during the 75-update R2 training trajectory does the behavioural / actor / critic
> divergence emerge, and can its onset be localized to a specific update interval?**

This question is currently unanswerable. Phase 0/1 established that the artifact corpus
contains exactly one policy snapshot per arm (`R3_batch32_best.pth`, at update 45, is the sole
mid-training checkpoint in the repository), that R2's `_best` and final checkpoints hold
**identical** weights and optimiser state, and that every mechanism instrument in the corpus is
a single-endpoint measurement which orders the arms backwards against the pre-registered
behavioural metric (20 readings, 20 negative, mean Spearman −0.900). The blocker is a **missing
observable**, not a missing hypothesis.

## 2. Exact hypothesis being tested

This rung tests **no mechanism hypothesis**. It tests one reproducibility hypothesis:

> **H(repro).** Re-running R2's exact configuration on this machine reproduces R2's training
> computation bit-for-bit, so that checkpoints saved at each update boundary are *provably* the
> trajectory that produced `mappo_R2_mc_target.pth`.

H(repro) is falsifiable and is falsified by any failure of the bit-identity criteria in §13.
If H(repro) is **false**, that is a reproducibility defect in the Sprint 7 corpus and is itself
the result of this rung — it would mean no previously reported single-seed number in Sprint 7 is
re-derivable, and the corpus would need re-grounding before any further causal work.

**Explicitly NOT claimed by this rung.** Temporal precedence is not causality. Localizing the
onset of a divergence to an update interval does **not** identify a cause, does not rank
mechanisms, and does not license an intervention. This rung produces an observable; a later,
separately pre-registered rung must use it.

## 3. Exact R2 configuration

Verified during Phase 4 preparation by resolving the planned command line through the
production `marl.train.apply_args(Sprint6Config(), parse_args(argv))` and diffing the resulting
`cfg.to_dict()` against the `config` block of `mappo_R2_mc_target_config.json`:

**91 fields compared. 91 keys present on both sides. Exactly 1 field differs: `train.tag`.**

| field | value |
|---|---|
| `critic_target` | `mc` |
| `rollout_episodes` | 8 |
| `episodes` | 600 |
| `seed` | 20260818 (`cfg.train.seed` = `cfg.env.seed`) |
| `device` | `cpu` |
| `gamma` | 0.999 |
| `gae_lambda` | 0.995 |
| `entropy_coef` | 0.02 |
| `clip_eps` | 0.2 |
| `value_clip_eps` | 0.2 |
| `value_coef` | 0.5 |
| `max_grad_norm` | 0.5 |
| `lr_actor` | 7e-4 |
| `lr_critic` | 1e-3 |
| `normalise_advantages` | True |
| `anneal_lr` | True |
| `episode_steps` | 400 |
| `risk_source` | `oof` |
| `ppo_epochs` | 4 |
| `minibatches` | 4 |
| `actor_hidden` / `critic_hidden` | [128,128] / [256,256], `separate_actors=True` |
| `start_frac_lo`, `start_frac_hi` | 0.0, 0.7 (set by `train.main`, `TRAIN_FRAC`) |
| dims | `n_agents` 10, `obs_dim` 48, `state_dim` 489 |

**Correction to the Phase 0 report, recorded rather than silently fixed.**
`SPRINT_7_PHASE0_RECONSTRUCTION.md` §1.3 lists `critic_target "mc"` among "frozen production
values". That is wrong: `config.py:505` still defaults to `"lambda"` (A0 behaviour), exactly as
`train.py`'s own help text states. `"mc"` is R2's **flag-selected** value, not the production
default. `--critic-target mc` is therefore **mandatory** on the command line; omitting it would
silently run an A0-target arm. The Phase 0 §1.3 error is noted here and the report is left
unedited.

**Second correction.** Phase 0 §1.1 says "`R2_best` is **bit-identical** to `R2`". The
*files* are not: `mappo_R2_mc_target.pth` is 5,207,250 bytes / md5 `00da5284…` and
`mappo_R2_mc_target_best.pth` is 5,208,669 bytes / md5 `cbf53ca7…`. What is identical is the
**payload**: actor (60 tensors), critic (7 tensors), `opt_actor`, `opt_critic` and `mappo_cfg`
all hash equal. The files differ only in `extra` (`kind` "best" vs "final", `mean_reward`
12.653897495678393 vs 4.6830271569342585) and in the zip archive stem (see §13). Both were
written at `episode 600`, i.e. R2's best update **was** its last update — which is why R2 has
only one distinct policy snapshot. The substantive Phase 0 claim stands; the word
"bit-identical" was imprecise.

## 4. Expected number of updates

`n_updates = max(1, episodes // rollout_episodes) = 600 // 8 = ` **75**.

Confirmed against the frozen artifact: `mappo_R2_mc_target_updates.csv` has **76 lines**
(1 header + 75 update rows), first row `update=1, episode=8`, last row `update=75, episode=600`.
`mappo_R2_mc_target_history.csv` has **601 lines** (1 header + 600 episodes).

## 5. Exact checkpoint schedule — 76 checkpoints

| index | when written | contents |
|---|---|---|
| `u000` | on entry to update 1, **before** `MAPPO.update` runs | the untrained initial state |
| `u001` … `u075` | immediately after each `MAPPO.update` returns | post-update state |

**`u000` is recoverable, and this was verified offline before the run.** Parameters change only
inside `MAPPO.update`; rollout (`act`, `value`) is under `@torch.no_grad()` and touches no
optimiser. `train.py:211` computes `frac = 1.0 - (1-1)/75 = 1.0` for update 1, and
`set_lr_scale(1.0)` is an exact identity on the learning rates (verified: lr stays 7e-4 /
1e-3). Therefore the state captured at `u000` is exactly the state `train.py` reaches before
its first update.

**AMENDED 2026-08-30, before the run, by the smoke test — that state is NOT a bare
construction.** `train.py:129` calls `MAPPO.assert_actors_independent()`, which perturbs actor
0's first weight matrix with `first.add_(1.0)` and then restores it with `first.sub_(1.0)`
(`mappo.py:530-537`). In float32 that round trip is **not exactly invertible**: at |w| ≈ 0.10
the ULP is ≈1.5e-8, but inside [1, 2) it is 2⁻²³, so roughly three low-order bits are rounded
away. Measured for seed 20260818: **5,476 of 6,144 elements move, max |Δ| = 5.96e-08**, and the
operation is idempotent thereafter (a second and third application change nothing). Two
consequences, both recorded here rather than silently fixed:

1. Every checkpoint in the Sprint 7 corpus carries this ~1e-8 perturbation on agent 0's first
   layer. It is far too small to matter behaviourally, but it is a real, permanent asymmetry
   between agent 0 and agents 1–9 that exists in every arm. Noted, not claimed as relevant.
2. The `u000` expected value below had to be recomputed against the **post-probe** state.

The post-probe state is seed-determined and was measured **in advance** — construct
`MAPPO(10, 48, 489, cfg.mappo, device="cpu", seed=20260818)` under
`torch.manual_seed(20260818)`, then call `assert_actors_independent()` once, and hash with
`marl/diag/_phase4_verify.py::weight_md5` (actor then critic `state_dict`s, keys sorted, each
tensor's bytes prefixed by `name + key`):

```
u000 expected weight md5 (POST-probe, weight_md5 convention) = ab714064bdf1ac56daabf5c163c92215
        same convention, PRE-probe (i.e. a bare construction) = 3bfda0327eaf93a16ff7c4d195e37d9f
```

The value published in the first draft of this document, `cfff208ba73247493e66f5eb97649bc8`,
is **SUPERSEDED**. It was computed by an ad-hoc snippet against a bare (pre-probe)
construction, and its hashing convention could not be reproduced by any of four candidates
tried (`weight_md5`; no key-name prefixes in `state_dict` order; `parameters()` only, which
excludes the critic's constant `eye` buffer; no key-name prefixes with sorted keys). Whatever
convention produced it, it was applied to the wrong state, so B7 as first written would have
**failed a faithful replication**. The smoke test caught this before any training was run.
Both replacement values above are pre-registered: computed before the run and independent of
it, and hard-coded in `_phase4_verify.py`.

## 6. This is a replication, not a new treatment

Under RULE 1, the number of manipulated production variables is **zero**. Same seed, same
episode count, same batching, same critic target, same device, same 91-field resolved config
save for the output tag. There is no treatment arm and no control arm: the *existing* R2
artifacts are the control, and the replication is compared against them.

Consequently this rung has **no behavioural success criterion**. It cannot succeed or fail at
"risk-aware migration". It can only succeed or fail at *faithfulness*.

## 7. The only intended difference

> After each of the 75 `agent.update(buf)` calls (and once on entry to update 1), an additional
> checkpoint is written to a new namespace.

Nothing else. Not the RNG handling, rollout generation, environment stepping, optimiser
behaviour, learning rates, annealing, PPO epochs, minibatch count, advantage computation,
critic target, reward, architecture, evaluation, checkpoint loading, seeds, episode ordering,
tensor operations, or logging calculations. The minibatch-tail issue (RULE 7) is **not** touched.

## 8. No production-code modification

None of `marl/mappo.py`, `marl/train.py`, `marl/env.py`, `marl/rollout.py`, `marl/config.py`,
`marl/evaluate.py`, `marl/risk_provider.py` is edited. The stale comment at `mappo.py:447`
(GAE horizon "~20 steps", correct only for the retired λ=0.95) remains unfixed, as in Phase 0.

**Interception point.** `train.py` has no per-update hook: the update boundary is
`stats = agent.update(buf)` at `train.py:213`, inside `main`, and the only checkpoints it writes
are `{tag}_best.pth` (line 232, gated on `mean_r > best_mean`) and `{tag}.pth` (line 252). The
smallest safe interception is therefore to **rebind `MAPPO.update` in the driver process** to a
wrapper that delegates to the original method and then saves, and to call the unmodified
`marl.train.main(argv)`. The driver additionally reuses `train.parse_args` / `train.apply_args`
to resolve the same config for its own bookkeeping, so no training logic is reimplemented.

**Why this is safe for determinism.** `MAPPO.save` (`mappo.py:475–489`) is a bare `torch.save`
of four `state_dict()`s plus static metadata — no timestamp, no wall clock, no RNG draw.
Measured during preparation: `torch.save` leaves `torch.get_rng_state()` and
`np.random.get_state()` **unchanged**. The driver additionally asserts this at runtime around
every one of the 76 saves and aborts if it is ever violated, so the claim is enforced, not
assumed. All driver bookkeeping that could conceivably touch a global RNG happens *before*
`train.main` calls `torch.manual_seed` / `np.random.seed`, which reset the global generators.

## 9. No new success criteria

The behavioural metric remains `Δ = π(EDGE | risk ≥ 0.6) − π(EDGE | risk < 0.2)` on the frozen
policy-independent RANDOM and UNION state sets, constructed exactly as recorded and **not**
redefined. No metric is introduced by this rung. The faithfulness criteria in §13 are integrity
checks against frozen artifacts, not new scientific metrics.

## 10. No R4 design

No intervention, mechanism, or treatment arm is proposed here. Deciding what to do with the
trajectory is out of scope for Phase 4 and Phase 5.

## 11. No inference of causality from temporal ordering alone

Written in advance so it cannot be relaxed later: if the trajectory shows quantity A diverging
at update *i* and quantity B at update *j > i*, that is **not** evidence that A causes B. The
divergence phase already recorded a precedence pattern of this kind (critic `explained_var`
onset 34, reward onset 47, `decision_frac` onset 25) and explicitly declined to call it causal.
The permitted use of this trajectory is: (a) to *calibrate* instruments against behaviour as a
function of training time, which §3 of the Phase 0 report shows no endpoint instrument passes;
(b) to *localize* the arg-max collapse currently bracketed to updates 45–75; (c) to generate
hypotheses that a later rung must test by manipulation.

## 12. Integrity requirements

Before-snapshot taken **before** any file was created this phase, in
`saved_models/marl/_PHASE4_integrity/`:

| manifest | root | entries |
|---|---|---|
| `SPRINT_7_P4_code_before.md5` | `python-ai/marl` | 48 `.py` |
| `SPRINT_7_P4_artifacts_before.md5` | `saved_models/marl` (depth 1) | 147 |
| `SPRINT_7_P4_checkpoints_before.md5` | `saved_models/marl` (depth 1) | 23 `.pth` |
| `SPRINT_7_P4_inputs_before.md5` | repo root | 3 |
| `SPRINT_7_P4_prior_manifests.md5` | `saved_models/marl` (depth 2) | 46 `.md5` |

Requirements:

1. **Production source unchanged.** All 48 code hashes must re-verify after the driver is
   written and after the run (the driver and smoke test are *new* entries, checked as additions).
2. **No checkpoint overwritten.** All 23 pre-existing `.pth` hashes must re-verify after the run.
3. **No artifact overwritten.** All 147 depth-1 artifact hashes must re-verify after the run.
   The driver refuses to start if any output path it would write already exists.
4. **Additive only.** Every new file must be a new path in a new namespace (§ below).
5. Git is a second, stronger baseline: HEAD `a8df6d9`, and `git status` must show new files as
   untracked additions with **zero** modified or deleted tracked paths.

**Namespace — nothing existing may be touched.** Verified: no path matching `R2_traj` or
`trajectory` exists in `saved_models/marl`.

| new path | role |
|---|---|
| `saved_models/marl/R2_trajectory/R2_trajectory_u000.pth` … `_u075.pth` | 76 trajectory checkpoints |
| `saved_models/marl/R2_trajectory/SPRINT_7_P4_trajectory_manifest.jsonl` | per-checkpoint manifest, flushed per update |
| `saved_models/marl/R2_trajectory/SPRINT_7_P4_trajectory_summary.json` | run summary |
| `saved_models/marl/R2_traj_repro.pth` / `_best.pth` / `_config.json` / `_history.csv` / `_updates.csv` | the run's own artifacts, tag `R2_traj_repro` |
| `marl/diag/_phase4_r2_trajectory.py` | the additive driver |
| `marl/diag/_phase4_smoke.py` | the smoke test |
| `marl/diag/_phase4_verify.py` | the B1–B10 verifier (read-only; writes only to a temp dir) |
| `marl/diag/_phase4_equiv.py` | the Phase 4.1 exact-equivalence verifier for B1a–B1e (read-only; writes only to a temp dir) |
| `saved_models/marl/SPRINT_7_P4_SMOKE_REPORT.json` | the smoke test's result, the one artifact it leaves behind |
| `saved_models/marl/SPRINT_7_P4_EQUIV_SELFTEST.txt` | captured console output of `_phase4_equiv --inspect --self-test` (written by shell redirection, not by the module) |
| `saved_models/marl/_PHASE41_integrity/SPRINT_7_P41_*.md5` | the Phase 4.1 integrity manifests (RULE 12); the `_PHASE4_integrity/*_after` set is left untouched and serves as Phase 4.1's *before* state |

The five protected R2 artifacts — `mappo_R2_mc_target.pth`, `_best.pth`, `_config.json`,
`_history.csv`, `_updates.csv` — are **not** in that list and are never opened for writing.

**Disk.** 5,207,250 bytes per checkpoint × 76 = **377.4 MiB**. 83 GB free. Note that
`.gitignore:14` has `# *.pth` **commented out**, so `.pth` files are tracked by this repository;
whether to commit 377 MiB of trajectory checkpoints is a decision for the operator, not for the
driver, and the driver makes no change to `.gitignore`.

## 13. Bit-identity requirements

**A mechanical finding that changes how this must be stated.** `torch.save` is byte-deterministic
for a given payload, but the byte stream depends on the **file's stem**, because
`PyTorchFileWriter` uses the basename as the zip archive prefix. Measured during preparation:
saving the same object to three different directories under the *same* basename
`mappo_R2_mc_target.pth` gives one md5, three times; saving it under different basenames gives
different md5s. And — decisively — **loading `mappo_R2_mc_target.pth` and re-saving it under the
same basename reproduces its exact md5 `00da5284504cee8c1687866b315d6194`**, so `torch.load` →
`torch.save` is a lossless bit-exact round trip on this file.

Consequently: whole-file md5 equality between the replication's final checkpoint and
`mappo_R2_mc_target.pth` is **impossible** without also matching the tag, because
`extra["config"]` embeds `train.tag` (and `train.out_dir`) and the filename sets the archive
prefix — and matching the tag would overwrite the protected artifact, which §12 forbids. The
difference is confined to **exactly two strings plus the file stem**, and that is provable rather
than asserted, via the round-trip property above.

> **AMENDED 2026-08-30, before the run (Phase 4.1).** The statement above was correct but
> under-verified: it asserted that the difference is "confined" to the tag and the stem without
> measuring how far either actually reaches, and it tested the claim only at whole-file md5
> level. B1 is therefore **strengthened**, not weakened, into four sub-criteria B1a–B1d plus a
> new B1e, backed by a measured decomposition of the two causes. The original B1 survives
> verbatim as **B1d**. Verifier: `marl/diag/_phase4_equiv.py`; audit log
> `SPRINT_7_P4_EQUIV_SELFTEST.txt`; see §20.

**MEASURED BLAST RADIUS OF THE TWO NORMALISED ITEMS** (read-only probes on the existing R2 file,
2026-08-30, before the run). `mappo_R2_mc_target.pth` is a zip archive of **271 members**, of
which **265 are tensor storages** — exactly one per tensor in the unpickled graph, which has
**476 leaves and 93 container nodes and 0 values of unrecognised type**.

| cause | what it changes | what it provably cannot change |
|---|---|---|
| **N1** the tag string | `data.pkl` (by exactly the tag's length delta, −5 bytes for `R2_traj_repro`) and `.data/serialization_id` (a content hash of the members) | **0 of the 265 tensor storages** |
| **N2** the archive stem | member **names** only. With an identical payload under two different stems, all 271 members have byte-identical *content*, including `data.pkl` and `serialization_id`; the file size shifts by `n_members ×` the stem length delta (271 × 5 = 1,355 bytes), and an equal-length stem gives an identical size with different bytes | any member's content |

The two causes are **disjoint**, and neither can reach a single byte of a weight, an optimiser
moment, or any scalar. `tag` also reaches nothing numeric in the source: across the seven
production files it occurs only as the field declaration (`config.py:526`), argparse plumbing
(`train.py:68,93`), one local alias (`train.py:121`), five output-path f-strings
(`train.py:142,143,233,253,256`) and a default checkpoint path in `evaluate.py:85,189`. It is
never passed to `MAPPO`, `DTMarlEnv`, the rollout, or any seed. (Every other grep hit for the
substring `tag` in `mappo.py` / `env.py` / `config.py` is inside the word "advan-**tag**-e".)

**THE ACCEPTANCE RULE.** B1 = PASS **only if** the trajectory's final checkpoint is bit-identical
to `mappo_R2_mc_target.pth` after normalising **only** (N1) `extra.config.train.tag` and (N2) the
archive filename/stem. Everything else must match **exactly, with no tolerance of any kind**.
All four sub-criteria must pass, and B2–B10 remain independent and must also pass.

**B1a — container, raw, un-normalised.** Compare the two `.pth` files member by member as **raw
bytes**. Require: member names identical after stripping the single stem segment, and the set of
members whose content differs is a **subset of `{data.pkl, .data/serialization_id}`** — i.e. all
**265 tensor storages byte-identical with no normalisation applied to them at all**. Any other
differing member fails.

**B1b — container, normalised.** Substitute the tag back, resave under basename
`mappo_R2_mc_target.pth`, and require **0 of 271 members differ**.

**B1c — structural, exact.** Flatten both unpickled graphs and compare every entry exactly:
container subtype (`OrderedDict` vs `dict`), **key order**, key types, sequence length and type
(`tuple` vs `list`), tensor dtype / shape / stride / contiguity / numel / `requires_grad` / **raw
bytes**, float **bit patterns**, and int/bool/str/bytes/None identity. Require **0 differences**,
**0 leaves of unrecognised type**, and **exactly one** allowlisted leaf accounted for. Compared
this way: all 60 actor tensors, all 7 critic tensors (including the constant `eye` buffer), both
optimiser `state` dicts (60 + 6 entries × `exp_avg`, `exp_avg_sq`, `step`), both
`param_groups[0]` (12 fields each, including `lr`, `betas`, `eps`, `params`), all 17 `mappo_cfg`
fields, `n_agents` / `obs_dim` / `state_dim`, `extra.episode` / `extra.mean_reward` /
`extra.kind`, and **95 of the 96 config leaves**. **There is no target-critic `state_dict` in
this schema** — this MAPPO has no target critic; the 9 top-level keys are `actor`, `critic`,
`opt_actor`, `opt_critic`, `mappo_cfg`, `n_agents`, `obs_dim`, `state_dim`, `extra`, and an
unexpected tenth key would fail B1c as a node difference.

**No tolerances, and raw bytes rather than `torch.equal`.** `torch.equal(nan, nan)` is `False`
and `torch.equal(0.0, -0.0)` is `True`; both are wrong for a bit-identity claim. Raw-byte
comparison is strictly stronger **in both directions**, and self-test case E0 proves it on those
two exact cases.

**B1d — whole file** (this is B1 as originally written, unchanged). Load the replication's final
`R2_traj_repro.pth`; substitute `extra["config"]["train"]["tag"] → "mappo_R2_mc_target"` and
`extra["config"]["train"]["out_dir"] →` R2's recorded `out_dir`; `torch.save` to a scratch path
whose basename is `mappo_R2_mc_target.pth`. **Require md5 == `00da5284504cee8c1687866b315d6194`.**

**B1e — the best checkpoint, same three layers.** Apply B1a/B1b/B1c/B1d to
`R2_traj_repro_best.pth` vs `mappo_R2_mc_target_best.pth`. Pre-registered reference: literal md5
`cbf53ca75d36ed908a974da2983df3ed` at 5,208,669 bytes; **normalised md5
`4937a745120601ff79f3634b4b4b5d71`** (the normalised form is the comparable quantity because the
`_best` stem is 5 characters longer). Measured 2026-08-30, before the run.

*Printed criterion names.* `_phase4_equiv --tag <tag>` emits the final checkpoint's four layers as
`B1a B1b B1c B1d` and the best checkpoint's as `B1ea B1eb B1ec B1ed`; B1e passes iff all four of
the latter pass. `_phase4_verify --tag <tag>` independently re-computes the whole-file layer and
also emits it as **B1d** — the two modules must agree, which is a deliberate redundant check.

**The substitution must be a MINIMAL COPY** — only the four dicts on the path
(`ck`, `extra`, `config`, `train`) may be rebuilt; every other object, including every tensor
and every string elsewhere in the config, must be passed through **by identity**. Established
while building the harness: a `json.loads(json.dumps(...))` deep copy of the config produced an
`==`-equal config whose pickle was **320 bytes larger**, because pickle memoises repeated string
objects and a deep copy destroys the shared identities. That version failed the self-test.
In practice `out_dir` is identical for both runs (there is no `--out-dir` flag, so both take the
config default), so the substitution reduces to a single string; the code substitutes `out_dir`
only if it differs.

**WHY THE NORMALISATION CANNOT HIDE A REAL TRAINING DIFFERENCE.** Six independent reasons, all
verified before the run:

1. The allowlist is **one literal leaf path** — `extra.config.train.tag`, i.e. **1 of 476 leaves
   (0.21%)** — not a prefix, not a pattern, not a type rule.
2. It applies **only** when the leaf is present on **both** sides and **both** are `str`. A tag
   that vanished, changed type, or moved falls through and is reported as a difference.
3. The **265 tensor storages are compared with no normalisation whatsoever** (B1a), before any
   substitution is performed.
4. The normalised count is reported alongside the total leaf count, and B1c additionally requires
   the allowlisted leaf to be *accounted for* — so "no differences" cannot be achieved by the
   leaf being absent.
5. Self-test case **E15 changes the tag AND a weight by 1 ULP together** and is still detected —
   normalisation does not mask a co-occurring substantive change.
6. `tag` is inert in the source (table above): it never reaches the environment, the networks,
   the optimiser, or any seed.

**Harness self-tests (must pass first; both DID pass on 2026-08-30, before the run):**

`python -m marl.diag._phase4_verify --self-test` — **4/4**:

| | check | result |
|---|---|---|
| S1 | `mappo_R2_mc_target.pth` md5 == `00da5284…` | pass |
| S2 | identity substitution + resave reproduces `00da5284…` byte-for-byte | pass |
| S3 | substituting tag *and* basename to `R2_traj_repro` **does** change the bytes, so B1 is not vacuous | pass |
| S4 | forward substitution then inversion returns exactly `00da5284…` — the transform is lossless | pass |

`python -m marl.diag._phase4_equiv --inspect --self-test` — **15/15** (see §20).

If either self-test fails, B1 is uninterpretable and the run is not judged by it.

**B2** — `R2_traj_repro_updates.csv` md5 == `0181e2e93d8ae8d9b2266335de9e8156`
(7,158 bytes, 76 lines). No substitution: this file contains no tag or path.
Reference first and last rows:
```
1,8,-12.0724,-0.013742,14.670667,0.8599,0.015996,0.1625,3.4471,-0.4721,0.0379,1.0000,0.0268
75,600,12.6539,-0.031048,5.770973,0.1380,0.000008,0.0000,2.4680,1.7241,0.1767,0.0133,0.7409
```

**B3** — `R2_traj_repro_history.csv` md5 == `42b44b4ad8e240d56a521d93440024be`
(29,824 bytes, 601 lines). No substitution.

**B4** — `R2_traj_repro_config.json` equals `mappo_R2_mc_target_config.json`
(md5 `cd36140bb5124197cc7dd9c65776bf1f`) after removing `wall_time_s` (R2: 1585.6 s) and
normalising `config.train.tag`. `train_start_window`, `train_frac` (0.7), `device` (`cpu`),
`episodes` (600) and the `risk` block must match exactly.

**B5** — `u075` payload equals the final checkpoint payload on all four `state_dict`s
(no parameter change occurs between update 75 at episode 600 and the final save).

**B6** — `u075` `opt_actor.param_groups[0]["lr"] == 9.333333333333316e-06` and
`opt_critic` lr `== 1.3333333333333309e-05`, i.e. `lr_scale = 1 − 74/75 = 0.013333…`, matching
R2's stored optimiser state and `updates.csv` row 75 `lr_scale=0.0133`.

**B7** — `u000` weight md5 (`weight_md5` convention) == `ab714064bdf1ac56daabf5c163c92215`,
the **post-probe** initial state (§5). A bare construction would give
`3bfda0327eaf93a16ff7c4d195e37d9f`; if B7 reports that value instead, the fault is in the
expectation, not the run.

**B8** — `R2_traj_repro_best.pth` payload equals the final payload, and its `extra` records
`episode 600`, `kind "best"`, `mean_reward 12.653897495678393` — reproducing the fact that R2's
best update was its last.

**B9** — 76 checkpoints exist, filenames unique, every one loads via `MAPPO.load`, every one
contains non-empty `opt_actor["state"]` and `opt_critic["state"]` **except `u000`**, whose
optimiser `state` is legitimately empty because Adam has taken no step yet. The manifest's
recorded size and md5 must re-verify against the files on disk.

**B10** — the driver's runtime RNG-neutrality assertion passed on all 76 saves
(`rng_checks_passed == 76`).

**Interpretation.** B1 (a–e) and B2–B4 together are the faithfulness test. B5–B9 are internal
consistency. B10 is the safety property of the instrumentation itself.

## 14. Failure conditions

| # | condition | action |
|---|---|---|
| F1 | B1 (any of B1a–B1e) fails but B2 and B3 pass | Contradiction — CSVs record the same losses while the weights differ. Stop, do not interpret the trajectory, diagnose serialization/precision before anything else. |
| F2 | B2 or B3 fails | **H(repro) is falsified.** Stop. Report the first diverging update row. Do not interpret the trajectory as R2's. This is a reproducibility defect and a reportable negative result in its own right. |
| F3 | B2 fails at update 1 | Environment-level nondeterminism, not drift. Stop immediately; the trajectory is worthless. Cheap: update 1 lands ≈20 s into a ≈26 min run. |
| F4 | B10 fails | The instrumentation perturbs the run. Abort; the design premise is wrong. |
| F5 | any pre-existing artifact, checkpoint or production source hash changes | RULE 12 violation. Stop, restore from git HEAD `a8df6d9`, report. |
| F6 | fewer than 76 checkpoints, or any duplicate/missing index | Instrumentation bug. Discard and fix before re-running. |
| F7 | B5/B6/B7/B8/B9 fails while B1–B4 pass | Manifest or schedule bug, not a training defect. Fix the driver; the run need not be repeated if the checkpoints themselves verify. |
| F8 | wall time greatly exceeds R2's 1585.6 s | Note it; hashing 377 MiB adds a few seconds and is expected. Not a failure by itself. |
| F9 | B1a reports a differing member **outside** `{data.pkl, .data/serialization_id}` | An unexplained container difference — the measured blast radius of N1/N2 is wrong. Stop and diagnose the serialization layer before interpreting anything, including B1b–B1d. |
| F10 | B1a passes (all 265 storages byte-identical) but B1c reports a difference | Weights, moments and every tensor replicated exactly while some scalar or metadata field did not. Report the exact leaf path; do **not** normalise it away and do **not** re-run until it is explained. |

**Known risk that cannot be resolved before the run.** `torch.get_num_threads()` is 4 on this
machine and nothing in the codebase pins it; `torch.are_deterministic_algorithms_enabled()` is
False. If R2's original run used a different thread count, CPU reduction order could differ and
B1–B3 could fail for a reason unrelated to the instrumentation. The repository contains **no
prior demonstration of bit-exact CPU→CPU replication** — `mappo_A0_cpu_repro.pth` was a
*device* re-run after CUDA broke seed reproducibility, not a bit-exact re-run of an existing CPU
arm — so H(repro) is genuinely untested here. This is why H(repro) is stated as falsifiable and
why F2/F3 are written as stop conditions rather than as retries. The thread count at run time
will be recorded in the summary.

## 15. What will be measured after the run

Only these, and nothing that requires a new metric:

1. **Faithfulness.** B1a–B1e and B2–B10, reported pass/fail individually, with the first diverging
   `updates.csv` row if B2 fails.
2. **The manifest.** 76 rows: update index, filename, bytes, md5, sha256, episode, `lr_scale`,
   `lr_actor`, `lr_critic`, and the update's own returned `stats` (`actor_loss`, `critic_loss`,
   `entropy`, `approx_kl`, `clip_frac`, `adv_mean`, `adv_std`, `value_mean`, `decision_frac`,
   `explained_var`) — all values the production `update` already returns, captured without
   modifying it.
3. **Nothing else in Phase 4 or 5.** Turning the 76 checkpoints into behavioural curves on the
   frozen RANDOM/UNION sets is Phase 6 and is not pre-registered here. It requires no training
   and will be pre-registered separately, because choosing what to measure on a trajectory after
   seeing the trajectory is exactly the cherry-picking RULE 9 forbids.

## 16. How the trajectory is compared with the existing R2 artifacts

| existing R2 artifact | comparison | criterion |
|---|---|---|
| `mappo_R2_mc_target.pth` | payload, via the tag/out_dir normalisation of §13 | B1 |
| `mappo_R2_mc_target_updates.csv` | raw bytes | B2 |
| `mappo_R2_mc_target_history.csv` | raw bytes | B3 |
| `mappo_R2_mc_target_config.json` | field-wise, minus `wall_time_s` and `tag` | B4 |
| `mappo_R2_mc_target_best.pth` | payload | B8 |
| stored `opt_actor` / `opt_critic` LRs | numeric | B6 |
| 75 rows of `updates.csv` | vs the manifest's per-update `stats` | consistency of the manifest with the CSV the production code wrote |

If **all** of B1–B4 pass, the 76 checkpoints are the trajectory that produced
`mappo_R2_mc_target.pth`, and every historical R2 result transfers to them without re-running
anything. If any of B1–B4 fails, the trajectory is *a* trajectory but not *R2's*, the
faithfulness hypothesis is falsified, and per RULE 10 the correct response is to report that and
stop — not to tune the driver until the hashes agree.

## 17. Replication (RULE 11)

Sprint 7 has used one seed throughout. This rung does not change that and adds no causal claim,
so it does not need ≥3 seeds. Any *causal* claim later built on this trajectory does.

## 18. Smoke test — mechanical only, run before the real run

`marl/diag/_phase4_smoke.py`, run 2026-08-30 on a deliberately tiny configuration
(`--episodes 6 --rollout-episodes 3 --episode-steps 5 --seed 999 --tag P4_SMOKE
--critic-target mc --device cpu` → 2 updates → 3 checkpoints). **Seed 999 and 6 episodes: this
is NOT the R2 arm and none of its numbers carry scientific content.** Reward, loss and entropy
are deliberately not reported by it. It writes into an isolated `_P4_SMOKE_trajectory`
namespace, so the real run's `R2_trajectory` preflight guard stays meaningful even if the smoke
test crashes, and it deletes every artifact it created except
`SPRINT_7_P4_SMOKE_REPORT.json`.

Result: **7/7 checks pass.**

| # | required check | how it was checked |
|---|---|---|
| 1 | saving occurs at the intended update boundary | `u000` == a freshly seeded `MAPPO` **after** `train.py:129`'s probe (and ≠ before it, so the check is not vacuous); weights differ from the previous checkpoint at every one of u001–u002, so each save is post-update; u002 weights == final `P4_SMOKE.pth`; `episode` = [0, 3, 6]; `buffer_T` = [None, 15, 15] = 3 episodes × 5 steps |
| 2 | checkpoint count is correct | 3 files, 3 manifest rows, 2 observed updates |
| 3 | filenames are unique | 3 distinct on disk, 3 distinct in the manifest, indices contiguous from 000 |
| 4 | `torch.save` does not alter RNG state | independent before/after diff of the torch **and** numpy global generators around a real 5 MB save, *plus* the driver's own runtime guard firing on 3/3 saves |
| 5 | the driver does not modify production source | all 7 forbidden files hash-identical across the run, and `MAPPO.update` is the original function object again after `uninstall()` |
| 6 | checkpoints load successfully | `MAPPO.load` on all 3 |
| 7 | optimizer state present, existing schema preserved | top-level keys identical to `mappo_R2_mc_target.pth` for every checkpoint; Adam `state` non-empty in u001–u002 and empty in u000 (no step taken yet); `extra` reproduces `train.py`'s schema except `mean_reward` |

**The one intended schema difference.** `train.py` records `mean_reward` in its checkpoints'
`extra`. It is computed in `train.main` *after* `update()` has already returned, so it is
unreachable from the interception point without editing production code — which is forbidden.
Every per-update mean reward is in `{tag}_updates.csv` and `{tag}_history.csv` instead, so
nothing is lost. All other `extra` keys are present.

**What the smoke test changed in this document.** It falsified the original B7 (see §5) before
any training was run. That is the whole point of running it first, and B7 is the only
pre-registration value that has been *corrected*; the later Phase 4.1 amendments (§20) add
criteria rather than change existing ones.

## 19. Amendment log

Amendments made **before** the run, with no result in hand. Nothing below was changed in
response to an outcome.

| date | section | change | reason |
|---|---|---|---|
| 2026-08-30 | §5, B7 | `u000` expected weight md5 `cfff208ba73247493e66f5eb97649bc8` → `ab714064bdf1ac56daabf5c163c92215` | the original was computed against a bare construction; `train.py:129`'s float32-lossy independence probe means `u000` is the post-probe state. Caught by smoke check 1. |
| 2026-08-30 | B1 | substitution specified as a **minimal copy** (four dicts only, everything else by identity) | a json deep copy of the config produced an `==`-equal object with a 320-byte-larger pickle (pickle string memoisation), which failed the S2 self-test |
| 2026-08-30 | B1 | self-test expanded from one check to S1–S4 and recorded as passing | S2 alone does not show the substitution is non-trivial (S3) or invertible (S4) |
| 2026-08-30 | §12 | `_phase4_verify.py` and `SPRINT_7_P4_SMOKE_REPORT.json` added to the new-path table | both are additive files this phase creates |
| 2026-08-30 | §13 | **B1 split into B1a–B1d and a new B1e**, backed by a measured blast-radius decomposition of the tag and the stem; original B1 kept verbatim as B1d | the original wording asserted the difference was "confined" to the tag and the stem without measuring either, and tested the claim only at whole-file md5 level. Strengthening, not weakening: 265 tensor storages are now compared as raw bytes with no normalisation at all, and 476 leaves + 93 nodes are compared exactly. |
| 2026-08-30 | §13 | pre-registered `mappo_R2_mc_target_best.pth` normalised md5 `4937a745120601ff79f3634b4b4b5d71` (literal `cbf53ca75d36ed908a974da2983df3ed`, 5,208,669 bytes) | B1e needs a reference measured **before** the run, not derived from it afterwards |
| 2026-08-30 | §14 | failure conditions **F9** and **F10** added; F1 reworded to name B1a–B1e | the new sub-criteria create two new distinguishable failure modes: an unexplained container difference, and tensors replicating exactly while metadata does not |
| 2026-08-30 | §12 | `_phase4_equiv.py` and `SPRINT_7_P4_EQUIV_SELFTEST.txt` added to the new-path table | additive files created by Phase 4.1 |
| 2026-08-30 | §20 | new section recording the Phase 4.1 self-test (15/15) | the equivalence machinery must be demonstrated before the run, on the existing checkpoint |
| 2026-08-30 | §13 | printed criterion names fixed: `_phase4_equiv` emits `B1a B1b B1c B1d` for the final and `B1ea B1eb B1ec B1ed` for `_best`; `_phase4_verify` re-emits the whole-file layer as `B1d` | one criterion name was defined in two modules; the redundancy is kept deliberately but must be unambiguous in the log |
| 2026-08-30 | §12, §20.1 | Phase 4.1 integrity manifests added under `_PHASE41_integrity/`; `*.txt` added to the artifacts scope | RULE 12 requires hashing before and after every phase; the `_PHASE4_integrity/*_after` set is left intact as the before-state, and `.txt` is the first artifact of that type in the directory |

## 20. Phase 4.1 — equivalence machinery, demonstrated before the run

`python -m marl.diag._phase4_equiv --inspect --self-test` → **15/15**, console output captured
verbatim in `SPRINT_7_P4_EQUIV_SELFTEST.txt`. The verifier is read-only and writes nothing outside
a system temp directory; criterion **RO** re-hashes `mappo_R2_mc_target.pth` at the end and
confirms `00da5284504cee8c1687866b315d6194` unchanged.

| | check | result |
|---|---|---|
| A | loaded `mappo_R2_mc_target.pth`, md5 == pre-registered reference | pass |
| I6 | `torch.load` → `torch.save` under the **same** basename reproduces the original md5 exactly | pass |
| B | saved an equivalent payload as `R2_traj_repro.pth` (tag *and* stem `R2_traj_repro`), 5,205,895 bytes | pass |
| C | literal whole-file md5 **differs** (`99fa9d59…` ≠ `00da5284…`) — a literal-identity criterion is unsatisfiable by construction | pass |
| D1a | 271 members, names identical modulo stem; **265/265 tensor storages byte-identical with no normalisation**; content differs only in `data.pkl` and `.data/serialization_id` | pass |
| D1b | after normalisation, **0 of 271** members differ | pass |
| D1c | **476 leaves + 93 nodes** compared exactly; **0 differences**; exactly 1 normalised leaf (`extra.config.train.tag`); **0 leaves of unrecognised type** | pass |
| D1d | normalised whole-file md5 == `00da5284504cee8c1687866b315d6194` | pass |
| E0 | leaf comparator separates `+0.0` from `-0.0` (which `torch.equal` calls equal) **and** calls NaN equal to an identically-encoded NaN (which `torch.equal` does not) — strictly stronger in both directions | pass |
| E | **15/15 mutation cases** behaved as specified: 14 detected, 1 normalised | pass |
| N1 | allowlist is exactly **one** literal leaf path, 1/476 leaves (0.21%); graph census matches the pre-registered 476/93/0 | pass |
| N2 | all 265 tensor-storage members (of 271 total) compared as raw bytes with no normalisation at any point | pass |
| N3 | the tag's measured blast radius is 2 container members; the stem's is member names only | pass |
| RO | the original R2 checkpoint is unmodified | pass |

**The 15 mutation cases (E).** Each is the smallest representable change of its kind; each must be
caught by **both** the structural layer and the whole-file layer, except E14 which must be reported
as *normalised* and must **not** flag.

| | mutation | expected | L2 diffs | L3 flags | first differing path |
|---|---|---|---|---|---|
| E1 | 1-ULP flip in an actor weight | detect | 1 | yes | `actor.actors.0.0.weight` |
| E2 | 1-ULP change in `extra.mean_reward` | detect | 1 | yes | `extra.mean_reward` |
| E3 | a config field **other than tag** (`train.seed` +1) | detect | 1 | yes | `extra.config.train.seed` |
| E4 | 1-ULP change in `mappo_cfg.gamma` | detect | 1 | yes | `mappo_cfg.gamma` |
| E5 | 1-ULP flip in `opt_actor.state[0].exp_avg` | detect | 1 | yes | `opt_actor.state.0.exp_avg` |
| E6 | optimiser step count +1 | detect | 1 | yes | `opt_actor.state.0.step` |
| E7 | `opt_critic.param_groups[0].lr` ×2 | detect | 1 | yes | `opt_critic.param_groups[0].lr` |
| E8 | `n_agents` +1 | detect | 1 | yes | `n_agents` |
| E9 | sign bit flipped in a critic bias element (covers `0.0 → -0.0`) | detect | 1 | yes | `critic.net.0.bias` |
| E10 | NaN injected into one actor bias element | detect | 1 | yes | `actor.actors.0.0.bias` |
| E11 | dict **key order** reversed in `extra` (schema drift) | detect | 1 | yes | `extra` |
| E12 | tensor dtype `float32 → float64` | detect | 1 | yes | `critic.net.0.bias` |
| E13 | an extra top-level key added | detect | 2 | yes | `` (root node) |
| E14 | `extra.config.train.tag` **only** | **normalise** | **0** | **no** | — |
| E15 | tag **and** a 1-ULP weight flip together | detect | 1 | yes | `actor.actors.0.0.weight` |

E14 and E15 are the pair that matter for §13's "cannot hide a real difference": the allowed
normalisation passes exactly one string through, and a substantive change occurring *alongside*
it is still caught.

**What Phase 4.1 changed in this document.** It strengthened B1 and added B1e, F9, F10 and this
section. It changed **no** methodology, threshold, metric, state set, seed or config, produced no
R4 proposal, and touched no production source. Nothing was relaxed: every criterion that existed
before still exists, and four new ones sit above it.

### 20.1 Phase 4.1 integrity (RULE 12)

The `_PHASE4_integrity/SPRINT_7_P4_*_after.md5` set is **not overwritten**; it is the recorded
state at the end of Phase 4 and therefore *is* Phase 4.1's before-state. Phase 4.1 writes a new
additive directory `_PHASE41_integrity/`. Manifest scopes, reproducible verbatim:

| manifest | command (from `python-ai/`) | lines |
|---|---|---|
| `SPRINT_7_P41_code_after.md5` | `cd marl && find . -name "*.py" -type f \| sort \| xargs md5sum` | 52 |
| `SPRINT_7_P41_artifacts_after.md5` | `cd saved_models/marl && find . -maxdepth 1 -type f \( -name "*.md" -o -name "*.json" -o -name "*.csv" -o -name "*.pth" -o -name "*.txt" \) \| sort \| xargs md5sum` | 150 |
| `SPRINT_7_P41_checkpoints_after.md5` | `cd saved_models/marl && find . -maxdepth 1 -type f -name "*.pth" \| sort \| xargs md5sum` | 23 |
| `SPRINT_7_P41_inputs_after.md5` | `cd .. && md5sum simulation/failure_history.csv simulation/failure_log.csv python-ai/saved_models/failure_predictor_oof.npz` | 3 |
| `SPRINT_7_P41_prior_manifests.md5` | `cd saved_models/marl && find . -mindepth 2 -type f -name "*.md5" -not -path "./_PHASE41_integrity/*" \| sort \| xargs md5sum` | 52 |
| `SPRINT_7_P41_newcode.md5` / `_newartifacts.md5` / `_changed.md5` | the deltas below | 1 / 1 / 2 |

`*.txt` was added to the artifacts scope because Phase 4.1 introduced the first `.txt` artifact in
this directory; the extension set is otherwise unchanged, and the addition can only *widen*
coverage.

**Verdict — Phase 4 after → Phase 4.1 after, by `diff`:**

| manifest | added | modified | removed |
|---|---|---|---|
| code (52 `.py` under `marl/`) | `diag/_phase4_equiv.py` | `diag/_phase4_verify.py` | none |
| artifacts (150) | `SPRINT_7_P4_EQUIV_SELFTEST.txt` | `SPRINT_7_PHASE4_PREREG.md` | none |
| checkpoints (23 `.pth`) | none | **none** | none |
| inputs (3) | none | **none** | none |

So: 2 additions, 2 modifications, 0 removals; **0 checkpoints and 0 inputs touched**, and the
seven forbidden production files hash exactly as they did at the start of Sprint 7
(`mappo.py 8614d016…`, `train.py 8ef42424…`, `env.py 4e565282…`, `rollout.py 3a829646…`,
`config.py 3adca5da…`, `evaluate.py 79c8e984…`, `risk_provider.py 02219832…`). `git status`
reports **no** modified tracked file — every Phase 4 / 4.1 path is new and untracked.

---

**Pre-registration ends here.** The command that would be run is recorded in the Phase 4 report
and is not executed until explicitly approved.
