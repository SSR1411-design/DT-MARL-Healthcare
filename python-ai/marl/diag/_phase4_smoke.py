"""Sprint 7 PHASE 4 -- smoke test for the trajectory instrumentation.  ADDITIVE.

Checks the seven mechanical properties the Phase 4 brief requires before the real
run, on a deliberately tiny configuration (6 episodes / 3 per update / 5 steps,
seed 999 -- NOT the R2 arm):

    1. checkpoint saving occurs at the intended update boundary (post-update)
    2. the checkpoint count is correct
    3. checkpoint filenames are unique
    4. torch.save does not alter RNG state
    5. the driver does not modify production source
    6. checkpoints load successfully
    7. optimizer state is present, in the existing checkpoint schema

NONE OF THESE NUMBERS ARE SCIENTIFIC. Six episodes of five steps cannot say
anything about learning; the run exists only to exercise the save path. Reward,
loss and entropy are deliberately not reported.

NAMESPACE ISOLATION. The driver writes into {out_dir}/R2_trajectory and there is
no --out-dir flag, so this test rebinds the driver's SUBDIR/MANIFEST/SUMMARY
module constants to smoke-labelled names. That keeps the R2_trajectory namespace
empty, so the real run's preflight guard is still meaningful, and it means a
crashed smoke test cannot poison the real run's directory. All smoke artifacts
are deleted afterwards; only SPRINT_7_P4_SMOKE_REPORT.json is left behind.

    python -m marl.diag._phase4_smoke
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.config import MappoConfig                      # noqa: E402
from marl.mappo import MAPPO                             # noqa: E402
from marl.diag import _phase4_r2_trajectory as _traj     # noqa: E402

M = _ROOT / "saved_models" / "marl"
R2 = M / "mappo_R2_mc_target.pth"
TAG = "P4_SMOKE"
SMOKE_SUBDIR = "_P4_SMOKE_trajectory"
REPORT = M / "SPRINT_7_P4_SMOKE_REPORT.json"

# The seven files the brief forbids editing. Hashed before and after (check 5).
PRODUCTION = ["mappo.py", "train.py", "env.py", "rollout.py", "config.py",
              "evaluate.py", "risk_provider.py"]

EPISODES, PER_UPDATE, STEPS, SEED = 6, 3, 5, 999
N_UPDATES = EPISODES // PER_UPDATE            # 2
N_CKPT = N_UPDATES + 1                        # 3  (u000 u001 u002)

_res = []


def check(n, name, parts):
    """`parts` is a list of (ok, description) so a failure names itself."""
    ok = all(p[0] for p in parts)
    _res.append({"check": n, "name": name, "pass": bool(ok),
                 "parts": [{"pass": bool(o), "detail": d} for o, d in parts]})
    print(f"  [{'PASS' if ok else 'FAIL'}] {n}. {name}")
    for o, d in parts:
        print(f"          {'ok  ' if o else 'FAIL'}  {d}")
    return ok


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def wmd5(ck):
    m = hashlib.md5()
    for name in ("actor", "critic"):
        for k in sorted(ck[name]):
            m.update((name + k).encode())
            m.update(ck[name][k].detach().cpu().numpy().tobytes())
    return m.hexdigest()


def smoke_paths():
    """Every path this test may create -- and therefore may delete."""
    ps = [M / f"{TAG}{s}" for s in _traj.TAG_SUFFIXES] + [REPORT]
    return ps, M / SMOKE_SUBDIR


def sweep(label):
    files, sub = smoke_paths()
    gone = [p.name for p in files if p.name != REPORT.name and p.exists()]
    for p in files:
        if p.name == REPORT.name:
            continue
        assert p.name.startswith(TAG) and p.name not in _traj.PROTECTED, p
        p.unlink(missing_ok=True)
    if sub.exists():
        assert sub.name == SMOKE_SUBDIR
        gone.append(sub.name + "/")
        shutil.rmtree(sub)
    if gone:
        print(f"  ({label}: removed {', '.join(gone)})")


def main():
    print("=" * 78)
    print("SPRINT 7 PHASE 4 -- INSTRUMENTATION SMOKE TEST (not a scientific run)")
    print("=" * 78)

    # Isolate the namespace before anything touches the filesystem.
    _traj.SUBDIR = SMOKE_SUBDIR
    _traj.MANIFEST = "SPRINT_7_P4_SMOKE_manifest.jsonl"
    _traj.SUMMARY = "SPRINT_7_P4_SMOKE_summary.json"
    sub = M / SMOKE_SUBDIR
    sweep("leftovers from an earlier smoke run")

    before_src = {f: md5(_ROOT / "marl" / f) for f in PRODUCTION}
    before_update = _traj.MAPPO.update

    argv = ["--tag", TAG, "--critic-target", "mc",
            "--episodes", str(EPISODES), "--rollout-episodes", str(PER_UPDATE),
            "--episode-steps", str(STEPS), "--seed", str(SEED), "--device", "cpu"]
    print(f"  driver argv: {' '.join(argv)}\n")
    rc = _traj.main(argv)
    print()

    paths = sorted(sub.glob(f"{_traj.PREFIX}*.pth"))
    idx = {int(p.stem.split("_u")[-1]): p for p in paths}
    summary = json.load(open(sub / _traj.SUMMARY))
    rows = [json.loads(l) for l in open(sub / _traj.MANIFEST) if l.strip()]
    cks = {i: torch.load(p, map_location="cpu", weights_only=False)
           for i, p in idx.items()}
    final = torch.load(M / f"{TAG}.pth", map_location="cpu", weights_only=False)

    # ---- 2. count ------------------------------------------------------
    check(2, "checkpoint count is correct", [
        (len(paths) == N_CKPT, f"{len(paths)} checkpoint files (want {N_CKPT})"),
        (len(rows) == N_CKPT, f"{len(rows)} manifest rows (want {N_CKPT})"),
        (summary["observed_updates"] == N_UPDATES,
         f"{summary['observed_updates']} updates observed (want {N_UPDATES})"),
    ])

    # ---- 3. uniqueness -------------------------------------------------
    check(3, "checkpoint filenames are unique", [
        (len({p.name for p in paths}) == len(paths),
         f"{len({p.name for p in paths})} distinct filenames on disk"),
        (len({r["file"] for r in rows}) == len(rows),
         f"{len({r['file'] for r in rows})} distinct filenames in the manifest"),
        (sorted(idx) == list(range(N_CKPT)),
         f"indices contiguous from 000: {sorted(idx)}"),
    ])

    # ---- 1. the save happens AT and AFTER the update boundary ----------
    # u000 must be the state at entry to update 1. Rebuild it from the seed --
    # which means replaying train.py's startup exactly, INCLUDING the probe at
    # train.py:129. assert_actors_independent() perturbs actor 0's first weight
    # matrix with add_(1.0)/sub_(1.0), and in float32 that round trip loses ~3
    # low bits (ULP jumps from ~1.5e-8 at |w|~0.1 to 2^-23 in [1,2)): 5497/6144
    # elements move by up to 5.96e-08. So a bare construction is NOT u000, and
    # this test checks both -- equal to post-probe, different from pre-probe.
    ck0 = cks[0]
    torch.manual_seed(SEED)
    fresh = MAPPO(ck0["n_agents"], ck0["obs_dim"], ck0["state_dim"],
                  MappoConfig(**ck0["mappo_cfg"]), "cpu", SEED)
    pre_probe = wmd5({"actor": fresh.actor.state_dict(),
                      "critic": fresh.critic.state_dict()})
    fresh.assert_actors_independent()
    post_probe = wmd5({"actor": fresh.actor.state_dict(),
                       "critic": fresh.critic.state_dict()})
    moved = [i for i in range(1, N_CKPT) if wmd5(cks[i]) != wmd5(cks[i - 1])]
    eps = [r["episode"] for r in rows]
    bts = [r["buffer_T"] for r in rows]
    check(1, "saving occurs at the intended update boundary", [
        (wmd5(ck0) == post_probe,
         f"u000 is the pre-update state at entry to update 1: {wmd5(ck0)[:12]} "
         f"== freshly seeded MAPPO after train.py:129's probe {post_probe[:12]}"),
        (pre_probe != post_probe,
         f"and that probe is not a no-op, so the check is not vacuous: bare "
         f"construction hashes {pre_probe[:12]}"),
        (moved == list(range(1, N_CKPT)),
         f"weights differ from the previous checkpoint at u{moved} "
         f"(want {list(range(1, N_CKPT))}) -- so each save is POST-update"),
        (wmd5(cks[N_UPDATES]) == wmd5(final),
         f"u{N_UPDATES:03d} weights == final {TAG}.pth -- the last save sits on "
         "the last update boundary and nothing ran after it"),
        (eps == [i * PER_UPDATE for i in range(N_CKPT)],
         f"episode bookkeeping {eps} (want {[i * PER_UPDATE for i in range(N_CKPT)]})"),
        (bts[1:] == [PER_UPDATE * STEPS] * N_UPDATES,
         f"buffer_T at each update {bts} (want None then "
         f"{PER_UPDATE}x{STEPS}={PER_UPDATE * STEPS})"),
    ])

    # ---- 4. torch.save does not alter RNG state ------------------------
    # Independent of the driver's own guard: save a real payload and diff both
    # global generators. Then confirm the driver's runtime guard also fired.
    t_before, n_before = torch.get_rng_state().clone(), np.random.get_state()
    scratch = sub / "_rngprobe.pth"
    torch.save(cks[N_UPDATES], scratch)
    t_after, n_after = torch.get_rng_state(), np.random.get_state()
    scratch.unlink()
    direct = (torch.equal(t_before, t_after) and n_before[0] == n_after[0]
              and np.array_equal(n_before[1], n_after[1])
              and n_before[2:] == n_after[2:])
    check(4, "torch.save does not alter RNG state", [
        (direct, "independent measurement: saving a real 5 MB checkpoint left "
                 "both the torch and the numpy global generator bit-identical"),
        (summary["rng_checks_passed"] == N_CKPT,
         f"the driver's own runtime guard passed on "
         f"{summary['rng_checks_passed']}/{N_CKPT} saves"),
    ])

    # ---- 5. production source untouched --------------------------------
    after_src = {f: md5(_ROOT / "marl" / f) for f in PRODUCTION}
    changed = [f for f in PRODUCTION if before_src[f] != after_src[f]]
    check(5, "the driver does not modify production source", [
        (not changed, f"all {len(PRODUCTION)} forbidden files hash-identical "
                      f"across the run (changed: {changed})"),
        (_traj.MAPPO.update is before_update,
         "MAPPO.update is the original function object again after uninstall"),
    ])

    # ---- 6. loadable ---------------------------------------------------
    failed = []
    for i, p in sorted(idx.items()):
        try:
            MAPPO.load(p, device="cpu")
        except Exception as e:                                   # noqa: BLE001
            failed.append((i, repr(e)[:70]))
    check(6, "checkpoints load successfully", [
        (not failed, f"MAPPO.load succeeded on all {len(idx)} checkpoints "
                     f"(failures: {failed})"),
    ])

    # ---- 7. schema parity + optimizer state ----------------------------
    r2 = torch.load(R2, map_location="cpu", weights_only=False)
    bad_schema = [i for i in idx if set(cks[i]) != set(r2)]
    no_opt = [i for i in range(1, N_CKPT)
              if not (cks[i]["opt_actor"]["state"] and cks[i]["opt_critic"]["state"])]
    # `mean_reward` is computed in train.main AFTER update() has returned, so it
    # is unreachable from the interception point without editing production code.
    # Its absence is intended and documented in the driver; it is recorded in
    # {tag}_updates.csv and {tag}_history.csv instead. Nothing else may be missing.
    missing = set(final["extra"]) - set(cks[1]["extra"])
    check(7, "optimizer state present, existing schema preserved", [
        (not bad_schema,
         f"top-level keys identical to mappo_R2_mc_target.pth for every "
         f"checkpoint (mismatches: {bad_schema}); keys {sorted(set(r2))}"),
        (not no_opt, f"Adam state non-empty in u001..u{N_UPDATES:03d} "
                     f"(empty in: {no_opt})"),
        (not cks[0]["opt_actor"]["state"],
         "u000 Adam state is empty, as it must be: no step has been taken yet"),
        (missing == {"mean_reward"},
         f"extra reproduces train.py's schema except the documented, "
         f"unreachable mean_reward (missing: {sorted(missing)})"),
    ])

    n_bad = sum(1 for r in _res if not r["pass"])
    json.dump({"probe": "_phase4_smoke.py", "prereg": "SPRINT_7_PHASE4_PREREG.md",
               "what": "mechanical smoke test of the Phase 4 instrumentation; "
                       "NOT a scientific run and its numbers must not be "
                       "interpreted",
               "config": {"episodes": EPISODES, "rollout_episodes": PER_UPDATE,
                          "episode_steps": STEPS, "seed": SEED,
                          "critic_target": "mc", "device": "cpu"},
               "driver_rc": rc, "checks": _res,
               "passed": len(_res) - n_bad, "total": len(_res),
               "artifacts_removed_after_test": True},
              open(REPORT, "w"), indent=2)

    del cks, final, r2, fresh
    sweep("smoke artifacts")
    print("-" * 78)
    print(f"  {len(_res) - n_bad}/{len(_res)} checks passed   report: "
          f"{REPORT.name}")
    if n_bad:
        print("  FAILED:", ", ".join(str(r["check"]) for r in _res
                                     if not r["pass"]))
    print("  These numbers are mechanical only and carry no scientific content.")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
