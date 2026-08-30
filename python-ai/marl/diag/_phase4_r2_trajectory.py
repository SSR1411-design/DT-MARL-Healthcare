"""Sprint 7 PHASE 4 -- checkpoint-instrumented replication of R2.  ADDITIVE.

Pre-registration: saved_models/marl/SPRINT_7_PHASE4_PREREG.md  (written first).

WHAT THIS IS. A zero-variable replication of the R2 arm that additionally persists
the learning state at every PPO update boundary. The artifact corpus currently
holds exactly ONE policy snapshot per arm, so no question about WHEN anything
happened during training is answerable. This driver produces that missing
observable and nothing else.

WHAT THIS IS NOT. Not a treatment. Not a new metric. Not an R4. It manipulates
zero production variables and has no behavioural success criterion -- only the
faithfulness criteria B1-B10 of the pre-registration.

HOW IT AVOIDS TOUCHING PRODUCTION CODE. train.py exposes no per-update hook: the
boundary is `stats = agent.update(buf)` at train.py:213, buried inside main(),
which persists only {tag}_best.pth and {tag}.pth. Rather than copy main() -- which
would create a second implementation of training and make every faithfulness claim
vacuous -- this module rebinds MAPPO.update on the class to a wrapper that
delegates to the ORIGINAL unbound method and then saves. The production call site,
control flow, RNG stream and buffer lifetime are untouched; `marl.train.main` runs
verbatim, and the resolved config comes from train.parse_args/apply_args.

WHY THE SAVE CANNOT PERTURB THE RUN. MAPPO.save is a bare torch.save of four
state_dicts plus static metadata -- no clock, no RNG draw. Measured before this
file existed: torch.save leaves torch.get_rng_state() and np.random.get_state()
unchanged. This module does not take that on trust: it snapshots both generator
states around every save and ABORTS on any difference, so the premise is enforced
rather than assumed (criterion B10).

READ-ONLY WRT EVERY EXISTING ARTIFACT. Refuses to start if any path it would write
already exists, and hard-refuses the five protected R2 filenames.

    python -m marl.diag._phase4_r2_trajectory --tag R2_traj_repro \
        --critic-target mc --rollout-episodes 8 --episodes 600 \
        --seed 20260818 --device cpu
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl import train as _train                    # noqa: E402  (NOT modified)
from marl.config import Sprint6Config               # noqa: E402
from marl.mappo import MAPPO                        # noqa: E402

SUBDIR = "R2_trajectory"
PREFIX = "R2_trajectory_u"
MANIFEST = "SPRINT_7_P4_trajectory_manifest.jsonl"
SUMMARY = "SPRINT_7_P4_trajectory_summary.json"
# Never writable by this driver, whatever the caller passes.
PROTECTED = {
    "mappo_R2_mc_target.pth", "mappo_R2_mc_target_best.pth",
    "mappo_R2_mc_target_config.json", "mappo_R2_mc_target_history.csv",
    "mappo_R2_mc_target_updates.csv",
}
TAG_SUFFIXES = (".pth", "_best.pth", "_config.json", "_history.csv",
                "_updates.csv")


def _digest(path):
    """md5 and sha256 in one pass; the checkpoints are ~5 MB each."""
    m, s = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            m.update(chunk)
            s.update(chunk)
    return m.hexdigest(), s.hexdigest()


def _rng_state():
    """Every global generator the run could conceivably draw from."""
    n = np.random.get_state()
    return torch.get_rng_state().clone(), (n[0], n[1].copy(), n[2], n[3], n[4])


def _rng_same(a, b):
    if not torch.equal(a[0], b[0]):
        return False
    x, y = a[1], b[1]
    return (x[0] == y[0] and bool(np.array_equal(x[1], y[1]))
            and x[2] == y[2] and x[3] == y[3] and x[4] == y[4])


class TrajectoryRecorder:
    """Rebinds MAPPO.update at the class level; the production body is untouched."""

    def __init__(self, cfg, n_updates):
        self.cfg_dict = cfg.to_dict()
        self.per_update = cfg.train.rollout_episodes
        self.n_updates = n_updates
        self.n_expected = n_updates + 1          # u000 .. u{n_updates}
        self.dir = Path(cfg.train.out_dir) / SUBDIR
        self.manifest = self.dir / MANIFEST
        self.rows = []
        self.update_id = 0
        self.rng_checks = 0
        self._saved_init = False
        self._orig = None

    # ---- guards ---------------------------------------------------------

    def preflight(self):
        """Nothing existing may be overwritten. Runs before any training."""
        if self.dir.exists() and any(self.dir.iterdir()):
            raise SystemExit(f"REFUSING: {self.dir} exists and is not empty; "
                             "trajectory checkpoints must be additive.")
        for i in range(self.n_expected):
            p = self._path(i)
            if p.name in PROTECTED or p.exists():
                raise SystemExit(f"REFUSING: would write {p}")
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, i):
        return self.dir / f"{PREFIX}{i:03d}.pth"

    # ---- the instrumented boundary --------------------------------------

    def install(self):
        orig = MAPPO.update
        self._orig = orig
        rec = self

        def update(agent, buf):
            # u000: the pre-training state, captured on entry to update 1 only.
            # Parameters change ONLY inside update(); rollout is @torch.no_grad,
            # and set_lr_scale(1.0) is an exact identity on the LRs -- so this IS
            # the post-__init__ state (pre-registration section 5).
            if not rec._saved_init:
                rec._saved_init = True
                rec._save(agent, 0, stats=None, buf_T=None)

            stats = orig(agent, buf)              # <-- production body, verbatim

            rec.update_id += 1
            rec._save(agent, rec.update_id, stats=stats, buf_T=len(buf))
            return stats

        MAPPO.update = update

    def uninstall(self):
        if self._orig is not None:
            MAPPO.update = self._orig
            self._orig = None

    # ---- persistence ----------------------------------------------------

    def _save(self, agent, i, stats, buf_T):
        # Mirrors train.py:211 for bookkeeping only; never feeds back into the run.
        frac = 1.0 if i == 0 else 1.0 - (i - 1) / self.n_updates
        episode = 0 if i == 0 else i * self.per_update
        path = self._path(i)
        stats = None if stats is None else {k: float(v) for k, v in stats.items()}

        before = _rng_state()
        # Same extra schema as train.py's checkpoints (config / episode / kind),
        # minus mean_reward, which is computed in train.main after update() has
        # already returned and so is not reachable from here without editing
        # production code. It is in {tag}_updates.csv and {tag}_history.csv.
        agent.save(path, extra=dict(
            config=self.cfg_dict, episode=episode, kind="trajectory",
            update=i, lr_scale=frac, update_stats=stats,
            sprint="7", rung="R2-TRAJ",
            note="checkpoint-instrumented replication of R2; zero manipulated "
                 "production variables; see SPRINT_7_PHASE4_PREREG.md"))
        after = _rng_state()

        # B10. If saving ever perturbs a generator the design premise is wrong and
        # the run must not be interpreted -- so abort rather than warn.
        if not _rng_same(before, after):
            raise SystemExit(
                f"ABORT: saving checkpoint {i} perturbed a global RNG state. "
                "The instrumentation is not neutral; criterion B10 failed.")
        self.rng_checks += 1

        md5, sha = _digest(path)
        row = {"update": i, "file": path.name, "bytes": path.stat().st_size,
               "md5": md5, "sha256": sha, "episode": episode, "lr_scale": frac,
               "lr_actor": float(agent.opt_actor.param_groups[0]["lr"]),
               "lr_critic": float(agent.opt_critic.param_groups[0]["lr"]),
               "buffer_T": buf_T, "stats": stats}
        self.rows.append(row)
        # Appended per update, not buffered: three earlier diagnostic runs were
        # killed mid-training and lost everything they had not flushed.
        with open(self.manifest, "a") as f:
            f.write(json.dumps(row) + "\n")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    # Resolve with the PRODUCTION parser/applier: no config logic is
    # reimplemented, so this is exactly the config train.main will build.
    cfg = _train.apply_args(Sprint6Config(), _train.parse_args(argv))
    cfg.env.start_frac_lo, cfg.env.start_frac_hi = 0.0, _train.TRAIN_FRAC
    n_updates = max(1, cfg.train.episodes // cfg.train.rollout_episodes)

    tag = cfg.train.tag
    if tag == "mappo" or tag.startswith("mappo_R2_mc_target"):
        raise SystemExit(f"REFUSING: tag {tag!r} collides with existing "
                         "artifacts. Use an isolated tag, e.g. R2_traj_repro.")
    out = Path(cfg.train.out_dir)
    for suffix in TAG_SUFFIXES:
        p = out / f"{tag}{suffix}"
        if p.name in PROTECTED:
            raise SystemExit(f"REFUSING: {p.name} is a protected R2 artifact.")
        if p.exists():
            raise SystemExit(f"REFUSING: {p} already exists (RULE 12 is "
                             "additive-only). Choose another tag.")

    rec = TrajectoryRecorder(cfg, n_updates)
    rec.preflight()

    print("=" * 78)
    print("SPRINT 7 PHASE 4 -- R2 TRAJECTORY (zero-variable replication)")
    print("=" * 78)
    print(f"  tag              : {tag}")
    print(f"  updates expected : {n_updates}  ({cfg.train.episodes} episodes / "
          f"{cfg.train.rollout_episodes} per update)")
    print(f"  checkpoints      : {rec.n_expected}  (u000 .. u{n_updates:03d})")
    print(f"  critic_target    : {cfg.mappo.critic_target}   seed "
          f"{cfg.train.seed}   device {cfg.train.device}")
    print(f"  trajectory dir   : {rec.dir}")
    print(f"  torch            : {torch.__version__}  threads "
          f"{torch.get_num_threads()}/{torch.get_num_interop_threads()}  "
          f"deterministic={torch.are_deterministic_algorithms_enabled()}")
    print("=" * 78)

    t0 = time.time()
    rec.install()
    try:
        # Verbatim production entrypoint. train.main re-parses argv and re-seeds
        # both global generators, so nothing above can have biased the stream.
        rc = _train.main(argv)
    finally:
        rec.uninstall()
    wall = time.time() - t0

    summary = {
        "probe": "_phase4_r2_trajectory.py", "rung": "R2-TRAJ",
        "prereg": "SPRINT_7_PHASE4_PREREG.md",
        "what": "checkpoint-instrumented replication of R2; zero manipulated "
                "production variables; produces an observable, localizes "
                "nothing by itself and claims no causality",
        "tag": tag,
        "expected_updates": n_updates, "observed_updates": rec.update_id,
        "expected_checkpoints": rec.n_expected,
        "observed_checkpoints": len(rec.rows),
        "unique_filenames": len({r["file"] for r in rec.rows}),
        "rng_checks_passed": rec.rng_checks,
        "wall_time_s": round(wall, 1),
        "torch_version": torch.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "config": cfg.to_dict(), "checkpoints": rec.rows,
    }
    with open(rec.dir / SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 78)
    print(f"  checkpoints written : {len(rec.rows)} / {rec.n_expected} expected")
    print(f"  unique filenames    : {summary['unique_filenames']}")
    print(f"  RNG-neutrality      : {rec.rng_checks}/{len(rec.rows)} saves verified")
    print(f"  manifest            : {rec.manifest}")
    print("  faithfulness (B1-B10) is judged by _phase4_verify.py, NOT by this "
          "run completing.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
