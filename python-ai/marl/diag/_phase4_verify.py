"""Sprint 7 PHASE 4 -- faithfulness verifier for the R2 trajectory run.  ADDITIVE.

Implements criteria B1-B10 of saved_models/marl/SPRINT_7_PHASE4_PREREG.md. Every
reference value below was measured from the frozen R2 artifacts BEFORE the
replication was run and is hard-coded here so that it cannot be adjusted after
seeing a result (RULE 3, RULE 9).

READ-ONLY. Writes nothing except into a system temp directory.

    python -m marl.diag._phase4_verify --self-test     # runnable before training
    python -m marl.diag._phase4_verify --tag R2_traj_repro

WHY B1 NEEDS A NORMALISATION. torch.save is byte-deterministic for a given payload
but the byte stream depends on the file's STEM, because PyTorchFileWriter uses the
basename as the zip archive prefix. And extra["config"] embeds train.tag and
train.out_dir. So a replication that must NOT overwrite mappo_R2_mc_target.pth
cannot produce a byte-identical file: two strings and the stem differ by
construction. B1 therefore substitutes exactly those two strings back and saves
under the original basename, which restores true whole-file byte-identity.
--self-test proves that machinery is lossless by round-tripping the ORIGINAL file
through the same substitution in both directions.

SCOPE. This module owns B1d (normalised whole-file md5) and B2-B10. Phase 4.1 split
B1 into four sub-criteria plus B1e; B1a (container comparison with NO normalisation),
B1b (container, normalised), B1c (exact structural comparison of the unpickled graph)
and B1e (the same layers on _best.pth) are implemented in _phase4_equiv.py, which
imports this module for the frozen reference values. The dependency is one-way, so
after a run BOTH must be executed:

    python -m marl.diag._phase4_verify --self-test --tag R2_traj_repro
    python -m marl.diag._phase4_equiv  --self-test --tag R2_traj_repro
"""
import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.mappo import MAPPO                        # noqa: E402  (load only)

M = _ROOT / "saved_models" / "marl"
TRAJ = M / "R2_trajectory"

# ---- pre-registered reference values (frozen before the run) ---------------
R2_STEM = "mappo_R2_mc_target"
REPRO_TAG = "R2_traj_repro"
REF = {
    "pth_md5": "00da5284504cee8c1687866b315d6194",
    "pth_bytes": 5207250,
    "updates_md5": "0181e2e93d8ae8d9b2266335de9e8156",
    "updates_bytes": 7158,
    "updates_lines": 76,
    "history_md5": "42b44b4ad8e240d56a521d93440024be",
    "history_bytes": 29824,
    "history_lines": 601,
    "config_md5": "cd36140bb5124197cc7dd9c65776bf1f",
    # POST-PROBE init, hashed by weight_md5 below. train.py:129 calls
    # MAPPO.assert_actors_independent(), which does first.add_(1.0) then
    # first.sub_(1.0) on actor 0's first weight matrix. In float32 that round
    # trip is NOT exactly invertible: at |w| ~ 0.10 the ULP is ~1.5e-8, but in
    # [1,2) it is 2^-23, so ~3 low bits are rounded away. Measured: 5476/6144
    # elements of that one matrix move, max |delta| 5.96e-08, and the operation
    # is idempotent afterwards. u000 is therefore the POST-probe state, not a
    # bare construction. The Phase 0 report pre-computed this against a bare
    # construction; that value (cfff208ba73247493e66f5eb97649bc8, whose hashing
    # convention could not be reproduced by any of four candidates) is
    # SUPERSEDED. Pre-probe under this convention is 3bfda0327eaf93a16ff7c4d195e37d9f.
    "u000_weight_md5": "ab714064bdf1ac56daabf5c163c92215",
    "final_lr_actor": 9.333333333333316e-06,
    "final_lr_critic": 1.3333333333333309e-05,
    "final_lr_scale": 0.013333333333333308,
    "best_mean_reward": 12.653897495678393,
    "final_mean_reward": 4.6830271569342585,
    "n_updates": 75,
    "n_checkpoints": 76,
    "out_dir": None,          # filled from the original file; machine-specific
    # ---- Phase 4.1 additions (measured 2026-08-30, still BEFORE the run) ----
    # Serialization structure of mappo_R2_mc_target.pth, used by _phase4_equiv:
    # 271 zip members of which 265 are tensor storages, one per tensor in the
    # graph; the unpickled graph has 476 leaves and 93 container nodes, 0 of
    # unrecognised type. The equivalence criterion normalises exactly 1 of those
    # 476 leaves (extra.config.train.tag) and the archive stem, nothing else.
    "n_zip_members": 271,
    "n_zip_storages": 265,
    "n_graph_leaves": 476,
    "n_graph_nodes": 93,
    # mappo_R2_mc_target_best.pth after the SAME normalisation (tag and stem
    # rewritten to mappo_R2_mc_target). Its literal md5 is
    # cbf53ca75d36ed908a974da2983df3ed at 5208669 bytes; the normalised form is
    # the comparable quantity because the _best stem is 5 characters longer.
    "best_literal_md5": "cbf53ca75d36ed908a974da2983df3ed",
    "best_norm_md5": "4937a745120601ff79f3634b4b4b5d71",
    "best_bytes": 5208669,
}


def md5_bytes(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def weight_md5(ck):
    """Hash actor+critic tensors only -- the quantity that is machine-portable."""
    m = hashlib.md5()
    for name in ("actor", "critic"):
        sd = ck[name]
        for k in sorted(sd):
            m.update((name + k).encode())
            m.update(sd[k].detach().cpu().numpy().tobytes())
    return m.hexdigest()


def payload_md5(ck):
    """Hash all four state_dicts plus the static metadata; ignores `extra`."""
    m = hashlib.md5()
    for name in ("actor", "critic"):
        sd = ck[name]
        for k in sorted(sd):
            m.update((name + k).encode())
            m.update(sd[k].detach().cpu().numpy().tobytes())
    for name in ("opt_actor", "opt_critic"):
        o = ck[name]
        for g in o["param_groups"]:
            m.update(repr(sorted((k, v) for k, v in g.items()
                                 if k != "params")).encode())
            m.update(repr(g.get("params")).encode())
        for k in sorted(o["state"], key=str):
            st = o["state"][k]
            for kk in sorted(st, key=str):
                v = st[kk]
                m.update((str(k) + str(kk)).encode())
                m.update(v.detach().cpu().numpy().tobytes()
                         if torch.is_tensor(v) else repr(v).encode())
    m.update(repr(sorted(ck["mappo_cfg"].items())).encode())
    m.update(repr((ck["n_agents"], ck["obs_dim"], ck["state_dim"])).encode())
    return m.hexdigest()


def substitute(ck, tag, out_dir):
    """Return `ck` with extra.config.train.{tag,out_dir} replaced.

    MINIMAL COPY, deliberately. Only the four dicts on the path to the two fields
    get new identities; every other object -- including every tensor and every
    string elsewhere in the config -- is passed through by identity. A json
    deep-copy of the config was tried first and FAILED the S2 identity self-test:
    it produced an `==`-equal config whose pickle was 320 bytes LARGER, because
    pickle memoises repeated string objects and a deep copy destroys the shared
    identities. The minimal copy preserves them, and S2/S4 prove it.
    """
    ck = dict(ck)
    ex = dict(ck["extra"])
    cf = dict(ex["config"])
    tr = dict(cf["train"])
    tr["tag"] = tag
    if tr["out_dir"] != out_dir:          # in practice a no-op: there is no
        tr["out_dir"] = out_dir           # --out-dir flag, so both runs share it
    cf["train"] = tr
    ex["config"] = cf
    ck["extra"] = ex
    return ck


def save_as(obj, stem, scratch, slot):
    """Save under basename `stem`.pth -- the stem IS part of the byte stream."""
    p = Path(scratch) / slot
    p.mkdir(parents=True, exist_ok=True)
    p = p / f"{stem}.pth"
    torch.save(obj, p)
    return md5_bytes(p), p


def normalised_md5(ck, out_dir, scratch, slot):
    """md5 of `ck` renamed back onto R2's tag and saved under R2's basename."""
    return save_as(substitute(ck, R2_STEM, out_dir), R2_STEM, scratch, slot)[0]


def check(results, name, ok, detail=""):
    results.append({"criterion": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:5s}  {detail}")
    return ok


# ---------------------------------------------------------------------------


def self_test(scratch):
    """B1 harness soundness. Runnable before any training exists."""
    print("B1 HARNESS SELF-TEST (no training required)")
    orig = M / f"{R2_STEM}.pth"
    ck = torch.load(orig, map_location="cpu", weights_only=False)
    out_dir = ck["extra"]["config"]["train"]["out_dir"]
    REF["out_dir"] = out_dir
    res = []

    check(res, "S1", md5_bytes(orig) == REF["pth_md5"],
          f"original md5 matches the pre-registered reference {REF['pth_md5']}")
    check(res, "S2", normalised_md5(ck, out_dir, scratch, "s2") == REF["pth_md5"],
          "identity substitution + resave reproduces the original byte-for-byte")

    # Forward: rewrite it as if it were the replication's own file -- different
    # tag AND different basename -- then invert and require the original bytes.
    fwd, fwd_p = save_as(substitute(ck, REPRO_TAG, out_dir), REPRO_TAG,
                         scratch, "s3")
    ck2 = torch.load(fwd_p, map_location="cpu", weights_only=False)
    back = normalised_md5(ck2, out_dir, scratch, "s4")
    check(res, "S3", fwd != REF["pth_md5"],
          "tag/stem substitution does change the bytes (so B1 is a real test)")
    check(res, "S4", back == REF["pth_md5"],
          "round trip through the substitution is lossless -> B1 is sound")
    return res


def verify(tag, scratch):
    print(f"FAITHFULNESS VERIFICATION for tag {tag!r}")
    res = []
    orig = torch.load(M / f"{R2_STEM}.pth", map_location="cpu",
                      weights_only=False)
    out_dir = orig["extra"]["config"]["train"]["out_dir"]
    fin_p = M / f"{tag}.pth"
    if not fin_p.exists():
        raise SystemExit(f"{fin_p} not found -- has the run been executed?")
    fin = torch.load(fin_p, map_location="cpu", weights_only=False)

    # B1d -- normalised whole-file byte-identity. B1a (container, un-normalised),
    # B1b (container, normalised), B1c (structural, exact) and B1e (the same three
    # layers on _best.pth) live in _phase4_equiv.py, which is not imported here to
    # keep the dependency one-way. Run both:
    #     python -m marl.diag._phase4_equiv --tag <tag>
    got = normalised_md5(fin, out_dir, scratch, "b1")
    check(res, "B1d", got == REF["pth_md5"],
          f"normalised md5 {got} vs reference {REF['pth_md5']}  "
          f"(B1a/B1b/B1c/B1e: run _phase4_equiv --tag {tag})")

    # B2 / B3 -- raw byte-identity of the CSVs (no substitution possible or needed)
    for crit, suffix, key in (("B2", "_updates.csv", "updates"),
                              ("B3", "_history.csv", "history")):
        p = M / f"{tag}{suffix}"
        ref_p = M / f"{R2_STEM}{suffix}"
        got = md5_bytes(p) if p.exists() else "MISSING"
        ok = got == REF[key + "_md5"]
        detail = f"{suffix} md5 {got}"
        if not ok and p.exists():
            a = ref_p.read_bytes().split(b"\n")
            b = p.read_bytes().split(b"\n")
            first = next((i for i in range(max(len(a), len(b)))
                          if a[i:i + 1] != b[i:i + 1]), None)
            detail += (f" | first differing line {first}: "
                       f"R2={a[first] if first is not None and first < len(a) else None!r} "
                       f"got={b[first] if first is not None and first < len(b) else None!r}")
        check(res, crit, ok, detail)

    # B4 -- config.json modulo wall_time_s and the tag
    cp, rp = M / f"{tag}_config.json", M / f"{R2_STEM}_config.json"
    a, b = json.load(open(rp)), json.load(open(cp))
    for d in (a, b):
        d.pop("wall_time_s", None)
    a["config"]["train"]["tag"] = b["config"]["train"]["tag"] = "<tag>"
    check(res, "B4", a == b, "config.json identical after removing wall_time_s "
                             "and normalising train.tag")

    # B5 / B6 / B7 / B9 -- the trajectory itself
    paths = sorted(TRAJ.glob("R2_trajectory_u*.pth"))
    check(res, "B9a", len(paths) == REF["n_checkpoints"],
          f"{len(paths)} checkpoints (expected {REF['n_checkpoints']})")
    check(res, "B9b", len({p.name for p in paths}) == len(paths),
          "all checkpoint filenames unique")

    idx = {int(p.stem.split("_u")[-1]): p for p in paths}
    missing = [i for i in range(REF["n_checkpoints"]) if i not in idx]
    check(res, "B9c", not missing, f"contiguous u000..u075 (missing: {missing})")

    man = TRAJ / "SPRINT_7_P4_trajectory_manifest.jsonl"
    rows = [json.loads(l) for l in open(man) if l.strip()] if man.exists() else []
    bad = [r["file"] for r in rows
           if not (TRAJ / r["file"]).exists()
           or (TRAJ / r["file"]).stat().st_size != r["bytes"]
           or md5_bytes(TRAJ / r["file"]) != r["md5"]]
    check(res, "B9d", bool(rows) and not bad,
          f"{len(rows)} manifest rows re-verify on disk (bad: {bad[:3]})")

    loadable, no_opt = [], []
    for i, p in sorted(idx.items()):
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            MAPPO.load(p, device="cpu")
        except Exception as e:                                  # noqa: BLE001
            loadable.append((i, repr(e)[:60]))
            continue
        # u000 legitimately has empty Adam state: no step has been taken yet.
        if i > 0 and not (ck["opt_actor"]["state"] and ck["opt_critic"]["state"]):
            no_opt.append(i)
    check(res, "B9e", not loadable, f"every checkpoint loads (failures: {loadable[:3]})")
    check(res, "B9f", not no_opt,
          f"optimizer state present in u001..u075 (empty in: {no_opt[:5]})")

    u000 = torch.load(idx[0], map_location="cpu", weights_only=False)
    check(res, "B7", weight_md5(u000) == REF["u000_weight_md5"],
          f"u000 weight md5 {weight_md5(u000)} vs the pre-registered post-probe "
          f"init {REF['u000_weight_md5']}")

    last = torch.load(idx[REF["n_updates"]], map_location="cpu",
                      weights_only=False)
    check(res, "B5", payload_md5(last) == payload_md5(fin),
          "u075 payload identical to the final checkpoint")
    la = last["opt_actor"]["param_groups"][0]["lr"]
    lc = last["opt_critic"]["param_groups"][0]["lr"]
    check(res, "B6", la == REF["final_lr_actor"] and lc == REF["final_lr_critic"],
          f"u075 lr_actor {la!r} lr_critic {lc!r}")

    # B8 -- best checkpoint reproduces "best update was the last update"
    bp = M / f"{tag}_best.pth"
    bst = torch.load(bp, map_location="cpu", weights_only=False)
    check(res, "B8", (payload_md5(bst) == payload_md5(fin)
                      and bst["extra"]["episode"] == 600
                      and bst["extra"]["kind"] == "best"
                      and bst["extra"]["mean_reward"] == REF["best_mean_reward"]),
          f"best: episode {bst['extra']['episode']}, "
          f"mean_reward {bst['extra']['mean_reward']!r}")

    # B10 -- instrumentation neutrality, from the run's own summary
    sp = TRAJ / "SPRINT_7_P4_trajectory_summary.json"
    s = json.load(open(sp)) if sp.exists() else {}
    check(res, "B10", s.get("rng_checks_passed") == REF["n_checkpoints"],
          f"rng_checks_passed {s.get('rng_checks_passed')} / "
          f"{REF['n_checkpoints']}  (threads {s.get('torch_num_threads')})")
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--self-test", action="store_true",
                   help="verify the B1 harness only; needs no training run")
    p.add_argument("--tag", default=None, help="tag of the replication run")
    a = p.parse_args(argv)
    if not a.self_test and not a.tag:
        p.error("give --self-test or --tag")

    scratch = tempfile.mkdtemp(prefix="p4verify_")
    try:
        res = self_test(scratch) if a.self_test else None
        if a.tag:
            if res:
                print()
            res = (res or []) + verify(a.tag, scratch)
        n_bad = sum(1 for r in res if not r["pass"])
        print("-" * 70)
        print(f"  {len(res) - n_bad}/{len(res)} criteria passed")
        if n_bad:
            print("  FAILURES:", ", ".join(r["criterion"] for r in res
                                           if not r["pass"]))
        return 1 if n_bad else 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
