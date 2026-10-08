"""Sprint 7 PHASE 4.1 -- EXACT equivalence verifier for the R2 replication.  ADDITIVE.

Resolves the open uncertainty H.1 of the Phase 4 report: literal whole-file md5
equality between the replication's final checkpoint and mappo_R2_mc_target.pth is
impossible without overwriting the protected R2 artifact, because (a) the pickled
payload embeds extra.config.train.tag and (b) the PyTorch zip archive uses the
file's basename as its member-name prefix. This module makes the equivalence claim
verifiable WITHOUT weakening it, by measuring exactly how far those two causes
reach and comparing everything else with zero tolerance.

MEASURED DECOMPOSITION (read-only probes on the existing R2 file, before any run):

  cause (a)  the tag string  -> changes `data.pkl` (by exactly the tag's length
             delta) and `.data/serialization_id` (a content hash of the members).
             ZERO of the 265 tensor-storage members change.
  cause (b)  the archive stem -> changes only member NAMES. With an identical
             payload saved under two different stems, all 271 members have
             byte-identical CONTENT, including data.pkl and serialization_id;
             the file size shifts by n_members x the stem length delta, and a
             same-length stem gives an identical size with different bytes.

So the two causes are disjoint and neither can reach a single byte of a weight,
an optimiser moment, or any scalar. The verifier below exploits that:

  L1  CONTAINER.   Compare the two .pth files as zip archives, member by member,
      as RAW BYTES. Pass requires: names identical after stripping the one-segment
      stem prefix, and the set of members whose content differs is a subset of
      {data.pkl, .data/serialization_id} -- i.e. every tensor storage matches with
      NO normalisation applied to it at all. Then repeat after normalising the tag
      and the stem, and require ZERO differing members.
  L2  STRUCTURAL.  Flatten both unpickled object graphs to (node, leaf) maps and
      compare every entry exactly: container subtype, key order, key types,
      sequence length, tensor dtype/shape/stride/contiguity/numel/raw bytes,
      float bit patterns, int/bool/str/bytes identity. Exactly ONE leaf path may
      be normalised: extra.config.train.tag. Anything unrecognised is reported as
      `opaque`, never silently accepted.
  L3  WHOLE FILE.  Substitute the tag back, resave under the original basename,
      and require the pre-registered whole-file md5. This is criterion B1 as
      originally written; it survives here as B1d.

WHY NO TOLERANCES, AND WHY RAW BYTES RATHER THAN torch.equal. torch.equal(nan, nan)
is False and torch.equal(0.0, -0.0) is True; both are wrong for a bit-identity
claim. Raw-byte comparison is strictly stronger in both directions, and the
self-test proves it on those two exact cases.

WHY THE NORMALISATION CANNOT HIDE A TRAINING DIFFERENCE. `tag` reaches nothing
numeric: in the seven production files it occurs only as the config field
declaration (config.py:526), argparse plumbing (train.py:68,93), one local alias
(train.py:121), five output-path f-strings (train.py:142,143,233,253,256) and a
default checkpoint path in evaluate.py. It is never passed to MAPPO, DTMarlEnv,
the rollout, or any seed. (Every other grep hit for the substring "tag" in
mappo.py / env.py / config.py is inside the word "advan-tag-e".) The allowlist is
one exact leaf path, not a pattern; it applies only when both sides are `str`;
the verifier reports the normalised count alongside the total leaf count; and
self-test case E15 changes the tag AND a tensor together to prove a co-occurring
substantive change is still caught.

READ-ONLY. Writes nothing outside a system temp directory. Never opens any
existing artifact for writing.

    python -m marl.diag._phase4_equiv --inspect     # serialization structure
    python -m marl.diag._phase4_equiv --self-test   # demonstrations A-E, runnable now
    python -m marl.diag._phase4_equiv --tag R2_traj_repro   # after the run
"""
import argparse
import hashlib
import shutil
import struct
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Single source of truth for the pre-registered constants and the tag substitution
# whose losslessness is already proved by _phase4_verify's S1-S4.
from marl.diag._phase4_verify import (                        # noqa: E402
    M, R2_STEM, REF, REPRO_TAG, md5_bytes, save_as, substitute,
)

# ---- the frozen normalisation allowlist ------------------------------------
# N1: one exact leaf path in the unpickled graph. Not a prefix, not a pattern.
# N2: the zip member-name prefix (the file's basename), handled structurally by
#     stripping the first path segment -- it has no representation in the graph.
ALLOW_LEAF_PATHS = ("extra.config.train.tag",)
# Members whose content is ALLOWED to differ before normalisation, and only
# because of N1: data.pkl holds the pickled tag, and serialization_id is a hash
# of the member contents. Nothing else may ever differ.
TAG_AFFECTED_MEMBERS = {"data.pkl", ".data/serialization_id"}

_res = []


def check(name, ok, detail=""):
    _res.append({"criterion": name, "pass": bool(ok), "detail": detail})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:5s}  {detail}")
    return bool(ok)


# ===========================================================================
# leaf / node description -- exact, no tolerances
# ===========================================================================

def describe_leaf(v):
    """Exact descriptor for a non-container value, or None if it is a container."""
    if v is None:
        return ("none",)
    if isinstance(v, bool):                 # before int: bool is an int subclass
        return ("bool", v)
    if isinstance(v, int):
        return ("int", v)
    if isinstance(v, float):
        # Bit pattern, not value: distinguishes 0.0 from -0.0 and compares NaN
        # payloads exactly. `==` would call -0.0 equal and NaN unequal to itself.
        return ("float", struct.pack("<d", v))
    if isinstance(v, str):
        return ("str", v)
    if isinstance(v, (bytes, bytearray)):
        return ("bytes", bytes(v))
    if torch.is_tensor(v):
        t = v.detach().cpu()
        try:
            raw = t.contiguous().numpy().tobytes()
        except Exception:                                        # noqa: BLE001
            raw = b"<no-numpy>" + repr(t.flatten().tolist()).encode()
        return ("tensor", str(t.dtype), tuple(t.shape), tuple(t.stride()),
                bool(t.is_contiguous()), int(t.numel()),
                bool(v.requires_grad), raw)
    if isinstance(v, np.ndarray):
        return ("ndarray", str(v.dtype), tuple(v.shape), v.tobytes())
    return None


def short(d):
    """Human-readable rendering of a descriptor; never used for comparison."""
    if d is None:
        return "<absent>"
    k = d[0]
    if k == "tensor":
        return (f"tensor {d[1]} shape={d[2]} stride={d[3]} numel={d[5]} "
                f"sha256={hashlib.sha256(d[7]).hexdigest()[:16]}")
    if k == "ndarray":
        return (f"ndarray {d[1]} shape={d[2]} "
                f"sha256={hashlib.sha256(d[3]).hexdigest()[:16]}")
    if k == "float":
        return f"float {struct.unpack('<d', d[1])[0]!r} bits={d[1].hex()}"
    if k == "bytes":
        return f"bytes[{len(d[1])}] sha256={hashlib.sha256(d[1]).hexdigest()[:16]}"
    if k == "dict":
        return f"dict({d[1]}) n={len(d[2])} keys={[x[1] for x in d[2]][:6]}"
    if k == "seq":
        return f"{d[1]}[{d[2]}]"
    return f"{k} {d[1]!r}" if len(d) > 1 else k


def flatten(obj, path="", nodes=None, leaves=None, opaque=None):
    """Flatten an object graph into exact node and leaf descriptor maps."""
    if nodes is None:
        nodes, leaves, opaque = {}, {}, []
    if isinstance(obj, dict):
        # Key ORDER and key TYPES are recorded: pickle byte-identity depends on
        # insertion order, and {0: ...} is not {"0": ...}.
        nodes[path] = ("dict", type(obj).__name__,
                       tuple((type(k).__name__, str(k)) for k in obj))
        for k, v in obj.items():
            flatten(v, f"{path}.{k}" if path else str(k), nodes, leaves, opaque)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        seq = list(obj)
        nodes[path] = ("seq", type(obj).__name__, len(seq))
        for i, v in enumerate(seq):
            flatten(v, f"{path}[{i}]", nodes, leaves, opaque)
    else:
        d = describe_leaf(obj)
        if d is None:
            # Unrecognised type: recorded and reported, never treated as equal
            # just because repr() happens to match.
            d = ("opaque", f"{type(obj).__module__}.{type(obj).__name__}",
                 repr(obj)[:200])
            opaque.append(path)
        leaves[path] = d
    return nodes, leaves, opaque


def structural_compare(a, b):
    """Exact comparison of two graphs; only ALLOW_LEAF_PATHS may be normalised."""
    na, la, oa = flatten(a)
    nb, lb, ob = flatten(b)
    diffs, normalised, allow_equal = [], [], []

    for p in dict.fromkeys(list(na) + list(nb)):
        if na.get(p) != nb.get(p):
            diffs.append(("node", p, short(na.get(p)), short(nb.get(p))))

    for p in dict.fromkeys(list(la) + list(lb)):
        da, db = la.get(p), lb.get(p)
        if da == db:
            # An allowlisted path that already matches was NOT normalised; it is
            # tracked separately so the normalised count means what it says.
            if p in ALLOW_LEAF_PATHS:
                allow_equal.append(p)
            continue
        # The allowance is narrow on purpose: the exact path AND both sides
        # present AND both sides strings. A tag that vanished or changed type
        # falls through and is reported as a difference.
        if (p in ALLOW_LEAF_PATHS and da is not None and db is not None
                and da[0] == "str" and db[0] == "str"):
            normalised.append((p, da[1], db[1]))
            continue
        diffs.append(("leaf", p, short(da), short(db)))

    return {"diffs": diffs, "normalised": normalised, "allow_equal": allow_equal,
            "n_nodes": len(dict.fromkeys(list(na) + list(nb))),
            "n_leaves": len(dict.fromkeys(list(la) + list(lb))),
            "opaque": sorted(set(oa) | set(ob))}


# ===========================================================================
# container (zip) comparison -- raw bytes
# ===========================================================================

def zip_members(path):
    """(stem, {member_name_without_stem: raw_bytes}). Asserts a single prefix."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        stems = {n.split("/", 1)[0] for n in names}
        if len(stems) != 1:
            raise SystemExit(f"{path}: expected one archive prefix, got {stems}")
        return stems.pop(), {n.split("/", 1)[1]: z.read(n) for n in names}


def container_compare(ref_path, cand_path):
    stem_a, ma = zip_members(ref_path)
    stem_b, mb = zip_members(cand_path)
    differing = sorted(k for k in ma if k in mb and ma[k] != mb[k])
    storages = sorted(k for k in ma if k.startswith("data/"))
    return {
        "stem_ref": stem_a, "stem_cand": stem_b,
        "n_members_ref": len(ma), "n_members_cand": len(mb),
        "names_equal_modulo_stem": set(ma) == set(mb),
        "only_in_ref": sorted(set(ma) - set(mb)),
        "only_in_cand": sorted(set(mb) - set(ma)),
        "differing": differing,
        "n_storages": len(storages),
        "storages_differing": sorted(k for k in storages
                                     if k in mb and ma[k] != mb[k]),
        "bytes_ref": Path(ref_path).stat().st_size,
        "bytes_cand": Path(cand_path).stat().st_size,
    }


# ===========================================================================
# the three-layer equivalence verdict
# ===========================================================================

def normalise_file(cand_path, scratch, slot):
    """Rewrite `cand_path` with R2's tag, under R2's basename. Temp dir only."""
    ck = torch.load(cand_path, map_location="cpu", weights_only=False)
    out_dir = ck["extra"]["config"]["train"]["out_dir"]
    _, p = save_as(substitute(ck, R2_STEM, out_dir), R2_STEM, scratch, slot)
    return p


def equivalence(ref_path, cand_path, scratch, slot, label):
    """L1/L2/L3 on one pair of files. Returns a dict of findings, prints nothing."""
    raw = container_compare(ref_path, cand_path)
    norm_p = normalise_file(cand_path, scratch, f"{slot}_norm")
    nrm = container_compare(ref_path, norm_p)
    st = structural_compare(
        torch.load(ref_path, map_location="cpu", weights_only=False),
        torch.load(cand_path, map_location="cpu", weights_only=False))
    return {
        "label": label, "ref": str(ref_path), "cand": str(cand_path),
        "L1_raw": raw, "L1_norm": nrm, "L2": st,
        "L3_md5": md5_bytes(norm_p),
        "literal_md5_ref": md5_bytes(ref_path),
        "literal_md5_cand": md5_bytes(cand_path),
    }


def report(e, ref_md5, prefix):
    """Print and score one equivalence result under criteria {prefix}a..d."""
    raw, nrm, st = e["L1_raw"], e["L1_norm"], e["L2"]
    ok = True

    extra_members = [k for k in raw["differing"] if k not in TAG_AFFECTED_MEMBERS]
    ok &= check(f"{prefix}a",
                raw["names_equal_modulo_stem"] and not extra_members
                and not raw["storages_differing"],
                f"L1 raw: {raw['n_members_ref']} members, names identical modulo "
                f"stem ({raw['stem_ref']!r} vs {raw['stem_cand']!r}); "
                f"{raw['n_storages']}/{raw['n_storages']} tensor storages "
                f"byte-identical with NO normalisation; content differs only in "
                f"{raw['differing'] or 'nothing'}"
                + (f"; UNEXPECTED: {extra_members}" if extra_members else ""))

    ok &= check(f"{prefix}b", nrm["names_equal_modulo_stem"] and not nrm["differing"],
                f"L1 normalised: 0 of {nrm['n_members_ref']} members differ "
                f"(differing: {nrm['differing'] or 'none'})")

    n_norm = len(st["normalised"])
    # The allowlisted leaf must be PRESENT on both sides exactly once, either
    # already equal or normalised -- so a vanished tag cannot pass as "no diff".
    accounted = n_norm + len(st["allow_equal"]) == len(ALLOW_LEAF_PATHS)
    ok &= check(f"{prefix}c",
                not st["diffs"] and not st["opaque"] and accounted,
                f"L2 structural: {st['n_leaves']} leaves + {st['n_nodes']} nodes "
                f"compared exactly; {len(st['diffs'])} differences; {n_norm} "
                f"normalised {[x[0] for x in st['normalised']]}; "
                f"{len(st['opaque'])} opaque leaves; allowlist accounted "
                f"{accounted}")
    for kind, p, a, b in st["diffs"][:12]:
        print(f"            DIFF ({kind}) {p}\n              ref : {a}\n"
              f"              cand: {b}")
    for p, a, b in st["normalised"]:
        print(f"            NORM {p}: {a!r} -> {b!r}")

    ok &= check(f"{prefix}d", e["L3_md5"] == ref_md5,
                f"L3 whole-file: normalised md5 {e['L3_md5']} vs pre-registered "
                f"{ref_md5}   (literal cand md5 {e['literal_md5_cand']} "
                f"{'==' if e['literal_md5_cand'] == ref_md5 else '!='} ref)")
    return ok


# ===========================================================================
# --inspect
# ===========================================================================

def inspect():
    p = M / f"{R2_STEM}.pth"
    print(f"SERIALIZATION STRUCTURE of {p.name}")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    nodes, leaves, opaque = flatten(ck)
    stem, mem = zip_members(p)
    storages = [k for k in mem if k.startswith("data/")]
    kinds = {}
    for d in leaves.values():
        kinds[d[0]] = kinds.get(d[0], 0) + 1
    print(f"  file            : {p.stat().st_size} bytes  md5 {md5_bytes(p)}")
    print(f"  archive stem    : {stem!r}   {len(mem)} members "
          f"({len(storages)} tensor storages)")
    print(f"  non-storage     : {sorted(k for k in mem if not k.startswith('data/'))}")
    print(f"  top-level keys  : {list(ck)}")
    for k, v in ck.items():
        print(f"     {k:11s} {type(v).__name__}"
              + (f"[{len(v)}]" if isinstance(v, dict) else f" = {v!r}"[:60]))
    print(f"  graph           : {len(nodes)} container nodes, "
          f"{len(leaves)} leaves, {len(opaque)} opaque")
    print(f"  leaf kinds      : {dict(sorted(kinds.items()))}")
    print(f"  opaque leaves   : {opaque or 'none'}")
    print(f"  tag leaf        : {ALLOW_LEAF_PATHS[0]} = "
          f"{leaves[ALLOW_LEAF_PATHS[0]][1]!r}")
    others = [q for q in leaves if q.startswith("extra.config.") and q not in ALLOW_LEAF_PATHS]
    print(f"  config leaves   : {len(others)} others, all compared exactly")
    return []


# ===========================================================================
# --self-test : demonstrations A-E, item 6, item 8
# ===========================================================================

def deep_set(obj, path, value):
    """Minimal-copy set at a key path; preserves dict/list subtypes."""
    if not path:
        return value
    k, rest = path[0], path[1:]
    if isinstance(obj, dict):
        new = type(obj)(obj)
        new[k] = deep_set(obj[k], rest, value)
        return new
    if isinstance(obj, list):
        new = list(obj)
        new[k] = deep_set(obj[k], rest, value)
        return new
    raise TypeError(f"cannot descend into {type(obj).__name__} at {k!r}")


def deep_get(obj, path):
    for k in path:
        obj = obj[k]
    return obj


def tensor_with(t, i, fn):
    """Clone `t` and replace flat element i by fn(old value). Exact, in-place free."""
    t2 = t.clone()
    flat = t2.view(-1)
    flat[i] = fn(float(flat[i]))
    return t2


def self_test(scratch):
    print("PHASE 4.1 SELF-TEST -- equivalence machinery, no training required")
    orig = M / f"{R2_STEM}.pth"
    before = md5_bytes(orig)

    # ---- A. load the existing R2 checkpoint ---------------------------------
    ck = torch.load(orig, map_location="cpu", weights_only=False)
    out_dir = ck["extra"]["config"]["train"]["out_dir"]
    check("A", before == REF["pth_md5"],
          f"loaded {orig.name}: md5 {before} == pre-registered reference")

    # ---- item 6. same-basename round trip is byte-exact --------------------
    same, same_p = save_as(ck, R2_STEM, scratch, "rt")
    check("I6", same == REF["pth_md5"],
          f"torch.load -> torch.save under the SAME basename reproduces the "
          f"original md5 exactly ({same})")

    # ---- B. save an equivalent payload under a different basename ----------
    # This is exactly the shape the replication's own final file will have:
    # identical payload, tag = the replication tag, stem = the replication tag.
    cand = substitute(ck, REPRO_TAG, out_dir)
    cand_md5, cand_p = save_as(cand, REPRO_TAG, scratch, "b")
    check("B", cand_p.exists(),
          f"saved an equivalent payload as {cand_p.name} "
          f"(tag and stem both 'R2_traj_repro'), {cand_p.stat().st_size} bytes")

    # ---- C. literal md5 differs -------------------------------------------
    check("C", cand_md5 != before,
          f"literal whole-file md5 DIFFERS: {cand_md5} != {before}  "
          f"-- so a literal-identity criterion is unsatisfiable by construction")

    # ---- D. the normalised comparison is exact ----------------------------
    e = equivalence(orig, cand_p, scratch, "d", "R2 vs equivalent-payload copy")
    d_ok = report(e, REF["pth_md5"], "D1")
    check("D", d_ok, "L1+L2+L3 all exact on a payload-equivalent file")

    # ---- E. substantive changes are DETECTED ------------------------------
    # E0 first: prove the leaf comparator is strictly stronger than torch.equal
    # in BOTH directions, which is the reason it compares raw bytes.
    zp, zn = torch.zeros(1), torch.tensor([-0.0])
    n1, n2 = torch.tensor([float("nan")]), torch.tensor([float("nan")])
    check("E0", (describe_leaf(zp) != describe_leaf(zn)) and torch.equal(zp, zn)
          and (describe_leaf(n1) == describe_leaf(n2)) and not torch.equal(n1, n2),
          "leaf comparator separates +0.0 from -0.0 (torch.equal calls them equal) "
          "and calls NaN equal to an identically-encoded NaN (torch.equal does not)")

    # Each mutation below is the smallest representable change of its kind. Every
    # one must be caught; the tag-only case must be reported as normalised.
    W = ("actor", "actors.0.0.weight")
    up = lambda v: float(np.nextafter(np.float32(v), np.float32(np.inf)))
    nx = lambda v: float(np.nextafter(float(v), np.inf))
    cases = [
        ("E1  1-ULP flip in an actor weight",
         lambda c: deep_set(c, W, tensor_with(deep_get(c, W), 0, up)), "detect"),
        ("E2  1-ULP change in extra.mean_reward",
         lambda c: deep_set(c, ("extra", "mean_reward"),
                            nx(deep_get(c, ("extra", "mean_reward")))), "detect"),
        ("E3  a config field other than tag (train.seed +1)",
         lambda c: deep_set(c, ("extra", "config", "train", "seed"),
                            deep_get(c, ("extra", "config", "train", "seed")) + 1),
         "detect"),
        ("E4  1-ULP change in mappo_cfg.gamma",
         lambda c: deep_set(c, ("mappo_cfg", "gamma"),
                            nx(deep_get(c, ("mappo_cfg", "gamma")))), "detect"),
        ("E5  1-ULP flip in opt_actor.state[0].exp_avg (optimiser moment)",
         lambda c: deep_set(c, ("opt_actor", "state", 0, "exp_avg"),
                            tensor_with(deep_get(c, ("opt_actor", "state", 0,
                                                     "exp_avg")), 0, up)), "detect"),
        ("E6  optimiser step count (state[0].step + 1)",
         lambda c: deep_set(c, ("opt_actor", "state", 0, "step"),
                            deep_get(c, ("opt_actor", "state", 0, "step")) + 1),
         "detect"),
        ("E7  optimiser lr in opt_critic.param_groups[0]",
         lambda c: deep_set(c, ("opt_critic", "param_groups", 0, "lr"),
                            deep_get(c, ("opt_critic", "param_groups", 0, "lr"))
                            * 2.0), "detect"),
        ("E8  scalar metadata n_agents + 1",
         lambda c: deep_set(c, ("n_agents",), deep_get(c, ("n_agents",)) + 1),
         "detect"),
        ("E9  sign bit flipped in a critic bias element (covers 0.0 -> -0.0)",
         lambda c: deep_set(c, ("critic", "net.0.bias"),
                            tensor_with(deep_get(c, ("critic", "net.0.bias")), 0,
                                        lambda v: -v)),
         "detect"),
        ("E10 NaN injected into one actor bias element",
         lambda c: deep_set(c, ("actor", "actors.0.0.bias"),
                            tensor_with(deep_get(c, ("actor", "actors.0.0.bias")),
                                        0, lambda v: float("nan"))), "detect"),
        ("E11 dict key ORDER reversed in extra (schema drift)",
         lambda c: deep_set(c, ("extra",),
                            {k: deep_get(c, ("extra", k))
                             for k in reversed(list(deep_get(c, ("extra",))))}),
         "detect"),
        ("E12 tensor dtype float32 -> float64 in a critic bias",
         lambda c: deep_set(c, ("critic", "net.0.bias"),
                            deep_get(c, ("critic", "net.0.bias")).double()),
         "detect"),
        ("E13 an extra top-level key added",
         lambda c: {**c, "smuggled": 1}, "detect"),
        ("E14 extra.config.train.tag ONLY (the one allowed normalisation)",
         lambda c: substitute(c, "SOME_OTHER_TAG", out_dir), "normal"),
        ("E15 tag AND a 1-ULP weight flip together",
         lambda c: deep_set(substitute(c, "SOME_OTHER_TAG", out_dir), W,
                            tensor_with(deep_get(c, W), 1, up)), "detect"),
    ]

    rows = []
    for i, (name, fn, expect) in enumerate(cases):
        mutated = fn(ck)
        st = structural_compare(ck, mutated)
        # Also confirm the whole-file layer agrees, so a "detected" claim is not
        # resting on the structural walk alone.
        m, _ = save_as(substitute(mutated, R2_STEM, out_dir), R2_STEM,
                       scratch, f"e{i}")
        l3_flags = m != REF["pth_md5"]
        n_diff, n_norm = len(st["diffs"]), len(st["normalised"])
        if expect == "detect":
            ok = n_diff > 0 and l3_flags
        else:                       # normalised: no diffs, exactly one normalised
            ok = n_diff == 0 and n_norm == 1 and not l3_flags
        rows.append((name, expect, n_diff, n_norm, l3_flags, ok))
        first = st["diffs"][0][1] if st["diffs"] else "-"
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        print(f"            expect={expect:6s} L2 diffs={n_diff} normalised={n_norm} "
              f"L3 flags={l3_flags}  first diff at {first}")

    check("E", all(r[5] for r in rows),
          f"{sum(r[5] for r in rows)}/{len(rows)} mutation cases behaved as "
          f"specified ({sum(1 for r in rows if r[1] == 'detect')} must be detected, "
          f"{sum(1 for r in rows if r[1] == 'normal')} must be normalised)")

    # ---- item 8. the normalisation cannot hide a training difference -------
    _, mem = zip_members(orig)
    n_storage = sum(1 for k in mem if k.startswith("data/"))
    nodes, leaves, opq = flatten(ck)
    check("N1", len(ALLOW_LEAF_PATHS) == 1
          and ALLOW_LEAF_PATHS[0] == "extra.config.train.tag"
          and len(leaves) == REF["n_graph_leaves"]
          and len(nodes) == REF["n_graph_nodes"] and not opq,
          f"allowlist is exactly ONE literal leaf path -- {1}/{len(leaves)} leaves "
          f"({100.0 / len(leaves):.2f}%) -- and the graph matches the "
          f"pre-registered census ({len(leaves)} leaves, {len(nodes)} nodes, "
          f"{len(opq)} of unrecognised type)")
    check("N2", n_storage == REF["n_zip_storages"]
          and len(mem) == REF["n_zip_members"] and n_storage == 265,
          f"all {n_storage} tensor-storage members (one per tensor, of "
          f"{len(mem)} total) are compared as raw bytes with NO normalisation "
          f"applied to them at any point -- criterion *a")
    check("N3", TAG_AFFECTED_MEMBERS == {"data.pkl", ".data/serialization_id"},
          "the tag's measured blast radius is 2 container members (the pickle "
          "stream and its content hash); the stem's is member NAMES only")

    after = md5_bytes(orig)
    check("RO", after == before,
          f"the original R2 checkpoint is unmodified: md5 {after}")
    return _res


def verify_run(tag, scratch):
    print(f"PHASE 4.1 EQUIVALENCE for tag {tag!r}")
    pairs = [(M / f"{R2_STEM}.pth", M / f"{tag}.pth", REF["pth_md5"],
              "B1", "final"),
             (M / f"{R2_STEM}_best.pth", M / f"{tag}_best.pth",
              REF["best_norm_md5"], "B1e", "best")]
    for ref_p, cand_p, ref_md5, prefix, label in pairs:
        if not cand_p.exists():
            check(prefix, False, f"{cand_p.name} not found -- has the run been run?")
            continue
        print(f"  -- {label}: {ref_p.name}  vs  {cand_p.name}")
        report(equivalence(ref_p, cand_p, scratch, prefix, label), ref_md5, prefix)
    return _res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--tag", default=None)
    a = p.parse_args(argv)
    if not (a.inspect or a.self_test or a.tag):
        p.error("give --inspect, --self-test or --tag")

    scratch = tempfile.mkdtemp(prefix="p4equiv_")
    try:
        if a.inspect:
            inspect()
            print()
        if a.self_test:
            self_test(scratch)
        if a.tag:
            if a.self_test:
                print()
            verify_run(a.tag, scratch)
        n_bad = sum(1 for r in _res if not r["pass"])
        if _res:
            print("-" * 74)
            print(f"  {len(_res) - n_bad}/{len(_res)} criteria passed")
            if n_bad:
                print("  FAILURES:", ", ".join(r["criterion"] for r in _res
                                               if not r["pass"]))
        return 1 if n_bad else 0
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
