"""
SCRATCH validation of the regenerated host-failure dataset.

Checks the simulation-layer acceptance criteria only. Does NOT train
anything and does NOT touch the ML pipeline.
"""
import numpy as np
import pandas as pd

CSV = "simulation/failure_history.csv"
LOG = "simulation/failure_log.csv"

OBS = ["cpu", "ram", "bandwidth", "energy", "runningTasks", "active",
       "degraded", "linkUp", "linkBandwidthMbps", "linkLatencyMs",
       "linkPacketLoss", "underAttack"]

df = pd.read_csv(CSV)
log = pd.read_csv(LOG)

print("=" * 72)
print("1) DATASET SHAPE")
print("=" * 72)
ticks = df["time"].nunique()
print(f"rows={len(df)}  nodes={df.nodeId.nunique()}  distinct ticks={ticks}")
print(f"time range: {df.time.min():.2f} -> {df.time.max():.2f}s")
print(f"positives={int(df.willFailSoon.sum())}  "
      f"negatives={int((df.willFailSoon == 0).sum())}  "
      f"positive rate={df.willFailSoon.mean()*100:.2f}%")

print("\nevent counts in failure_log.csv:")
print(log.type.value_counts().to_string())

hf = log[log.type == "HOST_FAILURE"]
print(f"\nHOST_FAILURE events: {len(hf)} across {hf.nodeId.nunique()} nodes")
print("per node:", hf.groupby("nodeId").size().to_dict())
print("causes  :", hf.cause.value_counts().to_dict())
rec = log[log.type == "HOST_RECOVERED"]
print(f"HOST_RECOVERED events: {len(rec)}  -> repeated failure/recovery cycles")

print("\n" + "=" * 72)
print("2) OBSERVABLE COLUMN VARIANCE (was: 5 of 12 constant)")
print("=" * 72)
for c in OBS:
    v = df[c]
    print(f"  {c:20s} std={v.std():10.4f}  min={v.min():9.3f}  max={v.max():9.3f}"
          f"  {'CONSTANT' if v.std() < 1e-9 else ''}")

print("\n" + "=" * 72)
print("3) DOES DEGRADATION PRECEDE FAILURE?")
print("=" * 72)
leads = []
for _, ev in hf.iterrows():
    n, ft = int(ev.nodeId), ev.time
    g = df[(df.nodeId == n) & (df.time < ft)].sort_values("time")
    # walk back while state is a degradation state, uninterrupted
    seq = g.audit_healthState.values
    k = 0
    while k < len(seq) and seq[-1 - k] in ("DEGRADING", "CRITICAL"):
        k += 1
    leads.append((n, ft, k))

lead_arr = np.array([l[2] for l in leads])
print(f"observable-degradation lead time before each HOST_FAILURE (ticks):")
print(f"  mean={lead_arr.mean():.1f}  median={np.median(lead_arr):.1f}  "
      f"min={lead_arr.min()}  max={lead_arr.max()}")
print(f"  failures with >=10 ticks of warning (>= label horizon): "
      f"{(lead_arr >= 10).sum()}/{len(lead_arr)}")
print(f"  failures with ZERO warning (abrupt): {(lead_arr == 0).sum()}/{len(lead_arr)}")
print("  distribution:", np.sort(lead_arr).tolist())

print("\n" + "=" * 72)
print("4) EXAMPLE TRAJECTORIES (last 14 ticks before failure)")
print("=" * 72)
shown = 0
for n, ft, k in leads:
    if k < 12 or shown >= 3:
        continue
    g = df[(df.nodeId == n) & (df.time <= ft)].sort_values("time").tail(15)
    print(f"\n--- node {n}, failure at t={ft:.2f}, {k} ticks of visible degradation ---")
    print(g[["time", "cpu", "ram", "bandwidth", "energy", "runningTasks",
             "linkLatencyMs", "linkPacketLoss", "linkBandwidthMbps",
             "degraded", "willFailSoon", "audit_healthState", "audit_wear"]]
          .to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
    shown += 1

print("\n" + "=" * 72)
print("5) ARE TRAJECTORIES DIVERSE? (per-failure signal deltas over last 10 ticks)")
print("=" * 72)
rows = []
for n, ft, k in leads:
    g = df[(df.nodeId == n) & (df.time <= ft)].sort_values("time").tail(11)
    if len(g) < 11:
        continue
    a, b = g.iloc[0], g.iloc[-2]  # exclude the failure row itself
    rows.append(dict(node=n, t=round(ft, 1), lead=k,
                     d_cpu=b.cpu - a.cpu, d_ram=b.ram - a.ram,
                     d_bw=b.bandwidth - a.bandwidth,
                     d_energy=b.energy - a.energy,
                     d_lat=b.linkLatencyMs - a.linkLatencyMs,
                     d_loss=b.linkPacketLoss - a.linkPacketLoss))
tr = pd.DataFrame(rows)
print(tr.to_string(index=False, float_format=lambda x: f"{x:8.2f}"))
print("\nper-channel spread across failure events (std of the deltas):")
print(tr[["d_cpu", "d_ram", "d_bw", "d_energy", "d_lat", "d_loss"]]
      .std().to_string())

print("\n" + "=" * 72)
print("6) LEAKAGE CHECK: is any observable column a function of the")
print("   scheduled/realised failure time?")
print("=" * 72)
alive = df[df.active == 1]
print("Pearson r between each OBSERVABLE column and audit_secondsToFailure,")
print("restricted to rows where a future failure exists (alive rows only):")
sub = alive[alive.audit_secondsToFailure >= 0]
for c in OBS:
    if sub[c].std() < 1e-12:
        print(f"  {c:20s}   n/a (constant on this subset)")
        continue
    r = np.corrcoef(sub[c], sub.audit_secondsToFailure)[0, 1]
    print(f"  {c:20s} r={r:+.4f}")

exact = [c for c in OBS
         if np.allclose(df[c].values, df.audit_nextFailureTime.values)
         or np.allclose(df[c].values, df.audit_secondsToFailure.values)]
print(f"\ncolumns byte-identical to a failure-time column: {exact or 'NONE'}")

print("\ncheck that willFailSoon is reconstructible from the event log alone:")
ok = True
for _, r in df.iterrows():
    exp = 1 if (0 < r.audit_secondsToFailure <= r.audit_predictionHorizon) else 0
    if exp != r.willFailSoon:
        ok = False
        break
print(f"  willFailSoon == (0 < secondsToFailure <= horizon) for every row: {ok}")

print("\n" + "=" * 72)
print("7) TRIVIAL-RULE SANITY (the old dataset collapsed to 'linkUp==1')")
print("=" * 72)
y = df.willFailSoon.values
for name, pred in [
    ("always-negative", np.zeros_like(y)),
    ("linkUp==1", (df.linkUp == 1).astype(int).values),
    ("active==1", (df.active == 1).astype(int).values),
    ("degraded==1", (df.degraded == 1).astype(int).values),
]:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (pred == y).mean() * 100
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"  {name:16s} acc={acc:6.2f}%  prec={prec:.4f}  rec={rec:.4f}  F1={f1:.4f}")

print("\n" + "=" * 72)
print("8) DUPLICATE-WINDOW CEILING on the raw observable rows")
print("=" * 72)
key = df[OBS].round(2).astype(str).agg("|".join, axis=1)
grp = pd.DataFrame({"k": key, "y": y}).groupby("k").y.agg(["sum", "count"])
amb = grp[(grp["sum"] > 0) & (grp["sum"] < grp["count"])]
amb_pos = int(amb["sum"].sum())
print(f"distinct rounded observation vectors: {len(grp)}")
print(f"positives sitting on an ambiguous vector (same obs, both labels): "
      f"{amb_pos}/{int(y.sum())} = {amb_pos/max(int(y.sum()),1)*100:.1f}%")
print("(previously 100% at the single-row level; lower is better)")
