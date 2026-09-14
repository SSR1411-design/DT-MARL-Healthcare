"""
PER-EVENT lead-time evaluation for the host failure predictor.

Answers one question per failure event: did the model raise an alert on the
right host, before the failure, inside the labelled warning horizon, and was
that alert actually informative rather than an alarm that was already stuck
on beforehand?

WHY THIS SCORES OUT-OF-FOLD PREDICTIONS

The previous version loaded the saved deployment model and scanned the whole
trace. That model is trained on 4/5 of the windows, so most of the events it
"detected early" were events it had trained on — an in-sample lead time. This
version reads the pooled out-of-fold probabilities written by
train_failure_predictor.py, so every window's score comes from a model that
never saw that window's temporal block, and each window is thresholded with
its own fold's validation-selected threshold. Lead time and the headline
F1/PR-AUC therefore describe the same system under the same protocol.

WHY A CONSTANT ALARM CANNOT SCORE HERE

"First alert before the failure" is trivially maximised by an alarm that is
on from t=0. Every event is therefore also given a QUIET-PERIOD alarm rate:
the model's alert rate on that same host over the 60 ticks that precede the
warning horizon, all of which are labelled negative. A detection is credited
as early warning only if that rate is below 0.5 — i.e. the alarm was not
already continuously active before the predictable window opened. Detections
that fail this test are reported as STANDING_ALARM and are excluded from the
credited lead-time statistics, not quietly folded into them.

The audit_healthState column is read here for CONTEXT ONLY — to report how
many ticks of observable degradation each event actually had. It is never a
feature; see data/failure_dataset.assert_no_leakage.
"""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "training"))

from data.failure_dataset import (  # noqa: E402
    build_windows, OBSERVABLE_COLUMNS, LEGACY_RAW_FEATURES,
)
from host_cv import CSV_PATH, SEQ_LEN  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--features", choices=["observable", "legacy"],
                default="observable")
ap.add_argument("--csv", default=str(CSV_PATH),
                help="telemetry export the OOF file was produced from")
ap.add_argument("--log", default=None,
                help="failure_log.csv holding the HOST_FAILURE events; "
                     "defaults to the sibling of --csv")
ap.add_argument("--tag", default=None, help="artifact suffix override")
ap.add_argument("--event-types", default="HOST_FAILURE,NETWORK_FAILURE",
                help="comma-separated failure types to score. willFailSoon is "
                     "labelled from any node outage, so both types belong here; "
                     "the new export happens to contain only HOST_FAILURE.")
args = ap.parse_args()

RAW = OBSERVABLE_COLUMNS if args.features == "observable" else LEGACY_RAW_FEATURES
if args.tag is not None:
    SUFFIX = f"_{args.tag}"
else:
    SUFFIX = "" if args.features == "observable" else "_legacy"

# Label horizon used by the simulator's exportLabeledCsv(..., 10.0).
HORIZON = 10.0
# Length of the labelled-negative "quiet" stretch used for the standing-alarm
# test, immediately before the horizon opens.
QUIET_TICKS = 60.0
# A detection is credited only if the quiet-period alarm rate is below this.
STANDING_ALARM_RATE = 0.5

LOG_PATH = (Path(args.log) if args.log
            else Path(args.csv).resolve().parent / "failure_log.csv")
save_dir = ROOT / "saved_models"

# --------------------------------------------------------------------------
# Load out-of-fold predictions and align them to (node, tick).
# --------------------------------------------------------------------------

oof = np.load(save_dir / f"failure_predictor{SUFFIX}_oof.npz")
X, y, node_ids, event_ids, end_times = build_windows(
    args.csv, SEQ_LEN, raw_features=RAW, return_times=True)

if not np.array_equal(y, oof["y"]) or not np.array_equal(node_ids, oof["node_ids"]):
    raise SystemExit("OOF file does not match the current dataset/feature arm; "
                     "re-run train_failure_predictor.py first.")

prob, pred, covered = oof["prob"], oof["pred"], oof["covered"]
print(f"Feature arm       : {args.features} ({len(RAW)} raw columns)")
print(f"Fold thresholds   : "
      f"{np.round(oof['fold_thresholds'], 3).tolist()}")
print(f"Windows scored OOF: {covered.sum()}/{len(y)}")
print(f"Global alert rate : {(pred[covered] == 1).mean()*100:.2f}% of all "
      f"windows  <- a value near 100% means every 'lead time' below is an "
      f"artefact")

df = pd.read_csv(args.csv)
log = pd.read_csv(LOG_PATH)
WANTED = [t.strip() for t in args.event_types.split(",") if t.strip()]
events = (log[log.type.isin(WANTED)][["nodeId", "time", "type"]]
          .sort_values(["time"]).reset_index(drop=True))
print(f"Failure events    : {len(events)} across "
      f"{events.nodeId.nunique()} hosts  "
      f"({log[log.type.isin(WANTED)].type.value_counts().to_dict()})")

# --------------------------------------------------------------------------
# Per-node views, indexed by window-end tick.
# --------------------------------------------------------------------------

per_node = {}
for n in np.unique(node_ids):
    m = node_ids == n
    order = np.argsort(end_times[m])
    per_node[int(n)] = dict(
        t=end_times[m][order],
        y=y[m][order],
        p=prob[m][order],
        a=pred[m][order],
        cov=covered[m][order],
    )

# Observable-degradation lead per event, for context only. The pre-degradation
# export has no audit_healthState column at all (that failure model had no
# health state to record), so this degrades to 0 rather than failing — which is
# itself the honest answer for that dataset: no observable precursor existed.
has_health = "audit_healthState" in df.columns
deg_lead = {}
for _, ev in events.iterrows():
    n, ft = int(ev.nodeId), float(ev.time)
    if not has_health:
        deg_lead[(n, round(ft, 2))] = 0
        continue
    seq = df[(df.nodeId == n) & (df.time < ft)].sort_values("time")
    states = seq["audit_healthState"].values
    k = 0
    while k < len(states) and states[-1 - k] in ("DEGRADING", "CRITICAL"):
        k += 1
    deg_lead[(n, round(ft, 2))] = k
if not has_health:
    print("NOTE: no audit_healthState column in this export; the 'deg' column "
          "is reported as 0 throughout.")

# --------------------------------------------------------------------------
# Score every event.
# --------------------------------------------------------------------------

rows = []
for _, ev in events.iterrows():
    n, ft = int(ev.nodeId), float(ev.time)
    v = per_node[n]

    # The labelled warning band: ticks t with ft in (t, t+HORIZON].
    band = (v["t"] >= ft - HORIZON) & (v["t"] < ft) & v["cov"]
    # The labelled-negative stretch just before it (standing-alarm test).
    quiet = ((v["t"] >= ft - HORIZON - QUIET_TICKS)
             & (v["t"] < ft - HORIZON) & v["cov"] & (v["y"] == 0))

    n_band = int(band.sum())
    if n_band == 0:
        rows.append(dict(node=n, t_fail=ft, verdict="UNMEASURABLE",
                         lead=np.nan, run_lead=np.nan, band_alarm=np.nan,
                         quiet_alarm=np.nan, n_band=0,
                         deg=deg_lead[(n, round(ft, 2))], max_p=np.nan))
        continue

    band_alerts = v["a"][band] == 1
    quiet_rate = (float((v["a"][quiet] == 1).mean())
                  if quiet.sum() > 0 else np.nan)

    if not band_alerts.any():
        rows.append(dict(node=n, t_fail=ft, verdict="MISSED", lead=np.nan,
                         run_lead=np.nan, band_alarm=0.0,
                         quiet_alarm=quiet_rate, n_band=n_band,
                         deg=deg_lead[(n, round(ft, 2))],
                         max_p=float(v["p"][band].max())))
        continue

    band_t = v["t"][band]
    first_t = float(band_t[np.argmax(band_alerts)])
    lead = ft - first_t

    # Walk backwards from that alert through the contiguous alarm run, to see
    # how much earlier the alarm actually came on.
    gi = int(np.searchsorted(v["t"], first_t))
    j = gi
    while j - 1 >= 0 and v["cov"][j - 1] and v["a"][j - 1] == 1:
        j -= 1
    run_lead = ft - float(v["t"][j])

    standing = (not np.isnan(quiet_rate)) and quiet_rate >= STANDING_ALARM_RATE
    rows.append(dict(node=n, t_fail=ft,
                     verdict="STANDING_ALARM" if standing else "EARLY_WARNING",
                     lead=lead, run_lead=run_lead,
                     band_alarm=float(band_alerts.mean()),
                     quiet_alarm=quiet_rate, n_band=n_band,
                     deg=deg_lead[(n, round(ft, 2))],
                     max_p=float(v["p"][band].max())))

res = pd.DataFrame(rows)

print("\n" + "=" * 88)
print("PER-EVENT DETECTION (out-of-fold predictions only)")
print("=" * 88)
print("  deg        = ticks of observable degradation before the failure "
      "(audit; context only)")
print("  lead       = ticks between the first in-horizon alert and the failure")
print("  run_lead   = ticks back to the start of that contiguous alarm run")
print("  band_alarm = share of the 10 in-horizon windows that alerted")
print("  quiet_alarm= alert rate over the 60 labelled-negative ticks before "
      "the horizon")
print()
hdr = (f"{'node':>4} {'t_fail':>9} {'deg':>4} {'lead':>5} {'run':>5} "
       f"{'band':>5} {'quiet':>6} {'maxP':>6}  verdict")
print(hdr)
print("-" * len(hdr))
for _, r in res.iterrows():
    print(f"{r.node:>4d} {r.t_fail:>9.2f} {r.deg:>4.0f} "
          f"{r.lead if not np.isnan(r.lead) else float('nan'):>5.1f} "
          f"{r.run_lead if not np.isnan(r.run_lead) else float('nan'):>5.1f} "
          f"{r.band_alarm*100 if not np.isnan(r.band_alarm) else float('nan'):>4.0f}% "
          f"{r.quiet_alarm*100 if not np.isnan(r.quiet_alarm) else float('nan'):>5.1f}% "
          f"{r.max_p:>6.3f}  {r.verdict}")

# --------------------------------------------------------------------------
# Aggregates.
# --------------------------------------------------------------------------

ok = res[res.verdict == "EARLY_WARNING"]
standing = res[res.verdict == "STANDING_ALARM"]
missed = res[res.verdict == "MISSED"]
unmeas = res[res.verdict == "UNMEASURABLE"]

print("\n" + "=" * 88)
print("EVENT-LEVEL SUMMARY")
print("=" * 88)
print(f"  failure events                      : {len(res)}")
print(f"  detected as genuine early warning   : {len(ok)}")
print(f"  detected but STANDING ALARM (voided): {len(standing)}")
print(f"  missed entirely                     : {len(missed)}")
print(f"  unmeasurable (no covered window)    : {len(unmeas)}")
print(f"  event-level recall (credited only)  : "
      f"{len(ok)/max(len(res),1)*100:.1f}%")

if len(ok):
    print(f"\n  credited lead time (ticks = simulated seconds):")
    print(f"    mean={ok.lead.mean():.2f}  median={ok.lead.median():.2f}  "
          f"min={ok.lead.min():.0f}  max={ok.lead.max():.0f}")
    print(f"    distribution: {sorted(ok.lead.astype(int).tolist())}")
    print(f"  alarm-run start lead (how early the alarm first came on):")
    print(f"    mean={ok.run_lead.mean():.2f}  median={ok.run_lead.median():.2f}"
          f"  max={ok.run_lead.max():.0f}")
    print(f"  mean quiet-period alarm rate on credited events: "
          f"{ok.quiet_alarm.mean()*100:.2f}%")

# Detection vs how much warning the physics actually gave.
print("\n  detection rate by available observable degradation:")
for lo, hi, lab in [(0, 3, "0-2 ticks (abrupt, no usable precursor)"),
                    (3, 10, "3-9 ticks (shorter than the label horizon)"),
                    (10, 10 ** 9, ">=10 ticks (full horizon of warning)")]:
    sub = res[(res.deg >= lo) & (res.deg < hi)]
    if len(sub) == 0:
        continue
    got = (sub.verdict == "EARLY_WARNING").sum()
    print(f"    {lab:44s} {got}/{len(sub)} detected")

print("\n  VALIDITY CHECKS")
glob_rate = (pred[covered] == 1).mean()
print(f"    global out-of-fold alert rate        : {glob_rate*100:.2f}% "
      f"({'OK' if glob_rate < 0.10 else 'TOO HIGH - alarm is near-constant'})")
if len(res[~res.quiet_alarm.isna()]):
    mq = res.quiet_alarm.mean(skipna=True)
    print(f"    mean quiet-period alarm rate (all)   : {mq*100:.2f}% "
          f"({'OK' if mq < STANDING_ALARM_RATE else 'FAIL - standing alarm'})")
print(f"    events credited on a standing alarm  : 0 by construction "
      f"({len(standing)} were detected but voided)")

res.to_csv(save_dir / f"host_lead_time{SUFFIX}.csv", index=False)
print(f"\nWrote saved_models/host_lead_time{SUFFIX}.csv")
