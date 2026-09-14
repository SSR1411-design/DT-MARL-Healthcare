import json

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------------------
# Host / network failure-prediction feature pipeline.
#
# The Java simulator exports failure_history.csv with 12 observable
# telemetry columns plus a willFailSoon label and five audit_* columns.
#
# WHICH COLUMNS ARE ALLOWED TO BE FEATURES
#
#   * OBSERVABLE_COLUMNS — everything a real monitoring agent could read off
#     a live host. This is the whitelist.
#
#   * AUDIT_COLUMNS — never features, under any circumstance. They exist so
#     that the *dataset* can be validated (does degradation actually precede
#     failure? is the label reconstructible?), and every one of them encodes
#     the future: audit_healthState is the simulator's latent wear state,
#     audit_secondsToFailure / audit_nextFailureTime are the failure time
#     itself. `assert_no_leakage` is called on every feature list built here
#     so a typo cannot silently admit one.
#
# WHY THE WHITELIST CHANGED
#
#   The previous version of this file hardcoded 7 raw columns, having
#   dropped cpu / ram / bandwidth / runningTasks / underAttack because they
#   were CONSTANT for the whole run in the old dataset — zero variance, zero
#   signal, and a divide-by-zero in the scaler.
#
#   That was a property of the old failure model (instant death at a
#   scheduled time, no symptoms), not of the columns. The simulator now runs
#   a latent-wear + fault-mechanism model that writes a mechanism-specific
#   symptom overlay onto exactly those channels, and 0 of the 12 observable
#   columns are constant any more. Applying the same rule ("keep the columns
#   that vary") to the new export therefore readmits all 12.
#
#   LEGACY_RAW_FEATURES is kept so the old 7-column arm can still be run
#   under an identical protocol and the two reported side by side.
#
# ENGINEERED FEATURES are causal first differences plus "time since the node
# was last seen active / link last seen up". They use only ticks <= t, so
# they leak no future information; they are the standard age / elapsed-time
# hazard features from reliability engineering.
#
#   The old file carried a serious caveat here: because hosts never
#   recovered, `active` was a monotone step function and time_since_active
#   was effectively a per-node clock, so the model could memorise "node k
#   alerts at counter value K". That no longer holds — hosts now fail AND
#   recover repeatedly (32 failures, 32 recoveries over 10 nodes), the
#   counters reset on every recovery, and their observed range is 0..44
#   rather than growing without bound. The counters are now genuinely
#   "time since this node's last outage".
# --------------------------------------------------------------------------

# Every column a live monitoring agent could observe. The whitelist.
OBSERVABLE_COLUMNS = [
    "cpu", "ram", "bandwidth", "energy", "runningTasks",
    "active", "degraded", "linkUp",
    "linkBandwidthMbps", "linkLatencyMs", "linkPacketLoss", "underAttack",
]

# Simulator-internal ground truth. NEVER a feature — see assert_no_leakage.
AUDIT_COLUMNS = [
    "audit_healthState", "audit_wear", "audit_nextFailureTime",
    "audit_secondsToFailure", "audit_predictionHorizon",
    "degradationStartTime",
]

# Also forbidden as features: the label itself and the bookkeeping columns.
FORBIDDEN_COLUMNS = AUDIT_COLUMNS + ["willFailSoon", "time", "nodeId"]

# The 7-column list the old (pre-degradation) dataset's variance audit left
# behind. Retained only so that arm can be reproduced for comparison.
LEGACY_RAW_FEATURES = [
    "energy", "active", "degraded", "linkUp",
    "linkBandwidthMbps", "linkLatencyMs", "linkPacketLoss",
]

# Default raw block: the whole whitelist, since nothing is constant now.
RAW_FEATURES = OBSERVABLE_COLUMNS


def assert_no_leakage(raw_features):
    """
    Hard guard: refuse to build features from any audit-only / label /
    bookkeeping column. Cheap, and it makes a leak impossible to introduce
    by editing a list.
    """
    bad = [c for c in raw_features
           if c in FORBIDDEN_COLUMNS or c.startswith("audit_")]
    if bad:
        raise ValueError(
            f"refusing to use non-observable columns as features: {bad}")
    return list(raw_features)


def feature_columns(raw_features=None):
    """Engineered feature order (raw + causal deltas + causal counters)."""
    raw = assert_no_leakage(raw_features or RAW_FEATURES)
    return raw + [f"d_{c}" for c in raw] + [
        "time_since_active", "time_since_linkup"]


# Default engineered feature order / width.
FEATURE_COLUMNS = feature_columns(RAW_FEATURES)

NUM_FEATURES = len(FEATURE_COLUMNS)  # 26 (12 raw + 12 deltas + 2 counters)


def engineer_group(group, raw_features=None):
    """
    Build the engineered feature matrix for a single node's time-ordered
    telemetry. Every derived feature is CAUSAL (uses only ticks <= t), so
    no future information is leaked into any window.

    Returns a (T, F) float32 array aligned with the group rows.
    """

    raw_features = assert_no_leakage(raw_features or RAW_FEATURES)

    group = group.sort_values("time").reset_index(drop=True)

    raw = group[raw_features].values.astype(np.float64)

    # First differences: prepend the first row so delta[0] == 0 (causal;
    # no look-ahead). Captures "energy just dropped", "link just went down".
    deltas = np.diff(raw, axis=0, prepend=raw[[0]])

    # Time since the node was last observed active (resets to 0 whenever
    # active==1). A pure past-only counter — the reliability "age" feature.
    active = group["active"].values
    tsa = np.zeros(len(group))
    c = 0
    for i in range(len(group)):
        c = 0 if active[i] == 1 else c + 1
        tsa[i] = c

    # Time since the link was last observed up.
    link_up = group["linkUp"].values
    tsl = np.zeros(len(group))
    c = 0
    for i in range(len(group)):
        c = 0 if link_up[i] == 1 else c + 1
        tsl[i] = c

    feats = np.concatenate(
        [raw, deltas, tsa[:, None], tsl[:, None]], axis=1
    )

    return feats.astype(np.float32)


def build_windows(csv_path, sequence_length=10, raw_features=None,
                  return_times=False):
    """
    Build sliding-window sequences per node.

    Each sample is (sequence_length, F); the label is willFailSoon at the
    LAST tick of the window ("given the last N ticks, is this host about to
    fail?").

    Also returns, per window:
      * node_ids   — the node the window belongs to.
      * event_ids  — a leakage-safe group id. Windows are grouped into
        contiguous same-node temporal blocks so that a grouped
        train/test split never puts two OVERLAPPING windows (which share
        up to sequence_length-1 ticks) on opposite sides of the split.
      * end_times  — (only with return_times=True) the simulation clock of
        the window's LAST tick. Needed to line predictions up against the
        failure event log for per-event lead time; it is bookkeeping, never
        a feature.
    """

    raw_features = assert_no_leakage(raw_features or RAW_FEATURES)

    df = pd.read_csv(csv_path)

    sequences, labels, node_ids, event_ids, end_times = [], [], [], [], []

    for node_id, group in df.groupby("nodeId"):

        feats = engineer_group(group, raw_features)
        ordered = group.sort_values("time").reset_index(drop=True)
        targets = ordered["willFailSoon"].values.astype(np.int64)
        times = ordered["time"].values.astype(np.float64)

        for i in range(len(feats) - sequence_length + 1):
            end = i + sequence_length - 1
            sequences.append(feats[i:i + sequence_length])
            labels.append(targets[end])
            node_ids.append(int(node_id))
            end_times.append(times[end])
            # Block size == sequence_length keeps within-block overlap
            # contained; combined with the purge in split_grouped this
            # removes cross-split window overlap.
            event_ids.append(int(node_id) * 100000 + end // sequence_length)

    X = np.stack(sequences).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    node_ids = np.array(node_ids, dtype=np.int64)
    event_ids = np.array(event_ids, dtype=np.int64)

    if return_times:
        return X, y, node_ids, event_ids, np.array(end_times, dtype=np.float64)

    return X, y, node_ids, event_ids


def purge_overlap(train_idx, test_idx, node_ids, event_ids):
    """
    Drop any training window that temporally overlaps a test window of the
    SAME node.

    Adjacent sliding windows share up to sequence_length-1 ticks, so a
    grouped split alone is not sufficient: a training window one block away
    from a test window still overlaps it. Block ids are consecutive integers
    within a node (see build_windows), and a block spans sequence_length
    consecutive window ends, so overlap reaches exactly one block either
    side — purging |block delta| <= 1 is both necessary and sufficient.

    Lives here (rather than in the training script) so that every evaluation
    script applies the identical purge and the numbers stay comparable.
    """

    test_blocks = {}
    for j in test_idx:
        test_blocks.setdefault(node_ids[j], set()).add(event_ids[j])

    keep = [
        i for i in train_idx
        if not any(abs(event_ids[i] - tb) <= 1
                   for tb in test_blocks.get(node_ids[i], ()))
    ]

    return np.array(keep, dtype=int)


# --------------------------------------------------------------------------
# Normalization — fit ONLY on training windows, applied to everything else.
# Persisted so lead-time evaluation and Sprint 6 use identical scaling.
# --------------------------------------------------------------------------

class Standardizer:

    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def fit(self, X):
        # X: (N, T, F) -> stats over (N*T) per feature.
        flat = X.reshape(-1, X.shape[-1])
        self.mean = flat.mean(axis=0)
        self.std = flat.std(axis=0)
        self.std[self.std < 1e-8] = 1.0  # guard constant/near-constant
        return self

    def transform(self, X):
        return ((X - self.mean) / self.std).astype(np.float32)

    def save(self, path):
        np.savez(path, mean=self.mean, std=self.std)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(mean=d["mean"], std=d["std"])


def save_meta(path, threshold, feature_columns, sequence_length):
    with open(path, "w") as f:
        json.dump(
            {
                "threshold": float(threshold),
                "feature_columns": list(feature_columns),
                "sequence_length": int(sequence_length),
                "num_features": len(feature_columns),
            },
            f,
            indent=2,
        )


def load_meta(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Backward-compatible Dataset wrapper (kept so existing imports still work).
# Now uses engineered features. Normalization is applied if a fitted
# Standardizer is supplied.
# --------------------------------------------------------------------------

class FailureHistoryDataset(Dataset):

    def __init__(self, csv_path, sequence_length=10, standardizer=None):
        X, y, node_ids, event_ids = build_windows(csv_path, sequence_length)
        if standardizer is not None:
            X = standardizer.transform(X)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int64)
        self.node_ids = node_ids
        self.event_ids = event_ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(csv_path, sequence_length=10, batch_size=32, train_split=0.8):
    """
    Kept for backward compatibility. NOTE: the training script now uses
    grouped, leakage-safe cross-validation in build_windows/split_grouped
    instead of this random split. This helper standardizes using train
    statistics only, so it no longer leaks scale information, but it still
    splits windows randomly and is retained only for legacy callers.
    """

    X, y, _, _ = build_windows(csv_path, sequence_length)

    n = len(y)
    idx = np.arange(n)
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    cut = int(n * train_split)
    tr_idx, te_idx = idx[:cut], idx[cut:]

    std = Standardizer().fit(X[tr_idx])
    Xtr = torch.tensor(std.transform(X[tr_idx]))
    Xte = torch.tensor(std.transform(X[te_idx]))
    ytr = torch.tensor(y[tr_idx])
    yte = torch.tensor(y[te_idx])

    train_ds = torch.utils.data.TensorDataset(Xtr, ytr)
    test_ds = torch.utils.data.TensorDataset(Xte, yte)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
