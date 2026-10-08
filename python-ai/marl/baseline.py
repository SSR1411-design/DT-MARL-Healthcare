"""
Baseline policies.

Sprint 6 only needs the minimum baseline infrastructure, and specifically a
REACTIVE scheduler that does not use predicted failure risk. That is
`ReactiveThresholdPolicy`. The full Sprint 9 baseline suite (HTCF, TGNN,
heuristic schedulers, other RL algorithms) is deliberately not implemented
here.

Two extra references are included because they cost a few lines each and
without them a learned result cannot be interpreted:

  * `NoMigrationPolicy`   — the do-nothing floor. Any policy that cannot beat
                            this has learned nothing useful.
  * `RiskThresholdPolicy` — a fixed threshold on predicted_failure_risk. This
                            is a HEURISTIC REFERENCE, not a Sprint 9 baseline:
                            it answers "does MAPPO do anything a one-line
                            threshold on the risk channel would not?", which
                            is the question the host-predictor audit taught us
                            to ask before believing any learned result.

Every policy takes (env, obs, masks) and returns one action per agent. All of
them respect the action mask, so no policy gets credit for an illegal move.

Sprint 9 adds `DetectionRecoveryPolicy`.  It deliberately requests no
proactive relocation: the environment's opt-in detection-recovery hook acts
only after its current availability physics observes a host outage.  Keeping
the action policy inert makes the causal boundary explicit: recovery is
failure detection -> restart/requeue, never prediction -> anticipation.
"""

import numpy as np

from marl.config import (
    ACTION_STAY, ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD,
    ACTION_PREEMPTIVE_REROUTE,
)


def _first_legal(mask, preference):
    for a in preference:
        if mask[a]:
            return a
    return ACTION_STAY


class BasePolicy:
    name = "base"
    uses_predicted_risk = False

    def reset(self):
        pass

    def act(self, env, obs, masks) -> np.ndarray:      # pragma: no cover
        raise NotImplementedError


class NoMigrationPolicy(BasePolicy):
    """Never relocate. The do-nothing floor."""

    name = "static-no-migration"

    def act(self, env, obs, masks):
        return np.zeros(env.n_agents, dtype=np.int64)


class RandomPolicy(BasePolicy):
    """Uniform over legal actions. Sanity reference for reward scale."""

    name = "random-legal"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, env, obs, masks):
        return np.array(
            [self.rng.choice(np.flatnonzero(masks[i])) for i in range(env.n_agents)],
            dtype=np.int64)


class ReactiveThresholdPolicy(BasePolicy):
    """
    REACTIVE SCHEDULING BASELINE — the one Sprint 6 asks for.

    Relocates a task only once the host it sits on shows a symptom that is
    already visible: `degraded` set, CPU above the ops threshold, or packet
    loss above the ops threshold (EnvConfig.reactive_*). It never reads
    predicted_failure_risk, so it can only respond after degradation has
    become measurable, not before.

    Prefers a reroute when the task has not started (free) and an edge
    migration otherwise, falling back to cloud.
    """

    name = "reactive-threshold"
    uses_predicted_risk = False

    def act(self, env, obs, masks):
        a = np.zeros(env.n_agents, dtype=np.int64)
        for i in range(env.n_agents):
            if env.focus_task(i) is None:
                continue
            if not env.symptomatic(i):
                continue
            a[i] = _first_legal(masks[i], (ACTION_PREEMPTIVE_REROUTE,
                                           ACTION_MIGRATE_EDGE,
                                           ACTION_MIGRATE_CLOUD))
        return a


class DetectionRecoveryPolicy(BasePolicy):
    """Detection-only self-healing control for Sprint 9 Baseline 3.

    The policy always selects ``STAY``.  A real recovery action is performed
    only by ``DTMarlEnv(detection_recovery=True)`` after the current transition
    observes a host outage.  That environment hook restarts affected work on a
    currently healthy edge node and appends a RecoveryRecord.  This class never
    reads risk, prediction windows, future trace values, or failure labels.
    """

    name = "detection-recovery-no-prediction"
    uses_predicted_risk = False

    def act(self, env, obs, masks):
        del env, obs
        return np.zeros(masks.shape[0], dtype=np.int64)


class RiskThresholdPolicy(BasePolicy):
    """
    HEURISTIC REFERENCE (not a Sprint 9 baseline).

    Relocate when predicted_failure_risk on the local node exceeds
    `threshold`. Optionally require the task to be at least
    `min_severity` severe, and send critical tasks to cloud.
    """

    name = "risk-threshold"
    uses_predicted_risk = True

    def __init__(self, threshold=0.18, min_severity=0.0, cloud_for_critical=0.5):
        # 0.18 is the operating point train_failure_predictor.py selected on
        # validation folds and stored in failure_predictor_meta.json.
        self.threshold = threshold
        self.min_severity = min_severity
        self.cloud_for_critical = cloud_for_critical
        self.name = f"risk-threshold@{threshold:g}"

    def act(self, env, obs, masks):
        a = np.zeros(env.n_agents, dtype=np.int64)
        for i in range(env.n_agents):
            t = env.focus_task(i)
            if t is None:
                continue
            if env.risk_at(i) <= self.threshold:
                continue
            if t.spec.severity < self.min_severity:
                continue
            if t.spec.severity >= self.cloud_for_critical:
                pref = (ACTION_MIGRATE_CLOUD, ACTION_MIGRATE_EDGE,
                        ACTION_PREEMPTIVE_REROUTE)
            else:
                pref = (ACTION_PREEMPTIVE_REROUTE, ACTION_MIGRATE_EDGE,
                        ACTION_MIGRATE_CLOUD)
            a[i] = _first_legal(masks[i], pref)
        return a


BASELINES = {
    "static": NoMigrationPolicy,
    "random": RandomPolicy,
    "reactive": ReactiveThresholdPolicy,
    "detection-recovery": DetectionRecoveryPolicy,
    "risk-threshold": RiskThresholdPolicy,
}

__all__ = ["BasePolicy", "NoMigrationPolicy", "RandomPolicy",
           "ReactiveThresholdPolicy", "DetectionRecoveryPolicy",
           "RiskThresholdPolicy", "BASELINES"]
