"""
Sprint 6 configuration: environment, reward, MAPPO, training.

Everything tunable lives here as a dataclass field with a documented default.
No coefficient is hardcoded inside the environment, the reward, or the
learner. `asdict` on any of these is JSON-serialisable, so every run saves
the exact configuration that produced it.

---------------------------------------------------------------------------
REWARD EQUATION (implemented in env.py::DTMarlEnv._agent_reward)
---------------------------------------------------------------------------

For agent i at decision step t, holding focus task k, over the transition
(t -> t+1):

    crit_i       = 1 + w_criticality           * severity(k)
    crit_migr_i  = 1 + w_criticality_migration * severity(k_moved)

where k is the focus task for outcome terms and k_moved is the task actually
relocated this step (event field `migration_severity`; 0 when nothing moved, so
crit_migr_i = 1). The two are the same task whenever a relocation happens --
relocation always acts on the focus task -- but they are tracked in separate
event fields on purpose: agent i can move one task while a DIFFERENT task it
owns completes or is lost in the same step, and writing the moved task's
severity into the shared `severity` field would silently re-weight the
completion/loss/SLA terms. Sprint 6 read the shared field here, which is never
populated on the relocation path, so crit_migr_i was identically 1.0 and
w_criticality_migration was dead code (see Sprint 6.5 diagnosis).

    r_i =   R_complete    * crit_i      * 1[k completed during the step]
          + R_progress    * sum_j crit_j * work_fraction_j(t)
          - P_task_lost   * crit_i      * 1[k lost during the step]
          - P_sla         * crit_i      * 1[k breached its deadline this step]
          - P_migration   * crit_migr_i * migration_cost(action, distance)
          - P_energy                    * energy_i(t)
          - P_risk_expose * crit_i      * risk_i(t) * 1[k left resident on i]
          - P_infeasible                * 1[action was infeasible]
          - P_overload                  * max(0, load_i(t+1) - overload_target)
          + team_reward_share * ( global_step_reward(t) / n_edge_nodes )

    global_step_reward(t) = R_complete * (#completions this step)
                          - P_task_lost * (#losses this step)
                          - P_balance   * load_imbalance(t+1)

severity(k) is the Sprint 5 clinical severity, reused verbatim (see
marl/criticality.py). risk_i(t) is the continuous predicted_failure_risk at
the CURRENT step (never a future value, never a ground-truth label).
energy_i(t) is the node power recorded in the Digital Twin trace at t.
work_fraction_j(t) is the share of task j's total length completed during this
step, summed over every task j owned by i and weighted by that task's OWN
criticality crit_j rather than by the focus task's (see RewardConfig.R_progress
for why the term exists).
Every outcome term is realised no later than t+1.

---------------------------------------------------------------------------
WHO IS CHARGED: reward attribution
---------------------------------------------------------------------------

Outcome terms (completion, loss, SLA, energy) are charged to the task's
`reward_owner`, which is the agent accountable for the task's CURRENT
placement, not merely whichever node it happens to sit on:

  * admitted        -> the arrival node
  * relocation      -> the DECIDING agent, for the whole transfer, so choosing
                       a destination that then fails costs the chooser
  * lands on edge d -> agent d, which can now act on the task
  * lands on cloud  -> stays with the agent that sent it there, because no
                       agent is resident on the cloud

This matters more than it looks. Deriving the owner from the residency field
credits nothing for a cloud-resident task (its node id is the CLOUD_NODE_ID
sentinel, not an agent index), so a cloud offload erased the task's completion,
loss, SLA and energy from every agent's ledger. Measured on 6 fixed episodes, a
random-legal policy completed 27.5 of 40 tasks but was credited for only ~8.8 of
them, while a never-migrate policy completed 18 and was credited for all 18.
Relocating therefore looked strictly worse than it was, and MAPPO correctly
learned to collapse onto STAY. Exposure and overload remain charged to the
RESIDENT agent, because those are properties of where the task physically sits.

CAUSALITY. Every symbol above is either (a) known at t, or (b) an outcome of
the action taken at t observed at t+1. Nothing in the reward reads the
failure event log ahead of the current step, `willFailSoon`, or any audit_*
column. The trace's *availability* channel drives the environment's physics
(a host that the simulator recorded as down at t is down at t), but that is
the environment transition, not an observation and not a reward look-ahead.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parents[1]          # python-ai/
REPO = ROOT.parent                                   # repository root


# ==========================================================================
# Environment
# ==========================================================================

@dataclass
class EnvConfig:
    """Physical / workload configuration of the DT-MARL replay environment."""

    # ---- data sources (recorded Digital Twin trace from the Java sim) ----
    trace_csv: str = str(REPO / "simulation" / "failure_history.csv")
    event_log_csv: str = str(REPO / "simulation" / "failure_log.csv")

    # ---- risk source -----------------------------------------------------
    # "oof"   : leakage-safe out-of-fold sigmoid scores (default). Every
    #           window was scored by a model that never trained on it.
    # "model" : live inference with saved_models/failure_predictor.pth. In
    #           sample for 4/5 of this trace, so only for deployment demos.
    # "zero"  : risk channel forced to 0 - the ablation that shows whether
    #           the policy actually uses the prediction.
    risk_source: str = "oof"
    oof_npz: str = str(ROOT / "saved_models" / "failure_predictor_oof.npz")
    model_path: str = str(ROOT / "saved_models" / "failure_predictor.pth")
    scaler_path: str = str(ROOT / "saved_models" / "failure_predictor_scaler.npz")
    meta_path: str = str(ROOT / "saved_models" / "failure_predictor_meta.json")
    sequence_length: int = 10

    # ---- cluster ---------------------------------------------------------
    n_edge_nodes: int = 10          # == HostManager.HOST_COUNT
    vms_per_node: int = 2           # == VmManager.VMS_PER_HOST
    vm_mips: float = 2000.0         # VmSimple(1000 MIPS, 2 PEs)
    queue_capacity: int = 4         # resident-but-not-running tasks per node

    # ---- cloud tier ------------------------------------------------------
    cloud_slots: int = 8
    cloud_mips: float = 4000.0
    cloud_wan_latency_ms: float = 40.0
    cloud_energy_multiplier: float = 1.6

    # ---- workload (Sprint 5 CloudletManager constants) -------------------
    n_tasks: int = 40               # CloudletManager.TASK_COUNT
    n_patients: int = 10            # CloudletManager.PATIENT_COUNT
    task_length_mi: float = 600_000.0
    task_pes: int = 2
    # Sprint 5's absolute deadlines (60 + 5k s) are shorter than the nominal
    # runtime of a 600k MI task on a 2000 MIPS VM (300 s), so replaying them
    # literally would make every task breach its SLA and the metric carry no
    # information. The env therefore expresses the deadline as slack over
    # nominal runtime while preserving Sprint 5's *relative* ordering
    # (task k always gets more slack than task k-1). Clinical severity is
    # untouched - it depends only on patient attributes.
    #   nominal runtime = 600_000 MI / 2000 MIPS = 300 s
    #   deadline        = arrival + 300 s * 1.35 = arrival + 405 s
    # One edge migration (30 s) still fits; three (90 s) do not. That is the
    # trade-off the SLA term is supposed to express.
    sla_slack_factor: float = 1.35
    deadline_slack_per_task_s: float = 5.0

    # ---- episode / stepping ---------------------------------------------
    ticks_per_step: int = 2         # decision interval = 2 recorded ticks = 2 s
    episode_steps: int = 400        # 800 recorded ticks of cluster time
    # Arrivals are spread across most of the episode rather than bunched at
    # the start. Two reasons: a continuous admission stream is the realistic
    # regime, and it keeps agents holding relocatable tasks throughout the
    # episode instead of idling once the early cohort drains. Measured under a
    # non-relocating policy, the fraction of agent-steps that offer a real
    # choice (>= 2 legal actions) rose 0.566 -> 0.728 going from 150 to 220.
    # Last arrival at step 220 + 150 steps nominal runtime = 370 < 400.
    #
    # NOTE for anyone reading MAPPO's `decision_frac` diagnostic: under a
    # policy that relocates heavily that figure is ~0.04, and that is not a
    # masking defect. A task in transit is resident on no node, so no agent
    # holds it as a focus task; 88 relocations x 15+ steps of transfer removes
    # most tasks from the decision set. Hop-cap exhaustion accounts for a
    # further ~3 agents/step. Density recovers as the policy learns to stay.
    task_arrival_span_steps: int = 220
    random_episode_start: bool = True
    # Episode-start window, in recorded ticks, as a fraction of the usable
    # range. Training and evaluation are given DISJOINT windows so reported
    # results are not measured on the same stretch of trace the policy trained
    # on. Per-node actors can otherwise specialise to the empirical failure
    # times of their own node, and a single 1500-tick trace gives no held-out
    # cluster to test that against.
    start_frac_lo: float = 0.0
    start_frac_hi: float = 1.0

    # ---- physics ---------------------------------------------------------
    # A recorded `degraded` host is slower (thermal throttling / disk
    # saturation). Observable-derived, so it is legitimate physics.
    degraded_throughput_penalty: float = 0.35
    # MIGRATION MODEL: checkpoint-and-resume, not pre-copy. The task's state
    # is transferred and execution resumes on the destination, so once the
    # transfer starts the task no longer shares the source's fate — that is
    # what makes proactive migration protective at all. What it costs is
    # TRANSFER TIME, during which the task makes no progress, plus energy and
    # the explicit migration penalty. If the DESTINATION fails mid-transfer
    # the task is lost, so migrating into a bad node is punished.
    #   VmSimple size = 10 000 MB image / 2048 MB RAM. Moving ~2 GB of task
    #   state over the recorded ~500 Mbps LAN is tens of seconds; over a WAN
    #   to cloud, a few times that. Hence 30 s edge / 80 s cloud.
    migration_latency_steps_edge: int = 15      # 30 s
    migration_latency_steps_cloud: int = 40     # 80 s
    # A reroute only applies to a task that has not started computing: there
    # is no state to copy, so it lands on the next step.
    reroute_latency_steps: int = 1
    max_migrations_per_task: int = 3

    # ---- reactive-symptom trigger (shared by the baseline and by the
    #      preemptive/reactive classification of MARL migrations) ----------
    # These are ordinary ops thresholds on OBSERVABLE channels. They contain
    # no predicted risk, which is what makes the reactive baseline a genuine
    # non-predictive control.
    reactive_cpu_threshold: float = 85.0
    reactive_packet_loss_threshold: float = 20.0

    # ---- destination selector weights (observable-only scoring) ----------
    dest_w_risk: float = 1.0
    dest_w_free_capacity: float = 0.6
    dest_w_link_latency: float = 0.3
    dest_w_load: float = 0.4

    # ---- topology --------------------------------------------------------
    # The repository has no pairwise adjacency model (mirrorNetworkLinks
    # creates one link per node), so neighbourhood is an explicit, documented
    # configuration choice: a ring with the given offsets.
    neighbour_offsets: List[int] = field(default_factory=lambda: [-2, -1, 1, 2])

    # ---- normalisation ---------------------------------------------------
    energy_norm_w: float = 250.0
    running_tasks_norm: float = 8.0          # observed max VMs per host
    link_bw_norm_mbps: float = 500.0
    link_latency_norm_ms: float = 10_000.0   # 9999 is the link-down sentinel
    overload_target: float = 1.0             # running tasks per node

    # EVALUATION ONLY. Window (in decision steps) after a migration in which
    # a real recorded failure of the SOURCE host counts the migration as
    # having protected the task. Reads the trace ahead of the migration, so it
    # is computed strictly in the post-episode metrics path and never enters
    # an observation, an action mask, or the reward.
    protection_window_steps: int = 30        # 60 s

    # ---- Sprint 6.5 ablation switches (BOTH DEFAULT OFF = fixed behaviour) --
    # These exist so the Sprint 6 baseline (arm A0) stays exactly re-runnable
    # from the same codebase after the two defects it contained were fixed, and
    # so each fix can be attributed independently. Turn them ON to reproduce
    # the DEFECTIVE Sprint 6 behaviour. They are not tuning knobs.
    #
    # legacy_fixed_apply_order: apply agent actions in the fixed order
    #   0..n-1 instead of rotating by step index. A committed transfer
    #   immediately consumes a destination slot, so the fixed order handed
    #   every contested slot to the lowest-index demander. Measured: the
    #   lowest-index demander won 100% of contentions, versus 3/21 after the
    #   fix. With separate per-agent actors that gives ten nominally
    #   homogeneous agents ten different effective action spaces.
    #
    # legacy_dead_migration_criticality: read the migration charge's severity
    #   from the shared `severity` event field, which the relocation path never
    #   writes, instead of from `migration_severity`. That pinned crit_migr to
    #   1.0 at every migration, so w_criticality_migration was dead code and
    #   patient criticality could not influence migration cost at all.
    #
    # NOTE: neither switch changes the refusal RATE under contention. That was
    # ~70% before and after, and it is inherent to simultaneous action on a
    # scarce shared destination -- each agent's mask is necessarily computed
    # without knowing what the others will choose in the same step.
    legacy_fixed_apply_order: bool = False
    legacy_dead_migration_criticality: bool = False

    seed: int = 20260818


# ==========================================================================
# Reward
# ==========================================================================

@dataclass
class RewardConfig:
    """
    Multi-objective reward coefficients. See the module docstring for the
    full equation. Magnitudes are deliberately of the same order so that no
    single objective dominates by construction; the intent is a genuine
    trade-off, not a disguised single-objective problem.
    """

    R_complete: float = 10.0
    # WORK-IN-PROGRESS credit, paid as computation is actually performed: a task
    # advanced by a fraction f of its total length this step earns
    # R_progress * crit * f. Integrated over a task carried from 0 to done this
    # totals exactly R_progress * crit, so it is a REDISTRIBUTION of completion
    # value over the task's lifetime rather than an extra prize on top of it.
    #
    # WHY IT EXISTS. The delay, not the magnitude, was the problem. Median task
    # lifetime is 131-149 decision steps, while GAE's own averaging horizon is
    # 1/(1 - gamma*lambda) ~ 20 steps and the critic measured only +0.08
    # explained variance - so nothing carried the completion bonus back to the
    # placement decision that earned it. Every term the agent COULD see inside
    # its horizon was a cost, and three training runs duly learned to hold fewer
    # tasks (completions 30 -> 20, losses 6 -> 16). This puts a positive term
    # inside the horizon.
    #
    # SIZING at R_complete, i.e. a task's total value is split half between
    # finishing it and half between the work of getting there. It is charged to
    # the task's `reward_owner`, so a task migrated mid-flight simply stops
    # earning during the transfer and resumes under whoever owns it next; there
    # is deliberately NO clawback of progress already credited. A potential-based
    # formulation would be provably policy-invariant but requires exactly that
    # clawback on handoff, which reintroduces the immediate anti-migration cost
    # this term is meant to offset. This is therefore a genuine change to the
    # objective, documented here and guarded by tests_env.py::t15.
    R_progress: float = 10.0
    P_task_lost: float = 20.0
    P_sla: float = 5.0
    # A migration costs ~1/5 of losing a task: you should be willing to move a
    # task several times to save it, but not to spray migrations for free.
    P_migration: float = 4.0
    P_energy: float = 0.1
    # Dense shaping for "a task left sitting on a node the predictor flags is
    # accumulating exposure".
    #
    # SIZING, because this term is charged EVERY step while the one-off
    # migration charge is paid ONCE, and getting the ratio wrong inverts the
    # decision. A task runs for ~150 decision steps, so for a mid-severity task
    # (severity 0.5 -> crit 2.0, crit_m 1.5):
    #
    #   lifetime exposure of staying  = P_risk_expose * 2.0 * risk * 150
    #   cost of one edge migration    = P_migration(4.0) * 1.5 * 1.0 = 6.0
    #
    # The sign should flip near the risk level at which moving is actually
    # worthwhile. Setting P_risk_expose = 0.15 puts the crossover at
    # 0.15*2*150*risk = 6.0 -> risk = 0.13, and at risk 0.9 exposure is 40.5
    # against a cost of 6.0, so high risk still strongly favours moving.
    # At the previous value of 1.0 the crossover sat at risk = 0.02, i.e. the
    # shaping term demanded migration at essentially any risk at all and
    # swamped the true objective it was meant to support.
    P_risk_expose: float = 0.15
    P_infeasible: float = 0.5
    P_overload: float = 0.5
    P_balance: float = 1.0

    # Criticality amplifiers. A maximally severe patient's task counts
    # (1 + w) times a zero-severity one for loss / SLA / completion, and
    # unnecessary movement of a critical task costs more than of a routine
    # one - which is what "migrated unnecessarily" has to mean numerically.
    w_criticality: float = 2.0
    w_criticality_migration: float = 1.0

    # Relative cost of the three ways a task can move.
    migration_cost_edge: float = 1.0
    migration_cost_cloud: float = 2.0
    reroute_cost: float = 0.25

    # Fraction of the global (team) step reward mixed into every agent's
    # reward, so credit assignment is not purely local.
    team_reward_share: float = 0.3

    reward_scale: float = 0.1     # keeps returns in a range PPO likes


# ==========================================================================
# MAPPO
# ==========================================================================

@dataclass
class MappoConfig:
    """
    MAPPO hyper-parameters. `separate_actors=True` means each agent owns its
    own actor parameters - not one shared network evaluated N times.
    """

    actor_hidden: List[int] = field(default_factory=lambda: [128, 128])
    critic_hidden: List[int] = field(default_factory=lambda: [256, 256])
    separate_actors: bool = True

    lr_actor: float = 7e-4
    lr_critic: float = 1e-3
    # DISCOUNT HORIZON. This has to be large, and the reason is measured, not
    # assumed: the outcome a placement decision causes is whether the task
    # COMPLETES, and a task takes a median of 131-149 decision steps to finish
    # (measured over 6 episodes under two fixed policies). So the completion
    # bonus lands ~150 steps after the decision that earns it, while every cost
    # it trades against - migration charge, exposure, energy - is paid
    # immediately. gamma must keep that delayed bonus visible:
    #
    #   gamma    0.95^149     0.99^149     0.997^149    0.999^149
    #   weight   0.00048      0.2237       0.6391       0.8615
    #
    # WHY THIS IS NOT A TUNING CHOICE. Ranking two fixed policies (no-migration
    # vs random-legal) by mean return-from-decision-step:
    #
    #   gamma          no-migr    random-legal   random better by
    #   undiscounted   -93.68     -57.98         +35.70   <- the reported objective
    #   0.95            -6.56     -11.10          -4.55   <- RANKING INVERTED
    #   0.99           -22.12     -21.92          +0.20   <- no signal
    #   0.997          -34.20     -24.53          +9.67
    #   0.999          -39.70     -24.19         +15.51
    #
    # At 0.95 the discounted objective PREFERS the worse policy, so PPO
    # correctly ascending its own objective necessarily descends the reported
    # one. At 0.99 the two are indistinguishable. Both predictions were
    # confirmed the expensive way: three training runs degraded monotonically
    # (gamma 0.99: -121 -> -173; gamma 0.95: -90 -> -170; gamma 0.95 again after
    # the reward-attribution fix: -38 -> -92, with completions falling 30 -> 20
    # and losses rising 6 -> 16 exactly as a cost-only visible horizon predicts).
    # The attribution fix in env.py was necessary but not sufficient; this is the
    # other half. See marl/_diag_horizon.py for the measurement and
    # tests_env.py::t15 for the regression guard.
    #
    # An earlier comment here justified 0.95 on the grounds that a relocation
    # "only has causal consequences for ~15 steps of transfer plus the ~30-step
    # protection window". That was the error: it described the duration of the
    # MECHANISM, not the horizon of the OUTCOME. It is recorded rather than
    # deleted because it is the mistake that cost three runs.
    #
    # 0.999 rather than 0.997: measured on 6 fixed episodes, the SEPARATION
    # between the two policies, relative to the baseline's own magnitude, keeps
    # growing as gamma -> 1, and 0.997 gives up a third of it.
    #
    #   gamma    no-migr   random   margin   margin / |no-migr|
    #   0.99      -2.212   -2.192   +0.020   0.0089   <- no signal at all
    #   0.997     -3.420   -2.453   +0.967   0.2828
    #   0.999     -3.970   -2.419   +1.551   0.3907
    #   0.9995    -4.128   -2.391   +1.737   0.4208
    #   1.0       -4.296   -2.353   +1.943   0.4522
    #
    # 0.999 keeps 86% of the separation available at gamma=1. Its effective
    # horizon (1000 steps) is longer than the 400-step episode, so discounting
    # is nearly inert WITHIN an episode - which is the point, because the
    # reported objective is the undiscounted episode return. A little
    # discounting is retained rather than going to exactly 1.0 as a numerical
    # safeguard on the time-limit bootstrap term.
    gamma: float = 0.999
    # GAE lambda. The advantage's own averaging horizon is 1/(1-gamma*lambda),
    # and THAT, not gamma, is what bounds how far a consequence can reach back to
    # the action that caused it. At gamma=0.999:
    #
    #   lambda   1/(1-gamma*lambda)     vs median task lifetime 131-149 steps
    #   0.90       9.1 steps            16x too short
    #   0.95      19.6 steps             7.6x too short   <- the failed run
    #   0.99      91.0 steps             still short
    #   0.995    166.9 steps             MATCHES the measured lifetime
    #   1.0      episode (pure MC)       unbiased, highest variance
    #
    # An earlier comment here argued for 0.95 as "the standard MAPPO value" and
    # the right division of labour, on the grounds that the centralised critic
    # would carry the long delay because the global state contains every task's
    # remaining work. That reasoning is recorded rather than deleted because the
    # run it produced falsified it, and the way it failed is worth keeping:
    #
    #   The critic's regression target is ret = adv + V, i.e. a LAMBDA-return.
    #   With lambda=0.95 that target only looks ~20 steps ahead and then trusts
    #   V for the rest, so the critic converges to a fixed point of its own
    #   short-sighted operator. Explained variance reached +0.84, which looks
    #   like a healthy critic but only measures self-consistency against a biased
    #   target - it is not evidence that the critic knows the true 150-step
    #   value. Propagating value back 150 steps at ~20 steps per update needs
    #   ~8 sequential update-hops through a policy that is changing underneath
    #   it. Over 75 updates that never converged: value_mean rose -0.51 -> +0.83
    #   (PPO improving its own objective) while the undiscounted episode return
    #   fell +8.59 -> -25.80 (the reported objective getting worse), with
    #   relocations flat at ~88 and losses climbing 11.6 -> 15.5.
    #
    # 0.995 sets the advantage horizon to ~167 steps, chosen to cover the
    # MEASURED task lifetime rather than picked for numerical taste. The cost is
    # variance, which is affordable here: 8 episodes x 400 steps x 10 agents =
    # 32,000 agent-steps per update, and advantages are normalised.
    gae_lambda: float = 0.995
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    # Raised from 0.01: with 10 independent actors each seeing only ~6k
    # decisions, the first run's entropy fell 0.86 -> 0.20 while performance got
    # worse, i.e. it committed before it had learned anything.
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatches: int = 4
    normalise_advantages: bool = True
    # SPRINT 7 RUNG 2. The CRITIC's regression target. The actor's GAE is not
    # affected by this flag in any way: gamma, gae_lambda and the advantage
    # returned by compute_gae are identical under both settings.
    #
    #   "lambda"  ret = adv + V, i.e. the lambda-return. PRODUCTION / A0
    #             DEFAULT. Self-referential: the target is a function of the
    #             critic's own current predictions.
    #   "mc"      within-episode Monte-Carlo discounted return, bootstrapped
    #             ONLY on genuine time-limit truncation. Contains no critic
    #             term for an episode that ends in a true terminal state.
    #
    # WHY THIS FLAG EXISTS. The gae_lambda comment above already recorded the
    # suspicion that "the critic converges to a fixed point of its own
    # short-sighted operator", and responded by raising lambda 0.95 -> 0.995.
    # Sprint 7 Rung 1 measured that this was necessary but not sufficient. On
    # arm A0, refitting ONLY the critic on MC returns and changing nothing else:
    #
    #                                   lambda-return (A0)   MC refit
    #   high-risk GAE sign agreement          26.3%            60.7%
    #   high-risk MIGRATE_EDGE advantage     -2.867           +0.840   (truth +3.363)
    #   residual-vs-risk slope, hi-only      -5.596           -0.254
    #   val explained variance                0.681            0.821
    #
    # The production critic predicts V=+3.66 for high-risk states whose true
    # return is -0.72, and that risk-correlated over-valuation is what inverts
    # the sign of the learning target. An arm that kept the clipped value loss
    # and changed only the target also passed, so value_clip_eps is NOT the
    # cause and is deliberately left alone. See
    # saved_models/marl/SPRINT_7_RUNG1_REPORT.md.
    #
    # The cost being accepted is target VARIANCE: MC returns are unbiased but
    # noisier than lambda-returns. Rung 1's val EV of 0.821 says they remain
    # fittable at this scale, and that is the main way this can still fail.
    critic_target: str = "lambda"
    # Linear decay of both learning rates to 0 over training. Standard PPO
    # practice; materially steadies the late-training policy.
    anneal_lr: bool = True


@dataclass
class TrainConfig:
    """Training loop configuration. Deliberately small: research prototype."""

    episodes: int = 600
    # 8 episodes per update rather than 4. Only ~4% of agent-steps are genuine
    # decisions (see EnvConfig.task_arrival_span_steps), so a 4-episode rollout
    # gave each PPO minibatch only ~160 decision entries to average over - far
    # too few for a stable policy-gradient step across 10 separate actors.
    rollout_episodes: int = 8      # episodes per PPO update
    seed: int = 20260818
    device: str = "auto"           # "auto" | "cpu" | "cuda"
    log_every: int = 10
    eval_episodes: int = 20
    out_dir: str = str(ROOT / "saved_models" / "marl")
    tag: str = "mappo"


@dataclass
class Sprint6Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    mappo: MappoConfig = field(default_factory=MappoConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self):
        return asdict(self)


# ==========================================================================
# Action space
# ==========================================================================

ACTION_STAY = 0
ACTION_MIGRATE_EDGE = 1
ACTION_MIGRATE_CLOUD = 2
ACTION_PREEMPTIVE_REROUTE = 3

ACTION_NAMES = {
    ACTION_STAY: "STAY",
    ACTION_MIGRATE_EDGE: "MIGRATE_TO_NEIGHBOR_EDGE",
    ACTION_MIGRATE_CLOUD: "MIGRATE_TO_CLOUD",
    ACTION_PREEMPTIVE_REROUTE: "PREEMPTIVE_REROUTE",
}

N_ACTIONS = 4

CLOUD_NODE_ID = -2      # distinct from HealthcareTask.UNASSIGNED_NODE (-1)


def resolve_device(spec: str = "auto") -> str:
    if spec != "auto":
        return spec
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:                                   # pragma: no cover
        return "cpu"


__all__ = [
    "EnvConfig", "RewardConfig", "MappoConfig", "TrainConfig", "Sprint6Config",
    "ACTION_STAY", "ACTION_MIGRATE_EDGE", "ACTION_MIGRATE_CLOUD",
    "ACTION_PREEMPTIVE_REROUTE", "ACTION_NAMES", "N_ACTIONS", "CLOUD_NODE_ID",
    "resolve_device", "ROOT", "REPO",
]
