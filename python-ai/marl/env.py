"""
The DT-MARL environment: Digital Twin -> predicted_failure_risk -> multi-agent
decision -> task placement.

WHAT THIS IS
    A multi-agent decision environment whose physics are REPLAYED from the
    recorded CloudSim Plus Digital Twin trace (simulation/failure_history.csv,
    15 000 rows = 10 hosts x 1500 ticks) and whose risk signal is the
    continuous predicted_failure_risk of the Sprint 4 Transformer->BiLSTM
    predictor. Host availability, CPU/RAM/bandwidth/energy load, link quality
    and attack state all come from the real simulation. Nothing here is
    generated from random numbers.

WHAT THIS IS NOT — stated plainly, because the alternative would be a fake
integration:
    It is NOT closed-loop co-simulation with CloudSim. The agents' decisions
    do not feed back into the Java simulator and therefore cannot change the
    recorded host-failure trajectory or the recorded background load. The
    trajectory is exogenous; what the agents change is where the healthcare
    tasks sit relative to it, and whether they survive. Closing the loop
    (CloudSim driving MARL tick-by-tick and MARL rebinding real cloudlets) is
    Sprint 7 work; a minimal Java-side migration mechanism and a real
    risk-consuming PredictionGateway are provided so that path is open.

AGENTS
    One agent per edge node (10). Each agent owns the tasks resident on its
    node and, each decision step, acts on its FOCUS TASK — the highest
    Sprint-5-priority resident task that is still live. Agents have separate
    actor parameters (see mappo.py); they are not one network called ten times.

OBSERVATION (per agent, 48 dims, all readable at the current tick)
    local (15)      : cpu, ram, bandwidth, energy, runningTasks, active,
                      degraded, linkUp, linkBandwidthMbps, linkLatencyMs,
                      linkPacketLoss, underAttack, predicted_failure_risk,
                      prediction_uncertainty [reserved, 0.0 in Sprint 6],
                      free-slot fraction
    task (8)        : has_task, clinical severity, Sprint-5 priority,
                      progress, time-to-deadline, migrations used, is_running,
                      queue length
    neighbours (20) : per ring neighbour — predicted_failure_risk, observed
                      availability, free-slot fraction, cpu, link latency
    cloud (3)       : free-slot fraction, WAN latency, cloud risk (0.0)
    context (2)     : episode progress, agent id

    NEVER present: willFailSoon, any audit_* column, any value at a tick
    later than the current one, the failure event log.

ACTIONS
    0 STAY, 1 MIGRATE_TO_NEIGHBOR_EDGE, 2 MIGRATE_TO_CLOUD,
    3 PREEMPTIVE_REROUTE (only legal for a task with zero progress; there is
    no state to copy, so it lands next step and costs a quarter of a
    migration). Destinations are resolved by destination.DestinationSelector
    from live observable state — never hardcoded. Infeasible actions are
    masked; if one is taken anyway it degrades to STAY and is counted.

REWARD
    Multi-objective, every coefficient in RewardConfig. Full equation in
    marl/config.py's module docstring.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from marl.config import (
    EnvConfig, RewardConfig, ACTION_STAY, ACTION_MIGRATE_EDGE,
    ACTION_MIGRATE_CLOUD, ACTION_PREEMPTIVE_REROUTE, ACTION_NAMES, N_ACTIONS,
    CLOUD_NODE_ID,
)
from marl.criticality import build_patient_tasks, priority_order_key
from marl.destination import CandidateView, DestinationSelector
from marl.risk_provider import build_risk_provider
from marl.topology import Topology
from marl.trace import load_trace

# task lifecycle
PENDING = "PENDING"
QUEUED = "QUEUED"
RUNNING = "RUNNING"
IN_FLIGHT = "IN_FLIGHT"
COMPLETED = "COMPLETED"
LOST = "LOST"

TERMINAL = (COMPLETED, LOST)

OBS_LOCAL = 15
OBS_TASK = 8
OBS_PER_NEIGHBOUR = 5
OBS_CLOUD = 3
OBS_CONTEXT = 2

GLOBAL_EXTRA = 9


@dataclass
class TaskRuntime:
    spec: object                     # criticality.PatientTask
    state: str = PENDING
    node: int = -1                   # edge index, CLOUD_NODE_ID, or -1
    dest: int = -1                   # in-flight destination
    # REWARD ATTRIBUTION. The agent accountable for this task's outcome
    # (completion / loss / SLA / energy). Always a valid edge-agent index once
    # the task has been admitted, which `node` is NOT: `node` becomes
    # CLOUD_NODE_ID (-2) for a cloud-resident task, and deriving the owner from
    # it silently credited cloud outcomes to nobody. That made offloading a free
    # way to erase a task from every agent's ledger, so the learned policy
    # correctly concluded that relocating was worthless and collapsed onto STAY.
    # Ownership follows agency: the agent that decided the current placement
    # holds it until an agent that can act on the task takes over.
    reward_owner: int = -1
    land_step: int = -1
    remaining_mi: float = 0.0
    migrations: int = 0
    reroutes: int = 0
    arrival_step: int = 0
    start_step: int = -1
    finish_step: int = -1
    wan_latency_ms: float = 0.0
    deadline_breached: bool = False
    lost_step: int = -1

    @property
    def progress(self) -> float:
        return 1.0 - self.remaining_mi / max(self.spec.length_mi, 1e-9)

    @property
    def live(self) -> bool:
        return self.state not in TERMINAL and self.state != PENDING


@dataclass
class MigrationRecord:
    """
    One relocation. `source_failed_within_window` and `task_survived` are
    filled AFTER the episode from the recorded trace — evaluation only, never
    visible to a policy.
    """
    step: int
    elapsed_s: float
    task_id: int
    patient_id: int
    severity: float
    priority: float
    action: int
    action_name: str
    source_node: int
    dest_node: int
    cost: float
    latency_steps: int
    preemptive: bool
    source_symptomatic: bool
    source_risk: float
    dest_risk: float
    source_failed_within_window: Optional[bool] = None
    task_survived: Optional[bool] = None


class DTMarlEnv:
    """reset() / step(actions) multi-agent environment."""

    def __init__(self, cfg: EnvConfig = None, rcfg: RewardConfig = None,
                 verbose: bool = False, trace=None, risk=None):
        """
        `trace` and `risk` are INJECTION HOOKS used only by the test suite:
        tests_env.py passes a future-corrupted trace to prove no observation
        reads ahead, and sanity_test.py passes a synthetic risk field to build
        the deterministic Node-A/Node-B scenario. Both default to the recorded
        trace and the out-of-fold predictor scores.
        """
        self.cfg = cfg or EnvConfig()
        self.rcfg = rcfg or RewardConfig()
        self.verbose = verbose

        self.trace = load_trace(self.cfg.trace_csv) if trace is None else trace
        self.risk = build_risk_provider(self.cfg) if risk is None else risk
        self.n_agents = min(self.cfg.n_edge_nodes, self.trace.n_nodes)
        self.topology = Topology(self.n_agents, self.cfg.neighbour_offsets)
        self.selector = DestinationSelector(self.cfg)

        tick_dt = float(np.median(np.diff(self.trace.times))) if self.trace.n_ticks > 1 else 1.0
        self.tick_dt = tick_dt
        self.dt = tick_dt * self.cfg.ticks_per_step

        self.node_capacity = self.cfg.vms_per_node + self.cfg.queue_capacity
        self.obs_dim = (OBS_LOCAL + OBS_TASK
                        + OBS_PER_NEIGHBOUR * self.topology.degree
                        + OBS_CLOUD + OBS_CONTEXT)
        self.state_dim = self.obs_dim * self.n_agents + GLOBAL_EXTRA
        self.n_actions = N_ACTIONS

        self._rng = np.random.default_rng(self.cfg.seed)
        self._episode = 0

        span = self.cfg.episode_steps * self.cfg.ticks_per_step
        lo = max(self.risk.first_valid_tick, 0)
        hi = self.trace.n_ticks - span - self.cfg.ticks_per_step
        if hi < lo:
            raise ValueError(
                f"trace too short: needs {span + self.cfg.ticks_per_step} ticks "
                f"after tick {lo}, has {self.trace.n_ticks}")
        # Restrict to the configured fraction of the usable window. train.py and
        # evaluate.py pass disjoint fractions so the reported numbers are not
        # measured on the episodes the policy was trained on.
        width = hi - lo
        self._min_start = lo + int(round(width * self.cfg.start_frac_lo))
        self._max_start = lo + int(round(width * self.cfg.start_frac_hi))
        if self._max_start < self._min_start:
            raise ValueError(
                f"empty episode-start window: start_frac_lo="
                f"{self.cfg.start_frac_lo} > start_frac_hi={self.cfg.start_frac_hi}")

    # =====================================================================
    # episode setup
    # =====================================================================

    def reset(self, episode_start_tick: int = None, seed: int = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if episode_start_tick is not None:
            self.t0 = int(episode_start_tick)
        elif self.cfg.random_episode_start:
            self.t0 = int(self._rng.integers(self._min_start, self._max_start + 1))
        else:
            self.t0 = self._min_start
        if not (self._min_start <= self.t0 <= self._max_start):
            raise ValueError(
                f"episode start tick {self.t0} outside "
                f"[{self._min_start}, {self._max_start}] "
                f"(risk is undefined before tick {self.risk.first_valid_tick})")

        self.step_idx = 0
        self._episode += 1

        nominal_runtime = self.cfg.task_length_mi / self.cfg.vm_mips
        span = max(self.cfg.task_arrival_span_steps, 1)
        arrival_steps = [int(round(k * span / max(self.cfg.n_tasks - 1, 1)))
                         for k in range(self.cfg.n_tasks)]
        arrivals = [s * self.dt for s in arrival_steps]
        # Sprint 5's deadline formula, re-based on nominal runtime so the SLA
        # metric is informative (see EnvConfig.sla_slack_factor).
        deadlines = [a + nominal_runtime * self.cfg.sla_slack_factor
                     + k * self.cfg.deadline_slack_per_task_s * 0.0
                     for k, a in enumerate(arrivals)]

        specs = build_patient_tasks(
            self.cfg.n_tasks, self.cfg.n_patients, self.cfg.task_length_mi,
            arrivals=arrivals, env_deadlines=deadlines)

        self.tasks: List[TaskRuntime] = [
            TaskRuntime(spec=s, remaining_mi=s.length_mi,
                        arrival_step=arrival_steps[i])
            for i, s in enumerate(specs)]

        self.residents: List[List[int]] = [[] for _ in range(self.n_agents)]
        self.cloud_residents: List[int] = []
        self.inbound = np.zeros(self.n_agents + 1, dtype=np.int64)  # last = cloud

        self.migrations: List[MigrationRecord] = []
        self.ep = dict(completed=0, lost=0, sla_breaches=0, infeasible=0,
                       stay=0, migrate_edge=0, migrate_cloud=0, reroute=0,
                       energy=0.0, reward=0.0, progress=0.0,
                       lost_severity=0.0, completed_severity=0.0,
                       critical_lost=0, critical_completed=0,
                       lost_on_resident_host=0, lost_in_flight=0)

        self._admit_arrivals(0)
        self._assign_running(0)
        self._refresh_derived()
        return self._observations(), self._global_state(), self.action_masks()

    # =====================================================================
    # time / trace helpers
    # =====================================================================

    def tick(self, step: int = None) -> int:
        s = self.step_idx if step is None else step
        return self.t0 + s * self.cfg.ticks_per_step

    @property
    def elapsed_s(self) -> float:
        return self.step_idx * self.dt

    def _sim_time(self, step: int = None) -> float:
        return float(self.trace.times[self.tick(step)])

    def obs_channel(self, name: str, node: int, step: int = None) -> float:
        return float(self.trace.ch(name)[node, self.tick(step)])

    def risk_at(self, node: int, step: int = None) -> float:
        """Continuous predicted_failure_risk for an edge node. Never binarised."""
        return self.risk.at(node, self.tick(step))

    def uncertainty_at(self, node: int, step: int = None) -> float:
        """Reserved Sprint 6.5 slot; identically 0.0 in Sprint 6."""
        return self.risk.uncertainty_at(node, self.tick(step))

    def observed_up(self, node: int, step: int = None) -> bool:
        return self.trace.is_up(node, self.tick(step))

    def _link_latency_norm(self, node: int, step: int = None) -> float:
        ms = self.obs_channel("linkLatencyMs", node, step)
        return float(np.log1p(max(ms, 0.0)) / np.log1p(self.cfg.link_latency_norm_ms))

    def symptomatic(self, node: int, step: int = None) -> bool:
        """
        Reactive ops trigger: a symptom visible RIGHT NOW, with no predicted
        risk involved. Shared by the reactive baseline and by the
        preemptive/reactive classification of migrations.
        """
        return bool(
            self.obs_channel("degraded", node, step) >= 0.5
            or self.obs_channel("cpu", node, step) > self.cfg.reactive_cpu_threshold
            or self.obs_channel("linkPacketLoss", node, step)
            > self.cfg.reactive_packet_loss_threshold)

    # =====================================================================
    # occupancy
    # =====================================================================

    def _occupancy(self, node: int) -> int:
        if node == CLOUD_NODE_ID:
            return len(self.cloud_residents) + int(self.inbound[self.n_agents])
        return len(self.residents[node]) + int(self.inbound[node])

    def _capacity(self, node: int) -> int:
        return self.cfg.cloud_slots if node == CLOUD_NODE_ID else self.node_capacity

    def _free_fraction(self, node: int) -> float:
        cap = self._capacity(node)
        return max(0.0, (cap - self._occupancy(node)) / cap)

    def _load_fraction(self, node: int) -> float:
        if node == CLOUD_NODE_ID:
            running = sum(1 for i in self.cloud_residents
                          if self.tasks[i].state == RUNNING)
            return running / max(self.cfg.cloud_slots, 1)
        running = sum(1 for i in self.residents[node]
                      if self.tasks[i].state == RUNNING)
        return running / max(self.cfg.vms_per_node, 1)

    def _slots(self, node: int) -> int:
        return self.cfg.cloud_slots if node == CLOUD_NODE_ID else self.cfg.vms_per_node

    def _mips(self, node: int) -> float:
        return self.cfg.cloud_mips if node == CLOUD_NODE_ID else self.cfg.vm_mips

    # =====================================================================
    # candidate views (observable-only)
    # =====================================================================

    def _candidate(self, node: int) -> CandidateView:
        if node == CLOUD_NODE_ID:
            return CandidateView(
                node_id=CLOUD_NODE_ID,
                # The cloud tier is modelled as non-failing in Sprint 6; there
                # is no cloud host in the trace, so there is nothing to
                # predict. Stated as a limitation.
                risk=0.0,
                observed_up=True,
                free_capacity_fraction=self._free_fraction(CLOUD_NODE_ID),
                load_fraction=self._load_fraction(CLOUD_NODE_ID),
                link_latency_norm=float(
                    np.log1p(self.cfg.cloud_wan_latency_ms)
                    / np.log1p(self.cfg.link_latency_norm_ms)),
                is_cloud=True)
        return CandidateView(
            node_id=node,
            risk=self.risk_at(node),
            observed_up=self.observed_up(node),
            free_capacity_fraction=self._free_fraction(node),
            load_fraction=self._load_fraction(node),
            link_latency_norm=self._link_latency_norm(node))

    def _edge_candidates(self, source: int) -> List[CandidateView]:
        return [self._candidate(j) for j in self.topology.neighbours(source)]

    # =====================================================================
    # focus task, masks, observations
    # =====================================================================

    def _refresh_derived(self):
        now = self.elapsed_s
        self._focus: List[int] = []
        for i in range(self.n_agents):
            live = [k for k in self.residents[i] if self.tasks[k].live]
            if not live:
                self._focus.append(-1)
            else:
                self._focus.append(
                    min(live, key=lambda k: priority_order_key(self.tasks[k].spec, now)))

    def focus_task(self, agent: int) -> Optional[TaskRuntime]:
        k = self._focus[agent]
        return None if k < 0 else self.tasks[k]

    def action_masks(self) -> np.ndarray:
        """(n_agents, 4) bool. Masks derive only from observable state."""
        m = np.zeros((self.n_agents, N_ACTIONS), dtype=bool)
        for i in range(self.n_agents):
            m[i] = self.agent_mask(i)
        return m

    def agent_mask(self, i: int) -> np.ndarray:
        """
        Legal actions for ONE agent, evaluated against the occupancy as it
        stands right now.

        Sprint 6.5: split out of action_masks() so step() can re-check the
        acting agent against CURRENT occupancy. The joint mask is computed once
        per step for the policy to condition on, but earlier agents in the same
        step commit inbound transfers that consume destination slots
        (_occupancy counts self.inbound), so a mask taken at step start can
        promise a destination that no longer exists by the time a later agent
        acts. Sprint 6 measured the consequence: 70% of relocation attempts
        were refused, every one of them legal at mask time and feasible had no
        co-agent gone first. The agent paid P_infeasible and stayed put, which
        made MIGRATE_TO_CLOUD a cheap pseudo-action and taught the policy to
        prefer it. Re-checking here removes that; it does not change what the
        mask is *made of*.
        """
        m = np.zeros(N_ACTIONS, dtype=bool)
        m[ACTION_STAY] = True
        k = self._focus[i]
        if k < 0:
            return m
        t = self.tasks[k]
        if t.migrations + t.reroutes >= self.cfg.max_migrations_per_task:
            return m
        edge_ok = self.selector.select(self._edge_candidates(i)) is not None
        cloud_c = self._candidate(CLOUD_NODE_ID)
        cloud_ok = self.selector.select([cloud_c]) is not None
        m[ACTION_MIGRATE_EDGE] = edge_ok
        m[ACTION_MIGRATE_CLOUD] = cloud_ok
        # Reroute is only meaningful before any computation has happened.
        m[ACTION_PREEMPTIVE_REROUTE] = (
            t.remaining_mi >= t.spec.length_mi - 1e-6
            and (edge_ok or cloud_ok))
        return m

    def _observations(self) -> np.ndarray:
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        c = self.cfg
        prog = self.step_idx / max(c.episode_steps, 1)
        now = self.elapsed_s

        for i in range(self.n_agents):
            v = []
            # ---- local node (15) ----
            v += [self.obs_channel("cpu", i) / 100.0,
                  self.obs_channel("ram", i) / 100.0,
                  self.obs_channel("bandwidth", i) / 100.0,
                  self.obs_channel("energy", i) / c.energy_norm_w,
                  self.obs_channel("runningTasks", i) / c.running_tasks_norm,
                  self.obs_channel("active", i),
                  self.obs_channel("degraded", i),
                  self.obs_channel("linkUp", i),
                  self.obs_channel("linkBandwidthMbps", i) / c.link_bw_norm_mbps,
                  self._link_latency_norm(i),
                  self.obs_channel("linkPacketLoss", i) / 100.0,
                  self.obs_channel("underAttack", i),
                  self.risk_at(i),                    # continuous, unthresholded
                  self.uncertainty_at(i),             # reserved (Sprint 6.5)
                  self._free_fraction(i)]

            # ---- focus task (8) ----
            k = self._focus[i]
            if k < 0:
                v += [0.0] * OBS_TASK
            else:
                t = self.tasks[k]
                ttd = t.spec.deadline - now
                v += [1.0,
                      t.spec.severity,
                      t.spec.priority_at(now) / 1.5,
                      t.progress,
                      float(np.clip(ttd / 300.0, -1.0, 1.0)),
                      t.migrations / max(c.max_migrations_per_task, 1),
                      1.0 if t.state == RUNNING else 0.0,
                      len(self.residents[i]) / self.node_capacity]

            # ---- neighbours (5 each) ----
            for j in self.topology.neighbours(i):
                v += [self.risk_at(j),
                      self.obs_channel("active", j),
                      self._free_fraction(j),
                      self.obs_channel("cpu", j) / 100.0,
                      self._link_latency_norm(j)]

            # ---- cloud (3) ----
            v += [self._free_fraction(CLOUD_NODE_ID),
                  float(np.log1p(c.cloud_wan_latency_ms)
                        / np.log1p(c.link_latency_norm_ms)),
                  0.0]

            # ---- context (2) ----
            v += [prog, i / max(self.n_agents, 1)]

            obs[i] = np.asarray(v, dtype=np.float32)

        if not np.isfinite(obs).all():                       # pragma: no cover
            raise FloatingPointError("non-finite observation")
        return obs

    def _global_state(self) -> np.ndarray:
        """
        Centralized-critic input: every agent's observation plus a small
        global block. Training-time only (CTDE). Still contains no future
        information — the joint state is the union of what the agents can
        each see now, which keeps the causality claim simple and total.
        """
        obs = self._observations().reshape(-1)
        up = np.mean([1.0 if self.observed_up(i) else 0.0
                      for i in range(self.n_agents)])
        risks = np.array([self.risk_at(i) for i in range(self.n_agents)])
        loads = np.array([self._load_fraction(i) for i in range(self.n_agents)])
        n = max(self.cfg.n_tasks, 1)
        extra = np.array([
            up, risks.mean(), risks.max(),
            self.ep["completed"] / n, self.ep["lost"] / n,
            sum(1 for t in self.tasks if t.state == PENDING) / n,
            loads.mean(), loads.std(),
            self.step_idx / max(self.cfg.episode_steps, 1),
        ], dtype=np.float32)
        return np.concatenate([obs, extra]).astype(np.float32)

    # =====================================================================
    # step
    # =====================================================================

    def step(self, actions) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool, dict]:
        """
        Apply one joint action.

        Returns (next_obs, next_global_state, rewards, done, info).
        `info["action_masks"]` carries the next step's masks.
        """
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        if actions.shape[0] != self.n_agents:
            raise ValueError(f"expected {self.n_agents} actions, got {actions.shape[0]}")

        ev = [self._blank_event() for _ in range(self.n_agents)]

        # ---- 1. apply actions -------------------------------------------
        # Sprint 6.5: apply in a rotating order, and re-check the acting
        # agent's mask against CURRENT occupancy.
        #
        # Sprint 6 applied agents in the fixed order 0..n-1. Because a
        # committed transfer immediately consumes a destination slot
        # (_occupancy counts self.inbound), agent 0 always won every
        # contention and agent 9 always lost it. With separate per-agent
        # actors that is not a tie-break detail: each actor faces a different
        # effective action space determined by its index, so ten agents that
        # the design calls homogeneous are not. Rotating by step index shares
        # the contention instead of assigning it by id. It changes no reward
        # term and no legality rule.
        order = ([(i + self.step_idx) % self.n_agents
                  for i in range(self.n_agents)]
                 if not self.cfg.legacy_fixed_apply_order
                 else list(range(self.n_agents)))
        for i in order:
            a = int(actions[i])
            if not self.agent_mask(i)[a]:
                self.ep["infeasible"] += 1
                ev[i]["infeasible"] = True
                a = ACTION_STAY
            ev[i]["action"] = a
            if a == ACTION_STAY:
                self.ep["stay"] += 1
                k = self._focus[i]
                if k >= 0:
                    ev[i]["stayed_resident"] = True
            else:
                self._relocate(i, a, ev[i])

        t_from = self.tick()
        t_to = self.tick(self.step_idx + 1)

        # ---- 2. environment physics over [t_from, t_to) -----------------
        self._land_and_kill(t_from, t_to, ev)
        self._admit_arrivals(self.step_idx + 1)
        self._assign_running(self.step_idx + 1)
        self._advance_compute(t_from, t_to, ev)
        self._check_deadlines(ev)

        # ---- 3. reward ---------------------------------------------------
        rewards = self._rewards(ev)

        # ---- 4. advance clock -------------------------------------------
        self.step_idx += 1
        self._refresh_derived()

        done = (self.step_idx >= self.cfg.episode_steps
                or all(t.state in TERMINAL for t in self.tasks))

        self.ep["reward"] += float(rewards.sum())
        info = dict(action_masks=self.action_masks(),
                    events=ev,
                    episode=dict(self.ep))
        return self._observations(), self._global_state(), rewards, done, info

    @staticmethod
    def _blank_event():
        return dict(action=ACTION_STAY, infeasible=False, stayed_resident=False,
                    migration_cost=0.0, completed=0, lost=0, sla=0,
                    energy=0.0, progress=0.0, progress_w=0.0,
                    severity=0.0, migration_severity=0.0,
                    task_id=-1, dest=-1)

    # ---- action application ---------------------------------------------

    def _relocate(self, agent: int, action: int, ev: dict):
        k = self._focus[agent]
        t = self.tasks[k]
        cfg, r = self.cfg, self.rcfg

        if action == ACTION_MIGRATE_EDGE:
            chosen = self.selector.select(self._edge_candidates(agent))
            latency, cost = cfg.migration_latency_steps_edge, r.migration_cost_edge
        elif action == ACTION_MIGRATE_CLOUD:
            chosen = self.selector.select([self._candidate(CLOUD_NODE_ID)])
            latency, cost = cfg.migration_latency_steps_cloud, r.migration_cost_cloud
        else:  # PREEMPTIVE_REROUTE — pick the best of edge neighbours OR cloud
            chosen = self.selector.select(
                self._edge_candidates(agent) + [self._candidate(CLOUD_NODE_ID)])
            latency, cost = cfg.reroute_latency_steps, r.reroute_cost

        if chosen is None:                       # became infeasible: degrade
            self.ep["infeasible"] += 1
            self.ep["stay"] += 1
            ev["infeasible"] = True
            ev["action"] = ACTION_STAY
            ev["stayed_resident"] = True
            return

        dest = chosen.node_id
        # detach from source; the task now depends on the DESTINATION's fate
        self.residents[agent].remove(k)
        t.state = IN_FLIGHT
        t.node = agent
        t.dest = dest
        # The deciding agent owns the outcome for the whole transfer. If it
        # chose a destination that fails mid-flight, it pays the loss.
        t.reward_owner = agent
        t.land_step = self.step_idx + latency
        self.inbound[self.n_agents if dest == CLOUD_NODE_ID else dest] += 1
        if action == ACTION_PREEMPTIVE_REROUTE:
            t.reroutes += 1
            self.ep["reroute"] += 1
        else:
            t.migrations += 1
            self.ep["migrate_cloud" if dest == CLOUD_NODE_ID else "migrate_edge"] += 1
        if dest == CLOUD_NODE_ID:
            t.wan_latency_ms += cfg.cloud_wan_latency_ms

        ev["migration_cost"] = cost
        ev["dest"] = dest
        ev["task_id"] = t.spec.task_id
        # Sprint 6.5: the reward charges P_migration * (1 +
        # w_criticality_migration * <severity>), but the field it read,
        # ev["severity"], was only ever written by _kill and by the completion
        # path -- never here. So crit_m was identically 1.0 at every migration
        # and w_criticality_migration was dead code: patient criticality could
        # not influence the cost of moving a patient's task at all.
        # _diag_counterfactual P8 exposed it, showing the migration term pinned
        # to exactly -0.4000 = P_migration * 1.0 * cost_edge * reward_scale.
        #
        # This uses its OWN field rather than ev["severity"] on purpose.
        # ev["severity"] also feeds `crit`, which weights the completion, loss
        # and SLA terms; writing the migrated task's severity there would
        # silently re-weight those terms in any step where an agent both moved
        # one task and lost or finished another. The defect is confined to the
        # migration charge, so the fix is too.
        ev["migration_severity"] = max(
            ev["migration_severity"], t.spec.severity)

        symptomatic = self.symptomatic(agent)
        self.migrations.append(MigrationRecord(
            step=self.step_idx, elapsed_s=self.elapsed_s,
            task_id=t.spec.task_id, patient_id=t.spec.patient_id,
            severity=t.spec.severity, priority=t.spec.priority_at(self.elapsed_s),
            action=action, action_name=ACTION_NAMES[action],
            source_node=agent, dest_node=dest, cost=cost, latency_steps=latency,
            # "Preemptive" = moved with NO present symptom on the source, i.e.
            # on anticipation rather than on a threshold breach.
            preemptive=not symptomatic, source_symptomatic=symptomatic,
            source_risk=self.risk_at(agent),
            dest_risk=chosen.risk))

    # ---- physics ---------------------------------------------------------

    def _land_and_kill(self, t_from: int, t_to: int, ev: List[dict]):
        for k, t in enumerate(self.tasks):
            if t.state == IN_FLIGHT:
                dest = t.dest
                # in flight: vulnerable to the DESTINATION going down
                down = (False if dest == CLOUD_NODE_ID
                        else self.trace.any_down_during(dest, t_from, t_to))
                if down:
                    self._kill(k, ev, reason="destination failed in flight")
                    continue
                if self.step_idx + 1 >= t.land_step:
                    self.inbound[self.n_agents if dest == CLOUD_NODE_ID else dest] -= 1
                    t.node = dest
                    t.dest = -1
                    t.state = QUEUED
                    if dest == CLOUD_NODE_ID:
                        self.cloud_residents.append(k)
                        # No agent is resident on the cloud, so ownership stays
                        # with the agent that sent it there. It keeps both the
                        # upside (a cloud completion) and the downside (an SLA
                        # breach from WAN latency, and the cloud's energy).
                    else:
                        self.residents[dest].append(k)
                        # The destination agent can now act on this task, so it
                        # takes over accountability for what happens next.
                        t.reward_owner = dest
            elif t.state in (QUEUED, RUNNING) and t.node != CLOUD_NODE_ID:
                if self.trace.any_down_during(t.node, t_from, t_to):
                    self._kill(k, ev, reason="host failed under a resident task")

    def _kill(self, k: int, ev: List[dict], reason: str = ""):
        t = self.tasks[k]
        home = t.node
        if t.state in (QUEUED, RUNNING):
            if home == CLOUD_NODE_ID:
                if k in self.cloud_residents:
                    self.cloud_residents.remove(k)
            elif 0 <= home < self.n_agents and k in self.residents[home]:
                self.residents[home].remove(k)
        elif t.state == IN_FLIGHT:
            d = self.n_agents if t.dest == CLOUD_NODE_ID else t.dest
            self.inbound[d] -= 1
        t.state = LOST
        t.lost_step = self.step_idx
        self.ep["lost"] += 1
        self.ep["lost_in_flight" if "in flight" in reason
                else "lost_on_resident_host"] += 1
        self.ep["lost_severity"] += t.spec.severity
        if t.spec.severity >= 0.5:
            self.ep["critical_lost"] += 1
        owner = t.reward_owner if 0 <= t.reward_owner < self.n_agents else -1
        if owner >= 0:
            ev[owner]["lost"] += 1
            ev[owner]["severity"] = max(ev[owner]["severity"], t.spec.severity)
        if self.verbose:
            print(f"    [t={self.elapsed_s:6.1f}s] task {t.spec.task_id} LOST "
                  f"({reason})")

    def place_task(self, k: int, node: int, state: str = QUEUED):
        """
        Make task `k` resident on edge node `node`, consistently.

        The single entry point for placing a task, so that residency and reward
        ownership can never drift apart. They did once: a hand-built scenario set
        `node` directly and left `reward_owner` at its -1 default, which silently
        charged that task's outcome to no agent at all. Anything that places a
        task must go through here.
        """
        t = self.tasks[k]
        t.state = state
        t.node = node
        t.dest = -1
        t.reward_owner = node
        if k not in self.residents[node]:
            self.residents[node].append(k)

    def unowned_live_tasks(self) -> List[int]:
        """
        Live tasks with no valid reward owner - i.e. whose completion or loss
        would be charged to nobody. Should always be empty; used as an invariant
        check by the tests rather than asserted in the hot loop.
        """
        return [k for k, t in enumerate(self.tasks)
                if t.live and not (0 <= t.reward_owner < self.n_agents)]

    def _admit_arrivals(self, step: int):
        """
        Initial placement is NOT the agents' job in Sprint 6 (that is
        scheduling, Sprint 9). Arrivals are admitted by the same
        least-loaded-observed-up rule for every policy, so the only thing a
        policy changes is the relocation decision.
        """
        arriving = [k for k, t in enumerate(self.tasks)
                    if t.state == PENDING and t.arrival_step <= step]
        if not arriving:
            return
        now = step * self.dt
        arriving.sort(key=lambda k: priority_order_key(self.tasks[k].spec, now))
        for k in arriving:
            cands = [i for i in range(self.n_agents)
                     if self.trace.is_up(i, min(self.tick(step), self.trace.n_ticks - 1))
                     and self._occupancy(i) < self.node_capacity]
            if not cands:
                continue                     # stays PENDING, retried next step
            home = min(cands, key=lambda i: (self._occupancy(i), i))
            self.place_task(k, home)

    def _assign_running(self, step: int):
        """Top `vms_per_node` resident tasks by Sprint-5 priority execute."""
        now = step * self.dt
        for i in range(self.n_agents):
            live = [k for k in self.residents[i] if self.tasks[k].live]
            live.sort(key=lambda k: priority_order_key(self.tasks[k].spec, now))
            for rank, k in enumerate(live):
                t = self.tasks[k]
                t.state = RUNNING if rank < self.cfg.vms_per_node else QUEUED
                if t.state == RUNNING and t.start_step < 0:
                    t.start_step = step
        live = [k for k in self.cloud_residents if self.tasks[k].live]
        live.sort(key=lambda k: priority_order_key(self.tasks[k].spec, now))
        for rank, k in enumerate(live):
            t = self.tasks[k]
            t.state = RUNNING if rank < self.cfg.cloud_slots else QUEUED
            if t.state == RUNNING and t.start_step < 0:
                t.start_step = step

    def _advance_compute(self, t_from: int, t_to: int, ev: List[dict]):
        for k, t in enumerate(self.tasks):
            if t.state != RUNNING:
                continue
            node = t.node
            if node == CLOUD_NODE_ID:
                throughput = 1.0
                power = float(np.nanmean(
                    self.trace.ch("energy")[:, t_from])) * self.cfg.cloud_energy_multiplier
            else:
                degraded = self.trace.ch("degraded")[node, t_from] >= 0.5
                throughput = 1.0 - (self.cfg.degraded_throughput_penalty
                                    if degraded else 0.0)
                power = float(self.trace.ch("energy")[node, t_from])
            before_mi = t.remaining_mi
            t.remaining_mi -= self._mips(node) * self.dt * throughput

            owner = t.reward_owner if 0 <= t.reward_owner < self.n_agents else -1
            e = power / self.cfg.energy_norm_w
            self.ep["energy"] += e
            if owner >= 0:
                ev[owner]["energy"] += e

            # PROGRESS, credited as the work is actually performed rather than
            # only when the task finishes. `frac` is the share of this task's
            # total work completed during this step, measured from the ACTUAL
            # work done (clamped at the finish line, so a task that overshoots on
            # its last step is not paid for work it never had to do). A task
            # carried from 0 to done therefore earns exactly R_progress * crit in
            # total, spread over the ~150 steps it really takes.
            #
            # This exists because the delay, not the magnitude, was the problem.
            # A completion bonus paid ~150 steps after the placement decision
            # that earned it is invisible to the advantage: GAE's horizon is
            # 1/(1 - gamma*lambda) ~ 20 steps and the critic measured only 0.08
            # explained variance, so nothing bridged the gap. Every term the
            # agent COULD see was a cost, and three training runs duly learned to
            # hold fewer tasks (completions 30 -> 20, losses 6 -> 16). Paying for
            # work as it happens puts a positive term inside the horizon.
            #
            # It is accumulated ALREADY WEIGHTED by each task's own criticality,
            # in `progress_w`, rather than going through the event's shared
            # `severity` field. Every running task makes progress every step, so
            # folding them into that shared max would raise the crit multiplier
            # applied to this agent's loss and SLA terms as well, silently
            # re-scaling penalties that have nothing to do with this term.
            done_mi = before_mi - max(t.remaining_mi, 0.0)
            frac = done_mi / max(t.spec.length_mi, 1e-9)
            self.ep["progress"] += frac
            if owner >= 0:
                ev[owner]["progress"] += frac
                ev[owner]["progress_w"] += frac * (
                    1.0 + self.rcfg.w_criticality * t.spec.severity)

            if t.remaining_mi <= 0.0:
                t.remaining_mi = 0.0
                t.state = COMPLETED
                t.finish_step = self.step_idx + 1
                self.ep["completed"] += 1
                self.ep["completed_severity"] += t.spec.severity
                if t.spec.severity >= 0.5:
                    self.ep["critical_completed"] += 1
                if owner >= 0:
                    ev[owner]["completed"] += 1
                    ev[owner]["severity"] = max(ev[owner]["severity"], t.spec.severity)
                if node == CLOUD_NODE_ID and k in self.cloud_residents:
                    self.cloud_residents.remove(k)
                elif owner >= 0 and k in self.residents[owner]:
                    self.residents[owner].remove(k)

    def _check_deadlines(self, ev: List[dict]):
        now = (self.step_idx + 1) * self.dt
        for t in self.tasks:
            if t.deadline_breached or t.state in (PENDING, COMPLETED, LOST):
                continue
            effective = now + t.wan_latency_ms / 1000.0
            if effective > t.spec.deadline:
                t.deadline_breached = True
                self.ep["sla_breaches"] += 1
                owner = t.reward_owner if 0 <= t.reward_owner < self.n_agents else -1
                if owner >= 0:
                    ev[owner]["sla"] += 1
                    ev[owner]["severity"] = max(ev[owner]["severity"], t.spec.severity)

    # ---- reward ----------------------------------------------------------

    def _rewards(self, ev: List[dict]) -> np.ndarray:
        r = self.rcfg
        loads = np.array([self._load_fraction(i) for i in range(self.n_agents)])
        team = (r.R_complete * sum(e["completed"] for e in ev)
                - r.P_task_lost * sum(e["lost"] for e in ev)
                - r.P_balance * float(loads.std()))
        team_each = r.team_reward_share * team / max(self.n_agents, 1)

        out = np.zeros(self.n_agents, dtype=np.float32)
        for i, e in enumerate(ev):
            sev = e["severity"]
            crit = 1.0 + r.w_criticality * sev
            # Sprint 6.5: the migration charge reads the severity of the task
            # that was actually MOVED (set in _relocate), not the severity of
            # whatever task this agent happened to finish or lose in the same
            # step. Before the fix this field did not exist and crit_m was
            # always 1.0. See _relocate.
            crit_m = 1.0 + r.w_criticality_migration * (
                e["severity"] if self.cfg.legacy_dead_migration_criticality
                else e["migration_severity"])

            val = 0.0
            val += r.R_complete * crit * e["completed"]
            # already criticality-weighted per task; see _advance_compute
            val += r.R_progress * e["progress_w"]
            val -= r.P_task_lost * crit * e["lost"]
            val -= r.P_sla * crit * e["sla"]
            val -= r.P_migration * crit_m * e["migration_cost"]
            val -= r.P_energy * e["energy"]
            val -= r.P_infeasible * (1.0 if e["infeasible"] else 0.0)
            val -= r.P_overload * max(0.0, loads[i] - self.cfg.overload_target)
            if e["stayed_resident"]:
                # exposure to the CURRENT predicted risk of staying put
                k = self._focus[i]
                s = self.tasks[k].spec.severity if k >= 0 else 0.0
                val -= (r.P_risk_expose * (1.0 + r.w_criticality * s)
                        * self.risk_at(i))
            out[i] = (val + team_each) * r.reward_scale
        return out

    # =====================================================================
    # post-episode metrics (evaluation only)
    # =====================================================================

    def finalise_migration_outcomes(self):
        """
        Annotate migrations with whether the SOURCE host actually failed soon
        afterwards, and whether the task survived.

        This reads the recorded trace AHEAD of the migration instant. It is
        run once, after the episode, purely to compute "tasks protected before
        failure". It is never called from step(), never observed by a policy,
        and never enters the reward.
        """
        win = self.cfg.protection_window_steps * self.cfg.ticks_per_step
        by_id = {t.spec.task_id: t for t in self.tasks}
        for m in self.migrations:
            t0 = self.tick(m.step)
            m.source_failed_within_window = self.trace.any_down_during(
                m.source_node, t0, min(t0 + win, self.trace.n_ticks))
            st = by_id[m.task_id]
            m.task_survived = st.state != LOST

    def episode_metrics(self) -> dict:
        self.finalise_migration_outcomes()
        n = self.cfg.n_tasks
        done = [t for t in self.tasks if t.state == COMPLETED]
        lost = [t for t in self.tasks if t.state == LOST]
        lat = [(t.finish_step - t.arrival_step) * self.dt + t.wan_latency_ms / 1000.0
               for t in done]
        crit_all = [t for t in self.tasks if t.spec.severity >= 0.5]
        protected = [m for m in self.migrations
                     if m.source_failed_within_window and m.task_survived]
        return dict(
            episode_reward=self.ep["reward"],
            steps=self.step_idx,
            episode_start_tick=self.t0,
            tasks=n,
            completed=len(done),
            lost=len(lost),
            unfinished=n - len(done) - len(lost),
            task_success_rate=len(done) / n,
            sla_violations=self.ep["sla_breaches"],
            sla_violation_rate=self.ep["sla_breaches"] / n,
            migrations=self.ep["migrate_edge"] + self.ep["migrate_cloud"],
            migrations_edge=self.ep["migrate_edge"],
            migrations_cloud=self.ep["migrate_cloud"],
            reroutes=self.ep["reroute"],
            relocations=len(self.migrations),
            preemptive_relocations=sum(1 for m in self.migrations if m.preemptive),
            reactive_relocations=sum(1 for m in self.migrations if not m.preemptive),
            tasks_protected_before_failure=len(protected),
            tasks_lost_on_resident_host=self.ep["lost_on_resident_host"],
            tasks_lost_in_flight=self.ep["lost_in_flight"],
            avg_task_latency_s=float(np.mean(lat)) if lat else float("nan"),
            energy_cost=self.ep["energy"],
            failed_critical_tasks=self.ep["critical_lost"],
            critical_tasks=len(crit_all),
            critical_success_rate=(self.ep["critical_completed"] / len(crit_all)
                                   if crit_all else float("nan")),
            infeasible_actions=self.ep["infeasible"],
            action_counts=dict(stay=self.ep["stay"],
                               migrate_edge=self.ep["migrate_edge"],
                               migrate_cloud=self.ep["migrate_cloud"],
                               reroute=self.ep["reroute"]),
        )

    # =====================================================================
    def describe(self) -> str:
        return "\n".join([
            "DT-MARL environment (trace-driven replay of the CloudSim Digital Twin)",
            f"  agents            : {self.n_agents} (one per edge node)",
            f"  topology          : {self.topology.describe()}",
            f"  obs dim / agent   : {self.obs_dim}",
            f"  centralized state : {self.state_dim}",
            f"  actions           : {N_ACTIONS} "
            f"({', '.join(ACTION_NAMES[a] for a in range(N_ACTIONS))})",
            f"  decision interval : {self.dt:.1f} s "
            f"({self.cfg.ticks_per_step} recorded ticks)",
            f"  episode           : {self.cfg.episode_steps} steps "
            f"= {self.cfg.episode_steps * self.dt:.0f} s of cluster time",
            f"  trace             : {self.trace.n_nodes} nodes x "
            f"{self.trace.n_ticks} ticks",
            f"  episode starts    : ticks [{self._min_start}, {self._max_start}]",
            f"  risk              : {self.risk.summary()}",
            f"                      {self.risk.notes}",
            f"  workload          : {self.cfg.n_tasks} healthcare tasks, "
            f"Sprint-5 criticality",
        ])


__all__ = ["DTMarlEnv", "TaskRuntime", "MigrationRecord",
           "PENDING", "QUEUED", "RUNNING", "IN_FLIGHT", "COMPLETED", "LOST"]
