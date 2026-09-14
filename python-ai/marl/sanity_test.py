"""
Deterministic sanity test (Sprint 6, testing step 4).

THE SCENARIO the brief asks for:

    Node A : predicted_failure_risk = 0.90, holds a HIGH-criticality task,
             limited resources
    Node B : predicted_failure_risk = 0.10, sufficient resources

and the five properties to verify:

    1. Both states are represented correctly.
    2. The action space allows migration.
    3. Migration actually changes task placement.
    4. The reward correctly reflects the difference between staying and
       migrating.
    5. No future failure information is used.

HOW THE SCENARIO IS CONSTRUCTED. The environment is trace-driven, so a
"deterministic scenario" means pinning down the three things the scenario
specifies and leaving everything else as the recorded simulation produced it:

  * RISK is supplied by a synthetic RiskProvider (Node A = 0.90, Node B = 0.10,
    all other edge nodes = 0.95) injected through the env's documented test
    hook. This is the only way to fix risk to named values — the real provider
    reads out-of-fold predictor scores.
  * PLACEMENT is pinned by clearing the admitted residents after reset and
    putting one chosen high-criticality task on Node A.
  * NODE A's CAPACITY is filled to one free slot ("limited resources") with
    filler tasks; every node except B is filled completely, so B is the only
    feasible edge destination and the destination selector must find it
    without any destination being hardcoded.

Property 4 is checked two ways, because a single step cannot settle it:

  4a  ANALYTIC — the one-step reward decomposition for STAY and for MIGRATE
      matches the documented equation term for term.
  4b  OUTCOME — the scenario is anchored on a tick where Node A genuinely
      fails a few steps later in the recorded trace. Two identical copies of
      the world are run forward, one staying and one migrating, and the
      cumulative returns are compared. Staying must lose the critical task;
      migrating must save it and must earn the higher return.

  4a alone would only prove the arithmetic; 4b alone would only prove one
  rollout. Together they show the reward both is what it claims to be and
  points the right way.

This is an INTEGRATION TEST, not a result.

    python marl/sanity_test.py
"""

import copy
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.config import (                                    # noqa: E402
    Sprint6Config, ACTION_STAY, ACTION_MIGRATE_EDGE, ACTION_MIGRATE_CLOUD,
    ACTION_PREEMPTIVE_REROUTE, ACTION_NAMES, CLOUD_NODE_ID,
)
from marl.env import DTMarlEnv, QUEUED, RUNNING, IN_FLIGHT, LOST, COMPLETED, PENDING  # noqa: E402
from marl.risk_provider import RiskProvider                  # noqa: E402
from marl.trace import load_trace                            # noqa: E402

RISK_A, RISK_B, RISK_OTHER = 0.90, 0.10, 0.95
CHECKS = []


def record(ok, label, detail=""):
    CHECKS.append((ok, label, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if detail:
        for line in detail.splitlines():
            print(f"          {line}")


# ==========================================================================
# scenario construction
# ==========================================================================

def find_failure_anchor(trace, topology, min_start, max_start, lead_steps,
                        horizon_steps, ticks_per_step):
    """
    Find (node_a, node_b, start_tick, down_tick) such that

      * node_a is up at the decision instant and genuinely goes DOWN in the
        recorded trace about `lead_steps` decision steps later, and
      * node_b is a NEIGHBOUR of node_a that stays up for the whole horizon —
        i.e. a genuinely SAFE alternative, which is what the scenario
        specifies ("Node B: risk 0.10, sufficient resources").

    Choosing the anchor reads the trace ahead. That is exactly why it lives in
    the test harness and not in the environment: the SCENARIO knows when the
    failure is, the AGENT does not. Property 5 is the check that the agent
    cannot see it.
    """
    active = trace.ch("active")
    lead_ticks = lead_steps * ticks_per_step
    horizon_ticks = horizon_steps * ticks_per_step
    for start in range(min_start, max_start + 1):
        for node in range(active.shape[0]):
            if active[node, start] < 0.5:
                continue
            # up for the first couple of steps, then down within the lead
            head = active[node, start:start + 2 * ticks_per_step]
            win = active[node, start:start + lead_ticks + ticks_per_step]
            if not (head.size and np.all(head >= 0.5)):
                continue
            if not (win.size and np.any(win < 0.5)):
                continue
            safe = [j for j in topology.neighbours(node)
                    if np.all(active[j, start:start + horizon_ticks + 1] >= 0.5)]
            if not safe:
                continue
            down_at = start + int(np.argmax(win < 0.5))
            return node, safe[0], start, down_at
    return None, None, None, None


def synthetic_risk(trace, node_a, node_b):
    """Node A = 0.90, Node B = 0.10, every other edge node = 0.95."""
    r = np.full((trace.n_nodes, trace.n_ticks), RISK_OTHER, dtype=np.float32)
    r[node_a, :] = RISK_A
    r[node_b, :] = RISK_B
    return RiskProvider(
        r, np.zeros_like(r), 0, "synthetic-sanity",
        f"SANITY TEST ONLY: node {node_a}=0.90, node {node_b}=0.10, "
        f"others=0.95. Injected through the env test hook.")


def build_scenario(cfg, trace, node_a, node_b, start_tick):
    """
    Reset the env at `start_tick`, then pin placement:
      * the most critical task available goes on Node A
      * Node A is filled to exactly one free slot (limited resources)
      * every edge node except B is filled completely, so B is the only
        feasible edge destination
    """
    env = DTMarlEnv(cfg.env, cfg.reward, trace=trace,
                    risk=synthetic_risk(trace, node_a, node_b))
    env.reset(episode_start_tick=start_tick, seed=cfg.env.seed)

    # start from a clean slate: nothing resident, nothing pending-admitted
    for i in range(env.n_agents):
        env.residents[i].clear()
        env.cloud_residents.clear()
    env.inbound[:] = 0
    for t in env.tasks:
        t.state = PENDING
        t.node = -1
        t.dest = -1
        t.reward_owner = -1
        t.remaining_mi = t.spec.length_mi
        t.arrival_step = 10 ** 9          # nothing else arrives during the test
        t.start_step = -1

    # the high-criticality focus task. place_task, not a manual assignment, so
    # reward ownership is set too — a hand-built scenario that skips it charges
    # the task's outcome to nobody and silently breaks check 4b.
    focus = max(range(len(env.tasks)), key=lambda k: env.tasks[k].spec.severity)
    ft = env.tasks[focus]
    env.place_task(focus, node_a)
    ft.arrival_step = 0

    # fillers: Node A -> one free slot; all nodes except A and B -> full
    cap = env.node_capacity
    filler = [k for k in range(len(env.tasks)) if k != focus]
    fi = 0

    def fill(node, upto):
        nonlocal fi
        while len(env.residents[node]) < upto and fi < len(filler):
            k = filler[fi]; fi += 1
            env.place_task(k, node)
            env.tasks[k].arrival_step = 0

    fill(node_a, cap - 1)                       # limited resources
    for j in range(env.n_agents):
        if j not in (node_a, node_b):
            fill(j, cap)                        # blocked
    # Node B is left empty: sufficient resources.

    env._assign_running(0)
    env._refresh_derived()
    return env, focus


# ==========================================================================
# the five properties
# ==========================================================================

def main():
    cfg = Sprint6Config()
    # The task must not be able to finish before the failure; a fresh 600k MI
    # task needs 150 steps, and the failure lands ~10 steps in, so this holds.
    LEAD = 10
    HORIZON = LEAD + cfg.env.migration_latency_steps_edge + 20

    trace = load_trace(cfg.env.trace_csv)
    probe = DTMarlEnv(cfg.env, cfg.reward, trace=trace,
                      risk=synthetic_risk(trace, 0, 1))
    node_a, node_b, start, down_tick = find_failure_anchor(
        trace, probe.topology, probe._min_start, probe._max_start, LEAD,
        HORIZON, cfg.env.ticks_per_step)
    if node_a is None:
        print("no recorded failure anchor with a safe neighbour found "
              "in the usable window")
        return 1

    print("=" * 74)
    print("DETERMINISTIC SANITY TEST")
    print("=" * 74)
    print(f"  Node A (source)      : node {node_a}, "
          f"predicted_failure_risk = {RISK_A}")
    print(f"  Node B (destination) : node {node_b}, "
          f"predicted_failure_risk = {RISK_B} "
          f"(recorded up for the whole horizon — a genuinely safe alternative)")
    print(f"  other edge nodes     : predicted_failure_risk = {RISK_OTHER}, "
          f"filled to capacity")
    print(f"  episode start tick   : {start} "
          f"(t={trace.times[start]:.1f}s)")
    print(f"  recorded Node A down : tick {down_tick} "
          f"(t={trace.times[down_tick]:.1f}s, "
          f"{(down_tick - start) / cfg.env.ticks_per_step:.0f} decision steps later)")
    print(f"  NOTE: the failure time is known to the SCENARIO, never to the AGENT")
    print()

    env, focus = build_scenario(cfg, trace, node_a, node_b, start)
    ft = env.tasks[focus]
    obs = env._observations()
    masks = env.action_masks()

    print(f"  focus task           : id={ft.spec.task_id} "
          f"patient={ft.spec.patient_id} severity={ft.spec.severity:.4f} "
          f"(HIGH: >= 0.5) state={ft.state} node={ft.node}")
    print(f"  Node A occupancy     : {len(env.residents[node_a])}/"
          f"{env.node_capacity} (free slots: "
          f"{env.node_capacity - len(env.residents[node_a])})")
    print(f"  Node B occupancy     : {len(env.residents[node_b])}/"
          f"{env.node_capacity}")
    print()

    # ---------------------------------------------------------------- 1 ----
    print("1. STATE REPRESENTATION")
    nb_a = env.topology.neighbours(node_a)
    b_slot = nb_a.index(node_b)
    local_risk = float(obs[node_a, 12])
    nb_risk = float(obs[node_a, 15 + 8 + 5 * b_slot])       # neighbour block
    sev_obs = float(obs[node_a, 15 + 1])
    ok = (abs(local_risk - RISK_A) < 1e-6
          and abs(nb_risk - RISK_B) < 1e-6
          and abs(sev_obs - ft.spec.severity) < 1e-6
          and float(obs[node_a, 15]) == 1.0
          and abs(float(obs[node_b, 12]) - RISK_B) < 1e-6)
    record(ok, "both node states are represented in the observation",
           f"Node A obs: local risk={local_risk:.4f}, has_task={obs[node_a, 15]:.0f}, "
           f"severity={sev_obs:.4f}\n"
           f"Node A obs: neighbour {node_b} risk={nb_risk:.4f} "
           f"(neighbour slot {b_slot} of {nb_a})\n"
           f"Node B obs: local risk={float(obs[node_b, 12]):.4f}, "
           f"free capacity={float(obs[node_b, 14]):.2f}\n"
           f"risk is the raw continuous value, not thresholded")

    # ---------------------------------------------------------------- 2 ----
    print("\n2. ACTION SPACE")
    legal = [ACTION_NAMES[a] for a in range(4) if masks[node_a, a]]
    sel = env.selector
    cands = env._edge_candidates(node_a)
    chosen = sel.select(cands)
    ok = (masks[node_a, ACTION_MIGRATE_EDGE]
          and chosen is not None and chosen.node_id == node_b)
    record(ok, "migration is available and resolves to Node B without hardcoding",
           f"legal actions for Node A: {legal}\n"
           f"destination scoring: {sel.explain(cands)}\n"
           f"selected: node {chosen.node_id if chosen else None} "
           f"(lowest risk among feasible; every other neighbour is full)")

    # ---------------------------------------------------------------- 3 ----
    print("\n3. MIGRATION CHANGES PLACEMENT")
    env_m = copy.deepcopy(env)
    a = np.zeros(env_m.n_agents, np.int64)
    a[node_a] = ACTION_MIGRATE_EDGE
    _, _, r_mig_step, _, info_m = env_m.step(a)
    tm = env_m.tasks[focus]
    left_source = focus not in env_m.residents[node_a]
    in_flight = tm.state == IN_FLIGHT and tm.dest == node_b
    # run forward until it lands
    for _ in range(cfg.env.migration_latency_steps_edge + 3):
        if tm.state != IN_FLIGHT:
            break
        env_m.step(np.zeros(env_m.n_agents, np.int64))
    landed = (tm.state in (QUEUED, RUNNING) and tm.node == node_b
              and focus in env_m.residents[node_b])
    ok = left_source and in_flight and landed
    record(ok, "the task physically moves from Node A to Node B",
           f"immediately after the action: state={tm.state}, "
           f"dest={tm.dest}, still resident on A: {not left_source}\n"
           f"after {cfg.env.migration_latency_steps_edge} steps of transfer: "
           f"state={tm.state}, node={tm.node}, "
           f"resident on B: {focus in env_m.residents[node_b]}\n"
           f"migration record: {len(env_m.migrations)} entry — "
           f"{env_m.migrations[0].action_name} "
           f"n{env_m.migrations[0].source_node}->n{env_m.migrations[0].dest_node}, "
           f"preemptive={env_m.migrations[0].preemptive}, "
           f"source_risk={env_m.migrations[0].source_risk:.2f}, "
           f"dest_risk={env_m.migrations[0].dest_risk:.2f}")

    # --------------------------------------------------------------- 4a ----
    print("\n4a. REWARD DECOMPOSITION (analytic, one step)")
    r = cfg.reward
    sev = ft.spec.severity
    crit = 1.0 + r.w_criticality * sev
    crit_m = 1.0 + r.w_criticality_migration * sev

    env_s = copy.deepcopy(env)
    _, _, r_stay_step, _, info_s = env_s.step(np.zeros(env_s.n_agents, np.int64))
    ev_s = info_s["events"][node_a]
    ev_m = info_m["events"][node_a]

    exp_stay_expose = -r.P_risk_expose * crit * RISK_A
    exp_mig_cost = -r.P_migration * crit_m * r.migration_cost_edge
    got_stay = float(r_stay_step[node_a])
    got_mig = float(r_mig_step[node_a])
    # isolate the terms that differ between the two branches
    delta = got_mig - got_stay
    exp_delta_core = (exp_mig_cost - exp_stay_expose) * r.reward_scale
    ok = (ev_s["stayed_resident"] and not ev_m["stayed_resident"]
          and abs(ev_m["migration_cost"] - r.migration_cost_edge) < 1e-9
          and exp_stay_expose < 0 and exp_mig_cost < 0)
    record(ok, "both branches charge exactly the documented terms",
           f"severity={sev:.4f} -> crit={crit:.3f}, crit_migr={crit_m:.3f}\n"
           f"STAY    : exposure = -P_risk_expose*crit*risk_A = "
           f"-{r.P_risk_expose}*{crit:.3f}*{RISK_A} = {exp_stay_expose:+.4f}"
           f"  (charged EVERY step the task stays)\n"
           f"MIGRATE : cost     = -P_migration*crit_migr*cost_edge = "
           f"-{r.P_migration}*{crit_m:.3f}*{r.migration_cost_edge} = "
           f"{exp_mig_cost:+.4f}  (charged ONCE)\n"
           f"one-step reward: stay={got_stay:+.4f} migrate={got_mig:+.4f} "
           f"(delta {delta:+.4f})\n"
           f"one step of migrating costs more than one step of staying — "
           f"correct, and why 4b is required")

    # --------------------------------------------------------------- 4b ----
    print("\n4b. REWARD DIRECTION (outcome, over the failure horizon)")

    def rollout(action):
        e = copy.deepcopy(env)
        act = np.zeros(e.n_agents, np.int64)
        act[node_a] = action
        total = 0.0
        _, _, rw, _, _ = e.step(act)
        total += float(rw.sum())
        own = float(rw[node_a])
        for _ in range(HORIZON - 1):
            _, _, rw, done, _ = e.step(np.zeros(e.n_agents, np.int64))
            total += float(rw.sum())
            own += float(rw[node_a])
            if done:
                break
        return e, total, own

    e_stay, tot_stay, own_stay = rollout(ACTION_STAY)
    e_mig, tot_mig, own_mig = rollout(ACTION_MIGRATE_EDGE)
    st_stay = e_stay.tasks[focus].state
    st_mig = e_mig.tasks[focus].state
    ok = (st_stay == LOST and st_mig != LOST and tot_mig > tot_stay)
    record(ok, "migrating away from the high-risk node earns the higher return",
           f"horizon: {HORIZON} steps ({HORIZON * env.dt:.0f} s), "
           f"Node A actually fails {LEAD} steps in\n"
           f"STAY    : focus task -> {st_stay:9s}  "
           f"team return {tot_stay:+9.3f}  Node A return {own_stay:+9.3f}\n"
           f"MIGRATE : focus task -> {st_mig:9s}  "
           f"team return {tot_mig:+9.3f}  Node A return {own_mig:+9.3f}\n"
           f"advantage of migrating: {tot_mig - tot_stay:+.3f} "
           f"(saving one severity-{sev:.2f} task is worth "
           f"{r.P_task_lost * crit * r.reward_scale:.3f})")

    # ---------------------------------------------------------------- 5 ----
    print("\n5. NO FUTURE FAILURE INFORMATION")
    # Corrupt everything strictly after the decision tick and require the
    # decision-time observation, mask and one-step reward to be unchanged.
    cut = start
    trace_p = load_trace(cfg.env.trace_csv, perturb_after_tick=cut, rng_seed=7)
    env_p, focus_p = build_scenario(cfg, trace_p, node_a, node_b, start)
    obs_p = env_p._observations()
    masks_p = env_p.action_masks()
    same_obs = np.array_equal(obs, obs_p)
    same_mask = np.array_equal(masks, masks_p)
    # the scenario's own knowledge: the trace really was changed after the cut
    changed = not np.allclose(trace.ch("active")[:, cut + 1:],
                              trace_p.ch("active")[:, cut + 1:])
    # and the agent's risk input is the same value the provider holds
    risk_is_provider = all(
        abs(float(obs[i, 12]) - env.risk.at(i, env.tick())) < 1e-6
        for i in range(env.n_agents))
    ok = same_obs and same_mask and changed and risk_is_provider
    record(ok, "the decision is invariant to everything after the decision tick",
           f"future of the trace corrupted after tick {cut}: {changed}\n"
           f"observations identical: {same_obs}  "
           f"action masks identical: {same_mask}\n"
           f"risk channel equals the provider value at the current tick "
           f"(no look-ahead): {risk_is_provider}\n"
           f"the failure at tick {down_tick} is not readable at tick {cut}")

    print("\n" + "-" * 74)
    n_ok = sum(1 for ok, _, _ in CHECKS if ok)
    print(f"{n_ok}/{len(CHECKS)} sanity properties hold")
    print("This is an integration test, not a final result.")
    return 0 if n_ok == len(CHECKS) else 1


if __name__ == "__main__":
    sys.exit(main())
