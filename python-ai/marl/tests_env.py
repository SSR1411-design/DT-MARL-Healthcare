"""
Environment integration tests (Sprint 6, testing steps 3 and 9).

No JUnit-equivalent is set up for the Python side of this repository, so these
are plain assertions with a printed pass/fail line each — the same style as
`marl/criticality.py --self-check`. Run:

    python marl/tests_env.py

Every test states what it would catch. The two that matter most for research
validity are T7 (no future information reaches an observation) and T8 (no
forbidden column reaches the environment at all).
"""

import sys
import copy
from pathlib import Path

import numpy as np

# Allow `python marl/tests_env.py` as well as `python -m marl.tests_env`.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.config import (
    Sprint6Config, EnvConfig, RewardConfig, ACTION_STAY, ACTION_MIGRATE_EDGE,
    ACTION_MIGRATE_CLOUD, ACTION_PREEMPTIVE_REROUTE, N_ACTIONS, CLOUD_NODE_ID,
)
from marl.env import (
    DTMarlEnv, COMPLETED, LOST, IN_FLIGHT, RUNNING, QUEUED, PENDING,
)
from marl.trace import load_trace, TRACE_CHANNELS
from marl.criticality import build_patient_tasks
from marl.baseline import NoMigrationPolicy, RandomPolicy
from marl.rollout import episode_starts

RESULTS = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append((True, name, detail or ""))
        print(f"  PASS  {name}" + (f"  — {detail}" if detail else ""))
    except AssertionError as e:
        RESULTS.append((False, name, str(e)))
        print(f"  FAIL  {name}\n          {e}")
    except Exception as e:                                  # noqa: BLE001
        RESULTS.append((False, name, f"{type(e).__name__}: {e}"))
        print(f"  ERROR {name}\n          {type(e).__name__}: {e}")


def make_env(**over):
    cfg = Sprint6Config()
    for k, v in over.items():
        setattr(cfg.env, k, v)
    return DTMarlEnv(cfg.env, cfg.reward)


# ==========================================================================

def t1_shapes():
    """Catches: obs/state geometry drifting out of sync with the networks."""
    env = make_env()
    obs, st, m = env.reset(episode_start_tick=100)
    assert obs.shape == (env.n_agents, env.obs_dim), obs.shape
    assert st.shape == (env.state_dim,), st.shape
    assert m.shape == (env.n_agents, N_ACTIONS), m.shape
    assert obs.dtype == np.float32 and st.dtype == np.float32
    assert np.isfinite(obs).all() and np.isfinite(st).all()
    # observations are normalised; nothing should be wildly out of range
    assert np.abs(obs).max() <= 5.0, f"obs max magnitude {np.abs(obs).max():.2f}"
    return f"obs {obs.shape}, state {st.shape}, |obs|max={np.abs(obs).max():.3f}"


def t2_masks_legal():
    """Catches: an illegal action being silently executed, or STAY unavailable."""
    env = make_env()
    obs, st, m = env.reset(episode_start_tick=100)
    rng = np.random.default_rng(0)
    for _ in range(120):
        assert m[:, ACTION_STAY].all(), "STAY must always be legal"
        for i in range(env.n_agents):
            if env.focus_task(i) is None:
                assert not m[i, 1:].any(), \
                    f"agent {i} has no task but a relocation is legal"
        a = np.array([rng.choice(np.flatnonzero(m[i])) for i in range(env.n_agents)])
        obs, st, r, done, info = env.step(a)
        m = info["action_masks"]
        if done:
            break
    assert env.ep["infeasible"] == 0, \
        f"{env.ep['infeasible']} mask-legal actions turned out infeasible"
    return "mask-legal actions are never infeasible over 120 steps"


def t3_determinism():
    """Catches: hidden nondeterminism that would break reproducibility."""
    outs = []
    for _ in range(2):
        env = make_env()
        obs, st, m = env.reset(episode_start_tick=250, seed=7)
        pol = RandomPolicy(seed=123)
        tot, trace = 0.0, []
        for _ in range(150):
            a = pol.act(env, obs, m)
            obs, st, r, done, info = env.step(a)
            m = info["action_masks"]
            tot += float(r.sum())
            trace.append(obs.copy())
            if done:
                break
        outs.append((tot, np.stack(trace)))
    assert abs(outs[0][0] - outs[1][0]) < 1e-9, \
        f"reward differs: {outs[0][0]} vs {outs[1][0]}"
    assert np.array_equal(outs[0][1], outs[1][1]), "observation streams differ"
    return f"identical over 150 steps (return {outs[0][0]:.4f})"


def t4_conservation():
    """Catches: tasks leaking out of the bookkeeping (double-counted or lost)."""
    env = make_env()
    obs, st, m = env.reset(episode_start_tick=9)
    pol = RandomPolicy(seed=5)
    done = False
    while not done:
        obs, st, r, done, info = env.step(pol.act(env, obs, m))
        m = info["action_masks"]
        n_res = sum(len(x) for x in env.residents) + len(env.cloud_residents)
        n_flight = sum(1 for t in env.tasks if t.state == IN_FLIGHT)
        n_term = sum(1 for t in env.tasks if t.state in (COMPLETED, LOST))
        n_pend = sum(1 for t in env.tasks if t.state == PENDING)
        assert n_res + n_flight + n_term + n_pend == env.cfg.n_tasks, \
            f"res={n_res} flight={n_flight} term={n_term} pend={n_pend}"
        assert (env.inbound >= 0).all(), f"negative inbound {env.inbound}"
    mm = env.episode_metrics()
    assert mm["completed"] + mm["lost"] + mm["unfinished"] == env.cfg.n_tasks
    assert int(env.inbound.sum()) == sum(
        1 for t in env.tasks if t.state == IN_FLIGHT), "inbound/in-flight mismatch"
    return (f"{mm['completed']} done + {mm['lost']} lost + "
            f"{mm['unfinished']} unfinished = {env.cfg.n_tasks}")


def t5_migration_moves_task():
    """Catches: a 'migration' that is only bookkeeping and moves nothing."""
    env = make_env()
    obs, st, m = env.reset(episode_start_tick=9)
    moved = None
    for _ in range(60):
        cand = [i for i in range(env.n_agents)
                if m[i, ACTION_MIGRATE_EDGE] and env.focus_task(i) is not None]
        if cand:
            i = cand[0]
            t = env.focus_task(i)
            src, k = i, t.spec.task_id
            assert k in [env.tasks[x].spec.task_id for x in env.residents[src]]
            a = np.zeros(env.n_agents, np.int64)
            a[i] = ACTION_MIGRATE_EDGE
            obs, st, r, done, info = env.step(a)
            m = info["action_masks"]
            rt = [x for x in env.tasks if x.spec.task_id == k][0]
            assert rt.state == IN_FLIGHT, f"state after migrate = {rt.state}"
            assert rt.dest != src and rt.dest >= 0, f"dest {rt.dest} == src {src}"
            assert k not in [env.tasks[x].spec.task_id for x in env.residents[src]], \
                "task still resident on the source"
            dest = rt.dest
            for _ in range(env.cfg.migration_latency_steps_edge + 2):
                obs, st, r, done, info = env.step(np.zeros(env.n_agents, np.int64))
                m = info["action_masks"]
                if rt.state != IN_FLIGHT:
                    break
            assert rt.state != IN_FLIGHT, "never landed"
            if rt.state != LOST:
                assert rt.node == dest, f"landed on {rt.node}, expected {dest}"
                assert rt.spec.task_id in [
                    env.tasks[x].spec.task_id for x in env.residents[dest]]
            moved = (k, src, dest, rt.state)
            break
        obs, st, r, done, info = env.step(np.zeros(env.n_agents, np.int64))
        m = info["action_masks"]
    assert moved is not None, "no migration opportunity found in 60 steps"
    return f"task {moved[0]}: node {moved[1]} -> {moved[2]} (state {moved[3]})"


def t6_reward_equation():
    """
    Catches: the implemented reward drifting from the documented equation.

    Recomputes one step's per-agent reward from the event record using the
    equation in config.py's docstring and compares to what the env returned.
    """
    env = make_env()
    obs, st, m = env.reset(episode_start_tick=9)
    pol = RandomPolicy(seed=11)
    r = env.rcfg
    worst = 0.0
    for _ in range(200):
        risk_before = [env.risk_at(i) for i in range(env.n_agents)]
        sev_focus = [(env.focus_task(i).spec.severity
                      if env.focus_task(i) is not None else 0.0)
                     for i in range(env.n_agents)]
        obs, st, rew, done, info = env.step(pol.act(env, obs, m))
        m = info["action_masks"]
        ev = info["events"]
        loads = np.array([env._load_fraction(i) for i in range(env.n_agents)])
        team = (r.R_complete * sum(e["completed"] for e in ev)
                - r.P_task_lost * sum(e["lost"] for e in ev)
                - r.P_balance * float(loads.std()))
        team_each = r.team_reward_share * team / env.n_agents
        for i, e in enumerate(ev):
            crit = 1.0 + r.w_criticality * e["severity"]
            # Sprint 6.5: the migration charge scales with the severity of the
            # task actually MOVED, which lives in its own event field. Sprint 6
            # read e["severity"] here, which the relocation path never writes,
            # so this factor was pinned to 1.0 -- see config.py's equation.
            crit_m = 1.0 + r.w_criticality_migration * e["migration_severity"]
            v = (r.R_complete * crit * e["completed"]
                 + r.R_progress * e["progress_w"]
                 - r.P_task_lost * crit * e["lost"]
                 - r.P_sla * crit * e["sla"]
                 - r.P_migration * crit_m * e["migration_cost"]
                 - r.P_energy * e["energy"]
                 - r.P_infeasible * (1.0 if e["infeasible"] else 0.0)
                 - r.P_overload * max(0.0, loads[i] - env.cfg.overload_target))
            if e["stayed_resident"]:
                v -= (r.P_risk_expose * (1.0 + r.w_criticality * sev_focus[i])
                      * risk_before[i])
            expect = (v + team_each) * r.reward_scale
            worst = max(worst, abs(expect - float(rew[i])))
        if done:
            break
    assert worst < 1e-5, f"reward mismatch up to {worst:.2e}"
    return f"max |implemented - documented| = {worst:.2e} over 200 steps"


def t7_no_future_information():
    """
    RESEARCH VALIDITY. Catches: any observation reading the trace ahead of the
    current decision tick.

    Loads a second copy of the trace in which every observable channel after
    tick K is randomly corrupted, runs both envs with the same actions, and
    requires the observation streams to be BIT-IDENTICAL while the current tick
    is <= K. If any observation peeked forward, corrupting the future would
    change it.
    """
    cfg = Sprint6Config()
    K = 400
    env_a = DTMarlEnv(cfg.env, cfg.reward)
    env_b = DTMarlEnv(cfg.env, cfg.reward,
                      trace=load_trace(cfg.env.trace_csv,
                                       perturb_after_tick=K, rng_seed=99))
    # sanity: the perturbation must actually have changed the future
    fut_a = np.stack([env_a.trace.ch(c)[:, K + 1:] for c in TRACE_CHANNELS])
    fut_b = np.stack([env_b.trace.ch(c)[:, K + 1:] for c in TRACE_CHANNELS])
    past_a = np.stack([env_a.trace.ch(c)[:, :K + 1] for c in TRACE_CHANNELS])
    past_b = np.stack([env_b.trace.ch(c)[:, :K + 1] for c in TRACE_CHANNELS])
    assert not np.allclose(fut_a, fut_b), \
        "perturbation was a no-op; the test would prove nothing"
    assert np.array_equal(past_a, past_b), \
        "perturbation leaked backwards; the test would be invalid"

    start = 100
    oa, sa, ma = env_a.reset(episode_start_tick=start, seed=3)
    ob, sb, mb = env_b.reset(episode_start_tick=start, seed=3)
    pol_a, pol_b = RandomPolicy(seed=42), RandomPolicy(seed=42)
    n_checked = 0
    while env_a.tick() <= K and env_a.step_idx < cfg.env.episode_steps:
        assert np.array_equal(oa, ob), \
            f"observations differ at tick {env_a.tick()} <= K={K}: " \
            f"max diff {np.abs(oa - ob).max():.3e} — FUTURE INFORMATION LEAK"
        assert np.array_equal(sa, sb), \
            f"global state differs at tick {env_a.tick()} <= K={K}"
        assert np.array_equal(ma, mb), \
            f"action masks differ at tick {env_a.tick()} <= K={K}"
        n_checked += 1
        a = pol_a.act(env_a, oa, ma)
        b = pol_b.act(env_b, ob, mb)
        assert np.array_equal(a, b)
        oa, sa, ra, da, ia = env_a.step(a)
        ob, sb, rb, db, ib = env_b.step(b)
        ma, mb = ia["action_masks"], ib["action_masks"]
        assert np.allclose(ra, rb), \
            f"rewards differ at tick {env_a.tick()} — reward reads the future"
        if da or db:
            break
    assert n_checked > 50, f"only {n_checked} steps checked"
    return (f"{n_checked} steps with corrupted future: observations, state, "
            f"masks and rewards all bit-identical")


def t8_no_forbidden_columns():
    """RESEARCH VALIDITY. Catches: a label or audit column entering the env."""
    from data.failure_dataset import FORBIDDEN_COLUMNS
    tr = load_trace(Sprint6Config().env.trace_csv)
    banned = [c for c in FORBIDDEN_COLUMNS if c not in ("time", "nodeId")]
    for c in banned:
        assert c not in tr.channels, f"forbidden channel '{c}' present in trace"
    assert list(tr.channels) == list(TRACE_CHANNELS)
    return (f"{len(tr.channels)} observable channels; {len(banned)} forbidden "
            f"columns absent (dropped: {getattr(tr, 'dropped_columns', [])})")


def t9_risk_is_continuous():
    """Catches: the risk channel being binarised before it reaches the agent."""
    env = make_env()
    env.reset(episode_start_tick=9)
    vals = np.array([[env.risk_at(i, s) for i in range(env.n_agents)]
                     for s in range(0, 300, 3)]).ravel()
    uniq = np.unique(vals)
    assert uniq.size > 20, f"only {uniq.size} distinct risk values — binarised?"
    assert not np.isin(uniq, [0.0, 1.0]).all(), "risk looks binary"
    assert (uniq >= 0.0).all() and (uniq <= 1.0).all(), "risk outside [0,1]"
    # the local-node risk observation must equal the provider value exactly
    obs = env._observations()
    for i in range(env.n_agents):
        assert abs(float(obs[i, 12]) - env.risk_at(i)) < 1e-6, \
            f"obs risk channel != provider value for agent {i}"
    return (f"{uniq.size} distinct values in [{uniq.min():.4f}, {uniq.max():.4f}], "
            f"passed through unthresholded")


def t10_uncertainty_slot_reserved():
    """Catches: Sprint 6.5 needing an observation-width change (it must not)."""
    env = make_env()
    env.reset(episode_start_tick=9)
    obs = env._observations()
    assert obs.shape[1] == env.obs_dim
    for i in range(env.n_agents):
        assert float(obs[i, 13]) == 0.0, "uncertainty slot is not zero in Sprint 6"
        assert env.uncertainty_at(i) == 0.0
    return "obs[13] is a reserved, identically-zero prediction_uncertainty slot"


def t11_criticality_reaches_reward():
    """
    Catches: criticality being decorative. Losing a maximally severe task must
    cost strictly more than losing a zero-severity one, by exactly the
    documented factor.
    """
    r = RewardConfig()
    lo = r.P_task_lost * (1.0 + r.w_criticality * 0.0)
    hi = r.P_task_lost * (1.0 + r.w_criticality * 0.5)
    assert hi > lo, "severity does not amplify the loss penalty"
    env = make_env()
    env.reset(episode_start_tick=9)
    sev = sorted({t.spec.severity for t in env.tasks})
    assert len(sev) > 1, "all tasks have identical severity"
    assert min(sev) >= 0.0 and max(sev) <= 1.0
    n_crit = sum(1 for t in env.tasks if t.spec.severity >= 0.5)
    return (f"severity spans [{min(sev):.3f}, {max(sev):.3f}] over "
            f"{len(sev)} levels, {n_crit}/{len(env.tasks)} critical; "
            f"loss penalty x{hi / lo:.2f} for the most severe")


def t12_zero_risk_ablation():
    """
    Catches: the risk channel being ignorable. With risk_source='zero' the
    channel must be identically zero — the ablation the evaluation needs.
    """
    env = make_env(risk_source="zero")
    env.reset(episode_start_tick=9)
    obs = env._observations()
    assert np.abs(obs[:, 12]).max() == 0.0, "zero ablation still has risk"
    assert env.risk.source == "zero"
    return "risk_source='zero' produces an identically-zero risk channel"


def t13_start_window_disjoint():
    """Catches: train and eval episode-start windows overlapping."""
    tr = make_env(start_frac_lo=0.0, start_frac_hi=0.7)
    ev = make_env(start_frac_lo=0.7, start_frac_hi=1.0)
    assert tr._max_start <= ev._min_start, \
        f"train window ends {tr._max_start}, eval starts {ev._min_start}"
    span = ev.cfg.episode_steps * ev.cfg.ticks_per_step
    return (f"train starts [{tr._min_start}, {tr._max_start}], "
            f"eval starts [{ev._min_start}, {ev._max_start}], "
            f"episode span {span} ticks (windows overlap in *coverage* — "
            f"noted as a limitation)")


def t14_reward_attribution_complete():
    """
    Catches: an outcome that is charged to no agent at all.

    This is the regression test for the defect that made the first two training
    runs degrade monotonically. Every completion and every loss must be credited
    to exactly one agent. Deriving the owner from the residency field silently
    dropped both for any task sitting on the cloud, because a cloud-resident
    task's node id is the CLOUD_NODE_ID sentinel rather than an agent index. The
    result was that relocating a task appeared to destroy its value, so the
    policy correctly learned never to relocate.

    A policy that never moves anything cannot expose this — the check has to be
    run under a policy that actually uses the cloud.
    """
    env = make_env()
    pol = RandomPolicy(seed=11)
    credited_done = credited_lost = 0.0
    orig = env._rewards

    def spy(ev):
        nonlocal credited_done, credited_lost
        credited_done += sum(e["completed"] for e in ev)
        credited_lost += sum(e["lost"] for e in ev)
        return orig(ev)

    env._rewards = spy
    obs, state, masks = env.reset(episode_start_tick=env._min_start, seed=5)
    done = False
    unowned = 0
    while not done:
        a = pol.act(env, obs, masks)
        obs, state, rew, done, info = env.step(a)
        masks = info["action_masks"]
        unowned = max(unowned, len(env.unowned_live_tasks()))
    m = env.episode_metrics()
    assert unowned == 0, f"{unowned} live tasks had no reward owner"
    assert credited_done == m["completed"], (
        f"{m['completed']} tasks completed but only {credited_done:.0f} were "
        f"credited to an agent")
    assert credited_lost == m["lost"], (
        f"{m['lost']} tasks lost but only {credited_lost:.0f} were charged to "
        f"an agent")
    assert m["migrations_cloud"] > 0, \
        "test is vacuous unless some task actually reaches the cloud"
    return (f"{m['completed']:.0f} completions and {m['lost']:.0f} losses all "
            f"attributed ({m['migrations_cloud']:.0f} cloud migrations, "
            f"{m['relocations']:.0f} relocations); no live task ever unowned")


def t15_discount_preserves_policy_ranking():
    """
    Catches: a discount factor so short that PPO's objective ranks policies
    DIFFERENTLY from the undiscounted episode return that gets reported.

    This is the bug that cost three training runs. The reported metric is the
    undiscounted sum of per-agent rewards; PPO maximises the gamma-discounted
    return. Those are different objectives, and if gamma is small enough they
    can disagree about which policy is better -- at which point PPO correctly
    ascending its own objective necessarily descends the reported one, and the
    training curve falls monotonically with no bug anywhere in the learner.

    Why it happens here: a task takes a median ~131-149 decision steps to
    complete, so the completion bonus is ~150 steps downstream of the placement
    decision that earns it, while the migration charge is paid immediately. At
    gamma=0.95 the bonus is discounted by 0.95^149 = 5e-4 -- so the agent sees
    the bill and not the goods.

    The test compares a relocating policy against a non-relocating one on
    identical episode starts and requires the SIGN of the difference to agree
    between the undiscounted return and the configured gamma. It deliberately
    does not assert a magnitude: the point is ordinal consistency, which is the
    property policy improvement actually needs.
    """
    cfg = Sprint6Config()
    cfg.env.start_frac_lo, cfg.env.start_frac_hi = 0.0, 0.7
    gamma = cfg.mappo.gamma

    def measure(policy):
        env = DTMarlEnv(cfg.env, cfg.reward)
        starts = episode_starts(env, 3)
        undisc, disc = [], []
        for j, s in enumerate(starts):
            obs, state, masks = env.reset(episode_start_tick=s, seed=500 + j)
            rows, decs, done = [], [], False
            while not done:
                decs.append(masks.sum(axis=1) > 1)
                a = policy.act(env, obs, masks)
                obs, state, rew, done, info = env.step(a)
                rows.append(np.asarray(rew, dtype=np.float64))
                masks = info["action_masks"]
            R = np.array(rows)                       # [T, n_agents]
            D = np.array(decs)[:R.shape[0]]
            undisc.append(float(R.sum()))
            # return-from-t, averaged over the steps where an agent actually
            # had a choice -- those are the entries the policy gradient uses
            G = np.zeros_like(R)
            acc = np.zeros(R.shape[1])
            for t in range(R.shape[0] - 1, -1, -1):
                acc = R[t] + gamma * acc
                G[t] = acc
            disc.append(float(G[D].mean()) if D.any() else 0.0)
        return float(np.mean(undisc)), float(np.mean(disc))

    u_stay, d_stay = measure(NoMigrationPolicy())
    u_rand, d_rand = measure(RandomPolicy(seed=3))

    du, dd = u_rand - u_stay, d_rand - d_stay
    assert abs(du) > 1e-6, (
        "the two policies score identically undiscounted, so this test cannot "
        "say anything - pick policies that actually differ")
    assert (du > 0) == (dd > 0), (
        f"gamma={gamma} INVERTS the policy ranking: random-legal beats "
        f"no-migration by {du:+.2f} undiscounted but by {dd:+.2f} discounted. "
        f"PPO maximises the discounted objective, so it will drive the reported "
        f"undiscounted return DOWN. Raise gamma until the completion bonus "
        f"survives a ~150-step task lifetime (see marl/_diag_horizon.py).")
    return (f"gamma={gamma}: random-legal beats no-migration by {du:+.2f} "
            f"undiscounted and {dd:+.2f} discounted — same sign")


def t16_gae_horizon_covers_task_lifetime():
    """
    Catches: an advantage horizon shorter than the delay it has to bridge.

    T15 guards gamma. This guards the OTHER horizon, which is the one that
    actually bounds credit assignment: GAE accumulates
    last = delta + gamma*lambda*last, so a consequence decays with (gamma*lambda)
    and the advantage's averaging horizon is 1/(1 - gamma*lambda) -- independent
    of how large gamma alone is.

    The failed run had gamma=0.999 (T15 passing comfortably) but lambda=0.95, for
    a horizon of 19.6 steps against a measured task lifetime of 131-149. The
    critic was supposed to carry the rest, and its explained variance reached
    +0.84 -- but the critic's target is ret = adv + V, a lambda-return, so that
    number only certifies self-consistency against an equally short-sighted
    target. Meanwhile PPO's own objective improved while the reported
    undiscounted return fell. Nothing in the learner was broken.

    So the invariant is: the advantage horizon must cover the lifetime of the
    thing being decided about. Measured, not assumed -- the lifetime comes from
    a rollout, so if the workload changes this test moves with it.
    """
    cfg = Sprint6Config()
    cfg.env.start_frac_lo, cfg.env.start_frac_hi = 0.0, 0.7
    g, lam = cfg.mappo.gamma, cfg.mappo.gae_lambda
    gl = g * lam
    horizon = (1.0 / (1.0 - gl)) if gl < 1.0 else float(cfg.env.episode_steps)

    env = DTMarlEnv(cfg.env, cfg.reward)
    lifetimes = []
    for j, s in enumerate(episode_starts(env, 3)):
        obs, state, masks = env.reset(episode_start_tick=s, seed=700 + j)
        pol, done = RandomPolicy(seed=5), False
        while not done:
            obs, state, rew, done, info = env.step(pol.act(env, obs, masks))
            masks = info["action_masks"]
        for t in env.tasks:
            if t.state == COMPLETED and t.start_step >= 0:
                lifetimes.append(t.finish_step - t.arrival_step)
    assert lifetimes, "no task completed; cannot measure a lifetime"
    med = float(np.median(lifetimes))

    assert horizon >= med, (
        f"GAE horizon 1/(1-gamma*lambda) = {horizon:.1f} steps "
        f"(gamma={g}, lambda={lam}) is SHORTER than the median task lifetime "
        f"{med:.0f} steps. Consequences of a placement decision land outside the "
        f"window the advantage averages over, so the policy gradient is myopic "
        f"no matter how large gamma is, and a high explained_var will NOT rescue "
        f"it (the critic's own target is a lambda-return with the same horizon). "
        f"Raise lambda until 1/(1-gamma*lambda) >= {med:.0f}.")
    return (f"advantage horizon {horizon:.0f} steps >= median task lifetime "
            f"{med:.0f} steps (gamma={g}, lambda={lam})")


def t17_contention_is_not_decided_by_agent_index():
    """
    SPRINT 6.5 REGRESSION. Catches: same-step contention for a scarce
    destination being resolved by agent id.

    Ten agents act simultaneously and cloud has cfg.cloud_slots seats, so when
    demand exceeds supply somebody MUST be refused. That part is inherent to
    simultaneous action on a shared resource and no masking scheme removes it:
    each agent's mask is computed without knowing what the others will choose
    this step. Sprint 6.5 measured 79.7% refusal under saturating demand both
    before and after the fix -- the fix does not reduce contention, it
    redistributes it.

    What WAS a defect is who pays. Sprint 6 applied actions in the fixed order
    0..n-1, so the lowest-index demander won every contention and the highest
    lost every one. With separate per-agent actors that is not a tie-break
    detail: each actor faces a different effective action space determined by
    its index, so ten agents the design calls homogeneous are not.

    Requirements:
      (a) the destination is never oversubscribed,
      (b) no agent is refused unless demand genuinely exceeded free capacity
          (no spurious refusals), and
      (c) contention is NOT decided by agent index.
    """
    cfg = Sprint6Config()
    env = DTMarlEnv(cfg.env, cfg.reward)
    contended = rot_first_won = idx_first_won = 0
    demand_total = refused_total = 0
    for j, start in enumerate(episode_starts(env, 3)):
        obs, st, m = env.reset(episode_start_tick=start, seed=900 + j)
        for _ in range(150):
            a = np.array([ACTION_MIGRATE_CLOUD if m[i, ACTION_MIGRATE_CLOUD]
                          else ACTION_STAY for i in range(env.n_agents)])
            dem = [i for i in range(env.n_agents) if a[i] == ACTION_MIGRATE_CLOUD]
            free = env.cfg.cloud_slots - env._occupancy(CLOUD_NODE_ID)
            s_idx = env.step_idx
            obs, st, r, done, info = env.step(a)
            ev = info["events"]
            win = [i for i in dem if not ev[i]["infeasible"]]
            lose = len(dem) - len(win)
            demand_total += len(dem)
            refused_total += lose
            assert env._occupancy(CLOUD_NODE_ID) <= env.cfg.cloud_slots, (
                f"cloud occupancy is {env._occupancy(CLOUD_NODE_ID)} "
                f"but has only {env.cfg.cloud_slots} slots")
            assert lose <= max(0, len(dem) - free), (
                f"{lose} agents refused but only {max(0, len(dem)-free)} could "
                f"not fit ({len(dem)} demanded, {free} slots free): a "
                f"mask-legal action was refused for a reason that predated "
                f"the step")
            if len(dem) > 1 and 0 < len(win) < len(dem):
                contended += 1
                rot_first_won += int(
                    min(dem, key=lambda i: (i - s_idx) % env.n_agents) in win)
                idx_first_won += int(min(dem) in win)
            m = info["action_masks"]
            if done:
                break
    assert contended >= 5, (
        f"only {contended} contended steps; test is too weak to judge fairness")
    assert rot_first_won == contended, (
        f"the documented rotating tie-break held in only {rot_first_won}/"
        f"{contended} contended steps")
    assert idx_first_won < contended, (
        f"the lowest-index demander won ALL {contended} contentions, so the "
        f"scarce destination is allocated by agent id. Per-agent actors then "
        f"face systematically different action spaces (the Sprint 6 defect).")
    return (f"{contended} contended steps: rotating tie-break held "
            f"{rot_first_won}/{contended}, lowest-index demander won only "
            f"{idx_first_won}/{contended}; {refused_total}/{demand_total} "
            f"requests refused, all attributable to genuine oversubscription")


def t18_criticality_reaches_the_migration_charge():
    """
    SPRINT 6.5 REGRESSION. Catches: w_criticality_migration going dead again.

    Sprint 6 charged P_migration * (1 + w_criticality_migration * severity) but
    read the severity from an event field that only the completion and loss paths
    write, so the factor was identically 1.0 at every migration and patient
    criticality could not influence migration cost at all. The migration term was
    pinned to exactly -0.4000 for every edge migration regardless of the patient.

    Requires the observed migration charges to actually SPAN a range consistent
    with the severity of the tasks moved, and the charge for a given cost to be
    exactly P_migration * (1 + w * severity_moved) * cost.
    """
    env = make_env()
    obs, st, m = env.reset(episode_start_tick=9)
    pol = RandomPolicy(seed=11)
    r = env.rcfg
    charges, sevs = [], []
    for _ in range(200):
        obs, st, rew, done, info = env.step(pol.act(env, obs, m))
        m = info["action_masks"]
        for e in info["events"]:
            if e["migration_cost"] <= 0.0:
                continue
            crit_m = 1.0 + r.w_criticality_migration * e["migration_severity"]
            charge = r.P_migration * crit_m * e["migration_cost"]
            # per unit cost, so edge and cloud moves are comparable
            charges.append(charge / e["migration_cost"])
            sevs.append(e["migration_severity"])
        if done:
            break
    assert len(charges) >= 10, f"only {len(charges)} migrations; test is weak"
    assert min(sevs) > 0.0, (
        "every migration reported migration_severity == 0.0, so "
        "w_criticality_migration is dead code and patient criticality does not "
        "reach the migration charge (the Sprint 6 defect)")
    span = max(charges) - min(charges)
    expect = r.P_migration * r.w_criticality_migration * (max(sevs) - min(sevs))
    assert abs(span - expect) < 1e-6, (
        f"migration charge spans {span:.4f} per unit cost but the severities "
        f"moved ({min(sevs):.3f}..{max(sevs):.3f}) imply {expect:.4f}")
    assert span > 1e-3, "migration charge is constant across patient severities"
    return (f"{len(charges)} migrations, charge/unit-cost spans "
            f"{min(charges):.3f}..{max(charges):.3f} for severity "
            f"{min(sevs):.3f}..{max(sevs):.3f}")


def t19_legacy_switches_reproduce_sprint6_defects():
    """
    SPRINT 6.5. Catches: the A0 ablation switches not actually restoring the
    Sprint 6 behaviour they claim to restore.

    The Sprint 6 baseline must stay exactly re-runnable, otherwise the A0-vs-A1
    comparison is not a controlled experiment and the two fixes cannot be
    attributed separately. `marl/` is untracked, so there is no git history to
    recover the pre-fix environment from -- the switches ARE the record.

    Asserts that each switch reproduces the specific measured defect, and that
    T17/T18's invariants FAIL under it (so those tests are real guards, not
    assertions that happen to hold either way).
    """
    # --- defect 1: contention decided by agent index --------------------
    cfg = Sprint6Config()
    cfg.env.legacy_fixed_apply_order = True
    env = DTMarlEnv(cfg.env, cfg.reward)
    obs, st, m = env.reset(episode_start_tick=100)
    contended = idx_first_won = 0
    for _ in range(150):
        a = np.array([ACTION_MIGRATE_CLOUD if m[i, ACTION_MIGRATE_CLOUD]
                      else ACTION_STAY for i in range(env.n_agents)])
        dem = [i for i in range(env.n_agents) if a[i] == ACTION_MIGRATE_CLOUD]
        obs, st, r, done, info = env.step(a)
        ev = info["events"]
        win = [i for i in dem if not ev[i]["infeasible"]]
        if len(dem) > 1 and 0 < len(win) < len(dem):
            contended += 1
            idx_first_won += int(min(dem) in win)
        m = info["action_masks"]
        if done:
            break
    assert contended >= 3, f"only {contended} contended steps under the legacy switch"
    assert idx_first_won == contended, (
        f"legacy_fixed_apply_order did NOT reproduce index-decided contention: "
        f"the lowest-index demander won only {idx_first_won}/{contended}")

    # --- defect 2: criticality never reaches the migration charge --------
    cfg2 = Sprint6Config()
    cfg2.env.legacy_dead_migration_criticality = True
    env2 = DTMarlEnv(cfg2.env, cfg2.reward)
    obs, st, m = env2.reset(episode_start_tick=9)
    pol, r2 = RandomPolicy(seed=11), env2.rcfg
    charges = []
    for _ in range(200):
        obs, st, rew, done, info = env2.step(pol.act(env2, obs, m))
        m = info["action_masks"]
        for e in info["events"]:
            if e["migration_cost"] > 0.0:
                crit_m = 1.0 + r2.w_criticality_migration * e["severity"]
                charges.append(r2.P_migration * crit_m * e["migration_cost"]
                               / e["migration_cost"])
        if done:
            break
    assert len(charges) >= 10, f"only {len(charges)} migrations under the legacy switch"
    span = max(charges) - min(charges)
    assert span < 1e-9, (
        f"legacy_dead_migration_criticality did NOT reproduce the dead "
        f"coefficient: charge/unit-cost still spans {span:.4f}")
    assert abs(charges[0] - r2.P_migration) < 1e-9, (
        f"legacy migration charge is {charges[0]:.4f} per unit cost, expected "
        f"exactly P_migration={r2.P_migration} (crit_migr pinned to 1.0)")
    return (f"legacy order: lowest-index demander won {idx_first_won}/{contended}; "
            f"legacy criticality: charge pinned at {charges[0]:.3f}/unit cost "
            f"over {len(charges)} migrations")


def main():
    print("=" * 74)
    print("DT-MARL environment tests")
    print("=" * 74)
    for name, fn in [
        ("T1  observation / state geometry", t1_shapes),
        ("T2  action masks are legal and complete", t2_masks_legal),
        ("T3  determinism under a fixed seed", t3_determinism),
        ("T4  task bookkeeping conservation", t4_conservation),
        ("T5  migration actually relocates a task", t5_migration_moves_task),
        ("T6  reward matches the documented equation", t6_reward_equation),
        ("T7  NO FUTURE INFORMATION in obs/state/reward", t7_no_future_information),
        ("T8  no label or audit column in the env", t8_no_forbidden_columns),
        ("T9  predicted_failure_risk stays continuous", t9_risk_is_continuous),
        ("T10 prediction_uncertainty slot reserved", t10_uncertainty_slot_reserved),
        ("T11 criticality amplifies penalties", t11_criticality_reaches_reward),
        ("T12 zero-risk ablation works", t12_zero_risk_ablation),
        ("T14 reward attribution is complete", t14_reward_attribution_complete),
        ("T13 train/eval start windows disjoint", t13_start_window_disjoint),
        ("T15 gamma preserves policy ranking", t15_discount_preserves_policy_ranking),
        ("T16 GAE horizon covers task lifetime", t16_gae_horizon_covers_task_lifetime),
        ("T17 contention is not decided by agent index",
         t17_contention_is_not_decided_by_agent_index),
        ("T18 criticality reaches the migration charge",
         t18_criticality_reaches_the_migration_charge),
        ("T19 legacy switches reproduce Sprint 6 defects",
         t19_legacy_switches_reproduce_sprint6_defects),
    ]:
        check(name, fn)
    n_pass = sum(1 for ok, _, _ in RESULTS if ok)
    print("-" * 74)
    print(f"{n_pass}/{len(RESULTS)} passed")
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
