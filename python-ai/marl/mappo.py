"""
MAPPO: Multi-Agent PPO with centralised training and decentralised execution.

WHAT MAKES THIS ACTUALLY MAPPO (rather than something merely named MAPPO):

  * SEPARATE ACTORS. `MultiAgentActor` holds an `nn.ModuleList` of
    `n_agents` independent MLPs with independent parameters and independent
    optimiser state. Agent i's gradient never touches agent j's weights.
    Verified by `assert_actors_independent()`.

  * CENTRALISED CRITIC. One value function that sees the GLOBAL state
    (every node's telemetry, risk, load, plus cluster aggregates) which no
    actor can see. Following Yu et al., the critic input is the
    agent-specific global state — global_state concatenated with a one-hot
    agent id — so it can express V^i(s) and give per-agent credit while
    still being a single shared network.

  * DECENTRALISED EXECUTION. `act()` and `act_greedy()` read ONLY
    `obs[i]` and `masks[i]`. The global state is never touched at execution
    time; it is a training-time-only input. `act_greedy` is what evaluation
    uses, so the reported policy is genuinely locally executable.

  * PPO CLIPPED SURROGATE with ratio clipping at `clip_eps`.
  * VALUE LOSS with the clipped-value variant at `value_clip_eps`.
  * ENTROPY REGULARISATION at `entropy_coef`.
  * GAE(lambda) advantage estimation, bootstrapped with V(s_T) because the
    episode ends on a time limit (a truncation, not a true terminal state) —
    treating it as terminal would bias every value target near the horizon.
  * MULTIPLE EPOCHS over MINIBATCHES with gradient-norm clipping.
  * ACTION MASKING inside the distribution, so an illegal action can never
    be sampled and never receives gradient.

LOSS AGGREGATION. An agent holding no task has exactly one legal action
(STAY). Its distribution is degenerate: ratio == 1, entropy == 0, so it
contributes no actor gradient, but it would still pollute the advantage
statistics and consume minibatch capacity. The actor and entropy losses are
therefore averaged over *decision* entries only (agents with >= 2 legal
actions). The critic is trained on every timestep, because the reward of an
idle agent (energy, exposure, team share) is still real return to predict.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marl.config import MappoConfig, N_ACTIONS

# Finite, not -inf: Categorical.entropy computes logits * probs, and
# (-inf) * 0 is NaN whereas (-1e8) * 0 is 0.
MASK_FILL = -1e8


def mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Sequential:
    layers, d = [], in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers += [nn.Linear(d, out_dim)]
    return nn.Sequential(*layers)


def orthogonal_init(module: nn.Module, final_gain: float = 0.01):
    """Orthogonal init with a small final layer — the standard PPO recipe."""
    linears = [m for m in module if isinstance(m, nn.Linear)]
    for i, m in enumerate(linears):
        gain = final_gain if i == len(linears) - 1 else np.sqrt(2)
        nn.init.orthogonal_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0.0)


def masked_dist(logits: torch.Tensor, mask: torch.Tensor):
    """Categorical over legal actions only."""
    logits = logits.masked_fill(~mask.bool(), MASK_FILL)
    return torch.distributions.Categorical(logits=logits)


# ==========================================================================
# networks
# ==========================================================================

class MultiAgentActor(nn.Module):
    """One independent policy network per agent. No parameter sharing."""

    def __init__(self, n_agents: int, obs_dim: int, hidden: List[int],
                 n_actions: int = N_ACTIONS, separate: bool = True):
        super().__init__()
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.separate = separate
        k = n_agents if separate else 1
        self.actors = nn.ModuleList(
            [mlp(obs_dim, hidden, n_actions) for _ in range(k)])
        for a in self.actors:
            orthogonal_init(a)

    def net(self, agent: int) -> nn.Module:
        return self.actors[agent if self.separate else 0]

    def logits(self, agent: int, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, obs_dim) rows belonging to `agent` only."""
        return self.net(agent)(obs)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """obs: (B, n_agents, obs_dim) -> logits (B, n_agents, n_actions)."""
        return torch.stack(
            [self.logits(i, obs[:, i, :]) for i in range(self.n_agents)], dim=1)


class CentralisedCritic(nn.Module):
    """
    V(s, i): agent-specific centralised value function.

    Input is the global state (unavailable to any actor) plus a one-hot agent
    id. Training-time only.
    """

    def __init__(self, state_dim: int, n_agents: int, hidden: List[int]):
        super().__init__()
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.net = mlp(state_dim + n_agents, hidden, 1)
        orthogonal_init(self.net, final_gain=1.0)
        self.register_buffer("eye", torch.eye(n_agents))

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, state_dim) -> values (B, n_agents)."""
        b = state.shape[0]
        s = state.unsqueeze(1).expand(b, self.n_agents, self.state_dim)
        ids = self.eye.unsqueeze(0).expand(b, self.n_agents, self.n_agents)
        return self.net(torch.cat([s, ids], dim=-1)).squeeze(-1)


# ==========================================================================
# rollout storage
# ==========================================================================

class RolloutBuffer:
    """
    Fixed-capacity storage for one PPO update's worth of transitions.

    The global state is stored once per timestep, not once per agent — it is
    identical for all agents by construction, and storing it per agent would
    multiply memory by n_agents for no information gain.
    """

    def __init__(self, capacity: int, n_agents: int, obs_dim: int,
                 state_dim: int, n_actions: int = N_ACTIONS):
        self.cap = capacity
        self.n = n_agents
        self.obs = np.zeros((capacity, n_agents, obs_dim), np.float32)
        self.state = np.zeros((capacity, state_dim), np.float32)
        self.act = np.zeros((capacity, n_agents), np.int64)
        self.logp = np.zeros((capacity, n_agents), np.float32)
        self.val = np.zeros((capacity, n_agents), np.float32)
        self.rew = np.zeros((capacity, n_agents), np.float32)
        self.mask = np.zeros((capacity, n_agents, n_actions), np.float32)
        self.decision = np.zeros((capacity, n_agents), np.float32)
        # 1.0 when the NEXT state continues the same episode.
        self.cont = np.zeros((capacity, 1), np.float32)
        # bootstrap value of the state after each episode's last step
        self.boot = np.zeros((capacity, n_agents), np.float32)
        # 1.0 when that episode end was a genuine TIME-LIMIT TRUNCATION rather
        # than a true terminal state. Read ONLY by compute_mc_returns;
        # compute_gae ignores it and bootstraps unconditionally, exactly as
        # before, so the A0 control is unaffected by its existence.
        self.trunc = np.zeros((capacity, 1), np.float32)
        self.ptr = 0

    def add(self, obs, state, act, logp, val, rew, mask, cont):
        i = self.ptr
        if i >= self.cap:
            raise RuntimeError("RolloutBuffer overflow")
        self.obs[i] = obs
        self.state[i] = state
        self.act[i] = act
        self.logp[i] = logp
        self.val[i] = val
        self.rew[i] = rew
        self.mask[i] = mask
        self.decision[i] = (mask.sum(axis=-1) > 1.5).astype(np.float32)
        self.cont[i] = cont
        # Cleared here so a stale flag from a previous update can never be read
        # after clear(); set_bootstrap writes it at every episode end.
        self.trunc[i] = 0.0
        self.ptr += 1

    def set_bootstrap(self, idx: int, values, truncated: bool = True):
        self.boot[idx] = values
        self.trunc[idx] = 1.0 if truncated else 0.0

    def clear(self):
        self.ptr = 0

    def __len__(self):
        return self.ptr


# ==========================================================================
# learner
# ==========================================================================

class MAPPO:

    def __init__(self, n_agents: int, obs_dim: int, state_dim: int,
                 cfg: Optional[MappoConfig] = None, device: str = "cpu",
                 seed: int = 0):
        self.cfg = cfg or MappoConfig()
        self.device = torch.device(device)
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        torch.manual_seed(seed)

        self.actor = MultiAgentActor(
            n_agents, obs_dim, self.cfg.actor_hidden, N_ACTIONS,
            separate=self.cfg.separate_actors).to(self.device)
        self.critic = CentralisedCritic(
            state_dim, n_agents, self.cfg.critic_hidden).to(self.device)

        # One optimiser over all actor parameters. Because the actors share no
        # parameters, and Adam's moments are per-parameter, this is exactly
        # equivalent to n_agents independent optimisers — just cheaper.
        self.opt_actor = torch.optim.Adam(self.actor.parameters(),
                                          lr=self.cfg.lr_actor, eps=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(),
                                           lr=self.cfg.lr_critic, eps=1e-5)

    # ---------------- execution (decentralised) -------------------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, masks: np.ndarray):
        """
        Sample one action per agent. DECENTRALISED: agent i sees obs[i] only.

        Returns (actions, logprobs, entropies-unused) as numpy arrays.
        """
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        m = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        acts = np.zeros(self.n_agents, np.int64)
        logps = np.zeros(self.n_agents, np.float32)
        for i in range(self.n_agents):
            d = masked_dist(self.actor.logits(i, o[i:i + 1]), m[i:i + 1])
            a = d.sample()
            acts[i] = int(a.item())
            logps[i] = float(d.log_prob(a).item())
        return acts, logps

    @torch.no_grad()
    def act_greedy(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Argmax over legal actions — exploration disabled, for evaluation."""
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        m = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        acts = np.zeros(self.n_agents, np.int64)
        for i in range(self.n_agents):
            lg = self.actor.logits(i, o[i:i + 1]).masked_fill(
                ~m[i:i + 1].bool(), MASK_FILL)
            acts[i] = int(lg.argmax(dim=-1).item())
        return acts

    @torch.no_grad()
    def action_probs(self, obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Per-agent action distribution — used by the behaviour probes."""
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        m = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
        out = np.zeros((self.n_agents, N_ACTIONS), np.float32)
        for i in range(self.n_agents):
            d = masked_dist(self.actor.logits(i, o[i:i + 1]), m[i:i + 1])
            out[i] = d.probs[0].cpu().numpy()
        return out

    # ---------------- centralised value ---------------------------------

    @torch.no_grad()
    def value(self, state: np.ndarray) -> np.ndarray:
        s = torch.as_tensor(state, dtype=torch.float32,
                            device=self.device).unsqueeze(0)
        return self.critic(s)[0].cpu().numpy()

    # ---------------- advantage estimation ------------------------------

    def compute_gae(self, buf: RolloutBuffer):
        """
        GAE(lambda) per agent, walking backwards through the buffer.

        `cont[t] == 0` marks the last step of an episode; there the next value
        is the stored bootstrap V(s_T) rather than val[t+1]. The episode ends
        on a time limit, so bootstrapping (not zeroing) is the correct
        treatment — every task still in flight has real expected return.
        """
        T = len(buf)
        g, lam = self.cfg.gamma, self.cfg.gae_lambda
        adv = np.zeros((T, self.n_agents), np.float32)
        last = np.zeros(self.n_agents, np.float32)
        for t in reversed(range(T)):
            cont = buf.cont[t, 0]
            next_v = buf.val[t + 1] if (cont > 0.5 and t + 1 < T) else buf.boot[t]
            delta = buf.rew[t] + g * next_v - buf.val[t]
            last = delta + g * lam * cont * last
            adv[t] = last
        ret = adv + buf.val[:T]
        return adv, ret

    def compute_mc_returns(self, buf: RolloutBuffer):
        """
        Within-episode Monte-Carlo discounted return — the CRITIC's regression
        target when cfg.critic_target == "mc". Sprint 7 Rung 2.

        Deliberately NOT a function of the critic's own predictions, except for
        the one case where a critic-free target does not exist: a genuine
        time-limit truncation, where the remaining return is real but
        unobserved. `env.step` ends an episode either at the step limit OR
        because every task reached a terminal state (env.py:594); only the
        former is a truncation, and only there is `boot` used. Bootstrapping the
        terminal case as well would reintroduce exactly the critic dependence
        this target exists to remove.

        Same discount as everything else (cfg.gamma). No lambda appears here —
        that is the whole point, and the actor's GAE(lambda) is left untouched.
        """
        T = len(buf)
        g = self.cfg.gamma
        ret = np.zeros((T, self.n_agents), np.float32)
        for t in reversed(range(T)):
            if buf.cont[t, 0] > 0.5 and t + 1 < T:
                tail = ret[t + 1]                 # same episode continues
            elif buf.trunc[t, 0] > 0.5:
                tail = buf.boot[t]                # time-limit truncation
            else:
                tail = 0.0                        # true terminal state
            ret[t] = buf.rew[t] + g * tail
        return ret

    # ---------------- PPO update ----------------------------------------

    def update(self, buf: RolloutBuffer) -> dict:
        T = len(buf)
        adv_np, ret_np = self.compute_gae(buf)
        # SPRINT 7 RUNG 2: swap ONLY the critic's regression target. `adv_np` —
        # the actor's GAE(gamma=0.999, lambda=0.995) — is untouched either way.
        if self.cfg.critic_target == "mc":
            ret_np = self.compute_mc_returns(buf)

        obs = torch.as_tensor(buf.obs[:T], device=self.device)
        state = torch.as_tensor(buf.state[:T], device=self.device)
        act = torch.as_tensor(buf.act[:T], device=self.device)
        old_lp = torch.as_tensor(buf.logp[:T], device=self.device)
        old_v = torch.as_tensor(buf.val[:T], device=self.device)
        mask = torch.as_tensor(buf.mask[:T], device=self.device)
        dec = torch.as_tensor(buf.decision[:T], device=self.device)
        adv = torch.as_tensor(adv_np, device=self.device)
        ret = torch.as_tensor(ret_np, device=self.device)

        if self.cfg.normalise_advantages:
            # Normalise over decision entries only: idle agents' advantages
            # carry no actor gradient and would distort the statistics.
            sel = dec > 0.5
            if sel.any():
                a = adv[sel]
                adv = (adv - a.mean()) / (a.std() + 1e-8)

        n_mb = max(1, self.cfg.minibatches)
        mb_size = max(1, T // n_mb)
        rng = np.random.default_rng(0)
        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0,
                 "approx_kl": 0.0, "clip_frac": 0.0, "n": 0}

        for _ in range(self.cfg.ppo_epochs):
            order = rng.permutation(T)
            for start in range(0, T, mb_size):
                idx = torch.as_tensor(order[start:start + mb_size],
                                      dtype=torch.long, device=self.device)
                if idx.numel() == 0:
                    continue

                # ---- actor: minibatch over TIME, all agents kept, each
                #      evaluated by its own network.
                new_lp, ent = [], []
                for i in range(self.n_agents):
                    d = masked_dist(self.actor.logits(i, obs[idx, i, :]),
                                    mask[idx, i, :])
                    new_lp.append(d.log_prob(act[idx, i]))
                    ent.append(d.entropy())
                new_lp = torch.stack(new_lp, dim=1)
                ent = torch.stack(ent, dim=1)

                d_mb = dec[idx]
                denom = d_mb.sum().clamp(min=1.0)

                ratio = torch.exp(new_lp - old_lp[idx])
                s1 = ratio * adv[idx]
                s2 = torch.clamp(ratio, 1.0 - self.cfg.clip_eps,
                                 1.0 + self.cfg.clip_eps) * adv[idx]
                pg = -(torch.min(s1, s2) * d_mb).sum() / denom
                ent_loss = -(ent * d_mb).sum() / denom
                actor_loss = pg + self.cfg.entropy_coef * ent_loss

                self.opt_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(),
                                         self.cfg.max_grad_norm)
                self.opt_actor.step()

                # ---- centralised critic: every timestep, every agent.
                v = self.critic(state[idx])
                v_clipped = old_v[idx] + torch.clamp(
                    v - old_v[idx], -self.cfg.value_clip_eps,
                    self.cfg.value_clip_eps)
                vl = torch.max(F.mse_loss(v, ret[idx], reduction="none"),
                               F.mse_loss(v_clipped, ret[idx], reduction="none"))
                critic_loss = self.cfg.value_coef * vl.mean()

                self.opt_critic.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),
                                         self.cfg.max_grad_norm)
                self.opt_critic.step()

                with torch.no_grad():
                    kl = ((old_lp[idx] - new_lp) * d_mb).sum() / denom
                    cf = (((ratio - 1.0).abs() > self.cfg.clip_eps).float()
                          * d_mb).sum() / denom
                stats["actor_loss"] += float(pg.item())
                stats["critic_loss"] += float(critic_loss.item())
                stats["entropy"] += float(-ent_loss.item())
                stats["approx_kl"] += float(kl.item())
                stats["clip_frac"] += float(cf.item())
                stats["n"] += 1

        n = max(1, stats.pop("n"))
        out = {k: v / n for k, v in stats.items()}
        out["adv_mean"] = float(adv_np.mean())
        out["adv_std"] = float(adv_np.std())
        out["value_mean"] = float(buf.val[:T].mean())
        out["decision_frac"] = float(buf.decision[:T].mean())
        # EXPLAINED VARIANCE of the critic, 1 - Var(ret - v) / Var(ret).
        #
        # This is the load-bearing diagnostic for this environment, not a nicety.
        # A task's completion bonus lands ~150 decision steps after the placement
        # decision that earns it, while GAE's own averaging horizon is only
        # 1/(1 - gamma*lambda) ~ 20 steps. Everything beyond that window reaches
        # the advantage THROUGH THE CRITIC. So if this number is near 0 the
        # advantages are effectively myopic again no matter how large gamma is,
        # and the fix is a longer lambda or a denser reward - not more episodes.
        # Read it as: <=0 the critic is worthless, ~0.3-0.6 usable, >0.8 good.
        with torch.no_grad():
            r_flat = ret.reshape(-1)
            v_now = self.critic(state).reshape(-1)
            var_r = torch.var(r_flat)
            out["explained_var"] = float(
                1.0 - torch.var(r_flat - v_now) / var_r) if float(var_r) > 1e-12 \
                else float("nan")
        return out

    # ---------------- schedules ------------------------------------------

    def set_lr_scale(self, frac: float):
        """Linear LR annealing. `frac` goes 1.0 -> 0.0 over training."""
        if not self.cfg.anneal_lr:
            return
        f = float(max(frac, 0.0))
        for g in self.opt_actor.param_groups:
            g["lr"] = self.cfg.lr_actor * f
        for g in self.opt_critic.param_groups:
            g["lr"] = self.cfg.lr_critic * f

    # ---------------- persistence ---------------------------------------

    def save(self, path, extra: dict = None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_critic": self.opt_critic.state_dict(),
            "mappo_cfg": asdict(self.cfg),
            "n_agents": self.n_agents,
            "obs_dim": self.obs_dim,
            "state_dim": self.state_dim,
            "extra": extra or {},
        }, path)
        return str(path)

    @classmethod
    def load(cls, path, device: str = "cpu"):
        ck = torch.load(path, map_location=device, weights_only=False)
        cfg = MappoConfig(**ck["mappo_cfg"])
        agent = cls(ck["n_agents"], ck["obs_dim"], ck["state_dim"],
                    cfg=cfg, device=device)
        agent.actor.load_state_dict(ck["actor"])
        agent.critic.load_state_dict(ck["critic"])
        agent.opt_actor.load_state_dict(ck["opt_actor"])
        agent.opt_critic.load_state_dict(ck["opt_critic"])
        agent.eval()
        return agent, ck.get("extra", {})

    def train_mode(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    # ---------------- structural self-checks -----------------------------

    def assert_actors_independent(self):
        """
        Prove the actors are not one shared network. Perturbing agent 0's
        first weight must change agent 0's logits and NOTHING else.
        """
        if not self.cfg.separate_actors:
            return "separate_actors=False (parameter sharing is ON)"
        ids = {id(p) for p in self.actor.actors[0].parameters()}
        for j in range(1, self.n_agents):
            other = {id(p) for p in self.actor.actors[j].parameters()}
            if ids & other:
                raise AssertionError(f"actors 0 and {j} share parameters")
        # Non-zero probe: with an all-zero observation the first layer returns
        # only its bias, so perturbing a weight matrix would look like a no-op.
        obs = torch.ones(1, self.obs_dim, device=self.device)
        before = [self.actor.logits(i, obs).detach().clone()
                  for i in range(self.n_agents)]
        with torch.no_grad():
            first = next(self.actor.actors[0].parameters())
            first.add_(1.0)
        after = [self.actor.logits(i, obs).detach().clone()
                 for i in range(self.n_agents)]
        with torch.no_grad():
            first.sub_(1.0)
        if torch.allclose(before[0], after[0]):
            raise AssertionError("perturbing actor 0 did not change its output")
        for i in range(1, self.n_agents):
            if not torch.allclose(before[i], after[i]):
                raise AssertionError(
                    f"perturbing actor 0 changed actor {i} — parameters shared")
        n_actor = sum(p.numel() for p in self.actor.parameters())
        n_critic = sum(p.numel() for p in self.critic.parameters())
        return (f"{self.n_agents} independent actors "
                f"({n_actor:,} params total, {n_actor // self.n_agents:,} each), "
                f"1 centralised critic ({n_critic:,} params)")

    def describe(self) -> str:
        return "\n".join([
            "MAPPO (centralised training, decentralised execution)",
            f"  actors            : {self.n_agents} x MLP"
            f"({self.obs_dim} -> {self.cfg.actor_hidden} -> {N_ACTIONS}),"
            f" separate={self.cfg.separate_actors}",
            f"  centralised critic: MLP({self.state_dim}+{self.n_agents} -> "
            f"{self.cfg.critic_hidden} -> 1)",
            f"  clip={self.cfg.clip_eps} value_clip={self.cfg.value_clip_eps} "
            f"ent={self.cfg.entropy_coef} vf={self.cfg.value_coef}",
            f"  gamma={self.cfg.gamma} lambda={self.cfg.gae_lambda} "
            f"epochs={self.cfg.ppo_epochs} minibatches={self.cfg.minibatches}",
            f"  critic_target={self.cfg.critic_target}",
            f"  lr_actor={self.cfg.lr_actor} lr_critic={self.cfg.lr_critic} "
            f"grad_clip={self.cfg.max_grad_norm}",
            f"  device={self.device}",
        ])


class MappoPolicy:
    """
    Adapter so a trained MAPPO agent can be evaluated by the same loop as the
    baselines. Uses `act_greedy` — exploration disabled — and reads only local
    observations and masks.
    """

    uses_predicted_risk = True

    def __init__(self, agent: MAPPO, name: str = "mappo", greedy: bool = True):
        self.agent = agent
        self.name = name
        self.greedy = greedy

    def reset(self):
        pass

    def act(self, env, obs, masks):
        if self.greedy:
            return self.agent.act_greedy(obs, masks)
        return self.agent.act(obs, masks)[0]


__all__ = ["MAPPO", "MappoPolicy", "MultiAgentActor", "CentralisedCritic",
           "RolloutBuffer", "masked_dist"]
