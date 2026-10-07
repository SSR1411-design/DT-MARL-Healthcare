"""Independent centralized single-agent PPO for Sprint 9 Baseline 2.

This is intentionally not a MAPPO wrapper.  One controller receives the
environment's global state, produces ten masked categorical decisions (one
per edge node), sums their legal log probabilities into one joint-action log
probability, and learns one scalar team value.  The factored representation
keeps the joint action tractable (10 x 4 choices) without pretending the
``4**10`` Cartesian product is a practical categorical action space.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from marl.config import N_ACTIONS


MASK_FILL = -1e8  # finite: categorical entropy is stable for masked choices
CHECKPOINT_KIND = "sprint9_single_agent_ppo"


@dataclass
class SingleAgentPPOConfig:
    """PPO settings owned by the independently identifiable baseline."""

    actor_hidden: List[int] = None
    critic_hidden: List[int] = None
    lr_actor: float = 7e-4
    lr_critic: float = 1e-3
    gamma: float = 0.999
    gae_lambda: float = 0.995
    clip_eps: float = 0.2
    value_clip_eps: float = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatches: int = 4
    normalise_advantages: bool = True
    anneal_lr: bool = True

    def __post_init__(self):
        if self.actor_hidden is None:
            self.actor_hidden = [256, 256]
        if self.critic_hidden is None:
            self.critic_hidden = [256, 256]
        if self.gamma <= 0.0 or self.gamma > 1.0:
            raise ValueError("gamma must be in (0, 1]")
        if self.gae_lambda <= 0.0 or self.gae_lambda > 1.0:
            raise ValueError("gae_lambda must be in (0, 1]")


def _mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Sequential:
    layers, width = [], in_dim
    for h in hidden:
        layers.extend([nn.Linear(width, h), nn.Tanh()])
        width = h
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


def _orthogonal_init(module: nn.Module, final_gain: float = 0.01) -> None:
    for layer in (item for item in module.modules() if isinstance(item, nn.Linear)):
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
        nn.init.constant_(layer.bias, 0.0)
    last = [item for item in module.modules() if isinstance(item, nn.Linear)][-1]
    nn.init.orthogonal_(last.weight, gain=final_gain)


def _masked_dist(logits: torch.Tensor, masks: torch.Tensor):
    """Return one categorical distribution per node with legal actions only."""
    if logits.shape != masks.shape:
        raise ValueError(f"logit/mask shape mismatch: {logits.shape} vs {masks.shape}")
    if not torch.all(masks.bool().any(dim=-1)):
        raise ValueError("each node must have at least one legal action")
    return torch.distributions.Categorical(logits=logits.masked_fill(~masks.bool(), MASK_FILL))


class FactoredGlobalActor(nn.Module):
    """One centralized network -> ``(node, action)`` logits, not ten actors."""

    def __init__(self, state_dim: int, n_nodes: int, hidden: List[int],
                 n_actions: int = N_ACTIONS):
        super().__init__()
        self.state_dim, self.n_nodes, self.n_actions = state_dim, n_nodes, n_actions
        self.net = _mlp(state_dim, hidden, n_nodes * n_actions)
        _orthogonal_init(self.net)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).view(-1, self.n_nodes, self.n_actions)


class ScalarCritic(nn.Module):
    """One team-value critic V(global_state)."""

    def __init__(self, state_dim: int, hidden: List[int]):
        super().__init__()
        self.net = _mlp(state_dim, hidden, 1)
        _orthogonal_init(self.net, final_gain=1.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class SingleAgentRolloutBuffer:
    """One scalar-reward trajectory buffer for a centralized PPO controller."""

    def __init__(self, capacity: int, state_dim: int, n_nodes: int,
                 n_actions: int = N_ACTIONS):
        if capacity < 1:
            raise ValueError("rollout capacity must be positive")
        self.capacity = capacity
        self.states = np.zeros((capacity, state_dim), np.float32)
        self.actions = np.zeros((capacity, n_nodes), np.int64)
        self.logp = np.zeros(capacity, np.float32)
        self.values = np.zeros(capacity, np.float32)
        self.rewards = np.zeros(capacity, np.float32)
        self.masks = np.zeros((capacity, n_nodes, n_actions), np.float32)
        self.continuation = np.zeros(capacity, np.float32)
        self.bootstrap = np.zeros(capacity, np.float32)
        self.ptr = 0

    def add(self, state, action, logp, value, reward, masks, continuation: float) -> None:
        if self.ptr >= self.capacity:
            raise RuntimeError("SingleAgentRolloutBuffer overflow")
        i = self.ptr
        self.states[i] = state
        self.actions[i] = action
        self.logp[i] = logp
        self.values[i] = value
        self.rewards[i] = reward
        self.masks[i] = masks
        self.continuation[i] = continuation
        self.bootstrap[i] = 0.0
        self.ptr += 1

    def set_bootstrap(self, value: float) -> None:
        if self.ptr < 1:
            raise RuntimeError("cannot bootstrap an empty rollout")
        self.bootstrap[self.ptr - 1] = float(value)

    def clear(self) -> None:
        self.ptr = 0

    def __len__(self) -> int:
        return self.ptr


class SingleAgentPPO:
    """Factored-action PPO with one global policy and one scalar critic."""

    def __init__(self, state_dim: int, n_nodes: int,
                 cfg: Optional[SingleAgentPPOConfig] = None,
                 device: str = "cpu", seed: int = 0):
        if state_dim < 1 or n_nodes < 1:
            raise ValueError("state_dim and n_nodes must be positive")
        self.cfg = cfg or SingleAgentPPOConfig()
        self.device = torch.device(device)
        self.state_dim, self.n_nodes, self.n_actions = state_dim, n_nodes, N_ACTIONS
        torch.manual_seed(int(seed))
        self.actor = FactoredGlobalActor(state_dim, n_nodes, self.cfg.actor_hidden,
                                         self.n_actions).to(self.device)
        self.critic = ScalarCritic(state_dim, self.cfg.critic_hidden).to(self.device)
        self.opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.cfg.lr_actor,
                                          eps=1e-5)
        self.opt_critic = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.lr_critic,
                                           eps=1e-5)

    def _tensors(self, state: np.ndarray, masks: np.ndarray):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        masks_t = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
        if state_t.ndim == 1:
            state_t = state_t.unsqueeze(0)
        if masks_t.ndim == 2:
            masks_t = masks_t.unsqueeze(0)
        if state_t.shape != (state_t.shape[0], self.state_dim):
            raise ValueError("global state has wrong dimensionality")
        if masks_t.shape[1:] != (self.n_nodes, self.n_actions):
            raise ValueError("action masks have wrong shape")
        return state_t, masks_t

    @torch.no_grad()
    def act(self, state: np.ndarray, masks: np.ndarray):
        """Sample a single factored joint action and its summed log probability."""
        state_t, masks_t = self._tensors(state, masks)
        dist = _masked_dist(self.actor(state_t), masks_t)
        action = dist.sample()
        joint_logp = dist.log_prob(action).sum(dim=-1)
        return action[0].cpu().numpy().astype(np.int64), float(joint_logp.item())

    @torch.no_grad()
    def act_greedy(self, state: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """Greedy centralized evaluation adapter for the future trained model."""
        state_t, masks_t = self._tensors(state, masks)
        logits = self.actor(state_t).masked_fill(~masks_t, MASK_FILL)
        return logits.argmax(dim=-1)[0].cpu().numpy().astype(np.int64)

    @torch.no_grad()
    def value(self, state: np.ndarray) -> float:
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        return float(self.critic(tensor)[0].item())

    def joint_logprob_and_entropy(self, states: torch.Tensor, actions: torch.Tensor,
                                  masks: torch.Tensor):
        """Joint log p is sum(log p_i); joint entropy is sum(H_i)."""
        dist = _masked_dist(self.actor(states), masks)
        return dist.log_prob(actions).sum(dim=-1), dist.entropy().sum(dim=-1)

    def compute_gae(self, buffer: SingleAgentRolloutBuffer):
        count = len(buffer)
        if count == 0:
            raise ValueError("cannot compute GAE from an empty buffer")
        adv = np.zeros(count, np.float32)
        last = 0.0
        for i in reversed(range(count)):
            cont = float(buffer.continuation[i])
            next_value = (buffer.values[i + 1]
                          if cont > 0.5 and i + 1 < count else buffer.bootstrap[i])
            delta = buffer.rewards[i] + self.cfg.gamma * next_value - buffer.values[i]
            last = delta + self.cfg.gamma * self.cfg.gae_lambda * cont * last
            adv[i] = last
        return adv, adv + buffer.values[:count]

    def update(self, buffer: SingleAgentRolloutBuffer) -> Dict[str, float]:
        """Perform genuine PPO updates once a later manual training run fills a buffer."""
        count = len(buffer)
        if count == 0:
            raise ValueError("cannot update from an empty buffer")
        advantages_np, returns_np = self.compute_gae(buffer)
        states = torch.as_tensor(buffer.states[:count], device=self.device)
        actions = torch.as_tensor(buffer.actions[:count], device=self.device)
        old_logp = torch.as_tensor(buffer.logp[:count], device=self.device)
        old_values = torch.as_tensor(buffer.values[:count], device=self.device)
        masks = torch.as_tensor(buffer.masks[:count], dtype=torch.bool, device=self.device)
        advantages = torch.as_tensor(advantages_np, device=self.device)
        returns = torch.as_tensor(returns_np, device=self.device)
        if self.cfg.normalise_advantages and count > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        minibatch = max(1, count // max(1, self.cfg.minibatches))
        generator = np.random.default_rng(0)
        stats = {key: 0.0 for key in (
            "actor_loss", "critic_loss", "entropy", "approx_kl", "clip_frac")}
        updates = 0
        for _ in range(self.cfg.ppo_epochs):
            order = generator.permutation(count)
            for first in range(0, count, minibatch):
                indices = order[first:first + minibatch]
                if len(indices) == 0:
                    continue
                idx = torch.as_tensor(indices, dtype=torch.long, device=self.device)
                new_logp, entropy = self.joint_logprob_and_entropy(
                    states[idx], actions[idx], masks[idx])
                ratio = torch.exp(new_logp - old_logp[idx])
                unclipped = ratio * advantages[idx]
                clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_eps,
                                      1.0 + self.cfg.clip_eps) * advantages[idx]
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                entropy_loss = -entropy.mean()
                actor_loss = policy_loss + self.cfg.entropy_coef * entropy_loss
                self.opt_actor.zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
                self.opt_actor.step()

                value = self.critic(states[idx])
                clipped_value = old_values[idx] + torch.clamp(
                    value - old_values[idx], -self.cfg.value_clip_eps, self.cfg.value_clip_eps)
                value_loss = torch.maximum(F.mse_loss(value, returns[idx], reduction="none"),
                                           F.mse_loss(clipped_value, returns[idx], reduction="none"))
                critic_loss = self.cfg.value_coef * value_loss.mean()
                self.opt_critic.zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
                self.opt_critic.step()

                with torch.no_grad():
                    stats["actor_loss"] += float(policy_loss.item())
                    stats["critic_loss"] += float(critic_loss.item())
                    stats["entropy"] += float(entropy.mean().item())
                    stats["approx_kl"] += float((old_logp[idx] - new_logp).mean().item())
                    stats["clip_frac"] += float(
                        ((ratio - 1.0).abs() > self.cfg.clip_eps).float().mean().item())
                updates += 1
        for key in stats:
            stats[key] /= max(updates, 1)
        stats.update(
            adv_mean=float(advantages_np.mean()), adv_std=float(advantages_np.std()),
            value_mean=float(buffer.values[:count].mean()),
            joint_action_dimensions=float(self.n_nodes),
        )
        with torch.no_grad():
            predicted = self.critic(states)
            variance = torch.var(returns)
            stats["explained_var"] = (
                float(1.0 - torch.var(returns - predicted) / variance)
                if float(variance) > 1e-12 else float("nan"))
        return stats

    def set_lr_scale(self, fraction: float) -> None:
        if not self.cfg.anneal_lr:
            return
        fraction = max(float(fraction), 0.0)
        for group in self.opt_actor.param_groups:
            group["lr"] = self.cfg.lr_actor * fraction
        for group in self.opt_critic.param_groups:
            group["lr"] = self.cfg.lr_critic * fraction

    def save(self, path: str | Path, extra: Optional[dict] = None) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "checkpoint_kind": CHECKPOINT_KIND,
            "single_agent_ppo_cfg": asdict(self.cfg), "state_dim": self.state_dim,
            "n_nodes": self.n_nodes, "n_actions": self.n_actions,
            "actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(), "opt_critic": self.opt_critic.state_dict(),
            "extra": extra or {},
        }, path)
        return str(path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "SingleAgentPPO":
        checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
        if checkpoint.get("checkpoint_kind") != CHECKPOINT_KIND:
            raise ValueError(
                "not a Sprint 9 single-agent PPO checkpoint; MAPPO checkpoints "
                "are intentionally incompatible with this baseline")
        agent = cls(int(checkpoint["state_dim"]), int(checkpoint["n_nodes"]),
                    SingleAgentPPOConfig(**checkpoint["single_agent_ppo_cfg"]), device=device)
        if int(checkpoint["n_actions"]) != N_ACTIONS:
            raise ValueError("single-agent checkpoint action definition is incompatible")
        agent.actor.load_state_dict(checkpoint["actor"])
        agent.critic.load_state_dict(checkpoint["critic"])
        agent.opt_actor.load_state_dict(checkpoint["opt_actor"])
        agent.opt_critic.load_state_dict(checkpoint["opt_critic"])
        agent.eval()
        return agent

    def train_mode(self) -> None:
        self.actor.train()
        self.critic.train()

    def eval(self) -> None:
        self.actor.eval()
        self.critic.eval()

    def describe(self) -> str:
        return ("Single-agent PPO (centralized global policy)\\n"
                f"  actor: MLP({self.state_dim} -> {self.cfg.actor_hidden} -> "
                f"{self.n_nodes} x {self.n_actions})\\n"
                f"  critic: MLP({self.state_dim} -> {self.cfg.critic_hidden} -> 1)\\n"
                f"  joint action: factored MultiDiscrete[{self.n_actions}] x {self.n_nodes}; "
                "log p = sum per-node log p")


class SingleAgentPolicy:
    """Evaluation adapter; it deliberately uses the global state, not MAPPO APIs."""

    uses_predicted_risk = True

    def __init__(self, agent: SingleAgentPPO, name: str = "single-agent-ppo",
                 greedy: bool = True):
        self.agent, self.name, self.greedy = agent, name, greedy

    def reset(self) -> None:
        pass

    def act(self, env, obs, masks) -> np.ndarray:
        del obs
        state = env._global_state()
        if self.greedy:
            return self.agent.act_greedy(state, masks)
        return self.agent.act(state, masks)[0]


__all__ = [
    "CHECKPOINT_KIND", "SingleAgentPPOConfig", "FactoredGlobalActor", "ScalarCritic",
    "SingleAgentRolloutBuffer", "SingleAgentPPO", "SingleAgentPolicy",
]
