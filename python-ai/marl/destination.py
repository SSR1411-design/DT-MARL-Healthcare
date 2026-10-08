"""
Destination selection for migration / reroute actions.

WHY THIS EXISTS. The policy chooses an action *class* (stay / move to a
neighbouring edge node / move to cloud / reroute), not a concrete target. A
flat action space over 10 nodes x 4 classes would be both larger and less
transferable, and hardcoding "always migrate to node 0" or "always to the
next node in the ring" is exactly what the Sprint 6 brief forbids.

So placement is a two-level factorisation:

    learned  : which KIND of relocation is worth its cost, given local
               telemetry, predicted_failure_risk, task criticality and
               neighbour state
    resolved : which concrete host best satisfies that choice right now,
               scored transparently from live observable state

The selector is deterministic and reproducible, uses ONLY quantities
observable at the current decision tick, and its weights live in EnvConfig.
Nothing about it is a fixed destination: the argmax moves as risk, capacity
and link conditions move.

    score(j) = - w_risk         * predicted_failure_risk(j, t)
               + w_free_cap     * free_capacity_fraction(j)
               - w_link_latency * normalised_link_latency(j, t)
               - w_load         * load_fraction(j)

Candidates that are observed DOWN at t, or that have no free capacity, are
not scored at all. If no candidate survives, the action is infeasible and the
environment falls back to STAY with the configured infeasibility penalty —
which is reported, not hidden.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class CandidateView:
    """Everything the selector may look at. All observable at tick t."""

    node_id: int
    risk: float                  # predicted_failure_risk in [0, 1]
    observed_up: bool            # `active` channel at t (a present reading)
    free_capacity_fraction: float
    load_fraction: float
    link_latency_norm: float
    is_cloud: bool = False


class DestinationSelector:

    def __init__(self, cfg):
        self.w_risk = cfg.dest_w_risk
        self.w_cap = cfg.dest_w_free_capacity
        self.w_lat = cfg.dest_w_link_latency
        self.w_load = cfg.dest_w_load

    def score(self, c: CandidateView) -> float:
        return (-self.w_risk * c.risk
                + self.w_cap * c.free_capacity_fraction
                - self.w_lat * c.link_latency_norm
                - self.w_load * c.load_fraction)

    def feasible(self, candidates: List[CandidateView]) -> List[CandidateView]:
        return [c for c in candidates
                if c.observed_up and c.free_capacity_fraction > 0.0]

    def select(self, candidates: List[CandidateView]) -> Optional[CandidateView]:
        """Best feasible candidate, or None if the action is infeasible."""
        feas = self.feasible(candidates)
        if not feas:
            return None
        # Tie-break on node id so the choice is reproducible.
        return max(feas, key=lambda c: (self.score(c), -c.node_id))

    def explain(self, candidates: List[CandidateView]) -> str:
        parts = []
        for c in candidates:
            tag = "cloud" if c.is_cloud else f"n{c.node_id}"
            ok = "" if (c.observed_up and c.free_capacity_fraction > 0) else " [infeasible]"
            parts.append(f"{tag}: risk={c.risk:.3f} cap={c.free_capacity_fraction:.2f} "
                         f"load={c.load_fraction:.2f} lat={c.link_latency_norm:.3f} "
                         f"score={self.score(c):+.4f}{ok}")
        return " | ".join(parts)


__all__ = ["CandidateView", "DestinationSelector"]
