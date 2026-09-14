"""
Edge-node neighbourhood topology.

The repository has NO pairwise adjacency model: DigitalTwinManager
`mirrorNetworkLinks(int)` creates one NetworkLink per node (an access link),
not a node-to-node graph, and nothing anywhere records which edge nodes are
peers. Rather than invent an adjacency matrix and present it as recovered
from the simulator, the neighbourhood is an EXPLICIT CONFIGURATION CHOICE:
a ring over node ids with the offsets in `EnvConfig.neighbour_offsets`
(default [-2, -1, +1, +2], so degree 4).

This is stated as an assumption in the Sprint 6 limitations. Replacing it
with a real topology later means supplying a different `Topology` — the
observation width is fixed by `len(neighbour_offsets)`, so a same-degree
topology needs no retraining-time change at all.
"""

from typing import List

from marl.config import CLOUD_NODE_ID


class Topology:
    """Fixed ring neighbourhood plus a single cloud tier."""

    def __init__(self, n_nodes: int, offsets: List[int]):
        if n_nodes < 2:
            raise ValueError("need at least 2 edge nodes")
        bad = [o for o in offsets if o == 0]
        if bad:
            raise ValueError("neighbour offset 0 would make a node its own neighbour")
        self.n_nodes = n_nodes
        self.offsets = list(offsets)
        self.degree = len(self.offsets)
        self._nb = [
            [(i + o) % n_nodes for o in self.offsets] for i in range(n_nodes)
        ]
        # Sanity: a node must not appear twice in its own neighbour list, or
        # the observation would double-count it.
        for i, nb in enumerate(self._nb):
            if len(set(nb)) != len(nb) or i in nb:
                raise ValueError(
                    f"degenerate neighbourhood for node {i}: {nb} "
                    f"(offsets {self.offsets} vs n_nodes {n_nodes})")

    def neighbours(self, node: int) -> List[int]:
        return self._nb[node]

    def is_cloud(self, node_id: int) -> bool:
        return node_id == CLOUD_NODE_ID

    def describe(self) -> str:
        return (f"ring topology, n={self.n_nodes}, offsets={self.offsets}, "
                f"degree={self.degree}, plus 1 cloud tier "
                f"(id {CLOUD_NODE_ID})")


__all__ = ["Topology"]
