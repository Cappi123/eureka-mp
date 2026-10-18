"""Probabilistic Roadmap (PRM), a multi-query planner.

The roadmap is built once by :meth:`PRM.build`; each call to :meth:`PRM.query`
then only connects the start and goal to the roadmap and runs a graph search,
so the build cost is amortised over many queries.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from planner2d.collision import CollisionChecker


@dataclass(frozen=True)
class RoadmapStats:
    n_nodes: int
    n_edges: int
    build_time: float


class PRM:
    """Probabilistic Roadmap with radius-based neighbour connection.

    Args:
        checker: Free-space oracle.
        n_samples: Number of collision-free vertices in the roadmap.
        radius: Connection attempts are made between vertices closer than this.
        rng: Random generator; pass a seeded one for reproducible roadmaps.
    """

    def __init__(
        self,
        checker: CollisionChecker,
        n_samples: int = 1000,
        radius: float = 0.8,
        rng: np.random.Generator | None = None,
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        if radius <= 0:
            raise ValueError(f"radius must be positive, got {radius}")
        self.checker = checker
        self.n_samples = n_samples
        self.radius = radius
        self.rng = rng if rng is not None else np.random.default_rng()

        self.nodes = np.empty((0, 2))
        self.edges = np.empty((0, 2), dtype=int)
        self._adjacency: list[list[tuple[int, float]]] = []
        self._kdtree: cKDTree | None = None
        self.stats: RoadmapStats | None = None

    @property
    def is_built(self) -> bool:
        return self._kdtree is not None

    def build(self) -> RoadmapStats:
        """Sample the roadmap vertices and connect neighbours by straight edges."""
        t0 = time.perf_counter()
        self.nodes = self.checker.sample_free(self.rng, self.n_samples)
        self._kdtree = cKDTree(self.nodes)

        candidates = self._kdtree.query_pairs(self.radius, output_type="ndarray")
        free = self.checker.segments_free(
            self.nodes[candidates[:, 0]], self.nodes[candidates[:, 1]]
        )
        self.edges = candidates[free].reshape(-1, 2)

        self._adjacency = [[] for _ in range(len(self.nodes))]
        for i, j in self.edges:
            d = float(np.linalg.norm(self.nodes[i] - self.nodes[j]))
            self._adjacency[i].append((j, d))
            self._adjacency[j].append((i, d))

        self.stats = RoadmapStats(len(self.nodes), len(self.edges), time.perf_counter() - t0)
        return self.stats

    def query(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
        """Return a collision-free path from ``start`` to ``goal``, or None.

        The path is the shortest one in the roadmap augmented with the start
        and goal, found by A* with the Euclidean distance as heuristic.
        """
        if not self.is_built:
            raise RuntimeError("Call build() before query()")
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        if not (self.checker.is_free(start) and self.checker.is_free(goal)):
            return None
        if self.checker.is_segment_free(start, goal):
            return np.vstack([start, goal])

        # Temporary vertices for this query; the roadmap itself is not modified.
        s_id, g_id = len(self.nodes), len(self.nodes) + 1
        positions = np.vstack([self.nodes, start, goal])
        extra: dict[int, list[tuple[int, float]]] = {}
        for q_id, q in ((s_id, start), (g_id, goal)):
            near = np.asarray(self._kdtree.query_ball_point(q, self.radius), dtype=int)
            free = self.checker.segments_free(np.broadcast_to(q, (len(near), 2)), self.nodes[near])
            for j in near[free]:
                d = float(np.linalg.norm(q - self.nodes[j]))
                extra.setdefault(q_id, []).append((int(j), d))
                extra.setdefault(int(j), []).append((q_id, d))

        route = self._astar(s_id, g_id, positions, extra)
        return None if route is None else positions[route]

    def _neighbours(
        self, i: int, extra: dict[int, list[tuple[int, float]]]
    ) -> list[tuple[int, float]]:
        base = self._adjacency[i] if i < len(self._adjacency) else []
        return base + extra.get(i, [])

    def _astar(
        self,
        source: int,
        target: int,
        positions: np.ndarray,
        extra: dict[int, list[tuple[int, float]]],
    ) -> list[int] | None:
        def heuristic(i: int) -> float:
            return float(np.linalg.norm(positions[i] - positions[target]))

        cost = {source: 0.0}
        parent: dict[int, int] = {}
        frontier = [(heuristic(source), source)]
        closed: set[int] = set()
        while frontier:
            _, i = heapq.heappop(frontier)
            if i == target:
                route = [i]
                while route[-1] != source:
                    route.append(parent[route[-1]])
                return route[::-1]
            if i in closed:
                continue
            closed.add(i)
            for j, d in self._neighbours(i, extra):
                new_cost = cost[i] + d
                if new_cost < cost.get(j, np.inf):
                    cost[j] = new_cost
                    parent[j] = i
                    heapq.heappush(frontier, (new_cost + heuristic(j), j))
        return None
