"""Rapidly-exploring Random Trees: single-tree RRT and bidirectional RRT-Connect.

Both are single-query planners: a fresh tree is grown for each query.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from planner2d.collision import CollisionChecker


class Tree:
    """A tree of 2D configurations stored in growable arrays."""

    def __init__(self, root: np.ndarray, capacity: int = 1024) -> None:
        self._nodes = np.empty((capacity, 2))
        self._parents = np.empty(capacity, dtype=int)
        self._nodes[0] = root
        self._parents[0] = -1
        self._size = 1

    def __len__(self) -> int:
        return self._size

    @property
    def nodes(self) -> np.ndarray:
        return self._nodes[: self._size]

    @property
    def parents(self) -> np.ndarray:
        return self._parents[: self._size]

    @property
    def root(self) -> np.ndarray:
        return self._nodes[0]

    def add(self, q: np.ndarray, parent: int) -> int:
        if self._size == len(self._nodes):
            self._nodes = np.resize(self._nodes, (2 * self._size, 2))
            self._parents = np.resize(self._parents, 2 * self._size)
        self._nodes[self._size] = q
        self._parents[self._size] = parent
        self._size += 1
        return self._size - 1

    def nearest(self, q: np.ndarray) -> int:
        """Index of the vertex closest to ``q`` (brute force, O(n))."""
        return int(np.argmin(np.sum((self.nodes - q) ** 2, axis=1)))

    def path_from_root(self, idx: int) -> np.ndarray:
        route = [idx]
        while self._parents[route[-1]] != -1:
            route.append(int(self._parents[route[-1]]))
        return self._nodes[route[::-1]].copy()


@dataclass
class RRTResult:
    path: np.ndarray | None
    iterations: int
    trees: list[Tree] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.path is not None


class RRT:
    """Single-tree RRT, grown from the start with a small bias toward the goal.

    Args:
        checker: Free-space oracle.
        step_size: Maximum edge length; this is ``r`` in the STEER function.
        goal_bias: Probability of using the goal as the random sample.
        max_iterations: EXTEND calls before giving up.
        rng: Random generator; pass a seeded one for reproducible runs.
    """

    def __init__(
        self,
        checker: CollisionChecker,
        step_size: float = 0.3,
        goal_bias: float = 0.05,
        max_iterations: int = 10_000,
        rng: np.random.Generator | None = None,
    ) -> None:
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if not 0.0 <= goal_bias <= 1.0:
            raise ValueError(f"goal_bias must be in [0, 1], got {goal_bias}")
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}")
        self.checker = checker
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iterations = max_iterations
        self.rng = rng if rng is not None else np.random.default_rng()

    def plan(self, start: np.ndarray, goal: np.ndarray) -> RRTResult:
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        tree = Tree(start)
        if not (self.checker.is_free(start) and self.checker.is_free(goal)):
            return RRTResult(None, 0, [tree])

        for iteration in range(1, self.max_iterations + 1):
            use_goal = self.rng.random() < self.goal_bias
            q_rand = goal if use_goal else self.checker.sample_uniform(self.rng)
            i_new = self._extend(tree, q_rand)
            if i_new is None:
                continue
            q_new = tree.nodes[i_new]
            if np.array_equal(q_new, goal):
                return RRTResult(tree.path_from_root(i_new), iteration, [tree])
            if np.linalg.norm(goal - q_new) <= self.step_size and self.checker.is_segment_free(
                q_new, goal
            ):
                i_goal = tree.add(goal, i_new)
                return RRTResult(tree.path_from_root(i_goal), iteration, [tree])
        return RRTResult(None, self.max_iterations, [tree])

    def _extend(self, tree: Tree, q_target: np.ndarray) -> int | None:
        """EXTEND: grow ``tree`` one step toward ``q_target``; return new vertex index."""
        i_near = tree.nearest(q_target)
        q_new = self._steer(tree.nodes[i_near], q_target)
        return None if q_new is None else tree.add(q_new, i_near)

    def _steer(self, q_near: np.ndarray, q_target: np.ndarray) -> np.ndarray | None:
        """STEER: move from ``q_near`` toward ``q_target`` by at most ``step_size``.

        Returns the reached configuration, or None if the motion collides
        or makes no progress.
        """
        delta = q_target - q_near
        dist = float(np.linalg.norm(delta))
        if dist == 0.0:
            return None
        q_new = q_target if dist <= self.step_size else q_near + delta * (self.step_size / dist)
        return q_new.copy() if self.checker.is_segment_free(q_near, q_new) else None


class BiRRT(RRT):
    """Bidirectional RRT-Connect (Kuffner & LaValle, 2000).

    One tree is rooted at the start and one at the goal. Each iteration
    extends one tree toward a random sample, then greedily extends the other
    tree toward the new vertex until it connects or is blocked. The trees
    swap roles every iteration. ``goal_bias`` is unused: the opposite tree
    already provides the pull toward the goal.
    """

    def plan(self, start: np.ndarray, goal: np.ndarray) -> RRTResult:
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        start_tree, goal_tree = Tree(start), Tree(goal)
        if not (self.checker.is_free(start) and self.checker.is_free(goal)):
            return RRTResult(None, 0, [start_tree, goal_tree])
        if self.checker.is_segment_free(start, goal):
            return RRTResult(np.vstack([start, goal]), 0, [start_tree, goal_tree])

        tree_a, tree_b = start_tree, goal_tree
        for iteration in range(1, self.max_iterations + 1):
            i_new = self._extend(tree_a, self.checker.sample_uniform(self.rng))
            if i_new is not None:
                q_new = tree_a.nodes[i_new]
                j_reached = self._connect(tree_b, q_new)
                if j_reached is not None:
                    half_a = tree_a.path_from_root(i_new)
                    half_b = tree_b.path_from_root(j_reached)
                    path = np.vstack([half_a, half_b[-2::-1]])  # q_new appears once
                    if tree_a is goal_tree:
                        path = path[::-1]
                    return RRTResult(path, iteration, [start_tree, goal_tree])
            tree_a, tree_b = tree_b, tree_a
        return RRTResult(None, self.max_iterations, [start_tree, goal_tree])

    def _connect(self, tree: Tree, q_target: np.ndarray) -> int | None:
        """CONNECT: extend repeatedly toward ``q_target``; index if reached, else None."""
        i_near = tree.nearest(q_target)
        if np.array_equal(tree.nodes[i_near], q_target):
            return i_near
        while True:
            q_new = self._steer(tree.nodes[i_near], q_target)
            if q_new is None:
                return None
            i_near = tree.add(q_new, i_near)
            if np.array_equal(q_new, q_target):
                return i_near
