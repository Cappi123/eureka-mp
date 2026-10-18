"""Reproducible environments and queries, generated exactly as in the course notes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planner2d.environment_2d import Environment

# Defaults from the exercise: a 10 x 6 workspace with 5 triangular obstacles.
SIZE_X, SIZE_Y, N_OBSTACLES = 10, 6, 5


@dataclass(frozen=True)
class Query:
    start: np.ndarray
    goal: np.ndarray


def make_scenario(
    seed: int, n_queries: int = 1, n_obstacles: int = N_OBSTACLES
) -> tuple[Environment, list[Query]]:
    """Build ``Environment`` and queries from NumPy's global RNG seeded with ``seed``.

    ``environment_2d`` draws from the global RNG, so this reproduces the
    course snippet exactly: ``make_scenario(4)`` gives the environment and
    query shown in the exercise figure.
    """
    np.random.seed(seed)  # noqa: NPY002 -- environment_2d uses the global RNG
    env = Environment(SIZE_X, SIZE_Y, n_obstacles)
    queries = []
    for _ in range(n_queries):
        q = env.random_query()
        if q is None:
            raise RuntimeError(f"Could not sample a free query for seed {seed}")
        x_start, y_start, x_goal, y_goal = q
        queries.append(Query(np.array([x_start, y_start]), np.array([x_goal, y_goal])))
    return env, queries
