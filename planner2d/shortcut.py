"""Path post-processing by random shortcutting (course Algorithm PATH_SHORTCUTTING)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planner2d.collision import CollisionChecker
from planner2d.geometry import cumulative_length, point_at, remove_duplicate_waypoints


@dataclass
class ShortcutResult:
    path: np.ndarray
    length_history: np.ndarray  # path length after each repetition, index 0 = input
    n_accepted: int


def shortcut_path(
    path: np.ndarray,
    checker: CollisionChecker,
    max_repetitions: int = 200,
    rng: np.random.Generator | None = None,
) -> ShortcutResult:
    """Shorten a collision-free path by replacing random sub-paths with straight segments.

    Each repetition picks two arc-length positions ``t1 < t2`` uniformly along
    the current path. If the segment ``[q(t1), q(t2)]`` is collision-free, the
    portion of the path between them is replaced by it. By the triangle
    inequality the length never increases, and since ``q(t1)`` and ``q(t2)``
    lie on a collision-free path the result stays collision-free.

    The input path is not modified.
    """
    if max_repetitions < 0:
        raise ValueError(f"max_repetitions must be >= 0, got {max_repetitions}")
    rng = rng if rng is not None else np.random.default_rng()
    path = remove_duplicate_waypoints(np.asarray(path, dtype=float))
    if len(path) < 2:
        raise ValueError("A path needs at least two distinct waypoints")

    cumlen = cumulative_length(path)
    history = [cumlen[-1]]
    n_accepted = 0
    for _ in range(max_repetitions):
        t1, t2 = np.sort(rng.uniform(0.0, cumlen[-1], size=2))
        i1, q1 = point_at(path, cumlen, t1)
        i2, q2 = point_at(path, cumlen, t2)
        # Both points on the same segment: that portion is already straight.
        if i1 != i2 and checker.is_segment_free(q1, q2):
            path = remove_duplicate_waypoints(np.vstack([path[: i1 + 1], q1, q2, path[i2 + 1 :]]))
            cumlen = cumulative_length(path)
            n_accepted += 1
        history.append(cumlen[-1])
    return ShortcutResult(path, np.asarray(history), n_accepted)
