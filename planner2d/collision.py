"""Collision checking for the 2D triangular-obstacle environment.

The planners only ever talk to the environment through :class:`CollisionChecker`,
which plays the role of the free-space oracle described in the course notes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid importing pylab at runtime just for a type hint
    from planner2d.environment_2d import Environment


class CollisionChecker:
    """Vectorised point and segment validity checks.

    A point is *free* if it lies inside the workspace rectangle and not
    strictly inside any triangular obstacle. The obstacle test reproduces
    ``Environment.check_collision`` exactly, but evaluates many points in a
    single NumPy call instead of looping in Python.

    Segments are checked exactly rather than by sampling points along them:
    each triangle is the intersection of three open half-planes, so the part
    of a segment inside it is an open interval of the segment parameter that
    can be computed in closed form. This cannot miss thin obstacle corners,
    and it lets a whole batch of segments be checked in one NumPy call.

    The ``n_point_checks`` and ``n_segment_checks`` counters are provided so
    planners can be compared by work done, not just by wall-clock time.
    """

    def __init__(self, env: Environment) -> None:
        self.lower = np.zeros(2)
        self.upper = np.array([env.size_x, env.size_y], dtype=float)
        if env.obs:
            # Obstacle m is {p : A[m] @ p - C[m] > 0 componentwise}.
            self._A = np.stack([ob.A for ob in env.obs])  # (m, 3, 2)
            self._C = np.stack([ob.C for ob in env.obs])  # (m, 3)
        else:
            self._A = np.zeros((0, 3, 2))
            self._C = np.zeros((0, 3))
        self.reset_counters()

    def reset_counters(self) -> None:
        self.n_point_checks = 0
        self.n_segment_checks = 0

    def points_free(self, points: np.ndarray) -> np.ndarray:
        """Return a boolean mask: True where each (x, y) row is collision-free."""
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        self.n_point_checks += len(pts)
        in_bounds = np.all((pts >= self.lower) & (pts <= self.upper), axis=1)
        residuals = np.einsum("mkd,nd->nmk", self._A, pts) - self._C  # (n, m, 3)
        in_obstacle = np.any(np.all(residuals > 0, axis=2), axis=1)
        return in_bounds & ~in_obstacle

    def is_free(self, point: np.ndarray) -> bool:
        return bool(self.points_free(point)[0])

    def segments_free(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Return a boolean mask: True where segment [a[i], b[i]] is collision-free.

        The workspace is convex, so a segment is in bounds iff its endpoints are.
        """
        a = np.atleast_2d(np.asarray(a, dtype=float))
        b = np.atleast_2d(np.asarray(b, dtype=float))
        self.n_segment_checks += len(a)
        in_bounds = np.all(
            (a >= self.lower) & (a <= self.upper) & (b >= self.lower) & (b <= self.upper), axis=1
        )
        # Along p(t) = a + t (b - a), constraint k of obstacle m reads
        # f0 + t * f1 > 0, with f0, f1 of shape (n, m, 3).
        f0 = np.einsum("mkd,nd->nmk", self._A, a) - self._C
        f1 = np.einsum("mkd,nd->nmk", self._A, b - a)
        with np.errstate(divide="ignore", invalid="ignore"):
            root = -f0 / f1
        # Each constraint keeps t > root (f1 > 0), t < root (f1 < 0),
        # or all t / no t (f1 == 0, depending on the sign of f0).
        t_lo = np.where(f1 > 0, root, np.where((f1 == 0) & (f0 <= 0), np.inf, -np.inf))
        t_hi = np.where(f1 < 0, root, np.inf)
        lo = np.maximum(t_lo.max(axis=2), 0.0)
        hi = np.minimum(t_hi.min(axis=2), 1.0)
        # The constraints are strict, so an empty-or-single-point overlap is free.
        hits_obstacle = np.any(lo < hi, axis=1)
        return in_bounds & ~hits_obstacle

    def is_segment_free(self, a: np.ndarray, b: np.ndarray) -> bool:
        """Check the straight segment [a, b], endpoints included."""
        return bool(self.segments_free(a, b)[0])

    def sample_uniform(self, rng: np.random.Generator, n: int | None = None) -> np.ndarray:
        """Sample uniformly in the workspace rectangle (free or not)."""
        shape = (2,) if n is None else (n, 2)
        return rng.uniform(self.lower, self.upper, size=shape)

    def sample_free(self, rng: np.random.Generator, n: int, max_batches: int = 1000) -> np.ndarray:
        """Rejection-sample ``n`` collision-free points.

        Raises:
            RuntimeError: if the free space is too small to fill the request
                within ``max_batches`` batches of ``n`` draws.
        """
        accepted: list[np.ndarray] = []
        count = 0
        for _ in range(max_batches):
            batch = self.sample_uniform(rng, n)
            free = batch[self.points_free(batch)]
            accepted.append(free)
            count += len(free)
            if count >= n:
                return np.concatenate(accepted)[:n]
        raise RuntimeError(f"Could only sample {count}/{n} free configurations")
