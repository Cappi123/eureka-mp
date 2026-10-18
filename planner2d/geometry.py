"""Helpers for piecewise-linear paths, represented as (k, 2) arrays of waypoints."""

from __future__ import annotations

import numpy as np


def segment_lengths(path: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.diff(path, axis=0), axis=1)


def path_length(path: np.ndarray) -> float:
    return float(segment_lengths(path).sum())


def cumulative_length(path: np.ndarray) -> np.ndarray:
    """Arc length at each waypoint; first entry is 0, last is the path length."""
    return np.concatenate(([0.0], np.cumsum(segment_lengths(path))))


def point_at(path: np.ndarray, cumlen: np.ndarray, s: float) -> tuple[int, np.ndarray]:
    """Locate arc length ``s`` on the path.

    Returns ``(i, p)`` where ``p`` is the point at arc length ``s`` and ``i`` is
    the index of the segment ``[path[i], path[i + 1]]`` containing it.
    """
    i = int(np.clip(np.searchsorted(cumlen, s, side="right") - 1, 0, len(path) - 2))
    seg = cumlen[i + 1] - cumlen[i]
    alpha = 0.0 if seg == 0 else (s - cumlen[i]) / seg
    return i, path[i] + alpha * (path[i + 1] - path[i])


def remove_duplicate_waypoints(path: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Drop waypoints that coincide with their predecessor."""
    keep = np.concatenate(([True], segment_lengths(path) > tol))
    return path[keep]
