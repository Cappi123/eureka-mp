"""Matplotlib helpers. Every function draws on an explicit ``Axes``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

if TYPE_CHECKING:
    from planner2d.environment_2d import Environment
    from planner2d.prm import PRM
    from planner2d.rrt import Tree


def plot_environment(ax: Axes, env: Environment) -> None:
    ax.plot([0, env.size_x, env.size_x, 0, 0], [0, 0, env.size_y, env.size_y, 0], "k", lw=1.5)
    for ob in env.obs:
        corners = [(ob.x0, ob.y0), (ob.x1, ob.y1), (ob.x2, ob.y2)]
        ax.add_patch(Polygon(corners, closed=True, fc="tab:red", ec="darkred", alpha=0.6))
    ax.set_xlim(-0.2, env.size_x + 0.2)
    ax.set_ylim(-0.2, env.size_y + 0.2)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def plot_query(ax: Axes, start: np.ndarray, goal: np.ndarray) -> None:
    ax.plot(*start, "o", color="tab:green", ms=9, mec="k", label="start", zorder=5)
    ax.plot(*goal, "*", color="tab:blue", ms=15, mec="k", label="goal", zorder=5)


def plot_path(ax: Axes, path: np.ndarray, **kwargs) -> None:
    style = {"color": "black", "lw": 2.0, "zorder": 4} | kwargs
    ax.plot(path[:, 0], path[:, 1], **style)


def plot_roadmap(ax: Axes, prm: PRM) -> None:
    segments = prm.nodes[prm.edges]
    ax.add_collection(LineCollection(segments, colors="0.7", linewidths=0.5, zorder=1))
    ax.plot(prm.nodes[:, 0], prm.nodes[:, 1], ".", color="0.45", ms=2.5, zorder=2)


def plot_tree(ax: Axes, tree: Tree, color: str = "tab:orange") -> None:
    child = np.arange(1, len(tree))
    segments = np.stack([tree.nodes[tree.parents[child]], tree.nodes[child]], axis=1)
    ax.add_collection(LineCollection(segments, colors=color, linewidths=0.6, alpha=0.8, zorder=1))
