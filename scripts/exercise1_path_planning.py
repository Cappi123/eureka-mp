"""Exercise 1: solve the 2D planning problem with PRM, RRT and Bi-RRT.

Produces:
  figures/ex1_course_instance.png   the instance from the course notes, all three planners
  figures/ex1_prm_multi_query.png   one PRM roadmap answering several queries
  figures/ex1_gallery.png           other environments and queries
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from planner2d import PRM, RRT, BiRRT, CollisionChecker, path_length  # noqa: E402
from planner2d.plotting import (  # noqa: E402
    plot_environment,
    plot_path,
    plot_query,
    plot_roadmap,
    plot_tree,
)
from planner2d.scenarios import make_scenario  # noqa: E402

TREE_COLORS = ("tab:orange", "tab:purple")


def solve_and_plot(ax, planner_name, env, checker, query, rng) -> None:
    """Run one planner on one query and draw its search structure and path."""
    t0 = time.perf_counter()
    if planner_name == "PRM":
        prm = PRM(checker, rng=rng)
        prm.build()
        path = prm.query(query.start, query.goal)
        elapsed = time.perf_counter() - t0
        plot_environment(ax, env)
        plot_roadmap(ax, prm)
        detail = f"{prm.stats.n_nodes} nodes, {prm.stats.n_edges} edges"
    else:
        planner_cls = RRT if planner_name == "RRT" else BiRRT
        result = planner_cls(checker, rng=rng).plan(query.start, query.goal)
        elapsed = time.perf_counter() - t0
        path = result.path
        plot_environment(ax, env)
        for tree, color in zip(result.trees, TREE_COLORS):
            plot_tree(ax, tree, color)
        detail = f"{sum(len(t) for t in result.trees)} nodes, {result.iterations} iters"
    plot_query(ax, query.start, query.goal)
    if path is None:
        ax.set_title(f"{planner_name}: no path found\n{detail}, {elapsed * 1e3:.0f} ms", fontsize=9)
    else:
        plot_path(ax, path)
        ax.set_title(
            f"{planner_name}: length {path_length(path):.2f}\n{detail}, {elapsed * 1e3:.0f} ms",
            fontsize=9,
        )


def figure_course_instance(out_dir: Path, seed: int) -> None:
    env, (query,) = make_scenario(4)
    checker = CollisionChecker(env)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    for ax, name in zip(axes, ("PRM", "RRT", "Bi-RRT")):
        solve_and_plot(ax, name, env, checker, query, np.random.default_rng(seed))
    fig.suptitle("Course instance (np.random.seed(4))")
    fig.tight_layout()
    fig.savefig(out_dir / "ex1_course_instance.png", dpi=150)
    plt.close(fig)


def figure_prm_multi_query(out_dir: Path, seed: int, n_queries: int = 6) -> None:
    env, queries = make_scenario(4, n_queries=n_queries)
    checker = CollisionChecker(env)
    prm = PRM(checker, rng=np.random.default_rng(seed))
    prm.build()
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_environment(ax, env)
    plot_roadmap(ax, prm)
    colors = plt.cm.tab10(np.arange(n_queries))
    n_solved = 0
    for query, color in zip(queries, colors):
        path = prm.query(query.start, query.goal)
        ax.plot(*query.start, "o", color=color, mec="k", ms=8, zorder=5)
        ax.plot(*query.goal, "*", color=color, mec="k", ms=13, zorder=5)
        if path is not None:
            plot_path(ax, path, color=color)
            n_solved += 1
    ax.set_title(f"One roadmap, {n_queries} queries ({n_solved} solved); o = start, * = goal")
    fig.tight_layout()
    fig.savefig(out_dir / "ex1_prm_multi_query.png", dpi=150)
    plt.close(fig)


def figure_gallery(out_dir: Path, seed: int, env_seeds: tuple[int, ...] = (0, 1, 2, 3)) -> None:
    names = ("PRM", "RRT", "Bi-RRT")
    fig, axes = plt.subplots(len(env_seeds), len(names), figsize=(15, 3.7 * len(env_seeds)))
    for row, env_seed in zip(axes, env_seeds):
        env, (query,) = make_scenario(env_seed, n_obstacles=6)
        checker = CollisionChecker(env)
        for ax, name in zip(row, names):
            solve_and_plot(ax, name, env, checker, query, np.random.default_rng(seed))
        row[0].set_ylabel(f"environment seed {env_seed}")
    fig.tight_layout()
    fig.savefig(out_dir / "ex1_gallery.png", dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="planner random seed")
    parser.add_argument("--out", type=Path, default=Path("figures"), help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    figure_course_instance(args.out, args.seed)
    figure_prm_multi_query(args.out, args.seed)
    figure_gallery(args.out, args.seed)
    print(f"Figures written to {args.out}/")


if __name__ == "__main__":
    main()
