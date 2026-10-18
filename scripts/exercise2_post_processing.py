"""Exercise 2: post-process PRM, RRT and Bi-RRT paths by random shortcutting.

Produces:
  figures/ex2_shortcut_course_instance.png   raw vs shortcut paths, course instance
  figures/ex2_shortcut_convergence.png       path length vs shortcut repetitions
and prints length statistics over many environments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from planner2d import PRM, RRT, BiRRT, CollisionChecker, path_length, shortcut_path  # noqa: E402
from planner2d.plotting import plot_environment, plot_path, plot_query  # noqa: E402
from planner2d.scenarios import make_scenario  # noqa: E402

PLANNER_NAMES = ("PRM", "RRT", "Bi-RRT")


def plan(name: str, checker: CollisionChecker, start, goal, rng) -> np.ndarray | None:
    if name == "PRM":
        prm = PRM(checker, rng=rng)
        prm.build()
        return prm.query(start, goal)
    planner_cls = RRT if name == "RRT" else BiRRT
    return planner_cls(checker, rng=rng).plan(start, goal).path


def figures_course_instance(out_dir: Path, seed: int, repetitions: int) -> None:
    env, (query,) = make_scenario(4)
    checker = CollisionChecker(env)
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.9))
    fig_conv, ax_conv = plt.subplots(figsize=(7, 4))
    for ax, name in zip(axes, PLANNER_NAMES):
        rng = np.random.default_rng(seed)
        raw = plan(name, checker, query.start, query.goal, rng)
        result = shortcut_path(raw, checker, max_repetitions=repetitions, rng=rng)
        plot_environment(ax, env)
        plot_path(ax, raw, color="0.55", lw=1.2, label=f"raw ({path_length(raw):.2f})")
        plot_path(
            ax,
            result.path,
            color="black",
            lw=2.2,
            label=f"shortcut ({path_length(result.path):.2f})",
        )
        plot_query(ax, query.start, query.goal)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
        ax.set_title(
            f"{name}: {len(raw)} -> {len(result.path)} waypoints, "
            f"{result.n_accepted}/{repetitions} shortcuts accepted",
            fontsize=9,
        )
        ax_conv.plot(result.length_history, label=name)
    fig.suptitle(f"Path shortcutting, {repetitions} repetitions (course instance)")
    fig.tight_layout()
    fig.savefig(out_dir / "ex2_shortcut_course_instance.png", dpi=150)
    plt.close(fig)

    ax_conv.set_xlabel("repetition")
    ax_conv.set_ylabel("path length")
    ax_conv.set_title("Shortcutting convergence (course instance)")
    ax_conv.grid(True, alpha=0.3)
    ax_conv.legend()
    fig_conv.tight_layout()
    fig_conv.savefig(out_dir / "ex2_shortcut_convergence.png", dpi=150)
    plt.close(fig_conv)


def print_statistics(seed: int, repetitions: int, n_envs: int) -> None:
    """Mean length before/after shortcutting over random environments."""
    print(f"\nShortcutting over {n_envs} random environments ({repetitions} repetitions):")
    print(f"{'planner':<8} {'solved':>7} {'raw length':>11} {'shortcut':>9} {'reduction':>10}")
    for name in PLANNER_NAMES:
        raw_lengths, short_lengths = [], []
        for env_seed in range(n_envs):
            env, (query,) = make_scenario(env_seed)
            checker = CollisionChecker(env)
            rng = np.random.default_rng(seed + env_seed)
            raw = plan(name, checker, query.start, query.goal, rng)
            if raw is None:
                continue
            short = shortcut_path(raw, checker, max_repetitions=repetitions, rng=rng).path
            raw_lengths.append(path_length(raw))
            short_lengths.append(path_length(short))
        raw_mean, short_mean = np.mean(raw_lengths), np.mean(short_lengths)
        print(
            f"{name:<8} {len(raw_lengths):>4}/{n_envs:<2} {raw_mean:>11.2f} {short_mean:>9.2f}"
            f" {100 * (1 - short_mean / raw_mean):>9.1f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=0, help="planner random seed")
    parser.add_argument("--repetitions", type=int, default=200, help="shortcut repetitions")
    parser.add_argument("--n-envs", type=int, default=30, help="environments for statistics")
    parser.add_argument("--out", type=Path, default=Path("figures"), help="output directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    figures_course_instance(args.out, args.seed, args.repetitions)
    print(f"Figures written to {args.out}/")
    print_statistics(args.seed, args.repetitions, args.n_envs)


if __name__ == "__main__":
    main()
