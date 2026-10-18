"""Compare PRM, RRT and Bi-RRT running times on single- and multi-query problems.

PRM pays for its roadmap once and then answers each query cheaply; RRT and
Bi-RRT grow new trees for every query. For each environment we time the PRM
build and every query, so the cumulative cost of answering the first k
queries can be read off for every k up to --max-queries.

Produces:
  results/benchmark.csv               one row per (environment, planner, query)
  results/benchmark.md                summary table
  figures/benchmark_cumulative_time.png
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from planner2d import PRM, RRT, BiRRT, CollisionChecker, path_length  # noqa: E402
from planner2d.scenarios import make_scenario  # noqa: E402

PLANNER_NAMES = ("PRM", "RRT", "Bi-RRT")


@dataclass
class Record:
    env_seed: int
    planner: str
    query_index: int
    time_s: float  # for PRM, query 0 includes the roadmap build
    segment_checks: int  # likewise
    success: bool
    path_length: float


def run_environment(env_seed: int, n_queries: int, planner_seed: int) -> list[Record]:
    env, queries = make_scenario(env_seed, n_queries=n_queries)
    checker = CollisionChecker(env)
    records: list[Record] = []

    prm = PRM(checker, rng=np.random.default_rng(planner_seed))
    checker.reset_counters()
    build_time = prm.build().build_time
    for k, q in enumerate(queries):
        if k > 0:  # query 0 is charged for the build
            checker.reset_counters()
        t0 = time.perf_counter()
        path = prm.query(q.start, q.goal)
        elapsed = time.perf_counter() - t0 + (build_time if k == 0 else 0.0)
        records.append(_record(env_seed, "PRM", k, elapsed, checker, path))

    for name, planner_cls in (("RRT", RRT), ("Bi-RRT", BiRRT)):
        planner = planner_cls(checker, rng=np.random.default_rng(planner_seed))
        for k, q in enumerate(queries):
            checker.reset_counters()
            t0 = time.perf_counter()
            path = planner.plan(q.start, q.goal).path
            elapsed = time.perf_counter() - t0
            records.append(_record(env_seed, name, k, elapsed, checker, path))
    return records


def _record(env_seed, planner, k, elapsed, checker, path) -> Record:
    length = path_length(path) if path is not None else float("nan")
    return Record(env_seed, planner, k, elapsed, checker.n_segment_checks, path is not None, length)


def cumulative(
    records: list[Record], name: str, field: str, n_envs: int, n_queries: int
) -> np.ndarray:
    """(n_envs, n_queries) array: total ``field`` spent answering the first k+1 queries."""
    values = np.zeros((n_envs, n_queries))
    for r in records:
        if r.planner == name:
            values[r.env_seed, r.query_index] = getattr(r, field)
    return np.cumsum(values, axis=1)


def write_outputs(records, n_envs, n_queries, results_dir: Path, figures_dir: Path) -> str:
    with open(results_dir / "benchmark.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0])))
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)

    checkpoints = [k for k in (1, 5, 10, 20, 50, 100) if k <= n_queries]
    times = {n: cumulative(records, n, "time_s", n_envs, n_queries) for n in PLANNER_NAMES}
    checks = {n: cumulative(records, n, "segment_checks", n_envs, n_queries) for n in PLANNER_NAMES}

    header = "| planner | " + " | ".join(f"k = {k}" for k in checkpoints) + " |"
    rule = "|---|" + "---:|" * len(checkpoints)
    lines = [
        f"Means over {n_envs} environments, answering the first k queries on each.",
        "PRM is charged once for building its roadmap.",
        "",
        "**Total wall-clock time (ms)**",
        "",
        header,
        rule,
    ]
    for name in PLANNER_NAMES:
        cells = [f"{1e3 * times[name][:, k - 1].mean():.0f}" for k in checkpoints]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "**Total segment collision checks**", "", header, rule]
    for name in PLANNER_NAMES:
        cells = [f"{checks[name][:, k - 1].mean():,.0f}" for k in checkpoints]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "**Success rate and path length over all queries**", ""]
    lines += ["| planner | success rate | mean length (solved) |", "|---|---:|---:|"]
    for name in PLANNER_NAMES:
        mine = [r for r in records if r.planner == name]
        success = 100 * np.mean([r.success for r in mine])
        length = np.mean([r.path_length for r in mine if r.success])
        lines.append(f"| {name} | {success:.1f}% | {length:.2f} |")
    table = "\n".join(lines)
    (results_dir / "benchmark.md").write_text(table + "\n")

    fig, (ax_time, ax_checks) = plt.subplots(1, 2, figsize=(12, 4.5))
    ks = np.arange(1, n_queries + 1)
    for name in PLANNER_NAMES:
        ax_time.plot(ks, 1e3 * times[name].mean(axis=0), marker="o", ms=3, label=name)
        ax_checks.plot(ks, checks[name].mean(axis=0), marker="o", ms=3, label=name)
    ax_time.set_ylabel("cumulative planning time [ms]")
    ax_checks.set_ylabel("cumulative segment collision checks")
    for ax in (ax_time, ax_checks):
        ax.set_xlabel("number of queries on the same environment")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Single- vs multi-query cost (mean over {n_envs} environments)")
    fig.tight_layout()
    fig.savefig(figures_dir / "benchmark_cumulative_time.png", dpi=150)
    plt.close(fig)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-envs", type=int, default=20)
    parser.add_argument("--max-queries", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="planner random seed")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figures", type=Path, default=Path("figures"))
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)

    records: list[Record] = []
    for env_seed in range(args.n_envs):
        records += run_environment(env_seed, args.max_queries, args.seed)
        print(f"environment {env_seed + 1}/{args.n_envs} done", flush=True)
    print()
    print(write_outputs(records, args.n_envs, args.max_queries, args.results, args.figures))


if __name__ == "__main__":
    main()
