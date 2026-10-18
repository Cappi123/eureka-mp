import numpy as np
import pytest

from conftest import assert_valid_path, make_env
from planner2d.collision import CollisionChecker
from planner2d.prm import PRM
from planner2d.rrt import RRT, BiRRT, Tree


def plan(planner_cls, checker, start, goal, seed=0):
    rng = np.random.default_rng(seed)
    if planner_cls is PRM:
        prm = PRM(checker, rng=rng)
        prm.build()
        return prm.query(start, goal)
    return planner_cls(checker, rng=rng).plan(start, goal).path


PLANNERS = [PRM, RRT, BiRRT]


@pytest.mark.parametrize("planner_cls", PLANNERS)
def test_finds_path_around_wall(planner_cls, wall_env):
    checker = CollisionChecker(wall_env)
    start, goal = np.array([1.0, 1.0]), np.array([9.0, 1.0])
    path = plan(planner_cls, checker, start, goal)
    assert_valid_path(path, checker, start, goal)
    assert np.max(path[:, 1]) >= 5.0  # it must go over the wall


@pytest.mark.parametrize("planner_cls", PLANNERS)
def test_course_instance(planner_cls, course_scenario):
    _, checker, query = course_scenario
    path = plan(planner_cls, checker, query.start, query.goal, seed=3)
    assert_valid_path(path, checker, query.start, query.goal)


@pytest.mark.parametrize("planner_cls", PLANNERS)
def test_reports_failure_when_goal_is_enclosed(planner_cls):
    # Goal sits inside a box of four thin triangular walls. Obstacles are open
    # sets, so the walls overlap past the corners to leave no gap.
    box = [
        (2.5, 2, 7.5, 2, 5, 1.8),
        (2.5, 4, 7.5, 4, 5, 4.2),
        (3, 1.5, 3, 4.5, 2.8, 3),
        (7, 1.5, 7, 4.5, 7.2, 3),
    ]
    checker = CollisionChecker(make_env(box))
    rng = np.random.default_rng(0)
    if planner_cls is PRM:
        prm = PRM(checker, n_samples=300, rng=rng)
        prm.build()
        assert prm.query([1, 1], [5, 3]) is None
    else:
        result = planner_cls(checker, max_iterations=300, rng=rng).plan([1, 1], [5, 3])
        assert not result.success
        assert result.iterations == 300


@pytest.mark.parametrize("planner_cls", PLANNERS)
def test_rejects_start_in_collision(planner_cls):
    checker = CollisionChecker(make_env([(4, 2, 6, 2, 5, 4)]))
    assert plan(planner_cls, checker, [5, 2.5], [1, 1]) is None


def test_prm_reuses_roadmap_across_queries(course_scenario):
    _, checker, query = course_scenario
    prm = PRM(checker, rng=np.random.default_rng(3))
    prm.build()
    nodes_before = prm.nodes.copy()
    for _ in range(3):
        assert prm.query(query.start, query.goal) is not None
    np.testing.assert_array_equal(prm.nodes, nodes_before)
    assert len(prm._adjacency) == len(prm.nodes)


def test_prm_query_before_build_raises(wall_env):
    with pytest.raises(RuntimeError):
        PRM(CollisionChecker(wall_env)).query([1, 1], [9, 1])


def test_rrt_edges_respect_step_size(wall_env):
    checker = CollisionChecker(wall_env)
    result = RRT(checker, step_size=0.25, rng=np.random.default_rng(1)).plan([1, 1], [9, 1])
    tree = result.trees[0]
    child = np.arange(1, len(tree))
    lengths = np.linalg.norm(tree.nodes[child] - tree.nodes[tree.parents[child]], axis=1)
    assert np.all(lengths <= 0.25 + 1e-9)


def test_tree_grows_past_initial_capacity():
    tree = Tree(np.zeros(2), capacity=2)
    for i in range(10):
        tree.add(np.array([i + 1.0, 0.0]), parent=i)
    assert len(tree) == 11
    np.testing.assert_array_equal(tree.path_from_root(10)[:, 0], np.arange(11.0))


@pytest.mark.parametrize("kwargs", [{"step_size": 0}, {"goal_bias": 1.5}, {"max_iterations": 0}])
def test_rrt_validates_parameters(kwargs, wall_env):
    with pytest.raises(ValueError):
        RRT(CollisionChecker(wall_env), **kwargs)
