import numpy as np
import pytest

from conftest import make_env
from planner2d.collision import CollisionChecker
from planner2d.scenarios import make_scenario


@pytest.mark.parametrize("seed", range(10))
def test_point_check_matches_environment_oracle(seed):
    env, _ = make_scenario(seed)
    checker = CollisionChecker(env)
    points = checker.sample_uniform(np.random.default_rng(seed), 2000)
    expected = np.array([not env.check_collision(x, y) for x, y in points])
    np.testing.assert_array_equal(checker.points_free(points), expected)


def test_points_outside_workspace_are_not_free():
    checker = CollisionChecker(make_env([]))
    assert not checker.is_free([-0.1, 1.0])
    assert not checker.is_free([1.0, 6.1])
    assert checker.is_free([10.0, 6.0])  # boundary is part of the workspace


def test_segment_through_triangle_collides():
    checker = CollisionChecker(make_env([(4, 2, 6, 2, 5, 4)]))
    assert not checker.is_segment_free([0, 2.5], [10, 2.5])
    assert checker.is_segment_free([0, 5], [10, 5])


def test_segment_clipping_only_a_corner_collides():
    # A thin sliver: a sampled check with coarse resolution would miss it.
    checker = CollisionChecker(make_env([(4, 2, 6, 2, 5, 2.001)]))
    assert not checker.is_segment_free([5, 0], [5, 4])


def test_segment_along_obstacle_edge_is_free():
    # Obstacles are open sets, as in Environment.check_collision.
    checker = CollisionChecker(make_env([(4, 2, 6, 2, 5, 4)]))
    assert checker.is_segment_free([3, 2], [7, 2])


def test_segment_ending_inside_obstacle_collides():
    checker = CollisionChecker(make_env([(4, 2, 6, 2, 5, 4)]))
    assert not checker.is_segment_free([5, 0], [5, 2.5])


@pytest.mark.parametrize("seed", range(5))
def test_exact_segment_check_agrees_with_dense_sampling(seed):
    env, _ = make_scenario(seed)
    checker = CollisionChecker(env)
    rng = np.random.default_rng(seed)
    a = checker.sample_uniform(rng, 300)
    b = np.clip(a + rng.normal(scale=1.5, size=a.shape), checker.lower, checker.upper)
    t = np.linspace(0.0, 1.0, 20_001)[:, None]
    sampled = np.array([checker.points_free(ai + t * (bi - ai)).all() for ai, bi in zip(a, b)])
    exact = checker.segments_free(a, b)
    # Sampling can only miss collisions, never invent them.
    assert not np.any(exact & ~sampled)
    assert np.mean(exact == sampled) > 0.99


def test_sample_free_returns_only_free_points(course_scenario):
    _, checker, _ = course_scenario
    points = checker.sample_free(np.random.default_rng(0), 500)
    assert points.shape == (500, 2)
    assert np.all(checker.points_free(points))
