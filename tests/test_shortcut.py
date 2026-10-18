import numpy as np
import pytest

from conftest import assert_valid_path
from planner2d.collision import CollisionChecker
from planner2d.geometry import path_length
from planner2d.rrt import RRT
from planner2d.shortcut import shortcut_path


def test_zigzag_in_free_space_becomes_straight(wall_env):
    checker = CollisionChecker(wall_env)
    zigzag = np.array([[1, 1], [1.5, 3], [2, 1], [2.5, 3], [3, 1]], dtype=float)
    result = shortcut_path(zigzag, checker, max_repetitions=300, rng=np.random.default_rng(0))
    assert path_length(result.path) == pytest.approx(2.0, abs=0.05)


@pytest.mark.parametrize("seed", range(5))
def test_shortcut_is_monotone_and_collision_free(seed, course_scenario):
    _, checker, query = course_scenario
    raw = RRT(checker, rng=np.random.default_rng(seed)).plan(query.start, query.goal).path
    assert raw is not None
    result = shortcut_path(raw, checker, rng=np.random.default_rng(seed))
    assert_valid_path(result.path, checker, query.start, query.goal)
    assert np.all(np.diff(result.length_history) <= 1e-12)
    assert result.length_history[0] == pytest.approx(path_length(raw))
    assert result.length_history[-1] == pytest.approx(path_length(result.path))
    assert result.length_history[-1] < result.length_history[0]


def test_input_path_is_not_modified(wall_env):
    checker = CollisionChecker(wall_env)
    path = np.array([[1, 1], [2, 3], [3, 1]], dtype=float)
    original = path.copy()
    shortcut_path(path, checker, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(path, original)


def test_two_point_path_is_unchanged(wall_env):
    checker = CollisionChecker(wall_env)
    path = np.array([[1, 1], [3, 1]], dtype=float)
    result = shortcut_path(path, checker, rng=np.random.default_rng(0))
    np.testing.assert_array_equal(result.path, path)
    assert result.n_accepted == 0


def test_degenerate_path_raises(wall_env):
    with pytest.raises(ValueError):
        shortcut_path(np.array([[1.0, 1.0], [1.0, 1.0]]), CollisionChecker(wall_env))
