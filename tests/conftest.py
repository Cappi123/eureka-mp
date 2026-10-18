import matplotlib

matplotlib.use("Agg")  # environment_2d imports pylab; keep tests headless

import numpy as np
import pytest

from planner2d.collision import CollisionChecker
from planner2d.environment_2d import Environment, TriangularObstacle
from planner2d.scenarios import make_scenario


def make_env(triangles, size_x=10.0, size_y=6.0):
    """Environment with hand-placed obstacles instead of random ones."""
    env = Environment(size_x, size_y, 0)
    env.obs = [TriangularObstacle(*t) for t in triangles]
    return env


@pytest.fixture
def wall_env():
    """A vertical wall from y=0 up to y=5, leaving a gap at the top (y in [5, 6])."""
    return make_env([(4.9, 0.0, 5.1, 0.0, 5.0, 5.0), (4.9, 0.0, 5.1, 5.0, 5.0, 5.0)])


@pytest.fixture
def course_scenario():
    """The environment and query shown in the course exercise (seed 4)."""
    env, (query,) = make_scenario(4)
    return env, CollisionChecker(env), query


def assert_valid_path(path, checker, start, goal):
    assert path is not None
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], goal)
    assert np.all(checker.segments_free(path[:-1], path[1:]))
