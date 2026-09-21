import numpy as np

from src.physics.analysis import deformation
from src.physics.state import make_ball, to_vector


def test_deformation_zero_when_balls_not_touching() -> None:
    ball1 = make_ball(5, 1, (0, 0), (0, 0))
    ball2 = make_ball(5, 1, (8, 0), (0, 0))
    y = to_vector(ball1, ball2)[:, np.newaxis]

    result = deformation(y, ball1, ball2)

    assert result[0] == 0.0


def test_deformation_positive_during_overlap() -> None:
    ball1 = make_ball(5, 1, (0, 0), (0, 0))
    ball2 = make_ball(5, 1, (1.5, 0), (0, 0))
    y = to_vector(ball1, ball2)[:, np.newaxis]

    result = deformation(y, ball1, ball2)

    assert result[0] == 0.5


def test_deformation_matches_formula_along_trajectory() -> None:
    ball1 = make_ball(5, 1, (0, 0), (2, 0))
    ball2 = make_ball(5, 1, (8, 0), (0, 0))

    positions1 = np.array([[0.0, 3.0, 7.0], [0.0, 0.0, 0.0]])
    positions2 = np.array([[8.0, 8.0, 8.0], [0.0, 0.0, 0.0]])
    velocities = np.zeros((2, 3))
    y = np.vstack([positions1, velocities, positions2, velocities])

    distances = np.linalg.norm(positions1 - positions2, axis=0)
    expected = np.maximum(ball1.radius + ball2.radius - distances, 0.0)

    np.testing.assert_allclose(deformation(y, ball1, ball2), expected)
