import numpy as np
import pytest

from src.analysis import expected_velocities_after_collision
from tests.conftest import SimulationResult


def test_conservation_laws(simulation: SimulationResult) -> None:
    assert simulation.conservation.momentum_conserved
    assert simulation.conservation.energy_conserved

def test_velocity_vectors_after_collision(simulation: SimulationResult) -> None:
    expected_velocities = expected_velocities_after_collision(
        simulation.case.ball1,
        simulation.case.ball2,
    )
    if expected_velocities is None:
        pytest.skip("Аналитическая проверка неприменима: столкновения нет")

    expected_velocity1, expected_velocity2 = expected_velocities

    np.testing.assert_allclose(
        simulation.y[2:4, -1],
        expected_velocity1,
        rtol=simulation.case.velocity_rtol,
        atol=simulation.case.velocity_atol,
    )
    np.testing.assert_allclose(
        simulation.y[6:8, -1],
        expected_velocity2,
        rtol=simulation.case.velocity_rtol,
        atol=simulation.case.velocity_atol,
    )
