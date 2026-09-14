from dataclasses import dataclass

import numpy as np
import pytest

from src.physics.analysis import ConservationResult, check_conservation
from src.physics.integrator import integrator
from src.physics.scenarios import SCENARIOS, Scenario


@dataclass(frozen=True)
class SimulationResult:
    case: Scenario
    t: np.ndarray
    y: np.ndarray
    conservation: ConservationResult


@pytest.fixture(
    scope="module",
    params=SCENARIOS,
    ids=lambda case: getattr(case, "name", None),
)
def simulation(request: pytest.FixtureRequest) -> SimulationResult:
    case: Scenario = request.param
    t, y = integrator(case.ball1, case.ball2, case.k, case.t_span)
    conservation = check_conservation(
        y,
        case.ball1,
        case.ball2,
        case.k,
        rtol=case.rtol,
        atol=case.atol,
    )
    return SimulationResult(case, t, y, conservation)
