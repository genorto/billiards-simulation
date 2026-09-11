from typing import Any

import numpy as np
from scipy import integrate

from src.physics.dynamics import dynamics
from src.physics.state import Ball, to_vector


def integrator(
    ball1: Ball, ball2: Ball, k: float, t_span: tuple[float, float]
) -> tuple[
    np.ndarray[tuple[int], np.dtype[np.float64]],
    np.ndarray[tuple[int, int], np.dtype[Any]],
]:
    args = (ball1.mass, ball1.radius, ball2.mass, ball2.radius, k)

    y0 = to_vector(ball1, ball2)

    sol = integrate.solve_ivp(
        dynamics,
        t_span,
        y0,
        args=args,
    )

    return sol.t, sol.y
