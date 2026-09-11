from typing import Any, TypeAlias

import numpy as np

from src.physics.state import Ball

Vec2: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]


def hertz_force(k: float, delta_x: float) -> float:
    if delta_x < 0:
        return 0.0
    return k * (delta_x**1.5)


def apply_force_along_normal(
    force: float, ball1: Ball, ball2: Ball
) -> tuple[Vec2, Vec2]:
    r = ball1.position - ball2.position
    r_length = np.linalg.norm(r)

    r_norm = r / r_length

    ball1_force = r_norm * force
    ball2_force = -ball1_force

    return ball1_force, ball2_force
