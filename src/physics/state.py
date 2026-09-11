from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

Vec2: TypeAlias = np.ndarray[Any, np.dtype[np.float64]]


@dataclass(frozen=True)
class Ball:
    mass: float
    radius: float
    position: Vec2
    velocity: Vec2


def to_vector(ball1: Ball, ball2: Ball) -> np.ndarray:
    return np.concatenate(
        [
            ball1.position,
            ball1.velocity,
            ball2.position,
            ball2.velocity,
        ]
    )


def from_vector(
    y: np.ndarray, m1: float, r1: float, m2: float, r2: float
) -> tuple[Ball, Ball]:
    p1 = y[0:2]
    v1 = y[2:4]
    p2 = y[4:6]
    v2 = y[6:8]

    ball1 = Ball(m1, r1, p1, v1)
    ball2 = Ball(m2, r2, p2, v2)

    return ball1, ball2
