import numpy as np

from src.physics import state
from src.physics import forces


def dynamics(
    t: float, y: np.ndarray, m1: float, r1: float, m2: float, r2: float, k: float
) -> np.ndarray:
    ball1, ball2 = state.from_vector(y, m1, r1, m2, r2)

    v1 = ball1.velocity
    v2 = ball2.velocity

    r = ball1.position - ball2.position
    distance = np.linalg.norm(r)
    delta_x = r1 + r2 - distance

    force = forces.hertz_force(k, delta_x)
    F1, F2 = forces.apply_force_along_normal(force, ball1, ball2)

    a1 = F1 / m1
    a2 = F2 / m2

    return np.concatenate([v1, a1, v2, a2])
