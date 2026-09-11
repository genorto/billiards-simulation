import numpy as np

from src.physics import state
from src.physics.integrator import integrator

if __name__ == "__main__":
    ball1 = state.Ball(5, 1, np.array([0, 0]), np.array([1, 1]))
    ball2 = state.Ball(5, 1, np.array([10, 10]), np.array([1, 1]))

    t, y = integrator(ball1, ball2, 500000000, (0, 10))

    print(t)
    print(y)
