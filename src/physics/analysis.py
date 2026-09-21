from dataclasses import dataclass

import numpy as np

from src.physics.state import Ball


@dataclass(frozen=True)
class ConservationResult:
    momentum: np.ndarray
    energy: np.ndarray
    momentum_errors: np.ndarray
    relative_momentum_errors: np.ndarray
    energy_errors: np.ndarray
    relative_energy_errors: np.ndarray
    max_momentum_error: float
    max_relative_momentum_error: float
    max_energy_error: float
    max_relative_energy_error: float
    momentum_conserved: bool
    energy_conserved: bool


def total_momentum(y: np.ndarray, ball1: Ball, ball2: Ball) -> np.ndarray:
    velocity1 = y[2:4]
    velocity2 = y[6:8]
    return ball1.mass * velocity1 + ball2.mass * velocity2


def kinetic_energy(y: np.ndarray, ball1: Ball, ball2: Ball) -> np.ndarray:
    velocity1 = y[2:4]
    velocity2 = y[6:8]

    energy1 = 0.5 * ball1.mass * np.sum(velocity1**2, axis=0)
    energy2 = 0.5 * ball2.mass * np.sum(velocity2**2, axis=0)
    return energy1 + energy2


def _overlap(y: np.ndarray, ball1: Ball, ball2: Ball) -> np.ndarray:
    position1 = y[0:2]
    position2 = y[4:6]

    distance = np.linalg.norm(position1 - position2, axis=0)
    return np.maximum(ball1.radius + ball2.radius - distance, 0.0)


def deformation(y: np.ndarray, ball1: Ball, ball2: Ball) -> np.ndarray:
    return _overlap(y, ball1, ball2)


def hertz_potential_energy(
    y: np.ndarray, ball1: Ball, ball2: Ball, k: float
) -> np.ndarray:
    overlap = _overlap(y, ball1, ball2)
    return (2.0 / 5.0) * k * overlap**2.5


def total_energy(
    y: np.ndarray, ball1: Ball, ball2: Ball, k: float
) -> np.ndarray:
    return kinetic_energy(y, ball1, ball2) + hertz_potential_energy(
        y, ball1, ball2, k
    )


def expected_velocities_after_collision(
    ball1: Ball,
    ball2: Ball,
) -> tuple[np.ndarray, np.ndarray] | None:
    relative_position = ball2.position - ball1.position
    relative_velocity = ball2.velocity - ball1.velocity
    contact_distance = ball1.radius + ball2.radius

    a = np.dot(relative_velocity, relative_velocity)
    b = 2 * np.dot(relative_position, relative_velocity)
    c = np.dot(relative_position, relative_position) - contact_distance**2

    if c < 0:
        return None
    if a == 0:
        return None

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return None

    square_root = np.sqrt(discriminant)
    contact_time = (-b - square_root) / (2 * a)
    if contact_time < 0:
        return None

    contact_vector = relative_position + relative_velocity * contact_time
    normal = contact_vector / np.linalg.norm(contact_vector)
    normal_relative_speed = np.dot(ball1.velocity - ball2.velocity, normal)

    if normal_relative_speed <= 0:
        return None

    total_mass = ball1.mass + ball2.mass
    velocity1 = ball1.velocity - (
        2 * ball2.mass / total_mass * normal_relative_speed * normal
    )
    velocity2 = ball2.velocity + (
        2 * ball1.mass / total_mass * normal_relative_speed * normal
    )
    return velocity1, velocity2


def check_conservation(
    y: np.ndarray,
    ball1: Ball,
    ball2: Ball,
    k: float,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-9,
) -> ConservationResult:
    if y.ndim != 2 or y.shape[0] != 8:
        raise ValueError("Ожидается массив y формы (8, количество моментов времени)")

    momentum = total_momentum(y, ball1, ball2)
    energy = total_energy(y, ball1, ball2, k)

    initial_momentum = momentum[:, 0]
    initial_energy = energy[0]

    momentum_errors = np.linalg.norm(
        momentum - initial_momentum[:, np.newaxis], axis=0
    )
    energy_errors = np.abs(energy - initial_energy)

    max_momentum_error = float(np.max(momentum_errors))
    max_energy_error = float(np.max(energy_errors))

    initial_momentum_norm = np.linalg.norm(initial_momentum)
    if initial_momentum_norm == 0.0:
        relative_momentum_errors = np.where(
            momentum_errors == 0.0, 0.0, np.inf
        )
    else:
        relative_momentum_errors = momentum_errors / initial_momentum_norm

    if initial_energy == 0.0:
        relative_energy_errors = np.where(energy_errors == 0.0, 0.0, np.inf)
    else:
        relative_energy_errors = energy_errors / abs(initial_energy)

    max_relative_momentum_error = float(np.max(relative_momentum_errors))
    max_relative_energy_error = float(np.max(relative_energy_errors))

    return ConservationResult(
        momentum=momentum,
        energy=energy,
        momentum_errors=momentum_errors,
        relative_momentum_errors=relative_momentum_errors,
        energy_errors=energy_errors,
        relative_energy_errors=relative_energy_errors,
        max_momentum_error=max_momentum_error,
        max_relative_momentum_error=max_relative_momentum_error,
        max_energy_error=max_energy_error,
        max_relative_energy_error=max_relative_energy_error,
        momentum_conserved=bool(
            np.allclose(momentum, initial_momentum[:, np.newaxis], rtol=rtol, atol=atol)
        ),
        energy_conserved=bool(
            np.allclose(energy, initial_energy, rtol=rtol, atol=atol)
        ),
    )


def print_conservation_report(result: ConservationResult) -> None:
    print(f"ЗСИ выполняется: {'да' if result.momentum_conserved else 'нет'}")
    print(
        "Максимальная абсолютная ошибка импульса: "
        f"{result.max_momentum_error:.3e} кг·м/с"
    )
    print(
        "Максимальная относительная ошибка импульса: "
        f"{result.max_relative_momentum_error * 100:.3e} %"
    )
    print(f"ЗСЭ выполняется: {'да' if result.energy_conserved else 'нет'}")
    print(
        "Максимальная абсолютная ошибка энергии: "
        f"{result.max_energy_error:.3e} Дж"
    )
    print(
        "Максимальная относительная ошибка энергии: "
        f"{result.max_relative_energy_error * 100:.3e} %"
    )
