from dataclasses import dataclass

from src.state import Ball, make_ball

DEFAULT_K = 6.4e8


@dataclass(frozen=True)
class Scenario:
    name: str
    ball1: Ball
    ball2: Ball
    t_span: tuple[float, float]
    k: float = DEFAULT_K
    rtol: float = 1e-3
    atol: float = 1e-9
    velocity_rtol: float = 5e-3
    velocity_atol: float = 1e-6


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        name="head_on_equal_masses_both_moving",
        ball1=make_ball(5, 1, (0, 0), (1, 0)),
        ball2=make_ball(5, 1, (10, 0), (-1, 0)),
        t_span=(0, 10),
    ),
    Scenario(
        name="head_on_different_masses_both_moving",
        ball1=make_ball(5, 1, (0, 0), (1, 0)),
        ball2=make_ball(15, 1, (10, 0), (-1, 0)),
        t_span=(0, 10),
    ),
    Scenario(
        name="head_on_equal_masses_second_at_rest",
        ball1=make_ball(5, 1, (0, 0), (2, 0)),
        ball2=make_ball(5, 1, (8, 0), (0, 0)),
        t_span=(0, 8),
    ),
    Scenario(
        name="head_on_light_ball_hits_heavy_ball_at_rest",
        ball1=make_ball(5, 1, (0, 0), (2, 0)),
        ball2=make_ball(20, 1, (8, 0), (0, 0)),
        t_span=(0, 8),
    ),
    Scenario(
        name="angled_equal_masses_second_at_rest",
        ball1=make_ball(5, 1, (0, 0), (2, 0)),
        ball2=make_ball(5, 1, (8, 1.2), (0, 0)),
        t_span=(0, 8),
    ),
    Scenario(
        name="angled_different_masses_both_moving",
        ball1=make_ball(5, 1, (0, 0), (1.5, 0.4)),
        ball2=make_ball(20, 1, (9, 4), (-1, -0.2)),
        t_span=(0, 8),
    ),
    Scenario(
        name="rear_end_equal_masses",
        ball1=make_ball(5, 1, (0, 0), (2, 0)),
        ball2=make_ball(5, 1, (6, 0), (0.5, 0)),
        t_span=(0, 6),
    ),
    Scenario(
        name="different_masses_and_radii_both_moving",
        ball1=make_ball(8, 0.5, (0, 0), (2, 1)),
        ball2=make_ball(3, 1.5, (8, 4), (-0.5, -0.25)),
        t_span=(0, 6),
    ),
)


def find_scenario(name: str) -> Scenario | None:
    return next((scenario for scenario in SCENARIOS if scenario.name == name), None)
