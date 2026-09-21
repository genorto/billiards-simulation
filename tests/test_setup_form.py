import pytest

from src.graphics.setup_form import (
    default_t_span,
    parse_ball_from_texts,
    validate_balls,
)
from src.physics.state import make_ball


def test_parse_ball_from_texts_valid() -> None:
    texts = {"mass": "5.0", "radius": "1.0", "x": "0", "y": "0", "vx": "2", "vy": "0"}
    ball = parse_ball_from_texts(1, texts)
    assert ball.mass == 5.0
    assert ball.radius == 1.0
    assert list(ball.position) == [0.0, 0.0]
    assert list(ball.velocity) == [2.0, 0.0]


def test_parse_ball_from_texts_accepts_comma_decimal() -> None:
    texts = {"mass": "5,5", "radius": "1.0", "x": "0", "y": "0", "vx": "2", "vy": "0"}
    ball = parse_ball_from_texts(1, texts)
    assert ball.mass == 5.5


def test_parse_ball_from_texts_rejects_invalid_number() -> None:
    texts = {"mass": "abc", "radius": "1.0", "x": "0", "y": "0", "vx": "0", "vy": "0"}
    with pytest.raises(ValueError, match="Масса"):
        parse_ball_from_texts(1, texts)


@pytest.mark.parametrize("field, value", [("mass", "0"), ("radius", "-1")])
def test_parse_ball_from_texts_rejects_non_positive(field: str, value: str) -> None:
    texts = {"mass": "5.0", "radius": "1.0", "x": "0", "y": "0", "vx": "0", "vy": "0"}
    texts[field] = value
    with pytest.raises(ValueError):
        parse_ball_from_texts(1, texts)


def test_validate_balls_rejects_overlap() -> None:
    ball1 = make_ball(5, 1, (0, 0), (1, 0))
    ball2 = make_ball(5, 1, (1, 0), (0, 0))
    with pytest.raises(ValueError, match="пересекаются"):
        validate_balls(ball1, ball2)


def test_validate_balls_accepts_separated_balls() -> None:
    ball1 = make_ball(5, 1, (0, 0), (1, 0))
    ball2 = make_ball(5, 1, (8, 0), (0, 0))
    validate_balls(ball1, ball2)


def test_default_t_span_uses_closing_speed() -> None:
    ball1 = make_ball(5, 1, (0, 0), (2, 0))
    ball2 = make_ball(5, 1, (8, 0), (0, 0))
    start, end = default_t_span(ball1, ball2)
    assert start == 0.0
    assert end > 0.0


def test_default_t_span_falls_back_when_not_approaching() -> None:
    ball1 = make_ball(5, 1, (0, 0), (-1, 0))
    ball2 = make_ball(5, 1, (8, 0), (0, 0))
    assert default_t_span(ball1, ball2) == (0.0, 10.0)
