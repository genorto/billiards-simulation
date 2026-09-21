import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox

from src.physics.scenarios import DEFAULT_K, Scenario
from src.physics.state import Ball, make_ball

FIELDS: tuple[tuple[str, str], ...] = (
    ("mass", "Масса, кг"),
    ("radius", "Радиус, м"),
    ("x", "x, м"),
    ("y", "y, м"),
    ("vx", "vx, м/с"),
    ("vy", "vy, м/с"),
)

DEFAULT_VALUES: dict[int, dict[str, str]] = {
    1: {"mass": "5.0", "radius": "1.0", "x": "0.0", "y": "0.0", "vx": "2.0", "vy": "0.0"},
    2: {"mass": "5.0", "radius": "1.0", "x": "8.0", "y": "0.0", "vx": "0.0", "vy": "0.0"},
}


def parse_ball_from_texts(ball_id: int, texts: dict[str, str]) -> Ball:
    values: dict[str, float] = {}
    for key, label in FIELDS:
        raw = texts[key].strip().replace(",", ".")
        try:
            values[key] = float(raw)
        except ValueError:
            raise ValueError(f"Шар {ball_id}: некорректное значение «{label}»") from None

    if values["mass"] <= 0:
        raise ValueError(f"Шар {ball_id}: масса должна быть положительной")
    if values["radius"] <= 0:
        raise ValueError(f"Шар {ball_id}: радиус должен быть положительным")

    return make_ball(
        values["mass"],
        values["radius"],
        (values["x"], values["y"]),
        (values["vx"], values["vy"]),
    )


def validate_balls(ball1: Ball, ball2: Ball) -> None:
    separation = float(np.linalg.norm(ball1.position - ball2.position))
    if separation <= ball1.radius + ball2.radius:
        raise ValueError(
            "Шары пересекаются в начальный момент — увеличьте расстояние между ними"
        )


def default_t_span(ball1: Ball, ball2: Ball) -> tuple[float, float]:
    offset = ball2.position - ball1.position
    separation = float(np.linalg.norm(offset))
    direction = offset / separation
    closing_speed = float(np.dot(ball1.velocity - ball2.velocity, direction))

    if closing_speed <= 1e-9:
        return (0.0, 10.0)

    time_to_contact = separation / closing_speed
    return (0.0, max(2.5 * time_to_contact, 5.0))


def run_setup_form() -> Scenario | None:
    fig = plt.figure(figsize=(8, 6))
    fig.suptitle("Настройка параметров столкновения")

    fig.text(0.16, 0.90, "Шар 1", fontsize=12, fontweight="bold")
    fig.text(0.66, 0.90, "Шар 2", fontsize=12, fontweight="bold")

    row_positions = (0.80, 0.70, 0.60, 0.50, 0.40, 0.30)
    textboxes: dict[tuple[int, str], TextBox] = {}
    for (key, label), y in zip(FIELDS, row_positions):
        for ball_id, x in ((1, 0.16), (2, 0.66)):
            ax = fig.add_axes((x, y, 0.18, 0.045))
            textboxes[(ball_id, key)] = TextBox(
                ax, label, initial=DEFAULT_VALUES[ball_id][key]
            )

    error_text = fig.text(0.5, 0.16, "", ha="center", color="#c1121f", wrap=True)

    button_ax = fig.add_axes((0.4, 0.05, 0.2, 0.06))
    button = Button(button_ax, "Запустить")

    result: list[Scenario | None] = [None]

    def on_submit(event: object) -> None:
        try:
            texts1 = {key: textboxes[(1, key)].text for key, _ in FIELDS}
            texts2 = {key: textboxes[(2, key)].text for key, _ in FIELDS}
            ball1 = parse_ball_from_texts(1, texts1)
            ball2 = parse_ball_from_texts(2, texts2)
            validate_balls(ball1, ball2)
        except ValueError as exc:
            error_text.set_text(str(exc))
            fig.canvas.draw_idle()
            return

        result[0] = Scenario(
            name="custom",
            ball1=ball1,
            ball2=ball2,
            t_span=default_t_span(ball1, ball2),
            k=DEFAULT_K,
        )
        plt.close(fig)

    button.on_clicked(on_submit)
    plt.show()
    return result[0]
