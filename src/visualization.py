import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from matplotlib.text import Text

from src.analysis import ConservationResult
from src.plotting import draw_conservation_errors
from src.state import Ball


def _square_limits(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    ball1: Ball,
    ball2: Ball,
) -> tuple[tuple[float, float], tuple[float, float]]:
    left = min(x1 - ball1.radius, x2 - ball2.radius)
    right = max(x1 + ball1.radius, x2 + ball2.radius)
    bottom = min(y1 - ball1.radius, y2 - ball2.radius)
    top = max(y1 + ball1.radius, y2 + ball2.radius)

    minimum_size = 8 * max(ball1.radius, ball2.radius)
    size = 1.2 * max(right - left, top - bottom, minimum_size)
    center_x = 0.5 * (left + right)
    center_y = 0.5 * (bottom + top)
    half_size = 0.5 * size
    return (
        (center_x - half_size, center_x + half_size),
        (center_y - half_size, center_y + half_size),
    )


def animate_simulation(
    t: np.ndarray,
    y: np.ndarray,
    ball1: Ball,
    ball2: Ball,
    conservation: ConservationResult,
    *,
    fps: int = 60,
    playback_speed: float = 1.0,
    collision_slowdown: float = 100.0,
    show: bool = True,
) -> FuncAnimation:
    if fps <= 0 or playback_speed <= 0 or collision_slowdown <= 0:
        raise ValueError(
            "fps, playback_speed и collision_slowdown должны быть положительными"
        )

    distances = np.linalg.norm(y[0:2] - y[4:6], axis=0)
    contact_distance = ball1.radius + ball2.radius
    contact_margin = 0.005 * min(ball1.radius, ball2.radius)
    near_contact = np.minimum(distances[:-1], distances[1:]) <= (
        contact_distance + contact_margin
    )

    display_intervals = np.diff(t) / playback_speed
    display_intervals = np.where(
        near_contact,
        display_intervals * collision_slowdown,
        display_intervals,
    )
    display_times = np.concatenate(([0.0], np.cumsum(display_intervals)))
    frame_count = int(np.ceil(display_times[-1] * fps)) + 1
    frame_display_times = np.linspace(0.0, display_times[-1], frame_count)
    frame_times = np.interp(frame_display_times, display_times, t)

    x1 = np.interp(frame_times, t, y[0])
    y1 = np.interp(frame_times, t, y[1])
    x2 = np.interp(frame_times, t, y[4])
    y2 = np.interp(frame_times, t, y[5])

    figure = plt.figure(figsize=(16, 9), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1.25, 1, 1))
    collision_ax = figure.add_subplot(grid[:, 0])
    error_axes = np.array(
        [
            [figure.add_subplot(grid[0, 1]), figure.add_subplot(grid[0, 2])],
            [figure.add_subplot(grid[1, 1]), figure.add_subplot(grid[1, 2])],
        ]
    )

    x_limits, y_limits = _square_limits(
        x1[0], y1[0], x2[0], y2[0], ball1, ball2
    )
    collision_ax.set_xlim(*x_limits)
    collision_ax.set_ylim(*y_limits)
    collision_ax.set_aspect("equal", adjustable="box")
    collision_ax.set_box_aspect(1)
    collision_ax.set_xlabel("x, м")
    collision_ax.set_ylabel("y, м")
    collision_ax.set_title("Столкновение шаров")
    collision_ax.set_facecolor("#176b3a")
    collision_ax.grid(color="white", alpha=0.15)

    circle1 = Circle((x1[0], y1[0]), ball1.radius, color="#f4f1de")
    circle2 = Circle((x2[0], y2[0]), ball2.radius, color="#e63946")
    collision_ax.add_patch(circle1)
    collision_ax.add_patch(circle2)

    time_text = collision_ax.text(
        0.03,
        0.97,
        "",
        color="white",
        ha="left",
        va="top",
        transform=collision_ax.transAxes,
    )

    draw_conservation_errors(error_axes, t, conservation)
    time_markers = [
        ax.axvline(t[0], color="#e63946", linewidth=1)
        for ax in error_axes.flat
        if ax.axison
    ]

    figure.suptitle("Моделирование столкновения и проверка законов сохранения")

    def update(frame: int) -> tuple[Circle | Text, ...]:
        current_time = frame_times[frame]
        circle1.center = (x1[frame], y1[frame])
        circle2.center = (x2[frame], y2[frame])
        x_limits, y_limits = _square_limits(
            x1[frame], y1[frame], x2[frame], y2[frame], ball1, ball2
        )
        collision_ax.set_xlim(*x_limits)
        collision_ax.set_ylim(*y_limits)
        current_distance = np.hypot(x1[frame] - x2[frame], y1[frame] - y2[frame])
        contact_text = " · контакт" if current_distance <= contact_distance else ""
        time_text.set_text(f"t = {current_time:.4f} с{contact_text}")
        for marker in time_markers:
            marker.set_xdata([current_time, current_time])
        return circle1, circle2, time_text, *time_markers

    animation = FuncAnimation(
        figure,
        update,
        frames=frame_count,
        interval=1000 / fps,
        blit=False,
        cache_frame_data=False,
    )

    if show:
        plt.show()
    return animation
