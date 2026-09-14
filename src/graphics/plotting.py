import numpy as np
from matplotlib.axes import Axes

from src.physics.analysis import ConservationResult


def _plot_error(
    ax: Axes,
    t: np.ndarray,
    values: np.ndarray,
    ylabel: str,
) -> None:
    if np.any(np.isinf(values)):
        ax.set_axis_off()
        ax.text(
            0.5,
            0.5,
            "График не построен:\nотносительная ошибка содержит inf",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    ax.plot(t, values)
    ax.set_xlabel("Время, с")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def draw_conservation_errors(
    axes: np.ndarray,
    t: np.ndarray,
    result: ConservationResult,
) -> None:
    if axes.shape != (2, 2):
        raise ValueError("Ожидается массив осей формы (2, 2)")

    _plot_error(axes[0, 0], t, result.momentum_errors, "|ΔP|, кг·м/с")
    _plot_error(axes[0, 1], t, result.relative_momentum_errors * 100, "ΔP, %")
    _plot_error(axes[1, 0], t, result.energy_errors, "|ΔE|, Дж")
    _plot_error(axes[1, 1], t, result.relative_energy_errors * 100, "ΔE, %")
