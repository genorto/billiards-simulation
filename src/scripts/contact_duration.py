import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import LogLocator

from src.physics.integrator import integrator
from src.physics.scenarios import DEFAULT_K
from src.physics.state import make_ball

m1 = m2 = 5.0
r1 = r2 = 1.0
k = DEFAULT_K

speeds = np.geomspace(0.2, 4.0, 18)
duration_values: list[float] = []

for v in speeds:
    ball1 = make_ball(m1, r1, (0, 0), (v, 0))
    ball2 = make_ball(m2, r2, (8, 0), (0, 0))
    t_span = (0, 16 / v + 2)
    t, y = integrator(ball1, ball2, k, t_span)

    pos1 = y[0:2]
    pos2 = y[4:6]
    distance = np.linalg.norm(pos1 - pos2, axis=0)
    in_contact = distance < (r1 + r2)
    if not np.any(in_contact):
        duration_values.append(np.nan)
        continue
    idx = np.where(in_contact)[0]
    duration_values.append(t[idx[-1]] - t[idx[0]])

durations = np.array(duration_values)

mask = ~np.isnan(durations)
slope, intercept = np.polyfit(np.log(speeds[mask]), np.log(durations[mask]), 1)
print(f"fitted slope (theory: -1/5 = -0.2): {slope:.4f}")

def _labels_overlap(fig: plt.Figure, ax: plt.Axes) -> bool:
    fig.canvas.draw()
    assert isinstance(fig.canvas, FigureCanvasAgg)
    renderer = fig.canvas.get_renderer()
    boxes = [
        label.get_window_extent(renderer=renderer)
        for label in ax.get_xticklabels(which="both")
        if label.get_text()
    ]
    boxes.sort(key=lambda b: b.x0)
    return any(a.x1 > b.x0 for a, b in zip(boxes, boxes[1:]))


def _fit_xticks(fig: plt.Figure, ax: plt.Axes) -> None:
    # Подписи промежуточных (minor) делений лог-шкалы перекрываются при узком
    # диапазоне (< 2 декад) — подбираем самый частый набор чисел, который ещё
    # помещается без наложения, вместо того чтобы гадать один вариант.
    candidate_subs = [
        (1, 2, 3, 4, 5, 6, 7, 8, 9),
        (1, 2, 3, 5, 7),
        (1, 2, 5),
        (1, 3),
        (1,),
    ]
    for subs in candidate_subs:
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=subs))
        if not _labels_overlap(fig, ax):
            return


fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(speeds, durations, "o", ms=5, color="tab:orange")
fit_line = np.exp(intercept) * speeds**slope
ax.loglog(speeds, fit_line, "--", color="tab:blue")
ax.set_xlabel(r"Относительная скорость сближения $v$ (м/с)")
ax.set_ylabel(r"Длительность контакта $\tau$ (с)")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
_fit_xticks(fig, ax)
fig.tight_layout()

out_path = ROOT / "docs" / "contact_duration.png"
fig.savefig(out_path, dpi=130)
