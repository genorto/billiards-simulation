import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(speeds, durations, "o", ms=5, color="tab:orange")
fit_line = np.exp(intercept) * speeds**slope
ax.loglog(speeds, fit_line, "--", color="tab:blue")
ax.set_xlabel(r"Относительная скорость сближения $v$ (м/с)")
ax.set_ylabel(r"Длительность контакта $\tau$ (с)")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()

out_path = ROOT / "docs" / "contact_duration.png"
fig.savefig(out_path, dpi=130)
