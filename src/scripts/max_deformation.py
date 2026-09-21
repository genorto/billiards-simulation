import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.physics.analysis import deformation
from src.physics.integrator import integrator
from src.physics.scenarios import DEFAULT_K
from src.physics.state import make_ball

m1 = m2 = 5.0
r1 = r2 = 1.0
k = DEFAULT_K

speeds = np.geomspace(0.5, 10.0, 20)
max_deformation_values: list[float] = []

for v in speeds:
    ball1 = make_ball(m1, r1, (0, 0), (v, 0))
    ball2 = make_ball(m2, r2, (8, 0), (0, 0))
    t_span = (0, 16 / v + 2)
    t, y = integrator(ball1, ball2, k, t_span)
    max_deformation_values.append(float(np.max(deformation(y, ball1, ball2))))

max_deformation = np.array(max_deformation_values) * 1000.0  # мм

slope, intercept = np.polyfit(np.log(speeds), np.log(max_deformation), 1)
print(f"fitted slope (theory: 4/5 = 0.8): {slope:.4f}")

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(speeds, max_deformation, "o", ms=5, color="tab:green")
fit_line = np.exp(intercept) * speeds**slope
ax.loglog(speeds, fit_line, "--", color="tab:blue")
ax.set_xlabel(r"Относительная скорость сближения $v$ (м/с)")
ax.set_ylabel(r"Максимальная деформация $\delta_{max}$ (мм)")
ax.grid(alpha=0.3, which="both")
fig.tight_layout()

out_path = ROOT / "docs" / "max_deformation.png"
fig.savefig(out_path, dpi=130)
