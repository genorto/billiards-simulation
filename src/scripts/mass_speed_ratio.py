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

m1 = 5.0
v1 = 2.0
r = 1.0
k = DEFAULT_K

ratios = np.geomspace(0.1, 10, 25)
v1_final_sim_values: list[float] = []
v1_final_theory_values: list[float] = []

for q in ratios:
    m2 = m1 * q
    ball1 = make_ball(m1, r, (0, 0), (v1, 0))
    ball2 = make_ball(m2, r, (8, 0), (0, 0))
    t, y = integrator(ball1, ball2, k, (0, 8))
    v1_final_sim_values.append(y[2, -1])
    v1_final_theory_values.append(v1 * (m1 - m2) / (m1 + m2))

v1_final_sim = np.array(v1_final_sim_values)
v1_final_theory = np.array(v1_final_theory_values)
max_abs_err = np.max(np.abs(v1_final_sim - v1_final_theory))
print("max |sim - theory| for v1':", max_abs_err)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.axhline(0, color="gray", linewidth=0.8)
ax.axvline(1, color="gray", linewidth=0.8, linestyle="--")
ax.plot(ratios, v1_final_theory, lw=2)
ax.plot(ratios, v1_final_sim, "o", ms=4, color="tab:orange")
ax.set_xscale("log")
ax.set_xlabel(r"Отношение масс $q = m_2/m_1$")
ax.set_ylabel(r"Конечная скорость шара 1, $v_1'$ (м/с)")
ax.grid(alpha=0.3)
fig.tight_layout()

out_path = ROOT / "docs" / "mass_speed_ratio.png"
fig.savefig(out_path, dpi=130)
