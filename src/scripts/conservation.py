import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.graphics.plotting import draw_conservation_errors
from src.physics.analysis import check_conservation
from src.physics.integrator import integrator
from src.physics.scenarios import find_scenario

sc = find_scenario("angled_different_masses_both_moving")
if sc is None:
    raise ValueError("Сценарий не найден")

t, y = integrator(sc.ball1, sc.ball2, sc.k, sc.t_span)
result = check_conservation(y, sc.ball1, sc.ball2, sc.k, rtol=sc.rtol, atol=sc.atol)

fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
draw_conservation_errors(axes, t, result)
fig.tight_layout()

out_path = ROOT / "docs" / "conservation.png"
fig.savefig(out_path, dpi=130)
