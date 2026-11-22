"""Add 50-serve landing scatters for each float-serve preset to data.json.

Run after export_demo_data.py:
    uv run python -I site/scripts/mc_preset.py <data.json>
"""
import json, sys
import numpy as np
from vb_coach.serving import CANONICAL_FLOAT_SERVES
from vb_coach.trajectory import simulate_stochastic
out = {}
for p in CANONICAL_FLOAT_SERVES:
    rs = p.release_state()
    L = np.array([simulate_stochastic(rs, rng=np.random.default_rng(s), t_max=2.0, dt=0.005).position[-1] for s in range(50)])
    ref = simulate_stochastic(rs, rng=None, t_max=2.0, dt=0.005).position[-1]
    out[p.slug] = {"speed": p.speed, "landings": np.round(L[:, :2], 3).tolist(), "reference": np.round(ref[:2], 3).tolist(),
                   "sigma_x_cm": round(float(np.std(L[:, 0]))*100, 1), "sigma_y_cm": round(float(np.std(L[:, 1]))*100, 1)}
    print(p.slug, p.speed, out[p.slug]["sigma_x_cm"], out[p.slug]["sigma_y_cm"], ref[:2])
d = json.load(open(sys.argv[1])); d["float"]["montecarlo_presets"] = out; json.dump(d, open(sys.argv[1], "w"), separators=(",", ":"))
