"""Add several engine-computed takes of each float-serve preset to data.json.

Each take uses its own random seed for the lateral (knuckle) force and a small
random aim to the left or right (lateral launch speed). The no-noise reference
of the same release is stored with it.

Run: uv run --project <vb-coach> python -I export_float_variants.py <data.json>
"""

from __future__ import annotations

import json
import sys

import numpy as np

from vb_coach.serving import CANONICAL_FLOAT_SERVES
from vb_coach.trajectory import simulate_stochastic

TAKES = 8
DT_OUT = 0.02
AIM_MAX = 0.6  # m/s of lateral launch speed


def resample(traj, dt: float) -> dict:
    t = traj.time
    grid = np.append(np.arange(0.0, t[-1], dt), t[-1])
    pos = np.stack([np.interp(grid, t, traj.position[:, i]) for i in range(3)], axis=1)
    vel = np.stack([np.interp(grid, t, traj.velocity[:, i]) for i in range(3)], axis=1)
    return {
        "t": np.round(grid, 4).tolist(),
        "p": np.round(pos, 3).tolist(),
        "speed": np.round(np.linalg.norm(vel, axis=1), 2).tolist(),
    }


def main() -> None:
    path = sys.argv[1]
    with open(path) as f:
        data = json.load(f)
    aim_rng = np.random.default_rng(2026)
    variants = {}
    for preset in CANONICAL_FLOAT_SERVES:
        takes = []
        for k in range(TAKES):
            lateral = float(np.round(aim_rng.uniform(-AIM_MAX, AIM_MAX), 2))
            seed = 100 + k
            rs = preset.release_state(lateral_speed=lateral)
            noisy = simulate_stochastic(rs, rng=np.random.default_rng(seed), t_max=2.0, dt=0.005)
            ref = simulate_stochastic(rs, rng=None, t_max=2.0, dt=0.005)
            takes.append({"seed": seed, "lateral_speed": lateral, "traj": resample(noisy, DT_OUT), "ref": resample(ref, DT_OUT)})
            end, rend = noisy.position[-1], ref.position[-1]
            print(f"{preset.slug} seed={seed} aim={lateral:+.2f} land=({end[0]:+.2f},{end[1]:.2f}) "
                  f"drift={abs(end[0] - rend[0]) * 100:.0f}cm")
        variants[preset.slug] = takes
    data["float"]["variants"] = variants
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
