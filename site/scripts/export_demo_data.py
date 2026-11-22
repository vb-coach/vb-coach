"""Export real vb-coach trajectories for the homepage animations.

The homepage reads site/shared/data.js, which wraps the JSON built here.
Regenerate it from the repository root with:

    uv run python -I site/scripts/export_demo_data.py /tmp/data.json
    uv run python -I site/scripts/mc_preset.py /tmp/data.json
    uv run python -I site/scripts/export_float_variants.py /tmp/data.json
    uv run python -I site/scripts/export_max_float.py /tmp/data.json
    { printf 'window.VB_DATA = '; cat /tmp/data.json; printf ';\n'; } > site/shared/data.js

The homepage draws the net at 2.35 m; set "net_height" in the JSON before
wrapping it (the engine's physics and scores do not use the net height).
sweep_float.py is the launch-speed sweep behind the 22 m/s serve.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict

import numpy as np

from vb_coach.constants import NET_HEIGHT_MEN
from vb_coach.serving import CANONICAL_FLOAT_SERVES
from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.attacks_position4 import score_half_ball, score_high_ball, score_shoot_ball
from vb_coach.setting.hit_window import position4_hit_window
from vb_coach.trajectory import ReleaseState, simulate, simulate_stochastic


def resample(traj, dt: float) -> dict:
    t = traj.time
    grid = np.arange(0.0, t[-1], dt)
    grid = np.append(grid, t[-1])
    pos = np.stack([np.interp(grid, t, traj.position[:, i]) for i in range(3)], axis=1)
    vel = np.stack([np.interp(grid, t, traj.velocity[:, i]) for i in range(3)], axis=1)
    return {
        "t": np.round(grid, 4).tolist(),
        "p": np.round(pos, 4).tolist(),
        "speed": np.round(np.linalg.norm(vel, axis=1), 3).tolist(),
    }


def sets() -> dict:
    attacker = DEFAULT_AMATEUR_MALE
    window = position4_hit_window(attacker)
    # Same release states as examples/position4_all_attacks_demo.py.
    scenarios = [
        ("high", [0.5, 0.8, 2.2], [-5.0, -0.7, 5.5]),
        ("half", [0.5, 0.8, 2.2], [-7.0, -0.75, 4.4]),
        ("shoot", [-1.5, 0.5, 2.3], [-8.5, -0.6, 3.0]),
    ]
    scorers = {"high": score_high_ball, "half": score_half_ball, "shoot": score_shoot_ball}
    out = []
    for label, p0, v0 in scenarios:
        rs = ReleaseState(position=np.array(p0), velocity=np.array(v0), spin=np.zeros(3))
        traj = simulate(rs, t_max=3.0)
        scores = {}
        for name, fn in scorers.items():
            s = fn(traj, window)
            scores[name] = {
                "score": round(float(s.score), 3),
                "dwell_seconds": round(float(s.dwell_seconds), 3),
                "vz_contact": round(float(s.velocity_at_contact_z), 2),
                "drift_m": round(float(s.lateral_drift_m), 3),
            }
        out.append({
            "label": label,
            "release": {"position": p0, "velocity": v0, "speed": round(float(np.linalg.norm(v0)), 2)},
            "traj": resample(traj, 0.02),
            "scores": scores,
        })
    return {
        "attacker": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in asdict(attacker).items()},
        "window": {
            "center": np.round(window.center, 3).tolist(),
            "r_x": window.r_x,
            "r_y_front": window.r_y_front,
            "r_y_back": window.r_y_back,
            "r_z_up": window.r_z_up,
            "r_z_down": window.r_z_down,
        },
        "sets": out,
    }


def floats() -> dict:
    out = []
    for preset in CANONICAL_FLOAT_SERVES:
        rs = preset.release_state()
        noisy = simulate_stochastic(rs, rng=np.random.default_rng(preset.seed), t_max=2.0, dt=0.005)
        ref = simulate_stochastic(rs, rng=None, t_max=2.0, dt=0.005)
        out.append({
            "slug": preset.slug,
            "label": preset.label,
            "speed": preset.speed,
            "contact_height": preset.contact_height,
            "traj": resample(noisy, 0.01),
            "ref": resample(ref, 0.01),
        })
    # Monte Carlo as in examples/float_serve_demo.py: 50 seeds at 16 m/s, no spin.
    rs = ReleaseState(position=np.array([0.0, -9.0, 2.4]), velocity=np.array([0.0, 16.0, 0.5]), spin=np.zeros(3))
    landings = np.array([
        simulate_stochastic(rs, rng=np.random.default_rng(seed), t_max=1.5, dt=0.005).position[-1]
        for seed in range(50)
    ])
    ref_landing = simulate_stochastic(rs, rng=None, t_max=1.5, dt=0.005).position[-1]
    return {
        "serves": out,
        "montecarlo": {
            "speed": 16.0,
            "landings": np.round(landings[:, :2], 3).tolist(),
            "reference": np.round(ref_landing[:2], 3).tolist(),
            "sigma_x_cm": round(float(np.std(landings[:, 0])) * 100, 1),
            "sigma_y_cm": round(float(np.std(landings[:, 1])) * 100, 1),
        },
    }


def main() -> None:
    data = {"net_height": NET_HEIGHT_MEN, **sets(), "float": floats()}
    with open(sys.argv[1], "w") as f:
        json.dump(data, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
