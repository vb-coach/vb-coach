"""Add engine takes of the largest-drift legal float serve to data.json.

A sweep over launch speed and vertical speed (contact 2.8 m at the end line,
must clear a 2.35 m net and land in court) found 22 m/s with vz = 1.5 m/s
drifts most at the receiver's depth: RMS 50 cm over 100 seeds. This keeps
seeds whose offset from the no-noise path at y = 7.2 m is at least 60 cm,
half drifting each way, so the float is easy to see.

Run: uv run --project <vb-coach> python -I export_max_float.py <data.json>
"""

from __future__ import annotations

import json
import sys

import numpy as np

from vb_coach.trajectory import ReleaseState, simulate_stochastic

SPEED, VZ, CONTACT, DEPTH = 22.0, 1.5, 2.8, 7.2
MIN_OFFSET, TAKES, DT_OUT = 0.60, 8, 0.02


def resample(traj, dt: float) -> dict:
    t = traj.time
    grid = np.append(np.arange(0.0, t[-1], dt), t[-1])
    pos = np.stack([np.interp(grid, t, traj.position[:, i]) for i in range(3)], axis=1)
    vel = np.stack([np.interp(grid, t, traj.velocity[:, i]) for i in range(3)], axis=1)
    return {"t": np.round(grid, 4).tolist(), "p": np.round(pos, 3).tolist(),
            "speed": np.round(np.linalg.norm(vel, axis=1), 2).tolist()}


def x_at_depth(traj) -> float:
    p = traj.position
    i = int(np.argmax(p[:, 1] >= DEPTH)) if np.any(p[:, 1] >= DEPTH) else len(p) - 1
    return float(p[i, 0])


def main() -> None:
    path = sys.argv[1]
    data = json.load(open(path))
    rs = ReleaseState(position=np.array([0.0, -9.0, CONTACT]), velocity=np.array([0.0, SPEED, VZ]), spin=np.zeros(3))
    ref = simulate_stochastic(rs, rng=None, t_max=2.5, dt=0.005)
    x_ref = x_at_depth(ref)
    left, right = [], []
    for seed in range(2000, 2200):
        tr = simulate_stochastic(rs, rng=np.random.default_rng(seed), t_max=2.5, dt=0.005)
        off = x_at_depth(tr) - x_ref
        if abs(off) < MIN_OFFSET or abs(tr.position[-1, 0]) > 4.5 or tr.position[-1, 1] > 9.0:
            continue
        (left if off < 0 else right).append((seed, off, tr))
    picked = left[: TAKES // 2] + right[: TAKES // 2]
    takes = []
    for seed, off, tr in picked:
        print(f"seed={seed} offset@7.2m={off * 100:+.0f}cm land=({tr.position[-1, 0]:+.2f},{tr.position[-1, 1]:.2f})")
        takes.append({"seed": seed, "lateral_speed": 0.0, "offset_cm": round(off * 100), "traj": resample(tr, DT_OUT), "ref": resample(ref, DT_OUT)})
    print(f"candidates: left {len(left)}, right {len(right)} of 200 seeds")
    data["float"].setdefault("variants", {})["max22"] = takes
    data["float"]["max22"] = {"speed": SPEED, "vz": VZ, "contact_height": CONTACT}
    json.dump(data, open(path, "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
