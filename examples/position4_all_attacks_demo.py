"""Demo: all three 4号位 attack types scored on three sample sets.

Cross-product evaluation:
- Three trajectories: tuned for high, half, and shoot respectively
- Three scoring functions: score_high_ball, score_half_ball, score_shoot_ball
- Output: 3x3 table of scores and a side-view plot showing all three
  trajectories with all three hit windows overlaid.

Run:
    uv run python examples/position4_all_attacks_demo.py

Produces:
    position4_all_attacks_demo.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vb_coach.constants import NET_HEIGHT_MEN
from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.attacks_position4 import (
    score_high_ball,
    score_half_ball,
    score_shoot_ball,
)
from vb_coach.setting.hit_window import (
    position4_high_window,
    position4_half_window,
    position4_shoot_window,
)
from vb_coach.trajectory import ReleaseState, simulate


def main() -> None:
    attacker = DEFAULT_AMATEUR_MALE
    high_win = position4_high_window(attacker)
    half_win = position4_half_window(attacker)
    shoot_win = position4_shoot_window(attacker)

    scenarios = [
        ("high",  np.array([0.5, 0.8, 2.2]),  np.array([-4.8, -0.5, 5.5])),
        ("half",  np.array([0.5, 0.8, 2.2]),  np.array([-6.5, -0.5, 4.5])),
        ("shoot", np.array([-1.5, 0.5, 2.3]), np.array([-7.5, -0.5, 3.5])),
    ]

    trajectories: list[tuple[str, np.ndarray, np.ndarray, object]] = []
    for label, position, velocity in scenarios:
        rs = ReleaseState(position=position, velocity=velocity, spin=np.zeros(3))
        traj = simulate(rs, t_max=3.0)
        trajectories.append((label, position, velocity, traj))

    # Build score table
    print(f"{'set type':>8} | {'high':>8} {'half':>8} {'shoot':>8}")
    print("-" * 42)
    for label, _, _, traj in trajectories:
        sh = score_high_ball(traj, high_win)
        sm = score_half_ball(traj, half_win)
        sf = score_shoot_ball(traj, shoot_win)
        print(f"{label:>8} | {sh.score:>8.3f} {sm.score:>8.3f} {sf.score:>8.3f}")

    # Build side-view plot
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = {"high": "tab:blue", "half": "tab:orange", "shoot": "tab:green"}
    for label, _, _, traj in trajectories:
        ax.plot(traj.position[:, 1], traj.position[:, 2], lw=2,
                label=f"{label} set", color=colors[label])

    # Net
    ax.plot([0.0, 0.0], [0.0, NET_HEIGHT_MEN], color="gray", lw=2, label="net")

    # Three hit-window y-z slices (overlapping at y center 0.2, different z extents)
    for win, name, fill_color in [
        (high_win, "high window", "tab:blue"),
        (half_win, "half window", "tab:orange"),
        (shoot_win, "shoot window", "tab:green"),
    ]:
        y0, y1 = win.center[1] - win.dy, win.center[1] + win.dy
        z0, z1 = win.center[2] - win.dz, win.center[2] + win.dz
        ax.fill_between([y0, y1], z0, z1, color=fill_color, alpha=0.12,
                        label=f"{name} y-z slice")

    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xlabel("y [m] (across court)")
    ax.set_ylabel("z [m]")
    ax.set_title("4号位 attacks: three sets, three windows")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("position4_all_attacks_demo.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved: position4_all_attacks_demo.png")


if __name__ == "__main__":
    main()
