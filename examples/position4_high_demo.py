"""Demo: score multiple 4号位高球 trajectories side by side.

Runs three release states (a good high ball, a flat low set, and a poorly
aimed set) for the default amateur male attacker, scores each with
score_high_ball, and produces a comparison plot.

Run:
    uv run python examples/position4_high_demo.py

Produces:
    position4_high_demo.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vb_coach.constants import NET_HEIGHT_MEN
from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.attacks_position4 import score_high_ball
from vb_coach.setting.hit_window import position4_hit_window
from vb_coach.trajectory import ReleaseState, simulate


_RELEASE_POSITION = np.array([0.5, 0.8, 2.2])


def main() -> None:
    attacker = DEFAULT_AMATEUR_MALE
    window = position4_hit_window(attacker)

    scenarios = [
        ("good high ball (v0=-5.0,-0.7,5.5)", np.array([-5.0, -0.7, 5.5])),
        ("low / flat set (v0=-4.5,-0.5,4.0)", np.array([-4.5, -0.5, 4.0])),
        ("off-line set (v0=-3,-1.5,7.5)",     np.array([-3.0, -1.5, 7.5])),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, v0 in scenarios:
        rs = ReleaseState(position=_RELEASE_POSITION.copy(), velocity=v0,
                          spin=np.array([0.0, 0.0, 0.0]))
        traj = simulate(rs, t_max=3.0)
        result = score_high_ball(traj, window)
        line, = ax.plot(traj.position[:, 1], traj.position[:, 2], lw=2,
                        label=f"{label}: score={result.score:.2f}")
        if result.contact_index >= 0:
            cp = traj.position[result.contact_index]
            ax.scatter([cp[1]], [cp[2]], color=line.get_color(), marker="o", s=60)
        print(f"{label}: score={result.score:.3f} "
              f"dwell={result.dwell_seconds:.2f}s "
              f"vz={result.velocity_at_contact_z:.2f}m/s "
              f"drift={result.lateral_drift_m*100:.1f}cm")

    # Net marker
    ax.plot([0.0, 0.0], [0.0, NET_HEIGHT_MEN], color="gray", lw=2, label="net")

    # Hit-window y-z slice (at the window's x center)
    y0 = window.center[1] - window.r_y_back
    y1 = window.center[1] + window.r_y_front
    z0 = window.center[2] - window.r_z_down
    z1 = window.center[2] + window.r_z_up
    ax.fill_between([y0, y1], z0, z1, color="green", alpha=0.18,
                    label="hit window (y-z slice)")

    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xlabel("y [m] (across court)")
    ax.set_ylabel("z [m]")
    ax.set_title("4号位高球 — three sets, side view")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("position4_high_demo.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved: position4_high_demo.png")


if __name__ == "__main__":
    main()
