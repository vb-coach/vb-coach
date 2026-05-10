"""Demo: setter at zone 2/3 sets a high ball to 4号位 (left front).

Run:
    uv run python examples/basic_set.py

Produces basic_set_3d.png and basic_set_side.png in the current directory.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from vb_coach.constants import NET_HEIGHT_MEN
from vb_coach.plotting import plot_trajectory_3d, plot_trajectory_side
from vb_coach.trajectory import ReleaseState, simulate


def main() -> None:
    release = ReleaseState(
        position=np.array([0.5, 0.8, 2.2]),
        velocity=np.array([-3.0, -1.0, 7.5]),
        spin=np.array([2.0, 0.0, 0.0]),
    )
    traj = simulate(release, t_max=3.0)

    print(f"Flight time: {traj.time[-1]:.2f} s")
    print(f"Apex height: {traj.apex_height():.2f} m")
    print(f"Apex at t=:  {traj.time[traj.apex_index()]:.2f} s")
    print(
        f"Landing:     x={traj.position[-1, 0]:.2f}  "
        f"y={traj.position[-1, 1]:.2f}  z={traj.position[-1, 2]:.3f}"
    )

    fig3d = plot_trajectory_3d(traj, net_height=NET_HEIGHT_MEN)
    fig3d.savefig("basic_set_3d.png", dpi=120, bbox_inches="tight")
    plt.close(fig3d)

    fig_side = plot_trajectory_side(traj, net_height=NET_HEIGHT_MEN)
    fig_side.savefig("basic_set_side.png", dpi=120, bbox_inches="tight")
    plt.close(fig_side)

    print("Saved: basic_set_3d.png, basic_set_side.png")


if __name__ == "__main__":
    main()
