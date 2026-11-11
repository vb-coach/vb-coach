"""Basic matplotlib plotting for trajectories.

Importing matplotlib happens only in this module so the core library has no
plot dependency at import time. Plot functions return the Figure so the
caller can save, show, or compose.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection

from vb_coach.trajectory import Trajectory


def plot_trajectory_3d(traj: Trajectory, net_height: float | None = None) -> plt.Figure:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj.position[:, 0], traj.position[:, 1], traj.position[:, 2], lw=2)
    ax.scatter(*traj.position[0], color="green", s=40, label="release")
    apex = traj.apex_index()
    ax.scatter(*traj.position[apex], color="orange", s=40, label="apex")
    ax.scatter(*traj.position[-1], color="red", s=40, label="end")
    if net_height is not None:
        xs = [-4.5, 4.5, 4.5, -4.5, -4.5]
        ys = [0.0, 0.0, 0.0, 0.0, 0.0]
        zs = [0.0, 0.0, net_height, net_height, 0.0]
        ax.plot(xs, ys, zs, color="gray", lw=1, alpha=0.6)
    ax.set_xlabel("x [m] (along net)")
    ax.set_ylabel("y [m] (across court)")
    ax.set_zlabel("z [m]")
    ax.legend()
    return fig


def plot_trajectory_side(traj: Trajectory, net_height: float | None = None) -> plt.Figure:
    """Plot height against across-court distance, in meters."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(traj.position[:, 1], traj.position[:, 2], lw=2)
    ax.scatter(traj.position[0, 1], traj.position[0, 2], color="green", s=40, label="release")
    apex = traj.apex_index()
    ax.scatter(traj.position[apex, 1], traj.position[apex, 2], color="orange", s=40, label="apex")
    ax.scatter(traj.position[-1, 1], traj.position[-1, 2], color="red", s=40, label="end")
    if net_height is not None:
        ax.plot([0.0, 0.0], [0.0, net_height], color="gray", lw=2)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xlabel("y [m] (across court)")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig
