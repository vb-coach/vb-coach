"""Generic animated trajectory demo.

This intentionally uses only ReleaseState -> Trajectory -> Plotly animation.
No attacker, hit window, or serve/set-specific scoring is required, so it
serves as the minimal reusable entry point for arbitrary scenes.

Run:
    uv run python examples/animated_trajectory_demo.py

Produces:
    animated_trajectory_demo.html
"""

from __future__ import annotations

import os

import numpy as np

from vb_coach.trajectory import ReleaseState, simulate
from vb_coach.viz.plotly3d import AnimationConfig, plot_trajectory_animation


def main() -> None:
    release = ReleaseState(
        position=np.array([0.0, 0.7, 2.2]),
        velocity=np.array([-4.5, -0.5, 5.2]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj = simulate(release, t_max=3.0)
    fig = plot_trajectory_animation(
        traj,
        title="Generic volleyball trajectory",
        config=AnimationConfig(frame_count=70, frame_duration_ms=35),
    )
    filename = "animated_trajectory_demo.html"
    fig.write_html(filename, include_plotlyjs="cdn")
    size_kb = os.path.getsize(filename) / 1024
    print(f"Saved: {filename} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
