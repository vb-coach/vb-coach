"""Interactive animated demo: 4号位 high / half / shoot attacks in Plotly 3D HTML.

Generates one HTML file per attack type. Each file is browser-renderable
(double-click to open) and shows the court, moving ball, growing trail,
the hit window, attacker silhouette, hover tooltips, and score breakdown.

Run:
    uv run python examples/interactive_attacks_demo.py

Produces (in current directory):
    animated_position4_high.html
    animated_position4_half.html
    animated_position4_shoot.html
"""

from __future__ import annotations

import os

import numpy as np

from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.attacks_position4 import (
    score_high_ball,
    score_half_ball,
    score_shoot_ball,
)
from vb_coach.setting.hit_window import position4_hit_window
from vb_coach.trajectory import ReleaseState, simulate
from vb_coach.viz.plotly3d import AnimationConfig, plot_trajectory_animation


SCENARIOS = [
    ("4号位高球",   "high",
     np.array([0.5, 0.8, 2.2]),  np.array([-5.0, -0.7, 5.5]),
     score_high_ball),
    ("4号位半高",   "half",
     np.array([0.5, 0.8, 2.2]),  np.array([-7.0, -0.75, 4.4]),
     score_half_ball),
    ("4号位平拉开", "shoot",
     np.array([-1.5, 0.5, 2.3]), np.array([-8.5, -0.6, 3.0]),
     score_shoot_ball),
]


def main() -> None:
    attacker = DEFAULT_AMATEUR_MALE
    window = position4_hit_window(attacker)

    for label, slug, position, velocity, scorer in SCENARIOS:
        release = ReleaseState(position=position, velocity=velocity, spin=np.zeros(3))
        traj = simulate(release, t_max=3.0)
        score = scorer(traj, window)

        fig = plot_trajectory_animation(
            traj,
            attacker=attacker,
            window=window,
            score=score,
            title=label,
            config=AnimationConfig(frame_count=70, frame_duration_ms=35),
        )

        filename = f"animated_position4_{slug}.html"
        fig.write_html(filename, include_plotlyjs="cdn")
        size_kb = os.path.getsize(filename) / 1024
        print(f"{label:>12}: score={score.score:.3f}  ->  {filename}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
