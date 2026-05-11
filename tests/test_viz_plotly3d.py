import numpy as np
import plotly.graph_objects as go

from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.attacks_position4 import score_high_ball
from vb_coach.setting.hit_window import position4_hit_window
from vb_coach.trajectory import ReleaseState, simulate
from vb_coach.viz.plotly3d import plot_attack_3d


def _sample_trajectory():
    rs = ReleaseState(
        position=np.array([0.5, 0.8, 2.2]),
        velocity=np.array([-4.8, -0.5, 5.5]),
        spin=np.zeros(3),
    )
    return simulate(rs, t_max=3.0)


def test_plot_attack_3d_returns_figure():
    fig = plot_attack_3d(_sample_trajectory())
    assert isinstance(fig, go.Figure)


def test_plot_attack_3d_with_court_traces():
    fig = plot_attack_3d(_sample_trajectory())
    # Court has at least 5 traces (boundary, 2 attack lines, net line, net mesh, 2 antennae)
    assert len(fig.data) >= 7


def test_plot_attack_3d_with_window_and_attacker():
    traj = _sample_trajectory()
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    fig = plot_attack_3d(traj, attacker=DEFAULT_AMATEUR_MALE, window=window)
    # With window + attacker we get more traces
    base = len(plot_attack_3d(traj).data)
    assert len(fig.data) > base


def test_plot_attack_3d_with_score_shows_breakdown_in_title():
    traj = _sample_trajectory()
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    score = score_high_ball(traj, window)
    fig = plot_attack_3d(traj, attacker=DEFAULT_AMATEUR_MALE, window=window, score=score)
    title = fig.layout.title.text
    assert "score=" in title
    assert "dwell" in title
    assert "velocity" in title
    assert "lateral" in title


def test_plot_attack_3d_writes_html(tmp_path):
    fig = plot_attack_3d(_sample_trajectory())
    html_path = tmp_path / "attack.html"
    fig.write_html(str(html_path))
    assert html_path.exists()
    assert html_path.stat().st_size > 10_000  # plotly HTML is at least 10 KB
