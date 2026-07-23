import numpy as np
import plotly.graph_objects as go
import pytest

from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.attacks_position4 import score_high_ball
from vb_coach.setting.hit_window import position4_hit_window
from vb_coach.trajectory import ReleaseState, simulate
from vb_coach.viz.plotly3d import (
    AnimationConfig,
    plot_attack_3d,
    plot_trajectory_animation,
    plot_trajectory_3d,
)


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


def test_plot_trajectory_3d_returns_figure_without_attack_overlays():
    fig = plot_trajectory_3d(_sample_trajectory())
    assert isinstance(fig, go.Figure)
    names = {trace.name for trace in fig.data}
    assert "trajectory" in names
    assert "hit window" not in names


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


def test_plot_trajectory_animation_has_frames_and_controls():
    fig = plot_trajectory_animation(
        _sample_trajectory(),
        config=AnimationConfig(frame_count=12, frame_duration_ms=20),
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.frames) == 12
    assert len(fig.layout.updatemenus) == 1
    assert len(fig.layout.sliders) == 1
    assert len(fig.layout.sliders[0].steps) == 12
    names = {trace.name for trace in fig.data}
    assert "court boundary" in names
    assert "animated trail" in names
    assert "ball" in names


def test_plot_trajectory_animation_has_adjustable_playback_speed():
    fig = plot_trajectory_animation(
        _sample_trajectory(),
        config=AnimationConfig(
            frame_count=8,
            frame_duration_ms=40,
            playback_speeds=(0.5, 1.0, 2.0),
        ),
    )
    buttons = fig.layout.updatemenus[0].buttons
    assert [button.label for button in buttons] == [
        "Play 0.5x",
        "Play 1x",
        "Play 2x",
        "Pause",
    ]
    assert [button.args[1]["frame"]["duration"] for button in buttons[:3]] == [
        80,
        40,
        20,
    ]


def test_animation_config_rejects_invalid_playback_speed():
    with pytest.raises(ValueError, match="playback_speeds"):
        AnimationConfig(playback_speeds=(1.0, 0.0))


def test_plot_trajectory_animation_with_attack_overlays():
    traj = _sample_trajectory()
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    score = score_high_ball(traj, window)
    fig = plot_trajectory_animation(
        traj,
        reference_trajectory=traj,
        attacker=DEFAULT_AMATEUR_MALE,
        window=window,
        score=score,
        config=AnimationConfig(frame_count=8),
    )
    names = {trace.name for trace in fig.data}
    assert "no-float reference" in names
    assert "hit window" in names
    assert any(str(name).startswith("attacker") for name in names)
    assert "score=" in fig.layout.title.text


def test_plot_trajectory_animation_writes_html(tmp_path):
    fig = plot_trajectory_animation(
        _sample_trajectory(),
        config=AnimationConfig(frame_count=8),
    )
    html_path = tmp_path / "animated.html"
    fig.write_html(str(html_path))
    text = html_path.read_text()
    assert "frames" in text
    assert "animated trail" in text
    assert html_path.stat().st_size > 10_000
