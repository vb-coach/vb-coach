import numpy as np

from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.attacks_position4 import HalfBallScore, score_half_ball
from vb_coach.setting.hit_window import position4_hit_window
from vb_coach.trajectory import ReleaseState, simulate


def _set_release(velocity: np.ndarray, position: np.ndarray | None = None) -> ReleaseState:
    if position is None:
        position = np.array([0.5, 0.8, 2.2])
    return ReleaseState(position=position, velocity=velocity, spin=np.array([0.0, 0.0, 0.0]))


def test_score_half_returns_zero_when_trajectory_misses_window():
    rs = _set_release(velocity=np.array([0.0, 0.0, 5.0]))
    traj = simulate(rs, t_max=3.0)
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    result = score_half_ball(traj, window)
    assert result.score == 0.0


def test_score_half_returns_zero_when_only_ascent_intersects():
    rs = _set_release(
        position=np.array([0.5, 0.0, 1.5]),
        velocity=np.array([-4.5, 0.4, 6.0]),
    )
    traj = simulate(rs, t_max=3.0)
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    result = score_half_ball(traj, window)
    assert result.score == 0.0


def test_score_half_good_semi_high_above_threshold():
    # Retuned for unified ellipsoid (r_z_up=0.05, r_y_back=0.15): adjusted vx/vy/vz
    # so the ball crosses x=-4 near y=0.2 at z~2.75 (inside tighter window). Scores ~0.71.
    rs = _set_release(velocity=np.array([-7.0, -0.75, 4.4]))
    traj = simulate(rs, t_max=3.0)
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    result = score_half_ball(traj, window)
    assert result.score > 0.5


def test_score_half_high_ball_velocity_scores_low():
    # A textbook HIGH ball should score LOW against the HALF criteria because it
    # descends too fast (best -2.5 m/s, band 1.5). This is the release that
    # test_score_good_high_ball_above_threshold scores > 0.5 as a high ball.
    # The previous release (-4.8, -0.5, 5.5) never entered the window, so the
    # velocity penalty was never exercised.
    rs = _set_release(velocity=np.array([-5.0, -0.7, 5.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    result = score_half_ball(traj, window)
    assert result.contact_index >= 0
    assert result.velocity_at_contact_z < -4.0
    assert result.velocity_score == 0.0
    assert result.score < 0.5


def test_score_half_contact_is_midpoint_of_descent_segment():
    # HalfBallScore documents the contact as the midpoint of the in-window
    # descent segment; velocity_score is evaluated there.
    rs = _set_release(velocity=np.array([-7.0, -0.75, 4.4]))
    traj = simulate(rs, t_max=3.0)
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    result = score_half_ball(traj, window)
    inside = window.contains_batch(traj.position)
    segment = np.where(inside & (np.arange(len(traj)) > traj.apex_index()))[0]
    assert segment.size >= 3
    assert result.contact_index == int(segment[segment.size // 2])
    assert result.velocity_at_contact_z == float(traj.velocity[result.contact_index, 2])


def test_score_half_is_in_unit_interval():
    # Uses a release that enters the window; (-6.5, -0.5, 4.5) missed it and
    # only checked the all-zero result.
    rs = _set_release(velocity=np.array([-7.0, -0.75, 4.4]))
    traj = simulate(rs, t_max=3.0)
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    result = score_half_ball(traj, window)
    assert result.contact_index >= 0
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.dwell_score <= 1.0
    assert 0.0 <= result.velocity_score <= 1.0
    assert 0.0 <= result.lateral_score <= 1.0


def test_score_half_returns_diagnostics():
    rs = _set_release(velocity=np.array([-7.0, -0.75, 4.4]))
    traj = simulate(rs, t_max=3.0)
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    result = score_half_ball(traj, window)
    assert result.contact_index >= 0
    assert isinstance(result.score, float)
    assert isinstance(result.dwell_seconds, float)
    assert isinstance(result.velocity_at_contact_z, float)
    assert isinstance(result.lateral_drift_m, float)
    assert isinstance(result.contact_index, int)
