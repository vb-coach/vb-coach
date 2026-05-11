import numpy as np

from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.attacks_position4 import HalfBallScore, score_half_ball
from vb_coach.setting.hit_window import position4_half_window
from vb_coach.trajectory import ReleaseState, simulate


def _set_release(velocity: np.ndarray, position: np.ndarray | None = None) -> ReleaseState:
    if position is None:
        position = np.array([0.5, 0.8, 2.2])
    return ReleaseState(position=position, velocity=velocity, spin=np.array([0.0, 0.0, 0.0]))


def test_score_half_returns_zero_when_trajectory_misses_window():
    rs = _set_release(velocity=np.array([0.0, 0.0, 5.0]))
    traj = simulate(rs, t_max=3.0)
    window = position4_half_window(DEFAULT_AMATEUR_MALE)
    result = score_half_ball(traj, window)
    assert result.score == 0.0


def test_score_half_returns_zero_when_only_ascent_intersects():
    rs = _set_release(
        position=np.array([0.5, 0.0, 1.5]),
        velocity=np.array([-4.5, 0.4, 6.0]),
    )
    traj = simulate(rs, t_max=3.0)
    window = position4_half_window(DEFAULT_AMATEUR_MALE)
    result = score_half_ball(traj, window)
    assert result.score == 0.0


def test_score_half_good_semi_high_above_threshold():
    # Tune for 半高: lower apex (~3-4m), hit during descent at v_z ~ -2.5 m/s.
    # The implementer should retune this velocity if the first guess doesn't
    # produce a clear-win scenario, just like Task A1.1 retuned the high-ball
    # velocity. Goal: score > 0.5.
    rs = _set_release(velocity=np.array([-6.5, -0.5, 4.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_half_window(DEFAULT_AMATEUR_MALE)
    result = score_half_ball(traj, window)
    assert result.score > 0.5


def test_score_half_high_ball_velocity_scores_low():
    # A textbook HIGH ball (apex ~5m, descent v_z ~ -4 m/s) should score LOW
    # against the HALF window because the velocity is too fast for "half-high"
    # criteria (v_z = -4 vs -HALF_VELOCITY_BEST = -2.5, band = 1.5).
    rs = _set_release(velocity=np.array([-4.8, -0.5, 5.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_half_window(DEFAULT_AMATEUR_MALE)
    result = score_half_ball(traj, window)
    # Either misses the (smaller) window, or the v_z penalty pulls score low.
    assert result.score < 0.5


def test_score_half_is_in_unit_interval():
    rs = _set_release(velocity=np.array([-6.5, -0.5, 4.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_half_window(DEFAULT_AMATEUR_MALE)
    result = score_half_ball(traj, window)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.dwell_score <= 1.0
    assert 0.0 <= result.velocity_score <= 1.0
    assert 0.0 <= result.lateral_score <= 1.0


def test_score_half_returns_diagnostics():
    rs = _set_release(velocity=np.array([-6.5, -0.5, 4.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_half_window(DEFAULT_AMATEUR_MALE)
    result = score_half_ball(traj, window)
    assert isinstance(result.score, float)
    assert isinstance(result.dwell_seconds, float)
    assert isinstance(result.velocity_at_contact_z, float)
    assert isinstance(result.lateral_drift_m, float)
    assert isinstance(result.contact_index, int)
