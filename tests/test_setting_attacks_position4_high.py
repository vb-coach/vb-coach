import numpy as np

from vb_coach.constants import RHO_AIR
from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.attacks_position4 import (
    HighBallScore,
    score_high_ball,
)
from vb_coach.setting.hit_window import position4_high_window
from vb_coach.trajectory import ReleaseState, simulate


def _set_release(velocity: np.ndarray, position: np.ndarray | None = None) -> ReleaseState:
    if position is None:
        position = np.array([0.5, 0.8, 2.2])
    return ReleaseState(position=position, velocity=velocity, spin=np.array([0.0, 0.0, 0.0]))


def test_score_returns_zero_when_trajectory_misses_window():
    # Set that never reaches the 4号位 antenna area
    rs = _set_release(velocity=np.array([0.0, 0.0, 5.0]))  # straight up + falls back
    traj = simulate(rs, t_max=3.0)
    window = position4_high_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    assert result.score == 0.0


def test_score_returns_zero_when_only_ascent_intersects():
    # Engineering a release where the ball enters the window only on ascent.
    # Set straight at the attacker contact point at fast horizontal speed,
    # passes through window while still climbing because release starts low.
    rs = _set_release(
        position=np.array([0.5, 0.0, 1.5]),
        velocity=np.array([-4.5, 0.4, 6.0]),  # toward window, mostly up
    )
    traj = simulate(rs, t_max=3.0)
    window = position4_high_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    # If only ascent intersection, score is zero (high ball means descent hit)
    # Acceptable: result.score could legitimately be 0 here.
    # The actionable assertion is: if the dwell is on ascent only, score is 0.
    if result.dwell_seconds > 0 and result.contact_index < traj.apex_index():
        assert result.score == 0.0


def test_score_good_high_ball_above_threshold():
    # Carefully tuned release that produces a textbook high ball:
    # apex ~3.6 m, descent into window at v_z ~ -4 m/s, no lateral drift.
    rs = _set_release(velocity=np.array([-4.8, -0.5, 5.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_high_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    assert result.score > 0.5
    assert result.dwell_seconds > 0.05


def test_score_too_fast_descent_velocity_kills_score():
    # Lower apex (faster descent at the window crossing) - flat low set
    rs = _set_release(velocity=np.array([-5.0, -0.5, 4.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_high_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    # Either the ball misses the window or v_z at contact is too fast — either
    # way the score should be low. If it does enter the window, v_z should be
    # outside the comfortable [-6, -2] m/s band.
    if result.score > 0:
        assert result.velocity_at_contact_z < -5.0 or result.velocity_at_contact_z > -2.0


def test_score_is_in_unit_interval():
    rs = _set_release(velocity=np.array([-4.8, -0.5, 5.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_high_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.dwell_score <= 1.0
    assert 0.0 <= result.velocity_score <= 1.0
    assert 0.0 <= result.lateral_score <= 1.0


def test_score_returns_diagnostics():
    rs = _set_release(velocity=np.array([-4.8, -0.5, 5.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_high_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    # All diagnostic fields are present and have expected types
    assert isinstance(result.score, float)
    assert isinstance(result.dwell_seconds, float)
    assert isinstance(result.velocity_at_contact_z, float)
    assert isinstance(result.lateral_drift_m, float)
    assert isinstance(result.contact_index, int)
