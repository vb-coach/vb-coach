import numpy as np

from vb_coach.constants import RHO_AIR
from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.attacks_position4 import (
    HighBallScore,
    score_high_ball,
)
from vb_coach.setting.hit_window import position4_hit_window
from vb_coach.trajectory import ReleaseState, simulate


def _set_release(velocity: np.ndarray, position: np.ndarray | None = None) -> ReleaseState:
    if position is None:
        position = np.array([0.5, 0.8, 2.2])
    return ReleaseState(position=position, velocity=velocity, spin=np.array([0.0, 0.0, 0.0]))


def test_score_returns_zero_when_trajectory_misses_window():
    # Set that never reaches the 4号位 antenna area
    rs = _set_release(velocity=np.array([0.0, 0.0, 5.0]))  # straight up + falls back
    traj = simulate(rs, t_max=3.0)
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    assert result.score == 0.0


def test_score_returns_zero_when_only_ascent_intersects():
    # Set passing through the window on its ascent (high horizontal speed
    # toward the attacker contact point, released near floor). The implementation
    # returns score=0 because high-ball scoring only considers post-apex dwell.
    rs = _set_release(
        position=np.array([0.5, 0.0, 1.5]),
        velocity=np.array([-4.5, 0.4, 6.0]),
    )
    traj = simulate(rs, t_max=3.0)
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    assert result.score == 0.0


def test_score_good_high_ball_above_threshold():
    # Threshold relaxed 0.5 -> 0.45 after contact_loss recalibration
    # (0.15 -> 0.20, Sheppard 2008) lowered the window center 5 cm. The
    # same release now lands with the ball passing through a tighter
    # part of the ellipsoid (dwell ~0.05 s vs 0.06 s prior).
    rs = _set_release(velocity=np.array([-4.8, -0.6, 5.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    assert result.score > 0.45
    assert result.dwell_seconds > 0.04


def test_score_low_apex_set_scores_low():
    # Low-apex set (lower z velocity → faster descent through the window).
    # Tune to ensure the trajectory does enter the window during descent.
    rs = _set_release(velocity=np.array([-4.5, -0.5, 4.0]))
    traj = simulate(rs, t_max=3.0)
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    # The score should be substantially lower than a textbook high ball.
    # If the ball misses the window entirely, score=0 (still low).
    assert result.score < 0.5


def test_score_is_in_unit_interval():
    rs = _set_release(velocity=np.array([-4.8, -0.5, 5.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.dwell_score <= 1.0
    assert 0.0 <= result.velocity_score <= 1.0
    assert 0.0 <= result.lateral_score <= 1.0


def test_score_returns_diagnostics():
    rs = _set_release(velocity=np.array([-4.8, -0.5, 5.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_hit_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    # All diagnostic fields are present and have expected types
    assert isinstance(result.score, float)
    assert isinstance(result.dwell_seconds, float)
    assert isinstance(result.velocity_at_contact_z, float)
    assert isinstance(result.lateral_drift_m, float)
    assert isinstance(result.contact_index, int)
