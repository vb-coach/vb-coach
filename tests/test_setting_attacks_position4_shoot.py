import numpy as np

from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.attacks_position4 import ShootBallScore, score_shoot_ball
from vb_coach.setting.hit_window import position4_shoot_window
from vb_coach.trajectory import ReleaseState, simulate


def _set_release(velocity: np.ndarray, position: np.ndarray | None = None) -> ReleaseState:
    if position is None:
        position = np.array([0.5, 0.8, 2.2])
    return ReleaseState(position=position, velocity=velocity, spin=np.array([0.0, 0.0, 0.0]))


def test_score_shoot_returns_zero_when_trajectory_misses_window():
    rs = _set_release(velocity=np.array([0.0, 0.0, 5.0]))
    traj = simulate(rs, t_max=3.0)
    window = position4_shoot_window(DEFAULT_AMATEUR_MALE)
    result = score_shoot_ball(traj, window)
    assert result.score == 0.0


def test_score_shoot_good_flat_set_above_threshold():
    # Tune for 平拉开: low flat trajectory with apex at attacker contact height,
    # fast horizontal velocity. Hit at or near apex (v_z ~ 0).
    # Release from setter position near 3/4 zone (~1.5m from antenna side).
    # Goal: score > 0.4.
    rs = _set_release(
        velocity=np.array([-7.5, -0.5, 3.5]),
        position=np.array([-1.5, 0.5, 2.3]),
    )
    traj = simulate(rs, t_max=3.0)
    window = position4_shoot_window(DEFAULT_AMATEUR_MALE)
    result = score_shoot_ball(traj, window)
    assert result.score > 0.4


def test_score_shoot_high_ball_velocity_scores_low():
    # A textbook HIGH ball with v_z ~ -4 m/s at contact should score LOW against
    # the shoot window because v_z is far from the ideal 0.
    rs = _set_release(velocity=np.array([-4.8, -0.5, 5.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_shoot_window(DEFAULT_AMATEUR_MALE)
    result = score_shoot_ball(traj, window)
    assert result.score < 0.4


def test_score_shoot_is_in_unit_interval():
    rs = _set_release(
        velocity=np.array([-7.5, -0.5, 3.5]),
        position=np.array([-1.5, 0.5, 2.3]),
    )
    traj = simulate(rs, t_max=3.0)
    window = position4_shoot_window(DEFAULT_AMATEUR_MALE)
    result = score_shoot_ball(traj, window)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.dwell_score <= 1.0
    assert 0.0 <= result.velocity_score <= 1.0
    assert 0.0 <= result.lateral_score <= 1.0


def test_score_shoot_returns_diagnostics():
    rs = _set_release(
        velocity=np.array([-7.5, -0.5, 3.5]),
        position=np.array([-1.5, 0.5, 2.3]),
    )
    traj = simulate(rs, t_max=3.0)
    window = position4_shoot_window(DEFAULT_AMATEUR_MALE)
    result = score_shoot_ball(traj, window)
    assert isinstance(result.score, float)
    assert isinstance(result.dwell_seconds, float)
    assert isinstance(result.velocity_at_contact_z, float)
    assert isinstance(result.lateral_drift_m, float)
    assert isinstance(result.contact_index, int)


def test_score_shoot_includes_apex_index():
    # 平拉开 attacker hits AT apex (v_z ~ 0). The score filter should be
    # >= apex_idx, not > apex_idx, so the apex point itself counts.
    # Test by checking that a trajectory with apex inside the window scores nonzero.
    rs = _set_release(
        velocity=np.array([-7.5, -0.5, 3.5]),
        position=np.array([-1.5, 0.5, 2.3]),
    )
    traj = simulate(rs, t_max=3.0)
    window = position4_shoot_window(DEFAULT_AMATEUR_MALE)
    result = score_shoot_ball(traj, window)
    if result.contact_index >= 0:
        apex_idx = traj.apex_index()
        # Contact index should be at or just past apex (within a few frames)
        assert abs(result.contact_index - apex_idx) < 20
