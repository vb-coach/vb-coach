import numpy as np
import pytest

from vb_coach.trajectory import ReleaseState, Trajectory


def test_release_state_construction():
    rs = ReleaseState(
        position=np.array([0.0, 0.8, 2.2]),
        velocity=np.array([0.0, -3.0, 6.0]),
        spin=np.array([0.0, 10.0, 0.0]),
    )
    assert rs.position.shape == (3,)
    assert rs.velocity.shape == (3,)
    assert rs.spin.shape == (3,)


def test_release_state_rejects_wrong_shape():
    with pytest.raises(ValueError):
        ReleaseState(
            position=np.array([0.0, 0.8]),
            velocity=np.array([0.0, -3.0, 6.0]),
            spin=np.array([0.0, 10.0, 0.0]),
        )


def test_trajectory_consistent_lengths():
    t = np.linspace(0, 1, 11)
    p = np.zeros((11, 3))
    v = np.zeros((11, 3))
    s = np.zeros((11, 3))
    traj = Trajectory(time=t, position=p, velocity=v, spin=s)
    assert len(traj) == 11


def test_trajectory_rejects_inconsistent_lengths():
    t = np.linspace(0, 1, 11)
    p = np.zeros((10, 3))
    v = np.zeros((11, 3))
    s = np.zeros((11, 3))
    with pytest.raises(ValueError):
        Trajectory(time=t, position=p, velocity=v, spin=s)


def test_trajectory_apex_index():
    t = np.linspace(0, 1, 101)
    z = 1 - (t - 0.5) ** 2 * 4
    p = np.column_stack([np.zeros_like(t), np.zeros_like(t), z])
    v = np.zeros_like(p)
    s = np.zeros_like(p)
    traj = Trajectory(time=t, position=p, velocity=v, spin=s)
    assert abs(traj.time[traj.apex_index()] - 0.5) < 0.02
