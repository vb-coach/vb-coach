import numpy as np
import pytest

from vb_coach.trajectory import ReleaseState, Trajectory, simulate
from vb_coach.constants import RHO_AIR


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


def test_simulate_returns_trajectory():
    rs = ReleaseState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.array([0.0, 0.0, 5.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj = simulate(rs, t_max=2.0)
    assert isinstance(traj, Trajectory)
    assert len(traj) > 10


def test_simulate_stops_at_floor():
    rs = ReleaseState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.array([0.0, 0.0, 5.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj = simulate(rs, t_max=5.0)
    assert abs(traj.position[-1, 2]) < 0.01


def test_simulate_pure_vertical_in_vacuum_matches_analytic():
    # In vacuum (rho=0) and no spin: z(t) = z0 + v0*t - 0.5*g*t^2
    # Apex height = z0 + v0^2/(2g) = 2 + 25/19.62 = 3.274 m
    rs = ReleaseState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.array([0.0, 0.0, 5.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj = simulate(rs, t_max=2.0, rho=0.0)
    assert abs(traj.apex_height() - 3.274) < 0.01


def test_simulate_no_spin_no_curvature_in_horizontal_plane():
    rs = ReleaseState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.array([0.0, -3.0, 5.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj = simulate(rs, t_max=2.0)
    assert float(np.max(np.abs(traj.position[:, 0]))) < 1e-6


def test_simulate_back_spin_lifts_ball():
    # Ball travels in +y. Back-spin means top of ball moves in -y.
    # By right-hand rule, omega in +x.
    # Magnus: omega_hat x v_hat = (+x) x (+y) = +z (lift) -> ball flies farther.
    base = ReleaseState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.array([0.0, 5.0, 4.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    spun = ReleaseState(
        position=base.position.copy(),
        velocity=base.velocity.copy(),
        spin=np.array([30.0, 0.0, 0.0]),
    )
    traj_base = simulate(base, t_max=3.0)
    traj_spun = simulate(spun, t_max=3.0)
    assert traj_spun.position[-1, 1] > traj_base.position[-1, 1]
