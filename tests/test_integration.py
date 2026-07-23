"""End-to-end physical sanity tests for the trajectory pipeline."""

import numpy as np

from vb_coach.aerodynamics import reynolds
from vb_coach.constants import BALL_DIAMETER, NET_HEIGHT_MEN, NU_AIR, RHO_AIR
from vb_coach.serving import CANONICAL_FLOAT_SERVES
from vb_coach.trajectory import ReleaseState, simulate, simulate_stochastic


def test_reynolds_for_typical_set_is_subcritical():
    # Set speed ~6 m/s, Re ~84k -> well below the drag crisis (~250k)
    re = reynolds(6.0, BALL_DIAMETER, NU_AIR)
    assert re < 200_000


def test_high_ball_to_position_4_reaches_realistic_apex():
    # Setter at (0.0, 0.8, 2.2); high ball toward 4号位.
    # Tuned to produce a typical men's high-ball arc.
    rs = ReleaseState(
        position=np.array([0.0, 0.8, 2.2]),
        velocity=np.array([-3.0, -1.0, 7.5]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj = simulate(rs, t_max=3.0)
    apex_z = traj.apex_height()
    # Realistic high-ball apex band
    assert 4.5 < apex_z < 6.5
    # Total flight time should be 1.0-3.0 s
    assert 1.0 < traj.time[-1] < 3.0


def test_drag_reduces_apex_compared_to_vacuum():
    rs = ReleaseState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.array([0.0, 0.0, 7.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj_air = simulate(rs, t_max=2.5, rho=RHO_AIR)
    traj_vac = simulate(rs, t_max=2.5, rho=0.0)
    # Vacuum apex from analytic: z0 + v0^2/(2g) = 2 + 49/19.62 = 4.497
    assert traj_vac.apex_height() > 4.4
    assert traj_air.apex_height() < traj_vac.apex_height()


def test_top_spin_drops_ball_earlier_than_no_spin():
    # Top-spin on a ball moving +y: top of ball moves +y faster than translation.
    # By right-hand rule, omega in -x. Magnus: omega_hat x v_hat = (-x) x (+y) = -z (down).
    base = ReleaseState(
        position=np.array([0.0, 0.0, 2.2]),
        velocity=np.array([0.0, 5.0, 5.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    top_spin = ReleaseState(
        position=base.position.copy(),
        velocity=base.velocity.copy(),
        spin=np.array([-30.0, 0.0, 0.0]),
    )
    traj_base = simulate(base, t_max=3.0)
    traj_top = simulate(top_spin, t_max=3.0)
    # Top-spin should land sooner and shorter (smaller y at landing)
    assert traj_top.time[-1] < traj_base.time[-1]
    assert traj_top.position[-1, 1] < traj_base.position[-1, 1]


def test_canonical_float_serves_clear_net_and_land_in_bounds():
    for preset in CANONICAL_FLOAT_SERVES:
        traj = simulate_stochastic(
            preset.release_state(),
            rng=np.random.default_rng(preset.seed),
            t_max=2.0,
            dt=0.005,
        )
        net_idx = int(np.argmin(np.abs(traj.position[:, 1])))
        landing = traj.position[-1]
        assert traj.position[net_idx, 2] > NET_HEIGHT_MEN
        assert landing[2] <= 0.05
        assert abs(landing[0]) <= 4.5
        assert 0.0 <= landing[1] <= 9.0
