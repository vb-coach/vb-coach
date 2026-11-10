import math

import numpy as np

from vb_coach.physics import gravity_force, drag_force, magnus_force, total_acceleration, knuckle_force
from vb_coach.constants import BALL_MASS, G, RHO_AIR, BALL_CROSS_SECTION
from vb_coach.aerodynamics.knuckle import OUState


def test_gravity_points_down_with_magnitude_mg():
    f = gravity_force(BALL_MASS)
    np.testing.assert_array_almost_equal(f, np.array([0.0, 0.0, -BALL_MASS * G]))


def test_drag_force_zero_at_rest():
    v = np.array([0.0, 0.0, 0.0])
    f = drag_force(v, rho=RHO_AIR, area=BALL_CROSS_SECTION, cd=0.45)
    np.testing.assert_array_almost_equal(f, np.zeros(3))


def test_drag_force_opposes_velocity():
    v = np.array([5.0, 0.0, 0.0])
    f = drag_force(v, rho=RHO_AIR, area=BALL_CROSS_SECTION, cd=0.45)
    assert f[0] < 0
    assert math.isclose(f[1], 0.0, abs_tol=1e-12)
    assert math.isclose(f[2], 0.0, abs_tol=1e-12)


def test_drag_force_magnitude():
    v = np.array([6.0, 0.0, 0.0])
    f = drag_force(v, rho=RHO_AIR, area=BALL_CROSS_SECTION, cd=0.45)
    assert math.isclose(np.linalg.norm(f), 0.336, abs_tol=0.01)


def test_magnus_zero_when_no_spin():
    v = np.array([6.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 0.0])
    f = magnus_force(v, omega, rho=RHO_AIR, area=BALL_CROSS_SECTION, radius=0.105)
    np.testing.assert_array_almost_equal(f, np.zeros(3))


def test_magnus_zero_when_no_velocity():
    v = np.array([0.0, 0.0, 0.0])
    omega = np.array([0.0, 10.0, 0.0])
    f = magnus_force(v, omega, rho=RHO_AIR, area=BALL_CROSS_SECTION, radius=0.105)
    np.testing.assert_array_almost_equal(f, np.zeros(3))


def test_magnus_back_spin_lifts_ball():
    # Ball moves +x. Back-spin means the top of the ball moves opposite to
    # translation (-x), which by right-hand rule means omega points in -y.
    # omega_hat x v_hat = (-y) x (+x) = +z, so Magnus lift is in +z.
    v = np.array([6.0, 0.0, 0.0])
    omega = np.array([0.0, -31.4, 0.0])
    f = magnus_force(v, omega, rho=RHO_AIR, area=BALL_CROSS_SECTION, radius=0.105)
    assert f[2] > 0
    assert math.isclose(f[0], 0.0, abs_tol=1e-9)
    assert math.isclose(f[1], 0.0, abs_tol=1e-9)


def test_magnus_top_spin_pushes_down():
    # Top-spin: top of ball moves with translation (+x). omega in +y.
    # omega_hat x v_hat = (+y) x (+x) = -z.
    v = np.array([6.0, 0.0, 0.0])
    omega = np.array([0.0, 31.4, 0.0])
    f = magnus_force(v, omega, rho=RHO_AIR, area=BALL_CROSS_SECTION, radius=0.105)
    assert f[2] < 0


def test_total_acceleration_includes_gravity():
    v = np.array([0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 0.0])
    a = total_acceleration(v, omega, mass=BALL_MASS, rho=RHO_AIR,
                           area=BALL_CROSS_SECTION, radius=0.105)
    np.testing.assert_array_almost_equal(a, np.array([0.0, 0.0, -G]))


def test_knuckle_force_zero_at_low_speed():
    # Below the cutoff (0.1 m/s) the force is suppressed
    v = np.array([0.05, 0.0, 0.0])
    ou = OUState(chan_h=1.0, chan_v=1.0)
    f = knuckle_force(v, ou, rho=1.2, area=0.0346, re=2.0e5)
    np.testing.assert_array_almost_equal(f, np.zeros(3))


def test_knuckle_force_perpendicular_to_velocity():
    # h_hat and vert_hat are both perpendicular to v_hat by construction
    v = np.array([0.0, 15.0, 0.5])
    ou = OUState(chan_h=1.0, chan_v=1.0)
    f = knuckle_force(v, ou, rho=1.2, area=0.0346, re=2.0e5)
    dot = float(np.dot(f, v / np.linalg.norm(v)))
    assert abs(dot) < 1e-9


def test_knuckle_force_zero_when_ou_zero():
    v = np.array([0.0, 15.0, 0.5])
    ou = OUState(chan_h=0.0, chan_v=0.0)
    f = knuckle_force(v, ou, rho=1.2, area=0.0346, re=2.0e5)
    np.testing.assert_array_almost_equal(f, np.zeros(3))


def test_knuckle_force_zero_when_cl_lat_zero():
    # At Re far from the peak, cl_lat_rms ~ 0 -> force ~ 0
    v = np.array([0.0, 15.0, 0.5])
    ou = OUState(chan_h=1.0, chan_v=1.0)
    f = knuckle_force(v, ou, rho=1.2, area=0.0346, re=50_000)
    assert float(np.linalg.norm(f)) < 0.01


def test_total_acceleration_default_diameter_is_two_radii():
    # At 20 m/s Re sits in the drag crisis, so a wrong characteristic length
    # changes Cd; at rest it would not.
    from vb_coach.constants import BALL_DIAMETER, BALL_RADIUS

    v = np.array([20.0, 0.0, 0.0])
    args = (v, np.zeros(3), BALL_MASS, RHO_AIR, BALL_CROSS_SECTION, BALL_RADIUS)
    np.testing.assert_allclose(
        total_acceleration(*args), total_acceleration(*args, diameter=BALL_DIAMETER), rtol=1e-12
    )


def test_knuckle_force_magnitude_does_not_depend_on_pitch():
    # Steep flight: |v_xy| / |v| = 0.45. The lateral basis must stay unit length,
    # so |F| = 0.5 * rho * A * Cl * |v|^2 for a unit OU channel.
    from vb_coach.aerodynamics import reynolds
    from vb_coach.aerodynamics.knuckle import cl_lat_rms
    from vb_coach.constants import BALL_DIAMETER, NU_AIR

    v = np.array([0.0, 10.0, 20.0])
    speed = float(np.linalg.norm(v))
    re = reynolds(speed, BALL_DIAMETER, NU_AIR)
    f = knuckle_force(v, OUState(chan_h=1.0, chan_v=0.0), rho=RHO_AIR, area=BALL_CROSS_SECTION, re=re)
    expected = 0.5 * RHO_AIR * BALL_CROSS_SECTION * cl_lat_rms(re) * speed**2
    assert math.isclose(float(np.linalg.norm(f)), expected, rel_tol=1e-9)
