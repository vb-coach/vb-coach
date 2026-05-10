import math

import numpy as np

from vb_coach.physics import gravity_force, drag_force, magnus_force, total_acceleration
from vb_coach.constants import BALL_MASS, G, RHO_AIR, BALL_CROSS_SECTION


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
