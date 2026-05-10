import math

from vb_coach.aerodynamics import reynolds, spin_parameter, cd_subcritical, cl_magnus
from vb_coach.constants import BALL_DIAMETER, BALL_RADIUS, NU_AIR


def test_reynolds_typical_set_speed():
    re = reynolds(6.0, BALL_DIAMETER, NU_AIR)
    assert 80_000 < re < 90_000


def test_reynolds_zero_velocity_is_zero():
    assert reynolds(0.0, BALL_DIAMETER, NU_AIR) == 0.0


def test_spin_parameter_zero_spin():
    assert spin_parameter(0.0, 6.0, BALL_RADIUS) == 0.0


def test_spin_parameter_known_value():
    s = spin_parameter(31.4, 6.0, BALL_RADIUS)
    assert math.isclose(s, 0.5495, abs_tol=1e-3)


def test_spin_parameter_zero_velocity_returns_zero():
    assert spin_parameter(10.0, 0.0, BALL_RADIUS) == 0.0


def test_cd_subcritical_constant_placeholder():
    assert cd_subcritical(50_000) == 0.45
    assert cd_subcritical(100_000) == 0.45


def test_cl_magnus_zero_spin_zero_lift():
    assert cl_magnus(0.0) == 0.0


def test_cl_magnus_saturates_at_high_spin():
    assert math.isclose(cl_magnus(0.5), 0.2, abs_tol=0.01)
    assert math.isclose(cl_magnus(1.0), 0.2, abs_tol=0.01)


def test_cl_magnus_linear_below_saturation():
    assert math.isclose(cl_magnus(0.25), 0.1, abs_tol=0.01)
