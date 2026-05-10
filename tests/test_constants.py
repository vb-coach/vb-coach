import vb_coach.constants as C


def test_gravity_magnitude():
    assert C.G == 9.81


def test_air_density_room_temp():
    assert 1.15 < C.RHO_AIR < 1.25


def test_air_kinematic_viscosity():
    assert 1.4e-5 < C.NU_AIR < 1.6e-5


def test_ball_diameter_volleyball():
    assert 0.205 < C.BALL_DIAMETER < 0.215


def test_ball_mass_volleyball():
    assert 0.255 < C.BALL_MASS < 0.285


def test_net_heights():
    assert C.NET_HEIGHT_MEN == 2.43
    assert C.NET_HEIGHT_WOMEN == 2.24
