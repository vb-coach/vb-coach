import numpy as np

from vb_coach.coords import mirror_across_net, on_team_a, on_team_b


def test_on_team_a_positive_y():
    assert on_team_a(np.array([0.0, 2.0, 1.0]))
    assert not on_team_a(np.array([0.0, -2.0, 1.0]))


def test_on_team_b_negative_y():
    assert on_team_b(np.array([0.0, -2.0, 1.0]))
    assert not on_team_b(np.array([0.0, 2.0, 1.0]))


def test_mirror_across_net_flips_y():
    p = np.array([1.5, 3.0, 2.4])
    m = mirror_across_net(p)
    np.testing.assert_array_equal(m, np.array([1.5, -3.0, 2.4]))


def test_mirror_is_involution():
    p = np.array([0.7, 4.0, 2.1])
    np.testing.assert_array_equal(mirror_across_net(mirror_across_net(p)), p)
