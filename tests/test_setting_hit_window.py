import math

import numpy as np
import pytest

from vb_coach.setting.attacker import Attacker, DEFAULT_AMATEUR_MALE
from vb_coach.setting.hit_window import HitWindow, position4_high_window


def test_hit_window_contains_center():
    w = HitWindow(center=np.array([0.0, 0.0, 2.8]), dx=0.5, dy=0.3, dz=0.2)
    assert w.contains(np.array([0.0, 0.0, 2.8]))


def test_hit_window_contains_near_edge():
    w = HitWindow(center=np.array([0.0, 0.0, 2.8]), dx=0.5, dy=0.3, dz=0.2)
    assert w.contains(np.array([0.49, 0.29, 2.99]))
    assert w.contains(np.array([-0.49, -0.29, 2.61]))


def test_hit_window_excludes_outside():
    w = HitWindow(center=np.array([0.0, 0.0, 2.8]), dx=0.5, dy=0.3, dz=0.2)
    assert not w.contains(np.array([0.51, 0.0, 2.8]))   # outside x
    assert not w.contains(np.array([0.0, 0.31, 2.8]))   # outside y
    assert not w.contains(np.array([0.0, 0.0, 3.01]))   # outside z


def test_hit_window_contains_batch():
    w = HitWindow(center=np.array([0.0, 0.0, 2.8]), dx=0.5, dy=0.3, dz=0.2)
    points = np.array([
        [0.0, 0.0, 2.8],     # inside
        [0.6, 0.0, 2.8],     # outside x
        [0.0, 0.0, 3.5],     # outside z
        [0.3, 0.2, 2.7],     # inside
    ])
    mask = w.contains_batch(points)
    np.testing.assert_array_equal(mask, np.array([True, False, False, True]))


def test_position4_high_window_centered_at_attacker_contact_height():
    a = DEFAULT_AMATEUR_MALE
    w = position4_high_window(a)
    # Center z should equal the attacker's contact height (2.95 - 0.15 = 2.80)
    assert math.isclose(w.center[2], a.contact_height, abs_tol=1e-9)


def test_position4_high_window_near_left_antenna():
    a = DEFAULT_AMATEUR_MALE
    w = position4_high_window(a)
    # 4号位 attack lands near x = -4.0 m
    assert -4.5 < w.center[0] < -3.5
    # Just on team A's side of the net (small positive y)
    assert 0.0 < w.center[1] < 0.6


def test_position4_high_window_dimensions_reasonable_for_amateur():
    a = DEFAULT_AMATEUR_MALE
    w = position4_high_window(a)
    # Amateur swing arc and timing tolerance — 0.3-0.7 m in lateral, 0.15-0.3 m vertical
    assert 0.3 < w.dx < 0.8
    assert 0.2 < w.dy < 0.5
    assert 0.15 < w.dz < 0.35
