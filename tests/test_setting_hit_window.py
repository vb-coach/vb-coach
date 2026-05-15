import math

import numpy as np
import pytest

from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.hit_window import (
    HitWindow,
    position2_hit_window,
    position4_hit_window,
)


def test_hit_window_contains_center():
    w = HitWindow(
        center=np.array([0.0, 0.0, 2.8]),
        r_x=0.40, r_y_front=0.30, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
    )
    assert w.contains(np.array([0.0, 0.0, 2.8]))


def test_hit_window_excludes_far_outside():
    w = HitWindow(
        center=np.array([0.0, 0.0, 2.8]),
        r_x=0.40, r_y_front=0.30, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
    )
    assert not w.contains(np.array([1.0, 0.0, 2.8]))
    assert not w.contains(np.array([0.0, 1.0, 2.8]))
    assert not w.contains(np.array([0.0, 0.0, 4.0]))


def test_hit_window_front_asymmetry():
    # r_y_front = 0.30 (forward = -y direction from center)
    # r_y_back  = 0.15 (backward = +y direction)
    w = HitWindow(
        center=np.array([0.0, 0.0, 2.8]),
        r_x=0.40, r_y_front=0.30, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
    )
    assert w.contains(np.array([0.0, -0.25, 2.8]))      # 0.25 m forward → inside
    assert not w.contains(np.array([0.0, +0.20, 2.8]))  # 0.20 m back → outside (>0.15)


def test_hit_window_vertical_asymmetry():
    w = HitWindow(
        center=np.array([0.0, 0.0, 2.8]),
        r_x=0.40, r_y_front=0.30, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
    )
    assert w.contains(np.array([0.0, 0.0, 2.60]))       # 0.20 m below → inside
    assert not w.contains(np.array([0.0, 0.0, 2.90]))   # 0.10 m above → outside


def test_hit_window_contains_batch():
    w = HitWindow(
        center=np.array([0.0, 0.0, 2.8]),
        r_x=0.40, r_y_front=0.30, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
    )
    points = np.array([
        [0.0, 0.0, 2.8],     # inside (center)
        [0.0, -0.25, 2.8],   # inside (forward)
        [0.0, +0.20, 2.8],   # outside (too far back)
        [0.0, 0.0, 2.90],    # outside (above max reach)
        [0.0, 0.0, 2.60],    # inside (below center)
    ])
    mask = w.contains_batch(points)
    np.testing.assert_array_equal(mask, np.array([True, True, False, False, True]))


def test_hit_window_rejects_negative_radii():
    with pytest.raises(ValueError):
        HitWindow(
            center=np.array([0.0, 0.0, 2.8]),
            r_x=-0.1, r_y_front=0.3, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
        )


def test_hit_window_rejects_wrong_center_shape():
    with pytest.raises(ValueError):
        HitWindow(
            center=np.array([0.0, 2.8]),
            r_x=0.40, r_y_front=0.30, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
        )


def test_position4_hit_window_centered_at_contact_height():
    a = DEFAULT_AMATEUR_MALE
    w = position4_hit_window(a)
    assert math.isclose(w.center[2], a.contact_height, abs_tol=1e-9)


def test_position4_hit_window_at_antenna_position():
    a = DEFAULT_AMATEUR_MALE
    w = position4_hit_window(a)
    assert -4.5 < w.center[0] < -3.5
    assert 0.0 < w.center[1] < 0.5


def test_position4_hit_window_amateur_radii_match_literature():
    a = DEFAULT_AMATEUR_MALE
    w = position4_hit_window(a)
    assert w.r_y_front > w.r_y_back     # Wagner + coaching front-bias
    assert w.r_z_down > w.r_z_up        # max-reach upper bound
    assert 0.25 <= w.r_x <= 0.50        # lateral with footwork tolerance
    assert w.r_z_up <= 0.10             # tight above-reach


def test_position4_hit_window_taller_attacker_higher_center():
    from vb_coach.setting.attacker import Attacker
    short_a = Attacker(height=1.65, standing_reach=2.18, approach_reach=2.55)
    tall_a = Attacker(height=1.95, standing_reach=2.55, approach_reach=3.20)
    short_w = position4_hit_window(short_a)
    tall_w = position4_hit_window(tall_a)
    assert tall_w.center[2] > short_w.center[2]
    # Same level defaults: r_x identical
    assert short_w.r_x == tall_w.r_x
    assert short_w.r_y_front == tall_w.r_y_front


def test_position4_hit_window_elite_has_tighter_r_x_than_amateur():
    # Footwork tolerance: amateur +20 cm, college +10, elite +5
    from vb_coach.setting.attacker import Attacker
    amateur = Attacker(height=1.80, standing_reach=2.34, approach_reach=2.95)
    college = Attacker(height=1.80, standing_reach=2.34, approach_reach=2.95, level="college")
    elite   = Attacker(height=1.80, standing_reach=2.34, approach_reach=2.95, level="elite")
    w_amateur = position4_hit_window(amateur)
    w_college = position4_hit_window(college)
    w_elite   = position4_hit_window(elite)
    assert math.isclose(w_amateur.r_x, 0.40)
    assert math.isclose(w_college.r_x, 0.30)
    assert math.isclose(w_elite.r_x,   0.25)
    # Other radii unchanged across levels (footwork only affects along-net)
    assert w_amateur.r_y_front == w_elite.r_y_front
    assert w_amateur.r_z_up == w_elite.r_z_up


def test_position2_hit_window_mirrors_position4():
    # 2号位 is the right-pin mirror of 4号位 across the net center.
    a = DEFAULT_AMATEUR_MALE
    w4 = position4_hit_window(a)
    w2 = position2_hit_window(a)
    assert math.isclose(w2.center[0], -w4.center[0])      # mirror x
    assert math.isclose(w2.center[1],  w4.center[1])      # same y
    assert math.isclose(w2.center[2],  w4.center[2])      # same z
    assert w2.r_x        == w4.r_x
    assert w2.r_y_front  == w4.r_y_front
    assert w2.r_y_back   == w4.r_y_back
    assert w2.r_z_up     == w4.r_z_up
    assert w2.r_z_down   == w4.r_z_down


def test_position2_hit_window_at_right_antenna():
    a = DEFAULT_AMATEUR_MALE
    w = position2_hit_window(a)
    assert 3.5 < w.center[0] < 4.5    # right side of net (positive x)
