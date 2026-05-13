import math

import pytest

from vb_coach.setting.attacker import Attacker, DEFAULT_AMATEUR_MALE


def test_attacker_construction_with_explicit_values():
    a = Attacker(height=1.80, standing_reach=2.34, approach_reach=2.95)
    assert a.height == 1.80
    assert a.standing_reach == 2.34
    assert a.approach_reach == 2.95


def test_attacker_contact_height_uses_default_loss():
    # contact_height = approach_reach - contact_loss (default 0.20)
    a = Attacker(height=1.80, standing_reach=2.34, approach_reach=2.95)
    assert math.isclose(a.contact_height, 2.75, abs_tol=1e-9)


def test_attacker_contact_height_with_custom_loss():
    a = Attacker(
        height=1.80,
        standing_reach=2.34,
        approach_reach=2.95,
        contact_loss=0.25,
    )
    assert math.isclose(a.contact_height, 2.70, abs_tol=1e-9)


def test_attacker_rejects_implausible_proportions():
    # standing_reach should exceed height (arms-up always reaches above head)
    with pytest.raises(ValueError):
        Attacker(height=1.80, standing_reach=1.60, approach_reach=2.50)


def test_attacker_rejects_approach_below_standing():
    # Approach reach must be >= standing reach (you don't jump downward)
    with pytest.raises(ValueError):
        Attacker(height=1.80, standing_reach=2.34, approach_reach=2.30)


def test_default_amateur_male_contact_in_data_band():
    # Per project amateur-reach reference: trained recreational male
    # contact_height median ~ 2.95 m (mean from VolleyTalk dataset).
    # Default value should fall within the trained-amateur band 2.71-3.19 m.
    a = DEFAULT_AMATEUR_MALE
    assert 2.71 <= a.contact_height <= 3.19
