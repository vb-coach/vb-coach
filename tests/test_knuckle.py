import math

import numpy as np
import pytest

from vb_coach.aerodynamics.knuckle import (
    TAU_KNUCKLE,
    OUState,
    cl_lat_rms,
    step_ou,
)


def test_cl_lat_rms_peak_at_2e5():
    # Gaussian peak at Re = 2.0e5; max value should be the documented 0.12
    cl_peak = cl_lat_rms(2.0e5)
    assert math.isclose(cl_peak, 0.12, abs_tol=1e-6)


def test_cl_lat_rms_decays_far_from_peak():
    # Far below and far above the critical band: Cl_lat -> ~0
    assert cl_lat_rms(50_000) < 0.01
    assert cl_lat_rms(500_000) < 0.01


def test_cl_lat_rms_symmetric_around_peak():
    # Gaussian shape: equal distance below and above peak should give equal Cl
    cl_low = cl_lat_rms(2.0e5 - 0.3e5)
    cl_high = cl_lat_rms(2.0e5 + 0.3e5)
    assert math.isclose(cl_low, cl_high, abs_tol=1e-6)


def test_ou_state_defaults_to_zero():
    ou = OUState()
    assert ou.chan_h == 0.0
    assert ou.chan_v == 0.0


def test_step_ou_zero_dt_keeps_state():
    # With dt = 0, decay = 1 and noise_scale = 0 -> state unchanged
    ou = OUState(chan_h=1.5, chan_v=-0.7)
    rng = np.random.default_rng(0)
    step_ou(ou, 0.0, rng)
    assert math.isclose(ou.chan_h, 1.5, abs_tol=1e-12)
    assert math.isclose(ou.chan_v, -0.7, abs_tol=1e-12)


def test_step_ou_large_dt_drives_toward_unit_variance():
    # After many steps with dt > tau, variance should approach 1
    rng = np.random.default_rng(42)
    samples = []
    ou = OUState()
    for _ in range(20_000):
        step_ou(ou, TAU_KNUCKLE, rng)
        samples.append(ou.chan_h)
    var = float(np.var(samples))
    assert 0.7 < var < 1.3
