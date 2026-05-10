"""Drag coefficient and Reynolds number for a volleyball."""

from __future__ import annotations

import math


def reynolds(velocity_magnitude: float, diameter: float, kinematic_viscosity: float) -> float:
    return velocity_magnitude * diameter / kinematic_viscosity


def cd_subcritical(reynolds_number: float) -> float:
    """Constant Cd ~ 0.45 across the subcritical band (Re < ~1.5e5).

    Kept as a function for analytical clarity and historical compatibility;
    new code should call cd_full_regime, which agrees with this in the
    subcritical band and additionally covers the drag crisis.
    """
    return 0.45


_RE_CRIT_LOW = 1.5e5
_RE_CRIT_HIGH = 2.5e5
_CD_SUBCRITICAL = 0.45
_CD_SUPERCRITICAL = 0.18


def cd_full_regime(reynolds_number: float) -> float:
    """Piecewise Cd(Re) for a FIVB-spec volleyball spanning the drag crisis.

    Tuned against Asai et al. 2010 wind-tunnel data (Mikasa MVA200):
        Re <= 1.5e5     : Cd = 0.45                  (subcritical plateau)
        1.5e5 < Re < 2.5e5 : Cd 0.45 -> 0.18 via cosine taper  (drag crisis)
        Re >= 2.5e5     : Cd = 0.18                  (supercritical plateau)
    The cosine transition avoids integrator stiffness at the band edges.
    """
    if reynolds_number <= _RE_CRIT_LOW:
        return _CD_SUBCRITICAL
    if reynolds_number >= _RE_CRIT_HIGH:
        return _CD_SUPERCRITICAL
    phase = math.pi * (reynolds_number - _RE_CRIT_LOW) / (_RE_CRIT_HIGH - _RE_CRIT_LOW)
    return _CD_SUPERCRITICAL + 0.5 * (_CD_SUBCRITICAL - _CD_SUPERCRITICAL) * (1 + math.cos(phase))
