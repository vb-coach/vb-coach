"""Drag coefficient and Reynolds number for a volleyball.

Values calibrated against Hong et al. 2018 (ISBS) and Hong & Asai 2020
(Applied Sciences 10:3256) wind-tunnel measurements of FIVB match balls
(Mikasa MVA200/V200W, Molten Flistatec, Mizuno 18-panel). See
docs/research/calibration-priors.md for the full citation trail.
"""

from __future__ import annotations

import math


# Subcritical plateau: Hong 2018 fig 2 / Dumek 2018 fig 4 measure ~0.50-0.55
# for all panel designs at Re < 1.5e5.
_CD_SUBCRITICAL = 0.50

# Supercritical plateau: Hong & Asai 2020 give 0.15-0.17 for Mikasa V200W;
# 0.16 is the median across modern 8-panel and hex-panel balls.
_CD_SUPERCRITICAL = 0.16

# Drag crisis onset/end: Hong 2018 observes the transition between
# Re ~ 2.0e5 (onset for most panel designs) and ~3.3e5 (Mikasa MVA200 end).
# Panel design narrows / widens this band by ~15%; this is the median.
_RE_CRIT_LOW = 2.0e5
_RE_CRIT_HIGH = 3.3e5


def reynolds(velocity_magnitude: float, diameter: float, kinematic_viscosity: float) -> float:
    return velocity_magnitude * diameter / kinematic_viscosity


def cd_subcritical(reynolds_number: float) -> float:
    """Constant Cd ~ 0.50 across the subcritical band (Re < ~2.0e5).

    Kept as a function for analytical clarity and historical compatibility;
    new code should call cd_full_regime, which agrees with this in the
    subcritical band and additionally covers the drag crisis.
    """
    return _CD_SUBCRITICAL


def cd_full_regime(reynolds_number: float) -> float:
    """Piecewise Cd(Re) for a FIVB-spec volleyball spanning the drag crisis.

    Calibrated from Hong et al. 2018 / Hong & Asai 2020 wind-tunnel data:
        Re <= 2.0e5     : Cd = 0.50                  (subcritical plateau)
        2.0e5 < Re < 3.3e5 : Cd 0.50 -> 0.16 via cosine taper  (drag crisis)
        Re >= 3.3e5     : Cd = 0.16                  (supercritical plateau)
    The cosine transition avoids integrator stiffness at the band edges.
    """
    if reynolds_number <= _RE_CRIT_LOW:
        return _CD_SUBCRITICAL
    if reynolds_number >= _RE_CRIT_HIGH:
        return _CD_SUPERCRITICAL
    phase = math.pi * (reynolds_number - _RE_CRIT_LOW) / (_RE_CRIT_HIGH - _RE_CRIT_LOW)
    return _CD_SUPERCRITICAL + 0.5 * (_CD_SUBCRITICAL - _CD_SUPERCRITICAL) * (1 + math.cos(phase))
