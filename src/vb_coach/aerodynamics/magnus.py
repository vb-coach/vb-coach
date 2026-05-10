"""Magnus lift coefficient and spin parameter for a volleyball.

Cl(S) is modelled as linear in spin parameter S = omega * R / |v| up to
S ~ 0.5, then saturating at 0.2 - the shape sports-ball wind-tunnel data
shows in this Re range.
"""

from __future__ import annotations


def spin_parameter(spin_rate_rad_s: float, velocity_magnitude: float, radius: float) -> float:
    if velocity_magnitude == 0.0:
        return 0.0
    return spin_rate_rad_s * radius / velocity_magnitude


_CL_SATURATION = 0.20
_CL_HALF_SATURATION_S = 0.25


def cl_magnus(spin_parameter_value: float) -> float:
    s = abs(spin_parameter_value)
    if s >= 2 * _CL_HALF_SATURATION_S:
        return _CL_SATURATION
    return _CL_SATURATION * s / (2 * _CL_HALF_SATURATION_S)
