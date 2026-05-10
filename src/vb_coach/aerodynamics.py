"""Aerodynamic coefficients for a volleyball at set speeds.

Per the subcritical-aero literature review (Darbois Texier 2016, Asai 2010,
Norman & McKeon 2011, cricket-ball analog Jackson 2024), at set speeds
(4-8 m/s, Re ~50k-120k) the ball behaves essentially as a smooth sphere:
- Cd is roughly constant at ~0.45-0.5 across the subcritical band
- Mean lateral (panel-induced) force is zero
- Unsteady lateral deflection over a 1 s flight is at most a few cm

Cd is therefore exposed as a Reynolds-indexed function so the call site
remains uniform if a regime-aware model is ever added, but the current
implementation returns a constant.

The Magnus lift coefficient Cl(S) is modelled as linear in spin parameter
S = omega * R / |v| up to S ~ 0.5, then saturating at 0.2 - the same shape
sports-ball wind-tunnel data shows in this Re range.
"""

from __future__ import annotations


def reynolds(velocity_magnitude: float, diameter: float, kinematic_viscosity: float) -> float:
    return velocity_magnitude * diameter / kinematic_viscosity


def spin_parameter(spin_rate_rad_s: float, velocity_magnitude: float, radius: float) -> float:
    if velocity_magnitude == 0.0:
        return 0.0
    return spin_rate_rad_s * radius / velocity_magnitude


def cd_subcritical(reynolds_number: float) -> float:
    return 0.45


_CL_SATURATION = 0.20
_CL_HALF_SATURATION_S = 0.25


def cl_magnus(spin_parameter_value: float) -> float:
    s = abs(spin_parameter_value)
    if s >= 2 * _CL_HALF_SATURATION_S:
        return _CL_SATURATION
    return _CL_SATURATION * s / (2 * _CL_HALF_SATURATION_S)
