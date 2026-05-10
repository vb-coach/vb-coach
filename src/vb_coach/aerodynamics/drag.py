"""Drag coefficient and Reynolds number for a volleyball."""

from __future__ import annotations


def reynolds(velocity_magnitude: float, diameter: float, kinematic_viscosity: float) -> float:
    return velocity_magnitude * diameter / kinematic_viscosity


def cd_subcritical(reynolds_number: float) -> float:
    """Constant Cd ~ 0.45 across the subcritical band (Re < ~1.5e5).

    Kept as a function for analytical clarity and historical compatibility;
    new code should call cd_full_regime, which agrees with this in the
    subcritical band and additionally covers the drag crisis.
    """
    return 0.45
