"""Aerodynamic coefficients for a volleyball.

Submodules:
    drag    - Re-dependent drag and (eventually) full-regime Cd(Re)
    magnus  - spin-induced lift (Cl(S))
    knuckle - panel-induced unsteady lateral force (added in this refactor)
"""

from vb_coach.aerodynamics.drag import reynolds, cd_subcritical, cd_full_regime
from vb_coach.aerodynamics.magnus import spin_parameter, cl_magnus

__all__ = [
    "reynolds",
    "cd_subcritical",
    "cd_full_regime",
    "spin_parameter",
    "cl_magnus",
]
