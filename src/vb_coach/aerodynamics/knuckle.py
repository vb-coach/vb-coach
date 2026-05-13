"""Panel-induced unsteady lateral force model for a volleyball at high Re.

The float-serve "knuckle" effect arises from asymmetric flow separation
around the ball's panels, which produces a side force whose magnitude
oscillates with vortex-shedding-like dynamics. This module captures it as:

- cl_lat_rms(Re): the magnitude envelope, peaked in the drag-crisis band
- OUState + step_ou: an Ornstein-Uhlenbeck process that produces correlated
  random fluctuations with coherence time TAU_KNUCKLE matching Cohen 2014
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# Coherence time of the lateral force process. Darbois Texier et al. 2016
# (NJP 18:073027) section 4.1 gives lambda/U ~ 8m / 16 m/s ~ 0.5 s and
# their Table 2 row "Volleyball" gives tau ~ 0.33 s; we adopt 0.35 s as
# the representative value across the drag-crisis band.
TAU_KNUCKLE: float = 0.35


# Norman & McKeon 2011 (Exp Fluids 51:1439) report z-direction RMS Cl
# peaking at 0.10-0.11 at Re ~ 3.0e5, falling to ~0.05 in the subcritical
# band and ~0.03 in the deep supercritical. Re_lat_peak is co-located
# with the drag-crisis center (Darbois Texier 2016 fig 7b).
_CL_LAT_PEAK: float = 0.10
_RE_LAT_PEAK: float = 3.0e5
_RE_LAT_WIDTH: float = 0.6e5


def cl_lat_rms(reynolds_number: float) -> float:
    """RMS lateral lift coefficient as a function of Re.

    Gaussian peaked in the drag-crisis band where panel-orientation effects
    dominate; ~0 outside it (subcritical and deep supercritical).
    """
    return _CL_LAT_PEAK * math.exp(
        -((reynolds_number - _RE_LAT_PEAK) / _RE_LAT_WIDTH) ** 2
    )


@dataclass
class OUState:
    """Two-channel OU state. chan_h drives in-plane sideways force;
    chan_v drives the (weaker) out-of-plane perturbation."""

    chan_h: float = 0.0
    chan_v: float = 0.0


def step_ou(ou: OUState, dt: float, rng: np.random.Generator) -> OUState:
    """Advance the OU process by dt. Steady-state variance is 1 per channel.

    Mutates and returns the input state for ergonomic chaining.
    """
    decay = math.exp(-dt / TAU_KNUCKLE)
    noise_scale = math.sqrt(1.0 - decay ** 2)
    ou.chan_h = decay * ou.chan_h + noise_scale * rng.standard_normal()
    ou.chan_v = decay * ou.chan_v + noise_scale * rng.standard_normal()
    return ou
