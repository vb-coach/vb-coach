"""Force model for a flying volleyball.

F_total = F_gravity + F_drag + F_Magnus

  F_d = -0.5 * rho * |v|^2 * A * Cd(Re) * v_hat
  F_m =  0.5 * rho * |v|^2 * A * Cl(S) * (omega_hat x v_hat)

Cd and Cl come from aerodynamics.py; both are evaluated each call so the
ODE integrator always sees the current velocity-dependent Re and S.
"""

from __future__ import annotations

import numpy as np

from vb_coach.aerodynamics import (
    cd_subcritical,
    cl_magnus,
    reynolds,
    spin_parameter,
)
from vb_coach.aerodynamics.knuckle import OUState, cl_lat_rms
from vb_coach.constants import G, NU_AIR


def gravity_force(mass: float) -> np.ndarray:
    return np.array([0.0, 0.0, -mass * G])


def drag_force(velocity: np.ndarray, rho: float, area: float, cd: float) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    if speed == 0.0:
        return np.zeros(3)
    v_hat = velocity / speed
    magnitude = 0.5 * rho * speed * speed * area * cd
    return -magnitude * v_hat


def magnus_force(
    velocity: np.ndarray,
    spin: np.ndarray,
    rho: float,
    area: float,
    radius: float,
) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    spin_mag = float(np.linalg.norm(spin))
    if speed == 0.0 or spin_mag == 0.0:
        return np.zeros(3)
    v_hat = velocity / speed
    omega_hat = spin / spin_mag
    s = spin_parameter(spin_mag, speed, radius)
    cl = cl_magnus(s)
    return 0.5 * rho * speed * speed * area * cl * np.cross(omega_hat, v_hat)


def total_acceleration(
    velocity: np.ndarray,
    spin: np.ndarray,
    mass: float,
    rho: float,
    area: float,
    radius: float,
    diameter: float | None = None,
) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    d = diameter if diameter is not None else 2 * radius
    re = reynolds(speed, d, NU_AIR)
    cd = cd_subcritical(re)
    f = (
        gravity_force(mass)
        + drag_force(velocity, rho, area, cd)
        + magnus_force(velocity, spin, rho, area, radius)
    )
    return f / mass


_KNUCKLE_SPEED_CUTOFF: float = 0.1
_KNUCKLE_VERTICAL_FRACTION: float = 0.5


def knuckle_force(
    velocity: np.ndarray,
    ou: OUState,
    rho: float,
    area: float,
    re: float,
) -> np.ndarray:
    """Panel-induced lateral force on a moving ball.

    Decomposes the lateral force into two flight-perpendicular channels:
    h_hat (horizontal-perpendicular, in the x-y plane) and vert_hat
    (out-of-plane vertical perturbation). The vertical channel is weighted
    at 0.5 because observed wobble is dominated by horizontal deflection.

    NOTE: this function does not step the OU process. The integrator calls
    step_ou separately so dt is owned at the simulator level.
    """
    speed = float(np.linalg.norm(velocity))
    if speed < _KNUCKLE_SPEED_CUTOFF:
        return np.zeros(3)
    v_hat = velocity / speed
    cl = cl_lat_rms(re)
    f_mag = 0.5 * rho * area * cl * speed * speed
    vxy = np.array([velocity[0], velocity[1], 0.0])
    vxy_norm = float(np.linalg.norm(vxy))
    if vxy_norm > 1e-6:
        h_hat = np.array([-vxy[1], vxy[0], 0.0]) / vxy_norm
    else:
        h_hat = np.array([0.0, 1.0, 0.0])
    vert_hat = np.cross(v_hat, h_hat)
    return f_mag * (ou.chan_h * h_hat + _KNUCKLE_VERTICAL_FRACTION * ou.chan_v * vert_hat)
