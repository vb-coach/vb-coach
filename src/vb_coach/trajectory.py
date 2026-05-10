"""Trajectory data types and simulation entry point.

ReleaseState captures the moment the ball leaves the setter's hands.
Trajectory captures the integrated path: arrays for time, position,
velocity, and spin sampled at the integrator's output times.

simulate() is implemented further down.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReleaseState:
    position: np.ndarray
    velocity: np.ndarray
    spin: np.ndarray

    def __post_init__(self) -> None:
        for name in ("position", "velocity", "spin"):
            arr = getattr(self, name)
            if arr.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), got {arr.shape}")


@dataclass
class Trajectory:
    time: np.ndarray
    position: np.ndarray
    velocity: np.ndarray
    spin: np.ndarray

    def __post_init__(self) -> None:
        n = self.time.shape[0]
        for name in ("position", "velocity", "spin"):
            arr = getattr(self, name)
            if arr.shape != (n, 3):
                raise ValueError(
                    f"{name} must have shape ({n}, 3), got {arr.shape}"
                )

    def __len__(self) -> int:
        return int(self.time.shape[0])

    def apex_index(self) -> int:
        return int(np.argmax(self.position[:, 2]))

    def apex_height(self) -> float:
        return float(self.position[self.apex_index(), 2])


from scipy.integrate import solve_ivp

from vb_coach.aerodynamics import cd_full_regime, cl_magnus, reynolds, spin_parameter
from vb_coach.constants import (
    BALL_CROSS_SECTION,
    BALL_DIAMETER,
    BALL_MASS,
    BALL_RADIUS,
    G,
    NU_AIR,
    RHO_AIR,
)


def _rhs(
    t: float,
    y: np.ndarray,
    mass: float,
    rho: float,
    area: float,
    radius: float,
    diameter: float,
    spin: np.ndarray,
) -> np.ndarray:
    # State y = [x, y, z, vx, vy, vz]. Spin is held constant during the flight
    # (decay over <2 s is negligible compared to the model's other uncertainties).
    v = y[3:6]
    speed = float(np.linalg.norm(v))
    a = np.array([0.0, 0.0, -G])
    if speed > 0.0:
        v_hat = v / speed
        re = reynolds(speed, diameter, NU_AIR)
        cd = cd_full_regime(re)
        a = a - (0.5 * rho * speed * speed * area * cd / mass) * v_hat
        spin_mag = float(np.linalg.norm(spin))
        if spin_mag > 0.0:
            omega_hat = spin / spin_mag
            s = spin_parameter(spin_mag, speed, radius)
            cl = cl_magnus(s)
            a = a + (0.5 * rho * speed * speed * area * cl / mass) * np.cross(
                omega_hat, v_hat
            )
    return np.concatenate([v, a])


def _floor_event(t: float, y: np.ndarray, *args) -> float:
    return float(y[2])


_floor_event.terminal = True
_floor_event.direction = -1


def simulate(
    release: ReleaseState,
    t_max: float = 3.0,
    mass: float = BALL_MASS,
    rho: float = RHO_AIR,
    area: float = BALL_CROSS_SECTION,
    radius: float = BALL_RADIUS,
    diameter: float = BALL_DIAMETER,
    max_step: float = 0.01,
) -> Trajectory:
    y0 = np.concatenate([release.position, release.velocity])
    sol = solve_ivp(
        _rhs,
        (0.0, t_max),
        y0,
        method="RK45",
        events=_floor_event,
        args=(mass, rho, area, radius, diameter, release.spin),
        max_step=max_step,
        rtol=1e-8,
        atol=1e-9,
    )
    n = sol.t.shape[0]
    pos = sol.y[0:3, :].T
    vel = sol.y[3:6, :].T
    spins = np.broadcast_to(release.spin, (n, 3)).copy()
    return Trajectory(time=sol.t, position=pos, velocity=vel, spin=spins)
