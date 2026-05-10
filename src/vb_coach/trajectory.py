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
