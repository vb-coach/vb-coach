"""Reusable float-serve scenario presets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vb_coach.trajectory import ReleaseState


@dataclass(frozen=True)
class FloatServePreset:
    """A reproducible float-serve release state and stochastic seed."""

    slug: str
    label: str
    speed: float
    contact_height: float
    vertical_speed: float
    seed: int

    def release_state(self, lateral_speed: float = 0.0) -> ReleaseState:
        return ReleaseState(
            position=np.array([0.0, -9.0, self.contact_height]),
            velocity=np.array([lateral_speed, self.speed, self.vertical_speed]),
            spin=np.zeros(3),
        )


CANONICAL_FLOAT_SERVES = (
    FloatServePreset(
        slug="persistent",
        label="Critical-band wobble",
        speed=21.0,
        contact_height=2.8,
        vertical_speed=2.0,
        seed=7,
    ),
    FloatServePreset(
        slug="drop",
        label="Drop floater",
        speed=19.0,
        contact_height=2.8,
        vertical_speed=2.5,
        seed=3,
    ),
    FloatServePreset(
        slug="below_sweet_spot",
        label="Below sweet spot",
        speed=13.0,
        contact_height=2.8,
        vertical_speed=4.2,
        seed=12,
    ),
)
