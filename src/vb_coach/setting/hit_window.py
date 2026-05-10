"""Hit-window geometry for attack-side analysis.

A HitWindow is an axis-aligned 3D box in world coordinates inside which
an attacker can effectively contact a ball. Outside the box, contact is
either physically impossible (out of reach) or tactically ineffective
(off-axis, mistimed). The shape is a simplification — real swing arcs
are ellipsoidal — but it captures the dominant constraints (reach
height, lateral footprint at the antenna, depth window across the net).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vb_coach.setting.attacker import Attacker


_POSITION4_CONTACT_X: float = -4.0   # near left antenna in world coords
_POSITION4_CONTACT_Y: float = 0.2    # just on team A's side of net
_POSITION4_HIGH_DX: float = 0.5      # lateral tolerance along net (m)
_POSITION4_HIGH_DY: float = 0.3      # depth tolerance across net (m)
_POSITION4_HIGH_DZ: float = 0.20     # vertical tolerance around contact height (m)


@dataclass
class HitWindow:
    """Axis-aligned 3D box defining the attacker's effective contact volume.

    Attributes
    ----------
    center
        World-coordinate center of the box, shape (3,).
    dx, dy, dz
        Half-extents along each world axis. A point p is inside iff
        |p[i] - center[i]| <= half_extent[i] for all i.
    """

    center: np.ndarray
    dx: float
    dy: float
    dz: float

    def __post_init__(self) -> None:
        if self.center.shape != (3,):
            raise ValueError(f"center must have shape (3,), got {self.center.shape}")

    def contains(self, point: np.ndarray) -> bool:
        return bool(
            abs(point[0] - self.center[0]) <= self.dx
            and abs(point[1] - self.center[1]) <= self.dy
            and abs(point[2] - self.center[2]) <= self.dz
        )

    def contains_batch(self, points: np.ndarray) -> np.ndarray:
        diff = np.abs(points - self.center)
        return (
            (diff[:, 0] <= self.dx)
            & (diff[:, 1] <= self.dy)
            & (diff[:, 2] <= self.dz)
        )


def position4_high_window(attacker: Attacker) -> HitWindow:
    """Hit-window for a 4号位 high-ball attack by the given attacker.

    Vertical center = attacker.contact_height. Lateral/depth center is the
    standard 4号位 attack position near the left antenna, just on team A's
    side of the net.
    """
    center = np.array([
        _POSITION4_CONTACT_X,
        _POSITION4_CONTACT_Y,
        attacker.contact_height,
    ])
    return HitWindow(
        center=center,
        dx=_POSITION4_HIGH_DX,
        dy=_POSITION4_HIGH_DY,
        dz=_POSITION4_HIGH_DZ,
    )
