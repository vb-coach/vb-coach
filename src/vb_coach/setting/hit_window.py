"""Hit-window geometry for attack-side analysis.

Models the effective-contact envelope around an attacker's optimal contact
point as an asymmetric axis-aligned ellipsoid. The envelope captures where
contact with the ball produces a viable spike — not just where the hand
can geometrically reach.

The asymmetry is literature-backed (see project memory
`reference_hit_window_biomechanics.md`):
  - Wagner et al. 2020: +/-20 cm horizontal deviations from optimum cause
    ~70% increase in shoulder muscle activation (effective tolerance scale).
  - Coaching consensus + Wagner posterior penalty: contact tolerance is
    larger toward the net (forward, in the direction of arm swing) than
    behind the optimum.
  - Coaching consensus: max approach reach is a hard upper bound (cannot
    contact above peak); below the optimum, the attacker can lower the
    jump or contact later in the arc, so down-tolerance is larger.

The body's facing direction is implicitly "toward the net" — i.e., -y for
a team-A attacker. So the "forward" half-axis applies when
(p[1] - center[1]) is negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vb_coach.setting.attacker import Attacker


_POSITION4_CONTACT_X: float = -4.0
_POSITION2_CONTACT_X: float = +4.0
_POSITION_CONTACT_Y: float = 0.2

# Wagner 2020 baseline (intrinsic upper-limb contact tolerance, ~20 cm
# lateral). Footwork tolerance is added on top per level: amateurs need
# more slack because they don't pre-adjust as cleanly as pros.
_WAGNER_R_X_BASE: float = 0.20
_FOOTWORK_R_X_BY_LEVEL: dict[str, float] = {
    "amateur": 0.20,
    "college": 0.10,
    "elite":   0.05,
}

_R_Y_FRONT: float = 0.30
_R_Y_BACK: float = 0.15
_R_Z_UP: float = 0.05
_R_Z_DOWN: float = 0.30


@dataclass
class HitWindow:
    """Asymmetric axis-aligned ellipsoidal effective-contact envelope.

    A point p is inside iff
        (dx / r_x)^2 + (dy / r_y_eff)^2 + (dz / r_z_eff)^2 <= 1
    where:
        dx, dy, dz = p - center
        r_y_eff   = r_y_front if dy < 0 else r_y_back   (forward = -y, world frame)
        r_z_eff   = r_z_up    if dz > 0 else r_z_down
    """

    center: np.ndarray
    r_x: float
    r_y_front: float
    r_y_back: float
    r_z_up: float
    r_z_down: float

    def __post_init__(self) -> None:
        if self.center.shape != (3,):
            raise ValueError(f"center must have shape (3,), got {self.center.shape}")
        for name in ("r_x", "r_y_front", "r_y_back", "r_z_up", "r_z_down"):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")

    def contains(self, point: np.ndarray) -> bool:
        dx = point[0] - self.center[0]
        dy = point[1] - self.center[1]
        dz = point[2] - self.center[2]
        r_y = self.r_y_front if dy < 0 else self.r_y_back
        r_z = self.r_z_up if dz > 0 else self.r_z_down
        return bool((dx / self.r_x) ** 2 + (dy / r_y) ** 2 + (dz / r_z) ** 2 <= 1.0)

    def contains_batch(self, points: np.ndarray) -> np.ndarray:
        dx = points[:, 0] - self.center[0]
        dy = points[:, 1] - self.center[1]
        dz = points[:, 2] - self.center[2]
        r_y = np.where(dy < 0, self.r_y_front, self.r_y_back)
        r_z = np.where(dz > 0, self.r_z_up, self.r_z_down)
        return (dx / self.r_x) ** 2 + (dy / r_y) ** 2 + (dz / r_z) ** 2 <= 1.0


def _pin_hit_window(attacker: Attacker, contact_x: float) -> HitWindow:
    """Shared geometry for left-pin (4号位) and right-pin (2号位) windows.

    The two pin attacks are mirrors across the net center: same y-offset,
    same vertical asymmetry, same level-dependent r_x scaling. Only the
    x-coordinate of the contact point differs.
    """
    center = np.array([
        contact_x,
        _POSITION_CONTACT_Y,
        attacker.contact_height,
    ])
    r_x = _WAGNER_R_X_BASE + _FOOTWORK_R_X_BY_LEVEL[attacker.level]
    return HitWindow(
        center=center,
        r_x=r_x,
        r_y_front=_R_Y_FRONT,
        r_y_back=_R_Y_BACK,
        r_z_up=_R_Z_UP,
        r_z_down=_R_Z_DOWN,
    )


def position4_hit_window(attacker: Attacker) -> HitWindow:
    """Hit-window for a 4号位 (left-pin) attack by the given attacker.

    The same window applies to all 4号位 attack types (高球 / 半高 / 平拉开)
    — the difference between attack types lives entirely in the score
    functions' velocity / dwell / drift thresholds, not in geometry.

    Along-net radius (r_x) scales with attacker.level: elite players have
    faster footwork adjustment, so their effective contact envelope is
    tighter (Wagner 20 cm + footwork 5 cm) compared with amateur (20 + 20).
    """
    return _pin_hit_window(attacker, _POSITION4_CONTACT_X)


def position2_hit_window(attacker: Attacker) -> HitWindow:
    """Hit-window for a 2号位 (right-pin) attack by the given attacker.

    Mirror of position4_hit_window across the net center (x = 0). All
    other geometry (vertical asymmetry, y-offset, footwork scaling) is
    identical to 4号位.
    """
    return _pin_hit_window(attacker, _POSITION2_CONTACT_X)
