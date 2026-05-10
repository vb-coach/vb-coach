"""Attacker physical parameters.

Captures the height-related inputs needed to determine where in 3D space an
attacker can effectively strike a ball. Biomechanics (approach speed, jump
timing) is deferred to Phase B; this module is purely geometric.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_CONTACT_LOSS: float = 0.15


@dataclass
class Attacker:
    """Attacker height + reach parameters in meters.

    Attributes
    ----------
    height
        Body height (standing, no shoes).
    standing_reach
        Highest fingertip reach with arm extended overhead, flat-footed.
    approach_reach
        Highest fingertip reach at the top of a running-approach jump.
    contact_loss
        Distance from approach_reach down to ball center at contact
        (ball radius + hand wrap + non-full extension). 15 cm by default.
    """

    height: float
    standing_reach: float
    approach_reach: float
    contact_loss: float = DEFAULT_CONTACT_LOSS

    def __post_init__(self) -> None:
        if self.standing_reach <= self.height:
            raise ValueError(
                f"standing_reach ({self.standing_reach}) must exceed height "
                f"({self.height})"
            )
        if self.approach_reach < self.standing_reach:
            raise ValueError(
                f"approach_reach ({self.approach_reach}) must be >= "
                f"standing_reach ({self.standing_reach})"
            )

    @property
    def contact_height(self) -> float:
        return self.approach_reach - self.contact_loss


DEFAULT_AMATEUR_MALE: Attacker = Attacker(
    height=1.78,
    standing_reach=2.32,
    approach_reach=2.95,
)
