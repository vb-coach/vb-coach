"""Attacker physical parameters.

Captures the height-related inputs needed to determine where in 3D space an
attacker can effectively strike a ball. Biomechanics (approach speed, jump
timing) is deferred to Phase B; this module is purely geometric.

Preset anthropometry values are anchored to Palao, Manzanares & Valades
2014 (J Hum Kinetics, n=1440 male / 1459 female FIVB-level players);
Sattler et al. 2015 (JSCR, Slovenian D1/D2 jump data); and Pocek et al.
2021 (IJERPH, junior female). See docs/research/calibration-priors.md.
"""

from __future__ import annotations

from dataclasses import dataclass


# Sheppard et al. 2008 (AIS) report in-game contact height = spike reach
# minus 18-22 cm (ball-center position, accounting for ball radius +
# hand wrap + elbow/shoulder/trunk-tilt non-full extension). 0.20 m is
# the median; raise to 0.22 m for elite attackers who can contact a hair
# higher on the ball.
DEFAULT_CONTACT_LOSS: float = 0.20


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
        (ball radius 10.5 cm + hand wrap ~5 cm + arm/shoulder non-full
        extension ~5 cm). 20 cm by default (Sheppard 2008).
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


# Trained recreational male: club-level player who can spike. Picked as
# the project's default because most "amateur" tooling questions come
# from this band.
DEFAULT_AMATEUR_MALE: Attacker = Attacker(
    height=1.78,
    standing_reach=2.32,
    approach_reach=2.95,
)

# More representative of an average recreational male (height closer to
# global adult-male mean, modest jump).
RECREATIONAL_MALE: Attacker = Attacker(
    height=1.75,
    standing_reach=2.27,
    approach_reach=2.78,
)

# College / advanced club male (NCAA D2-D3 OH median, Marques 2009 etc.).
COLLEGE_MALE: Attacker = Attacker(
    height=1.90,
    standing_reach=2.48,
    approach_reach=3.20,
)

# Elite male outside hitter (Palao 2014 OH median: height 1.97 m, spike
# reach 3.45 m). spike_reach maps to approach_reach in our model.
ELITE_MALE: Attacker = Attacker(
    height=1.97,
    standing_reach=2.58,
    approach_reach=3.45,
    contact_loss=0.22,
)

# Recreational female club player.
DEFAULT_AMATEUR_FEMALE: Attacker = Attacker(
    height=1.68,
    standing_reach=2.15,
    approach_reach=2.55,
    contact_loss=0.18,
)

# College / advanced female (NCAA D1 OH median).
COLLEGE_FEMALE: Attacker = Attacker(
    height=1.82,
    standing_reach=2.35,
    approach_reach=2.95,
    contact_loss=0.18,
)

# Elite female OH (Palao 2014 OH median: 1.84 m / 3.05 m spike reach).
ELITE_FEMALE: Attacker = Attacker(
    height=1.84,
    standing_reach=2.39,
    approach_reach=3.05,
    contact_loss=0.20,
)
