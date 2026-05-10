"""Physical constants and default volleyball parameters.

All SI units (kg, m, s, rad).
"""

import math

G: float = 9.81
RHO_AIR: float = 1.20
NU_AIR: float = 1.5e-5

BALL_DIAMETER: float = 0.21
BALL_RADIUS: float = BALL_DIAMETER / 2
BALL_MASS: float = 0.270
BALL_CROSS_SECTION: float = math.pi * BALL_RADIUS ** 2

COURT_LENGTH: float = 18.0
COURT_WIDTH: float = 9.0
ATTACK_LINE_FROM_NET: float = 3.0
NET_HEIGHT_MEN: float = 2.43
NET_HEIGHT_WOMEN: float = 2.24
ANTENNA_HEIGHT_ABOVE_NET: float = 0.80
