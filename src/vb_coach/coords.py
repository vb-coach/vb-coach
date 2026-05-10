"""World coordinate convention.

Right-handed frame:
  origin: on the floor under the center of the net
  x-axis: along the net, positive toward team A's right (and team B's left)
  y-axis: perpendicular to the net, positive into team A's court
  z-axis: vertical, positive up

Spin vector omega uses the right-hand rule. The sign of omega for a given
"spin name" depends on the direction of travel. Examples:
  ball travels +x with back-spin  -> omega in -y (top of ball moves in -x)
  ball travels +x with top-spin   -> omega in +y
  ball travels +y with back-spin  -> omega in +x
  ball travels +y with top-spin   -> omega in -x
The Magnus force is F_m = 0.5 * rho * |v|^2 * A * Cl(S) * (omega_hat x v_hat).
"""

import numpy as np


def on_team_a(point: np.ndarray) -> bool:
    return bool(point[1] > 0)


def on_team_b(point: np.ndarray) -> bool:
    return bool(point[1] < 0)


def mirror_across_net(point: np.ndarray) -> np.ndarray:
    out = point.copy()
    out[1] = -out[1]
    return out
