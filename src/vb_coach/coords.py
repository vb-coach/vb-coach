"""World coordinate convention.

Right-handed frame:
  origin: on the floor under the center of the net
  x-axis: along the net, positive toward team A's right (and team B's left)
  y-axis: perpendicular to the net, positive into team A's court
  z-axis: vertical, positive up

Spin vector omega uses the right-hand rule: pure back-spin on a ball
traveling in +x has omega in +y, so omega x v_hat points in +z (lift).
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
