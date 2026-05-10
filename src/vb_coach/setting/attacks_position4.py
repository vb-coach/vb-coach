"""Quality-scoring functions for 4号位 attacks.

Each scoring function maps a (Trajectory, HitWindow) pair to a HighBallScore
(or analogous score type) with a scalar `score` in [0, 1] plus diagnostic
sub-scores. Scores are the product of three orthogonal quality dimensions:
dwell time in the hit window, vertical velocity at contact, and lateral
stability through the dwell.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vb_coach.setting.hit_window import HitWindow
from vb_coach.trajectory import Trajectory


_DWELL_TARGET_SECONDS: float = 0.08
_VELOCITY_BEST: float = -4.0      # m/s, best vertical velocity at contact
_VELOCITY_BAND: float = 2.0       # m/s, falloff half-width (band is [-6, -2])
_LATERAL_MAX_DRIFT: float = 0.30  # m, drift above which lateral score is 0


@dataclass
class HighBallScore:
    """Diagnostic result of scoring a trajectory against a high-ball hit window.

    Attributes
    ----------
    score
        Product of dwell_score, velocity_score, and lateral_score. In [0, 1].
    dwell_score, velocity_score, lateral_score
        Sub-scores, each in [0, 1].
    dwell_seconds
        Total time the trajectory spent inside the hit window during descent.
    velocity_at_contact_z
        Vertical velocity at the midpoint of the in-window descent segment.
    lateral_drift_m
        |x| displacement of the ball during the in-window descent segment.
    contact_index
        Trajectory index used as the "contact" reference point (midpoint of
        in-window descent segment). -1 if the ball never entered the window
        during descent.
    """

    score: float
    dwell_score: float
    velocity_score: float
    lateral_score: float
    dwell_seconds: float
    velocity_at_contact_z: float
    lateral_drift_m: float
    contact_index: int


def _zero_score(contact_index: int = -1) -> HighBallScore:
    return HighBallScore(
        score=0.0,
        dwell_score=0.0,
        velocity_score=0.0,
        lateral_score=0.0,
        dwell_seconds=0.0,
        velocity_at_contact_z=0.0,
        lateral_drift_m=0.0,
        contact_index=contact_index,
    )


def score_high_ball(trajectory: Trajectory, window: HitWindow) -> HighBallScore:
    """Score how well a trajectory delivers a 4号位 high ball.

    The trajectory must enter the hit window during its descent (post-apex)
    for a non-zero score. The score combines:
      - dwell_score: linear 0..1 as in-window dwell goes 0..0.30 s
      - velocity_score: triangular peak at v_z = -4 m/s, zero at +/- 2 m/s away
      - lateral_score: linear 1..0 as in-window lateral drift goes 0..0.30 m

    Returns a HighBallScore with the combined score (product of sub-scores)
    and diagnostic fields.
    """
    inside_mask = window.contains_batch(trajectory.position)
    if not inside_mask.any():
        return _zero_score()
    apex_idx = trajectory.apex_index()
    descent_inside = np.where(inside_mask & (np.arange(len(trajectory)) > apex_idx))[0]
    if descent_inside.size == 0:
        return _zero_score()
    first = int(descent_inside[0])
    last = int(descent_inside[-1])
    contact = int(descent_inside[descent_inside.size // 2])

    dwell_seconds = float(trajectory.time[last] - trajectory.time[first])
    velocity_z = float(trajectory.velocity[contact, 2])
    drift = float(
        np.max(trajectory.position[descent_inside, 1])
        - np.min(trajectory.position[descent_inside, 1])
    )

    dwell_score = min(dwell_seconds / _DWELL_TARGET_SECONDS, 1.0)
    velocity_score = max(0.0, 1.0 - abs(velocity_z - _VELOCITY_BEST) / _VELOCITY_BAND)
    lateral_score = max(0.0, 1.0 - drift / _LATERAL_MAX_DRIFT)

    score = dwell_score * velocity_score * lateral_score
    return HighBallScore(
        score=float(score),
        dwell_score=float(dwell_score),
        velocity_score=float(velocity_score),
        lateral_score=float(lateral_score),
        dwell_seconds=dwell_seconds,
        velocity_at_contact_z=velocity_z,
        lateral_drift_m=drift,
        contact_index=contact,
    )
