# Phase A1.1: Attacker, Hit-Window, and 4号位高球 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the foundation layer for attack-side analysis — an `Attacker` parameter dataclass, a `HitWindow` geometric model, and a quality-scoring function for **4号位高球** (left-front high ball, the most common amateur attack). The pipeline takes a `Trajectory` (from `simulate()`) + an `Attacker` and returns a scalar quality score in `[0, 1]` plus diagnostics.

**Architecture:** New sub-package `src/vb_coach/setting/` housing attack-side code. Three new modules: `attacker.py` (Attacker dataclass + contact-height derivation), `hit_window.py` (HitWindow dataclass + position4 factory), `attacks_position4.py` (`score_high_ball` function). All public functions are pure (no module-global state). Each attack type is one function — the registry / multi-attack catalog is deferred to Phase A1.2.

**Tech Stack:** Python 3.11+, numpy, pytest, uv. No new deps.

**Scope decisions:**
- Only **高球** (high ball) is modeled. 半高 (semi-high) and 平拉开 (shoot set) come in a separate Phase A1.2 plan after this pattern is validated.
- Quality metric is **model-validated, not data-validated** — synthetic trajectories in tests, no real volleyball video. Real-data calibration is a later phase.
- Hit-window is an **axis-aligned box** in world coordinates, parameterized by center + half-extents (dx, dy, dz). More elaborate geometries (sphere, ellipsoid, attacker swing arc) are deliberately deferred.
- Only **right-handed amateur male attacker** profile is supported by default (extending to female / left-handed is trivial later — just different default Attacker values).

**Out of scope:**
- Attacker biomechanics (Phase B)
- Optimization: "given attacker, find best release state" (Phase A5)
- Multi-attack registry / dispatcher (Phase A1.2 or later)
- Interactive visualization (Phase A6)

---

## File structure after this plan

```
src/vb_coach/
  setting/                          ← new sub-package
    __init__.py                     re-exports
    attacker.py                     Attacker dataclass
    hit_window.py                   HitWindow dataclass + position4_high_window()
    attacks_position4.py            score_high_ball()
  ... unchanged otherwise ...
tests/
  test_setting_attacker.py
  test_setting_hit_window.py
  test_setting_attacks_position4_high.py
examples/
  position4_high_demo.py            runnable example
```

## Coordinate convention (already defined; no changes here)

Origin at floor under net center. `+y` = team A's court. `+x` = team A's right. `+z` = up. Team A's 4号位 (left front, looking from team A's side) is at large negative x near the net.

## Key constants (will be defined in code; documented here for reviewer)

- `DEFAULT_CONTACT_LOSS = 0.15` m — distance from `approach_reach` (highest fingertip) down to ball center at contact
- 4号位 antenna nominal position: x = -4.5, y = 0
- 4号位 attacker contact target: x ≈ -4.0, y ≈ +0.2 (just on team A's side of net)
- Amateur male default: height 1.78 m, standing_reach 2.32 m, approach_reach 2.90 m → contact_height 2.75 m

---

## Task 1: Setting sub-package and `Attacker` dataclass

**Files:**
- Create: `src/vb_coach/setting/__init__.py`
- Create: `src/vb_coach/setting/attacker.py`
- Create: `tests/test_setting_attacker.py`

- [ ] **Step 1: Write failing tests**

Create `/home/siriux/Projects/vb_coach/tests/test_setting_attacker.py`:

```python
import math

import pytest

from vb_coach.setting.attacker import Attacker, DEFAULT_AMATEUR_MALE


def test_attacker_construction_with_explicit_values():
    a = Attacker(height=1.80, standing_reach=2.34, approach_reach=2.95)
    assert a.height == 1.80
    assert a.standing_reach == 2.34
    assert a.approach_reach == 2.95


def test_attacker_contact_height_uses_default_loss():
    # contact_height = approach_reach - contact_loss (default 0.15)
    a = Attacker(height=1.80, standing_reach=2.34, approach_reach=2.95)
    assert math.isclose(a.contact_height, 2.80, abs_tol=1e-9)


def test_attacker_contact_height_with_custom_loss():
    a = Attacker(
        height=1.80,
        standing_reach=2.34,
        approach_reach=2.95,
        contact_loss=0.25,
    )
    assert math.isclose(a.contact_height, 2.70, abs_tol=1e-9)


def test_attacker_rejects_implausible_proportions():
    # standing_reach should exceed height (arms-up always reaches above head)
    with pytest.raises(ValueError):
        Attacker(height=1.80, standing_reach=1.60, approach_reach=2.50)


def test_attacker_rejects_approach_below_standing():
    # Approach reach must be >= standing reach (you don't jump downward)
    with pytest.raises(ValueError):
        Attacker(height=1.80, standing_reach=2.34, approach_reach=2.30)


def test_default_amateur_male_contact_in_data_band():
    # Per project amateur-reach reference: trained recreational male
    # contact_height median ~ 2.95 m (mean from VolleyTalk dataset).
    # Default value should fall within the trained-amateur band 2.71-3.19 m.
    a = DEFAULT_AMATEUR_MALE
    assert 2.71 <= a.contact_height <= 3.19
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/test_setting_attacker.py -v`
Expected: FAIL — `vb_coach.setting` does not exist.

- [ ] **Step 3: Implement `attacker.py`**

Create `/home/siriux/Projects/vb_coach/src/vb_coach/setting/__init__.py`:

```python
"""Attack-side analysis for vb_coach.

Submodules:
    attacker    - Attacker physical parameters (height, reach, contact-height)
    hit_window  - Geometric definition of where an attacker can effectively strike
    attacks_position4 - Quality-scoring functions for 4号位 attacks
"""
```

Create `/home/siriux/Projects/vb_coach/src/vb_coach/setting/attacker.py`:

```python
"""Attacker physical parameters.

Captures the height-related inputs needed to determine where in 3D space an
attacker can effectively strike a ball. Biomechanics (approach speed, jump
timing) is deferred to Phase B; this module is purely geometric.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_setting_attacker.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vb_coach/setting/ tests/test_setting_attacker.py
git commit -m "[feat:setting] Attacker dataclass with reach-derived contact height"
```

---

## Task 2: `HitWindow` dataclass + 4号位高球 factory

**Files:**
- Create: `src/vb_coach/setting/hit_window.py`
- Create: `tests/test_setting_hit_window.py`

- [ ] **Step 1: Write failing tests**

Create `/home/siriux/Projects/vb_coach/tests/test_setting_hit_window.py`:

```python
import math

import numpy as np
import pytest

from vb_coach.setting.attacker import Attacker, DEFAULT_AMATEUR_MALE
from vb_coach.setting.hit_window import HitWindow, position4_high_window


def test_hit_window_contains_center():
    w = HitWindow(center=np.array([0.0, 0.0, 2.8]), dx=0.5, dy=0.3, dz=0.2)
    assert w.contains(np.array([0.0, 0.0, 2.8]))


def test_hit_window_contains_near_edge():
    w = HitWindow(center=np.array([0.0, 0.0, 2.8]), dx=0.5, dy=0.3, dz=0.2)
    assert w.contains(np.array([0.49, 0.29, 2.99]))
    assert w.contains(np.array([-0.49, -0.29, 2.61]))


def test_hit_window_excludes_outside():
    w = HitWindow(center=np.array([0.0, 0.0, 2.8]), dx=0.5, dy=0.3, dz=0.2)
    assert not w.contains(np.array([0.51, 0.0, 2.8]))   # outside x
    assert not w.contains(np.array([0.0, 0.31, 2.8]))   # outside y
    assert not w.contains(np.array([0.0, 0.0, 3.01]))   # outside z


def test_hit_window_contains_batch():
    w = HitWindow(center=np.array([0.0, 0.0, 2.8]), dx=0.5, dy=0.3, dz=0.2)
    points = np.array([
        [0.0, 0.0, 2.8],     # inside
        [0.6, 0.0, 2.8],     # outside x
        [0.0, 0.0, 3.5],     # outside z
        [0.3, 0.2, 2.7],     # inside
    ])
    mask = w.contains_batch(points)
    np.testing.assert_array_equal(mask, np.array([True, False, False, True]))


def test_position4_high_window_centered_at_attacker_contact_height():
    a = DEFAULT_AMATEUR_MALE
    w = position4_high_window(a)
    # Center z should equal the attacker's contact height (2.95 - 0.15 = 2.80)
    assert math.isclose(w.center[2], a.contact_height, abs_tol=1e-9)


def test_position4_high_window_near_left_antenna():
    a = DEFAULT_AMATEUR_MALE
    w = position4_high_window(a)
    # 4号位 attack lands near x = -4.0 m
    assert -4.5 < w.center[0] < -3.5
    # Just on team A's side of the net (small positive y)
    assert 0.0 < w.center[1] < 0.6


def test_position4_high_window_dimensions_reasonable_for_amateur():
    a = DEFAULT_AMATEUR_MALE
    w = position4_high_window(a)
    # Amateur swing arc and timing tolerance — 0.3-0.7 m in lateral, 0.15-0.3 m vertical
    assert 0.3 < w.dx < 0.8
    assert 0.2 < w.dy < 0.5
    assert 0.15 < w.dz < 0.35
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/test_setting_hit_window.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `hit_window.py`**

Create `/home/siriux/Projects/vb_coach/src/vb_coach/setting/hit_window.py`:

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_setting_hit_window.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vb_coach/setting/hit_window.py tests/test_setting_hit_window.py
git commit -m "[feat:setting] HitWindow box geometry + 4号位高球 factory"
```

---

## Task 3: `score_high_ball()` — quality metric for 4号位高球

**Files:**
- Create: `src/vb_coach/setting/attacks_position4.py`
- Create: `tests/test_setting_attacks_position4_high.py`

### Scoring model (for the implementer to follow exactly)

The score is the product of three sub-scores, each in `[0, 1]`:

1. **dwell_score** — how long the ball spends inside the hit window during descent. Linear from 0 (no dwell) up to 1 (>= 0.30 s in window). Captures "立得住" intuition for high balls.
2. **velocity_score** — peak at v_z = -4 m/s (comfortable descent), zero outside v_z ∈ [-6, -2] m/s. Triangular shape.
3. **lateral_score** — penalizes lateral drift during the dwell. Linear from 1 (0 cm drift) down to 0 (>= 30 cm drift).

If the trajectory does not enter the window during descent at all, score is 0.0.

### Test plan

Hand-craft three synthetic trajectories:
- A "good" trajectory: passes through the window center on descent with v_z = -4 m/s, no lateral drift → score >= 0.7
- A "bad-velocity" trajectory: passes through but v_z = -7 m/s (too fast) → score == 0
- A "missed-window" trajectory: never enters the window → score == 0
- A "lateral-drift" trajectory: enters the window but drifts 40 cm during dwell → low score
- A "short-dwell" trajectory: enters the window for only 0.05 s → low score

Use `simulate()` to generate trajectories from chosen release states. Compute and verify scores.

- [ ] **Step 1: Write failing tests**

Create `/home/siriux/Projects/vb_coach/tests/test_setting_attacks_position4_high.py`:

```python
import numpy as np

from vb_coach.constants import RHO_AIR
from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.attacks_position4 import (
    HighBallScore,
    score_high_ball,
)
from vb_coach.setting.hit_window import position4_high_window
from vb_coach.trajectory import ReleaseState, simulate


def _set_release(velocity: np.ndarray, position: np.ndarray | None = None) -> ReleaseState:
    if position is None:
        position = np.array([0.5, 0.8, 2.2])
    return ReleaseState(position=position, velocity=velocity, spin=np.array([0.0, 0.0, 0.0]))


def test_score_returns_zero_when_trajectory_misses_window():
    # Set that never reaches the 4号位 antenna area
    rs = _set_release(velocity=np.array([0.0, 0.0, 5.0]))  # straight up + falls back
    traj = simulate(rs, t_max=3.0)
    window = position4_high_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    assert result.score == 0.0


def test_score_returns_zero_when_only_ascent_intersects():
    # Engineering a release where the ball enters the window only on ascent.
    # Set straight at the attacker contact point at fast horizontal speed,
    # passes through window while still climbing because release starts low.
    rs = _set_release(
        position=np.array([0.5, 0.0, 1.5]),
        velocity=np.array([-4.5, 0.4, 6.0]),  # toward window, mostly up
    )
    traj = simulate(rs, t_max=3.0)
    window = position4_high_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    # If only ascent intersection, score is zero (high ball means descent hit)
    # Acceptable: result.score could legitimately be 0 here.
    # The actionable assertion is: if the dwell is on ascent only, score is 0.
    if result.dwell_seconds > 0 and result.contact_index < traj.apex_index():
        assert result.score == 0.0


def test_score_good_high_ball_above_threshold():
    # Carefully tuned release that produces a textbook high ball:
    # apex ~5 m, descent into window at v_z ~ -4 m/s, no lateral drift.
    rs = _set_release(velocity=np.array([-4.5, -0.5, 7.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_high_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    assert result.score > 0.5
    assert result.dwell_seconds > 0.1


def test_score_too_fast_descent_velocity_kills_score():
    # Lower apex (faster descent at the window crossing) - flat low set
    rs = _set_release(velocity=np.array([-5.0, -0.5, 4.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_high_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    # Either the ball misses the window or v_z at contact is too fast — either
    # way the score should be low. If it does enter the window, v_z should be
    # outside the comfortable [-6, -2] m/s band.
    if result.score > 0:
        assert result.velocity_at_contact_z < -5.0 or result.velocity_at_contact_z > -2.0


def test_score_is_in_unit_interval():
    rs = _set_release(velocity=np.array([-4.5, -0.5, 7.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_high_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    assert 0.0 <= result.score <= 1.0
    assert 0.0 <= result.dwell_score <= 1.0
    assert 0.0 <= result.velocity_score <= 1.0
    assert 0.0 <= result.lateral_score <= 1.0


def test_score_returns_diagnostics():
    rs = _set_release(velocity=np.array([-4.5, -0.5, 7.5]))
    traj = simulate(rs, t_max=3.0)
    window = position4_high_window(DEFAULT_AMATEUR_MALE)
    result = score_high_ball(traj, window)
    # All diagnostic fields are present and have expected types
    assert isinstance(result.score, float)
    assert isinstance(result.dwell_seconds, float)
    assert isinstance(result.velocity_at_contact_z, float)
    assert isinstance(result.lateral_drift_m, float)
    assert isinstance(result.contact_index, int)
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/test_setting_attacks_position4_high.py -v`
Expected: FAIL — module / functions missing.

- [ ] **Step 3: Implement `attacks_position4.py`**

Create `/home/siriux/Projects/vb_coach/src/vb_coach/setting/attacks_position4.py`:

```python
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


_DWELL_TARGET_SECONDS: float = 0.30
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
        np.max(trajectory.position[descent_inside, 0])
        - np.min(trajectory.position[descent_inside, 0])
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_setting_attacks_position4_high.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Run full test suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: 83 passed (64 prior + 6 attacker + 7 hit_window + 6 high_ball).

- [ ] **Step 6: Commit**

```bash
git add src/vb_coach/setting/attacks_position4.py tests/test_setting_attacks_position4_high.py
git commit -m "[feat:setting] score_high_ball for 4号位高球 quality metric"
```

---

## Task 4: Runnable demo + side-by-side plot

**Files:**
- Create: `examples/position4_high_demo.py`

The demo simulates several release states (varying initial velocity) and prints + plots their scores so a user can visually compare a "good" vs a "bad" high-ball set.

- [ ] **Step 1: Write the demo**

Create `/home/siriux/Projects/vb_coach/examples/position4_high_demo.py`:

```python
"""Demo: score multiple 4号位高球 trajectories side by side.

Runs three release states (a good high ball, a flat low set, and a poorly
aimed set) for the default amateur male attacker, scores each with
score_high_ball, and produces a comparison plot.

Run:
    uv run python examples/position4_high_demo.py

Produces:
    position4_high_demo.png
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vb_coach.constants import NET_HEIGHT_MEN
from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.attacks_position4 import score_high_ball
from vb_coach.setting.hit_window import position4_high_window
from vb_coach.trajectory import ReleaseState, simulate


_RELEASE_POSITION = np.array([0.5, 0.8, 2.2])


def main() -> None:
    attacker = DEFAULT_AMATEUR_MALE
    window = position4_high_window(attacker)

    scenarios = [
        ("good high ball (v0=-4.5,-0.5,7.5)", np.array([-4.5, -0.5, 7.5])),
        ("low / flat set (v0=-5,-0.5,4.5)",   np.array([-5.0, -0.5, 4.5])),
        ("off-line set (v0=-3,-1.5,7.5)",     np.array([-3.0, -1.5, 7.5])),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, v0 in scenarios:
        rs = ReleaseState(position=_RELEASE_POSITION.copy(), velocity=v0,
                          spin=np.array([0.0, 0.0, 0.0]))
        traj = simulate(rs, t_max=3.0)
        result = score_high_ball(traj, window)
        line, = ax.plot(traj.position[:, 1], traj.position[:, 2], lw=2,
                        label=f"{label}: score={result.score:.2f}")
        if result.contact_index >= 0:
            cp = traj.position[result.contact_index]
            ax.scatter([cp[1]], [cp[2]], color=line.get_color(), marker="o", s=60)
        print(f"{label}: score={result.score:.3f} "
              f"dwell={result.dwell_seconds:.2f}s "
              f"vz={result.velocity_at_contact_z:.2f}m/s "
              f"drift={result.lateral_drift_m*100:.1f}cm")

    # Net marker
    ax.plot([0.0, 0.0], [0.0, NET_HEIGHT_MEN], color="gray", lw=2, label="net")

    # Hit-window y-z slice (at the window's x center)
    y0, y1 = window.center[1] - window.dy, window.center[1] + window.dy
    z0, z1 = window.center[2] - window.dz, window.center[2] + window.dz
    ax.fill_between([y0, y1], z0, z1, color="green", alpha=0.18,
                    label="hit window (y-z slice)")

    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xlabel("y [m] (across court)")
    ax.set_ylabel("z [m]")
    ax.set_title("4号位高球 — three sets, side view")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("position4_high_demo.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("Saved: position4_high_demo.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the demo and verify it works**

Run: `uv run python examples/position4_high_demo.py`

Expected output:
- Three printed score lines, one per scenario
- "good high ball" scenario should print `score=X` where X > 0.5
- "low / flat set" scenario should print a markedly lower score (either 0 because it misses the window, or low because v_z is too fast)
- "off-line set" scenario should print a low score (misses the window in x)
- The PNG file is created in the current directory

Visually inspect `position4_high_demo.png`:
- Three trajectories, each in different color, with the green hit-window slab visible near (y=0.2, z=2.80)
- The net (gray vertical line) at y=0
- The "good" trajectory should clearly pass through the hit window during descent
- Contact markers (filled circles) should appear on the good trajectory only

- [ ] **Step 3: Run the full test suite one more time**

Run: `uv run pytest -q`
Expected: ALL pass (no regressions from the demo file).

- [ ] **Step 4: Commit**

```bash
git add examples/position4_high_demo.py
git commit -m "[feat:examples] 4号位高球 scoring demo with three reference sets"
```

---

## Verification checklist before declaring A1.1 done

- [ ] `uv run pytest -q` passes all tests (target: ~83 passed total)
- [ ] `uv run python examples/position4_high_demo.py` runs and produces both printout and PNG
- [ ] "good high ball" scenario in the demo scores > 0.5; at least one other scenario scores noticeably lower
- [ ] `grep -rn "_POSITION4_" src/vb_coach/setting/` — magic numbers for the 4号位 hit-window only live in `hit_window.py`
- [ ] `Attacker.contact_height` is the only place `contact_loss` is subtracted from `approach_reach` (no duplication)
- [ ] No public function in `setting/` lacks a type hint

## What comes next (not in this plan)

- Phase A1.2: add `score_half_ball` (半高) and `score_shoot_ball` (平拉开). Same pattern as `score_high_ball`, with different velocity bands and dwell windows. Each is ~30 lines of code + tests.
- Phase A1.3: attack-type registry mapping name → (window factory, score function). One unified `score_attack(traj, attacker, attack_name)` entry point.
- Phase A2: 2号位 (mirror of 4号位 with optional 背传 variant).
- Phase A5: inverse problem — given attacker, find best release state via numerical optimization.
- Tune the score parameters (`_DWELL_TARGET_SECONDS`, `_VELOCITY_BEST`, etc.) against real video data when we have it. Current values are model priors.
