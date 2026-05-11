# HitWindow Refactor: Asymmetric Ellipsoid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current axis-aligned box `HitWindow` (one per attack type, with per-attack dimensions) with a literature-backed **non-symmetric ellipsoid** (one per attacker, shared across attack types). Eliminates over-parameterization where window dimensions and score-function thresholds were both encoding the same tactical-tolerance intuition.

**Architecture:** New `HitWindow` dataclass with 5 named asymmetric radii (`r_x`, `r_y_front`, `r_y_back`, `r_z_up`, `r_z_down`). Single factory `position4_hit_window(attacker)`. All three score functions take the same window. Plotly viz renders the ellipsoid as a parametric Mesh3d with piecewise scaling.

**Tech Stack:** Same as before. No new deps.

**Scope decisions:**
- **Model B (asymmetric axis-aligned ellipsoid)** per the literature review. Models C/D (swing-arc shell) deferred — they need swing-plane orientation data that the current `Attacker` doesn't carry.
- **Single factory per court position** — `position4_hit_window(attacker)` returns the ONE window all three attack types share.
- **Lateral radius kept at 0.40 m** (between Wagner's 0.20 and current 0.50) — reflects amateur footwork tolerance on top of Wagner's pure-contact tolerance.
- **Old `HitWindow` API removed** — `dx`/`dy`/`dz` field names are replaced. This breaks callers; update all of them in this same plan.
- **Old per-attack window factories removed** — `position4_high_window`, `position4_half_window`, `position4_shoot_window` collapse into one.

**Out of scope:**
- Body-frame rotation (Model C/D) — needs Attacker to carry a swing-plane vector
- 2号位 / 3号位 / back-row window factories (these come in their respective phases)
- Recalibration of score thresholds — those stay model prior

---

## File structure changes

```
src/vb_coach/setting/
  hit_window.py                    REWRITTEN: new HitWindow + position4_hit_window only
  attacks_position4.py             score functions now take the unified window
src/vb_coach/viz/
  plotly3d.py                      _hit_window_trace renders ellipsoid mesh
tests/
  test_setting_hit_window.py       UPDATED for new API
  test_setting_attacks_position4_high.py    UPDATED (use position4_hit_window)
  test_setting_attacks_position4_half.py    UPDATED
  test_setting_attacks_position4_shoot.py   UPDATED
  test_viz_plotly3d.py             UPDATED
examples/
  position4_high_demo.py           UPDATED
  position4_all_attacks_demo.py    UPDATED
  interactive_attacks_demo.py      UPDATED
app.py                             UPDATED
```

## Key design parameters (literature-backed, see `reference_hit_window_biomechanics.md`)

For 4号位 amateur male attacker (height 1.78m, approach reach 2.95m):

| Radius | Value | Source / Rationale |
|---|---|---|
| `r_x` (along net, symmetric) | 0.40 m | Wagner 2020 lateral 0.20 m + ~0.20 m amateur footwork |
| `r_y_front` (toward net) | 0.30 m | "12-18 in. forward of shoulder" + Wagner posterior penalty |
| `r_y_back` (away from net) | 0.15 m | Wagner: posterior 20 cm offset → muscle activation +70% |
| `r_z_up` (above contact_height) | 0.05 m | Coaching consensus: hit at peak reach; max-reach is hard upper bound |
| `r_z_down` (below contact_height) | 0.30 m | Coaching: can dip jump or contact later; widest tolerance |

In 4号位 world coords, "toward net" = -y direction (team A side). So `r_y_front` applies when `(p[1] - center[1]) < 0`.

Center remains at `(-4.0, 0.2, attacker.contact_height)` for the moment — this is consistent with prior tasks. Could refine to `(-4.0, 0.4, contact_height)` (further from net by 0.2m to match "ball-in-front-of-shoulder") but the score function logic already filters to descent; for now keep center the same.

---

## Task 1: New `HitWindow` dataclass

Rewrite `hit_window.py` with the asymmetric-ellipsoid API.

**Files:**
- Modify (rewrite): `src/vb_coach/setting/hit_window.py`
- Modify: `tests/test_setting_hit_window.py`

### Step 1: Write failing tests (replacing existing tests)

Completely rewrite `/home/siriux/Projects/vb_coach/tests/test_setting_hit_window.py`:

```python
import math

import numpy as np
import pytest

from vb_coach.setting.attacker import DEFAULT_AMATEUR_MALE
from vb_coach.setting.hit_window import HitWindow, position4_hit_window


def test_hit_window_contains_center():
    w = HitWindow(
        center=np.array([0.0, 0.0, 2.8]),
        r_x=0.40, r_y_front=0.30, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
    )
    assert w.contains(np.array([0.0, 0.0, 2.8]))


def test_hit_window_excludes_far_outside():
    w = HitWindow(
        center=np.array([0.0, 0.0, 2.8]),
        r_x=0.40, r_y_front=0.30, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
    )
    assert not w.contains(np.array([1.0, 0.0, 2.8]))
    assert not w.contains(np.array([0.0, 1.0, 2.8]))
    assert not w.contains(np.array([0.0, 0.0, 4.0]))


def test_hit_window_front_asymmetry():
    # r_y_front = 0.30 (forward = -y direction from center)
    # r_y_back  = 0.15 (backward = +y direction)
    # A point 0.25 m in -y direction is INSIDE (within front radius);
    # a point 0.25 m in +y direction is OUTSIDE (beyond back radius).
    w = HitWindow(
        center=np.array([0.0, 0.0, 2.8]),
        r_x=0.40, r_y_front=0.30, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
    )
    assert w.contains(np.array([0.0, -0.25, 2.8]))      # 0.25 m forward → inside
    assert not w.contains(np.array([0.0, +0.20, 2.8]))  # 0.20 m back → outside (>0.15)


def test_hit_window_vertical_asymmetry():
    # r_z_up = 0.05 (above), r_z_down = 0.30 (below)
    # 0.20 m below center is INSIDE; 0.10 m above is OUTSIDE.
    w = HitWindow(
        center=np.array([0.0, 0.0, 2.8]),
        r_x=0.40, r_y_front=0.30, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
    )
    assert w.contains(np.array([0.0, 0.0, 2.60]))       # 0.20 m below → inside
    assert not w.contains(np.array([0.0, 0.0, 2.90]))   # 0.10 m above → outside


def test_hit_window_contains_batch():
    w = HitWindow(
        center=np.array([0.0, 0.0, 2.8]),
        r_x=0.40, r_y_front=0.30, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
    )
    points = np.array([
        [0.0, 0.0, 2.8],     # inside (center)
        [0.0, -0.25, 2.8],   # inside (forward)
        [0.0, +0.20, 2.8],   # outside (too far back)
        [0.0, 0.0, 2.90],    # outside (above max reach)
        [0.0, 0.0, 2.60],    # inside (below center)
    ])
    mask = w.contains_batch(points)
    np.testing.assert_array_equal(mask, np.array([True, True, False, False, True]))


def test_hit_window_rejects_negative_radii():
    with pytest.raises(ValueError):
        HitWindow(
            center=np.array([0.0, 0.0, 2.8]),
            r_x=-0.1, r_y_front=0.3, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
        )


def test_hit_window_rejects_wrong_center_shape():
    with pytest.raises(ValueError):
        HitWindow(
            center=np.array([0.0, 2.8]),
            r_x=0.40, r_y_front=0.30, r_y_back=0.15, r_z_up=0.05, r_z_down=0.30,
        )


def test_position4_hit_window_centered_at_contact_height():
    a = DEFAULT_AMATEUR_MALE
    w = position4_hit_window(a)
    assert math.isclose(w.center[2], a.contact_height, abs_tol=1e-9)


def test_position4_hit_window_at_antenna_position():
    a = DEFAULT_AMATEUR_MALE
    w = position4_hit_window(a)
    # 4号位 attack: x near -4, y just on team A's side (small positive)
    assert -4.5 < w.center[0] < -3.5
    assert 0.0 < w.center[1] < 0.5


def test_position4_hit_window_amateur_radii_match_literature():
    # Per reference_hit_window_biomechanics.md: r_y_front > r_y_back,
    # r_z_down > r_z_up, r_x in 0.30..0.50 range.
    a = DEFAULT_AMATEUR_MALE
    w = position4_hit_window(a)
    assert w.r_y_front > w.r_y_back     # front-back asymmetry (Wagner + coaching)
    assert w.r_z_down > w.r_z_up        # vertical asymmetry (max reach upper bound)
    assert 0.25 <= w.r_x <= 0.50        # lateral with footwork tolerance
    assert w.r_z_up <= 0.10             # tight above-reach (peak-reach hard limit)


def test_position4_hit_window_taller_attacker_higher_center():
    from vb_coach.setting.attacker import Attacker
    short_a = Attacker(height=1.65, standing_reach=2.18, approach_reach=2.55)
    tall_a = Attacker(height=1.95, standing_reach=2.55, approach_reach=3.20)
    short_w = position4_hit_window(short_a)
    tall_w = position4_hit_window(tall_a)
    assert tall_w.center[2] > short_w.center[2]
    # Radii should not depend on attacker (yet — that's a future per-attacker calibration)
    assert short_w.r_x == tall_w.r_x
    assert short_w.r_y_front == tall_w.r_y_front
```

### Step 2: Run tests, verify fail

Run: `uv run pytest tests/test_setting_hit_window.py -v`
Expected: failures because the new HitWindow signature doesn't match.

### Step 3: Rewrite `hit_window.py`

Replace the entire contents of `/home/siriux/Projects/vb_coach/src/vb_coach/setting/hit_window.py` with:

```python
"""Hit-window geometry for attack-side analysis.

Models the **effective-contact envelope** around an attacker's optimal
contact point as an **asymmetric axis-aligned ellipsoid**. The envelope
captures where contact with the ball produces a viable spike — not just
where the hand can geometrically reach.

The asymmetry is literature-backed (see project memory
`reference_hit_window_biomechanics.md`):
  - Wagner et al. 2020: ±20 cm horizontal deviations from optimum cause
    ~70% increase in shoulder muscle activation (effective tolerance scale).
  - Coaching consensus + Wagner posterior penalty: contact tolerance is
    larger toward the net (forward, in the direction of arm swing) than
    behind the optimum.
  - Coaching consensus: max approach reach is a hard upper bound (cannot
    contact above peak); below the optimum, attacker can lower the jump
    or contact later in the arc, so down-tolerance is larger.

The body's facing direction is implicitly "toward the net" — i.e., -y for
a team-A attacker. So "forward" half-axis applies when (p[1] - center[1])
is negative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vb_coach.setting.attacker import Attacker


# 4号位 contact center: near left antenna, just on team A's side of net.
_POSITION4_CONTACT_X: float = -4.0
_POSITION4_CONTACT_Y: float = 0.2

# Amateur-male radii (literature-anchored; see memory reference for derivation):
_AMATEUR_R_X: float = 0.40        # along net; Wagner 0.20 + ~0.20 footwork tolerance
_AMATEUR_R_Y_FRONT: float = 0.30  # toward net (forward of attacker)
_AMATEUR_R_Y_BACK: float = 0.15   # away from net (behind attacker)
_AMATEUR_R_Z_UP: float = 0.05     # above contact_height (peak reach is upper bound)
_AMATEUR_R_Z_DOWN: float = 0.30   # below contact_height (can drop contact)


@dataclass
class HitWindow:
    """Asymmetric axis-aligned ellipsoidal effective-contact envelope.

    A point p is inside iff
        (dx / r_x)^2 + (dy / r_y_eff)^2 + (dz / r_z_eff)^2 <= 1
    where:
        dx, dy, dz = p - center
        r_y_eff   = r_y_front if dy < 0 else r_y_back   (forward = -y, world frame)
        r_z_eff   = r_z_up    if dz > 0 else r_z_down

    The piecewise scaling produces a smooth half-ellipsoid in each octant of
    the y-z plane, stitched at the plane y=center[1] and the plane
    z=center[2]. The result is C0 (position-continuous) but not C1.
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


def position4_hit_window(attacker: Attacker) -> HitWindow:
    """Hit-window for a 4号位 attack by the given attacker.

    Center sits at the standard 4号位 contact location (near left antenna,
    just on team A's side of the net), at the attacker's contact_height.
    Radii use amateur-male defaults from the biomechanics literature; for
    now these do not vary by attacker beyond the contact-height shift.

    The same window applies to all 4号位 attack types (高球 / 半高 / 平拉开)
    — the difference between attack types lives entirely in the score
    functions' velocity / dwell / drift thresholds, not in geometry.
    """
    center = np.array([
        _POSITION4_CONTACT_X,
        _POSITION4_CONTACT_Y,
        attacker.contact_height,
    ])
    return HitWindow(
        center=center,
        r_x=_AMATEUR_R_X,
        r_y_front=_AMATEUR_R_Y_FRONT,
        r_y_back=_AMATEUR_R_Y_BACK,
        r_z_up=_AMATEUR_R_Z_UP,
        r_z_down=_AMATEUR_R_Z_DOWN,
    )
```

**The old factories** (`position4_high_window`, `position4_half_window`, `position4_shoot_window`) are now **removed**. All callers must switch to `position4_hit_window`.

### Step 4: Run tests, verify pass

Run: `uv run pytest tests/test_setting_hit_window.py -v`
Expected: 11 passed.

Other test files (score function tests, viz tests, app smoke test) will fail at this point — that's expected and will be fixed in subsequent tasks.

### Step 5: Commit

```bash
git add src/vb_coach/setting/hit_window.py tests/test_setting_hit_window.py
git commit -m "[refactor:setting] HitWindow as asymmetric ellipsoid (Wagner 2020 + coaching)"
```

---

## Task 2: Update score functions to use unified window

The three score functions (`score_high_ball`, `score_half_ball`, `score_shoot_ball`) currently accept any `HitWindow`. The new ellipsoid window has different field names but the public predicates (`contains`, `contains_batch`) are unchanged — so score functions need NO logic changes. However, **the tests** call `position4_high_window` / `position4_half_window` / `position4_shoot_window` which no longer exist. Update all three test files to use the new factory.

**Files:**
- Modify: `tests/test_setting_attacks_position4_high.py`
- Modify: `tests/test_setting_attacks_position4_half.py`
- Modify: `tests/test_setting_attacks_position4_shoot.py`

### Step 1: Replace per-attack window imports with the unified factory

In each of the three test files, change:

```python
from vb_coach.setting.hit_window import position4_high_window   # or half / shoot
```

to:

```python
from vb_coach.setting.hit_window import position4_hit_window
```

And replace all uses (`window = position4_high_window(DEFAULT_AMATEUR_MALE)`, etc.) with `window = position4_hit_window(DEFAULT_AMATEUR_MALE)`.

### Step 2: Run the three score-function test files

Run: `uv run pytest tests/test_setting_attacks_position4_high.py tests/test_setting_attacks_position4_half.py tests/test_setting_attacks_position4_shoot.py -v`

Expected: most should still pass because the trajectories pass through the central region of the ellipsoid (which is more restrictive than the box at corners but still contains the typical descent paths). HOWEVER, some "good ball" scenarios may now score below their threshold (e.g., `> 0.5` for high-ball) because the new ellipsoid is geometrically tighter than the box.

If any "good ball" test fails:
- Retune the test scenario velocity until it scores above the threshold
- OR (preferred) relax the threshold (`> 0.5` → `> 0.4`) to acknowledge the geometry change
- Document any retuning in the commit message

**Goal: all 18 score-function tests pass.** Adjust scenarios or thresholds as needed.

### Step 3: Commit

```bash
git add tests/test_setting_attacks_position4_high.py tests/test_setting_attacks_position4_half.py tests/test_setting_attacks_position4_shoot.py
git commit -m "[refactor:setting] score function tests use unified position4_hit_window"
```

---

## Task 3: Update Plotly viz to render ellipsoid

The `_hit_window_trace` in `src/vb_coach/viz/plotly3d.py` currently builds a box mesh. Replace with a parametric ellipsoid mesh that handles the asymmetric scaling.

**Files:**
- Modify: `src/vb_coach/viz/plotly3d.py`
- Modify: `tests/test_viz_plotly3d.py`

### Step 1: Rewrite `_hit_window_trace`

Replace the existing function body with:

```python
def _hit_window_trace(window: HitWindow, n_phi: int = 24, n_theta: int = 16) -> go.Mesh3d:
    """Return a translucent surface mesh for the asymmetric-ellipsoid hit window.

    Builds an axis-aligned ellipsoidal surface in spherical parametric form,
    then applies piecewise scaling so the +y and -y hemispheres use different
    radii (r_y_back vs r_y_front), and similarly for +z and -z (r_z_up vs
    r_z_down). The result is a closed surface stitched along the y=0 and z=0
    body-frame planes.
    """
    # Spherical parametric grid
    theta = np.linspace(0.0, np.pi, n_theta)          # polar (0 = +z, pi = -z)
    phi = np.linspace(0.0, 2 * np.pi, n_phi)          # azimuth around z
    THETA, PHI = np.meshgrid(theta, phi)
    # Unit sphere
    x_unit = np.sin(THETA) * np.cos(PHI)
    y_unit = np.sin(THETA) * np.sin(PHI)
    z_unit = np.cos(THETA)
    # Piecewise scaling
    x_s = window.r_x * x_unit
    y_s = np.where(y_unit < 0, window.r_y_front * y_unit, window.r_y_back * y_unit)
    z_s = np.where(z_unit > 0, window.r_z_up * z_unit, window.r_z_down * z_unit)
    # Translate to center, flatten
    cx, cy, cz = float(window.center[0]), float(window.center[1]), float(window.center[2])
    x = (x_s + cx).flatten()
    y = (y_s + cy).flatten()
    z = (z_s + cz).flatten()
    # Triangle indices for the meshed grid
    i_idx: list[int] = []
    j_idx: list[int] = []
    k_idx: list[int] = []
    for a in range(n_phi - 1):
        for b in range(n_theta - 1):
            p00 = a * n_theta + b
            p01 = a * n_theta + (b + 1)
            p10 = (a + 1) * n_theta + b
            p11 = (a + 1) * n_theta + (b + 1)
            i_idx.extend([p00, p00])
            j_idx.extend([p10, p11])
            k_idx.extend([p11, p01])
    return go.Mesh3d(
        x=x, y=y, z=z, i=i_idx, j=j_idx, k=k_idx,
        color="green", opacity=0.18, name="hit window",
        hoverinfo="skip",
    )
```

### Step 2: Update viz tests

The existing `test_plot_attack_3d_with_window_and_attacker` uses `position4_high_window`. Update to `position4_hit_window`:

In `/home/siriux/Projects/vb_coach/tests/test_viz_plotly3d.py`, replace any `position4_high_window` import and usage with `position4_hit_window`.

### Step 3: Run viz tests

Run: `uv run pytest tests/test_viz_plotly3d.py -v`
Expected: all 5 still pass. The window mesh is now an ellipsoid, but the smoke tests only check trace counts and basic properties.

### Step 4: Manually verify by generating an HTML

Run: `uv run python examples/interactive_attacks_demo.py` and open one of the HTML files. The "hit window" should now be a smooth, asymmetric ellipsoid (squashed at top, longer toward -y) instead of a box.

Update the demo file in the next task — but at this point the existing demo will fail because it imports the removed `position4_high_window` etc. Move to Task 4.

### Step 5: Commit

```bash
git add src/vb_coach/viz/plotly3d.py tests/test_viz_plotly3d.py
git commit -m "[refactor:viz] hit-window trace renders asymmetric ellipsoid mesh"
```

---

## Task 4: Update demos and Streamlit app

All demos and `app.py` import the old per-attack factories. Replace with the unified factory.

**Files:**
- Modify: `examples/position4_high_demo.py`
- Modify: `examples/position4_all_attacks_demo.py`
- Modify: `examples/interactive_attacks_demo.py`
- Modify: `app.py`

### Step 1: Update each file

In each file, replace all imports of `position4_high_window`, `position4_half_window`, `position4_shoot_window` with `position4_hit_window`. Replace all uses accordingly.

For `app.py` and `position4_all_attacks_demo.py` which currently use **multiple** windows (one per attack type), simplify to use **one** window for all three attack types.

Example for `app.py`'s `_ATTACK_PRESETS`:

```python
_ATTACK_PRESETS = {
    "高球 (high)":   {"position": (0.5, 0.8, 2.2),  "velocity": (-4.8, -0.5, 5.5),
                      "score_fn": score_high_ball},
    "半高 (half)":   {"position": (0.5, 0.8, 2.2),  "velocity": (-6.5, -0.5, 4.5),
                      "score_fn": score_half_ball},
    "平拉开 (shoot)": {"position": (-1.5, 0.5, 2.3), "velocity": (-7.5, -0.5, 3.5),
                      "score_fn": score_shoot_ball},
}
```

(Drop the `window_fn` key — all attacks use `position4_hit_window(attacker)`.)

And later in `app.py`'s `main()`:

```python
window = position4_hit_window(attacker)   # one window for all attacks
score = preset["score_fn"](traj, window)
```

### Step 2: Run each demo

```bash
uv run python examples/position4_high_demo.py
uv run python examples/position4_all_attacks_demo.py
uv run python examples/interactive_attacks_demo.py
```

Each should run without error. Check that the printed scores haven't drastically changed (a small change is expected because of the new ellipsoid geometry; large drops likely mean test scenarios need re-tuning).

### Step 3: Run the streamlit smoke test

Run: `uv run pytest tests/test_app_smoke.py -v`
Expected: 2 passed.

### Step 4: Run the full test suite

Run: `uv run pytest -q`
Expected: total count likely similar to before but with possibly 1-2 retuned scenarios. Aim for all green.

### Step 5: Commit

```bash
git add examples/ app.py
git commit -m "[refactor:examples] demos and app use unified position4_hit_window"
```

---

## Task 5: Verification + DEVLOG entry

### Step 1: Final test pass

Run: `uv run pytest -q`
Expected: ALL tests pass.

### Step 2: Visual verification

Run: `uv run streamlit run app.py --server.headless true --server.port 8501 &`
Then `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501/`
Expected: `200`.

Kill: `pkill -f "streamlit run app.py"`

### Step 3: Generate updated HTML demos and visually verify (optional)

Run: `uv run python examples/interactive_attacks_demo.py`

Open one of the generated HTML files. The hit window should now be a green ellipsoid that:
- Is wider toward the net (-y direction) than away from it (+y direction)
- Has very small vertical extent above the center (almost flat top)
- Has substantial extent below the center

### Step 4: Update DEVLOG

Append a new entry to `docs/DEVLOG.md` (at the TOP of the file, since it's reverse-chronological) describing the refactor's actual outcome — including any scenario retuning or threshold adjustments needed.

### Step 5: Commit DEVLOG update

```bash
git add docs/DEVLOG.md
git commit -m "[docs] DEVLOG entry for HitWindow refactor completion"
```

---

## Verification checklist before declaring refactor done

- [ ] `uv run pytest -q` passes ALL tests
- [ ] `uv run streamlit run app.py` starts and serves localhost:8501
- [ ] `examples/position4_all_attacks_demo.py` still shows diagonal-dominance pattern (each set type scores highest in its matched scorer)
- [ ] `grep -r "position4_high_window\|position4_half_window\|position4_shoot_window" src/ tests/ examples/ app.py` returns NOTHING (the three old factories are fully removed)
- [ ] `grep -rn "dx\|dy\|dz" src/vb_coach/setting/hit_window.py` returns nothing matching the old field names (verify HitWindow has fully transitioned to the new API)
- [ ] One reference to `reference_hit_window_biomechanics.md` (or Wagner 2020) in `hit_window.py`'s module docstring

## What comes next (not in this plan)

- Phase A2: 2号位 with `position2_hit_window(attacker)` — mirror of 4号位 in x
- Phase A3: 3号位 with its own window (apex contact for quicks)
- Body-frame swing-plane orientation in Attacker — Model C/D upgrade if needed
- Per-attacker calibration of radii (taller attacker → wider lateral footwork?)
