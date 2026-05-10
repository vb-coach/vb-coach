# Phase A0: Trajectory Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python library that takes a volleyball release state (3D position, velocity, spin vector) and produces an accurate trajectory under gravity, drag, and Magnus forces, with basic plotting for sanity checks.

**Architecture:** numpy + scipy.integrate.solve_ivp for ODE integration. Aerodynamic coefficients (Cd, Cl) are isolated in a single module so they can be swapped for regime-correct functions later when the subcritical-aero research returns. Each physical concept gets its own module with one responsibility. ReleaseState and Trajectory are dataclasses; positions/velocities/spins are np.ndarray of shape (3,) (single state) or (N, 3) (trajectory).

**Tech Stack:** Python 3.11+, numpy, scipy, matplotlib, pytest, uv (package manager). No web/JS — that's the visualization phase later.

**Out of scope (deferred to later plans):**
- Court geometry and net interaction (added in attack-type plans A1–A4)
- Hit-window geometry and attacker model (Phase A5)
- Regime-dependent Cd(Re) — current plan ships a constant-Cd placeholder with a swap-in interface
- Panel-orientation lateral force / knuckle-set effects (depends on aero research)
- Phase B setter-biomechanics work

---

## File structure

After this plan ships, the repo will look like:

```
vb_coach/
├── pyproject.toml
├── .python-version
├── .gitignore
├── src/
│   └── vb_coach/
│       ├── __init__.py
│       ├── constants.py        # Air/ball/gravity defaults
│       ├── coords.py           # Coordinate system convention (doc + tiny helpers)
│       ├── aerodynamics.py     # Cd(Re), Cl(S), Reynolds, spin parameter
│       ├── physics.py          # Force model: gravity + drag + Magnus
│       ├── trajectory.py       # ReleaseState, Trajectory, simulate()
│       └── plotting.py         # Side-view + 3D plots
├── tests/
│   ├── __init__.py
│   ├── test_constants.py
│   ├── test_coords.py
│   ├── test_aerodynamics.py
│   ├── test_physics.py
│   ├── test_trajectory.py
│   └── test_integration.py     # End-to-end + analytic verification
└── examples/
    └── basic_set.py            # Runnable demo: setter → 4号位高球
```

**Module responsibilities:**
- `constants.py` holds only literal values (gravity, air density, ball mass, ball diameter, etc.). No logic.
- `coords.py` documents the world coordinate frame (origin under net center, x along net, y across court, z up) and provides any tiny helpers like `mirror_court(point)` if needed by tests.
- `aerodynamics.py` exposes pure functions `reynolds(v, D, nu)`, `spin_parameter(omega, v, R)`, `cd(re)`, `cl(s)`. These are the swap points when regime-aware models arrive.
- `physics.py` exposes pure force functions taking np.ndarray inputs.
- `trajectory.py` is the entry point: `simulate(release_state, params, t_max)` → `Trajectory`.
- `plotting.py` takes a Trajectory and produces matplotlib figures. Importing matplotlib is isolated here so the core library has no plot dep at import time.

---

## Coordinate system convention

World frame (right-handed):
- **Origin:** point on the floor directly below the net center
- **x-axis:** along the net, from setter's left antenna toward right antenna (positive x = right)
- **y-axis:** perpendicular to net, positive into team A's court (negative into team B's)
- **z-axis:** vertical, positive up

So a typical setter at zone 2/3 transition on team A: position ≈ (0.0, 0.8, 2.2) m at release. 4号位 antenna (team A's left front, opposite team B's antenna location at x = -4.5): the attacker hits near (-4.0, 0.5, ball_contact_height).

Spin vector ω uses right-hand rule. A pure back-spin set traveling in +x direction has ω pointing in +y direction (ω × v_hat points up → Magnus lifts the ball).

---

## Task 1: Project scaffold and constants module

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `src/vb_coach/__init__.py`
- Create: `src/vb_coach/constants.py`
- Create: `tests/__init__.py`
- Create: `tests/test_constants.py`

- [ ] **Step 1: Initialize git and uv project**

Run:
```bash
cd /home/siriux/Projects/vb_coach
git init
uv init --package --name vb_coach --python 3.11
```

If `uv` is not installed: `curl -LsSf https://astral.sh/uv/install.sh | sh` then re-source shell.

Expected: `pyproject.toml`, `.python-version`, `src/vb_coach/__init__.py` created by uv.

- [ ] **Step 2: Add runtime and dev dependencies**

Run:
```bash
uv add numpy scipy matplotlib
uv add --dev pytest
```

Expected: deps added to `pyproject.toml`, `uv.lock` created.

- [ ] **Step 3: Write `.gitignore`**

Create `/home/siriux/Projects/vb_coach/.gitignore`:

```
__pycache__/
*.pyc
.pytest_cache/
.venv/
dist/
build/
*.egg-info/
.coverage
htmlcov/
.DS_Store
```

- [ ] **Step 4: Write failing test for constants**

Create `/home/siriux/Projects/vb_coach/tests/__init__.py` (empty file).

Create `/home/siriux/Projects/vb_coach/tests/test_constants.py`:

```python
import vb_coach.constants as C


def test_gravity_magnitude():
    assert C.G == 9.81  # m/s^2, sea level


def test_air_density_room_temp():
    assert 1.15 < C.RHO_AIR < 1.25  # kg/m^3 at ~20°C


def test_air_kinematic_viscosity():
    assert 1.4e-5 < C.NU_AIR < 1.6e-5  # m^2/s at ~20°C


def test_ball_diameter_volleyball():
    # FIVB: 65-67 cm circumference -> diameter ~20.7-21.3 cm
    assert 0.205 < C.BALL_DIAMETER < 0.215


def test_ball_mass_volleyball():
    # FIVB: 260-280 g
    assert 0.255 < C.BALL_MASS < 0.285


def test_net_heights():
    assert C.NET_HEIGHT_MEN == 2.43
    assert C.NET_HEIGHT_WOMEN == 2.24
```

- [ ] **Step 5: Run test, verify ModuleNotFoundError or missing constants**

Run: `uv run pytest tests/test_constants.py -v`
Expected: FAIL — `vb_coach.constants` does not exist yet.

- [ ] **Step 6: Implement `constants.py`**

Create `/home/siriux/Projects/vb_coach/src/vb_coach/constants.py`:

```python
"""Physical constants and default volleyball parameters.

All SI units (kg, m, s, rad).
"""

# Environment (sea level, 20°C)
G: float = 9.81
RHO_AIR: float = 1.20
NU_AIR: float = 1.5e-5

# Ball (FIVB regulation midpoint)
BALL_DIAMETER: float = 0.21
BALL_RADIUS: float = BALL_DIAMETER / 2
BALL_MASS: float = 0.270
BALL_CROSS_SECTION: float = 3.141592653589793 * BALL_RADIUS ** 2

# Court
COURT_LENGTH: float = 18.0
COURT_WIDTH: float = 9.0
ATTACK_LINE_FROM_NET: float = 3.0
NET_HEIGHT_MEN: float = 2.43
NET_HEIGHT_WOMEN: float = 2.24
ANTENNA_HEIGHT_ABOVE_NET: float = 0.80
```

- [ ] **Step 7: Run tests, verify pass**

Run: `uv run pytest tests/test_constants.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore src/ tests/
git commit -m "feat: project scaffold and physical constants"
```

---

## Task 2: Coordinate-system module

**Files:**
- Create: `src/vb_coach/coords.py`
- Create: `tests/test_coords.py`

- [ ] **Step 1: Write failing test**

Create `/home/siriux/Projects/vb_coach/tests/test_coords.py`:

```python
import numpy as np
import pytest
from vb_coach.coords import mirror_across_net, on_team_a, on_team_b


def test_on_team_a_positive_y():
    assert on_team_a(np.array([0.0, 2.0, 1.0]))
    assert not on_team_a(np.array([0.0, -2.0, 1.0]))


def test_on_team_b_negative_y():
    assert on_team_b(np.array([0.0, -2.0, 1.0]))
    assert not on_team_b(np.array([0.0, 2.0, 1.0]))


def test_mirror_across_net_flips_y():
    p = np.array([1.5, 3.0, 2.4])
    m = mirror_across_net(p)
    np.testing.assert_array_equal(m, np.array([1.5, -3.0, 2.4]))


def test_mirror_is_involution():
    p = np.array([0.7, 4.0, 2.1])
    np.testing.assert_array_equal(mirror_across_net(mirror_across_net(p)), p)
```

- [ ] **Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_coords.py -v`
Expected: FAIL — `vb_coach.coords` does not exist.

- [ ] **Step 3: Implement `coords.py`**

Create `/home/siriux/Projects/vb_coach/src/vb_coach/coords.py`:

```python
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
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_coords.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vb_coach/coords.py tests/test_coords.py
git commit -m "feat: world coordinate system and net mirror helper"
```

---

## Task 3: Aerodynamics module (Reynolds, spin parameter, placeholder Cd/Cl)

**Files:**
- Create: `src/vb_coach/aerodynamics.py`
- Create: `tests/test_aerodynamics.py`

- [ ] **Step 1: Write failing test**

Create `/home/siriux/Projects/vb_coach/tests/test_aerodynamics.py`:

```python
import math
import numpy as np
import pytest
from vb_coach.aerodynamics import reynolds, spin_parameter, cd_subcritical, cl_magnus
from vb_coach.constants import BALL_DIAMETER, BALL_RADIUS, NU_AIR


def test_reynolds_typical_set_speed():
    # v = 6 m/s, D = 0.21 m, nu = 1.5e-5 m^2/s -> Re ~ 84,000
    re = reynolds(6.0, BALL_DIAMETER, NU_AIR)
    assert 80_000 < re < 90_000


def test_reynolds_zero_velocity_is_zero():
    assert reynolds(0.0, BALL_DIAMETER, NU_AIR) == 0.0


def test_spin_parameter_zero_spin():
    assert spin_parameter(0.0, 6.0, BALL_RADIUS) == 0.0


def test_spin_parameter_known_value():
    # omega = 31.4 rad/s (5 rev/s), v = 6 m/s, R = 0.105
    # S = 31.4 * 0.105 / 6 = 0.5495
    s = spin_parameter(31.4, 6.0, BALL_RADIUS)
    assert math.isclose(s, 0.5495, abs_tol=1e-3)


def test_spin_parameter_zero_velocity_returns_zero():
    # Avoid div-by-zero; ball at rest has no meaningful spin parameter
    assert spin_parameter(10.0, 0.0, BALL_RADIUS) == 0.0


def test_cd_subcritical_constant_placeholder():
    # Placeholder model: constant 0.45 across the subcritical range.
    # This is intentionally crude; real Cd(Re) plugs in after aero research.
    assert cd_subcritical(50_000) == 0.45
    assert cd_subcritical(100_000) == 0.45


def test_cl_magnus_zero_spin_zero_lift():
    assert cl_magnus(0.0) == 0.0


def test_cl_magnus_saturates_at_high_spin():
    # Above S ~ 0.5, Cl flattens around 0.2 for sports balls
    assert math.isclose(cl_magnus(0.5), 0.2, abs_tol=0.01)
    assert math.isclose(cl_magnus(1.0), 0.2, abs_tol=0.01)


def test_cl_magnus_linear_below_saturation():
    # At S = 0.25 (half-saturation) Cl ~ 0.1
    assert math.isclose(cl_magnus(0.25), 0.1, abs_tol=0.01)
```

- [ ] **Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_aerodynamics.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `aerodynamics.py`**

Create `/home/siriux/Projects/vb_coach/src/vb_coach/aerodynamics.py`:

```python
"""Aerodynamic coefficients for a volleyball.

The current Cd(Re) and Cl(S) models are intentionally crude placeholders:
- Cd: constant 0.45 across the subcritical regime (Re < ~200k)
- Cl: linear in spin parameter S up to S = 0.5, saturating at 0.2

These are swap points. Once subcritical wind-tunnel / CFD data is in hand
(see project notes), replace cd_subcritical and cl_magnus with regime-correct
functions. The simulate() entry point in trajectory.py reads coefficients
exclusively through this module, so a one-file change updates the model.
"""

from __future__ import annotations


def reynolds(velocity_magnitude: float, diameter: float, kinematic_viscosity: float) -> float:
    return velocity_magnitude * diameter / kinematic_viscosity


def spin_parameter(spin_rate_rad_s: float, velocity_magnitude: float, radius: float) -> float:
    if velocity_magnitude == 0.0:
        return 0.0
    return spin_rate_rad_s * radius / velocity_magnitude


def cd_subcritical(reynolds_number: float) -> float:
    # Placeholder: replace with Re-dependent model once data available.
    return 0.45


_CL_SATURATION = 0.20
_CL_HALF_SATURATION_S = 0.25


def cl_magnus(spin_parameter_value: float) -> float:
    s = abs(spin_parameter_value)
    if s >= 2 * _CL_HALF_SATURATION_S:
        return _CL_SATURATION
    return _CL_SATURATION * s / (2 * _CL_HALF_SATURATION_S)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_aerodynamics.py -v`
Expected: PASS, 9 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vb_coach/aerodynamics.py tests/test_aerodynamics.py
git commit -m "feat: aerodynamics module with placeholder Cd/Cl"
```

---

## Task 4: Force model (gravity, drag, Magnus)

**Files:**
- Create: `src/vb_coach/physics.py`
- Create: `tests/test_physics.py`

- [ ] **Step 1: Write failing test**

Create `/home/siriux/Projects/vb_coach/tests/test_physics.py`:

```python
import math
import numpy as np
import pytest
from vb_coach.physics import gravity_force, drag_force, magnus_force, total_acceleration
from vb_coach.constants import BALL_MASS, G, RHO_AIR, BALL_CROSS_SECTION


def test_gravity_points_down_with_magnitude_mg():
    f = gravity_force(BALL_MASS)
    np.testing.assert_array_almost_equal(f, np.array([0.0, 0.0, -BALL_MASS * G]))


def test_drag_force_zero_at_rest():
    v = np.array([0.0, 0.0, 0.0])
    f = drag_force(v, rho=RHO_AIR, area=BALL_CROSS_SECTION, cd=0.45)
    np.testing.assert_array_almost_equal(f, np.zeros(3))


def test_drag_force_opposes_velocity():
    v = np.array([5.0, 0.0, 0.0])
    f = drag_force(v, rho=RHO_AIR, area=BALL_CROSS_SECTION, cd=0.45)
    assert f[0] < 0
    assert math.isclose(f[1], 0.0, abs_tol=1e-12)
    assert math.isclose(f[2], 0.0, abs_tol=1e-12)


def test_drag_force_magnitude():
    v = np.array([6.0, 0.0, 0.0])
    f = drag_force(v, rho=RHO_AIR, area=BALL_CROSS_SECTION, cd=0.45)
    # |F_d| = 0.5 * 1.2 * 36 * 0.0346 * 0.45 ~ 0.336 N
    assert math.isclose(np.linalg.norm(f), 0.336, abs_tol=0.01)


def test_magnus_zero_when_no_spin():
    v = np.array([6.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 0.0])
    f = magnus_force(v, omega, rho=RHO_AIR, area=BALL_CROSS_SECTION, radius=0.105)
    np.testing.assert_array_almost_equal(f, np.zeros(3))


def test_magnus_zero_when_no_velocity():
    v = np.array([0.0, 0.0, 0.0])
    omega = np.array([0.0, 10.0, 0.0])
    f = magnus_force(v, omega, rho=RHO_AIR, area=BALL_CROSS_SECTION, radius=0.105)
    np.testing.assert_array_almost_equal(f, np.zeros(3))


def test_magnus_perpendicular_to_velocity_and_spin():
    v = np.array([6.0, 0.0, 0.0])
    omega = np.array([0.0, 31.4, 0.0])  # back-spin on a ball moving +x
    f = magnus_force(v, omega, rho=RHO_AIR, area=BALL_CROSS_SECTION, radius=0.105)
    # Back-spin in +y, motion in +x -> omega x v_hat points in +z -> lift up
    assert f[2] > 0
    assert math.isclose(f[0], 0.0, abs_tol=1e-9)
    assert math.isclose(f[1], 0.0, abs_tol=1e-9)


def test_total_acceleration_includes_gravity():
    v = np.array([0.0, 0.0, 0.0])
    omega = np.array([0.0, 0.0, 0.0])
    a = total_acceleration(v, omega, mass=BALL_MASS, rho=RHO_AIR,
                           area=BALL_CROSS_SECTION, radius=0.105)
    np.testing.assert_array_almost_equal(a, np.array([0.0, 0.0, -G]))
```

- [ ] **Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_physics.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `physics.py`**

Create `/home/siriux/Projects/vb_coach/src/vb_coach/physics.py`:

```python
"""Force model for a flying volleyball.

Force decomposition:
    F_total = F_gravity + F_drag + F_Magnus

Drag:
    F_d = -0.5 * rho * |v|^2 * A * Cd(Re) * v_hat

Magnus (lift from spinning sphere in flow):
    F_m = 0.5 * rho * A * Cl(S) * |v|^2 * (omega_hat x v_hat)

We compute coefficients on the fly each step so simulate() always sees the
current velocity-dependent Re and S. Cd and Cl come from aerodynamics.py.
"""

from __future__ import annotations
import numpy as np

from vb_coach.aerodynamics import (
    reynolds,
    spin_parameter,
    cd_subcritical,
    cl_magnus,
)
from vb_coach.constants import G, NU_AIR


def gravity_force(mass: float) -> np.ndarray:
    return np.array([0.0, 0.0, -mass * G])


def drag_force(velocity: np.ndarray, rho: float, area: float, cd: float) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    if speed == 0.0:
        return np.zeros(3)
    v_hat = velocity / speed
    magnitude = 0.5 * rho * speed * speed * area * cd
    return -magnitude * v_hat


def magnus_force(
    velocity: np.ndarray,
    spin: np.ndarray,
    rho: float,
    area: float,
    radius: float,
) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    spin_mag = float(np.linalg.norm(spin))
    if speed == 0.0 or spin_mag == 0.0:
        return np.zeros(3)
    v_hat = velocity / speed
    omega_hat = spin / spin_mag
    s = spin_parameter(spin_mag, speed, radius)
    cl = cl_magnus(s)
    return 0.5 * rho * speed * speed * area * cl * np.cross(omega_hat, v_hat)


def total_acceleration(
    velocity: np.ndarray,
    spin: np.ndarray,
    mass: float,
    rho: float,
    area: float,
    radius: float,
    diameter: float | None = None,
) -> np.ndarray:
    speed = float(np.linalg.norm(velocity))
    d = diameter if diameter is not None else 2 * radius
    re = reynolds(speed, d, NU_AIR)
    cd = cd_subcritical(re)
    f = gravity_force(mass) + drag_force(velocity, rho, area, cd) + magnus_force(
        velocity, spin, rho, area, radius
    )
    return f / mass
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_physics.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vb_coach/physics.py tests/test_physics.py
git commit -m "feat: gravity, drag, and Magnus force model"
```

---

## Task 5: ReleaseState and Trajectory data types

**Files:**
- Create: `src/vb_coach/trajectory.py` (types only — simulate() comes in Task 6)
- Create: `tests/test_trajectory.py` (types tests only — simulate tests in Task 6)

- [ ] **Step 1: Write failing test**

Create `/home/siriux/Projects/vb_coach/tests/test_trajectory.py`:

```python
import numpy as np
import pytest
from vb_coach.trajectory import ReleaseState, Trajectory


def test_release_state_construction():
    rs = ReleaseState(
        position=np.array([0.0, 0.8, 2.2]),
        velocity=np.array([0.0, -3.0, 6.0]),
        spin=np.array([0.0, 10.0, 0.0]),
    )
    assert rs.position.shape == (3,)
    assert rs.velocity.shape == (3,)
    assert rs.spin.shape == (3,)


def test_release_state_rejects_wrong_shape():
    with pytest.raises(ValueError):
        ReleaseState(
            position=np.array([0.0, 0.8]),
            velocity=np.array([0.0, -3.0, 6.0]),
            spin=np.array([0.0, 10.0, 0.0]),
        )


def test_trajectory_consistent_lengths():
    t = np.linspace(0, 1, 11)
    p = np.zeros((11, 3))
    v = np.zeros((11, 3))
    s = np.zeros((11, 3))
    traj = Trajectory(time=t, position=p, velocity=v, spin=s)
    assert len(traj) == 11


def test_trajectory_rejects_inconsistent_lengths():
    t = np.linspace(0, 1, 11)
    p = np.zeros((10, 3))  # mismatch
    v = np.zeros((11, 3))
    s = np.zeros((11, 3))
    with pytest.raises(ValueError):
        Trajectory(time=t, position=p, velocity=v, spin=s)


def test_trajectory_apex_index():
    # Synthetic z = 1 - (t - 0.5)^2 * 4, peak at t=0.5
    t = np.linspace(0, 1, 101)
    z = 1 - (t - 0.5) ** 2 * 4
    p = np.column_stack([np.zeros_like(t), np.zeros_like(t), z])
    v = np.zeros_like(p)
    s = np.zeros_like(p)
    traj = Trajectory(time=t, position=p, velocity=v, spin=s)
    assert abs(traj.time[traj.apex_index()] - 0.5) < 0.02
```

- [ ] **Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_trajectory.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement types in `trajectory.py`**

Create `/home/siriux/Projects/vb_coach/src/vb_coach/trajectory.py`:

```python
"""Trajectory data types and simulation entry point.

ReleaseState captures the moment the ball leaves the setter's hands.
Trajectory captures the integrated path: arrays for time, position,
velocity, and spin sampled at the integrator's output times.

simulate() is implemented in Task 6.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class ReleaseState:
    position: np.ndarray  # shape (3,), meters
    velocity: np.ndarray  # shape (3,), m/s
    spin: np.ndarray      # shape (3,), rad/s, right-hand rule

    def __post_init__(self) -> None:
        for name in ("position", "velocity", "spin"):
            arr = getattr(self, name)
            if arr.shape != (3,):
                raise ValueError(f"{name} must have shape (3,), got {arr.shape}")


@dataclass
class Trajectory:
    time: np.ndarray      # shape (N,), seconds
    position: np.ndarray  # shape (N, 3), meters
    velocity: np.ndarray  # shape (N, 3), m/s
    spin: np.ndarray      # shape (N, 3), rad/s

    def __post_init__(self) -> None:
        n = self.time.shape[0]
        for name in ("position", "velocity", "spin"):
            arr = getattr(self, name)
            if arr.shape != (n, 3):
                raise ValueError(
                    f"{name} must have shape ({n}, 3), got {arr.shape}"
                )

    def __len__(self) -> int:
        return int(self.time.shape[0])

    def apex_index(self) -> int:
        return int(np.argmax(self.position[:, 2]))

    def apex_height(self) -> float:
        return float(self.position[self.apex_index(), 2])
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_trajectory.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vb_coach/trajectory.py tests/test_trajectory.py
git commit -m "feat: ReleaseState and Trajectory dataclasses"
```

---

## Task 6: ODE integration — `simulate()` entry point

**Files:**
- Modify: `src/vb_coach/trajectory.py` (add `simulate()`)
- Modify: `tests/test_trajectory.py` (add simulate tests)

- [ ] **Step 1: Write failing tests for `simulate()`**

Append to `/home/siriux/Projects/vb_coach/tests/test_trajectory.py`:

```python
from vb_coach.trajectory import simulate
from vb_coach.constants import BALL_MASS, BALL_RADIUS, BALL_CROSS_SECTION, RHO_AIR, G


def test_simulate_returns_trajectory():
    rs = ReleaseState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.array([0.0, 0.0, 5.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj = simulate(rs, t_max=2.0)
    assert isinstance(traj, Trajectory)
    assert len(traj) > 10


def test_simulate_stops_at_floor():
    rs = ReleaseState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.array([0.0, 0.0, 5.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj = simulate(rs, t_max=5.0)
    # Floor event terminates integration when z=0
    assert abs(traj.position[-1, 2]) < 0.01


def test_simulate_pure_vertical_in_vacuum_matches_analytic():
    # In vacuum (rho=0) and no spin: z(t) = z0 + v0*t - 0.5*g*t^2
    rs = ReleaseState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.array([0.0, 0.0, 5.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj = simulate(rs, t_max=2.0, rho=0.0)
    # Apex: t_apex = v0/g = 5/9.81 = 0.5097s; z_apex = z0 + v0^2/(2g) = 2 + 25/19.62 = 3.274m
    assert abs(traj.apex_height() - 3.274) < 0.01


def test_simulate_no_spin_no_curvature_in_horizontal_plane():
    # No spin, no wind: y velocity stays at initial value, x stays at 0
    rs = ReleaseState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.array([0.0, -3.0, 5.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj = simulate(rs, t_max=2.0)
    # x must remain ~0 throughout
    assert float(np.max(np.abs(traj.position[:, 0]))) < 1e-6


def test_simulate_back_spin_lifts_ball():
    # Compare same release with and without back-spin: back-spin trajectory
    # should travel farther in +y before hitting floor.
    base = ReleaseState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.array([0.0, 5.0, 4.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    # Back-spin on a ball traveling in +y: omega in +x -> omega x v_hat = +z (lift)
    spun = ReleaseState(
        position=base.position.copy(),
        velocity=base.velocity.copy(),
        spin=np.array([30.0, 0.0, 0.0]),
    )
    traj_base = simulate(base, t_max=3.0)
    traj_spun = simulate(spun, t_max=3.0)
    assert traj_spun.position[-1, 1] > traj_base.position[-1, 1]
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/test_trajectory.py -v`
Expected: 5 prior pass; 5 new fail with `cannot import name 'simulate'`.

- [ ] **Step 3: Implement `simulate()` in `trajectory.py`**

Append to `/home/siriux/Projects/vb_coach/src/vb_coach/trajectory.py`:

```python
from scipy.integrate import solve_ivp

from vb_coach.constants import (
    BALL_MASS,
    BALL_RADIUS,
    BALL_CROSS_SECTION,
    BALL_DIAMETER,
    RHO_AIR,
    NU_AIR,
)
from vb_coach.aerodynamics import reynolds, spin_parameter, cd_subcritical, cl_magnus


def _rhs(t: float, y: np.ndarray, mass: float, rho: float, area: float,
         radius: float, diameter: float, spin: np.ndarray) -> np.ndarray:
    # State vector y = [x, y, z, vx, vy, vz]; spin held constant for short flights.
    v = y[3:6]
    speed = float(np.linalg.norm(v))
    a = np.array([0.0, 0.0, -9.81])
    if speed > 0.0:
        v_hat = v / speed
        re = reynolds(speed, diameter, NU_AIR)
        cd = cd_subcritical(re)
        a = a - (0.5 * rho * speed * speed * area * cd / mass) * v_hat
        spin_mag = float(np.linalg.norm(spin))
        if spin_mag > 0.0:
            omega_hat = spin / spin_mag
            s = spin_parameter(spin_mag, speed, radius)
            cl = cl_magnus(s)
            a = a + (0.5 * rho * speed * speed * area * cl / mass) * np.cross(omega_hat, v_hat)
    return np.concatenate([v, a])


def _floor_event(t: float, y: np.ndarray, *args) -> float:
    return float(y[2])


_floor_event.terminal = True
_floor_event.direction = -1


def simulate(
    release: ReleaseState,
    t_max: float = 3.0,
    mass: float = BALL_MASS,
    rho: float = RHO_AIR,
    area: float = BALL_CROSS_SECTION,
    radius: float = BALL_RADIUS,
    diameter: float = BALL_DIAMETER,
    max_step: float = 0.01,
) -> Trajectory:
    y0 = np.concatenate([release.position, release.velocity])
    sol = solve_ivp(
        _rhs,
        (0.0, t_max),
        y0,
        method="RK45",
        events=_floor_event,
        args=(mass, rho, area, radius, diameter, release.spin),
        max_step=max_step,
        rtol=1e-8,
        atol=1e-9,
    )
    n = sol.t.shape[0]
    pos = sol.y[0:3, :].T
    vel = sol.y[3:6, :].T
    spins = np.broadcast_to(release.spin, (n, 3)).copy()
    return Trajectory(time=sol.t, position=pos, velocity=vel, spin=spins)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_trajectory.py -v`
Expected: PASS, 10 tests total.

- [ ] **Step 5: Commit**

```bash
git add src/vb_coach/trajectory.py tests/test_trajectory.py
git commit -m "feat: ODE-based simulate() with floor termination event"
```

---

## Task 7: Plotting module

**Files:**
- Create: `src/vb_coach/plotting.py`
- Create: `tests/test_plotting.py`

- [ ] **Step 1: Write failing test**

Create `/home/siriux/Projects/vb_coach/tests/test_plotting.py`:

```python
import matplotlib
matplotlib.use("Agg")  # headless backend for CI
import numpy as np
import pytest

from vb_coach.trajectory import ReleaseState, simulate
from vb_coach.plotting import plot_trajectory_3d, plot_trajectory_side


@pytest.fixture
def sample_trajectory():
    rs = ReleaseState(
        position=np.array([0.0, 0.8, 2.2]),
        velocity=np.array([0.0, -4.0, 6.0]),
        spin=np.array([5.0, 0.0, 0.0]),
    )
    return simulate(rs, t_max=2.0)


def test_plot_trajectory_3d_returns_figure(sample_trajectory):
    fig = plot_trajectory_3d(sample_trajectory)
    assert fig is not None
    assert len(fig.axes) >= 1


def test_plot_trajectory_side_returns_figure(sample_trajectory):
    fig = plot_trajectory_side(sample_trajectory)
    assert fig is not None
    assert len(fig.axes) >= 1


def test_plot_with_net_marker_men(sample_trajectory):
    fig = plot_trajectory_side(sample_trajectory, net_height=2.43)
    assert fig is not None
```

- [ ] **Step 2: Run test, verify fail**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `plotting.py`**

Create `/home/siriux/Projects/vb_coach/src/vb_coach/plotting.py`:

```python
"""Basic matplotlib plotting for trajectories.

Importing matplotlib happens only here so the core library has no plot
dependency at import time. Plot functions always return the Figure so the
caller can save it, show it, or compose multiple plots.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3d projection

from vb_coach.trajectory import Trajectory


def plot_trajectory_3d(traj: Trajectory, net_height: float | None = None) -> plt.Figure:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj.position[:, 0], traj.position[:, 1], traj.position[:, 2], lw=2)
    ax.scatter(*traj.position[0], color="green", s=40, label="release")
    apex = traj.apex_index()
    ax.scatter(*traj.position[apex], color="orange", s=40, label="apex")
    ax.scatter(*traj.position[-1], color="red", s=40, label="end")
    if net_height is not None:
        xs = [-4.5, 4.5, 4.5, -4.5, -4.5]
        ys = [0.0, 0.0, 0.0, 0.0, 0.0]
        zs = [0.0, 0.0, net_height, net_height, 0.0]
        ax.plot(xs, ys, zs, color="gray", lw=1, alpha=0.6)
    ax.set_xlabel("x [m] (along net)")
    ax.set_ylabel("y [m] (across court)")
    ax.set_zlabel("z [m]")
    ax.legend()
    return fig


def plot_trajectory_side(traj: Trajectory, net_height: float | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(traj.position[:, 1], traj.position[:, 2], lw=2)
    ax.scatter(traj.position[0, 1], traj.position[0, 2], color="green", s=40, label="release")
    apex = traj.apex_index()
    ax.scatter(traj.position[apex, 1], traj.position[apex, 2], color="orange", s=40, label="apex")
    ax.scatter(traj.position[-1, 1], traj.position[-1, 2], color="red", s=40, label="end")
    if net_height is not None:
        ax.axvline(0.0, color="gray", lw=1, alpha=0.5)
        ax.plot([0.0, 0.0], [0.0, net_height], color="gray", lw=2)
    ax.axhline(0.0, color="black", lw=0.5)
    ax.set_xlabel("y [m] (across court)")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/vb_coach/plotting.py tests/test_plotting.py
git commit -m "feat: 2D side-view and 3D trajectory plots"
```

---

## Task 8: End-to-end integration tests

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write end-to-end tests**

Create `/home/siriux/Projects/vb_coach/tests/test_integration.py`:

```python
"""End-to-end physical sanity tests.

These tests check that the full pipeline produces values consistent with
back-of-envelope physics for realistic volleyball-setting scenarios.
"""

import math
import numpy as np
import pytest

from vb_coach.trajectory import ReleaseState, simulate
from vb_coach.constants import (
    BALL_MASS,
    BALL_CROSS_SECTION,
    RHO_AIR,
    NU_AIR,
    BALL_DIAMETER,
)
from vb_coach.aerodynamics import reynolds


def test_reynolds_for_typical_set_is_subcritical():
    # Set speed ~6 m/s, Re ~84k -> well below the drag crisis (~250k)
    re = reynolds(6.0, BALL_DIAMETER, NU_AIR)
    assert re < 200_000


def test_high_ball_to_position_4_reaches_realistic_apex():
    # Setter at (0.0, 0.8, 2.2); target apex above attack zone 4 at x = -3.5, y = -0.3
    # Initial velocity tuned to fly ~3.7m horizontally to apex over ~1.1s with apex ~5.5m
    rs = ReleaseState(
        position=np.array([0.0, 0.8, 2.2]),
        velocity=np.array([-3.0, -1.0, 7.5]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj = simulate(rs, t_max=3.0)
    apex_z = traj.apex_height()
    # Apex should be in the 4.5-6.5 m range — realistic high-ball
    assert 4.5 < apex_z < 6.5
    # Total flight time should be 1.5-2.5 s
    assert 1.0 < traj.time[-1] < 3.0


def test_drag_reduces_apex_compared_to_vacuum():
    rs = ReleaseState(
        position=np.array([0.0, 0.0, 2.0]),
        velocity=np.array([0.0, 0.0, 7.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj_air = simulate(rs, t_max=2.5, rho=RHO_AIR)
    traj_vac = simulate(rs, t_max=2.5, rho=0.0)
    # Vacuum apex from analytic: z0 + v0^2/(2g) = 2 + 49/19.62 = 4.497
    assert traj_vac.apex_height() > 4.4
    # Air resistance must reduce apex
    assert traj_air.apex_height() < traj_vac.apex_height()


def test_top_spin_drops_ball_earlier_than_no_spin():
    # Top-spin on a ball moving +y has omega in -x (right-hand rule):
    # omega_hat x v_hat = (-x) x (+y) = -z -> Magnus pushes down
    base = ReleaseState(
        position=np.array([0.0, 0.0, 2.2]),
        velocity=np.array([0.0, 5.0, 5.0]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    top_spin = ReleaseState(
        position=base.position.copy(),
        velocity=base.velocity.copy(),
        spin=np.array([-30.0, 0.0, 0.0]),
    )
    traj_base = simulate(base, t_max=3.0)
    traj_top = simulate(top_spin, t_max=3.0)
    # Top-spin should land sooner
    assert traj_top.time[-1] < traj_base.time[-1]
    # And land shorter (smaller y at landing)
    assert traj_top.position[-1, 1] < traj_base.position[-1, 1]
```

- [ ] **Step 2: Run tests, verify pass**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: ALL pass (constants 6 + coords 4 + aero 9 + physics 8 + trajectory 10 + plotting 3 + integration 4 = 44).

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: end-to-end physical sanity checks"
```

---

## Task 9: Runnable example — 4号位 high-ball set

**Files:**
- Create: `examples/basic_set.py`

- [ ] **Step 1: Write the example script**

Create `/home/siriux/Projects/vb_coach/examples/basic_set.py`:

```python
"""Demo: setter at zone 2/3 sets a high ball to 4号位 (left front).

Run:
    uv run python examples/basic_set.py

Produces:
    basic_set_3d.png and basic_set_side.png in the current directory.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vb_coach.trajectory import ReleaseState, simulate
from vb_coach.plotting import plot_trajectory_3d, plot_trajectory_side
from vb_coach.constants import NET_HEIGHT_MEN


def main() -> None:
    release = ReleaseState(
        position=np.array([0.5, 0.8, 2.2]),     # near zone 2/3, 0.8 m off net, hands at 2.2 m
        velocity=np.array([-3.0, -1.0, 7.5]),   # toward zone 4, slightly off net, big lift
        spin=np.array([2.0, 0.0, 0.0]),         # mild back-spin
    )
    traj = simulate(release, t_max=3.0)

    print(f"Flight time: {traj.time[-1]:.2f} s")
    print(f"Apex height: {traj.apex_height():.2f} m")
    print(f"Apex at t=:  {traj.time[traj.apex_index()]:.2f} s")
    print(f"Landing:     x={traj.position[-1, 0]:.2f}  y={traj.position[-1, 1]:.2f}  z={traj.position[-1, 2]:.3f}")

    fig3d = plot_trajectory_3d(traj, net_height=NET_HEIGHT_MEN)
    fig3d.savefig("basic_set_3d.png", dpi=120, bbox_inches="tight")
    plt.close(fig3d)

    fig_side = plot_trajectory_side(traj, net_height=NET_HEIGHT_MEN)
    fig_side.savefig("basic_set_side.png", dpi=120, bbox_inches="tight")
    plt.close(fig_side)

    print("Saved: basic_set_3d.png, basic_set_side.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the example**

Run: `uv run python examples/basic_set.py`
Expected: prints flight stats; creates `basic_set_3d.png` and `basic_set_side.png`. Open them and verify the trajectory looks like a high set: rises above 4 m, comes back down on the opposite side of the net from release.

- [ ] **Step 3: Commit**

```bash
git add examples/basic_set.py
git commit -m "feat: 4号位 high-ball example script"
```

---

## Verification checklist before declaring Phase A0 done

- [ ] `uv run pytest -v` passes all tests
- [ ] `uv run python examples/basic_set.py` runs and produces sensible numbers (apex 4.5-6.5 m, flight 1.5-2.5 s)
- [ ] Constants file has no logic, only literals
- [ ] `aerodynamics.py` is the only place coefficients are defined (grep confirms no Cd literal outside it)
- [ ] All public functions have type hints
- [ ] No file exceeds ~150 lines

## What comes next (not in this plan)

- A0.4: regime-correct Cd(Re) and any panel-orientation lateral force — wait for subcritical aero research agent results, then swap into `aerodynamics.py` without touching anything else
- A1: 4号位高球 attack model — hit-window geometry, attacker reach inputs, quality metric definition
- A2-A4: remaining attack types
- A5: attacker-adaptation layer
- A6: interactive visualization (Three.js or Plotly, separate from matplotlib used here)
