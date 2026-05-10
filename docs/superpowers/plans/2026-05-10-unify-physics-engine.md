# Physics Engine Unification Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the standalone `examples/float_serve_demo.py` to use the shared `vb_coach` library. Extend the library so it supports both the subcritical regime (sets, 4–8 m/s) and the supercritical / drag-crisis regime (float serves, 13–20 m/s) through one aerodynamics package and two simulators (deterministic RK45 + stochastic symplectic-Euler).

**Architecture:** `aerodynamics.py` becomes a package with three submodules: `drag` (`reynolds`, `cd_subcritical`, `cd_full_regime`), `magnus` (`spin_parameter`, `cl_magnus`), and `knuckle` (`cl_lat_rms`, `OUState`, `step_ou`). `physics.py` gains `knuckle_force()`. `trajectory.py` gains `simulate_stochastic()` that uses symplectic Euler so the OU noise process can be advanced in lockstep. The float-serve demo shrinks from 369 lines to ~150 by importing from the library.

**Tech Stack:** Python 3.11+, numpy, scipy, matplotlib, pytest, uv. No new deps.

**Out of scope:**
- Phase A1 attack-type modeling (separate plans later)
- Visualization redesign (matplotlib stays; Three.js / Plotly comes in A6)
- New physics — we are only *moving* existing physics, not adding new modeling

---

## File structure after refactor

```
src/vb_coach/
  aerodynamics/                 ← was aerodynamics.py (single file)
    __init__.py                 re-exports the public API
    drag.py                     reynolds, cd_subcritical, cd_full_regime
    magnus.py                   spin_parameter, cl_magnus
    knuckle.py                  cl_lat_rms, OUState, step_ou, TAU_KNUCKLE
  physics.py                    + knuckle_force()
  trajectory.py                 + simulate_stochastic()
  constants.py, coords.py, plotting.py  unchanged
tests/
  test_aerodynamics.py          extended with cd_full_regime tests
  test_knuckle.py               new: cl_lat_rms, OU process
  test_physics.py               extended with knuckle_force tests
  test_trajectory.py            extended with simulate_stochastic tests
examples/
  basic_set.py                  unchanged
  float_serve_demo.py           rewritten to use library
```

## Key API decisions

- **OUState mutation pattern preserved** from the original demo (`step_ou` mutates and returns) — keeps the existing demo's structure usable as-is and avoids redundant copies in the hot loop.
- **Symplectic Euler** for `simulate_stochastic`: velocity updates first, position uses the NEW velocity. This matches the original demo bit-for-bit and gives slightly better energy properties than pure forward Euler.
- **`cd_subcritical` retained** as a constant-0.45 function (legacy, useful for analytical reasoning). New code uses `cd_full_regime`. `simulate()` switches to `cd_full_regime` — behavior unchanged for Re < 1.5×10⁵, which covers every existing test case.
- **Backward-compatible imports**: `from vb_coach.aerodynamics import reynolds, cd_subcritical, cl_magnus, ...` keeps working because `__init__.py` re-exports.
- **Bit-for-bit reproducibility**: the rewritten float demo must produce the same canonical trajectories and the same Monte Carlo σ_x ≈ 30 cm at v=16 m/s with seeds 0–49.

---

## Task 1: Convert `aerodynamics.py` to a package

Pure refactor. No behavior change. All existing tests must still pass without modification.

**Files:**
- Delete: `src/vb_coach/aerodynamics.py`
- Create: `src/vb_coach/aerodynamics/__init__.py`
- Create: `src/vb_coach/aerodynamics/drag.py`
- Create: `src/vb_coach/aerodynamics/magnus.py`

- [ ] **Step 1: Confirm baseline tests pass**

Run: `uv run pytest -q`
Expected: 45 passed.

- [ ] **Step 2: Create `drag.py`**

Create `/home/siriux/Projects/vb_coach/src/vb_coach/aerodynamics/drag.py`:

```python
"""Drag coefficient and Reynolds number for a volleyball."""

from __future__ import annotations


def reynolds(velocity_magnitude: float, diameter: float, kinematic_viscosity: float) -> float:
    return velocity_magnitude * diameter / kinematic_viscosity


def cd_subcritical(reynolds_number: float) -> float:
    """Constant Cd ~ 0.45 across the subcritical band (Re < ~1.5e5).

    Kept as a function for analytical clarity and historical compatibility;
    new code should call cd_full_regime, which agrees with this in the
    subcritical band and additionally covers the drag crisis.
    """
    return 0.45
```

- [ ] **Step 3: Create `magnus.py`**

Create `/home/siriux/Projects/vb_coach/src/vb_coach/aerodynamics/magnus.py`:

```python
"""Magnus lift coefficient and spin parameter for a volleyball.

Cl(S) is modelled as linear in spin parameter S = omega * R / |v| up to
S ~ 0.5, then saturating at 0.2 - the shape sports-ball wind-tunnel data
shows in this Re range.
"""

from __future__ import annotations


def spin_parameter(spin_rate_rad_s: float, velocity_magnitude: float, radius: float) -> float:
    if velocity_magnitude == 0.0:
        return 0.0
    return spin_rate_rad_s * radius / velocity_magnitude


_CL_SATURATION = 0.20
_CL_HALF_SATURATION_S = 0.25


def cl_magnus(spin_parameter_value: float) -> float:
    s = abs(spin_parameter_value)
    if s >= 2 * _CL_HALF_SATURATION_S:
        return _CL_SATURATION
    return _CL_SATURATION * s / (2 * _CL_HALF_SATURATION_S)
```

- [ ] **Step 4: Create package `__init__.py` that re-exports**

Create `/home/siriux/Projects/vb_coach/src/vb_coach/aerodynamics/__init__.py`:

```python
"""Aerodynamic coefficients for a volleyball.

Submodules:
    drag    - Re-dependent drag and (eventually) full-regime Cd(Re)
    magnus  - spin-induced lift (Cl(S))
    knuckle - panel-induced unsteady lateral force (added in this refactor)
"""

from vb_coach.aerodynamics.drag import reynolds, cd_subcritical
from vb_coach.aerodynamics.magnus import spin_parameter, cl_magnus

__all__ = [
    "reynolds",
    "cd_subcritical",
    "spin_parameter",
    "cl_magnus",
]
```

- [ ] **Step 5: Delete old `aerodynamics.py`**

Run: `rm /home/siriux/Projects/vb_coach/src/vb_coach/aerodynamics.py`

- [ ] **Step 6: Run full test suite to verify backward compatibility**

Run: `uv run pytest -q`
Expected: 45 passed. Existing `from vb_coach.aerodynamics import ...` calls must continue to work via the package `__init__.py`.

- [ ] **Step 7: Commit**

```bash
git add src/vb_coach/aerodynamics/ src/vb_coach/aerodynamics.py
git commit -m "[refactor:aero] split aerodynamics into a package with drag and magnus submodules"
```

(Note: `git add src/vb_coach/aerodynamics.py` stages the deletion of the old file.)

---

## Task 2: Add `cd_full_regime` to `drag.py`

Adds the piecewise Cd(Re) that covers the drag crisis.

**Files:**
- Modify: `src/vb_coach/aerodynamics/drag.py`
- Modify: `src/vb_coach/aerodynamics/__init__.py`
- Modify: `tests/test_aerodynamics.py`

- [ ] **Step 1: Write failing tests**

Append to `/home/siriux/Projects/vb_coach/tests/test_aerodynamics.py`:

```python
import math

from vb_coach.aerodynamics import cd_full_regime


def test_cd_full_regime_subcritical_band():
    # Re below the drag crisis: Cd plateau at 0.45
    assert cd_full_regime(50_000) == 0.45
    assert cd_full_regime(100_000) == 0.45
    assert cd_full_regime(150_000) == 0.45


def test_cd_full_regime_supercritical_band():
    # Re above the drag crisis: Cd plateau at 0.18
    assert cd_full_regime(250_000) == 0.18
    assert cd_full_regime(500_000) == 0.18


def test_cd_full_regime_critical_band_monotonic():
    # Smooth cosine taper across 1.5e5 < Re < 2.5e5
    cd_low = cd_full_regime(1.6e5)
    cd_mid = cd_full_regime(2.0e5)
    cd_high = cd_full_regime(2.4e5)
    assert cd_low > cd_mid > cd_high
    # Midpoint should be the average of the two plateaus
    assert math.isclose(cd_mid, 0.5 * (0.45 + 0.18), abs_tol=0.005)
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/test_aerodynamics.py -k full_regime -v`
Expected: 3 failures, "cannot import name 'cd_full_regime'".

- [ ] **Step 3: Implement `cd_full_regime`**

Append to `/home/siriux/Projects/vb_coach/src/vb_coach/aerodynamics/drag.py`:

```python
import math


_RE_CRIT_LOW = 1.5e5
_RE_CRIT_HIGH = 2.5e5
_CD_SUBCRITICAL = 0.45
_CD_SUPERCRITICAL = 0.18


def cd_full_regime(reynolds_number: float) -> float:
    """Piecewise Cd(Re) for a FIVB-spec volleyball spanning the drag crisis.

    Tuned against Asai et al. 2010 wind-tunnel data (Mikasa MVA200):
        Re <= 1.5e5     : Cd = 0.45                  (subcritical plateau)
        1.5e5 < Re < 2.5e5 : Cd 0.45 -> 0.18 via cosine taper  (drag crisis)
        Re >= 2.5e5     : Cd = 0.18                  (supercritical plateau)
    The cosine transition avoids integrator stiffness at the band edges.
    """
    if reynolds_number <= _RE_CRIT_LOW:
        return _CD_SUBCRITICAL
    if reynolds_number >= _RE_CRIT_HIGH:
        return _CD_SUPERCRITICAL
    phase = math.pi * (reynolds_number - _RE_CRIT_LOW) / (_RE_CRIT_HIGH - _RE_CRIT_LOW)
    return _CD_SUPERCRITICAL + 0.5 * (_CD_SUBCRITICAL - _CD_SUPERCRITICAL) * (1 + math.cos(phase))
```

- [ ] **Step 4: Re-export from package `__init__.py`**

Edit `/home/siriux/Projects/vb_coach/src/vb_coach/aerodynamics/__init__.py`:

Replace the existing `from vb_coach.aerodynamics.drag import reynolds, cd_subcritical` with:

```python
from vb_coach.aerodynamics.drag import reynolds, cd_subcritical, cd_full_regime
```

And add `"cd_full_regime"` to `__all__`.

- [ ] **Step 5: Run tests, verify pass**

Run: `uv run pytest tests/test_aerodynamics.py -v`
Expected: 12 passed (9 original + 3 new).

- [ ] **Step 6: Commit**

```bash
git add src/vb_coach/aerodynamics/ tests/test_aerodynamics.py
git commit -m "[feat:aero] add cd_full_regime piecewise model covering the drag crisis"
```

---

## Task 3: Add `knuckle.py` module

Adds the panel-induced lateral force model (Cl_lat(Re) + Ornstein-Uhlenbeck process).

**Files:**
- Create: `src/vb_coach/aerodynamics/knuckle.py`
- Modify: `src/vb_coach/aerodynamics/__init__.py`
- Create: `tests/test_knuckle.py`

- [ ] **Step 1: Write failing tests**

Create `/home/siriux/Projects/vb_coach/tests/test_knuckle.py`:

```python
import math

import numpy as np
import pytest

from vb_coach.aerodynamics.knuckle import (
    TAU_KNUCKLE,
    OUState,
    cl_lat_rms,
    step_ou,
)


def test_cl_lat_rms_peak_at_2e5():
    # Gaussian peak at Re = 2.0e5; max value should be the documented 0.12
    cl_peak = cl_lat_rms(2.0e5)
    assert math.isclose(cl_peak, 0.12, abs_tol=1e-6)


def test_cl_lat_rms_decays_far_from_peak():
    # Far below and far above the critical band: Cl_lat -> ~0
    assert cl_lat_rms(50_000) < 0.01
    assert cl_lat_rms(500_000) < 0.01


def test_cl_lat_rms_symmetric_around_peak():
    # Gaussian shape: equal distance below and above peak should give equal Cl
    cl_low = cl_lat_rms(2.0e5 - 0.3e5)
    cl_high = cl_lat_rms(2.0e5 + 0.3e5)
    assert math.isclose(cl_low, cl_high, abs_tol=1e-6)


def test_ou_state_defaults_to_zero():
    ou = OUState()
    assert ou.chan_h == 0.0
    assert ou.chan_v == 0.0


def test_step_ou_zero_dt_keeps_state():
    # With dt = 0, decay = 1 and noise_scale = 0 -> state unchanged
    ou = OUState(chan_h=1.5, chan_v=-0.7)
    rng = np.random.default_rng(0)
    step_ou(ou, 0.0, rng)
    assert math.isclose(ou.chan_h, 1.5, abs_tol=1e-12)
    assert math.isclose(ou.chan_v, -0.7, abs_tol=1e-12)


def test_step_ou_large_dt_drives_toward_unit_variance():
    # After many steps with dt > tau, variance should approach 1
    rng = np.random.default_rng(42)
    samples = []
    ou = OUState()
    for _ in range(20_000):
        step_ou(ou, TAU_KNUCKLE, rng)
        samples.append(ou.chan_h)
    var = float(np.var(samples))
    assert 0.7 < var < 1.3
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/test_knuckle.py -v`
Expected: collection error, module missing.

- [ ] **Step 3: Implement `knuckle.py`**

Create `/home/siriux/Projects/vb_coach/src/vb_coach/aerodynamics/knuckle.py`:

```python
"""Panel-induced unsteady lateral force model for a volleyball at high Re.

The float-serve "knuckle" effect arises from asymmetric flow separation
around the ball's panels, which produces a side force whose magnitude
oscillates with vortex-shedding-like dynamics. This module captures it as:

- cl_lat_rms(Re): the magnitude envelope, peaked in the drag-crisis band
- OUState + step_ou: an Ornstein-Uhlenbeck process that produces correlated
  random fluctuations with coherence time TAU_KNUCKLE matching Cohen 2014
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# Coherence time of the lateral force process. Matches Cohen 2014
# high-speed video measurements of float-serve lateral wobble.
TAU_KNUCKLE: float = 0.25


_CL_LAT_PEAK: float = 0.12
_RE_LAT_PEAK: float = 2.0e5
_RE_LAT_WIDTH: float = 0.6e5


def cl_lat_rms(reynolds_number: float) -> float:
    """RMS lateral lift coefficient as a function of Re.

    Gaussian peaked in the drag-crisis band where panel-orientation effects
    dominate; ~0 outside it (subcritical and deep supercritical).
    """
    return _CL_LAT_PEAK * math.exp(
        -((reynolds_number - _RE_LAT_PEAK) / _RE_LAT_WIDTH) ** 2
    )


@dataclass
class OUState:
    """Two-channel OU state. chan_h drives in-plane sideways force;
    chan_v drives the (weaker) out-of-plane perturbation."""

    chan_h: float = 0.0
    chan_v: float = 0.0


def step_ou(ou: OUState, dt: float, rng: np.random.Generator) -> OUState:
    """Advance the OU process by dt. Steady-state variance is 1 per channel.

    Mutates and returns the input state for ergonomic chaining.
    """
    decay = math.exp(-dt / TAU_KNUCKLE)
    noise_scale = math.sqrt(1.0 - decay ** 2)
    ou.chan_h = decay * ou.chan_h + noise_scale * rng.standard_normal()
    ou.chan_v = decay * ou.chan_v + noise_scale * rng.standard_normal()
    return ou
```

- [ ] **Step 4: Re-export from package `__init__.py`**

Edit `/home/siriux/Projects/vb_coach/src/vb_coach/aerodynamics/__init__.py` to add:

```python
from vb_coach.aerodynamics.knuckle import (
    TAU_KNUCKLE,
    OUState,
    cl_lat_rms,
    step_ou,
)
```

And extend `__all__` with `["TAU_KNUCKLE", "OUState", "cl_lat_rms", "step_ou"]`.

- [ ] **Step 5: Run tests, verify pass**

Run: `uv run pytest tests/test_knuckle.py -v`
Expected: 6 passed.

- [ ] **Step 6: Commit**

```bash
git add src/vb_coach/aerodynamics/ tests/test_knuckle.py
git commit -m "[feat:aero] add knuckle module: cl_lat_rms + OU lateral-force process"
```

---

## Task 4: Add `knuckle_force()` to `physics.py`

Wires the OU lateral force into the force-summation API.

**Files:**
- Modify: `src/vb_coach/physics.py`
- Modify: `tests/test_physics.py`

- [ ] **Step 1: Write failing tests**

Append to `/home/siriux/Projects/vb_coach/tests/test_physics.py`:

```python
import numpy as np

from vb_coach.aerodynamics.knuckle import OUState
from vb_coach.physics import knuckle_force


def test_knuckle_force_zero_at_low_speed():
    # Below the cutoff (0.1 m/s) the force is suppressed
    v = np.array([0.05, 0.0, 0.0])
    ou = OUState(chan_h=1.0, chan_v=1.0)
    f = knuckle_force(v, ou, rho=1.2, area=0.0346, re=2.0e5)
    np.testing.assert_array_almost_equal(f, np.zeros(3))


def test_knuckle_force_perpendicular_to_velocity():
    # h_hat and vert_hat are both perpendicular to v_hat by construction
    v = np.array([0.0, 15.0, 0.5])
    ou = OUState(chan_h=1.0, chan_v=1.0)
    f = knuckle_force(v, ou, rho=1.2, area=0.0346, re=2.0e5)
    dot = float(np.dot(f, v / np.linalg.norm(v)))
    assert abs(dot) < 1e-9


def test_knuckle_force_zero_when_ou_zero():
    v = np.array([0.0, 15.0, 0.5])
    ou = OUState(chan_h=0.0, chan_v=0.0)
    f = knuckle_force(v, ou, rho=1.2, area=0.0346, re=2.0e5)
    np.testing.assert_array_almost_equal(f, np.zeros(3))


def test_knuckle_force_zero_when_cl_lat_zero():
    # At Re far from the peak, cl_lat_rms ~ 0 -> force ~ 0
    v = np.array([0.0, 15.0, 0.5])
    ou = OUState(chan_h=1.0, chan_v=1.0)
    f = knuckle_force(v, ou, rho=1.2, area=0.0346, re=80_000)
    assert float(np.linalg.norm(f)) < 0.01
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/test_physics.py -k knuckle -v`
Expected: 4 errors, "cannot import name 'knuckle_force'".

- [ ] **Step 3: Implement `knuckle_force()`**

Append to `/home/siriux/Projects/vb_coach/src/vb_coach/physics.py`:

```python
from vb_coach.aerodynamics.knuckle import OUState, cl_lat_rms


_KNUCKLE_SPEED_CUTOFF: float = 0.1
_KNUCKLE_VERTICAL_FRACTION: float = 0.5


def knuckle_force(
    velocity: np.ndarray,
    ou: OUState,
    rho: float,
    area: float,
    re: float,
) -> np.ndarray:
    """Panel-induced lateral force on a moving ball.

    Decomposes the lateral force into two flight-perpendicular channels:
    h_hat (horizontal-perpendicular, in the x-y plane) and vert_hat
    (out-of-plane vertical perturbation). The vertical channel is weighted
    at 0.5 because observed wobble is dominated by horizontal deflection.

    NOTE: this function does not step the OU process. The integrator calls
    step_ou separately so dt is owned at the simulator level.
    """
    speed = float(np.linalg.norm(velocity))
    if speed < _KNUCKLE_SPEED_CUTOFF:
        return np.zeros(3)
    v_hat = velocity / speed
    cl = cl_lat_rms(re)
    f_mag = 0.5 * rho * area * cl * speed * speed
    vxy = np.array([velocity[0], velocity[1], 0.0])
    vxy_norm = float(np.linalg.norm(vxy))
    if vxy_norm > 1e-6:
        h_hat = np.array([-vxy[1], vxy[0], 0.0]) / vxy_norm
    else:
        h_hat = np.array([0.0, 1.0, 0.0])
    vert_hat = np.cross(v_hat, h_hat)
    return f_mag * (ou.chan_h * h_hat + _KNUCKLE_VERTICAL_FRACTION * ou.chan_v * vert_hat)
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_physics.py -v`
Expected: 13 passed (9 original + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/vb_coach/physics.py tests/test_physics.py
git commit -m "[feat:physics] add knuckle_force for stochastic lateral perturbation"
```

---

## Task 5: Switch `simulate()` to use `cd_full_regime`

Replace the constant placeholder with the full-regime model. Behavior unchanged for current tests because cd_full_regime returns 0.45 for Re < 1.5×10⁵, which covers all setting scenarios.

**Files:**
- Modify: `src/vb_coach/trajectory.py`

- [ ] **Step 1: Edit `_rhs` in `trajectory.py`**

In `/home/siriux/Projects/vb_coach/src/vb_coach/trajectory.py`, replace the import line:

```python
from vb_coach.aerodynamics import cd_subcritical, cl_magnus, reynolds, spin_parameter
```

with:

```python
from vb_coach.aerodynamics import cd_full_regime, cl_magnus, reynolds, spin_parameter
```

And inside `_rhs`, replace the line:

```python
        cd = cd_subcritical(re)
```

with:

```python
        cd = cd_full_regime(re)
```

- [ ] **Step 2: Run full test suite, verify all 45+ original tests still pass**

Run: `uv run pytest -q`
Expected: 58 passed (45 original + 3 cd_full_regime + 6 knuckle + 4 knuckle_force).

- [ ] **Step 3: Commit**

```bash
git add src/vb_coach/trajectory.py
git commit -m "[refactor:trajectory] use cd_full_regime so simulate() handles high-Re drag"
```

---

## Task 6: Add `simulate_stochastic()` to `trajectory.py`

Forward / symplectic-Euler integrator that advances the OU process in lockstep with the trajectory. Used for float-serve regimes.

**Files:**
- Modify: `src/vb_coach/trajectory.py`
- Modify: `tests/test_trajectory.py`

- [ ] **Step 1: Write failing tests**

Append to `/home/siriux/Projects/vb_coach/tests/test_trajectory.py`:

```python
from vb_coach.trajectory import simulate_stochastic


def test_simulate_stochastic_returns_trajectory():
    rs = ReleaseState(
        position=np.array([0.0, -9.0, 2.4]),
        velocity=np.array([0.0, 15.0, 0.5]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    rng = np.random.default_rng(0)
    traj = simulate_stochastic(rs, rng=rng, t_max=1.5, dt=0.005)
    assert isinstance(traj, Trajectory)
    assert len(traj) > 10


def test_simulate_stochastic_terminates_on_floor():
    rs = ReleaseState(
        position=np.array([0.0, -9.0, 2.4]),
        velocity=np.array([0.0, 15.0, 0.5]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    rng = np.random.default_rng(0)
    traj = simulate_stochastic(rs, rng=rng, t_max=3.0, dt=0.005)
    assert traj.position[-1, 2] <= 0.01


def test_simulate_stochastic_deterministic_without_rng():
    # Without RNG, no lateral knuckle force. Two runs must match exactly.
    rs = ReleaseState(
        position=np.array([0.0, -9.0, 2.4]),
        velocity=np.array([0.0, 15.0, 0.5]),
        spin=np.array([0.0, 0.0, 0.0]),
    )
    traj_a = simulate_stochastic(rs, rng=None, t_max=1.5, dt=0.005)
    traj_b = simulate_stochastic(rs, rng=None, t_max=1.5, dt=0.005)
    np.testing.assert_array_equal(traj_a.position, traj_b.position)


def test_simulate_stochastic_monte_carlo_landing_scatter():
    # Regression: 50 seeds at v=16 m/s should produce sideline sigma ~30 cm
    # to match the float-serve research doc and the original demo's output.
    p0 = np.array([0.0, -9.0, 2.4])
    v0 = np.array([0.0, 16.0, 0.5])
    landings_x = []
    for seed in range(50):
        rs = ReleaseState(
            position=p0.copy(),
            velocity=v0.copy(),
            spin=np.array([0.0, 0.0, 0.0]),
        )
        rng = np.random.default_rng(seed)
        traj = simulate_stochastic(rs, rng=rng, t_max=1.5, dt=0.005)
        landings_x.append(float(traj.position[-1, 0]))
    sigma_x_cm = float(np.std(landings_x)) * 100
    # Tolerate +/- 8 cm of the 30 cm value reported in the research doc
    assert 22 < sigma_x_cm < 38
```

- [ ] **Step 2: Run tests, verify fail**

Run: `uv run pytest tests/test_trajectory.py -k stochastic -v`
Expected: 4 errors, "cannot import name 'simulate_stochastic'".

- [ ] **Step 3: Implement `simulate_stochastic()` in `trajectory.py`**

Append to `/home/siriux/Projects/vb_coach/src/vb_coach/trajectory.py`:

```python
from vb_coach.aerodynamics.knuckle import OUState, step_ou
from vb_coach.physics import knuckle_force


def simulate_stochastic(
    release: ReleaseState,
    rng: np.random.Generator | None,
    t_max: float = 1.5,
    dt: float = 0.005,
    mass: float = BALL_MASS,
    rho: float = RHO_AIR,
    area: float = BALL_CROSS_SECTION,
    radius: float = BALL_RADIUS,
    diameter: float = BALL_DIAMETER,
    y_bound: float = 20.0,
) -> Trajectory:
    """Symplectic Euler integration with optional OU lateral force.

    Used for float-serve regime where the panel-induced unsteady lateral
    force must be advanced in lockstep with the position. If rng is None
    the lateral force is zero and the result is deterministic (gravity +
    drag + Magnus only).

    Terminates when z < 0 or |y| > y_bound.
    """
    n_steps = int(t_max / dt)
    t = np.zeros(n_steps + 1)
    pos = np.zeros((n_steps + 1, 3))
    vel = np.zeros((n_steps + 1, 3))
    pos[0] = release.position.astype(float)
    vel[0] = release.velocity.astype(float)
    ou = OUState()

    for i in range(n_steps):
        v = vel[i]
        speed = float(np.linalg.norm(v))
        f_grav = gravity_force(mass)
        if speed > 0.0:
            re = reynolds(speed, diameter, NU_AIR)
            cd = cd_full_regime(re)
            f_drag = drag_force(v, rho, area, cd)
            f_mag = magnus_force(v, release.spin, rho, area, radius)
        else:
            re = 0.0
            f_drag = np.zeros(3)
            f_mag = np.zeros(3)
        if rng is not None and speed > 0.1:
            step_ou(ou, dt, rng)
            f_lat = knuckle_force(v, ou, rho, area, re)
        else:
            f_lat = np.zeros(3)
        a = (f_grav + f_drag + f_mag + f_lat) / mass
        vel[i + 1] = v + a * dt
        pos[i + 1] = pos[i] + vel[i + 1] * dt
        t[i + 1] = t[i] + dt
        if pos[i + 1, 2] < 0.0 or abs(pos[i + 1, 1]) > y_bound:
            stop = i + 2
            spins = np.broadcast_to(release.spin, (stop, 3)).copy()
            return Trajectory(
                time=t[:stop], position=pos[:stop], velocity=vel[:stop], spin=spins
            )
    spins = np.broadcast_to(release.spin, (n_steps + 1, 3)).copy()
    return Trajectory(time=t, position=pos, velocity=vel, spin=spins)
```

Also ensure these names are imported at the top of `trajectory.py`:

```python
from vb_coach.aerodynamics.drag import cd_full_regime
from vb_coach.physics import drag_force, gravity_force, magnus_force
```

If they aren't already imported there, add them with the existing imports.

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_trajectory.py -v`
Expected: 14 passed (10 original + 4 new).

- [ ] **Step 5: Commit**

```bash
git add src/vb_coach/trajectory.py tests/test_trajectory.py
git commit -m "[feat:trajectory] add simulate_stochastic for float-serve regime"
```

---

## Task 7: Rewrite `float_serve_demo.py` on the shared library

Removes duplicated physics. The demo becomes a thin scenario-runner + plotter.

**Files:**
- Modify: `examples/float_serve_demo.py`

- [ ] **Step 1: Replace the file's content**

Overwrite `/home/siriux/Projects/vb_coach/examples/float_serve_demo.py` with:

```python
"""Float-serve trajectory demonstration (library-backed).

Reproduces the three canonical float-serve trajectory types from
docs/research/float-serve.md, using the vb_coach trajectory engine:

    (A) Persistent wobble   - release v ~ 15 m/s, full flight in critical band
    (B) Drop floater        - release v ~ 19 m/s, crosses critical band
    (C) Below sweet spot    - release v ~ 13 m/s, sags but does not float

Plus a Monte Carlo experiment at v = 16 m/s showing landing scatter.

Run:
    uv run python examples/float_serve_demo.py

Outputs:
    docs/research/figures/float_trajectories.png
    docs/research/figures/float_montecarlo.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vb_coach.aerodynamics import cd_full_regime, cl_lat_rms
from vb_coach.constants import BALL_DIAMETER, NU_AIR
from vb_coach.trajectory import ReleaseState, simulate_stochastic


def _release(v0_mag: float, z0: float = 2.4, vz: float = 0.5) -> ReleaseState:
    return ReleaseState(
        position=np.array([0.0, -9.0, z0]),
        velocity=np.array([0.0, v0_mag, vz]),
        spin=np.array([0.0, 0.0, 0.0]),
    )


def plot_canonical() -> plt.Figure:
    scenarios = [
        ("A: persistent wobble (v0=15)", _release(15.0), "tab:blue", 7),
        ("B: drop floater (v0=19)", _release(19.0, z0=2.8, vz=-0.5), "tab:orange", 3),
        ("C: below sweet spot (v0=13)", _release(13.0), "tab:green", 12),
    ]

    fig = plt.figure(figsize=(14, 5))
    ax_top = fig.add_subplot(1, 3, 1)
    ax_side = fig.add_subplot(1, 3, 2)
    ax_re = fig.add_subplot(1, 3, 3)

    for label, rs, color, seed in scenarios:
        rng = np.random.default_rng(seed)
        traj = simulate_stochastic(rs, rng=rng, t_max=1.5, dt=0.005)
        pos = traj.position
        t = traj.time
        ax_top.plot(pos[:, 1], pos[:, 0], lw=2, label=label, color=color)
        ax_side.plot(pos[:, 1], pos[:, 2], lw=2, label=label, color=color)
        v = np.diff(pos, axis=0) / np.diff(t)[:, None]
        speed = np.linalg.norm(v, axis=1)
        re = speed * BALL_DIAMETER / NU_AIR
        ax_re.plot(t[1:], re, lw=2, label=label, color=color)

    ax_top.axhline(0, color="black", lw=0.5)
    ax_top.axvline(0, color="gray", lw=1, alpha=0.5, label="net (y=0)")
    ax_top.set_xlabel("y [m] (downcourt)")
    ax_top.set_ylabel("x [m] (sideline drift)")
    ax_top.set_title("Top-down: lateral wobble")
    ax_top.set_ylim(-0.8, 0.8)
    ax_top.legend(fontsize=8, loc="lower right")
    ax_top.grid(alpha=0.3)

    ax_side.axvline(0, color="gray", lw=1, alpha=0.5)
    ax_side.plot([0, 0], [0, 2.43], color="gray", lw=2, alpha=0.7, label="net")
    ax_side.axhline(0, color="black", lw=0.5)
    ax_side.set_xlabel("y [m] (downcourt)")
    ax_side.set_ylabel("z [m]")
    ax_side.set_title("Side view: drop and approach")
    ax_side.legend(fontsize=8, loc="upper right")
    ax_side.grid(alpha=0.3)

    ax_re.axhspan(1.5e5, 2.5e5, color="red", alpha=0.1, label="critical band")
    ax_re.set_xlabel("t [s]")
    ax_re.set_ylabel("Re")
    ax_re.set_title("Reynolds number through flight")
    ax_re.legend(fontsize=8, loc="upper right")
    ax_re.grid(alpha=0.3)

    fig.suptitle("Three canonical float-serve trajectories", fontsize=12)
    fig.tight_layout()
    return fig


def plot_montecarlo(n_trials: int = 50, v0_mag: float = 16.0):
    rs_template = _release(v0_mag)
    landings = []
    for seed in range(n_trials):
        rs = ReleaseState(
            position=rs_template.position.copy(),
            velocity=rs_template.velocity.copy(),
            spin=rs_template.spin.copy(),
        )
        rng = np.random.default_rng(seed)
        traj = simulate_stochastic(rs, rng=rng, t_max=1.5, dt=0.005)
        landings.append(traj.position[-1])
    landings = np.array(landings)

    traj_det = simulate_stochastic(rs_template, rng=None, t_max=1.5, dt=0.005)
    landing_det = traj_det.position[-1]

    fig, (ax_xy, ax_yz) = plt.subplots(1, 2, figsize=(12, 5))
    ax_xy.scatter(landings[:, 1], landings[:, 0], alpha=0.6, s=30,
                  label=f"float ({n_trials} trials)")
    ax_xy.scatter([landing_det[1]], [landing_det[0]], color="red",
                  marker="x", s=120, label="no-noise reference")
    ax_xy.set_xlabel("y [m] (downcourt landing)")
    ax_xy.set_ylabel("x [m] (sideline landing)")
    ax_xy.set_title(f"Landing scatter, v0 = {v0_mag} m/s, omega = 0")
    ax_xy.legend(fontsize=9)
    ax_xy.grid(alpha=0.3)
    ax_xy.set_aspect("equal", adjustable="datalim")

    x_std = float(np.std(landings[:, 0]))
    y_std = float(np.std(landings[:, 1]))
    z_std = float(np.std(landings[:, 2]))
    ax_xy.text(0.02, 0.98,
               f"sideline σ_x = {x_std*100:.1f} cm\n"
               f"range σ_y = {y_std*100:.1f} cm",
               transform=ax_xy.transAxes, verticalalignment="top",
               fontsize=10, bbox=dict(facecolor="white", alpha=0.8))

    ax_yz.hist(landings[:, 1], bins=15, alpha=0.7, color="tab:blue")
    ax_yz.axvline(landing_det[1], color="red", lw=2, label="no-noise reference")
    ax_yz.set_xlabel("y at landing [m]")
    ax_yz.set_ylabel("count")
    ax_yz.set_title("Distribution of downcourt landing point")
    ax_yz.legend(fontsize=9)
    ax_yz.grid(alpha=0.3)

    fig.tight_layout()
    return fig, {"x_std_cm": x_std * 100, "y_std_cm": y_std * 100, "z_std_cm": z_std * 100}


def main() -> None:
    out_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "docs", "research", "figures")
    )
    os.makedirs(out_dir, exist_ok=True)

    print("Generating canonical-trajectories figure ...")
    fig1 = plot_canonical()
    p1 = os.path.join(out_dir, "float_trajectories.png")
    fig1.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig1)
    print(f"  -> {p1}")

    print("Running Monte Carlo (50 trials at v=16 m/s) ...")
    fig2, stats = plot_montecarlo(n_trials=50, v0_mag=16.0)
    p2 = os.path.join(out_dir, "float_montecarlo.png")
    fig2.savefig(p2, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    print(f"  -> {p2}")
    print("Landing-point scatter at v=16 m/s, omega=0:")
    for k, v in stats.items():
        print(f"  {k} = {v:.1f}")

    print("\n--- Sanity checks (matching values quoted in research doc) ---")
    print(f"Cd at Re=84k:  {cd_full_regime(84_000):.3f}  (expect 0.45)")
    print(f"Cd at Re=200k: {cd_full_regime(200_000):.3f}  (expect ~0.31)")
    print(f"Cd at Re=300k: {cd_full_regime(300_000):.3f}  (expect 0.18)")
    print(f"Cl_lat,RMS at Re=200k: {cl_lat_rms(200_000):.3f}  (expect ~0.12)")
    print(f"Cl_lat,RMS at Re=80k:  {cl_lat_rms(80_000):.3f}   (expect ~0)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the rewritten demo**

Run: `uv run python examples/float_serve_demo.py`

Expected output includes:
```
Cd at Re=84k:  0.450  (expect 0.45)
Cd at Re=200k: 0.315  (expect ~0.31)
Cd at Re=300k: 0.180  (expect 0.18)
Cl_lat,RMS at Re=200k: 0.120  (expect ~0.12)
Cl_lat,RMS at Re=80k:  0.002  (expect ~0)
```

And `sigma_x ~30 cm` in the Monte Carlo (must be within 22-38 cm to match the regression test from Task 6).

- [ ] **Step 3: Verify the regenerated figures look the same**

Visually inspect:
- `docs/research/figures/float_trajectories.png`
- `docs/research/figures/float_montecarlo.png`

They should show the same three canonical trajectories and the same scatter pattern as before. Sub-pixel differences are acceptable; structural differences are not.

- [ ] **Step 4: Run the full test suite one final time**

Run: `uv run pytest -q`
Expected: ALL tests pass.

- [ ] **Step 5: Commit**

```bash
git add examples/float_serve_demo.py docs/research/figures/
git commit -m "[refactor:examples] rewrite float_serve_demo on shared physics library

Demo shrinks from 369 to ~150 lines. All physics now imported from
vb_coach.aerodynamics and vb_coach.trajectory. Canonical trajectories
and Monte Carlo landing scatter match the pre-refactor outputs."
```

---

## Verification checklist

- [ ] `uv run pytest -q` passes all tests (target: ~60 passed, all green)
- [ ] `uv run python examples/basic_set.py` still produces the 4号位 high-ball example
- [ ] `uv run python examples/float_serve_demo.py` produces both figures and matching sanity-check numbers
- [ ] `grep -r "cd = 0.45" src/` returns nothing — the constant only lives in `aerodynamics/drag.py`
- [ ] `grep -r "from vb_coach.aerodynamics" src/ tests/ examples/` shows imports from the package (not from the old single-file module)
- [ ] The float worktree at `.worktrees/float/` is untouched

## What comes next (not in this plan)

- Phase A1: define 击球窗口 geometry + per-attack-type quality metrics
- Phase A1 attack-type models (高球 / 半高 / 平拉开 first)
- Eventually a `serve/` package mirroring `setting/` for serve-side quality analysis if the float-serve research grows
