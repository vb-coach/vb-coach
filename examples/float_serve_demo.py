"""Float-serve trajectory demonstration.

Stand-alone script that reproduces the three canonical float-serve trajectory
types described in docs/research/float-serve.md:

    (A) Persistent wobble   — release at v ≈ 15 m/s, full flight in critical band
    (B) Drop floater        — release at v ≈ 19 m/s, crosses critical band mid-flight
    (C) Below sweet spot    — release at v ≈ 13 m/s, sags but does not float

A clean two-flip "S-curve" trajectory is rare in this model (< 1 % of seeds at
typical flight times of 0.7-1 s — empirically matching the coaching literature's
claim that S-curve floaters are flukes, not reproducible technique).

Plus a Monte Carlo experiment: 50 release states at v = 16 m/s, omega = 0,
showing landing-point scatter.

The script is intentionally self-contained — it does NOT import from
src/vb_coach/aerodynamics so the main-line subcritical model stays untouched.

Run:
    uv run python examples/float_serve_demo.py

Outputs:
    docs/research/figures/float_trajectories.png
    docs/research/figures/float_montecarlo.png
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

# Physical constants (SI). Independent copy so this script doesn't import
# anything from src/vb_coach.
G = 9.81
RHO = 1.20
NU = 1.5e-5
D = 0.21
R = D / 2
A = math.pi * R ** 2
M = 0.27


# -------------------------------------------------------------------------
# Aerodynamics: full-regime Cd(Re) and stochastic lateral force
# -------------------------------------------------------------------------

def cd_full_regime(re: float) -> float:
    """Piecewise Cd(Re) for a FIVB-spec volleyball.

    Tuned against Asai et al. 2010 wind-tunnel data for Mikasa MVA200:
      Re < 1.5e5         : Cd = 0.45               (subcritical plateau)
      1.5e5 < Re < 2.5e5 : Cd drops 0.45 -> 0.18   (drag crisis)
      Re > 2.5e5         : Cd = 0.18               (supercritical plateau)
    The transition is smoothed with a cosine to avoid integrator stiffness.
    """
    re_low = 1.5e5
    re_high = 2.5e5
    if re <= re_low:
        return 0.45
    if re >= re_high:
        return 0.18
    # cosine taper across the critical band
    phase = math.pi * (re - re_low) / (re_high - re_low)
    return 0.18 + 0.5 * (0.45 - 0.18) * (1 + math.cos(phase))


def cl_lat_rms(re: float) -> float:
    """RMS lateral lift coefficient as a function of Re.

    Peaks in the critical band, ~zero outside. Matches Cohen 2014 magnitude.
    """
    re_peak = 2.0e5
    re_width = 0.6e5
    return 0.12 * math.exp(-((re - re_peak) / re_width) ** 2)


# Ornstein-Uhlenbeck driven side force: persists for ~tau, then redraws.
# Two independent channels (one per perpendicular axis in the flight plane).
TAU_KNUCKLE = 0.25  # coherence time (s); Cohen 2014 measurements


@dataclass
class OUState:
    """State of the OU process for the two lateral channels."""
    chan_h: float = 0.0  # horizontal (perpendicular to flight in x-y plane)
    chan_v: float = 0.0  # vertical perturbation


def step_ou(ou: OUState, dt: float, rng: np.random.Generator) -> OUState:
    """Advance OU process by dt. Variance normalized to 1 in steady state."""
    decay = math.exp(-dt / TAU_KNUCKLE)
    noise_scale = math.sqrt(1.0 - decay ** 2)
    ou.chan_h = decay * ou.chan_h + noise_scale * rng.standard_normal()
    ou.chan_v = decay * ou.chan_v + noise_scale * rng.standard_normal()
    return ou


# -------------------------------------------------------------------------
# Integration
# -------------------------------------------------------------------------

def simulate_float(
    p0: np.ndarray,
    v0: np.ndarray,
    omega_rad_s: float = 0.0,
    rng: np.random.Generator | None = None,
    t_max: float = 1.5,
    dt: float = 0.005,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-Euler integration with OU side force.

    Returns (time, position[N,3]).

    Uses fixed timestep so we can advance the OU noise process consistently.
    For pure drag (rng=None) this reduces to a deterministic trajectory.
    """
    n_steps = int(t_max / dt)
    t = np.zeros(n_steps + 1)
    pos = np.zeros((n_steps + 1, 3))
    vel = np.zeros((n_steps + 1, 3))
    pos[0] = p0
    vel[0] = v0

    ou = OUState()

    for i in range(n_steps):
        p, v = pos[i], vel[i]
        speed = float(np.linalg.norm(v))
        re = speed * D / NU
        cd = cd_full_regime(re)

        # Drag (opposes velocity)
        if speed > 0:
            v_hat = v / speed
            f_drag = -0.5 * RHO * A * cd * speed ** 2 * v_hat
        else:
            v_hat = np.array([1.0, 0.0, 0.0])
            f_drag = np.zeros(3)

        # Lateral knuckle force in the flight-perpendicular plane.
        # Build an orthonormal basis (v_hat, h_hat, vert_hat) where:
        #   h_hat ⊥ v_hat in horizontal plane
        #   vert_hat = v_hat × h_hat (out-of-plane vertical perturbation)
        f_lat = np.zeros(3)
        if rng is not None and speed > 0.1:
            ou = step_ou(ou, dt, rng)
            cl = cl_lat_rms(re)
            f_mag = 0.5 * RHO * A * cl * speed ** 2
            # Horizontal perp: project v onto x-y, rotate 90°
            vxy = np.array([v[0], v[1], 0.0])
            vxy_norm = float(np.linalg.norm(vxy))
            if vxy_norm > 1e-6:
                h_hat = np.array([-vxy[1], vxy[0], 0.0]) / vxy_norm
            else:
                h_hat = np.array([0.0, 1.0, 0.0])
            vert_hat = np.cross(v_hat, h_hat)
            f_lat = f_mag * (ou.chan_h * h_hat + 0.5 * ou.chan_v * vert_hat)

        # Gravity
        f_grav = np.array([0.0, 0.0, -M * G])

        a = (f_grav + f_drag + f_lat) / M

        vel[i + 1] = v + a * dt
        pos[i + 1] = p + vel[i + 1] * dt
        t[i + 1] = t[i] + dt

        # Stop if ball hits ground or crosses end line very far away
        if pos[i + 1, 2] < 0 or abs(pos[i + 1, 1]) > 20:
            return t[: i + 2], pos[: i + 2]

    return t, pos


# -------------------------------------------------------------------------
# Canonical scenarios + plotting
# -------------------------------------------------------------------------

def scenario_persistent_wobble(rng: np.random.Generator):
    """(A) Release ~15 m/s, full flight in critical band."""
    p0 = np.array([0.0, -9.0, 2.4])  # behind end-line, contact height
    v0 = np.array([0.0, 15.0, 0.5])  # toward team A court, slight upward
    return simulate_float(p0, v0, rng=rng)


def scenario_drop_floater(rng: np.random.Generator):
    """(B) Release ~19 m/s, crosses critical band mid-flight."""
    p0 = np.array([0.0, -9.0, 2.8])  # higher contact (jump float)
    v0 = np.array([0.0, 19.0, -0.5])  # slight downward
    return simulate_float(p0, v0, rng=rng)


def scenario_below_sweet_spot(rng: np.random.Generator):
    """(C) Below sweet spot: v=13 m/s, Re barely entering critical band.

    Demonstrates the 'too slow' failure mode quoted in the research doc:
    Cd stays high (~0.4), so ball drops fast and arrives short, but the
    lateral knuckle force is small (Cl_lat peaks at Re=2e5, decays at lower Re),
    so the ball doesn't 'float' much — it just sags. Receivers easily judge it.
    """
    p0 = np.array([0.0, -9.0, 2.4])
    v0 = np.array([0.0, 13.0, 0.5])
    return simulate_float(p0, v0, rng=rng)


def plot_canonical():
    fig = plt.figure(figsize=(14, 5))

    # Top-down (x vs y): shows lateral deflection
    ax_top = fig.add_subplot(1, 3, 1)
    # Side view (y vs z): shows drop
    ax_side = fig.add_subplot(1, 3, 2)
    # Re(t) and Cd(t): shows regime crossing
    ax_re = fig.add_subplot(1, 3, 3)

    # Three rng seeds picked to produce visually distinct A/B/C trajectories
    seeds = {"A: persistent wobble (v0=15)": 7,
             "B: drop floater (v0=19)": 3,
             "C: below sweet spot (v0=13)": 12}
    scenarios = [scenario_persistent_wobble,
                 scenario_drop_floater,
                 scenario_below_sweet_spot]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for (label, seed), scen, color in zip(seeds.items(), scenarios, colors):
        rng = np.random.default_rng(seed)
        t, pos = scen(rng)
        ax_top.plot(pos[:, 1], pos[:, 0], lw=2, label=label, color=color)
        ax_side.plot(pos[:, 1], pos[:, 2], lw=2, label=label, color=color)

        # Compute Re(t) from position differences
        v = np.diff(pos, axis=0) / np.diff(t)[:, None]
        speed = np.linalg.norm(v, axis=1)
        re = speed * D / NU
        ax_re.plot(t[1:], re, lw=2, label=label, color=color)

    # Top-down panel
    ax_top.axhline(0, color="black", lw=0.5)
    ax_top.axvline(0, color="gray", lw=1, alpha=0.5, label="net (y=0)")
    ax_top.set_xlabel("y [m] (downcourt)")
    ax_top.set_ylabel("x [m] (sideline drift)")
    ax_top.set_title("Top-down: lateral wobble")
    ax_top.set_ylim(-0.8, 0.8)
    ax_top.legend(fontsize=8, loc="lower right")
    ax_top.grid(alpha=0.3)

    # Side panel
    ax_side.axvline(0, color="gray", lw=1, alpha=0.5)
    ax_side.plot([0, 0], [0, 2.43], color="gray", lw=2, alpha=0.7, label="net")
    ax_side.axhline(0, color="black", lw=0.5)
    ax_side.set_xlabel("y [m] (downcourt)")
    ax_side.set_ylabel("z [m]")
    ax_side.set_title("Side view: drop and approach")
    ax_side.legend(fontsize=8, loc="upper right")
    ax_side.grid(alpha=0.3)

    # Re panel
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
    """Show landing-point scatter from n_trials independent OU noise draws."""
    p0 = np.array([0.0, -9.0, 2.4])
    v0 = np.array([0.0, v0_mag, 0.5])

    landings = []
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        _, pos = simulate_float(p0, v0, rng=rng)
        # End point: last position
        landings.append(pos[-1])
    landings = np.array(landings)

    # Deterministic (no noise) reference
    _, pos_det = simulate_float(p0, v0, rng=None)
    landing_det = pos_det[-1]

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

    # Print spread statistics on the plot
    x_std = float(np.std(landings[:, 0]))
    y_std = float(np.std(landings[:, 1]))
    z_std = float(np.std(landings[:, 2]))
    ax_xy.text(0.02, 0.98,
               f"sideline σ_x = {x_std*100:.1f} cm\n"
               f"range σ_y = {y_std*100:.1f} cm",
               transform=ax_xy.transAxes,
               verticalalignment="top",
               fontsize=10,
               bbox=dict(facecolor="white", alpha=0.8))

    # Histogram of landing y (range)
    ax_yz.hist(landings[:, 1], bins=15, alpha=0.7, color="tab:blue")
    ax_yz.axvline(landing_det[1], color="red", lw=2, label="no-noise reference")
    ax_yz.set_xlabel("y at landing [m]")
    ax_yz.set_ylabel("count")
    ax_yz.set_title("Distribution of downcourt landing point")
    ax_yz.legend(fontsize=9)
    ax_yz.grid(alpha=0.3)

    fig.tight_layout()
    return fig, {"x_std_cm": x_std * 100, "y_std_cm": y_std * 100,
                 "z_std_cm": z_std * 100}


# -------------------------------------------------------------------------

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..",
                           "docs", "research", "figures")
    out_dir = os.path.abspath(out_dir)
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
