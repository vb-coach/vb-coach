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
    """Standard server position: at the end-line center, contact height z0, speed v0_mag downcourt."""
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


def plot_montecarlo(n_trials: int = 50, v0_mag: float = 16.0) -> tuple[plt.Figure, dict[str, float]]:
    landings = []
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        traj = simulate_stochastic(_release(v0_mag), rng=rng, t_max=1.5, dt=0.005)
        landings.append(traj.position[-1])
    landings = np.array(landings)

    traj_det = simulate_stochastic(_release(v0_mag), rng=None, t_max=1.5, dt=0.005)
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
