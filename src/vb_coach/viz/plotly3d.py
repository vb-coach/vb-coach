"""Interactive 3D plots for vb_coach attack analysis using Plotly.

The primary entry point is plot_attack_3d(), which assembles a court
scene + trajectory + hit window + attacker silhouette into one
rotatable/zoomable browser-renderable Plotly Figure.

Coordinate convention (from coords.py):
    origin under net center on the floor
    +x along the net (team A's right)
    +y across the net into team A's court
    +z up

To save as HTML:
    fig = plot_attack_3d(traj, attacker, window)
    fig.write_html("attack.html")
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from vb_coach.constants import (
    ANTENNA_HEIGHT_ABOVE_NET,
    ATTACK_LINE_FROM_NET,
    COURT_LENGTH,
    COURT_WIDTH,
    NET_HEIGHT_MEN,
)
from vb_coach.setting.attacker import Attacker
from vb_coach.setting.hit_window import HitWindow
from vb_coach.trajectory import Trajectory


def _court_traces(net_height: float) -> list[go.Scatter3d | go.Mesh3d]:
    """Return list of traces for court floor, boundary, net, antennae, attack lines."""
    half_x = COURT_WIDTH / 2          # 4.5
    half_y = COURT_LENGTH / 2         # 9.0
    antenna_top = net_height + ANTENNA_HEIGHT_ABOVE_NET
    traces = []

    # Court boundary (closed rectangle at z=0)
    bx = [-half_x, half_x, half_x, -half_x, -half_x]
    by = [-half_y, -half_y, half_y, half_y, -half_y]
    bz = [0, 0, 0, 0, 0]
    traces.append(go.Scatter3d(
        x=bx, y=by, z=bz, mode="lines",
        line=dict(color="#444444", width=4),
        name="court boundary", showlegend=False, hoverinfo="skip",
    ))

    # Attack lines (y = +/-3, full court width)
    for y in (-ATTACK_LINE_FROM_NET, ATTACK_LINE_FROM_NET):
        traces.append(go.Scatter3d(
            x=[-half_x, half_x], y=[y, y], z=[0, 0], mode="lines",
            line=dict(color="#888888", width=2, dash="dot"),
            name="attack line", showlegend=False, hoverinfo="skip",
        ))

    # Center line under the net (y=0)
    traces.append(go.Scatter3d(
        x=[-half_x, half_x], y=[0, 0], z=[0, 0], mode="lines",
        line=dict(color="#aaaaaa", width=2),
        name="net line", showlegend=False, hoverinfo="skip",
    ))

    # Net surface (rectangular mesh at y=0)
    net_x = [-half_x, half_x, half_x, -half_x]
    net_y = [0, 0, 0, 0]
    net_z = [0, 0, net_height, net_height]
    traces.append(go.Mesh3d(
        x=net_x, y=net_y, z=net_z, i=[0, 0], j=[1, 2], k=[2, 3],
        color="#bbbbbb", opacity=0.25, name="net", showlegend=True, hoverinfo="skip",
    ))

    # Antennae (vertical lines at +/-4.5 from net to antenna_top)
    for x in (-half_x, half_x):
        traces.append(go.Scatter3d(
            x=[x, x], y=[0, 0], z=[0, antenna_top], mode="lines",
            line=dict(color="#ff0000", width=4),
            name="antenna", showlegend=False, hoverinfo="skip",
        ))

    return traces


def _trajectory_trace(traj: Trajectory) -> go.Scatter3d:
    """Return a colored 3D line for the trajectory.

    Hover text shows t, x, y, z, and speed at each sampled point.
    """
    speed = np.linalg.norm(traj.velocity, axis=1)
    hover = [
        f"t={t:.2f}s<br>x={p[0]:.2f}m<br>y={p[1]:.2f}m<br>z={p[2]:.2f}m<br>|v|={v:.2f}m/s"
        for t, p, v in zip(traj.time, traj.position, speed)
    ]
    return go.Scatter3d(
        x=traj.position[:, 0],
        y=traj.position[:, 1],
        z=traj.position[:, 2],
        mode="lines+markers",
        marker=dict(size=2, color=traj.time, colorscale="Viridis",
                    colorbar=dict(title="t [s]", x=1.05)),
        line=dict(color="#1f77b4", width=4),
        name="trajectory",
        hovertext=hover,
        hoverinfo="text",
    )


def _key_point_markers(traj: Trajectory, contact_index: int | None) -> list[go.Scatter3d]:
    """Return markers for release (green), apex (orange), and contact (red, if provided)."""
    markers = []
    # Release
    p = traj.position[0]
    markers.append(go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
        marker=dict(size=8, color="green"), name="release",
    ))
    # Apex
    apex = traj.apex_index()
    p = traj.position[apex]
    markers.append(go.Scatter3d(
        x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
        marker=dict(size=8, color="orange"), name="apex",
    ))
    # Contact (if provided and valid)
    if contact_index is not None and contact_index >= 0:
        p = traj.position[contact_index]
        markers.append(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]], mode="markers",
            marker=dict(size=10, color="red", symbol="diamond"), name="contact",
        ))
    return markers


def _hit_window_trace(window: HitWindow, n_phi: int = 24, n_theta: int = 16) -> go.Mesh3d:
    """Return a translucent surface mesh for the asymmetric-ellipsoid hit window.

    Builds an axis-aligned ellipsoidal surface in spherical parametric form,
    then applies piecewise scaling so the -y and +y hemispheres use different
    radii (r_y_front vs r_y_back), and similarly for +z and -z (r_z_up vs
    r_z_down). The result is a closed surface stitched along the y=center[1]
    and z=center[2] planes.
    """
    theta = np.linspace(0.0, np.pi, n_theta)
    phi = np.linspace(0.0, 2 * np.pi, n_phi)
    THETA, PHI = np.meshgrid(theta, phi)
    x_unit = np.sin(THETA) * np.cos(PHI)
    y_unit = np.sin(THETA) * np.sin(PHI)
    z_unit = np.cos(THETA)
    x_s = window.r_x * x_unit
    y_s = np.where(y_unit < 0, window.r_y_front * y_unit, window.r_y_back * y_unit)
    z_s = np.where(z_unit > 0, window.r_z_up * z_unit, window.r_z_down * z_unit)
    cx, cy, cz = float(window.center[0]), float(window.center[1]), float(window.center[2])
    x = (x_s + cx).flatten()
    y = (y_s + cy).flatten()
    z = (z_s + cz).flatten()
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


def _attacker_silhouette_trace(attacker: Attacker, window: HitWindow) -> go.Scatter3d:
    """Return a simple stick figure for the attacker (vertical line at approach position)."""
    # Stand a stick behind the contact point on team A's side (y = window center + 0.4)
    x = float(window.center[0])
    y = float(window.center[1] + 0.4)
    return go.Scatter3d(
        x=[x, x], y=[y, y], z=[0.0, attacker.approach_reach],
        mode="lines+markers",
        line=dict(color="#8888ff", width=6),
        marker=dict(size=4, color="#8888ff"),
        name=f"attacker (reach={attacker.approach_reach:.2f}m)",
    )


def plot_attack_3d(
    trajectory: Trajectory,
    attacker: Attacker | None = None,
    window: HitWindow | None = None,
    score: object | None = None,
    net_height: float = NET_HEIGHT_MEN,
    title: str = "Volleyball attack trajectory",
) -> go.Figure:
    """Interactive 3D plot of a volleyball attack trajectory.

    Renders the court, net, antennae, attack lines, the trajectory (color
    graded by time), key points (release/apex/contact), the hit window
    (if provided), and an attacker silhouette (if provided). The score
    breakdown (if provided) is shown in the title.
    """
    traces: list = []
    traces.extend(_court_traces(net_height))

    # contact_index from score, if present
    contact_index = getattr(score, "contact_index", None)

    traces.append(_trajectory_trace(trajectory))
    traces.extend(_key_point_markers(trajectory, contact_index))

    if window is not None:
        traces.append(_hit_window_trace(window))
    if attacker is not None and window is not None:
        traces.append(_attacker_silhouette_trace(attacker, window))

    title_text = title
    if score is not None:
        s = getattr(score, "score", None)
        if s is not None:
            ds = getattr(score, "dwell_score", float("nan"))
            vs = getattr(score, "velocity_score", float("nan"))
            ls = getattr(score, "lateral_score", float("nan"))
            title_text = (
                f"{title}<br><sub>"
                f"score={s:.2f} = dwell {ds:.2f} * velocity {vs:.2f} * lateral {ls:.2f}"
                "</sub>"
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title_text,
        scene=dict(
            xaxis_title="x [m] (along net)",
            yaxis_title="y [m] (across court)",
            zaxis_title="z [m]",
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
    )
    return fig
