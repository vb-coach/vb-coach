"""Interactive 3D plots and animations for vb_coach using Plotly.

The primary generic entry points are plot_trajectory_3d() and
plot_trajectory_animation(). Both assemble a court scene plus a trajectory
into browser-renderable Plotly figures. Attack-specific overlays such as
hit windows, attacker silhouettes, and score diagnostics are optional data
layers, not assumptions baked into the renderer.

Coordinate convention (from coords.py):
    origin under net center on the floor
    +x along the net (team A's right)
    +y across the net into team A's court
    +z up

To save as HTML:
    fig = plot_trajectory_animation(traj, window=window)
    fig.write_html("attack.html")
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class AnimationConfig:
    """Playback settings for Plotly trajectory animations."""

    frame_count: int = 80
    frame_duration_ms: int = 35
    transition_ms: int = 0
    playback_speeds: tuple[float, ...] = (0.5, 1.0, 2.0)
    show_full_path: bool = True
    trail_width: int = 6
    ball_size: int = 7

    def __post_init__(self) -> None:
        if self.frame_count < 2:
            raise ValueError("frame_count must be at least 2")
        if self.frame_duration_ms <= 0:
            raise ValueError("frame_duration_ms must be positive")
        if not self.playback_speeds or any(speed <= 0 for speed in self.playback_speeds):
            raise ValueError("playback_speeds must contain positive values")


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


def _trajectory_hover(traj: Trajectory) -> list[str]:
    speed = np.linalg.norm(traj.velocity, axis=1)
    return [
        f"t={t:.2f}s<br>x={p[0]:.2f}m<br>y={p[1]:.2f}m<br>z={p[2]:.2f}m<br>|v|={v:.2f}m/s"
        for t, p, v in zip(traj.time, traj.position, speed)
    ]


def _trajectory_trace(traj: Trajectory) -> go.Scatter3d:
    """Return a colored 3D line for the trajectory.

    Hover text shows t, x, y, z, and speed at each sampled point.
    """
    return go.Scatter3d(
        x=traj.position[:, 0],
        y=traj.position[:, 1],
        z=traj.position[:, 2],
        mode="lines+markers",
        marker=dict(size=2, color=traj.time, colorscale="Viridis",
                    colorbar=dict(title="t [s]", x=1.05)),
        line=dict(color="#1f77b4", width=4),
        name="trajectory",
        hovertext=_trajectory_hover(traj),
        hoverinfo="text",
    )


def _reference_trajectory_trace(traj: Trajectory) -> go.Scatter3d:
    """Return a quiet dashed trajectory for comparison."""
    return go.Scatter3d(
        x=traj.position[:, 0],
        y=traj.position[:, 1],
        z=traj.position[:, 2],
        mode="lines",
        line=dict(color="#666666", width=3, dash="dash"),
        name="no-float reference",
        hovertext=_trajectory_hover(traj),
        hoverinfo="text",
    )


def _animated_trail_trace(traj: Trajectory, end_idx: int, width: int) -> go.Scatter3d:
    end = max(1, min(end_idx, len(traj) - 1))
    return go.Scatter3d(
        x=traj.position[: end + 1, 0],
        y=traj.position[: end + 1, 1],
        z=traj.position[: end + 1, 2],
        mode="lines",
        line=dict(color="#1f77b4", width=width),
        name="animated trail",
        hoverinfo="skip",
        showlegend=False,
    )


def _animated_ball_trace(traj: Trajectory, idx: int, size: int) -> go.Scatter3d:
    i = max(0, min(idx, len(traj) - 1))
    p = traj.position[i]
    return go.Scatter3d(
        x=[p[0]],
        y=[p[1]],
        z=[p[2]],
        mode="markers",
        marker=dict(size=size, color="#d62728"),
        name="ball",
        hovertext=[_trajectory_hover(traj)[i]],
        hoverinfo="text",
    )


def _sample_frame_indices(traj: Trajectory, frame_count: int) -> list[int]:
    if len(traj) == 0:
        raise ValueError("trajectory must contain at least one sample")
    n_frames = max(2, min(frame_count, len(traj)))
    return [int(i) for i in np.linspace(0, len(traj) - 1, n_frames)]


def _playback_buttons(cfg: AnimationConfig) -> list[dict]:
    buttons = []
    for speed in cfg.playback_speeds:
        duration = max(1, round(cfg.frame_duration_ms / speed))
        buttons.append(
            dict(
                label=f"Play {speed:g}x",
                method="animate",
                args=[
                    None,
                    {
                        "frame": {"duration": duration, "redraw": True},
                        "transition": {"duration": cfg.transition_ms},
                        "fromcurrent": True,
                        "mode": "immediate",
                    },
                ],
            )
        )
    buttons.append(
        dict(
            label="Pause",
            method="animate",
            args=[
                [None],
                {
                    "frame": {"duration": 0, "redraw": False},
                    "transition": {"duration": 0},
                    "mode": "immediate",
                },
            ],
        )
    )
    return buttons


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


def _score_title(title: str, score: object | None) -> str:
    if score is None:
        return title
    s = getattr(score, "score", None)
    if s is None:
        return title
    ds = getattr(score, "dwell_score", float("nan"))
    vs = getattr(score, "velocity_score", float("nan"))
    ls = getattr(score, "lateral_score", float("nan"))
    return (
        f"{title}<br><sub>"
        f"score={s:.2f} = dwell {ds:.2f} * velocity {vs:.2f} * lateral {ls:.2f}"
        "</sub>"
    )


def _apply_scene_layout(fig: go.Figure, title_text: str) -> None:
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


def _overlay_traces(
    trajectory: Trajectory,
    attacker: Attacker | None,
    window: HitWindow | None,
    score: object | None,
) -> list[go.Scatter3d | go.Mesh3d]:
    traces: list[go.Scatter3d | go.Mesh3d] = []
    contact_index = getattr(score, "contact_index", None)
    traces.extend(_key_point_markers(trajectory, contact_index))
    if window is not None:
        traces.append(_hit_window_trace(window))
    if attacker is not None and window is not None:
        traces.append(_attacker_silhouette_trace(attacker, window))
    return traces


def plot_trajectory_3d(
    trajectory: Trajectory,
    attacker: Attacker | None = None,
    window: HitWindow | None = None,
    score: object | None = None,
    net_height: float = NET_HEIGHT_MEN,
    title: str = "Volleyball trajectory",
) -> go.Figure:
    """Interactive static 3D plot for any volleyball trajectory.

    Renders the court, net, antennae, attack lines, trajectory, and optional
    overlays. The function is scenario-agnostic: serve, set, attack, or
    arbitrary release-state experiments all enter through the same Trajectory.
    """
    traces: list = []
    traces.extend(_court_traces(net_height))
    traces.append(_trajectory_trace(trajectory))
    traces.extend(_overlay_traces(trajectory, attacker, window, score))

    fig = go.Figure(data=traces)
    _apply_scene_layout(fig, _score_title(title, score))
    return fig


def plot_trajectory_animation(
    trajectory: Trajectory,
    attacker: Attacker | None = None,
    window: HitWindow | None = None,
    score: object | None = None,
    net_height: float = NET_HEIGHT_MEN,
    title: str = "Volleyball trajectory",
    config: AnimationConfig | None = None,
    reference_trajectory: Trajectory | None = None,
) -> go.Figure:
    """Interactive Plotly animation for any volleyball trajectory.

    The figure includes a rendered volleyball court, an optional comparison
    trajectory, optional overlays, a moving ball marker, a growing trail,
    playback-speed controls, and a time slider. It intentionally depends only
    on Trajectory plus optional generic overlays so different scenes can reuse
    it without renderer changes.
    """
    cfg = config or AnimationConfig()
    frame_indices = _sample_frame_indices(trajectory, cfg.frame_count)
    initial_idx = frame_indices[0]

    traces: list = []
    traces.extend(_court_traces(net_height))
    if cfg.show_full_path:
        full_path = _trajectory_trace(trajectory)
        full_path.update(opacity=0.22, name="full path", showlegend=True)
        traces.append(full_path)
    if reference_trajectory is not None:
        traces.append(_reference_trajectory_trace(reference_trajectory))
    traces.extend(_overlay_traces(trajectory, attacker, window, score))

    trail_trace_index = len(traces)
    traces.append(_animated_trail_trace(trajectory, initial_idx, cfg.trail_width))
    ball_trace_index = len(traces)
    traces.append(_animated_ball_trace(trajectory, initial_idx, cfg.ball_size))

    frames = [
        go.Frame(
            name=f"{trajectory.time[i]:.2f}s",
            data=[
                _animated_trail_trace(trajectory, i, cfg.trail_width),
                _animated_ball_trace(trajectory, i, cfg.ball_size),
            ],
            traces=[trail_trace_index, ball_trace_index],
        )
        for i in frame_indices
    ]

    fig = go.Figure(data=traces, frames=frames)
    _apply_scene_layout(fig, _score_title(title, score))
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                x=0.02,
                y=0.02,
                xanchor="left",
                yanchor="bottom",
                buttons=_playback_buttons(cfg),
            )
        ],
        sliders=[
            dict(
                active=0,
                x=0.16,
                y=0.03,
                len=0.78,
                currentvalue=dict(prefix="t = "),
                steps=[
                    dict(
                        label=f"{trajectory.time[i]:.2f}s",
                        method="animate",
                        args=[
                            [f"{trajectory.time[i]:.2f}s"],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                                "mode": "immediate",
                            },
                        ],
                    )
                    for i in frame_indices
                ],
            )
        ],
    )
    return fig


def plot_attack_3d(
    trajectory: Trajectory,
    attacker: Attacker | None = None,
    window: HitWindow | None = None,
    score: object | None = None,
    net_height: float = NET_HEIGHT_MEN,
    title: str = "Volleyball attack trajectory",
) -> go.Figure:
    """Interactive static 3D plot of a volleyball attack trajectory.

    Backward-compatible wrapper around plot_trajectory_3d().
    """
    return plot_trajectory_3d(
        trajectory,
        attacker=attacker,
        window=window,
        score=score,
        net_height=net_height,
        title=title,
    )
