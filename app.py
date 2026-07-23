"""Streamlit interactive web app for vb_coach trajectory experiments.

Run:
    uv run streamlit run app.py

Opens at localhost:8501 by default. The sidebar exposes:
  - Scene (pin attack / float serve)
  - Attack position (4号位 / 2号位)
  - Attacker preset (业余 / 大学 / 精英 × 男 / 女) plus per-parameter override
  - Setter or server release state (position, velocity, spin)
  - Attack type, float-serve preset, and net gender
"""

from __future__ import annotations

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from vb_coach.constants import NET_HEIGHT_MEN, NET_HEIGHT_WOMEN
from vb_coach.setting.attacker import (
    Attacker,
    COLLEGE_FEMALE,
    COLLEGE_MALE,
    DEFAULT_AMATEUR_FEMALE,
    DEFAULT_AMATEUR_MALE,
    ELITE_FEMALE,
    ELITE_MALE,
    RECREATIONAL_MALE,
)
from vb_coach.setting.attacks_position4 import (
    score_high_ball,
    score_half_ball,
    score_shoot_ball,
)
from vb_coach.setting.hit_window import (
    position2_hit_window,
    position4_hit_window,
)
from vb_coach.serving import CANONICAL_FLOAT_SERVES, FloatServePreset
from vb_coach.trajectory import ReleaseState, simulate, simulate_stochastic
from vb_coach.viz.plotly3d import AnimationConfig, plot_attack_3d, plot_trajectory_animation
from vb_coach.viz.threejs import (
    PerspectiveCameraConfig,
    ThreeJsAnimationConfig,
    render_trajectory_threejs,
)


_ATTACKER_PRESETS = {
    "业余男 (recreational male)":      RECREATIONAL_MALE,
    "业余男+ (trained amateur male)":  DEFAULT_AMATEUR_MALE,
    "大学男 (college male)":           COLLEGE_MALE,
    "精英男 (elite male, FIVB OH)":    ELITE_MALE,
    "业余女 (recreational female)":     DEFAULT_AMATEUR_FEMALE,
    "大学女 (college female)":          COLLEGE_FEMALE,
    "精英女 (elite female, FIVB OH)":   ELITE_FEMALE,
    "Custom": None,
}


# Preset release states for each (position, attack_type) cell. The 2号位
# entries are x-mirrors of the 4号位 ones.
_ATTACK_PRESETS = {
    ("4号位", "高球 (high)"):
        {"position": (0.5, 0.8, 2.2),  "velocity": (-5.0, -0.7, 5.5)},
    ("4号位", "半高 (half)"):
        {"position": (0.5, 0.8, 2.2),  "velocity": (-7.0, -0.75, 4.4)},
    ("4号位", "平拉开 (shoot)"):
        {"position": (-1.5, 0.5, 2.3), "velocity": (-8.5, -0.6, 3.0)},
    ("2号位", "高球 (high)"):
        {"position": (-0.5, 0.8, 2.2),  "velocity": (5.0, -0.7, 5.5)},
    ("2号位", "半高 (half)"):
        {"position": (-0.5, 0.8, 2.2),  "velocity": (7.0, -0.75, 4.4)},
    ("2号位", "平拉开 (shoot)"):
        {"position": (1.5, 0.5, 2.3), "velocity": (8.5, -0.6, 3.0)},
}

_SCORE_FN = {
    "高球 (high)":   score_high_ball,
    "半高 (half)":   score_half_ball,
    "平拉开 (shoot)": score_shoot_ball,
}

_WINDOW_FACTORY = {
    "4号位": position4_hit_window,
    "2号位": position2_hit_window,
}

_FLOAT_SERVE_LABELS = {
    "临界强飘 (critical-band wobble)": CANONICAL_FLOAT_SERVES[0],
    "后程下坠 (drop floater)": CANONICAL_FLOAT_SERVES[1],
    "低于甜区 (below sweet spot)": CANONICAL_FLOAT_SERVES[2],
}


def _attacker_inputs() -> Attacker:
    st.sidebar.subheader("攻手 Attacker")
    preset_name = st.sidebar.selectbox(
        "预设 Preset",
        list(_ATTACKER_PRESETS.keys()),
        index=1,  # 业余男+ default
    )
    preset = _ATTACKER_PRESETS[preset_name]

    if preset is None:
        # Custom: full slider control
        h_default, sr_default, ar_default, cl_default = 1.78, 2.32, 2.95, 0.20
        level_default = "amateur"
    else:
        h_default = preset.height
        sr_default = preset.standing_reach
        ar_default = preset.approach_reach
        cl_default = preset.contact_loss
        level_default = preset.level

    height = st.sidebar.slider("身高 height [m]", 1.50, 2.10, h_default, 0.01)
    standing = st.sidebar.slider(
        "站立摸高 standing reach [m]",
        max(height + 0.01, 2.00), 2.80,
        max(sr_default, height + 0.50), 0.01,
    )
    approach = st.sidebar.slider(
        "助跑摸高 approach reach [m]",
        max(standing, 2.30), 3.60,
        max(ar_default, standing + 0.40), 0.01,
    )
    contact_loss = st.sidebar.slider(
        "击球损失 contact loss [m]", 0.05, 0.30, cl_default, 0.01,
        help="approach_reach - contact_loss = ball-center height at contact (Sheppard 2008: 18-22 cm)",
    )
    level = st.sidebar.radio(
        "水平 Level (footwork tolerance)",
        ["amateur", "college", "elite"],
        index=["amateur", "college", "elite"].index(level_default),
        help="Hit-window r_x = Wagner 20 cm + footwork (amateur 20 / college 10 / elite 5)",
        horizontal=True,
    )
    return Attacker(
        height=height,
        standing_reach=standing,
        approach_reach=approach,
        contact_loss=contact_loss,
        level=level,
    )


def _release_inputs(preset_position, preset_velocity) -> ReleaseState:
    st.sidebar.subheader("二传 释放 Release")
    st.sidebar.write("**位置 Position [m]**")
    px = st.sidebar.slider("x (along net)", -2.0, 2.0, float(preset_position[0]), 0.1)
    py = st.sidebar.slider("y (across court)", -1.0, 2.0, float(preset_position[1]), 0.1)
    pz = st.sidebar.slider("z (height)", 1.5, 3.0, float(preset_position[2]), 0.05)
    st.sidebar.write("**速度 Velocity [m/s]**")
    vx = st.sidebar.slider("vx", -10.0, 10.0, float(preset_velocity[0]), 0.1)
    vy = st.sidebar.slider("vy", -3.0, 3.0, float(preset_velocity[1]), 0.1)
    vz = st.sidebar.slider("vz", 0.0, 10.0, float(preset_velocity[2]), 0.1)
    st.sidebar.write("**旋转 Spin [rad/s]** (0 = no spin)")
    sx = st.sidebar.slider("omega_x", -40.0, 40.0, 0.0, 1.0)
    sy = st.sidebar.slider("omega_y", -40.0, 40.0, 0.0, 1.0)
    sz = st.sidebar.slider("omega_z", -40.0, 40.0, 0.0, 1.0)
    return ReleaseState(
        position=np.array([px, py, pz]),
        velocity=np.array([vx, vy, vz]),
        spin=np.array([sx, sy, sz]),
    )


def _serve_inputs(preset: FloatServePreset) -> tuple[ReleaseState, bool, int]:
    st.sidebar.subheader("发球 Serve")
    speed = st.sidebar.slider(
        "向前速度 downcourt speed [m/s]", 10.0, 25.0, preset.speed, 0.5
    )
    height = st.sidebar.slider(
        "击球高度 contact height [m]", 1.8, 3.3, preset.contact_height, 0.05
    )
    vertical_speed = st.sidebar.slider(
        "垂直速度 vertical speed [m/s]",
        -2.0,
        6.0,
        preset.vertical_speed,
        0.1,
    )
    lateral_speed = st.sidebar.slider(
        "横向速度 lateral speed [m/s]", -2.0, 2.0, 0.0, 0.1
    )
    spin_x = st.sidebar.slider("前后旋 omega_x [rad/s]", -20.0, 20.0, 0.0, 1.0)
    spin_z = st.sidebar.slider("侧旋 omega_z [rad/s]", -20.0, 20.0, 0.0, 1.0)
    stochastic = st.sidebar.checkbox("模拟飘移 Float-force noise", value=True)
    seed = int(
        st.sidebar.number_input(
            "随机种子 Seed", min_value=0, max_value=10_000, value=preset.seed
        )
    )
    release = ReleaseState(
        position=np.array([0.0, -9.0, height]),
        velocity=np.array([lateral_speed, speed, vertical_speed]),
        spin=np.array([spin_x, 0.0, spin_z]),
    )
    return release, stochastic, seed


def _camera_inputs() -> PerspectiveCameraConfig:
    st.sidebar.subheader("视角 Camera")
    view_label = st.sidebar.radio(
        "机位 Position",
        ["接发球员 Receiver", "发球员 Server", "全场 Overview"],
    )
    view = {
        "接发球员 Receiver": "receiver",
        "发球员 Server": "server",
        "全场 Overview": "overview",
    }[view_label]
    default_height = 1.68 if view == "receiver" else 1.72
    default_pitch = 4.0 if view == "receiver" else 10.0
    eye_height = st.sidebar.slider(
        "眼高 Eye height [m]",
        1.20,
        2.20,
        default_height,
        0.01,
        key=f"camera_height_{view}",
    )
    lateral_position = st.sidebar.slider(
        "横向站位 Lateral position [m]",
        -4.5,
        4.5,
        0.0,
        0.1,
        key=f"camera_lateral_{view}",
    )
    pitch = st.sidebar.slider(
        "俯仰角 Pitch [deg]",
        -15.0,
        15.0,
        default_pitch,
        0.5,
        key=f"camera_pitch_{view}",
    )
    vertical_fov = st.sidebar.slider(
        "垂直视场 Vertical FOV [deg]",
        35.0,
        80.0,
        50.0,
        1.0,
        key=f"camera_fov_{view}",
    )
    return PerspectiveCameraConfig(
        view=view,
        eye_height_m=eye_height,
        lateral_position_m=lateral_position,
        pitch_deg=pitch,
        vertical_fov_deg=vertical_fov,
    )


def _render_attack(net_height: float) -> None:
    position = st.sidebar.radio(
        "位置 Position",
        ["4号位", "2号位"],
        horizontal=True,
    )
    attack_name = st.sidebar.selectbox(
        "战术 Attack type",
        list(_SCORE_FN.keys()),
        index=0,
    )
    preset = _ATTACK_PRESETS[(position, attack_name)]
    animated = st.sidebar.checkbox("动态播放 Animation", value=True)

    attacker = _attacker_inputs()
    release = _release_inputs(preset["position"], preset["velocity"])
    traj = simulate(release, t_max=3.0)
    window = _WINDOW_FACTORY[position](attacker)
    score = _SCORE_FN[attack_name](traj, window)

    col_plot, col_score = st.columns([3, 1])
    with col_plot:
        title = f"{position} {attack_name}"
        if animated:
            fig = plot_trajectory_animation(
                traj,
                attacker=attacker,
                window=window,
                score=score,
                net_height=net_height,
                title=title,
                config=AnimationConfig(frame_count=70, frame_duration_ms=35),
            )
        else:
            fig = plot_attack_3d(
                traj,
                attacker=attacker,
                window=window,
                score=score,
                net_height=net_height,
                title=title,
            )
        st.plotly_chart(fig, use_container_width=True)

    with col_score:
        st.subheader("分数 Score")
        st.metric("Total", f"{score.score:.3f}")
        st.write("**子分数 Sub-scores**")
        st.write(f"- dwell:    {score.dwell_score:.3f}")
        st.write(f"- velocity: {score.velocity_score:.3f}")
        st.write(f"- lateral:  {score.lateral_score:.3f}")
        st.write("**诊断 Diagnostics**")
        st.write(f"- dwell time: {score.dwell_seconds*1000:.0f} ms")
        st.write(f"- v_z at contact: {score.velocity_at_contact_z:.2f} m/s")
        st.write(f"- y-drift in window: {score.lateral_drift_m*100:.1f} cm")
        st.write(f"- attacker contact height: {attacker.contact_height:.2f} m")
        st.write(f"- attacker level: {attacker.level} (r_x = {window.r_x:.2f} m)")
        st.write(f"- apex height: {traj.apex_height():.2f} m")
        st.write(f"- flight time: {traj.time[-1]:.2f} s")


def _render_float_serve(net_height: float) -> None:
    preset_name = st.sidebar.selectbox(
        "发球预设 Serve preset", list(_FLOAT_SERVE_LABELS.keys())
    )
    release, stochastic, seed = _serve_inputs(_FLOAT_SERVE_LABELS[preset_name])
    view_mode = st.sidebar.radio(
        "视图 View",
        ["人眼视角 Human POV", "分析视角 Analysis"],
        horizontal=True,
    )
    camera = _camera_inputs() if view_mode.startswith("人眼") else None
    rng = np.random.default_rng(seed) if stochastic else None
    traj = simulate_stochastic(release, rng=rng, t_max=2.0, dt=0.005)
    reference_traj = simulate_stochastic(release, rng=None, t_max=2.0, dt=0.005)

    col_plot, col_metrics = st.columns([4, 1])
    with col_plot:
        if camera is not None:
            render_config = ThreeJsAnimationConfig(
                frame_count=180,
                canvas_height_px=620,
                autoplay=True,
            )
            html = render_trajectory_threejs(
                traj,
                reference_trajectory=reference_traj,
                net_height=net_height,
                camera=camera,
                config=render_config,
            )
            components.html(html, height=render_config.canvas_height_px, scrolling=False)
        else:
            fig = plot_trajectory_animation(
                traj,
                reference_trajectory=reference_traj,
                net_height=net_height,
                title=f"飘球发球 Float serve: {preset_name}",
                config=AnimationConfig(frame_count=100, frame_duration_ms=30),
            )
            st.plotly_chart(fig, use_container_width=True)

    landing = traj.position[-1]
    reference_x = np.interp(
        traj.time,
        reference_traj.time,
        reference_traj.position[:, 0],
    )
    max_float_drift = float(np.max(np.abs(traj.position[:, 0] - reference_x)))
    net_idx = int(np.argmin(np.abs(traj.position[:, 1])))
    reached_net = bool(traj.position[net_idx, 1] >= -0.1)
    net_clearance = float(traj.position[net_idx, 2] - net_height) if reached_net else float("nan")
    landed = bool(landing[2] <= 0.05)
    cleared_net = reached_net and net_clearance >= 0.0
    in_bounds = (
        landed
        and cleared_net
        and abs(landing[0]) <= 4.5
        and 0.0 <= landing[1] <= 9.0
    )

    with col_metrics:
        st.subheader("发球 Serve")
        st.metric("Total release speed", f"{np.linalg.norm(release.velocity):.1f} m/s")
        st.metric("Apex", f"{traj.apex_height():.2f} m")
        st.metric("Max float drift", f"{max_float_drift * 100:.0f} cm")
        if reached_net:
            st.metric("Net clearance", f"{net_clearance:.2f} m")
        st.write("**落点 Landing**")
        st.write(f"- x: {landing[0]:.2f} m")
        st.write(f"- y: {landing[1]:.2f} m")
        st.write(f"- flight time: {traj.time[-1]:.2f} s")
        st.write(f"- legal in-bounds serve: {'yes' if in_bounds else 'no'}")
        st.write(f"- stochastic float: {'on' if stochastic else 'off'}")


def main() -> None:
    st.set_page_config(page_title="vb_coach", layout="wide")
    st.title("vb_coach: trajectory lab")
    st.caption("按真实场地和球体尺寸观察轨迹；可切换球员人眼视角或分析视角。")

    scene = st.sidebar.radio(
        "场景 Scene",
        ["飘球发球 Float serve", "进攻 Attack"],
        horizontal=True,
    )
    net_gender = st.sidebar.radio("网高 Net", ["男 men (2.43m)", "女 women (2.24m)"])
    net_height = NET_HEIGHT_MEN if net_gender.startswith("男") else NET_HEIGHT_WOMEN
    if scene.startswith("进攻"):
        _render_attack(net_height)
    else:
        _render_float_serve(net_height)


if __name__ == "__main__":
    main()
