"""Streamlit interactive web app for vb_coach attack scoring.

Run:
    uv run streamlit run app.py

Opens at localhost:8501 by default. The sidebar exposes attacker params,
setter release state, attack type, and net gender. The main panel shows
the interactive 3D Plotly figure plus the score breakdown.
"""

from __future__ import annotations

import numpy as np
import streamlit as st

from vb_coach.constants import NET_HEIGHT_MEN, NET_HEIGHT_WOMEN
from vb_coach.setting.attacker import Attacker
from vb_coach.setting.attacks_position4 import (
    score_high_ball,
    score_half_ball,
    score_shoot_ball,
)
from vb_coach.setting.hit_window import (
    position4_high_window,
    position4_half_window,
    position4_shoot_window,
)
from vb_coach.trajectory import ReleaseState, simulate
from vb_coach.viz.plotly3d import plot_attack_3d


_ATTACK_PRESETS = {
    "高球 (high)":   {"position": (0.5, 0.8, 2.2),  "velocity": (-4.8, -0.5, 5.5),
                      "window_fn": position4_high_window,  "score_fn": score_high_ball},
    "半高 (half)":   {"position": (0.5, 0.8, 2.2),  "velocity": (-6.5, -0.5, 4.5),
                      "window_fn": position4_half_window,  "score_fn": score_half_ball},
    "平拉开 (shoot)": {"position": (-1.5, 0.5, 2.3), "velocity": (-7.5, -0.5, 3.5),
                      "window_fn": position4_shoot_window, "score_fn": score_shoot_ball},
}


def _attacker_inputs() -> Attacker:
    st.sidebar.subheader("攻手 Attacker")
    height = st.sidebar.slider("身高 height [m]", 1.50, 2.10, 1.78, 0.01)
    standing = st.sidebar.slider(
        "站立摸高 standing reach [m]",
        max(height + 0.01, 2.00), 2.80, max(2.32, height + 0.54), 0.01,
    )
    approach = st.sidebar.slider(
        "助跑摸高 approach reach [m]",
        max(standing, 2.30), 3.50, max(2.95, standing + 0.63), 0.01,
    )
    contact_loss = st.sidebar.slider(
        "击球损失 contact loss [m]", 0.05, 0.30, 0.15, 0.01,
        help="approach_reach - contact_loss = ball-center height at contact",
    )
    return Attacker(
        height=height,
        standing_reach=standing,
        approach_reach=approach,
        contact_loss=contact_loss,
    )


def _release_inputs(preset_position, preset_velocity) -> ReleaseState:
    st.sidebar.subheader("二传 释放 Release")
    st.sidebar.write("**位置 Position [m]**")
    px = st.sidebar.slider("x (along net)", -2.0, 2.0, float(preset_position[0]), 0.1)
    py = st.sidebar.slider("y (across court)", -1.0, 2.0, float(preset_position[1]), 0.1)
    pz = st.sidebar.slider("z (height)", 1.5, 3.0, float(preset_position[2]), 0.05)
    st.sidebar.write("**速度 Velocity [m/s]**")
    vx = st.sidebar.slider("vx", -10.0, 0.0, float(preset_velocity[0]), 0.1)
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


def main() -> None:
    st.set_page_config(page_title="vb_coach", layout="wide")
    st.title("vb_coach: 4号位 attack scoring")
    st.caption("拖动左侧滑块改变攻手 / 二传参数。3D 图实时更新，可旋转 / 缩放 / 悬停查看 t·x·y·z·|v|。")

    # Attack type selector at the top of the sidebar
    attack_name = st.sidebar.selectbox(
        "战术 Attack type",
        list(_ATTACK_PRESETS.keys()),
        index=0,
    )
    preset = _ATTACK_PRESETS[attack_name]
    net_gender = st.sidebar.radio("网高 Net", ["男 men (2.43m)", "女 women (2.24m)"])
    net_height = NET_HEIGHT_MEN if net_gender.startswith("男") else NET_HEIGHT_WOMEN

    attacker = _attacker_inputs()
    release = _release_inputs(preset["position"], preset["velocity"])

    # Simulate
    traj = simulate(release, t_max=3.0)
    window = preset["window_fn"](attacker)
    score = preset["score_fn"](traj, window)

    # Main panel layout
    col_plot, col_score = st.columns([3, 1])

    with col_plot:
        fig = plot_attack_3d(
            traj,
            attacker=attacker,
            window=window,
            score=score,
            net_height=net_height,
            title=attack_name,
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
        st.write(f"- apex height: {traj.apex_height():.2f} m")
        st.write(f"- flight time: {traj.time[-1]:.2f} s")


if __name__ == "__main__":
    main()
