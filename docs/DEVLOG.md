# vb_coach 开发日志

按时间倒序，最新在最上。每条记录包含：发生的事 + 关键决策 + 为什么。
不是 changelog（那是 git log 的事），而是**决策与思路的备忘**。

---

## 2026-05-10 — HitWindow 重构启动

**触发**：用户问"为什么三种球种的击球窗口不一样"。承认现有设计是 over-parameterized：窗口尺寸的差异（如 dx 0.5 vs 0.6，dz 0.18 vs 0.20）和评分函数的差异（dwell target、velocity band、lateral max）在惩罚同一件事。

**目标方向**：
- HitWindow = **纯几何**（攻手能不能碰到球），跟战术无关
- Score function = **战术品质**（球种相关）
- 三种球种共享同一个 window

**正在做**：
- 派研究 agent 找文献依据（reach envelope / swing arc / contact volume tolerance）
- 大概率不是矩形 —— ellipsoid 或 swing-arc segment 更接近真实

**待研究结果回来后决定**：
- 几何模型（轴对齐椭球？带朝向的椭球？球冠片段？）
- 参数化方式（继续用 attacker 参数，还是引入挥臂半径等新参数）
- 是否在 viz 里渲染 envelope（替换当前的绿色 box）

**为什么记下来**：这个重构跟之前的 unification refactor 一样，是修一个我自己在 plan 里写错的 over-engineering。下次写计划时要警惕"两层独立惩罚同一件事"。

---

## 2026-05-10 — Streamlit 交互式 web app

加 `app.py` —— 用户拖滑块调攻手参数 / 二传释放参数 / 球种，实时看 3D 轨迹和评分。

**为什么**：用户说"想体验一下"。HTML 静态文件不够 —— 没有实时调参的能力。Streamlit 是最快的方案（vs Dash / Flask + 自写 UI / Gradio）。

**架构**：sidebar 9 个滑块（位置 3 + 速度 3 + 旋转 3）+ 4 个攻手滑块 + 球种 / 网高选择器；主面板 3D 图（plotly_chart）+ 评分卡 + 诊断面板。每次交互 Streamlit 重跑整个脚本 —— simulate() 够快（10-100ms），不需要缓存。

**关键决策**：用 `include_plotlyjs="cdn"` 让单页 HTML 只有 30KB（不然 4.7MB inline）。本地实验不重要，但放云上时关键。

**运行**：`uv run streamlit run app.py` → localhost:8501

---

## 2026-05-10 — Phase A6：Plotly 3D 可视化

把 matplotlib 静态图升级成浏览器可旋转 / 缩放 / 悬停的 3D。

**关键设计**：`plot_attack_3d(trajectory, attacker?, window?, score?)` —— 输入只是数据对象，**所有位置 / 球种共用这一个函数**。这印证了用户那时候的直觉："对可视化来说，不同位置只是改一个输入参数的区别"。

**组件**：球场（边界 + 网 + 天线 + 进攻线 5+ 个 trace）+ 轨迹（按时间染色，hover 显示 t/x/y/z/|v|）+ 击球窗口（半透明 box）+ 攻手剪影（垂直线）+ 标题里嵌评分分解。

**意外好处**：中文标题在浏览器原生支持 UTF-8 —— matplotlib 之前的 CJK 字体问题自动消失。

---

## 2026-05-10 — Phase A1.1 + A1.2：4号位 三种球种评分

A1.1：foundation（`Attacker`、`HitWindow`、`score_high_ball`）+ demo。
A1.2：加 `score_half_ball`（半高）+ `score_shoot_ball`（平拉开）+ 3×3 unified demo。

**评分模型**：三个 [0,1] 子分相乘 —— dwell × velocity × lateral。
- dwell：球在窗口内停留时长 / 目标时长
- velocity：v_z 离最优值的距离（三角形 falloff）
- lateral：y 方向漂移 / 最大容差

**子 agent 抓到的 3 个 bug**（我在 plan 里写错的）：
1. 横向漂移本来写的 x 轴 —— 但 x 是球的主飞行方向（设到攻手），漂移永远 ~0.3m，score 永远 0。改成 y 轴。
2. `_DWELL_TARGET = 0.30s` —— 但窗口高 0.40m 在 v_z = -4m/s 下穿越只要 0.10s，0.30s 物理上达不到。改成 0.08s。
3. 平拉开击球点应该是 apex（v_z=0），不是 descent 中点。改用 apex-inclusive 过滤 + apex 作为 contact reference。

**diagonal-dominance 验证**：每种球种最高分都在它的匹配评分器列，off-diagonal 都很低或 0。物理上合理。

**遗留**：所有评分常数是 model prior，不是 data calibrated。要做 calibration 得有比赛视频 + 攻手主观评价或扣球成功率。

---

## 2026-05-10 — Unification refactor：物理引擎统一

把独立的 `examples/float_serve_demo.py`（369 行自包含飘球模拟）重构成调用主线库。

**关键变更**：
- `aerodynamics.py` 从单文件拆成 package：`drag/` + `magnus/` + `knuckle/`
- `physics.py` 加 `knuckle_force()`
- `trajectory.py` 加 `simulate_stochastic()`（symplectic Euler + OU 过程）
- demo 从 369 行 → 174 行

**最强回归证据**：重构后生成的 PNG 跟原版 byte-identical（MD5 一致）。

**关键决策**：保留两个 simulator —— `simulate()` 用 RK45 + 地面 event（确定性，平滑），`simulate_stochastic()` 用 fixed-dt Euler（必须，因为 OU 噪声要跟时间步同步）。

---

## 2026-05-10 — Phase A0：轨迹引擎核心

第一个真正的代码。从空目录到有 ODE 求解器 + 测试 + 示例。

**关键决策**：
- Python + numpy + scipy + matplotlib + pytest + uv —— 标准科学栈
- 坐标系：原点在网中心地面，x 沿网，y 跨网，z 向上
- ODE：scipy.integrate.solve_ivp RK45 + 地面终止事件
- 阻力：constant Cd ≈ 0.45（亚临界，文献支持）

**子 agent 抓到的 sign convention bug**：back-spin 在 +x 方向的球上对应 ω = **-y**，不是 +y。我在 plan 里写反了。物理就是物理。

---

## 2026-05-10 — 文献调研 + 数据收集

派两个后台 agent 做：
1. 亚临界排球气动 —— 结论：二传速度下用 Cd=0.45 光滑球模型，可忽略飘动效应（Darbois Texier 2016 量化估算 <3cm 偏移）
2. 业余 reach 数据 —— 30 个英文 forum 自报 + 中国高校招生标准 + FIVB 反推。业余男击球球心 ~280-315cm，女 ~240-285cm

**学到的事**：派 agent 同时做独立调研 + 自己写 plan，效率高。Agent 的结论以记忆形式保存，下次会话直接可用。

---

## 项目原则（沉淀下来）

1. **两阶段拆分**：A = 正向（释放状态 → 轨迹 → 品质）；B = 逆向（身体动作 → 释放状态），推迟
2. **业余优先**：按业余 reach 数据校准参数
3. **按战术分层**：每种球种"好球"的定义不同
4. **TDD + 频繁 commit**：每个 task 都先写测试再实现
5. **子 agent 驱动 + 两阶段审查**：implementer → spec review → quality review
6. **诚实记下偏差**：implementer 改 plan 时要解释为什么；评审者要判断改得对不对
