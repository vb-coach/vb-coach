# vb_coach 开发日志

按时间倒序，最新在最上。每条记录包含：发生的事 + 关键决策 + 为什么。
不是 changelog（那是 git log 的事），而是**决策与思路的备忘**。

---

## 2026-05-13 — Prior 文献校准扫盘

**触发**：用户问"哪些 prior 有论文或网络帖子做过研究，可以校准我们的模型"。承认 README 早就标了"评分模型是 prior, 不是 calibrated"——但具体哪些 prior 已经有数据可校准、哪些还是直觉，没系统盘过。

**方法**：派 4 个并行研究 agent，覆盖（1）气动 + Magnus + 飘球；（2）二传/扣球运动学；（3）攻手体型与跳跃；（4）评分判据与教练经验。每个 agent 都被告知现有 prior 数值，专注于补全空白与验证。

**主要发现**：

1. **气动层有大量精确文献可校准**（Hong 2018, Norman & McKeon 2011, Darbois Texier 2016）。我们之前的 Cd 和 Cl_lat 常数数量级正确但偏差明显——亚临界 Cd 是 0.50 而不是 0.45，drag crisis 中心在 Re ≈ 2.65×10⁵ 而不是 ~2×10⁵，横向力 RMS 峰值在 Re ≈ 3×10⁵ 而不是 2×10⁵。这些值改完后**飘球最佳速度从 16 m/s 移到 20–21 m/s**，更符合"职业男排发球最飘"的经验。

2. **攻手 contact_loss 实测是 18–22 cm**（Sheppard 2008 AIS in-game data），不是我们之前估的 15 cm。Wagner 2014 + Coleman 1993 等都给出 spike contact 距网平面 0.2–0.5 m，跟我们 0.2 m 一致。

3. **评分函数的 9 个常数（dwell × 3、velocity_best × 3、lateral_max × 3）几乎全是 data gap**。物理推导可以自洽（抛物线 + Wagner ±20 cm），但**没有任何论文测过球 v_z vs spike outcome、ball-in-window dwell time、attacker subjective rating**。Wagner 2020 还是项目里唯一真正校准的几何参数。

**应用了哪些**：
- `drag.py`：Cd_sub 0.45 → 0.50；Re_crit 1.5e5–2.5e5 → 2.0e5–3.3e5；Cd_super 0.18 → 0.16
- `knuckle.py`：Cl_lat_peak 0.12 → 0.10；Re_lat_peak 2.0e5 → 3.0e5；τ_knuckle 0.25 → 0.35 s
- `attacker.py`：DEFAULT_CONTACT_LOSS 0.15 → 0.20；新增 6 个 preset（RECREATIONAL_MALE、COLLEGE_MALE/FEMALE、ELITE_MALE/FEMALE 等，Palao 2014 锚定）
- 测试同步更新（106 测试全过）

**没应用的**（中置信度，留底）：
- Magnus `_CL_HALF_SATURATION_S` 0.25 → 0.15（Kao 1994 排球专测，但只覆盖发球速度区间）
- `_HALF_VELOCITY_BEST` -2.5 → -3.0（物理推导，无实验）
- `_SHOOT_VELOCITY_BAND` 1.5 → 1.0（战术文献描述，无定量）
- 4号位接触点 x −4.0 → −3.5（教练学共识，无 peer-reviewed）

**飘球速度迁移**：原 [float-serve.md](research/float-serve.md) 整篇基于"飘最厉害在 v=16 m/s, Re ≈ 2×10⁵"。新常数把临界中心移到 3×10⁵ → 飘最厉害在 v ≈ 21 m/s。在 float-serve.md 顶部加了校准注释，正文具体数字保留没动（结构性结论仍正确，只需把数字读上调一档）。Demo 没动；如果未来重跑飘球研究 demo，应该用 v ≈ 20 m/s 而非 16 m/s。

**为什么记下来**：
1. 第一次系统区分"已有文献支持的 prior" vs "纯直觉" vs "data gap"。建立了 [calibration-priors.md](research/calibration-priors.md) 作为以后改任何常数的 reference checklist。
2. 飘球速度从 16 → 21 m/s 是 nontrivial 的物理修正——之前的 prior 在面板球与光滑球的 Re_crit 差异上估错了（30% 偏低）。
3. 评分函数的 data gap 现在显式列出来了。下一步要么接受 prior 不变（标"未校准"），要么做 200-回合标注实验。

---

## 2026-05-10 — HitWindow 重构完成

5 任务跑完。最终状态：106 测试通过，3×3 unified demo 干净的对角占优（0.601 / 0.707 / 0.793，off-diagonal 全 0）。

**实际改动**：
- `HitWindow` 从 (dx, dy, dz) 轴对齐盒子 → (r_x, r_y_front, r_y_back, r_z_up, r_z_down) 非对称椭球
- 3 个 per-attack window factory → 1 个 `position4_hit_window(attacker)`
- 评分函数代码逻辑没变 —— 只是接收同一个 window，差异全在评分函数内部常数
- Plotly viz 从 12 三角形 box mesh → 690 三角形参数化椭球 mesh
- 4 个 demo / app 全部切到统一 window

**几个发现**：
1. **新椭球比老盒子在 back 方向显著更紧**（r_y_back=0.15 vs 旧 dy=0.30）。Task 2 调了 3 个 test velocity，让球在到达 contact 点时 y 更小（更靠近网）—— 这其实更贴近真实战术：好球应该在攻手前方一点，不是直接在头顶
2. **r_z_up = 0.05** 修了 latent bug：之前的对称模型让"球高于 max reach"也算 in-window，物理不可能。Wagner + 教练学都说峰值摸高是硬上限
3. Demo 第一版没改 release velocity，所有分数都 0 —— 重构正确，但 demo 用了 stale 数据。多花一个 commit 同步 demo velocity 跟测试一致

**对角占优 vs 重构前对比**：

```
重构前：
    high |    0.865    0.029    0.000
    half |    0.296    0.686    0.000
   shoot |    0.000    0.000    0.716

重构后：
    high |    0.601    0.000    0.000
    half |    0.226    0.707    0.000
   shoot |    0.000    0.000    0.793
```

绝对分数低了一点（椭球更紧 → corner 处掉分），但**对角占优更干净**（off-diagonal 不再有 0.296 这种"混淆值"）。这是好事 —— 评分判别力更强。

**遗留**：
- 横向半径 0.40 m 仍然是 Wagner 0.20 + 0.20 footwork tolerance 的混合，没有真正校准 footwork 项
- Per-attacker radii 调整（高个 vs 矮个挥臂半径不同）暂时没做 —— 当前所有 attacker 共用同一组半径，只 center 跟着 contact_height 走

**为什么记下来**：
1. 第一个**真正数据校准**的几何参数（Wagner ±20 cm） —— 前几个 phase 全是 model prior
2. 重构方法学验证：5 task subagent-driven 流程能干净跑过 over-parameterization 重构，每个 task 独立可验证

---

## 2026-05-10 — HitWindow 重构启动 + 文献调研结果

**触发**：用户问"为什么三种球种的击球窗口不一样"。承认现有设计是 over-parameterized：窗口尺寸的差异（如 dx 0.5 vs 0.6，dz 0.18 vs 0.20）和评分函数的差异（dwell target、velocity band、lateral max）在惩罚同一件事。

**目标方向**：
- HitWindow = **纯几何**（攻手能不能碰到球），跟战术无关
- Score function = **战术品质**（球种相关）
- 三种球种共享同一个 window

**研究 agent 回来了**：[reference_hit_window_biomechanics.md](.claude memory)。关键发现：
- **Wagner et al. 2020** 是唯一变化球位置测肌肉激活的论文。±20 cm 偏移 → 三角肌负荷涨 70%。这是 envelope 的尺度基准。
- **没有任何论文**测过 spike velocity / success rate vs contact position deviation —— 数据缺口。
- 教练学一致认同：接触在**肩前 30–46 cm**、**摸高峰值**处；前向比后向容差大；摸高是硬上限（无 "above max reach"）
- 跨项目参考：tennis ~80mm、baseball ±23cm 横向 —— **量级都是 ~20 cm**
- 推荐 **Model B：非对称椭球**（前 0.30 / 后 0.15 / 横 0.20 / 上 0.05 / 下 0.30 m，amateur male）
- Model C/D（球冠 / 非对称球冠）更忠实但需要 swing-plane 朝向，暂不需要

**重构决策**：
- 用 Model B 非对称椭球替换轴对齐盒子
- 删除三个球种各自的 window factory，改用 `position4_hit_window(attacker)` 单一工厂
- 当前 lateral 0.5m 是 Wagner 2× 宽 —— 隐含 footwork tolerance。保留较宽的横向（lateral=0.40m）反映业余攻手脚步调整
- z_up = 0.05 m 修了一个 latent bug：当前对称模型允许 ball 在 contact_height + 0.20 m 也算 in-window，但那高于 max reach，物理不可能
- 评分函数差异**只剩**：velocity_best、velocity_band、dwell_target、lateral_max_drift；窗口几何统一

**为什么记下来**：
1. 重构同样的 over-engineering 教训 —— 警惕"两层独立惩罚同一件事"
2. Wagner 的 ±20 cm 是项目里第一个**真正数据校准**的参数（之前的 dwell target 等都是直觉）
3. 文献承认缺口 —— 我们的评分仍是 model prior，不是 truth；要真校准需要 spike-success vs deviation 的实验

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
