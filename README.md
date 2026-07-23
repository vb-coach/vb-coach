# vb_coach

把排球教练里的定性概念（"立得住"、"球头"、"飘球"）变成**可计算、可可视化、可实验**的物理模型 —— 帮业余球员理解什么样的二传最好打，二传应该传成什么样。

## 这个项目想回答什么

排球场上经常听到这种话："这个球立得住"、"有球头"、"这个发球飘得厉害"、"这个二传不舒服"。但**没有数据** —— 只是凭感觉。

我们想把这些感觉变成数学：

1. **二传出球的物理参数**（出手位置 / 速度 / 高度 / 旋转）怎么决定了攻手的击球品质？
2. 攻手身高、摸高、扣球高度不同，**最适合打什么样的球**？
3. 飘球的"飘"到底有多大，可不可以预测？什么条件下飘得最厉害？
4. 同一组释放参数下，**三种攻击战术**（高球 / 半高 / 平拉开）哪个最适合这个攻手？

最终目标是一个**交互式工具**：业余教练或攻手输入自己的身体参数，看不同二传输出对应的 3D 轨迹和品质评分，搞清楚"哪种球该让二传给你传"。

## 已实现

### 物理引擎
- **轨迹模拟器**：`F = mg + F_drag + F_Magnus`，scipy RK45 + 地面终止事件 ([trajectory.py](src/vb_coach/trajectory.py))
- **亚临界 + 超临界气动**：`cd_full_regime(Re)` 跨 drag crisis，亚临界用 Cd=0.45（文献支持），飘球区段用余弦过渡 ([aerodynamics/drag.py](src/vb_coach/aerodynamics/drag.py))
- **飘球随机模型**：`simulate_stochastic()` 用 symplectic Euler + Ornstein-Uhlenbeck 过程模拟 panel 引起的横向涡脱落 ([aerodynamics/knuckle.py](src/vb_coach/aerodynamics/knuckle.py))

### 攻击评分（4 号位 + 2 号位 pin attacks）
- **Attacker 模型**：身高 + 站立摸高 + 助跑摸高 + level → 击球高度 + 窗口宽度 ([attacker.py](src/vb_coach/setting/attacker.py))
  - 7 个 preset：业余 / 大学 / 精英 × 男 / 女（Palao 2014 anthropometry 锚定）
  - `level` 影响 HitWindow 沿网半径：amateur +20 cm / college +10 / elite +5 cm 的 footwork tolerance
- **三种球种评分函数** ([attacks_position4.py](src/vb_coach/setting/attacks_position4.py)，pin-attack agnostic)：
  - `score_high_ball()` — **高球**（下降段击球，v_z ≈ -4 m/s）
  - `score_half_ball()` — **半高**（下降段早期击球，v_z ≈ -2.5 m/s）
  - `score_shoot_ball()` — **平拉开**（接近顶点击球，v_z ≈ 0）
- **HitWindow 工厂**：`position4_hit_window(attacker)` 和 `position2_hit_window(attacker)`（镜像）
- 每个返回 `score ∈ [0, 1]` + 诊断字段（停留时间 / 击球时 v_z / 横向漂移）
- 评分模型：`score = dwell_score × velocity_score × lateral_score`（任一为 0 总分为 0）

### 可视化
- **静态 matplotlib 图**（侧视图 + 3D，[plotting.py](src/vb_coach/plotting.py)）
- **交互式 Plotly 3D**（浏览器可旋转 / 缩放 / 悬停 / 播放动画，[plotly3d.py](src/vb_coach/viz/plotly3d.py)）
- **通用动画接口**：`plot_trajectory_animation(traj, ...)` 只依赖 `Trajectory`，可复用于发球、二传、扣球等任意场景
- **可调播放速度**：每个动画内置 `0.5x / 1x / 2x / Pause`，时间轴也可逐帧拖动
- **飘球对照轨迹**：随机飘球与无随机侧向力的虚线轨迹同图显示，直接观察横向偏移
- **米制 Three.js 人眼视角**：18×9 m 场地、5 cm 场线、1 m 网面、21 cm 球体均按真实尺寸建模；透视相机支持接发球员 / 发球员 / 全场机位和眼高、横向站位、俯仰角、视场角参数 ([threejs.py](src/vb_coach/viz/threejs.py))
- **Streamlit 本地 web app**（进攻评分 + 飘球发球两种场景，发球默认人眼视角并可切回 Plotly 分析视角，[app.py](app.py)）

### 飘球研究
独立研究分支，结论：飘球速度范围 13-20 m/s，最佳临界 Re ≈ 1.5-2.5×10⁵，σ_x 约 30 cm landing scatter ([docs/research/float-serve.md](docs/research/float-serve.md))。

## 快速开始

```bash
# 安装（用 uv）
uv sync

# 跑所有测试（应该 120 通过）
uv run pytest -q

# 静态示例
uv run python examples/basic_set.py                      # 4号位高球轨迹
uv run python examples/position4_all_attacks_demo.py     # 3 种球种 × 3 轨迹评分对比
uv run python examples/float_serve_demo.py               # 三种飘球轨迹 + 蒙特卡洛 + 3D 动画
uv run python examples/animated_trajectory_demo.py       # 通用动态轨迹 HTML
uv run python examples/interactive_attacks_demo.py       # 生成 3 个攻击动画 HTML

# 交互式 web app（推荐先玩这个）
uv run streamlit run app.py
# 然后浏览器打开 http://localhost:8501
```

Streamlit 里可以切换“进攻”和“飘球发球”。进攻场景可调攻手、二传释放状态、球种和网高；发球场景可调球速、击球高度、垂直/横向速度、旋转、飘移随机力和随机种子。人眼视角可切换接发球员 / 发球员 / 全场机位，并调整眼高、站位、俯仰角和视场角；动画支持 `0.5x / 1x / 2x` 播放或拖动时间轴。

## 项目结构

```
src/vb_coach/
  constants.py            物理常数 + 场地参数
  coords.py               坐标系（原点网中心，x 沿网，y 跨网，z 向上）
  aerodynamics/
    drag.py               Cd(Re) 模型（亚临界 + drag crisis）
    magnus.py             Cl(S) 旋转升力
    knuckle.py            Panel 引起的横向力 + OU 过程
  physics.py              gravity + drag + Magnus + knuckle force
  trajectory.py           simulate() (RK45) + simulate_stochastic() (Euler+OU)
  setting/
    attacker.py           Attacker dataclass
    hit_window.py         击球窗口几何
    attacks_position4.py  4号位评分函数
  plotting.py             matplotlib 2D/3D
  viz/plotly3d.py         Plotly 交互式 3D + 动画
  viz/threejs.py          米制 Three.js 场地 + 人眼透视相机
app.py                    Streamlit web app
docs/PLAN.md              总体规划
docs/DEVLOG.md            开发日志（决策记录）
docs/research/            研究文档（飘球等）
docs/superpowers/plans/   实施计划档案
examples/                 可运行示例 + 动画 HTML + 飘球研究 demo
tests/                    pytest 测试套件（120 个测试）
```

## 路线图

- [x] **Phase A0** — 轨迹引擎核心
- [x] **Unification** — 主线物理引擎 + 飘球研究统一到一套
- [x] **Phase A1.1** — Attacker + HitWindow + 4号位高球评分
- [x] **Phase A1.2** — 4号位 半高 + 平拉开评分
- [x] **Phase A6** — Plotly 3D 可视化 + Streamlit 交互式 app
- [x] **HitWindow 重构** — 非对称椭球（Wagner 2020 ±20 cm 锚定）
- [x] **Prior 校准扫盘** — Hong 2018 / Norman & McKeon 2011 等校准 drag + knuckle + contact_loss；level-dependent footwork tolerance
- [x] **Phase A2** — 2 号位 attacks（4 号位镜像）
- [ ] **Phase A3** — 3 号位 快攻（近体快 / 背快 / 背飞 / 短平）
- [ ] **Phase A4** — 后排攻
- [ ] **Phase A5** — 攻手适配层（输入 attacker 参数 → 自动推荐最优球种）
- [ ] **评分函数数据校准** — 通过 200-回合视频标注实验填掉 dwell / v_z / drift 真实数据空白（[annotation-protocol.md](docs/research/annotation-protocol.md)）
- [ ] **真实场地材质** — PBR 地板、体育馆环境、网柱/球网模型与灯光
- [ ] **真实排球材质** — 官方球款 UV 贴图，通过 `ball_texture_url` 接口加载
- [ ] **Phase B**（推迟）— 二传身体动作 → 释放状态的逆向研究

## 重要说明

- **评分模型是 prior，不是 calibrated**：所有评分常数（`_VELOCITY_BEST`、`_DWELL_TARGET` 等）是基于教练直觉 + 物理推导，没有用真实比赛数据校准。要做严格 calibration 需要比赛视频反推真实轨迹 + 攻手主观评价。
- **击球窗口是简化模型**：目前用轴对齐 3D 盒子。**正在按文献重构成更精确的非矩形几何**（椭球或挥臂弧线 envelope）。
- **业余优先**：默认参数按业余男攻手校准（身高 178cm、助跑摸高 295cm、击球点 280cm）。专业球员 / 女子需要单独参数。
- **飘球分支已合并**：飘球的物理是超临界区的事，跟二传速度（亚临界）的物理完全不同 —— 但共用同一个轨迹引擎和可视化层。

## 技术栈

Python 3.11+ / numpy / scipy / matplotlib / plotly / streamlit / Three.js / pytest / uv

## 测试

```bash
uv run pytest -q        # 全部 120 测试
uv run pytest -v        # 详细输出
```

## 协作约定

见 [CLAUDE.md](CLAUDE.md)。简而言之：commit 用 `[scope] content` 格式，代码英文，docstring 英文，沟通中文。

## License

未指定。学习 / 研究用途。
