# vb_coach 协作约定

## 环境

- 用 `uv` 管理 Python 环境和依赖，所有命令通过 `uv run ...` 跑
- 安装新依赖：`uv add <pkg>`（运行时）或 `uv add --dev <pkg>`（开发）
- 仓库在 iCloud 里，`.venv` 内的文件会被打上 macOS hidden 标志（来源未确认），Python 3.11.15 会跳过这样的 `.pth`，表现为 `ModuleNotFoundError: No module named 'vb_coach'`。把 venv 放到 iCloud 外：`export UV_PROJECT_ENVIRONMENT=~/.cache/uv-venvs/vb-coach`

## 语言

- **沟通**：中文回复用户
- **代码**：变量、函数、类名英文
- **commit / docstring / 注释**：英文

## Git commit 格式

格式：`[scope] content`，1-3 行尽可能精简。

scope 基础类型：`feat` / `fix` / `test` / `docs` / `refactor` / `chore`

可细分到模块：`[feat:trajectory]`、`[fix:aero]`、`[test:physics]`

范例：

- `[feat:plotting] 2D side-view and 3D trajectory plots`
- `[fix:physics] correct Magnus sign convention for back-spin`
- `[test:trajectory] add vacuum-projectile analytic verification`

## 代码审查

这个项目的主要失败方式是模拟结果错了但程序不报错。审查优先查：

- 符号和坐标约定，对照 `src/vb_coach/coords.py`。前例：+x 方向飞行的下旋球 ω 应为 -y，曾写反；横向漂移用错轴，评分恒为 0
- 物理可行性：目标值算得到吗，状态可能出现吗。前例：`_DWELL_TARGET = 0.30s` 而实际穿越只要 0.10s；球高于最大摸高仍算 in-window
- 单位，全部 SI
- 常数和校准值：新增或修改要有出处（`docs/research/` 或文献），同一常数只留一份。前例：Cd 亚临界常数、`KNUCKLE_SPEED_CUTOFF` 曾重复定义
- `simulate()` 和 `simulate_stochastic()`：改一个要确认另一个是否受影响
- 已有测试的断言被改时，要说明是旧预期错了，还是在迁就新结果

审查输出：

- 第一行写 finding 数量，没有问题就写 `0 findings`
- 每条给 `file:line` 和推理依据，不凭名字猜
- 标 Important / Nit / Pre-existing，Nit 最多 3 条
- 同类问题一次指出全部位置
- 不报格式和命名偏好，跳过 `uv.lock`、`docs/superpowers/`、生成的 `*.html`

收敛：最多三轮，第三轮还不过就停下重新考虑方案；第二轮起只审新增 diff，已提过的问题不再提。同一条意见反复出现时，改这一节或补测试。

风险分档：

- 高，人读 diff：`aerodynamics/`、`physics.py`、`trajectory.py`、`coords.py`、`constants.py`、`.github/workflows/`，以及任何改动已有测试断言的 PR
- 中，人读测试证据和 findings：`setting/`、`serving.py`、`app.py`
- 低，CI 绿且无 Important 即可合并：`docs/`、`viz/`、`plotting.py`、`examples/`

一个 PR 跨档时按最高档。push 前跑 `uv run pytest`，再用 `reviewer` subagent 或 `/code-review high` 复审。AI 通过不算人的批准，合并的人对结果负责。
