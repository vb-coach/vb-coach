# vb_coach 协作约定

## 环境

- 用 `uv` 管理 Python 环境和依赖，所有命令通过 `uv run ...` 跑
- 安装新依赖：`uv add <pkg>`（运行时）或 `uv add --dev <pkg>`（开发）

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
