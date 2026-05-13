# 视频标注实验方案 — 校准评分函数 prior

> 目标：填掉 [calibration-priors.md](calibration-priors.md) §7 列出的 9 个评分常数（dwell × 3、velocity_best × 3、lateral_max × 3）的真实数据空白。最小可行方案：200 个比赛回合 × 12 个字段。

## 视频源

- **自录训练**（MVP 推荐）：两台 iPhone 60 fps，一台侧视一台端视，一个训练馆下午能拍 100+ 回合
- **YouTube FIVB / Volleyball World** 4K 慢动作：补精英样本；缺 metric scale，用网长 9 m / 球径 21 cm 当参考
- **NCAA 公开录像**：中间水平

## 相机布置（自录）

- Camera 1（端视，三脚架 2.5 m）：对底线，拍 Y-Z 平面（球距网 + 弧线）
- Camera 2（侧视，三脚架 1.5 m）：对网中线垂直，拍 X-Z 平面（沿底线方向）
- 用拍手板做同步锚；赛前拍 5 秒空场作 4 角标定（9 × 18 m）
- 最低 60 fps；24 m/s 扣球 5 cm 位移 ≈ 2 ms

## 球检测

| 方法 | 工作量 | 适合阶段 |
|---|---|---|
| 手工标注（CVAT / VGG VIA）每回合 5–10 帧 | 200 × 2 min = ~7 h | MVP |
| 先手标 50 回合，训练 YOLOv8，剩下半自动 | +1 天 ML | 扩到 500+ |

MVP 用手工足够。

## 字段 schema

每回合一行 CSV：

```python
from pydantic import BaseModel
from typing import Literal

class RallyAnnotation(BaseModel):
    rally_id: str
    video_url: str
    timestamp_start: float                              # 视频内秒
    attacker_level: Literal["amateur", "college", "elite"]
    attacker_gender: Literal["M", "F"]
    attack_type: Literal["high", "half", "shoot"]
    spike_outcome: Literal["kill", "in_play", "error", "blocked"]
    attacker_rating: int | None                         # Likert 1-5; None 若不可得
    ball_v_z_at_contact_mps: float                      # 校准 _VELOCITY_BEST
    ball_y_at_contact_m: float                          # 验证 contact_y 默认
    lateral_drift_cm: float                             # 校准 _LATERAL_MAX_DRIFT
    dwell_seconds_in_window: float                      # 校准 _DWELL_TARGET
    attacker_contact_height_m: float                    # 验证 contact_loss
    attacker_max_reach_m: float                         # per-attacker scaling
```

**最高 ROI 三字段**：`attacker_rating`（直接因变量）、`ball_v_z_at_contact_mps`、`lateral_drift_cm`。

## 取样

200 回合按 3 球种 × 3 水平 = 9 cell 分层，每 cell ≥ 20 回合（共 180）+ 20 故意取"差球"锚低分端。同一攻手最多 5 回合避免偏差。

## 分析

最简版 — 拟合 `_VELOCITY_BEST`：

```python
import pandas as pd
from scipy.optimize import curve_fit

df = pd.read_csv("data/annotations/rallies_v1.csv")

for attack_type in ["high", "half", "shoot"]:
    sub = df[df.attack_type == attack_type]
    kill_rate_by_vz = sub.groupby(pd.cut(sub.ball_v_z_at_contact_mps, 10)).spike_outcome.apply(
        lambda x: (x == "kill").mean()
    )
    # Fit parabola; peak gives _VELOCITY_BEST
    vz_centers = [b.mid for b in kill_rate_by_vz.index]
    popt, _ = curve_fit(lambda v, a, v0, c: a*(v-v0)**2 + c, vz_centers, kill_rate_by_vz.values)
    print(f"{attack_type}: _VELOCITY_BEST = {popt[1]:.2f} m/s")
```

**单一因变量**：选 `kill_rate`（客观，需 200+）或 `attacker_rating`（信号强，但精英不可得）二选一，不要混。

## 拟合容量 vs 样本量

| 拟合目标 | 参数数 | 需要回合 |
|---|---|---|
| `_VELOCITY_BEST × 3` + band × 3 | 6 | ~150 |
| 加 dwell + lateral × 3 | +6 = 12 | ~300 |
| 加 amateur/college/elite scaling | +12 = 24 | ~600 |

→ **200 回合先拟合 velocity 一项**，后续扩样本再校准 dwell / lateral / level scaling。

## MVP 一周路径

| 天 | 任务 |
|---|---|
| 1 | 自录 30 业余回合（两台 iPhone）|
| 2 | 装 CVAT，标 30 回合 ≈ 2.5 h；定 schema |
| 3 | 写 `src/vb_coach/calibration/fit_scoring.py`，跑出 30 样本的 v_best 估计 |
| 4 | YouTube 补 30 个精英回合（用人物身高 / 网长校准 metric）|
| 5 | 60 回合重新拟合，对比 prior 偏差 |
| 6–7 | 决定继续扩到 200 还是当前结果已可用 |

60 回合就能验证 pipeline；若 `_VELOCITY_BEST` 拟合与现 prior 偏差 < 0.5 m/s，证明 200 够用。

## 代码框架（待实现）

```
src/vb_coach/calibration/
    schema.py        # RallyAnnotation pydantic
    fit_scoring.py   # 从 CSV 拟合 _VELOCITY_BEST / _BAND
    apply.py         # 写回 attacks_position4.py
data/annotations/
    rallies_v1.csv   # 实际标注数据（git LFS 或 .gitignore）
```
