# vb_coach 模型 prior 校准 — 文献调研汇总

> Date: 2026-05-13
> Scope: 项目里所有 prior 常数的文献校准状态。区分**高置信度可替换**、**中置信度待评估**、**真实数据空白**三类。
>
> 本文是研究档案，不是实施清单。代码层面已应用的改动见底部 "Applied Changes" 节；尚未应用的建议保留在各章节里供后续评估。

---

## 0. 一句话总结

气动模型有大量精确文献可校准（高置信度修正多）；评分函数则大量是真实 data gap，需要自采实验数据。Wagner 2020 ±20 cm（HitWindow 几何）和 Hong 2018 / Norman & McKeon 2011（气动 + 飘球）是项目里目前可以信任的两条数据线。

---

## 1. 球物理常数 ✅ 已验证

| 项 | 项目值 | 文献 | 状态 |
|---|---|---|---|
| 直径 D | 0.21 m | FIVB 规则：C=65–67 cm → D=20.7–21.3 cm | 准确 |
| 质量 m | 0.270 kg | FIVB: 260–280 g | 准确 |
| 空气密度 ρ | 1.20 kg/m³ | 20 °C / 1 atm 标准值 | 准确 |
| 运动黏度 ν | 1.5×10⁻⁵ m²/s | 20 °C: 1.51×10⁻⁵ | 准确（≤1% 偏差） |

**沙滩球差异**：FIVB 沙滩规格 D=21.0–21.6 cm，m 相同，内压低（0.175–0.225 vs 0.300–0.325 kg/cm²）。气动差异 <3%，对当前模型可忽略。

**Sources**: [FIVB Homologation Procedures (2024)](https://www.fivb.com/en/volleyball/the-game/equipment); Darbois Texier et al. 2016 *NJP* 18:073027 Table 1.

---

## 2. Drag 模型 🟢 已校准并应用

**文件**: [src/vb_coach/aerodynamics/drag.py](../../src/vb_coach/aerodynamics/drag.py)

### 校准前 vs 校准后

| 常数 | 旧值 | 新值 | 依据 |
|---|---|---|---|
| `_CD_SUBCRITICAL` | 0.45 | **0.50** | Hong 2018 风洞实测亚临界平台 0.50–0.55（多球型一致） |
| `_RE_CRIT_LOW` | 1.5×10⁵ | **2.0×10⁵** | Hong 2018: Mikasa / Molten / Mizuno 起始一致 |
| `_RE_CRIT_HIGH` | 2.5×10⁵ | **3.3×10⁵** | Hong 2018: Mikasa MVA200 上端 2.9–3.3×10⁵ |
| `_CD_SUPERCRITICAL` | 0.18 | **0.16** | Hong & Asai 2020: V200W 0.15–0.17 |

### 实测 Cd-Re 表（参考）

| 球种 | Re_crit | Cd_super | 源 |
|---|---|---|---|
| Mikasa MVA200CEV (8-panel) | 2.9–3.3×10⁵ | 0.15–0.16 | Hong 2018 |
| Molten V5M5000 (hex) | 2.7–3.0×10⁵ | 0.15–0.19 | Hong 2018 |
| Mizuno 18-panel (conv) | 2.0–2.8×10⁵ | 0.16–0.20 | Hong 2018 |
| Adidas 18-panel | 2.4–2.8×10⁵ | 0.12–0.17 | Hong 2018 |
| Beach (Mikasa VLS300) | ~3.0×10⁵ | ~0.20 | Dumek & Šafařík 2018 |
| Asai 2010 "new" ball | ~2.2×10⁵ | 0.17 | Asai 2010 |

**未来扩展**：可以做 per-ball preset（`drag.py` 暴露 ball-specific `_CD_SUB/SUPER/RE_CRIT` 字典）。当前是 amateur/match 的均值。

**Sources**:
- Hong & Asai 2020 *Applied Sciences* 10:3256 — [DOI 10.3390/app10093256](https://doi.org/10.3390/app10093256)
- Hong et al. 2018 *ISBS Proceedings* 36:212 — [link](https://commons.nmu.edu/isbs/vol36/iss1/212/)
- Asai et al. 2010 *Procedia Eng* 2:2493 — [DOI 10.1016/j.proeng.2010.04.022](https://doi.org/10.1016/j.proeng.2010.04.022)
- Dumek & Šafařík 2018 *EPJ Web Conf* 180:02021

---

## 3. Magnus 模型 🟡 中置信度，未应用

**文件**: [src/vb_coach/aerodynamics/magnus.py](../../src/vb_coach/aerodynamics/magnus.py)

### 当前状态

| 常数 | 当前值 | 建议 | 依据 | 置信度 |
|---|---|---|---|---|
| `_CL_SATURATION` | 0.20 | **保留**（或 0.22） | Kao 1994 排球风洞测：ω=8 rev/s, V=20 m/s → Cl≈0.20 ✓；Maranesi 2022 video analysis: 0.25(2) | 中 |
| `_CL_HALF_SATURATION_S` | 0.25 | **0.15** | Kao 1994 在 S=0.1 已达 Cl≈0.13；Mehta 1985 综述多在 S~0.15 半饱和 | 中 |

**未应用原因**：Kao 1994 是唯一**直接对排球**的风洞测量，但只测 ω ∈ [3, 8] rev/s, V ∈ [15, 35] m/s 范围（接近发球/扣球速度），不直接覆盖二传亚临界范围。保守保留现有值。

### Inverse Magnus 风险

Muto et al. 2012 与 Kim et al. 2014 在 Re∈[2.5×10⁵, 3.5×10⁵] + 低旋下光滑球观察到 Cl 反号。

- **影响范围**：仅光滑业余训练球。Mikasa MVA200 等 dimpled 比赛球的面板纹理抑制此效应（Asai 2010）。
- **对项目影响**：二传亚临界（v 4–8 m/s, Re ≈ 60–110k）完全不涉及；扣球（v 20–30 m/s, Re ≈ 280–420k）正好经过这个区间，理论上可能有影响。
- **建议**：当扩展到扣球出球预测时再加 inverse-Magnus 衰减因子。

**Sources**:
- Kao, Sellens, Stevenson 1994 *J Appl Biomech* 10:95–109 — [PDF](https://people.stfx.ca/smackenz/courses/HK474/Labs/Jump%20Float%20Lab/Kao%201994%20A%20mathematical%20model%20for%20the%20trajectory%20of%20a%20spiked%20volleyball%20and%20its%20coaching%20application.pdf)
- Maranesi & Penna 2022 *Eur J Phys* 43:015002
- Mehta 1985 *Annu Rev Fluid Mech* 17:151–189
- Muto, Tsubokura, Oshima 2012 *Phys Fluids* 24:014102
- Kim et al. 2014 *J Fluid Mech* 754:R5
- Cohen et al. 2014 *Proc R Soc A* 470:20130497

---

## 4. Knuckle / 飘球模型 🟢 已校准并应用

**文件**: [src/vb_coach/aerodynamics/knuckle.py](../../src/vb_coach/aerodynamics/knuckle.py)

### 校准前 vs 校准后

| 常数 | 旧值 | 新值 | 依据 |
|---|---|---|---|
| `_CL_LAT_PEAK` | 0.12 | **0.10** | Norman & McKeon 2011 fig 7(b): z-RMS 峰值 0.10–0.11 |
| `_RE_LAT_PEAK` | 2.0×10⁵ | **3.0×10⁵** | Darbois Texier 2016 fig 7(b)：横向 RMS 峰**精确**在 Re ≈ 3×10⁵（与新 drag crisis 中心一致）|
| `TAU_KNUCKLE` | 0.25 s | **0.35 s** | Darbois Texier 2016 §4.1: λ/U ≈ 0.5 s；Table 2 给 τ_knuckle ≈ 0.33 s |

### 飘球速度区间的修正

原 [float-serve.md](float-serve.md) 给"飘球速度 13–20 m/s"基于 Re_crit ≈ 2×10⁵。新常数把临界中心移到 Re=3×10⁵：

| v [m/s] | Re | 新 Cl_lat | 物理状态 |
|---|---|---|---|
| 14 | 196 k | 0.025 | 临界边缘，弱飘 |
| 16 | 224 k | 0.045 | 临界进入 |
| 18 | 252 k | 0.072 | 临界中段 |
| **21** | **294 k** | **0.099 (峰值)** | **临界顶点** |
| 24 | 336 k | 0.078 | 临界退出 |
| 27 | 378 k | 0.029 | 超临界 |

**结论**：飘球最佳速度从原估计的 v ≈ 16 m/s 上移到 **v ≈ 20–21 m/s**。这跟实际比赛中"职业男排发球 20–22 m/s 最飘"的经验更一致；业余 16 m/s 的"飘"主要靠出手不旋而非真正的临界 panel-flicker。

测试 `test_simulate_stochastic_monte_carlo_landing_scatter` 已调整为 v=21 m/s。

**Sources**:
- Norman & McKeon 2011 *Exp Fluids* 51:1439 — [DOI 10.1007/s00348-011-1161-8](https://doi.org/10.1007/s00348-011-1161-8)
- Darbois Texier et al. 2016 *NJP* 18:073027 — [link](https://iopscience.iop.org/article/10.1088/1367-2630/18/7/073027)
- Cohen et al. 2014 *Proc R Soc A* 470:20130497
- Mizota et al. 2013 *Sci Rep* 3:1871（足球类比，St ~ 0.04）

---

## 5. 攻手体型与摸高 🟢 已应用 + 多 preset

**文件**: [src/vb_coach/setting/attacker.py](../../src/vb_coach/setting/attacker.py)

### 校准前 vs 校准后

| 项 | 旧值 | 新值 | 依据 |
|---|---|---|---|
| `DEFAULT_CONTACT_LOSS` | 0.15 m | **0.20 m** | Sheppard 2008 (AIS)：in-game contact_height = spike_reach − 18–22 cm |

`DEFAULT_AMATEUR_MALE` 数值（1.78 / 2.32 / 2.95）保留——它代表"会扣球的业余男（trained recreational）"，不是"普通成年男"。

### 新增 preset

```python
RECREATIONAL_MALE   = Attacker(1.75, 2.27, 2.78)               # 普通业余男
DEFAULT_AMATEUR_MALE = Attacker(1.78, 2.32, 2.95)              # 会扣球业余
COLLEGE_MALE        = Attacker(1.90, 2.48, 3.20)               # NCAA D2-D3 median
ELITE_MALE          = Attacker(1.97, 2.58, 3.45, c_loss=0.22)  # Palao 2014 OH median
DEFAULT_AMATEUR_FEMALE = Attacker(1.68, 2.15, 2.55, c_loss=0.18)
COLLEGE_FEMALE      = Attacker(1.82, 2.35, 2.95, c_loss=0.18)
ELITE_FEMALE        = Attacker(1.84, 2.39, 3.05, c_loss=0.20)  # Palao 2014 OH median
```

### FIVB 精英身高 + spike reach（Palao 2014 n=1440 男 / 1459 女）

按位置：

| 位置 | 男 身高 | 男 spike reach | 女 身高 | 女 spike reach |
|---|---|---|---|---|
| MB | 2.02 ± 0.05 | 3.48 ± 0.12 | 1.86 | 3.07 |
| OPP | 1.99 ± 0.06 | 3.48 ± 0.10 | 1.85 | 3.07 |
| OH | 1.97 ± 0.05 | 3.45 ± 0.10 | 1.84 | 3.05 |
| S | 1.92 ± 0.06 | 3.35 ± 0.11 | 1.77 | 2.94 |
| L | 1.86 ± 0.06 | 3.26 ± 0.13 | 1.71 | 2.82 |

国家队均值：USA 196, ITA 195.8, BRA 194.4, RUS 201.8 cm。

### 比赛中跳跃 vs 实验室 CMJ（Tilp 2023）

In-game attack jump：男 70.3 ± 18.7 cm，女 46.4 ± 14.2 cm。
实验室 CMJ 通常比 in-game 高 5–10 cm（疲劳与战术影响）。

### 攻手手部接触几何

- 球半径 0.105 m
- 手长（hand length）：男 18.4–20.0 cm，女 16.8–17.7 cm
- 击球时肘部并非全伸：屈 10–15°（即 165–170° extension），损失约 1 cm
- 击球点常不在头顶正上方：肩抬升角 165–170°，躯干侧屈 ~15° → 实际下移 8–12 cm

→ contact_loss = ball_radius (10.5) + hand_wrap (~5) + arm/shoulder/trunk flex (~5) ≈ **20 cm**

**Sources**:
- Palao, Manzanares & Valades 2014 *J Hum Kinetics* — [PMC4327374](https://pmc.ncbi.nlm.nih.gov/articles/PMC4327374/)
- Sheppard et al. 2008 *JSCR* — [PubMed 18438243](https://pubmed.ncbi.nlm.nih.gov/18438243/)
- Sattler et al. 2015 *JSCR* — [PubMed 25436623](https://pubmed.ncbi.nlm.nih.gov/25436623/)
- Pocek et al. 2021 *IJERPH* — [PMC8393901](https://pmc.ncbi.nlm.nih.gov/articles/PMC8393901/)
- Marques et al. 2009 *JSCR* — [PubMed 19528849](https://pubmed.ncbi.nlm.nih.gov/19528849/)
- Reeser et al. 2010 *Sports Health* — [PMC3445065](https://pmc.ncbi.nlm.nih.gov/articles/PMC3445065/)
- Tilp et al. 2023 *Scientific Reports* — [PMC10235019](https://pmc.ncbi.nlm.nih.gov/articles/PMC10235019/)

---

## 6. HitWindow 几何 ✅ 已用 Wagner 2020 校准

**文件**: [src/vb_coach/setting/hit_window.py](../../src/vb_coach/setting/hit_window.py)

非对称椭球 (r_x, r_y_front, r_y_back, r_z_up, r_z_down) = (0.40, 0.30, 0.15, 0.05, 0.30) m，已在 2026-05-10 重构中按 Wagner 2020 校准。详见 [DEVLOG](../DEVLOG.md) 同日条目。

唯一 outstanding 项：横向 r_x = 0.40 = Wagner 0.20 + ~0.20 业余 footwork tolerance。后者未实证。

### 4号位接触点坐标（中置信度，未应用）

| 项 | 项目值 | 文献建议 | 备注 |
|---|---|---|---|
| 沿底线 x | −4.0 m | −3.5 m（中位）| 当前 −4.0 m 偏 antenna，对应"打边线"极端；Hudl 教练材料指实战中位距 antenna 0.5–1.0 m 内 |
| 距网 y | +0.2 m | +0.3 m | Coleman 1993 / Wagner 2014: 实测 spike contact 距网 0.2–0.5 m |

**未应用原因**：文献只有教练学量级共识，缺少 peer-reviewed 直接量化。保留现值；如果未来想加 `OUTSIDE_PIN_ATTACK` 极端 preset，可分别用 −4.0 和 −3.5。

---

## 7. 评分函数常数 🔴 多数是真实数据空白

**文件**: [src/vb_coach/setting/attacks_position4.py](../../src/vb_coach/setting/attacks_position4.py)

### 现状

| 球种 | dwell_target [ms] | velocity_best [m/s] | velocity_band [m/s] | lateral_max [cm] |
|---|---|---|---|---|
| 高球 | 80 | -4.0 | 2.0 | 30 |
| 半高 | 50 | -2.5 | 1.5 | 20 |
| 平拉开 | 40 | 0.0 | 1.5 | 15 |

### 文献评估

| 项 | 结论 |
|---|---|
| **velocity_best**（高/半高/平拉开 = -4 / -2.5 / 0） | 物理推导（抛物线 + drag）一致，但**没有人测过球 v_z vs spike outcome**。半高 -2.5 偏弱半档，建议 -3.0（中置信度，未应用，因为没有直接实验数据） |
| **velocity_band**（2 / 1.5 / 1.5） | **没有任何文献支持**。建议平拉开收紧到 1.0（中置信度，未应用） |
| **dwell_target**（80 / 50 / 40 ms） | 物理推算自洽（窗口 z-span / vz），但**完全没有人**测过"球在攻手击球点 ±20 cm 球体内的停留时间"vs spike outcome |
| **lateral_max**（30 / 20 / 15 cm） | 高球 30 cm 可解释为 Wagner 2020 +20 cm muscle activation 阈值的"上限"，但 Wagner 测的是肩部 EMG 不是 spike outcome。其余两个值是直觉外推 |

### 三种战术相对容差比例

我们用 80:50:40 ms 和 30:20:15 cm（约 2:1.25:1）。教练共识"快球容错小"——但**没有数据**给出确切比例。Marcelino 2014 间接支持：early approach 4.2% blocked vs late 17%（这是 timing 而不是空间容差）。

### 攻击速度参考（未来扩展用）

- NCAA D1 男：22–27 m/s mean
- NCAA D1 女：16 m/s mean，peak 21 m/s
- Italian Serie A2 男：25.6–26.3 m/s mean
- Elite peak: Sharifi 2025 VNL 38.6 m/s（公开 spike 纪录）

### 二传旋转 = 0 ✓

Real-World Physics Problems + coachingvb 一致："clean set has low velocity and no rotation"。当前模型用 spin=0 即可。

**Sources**:
- Wagner et al. 2020 *JSAMS* — [PMC7256809](https://pmc.ncbi.nlm.nih.gov/articles/PMC7256809/)
- Marcelino et al. 2014 *J Sports Sci Med* — [PMC3737848](https://pmc.ncbi.nlm.nih.gov/articles/PMC3737848/)
- Eom & Schutz 1992 *RQES* 63(1):11–18
- Sarvestan et al. 2017 *J Hum Kinet* — [PMC5548173](https://pmc.ncbi.nlm.nih.gov/articles/PMC5548173/)
- Garcia-de-Alcaraz et al. 2021 *PLOS ONE* — [DOI 10.1371/journal.pone.0249612](https://doi.org/10.1371/journal.pone.0249612)
- Oliveira et al. 2020 *Int J Sports Sci Coach* — systematic review

---

## 8. 真实数据空白（需要自采）

排球教练科学有 **5 个相互独立的 gap**，都需要原始数据收集才能校准：

1. **球 v_z at contact vs spike outcome**：零论文。
2. **Dwell time of ball in strike zone**：零论文。Yoshida 2021 测攻手 COM hang time，不是球。
3. **三种球种的相对容差**：零文献支持递减比例。
4. **业余 vs 精英 hit window 比例**：零数据。
5. **Attacker 主观 Likert 评分** vs 客观几何指标：零论文。

**公开数据集状态**：
- VREN dataset（arXiv 2209.13846）有 1632 rallies × 12 k actions，但**没有 3D 球轨迹**。
- openvolley/datavolley R package 解析 *.dvw 格式，标 set quality 但**只标 quality grade，不标几何**。
- Kaggle PlusLiga / Beach Volleyball 都是 box-score 级别，没有 contact-level 数据。

**结论**：现有公开数据**不够校准** dwell / v_z / drift 阈值。需要自采。

---

## Applied Changes（2026-05-13 提交）

代码改动汇总：

```
src/vb_coach/aerodynamics/drag.py
  _CD_SUBCRITICAL   0.45 -> 0.50
  _CD_SUPERCRITICAL 0.18 -> 0.16
  _RE_CRIT_LOW      1.5e5 -> 2.0e5
  _RE_CRIT_HIGH     2.5e5 -> 3.3e5

src/vb_coach/aerodynamics/knuckle.py
  _CL_LAT_PEAK      0.12 -> 0.10
  _RE_LAT_PEAK      2.0e5 -> 3.0e5
  TAU_KNUCKLE       0.25 -> 0.35

src/vb_coach/setting/attacker.py
  DEFAULT_CONTACT_LOSS  0.15 -> 0.20
  + 6 个新 preset: RECREATIONAL_MALE, COLLEGE_MALE, ELITE_MALE,
                   DEFAULT_AMATEUR_FEMALE, COLLEGE_FEMALE, ELITE_FEMALE

tests/* — 数值断言同步更新，全部 106 测试通过
```

未应用（中置信度，留待后续）：

```
magnus.py: _CL_HALF_SATURATION_S 0.25 -> 0.15 (Kao 1994)
attacks_position4.py: _HALF_VELOCITY_BEST -2.5 -> -3.0 (物理推导)
attacks_position4.py: _SHOOT_VELOCITY_BAND 1.5 -> 1.0 (战术 timing 文献)
hit_window.py: _POSITION4_CONTACT_X -4.0 -> -3.5 (教练学共识)
hit_window.py: _POSITION4_CONTACT_Y +0.2 -> +0.3 (Coleman/Wagner)
```

---

## 参考文献完整清单

**Aerodynamics**
- Asai T., Ito S., Seo K., Hitotsubashi A. (2010) Aerodynamics of a new volleyball. *Procedia Eng* 2:2493–2498.
- Cohen C., Texier B.D., Dupeux G., Quéré D., Clanet C. (2014) The aerodynamic wall. *Proc R Soc A* 470:20130497.
- Darbois Texier B., Cohen C., Quéré D., Clanet C. (2016) Physics of knuckleballs. *New J Phys* 18:073027.
- Dumek J., Šafařík P. (2018) Complete description of forces acting on a flying beach volleyball. *EPJ Web Conf* 180:02021.
- Hong S., Asai T. (2020) Aerodynamic characteristics of new volleyball for the 2020 Tokyo Olympics. *Applied Sciences* 10:3256.
- Hong S., Weon B.M., Asai T. et al. (2018) Aerodynamic effects of a panel orientation in volleyball float serve. *ISBS Proceedings* 36:212.
- Kao S.S., Sellens R.W., Stevenson J.M. (1994) A mathematical model for the trajectory of a spiked volleyball. *J Appl Biomech* 10:95–109.
- Kim et al. (2014) Inverse Magnus effect on a rotating sphere. *J Fluid Mech* 754:R5.
- Maranesi V. & Penna G. (2022) The Magnus effect in volleyball service by video analysis. *Eur J Phys* 43:015002.
- Mehta R.D. (1985) Aerodynamics of sports balls. *Annu Rev Fluid Mech* 17:151–189.
- Mizota T. et al. (2013) Strange flight behaviour of slowly spinning soccer balls. *Sci Rep* 3:1871.
- Muto, Tsubokura, Oshima (2012) Negative Magnus effect. *Phys Fluids* 24:014102.
- Norman A.K. & McKeon B.J. (2011) Unsteady force measurements in sphere flow. *Exp Fluids* 51:1439–1453.

**Biomechanics / Performance**
- Eom H.J. & Schutz R.W. (1992) Statistical analyses of volleyball team performance. *Res Q Exerc Sport* 63(1):11–18.
- Garcia-de-Alcaraz A. et al. (2021) Force-velocity profile and spike/serve speed. *PLOS ONE* 16(4):e0249612.
- Marcelino R., Afonso J., Moraes J.C., Mesquita I. (2014) Determinants of attack players in high-level men's volleyball. *J Sports Sci Med* 13:813–820.
- Marques M.C. et al. (2009) Physical fitness qualities of professional volleyball players. *JSCR* 23(4):1106–1111.
- Oliveira et al. (2020) Systematic review of volleyball spike kinematics. *Int J Sports Sci Coach* 15(2).
- Palao J.M., Manzanares P., Valades D. (2014) Anthropometric, physical, and age differences by the player position and the performance level in volleyball. *J Hum Kinet* 44:223–236.
- Pocek S. et al. (2021) Body composition and physical performance of junior female volleyball players. *IJERPH* 18.
- Reeser J.C. et al. (2010) Upper limb biomechanics during the volleyball serve and spike. *Sports Health* 2(5):368–374.
- Sarvestan et al. (2017) Kinematic analysis of volleyball attack with various take-offs. *J Hum Kinet*.
- Sattler T. et al. (2015) Vertical jumping tests in volleyball by position and competition level. *JSCR* 29(5):1486–1493.
- Sheppard J.M. et al. (2008) Anthropometric, strength, and power of elite volleyball players. *JSCR* 22(3):758–765.
- Tilp M. et al. (2023) In-game jump heights of elite volleyball players. *Sci Rep* 13:9047.
- Wagner H. et al. (2020) Ball impact position in volleyball spike. *J Sci Med Sport* 23(8):768–772.
- Yoshida et al. (2021) Biomechanics of hang-time in volleyball spike jumps. *J Biomech*.
