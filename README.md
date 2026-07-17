# Foliation Clinical Evidence Packs · 转化医学计算证据包

**EN / 中文** · `clinical_grade=false`

**EN:** Public **computational** evidence for multi-indication pharmacology. Docking ≠ wet
affinity. No fabricated IC50. See each pack's *"What we compute"* section for methodology.
**中文：** 多适应症药理学的公开**计算**证据。对接 ≠ 湿实验亲和力。不虚构 IC50。
方法学见各数据包的「我们在计算什么」章节。

## Featured packs · 精选数据包

| Pack | Focus · 方向 | Link |
|------|--------------|------|
| **Translational Medicine** | **ER-100** · Retina regen · Heme | [`Translational_Medicine/`](Translational_Medicine/) |
| **Wave-2 Expansion** | ER-100 links · 视网膜再造 · AML/淋巴瘤/CML · **M5 leads + figures + docking guide** | [`Wave2_Disease_Expansion/`](Wave2_Disease_Expansion/) |

### Fragment → lead figure · 片段到先导

![Warhead to lead overview](Wave2_Disease_Expansion/images/M5_overview.png)

### Snapshot · 快照（RDKit formulas · verified）

| Program | Lead | Formula · 分子式 | Fsp³ | Vina* |
|---------|------|------------------|------|-------|
| ER-100 / ESR1 | `serm_stilbene_amine` | **C18H21NO2** | — | -7.045 |
| Retina / GSK3β | `gsk_x10` | **C15H17FN4O** | — | -7.648 |
| FLT3 / M5 | `flt3_m5_11` | **C22H25FN4O2** | 0.3182 | -10.91 |
| BCL2 / M5 | `bcl2_m5_12` | **C24H27N3O4S** | 0.2917 | -10.14 |
| ABL1 / M5 | `abl1_m5_07` | **C21H25N5O2** | 0.3333 | -10.57 |

\* Vina is secondary / informational only · 对接分仅供参考。

**EN:** Heme M5 = locked mass-screen warheads + high-Fsp³ polar solvent tails. Each lead ships
a spec sheet, 2D figure, SDF, and a reproducible docking guide. Formulas RDKit-verified.
**中文：** 血液肿瘤 M5 = 锁定筛选弹头 + 高 Fsp³ 极性溶剂尾。每个先导附带说明书、结构图、SDF
及可复现对接指南；分子式经 RDKit 校验。

## Other releases · 其他发布

See [`releases/`](releases/) for historical Cryo-EM homology and clinical showcase packs.

Proprietary silicon RTL and internal synthesis deliverables are **omitted** · 专有硅实现与内部合成交付物不公开。
