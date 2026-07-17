# Foliation Clinical Evidence Packs · 转化医学计算证据包

**EN / 中文** · `clinical_grade=false`

**EN:** Public computational evidence for multi-indication pharmacology. Docking ≠ wet affinity. No fabricated IC50.  
**中文：** 多适应症药理学的公开计算证据。对接 ≠ 湿实验亲和力。不虚构 IC50。

## Featured packs · 精选数据包

| Pack | Focus · 方向 | Link |
|------|--------------|------|
| **Translational Medicine** | **ER-100** · Retina regen · Heme | [`Translational_Medicine/`](Translational_Medicine/) |
| **Wave-2 Expansion** | ER-100 links · **视网膜再造** · AML/淋巴瘤/CML · **M5 leads** | [`Wave2_Disease_Expansion/`](Wave2_Disease_Expansion/) |

### Snapshot · 快照（RDKit formulas · verified）

| Program | Lead | Formula · 分子式 | Fsp³ | Vina* |
|---------|------|------------------|------|-------|
| ER-100 / ESR1 | `serm_stilbene_amine` | **C18H21NO2** | — | -7.045 |
| Retina / GSK3β | `gsk_x10` | **C15H17FN4O** | — | -7.648 |
| Retina discovery | `DISC-RETINA_REGEN-emix_bicyclo_F-8a573f6edf` | **C17H24FNO2** | — | — |
| AML / FLT3 **M5** | `flt3_m5_11` | **C22H25FN4O2** | 0.3182 | -10.91 |
| Lymphoma / BCL2 **M5** | `bcl2_m5_12` | **C24H27N3O4S** | 0.2917 | -10.14 |
| CML / ABL1 **M5** | `abl1_m5_07` | **C21H25N5O2** | 0.3333 | -10.57 |

\* Vina is secondary / informational only · 对接分仅供参考.

**EN:** Heme M5 = locked R3 warheads + high-Fsp³ N-methylpiperazine ethoxy solvent tails (formulas RDKit-verified).  
**中文：** 血液肿瘤 M5 = 锁定 R3 弹头 + 高 Fsp³ N-甲基哌嗪乙氧基溶剂尾（分子式经 RDKit 校验）。

## Other releases · 其他发布

See [`releases/`](releases/) for historical Cryo-EM homology and clinical showcase packs.

Proprietary silicon RTL and internal synthesis deliverables are **omitted** · 专有硅实现与内部合成交付物不公开。
