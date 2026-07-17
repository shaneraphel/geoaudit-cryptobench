# Translational Medicine · 转化医学证据包

**EN / 中文 bilingual** · Generated: `2026-07-17T08:58:45.469076+00:00`

![Boolean pocket tensor × synthon ligand graph](hero_banner.png)

> **`clinical_grade = false`**
>
> **EN:** Chemical Sanity is primary. Docking affinity is secondary / informational only. Computational gates ≠ measured IC50.
>
> **中文：** 化学合理性门控优先；对接亲和力仅作次级参考。计算分数 ≠ 湿实验 IC50 / 临床疗效。

---

## Program map · 项目地图

| Lane · 赛道 | Gene · 靶点 | Indication · 适应症 | Pack · 数据包 |
|-------------|-------------|---------------------|---------------|
| **ER-100** | **ESR1** | ER+ breast · 雌激素受体阳性乳腺癌 | [`Pipeline_Matrix/`](Pipeline_Matrix/) · CRI **4/4** SERMs |
| **Retina regen · 视网膜再造** | **GSK3β** (+ discovery chemotypes) | Stem-cell / Pax6–Rax induction · 视网膜干细胞诱导 | [`../Wave2_Disease_Expansion/`](../Wave2_Disease_Expansion/) |
| **Heme · 血癌/淋巴癌** | **FLT3 / BCL2 / ABL1** | AML · B-cell lymphoma · CML | [`../Wave2_Disease_Expansion/`](../Wave2_Disease_Expansion/) · [`wetlab/`](wetlab/) |

---

## 1. ER-100 (ESR1) · 雌激素受体项目

**EN:** ER-100 is the flagship ER+ breast program. Leads are SERM-class graphs that pass the full Clinical Readiness Index (DepMap · TME pH · liability shields · Chemical Sanity). Docking is secondary.

**中文：** ER-100 为 ER+ 乳腺癌旗舰项目。候选为 SERM 类分子图，需通过完整临床就绪指数（DepMap · 肿瘤微环境 pH · 脱靶屏蔽 · 化学合理性）。对接分数仅作次级指标。

| ID | Formula · 分子式 | HA | MW | Vina (secondary) | CRI |
|----|------------------|----|----|------------------|-----|
| `serm_stilbene_amine` | **C18H21NO2** | 21 | 283.37 | -7.045 | 4/4 ACTIVE |
| `serm_biphenyl_amine` | **C16H19NO2** | 19 | 257.33 | -6.823 | 4/4 ACTIVE |

SMILES:
- `serm_stilbene_amine`: `Oc1ccc(/C=C/c2ccc(OCCN(C)C)cc2)cc1`
- `serm_biphenyl_amine`: `Oc1ccc(-c2ccc(OCCN(C)C)cc2)cc1`

ACTIVE pool (n=7) also includes `serm_oht_parent`, `serm_oht_fluorophenyl`, `serm_oht_tolyl`, `serm_oht_bcp`, `serm_oht_azaspiro` — see [`VALIDATED_CANDIDATE_POOL.json`](VALIDATED_CANDIDATE_POOL.json).

Hero visual (ESR1 64³ pocket × stilbene amine): [`hero_banner.png`](hero_banner.png)

---

## 2. Retinal regeneration · 视网膜再造

**EN:** Small-molecule induction chemotypes for retinal reprogramming (Pax6/Rax axis). GSK3β is treated as a **modulatory** (not plug-only) target; discovery SDFs are high-Fsp³ / oxetane–bicyclo motifs. Companion cryo maps: DOT1L (epigenetic co-target).

**中文：** 面向视网膜细胞重编程（Pax6/Rax）的小分子诱导化学型。GSK3β 按**调谐/变构逻辑**（非单纯堵孔）筛选；发现集为高 Fsp³ / 氧杂环丁烷–双环骨架。表观共靶 DOT1L 冷冻电镜图已归档于本地 vault。

### GSK3β modulatory lead · 调谐先导

| ID | Formula · 分子式 | HA | MW | Vina |
|----|------------------|----|----|------|
| `gsk_x10` | **C15H17FN4O** | 21 | 288.33 | -7.648 |

SMILES: `Cc1nc(N2CCOCC2)cc(Nc2ccccc2F)n1`

### Discovery chemotypes · 发现化学型（RETINA_REGEN）

| ID | Formula · 分子式 | HA | MW |
|----|------------------|----|----|
| `DISC-RETINA_REGEN-emix_bicyclo_F-8a573f6edf` | **C17H24FNO2** | 21 | 293.38 |
| `DISC-RETINA_REGEN-emix_oxetane_F-4d5df33d4a` | **C13H18FNO3** | 18 | 255.29 |
| `DISC-RETINA_REGEN-emix_pyridyl_oxetane-899943b081` | **C12H18N2O3** | 17 | 238.29 |

SDF files: [`../Wave2_Disease_Expansion/retina_ligands/`](../Wave2_Disease_Expansion/retina_ligands/)

---

## 3. Hematologic oncology · 血液肿瘤 / 淋巴瘤

**EN:** Mass Chemical Sanity + secondary Vina (R3: exhaustiveness 16 × 6 seeds). Formulas RDKit-verified.

**中文：** 大规模化学合理性筛选 + 次级对接（R3：exhaustiveness 16 × 6 随机种子）。分子式均经 RDKit 校验。

| Gene | Disease · 病种 | Best ID | Formula · 分子式 | Vina |
|------|----------------|---------|------------------|------|
| **FLT3** | AML / FLT3-ITD | `{r3map['FLT3']['id']}` | **{r3map['FLT3']['formula']}** | **{r3map['FLT3']['vina']}** |
| **BCL2** | B-cell lymphoma · B 细胞淋巴瘤 | `{r3map['BCL2']['id']}` | **{r3map['BCL2']['formula']}** | **{r3map['BCL2']['vina']}** |
| **ABL1** | CML / Ph+ leukemia · 慢粒 | `{r3map['ABL1']['id']}` | **{r3map['ABL1']['formula']}** | **{r3map['ABL1']['vina']}** |

Details: [`../Wave2_Disease_Expansion/README.md`](../Wave2_Disease_Expansion/README.md) · ABL1 wet-lab (K562): [`wetlab/`](wetlab/)

---

## 4. Pipeline gates · 管线门控（shared）

1. **DepMap Boolean causality** — [`DEPMAP_KILL_SWITCH_JUSTIFICATION.md`](DEPMAP_KILL_SWITCH_JUSTIFICATION.md)
2. **TCGA TME → ProtonationState lock** — breast milli-pH 680 · BM niche 720
3. **Chemical Sanity** — MAX_HEAVY_ATOMS ≤ 35 · PAINS · strain
4. **Clinical Readiness Index** — Boolean ranking (**not** Vina) — [`UNIFIED_CLINICAL_LEDGER.md`](UNIFIED_CLINICAL_LEDGER.md)

---

## 5. Compensatory atlas · 代偿网络

Targets that fail DepMap kill-switch are retained as escape / bypass nodes — [`Compensatory_Atlas/README.md`](Compensatory_Atlas/README.md).

---

## Disclaimer · 免责声明

**EN:** This repository publishes computational evidence only. No fabricated wet IC50 / TGI%. Proprietary silicon RTL omitted.

**中文：** 本仓库仅发布计算证据包。不虚构湿实验 IC50 / TGI%。专有硅实现细节不公开。
