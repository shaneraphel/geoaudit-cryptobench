# Wave-2 Disease Expansion · 疾病扩展包（中英双语）

Generated: `2026-07-17T08:58:45.469076+00:00` · **`clinical_grade = false`**

**EN:** Multi-indication expansion pack: ER-100 companion links, retinal regeneration, and hematologic oncology (AML / lymphoma / CML).  
**中文：** 多适应症扩展包：ER-100 关联、视网膜再造、血液肿瘤（AML / 淋巴瘤 / CML）。

---

## A. ER-100 (ESR1) · 乳腺癌旗舰（关联）

| ID | Formula · 分子式 | Vina | Note |
|----|------------------|------|------|
| `serm_stilbene_amine` | **C18H21NO2** | -7.045 | SERM · CRI 4/4 |
| `serm_biphenyl_amine` | **C16H19NO2** | -6.823 | SERM · CRI 4/4 |

Full ER-100 narrative: [`../Translational_Medicine/README.md`](../Translational_Medicine/README.md) · SDF: [`ligands/ESR1_serm_stilbene_amine.sdf`](ligands/ESR1_serm_stilbene_amine.sdf)

---

## B. Retinal regeneration · 视网膜再造

**EN:** Modulatory GSK3β + high-Fsp³ discovery chemotypes (oxetane / bicyclo) for stem-cell induction.  
**中文：** GSK3β 调谐型先导 + 高 Fsp³ 发现化学型（氧杂环丁烷 / 双环），用于干细胞诱导路径。

| ID | Formula · 分子式 | HA | Role · 角色 |
|----|------------------|----|-------------|
| `gsk_x10` | **C15H17FN4O** | 21 | GSK3β modulatory · 调谐先导 (Vina -7.648) |
| `DISC-RETINA_REGEN-emix_bicyclo_F-8a573f6edf` | **C17H24FNO2** | 21 | Discovery · 发现化学型 |
| `DISC-RETINA_REGEN-emix_oxetane_F-4d5df33d4a` | **C13H18FNO3** | 18 | Discovery · 发现化学型 |
| `DISC-RETINA_REGEN-emix_pyridyl_oxetane-899943b081` | **C12H18N2O3** | 17 | Discovery · 发现化学型 |

SMILES (GSK3β): `Cc1nc(N2CCOCC2)cc(Nc2ccccc2F)n1`

Retina SDFs: [`retina_ligands/`](retina_ligands/)

---

## C. Hematologic oncology · 血癌 / 淋巴癌（Mass R3）

| Gene | Disease · 病种 | Best ID | Formula · 分子式 | Vina |
|------|----------------|---------|------------------|------|
| **FLT3** | AML | `flt3_r3_x10` | **C15H10F2N2O** | **-10.34** |
| **BCL2** | B-cell lymphoma · 淋巴瘤 | `bcl2_r3_29` | **C17H13NO3S** | **-9.372** |
| **ABL1** | CML | `abl1_r3_23` | **C14H10FN3O** | **-10.52** |

SMILES:
- FLT3: `O=C(Nc1ccc(F)cc1F)c1c[nH]c2ccccc12`
- BCL2: `O=C(NS(=O)(=O)c1ccc2ccccc2c1)c1ccccc1`
- ABL1: `Fc1ccc(C(=O)Nc2n[nH]c3ccccc23)cc1`

**EN:** R3 budget = exhaustiveness 16 × 6 seeds × 8 workers. Formulas RDKit-verified.  
**中文：** R3 计算预算 = exhaustiveness 16 × 6 种子 × 8 线程。分子式均经 RDKit 校验。

---

## Files · 文件

- [`WAVE2_PUBLIC_LEDGER.json`](WAVE2_PUBLIC_LEDGER.json)
- [`HEMATOLOGY_FORMULA_AUDIT.json`](HEMATOLOGY_FORMULA_AUDIT.json)
- [`ligands/`](ligands/) — ESR1 / GSK3β / FLT3 / BCL2 / ABL1 / retina SDFs
- [`retina_ligands/`](retina_ligands/)

## Method · 方法

- Max heavy atoms ≤ 35 · PAINS · strain
- 64³ Boolean pocket occupancy
- Vina secondary only
