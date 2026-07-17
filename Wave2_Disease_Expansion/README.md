# Wave-2 Disease Expansion · 疾病扩展包（中英双语）

Generated: `2026-07-17T14:54:35.086389+00:00` · **`clinical_grade = false`**

**EN:** Multi-indication expansion pack: ER-100 companion links, retinal regeneration, and hematologic oncology (AML / lymphoma / CML) through Sprint M5 fragment→lead growth.  
**中文：** 多适应症扩展包：ER-100 关联、视网膜再造、血液肿瘤（AML / 淋巴瘤 / CML），含 Sprint M5 片段→先导扩展。

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

## C. Hematologic oncology · 血癌 / 淋巴癌

### C1. Mass R3 warheads · 片段弹头（Fsp³ ≈ 0）

| Gene | Disease · 病种 | Best ID | Formula · 分子式 | Vina |
|------|----------------|---------|------------------|------|
| **FLT3** | AML | `flt3_r3_x10` | **C15H10F2N2O** | **-10.34** |
| **BCL2** | B-cell lymphoma · 淋巴瘤 | `bcl2_r3_29` | **C17H13NO3S** | **-9.372** |
| **ABL1** | CML | `abl1_r3_23` | **C14H10FN3O** | **-10.52** |

### C2. Sprint M5 clinical leads · 片段→先导（高 Fsp³ 溶剂尾）

**EN:** Aromatic warheads locked; N-methylpiperazine / morpholine ethoxy solvent-channel tails; MAX_HEAVY=40 (M5 only); PAINS+strain retained. Formulas RDKit-verified before publish.  
**中文：** 芳香弹头锁定；N-甲基哌嗪 / 吗啉乙氧基溶剂通道侧链；本 sprint 放宽 MAX_HEAVY=40；保留 PAINS+应变。上架前分子式经 RDKit 校验。

| Gene | Indication | Warhead → Lead | Formula | Fsp³ | cLogP | Vina* |
|------|------------|----------------|---------|------|-------|-------|
| **FLT3** | AML / FLT3-ITD | `flt3_r3_x10` → **`flt3_m5_11`** | **C15H10F2N2O** → **C22H25FN4O2** | 0.0→**0.3182** (Δ0.3182) | 3.698→**3.186** (Δ-0.512) | **-10.91** |
| **BCL2** | B-cell lymphoma | `bcl2_r3_29` → **`bcl2_m5_12`** | **C17H13NO3S** → **C24H27N3O4S** | 0.0→**0.2917** (Δ0.2917) | 2.959→**2.585** (Δ-0.374) | **-10.14** |
| **ABL1** | CML / Ph+ leukemia | `abl1_r3_23` → **`abl1_m5_07`** | **C14H10FN3O** → **C21H25N5O2** | 0.0→**0.3333** (Δ0.3333) | 2.954→**2.441** (Δ-0.513) | **-10.57** |

SMILES (M5 leads):
- FLT3 `flt3_m5_11`: `O=C(Nc1ccc(OCCN2CCN(C)CC2)cc1F)c1c[nH]c2ccccc12`
- BCL2 `bcl2_m5_12`: `O=C(NS(=O)(=O)c1ccc2cc(OCCN3CCN(C)CC3)ccc2c1)c1ccccc1`
- ABL1 `abl1_m5_07`: `O=C(Nc1n[nH]c2ccccc12)c1ccc(OCCN2CCN(C)CC2)cc1`

SDFs: [`ligands/FLT3_flt3_m5_11.sdf`](ligands/FLT3_flt3_m5_11.sdf) · [`ligands/BCL2_bcl2_m5_12.sdf`](ligands/BCL2_bcl2_m5_12.sdf) · [`ligands/ABL1_abl1_m5_07.sdf`](ligands/ABL1_abl1_m5_07.sdf)

Ledger: [`SPRINT_M5_FRAGMENT_TO_LEAD.json`](SPRINT_M5_FRAGMENT_TO_LEAD.json) · Formula audit: [`SPRINT_M5_FORMULA_VERIFY.json`](SPRINT_M5_FORMULA_VERIFY.json)

---

## Files · 文件

- [`WAVE2_PUBLIC_LEDGER.json`](WAVE2_PUBLIC_LEDGER.json)
- [`HEMATOLOGY_FORMULA_AUDIT.json`](HEMATOLOGY_FORMULA_AUDIT.json)
- [`SPRINT_M5_FRAGMENT_TO_LEAD.json`](SPRINT_M5_FRAGMENT_TO_LEAD.json)
- [`ligands/`](ligands/) — ESR1 / GSK3β / FLT3 / BCL2 / ABL1 / retina / **M5** SDFs
- [`retina_ligands/`](retina_ligands/)

## Method · 方法

- Chemical Sanity · PAINS · strain (M5: MAX_HEAVY ≤ 40)
- 64³ Boolean pocket occupancy
- Hydration-shell proxy (solvent-tail polar synthons)
- Vina secondary only
