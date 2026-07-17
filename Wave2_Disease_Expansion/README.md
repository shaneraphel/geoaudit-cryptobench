# Hematologic Oncology Expansion · AML / Lymphoma

Generated: `2026-07-17T03:50:56.653825+00:00`

> **`clinical_grade = false`**
> Focus: **blood cancers** (AML / FLT3-ITD leukemia) and **B-cell lymphoma** (BCL2 apoptotic escape).
> Chemical Sanity is primary. Docking is secondary. All **molecular formulas are RDKit-verified**.

## Why this pack

Hematologic malignancies are the primary clinical lane in this release:

| Priority | Gene | Disease | Why it matters |
|----------|------|---------|----------------|
| **P0** | **FLT3** | AML / FLT3-ITD | Driver kinase in acute myeloid leukemia; ATP-site steric fill |
| **P0** | **BCL2** | B-cell lymphoma | Apoptotic buffering / escape after kinase pressure (venetoclax-class pocket) |
| P1 | ABL1 | CML / Ph+ leukemia | Existing wet-lab packs on K562 — see parent Translational_Medicine |

Companion (non-heme) wave-2 runs (fibrosis / retina) are listed at the bottom for completeness only.

## Hematology results

| Gene | Disease | N screened | Docked | Best ID | Formula | Best Vina |
|------|---------|------------|--------|---------|---------|-----------|
| **FLT3** | AML | 40 | 31 | `flt3_r2_00` | **C15H10F2N2O** | **-10.25** |
| **BCL2** | Lymphoma | 24 | 23 | `bcl2_naphthyl_SO2` | **C18H15NO3S** | **-9.174** |

## Lead molecules (formula-checked)

### FLT3 · AML

- ID: `flt3_r2_00`
- Formula: **C15H10F2N2O** (HA=20, MW=272.25)
- SMILES: `O=C(Nc1ccc(F)cc1)c1c[nH]c2ccc(F)cc12`
- Structure refs: PDB 4RT7 / 6JQR / 5X02 / 4XUF

### BCL2 · B-cell lymphoma

- ID: `bcl2_naphthyl_SO2`
- Formula: **C18H15NO3S** (HA=23, MW=325.39)
- SMILES: `O=C(NS(=O)(=O)c1ccc(C)cc1)c1ccc2ccccc2c1`
- Structure refs: PDB 6O0K BCL2

## Top FLT3 (AML) formulas

| ID | Formula | HA | Vina |
|----|---------|----|------|
| `flt3_r2_00` | C15H10F2N2O | 20 | -10.25 |
| `flt3_r2_06` | C15H11FN2O | 19 | -10.13 |
| `flt3_carboxamide_indole` | C15H12N2O | 18 | -9.991 |
| `flt3_r2_01` | C14H17F2N7 | 23 | -9.746 |
| `flt3_r2_04` | C17H17N5O | 23 | -9.721 |
| `flt3_x06` | C17H17FN2O | 21 | -9.697 |

## Top BCL2 (lymphoma) formulas

| ID | Formula | HA | Vina |
|----|---------|----|------|
| `bcl2_naphthyl_SO2` | C18H15NO3S | 23 | -9.174 |
| `bcl2_biphenyl_SO2` | C20H17NO3S | 25 | -9.116 |
| `bcl2_CF3_SO2` | C14H10F3NO3S | 22 | -8.904 |
| `bcl2_r2_00` | C17H18N2O4S | 24 | -8.455 |
| `bcl2_r2_01` | C14H11F2NO3S | 21 | -8.36 |
| `bcl2_fluoro_biphenyl` | C14H12FNO3S | 20 | -8.306 |

## Formula verification

All SMILES above were parsed with RDKit; `molecular_formula` = `CalcMolFormula`.  
Audit file: [`HEMATOLOGY_FORMULA_AUDIT.json`](HEMATOLOGY_FORMULA_AUDIT.json) · `formula_verified: true` on every lead.

## Files

- [`WAVE2_PUBLIC_LEDGER.json`](WAVE2_PUBLIC_LEDGER.json)
- [`ligands/`](ligands/) — SDF poses (FLT3 + BCL2)
- Parent: [`../Translational_Medicine/`](../Translational_Medicine/) (ABL1/K562 wet-lab packs)

## Method

- Max heavy atoms ≤ 35; PAINS / strain gates
- 64³ Boolean pocket occupancy
- Vina secondary: multi-seed, exhaustiveness ≥ 10
- BCL2 receptor: rigid PDBQT @ TME-like pH 7.2 (6O0K)

## Companion (not blood/lymphoma)

- **ALK5** (renal fibrosis): `alk5_x00` · C16H16FN3O2 · Vina -8.997
- **GSK3B** (retinal reprogramming (companion)): `gsk_x10` · C15H17FN4O · Vina -7.648
