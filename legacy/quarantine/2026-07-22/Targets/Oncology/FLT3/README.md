# FLT3 · flt3_m5_11 — clinical ledger entry

> `clinical_grade = false` · Computational topology. Docking (Vina) is a **secondary /
> informational** metric — not measured affinity, IC50, or clinical efficacy.

**Target:** Fms-like tyrosine kinase 3 (FLT3, UniProt [P36888](https://www.uniprot.org/uniprotkb/P36888/entry))
**Lane:** Oncology · **Indication:** Acute myeloid leukemia (FLT3-ITD)
**Structure:** [4RT7](https://www.rcsb.org/structure/4RT7) — FLT3 kinase domain with a small-molecule inhibitor · cross-dock [6JQR](https://www.rcsb.org/structure/6JQR)
**DepMap causality:** AML lineage dependency (FLT3-ITD driver)

![lead](../../Visualizations/FLT3_lead.png)

## Identity
| Field | Value |
|-------|-------|
| Lead ID | `flt3_m5_11` |
| SMILES | `CN1CCN(CCOc2ccc(NC(=O)c3c[nH]c4ccccc34)c(F)c2)CC1` |
| Formula | **C22H25FN4O2** |
| InChIKey | `BLOXIVIZRUDTHK-UHFFFAOYSA-N` |
| Warhead lineage | `flt3_r3_x10` (C15H10F2N2O, Fsp3 0.0) |

## Granular computational panel
| Metric | Value |
|--------|-------|
| Fsp3 | 0.3182 |
| Heavy-atom count (HAC) | 29 |
| MW (g/mol) | 396.47 |
| cLogP | 3.186 |
| TPSA (A^2) | 60.6 |
| QED | 0.6719 |
| Rotational barrier deltaE (kcal/mol) | 2.331 |
| Cavity Betti (b0, b1, b2) | (147, 316, 0) |
| Cavity Euler chi | -169 |
| TME pH prior (milli-pH) | 720 [physiological_prior] |

## Chemical Sanity auto-audit
- **Status:** PASS after **3** recursive attempt(s).
- **Strain:** 0.0 kcal/mol (ceiling 55.0).
- **PAINS / heavy-atom / Lipinski:** clear.

## Method
- Cavity topology: cubical complex (Z/2), 6-connectivity; chi=V-E+F-C; beta1=beta0+beta2-chi on the 64^3 pocket-occupancy tensor (18 A radius).
- Rotational barrier: MMFF relaxed (constrained) torsion scan; principal-hinge bond (30° steps).
- Docking box + reproduction: see [`../../Wave2_Disease_Expansion/DOCKING_GUIDE.md`](../../Wave2_Disease_Expansion/DOCKING_GUIDE.md).
