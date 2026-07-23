# BCL2 · bcl2_m5_12 — clinical ledger entry

> `clinical_grade = false` · Computational topology. Docking (Vina) is a **secondary /
> informational** metric — not measured affinity, IC50, or clinical efficacy.

**Target:** B-cell lymphoma 2 (apoptosis regulator) (BCL2, UniProt [P10415](https://www.uniprot.org/uniprotkb/P10415/entry))
**Lane:** Oncology · **Indication:** B-cell lymphoma / CLL (BH3 groove)
**Structure:** [6O0K](https://www.rcsb.org/structure/6O0K) — BCL-2 in complex with venetoclax
**DepMap causality:** Apoptosis dependency (BH3 groove)

![lead](../../Visualizations/BCL2_lead.png)

## Identity
| Field | Value |
|-------|-------|
| Lead ID | `bcl2_m5_12` |
| SMILES | `CN1CCN(CCOc2ccc3cc(S(=O)(=O)NC(=O)c4ccccc4)ccc3c2)CC1` |
| Formula | **C24H27N3O4S** |
| InChIKey | `UOVDAIIFOQHALZ-UHFFFAOYSA-N` |
| Warhead lineage | `bcl2_r3_29` (C17H13NO3S, Fsp3 0.0) |

## Granular computational panel
| Metric | Value |
|--------|-------|
| Fsp3 | 0.2917 |
| Heavy-atom count (HAC) | 32 |
| MW (g/mol) | 453.56 |
| cLogP | 2.585 |
| TPSA (A^2) | 78.95 |
| QED | 0.5925 |
| Rotational barrier deltaE (kcal/mol) | 2.674 |
| Cavity Betti (b0, b1, b2) | (188, 328, 0) |
| Cavity Euler chi | -140 |
| TME pH prior (milli-pH) | 720 [physiological_prior] |

## Chemical Sanity auto-audit
- **Status:** PASS after **3** recursive attempt(s).
- **Strain:** 0.456 kcal/mol (ceiling 55.0).
- **PAINS / heavy-atom / Lipinski:** clear.

## Method
- Cavity topology: cubical complex (Z/2), 6-connectivity; chi=V-E+F-C; beta1=beta0+beta2-chi on the 64^3 pocket-occupancy tensor (18 A radius).
- Rotational barrier: MMFF relaxed (constrained) torsion scan; principal-hinge bond (30° steps).
- Docking box + reproduction: see [`../../Wave2_Disease_Expansion/DOCKING_GUIDE.md`](../../Wave2_Disease_Expansion/DOCKING_GUIDE.md).
