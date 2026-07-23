# ABL1 · abl1_m5_07 — clinical ledger entry

> `clinical_grade = false` · Computational topology. Docking (Vina) is a **secondary /
> informational** metric — not measured affinity, IC50, or clinical efficacy.

**Target:** ABL proto-oncogene 1, tyrosine kinase (ABL1, UniProt [P00519](https://www.uniprot.org/uniprotkb/P00519/entry))
**Lane:** Oncology · **Indication:** Chronic myeloid leukemia / Ph+ (ATP site)
**Structure:** [3OXZ](https://www.rcsb.org/structure/3OXZ) — ABL kinase domain bound to ponatinib (AP24534)
**DepMap causality:** CML driver (BCR-ABL fusion)

![lead](../../Visualizations/ABL1_lead.png)

## Identity
| Field | Value |
|-------|-------|
| Lead ID | `abl1_m5_07` |
| SMILES | `CN1CCN(CCOc2ccc(C(=O)Nc3n[nH]c4ccccc34)cc2)CC1` |
| Formula | **C21H25N5O2** |
| InChIKey | `LWOOUBWLLABRNH-UHFFFAOYSA-N` |
| Warhead lineage | `abl1_r3_23` (C14H10FN3O, Fsp3 0.0) |

## Granular computational panel
| Metric | Value |
|--------|-------|
| Fsp3 | 0.3333 |
| Heavy-atom count (HAC) | 28 |
| MW (g/mol) | 379.46 |
| cLogP | 2.441 |
| TPSA (A^2) | 73.49 |
| QED | 0.6882 |
| Rotational barrier deltaE (kcal/mol) | 2.273 |
| Cavity Betti (b0, b1, b2) | (218, 448, 0) |
| Cavity Euler chi | -230 |
| TME pH prior (milli-pH) | 720 [physiological_prior] |

## Chemical Sanity auto-audit
- **Status:** PASS after **3** recursive attempt(s).
- **Strain:** 0.006 kcal/mol (ceiling 55.0).
- **PAINS / heavy-atom / Lipinski:** clear.

## Method
- Cavity topology: cubical complex (Z/2), 6-connectivity; chi=V-E+F-C; beta1=beta0+beta2-chi on the 64^3 pocket-occupancy tensor (18 A radius).
- Rotational barrier: MMFF relaxed (constrained) torsion scan; principal-hinge bond (30° steps).
- Docking box + reproduction: see [`../../Wave2_Disease_Expansion/DOCKING_GUIDE.md`](../../Wave2_Disease_Expansion/DOCKING_GUIDE.md).
