# AKT1 · akt1_lead — clinical ledger entry

> `clinical_grade = false` · Computational topology. Docking (Vina) is a **secondary /
> informational** metric — not measured affinity, IC50, or clinical efficacy.

**Target:** RAC-alpha serine/threonine-protein kinase (AKT1) (AKT1, UniProt [P31749](https://www.uniprot.org/uniprotkb/P31749/entry))
**Lane:** Oncology · **Indication:** PI3K-pathway cancer / PIK3CA-bypass (allosteric)
**Structure:** [3O96](https://www.rcsb.org/structure/3O96) — human AKT1 in complex with an allosteric inhibitor
**DepMap causality:** Compensatory escape for PIK3CA (PI3K/AKT axis)

![lead](../../Visualizations/AKT1_lead.png)

## Identity
| Field | Value |
|-------|-------|
| Lead ID | `akt1_lead` |
| SMILES | `Cc1nc(Nc2ccc(OCCN3CCN(C)CC3)cc2)cc(N2CCOCC2)n1` |
| Formula | **C22H32N6O2** |
| InChIKey | `FXTDRSFDAIEDGT-UHFFFAOYSA-N` |
| Warhead lineage | `akt_morpholino_pyrimidine` (C15H18N4O, Fsp3 0.3333) |

## Granular computational panel
| Metric | Value |
|--------|-------|
| Fsp3 | 0.5455 |
| Heavy-atom count (HAC) | 30 |
| MW (g/mol) | 412.54 |
| cLogP | 1.991 |
| TPSA (A^2) | 65.99 |
| QED | 0.7408 |
| Rotational barrier deltaE (kcal/mol) | 2.63 |
| Cavity Betti (b0, b1, b2) | (191, 347, 0) |
| Cavity Euler chi | -156 |
| TME pH prior (milli-pH) | 680 [physiological_prior] |

## Chemical Sanity auto-audit
- **Status:** PASS after **3** recursive attempt(s).
- **Strain:** 0.051 kcal/mol (ceiling 55.0).
- **PAINS / heavy-atom / Lipinski:** clear.

## Method
- Cavity topology: cubical complex (Z/2), 6-connectivity; chi=V-E+F-C; beta1=beta0+beta2-chi on the 64^3 pocket-occupancy tensor (18 A radius).
- Rotational barrier: MMFF relaxed (constrained) torsion scan; principal-hinge bond (30° steps).
- Docking box + reproduction: see [`../../Wave2_Disease_Expansion/DOCKING_GUIDE.md`](../../Wave2_Disease_Expansion/DOCKING_GUIDE.md).
