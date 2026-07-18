# GSK3B · gsk_x10 — clinical ledger entry

> `clinical_grade = false` · Computational topology. Docking (Vina) is a **secondary /
> informational** metric — not measured affinity, IC50, or clinical efficacy.

**Target:** Glycogen synthase kinase-3 beta (GSK3B, UniProt [P49841](https://www.uniprot.org/uniprotkb/P49841/entry))
**Lane:** Regeneration · **Indication:** Retinal regeneration (Pax6/Rax induction)
**Structure:** [1Q3D](https://www.rcsb.org/structure/1Q3D) — GSK-3 beta complexed with staurosporine
**DepMap causality:** Regeneration modulator (Wnt/GSK3 axis)

![lead](../../Visualizations/GSK3B_lead.png)

## Identity
| Field | Value |
|-------|-------|
| Lead ID | `gsk_x10` |
| SMILES | `Cc1nc(Nc2ccccc2F)cc(N2CCOCC2)n1` |
| Formula | **C15H17FN4O** |
| InChIKey | `GFMIMAHEKHTIDE-UHFFFAOYSA-N` |
| Warhead lineage | — |

## Granular computational panel
| Metric | Value |
|--------|-------|
| Fsp3 | 0.3333 |
| Heavy-atom count (HAC) | 21 |
| MW (g/mol) | 288.33 |
| cLogP | 2.504 |
| TPSA (A^2) | 50.28 |
| QED | 0.9401 |
| Rotational barrier deltaE (kcal/mol) | 8.836 |
| Cavity Betti (b0, b1, b2) | (150, 253, 0) |
| Cavity Euler chi | -103 |
| TME pH prior (milli-pH) | 740 [physiological_prior] |

## Chemical Sanity auto-audit
- **Status:** PASS after **3** recursive attempt(s).
- **Strain:** 0.0 kcal/mol (ceiling 55.0).
- **PAINS / heavy-atom / Lipinski:** clear.

## Method
- Cavity topology: cubical complex (Z/2), 6-connectivity; chi=V-E+F-C; beta1=beta0+beta2-chi on the 64^3 pocket-occupancy tensor (18 A radius).
- Rotational barrier: MMFF relaxed (constrained) torsion scan; principal-hinge bond (30° steps).
- Docking box + reproduction: see [`../../Wave2_Disease_Expansion/DOCKING_GUIDE.md`](../../Wave2_Disease_Expansion/DOCKING_GUIDE.md).
