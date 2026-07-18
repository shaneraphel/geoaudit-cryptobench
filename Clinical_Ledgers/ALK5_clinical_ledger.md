# ALK5 · alk5_x00 — clinical ledger entry

> `clinical_grade = false` · Computational topology. Docking (Vina) is a **secondary /
> informational** metric — not measured affinity, IC50, or clinical efficacy.

**Target:** TGF-beta receptor type-1 (TGFBR1/ALK5) (ALK5, UniProt [P36897](https://www.uniprot.org/uniprotkb/P36897/entry))
**Lane:** Fibrosis · **Indication:** Renal fibrosis / uremia (TGF-beta axis)
**Structure:** [3TZM](https://www.rcsb.org/structure/3TZM) — TGF-beta receptor type-1 in complex with SB431542
**DepMap causality:** Fibrosis signaling (TGF-beta, not a cancer dependency)

![lead](../../Visualizations/ALK5_lead.png)

## Identity
| Field | Value |
|-------|-------|
| Lead ID | `alk5_x00` |
| SMILES | `O=C(Nc1ccc(F)cc1)c1ccc(N2CCOCC2)nc1` |
| Formula | **C16H16FN3O2** |
| InChIKey | `FDQVNFVEHJBFLA-UHFFFAOYSA-N` |
| Warhead lineage | — |

## Granular computational panel
| Metric | Value |
|--------|-------|
| Fsp3 | 0.25 |
| Heavy-atom count (HAC) | 22 |
| MW (g/mol) | 301.32 |
| cLogP | 2.31 |
| TPSA (A^2) | 54.46 |
| QED | 0.945 |
| Rotational barrier deltaE (kcal/mol) | 3.126 |
| Cavity Betti (b0, b1, b2) | (187, 368, 0) |
| Cavity Euler chi | -181 |
| TME pH prior (milli-pH) | 710 [physiological_prior] |

## Chemical Sanity auto-audit
- **Status:** PASS after **3** recursive attempt(s).
- **Strain:** 0.0 kcal/mol (ceiling 55.0).
- **PAINS / heavy-atom / Lipinski:** clear.

## Method
- Cavity topology: cubical complex (Z/2), 6-connectivity; chi=V-E+F-C; beta1=beta0+beta2-chi on the 64^3 pocket-occupancy tensor (18 A radius).
- Rotational barrier: MMFF relaxed (constrained) torsion scan; principal-hinge bond (30° steps).
- Docking box + reproduction: see [`../../Wave2_Disease_Expansion/DOCKING_GUIDE.md`](../../Wave2_Disease_Expansion/DOCKING_GUIDE.md).
