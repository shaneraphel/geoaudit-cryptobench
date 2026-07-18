# EGFR · egfr_lead — clinical ledger entry

> `clinical_grade = false` · Computational topology. Docking (Vina) is a **secondary /
> informational** metric — not measured affinity, IC50, or clinical efficacy.

**Target:** Epidermal growth factor receptor (EGFR, UniProt [P00533](https://www.uniprot.org/uniprotkb/P00533/entry))
**Lane:** Oncology · **Indication:** NSCLC / KRAS-bypass resistance (ATP site)
**Structure:** [1M17](https://www.rcsb.org/structure/1M17) — EGFR tyrosine kinase domain with the 4-anilinoquinazoline inhibitor erlotinib
**DepMap causality:** Compensatory escape for KRAS (RTK bypass)

![lead](../../Visualizations/EGFR_lead.png)

## Identity
| Field | Value |
|-------|-------|
| Lead ID | `egfr_lead` |
| SMILES | `COc1cc2ncnc(Nc3ccc(F)cc3)c2cc1OCCN1CCN(C)CC1` |
| Formula | **C22H26FN5O2** |
| InChIKey | `PVMFDKNLULFNHJ-UHFFFAOYSA-N` |
| Warhead lineage | `egfr_anilinoquinazoline` (C14H10FN3, Fsp3 0.0) |

## Granular computational panel
| Metric | Value |
|--------|-------|
| Fsp3 | 0.3636 |
| Heavy-atom count (HAC) | 30 |
| MW (g/mol) | 411.48 |
| cLogP | 3.147 |
| TPSA (A^2) | 62.75 |
| QED | 0.6408 |
| Rotational barrier deltaE (kcal/mol) | 1.214 |
| Cavity Betti (b0, b1, b2) | (218, 492, 0) |
| Cavity Euler chi | -274 |
| TME pH prior (milli-pH) | 650 [physiological_prior] |

## Chemical Sanity auto-audit
- **Status:** PASS after **3** recursive attempt(s).
- **Strain:** 0.225 kcal/mol (ceiling 55.0).
- **PAINS / heavy-atom / Lipinski:** clear.

## Method
- Cavity topology: cubical complex (Z/2), 6-connectivity; chi=V-E+F-C; beta1=beta0+beta2-chi on the 64^3 pocket-occupancy tensor (18 A radius).
- Rotational barrier: MMFF relaxed (constrained) torsion scan; principal-hinge bond (30° steps).
- Docking box + reproduction: see [`../../Wave2_Disease_Expansion/DOCKING_GUIDE.md`](../../Wave2_Disease_Expansion/DOCKING_GUIDE.md).
