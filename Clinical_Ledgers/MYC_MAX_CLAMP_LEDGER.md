# MYC/MAX Molecular Clamp — Bounded Forge · Atomic Refine · Off-target Hypothesis

**Status:** computational hypothesis · `clinical_grade=false` · 2026-07-18  
**Target:** MYC/MAX heterodimer interface (intrinsically disordered, flat PPI surface).

> Scope honesty: this is a **bounded** enumeration + refinement, **not** a full
> exhaustion of chemical space. **No MYC/MAX receptor** was available locally and web
> fetch is disallowed, so **no receptor docking to MYC** was performed.

## Phase 1 — Bounded combinatorial enumeration
- Grid: 6 polar heads × 4 rigid cages = 24 scaffolds.
- Stereochemical space (bounded, by stereocenter count): **174** nominal isomers.
- Passed sanity filters (Fsp³>0.5, aromatic≤2, MW 200–420, PAINS-free, embeddable):
  **8** (representative embedded).
- This is a defined finite grid — larger chemistries exist and are **not** claimed covered.

## Phase 2 — Atomic refinement (top candidate)
| Field | Value |
|-------|-------|
| ID | `MYC_MAX_CLAMP_001` |
| SMILES | `CN1CCN(CC(=O)NC23CC4CC(CC(C4)C2)C3)CC1` |
| Formula | C17H29N3O |
| Fsp³ | 0.9412 |
| Aromatic rings | 0 |
| MW | 291.4 |
| QED | 0.852 |
| Conformers (ETKDGv3+MMFF) | 80 |
| Min MMFF energy (kcal/mol) | 112.25 |
| Ensemble spread (kcal/mol) | 15.71 |

Interface π–π / cation–π and interface H-bond networks are **not** computed (no MYC
complex). We report only the ligand's intrinsic capacity: HBD 1,
HBA 3, TPSA 35.6, aromatic rings
0,
non-aromatic N (cation-π/salt-bridge capacity)
3.

## Phase 3 — Off-target selectivity **hypothesis** (not a proof)
2D Morgan Tanimoto + 3D USRCAT shape vs local panel (EGFR/ALK5/FLT3/GSK3B/AKT1/ABL1/BCL2):

| Panel target | Tanimoto 2D | USRCAT shape 3D |
|--------------|-------------|-----------------|
| ABL1_lead | 0.224 | 0.08 |
| BCL2_lead | 0.224 | 0.079 |
| FLT3_lead | 0.205 | 0.081 |
| AKT1_lead | 0.169 | 0.093 |
| EGFR_lead | 0.167 | 0.108 |
| ALK5_lead | 0.106 | 0.122 |
| GSK3B_lead | 0.074 | 0.137 |

Max Tanimoto to panel: **0.224**; max shape: **0.137**.
Low similarity ⇒ distinct aliphatic-cage chemotype (selectivity **hypothesis**).

### What this does NOT prove
- Not proteome-wide inertness; not off-target safety.
- Not target engagement / lethality to MYC.
- Ligand-only; no MYC receptor docking. Wet kinome/selectivity panels required.

## Air-gap
No hardware / silicon-artifact identifiers in this ledger. Run `tools/public_leak_audit.py`.
