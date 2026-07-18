# MYC/MAX Orthogonal Scaffold-Hopping — Diversity Matrix

**Status:** computational hypothesis · `clinical_grade=false` · 2026-07-18  
**Target:** MYC/MAX flat PPI interface. Ligand-side diversity (no receptor docking; no local MYC structure).

Constraints enforced: distinct Murcko scaffolds (graph ban) · pairwise Morgan Tanimoto **< 0.35** · HAC **< 40** · PAINS-free · MMFF-embeddable.

## Three orthogonal motifs

| Class | SMILES | Formula | HAC | Fsp³ | Rings (max size) | Spiro | Bridgehead | MMFF min / spread (kcal/mol) |
|-------|--------|---------|-----|------|------------------|-------|------------|------------------------------|
| macrocycle | `O=C1CCCCCNC(=O)CCNC(=O)CCCCCN1` | C15H27N3O3 | 21 | 0.8 | 1 (18) | 0 | 0 | -71.79 / 9.19 |
| spiro | `O=C(NC1CC2(COC2)C1)C1CC2(CCC2)C1` | C14H21NO2 | 17 | 0.9286 | 4 (4) | 2 | 0 | 32.19 / 1.25 |
| bridged | `CN1CCN(CC(=O)NC23CC4CC(CC(C4)C2)C3)CC1` | C17H29N3O | 21 | 0.9412 | 5 (6) | 0 | 4 | 112.25 / 15.71 |

## Tanimoto orthogonality matrix (Morgan 2, 2048 bit)

| | macrocycle | spiro | bridged |
|--|--|--|--|
| macrocycle | 1.0 | 0.065 | 0.041 |
| spiro | 0.065 | 1.0 | 0.148 |
| bridged | 0.041 | 0.148 | 1.0 |

Max off-diagonal Tanimoto = **0.148** (< 0.35 ⇒ structurally orthogonal).

## Murcko scaffolds (distinct)
- macrocycle: `O=C1CCCCCNC(=O)CCNC(=O)CCCCCN1`
- spiro: `O=C(NC1CC2(COC2)C1)C1CC2(CCC2)C1`
- bridged: `O=C(CN1CCNCC1)NC12CC3CC(CC(C3)C1)C2`

## Honesty / boundary
- `clinical_grade=false`; ligand-side only; **no MYC receptor docking** (no local structure).
- Strain reported as MMFF conformer-ensemble min + spread; **not** claimed ≈0.
- Diversity is structural (2D/Murcko); target engagement requires wet validation.
