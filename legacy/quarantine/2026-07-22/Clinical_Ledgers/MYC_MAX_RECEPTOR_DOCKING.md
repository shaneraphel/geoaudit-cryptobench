# MYC/MAX Receptor Docking — real structure (honest)

`clinical_grade=false` · 2026-07-18T08:06:08.758773+00:00

## Receptor
- **1NKP** (c-Myc/Max bHLHZip heterodimer, RCSB), protein-only extract (DNA/water removed).
- Prep: prody + obabel (pH 7.4, rigid PDBQT).

## Pocket detection (fpocket 4.0)
| Pocket | Score | Druggability | Volume (A^3) |
|--------|-------|--------------|--------------|
| 22 | ? | 0.248 | 1177.141 |
| 1 | 0.081 | 0.158 | 277.035 |
| 26 | ? | 0.026 | 948.12 |
| 3 | 0.04 | 0.013 | 389.998 |
| 5 | 0.026 | 0.011 | 389.097 |

Top druggability = **0.248** — well below the ~0.5 "druggable" threshold. This is the honest, expected signature of a shallow MYC/MAX PPI/IDP interface.

## Docking (AutoDock Vina 1.2.5)
- Box center ['50.81', '29.58', '69.41'] (fpocket top pocket), size 22 A cube, exhaustiveness 8.
- Best affinity: **-6.229 kcal/mol** (9 modes).
- Interpretation: **weak/modest** score, consistent with the low druggability. A starting hypothesis, **not** a strong hit; docking is secondary geometry, not IC50.

## Honesty
Receptor docking to a real MYC/Max structure is now performed (earlier caveat removed). We do **not** claim a strong hit, target engagement, or selectivity proof. Wet validation required.
