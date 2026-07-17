# Wave-2 Disease Expansion · FLT3 / ALK5 / GSK3β

Generated: `2026-07-17T03:45:02.393379+00:00`

> **`clinical_grade = false`**
> Chemical Sanity is primary. Docking affinity is secondary / informational only.
> Computational gates ≠ measured IC50 or clinical efficacy.

## Indications

| Gene | Indication | Geometric logic |
|------|------------|-----------------|
| **FLT3** | AML / FLT3-ITD | ATP-site steric fill + Chemical Sanity |
| **ALK5** (TGFBR1) | Uremia / renal fibrosis | Kinase block + **CYP450 shield** (compromised clearance) |
| **GSK3β** | Retinal stem-cell reprogramming | Modulatory / allosteric chemotypes (not plug-only); DOT1L co-target maps archived |

## Results (R1+R2)

| Gene | Candidates screened | Docked | Best ID | Best Vina |
|------|---------------------|--------|---------|-----------|
| **FLT3** | 40 | 31 | `flt3_r2_00` | **-10.25** |
| **ALK5** | 42 | 39 | `alk5_x00` | **-8.997** |
| **GSK3B** | 44 | 39 | `gsk_x10` | **-7.648** |

## Best SMILES

- **FLT3** (`flt3_r2_00`): `O=C(Nc1ccc(F)cc1)c1c[nH]c2ccc(F)cc12`
- **ALK5** (`alk5_x00`): `O=C(Nc1ccc(F)cc1)c1ccc(N2CCOCC2)nc1`
- **GSK3B** (`gsk_x10`): `Cc1nc(N2CCOCC2)cc(Nc2ccccc2F)n1`

## Structures used (public PDB / EMDB labels)

- **FLT3**: PDB 4RT7 / 6JQR / 5X02 / 4XUF (kinase domain)
- **ALK5**: PDB 3TZM / 5E8S / 1VJY (TGFBR1 kinase); cryo ECD EMD-50519
- **GSK3B**: PDB 1Q3D / 6B8J / 4ACC; DOT1L cryo EMD-9843 / EMD-22692

## Files

- [`WAVE2_PUBLIC_LEDGER.json`](WAVE2_PUBLIC_LEDGER.json) — ranked top candidates
- [`ligands/`](ligands/) — SDF poses for top hits
- Parent pipeline: [`../Translational_Medicine/`](../Translational_Medicine/)

## Method notes

- Max heavy atoms ≤ 35; PAINS / strain gates
- 64³ Boolean pocket occupancy around co-crystal ligand centroid
- Vina: box 24 Å, exhaustiveness 12, seeds 41–44
- ALK5 rejects catechol / nitro / methylenedioxy motifs
