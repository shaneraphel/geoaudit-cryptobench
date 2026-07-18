# MYC / MAX — W6 First Clamp Forge

`clinical_grade = false`

## Protocol
- Surface topology from 1NKP → 64³ Boolean steric wall + clamp shell
- Bounded combinatorial enumeration (macrocycle / spiro / bridged)
- Gates: Fsp³ ≥ 0.55, aromatic rings = 0, SA < 4.5, Murcko Tanimoto ≤ 0.4
- MMFF embed + strain audit; steric clash against surface tensor rejected

## First-in-class SMILES
- `myc_spiro_ketal`: `O=C(C1CC2(CCC2)C1)N1CCC2(CC1)OCCO2` (SA 3.44, Fsp³ 0.933, Vina -7.868)
- `myc_azetidine_acetyl__bicyclo222_1yl`: `O=C(CN1CCC1)NC12CCC(CC1)CC2` (SA 3.38, Fsp³ 0.923, Vina -6.307)
- `myc_azetidine_acetyl__spiro33hept_2yl`: `O=C(CN1CCC1)NC1CC2(CCC2)C1` (SA 2.71, Fsp³ 0.917, Vina -6.546)
- `myc_azetidine_acetyl__norbornan_2yl`: `O=C(CN1CCC1)NC1CC2CCC1C2` (SA 3.73, Fsp³ 0.917, Vina -6.231)
- `myc_morpholine_acetyl__oxaspiro_amine`: `O=C(CN1CCOCC1)NC1CC2(CCOCC2)C1` (SA 2.93, Fsp³ 0.929, Vina -5.898)

## Boundaries
- ADMET / toxicity not wet-proven
- No FTO claim
- Vina is secondary pose-fit on a shallow PPI — modest scores are honest, not failure of reporting
