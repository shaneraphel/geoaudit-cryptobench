# Boolean Retrosynthesis — FLT3 cubane_carbox (clinical_grade=false)

Target: O=C(NC12CC3(CC1C1CC13)C2)c1ccc(F)cc1N1CCOCC1

## Retro tree (leaves commercial)
- G1 Amide coupling (EDC/HOBt or acyl chloride)
  - A1 4-fluoro-2-morpholinobenzoic acid
    - G2 SNAr: 2,4-difluorobenzoic acid [COMMERCIAL] + morpholine [COMMERCIAL]
  - A2 cubane-1-amine
    - G3 Curtius from cubane-1-carboxylic acid (<- cubane-1,4-dicarboxylic acid [SPECIALTY, catalogued])

## Gates (each a robust, verifiable reaction)
| Gate | Reaction | Robustness |
|------|----------|-----------|
| G1 | Amide coupling | very high |
| G2 | SNAr F-displacement (activated ortho to COOH) | high |
| G3 | Curtius (acid->Boc-amine->deprotect) | high |

## Honest note
Only hard node = cubane-1-amine (Curtius from commercial cubane dicarboxylic acid).
That is exactly why empirical SA=5.23. With cubane block in hand: 3 robust steps.
Route PROPOSAL, not executed synthesis; wet validation required.
