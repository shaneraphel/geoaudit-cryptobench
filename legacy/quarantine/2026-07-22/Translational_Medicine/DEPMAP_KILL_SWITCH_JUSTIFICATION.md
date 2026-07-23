# DepMap Boolean Causality · Kill-Switch Gate

Generated: `2026-07-16T10:33:47.076018+00:00`

> **clinical_grade = false**
> CRISPR Chronos gene-effect scores ≠ measured chemotype IC50.

## Decision rule (absolute Boolean)

- PASS if pan-cancer median < `-0.5`, **or**
- selective lethality (fraction score < −1 ≥ `0.02` and p10 < `-1.0`), **or**
- indication-lineage median < `-0.5`.
- Otherwise **ABORT** the target.

## Results

| Program | Gene | Decision | Pan med | Selective? | Lineage med | Wet cell line |
|---------|------|----------|---------|------------|-------------|---------------|
| `leukemia_ABL1` | **ABL1** | **PASS** | 0.042 | False | -1.8502399407237715 | K562 |
| `kras_G12C` | **KRAS** | **PASS** | -0.497 | True | -1.2710789999573375 | NCI-H358 |
| `pik3ca_H1047R` | **PIK3CA** | **PASS** | -0.454 | False | -1.0469504122001227 | T47D |
| `esr1_ER_positive_breast` | **ESR1** | **PASS** | -0.068 | False | -1.4622495827007858 | MCF7 |

## Lineage evidence (leukemia ABL1)

- `MEG-01` (`ACH-000604`): Chronos **-2.096**
- `KU812` (`ACH-000076`): Chronos **-2.031**
- `LAMA-84` (`ACH-000326`): Chronos **-1.850**
- `K562` (`ACH-000551`): Chronos **-1.626**
- `EM-2` (`ACH-000983`): Chronos **-1.549**

## VALIDATED_CANDIDATE_POOL

- `CRYOML-LEUK-pona_norbornyl_spiro-e8c146ed2e`
- `CRYOML-LEUK-pona_azaspiro_oxetane_distal-7840c84c41`
- `serm_biphenyl_amine`
- `serm_stilbene_amine`
- `serm_oht_parent`
- `serm_oht_tolyl`
- `serm_oht_fluorophenyl`
- `serm_oht_azaspiro`
- `serm_oht_bcp`
