# Translational Medicine · Pre-Clinical Asset Pack

Generated: `2026-07-15T04:38:54.343037+00:00`

> **`clinical_grade = false`**
> DepMap Chronos kill-switch gate → TCGA TME constraints → Wet-Lab Execution Package.
> Computational docking ≠ measured IC50. No regulatory claim.

## 1. DepMap Boolean causality (kill-switch gate)

See [`DEPMAP_KILL_SWITCH_JUSTIFICATION.md`](DEPMAP_KILL_SWITCH_JUSTIFICATION.md).

| Program | Gene | Decision | Pan median | Selective? | Lineage median | Wet line |
|---------|------|----------|------------|------------|----------------|----------|
| `leukemia_ABL1` | **ABL1** | **PASS** | 0.04200215621875679 | False | -1.8502399407237715 | K562 |
| `kras_G12C` | **KRAS** | **PASS** | -0.49668749740728335 | True | -1.2710789999573375 | NCI-H358 |
| `pik3ca_H1047R` | **PIK3CA** | **PASS** | -0.4539621313459162 | False | -1.0469504122001227 | T47D |

## 2. TCGA tumor microenvironment (after causality PASS)

- Co-mutation rule: `PIK3CA_H1047R_AND_PTEN_LOSS` · Boolean **True**
- H1047-like samples: **186** · H1047∩PTEN: **36**
- Dictionary milli-pH priors: `{"breast_ER_pos": 680, "leukemia_BM_niche": 720, "solid_hypoxic_core": 650}`

## 3. Clinical attrition → Boolean toxicity masks

| Mask | Banned | Failure class |
|------|--------|---------------|
| `anilide_primary_aniline` | True | reactive_metabolite_hepatotox |
| `nitroaromatic` | True | genotox_risk |
| `unsubstituted_michael` | True | covalent_offtarget |
| `catechol` | True | quinone_tox |
| `flat_polyaryl_basic_amine` | False | herg_qt |

## 4. VALIDATED_CANDIDATE_POOL (top orphans)

- `CRYOML-LEUK-pona_norbornyl_spiro-e8c146ed2e`
- `CRYOML-LEUK-pona_azaspiro_oxetane_distal-7840c84c41`

## 5. Wet-Lab Execution Packages

### `CRYOML-LEUK-pona_norbornyl_spiro-e8c146ed2e`

- Docking: -13.96 vs ponatinib (-10.04)
- Biochem: `KinaseGlo_or_ADPGlo_ABL1`
- DepMap cell line: **K562**
- SOP: [`wetlab/CRYOML-LEUK-pona_norbornyl_spiro-e8c146ed2e_WETLAB_SOP.md`](wetlab/CRYOML-LEUK-pona_norbornyl_spiro-e8c146ed2e_WETLAB_SOP.md)
- Retro: [`wetlab/CRYOML-LEUK-pona_norbornyl_spiro-e8c146ed2e_RETROSYNTHESIS.json`](wetlab/CRYOML-LEUK-pona_norbornyl_spiro-e8c146ed2e_RETROSYNTHESIS.json)

### `CRYOML-LEUK-pona_azaspiro_oxetane_distal-7840c84c41`

- Docking: -13.24 vs ponatinib (-10.04)
- Biochem: `KinaseGlo_or_ADPGlo_ABL1`
- DepMap cell line: **K562**
- SOP: [`wetlab/CRYOML-LEUK-pona_azaspiro_oxetane_distal-7840c84c41_WETLAB_SOP.md`](wetlab/CRYOML-LEUK-pona_azaspiro_oxetane_distal-7840c84c41_WETLAB_SOP.md)
- Retro: [`wetlab/CRYOML-LEUK-pona_azaspiro_oxetane_distal-7840c84c41_RETROSYNTHESIS.json`](wetlab/CRYOML-LEUK-pona_azaspiro_oxetane_distal-7840c84c41_RETROSYNTHESIS.json)

## 6. Refusals / IP air-gap

- Not a wet IC50 / PK / PDX result
- Not FDA efficacy
- No proprietary accelerator RTL, netlists, or hardware architecture disclosure

