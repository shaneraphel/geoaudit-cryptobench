# Translational Medicine · Pre-Clinical Asset Pack

Generated: `2026-07-15T04:07:43.944804+00:00`

> **`clinical_grade = false`**
> This pack connects **public clinical databases** (DepMap, TCGA/cBioPortal, ChEMBL) to
> a **Wet-Lab Execution Package** for the computational lead.
> It does **not** claim patient efficacy, measured IC50, or regulatory readiness.

## 1. Why these targets (DepMap + TCGA)

### DepMap CRISPR dependency (gene_dep API)

| Gene | Program | n lines | median dep | strong? |
|------|---------|---------|------------|---------|
| PIK3CA | PI3KA program | — | — | error |
| ESR1 | ER-100 program | — | — | error |
| ABL1 | leukemia BCR-ABL program | — | — | error |
| BCL2 | apoptosis / leukocyte adjacency | — | — | error |
| KRAS | KRAS G12C cryo program | — | — | error |

### TCGA co-mutation rule (cBioPortal)

- Rule: `PIK3CA_H1047R_AND_PTEN_LOSS`
- Boolean: **True**
- Note: TCGA breast/endometrial cohorts enrich PIK3CA hotspot with PTEN pathway loss — TME graph interventions must treat dual-hit as distinct manifold

## 2. Clinical attrition → Boolean toxicity masks

Phase-fail / withdrawn motifs flattened into **deterministic Boolean masks**
(no SoftMax at inference):

| Mask | Banned | Failure class |
|------|--------|---------------|
| `anilide_primary_aniline` | True | reactive_metabolite_hepatotox |
| `nitroaromatic` | True | genotox_risk |
| `unsubstituted_michael` | True | covalent_offtarget |
| `catechol` | True | quinone_tox |
| `flat_polyaryl_basic_amine` | False | herg_qt |

Tumor niche milli-pH priors: `{"breast_ER_pos": 680, "leukemia_BM_niche": 720, "solid_hypoxic_core": 650}`

## 3. Lead candidate (computational)

| Field | Value |
|-------|-------|
| ID | `CRYOML-LEUK-pona_norbornyl_spiro-e8c146ed2e` |
| Indication | leukemia / ABL1 (3OXZ) |
| Local Vina | -13.96 vs ponatinib (-10.04) |
| Beat local FDA docking baseline | **True** |
| DepMap gene link | `ABL1` |

## 4. Wet-Lab Execution Package

See:

- [`wetlab/CRYOML-LEUK-pona_norbornyl_spiro-e8c146ed2e_WETLAB_SOP.md`](wetlab/CRYOML-LEUK-pona_norbornyl_spiro-e8c146ed2e_WETLAB_SOP.md)
- [`wetlab/CRYOML-LEUK-pona_norbornyl_spiro-e8c146ed2e_RETROSYNTHESIS.json`](wetlab/CRYOML-LEUK-pona_norbornyl_spiro-e8c146ed2e_RETROSYNTHESIS.json)

### Assays specified

- Biochem: `KinaseGlo_or_ADPGlo_ABL1`
- Cell: `['K562', 'BaF3_BCR_ABL']`
- Safety: `hERG_patch_clamp`
- In vivo: `['K562_CDX', 'CML_PDX_if_available']`

### Retrosynthesis

RDKit template sketch (AiZynthFinder optional upgrade). Building-block classes and reagents listed in the JSON.

## 5. Refusals

- Not a wet IC50 / PK / PDX result
- Not FDA efficacy
- No proprietary accelerator RTL or collapsed netlists

