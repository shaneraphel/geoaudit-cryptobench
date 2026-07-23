# Unified Clinical Ledger · Clinical Readiness Index

Generated: `2026-07-16T10:33:47.079191+00:00`

> **clinical_grade = false** · Ranked by Clinical Readiness Index — **not** docking scores.

## Program freeze / activate (DepMap veto)

| Program | Gene | DepMap | Pipeline | TME niche |
|---------|------|--------|----------|-----------|
| `leukemia_ABL1` | **ABL1** | **PASS** | **ACTIVE** | `leukemia_BM_niche` |
| `kras_G12C` | **KRAS** | **PASS** | **ACTIVE** | `solid_hypoxic_core` |
| `pik3ca_H1047R` | **PIK3CA** | **PASS** | **ACTIVE** | `breast_ER_pos` |
| `esr1_ER_positive_breast` | **ESR1** | **PASS** | **ACTIVE** | `breast_ER_pos` |

## Candidates ranked by CRI (4 Boolean dims)

| Rank | ID | CRI | DepMap | TME pH | Off-target | Chem sanity | Status |
|------|----|-----|--------|--------|------------|-------------|--------|
| 1 | `serm_biphenyl_amine` | **4/4** | Pass | Pass | Pass | Pass | **ACTIVE** |
| 2 | `serm_stilbene_amine` | **4/4** | Pass | Pass | Pass | Pass | **ACTIVE** |
| 3 | `serm_oht_parent` | **4/4** | Pass | Pass | Pass | Pass | **ACTIVE** |
| 4 | `serm_oht_fluorophenyl` | **4/4** | Pass | Pass | Pass | Pass | **ACTIVE** |
| 5 | `serm_oht_tolyl` | **4/4** | Pass | Pass | Pass | Pass | **ACTIVE** |
| 6 | `serm_oht_bcp` | **4/4** | Pass | Pass | Pass | Pass | **ACTIVE** |
| 7 | `serm_oht_azaspiro` | **4/4** | Pass | Pass | Pass | Pass | **ACTIVE** |
| 8 | `serm_oht_piperidine` | **3/4** | Pass | Pass | Pass | Fail | **FROZEN** |
| 9 | `CRYOML-LEUK-pona_azaspiro_oxetane_distal-7840c84c41` | **3/4** | Pass | Pass | Pass | Fail | **FROZEN** |
| 10 | `CRYOML-LEUK-pona_norbornyl_spiro-e8c146ed2e` | **3/4** | Pass | Pass | Pass | Fail | **FROZEN** |

## ACTIVE VALIDATED_CANDIDATE_POOL (n=7)

- `serm_biphenyl_amine`
- `serm_stilbene_amine`
- `serm_oht_parent`
- `serm_oht_fluorophenyl`
- `serm_oht_tolyl`
- `serm_oht_bcp`
- `serm_oht_azaspiro`

## Truth boundary

Clinical Readiness Index is a computational Boolean gate stack (DepMap Chronos × TME pH × chemical sanity × off-target shields). It is not wet IC50, PK, QT, or regulatory efficacy. clinical_grade=false.

