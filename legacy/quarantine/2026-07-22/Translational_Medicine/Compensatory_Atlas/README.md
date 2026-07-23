# Compensatory Network Atlas · Polypharmacology Escape Routes

Generated: `2026-07-16T13:45:36.160187+00:00`

> **`clinical_grade = false`**
> Targets that **fail** the DepMap Boolean kill-switch are **not deleted**.
> They are re-indexed as `SECONDARY_BYPASS_NODE` — dormant assets for combination therapy.

## Policy

- Primary clinical leads = lineage-selective Chronos kill-switches (`PRIMARY_KILL_SWITCH`).
- Non-lethal / bypassable genes → `SECONDARY_BYPASS_NODE` with **generation frozen**.
- Edge type: `ESCAPE_ROUTE` (primary block → compensatory survival path).

## First three secondary targets linked to primary leads

| # | Primary (kill-switch) | Secondary (bypass) | Rule |
|---|----------------------|--------------------|------|
| 1 | **KRAS** | **EGFR** | `KRAS_BLOCK_EGFR_BYPASS` |
| 2 | **PIK3CA** | **AKT1** | `PIK3CA_BLOCK_AKT1_BYPASS` |
| 3 | **ABL1** | **BCL2** | `ABL1_BLOCK_BCL2_APOPTOSIS_BYPASS` |

### Rationale

**1. KRAS → EGFR**

KRAS-mutant lineages rewire through EGFR/RTK feedback after KRAS blockade; EGFR fails pan Chronos kill-switch → SECONDARY_BYPASS_NODE.

**2. PIK3CA → AKT1**

PI3Kα blockade escaped via AKT1 parallel signaling; TCGA PIK3CA_H1047R_AND_PTEN_LOSS sharpens this context.

**3. ABL1 → BCL2**

BCR-ABL inhibition selects for BCL2 apoptotic buffering in hematopoietic niches; BCL2 is not an ABL1-class Boolean kill-switch → ESCAPE_ROUTE.

## Full ESCAPE_ROUTE table

| Primary | Bypass | Bypass pan Chronos med | Fails pan kill-switch? | TCGA |
|---------|--------|------------------------|------------------------|------|
| **KRAS** | **EGFR** | -0.24189831971788384 | True | `—` |
| **PIK3CA** | **AKT1** | -0.03672761676032363 | True | `PIK3CA_H1047R_AND_PTEN_LOSS` |
| **ABL1** | **BCL2** | -0.013384857400670074 | True | `—` |
| **ESR1** | **EGFR** | -0.24189831971788384 | True | `—` |
| **ESR1** | **ERBB2** | -0.2694361475547698 | True | `—` |
| **KRAS** | **BRAF** | -0.058930504803426306 | True | `—` |
| **KRAS** | **MET** | -0.09146268334214958 | True | `—` |
| **ABL1** | **SRC** | -0.18473198105951824 | True | `—` |
| **PIK3CA** | **EGFR** | -0.24189831971788384 | True | `PIK3CA_H1047R_AND_PTEN_LOSS` |

## Secondary bypass nodes (generation frozen)

| Gene | Role | Pipeline | Pan median |
|------|------|----------|------------|
| **AKT1** | `SECONDARY_BYPASS_NODE` | `FROZEN_ATLAS` | -0.03672761676032363 |
| **BCL2** | `SECONDARY_BYPASS_NODE` | `FROZEN_ATLAS` | -0.013384857400670074 |
| **BRAF** | `SECONDARY_BYPASS_NODE` | `FROZEN_ATLAS` | -0.058930504803426306 |
| **EGFR** | `SECONDARY_BYPASS_NODE` | `FROZEN_ATLAS` | -0.24189831971788384 |
| **ERBB2** | `SECONDARY_BYPASS_NODE` | `FROZEN_ATLAS` | -0.2694361475547698 |
| **FGFR1** | `SECONDARY_BYPASS_NODE` | `FROZEN_ATLAS` | 0.01778708674470114 |
| **LYN** | `SECONDARY_BYPASS_NODE` | `FROZEN_ATLAS` | 0.03782982429222507 |
| **MAPK1** | `SECONDARY_BYPASS_NODE` | `FROZEN_ATLAS` | -0.14184942235857542 |
| **MET** | `SECONDARY_BYPASS_NODE` | `FROZEN_ATLAS` | -0.09146268334214958 |
| **RAF1** | `SECONDARY_BYPASS_NODE` | `FROZEN_ATLAS` | -0.15839538755121257 |
| **SRC** | `SECONDARY_BYPASS_NODE` | `FROZEN_ATLAS` | -0.18473198105951824 |

## Resistance narrative (partner-facing)

When a tumor develops resistance to a primary drug against a kill-switch target, this atlas lists the pre-linked compensatory nodes. Combination / polypharmacology campaigns can then prioritize geometric inhibitors against those `ESCAPE_ROUTE` genes without re-running blind target discovery.

## Truth boundary

ESCAPE_ROUTE links are deterministic oncology + Chronos co-dependency priors for combination-therapy planning. They are not measured clinical resistance outcomes or wet IC50. clinical_grade=false.

- Not a measured clinical resistance cohort outcome
- Not wet IC50 / PK for the combination
- No proprietary accelerator RTL or hardware architecture disclosure

