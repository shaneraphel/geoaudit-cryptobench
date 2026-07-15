# Target selection rationale (DepMap Boolean × TCGA)

1. **ABL1 (leukemia)** — PASS via **indication-lineage kill-switch** (K562 / KU812 / LAMA-84 / EM-2 / MEG-01 Chronos medians ≪ −0.5). Pan-cancer ABL1 is not a common essential; engineering is lineage-scoped.
2. **KRAS** — PASS via selective Chronos lethality (fraction score < −1 and p10 < −1).
3. **PIK3CA** — evaluated under gate; TCGA H1047×PTEN dual-hit defines TME manifold when program PASS.
4. Candidates enter `VALIDATED_CANDIDATE_POOL` only after DepMap PASS.

All selections remain **hypothesis-generating**; Chronos ≠ chemotype IC50.
