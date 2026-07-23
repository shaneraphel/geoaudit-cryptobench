# Target selection rationale (DepMap Boolean × TCGA × Chemical Sanity)

1. **ABL1 (leukemia)** — DepMap PASS via indication-lineage kill-switch (K562 / KU812 / LAMA-84 / EM-2 / MEG-01).
2. **KRAS** — DepMap PASS via selective Chronos lethality.
3. **PIK3CA** — DepMap PASS; TCGA H1047×PTEN dual-hit defines TME context.
4. **ESR1 / ER-100** — DepMap PASS via ER+ breast lineage kill-switch (MCF7 / T47D).
5. Candidates enter ACTIVE `VALIDATED_CANDIDATE_POOL` only after **Clinical Readiness Index = 4/4**
   (DepMap × TME pH × off-target orthogonality × chemical sanity ≤35 heavy atoms).
6. Docking scores are never used for ranking.

All selections remain **hypothesis-generating**; Chronos ≠ chemotype IC50.
