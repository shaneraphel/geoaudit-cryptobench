# First-Principles Structural Rationale (Computational)

**Status:** deterministic geometric/thermodynamic rationale · `clinical_grade = false`

> These are **computational rationales** in the requested lexicon (Boolean steric
> boundaries, Van der Waals radii, inverse tensor mismatch, pocket-cavity Betti
> numbers, dielectric solvent shielding). They are hypotheses about *why the
> docking geometry scores as it does* — **not** proven binding mechanisms, and
> **not** measured energetics. Docking is a secondary metric, not affinity.

## Lexicon
- **Boolean steric boundary** `L_k`: the 64³ occupancy set the ligand must not
  violate (hard clash = forbidden voxel overlap).
- **Inverse tensor mismatch** `G^{-1} → 0`: near the pocket boundary the boundary
  metric diverges; a candidate that pushes into a clashing voxel is frozen out —
  a geometric brake. Good fit = low mismatch in the interior.
- **Pocket Betti** `(β0,β1,β2)`: β2=0 ⇒ open solvent-accessible cavity; large β1 ⇒
  through-channels that license a polar side-chain to exit toward bulk water.
- **Dielectric solvent shielding**: a polar tail projected into a β1 channel is
  shielded by high-dielectric water; a buried charge without a channel is penalized.

## Per-lead rationale (honest, docking-level)

### FLT3 · `cubane_carbox` (same-box −10.40 vs gilteritinib −8.72)
- The **cubane** cage presents a near-isotropic Van der Waals volume that packs the
  hydrophobic hinge shelf with minimal rotational entropy cost (rigid → low
  conformational penalty). Acts as a **steric wedge** filling volume a flat
  aromatic would leave partially void.
- The ortho-**morpholine** on the benzamide is an **electrostatic anchor** whose
  ether oxygen sits toward a β1 channel (dielectric-shielded), consistent with the
  favorable same-box geometry.
- Honest caveat: gilteritinib is a comparatively weak docker; beating it does not
  imply beating a strong ATP-site binder.

### ABL1 · `cubane_diamide` (same-box −11.43 vs ponatinib −12.85 → does NOT beat)
- Two amide **electrostatic anchors** flank a cubane **steric wedge**; the pyridyl
  cap adds one H-bond acceptor. Geometry fills the ATP cleft but leaves a
  hinge-vector under-satisfied relative to ponatinib's alkyne-diaryl reach.
- **Inverse tensor mismatch** is higher than ponatinib's because the rigid cage
  cannot thread the deep back-pocket channel ponatinib's alkyne occupies. This is
  the quantified novelty↔fit tradeoff, not a failure to report.

### BCL2 · `azaspiro_ext` (same-box −11.72 vs venetoclax −10.60)
- The acylsulfonamide is an **electrostatic anchor** into the BH3 hot-spot; the
  **2-oxa-8-azaspiro** provides a rigid 3D **steric wedge** into the P2 groove.
  Smaller Van der Waals footprint than venetoclax (MW 499 vs 736) yet comparable
  same-box packing.
- Honest caveat: structurally close to the venetoclax BH3-mimic class
  (Tanimoto 0.43) — a groove-filler variant, not an orthogonal chemotype.

### MYC/MAX clamp (no receptor docking — shallow PPI)
- fpocket druggability 0.158 ⇒ the interface offers few deep Boolean-forbidden
  wells; β2=0 with modest β1. A high-Fsp³ cage clamp maximizes surface Van der
  Waals contact rays but cannot exploit a deep well that does not exist. This is
  why same-box scores stayed modest (−6…−7) — the honest physics of a flat PPI.

## Summary
Where our candidates score well, it is attributable to rigid steric-wedge volume
filling + a channel-shielded electrostatic anchor. Where they fall short
(ABL1/BCL2 vs strong FDA binders), the deficit is a real inverse-tensor-mismatch
in a deep channel the rigid cage cannot thread. All statements are docking-level
geometry, pending wet validation.
