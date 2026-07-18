# Roadmap — Biologics / Peptidomimetic Modality

**Status:** planning · `clinical_grade = false` · computational only

We are extending the deterministic geometry pipeline from small molecules to
**macrocyclic peptidomimetics** (cyclic peptides, stapled peptides, bicyclic
peptides) as a route to large flat protein-protein interfaces (PPI) that classic
small molecules cannot cover.

## What we have now (this iteration)
- 5 macrocyclic peptidomimetic candidates forged and 3D-embedded
  (`BIOLOGICS_CANDIDATES.json` + SDFs): an RGD integrin-mimetic, a cyclic
  pentapeptoid, a proline-rich macrocycle, a hydrocarbon-stapled peptide, and a
  bicyclic peptide. All PAINS-free; 12–21-membered rings; 4–6 backbone amides.

## Honest limitations (must not overstate)
- **No binding affinity computed.** Rigid-receptor AutoDock Vina is unreliable for
  large flexible macrocycles; we did **not** produce docking scores for these.
- No PD-1/PD-L1 or integrin receptor was docked this iteration.
- These are chemistry/geometry candidates only, not validated binders.

## Upcoming computational tasks (ordered)
1. **Macrocycle conformer sampling** — ensemble generation with macrocycle-aware
   ETKDG + MMFF; report accessible-surface and ring-pucker descriptors.
2. **PPI interface tensor matching** — map a chosen extracellular interface
   (e.g., PD-L1 face from a deposited complex) onto the 64³ occupancy grid; score
   shape complementarity (steric fit + Betti channel overlap), NOT affinity.
3. **Specialized macrocycle docking** — integrate a flexible-macrocycle docking
   engine (or restrained ensemble docking) before any affinity claim.
4. **Developability for peptides** — membrane permeability proxies (PSA, N-methyl
   count), protease-liability flags; label all as heuristic.
5. **Antibody-mimetic / large geometric folding** — long-horizon; requires a
   folding front-end; explicitly out of current scope.

## Guardrails
- No affinity/efficacy claims without proper macrocycle sampling + validation.
- IP air-gap: the internal generative engine stays private; only candidate
  structures + scalar evidence are shareable.
