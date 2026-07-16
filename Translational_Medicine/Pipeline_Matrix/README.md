# Pipeline Matrix · Multi-Target Geometric Chemistry

Generated: `2026-07-16T14:51:47.234305+00:00`

> **`clinical_grade = false`**
> Chemical Sanity (≤35 heavy atoms, PAINS, strain) is the primary filter.
> Local docking affinity is secondary / informational only.
> Computational gates ≠ measured IC50 / clinical efficacy.

## Pipeline Matrix

| Target | Role | Linked primary | Top candidate | SMILES | Chem sanity | Docking (secondary) |
|--------|------|----------------|---------------|--------|-------------|---------------------|
| **EGFR** | `SECONDARY_BYPASS_NODE` | KRAS | `egfr_anilinoquinazoline` | `Fc1ccc(Nc2ncnc3ccccc23)cc1` | Pass | -6.58 |
| **AKT1** | `SECONDARY_BYPASS_NODE` | PIK3CA | `akt_morpholino_pyrimidine` | `Cc1nc(Nc2ccccc2)cc(N3CCOCC3)n1` | Pass | -8.34 |
| **BCL2** | `SECONDARY_BYPASS_NODE` | ABL1 | `bcl2_sulfonylbenzamide` | `O=C(NS(=O)(=O)c1ccc(C)cc1)c1ccccc1` | Pass | -8.06 |
| **ESR1** | `PRIMARY_KILL_SWITCH` | — | `serm_stilbene_amine` | `Oc1ccc(/C=C/c2ccc(OCCN(C)C)cc2)cc1` | Pass | -7.04 |

## Design rules (enforced)

1. **MAX_HEAVY_ATOMS ≤ 35** + PAINS + thermodynamic strain gate
2. **Synthon-only** combination logic (named reactions on commercial building blocks)
3. **64³ Boolean pocket tensors** bound ligand clearance (geodesic / steric shell)
4. **DepMap / Compensatory Atlas** justification for every target

## Ligands

SDF files: [`ligands/`](ligands/)

## Visualization

PyMOL / ChimeraX scripts: [`visualization/`](visualization/)

```bash
pymol visualization/EGFR_view.pml
chimerax visualization/AKT1_view.cxc
```

## Compensatory context

EGFR / AKT1 / BCL2 are combination-therapy deployments against `SECONDARY_BYPASS_NODE` escape routes (see [`../Compensatory_Atlas/`](../Compensatory_Atlas/)). ESR1 is a primary ER+ breast Chronos kill-switch with crystal pocket tensors.

## Refusals / IP air-gap

- No proprietary accelerator RTL or hardware architecture
- No synthesis bitstream dumps
- No internal geometric-engine source disclosure
- Not wet IC50 / PK / PDX / regulatory claims

