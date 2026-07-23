# BCL2 · bcl2_m5_12 — molecule spec sheet

> `clinical_grade = false` · Computational topology. Docking is a **secondary**
> metric only; values are **not** measured affinity, IC50, or clinical efficacy.

**Target:** B-cell lymphoma 2 (apoptosis regulator) (BCL2, UniProt [P10415](https://www.uniprot.org/uniprotkb/P10415/entry))  
**Indication:** B-cell lymphoma / CLL (BH3 groove)  
**Design:** aromatic warhead locked; high-Fsp³ polar solvent-channel tail appended (Sprint M5).

![BCL2 warhead to lead](../images/BCL2_fragment_to_lead.png)

## Identity

| Field | Warhead `bcl2_r3_29` | Lead `bcl2_m5_12` |
|-------|----------------------|-------------------------------|
| SMILES | `O=C(NS(=O)(=O)c1ccc2ccccc2c1)c1ccccc1` | `CN1CCN(CCOc2ccc3cc(S(=O)(=O)NC(=O)c4ccccc4)ccc3c2)CC1` |
| Formula | **C17H13NO3S** | **C24H27N3O4S** |
| InChIKey | `KAHWLPGWHNTGGJ-UHFFFAOYSA-N` | `UOVDAIIFOQHALZ-UHFFFAOYSA-N` |
| MW (g/mol) | 311.36 | 453.56 |

## Physicochemical panel

| Property | Warhead | Lead | Δ (lead − warhead) |
|----------|---------|------|--------------------|
| Heavy atoms | 22 | 32 | +10 |
| **Fsp³** | 0.0 | **0.2917** | **+0.2917** |
| cLogP | 2.959 | 2.585 | -0.374 |
| TPSA (Å²) | 63.24 | 78.95 | +15.71 |
| H-bond donors | 1 | 1 | +0 |
| H-bond acceptors | 3 | 6 | +3 |
| Rotatable bonds | 3 | 7 | +4 |
| Aromatic rings | 3 | 3 | +0 |
| QED | 0.8086 | 0.5925 | -0.2161 |

## Drug-likeness rule compliance (lead)

| Rule | Result |
|------|--------|
| Lipinski Ro5 (MW≤500, cLogP≤5, HBD≤5, HBA≤10) | **PASS** |
| Veber (RotB≤10, TPSA≤140) | **PASS** |

## Fragment → lead rationale

- **Locked warhead:** the aromatic core (bcl2_r3_29) is frozen as the binding anchor.
- **Grown tail:** `naphthyl_6_ethoxy_MePip` — a polar, high-Fsp³ solubilizing group
  directed toward the solvent-exposed channel (highlighted green in the figure).
- **Effect:** Fsp³ **0.0 → 0.2917** and TPSA **63.24 → 78.95 Å²**,
  improving predicted solubility while keeping cLogP in range (2.959 → 2.585).

## Structure & docking box

| Field | Value |
|-------|-------|
| Primary PDB | [6O0K](https://www.rcsb.org/structure/6O0K) — BCL-2 in complex with a BH3-mimetic (venetoclax-class groove) |

| Box center (x,y,z Å) | [-10.291, 2.335, -9.415] |
| Box size (Å) | [24.0, 24.0, 24.0] |
| Clinical reference | Venetoclax (approved BCL-2 BH3-mimetic) |

Secondary docking (AutoDock Vina, informational): best **-10.14** kcal/mol
(mean -10.1). See [`../DOCKING_GUIDE.md`](../DOCKING_GUIDE.md) to reproduce.
