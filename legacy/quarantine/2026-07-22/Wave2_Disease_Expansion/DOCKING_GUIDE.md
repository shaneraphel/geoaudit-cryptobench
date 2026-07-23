# Docking reproduction guide · Hematology M5 leads

> `clinical_grade = false`. AutoDock Vina scores are a **secondary / informational**
> metric. They are **not** measured binding affinity, IC50, or clinical efficacy.
> This guide lets an independent reviewer regenerate the numbers in the ledger.

## 1. Software

- **AutoDock Vina** ≥ 1.2 (Trott & Olson, *J. Comput. Chem.* 2010; Eberhardt et al. 2021)
- **RDKit** (2023.09+) for ligand 3D embedding and property calculation
- `prepare_receptor` (ADFR suite / Meeko) for receptor PDBQT preparation

## 2. Receptor preparation

1. Download the structure from RCSB, e.g. `wget https://files.rcsb.org/download/4RT7.pdb`.
2. Retain a single protein chain; strip waters, ions, and the co-crystallized ligand.
3. Add polar hydrogens at physiological pH (BCL-2 was protonated at pH 7.2).
4. Convert to PDBQT (Gasteiger charges) with `prepare_receptor -r <clean>.pdb -o <rec>.pdbqt`.

The pocket search box is centered on the co-crystal ligand centroid (the ATP site
for the kinases, the BH3 groove for BCL-2).

## 3. Search box per target

| Target | PDB (source) | Box center (x,y,z Å) | Box size (Å) |
|--------|--------------|----------------------|--------------|
| FLT3 | [4RT7](https://www.rcsb.org/structure/4RT7) (+ cross-dock [6JQR](https://www.rcsb.org/structure/6JQR)) | [-40.601, 11.24, -13.828] | [24.0, 24.0, 24.0] |
| BCL2 | [6O0K](https://www.rcsb.org/structure/6O0K) | [-10.291, 2.335, -9.415] | [24.0, 24.0, 24.0] |
| ABL1 | [3OXZ](https://www.rcsb.org/structure/3OXZ) | [12.296, 0.003, 14.215] | [24.0, 24.0, 24.0] |

All boxes are 24 × 24 × 24 Å, matching the 18 Å-radius pocket occupancy tensor
(64³ Boolean grid) used for the geometric clash / hydration-shell pre-filter.

## 4. Ligand preparation

```python
from rdkit import Chem
from rdkit.Chem import AllChem
mol = Chem.AddHs(Chem.MolFromSmiles(SMILES))
AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
# then export to PDBQT (Meeko / OpenBabel)
```

## 5. Vina command (per seed)

```bash
vina --receptor <rec>.pdbqt --ligand <lig>.pdbqt \
     --center_x CX --center_y CY --center_z CZ \
     --size_x 24 --size_y 24 --size_z 24 \
     --exhaustiveness 8 --cpu 1 --seed SEED \
     --out <pose>.pdbqt
```

- **Sprint M5 docking budget:** exhaustiveness 8, seeds 41–43 (best-of-N reported).
- Reported `best` = minimum (most negative) `REMARK VINA RESULT` across seeds; `mean` = average.
- The upstream fragment screens (R3 / R4) used higher exhaustiveness (16–20) and 6–8 seeds.

## 6. Parsing

```bash
grep "REMARK VINA RESULT" <pose>.pdbqt | head -1   # kcal/mol of top pose
```

## 7. Pre-docking gates (applied before Vina)

1. **Chemical Sanity** — heavy atoms ≤ 40 (M5 growth), PAINS filter, MMFF strain ceiling.
2. **64³ Boolean pocket occupancy** — geometric clash check against the pocket wall.
3. **Hydration-shell proxy** — the appended polar Fsp³ tail must present H-bond
   acceptors toward the solvent channel (rigid proxy, not molecular dynamics).

Only candidates passing 1–3 are docked. Full per-candidate results:
[`SPRINT_M5_FRAGMENT_TO_LEAD.json`](SPRINT_M5_FRAGMENT_TO_LEAD.json).

## 8. Suggested wet-lab validation · 建议湿实验验证 (orthogonal, NOT yet performed)

> These are **proposed** assays for an independent lab to test the computational
> hypotheses. **No wet-lab data is included in this repository** — nothing below is a
> measured result. · 以下为**建议**试验，供第三方实验室验证计算假设；本仓库**不含**任何湿实验数据。

| Target | Biochemical | Cellular model | Positive control |
|--------|-------------|----------------|------------------|
| **FLT3** | FLT3 / FLT3-ITD kinase inhibition (ADP-Glo or mobility-shift), dose–response IC50 | MV4-11 (FLT3-ITD⁺) viability | quizartinib / gilteritinib |
| **BCL2** | TR-FRET / fluorescence-polarization BH3-peptide displacement; BH3 profiling | RS4;11 or SU-DHL-4 viability | venetoclax |
| **ABL1** | ABL1 (and T315I) kinase inhibition, dose–response IC50 | K562 (BCR-ABL⁺) viability | ponatinib / imatinib |

General practice · 通用规范:

1. Run a dose–response (≥ 8 points) and report IC50 with 95% CI; DMSO vehicle control.
2. Require assay quality **Z′ ≥ 0.5**; positive control IC50 within the historical range.
3. Confirm target engagement orthogonally (SPR / ITC / thermal shift) before claiming affinity.
4. Counter-screen for aggregation / PAINS-like promiscuity (detergent control).

The synthesis handle is the highlighted solvent tail; the locked aromatic warhead is the
shared pharmacophore across each series, so a small analog matrix around the tail is the
natural first medicinal-chemistry follow-up.
