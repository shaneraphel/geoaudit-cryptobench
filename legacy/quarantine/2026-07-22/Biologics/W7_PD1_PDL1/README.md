# W7 Biologics — PD-1 / PD-L1 Macrocycle Forge

`clinical_grade = false`

## Target
- PDB **4ZQK** (PD-1/PD-L1 extracellular interface)
- Extended planar 64³ Boolean surface tensor (`half_box = 30 Å`)

## Top candidates (true macrocycles, rigid unbound ensemble)
- `pd1_oxetane_peptide`: `N=C(N)NCCC[C@@H]1NC(=O)[C@H](CC(=O)O)NC(=O)[C@@H](C2COC2)NC(=O)[C@H](Cc2ccccc2)NC(=O)C(=O)CNC1=O`
  - ring=16 MW=616.6 ΔE_span=13.3 kcal (rigid=True)
- `pd1_macro_aib_staple`: `CC1(C)NC(=O)[C@H](Cc2ccccc2)NC(=O)CNC(=O)[C@H](CC(=O)O)NC(=O)[C@@H](CCCNC(=N)N)NC(=O)CCCCC=CCCCCCNC1=O`
  - ring=28 MW=755.9 ΔE_span=15.77 kcal (rigid=True)
- `pd1_macro_aib_pentapeptoid`: `CN1CC(=O)N[C@@H](Cc2ccccc2)C(=O)N[C@H](C(C)(C)C)C(=O)N[C@@H](CC(=O)O)C(=O)N[C@@H](CO)C(=O)NC1=O`
  - ring=17 MW=576.6 ΔE_span=17.59 kcal (rigid=True)

## Boundaries
- Small-molecule Vina suspended for this module
- ΔE is unbound torsional ensemble rigidity, not wet Kd / affinity
- No FTO claim
