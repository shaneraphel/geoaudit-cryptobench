# PyMOL — AKT1 conquest candidate (public showcase)
# Usage: pymol AKT1_view.pml
load ligands/AKT1_akt_morpholino_pyrimidine.sdf
# Optionally load a public PDB receptor separately (see structure note: PDB 3O96 AKT1 kinase)
show sticks, all
util.cbaw
orient
ray 1200,900
png figures/AKT1_ligand.png, dpi=150
