# PyMOL — BCL2 conquest candidate (public showcase)
# Usage: pymol BCL2_view.pml
load ligands/BCL2_bcl2_sulfonylbenzamide.sdf
# Optionally load a public PDB receptor separately (see structure note: PDB 6O0K BCL2)
show sticks, all
util.cbaw
orient
ray 1200,900
png figures/BCL2_ligand.png, dpi=150
