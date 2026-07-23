# PyMOL — EGFR conquest candidate (public showcase)
# Usage: pymol EGFR_view.pml
load ligands/EGFR_egfr_anilinoquinazoline.sdf
# Optionally load a public PDB receptor separately (see structure note: Public kinase pocket receptor (first-principles mass pack))
show sticks, all
util.cbaw
orient
ray 1200,900
png figures/EGFR_ligand.png, dpi=150
