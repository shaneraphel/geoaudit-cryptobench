# PyMOL — ESR1 conquest candidate (public showcase)
# Usage: pymol ESR1_view.pml
load ligands/ESR1_serm_stilbene_amine.sdf
# Optionally load a public PDB receptor separately (see structure note: PDB 3ERT ER-α + existing 64³ pocket wall tensor)
show sticks, all
util.cbaw
orient
ray 1200,900
png figures/ESR1_ligand.png, dpi=150
