# PyMOL :: ALK5 pocket + clinical lead  (clinical_grade=false; docking secondary)
# Usage: pymol ALK5_pocket.pml
fetch 3TZM, async=0
remove solvent
hide everything
show cartoon, polymer
color grey80, polymer
pseudoatom pocket_center, pos=[4.527, 8.717, 6.784]
show spheres, pocket_center
color orange, pocket_center
set sphere_scale, 0.6, pocket_center
load ../Targets/Fibrosis/ALK5/alk5_x00.sdf, lead
show sticks, lead
util.cbag lead
zoom lead, 8
bg_color white
set ray_opaque_background, 0
