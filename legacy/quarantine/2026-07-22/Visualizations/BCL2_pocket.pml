# PyMOL :: BCL2 pocket + clinical lead  (clinical_grade=false; docking secondary)
# Usage: pymol BCL2_pocket.pml
fetch 6O0K, async=0
remove solvent
hide everything
show cartoon, polymer
color grey80, polymer
pseudoatom pocket_center, pos=[-10.291, 2.335, -9.415]
show spheres, pocket_center
color orange, pocket_center
set sphere_scale, 0.6, pocket_center
load ../Targets/Oncology/BCL2/bcl2_m5_12.sdf, lead
show sticks, lead
util.cbag lead
zoom lead, 8
bg_color white
set ray_opaque_background, 0
