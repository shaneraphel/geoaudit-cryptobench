# PyMOL :: FLT3 pocket + clinical lead  (clinical_grade=false; docking secondary)
# Usage: pymol FLT3_pocket.pml
fetch 4RT7, async=0
remove solvent
hide everything
show cartoon, polymer
color grey80, polymer
pseudoatom pocket_center, pos=[-40.601, 11.240, -13.828]
show spheres, pocket_center
color orange, pocket_center
set sphere_scale, 0.6, pocket_center
load ../Targets/Oncology/FLT3/flt3_m5_11.sdf, lead
show sticks, lead
util.cbag lead
zoom lead, 8
bg_color white
set ray_opaque_background, 0
