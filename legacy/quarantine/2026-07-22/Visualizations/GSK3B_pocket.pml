# PyMOL :: GSK3B pocket + clinical lead  (clinical_grade=false; docking secondary)
# Usage: pymol GSK3B_pocket.pml
fetch 1Q3D, async=0
remove solvent
hide everything
show cartoon, polymer
color grey80, polymer
pseudoatom pocket_center, pos=[30.497, -5.134, 21.762]
show spheres, pocket_center
color orange, pocket_center
set sphere_scale, 0.6, pocket_center
load ../Targets/Regeneration/GSK3B/gsk_x10.sdf, lead
show sticks, lead
util.cbag lead
zoom lead, 8
bg_color white
set ray_opaque_background, 0
