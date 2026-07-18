# PyMOL :: AKT1 pocket + clinical lead  (clinical_grade=false; docking secondary)
# Usage: pymol AKT1_pocket.pml
fetch 3O96, async=0
remove solvent
hide everything
show cartoon, polymer
color grey80, polymer
pseudoatom pocket_center, pos=[8.373, -6.828, 12.622]
show spheres, pocket_center
color orange, pocket_center
set sphere_scale, 0.6, pocket_center
load ../Targets/Oncology/AKT1/akt1_lead.sdf, lead
show sticks, lead
util.cbag lead
zoom lead, 8
bg_color white
set ray_opaque_background, 0
