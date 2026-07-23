# PyMOL :: EGFR pocket + clinical lead  (clinical_grade=false; docking secondary)
# Usage: pymol EGFR_pocket.pml
fetch 1M17, async=0
remove solvent
hide everything
show cartoon, polymer
color grey80, polymer
pseudoatom pocket_center, pos=[23.271, 9.822, 59.343]
show spheres, pocket_center
color orange, pocket_center
set sphere_scale, 0.6, pocket_center
load ../Targets/Oncology/EGFR/egfr_lead.sdf, lead
show sticks, lead
util.cbag lead
zoom lead, 8
bg_color white
set ray_opaque_background, 0
