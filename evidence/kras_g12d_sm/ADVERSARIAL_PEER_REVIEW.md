# Adversarial Peer Review — TOP-3 KRAS G12D Candidates
# 对抗性同行评议 — Top-3 KRAS G12D 候选

**clinical_grade=false** · Blue = Author defense · Red = Lethal reviewer  
**Rule:** Red must land ≥2 fatal metabolic/biological/synthetic critiques with causal chemistry. Author may attempt **one** concrete structural repair per candidate (RDKit-validated).

---

## Bilingual verdict summary / 双语裁决总表

| Rank | ID | Verdict / 裁决 | Repair SMILES (if any) | InChIKey (repair) |
|------|-----|----------------|------------------------|-------------------|
| 1 | 00379 | **SURVIVED_WITH_MODIFICATION** / **经修改存活** | `Nc1cc2ccccc2c(COC3CN(C)CC(F)C3)n1` | `FEMGRWXZTHITSN-UHFFFAOYSA-N` |
| 2 | 02621 | **ASSASSINATED** / **否决** | *(N-acetyl attempted; Red rejects)* `CN1CCCC1OCc1ccc(F)c2sc(NC(C)=O)c(C#N)c12` | — |
| 3 | 00553 | **SURVIVED_WITH_MODIFICATION** / **经修改存活** | `Cc1cccc2cc(O)cc(OCCN3CCOCC3)c12` | `BBPVRZAZOXDXGZ-UHFFFAOYSA-N` |

None scored **SURVIVED_CLEAN**. One candidate (**02621**) is **ASSASSINATED**.

---

## Candidate Rank 1 — 00379 (`LSDREBDFWDRPJZ-CXAGYDPISA-N`)

**Parent SMILES:** `Nc1cc2ccccc2c(COC23CCCN2CC(F)C3)n1`

### Red — Lethal Review (≥2 fatals)

**Fatal 1 — Metabolic / chemical (aminal).**  
The bridgehead carbon in `COC23CCCN2…` is bonded to **both oxygen and nitrogen** (N,O-aminal). Under hepatic lysosomal / TME-like acidity this bond hydrolyzes to an **aldehyde + amino-alcohol**, destroying the ligand before sustained KRAS G12D engagement. This is a **PK death sentence**, not a soft flag ([reactive metabolite / aminal instability](https://pubmed.ncbi.nlm.nih.gov/17722901/)).

**Fatal 2 — Bioactivation (aminoisoquinoline).**  
The **3-aminoisoquinoline** is an aromatic aminoheteroarene: CYP N-oxidation → conjugated N–O species → **nitrenium** → covalent binding (hepatotoxicity / genotoxicity hypothesis) ([Kalgutkar et al.](https://pubmed.ncbi.nlm.nih.gov/15606127/)). QED cannot override this.

**Fatal 3 (synthetic, optional).**  
Stereodefined fluorinated azabicyclic aminal ethers are low-yielding and epimerizable at the aminal carbon — scale-up risk for wet-lab triage.

### Blue — One repair

**Repair intent:** Remove N,O-aminal by converting the azabicycle into an **N-methyl-5-fluoropiperidin-3-yl ether** while keeping the aminoisoquinoline–CH2O hinge hypothesized for 9BL0-class noncovalent occupancy.

**Repaired SMILES:** `Nc1cc2ccccc2c(COC3CN(C)CC(F)C3)n1`  
**RDKit parse:** OK · **InChIKey:** `FEMGRWXZTHITSN-UHFFFAOYSA-N`  
**Descriptors (RDKit):** MW **289.35** · cLogP **2.38** · TPSA **51.38** · QED **0.942** · formula `C16H20FN3O`

**Pharmacophore check:** Aminoisoquinoline + benzylic/heteroaryl-CH2–O–saturated amine retained; F retained on the amine ring; aminal carbon eliminated. Residual nitrenium hypothesis remains but aminal PK fatal is removed.

### Verdict

**SURVIVED_WITH_MODIFICATION**  
New identity: SMILES `Nc1cc2ccccc2c(COC3CN(C)CC(F)C3)n1` · InChIKey `FEMGRWXZTHITSN-UHFFFAOYSA-N`

---

## Candidate Rank 2 — 02621 (`FLRZLMYSWNATQP-GFCCVEGCSA-N`)

**Parent SMILES:** `CN1CCCC1OCc1ccc(F)c2sc(N)c(C#N)c12`

### Red — Lethal Review (≥2 fatals)

**Fatal 1 — Thiophene-S-oxide chain.**  
Electron-rich **2-aminothiophene** undergoes CYP/FMO **S-oxidation** to **thiophene-S-oxide**, a soft electrophile linked to idiosyncratic hepatotoxicity ([Gramec et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24655145/); [Kalgutkar](https://pubmed.ncbi.nlm.nih.gov/16101570/)). The 3-nitrile further polarizes the ring.

**Fatal 2 — Nitrenium from 2-amino.**  
The same motif enables amino → N-oxidation → **nitrenium** (genotoxicity/hepatotoxicity). Dual orthogonal bioactivation on one five-membered ring is unacceptable for a KRAS G12D lead hypothesis.

**Fatal 3 — Synthetic / IP chemotype.**  
2-Amino-3-cyanothiophenes are prolific screening hits with frequent assay interference and metabolic liability — poor developability chemotype even when PAINS SMARTS are silent ([Baell & Holloway context](https://pubmed.ncbi.nlm.nih.gov/20131845/)).

### Blue — One repair (attempted)

**Repair intent:** Cap the 2-amino as acetamide to block nitrenium initiation while preserving benzothiophene shape.

**Repaired SMILES:** `CN1CCCC1OCc1ccc(F)c2sc(NC(C)=O)c(C#N)c12`  
**RDKit parse:** OK · **InChIKey:** `XGZNVIGUQWQXDP-UHFFFAOYSA-N`  
**Descriptors:** MW **347.42** · cLogP **3.44** · TPSA **65.36** · QED **0.920** · `C17H18FN3O2S`

### Red — Rejection of repair

Acetylation may slow nitrenium formation but **does not remove the thiophene S-atom**. S-oxide bioactivation remains intact; MW/cLogP worsen; amide NH introduces new CYP/UGT soft spot. Replacing S with CH/O would abandon the parent chemotype’s electronic map vs KRAS G12D pocket hypotheses. **Repair insufficient.**

### Verdict

**ASSASSINATED**  
**Reason:** Irreducible dual bioactivation of **2-aminothiophene-3-carbonitrile** (S-oxide + nitrenium). Single allowed repair (N-acetyl) fails to extinguish S-oxidation; motif is incompatible with a KRAS G12D computational lead advancing past liability triage. **clinical_grade=false** — assassination is a hypothesis gate, not a regulatory finding.

---

## Candidate Rank 3 — 00553 (`JTQKZCJFKLCOKZ-UHFFFAOYSA-N`)

**Parent SMILES:** `Cc1cccc2cc(O)cc(OC3(N4CCOCC4)CC3)c12`

### Red — Lethal Review (≥2 fatals)

**Fatal 1 — Cyclopropane N,O-aminal.**  
`OC3(N4CCOCC4)CC3` places morpholine N and ether O on the **same cyclopropane carbon**. Acidic TME / gastric / lysosomal conditions release morpholine and a **carbonyl electrophile** — the molecule is a latent aldehyde/ketone generator ([aminal chemistry](https://pubmed.ncbi.nlm.nih.gov/17722901/)). Noncovalent KRAS occupancy cannot outrun hydrolysis.

**Fatal 2 — Naphthol → naphthoquinone.**  
The free naphthalen-2-ol is a CYP-ready phenol; two-electron oxidation yields **Michael-acceptor quinones** with hepatotoxicity precedent ([Bolton et al.](https://pubmed.ncbi.nlm.nih.gov/10725116/)).

### Blue — One repair

**Repair intent:** Replace the cyclopropane aminal with a **morpholinoethyl ether** (no shared N/O carbon), retaining naphthol methyl-naphthalene core for hydrophobic contact hypotheses.

**Repaired SMILES:** `Cc1cccc2cc(O)cc(OCCN3CCOCC3)c12`  
**RDKit parse:** OK · **InChIKey:** `BBPVRZAZOXDXGZ-UHFFFAOYSA-N`  
**Descriptors:** MW **287.36** · cLogP **2.56** · TPSA **41.93** · QED **0.938** · `C17H21NO3`

**Pharmacophore check:** Naphthol + morpholine vector retained; aminal deleted. Quinone hypothesis **remains** as a residual watch-item but is no longer paired with an acid-labile warhead — acceptable for continued computational triage with phenol blocking as a later iteration.

### Verdict

**SURVIVED_WITH_MODIFICATION**  
New identity: SMILES `Cc1cccc2cc(O)cc(OCCN3CCOCC3)c12` · InChIKey `BBPVRZAZOXDXGZ-UHFFFAOYSA-N`

---

## Author notes / 作者备注

- All repairs validated with RDKit 2026.03.3 (`MolFromSmiles` ≠ None; descriptors recomputed).
- DFT metrics for parents and repairs: `DFT_QUANTUM_METRICS.json` (see toxicology report Quantum section).
- Geometry reference for KRAS G12D noncovalent context: [PDB 9BL0](https://www.rcsb.org/structure/9BL0).
- **clinical_grade=false** throughout.
