# AAAI Zero-Trust Public Evidence Architecture

## 0. Manuscript methodology anchor

### O(1) Deterministic GF(4) Tensor Mapping

The manuscript label is implemented as one NumPy broadcast over a locked
`100×48×2` tensor. There is no alignment, stochastic search, model sampling,
or generative inference. Hashes establish exact reproduction for this fixed
matrix; they do not prove asymptotic O(1) behavior for arbitrary input size.

### Novelty and synthesizability versus affinity-oriented scores

The FLT3 record makes the tradeoff explicit:

- `flt3_cubane_carbox`: secondary score `−10.404`, SA `5.23`.
- `flt3_iso_norbornyl`: secondary score `−9.048`, SA `3.85`.
- Transparent yield: `1.356` score units to cross the `SA < 4` boundary.

The score remains secondary and is not experimental affinity.

### Dependency linkage

Target labels encode compensatory-bypass hypotheses. EGFR, for example, is
tracked as a possible bypass route under KRAS blockade. Public portal URLs are
preserved, but causality remains false without a release-pinned dataset,
model identifiers, score columns, and statistical analysis.

## 1. Air-gapped repository

```text
aaai_public_evidence/
├── README.md
├── src/
│   ├── w12_batch_algebraic_forge.py
│   ├── audit_forge.py
│   ├── novelty_scan.py
│   └── build_report.py
├── data/
│   ├── ncbi_refseq_accessions.json
│   ├── candidates.json
│   ├── candidate_matrix.json
│   ├── methodology_evidence.json
│   ├── references.json
│   └── provenance.json
├── sources/
│   └── SOURCE_URLS.json
├── logs/
│   ├── exact_form_null.json
│   ├── audit_details.json
│   ├── novelty_scan.json
│   ├── LEAK_AUDIT.json
│   └── AAAI_AUDIT_STAMP.log
└── PUBLIC_AUDIT_ARCHITECTURE.md
```

### Rationale

1. `src/` contains executable verification only; reviewers can inspect every
   transformation and rerun it locally.
2. `data/` separates immutable public inputs, source URLs, candidate records,
   and bounded comparison references from generated output.
3. `sources/` provides a dedicated, machine-readable provenance index.
4. `logs/` contains hashes, descriptor-level evidence, thresholds, failures,
   and the exact terminal stamp. Claims can therefore be traced to both code
   and input records.
5. The source audit scans the complete public tree for private identifiers,
   credentials, and absolute workstation paths.
6. `clinical_grade=false` is immutable. Structural similarity is reported as
   a bounded screen, never as legal clearance.

## 2. Core autonomous verification

Full executable: `src/audit_forge.py`

```python
mol = Chem.MolFromSmiles(candidate["smiles"])
formula = rdMolDescriptors.CalcMolFormula(mol)
exact_mass = Descriptors.ExactMolWt(mol)
fsp3 = rdMolDescriptors.CalcFractionCSP3(mol)
hac = mol.GetNumHeavyAtoms()
sa_score = sascorer.calculateScore(mol)

scaffold = MurckoScaffold.GetScaffoldForMol(mol)
fp = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048
).GetFingerprint(scaffold)
max_similarity = max(
    DataStructs.TanimotoSimilarity(fp, reference_fp)
    for reference_fp in reference_fingerprints
)

sa_pass = sa_score < 4.0
similarity_screen_pass = max_similarity < 0.40
ip_conflict = max_similarity >= 0.40
pains_pass = len(pains_catalog.GetMatches(mol)) == 0
```

Verification semantics:

- Exact mass, formula, Fsp3, HAC, PAINS, and SA are recomputed from SMILES.
- Rounded source MW/HAC/Fsp3 values are compared with recomputed descriptors.
- Source artifacts do not declare formulas, so independent formula matching is
  explicitly `not_testable`.
- The Murcko threshold is a finite-reference heuristic. It cannot establish
  patent clearance or freedom to operate.
- Runtime is bounded for the fixed matrix but is not asymptotically constant.

## 3. Live deterministic terminal output

```text
AAAI_AUDIT_BEGIN
ENV python=3.12.8 rdkit=2026.03.3 numpy=1.26.4
COMPILE gf4_solver=pass rdkit_auditor=pass
METHODOLOGY_SEQUENCE=O1_deterministic_GF4_fixed_100x48_broadcast probabilistic_generation=false
METHODOLOGY_CHEMISTRY=synthesis_first affinity_oriented_scores=secondary SA_limit=4.0
METHODOLOGY_DEPENDENCY=compensatory_bypass_hypothesis causality_claimed=false
GF4_BATCH shape=100x48 residual_zero=true path_sha256=ea672670f153560efa783ab7833bf7b17c3f4f5b09bb990ba97b854ea7cbec1d target_sha256=f8ce4961fa5054660dcb0d067ee404aa73559d6b786a10cf9572554d48d7246d
TARGET=EGFR CANDIDATE=egfr_anilinoquinazoline FORMULA=C14H10FN3 EXACT_MASS=239.08587554 FSP3=0.0000 HAC=18 SA=1.6710 SA_LT_4=true MURCKO_MAX_TANIMOTO=0.3125 MAX_REF=osimertinib SIMILARITY_LT_0_40=true IP_CONFLICT=false SOURCE_DESCRIPTOR_MATCH=true FORMULA_SOURCE_MATCH=not_testable PAINS=0 GF4_NULL=true REFSEQ=NM_005228.5 NCBI_URL=https://www.ncbi.nlm.nih.gov/nuccore/NM_005228.5 DEPMAP=PORTAL_LINK_ONLY_NO_RELEASE_PIN DEPMAP_URL=https://depmap.org/portal/gene/EGFR?tab=overview IP_CLEARANCE=not_established
TARGET=AKT1 CANDIDATE=akt_morpholino_pyrimidine FORMULA=C15H18N4O EXACT_MASS=270.14806120 FSP3=0.3333 HAC=20 SA=2.0233 SA_LT_4=true MURCKO_MAX_TANIMOTO=0.2206 MAX_REF=capivasertib SIMILARITY_LT_0_40=true IP_CONFLICT=false SOURCE_DESCRIPTOR_MATCH=true FORMULA_SOURCE_MATCH=not_testable PAINS=0 GF4_NULL=true REFSEQ=NM_001382430.1 NCBI_URL=https://www.ncbi.nlm.nih.gov/nuccore/NM_001382430.1 DEPMAP=PORTAL_LINK_ONLY_NO_RELEASE_PIN DEPMAP_URL=https://depmap.org/portal/gene/AKT1?tab=overview IP_CLEARANCE=not_established
TARGET=FLT3 CANDIDATE=flt3_iso_norbornyl FORMULA=C18H23FN2O2 EXACT_MASS=318.17435620 FSP3=0.6111 HAC=23 SA=3.8540 SA_LT_4=true MURCKO_MAX_TANIMOTO=0.1772 MAX_REF=gilteritinib SIMILARITY_LT_0_40=true IP_CONFLICT=false SOURCE_DESCRIPTOR_MATCH=true FORMULA_SOURCE_MATCH=not_testable PAINS=0 GF4_NULL=true REFSEQ=NM_004119.3 NCBI_URL=https://www.ncbi.nlm.nih.gov/nuccore/NM_004119.3 DEPMAP=PORTAL_LINK_ONLY_NO_RELEASE_PIN DEPMAP_URL=https://depmap.org/portal/gene/FLT3?tab=overview IP_CLEARANCE=not_established
TARGET=BCL2 CANDIDATE=bcl2_sulfonylbenzamide FORMULA=C14H13NO3S EXACT_MASS=275.06161428 FSP3=0.0714 HAC=19 SA=1.5202 SA_LT_4=true MURCKO_MAX_TANIMOTO=0.3158 MAX_REF=21Q SIMILARITY_LT_0_40=true IP_CONFLICT=false SOURCE_DESCRIPTOR_MATCH=true FORMULA_SOURCE_MATCH=not_testable PAINS=0 GF4_NULL=true REFSEQ=NM_000633.3 NCBI_URL=https://www.ncbi.nlm.nih.gov/nuccore/NM_000633.3 DEPMAP=PORTAL_LINK_ONLY_NO_RELEASE_PIN DEPMAP_URL=https://depmap.org/portal/gene/BCL2?tab=overview IP_CLEARANCE=not_established
LEAK_AUDIT=pass findings=0 files_scanned=18
COMPUTATIONAL_GATE=pass
COMPLEXITY_CLAIM=bounded_fixed_batch_not_asymptotic_O1
AAAI_CLAIM_GATE=fail unsupported=ip_clearance,depmap_causality,formula_source_match,O1_asymptotics
CLINICAL_GRADE=false AFFINITY_CLAIMED=false
AAAI_AUDIT_END
```

## 4. Heuristic novelty expansion

The bundled matrix contains **233** valid parsed candidates;
**27** satisfy all bounded heuristic thresholds:
`Fsp3 > 0.40`, `SA < 3.0`, `HAC ∈ [12,45]`, zero PAINS alerts, and maximum
bundled-reference Murcko Tanimoto `< 0.40`.

### 1. `myc_morpholine_acetyl__oxaspiro_amine` (MYC_MAX)

- SMILES: `O=C(CN1CCOCC1)NC1CC2(CCOCC2)C1`
- Formula / exact mass: `C14H24N2O3` / `268.17869263`
- Fsp3 / SA / HAC: `0.9286` / `2.9280` / `19`
- Maximum bounded-reference Murcko Tanimoto: `0.1370` vs `gilteritinib`
- Source: `validation/W6_MYC_CLAMP_LEDGER.json` at `first_in_class_candidates[4]`
- Dependency mapping: `MYC` / https://depmap.org/portal/gene/MYC?tab=overview / causality `unclaimed`
- Interpretation: A saturated oxa-spiro center and morpholine ring provide several non-coplanar vectors with no aromatic-ring dependence. The observation is restricted to the bundled reference set and does not establish patentability, non-infringement, synthesis, or activity.

### 2. `flt3_iso_spiro33hept` (FLT3_iso)

- SMILES: `O=C(NC1CC2(CCC2)C1)c1ccc(F)cc1N1CCOCC1`
- Formula / exact mass: `C18H23FN2O2` / `318.17435620`
- Fsp3 / SA / HAC: `0.6111` / `2.8138` / `23`
- Maximum bounded-reference Murcko Tanimoto: `0.1858` vs `venetoclax`
- Source: `validation/W5_FORGE_LEDGER.json` at `results.FLT3_iso.candidates[3]`
- Dependency mapping: `FLT3` / https://depmap.org/portal/gene/FLT3?tab=overview / causality `unclaimed`
- Interpretation: A compact spirocyclic amide replaces a flat hydrophobe with a shape-rich, conformationally bounded core. The observation is restricted to the bundled reference set and does not establish patentability, non-infringement, synthesis, or activity.

### 3. `brd4_dmisox_azaspiro` (BRD4)

- SMILES: `Cc1noc(C)c1-c1ccc(N2CCC3(CCOCC3)CC2)cc1`
- Formula / exact mass: `C20H26N2O2` / `326.19942807`
- Fsp3 / SA / HAC: `0.5500` / `2.9241` / `24`
- Maximum bounded-reference Murcko Tanimoto: `0.2319` vs `gilteritinib`
- Source: `validation/W5_FORGE_LEDGER.json` at `results.BRD4.candidates[2]`
- Dependency mapping: `BRD4` / https://depmap.org/portal/gene/BRD4?tab=overview / causality `unclaimed`
- Interpretation: An aza-spiro saturated center projects the dimethylisoxazole recognition element into a compact non-planar topology. The observation is restricted to the bundled reference set and does not establish patentability, non-infringement, synthesis, or activity.

## 5. Claim boundary

- Computational gate: **PASS**
- Public source audit: **PASS**
- Paper claim gate: **FAIL** until formula declarations, release-pinned
  dependency analysis, and a legal claim-chart review exist.
- These outputs establish reproducible descriptor calculations and bounded
  structural divergence only.

## 6. ALAIN_TELEMETRY

```text
GF4_TARGETS=100
GF4_RESIDUAL_STATUS=NULL
GF4_NONZERO_RESIDUALS=0
GF4_PATH_SHA256=ea672670f153560efa783ab7833bf7b17c3f4f5b09bb990ba97b854ea7cbec1d
GF4_TARGET_SHA256=f8ce4961fa5054660dcb0d067ee404aa73559d6b786a10cf9572554d48d7246d
MURCKO_SCOPE_CANDIDATES=233
MURCKO_WORST_CASE_TANIMOTO=0.594595
MURCKO_WORST_CANDIDATE=LEUKEMIA-007
MURCKO_WORST_TARGET=LEUKEMIA
MURCKO_WORST_REFERENCE=21Q
IP_CONFLICT=true
DEPMAP_1_CANDIDATE=myc_morpholine_acetyl__oxaspiro_amine
DEPMAP_1_GENE=MYC
DEPMAP_1_CAUSALITY=unclaimed
DEPMAP_1_URL=https://depmap.org/portal/gene/MYC?tab=overview
DEPMAP_2_CANDIDATE=flt3_iso_spiro33hept
DEPMAP_2_GENE=FLT3
DEPMAP_2_CAUSALITY=unclaimed
DEPMAP_2_URL=https://depmap.org/portal/gene/FLT3?tab=overview
DEPMAP_3_CANDIDATE=brd4_dmisox_azaspiro
DEPMAP_3_GENE=BRD4
DEPMAP_3_CAUSALITY=unclaimed
DEPMAP_3_URL=https://depmap.org/portal/gene/BRD4?tab=overview
```
