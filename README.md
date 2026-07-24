# ER100: multitarget multimodal computational chemistry paper

`clinical_grade=false`

This is the ER100 paper repository. The scientific program is multitarget and
multimodal computational chemistry for oncology resistance, not a single-target
showcase and not a cross-disease paper.

## Paper architecture

ER100 is organized as two linked private repositories:

| Repository | Role |
|---|---|
| `foliation-er100-oncology-data` (this repo) | Paper scope, claim boundary, ESR1 receptor-only pocket pilot appendix, and pointers to release evidence |
| [`foliation-er100-multimodal-chemistry`](https://github.com/shaneraphel/foliation-er100-multimodal-chemistry) | Auditable candidate evidence release: identities, chemistry gates, bilingual principles, source URLs, and checksums |

Candidate JSON dumps, docking pose archives, hardware acceleration packages, and
regulatory-readiness packages are intentionally **not** stored in this paper
tree. They live in the companion evidence repository when they are release-
ready.

## Shared principles, target-specific computation

All six panel targets share fail-closed identity, provenance, and claim
boundaries. Computation specializes by target pocket/family:

| Panel slot | Specialization focus |
|---|---|
| ESR1 | Nuclear-receptor LBD / Helix-12 / AF2; genotype-separated WT, Y537S, D538G |
| KRAS | G12C Switch-II; PDB `6OIM` (not BRD4 `4LYW`) |
| FLT3 | Kinase-domain activation-loop context; ITD is not a KD-crystal surrogate |
| PIM1 | Hinge / P-loop ATP site with macrocycle occupancy references |
| PIK3CA | Helical vs kinase-domain mutants kept separate; `8W9A` for H1047R allostery |
| CDK4/6 | Explicit CDK4 versus CDK6 isoform labeling with cyclin context |

Structure-defined modalities for the current evidence release are:

1. targeted small molecule
2. PROTAC
3. molecular glue
4. cyclic peptide / macrocycle

Each modality has its own validation protocol. Small-molecule single-pocket
logic is not reused as PROTAC ternary evidence or as a macrocycle surrogate.

## Companion evidence status

The companion release currently reports:

- Phase 1: 4,000 campaign-unique `IDENTITY_READY` graph identities
- Phase 2: 4,000 `CHEMISTRY_READY` records after modality-specific chemistry
  gates, with target allocation preserved
- Per-candidate bilingual chemistry / biology / pharmacology text
- HTTPS source URLs and inherited SHA256 digests
- Target-local structural observations without docking claims
- v8: one deterministic 4,000-record 2D SDF, a per-record bond-graph index,
  4,000 RDKit ligand-feature records (67,686 features), and a 24-structure
  modality × target gallery
- v8 internal diversity audit: dense analogue/template families are explicit;
  the largest macrocycle scaffold occupies 11.6% of that modality
- v8 public exact-match snapshot: 81/4,000 exact InChIKey matches in retrieved
  PubChem/ChEMBL snapshots (44 small molecules, 37 molecular glues); 3,919
  no-hits remain bounded public-database results, not global novelty or FTO
- v8 bounded accelerator sidecar: 24 ligand-only 3D representatives; 18
  force-field minimizations converged, NCGD convergence was 0/24 (`MAX_ITER`),
  GCU integer software-reference parity was 24/24; no target pose, physical
  hardware execution or new netlist is claimed
- v8.1 protocol requalification: all 24 target × modality cells and all 4,000
  records were re-evaluated; only 167 KRAS-G12C small-molecule records are
  geometry-route-ready, 3,833 remain structurally blocked, and 334 assignments
  to the 5T35/5FQD method templates are quarantined

`CHEMISTRY_READY` is a computational triage state. It is not experimental
acceptance, affinity, efficacy, degradation, safety, novelty, patentability,
or clinical evidence.

Merged companion v7 PR:

- https://github.com/shaneraphel/foliation-er100-multimodal-chemistry/pull/1

The SDF coordinates are depictions, not target poses. The internal diversity
audit is not a global novelty or patentability search. PROTAC ternary complexes,
molecular-glue partner interfaces and accepted macrocycle conformer ensembles
remain missing evidence, not implied results.

“Geometry-route-ready” does not mean that a target pose was computed. It means
that a target-local package may receive a future pose-plausibility calculation.
The v8.1 ledger contains English/Chinese protocol rationale, next steps and
HTTPS source URLs for every record. Specifically:

- ESR1 must split WT, Y537S and D538G; D538G `4Q13` remains observation-only.
- KRAS G12C may route through `6OIM`, but covalent chemistry is a separate
  reaction-model question.
- FLT3 D835-context structures do not represent ITD.
- PIM1 `5EOL` remains a macrocycle route and still needs conformer ensembles.
- PIK3CA E542K `8GUA` is not transferred to E545K.
- CDK4 `7SJ3` and CDK6 `5L2S` require isoform/cyclin assignment.
- 5T35 (BRD4/VHL) and 5FQD (CRBN/CK1A) are method templates, not panel-target
  PROTAC or molecular-glue evidence.

The requested scale of 1,000 records per target per structure-defined modality
is 24,000 records. The current release has 166–167 per cell. Neither a count nor
`CHEMISTRY_READY` establishes druggability; the paper therefore does not call
these 4,000 records “4,000 druggable candidates.”

The preliminary expansion gate retains 875 internal one-per-cluster/scaffold
diversity seeds after excluding public exact matches from novelty-focused
selection. Only 52 also have a route-ready structural protocol, all in the
KRAS-G12C small-molecule cell. Zero of 24 cells is currently ready for
1,000-record expansion.

The companion v8 contract also allocates 1,000 planning slots each for
biologics, gene medicines and engineered stem/progenitor-cell products. These
are not 3,000 materialized candidates. Sequence-defined proteins and
oligonucleotides receive a theoretical formula only when every chain/sequence,
terminus, covalent modification and charge convention is explicit. Living
engineered-cell products, heterogeneous biologics, viral vectors and delivery
assemblies do not have one molecular formula; they require product or component
specifications.

## Appendix A: ESR1 receptor-only pocket pilot

This repository retains a retrospective six-structure ESR1 receptor-only
pocket pilot as a methods appendix. It is not the whole ER100 paper.

Current appendix status:

- Foliation Top-1 DCA ≤4 Å: 0/6
- fpocket Top-1 DCA ≤4 Å: 0/6
- P2Rank Top-1 DCA ≤4 Å: 6/6
- fpocket and P2Rank executed successfully
- DeepPocket was unavailable and is not counted as a baseline miss

The original development / validation / test assignment is not
cluster-disjoint. Multiple cross-split receptor pairs have Cα RMSD ≤1.5 Å.
Accordingly, no hidden-pocket superiority or comparative-significance claim is
supported. Until a preregistered unseen cluster-disjoint holdout exists, this
appendix contribution is limited to deterministic receptor-geometry auditing.

### Leakage boundary for the pocket appendix

Prediction and scoring are separate commands:

1. prediction receives checksum-pinned, ATOM-only receptor PDBs;
2. scoring joins ligand-derived labels only after prediction files are closed.

Ligand coordinates may define evaluation labels, but may not define a pocket
anchor, candidate center, search grid, method feature, or baseline input.

`TOOL_UNAVAILABLE` makes the mandatory benchmark environment incomplete. It is
never a miss. Method `CRASH` or `EMPTY` is a failure in the
intention-to-evaluate denominator and is also reported in the failure rate.

## Repository layout

```text
contracts/           paper scope and companion-evidence pointers
src/                 receptor-only prediction and separate scoring (Appendix A)
data/manifests/      source URLs, checksums, split and clustering ledgers
configs/             frozen seeds and pilot hyperparameters
results/pilot/       retrospective Appendix A results; not locked evidence
tests/               leakage, accounting, schema, and scope tests
```

## Reproduction

```bash
make verify
make test
```

Full external-tool reproduction of Appendix A additionally requires pinned
fpocket and P2Rank installations. CI validates schemas, leakage guards,
statistics, and published checksums when available; GitHub Actions billing
failures do not invalidate local verification.

## Claim boundary

- ER100 is multitarget and multimodal; this paper tree does not collapse all
  targets into one ESR1 pocket score.
- The companion evidence release is computational research, not a clinical
  trial, IND package, or patent opinion.
- The Appendix A pocket dataset has six ESR1 structures, not 100 independent
  structures, and is not a cluster-disjoint locked holdout.
- Docking scores are not affinity, efficacy, or therapeutic evidence.
- Other organ or disease programs belong in separate paper repositories.

Sources:

- Companion evidence repo: https://github.com/shaneraphel/foliation-er100-multimodal-chemistry
- ESR1 UniProt: https://www.uniprot.org/uniprotkb/P03372/entry
- RCSB PDB downloads: https://files.rcsb.org/download/3ERT.pdb
- KRAS G12C structure: https://files.rcsb.org/download/6OIM.pdb
- fpocket: https://github.com/Discngine/fpocket
- P2Rank: https://github.com/rdk/p2rank
- Pocket benchmark metric context: https://doi.org/10.1093/bioinformatics/btaa105
