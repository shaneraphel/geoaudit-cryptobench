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

- Phase 1: 4,000 globally unique `IDENTITY_READY` graph identities
- Phase 2: 4,000 `CHEMISTRY_READY` records after modality-specific chemistry
  gates, with target allocation preserved
- Per-candidate bilingual chemistry / biology / pharmacology text
- HTTPS source URLs and inherited SHA256 digests
- Target-local structural observations without docking claims

`CHEMISTRY_READY` is a computational triage state. It is not experimental
acceptance, affinity, efficacy, degradation, safety, novelty, patentability,
or clinical evidence.

Open companion PR:

- https://github.com/shaneraphel/foliation-er100-multimodal-chemistry/pull/1

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
