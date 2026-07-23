# ER100: ESR1 receptor-only pocket benchmark

`clinical_grade=false`

This paper repository has one scientific scope: receptor-only pocket
prediction on experimental ESR1 structures. It contains no cross-domain
campaign material, hardware evidence, regulatory-readiness material, or other
disease programs.

## Current evidence status

The available receptor-only ESR1 run is a retrospective six-structure pilot,
not a locked test:

- Foliation Top-1 DCA ≤4 Å: 0/6
- fpocket Top-1 DCA ≤4 Å: 0/6
- P2Rank Top-1 DCA ≤4 Å: 6/6
- fpocket and P2Rank executed successfully
- DeepPocket was unavailable and is not counted as a baseline miss

The original development, validation, and test assignment is not
cluster-disjoint. Multiple cross-split receptor pairs have Cα RMSD ≤1.5 Å.
Accordingly, no hidden-pocket superiority or comparative-significance claim is
supported. Until a preregistered unseen cluster-disjoint holdout exists, the
method contribution is limited to deterministic receptor-geometry auditing.

## Leakage boundary

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
src/                 receptor-only prediction and separate scoring
data/manifests/      source URLs, checksums, split and clustering ledgers
configs/             frozen seeds and pilot hyperparameters
results/pilot/       retrospective results; not locked evidence
tests/               leakage, accounting, and schema tests
```

## Reproduction

```bash
make verify
make test
```

Full external-tool reproduction additionally requires pinned fpocket and
P2Rank installations. CI validates schemas, leakage guards, statistics, and
published checksums; it does not relabel an unavailable external tool as a
miss.

## Claim boundary

- The present dataset has six ESR1 structures, not 100 independent structures.
- Docking scores are not affinity, efficacy, or therapeutic evidence.
- The current split is not cluster-disjoint, so this is a retrospective pilot.
- Comparative significance and hidden-pocket superiority are not claimed.

Sources:

- ESR1 UniProt: https://www.uniprot.org/uniprotkb/P03372/entry
- RCSB PDB downloads: https://files.rcsb.org/download/3ERT.pdb
- fpocket: https://github.com/Discngine/fpocket
- P2Rank: https://github.com/rdk/p2rank
- Pocket benchmark metric context: https://doi.org/10.1093/bioinformatics/btaa105
