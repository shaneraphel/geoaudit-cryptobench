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
| KRAS | G12C Switch-II (`6OIM`) and G12D binary refs (`9BL0` noncovalent primary, `9GBJ` covalent observation-only); not BRD4 `4LYW` |
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

## v9 companion-evidence update / v9 伴随证据更新

The evidence repository now materializes 24,000 deterministic target ×
modality **design slots**: 4,000 immutable v7/v8 lineage references plus
20,000 empty slots. These are not 24,000 molecules. v9 has not yet promoted an
identity to `IDENTITY_READY`, and established druggable candidates remain 0.

伴随证据库现已物化 24,000 个确定性“靶点 × 模态”**设计槽位**：4,000 个不可变
v7/v8 谱系引用和 20,000 个空槽位。它们不是 24,000 个分子；v9 尚无
`IDENTITY_READY` 身份，已证明可成药候选仍为 0。

The non-synthetic source snapshot stores 420 RCSB/ChEMBL payloads
(244,652,474 bytes) in the configured iCloud data container and publishes
portable URLs and SHA-256 digests. It pins ChEMBL_37 (2026-05-01), completes
the declared non-null-pChEMBL pagination and yields 26,543 ChEMBL reported
reference identities and 9,002 deterministic diverse chemotype anchors with bilingual
chemistry, biology, clinical-boundary and wet-lab-control text. These are known
experimental references, not novel designs; their reported activity does not
transfer to derivatives.

真实来源快照把 420 个 RCSB/ChEMBL 载荷（244,652,474 字节）保存在配置的 iCloud
数据容器，并公开可移植 URL 与 SHA-256。快照固定 ChEMBL_37（2026-05-01）并完整
遍历声明的非空 pChEMBL 分页，由此得到 26,543 个 ChEMBL 报告参考身份和
9,002 个确定性多样化学型锚点，每条含中英文化学、生物、临床边界与湿实验对照说明。
它们是已知实验参考，不是新设计；效力不得转移给衍生物。

The v9 genotype/isoform registry separates ESR1 WT/Y537S/D538G, KRAS
G12C/G12D/G12V/G12R, FLT3 reference/D835/ITD, PIK3CA H1047R/E542K and CDK4
from CDK6. It adds source-backed reference structures for the ESR1
14-3-3/VHL hybrid MGPROTAC (`9SV3`), KRAS G12D binary complexes (`9BL0`,
`9GBJ`), KRAS PROTAC references (`8QU8`, `9RKE`, `9RKN`, `9RKC`) and
KRAS/CypA induced tri-complexes (`9BFY`, `9BI1`, `8TBM`, `9BGC`).
These do not license protocol transfer: PROTACs still require
target–PROTAC–E3 ensembles, glues require declared partner interfaces, and
macrocycles require target-matched multi-conformer/ring-strain packages.
Zero of 24 cells is yet safe-expansion-ready.

The paper therefore reports a stronger sourced foundation and clearer
negative results, not “1,000 druggable candidates per cell.” Exact
PubChem/ChEMBL no-hit remains snapshot-specific and is not global novelty,
patentability or FTO. No patient data, wet-lab measurement, clinical result,
regulatory-dossier readiness or organization-specific reporting status is
created by this update.

The companion OpenReview self-audit remains `MAJOR_REVISION`. All 4,000
historical identities are lineage-only and blocked from v9 migration: mixed
genotype/isoform packs, CDK `5L2I` route drift, conflicting geometry flags,
24-cell biology/pharmacology boilerplate and null-valued hard gates must be
repaired before any identity expansion or experimental-priority claim.

## v9.1 companion-evidence update / KRAS G12D computational priority

The companion evidence repository now ships an additive KRAS G12D
noncovalent small-molecule **computational-priority** campaign:

- Generated 1,335 source-backed medchem/BRICS variants from 84 MW≤800 G12D
  parents; accepted **494** identities under Morgan-0.8 / Murcko-cap-10 gates
  without quota forcing.
- Primary geometry `9BL0`; covalent `9GBJ` observation-only.
- Full packaging in companion
  `releases/v9_1/KRAS_G12D_SM_v2026-07-24/`: JSONL.gz, Parquet, 2D SDF.gz,
  top-100 3D ensemble SDF.gz, novelty/patent ledger, top-20 wet-lab plans.
- Novelty: ChEMBL exact hits = 3; PubChem successful exact queries found 0 CIDs;
  SureChEMBL remains remote-query metadata only (no 15GB snapshot claim).
- This does **not** clear the v9 `MAJOR_REVISION` lineage-migration block, does
  **not** claim 1,000 druggable candidates, and does **not** establish global
  novelty, FTO, affinity, or clinical readiness.

伴随证据库现已发布附加的 KRAS G12D 非共价小分子**计算优先**战役：由 84 个
MW≤800 G12D 亲本生成 1,335 个有来源药化/BRICS 变体，在 Morgan-0.8 /
Murcko≤10 硬门下接受 **494** 个身份且不强制凑满 1,000；主几何为 `9BL0`，
`9GBJ` 仅作共价观察。完整 JSONL/Parquet/SDF/三维/新颖性/湿实验包在伴随库
`releases/v9_1/KRAS_G12D_SM_v2026-07-24/`。这不解除 v9 `MAJOR_REVISION`
谱系迁移封锁，也不构成可成药、全球新颖性、FTO、亲和力或临床就绪主张。
