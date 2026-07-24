# ER100: multitarget, multimodal computational chemistry (single paper repo)

`clinical_grade=false`

This is the **single** ER100 paper repository. The scientific program is
multitarget and multimodal computational chemistry for oncology resistance — it
is not a single-target ESR1 showcase and not a cross-disease paper. This README
describes only the **current** state. Per-version internal history is kept
local and unpublished.

> One paper, one repo. Full bulk evidence (all accepted identities, 3D
> ensembles, ledgers) is kept in a **local private evidence tree**
> (`foliation-er100-multimodal-chemistry/`) mirrored to iCloud with SHA-256
> pointers. A curated top-N subset is folded into this repo so reviewers can
> reproduce the headline evidence without downloading the bulk archive.

## Scope: shared principles, target-specific computation

All six panel targets share fail-closed identity, provenance, and claim
boundaries. Computation specializes by target pocket/family:

| Panel slot | Target-specific computational focus |
|---|---|
| ESR1 | Nuclear-receptor LBD / Helix-12 / AF2; genotypes WT, Y537S, D538G kept separate |
| KRAS | G12C Switch-II (`6OIM`) vs G12D binary refs (`9BL0` noncovalent primary, `9GBJ` covalent observation-only) |
| FLT3 | Kinase-domain activation-loop context; ITD is not a KD-crystal surrogate |
| PIM1 | Hinge / P-loop ATP site with macrocycle occupancy references |
| PIK3CA | Helical vs kinase-domain mutants separated; `8W9A` for H1047R allostery |
| CDK4/6 | Explicit CDK4 vs CDK6 isoform labeling with cyclin context |

Structure-defined modalities:

1. targeted small molecule
2. PROTAC
3. molecular glue
4. cyclic peptide / macrocycle

### Modality protocols are not interchangeable

- **Small molecule** → single-pocket geometry (e.g. KRAS G12D `9BL0`).
- **PROTAC** → target–PROTAC–E3 ternary complex; a single-pocket score is
  never accepted as PROTAC evidence.
- **Molecular glue** → declared partner interface required.
- **Macrocycle** → only structures with a target-matched wall and a
  multi-conformer / ring-strain package; single-pocket Vina is not a
  macrocycle surrogate.

## Current materialized evidence

The current release materializes one fully worked cell — **KRAS G12D
noncovalent small molecule** — plus a sourced reference foundation for the
other cells:

- **798** accepted identities from 2,783 source-backed medicinal-chemistry /
  BRICS variants of MW≤800 G12D parents, under hard diversity gates
  (Morgan/ECFP Tanimoto ≤ 0.8, Murcko-scaffold cap 10) with **no quota
  forcing**. The shortfall against the 1,000-per-cell aspiration is reported,
  not hidden — a count is not druggability.
- Primary docking geometry `9BL0`; covalent `9GBJ` is observation-only.
- Pocket-wall clash check uses **per-element Bondi (1964) van der Waals radii**
  (element-aware), with a disclosed fixed-threshold `clash_severity_band`. This
  is a geometric observation, not affinity.
- **Toxicology screen**: RDKit structural-alert screen (literature-cited
  alerts) across all 798 candidates; a hERG (`5VA1`) central-cavity geometric
  compatibility probe. These are liability flags, not safety clearance.
- Each candidate carries **bilingual (EN/中文)** chemistry / biology /
  pharmacology rationale, HTTPS source URLs, and inherited SHA-256 digests.
- Source foundation: ChEMBL_37 (2026-05-01) reference identities and diverse
  chemotype anchors; **no synthetic assay or patient data** is used.

`CHEMISTRY_READY` / "accepted" is a computational triage state. It is **not**
experimental acceptance, affinity, efficacy, degradation, safety, novelty,
patentability, FTO, or clinical evidence.

## Appendix A: ESR1 receptor-only pocket pilot — **INVALIDATED, pending regeneration**

This repository retains a retrospective six-structure ESR1 receptor-only pocket
pilot as a methods appendix. **The previously reported pilot numbers are
invalidated** and are not citable:

- Root cause: labels were built by ligand *resname only*, merging every
  crystallographic copy of the ligand across all chains into one pseudo-ligand
  (e.g. 4Q50 `OHT` 29 → 232 heavy atoms ≈ 8 copies), so DCA could match the
  wrong copy.
- Fix: `pdb_io.ligand_heavy_coords` now selects a single chain-scoped ligand
  instance (regression test included). `results/pilot/CLAIMS.json` and the pilot
  report are marked `INVALIDATED_PENDING_LABEL_REGENERATION`.
- No pilot number (Foliation / fpocket / P2Rank / random, including the prior
  P2Rank 6/6) may be cited until labels are regenerated per chain and all
  methods are re-run on a preregistered, cluster-disjoint holdout.

Until then this appendix contributes only deterministic receptor-geometry
auditing, with no hidden-pocket or comparative-significance claim.

### Leakage boundary

Prediction and scoring are separate commands: prediction receives
checksum-pinned, ATOM-only receptor PDBs; scoring joins ligand-derived labels
only after prediction files are closed. `TOOL_UNAVAILABLE` is never a miss;
`CRASH`/`EMPTY` is a failure in the intention-to-evaluate denominator.

## Repository layout

```text
contracts/           paper scope and evidence pointers
src/                 receptor-only prediction + separate scoring (Appendix A)
data/manifests/      source URLs, checksums, split and clustering ledgers
configs/             frozen seeds and pilot hyperparameters
results/pilot/       retrospective Appendix A results (currently INVALIDATED)
tests/               leakage, accounting, schema, and scope tests
```

## Reproduction

```bash
make verify
make test
```

`make verify` enforces scope, leakage, `clinical_grade=false`, and the pilot
invalidation marker. `make test` runs the unit suite (including the ligand
chain-selection regression). Full Appendix A external-tool reproduction
additionally requires pinned fpocket and P2Rank.

## Claim boundary

- ER100 is multitarget and multimodal; this paper does not collapse all targets
  into one ESR1 pocket score.
- Docking scores are not affinity, efficacy, or therapeutic evidence.
- The Appendix A dataset has six ESR1 structures, is not a cluster-disjoint
  locked holdout, and is currently invalidated.
- Other organ or disease programs belong in separate paper repositories.

## Sources

- ESR1 UniProt: https://www.uniprot.org/uniprotkb/P03372/entry
- RCSB PDB (ESR1 example): https://files.rcsb.org/download/3ERT.pdb
- KRAS G12D binary: https://files.rcsb.org/download/9BL0.pdb
- KRAS G12C: https://files.rcsb.org/download/6OIM.pdb
- hERG cryo-EM: https://files.rcsb.org/download/5VA1.pdb
- ChEMBL: https://www.ebi.ac.uk/chembl/
- fpocket: https://github.com/Discngine/fpocket
- P2Rank: https://github.com/rdk/p2rank
- Pocket-benchmark metric context: https://doi.org/10.1093/bioinformatics/btaa105
- Bondi van der Waals radii (1964): https://doi.org/10.1021/j100785a001

---

## 中文说明（最新版本）

这是**唯一**的 ER100 论文仓库。研究主题是面向肿瘤耐药的**多靶点、多模态计算
化学**，既不是单靶点 ESR1 展示，也不是跨疾病论文。本 README 只描述**当前**状态；
逐版本的内部历史保存在本地、不发布。

- 六个靶点共享同一套 fail-closed 身份、来源与主张边界，但按靶点口袋/家族做计算
  特化（见上表）。四种结构定义模态各有独立协议：小分子用单口袋；PROTAC 用
  三元复合物；分子胶需声明伙伴界面；大环只用有匹配墙体与多构象/环张力包的结构。
- **当前物化证据**：KRAS G12D 非共价小分子单元，接受 **798** 个身份（Tanimoto
  ≤0.8、Murcko 上限 10，硬门、不强制凑满 1,000）；主几何 `9BL0`，`9GBJ` 仅共价
  观察；口袋墙体采用**逐元素 Bondi(1964) 范德华半径**并给出固定阈值
  `clash_severity_band`（几何观察，非亲和力）。
- **毒性审查**：对全部 798 条做 RDKit 结构预警筛查，并对 hERG(`5VA1`) 中心腔做
  几何相容性探针；这些是风险标记，不是安全放行。
- 每条候选附**中英双语**化学/生物/药理原理、HTTPS 来源 URL 与 SHA-256；来源为
  ChEMBL_37(2026-05-01)，**不使用合成实验或患者数据**。
- **附录 A（ESR1 口袋回顾性 pilot）当前作废**：旧标签仅按配体名合并了所有晶体
  拷贝（如 4Q50 `OHT` 29→232 重原子≈8 份），导致 DCA 可能匹配错误拷贝。已修复
  `ligand_heavy_coords`（按链选单一实例，含回归测试），并把 pilot 报告标记为
  `INVALIDATED_PENDING_LABEL_REGENERATION`。在按链重建标签并在预注册的
  cluster-disjoint 留出集上重跑之前，任何 pilot 数值（含 P2Rank 6/6）都不可引用。
- 全量证据（所有接受身份、三维构象、账本）保存在**本地私有证据树**
  `foliation-er100-multimodal-chemistry/` 并镜像到 iCloud（附 SHA-256 指针）；
  仓库内只并入 curated 子集，便于复现头部证据。
