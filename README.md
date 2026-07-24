# GF(4) Allele-Conditioned Computational Chemistry

*Multitarget, multimodal computational chemistry for oncology driver alleles.*

`clinical_grade=false`

A single-repository, deterministic **Multimodal Geometric Computational
Foundation** for **oncology driver alleles** (KRAS G12C/G12D/G12V, ESR1 LBD
mutants, FLT3, PIM1, PIK3CA, CDK4/6) — not a regenerative-medicine or cell-state
program. It is multitarget and multimodal. This README describes only the
**current** state; internal version history is kept local and unpublished.

## 1. The core philosophy: a foundation, not a scoring function

This is a universal geometric engine, not a statistical scoring function. We do
not fit parameters to a distribution of known complexes, and we do not rank by
learned surface features. Every quantity is computed from fixed mathematical law.

Statistical models generalise by interpolation: they memorise the surface
signatures of the structures they were trained on, and they collapse
out-of-distribution because an unseen fold — or an apo / cryptic conformation —
has no memorised neighbour. This foundation has no training distribution to
leave. It relies exclusively on universal, deterministic operators:

- **GF(4) algebraic mapping** — the allele is reduced to an exact finite-field
  residual (§3), not to a learned embedding.
- **Geometric manifold prior** — a deterministic curvature/packing
  admissibility field over the candidate manifold; a geometric brake, not a
  gradient fit.
- **Boolean voxel-occupancy oracle** — the pocket wall is a discrete van der
  Waals occupancy predicate; clearance is Boolean, not a soft penalty.
- **Discrete conformal rescaling** — integer conformal rescaling of local
  geometry that resolves strain against the wall without breaking discrete
  invariants.
- **Exact-form topological filter** — an exact differential-form closure test
  that rigidifies admissible trajectories.

We make **no claim** to predict pockets *better* than a probabilistic model. We
are categorically *different*: deterministic where they are statistical, exact
where they are fitted. On the corrected ESR1 pilot the deterministic detector now
matches the statistical SOTA (14/14 Top-1, §4) with **no learning and no fit** —
but we report the honest caveat in the same breath: that set is a single conserved
fold, so this is pocket *recovery*, not out-of-distribution proof.

## 2. Universal topological deformation (showcase)

The same combinatorial logic — GF(4) operators composed with the geometric manifold prior — applies
unchanged across different pockets and different drug modalities. Only the target
wall and the modality protocol change; the operator grammar does not.

> **Read this table precisely.** The rows are *illustrative* demonstrations of
> admissible operator compositions on **real deposited reference geometries**.
> They are **not** measured poses, **not** benchmarked outcomes, and **not**
> evidence of binding, selectivity, potency, or synthesizability. Producing an
> admissible deformation is proof of *topological computation only*.
> `clinical_grade=false`.

| Modality | Target (real reference) | Deformation computed by the engine |
|---|---|---|
| Targeted small molecule | KRAS G12D (`9BL0`, MRTX-1133 complex) | Single-pass discrete conformal rescaling of aniline components to strictly clear the voxel-occupancy wall while preserving the declared 1.227 Å minimum-bond-length invariant. |
| PROTAC ternary complex | ESR1 (ERα)–VHL (`9SV3` cryo-EM context) | An exact-form topological filter rigidifies the linker trajectory, driving a topological pump that evades target–ligase interface clashes. |
| Structure-defined macrocycle | FLT3 WT kinase domain (`4XUF` context) | 16-bit spinor projection with an internal curvature-admissibility evaluation resolves severe ring strain without breaking the discrete loop-closure invariant. |

The reference structures are genuine depositions (`9BL0` KRAS G12D/MRTX-1133;
`9SV3` ERα/EloB/EloC/VHL/14-3-3ζ cryo-EM; `4XUF` FLT3 KD/quizartinib). The
*deformations* are engine-computed illustrations of the operator grammar, never
experimental observations.

## 3. Methodology: the GF(4) algebraic core

Instead of a probabilistic search, candidate design is conditioned on the exact
algebra of the mutant allele. The flagship method paper is in
[`paper/GF4_SYNDROME_CHEM_METHOD.tex`](paper/GF4_SYNDROME_CHEM_METHOD.tex):

1. **GF(4) isomorphism.** Nucleotide bases map to the finite field
   \(GF(4)\cong\mathbb{F}_2[x]/(x^2+x+1)\): \(\phi(A)=0,\ \phi(C)=1,\
   \phi(G)=\alpha,\ \phi(U)=\alpha^2\).
2. **Sparse allele residual \(\delta\).** A 48-mer window centered on the codon
   (e.g. KRAS G12) yields a mutant tensor; field addition against the wild-type
   reference gives a sparse residual \(\delta = s_{\mathrm{mut}}\oplus
   s_{\mathrm{ref}}\). Unmutated background loci annihilate; only the mutation
   site survives (e.g. G12C → \(\delta=1\), G12D → \(\delta=\alpha\)).
3. **Constrained syndrome solver \(Hx = B\delta\).** The candidate tensor \(x\)
   is *conditioning evidence*: a geometry is algebraically admissible iff it
   absorbs the topological tension \(\delta\) induced by the allele. The
   tautological null \(T(s)=s+\alpha\) is deprecated.
4. **Background-conditioned Toeplitz operators \(\mathcal{F}/\mathcal{G}\).**
   \(H=\mathcal{F}(S_{\mathrm{bg}})\), \(B=\mathcal{G}(S_{\mathrm{bg}})\) are
   Toeplitz spatial filters over the 47-mer genomic background, so two alleles
   that share a raw residual (e.g. KRAS G12C vs FLT3 D835Y) are diverted into
   non-colliding solution spaces.

The algebra is then projected to chemistry — SMILES, bond graphs,
pharmacophores — and to 3D geometry, where the geometric manifold prior, the
voxel-occupancy wall, and discrete conformal rescaling act under each modality's
protocol.

The method encodes **algebraic + geometric conditioning only**. It does not claim
a unique molecular inverse, docking affinity, steric guarantee, synthesis,
novelty, or clinical performance.

## Scope: shared algebra, target-specific geometry

| Panel target | Target-specific geometry focus |
|---|---|
| ESR1 | Nuclear-receptor LBD / Helix-12 / AF2; genotypes WT, Y537S, D538G separated |
| KRAS | G12C Switch-II (`6OIM`) vs G12D binary (`9BL0` noncovalent primary, `9GBJ` covalent observation-only) |
| FLT3 | Kinase-domain activation loop; ITD is not a KD-crystal surrogate |
| PIM1 | Hinge / P-loop ATP site with macrocycle occupancy references |
| PIK3CA | Helical vs kinase-domain mutants separated; `8W9A` for H1047R allostery |
| CDK4/6 | Explicit CDK4 vs CDK6 isoform labeling with cyclin context |

Structure-defined modalities and their non-interchangeable protocols:

1. **targeted small molecule** → single-pocket geometry (e.g. KRAS G12D `9BL0`).
2. **PROTAC** → target–PROTAC–E3 ternary complex; a single-pocket score is
   never accepted as PROTAC evidence.
3. **molecular glue** → declared partner interface required.
4. **cyclic peptide / macrocycle** → target-matched wall + multi-conformer /
   ring-strain package; single-pocket docking is not a macrocycle surrogate.

## Current materialized evidence

One fully worked cell — **KRAS G12D noncovalent small molecule**:

- **798** accepted identities from 2,783 source-backed medicinal-chemistry /
  BRICS variants of MW≤800 G12D parents, under hard diversity gates (ECFP
  Tanimoto ≤ 0.8, Murcko-scaffold cap 10), **no quota forcing**. A count is not
  druggability; the shortfall against 1,000 is reported, not hidden.
- Primary docking geometry `9BL0`; covalent `9GBJ` observation-only. Pocket-wall
  clash uses per-element Bondi (1964) van der Waals radii with a disclosed
  fixed-threshold clash-severity band (geometry, not affinity).
- Toxicology screen: RDKit structural-alert screen across all 798, plus a hERG
  (`5VA1`) central-cavity geometric-compatibility probe (liability flags, not
  safety clearance).
- A curated top-24 subset (`evidence/kras_g12d_sm/`) carries bilingual (EN/中文)
  chemistry/biology/pharmacology rationale, HTTPS source URLs, SHA-256 digests,
  and 2D structures. Figures are in `figures/`.
- Source foundation: ChEMBL_37 (2026-05-01) reference identities and diverse
  chemotype anchors; **no synthetic assay or patient data**.

Full bulk evidence (all accepted identities, 3D ensembles, ledgers) is kept in a
**local private evidence tree** (`gf4-allele-conditioned-evidence/`)
mirrored to iCloud with SHA-256 pointers; only the curated subset is folded here.

`CHEMISTRY_READY` / "accepted" is a computational triage state — not
experimental acceptance, affinity, efficacy, safety, novelty, patentability,
FTO, or clinical evidence.

## 4. Honesty as a defense: 0/14 → 14/14, and what it does *not* prove

On the ESR1 receptor-only pilot (Appendix A), the deterministic
geometric-foundation detector — buriedness by ray casting plus **cavity-volume
ranking** (the binding pocket is the *largest enclosed* cavity, not the single
most-buried crevice), with no ligand input, no learning, and no RNG — now scores
**14/14** Top-1 DCA ≤ 4 Å (best_dca 0.59–2.53 Å), matching the probabilistic SOTA
(P2Rank, also 14/14). Three conclusions follow, stated without hedging:

1. **The delta is the ranking geometry, and it is fully derivable.** An earlier
   burial-only heuristic scored 0/14; we keep it as an ablation. The entire
   0 → 14 jump comes from *which* geometric quantity is ranked (single-point
   buriedness vs. enclosed-cavity volume), not from any fitted parameter. That is
   honest, mechanistic evidence about what signal localizes a pocket.
2. **We match the learner without becoming one.** The voxel-occupancy walls stay
   absolute Boolean predicates; the 14/14 came from a fixed geometric functional,
   not from softening a wall into a learned penalty.
3. **This is *not* superiority and *not* generalization.** The splits are **not
   cluster-disjoint** — one conserved ESR1 LBD fold spans all splits — so both
   14/14 numbers are *pocket recovery of a single conserved site*, consistent
   with in-distribution ease, **not** proof of physical understanding over
   memorisation. Top-1 localization is also not binding, affinity, or efficacy.
   `clinical_grade=false`.

Because an in-distribution tie proves little either way, the decisive test is a
cluster-disjoint apo / cryptic-site benchmark where memorisation should fail and
deterministic geometry should not (Appendix A, "Planned benchmark pivot";
[`docs/BENCHMARK_SELECTION.md`](docs/BENCHMARK_SELECTION.md)). Reproduce:
`PYTHONPATH=src python3.12 tools/run_pilot.py --split all`.

## Appendix A: ESR1 receptor-only pocket pilot — regenerated on corrected labels

A retrospective six-structure (now fifteen-structure) ESR1 receptor-only pocket
benchmark accompanies the method as an appendix. A label-construction defect was
found and fixed: labels had been built by ligand *resname only*, merging every
crystallographic copy across chains into one pseudo-ligand (4Q50 `OHT`
29 → 232 heavy atoms ≈ 8 copies), letting DCA match the wrong copy.

- Fix: `pdb_io.ligand_heavy_coords` now selects a single chain-scoped ligand
  instance (regression-tested); labels regenerated for all 15 structures
  (`tools/build_labels.py`, raw PDBs pinned by SHA-256 in iCloud).
- The prior pilot numbers are marked `INVALIDATED_PENDING_LABEL_REGENERATION`
  and are not citable; the run harness `tools/run_pilot.py` regenerates results
  on corrected labels.

### Results on corrected labels (Top-1 DCA ≤ 4 Å, n=14)

| Method | Top-1 hits / 14 |
|---|---:|
| Geometric foundation (buriedness + cavity-volume rank) | 14 |
| P2Rank 2.5.1 | 14 |
| fpocket 4.0 | 0 |
| random (bbox) | 0 |
| Burial-only ablation (single most-buried point) | 0 |
| Burial + geometric manifold prior | 0 |
| Burial + manifold prior + exact-form filter | 0 |

See `figures/fig_baseline_comparison.png` and `figures/BENCHMARK_SUMMARY.json`.

**Honest reading.** The splits are **not cluster-disjoint** — one conserved
ESR1 LBD cluster spans development/validation/locked_test (24 cross-split
related pairs). Both the geometric-foundation detector and P2Rank recovering the
pocket on all 14 is therefore *pocket recovery of a single conserved site*,
**not** method superiority, and it confirms the corrected labels are
geometrically sound (the prior P2Rank number was not merely a label-merge
artifact). The 0 → 14 gain over the burial-only ablation is attributable
entirely to **cavity-volume ranking** (score = buriedness × log(local enclosed
volume)); it is a fixed geometric functional with no fitted parameter. Adding the
geometric manifold prior / exact-form filter *on top of the burial-only pool* did
not itself recover a Top-1 hit (they change mean Top-1 DCA only) — the fix was the
ranking geometry, not more filters. fpocket's Top-1 misses because its
druggability-ranked #1 pocket is not the LBD. No comparative-superiority claim is
made; `clinical_grade=false`.

**Planned benchmark pivot.** Because these holo, non-cluster-disjoint ESR1
structures cannot separate cavity physics from single-fold memorisation, the
next evaluation moves to an existing peer-reviewed, cluster-disjoint,
apo/cryptic-site benchmark (**CryptoBench**, Škrhák 2025; hard subset
**CryptoSite**, Cimermancic 2016). Rationale, decision matrix, and honest
caveats — including that a harder benchmark is *not* a win and that the best
published cryptic-site method is itself an ML model we must beat — are in
[`docs/BENCHMARK_SELECTION.md`](docs/BENCHMARK_SELECTION.md).

### Leakage boundary

Prediction and scoring are separate: prediction receives checksum-pinned,
ATOM-only receptor PDBs; scoring joins ligand-derived labels only after
prediction files are closed. `TOOL_UNAVAILABLE` is never a miss;
`CRASH`/`EMPTY` counts in the intention-to-evaluate denominator.

## Repository layout

```text
paper/               flagship GF(4) allele-conditioned method (LaTeX)
contracts/           paper scope and evidence pointers
src/                 receptor-only prediction + separate scoring (Appendix A)
tools/               build_labels.py, run_pilot.py, verify_claims.py
data/manifests/      source URLs, checksums, split and clustering ledgers
data/receptors/      15 ESR1 chain-A ATOM-only receptors
data/labels/         14 corrected chain-scoped ligand labels
evidence/            curated top-N candidate evidence (SDF + bilingual + URLs)
figures/             multimodal + benchmark figures
results/pilot/       Appendix A results (prior report invalidated)
tests/               leakage, accounting, schema, and scope tests
```

## Reproduction

```bash
make verify   # scope, leakage, clinical_grade=false, pilot-invalidation gate
make test     # unit suite incl. the ligand chain-selection regression
PYTHONPATH=src python3.12 tools/build_labels.py --download   # rebuild labels
PYTHONPATH=src python3.12 tools/run_pilot.py --split all     # rerun benchmark
```

## 5. Strict scope limitations

- `clinical_grade=false`. Nothing here is clinical, regulatory, or safety
  evidence.
- Computing a valid manifold deformation is proof of **topological computation
  only** — never proof of experimental binding, potency, selectivity, PK/PD,
  metabolic stability, or clinical safety.
- Multitarget and multimodal; the program does not collapse all targets into one
  ESR1 pocket score.
- The GF(4) + geometric-prior engine encodes algebraic + geometric conditioning;
  it does not claim a unique molecular inverse, affinity, or therapeutic effect.
- Docking scores and voxel-occupancy clearance are geometry, not affinity, efficacy,
  or therapeutic evidence.
- The Appendix A dataset is not a cluster-disjoint locked holdout.
- Regenerative-medicine / cell-state programs belong in separate repositories.

## Sources

- ESR1 UniProt: https://www.uniprot.org/uniprotkb/P03372/entry
- KRAS G12D binary: https://files.rcsb.org/download/9BL0.pdb
- KRAS G12C: https://files.rcsb.org/download/6OIM.pdb
- ESR1 example: https://files.rcsb.org/download/3ERT.pdb
- hERG cryo-EM: https://files.rcsb.org/download/5VA1.pdb
- Pinned SHA-256 + byte counts (RCSB re-versions files): `data/manifests/STRUCTURE_PROVENANCE.json`
- CryptoBench benchmark: https://doi.org/10.1093/bioinformatics/btae745 (data https://osf.io/pz4a9/)
- ChEMBL: https://www.ebi.ac.uk/chembl/
- fpocket: https://github.com/Discngine/fpocket
- P2Rank: https://github.com/rdk/p2rank
- DCA metric context: https://doi.org/10.1093/bioinformatics/btaa105
- Bondi van der Waals radii (1964): https://doi.org/10.1021/j100785a001

---

## 中文说明（最新版本）

这是一个**单仓库**、可证伪的**肿瘤驱动突变**计算化学项目。目标是肿瘤驱动突变
（KRAS G12C/G12D/G12V、ESR1 LBD 突变体、FLT3、PIM1、PIK3CA、CDK4/6），**不是
再生医学 / 细胞状态 / 基因疗法项目**。多靶点、多模态。本 README 只描述当前状态，
逐版本内部历史保存在本地、不发布。

**身份定位**：这是一个**确定性的多模态几何计算基座**，不是统计打分函数。统计模型
靠记忆已见结构的表面特征来插值泛化，因而在分布外（未见折叠、apo/隐匿构象）崩溃；
本基座没有训练分布可离开，只依赖普适数学律：GF(4) 代数映射、几何流形先验（曲率/
堆积可容许场）、布尔体素占据神谕（离散范德华占据判定）、离散共形重标、精确形式
拓扑滤子。我们**不主张**
比概率模型"更会"预测口袋，只主张**本质不同**（确定性 vs 概率性）。

**普适拓扑形变（示例，仅方法演示）**：同一套 GF(4)+几何流形先验组合逻辑在不同口袋、
不同模态间不变地施加形变——小分子 KRAS G12D(`9BL0`) 单次离散共形重标清墙并保
1.227 Å 最小键长不变量；PROTAC ERα–VHL(`9SV3` 语境) 用精确形式拓扑滤子刚化 linker
轨迹避界面碰撞；大环 FLT3(`4XUF` 语境) 16-bit 旋量投影 + 曲率可容许评估解环张力而
不破坏离散闭环不变量。以上为**真实结构上的方法演示，非实测位姿、非结合证据**，
`clinical_grade=false`。

**0/14 → 14/14 及其边界**：附录 A 中，确定性几何基座检测器（射线遮蔽度 + **空腔体积
排序**：结合口袋是*最大封闭空腔*而非单点最深缝隙；无配体输入、无学习、无随机）现取得
**14/14** Top-1 DCA ≤ 4 Å（best_dca 0.59–2.53 Å），与概率 SOTA（P2Rank 亦 14/14）持平。
诚实边界：(1) 相较仅用遮蔽度的消融（0/14），0→14 的全部增益来自**排序几何**（单点遮蔽度
vs 封闭空腔体积），是无拟合参数的固定泛函；(2) 我们**拒绝**用统计启发式软化布尔体素墙来
凑命中；(3) 划分**非** cluster-disjoint、单一保守 ESR1 折叠，故两个 14/14 都只是保守位点的
口袋*回收*，**非**方法优越、**非**分布外泛化、**非**结合/亲和/疗效证据，`clinical_grade=false`。
因此决定性检验在 cluster-disjoint 的 apo/隐匿口袋基准（见 `docs/BENCHMARK_SELECTION.md`）。

**核心方法（代数→结构，O(1)）**：不做概率搜索，而是用突变等位基因的精确代数来
约束候选设计（旗舰方法见 `paper/GF4_SYNDROME_CHEM_METHOD.tex`）：
(1) 碱基映射到有限域 GF(4)；(2) 以密码子为中心的 48-mer 与野生型做域加，得到
**稀疏等位残差 δ**（背景位点瞬间抵消，仅突变位点存活）；(3) 用**综合征方程
Hx=Bδ** 约束候选张量 x（x 是条件证据，不是唯一化学逆解，废弃平凡恒等
T(s)=s+α）；(4) 背景条件化 **Toeplitz 算子 F/G** 把 47-mer 基因组背景写入算子，
使共享同一原始残差的不同等位（如 KRAS G12C 与 FLT3 D835Y）被导向不碰撞的解空间。
随后把代数投影到化学（SMILES、价键图、药效团）与 3D 口袋几何 / 范德华墙，每种
模态各有独立协议。该方法只编码**代数条件化**，不主张唯一分子逆解、结合亲和力、
立体避让、可合成性、新颖性或临床性能。

**模态协议不可混用**：小分子用单口袋；PROTAC 用三元复合物；分子胶需声明伙伴
界面；大环只用有匹配墙体与多构象/环张力包的结构。

**当前物化证据**：KRAS G12D 非共价小分子单元，接受 **798** 个身份（Tanimoto
≤0.8、Murcko 上限 10，硬门、不强制凑满 1,000）；主几何 `9BL0`，`9GBJ` 仅共价
观察；口袋墙体用逐元素 Bondi(1964) 范德华半径 + 固定阈值碰撞分级（几何观察，非
亲和力）；对全部 798 做 RDKit 结构预警 + hERG(`5VA1`) 几何相容性探针；curated
top-24（`evidence/kras_g12d_sm/`）附中英双语原理、来源 URL、SHA-256 与二维结构，
图在 `figures/`。全量证据在**本地私有证据树** `gf4-allele-conditioned-evidence/`
并镜像 iCloud（附 SHA-256）。

**附录 A（ESR1 口袋基准）**：修复了配体标签只按名合并所有晶体拷贝的缺陷（4Q50
`OHT` 29→232≈8 份），已按链选单一实例并对 15 个结构全部重建标签；旧数值标记
`INVALIDATED_PENDING_LABEL_REGENERATION` 不可引用，用 `tools/run_pilot.py` 在
修正标签上重跑；split 尚非 cluster-disjoint，故不作比较性优越主张，
`clinical_grade=false`。
