# GF(4) Allele-Conditioned Computational Chemistry

*Multitarget, multimodal computational chemistry for oncology driver alleles.*

`clinical_grade=false`

A single-repository, falsifiable computational-chemistry program for **oncology
driver alleles**. The scientific target is a tumor driver mutation (KRAS
G12C/G12D/G12V, ESR1 LBD mutants, FLT3, PIM1, PIK3CA, CDK4/6), not a
regenerative-medicine or cell-state program. It is multitarget and multimodal.
This README describes only the **current** state; internal version history is
kept local and unpublished.

## The idea: algebra → structure, in O(1)

Instead of a probabilistic search, we condition candidate design on the exact
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
pharmacophores — and to 3D pocket geometry / van der Waals walls, where each
modality gets its own protocol.

The method encodes **algebraic conditioning only**. It does not claim a unique
molecular inverse, docking affinity, steric clearance, synthesis, novelty, or
clinical performance.

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
| P2Rank 2.5.1 | 14 |
| fpocket 4.0 | 0 |
| random (bbox) | 0 |
| Foliation (burial) | 0 |
| Foliation + geometric manifold prior | 0 |
| Foliation + manifold prior + exact-form filter | 0 |

See `figures/fig_baseline_comparison.png` and `figures/BENCHMARK_SUMMARY.json`.

**Honest reading.** The splits are **not cluster-disjoint** — one conserved
ESR1 LBD cluster spans development/validation/locked_test (24 cross-split
related pairs). P2Rank recovering the pocket on all 14 is therefore *pocket
recovery of a single conserved site*, **not** method superiority, and it also
shows the corrected labels are geometrically sound (the prior P2Rank number was
not merely a label-merge artifact). A geometric manifold prior and an
exact-form filter re-rank the Foliation candidates (changing mean Top-1 DCA)
but recover **no** Top-1 hit; the bottleneck is candidate *generation* — only
1/14 structures has any Foliation candidate within 4 Å in the top-20 pool
(oracle ceiling). No comparative-superiority claim is made; `clinical_grade=false`.

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

## Claim boundary

- Multitarget and multimodal; the paper does not collapse all targets into one
  ESR1 pocket score.
- The GF(4) method encodes algebraic conditioning; it does not claim a unique
  molecular inverse, affinity, or therapeutic effect.
- Docking scores are not affinity, efficacy, or therapeutic evidence.
- The Appendix A dataset is not a cluster-disjoint locked holdout.
- Regenerative-medicine / cell-state programs belong in separate repositories.

## Sources

- ESR1 UniProt: https://www.uniprot.org/uniprotkb/P03372/entry
- KRAS G12D binary: https://files.rcsb.org/download/9BL0.pdb
- KRAS G12C: https://files.rcsb.org/download/6OIM.pdb
- ESR1 example: https://files.rcsb.org/download/3ERT.pdb
- hERG cryo-EM: https://files.rcsb.org/download/5VA1.pdb
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

**核心思想（代数→结构，O(1)）**：不做概率搜索，而是用突变等位基因的精确代数来
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
