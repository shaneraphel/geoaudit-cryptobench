# Chemistry Dictionary — KRAS G12D Top-24
# 化学词典 — KRAS G12D Top-24

**Schema:** `gf4cc.chemistry_dictionary.v1` · **clinical_grade:** `false`  
**Source:** `evidence/kras_g12d_sm/CURATED_TOP24.json` (`candidates`, N=24)  
**Computation:** RDKit 2026.03.3 — all values recomputed from each candidate’s `canonical_smiles` (no fabricated structures).

**Truth boundary / 真实性边界:** Computed structural and physicochemical description only. Not binding, affinity, efficacy, safety, or developability evidence.  
仅结构与理化计算描述；不构成结合、亲和力、疗效、安全性或可开发性证据。

Machine-readable twin: [`CHEMISTRY_DICTIONARY.json`](CHEMISTRY_DICTIONARY.json).

---

## (a) Glossary / 术语表

Each term: precise definition · why it matters for druggability / metabolism / tox (descriptive common sense, not a claim about these candidates).

| Term 术语 | Definition 定义 | Why it matters 为何重要 |
|---|---|---|
| **Chirality / stereocenter 手性 / 立体中心** | A tetrahedral atom (usually carbon) with four different substituents, yielding non-superimposable mirror-image configurations. Counted here with RDKit `FindMolChiralCenters(..., includeUnassigned=True)`. 具有四个不同取代基的四面体原子（常为碳），产生不可重合的镜像构型。 | Enantiomers can differ in target binding, off-target profile, and metabolic clearance; unspecified stereo blocks reproducible synthesis and assay interpretation. 对映体可在靶点结合、脱靶与代谢清除上不同；立体未指定则合成与实验不可复现。 |
| **R/S (Cahn–Ingold–Prelog)** | Absolute configuration labels for a stereocenter under CIP priority rules (R = rectus / clockwise; S = sinister / counterclockwise when the lowest-priority substituent points away). 立体中心的绝对构型标签（CIP 优先级：最低优先级指向远离观察者时，顺时针为 R，逆时针为 S）。 | Assigned R/S makes the chemical identity unambiguous for peer review and lot-to-lot comparison. 指定 R/S 使化学身份明确，便于同行评议与批次比对。 |
| **E/Z (alkene stereo)** | Configuration about a stereogenic double bond: E (entgegen, opposite) vs Z (zusammen, together) by CIP ranking of substituents on each sp² carbon. 立体双键构型：按 CIP 对两端取代基排序后的对侧 (E) / 同侧 (Z)。 | Geometric isomers can change shape and polarity enough to alter binding and ADME; unspecified E/Z is an identity gap. 几何异构可改变形状与极性，影响结合与 ADME；E/Z 未指定即身份缺口。 |
| **Murcko scaffold 骨架** | Bemis–Murcko framework: ring systems plus linker atoms between rings, with side-chain substituents removed (RDKit `MurckoScaffoldSmiles`). 去掉侧链后保留环系及环间连接原子的骨架。 | Groups analogs by core topology for series analysis without claiming bioisosterism. 按核心拓扑归类类似物，便于系列分析（不宣称生物电子等排）。 |
| **Generic (framework) scaffold 通用骨架** | Murcko scaffold with all heteroatoms converted to carbon and bond orders normalized (RDKit `MakeScaffoldGeneric`) — a pure connectivity framework. 将杂原子改为碳并归一化键级后的纯连接框架。 | Separates “shape of the ring graph” from heteroatom decoration when comparing chemotypes. 比较化学型时区分环图形状与杂原子装饰。 |
| **Ring system 环系** | A set of rings sharing atoms (SSSR rings; sizes listed per ring). Fused systems share ≥1 atom; spiro share exactly one atom. 共享原子的环集合（SSSR；记录各环大小）。稠环共享≥1原子；螺环恰共享1原子。 | Ring count/size modulate rigidity, solubility, and CYP recognition surfaces. 环数/大小影响刚性、溶解性与 CYP 识别面。 |
| **Spiro 螺环** | Two rings share exactly one atom (the spiro atom). Counted with `CalcNumSpiroAtoms`. 两环仅共享一个原子（螺原子）。 | Spiro centers add 3D bulk and can reduce flatness (often linked to solubility / promiscuity heuristics). 螺中心增加三维体积，可降低平面性（常与溶解性/杂泛性启发式相关）。 |
| **Fused 稠环** | At least one atom belongs to two or more SSSR rings. 至少一个原子属于两个或以上 SSSR 环。 | Fusion increases rigidity and aromatic surface; can raise stacking / clearance liabilities. 稠合增加刚性与芳香表面，可能提高堆积/清除相关风险启发式。 |
| **Macrocycle 大环** | Any ring with size ≥ 12 atoms (flagged in this dictionary). 任一环原子数 ≥ 12。 | Large rings change permeability and conformational ensembles versus typical small-ring drugs. 大环在通透性与构象系综上常不同于典型小环药物。 |
| **Heteroatom 杂原子** | Non-carbon, non-hydrogen heavy atom (N, O, S, F, Cl, …). 非碳非氢重原子。 | Heteroatoms drive H-bonding, ionization, metabolism (e.g. N/S oxidation), and tox alerts. 杂原子驱动氢键、电离、代谢（如 N/S 氧化）与毒性警示启发式。 |
| **FractionCSP3** | Fraction of carbon atoms that are sp³ hybridized (`Descriptors.FractionCSP3`). sp³ 杂化碳占总碳的比例。 | Higher Fsp³ often correlates with greater 3D character and, in some datasets, improved developability metrics (heuristic only). 较高 Fsp³ 常与更强三维性相关，在一些数据集中与可开发性指标改善相关（仅为启发式）。 |
| **Aniline / arylamine 苯胺 / 芳胺** | Amino group attached to a carbocyclic aromatic carbon (`[NX3]-c`), excluding amides. 氨基连在碳环芳香碳上（排除酰胺）。 | Classic anilines are associated with metabolic activation / reactive-metabolite heuristics; flag for awareness, not a verdict. 经典苯胺与代谢活化/活性代谢物启发式相关；仅作标记而非判决。 |
| **Aminoheteroarene 氨基杂芳** | Amino group on an aromatic ring that contains a heteroatom (e.g. aminopyridine, aminobenzothiophene). 氨基连在含杂原子的芳环上。 | Often used as hinge-/pocket-facing motifs; tautomerism and basicity differ from aniline. 常用作朝向口袋的母核；互变异构与碱性不同于苯胺。 |
| **Aminal / N,O-acetal** | Motif `N–C–O` with the carbon tetrahedral, here required on-ring (`[NX3][CX4][OX2]` and C in a ring) — includes hemiaminal-ether / N,O-acetal-like bridgeheads. 四面体碳上的 N–C–O 片段，且该碳在环上。 | Can be hydrolytically labile under acid; relevant to chemical stability / formulation common sense. 酸性下可能水解不稳定；与化学稳定性/制剂常识相关。 |
| **Nitrile 腈** | Carbon–nitrogen triple bond (`C≡N`). 碳氮三键。 | Compact H-bond acceptor; can be metabolized (nitrilase) or used as a bioisostere of carbonyls. 紧凑氢键受体；可被代谢或作羰基生物电子等排体。 |
| **TPSA** | Topological polar surface area (Å²) from polar fragment contributions (`Descriptors.TPSA`). 由极性片段贡献估算的拓扑极性表面积。 | Strong empirical correlate of passive permeability and efflux propensity (Veber-style heuristics). 与被动通透及外排倾向的经验相关（Veber 类启发式）。 |
| **HBD / HBA** | Hydrogen-bond donor / acceptor counts (Lipinski definitions via RDKit `NumHDonors` / `NumHAcceptors`). 氢键给体/受体计数。 | Excess donors/acceptors can hurt permeability; too few can weaken soluble recognition. 过多损害通透；过少可能削弱可溶识别。 |
| **QED** | Quantitative Estimate of Drug-likeness (Bickerton et al.; RDKit `QED.qed`) — a composite desirability score in [0,1]. 定量类药性估计（[0,1] 综合合意度分数）。 | Ranking aid for multi-property desirability; **not** a proof of activity or safety. 多性质合意度排序辅助；**不是**活性或安全性证明。 |
| **Lipinski (Ro5)** | Rule-of-five: MW≤500, cLogP≤5, HBD≤5, HBA≤10; pass = zero violations (this file recounts violations). 五规则；本文件重新计违规数，零违规为通过。 | Oral-space heuristic only; many useful molecules violate it. 仅为口服空间启发式；许多有用分子并不满足。 |
| **Veber** | Rotatable bonds ≤ 10 and TPSA ≤ 140 Å² (pass/fail in this file). 可旋转键 ≤ 10 且 TPSA ≤ 140。 | Oral bioavailability heuristic focused on flexibility and polarity. 聚焦柔性与极性的口服生物利用度启发式。 |

Additional tags used in the JSON (boolean SMARTS / ring checks): ether, alcohol, phenol, fluoroalkyl (aliphatic C–F), amide, amine 1°/2°/3°, aromatic heterocycle type list (pyridine, thiophene, quinazoline, …).  
JSON 中另有：醚、醇、酚、氟代烷基（脂肪族 C–F）、酰胺、1°/2°/3°胺、芳杂环类型列表。

---

## (b) Per-candidate summary / 逐候选摘要

† = `stereochemically_underspecified` (unassigned stereocenter and/or unspecified stereo double bond in the **canonical_smiles** used for computation).  
† = 用于计算的 **canonical_smiles** 中存在未指定立体中心和/或未指定立体双键。

| rank | id (InChIKey prefix) | scaffold class | #stereo (unassigned) | key functional groups | MW | cLogP | TPSA | QED | Lipinski | Veber |
|---:|---|---|---:|---|---:|---:|---:|---:|---|---|
| 1 | `LSDREBDFWDRPJZ` | isoquinoline+pyrrolidine | 2 (2)† | aminoheteroarene, ether, aminal/N,O-acetal, fluoroalkyl, 1° amine, 3° amine | 301.365 | 2.8675 | 51.38 | 0.9469 | pass | pass |
| 2 | `FLRZLMYSWNATQP` | benzothiophene+pyrrolidine | 1 (1)† | aminoheteroarene, nitrile, ether, aminal/N,O-acetal, 1° amine, 3° amine | 305.378 | 3.0626 | 62.28 | 0.9464 | pass | pass |
| 3 | `JTQKZCJFKLCOKZ` | naphthol+morpholine | 0 (0) | ether, phenol, aminal/N,O-acetal, 3° amine | 299.37 | 3.0549 | 41.93 | 0.9456 | pass | pass |
| 4 | `OZQGVDAAVLZDIZ` | isoquinoline+pyrrolidine | 2 (2)† | aminoheteroarene, ether, aminal/N,O-acetal, fluoroalkyl, 1° amine, 3° amine | 319.355 | 3.0066 | 51.38 | 0.9447 | pass | pass |
| 5 | `GQSHKRUHOFIMOU` | benzothiazole+pyrrolidine | 2 (2)† | aminoheteroarene, ether, aminal/N,O-acetal, fluoroalkyl, 1° amine, 3° amine | 325.384 | 3.0681 | 51.38 | 0.9422 | pass | pass |
| 6 | `DQLYEHADLXHFLQ` | naphthol+morpholine | 0 (0) | ether, phenol, 3° amine | 313.397 | 3.0974 | 41.93 | 0.9416 | pass | pass |
| 7 | `SIKXDYXRZOXFDX` | naphthol+morpholine | 0 (0) | ether, phenol, aminal/N,O-acetal, 3° amine | 313.397 | 3.1927 | 41.93 | 0.9411 | pass | pass |
| 8 | `SFGMSYUCFPPLEQ` | naphthol+pyrrolidine | 2 (2)† | ether, phenol, aminal/N,O-acetal, fluoroalkyl, 3° amine | 301.361 | 3.5959 | 32.7 | 0.94 | pass | pass |
| 9 | `IBQXRHJYHYFNOG` | pyridine+piperidine | 0 (0) | amide, 1° amine | 330.407 | 2.457 | 72.11 | 0.9382 | pass | pass |
| 10 | `OVNHSYBFROGGJB` | naphthol+pyrrolidine | 2 (2)† | ether, phenol, aminal/N,O-acetal, fluoroalkyl, 3° amine | 319.351 | 3.735 | 32.7 | 0.9354 | pass | pass |
| 11 | `HWZAIONOWFFRIX` | naphthol+pyrrolidine | 0 (0) | ether, phenol, aminal/N,O-acetal, 3° amine | 283.371 | 3.6479 | 32.7 | 0.933 | pass | pass |
| 12 | `UCYFDVLKOVGHOA` | pyridine+pyrrolidine | 0 (0) | aminoheteroarene, ether, aminal/N,O-acetal, fluoroalkyl, 1° amine, 3° amine | 315.339 | 3.0935 | 51.38 | 0.9311 | pass | pass |
| 13 | `LSAKXHWSTHUNMP` | pyridine+pyrrolidine | 0 (0) | aminoheteroarene, ether, aminal/N,O-acetal, fluoroalkyl, 1° amine, 3° amine | 315.339 | 3.0935 | 51.38 | 0.9311 | pass | pass |
| 14 | `BXGVXIAUXSHRGG` | naphthol+pyrrolidine | 1 (1)† | ether, phenol, aminal/N,O-acetal, 3° amine | 285.387 | 3.6761 | 32.7 | 0.9299 | pass | pass |
| 15 | `FXCSXWXNOXOGRQ` | pyridine+pyrrolidine | 1 (1)† | aminoheteroarene, ether, aminal/N,O-acetal, fluoroalkyl, 1° amine, 3° amine | 289.301 | 2.5593 | 51.38 | 0.9288 | pass | pass |
| 16 | `QOGITRBIAHREIY` | pyridine+pyrrolidine | 1 (1)† | aminoheteroarene, ether, aminal/N,O-acetal, fluoroalkyl, 1° amine, 3° amine | 289.301 | 2.5593 | 51.38 | 0.9288 | pass | pass |
| 17 | `XGMXXRODSZKRKH` | isoquinoline+pyrrolidine | 2 (2)† | aminoheteroarene, ether, aminal/N,O-acetal, fluoroalkyl, 1° amine, 3° amine | 305.328 | 2.8688 | 51.38 | 0.9267 | pass | pass |
| 18 | `JTOSVRRDMQTNDA` | benzothiazole+pyrrolidine | 2 (2)† | aminoheteroarene, ether, aminal/N,O-acetal, fluoroalkyl, 1° amine, 3° amine | 311.357 | 2.9303 | 51.38 | 0.9263 | pass | pass |
| 19 | `KXXDRSLBYYCTMX` | pyridine+pyrrolidine | 2 (2)† | aminoheteroarene, ether, aminal/N,O-acetal, 1° amine, 3° amine | 304.438 | 2.3072 | 54.62 | 0.9249 | pass | pass |
| 20 | `UOCPMXIGUGZOLU` | pyridine+pyrrolidine | 2 (2)† | aminoheteroarene, ether, aminal/N,O-acetal, 1° amine, 3° amine | 304.438 | 2.3072 | 54.62 | 0.9249 | pass | pass |
| 21 | `LGPVGJPISCANIW` | pyridine+pyrrolidine | 2 (2)† | aminoheteroarene, ether, aminal/N,O-acetal, 1° amine, 3° amine | 290.411 | 2.1694 | 54.62 | 0.9244 | pass | pass |
| 22 | `NPVNTVODHGEPEH` | benzothiophene+pyrrolidine | 1 (1)† | aminoheteroarene, nitrile, ether, aminal/N,O-acetal, 1° amine, 3° amine | 291.351 | 2.9248 | 62.28 | 0.9238 | pass | pass |
| 23 | `OFNCRXQRGNFZJI` | naphthol+pyrrolidine | 2 (2)† | ether, phenol, aminal/N,O-acetal, fluoroalkyl, 3° amine | 337.341 | 3.8741 | 32.7 | 0.9227 | pass | pass |
| 24 | `DJQMDUSVZGPTOJ` | naphthol+morpholine | 0 (0) | ether, phenol, aminal/N,O-acetal, 3° amine | 339.435 | 3.6818 | 41.93 | 0.9225 | pass | pass |

Full IDs follow `gpt56v91-sm-krasg12d-XXXXX-<InChIKey14>` in the JSON.  
完整 ID 见 JSON 中的 `candidate_id`。

---

## (c) Aggregates / 汇总

### Scaffold-class distribution / 骨架类别分布
- `pyridine+pyrrolidine`: 7/24
- `naphthol+pyrrolidine`: 5/24
- `naphthol+morpholine`: 4/24
- `isoquinoline+pyrrolidine`: 3/24
- `benzothiazole+pyrrolidine`: 2/24
- `benzothiophene+pyrrolidine`: 2/24
- `pyridine+piperidine`: 1/24

### Chirality distribution / 手性分布
- all_stereocenters_unassigned: 16/24
- achiral_0_centers: 8/24

- **Stereochemically underspecified:** **16/24** (important for peer-review reproducibility — RDKit canonicalization plus explicit stereo are required for bit-identical chemical identity).  
  **立体化学欠指定：** **16/24**（同行评议可复现性关键：RDKit 规范化与显式立体必须齐全，化学身份才能逐位一致）。
- Achiral (0 stereocenters): 8/24.
- Fully assigned R/S in canonical SMILES: 0/24.
- Note: several records also store `isomeric_smiles` with assigned stereo in the source curated file; this dictionary intentionally scores underspecification on the published **`canonical_smiles`** field so that field’s reproducibility is auditable.  
  注：策展源文件中部分记录另有带立体的 `isomeric_smiles`；本词典故意对公开的 **`canonical_smiles`** 字段计欠指定，以便审计该字段的可复现性。

### Element / heteroatom distribution / 元素与杂原子分布
- C: present in 24/24
- N: present in 24/24
- O: present in 24/24
- F: present in 15/24
- S: present in 4/24
- Mean heteroatom count (heavy, non-C): **5.375**
- F present in 15/24; S present in 4/24; Cl present in 0/24; every candidate has N and O.

### Drug-likeness recount / 类药性重算
- MW discrepancy vs curated input (`\|ΔMW\| > 1.0`): **0/24**
- All 24: Lipinski pass; all 24: Veber pass (recomputed).

---

## SMILES / stereo reproducibility note / SMILES 与立体可复现性说明

**Method:** for each candidate, `m = MolFromSmiles(canonical_smiles)`, `s1 = MolToSmiles(m)`, `s2 = MolToSmiles(MolFromSmiles(s1))`. Round-trip success requires `s1 == s2`.

| Check | Result |
|---|---|
| Round-trip failures (`s1 ≠ s2`) | **0** |
| Stored `canonical_smiles` ≠ RDKit `MolToSmiles` reordering | **0** |
| Stereochemically underspecified under canonical SMILES | **16/24** |

**Conclusion for a “does your chemistry reproduce?” review question:** every one of the 24 `canonical_smiles` strings round-trips identically through RDKit (`s1 == s2`). No SMILES failed round-trip; none lost additional stereo upon re-canonicalization beyond what the stored canonical string already omitted. Where `stereochemically_underspecified=true`, peer-reproducible absolute configuration requires the source `isomeric_smiles` (or an explicitly stereo-annotated SDF), not the stereo-stripped canonical string alone. InChIKeys recomputed from canonical (often stereo-free) SMILES therefore can differ from stereo-aware `inchi_key` values stored in the curated input — both are recorded in the JSON.

**结论（回应“化学能否复现”）：** 24 条 `canonical_smiles` 均经 RDKit 往返一致（`s1 == s2`）。无往返失败；除存储的 canonical 字符串本身已省略的立体外，再规范化未额外丢失立体。当 `stereochemically_underspecified=true` 时，对等可复现的绝对构型需依赖源 `isomeric_smiles`（或显式立体 SDF），不能仅靠去立体的 canonical 字符串。由（常无立体的）canonical SMILES 重算的 InChIKey 可能不同于策展输入中带立体的 `inchi_key`——二者均写入 JSON。

---

*Generated for descriptive chemistry inventory only. `clinical_grade=false`.*
