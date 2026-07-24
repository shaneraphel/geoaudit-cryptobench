# Toxicology Causal Reasoning Report — KRAS G12D Noncovalent Top-10
# 毒理学因果推理报告 — KRAS G12D 非共价 Top-10

**clinical_grade=false** · Liability **hypotheses** only · Not safety clearance  
**Scope:** KRAS G12D oncology small molecules from `CURATED_TOP24.json` ranks 1–10  
**Structural alerts:** RDKit FilterCatalog **Brenk + PAINS** recomputed in this session (all 10: **0 catalog hits**)  
**Note:** Absence of Brenk/PAINS flags does **not** clear mechanistic soft-spot liabilities below.

---

## Methods / 方法

- Identity source: `evidence/kras_g12d_sm/CURATED_TOP24.json` (`candidates`, ranks 1–10).
- Alerts: `rdkit.Chem.FilterCatalog` with `PAINS` + `BRENK`.
- Causal chains: CYP soft-spot → reactive intermediate → organ liability **hypotheses**, with peer-reviewed mechanism citations.
- Quantum cross-check (ranks 1–3 + repairs): see **Quantum DFT verification** at end (`DFT_QUANTUM_METRICS.json`).

---

## Batch summary table / 批次总表

| Rank | Short ID | Motif liability focus (EN) | 主要结构风险（中文） | Brenk/PAINS |
|------|----------|----------------------------|----------------------|-------------|
| 1 | 00379 | Azabicyclic N,O-aminal ether + 3-aminoisoquinoline | 氮杂双环 N,O-缩醛醚 + 3-氨基异喹啉 | 0 |
| 2 | 02621 | 2-Aminothiophene-3-carbonitrile + benzylic ether | 2-氨基噻吩-3-腈 + 苄醚 | 0 |
| 3 | 00553 | Cyclopropane N,O-aminal (morpholine) + naphthol | 环丙烷 N,O-缩醛（吗啉）+ 萘酚 | 0 |
| 4 | 01654 | Same scaffold as 00379 + aryl-F | 同 00379 骨架 + 芳氟 | 0 |
| 5 | 02323 | 2-Aminobenzothiazole + azabicyclic aminal ether | 2-氨基苯并噻唑 + 氮杂双环缩醛醚 | 0 |
| 6 | 00552 | Morpholinomethyl-cyclopropoxy naphthol (related aminal) | 吗啉甲基环丙氧基萘酚（相关缩醛） | 0 |
| 7 | 00532 | Cyclopropane N,O-aminal via CH2O linker + naphthol | CH2O 连接的环丙烷缩醛 + 萘酚 | 0 |
| 8 | 01537 | Naphthol + azabicyclic aminal ether | 萘酚 + 氮杂双环缩醛醚 | 0 |
| 9 | 00363 | 4-Aminopiperidine amide + fluoro-naphthyridine | 4-氨基哌啶酰胺 + 氟代萘啶 | 0 |
| 10 | 01355 | Fluoronaphthol + azabicyclic aminal ether | 氟萘酚 + 氮杂双环缩醛醚 | 0 |

---

## Rank 1 — `gpt56v91-sm-krasg12d-00379-LSDREBDFWDRPJZ`

**SMILES:** `Nc1cc2ccccc2c(COC23CCCN2CC(F)C3)n1`  
**InChIKey:** `LSDREBDFWDRPJZ-CXAGYDPISA-N` · MW 301.37 · cLogP 2.87 · TPSA 51.4 · QED 0.947  
**Brenk/PAINS (recomputed):** 0 hits.

### Metabolism (CYP soft spots)

**EN.** Two orthogonal soft spots:
1. **Bridgehead N,O-aminal ether** (`…COC23CCCN2…`): the carbon bonded to both **O** and **tertiary N** is acid- and CYP-vulnerable. CYP3A4/2D6-mediated single-electron oxidation at the tertiary amine, or direct hydrolytic cleavage of the aminal, yields an **amino-aldehyde / hemiaminal** pair and liberates the isoquinolinyl-CH2OH fragment — a high hepatic clearance hypothesis ([Guengerich, Chem. Res. Toxicol. 2001](https://pubmed.ncbi.nlm.nih.gov/11266157/); aminal/hemiaminal instability in drug design [Stachulski & Lennard, *J. Med. Chem.* reviews on reactive metabolites](https://pubmed.ncbi.nlm.nih.gov/17722901/)).
2. **3-Aminoisoquinoline:** CYP1A2/3A4 aromatic hydroxylation or **N-oxidation** of the primary aminoheteroarene can generate a **nitrenium-like** electrophile after sulfate/acetate conjugation (classic aromatic amine bioactivation; [Miller & Miller; Kalgutkar et al., Chem. Res. Toxicol.](https://pubmed.ncbi.nlm.nih.gov/15606127/)).

**Phase II:** Glucuronidation of any phenolic metabolite after ring hydroxylation; possible N-glucuronidation of the aminoisoquinoline.  
**Hepatic clearance hypothesis:** Combined aminal cleavage + oxidative N-dealkylation → **high CL_h**, short half-life unless formulation/prodrug strategies intervene (computational hypothesis only).

**中文.** （1）桥头 **N,O-缩醛醚** 可被酸解或 CYP 氧化裂解，释放醛/半缩醛与异喹啉甲醇片段，推测肝清除偏高；（2）**3-氨基异喹啉** 可经 CYP N-氧化/芳羟化后形成类硝鎓亲电体，关联肝毒性/遗传毒性假说。II 相以葡糖醛酸化为主。

### TME (pH ~6.5 / hypoxia)

**EN.** Tertiary bridgehead amine pKa typically ~8–9 → at pH 6.5 a larger **cationic fraction**, increasing trapping in acidic compartments but also accelerating **acid-catalyzed aminal hydrolysis**. Hypoxia does not reductively activate this scaffold (no nitro/quinone latent warhead); main TME risk is **chemical instability of the aminal**, not bioreductive activation.

**中文.** 桥头叔胺在 pH 6.5 质子化比例上升，酸催化缩醛水解加速；无氧下无典型硝基/醌还原活化。

### Reactive-intermediate chain

**EN.** Parent → (H+/CYP) aminal cleavage → **aldehyde** + amino-alcohol · **and/or** aminoisoquinoline → N-OH → O-conjugation → **nitrenium** → protein/DNA adducts (hepatotoxicity / genotoxicity hypotheses). hERG: basic amine + aromatic surface is a **structural** hERG alert class ([Sanguinetti & Tristani-Firouzi](https://pubmed.ncbi.nlm.nih.gov/16860700/)) — hypothesis only.

**中文.** 缩醛 → 醛；氨基异喹啉 → 硝鎓；碱性胺+芳环构成 hERG 结构假说风险。

**Citations:** [https://www.rcsb.org/structure/9BL0](https://www.rcsb.org/structure/9BL0) · [CHEMBL5418176](https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL5418176) · [PubMed 15606127](https://pubmed.ncbi.nlm.nih.gov/15606127/) · [PubMed 11266157](https://pubmed.ncbi.nlm.nih.gov/11266157/)

---

## Rank 2 — `gpt56v91-sm-krasg12d-02621-FLRZLMYSWNATQP`

**SMILES:** `CN1CCCC1OCc1ccc(F)c2sc(N)c(C#N)c12`  
**InChIKey:** `FLRZLMYSWNATQP-GFCCVEGCSA-N` · MW 305.38 · cLogP 3.06 · TPSA 62.3 · QED 0.946  
**Brenk/PAINS:** 0 hits (**catalog silent** on 2-aminothiophene-3-carbonitrile — still a severe mechanistic motif).

### Metabolism

**EN.**
1. **Thiophene S-oxidation (CYP2C9/3A4/FMO):** → **thiophene-S-oxide** → electrophilic addition / ring-opened thiols; documented hepatotoxic bioactivation path for thiophenes ([Dansette et al.; Gramec et al.](https://pubmed.ncbi.nlm.nih.gov/24655145/); [Kalgutkar, Curr. Drug Metab.](https://pubmed.ncbi.nlm.nih.gov/16101570/)).
2. **2-Amino group:** peroxidase/CYP → N-oxidation → **nitrenium** (aminoheteroarene).
3. **N-methylpyrrolidine:** CYP2D6/3A4 **N-dealkylation** / α-carbon hydroxylation → iminium; benzylic ether may undergo O-dealkylation to aldehyde + alcohol.

**Phase II / CL_h:** GSH trapping of S-oxide and nitrenium expected; high predicted metabolic turnover.

**中文.** 噻吩 **S-氧化** → 噻吩亚砜亲电体；2-氨基 → 硝鎓；N-甲基吡咯烷 N-脱烷基；综合肝清除与 GSH 耗竭假说。

### TME

**EN.** N-methylpyrrolidine protonation ↑ at pH 6.5. Hypoxia: no classic nitro-reductive trigger; electron-rich aminothiophene may still undergo **oxidative stress–coupled** activation in inflammatory TME (hypothesis). Nitrile is metabolically inert relative to the thiophene/amine.

**中文.** 酸性微环境增加胺质子化；低氧不典型还原活化，但缺电子应激下氨基噻吩仍可被氧化活化。

### Reactive-intermediate chain

**EN.** Parent → thiophene-**S-oxide** (hepatotoxicity) **and** amino → **nitrenium** (genotoxicity/hepatotoxicity). The **2-aminothiophene-3-carbonitrile** motif is a known privileged-but-risky chemotype in screening libraries ([Baell & Holloway PAINS context](https://pubmed.ncbi.nlm.nih.gov/20131845/) — note: PAINS SMARTS may miss this exact pattern; mechanism still stands).

**中文.** 双通道：噻吩亚砜 + 氨基硝鎓；该母核为高优先级剔除假说。

**Citations:** [CHEMBL5430198](https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL5430198) · [PubMed 24655145](https://pubmed.ncbi.nlm.nih.gov/24655145/) · [PubMed 16101570](https://pubmed.ncbi.nlm.nih.gov/16101570/) · [9BL0](https://www.rcsb.org/structure/9BL0)

---

## Rank 3 — `gpt56v91-sm-krasg12d-00553-JTQKZCJFKLCOKZ`

**SMILES:** `Cc1cccc2cc(O)cc(OC3(N4CCOCC4)CC3)c12`  
**InChIKey:** `JTQKZCJFKLCOKZ-UHFFFAOYSA-N` · MW 299.37 · cLogP 3.05 · TPSA 41.9 · QED 0.946  
**Brenk/PAINS:** 0 hits.

### Metabolism

**EN.**
1. **Cyclopropane N,O-aminal** `OC3(N-morpholino)CC3`: the cyclopropyl carbon attached to **both O and N** is a textbook **acid-labile aminal ether**. Hydrolysis → morpholine + **cyclopropanone hemiacetal / ring-opened carbonyl** electrophile ([aminal prodrug literature](https://pubmed.ncbi.nlm.nih.gov/17722901/)).
2. **Naphthalen-2-ol:** CYP1A2/3A4 → catechol / **naphthoquinone** path → Michael acceptor (hepatotoxicity) ([Bolton et al., Chem. Res. Toxicol.](https://pubmed.ncbi.nlm.nih.gov/10725116/)).
3. Morpholine: CYP **N-oxidation** / oxidative ring opening (secondary).

**CL_h hypothesis:** Rapid chemical + oxidative clearance; quinone GSH depletion.

**中文.** 环丙烷 **N,O-缩醛** 酸解释放吗啉与羰基亲电体；萘酚氧化至 **萘醌**；肝清除与 GSH 耗竭假说。

### TME

**EN.** Phenol pKa ~9–10 → mostly neutral at 6.5; **aminal hydrolysis strongly accelerated** in acidic TME. Hypoxia: quinone/hydroquinone redox cycling can couple to hypoxic stress (hypothesis).

**中文.** 酸性 TME 显著加速缩醛水解；醌/氢醌可与低氧应激耦合。

### Reactive intermediates

**EN.** Aminal → carbonyl electrophile; phenol → **o-/p-quinone**. hERG less concerning (weak base morpholine, TPSA-limited). Genotoxicity via quinone-DNA adducts — hypothesis.

**Citations:** [CHEMBL4857517](https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL4857517) · [PubMed 10725116](https://pubmed.ncbi.nlm.nih.gov/10725116/) · [9BL0](https://www.rcsb.org/structure/9BL0)

---

## Rank 4 — `gpt56v91-sm-krasg12d-01654-OZQGVDAAVLZDIZ`

**SMILES:** `Nc1cc2cccc(F)c2c(COC23CCCN2CC(F)C3)n1` · **InChIKey:** `OZQGVDAAVLZDIZ-SJKOYZFVSA-N`  
**Brenk/PAINS:** 0.

**EN.** Mechanistically **isosteric to rank 1** with an extra aryl-F: same **N,O-aminal** cleavage → aldehyde; same **aminoisoquinoline → nitrenium** path. Aryl-F can slow certain aromatic hydroxylations but does **not** remove aminal or N-oxidation liabilities. TME: identical aminal acid-lability.

**中文.** 与 00379 同因果链；芳氟仅调节羟化速率，不消除缩醛/硝鎓风险。

**Citations:** [CHEMBL5426739](https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL5426739) · [PubMed 15606127](https://pubmed.ncbi.nlm.nih.gov/15606127/)

---

## Rank 5 — `gpt56v91-sm-krasg12d-02323-GQSHKRUHOFIMOU`

**SMILES:** `Nc1nc2c(COC34CCCN3CC(F)C4)ccc(F)c2s1` · **InChIKey:** `GQSHKRUHOFIMOU-MEBBXXQBSA-N`  
**Brenk/PAINS:** 0.

**EN.** Dual: (1) **azabicyclic N,O-aminal ether** (as 00379); (2) **2-aminobenzothiazole** — S-oxidation / amino → nitrenium / benzothiazole epoxidation-like pathways ([Kalgutkar](https://pubmed.ncbi.nlm.nih.gov/16101570/)). Hepatic clearance: aminal + heteroarene oxidation. TME: acid-labile aminal.

**中文.** 缩醛醚 + 2-氨基苯并噻唑（S-氧化/硝鎓）双风险。

**Citations:** [CHEMBL5397425](https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL5397425) · [PubMed 24655145](https://pubmed.ncbi.nlm.nih.gov/24655145/)

---

## Rank 6 — `gpt56v91-sm-krasg12d-00552-DQLYEHADLXHFLQ`

**SMILES:** `Cc1cccc2cc(O)cc(OC3(CN4CCOCC4)CC3)c12` · **InChIKey:** `DQLYEHADLXHFLQ-UHFFFAOYSA-N`  
**Brenk/PAINS:** 0.

**EN.** Related to 00553 but morpholine is attached via **CH2** to the cyclopropane (`OC3(CN4…)CC3`). This is an **O,C-cyclopropyl ether with β-amino**, not a classical N,O-aminal on the same carbon — **chemically more stable** than 00553, yet still strained; CYP may hydroxylate cyclopropane/benzylic positions. Dominant liability shifts to **naphthol → naphthoquinone**. TME: phenol ionization minor; no aminal acid bomb.

**中文.** 相对 00553 缩醛风险降低，萘酚→萘醌仍是主因果链。

**Citations:** [CHEMBL4857517](https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL4857517) · [PubMed 10725116](https://pubmed.ncbi.nlm.nih.gov/10725116/)

---

## Rank 7 — `gpt56v91-sm-krasg12d-00532-SIKXDYXRZOXFDX`

**SMILES:** `Cc1cccc2cc(O)cc(COC3(N4CCOCC4)CC3)c12` · **InChIKey:** `SIKXDYXRZOXFDX-UHFFFAOYSA-N`  
**Brenk/PAINS:** 0.

**EN.** Restores **true N,O-aminal** on cyclopropane (`COC3(N-morpholino)CC3`) plus naphthol. Same fatal pair as 00553: acid/CYP aminal rupture + quinone path. Benzylic CH2 may additionally form aldehyde after O-dealkylation.

**中文.** 与 00553 同类：环丙烷缩醛 + 萘醌双通道。

**Citations:** [CHEMBL4857517](https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL4857517)

---

## Rank 8 — `gpt56v91-sm-krasg12d-01537-SFGMSYUCFPPLEQ`

**SMILES:** `Oc1cc(COC23CCCN2CC(F)C3)c2ccccc2c1` · **InChIKey:** `SFGMSYUCFPPLEQ-YJBOKZPZSA-N`  
**Brenk/PAINS:** 0.

**EN.** **Naphthol + azabicyclic aminal ether** (no aminoheteroarene). Soft spots: aminal → aldehyde; phenol → quinone; tertiary amine α-oxidation. TME acid instability of aminal dominates PK. hERG: basic amine + lipophilic naphthalene (cLogP 3.60) — structural hypothesis ([PubMed 16860700](https://pubmed.ncbi.nlm.nih.gov/16860700/)).

**中文.** 缩醛不稳定 + 萘酚醌化 + 碱性胺 hERG 假说。

**Citations:** [CHEMBL5093274](https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL5093274) · [9BL0](https://www.rcsb.org/structure/9BL0)

---

## Rank 9 — `gpt56v91-sm-krasg12d-00363-IBQXRHJYHYFNOG`

**SMILES:** `CCc1ncc2c(CC)cc(C(=O)N3CCC(N)CC3)nc2c1F` · **InChIKey:** `IBQXRHJYHYFNOG-UHFFFAOYSA-N`  
**Brenk/PAINS:** 0.

**EN.** Distinct chemotype: **fluoro-naphthyridine carboxamide** to **4-aminopiperidine**. Soft spots: (1) primary aliphatic amine — MAO/CYP → iminium/aldehyde; (2) ethyl side-chain benzylic oxidation; (3) amide hydrolysis (chemical/enzymatic) → naphthyridine acid + aminopiperidine. No thiophene/aminal. Nitrenium risk lower than aromatic amines but **aliphatic amine oxidation** can still yield reactive iminiums ([Sayre et al.](https://pubmed.ncbi.nlm.nih.gov/16651726/)). TME: amine fully protonated at 6.5 → permeability drop / lysosomal trapping. hERG: classic basic amine risk.

**中文.** 无缩醛/噻吩；主风险为脂肪胺氧化亚铵、酰胺水解与 hERG。

**Citations:** [CHEMBL6166393](https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL6166393) · [PubMed 16651726](https://pubmed.ncbi.nlm.nih.gov/16651726/)

---

## Rank 10 — `gpt56v91-sm-krasg12d-01355-OVNHSYBFROGGJB`

**SMILES:** `Oc1cc(COC23CCCN2CC(F)C3)c2c(F)cccc2c1` · **InChIKey:** `OVNHSYBFROGGJB-RDTXWAMCSA-N`  
**Brenk/PAINS:** 0.

**EN.** Analog of 01537 with aryl-F: same **aminal + naphthol→quinone** chain; F may retard some hydroxylation but not aminal cleavage. Highest cLogP in this batch (3.74) among top-10 → phospholipidosis/hERG structural concern (hypothesis).

**中文.** 同 01537 因果链；更高脂溶性加重 hERG/膜结合假说。

**Citations:** [CHEMBL5398980](https://www.ebi.ac.uk/chembl/explore/compound/CHEMBL5398980)

---

## Cross-cutting conclusions / 横切结论

1. **Aminal ethers** (00379, 01654, 02323, 00553, 00532, 01537, 01355) are the dominant **chemical-stability** liability under TME-like acidity — independent of Brenk/PAINS silence.
2. **2-Aminothiophene-3-carbonitrile (02621)** and **2-aminobenzothiazole (02323)** carry the strongest **reactive-metabolite** hepatotoxicity/genotoxicity hypotheses.
3. **Naphthols** (00553 family, 01537, 01355) share **quinone** bioactivation logic.
4. All statements are **clinical_grade=false** computational liability hypotheses for KRAS G12D triage — not developability clearance.

---

## Quantum DFT verification / 量子 DFT 验证

Live PySCF **B3LYP** single points after RDKit **ETKDGv3** embed + **MMFF94** optimize. Fukui indices by finite difference at fixed geometry using Mulliken charges: \(f^+=q_N-q_{N+1}\), \(f^-=q_{N-1}-q_N\), \(f^0=(f^++f^-)/2\). Machine-readable source: `DFT_QUANTUM_METRICS.json`. **clinical_grade=false**.

### Orbital metrics (Hartree and eV)

| Label | Basis | HOMO (Ha) | HOMO (eV) | LUMO (Ha) | LUMO (eV) | Gap (eV) | Neutral SCF |
|-------|-------|-----------|-----------|-----------|-----------|----------|-------------|
| 00379_parent | def2-SVP | −0.203162 | −5.5283 | −0.058335 | −1.5874 | **3.9409** | converged=true |
| 02621_parent | def2-SVP | −0.213538 | −5.8107 | −0.038871 | −1.0577 | **4.7529** | converged=true |
| 00553_parent | def2-SVP | −0.194115 | −5.2821 | −0.038411 | −1.0452 | **4.2369** | converged=true |
| 00379_repair | def2-SVP | −0.208085 | −5.6623 | −0.063418 | −1.7257 | **3.9366** | converged=true |
| 02621_repair | def2-SVP | −0.217109 | −5.9078 | −0.064617 | −1.7583 | **4.1495** | converged=true |
| 00553_repair | def2-SVP | −0.201343 | −5.4788 | −0.045744 | −1.2447 | **4.2341** | converged=true |

All six neutral SCFs **converged**. Repair Fukui cation (N−1) UKS: **NON-CONVERGENCE** (anion converged; reported honestly).

### Fukui top-2 atoms (parents) and metabolic mapping

**00379_parent** (charge_method=mulliken)
- **f+** top-2: C9 (0.0813), C7 (0.0799) — benzo/isoquinoline aryl carbons → electrophilic/oxidative attack on the fused aromatic core.
- **f−** top-2: C2 (0.0770), **N0 (0.0729, exocyclic NH₂)** — supports aminoisoquinoline oxidation / **nitrenium** soft-spot in Report §Rank 1.
- Aminal carbon is **not** among frontier Fukui tops: aminal liability remains primarily **acid/ionic hydrolysis**, not HOMO-driven oxidation.

**02621_parent** (charge_method=mulliken)
- **f+** top-2: **S14 (0.1417)**, N19 (nitrile N, 0.0967) — sulfur dominates electrophilic attack sites → strongly supports **thiophene-S-oxide** CYP/FMO hypothesis.
- **f−** top-2: **S14 (0.0978)**, N1 (pyrrolidine N, 0.0919) — S again dominant for nucleophilic/oxidative reactivity; tertiary amine α-site consistent with N-dealkylation.
- Maps directly onto Rank-2 dual-path assassination rationale (S-oxide + amine oxidation).

**00553_parent** (charge_method=mulliken)
- **f+** top-2: C10 (0.0810), C6 (0.0700) — naphthalene carbons toward oxidative activation.
- **f−** top-2: C6 (0.0869), **O8 (phenol O, 0.0514)** — phenol oxygen reactivity supports **naphthol → naphthoquinone** chain.
- Morpholine N / aminal C are secondary in Fukui ranking; aminal risk remains **acid-labile cleavage** (ionic), consistent with Rank-3 discussion.

### 中文摘要（DFT）

六分子中性态 B3LYP/def2-SVP 均收敛。能隙：00379≈3.94 eV，02621≈4.75 eV，00553≈4.24 eV；修复体能隙相近。Fukui：**02621 的 S 原子 f+/f− 最高**，验证噻吩 S-氧化假说；**00379 氨基 N 的 f− 偏高**，支持硝鎓路径；**00553 酚氧 f−** 支持萘醌路径。修复体质荷 UKS 阳离子未收敛，已如实记录。
