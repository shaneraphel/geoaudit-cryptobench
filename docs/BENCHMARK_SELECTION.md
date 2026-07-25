# Benchmark selection: pivoting the pocket-prediction evaluation

*GF(4) Allele-Conditioned Computational Chemistry — methodology note.*
*clinical_grade=false. This note describes an evaluation-design decision; it makes no efficacy, binding, or superiority claim.*

## 1. Why the current ESR1 appendix cannot decide the question

The Appendix-A ESR1 pocket pilot compares a geometry-only cavity predictor
against `fpocket`, `P2Rank`, and a random-box baseline over 14 ESR1 ligand-binding-domain
(LBD) **holo** structures. After the label-merge defect was fixed and labels were
regenerated per chain, the honest result is:

| Method | Top-1 DCA ≤ 4 Å |
|---|---|
| P2Rank | 14 / 14 |
| fpocket | 0 / 14 |
| random box | 0 / 14 |
| geometry-only predictor (this work) | 0 / 14 |

Two structural facts make this appendix **unable to test our hypothesis**, in either direction:

1. **No cluster-disjoint split.** Every structure is the ESR1 LBD fold. A single
   conserved sequence/structure cluster spans development / validation / locked-test.
   A statistical learner (P2Rank) can therefore succeed by recognising *this one
   fold*, not by understanding cavity physics. 14/14 is consistent with
   memorisation and tells us nothing about generalisation.
2. **Holo-only, no conformational challenge.** The pocket is already open and
   ligand-shaped in every input. A holo LBD does not stress a physical/geometric
   method's distinctive claim (that it can localise a site from boundary/topology
   rather than from a pre-formed, ML-recognisable surface signature).

We will **not** fix this by hand-building a split that flatters us. The correct
move is to adopt an existing, peer-reviewed benchmark whose *published* design
already isolates generalisation from memorisation.

## 2. Candidates considered

| Benchmark | Task | OOD pressure | Cluster-disjoint splits | Fit to a boundary/topology method | Verdict |
|---|---|---|---|---|---|
| PDBbind + strict/temporal split (LP-PDBbind) | affinity **scoring** | high (known ligand/protein-identity shortcut) | yes | weak — scores affinity, not *where* the pocket is | reject (wrong task) |
| Astex Diverse Set | docking pose, holo | low ("diverse" by construction, all holo) | no formal split | weak — holo, pose not detection | reject |
| Pocketome | conformational ensembles of druggable sites | moderate | no held-out protocol | partial — good for variability, not a scored benchmark | reject |
| CASP-CAPRI | complex/interface prediction | high | yes (blind targets) | partial — interface, now AlphaFold-dominated | reject (task drift, ML-saturated) |
| Apo–holo pair sets (generic) | site on apo | high | varies | strong | subsumed by CryptoBench/CryptoSite |
| **CryptoSite** (Cimermancic 2016) | cryptic site on apo | **very high** (large conformational change) | classic 93-pocket set | **strong** | keep as hard subset |
| **CryptoBench** (Škrhák 2025) | cryptic binding-site residues on apo | **high** | **yes — UniProt-grouped, sequence-identity clustered, predefined CV** | **strong** | **primary** |

## 3. Recommendation

**Primary benchmark: CryptoBench** (Škrhák et al., *Bioinformatics* 2025,
doi:10.1093/bioinformatics/btae745; data on OSF `pz4a9`).
**Hard subset: CryptoSite** (Cimermancic et al., *J. Mol. Biol.* 2016) for the
large-conformational-change cases.

Why this specific pair:

- **It is where holo-trained ML degrades, by construction and by measurement.**
  CryptoBench keeps only apo–holo pairs with *substantial binding-site
  conformational change*, then scores predictors on the **apo** state. The
  published baselines report exactly the degradation we predicted: P2Rank is
  markedly worse on apo than on its holo counterparts (CB-P2RANK-apo vs
  CB-P2RANK-holo), because on an apo cryptic site there is **no pre-formed,
  memorisable pocket surface** to key on. This is the honest, literature-grounded
  version of "memorisation fails," not a split we crafted.
- **Its splits already solve our #1 methodological gap.** Structures are grouped
  by UniProt ID and clustered by sequence identity with predefined
  cross-validation folds, so train/test are cluster-disjoint *as published*. We
  inherit that rigor instead of inventing it.
- **It rewards a boundary/topology approach in principle.** A cryptic site is a
  latent cavity: detecting it requires reasoning about van-der-Waals boundaries,
  packing defects, and cavity topology rather than recognising an open, ligand-shaped
  groove. That is exactly the regime our geometric-manifold / exact-form-topology
  prior is designed for.
- **It is standard and defensible.** Peer-reviewed, 1,107 structures, public
  data + code, and it is the current largest cryptic-binding-site benchmark — so
  a reviewer cannot dismiss it as self-serving.

## 4. Honest caveats (these are load-bearing, not boilerplate)

1. **A harder benchmark is not a win.** Our predictor scores 0/14 on *easy* holo
   ESR1 pockets. On apo cryptic sites it will very likely also fail until the
   candidate-generation stage is fixed — our own ablation showed generation, not
   ranking, is the bottleneck (the oracle ceiling over generated candidates is
   ~1/14). Adopting CryptoBench sets up the *correct experiment*; it does not by
   itself produce a positive result.
2. **"ML collapses on cryptic sites" is only partly true.** On CryptoBench the
   *best* reported method is still an ML model — a protein-language-model NN
   (pLM-NN) that beats both PocketMiner and P2Rank. What collapses is
   **general, holo-trained** ML (P2Rank), not all ML. To claim physical truth we
   must beat the published pLM-NN / PocketMiner numbers, not merely watch P2Rank
   drop. We will report our numbers against those baselines verbatim.
3. **Choosing a benchmark because an opponent fails there is itself a bias.** We
   adopt CryptoBench because it is the scientifically correct OOD,
   cluster-disjoint test of the cavity-physics hypothesis — and we commit to
   reporting our own apo numbers on it *even if we also fail*.

## 5. Concrete next steps

- Mirror CryptoBench apo receptors (OSF `pz4a9`) to iCloud with pinned SHA-256,
  reusing `tools/build_labels.py` provenance conventions.
- Score against the **published CV folds** (no re-splitting), residue-level
  metrics as defined by the benchmark (AUC / AUPRC / MCC / F1) plus our DCA/DCC.
- Report our geometry-only, +manifold-prior, and +manifold-prior+exact-form
  variants next to pLM-NN, PocketMiner, and P2Rank — apo and holo columns both.

## Sources

- Škrhák V. et al. *CryptoBench: cryptic protein–ligand binding sites dataset and
  benchmark.* Bioinformatics 41(1):btae745 (2025).
  https://doi.org/10.1093/bioinformatics/btae745 · data https://osf.io/pz4a9/ ·
  code https://github.com/skrhakv/CryptoBench
- Cimermancic P. et al. *CryptoSite: Expanding the Druggable Proteome by
  Characterization and Prediction of Cryptic Binding Sites.* J. Mol. Biol.
  428(4):709–719 (2016). https://doi.org/10.1016/j.jmb.2016.01.029
- Meller A. et al. *PocketMiner: predicting locations of cryptic pockets from
  single protein structures.* Nat. Commun. 14:1177 (2023).
