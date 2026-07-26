# GeoAudit — zero-shot geometric cryptic-pocket detection on CryptoBench

`clinical_grade=false`

Repository specification. This document defines scope, execution, data provenance,
telemetry schema, and CI gates. It is not a paper and makes no comparative or
clinical claim.

**Scope (single claim).** Evaluating zero-shot geometric cryptic-pocket detectors on
the official CryptoBench receptor-only benchmark.

- Task: recover cryptic binding-site residues from **apo receptors only**, using
  deterministic geometry with no training, no fitted parameters, and no exposure to
  any CryptoBench partition.
- Benchmark: the official CryptoBench test fold (MMseqs2 clustering at 10 % sequence
  identity, cluster-disjoint), loaded fail-closed with per-file SHA-256 verification.
- Primary metrics: per-residue `residue_auc`, `residue_pr_auc`, `residue_mcc`,
  `residue_f1`, each with a 95 % paired-bootstrap CI over structures.
- Not claimed: parity with trained baselines, binding affinity, candidate efficacy,
  or global novelty. See `contracts/GEOAUDIT_PAPER_SCOPE.json`.

Appendices are excluded from the primary claim and from every generalization
statement: **Appendix A** is a retrospective ESR1 receptor-only pilot (invalidated
pending label regeneration); **Appendix B** is a finite-field allele-conditioning
ablation retained as future work (algebraic negative control only).

---

## 1. Repository scope & execution

### 1.1 Environment

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e ".[test,spectral,chem]"   # core: numpy==1.26.4
```

| Extra | Package | Used by |
|---|---|---|
| (core) | `numpy==1.26.4` | detectors, metrics, GF(4) ablation |
| `spectral` | `scipy>=1.11` | anisotropic-shear modes (`sstar_pocket`) |
| `chem` | `rdkit>=2023.9` | chemistry dictionary / alert screen |
| `test` | `pytest>=8`, `jsonschema>=4.20` | test suite |

External baselines (not pip-installable), versions pinned in
`data/manifests/BASELINE_ENV.json`: `fpocket 4.0`, `p2rank 2.5.1` (+ OpenJDK 17).
`deeppocket` is declared `TOOL_UNAVAILABLE`.

### 1.2 Commands

| Command | Action | Output |
|---|---|---|
| `make verify` | run `tools/verify_claims.py` scope/science gates | stdout JSON, exit≠0 on fail |
| `make test` | `unittest discover -s tests` | test report |
| `bash tools/build_native.sh` | optional Rust geometry kernels (arm64 / amd64; bit-identical to NumPy) | `native/geoaudit_kernels/target/release/libgeoaudit_kernels.{dylib,so}` |
| `PYTHONPATH=src python3.12 tools/run_cryptobench_apo.py --dataset official --jobs 6` | **primary**: official CryptoBench test fold (fail-closed; never falls back to the pilot; `--jobs` = structure-parallel workers) | `results/cryptobench_official/{APO_BENCHMARK,TELEMETRY}.json` |
| `PYTHONPATH=src python3.12 tools/run_cryptobench_apo.py --dataset pilot` | apo pilot (n=15, Appendix only) | `results/cryptobench_apo/{APO_BENCHMARK,TELEMETRY}.json` |
| `PYTHONPATH=src python3.12 -m pocket_bench.metrics_bootstrap --dataset official` | paired bootstrap CIs on the official fold | `results/cryptobench_official/BOOTSTRAP_CI.json` (frozen copy: `results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP.json`) |
| `PYTHONPATH=src python3.12 tools/fetch_official_data.py --build-manifest --fold-file data/cryptobench_apo/_osf/test.json` | materialize official test fold from OSF (hash-verified) | `data/cryptobench_apo/official_manifest.json` (+ receptors/labels, gitignored) |
| `PYTHONPATH=src python3.12 tools/run_official_fold.py` | official test-fold residue metrics + bootstrap CIs | `results/official_fold/{OFFICIAL_FOLD_METRICS,PER_STRUCTURE}.json` |
| `PYTHONPATH=src python3.12 tools/gf4_allele_shuffle_ablation.py` | GF(4) wrong-allele control | `results/gf4_ablation/GF4_ALLELE_ABLATION.json` |
| `PYTHONPATH=src python3.12 tools/build_split_ledger.py` | rebuild split ledger | `data/manifests/SPLIT_LEDGER.json` |
| `PYTHONPATH=src python3.12 tools/build_labels.py --download` | ESR1 labels (Appendix A) | `data/labels/*_labels.json` |
| `PYTHONPATH=src python3.12 tools/run_pilot.py --split all` | ESR1 retrospective pilot | `results/pilot/REGENERATED_PILOT_REPORT.json` |

### 1.3 Fail-closed external inputs (`src/pocket_bench/adapters.py`)

| Adapter | Requires | On absence |
|---|---|---|
| `load_official_test_fold()` | `data/cryptobench_apo/official_manifest.json` (schema `cryptobench.official_test_fold.v1`) | raises `DataUnavailable` with fetch instructions; a stride pilot is never substituted |
| `load_pocketminer_scores(pdb,chain)` | `data/baselines/pocketminer/<pdb>_<chain>.{json,csv}` | raises `DataUnavailable`; never scored as zeros |

Availability probes (non-raising): `official_fold_available()`, `pocketminer_available()`.

---

## 2. Data provenance

All external inputs are pinned by SHA-256. RCSB re-versions coordinate files; a
later fetch that mismatches means RCSB re-released the entry.

### 2.1 CryptoBench apo pilot (`data/cryptobench_apo/`, n=15)

- Selection: deterministic stride (`stride=36`) over sorted keys of the label set;
  full record in `data/cryptobench_apo/PREDICTION_INPUT_MANIFEST.json`
  (`schema=gf4cc.cryptobench_apo.manifest.v1`), per-structure `raw_sha256`,
  `receptor_sha256`, `receptor_atoms`, `n_cryptic_residues`, `n_label_atoms`.
- Labels source: `https://raw.githubusercontent.com/skrhakv/CryptoBench/master/src/F-statistics/conservation/label_dataset.json`
  — `sha256=effd00f26af1cc614771321697e7ca6dbc9fb1d851c46c54e652b115d5f3c221`,
  358013 bytes. Citation: Skrhak et al., Bioinformatics 2025,
  doi:10.1093/bioinformatics/btae745; dataset `https://osf.io/pz4a9/`.
- Structures: `https://files.rcsb.org/download/<PDBID>.pdb`.
- Status: this is **not** the official cluster-disjoint test fold
  (`SPLIT_LEDGER.json` → `cryptobench_apo.is_official_cryptobench_test_fold=false`;
  `BOOTSTRAP_CI.json` → `is_official_mmseqs2_10pct_test_fold=false`).

### 2.2 Official CryptoBench MMseqs2 10% test fold (ingested)

- Source: OSF node `https://osf.io/pz4a9/` (Skrhak et al., Bioinformatics 2025,
  doi:10.1093/bioinformatics/btae745). Split construction: MMseqs2 @ 10% sequence
  identity, 80:20 train:test, test = 222 apo structures.
- Verified OSF files (SHA-256 checked against OSF-reported digests; pinned in
  `data/cryptobench_apo/PROVENANCE.json`):
  - `cryptobench-dataset/folds.json` — `sha256=ced97a50…3c6ac67d`, 17826 bytes
    (fold membership).
  - `cryptobench-dataset/folds/test.json` — `sha256=28f5630e…09a7560838`, 1705224
    bytes (test membership + `apo_pocket_selection` cryptic labels).
- Materialization: `tools/fetch_official_data.py` builds
  `data/cryptobench_apo/official_manifest.json` (schema
  `cryptobench.official_test_fold.v1`; `fold=="test"`, `clustering.method=="mmseqs2"`,
  `sequence_identity_threshold==0.10`). Each entry
  `{pdb, chain, cluster_id, receptor_path, receptor_sha256, label_path, label_sha256}`;
  receptors fetched chain-scoped and ligand-stripped from
  `https://files.rcsb.org/download/<PDBID>.pdb`; `cluster_id` = `uniprot_id`
  sequence-cluster surrogate. Loader `adapters.load_official_test_fold()` verifies
  every SHA-256 and rejects any `cluster_id` crossing splits.
- Evaluable scope: 222 test apo PDBs → 193 single-chain units; 38 multi-chain
  (compound `apo_chain`) units excluded (chain-agnostic `resseq` indexing would be
  ambiguous — recorded in the manifest, not silently dropped); `7nbc` skipped (RCSB
  serves mmCIF only) → **192 manifest entries**.
- The 35 MB materialized receptor set is git-ignored; it is regenerated from the
  hash-pinned fetcher on demand.

### 2.3 PocketMiner baseline (not present)

- Source: `https://github.com/Mickdub/gvp/tree/pocket_pred` (Meller et al.,
  Nat. Commun. 2023, doi:10.1038/s41467-023-36699-3).
- Required: `data/baselines/pocketminer/<pdb>_<chain>.json`
  (`{residue_scores:{resseq:prob}}`) or `.csv` (`resseq,score`); probabilities in [0,1].

### 2.4 KRAS / probe structures (`data/manifests/STRUCTURE_PROVENANCE.json`)

| PDB | Role | SHA-256 (prefix) | bytes |
|---|---|---|---|
| `9BL0` | KRAS G12D noncovalent primary docking geometry | `719c4a6c…b508fe0` | 289737 |
| `5VA1` | hERG cryo-EM geometric-compatibility probe | `946f89bb…3fc7f472` | 402732 |

### 2.5 ESR1 pilot structures (Appendix A) — `data/manifests/SOURCE_URLS.json`

UniProt `P03372`; RCSB `3ERT, 3OS8, 1UOM, 5U2B, 6CHW, 4Q50, 4Q13`. Baseline tools
`fpocket` (`https://github.com/Discngine/fpocket`), `p2rank`
(`https://github.com/rdk/p2rank`).

### 2.6 Fetch / manifest scripts

`tools/mirror_cryptobench_to_icloud.py` (raw PDB mirror + hashes),
`tools/refresh_manifests.py` (rebuild manifests), `tools/build_labels.py --download`
(ESR1 labels). Companion evidence SHA/pointers: `data/manifests/COMPANION_EVIDENCE.json`.

---

## 3. Telemetry & metrics schema

### 3.1 Per-(method,structure) row — `results/cryptobench_apo/TELEMETRY.json` (`geoaudit.telemetry.v1`)

| Field | Type | Definition |
|---|---|---|
| `method` | str | detector id (`geometric_foundation`,`sstar_pocket`,`fstar_pocket`,`p2rank`,`random_bbox`) |
| `pdb`,`split` | str | structure id; split label |
| `status` | enum | `OK` \| `EMPTY` \| `CRASH` \| `TOOL_UNAVAILABLE` |
| `tool`,`tool_version` | str | external tool name/version (empty for internal) |
| `seed`,`env_sha`,`runtime_s` | int/str/float | reproducibility |
| `top1_dca`,`best_dca` | float | Å from Top-1 centre to nearest cryptic-residue atom (localization proxy) |
| `top1_success` | bool | `best_dca ≤ 4.0` and `status==OK` |
| `dcc_top1` | float | centre-to-centre distance |
| `n_pockets` | int | pockets returned |
| `residue_auc` | float\|null | CryptoBench-faithful per-residue ROC-AUC |
| `residue_pr_auc` | float\|null | per-residue average precision |
| `residue_mcc` | float\|null | per-residue MCC (null when the confusion matrix is degenerate) |
| `residue_f1` | float\|null | per-residue F1 (null when threshold not defined) |
| `residue_metrics_available` | bool | whether label∩universe join produced metrics |
| `chain`,`unit_id` | str | evaluation unit `(pdb, chain)`; the bootstrap key |

### 3.2 Aggregate — same file, `per_method`

`intention_to_evaluate_denominator` (every structure attempted),
`ok`/`crash`/`empty`/`tool_unavailable`, `available_denominator = intention −
tool_unavailable`, `top1_hits`, `hits_over_intention`, `hits_over_available`.
Primary metric `top1_dca_le_4A`; faithful metrics
`[residue_auc, residue_pr_auc, residue_mcc, residue_f1]`.

### 3.3 Bootstrap — `results/cryptobench_apo/BOOTSTRAP_CI.json` (`geoaudit.bootstrap_report.v1`)

Per metric (`residue_auc`, `residue_pr_auc`, `residue_mcc`, `residue_f1`):
`per_method{point, ci_low, ci_high, n_structures_scored}` and
`paired_vs_baseline{delta_point, delta_ci_low, delta_ci_high,
p_two_sided_bootstrap, crosses_zero}`. Params `{n_boot=10000, seed, ci_level,
baseline}`. Paired resampling over units keyed on `unit_id`. Where a method has no
scored unit for a metric, every Δ field is `null` (an unestimated difference is never
rendered as an excluded-zero result).

### 3.4 Official test-fold freeze — `results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP.json`

Produced by `tools/run_cryptobench_apo.py --dataset official --jobs 6` followed by
`python -m pocket_bench.metrics_bootstrap --dataset official --baseline p2rank`
(`n_boot=10000`, seed `20260725`, 95 % CI, paired resampling over units).

**Execution guarantee.** The official evaluation emits exactly **960/960** telemetry
rows = **192 evaluation units × 5 methods**, with `residue_metrics_available=true` on
960/960 and **zero silent drops**. The evaluation unit is a `(pdb, chain)` pair
(`unit_id`), not a PDB entry: the 192 units span 190 distinct entries because `3lnz`
(chains C, O) and `3pfp` (chains A, B) each contribute two chains. All four metrics
resample over n=192; `per_structure_values` raises on a colliding `unit_id` rather
than overwriting a row. A further 38 fold units are multi-chain assemblies, excluded
at manifest build time and recorded in `official_manifest.json`
(`n_excluded_multichain=38`) — excluded before evaluation, never dropped during it.

Per-method status: `geometric_foundation`, `fstar_pocket`, `sstar_pocket`,
`random_bbox` = 192/192 `OK`; `p2rank` = 186/192 `OK` + 6 `EMPTY`. Cost recorded in
telemetry: 6769 s summed over the 960 `runtime_s` fields; the prediction phase
completes in ~11 min wall on an Apple M4 (10 cores) at `--jobs 6` with the optional
Rust kernels built. `--jobs` changes only scheduling: results are order-preserving
and identical to `--jobs 1`.

Point estimates [95 % CI], baseline `p2rank`:

| method | ROC-AUC | PR-AUC | MCC | F1 |
|---|---|---|---|---|
| `geometric_foundation` | 0.664 [0.643, 0.684] | 0.276 [0.246, 0.306] | 0.237 [0.208, 0.266] | 0.285 [0.257, 0.313] |
| `p2rank` | 0.621 [0.605, 0.637] | 0.264 [0.237, 0.291] | 0.220 [0.192, 0.249] | 0.251 [0.225, 0.278] |
| `fstar_pocket` | 0.595 [0.575, 0.614] | 0.203 [0.176, 0.231] | 0.129 [0.105, 0.153] | 0.196 [0.173, 0.219] |
| `sstar_pocket` | 0.559 [0.542, 0.575] | 0.133 [0.115, 0.151] | 0.082 [0.061, 0.103] | 0.155 [0.135, 0.176] |
| `random_bbox` | 0.500 [0.500, 0.500] | 0.093 [0.084, 0.103] | null | 0.000 [0.000, 0.000] |

`residue_mcc` is scored on n=186 for `p2rank` (6 `EMPTY` predictions give an
undefined confusion matrix) and on n=0 for `random_bbox` (no positive predictions at
the operating point); its point estimate, Δ, and `crosses_zero` are all `null`, never
0.

Paired Δ (`geometric_foundation` − `p2rank`), same resample indices:

| metric | Δ [95 % CI] | bootstrap p | `crosses_zero` |
|---|---|---|---|
| ROC-AUC | +0.043 [+0.025, +0.062] | ≈0.000 | false |
| F1 | +0.034 [+0.008, +0.059] | 0.007 | false |
| PR-AUC | +0.012 [−0.013, +0.037] | 0.368 | true |
| MCC | +0.017 [−0.011, +0.045] | 0.244 | true |

Two of four Δ CIs exclude 0; PR-AUC and MCC are statistically indistinguishable from
`p2rank` on this fold.

**Baseline status.** `p2rank 2.5.1` is executed locally (JVM, version pinned in
`BASELINE_ENV.json`) and is a real scored baseline here — it is **not**
`DATA_UNAVAILABLE`. `pocketminer` **is** `DATA_UNAVAILABLE`: OSF publishes only a
trained model binary, no per-residue predictions, so no paired CI against it is
computed and no value is imputed. `deeppocket` is declared `TOOL_UNAVAILABLE`.
A legacy null-only freeze (`OFFICIAL_FOLD_METRICS.json`, vs `random_residue`, keyed
on pdb) is retained for appendix comparison and is not the primary readout.

### 3.5 Appendix B artifact — allele-conditioning ablation (`results/gf4_ablation/GF4_ALLELE_ABLATION.json`, `geoaudit.gf4_allele_ablation.v1`)

Excluded from the primary claim; an algebraic negative control only, with no chemical,
binding or efficacy content. Schema:

`syndromes{allele:vector}`, `syndrome_nonzero_index`, `all_syndromes_distinct`,
`correct_allele_G12D_admissible`, `wrong_allele_fail_closed{allele:{admissible,
rejected,residual_weight}}`, `allele_shuffle{n_shuffles,seed,rejected,
passed_by_chance,rejection_rate}`.

---

## 4. Physical CI/CD gates

Enforced by `tools/verify_claims.py` (`make verify`) and `tests/` (`make test`);
all fail-closed.

| Gate | Enforcement | Fails if |
|---|---|---|
| Zero-leakage firewall | `ligand_leak_guard` on predictors; AST import-graph test | receptor has non-solvent HETATM / smuggled non-polymer ATOM, or a `methods.*` module imports the scorer |
| Cluster-disjoint splits | `SPLIT_LEDGER.json` + `test_split_cluster_disjoint.py` | a `cluster_disjoint_required` group shares a `cluster_id` across splits, or a non-disjoint group omits `split_integrity_passed=false`+reason |
| Denominator discipline | `telemetry.assert_denominator_discipline` + `test_denominator_discipline.py` | intention denominator ≠ attempts; `TOOL_UNAVAILABLE` used for a declared-present tool; internal method masked |
| Label integrity | `verify_claims` re-reads labels | any ligand label not 3–80 heavy atoms / missing centroid |
| ESR1 pilot boundary | `verify_claims` | Appendix A not marked `retrospective_pilot_only` / `comparative_claim_allowed=false` / not invalidated-pending-regeneration |
| Deterministic modes | eigenvector phase pin in `low_shear_modes` + `test_anisotropic_shear_oracle.py` | non-deterministic S\* mask |
| GF(4) wrong-allele ablation | `test_allele_ablation.py` | correct allele inadmissible, any wrong allele admitted, or shuffle rejection <0.90 |
| Provenance pinned | `verify_claims` | `9BL0`/`5VA1` SHA-256 or byte size unset |
| Scope hygiene | `verify_claims` regexes | out-of-scope terms, proprietary engine names, absolute local paths, or credential patterns in primary docs |

Current state: `verify_claims` all checks pass; test suite green.

---

## 5. Appendices and future work

Everything in this section is **outside the primary claim** and is excluded from every
generalization statement. Machine-enforced by
`contracts/GEOAUDIT_PAPER_SCOPE.json` and the CI gates above.

| Item | Status | Boundary |
|---|---|---|
| **Appendix A** — ESR1 receptor-only pilot | `retrospective_pilot_only`, invalidated pending label regeneration | `comparative_claim_allowed=false`; contributes nothing to the CryptoBench result |
| **Appendix B** — finite-field allele-conditioning ablation | `algebraic_ablation_future_work` | Algebraic negative control; no chemical, binding, or efficacy claim |
| Candidate generation / structure-defined modalities | future work | Bulk evidence lives in the companion tree `gf4-allele-conditioned-evidence` (`data/manifests/COMPANION_EVIDENCE.json`); not a claim of this repository |
| Localized anisotropic shear for void-absent pockets | future work | Global low-frequency modes caused surface drift; see `results/cryptobench_apo/` |

Numerical record: all counts and metrics are read from the JSON artifacts
(`results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP.json`,
`results/gf4_ablation/GF4_ALLELE_ABLATION.json`,
`data/manifests/COMPANION_EVIDENCE.json`). Where earlier narrative text disagreed with
those artifacts, the narrative has been deleted rather than reconciled; the JSON is the
sole source of truth.

---

## Repository layout

```text
paper/        MAIN_CRYPTOBENCH_GEOAUDIT.tex (primary manuscript),
              appendix_b_gf4_ablation.tex (\input appendix, not standalone)
contracts/    GEOAUDIT_PAPER_SCOPE.json (scope contract)
src/pocket_bench/
  methods/    receptor-only detectors (firewalled) + anisotropic_shear_oracle
  native.py   ctypes loader for optional Rust kernels (NumPy fallback)
  metrics.py metrics_bootstrap.py telemetry.py adapters.py paths.py pdb_io.py
native/       geoaudit_kernels (Rust cdylib: free-grid, buriedness, local_free_enclosed)
tools/        run_cryptobench_apo.py, build_native.sh, run_official_fold.py,
              fetch_official_data.py, run_pilot.py, build_labels.py,
              build_split_ledger.py, gf4_allele_shuffle_ablation.py, verify_claims.py
data/         cryptobench_apo/ (PROVENANCE.json + gitignored materialized fold),
              manifests/, labels/
results/      cryptobench_apo/, official_fold/, gf4_ablation/, pilot/
tests/        firewall, native kernels, split-disjoint, denominator, bootstrap,
              adapters, ablation
```

Local-only material (never published) is gitignored (`*.local.*`, `_local/`) and
excluded from scope gates.
