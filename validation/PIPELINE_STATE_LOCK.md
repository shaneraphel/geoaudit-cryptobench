# PIPELINE STATE LOCK — Anti-Drift Architectural Anchor

**Status:** IMMUTABLE METHOD LOCK · `clinical_grade = false`  
**Freeze date:** 2026-07-18  
**Scope:** W1–W6 computational oncology / undruggable PPI pipeline  
**Rule:** Any change to methods, seeds, boxes, or listed script bytes invalidates comparability until this lock is explicitly revised and re-hashed.

---

## 1. Methodological Freeze

### 1.1 Docking engine (secondary metric only)

| Parameter | Locked value |
|-----------|--------------|
| Engine | AutoDock Vina **1.2.5** (local `.tools/bin/vina`) |
| Exhaustiveness (strict) | **16** |
| Exhaustiveness (prefilter) | **8** (W3/W5/W6 prefilter only) |
| Random seeds (strict) | **41, 42, 43, 44** |
| Modes reported | best pose mode 1 mean ± CI where applicable |
| Interpretation | more-negative = better **pose-fit geometry in that box**; **NOT** affinity / IC50 / Kd / efficacy |

### 1.2 Same-box definition (W1 trust foundation)

1. Receptor = co-crystal PDB protein atoms (water / irrelevant HET stripped).
2. Box center = centroid of co-crystal ligand HET atoms.
3. Box size = axis-aligned extents of HET + padding as implemented in `tools/fda_baseline_redock.py` (`box_size`).
4. FDA reference and lead are docked into **identical** receptor + box.
5. Native redock of co-crystal ligand is the box-validity control; random-surface negative control is recorded when present.

Locked W1 pairs:

| Target | PDB | HET | FDA drug |
|--------|-----|-----|----------|
| FLT3 | 6JQR | C6F | gilteritinib |
| ABL1 | 3OXZ | 0LI | ponatinib |
| BCL2 | 6O0K | LBM | venetoclax |

### 1.3 Undruggable surface tensors (W6)

1. Fetch experimental PDB (AlphaFold companion optional; not required for lock).
2. Strip water / ions / non-protein HET; keep protein heavy atoms.
3. Interface seeds = inter-chain contacts ≤ 5.0 Å (or whole chain for single-chain surfaces).
4. Map into fixed **64³** Boolean grids:
   - **steric wall** — voxels within VdW 1.7 Å of protein heavy atoms
   - **clamp shell** — exterior ring (VdW … VdW+2.4 Å), mutually exclusive with steric, restricted to ≤ 8 Å of interface atoms
5. **No pocket-detection heuristic.** Flat PPI/IDP topology only.

### 1.4 Honesty constants (never relaxed by automation)

- `clinical_grade = false` on all ledgers
- ADMET / hERG / metabolism / PAINS = geometric or rule **proxies** (`is_measured=false`)
- No FTO / Freedom-to-Operate legal claims from Tanimoto / Murcko
- No wet superiority claims from Vina

---

## 2. File Dependency Tree + SHA-1 (byte hashes)

Hashes are **SHA-1 of file bytes on disk at freeze** (not git commit IDs). Silent edits change the hash → drift.

### 2.1 Active scripts (W1–W6)

| SHA-1 | Path | Role |
|-------|------|------|
| `fe4ca0f2ea45139082bf7f7fa211a538db021a79` | `tools/fda_baseline_redock.py` | W1 same-box FDA redock |
| `e9439f446405379632b5aff6b112b2137b420378` | `tools/w3_novel_forge.py` | W3 Murcko-novelty forge |
| `092acf8eef77ee96a9aa4209f7786927a245ce83` | `tools/w3_forge_benchmark.py` | W3 forge benchmark |
| `eb8ed691387a5f215c4819959781401abd4ed41c` | `tools/w4_profile.py` | W4 FLT3 cubane physchem/proxy profile |
| `852e9ef07fcffc94c706f16ba63c9d2b5f847353` | `tools/w4_geometric_audit.py` | W4 geometric off-target / hERG proxy |
| `6069e1e21c6b1c54fd36d61c67202d22d4430b39` | `tools/w5_forge.py` | W5 weak-box + FLT3 isostere |
| `e83e339d3dbe02000216cf5c035c958a338878b9` | `tools/mass_tensor_ingest.py` | W6 64³ surface ingest |
| `d3abf6d4b745e3d035973602e67f79f261b350ca` | `tools/w6_myc_clamp_forge.py` | W6 MYC clamp forge |
| `3b9cddb3d8ba2c5fc2be082ba7fb71d10312b6e0` | `tools/myc_scaffold_hopping.py` | MYC motif diversity |
| `d9915e20bfb70cec04725e3776ebf26848f3baf8` | `tools/myc_clamp_refine_offtarget.py` | MYC bounded clamp refine |
| `cee9dbbcfbf5aed3045c93aee1bb3ce7bb28ff0e` | `tools/myc_receptor_dock.py` | MYC/MAX receptor dock |
| `cf26549061e7585fe80231abef03fc31f8753f54` | `tools/public_leak_audit.py` | Deny-by-default leak audit |
| `e687c63f847561ac341ea6eb7ee644866fc1b89d` | `tools/generate_3d_viewer.py` | W5 3D viewer export |

### 2.2 Frozen ledgers / overview

| Artifact | Role |
|----------|------|
| `validation/FDA_SAMEBOX_REDOCK_SUMMARY.json` | W1 summary |
| `validation/W3_NOVEL_FORGE_LEDGER.json` | W3 novelty forge |
| `validation/W4_FLT3_CUBANE_PROFILE.json` | W4 cubane profile |
| `validation/W5_FORGE_LEDGER.json` | W5 winners |
| `experimental_v2/chem_pipeline/outputs/w6_mass_ingest/W6_MASS_INGEST_LEDGER.json` | W6 tensors |
| `experimental_v2/chem_pipeline/outputs/w6_myc_clamp/W6_MYC_CLAMP_LEDGER.json` | W6 MYC clamps |
| `W1_W5_HONEST_OVERVIEW.md` | External honesty one-pager |

### 2.3 Private staging mirror

`Targets/Oncology/{MYC_MAX,CTNNB1_TCF4,STAT3,TP53_MUT}/` in private `foliation-er100-oncology-data`.

---

## 3. Combinatorial Constraint Lock

### 3.1 Small-molecule gates (W3–W5; FLT3 isostere stricter)

| Gate | Locked value | Notes |
|------|--------------|-------|
| Murcko novelty | Tanimoto ≤ **0.40** vs blacklist scaffolds | Morgan radius 2, 2048 bits on Murcko |
| SA (FLT3 isostere W5) | **SA < 4.0** | Affinity↔SA trade (−10.40 → −9.05 ≈ **1.35 kcal**) |
| SA (W5 weak-box) | SA ≤ 6.0 library gate | Winners SA 2.5–2.9 |
| SA (W6 MYC clamp) | **SA < 4.5** | High-Fsp³ rigid clamps |
| Fsp³ (W6 MYC) | ≥ **0.55** | `ar_rings = 0` (no flat aromatic plates) |
| Cubane | Banned in W5+ libraries | W3 cubane = historical peak only |

### 3.2 64³ Boolean tensor mapping (exact)

```
GRID = 64
VDW  = 1.7 Å
SHELL = 2.4 Å
half_box = target-specific (MYC 22, β-cat/TCF 26, STAT3 24, p53 22) Å
spacing = (2 * half_box) / (GRID - 1)
steric = voxels with dist(protein) ≤ VDW
shell  = (VDW < dist ≤ VDW+SHELL) AND NOT steric AND dist(interface) ≤ 8.0 Å
```

Schema: `foliation.surface_steric_64.v1` (`.bin` + JSON). No pocket detection.

### 3.3 W6 MYC spatial pass

| Parameter | Locked value |
|-----------|--------------|
| `SHELL_FILL_MIN` | 0.25 |
| `CLASH_FRAC_MAX` | 0.30 |
| Placement | rigid-body search on clamp-shell centroid |

### 3.4 Philosophy lock (W5)

> We trade theoretical affinity for physical synthesizability and absolute structural novelty.

---

## 4. Biologics module boundary (W7+)

- Small-molecule same-box Vina protocol remains frozen for kinase/weak-box work.
- Biologics use separate ledgers; rigid-receptor Vina for large flexible rings is **unreliable** and is not affinity.
- Biologics rigidity: MMFF torsional ΔE (constrained vs relaxed), not wet Kd.

---

## 5. Lock revision protocol

1. Edit methods or scripts → recompute SHA-1 table here.
2. Bump `lock_revision` with UTC timestamp and reason.
3. Re-run affected acceptance commands.
4. LEAK_AUDIT must PASS before private push.

**lock_revision:** `v1-2026-07-18` — initial anti-drift freeze after W6 MYC clamps.
