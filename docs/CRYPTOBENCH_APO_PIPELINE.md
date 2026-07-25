# CryptoBench-Apo pipeline & the Multimodal Geometric Foundation

*Architecture plan. `clinical_grade=false`. This document specifies a pipeline and
a generation theory; it makes no binding, efficacy, novelty, or clinical claim.*

## 0. Why apo, and how scope is preserved (Bug 1)

CryptoBench (Škrhák 2025) scores predictors on **apo** conformations of
apo–holo pairs with substantial binding-site conformational change, using
UniProt-grouped, sequence-clustered, predefined CV folds. It is the correct
out-of-distribution test because an apo cryptic site has **no pre-formed,
memorisable surface** — the exact regime where a deterministic geometric engine
is differentiated from a fitted one.

Scope is preserved by splitting evaluation into two tracks, so the universal
geometry claim and the oncology claim never contaminate each other:

- **Track G (geometry / method).** Full CryptoBench-apo, target-agnostic. The
  foundation claims universality, so it is tested on every apo pocket, reporting
  the benchmark's own residue-level metrics (AUC/AUPRC/MCC/F1) plus our DCA/DCC.
  No drug-candidate claim is attached to Track G.
- **Track O (oncology candidates / claims).** Only the subset of CryptoBench-apo
  whose UniProt maps to the six-target panel (KRAS, ESR1, FLT3, PIM1, PIK3CA,
  CDK4/6) under the four structure-defined modalities. `N` is reported honestly;
  if the intersection is small, that number is published, not padded.

The scope gate in `tools/verify_claims.py` is a **hard CI fail** on the
out-of-scope keyword set (`OUT_OF_SCOPE`), which now also blocks the
out-of-panel-disease and ecosystem-scale framings in addition to the fabrication
and hardware terms. Any Track that tries to smuggle an out-of-panel disease into
the paper tree fails `make verify` before it can be committed. (The literal
blocked tokens live only inside the regex in `tools/verify_claims.py`, which the
scope scan deliberately excludes from itself.)

## 1. Data mirror & pinning — `tools/mirror_cryptobench_to_icloud.py` (Bug 4)

Every artifact is fetched to the iCloud cache and pinned by SHA-256 + byte count.
Fail-closed: a short read, a hash change, or a missing file aborts the run.

```python
# core logic (not the full script)
ICLOUD = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs" \
    / "Foliation-Engine-Archive/cryptobench_apo"
OSF_INDEX = "https://osf.io/pz4a9/"          # CryptoBench release (splits + CIF)
RCSB = "https://files.rcsb.org/download/{pdb}.pdb"

def fetch_pinned(url: str, dest: Path, attempts: int = 4, min_bytes: int = 5000) -> dict:
    for k in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "gf4cc/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310
                data = r.read()
            if len(data) < min_bytes:
                raise ValueError(f"{url} too small ({len(data)}B)")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            return {"url": url, "path": str(dest.relative_to(ICLOUD)),
                    "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        except Exception as e:                 # retry any transient read
            last = e
    raise RuntimeError(f"failed {url} after {attempts}: {last}")

def build_manifest(entries: list[dict]) -> None:
    # entries: [{pdb_id, chain, apo_pdb, holo_pdb, ligand_resname, ligand_resseq, fold}]
    records = []
    for e in entries:
        apo  = fetch_pinned(RCSB.format(pdb=e["apo_pdb"]),  ICLOUD / f"{e['apo_pdb']}.pdb")
        holo = fetch_pinned(RCSB.format(pdb=e["holo_pdb"]), ICLOUD / f"{e['holo_pdb']}.pdb")
        records.append({**e, "apo": apo, "holo": holo})
    manifest = {
        "schema": "gf4cc.cryptobench_apo.prediction_inputs.v1",
        "clinical_grade": False,
        "source": {"benchmark": "CryptoBench", "doi": "10.1093/bioinformatics/btae745",
                    "data": OSF_INDEX, "retrieved": date.today().isoformat()},
        "n_entries": len(records),
        "entries": records,       # every artifact carries sha256 + bytes
    }
    (ICLOUD / "PREDICTION_INPUT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
```

The paper tree stores only the manifest pointer (`data/manifests/`), never bulk
PDBs; iCloud holds the raw bytes. This is the same provenance contract already
used by `STRUCTURE_PROVENANCE.json` and `RAW_PDB_ICLOUD.json`.

## 2. Chain isolation — exact parsing logic (Bug 2)

A cryptic label is **one ligand residue instance** on **one receptor chain**.
The old defect matched `resname` alone and merged copies (4Q50 `OHT`
29 → 232 heavy atoms). The strict selector isolates a single
`(model, chain, resname, resseq, icode)` instance, drops the best altloc,
excludes solvent/buffer/ion HETATMs, and enforces a physical heavy-atom bound.

```python
# non-ligand HETATM blocklist (solvent, cryoprotectant, buffer, common ions)
_NON_LIGAND = {"HOH","DOD","SO4","PO4","GOL","EDO","PEG","PG4","MPD","ACT","DMS",
               "FMT","CL","NA","K","MG","CA","ZN","MN","FE","NI","CD","IOD","BR"}

def isolate_ligand(atoms, resname, *, chain, resseq=None, icode="", model=1,
                   min_heavy=6, max_heavy=100):
    """Heavy-atom coords of exactly ONE ligand instance. Fails closed."""
    if resname in _NON_LIGAND:
        raise ValueError(f"{resname} is solvent/ion, not a ligand label")
    inst: dict[tuple, dict[str, list]] = {}
    for a in atoms:
        if a["record"] != "HETATM" or a["resname"] != resname:      continue
        if a.get("model", 1) != model:                              continue
        if a["chain"] != chain:                                     continue
        if resseq is not None and a["resseq"] != resseq:            continue
        if icode and a.get("icode", "") != icode:                   continue
        if a["element"] == "H":                                     continue
        alt = a.get("altloc", "") or ""
        key = (a["chain"], a["resseq"], a.get("icode", ""))
        # keep the highest-occupancy altloc per atom name; ties -> altloc 'A'/''
        slot = inst.setdefault(key, {})
        prev = slot.get(a["name"])
        if prev is None or (a.get("occ", 1.0), alt in ("", "A")) > (prev["occ"], prev["altpref"]):
            slot[a["name"]] = {"xyz": [a["x"], a["y"], a["z"]],
                               "occ": a.get("occ", 1.0), "altpref": alt in ("", "A")}
    if not inst:
        raise ValueError(f"no HETATM instance for {resname} chain {chain} resseq {resseq}")
    # one instance only: the copy with the most heavy atoms on this chain,
    # deterministically tie-broken by (resseq, icode)
    key = max(inst, key=lambda k: (len(inst[k]), -k[1], k[2]))
    coords = [atom["xyz"] for atom in inst[key].values()]
    n = len(coords)
    if not (min_heavy <= n <= max_heavy):
        raise ValueError(f"ligand {resname} {key} has {n} heavy atoms "
                         f"(outside physical bound [{min_heavy},{max_heavy}])")
    return coords, key
```

Binding residues on the **apo** structure are then derived by aligning apo↔holo
by residue number + name (Kabsch on matched Cα) and taking apo residues whose
any heavy atom is within the contact cutoff of the isolated holo ligand — so the
label lives on apo coordinates while being defined by the holo ligand instance.

## 3. Science-invariant CI (Bug 3)

CI must **compute physics**, not scan text. Gates (extending the four already in
`verify_claims.py`):

```python
def gate_label_physical(labels: list[dict]) -> bool:
    # every label is a single instance within physical bounds
    return all(6 <= len(l["ligand_heavy_coords"]) < 100
               and len(l["ligand_centroid"]) == 3 for l in labels)

def gate_candidate_feasibility(cands: list[dict], oracle) -> bool:
    # every ACCEPTED candidate clears the Boolean voxel-occupancy wall and respects the
    # 1.227 Å minimum-bond invariant — recomputed here, not trusted from the file
    for c in cands:
        if any(oracle.occupied(p) for p in c["atom_xyz"]):          return False
        if min_pairwise_bond(c["bond_edges"]) < 1.227:              return False
    return True

def gate_baseline_honest(report: dict, labels: dict) -> bool:
    # recompute Top-1 DCA from stored predictions + labels; reported == recomputed
    for method, pred in report["per_method"].items():
        recomputed = sum(top1_dca(pred[p], labels[p]) <= 4.0 for p in labels)
        if recomputed != pred["top1_hits"]:                         return False
        denom = pred["ok"] + pred["unavailable"] + pred["crash_empty"]
        if denom != report["n_structures"]:                         return False
    return True   # a 0/N stays 0/N; the gate refuses silent inflation
```

The baseline gate is the anti-fabrication core: it re-derives every hit count
from raw predictions+labels, so a hand-edited "0 → 14" cannot pass CI.

## 4. Breaking the 1/14 ceiling: one-pass combinational algebraic projection

The ceiling is a **generation**, not a ranking, failure: on apo only ~1/14
pools contain any candidate within 4 Å. We do not fix this with stochastic
sampling or iterative refinement. We fold the physical boundaries **into the
initial map** so a candidate is feasible *by construction*, in a single
straight-line pass with no clock and no fixed-point loop.

### 4.1 Objects (all precomputed, deterministic)

- Grid `G = Z³ ∩ box`, spacing `h`. Bitplanes over `G`:
  - **Boolean voxel-occupancy wall** `O(v)=1` iff `‖center(v) − nearest_protein_heavy‖ <
    r_Bondi(element)`. Free space `F = ¬O`.
  - **Curvature-admissibility field** `K(v)=1` iff `κ(v) ≤ κ*`, where `κ` is a
    local curvature/packing proxy from the geometric manifold prior, precomputed once.
- **Allele seed** `s₀ = embed( solve(H x = B δ) )`, a bitplane of proposed atom
  centres from the GF(4) syndrome solution for residual `δ` (§ method paper).
- **Bond LUT** `Λ`: the fixed table of admissible neighbour offsets whose length
  lies in `[1.227 Å, d_max]`. Membership is a lookup, never a search.
- **Rescale set** `W`: a fixed, finite set of integer conformal rescales
  (discrete conformal rescaling) — bounded, evaluated in parallel, no loop.

### 4.2 The map (straight-line / clockless)

```
candidate(δ, structure) =
    Π_bond(Λ) ∘ Π_squeeze(W) ∘ ( s₀  AND  F*  AND  K )
```

1. **Mask** `m = s₀ ∧ F* ∧ K` — a bitwise AND of three bitplanes.
2. **Squeeze** `Π_squeeze` — apply each `w ∈ W`, keep the single `w` maximising
   retained mass by one `argmax` over the fixed set `|W|` (bounded, parallel).
3. **Bond snap** `Π_bond` — realise edges only where a center pair matches an
   offset in `Λ`; sub-1.227 Å edges are unrepresentable by construction.

Every stage is a bounded pure function of its input (bitwise ops, a finite LUT,
an `argmax` over a fixed finite set). The composition is therefore a
combinational, clockless straight-line map: identical `(δ, structure)` always
yield the identical candidate — no RNG, no iteration, no fixed point. This is the
"flattened logic-gate" formulation requested: constraints are satisfied at step
zero rather than filtered after sampling.

### 4.3 The apo-cryptic correction (the honest hard part)

A **rigid** oracle over an apo structure marks the closed cryptic pocket as
occupied, so `F` would exclude exactly the region the holo ligand occupies —
rigid feasibility is *anti-correlated* with the apo target. The fix keeps the map
one-pass while letting the wall breathe deterministically:

```
F* = OR over  m ∈ M  of  free_space( Weyl_rescale_m(wall) )
```

where `M` is a fixed, low-order set of elastic-network breathing modes at fixed
amplitudes (or fixed Weyl integer rescales). `F*` admits a voxel if it is free in
**at least one** low-energy breathing mode — a deterministic model of cryptic
opening with no molecular-dynamics sampling.

**This is a falsifiable physical bet, not a guarantee.** If a true cryptic site
is never free in `M`, the candidate pool misses it and we report **0** — we will
not enlarge `M` post hoc to manufacture a hit, and we will not soften `O` into a
learned penalty. The mode set `M` is fixed before scoring and pinned in the
manifest.

## 5. Novelty framing (Bug 5)

The "Foliation-lite burial heuristic" narrative is retired. The method is the
**Multimodal Geometric Foundation**: a deterministic engine that computes exact
topological deformations from universal law — GF(4) allele algebra, a geometric
manifold prior (curvature-admissibility field), the Boolean voxel-occupancy wall,
discrete conformal rescaling, and an exact-form topological filter — applied unchanged
across pockets and modalities. Burial counting is gone; feasibility is an
algebraic property of the one-pass map, not a heuristic score.

## 6. Showcase (README integration)

The same operator grammar, three modalities, real deposited references:

| Modality | Target (real reference) | Deformation |
|---|---|---|
| Small molecule | KRAS G12D `9BL0` | Single-pass discrete conformal rescaling on aniline components clears the voxel-occupancy wall while `Λ` preserves the 1.227 Å minimum-bond invariant. |
| PROTAC ternary | ESR1 (ERα)–VHL `9SV3` | An exact-form topological filter rigidifies the linker trajectory, driving a topological pump that evades the target–ligase interface. |
| Macrocycle | FLT3 WT `4XUF` | 16-bit spinor projection + curvature-admissibility evaluation resolves ring strain without breaking the discrete loop-closure invariant. |

These are **illustrative operator demonstrations on real geometries**, not
measured poses or binding evidence.

## 7. Honesty boundaries

- `clinical_grade=false`. A valid deformation is proof of *topological
  computation only* — never binding, potency, selectivity, PK/PD, or safety.
- A failed geometry scores **0**; 0/N is published as 0/N.
- The breathing mode set `M`, grid, Bondi radii, `κ*`, `Λ`, and `W` are all
  pinned before scoring; nothing is tuned on the test folds.
