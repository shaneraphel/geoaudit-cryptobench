"""Compile the Spatial Associative Memory from the CryptoBench TRAIN folds.

Streams every training residue into the 4096-cell quaternary table as two integer
counters. No weights are fitted and nothing is optimized: a cell's content is a
count of how many training residues addressed it and how many of those were
cryptic.

Leakage control: the training units come from ``train_manifest.json``, whose
builder verifies both pdb-id and uniprot-cluster disjointness against the test
manifest before writing. This tool additionally refuses to run if the two entry
sets intersect.

Usage: PYTHONPATH=src python3.12 tools/compile_resolution_field.py --jobs 8
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TRAIN_MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
TEST_MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
OUT_A = ROOT / "data/cryptobench_apo/RESOLUTION_FIELD.json"
CACHE = ROOT / "data/cryptobench_apo/_feature_cache.npz"
OUT_B = ROOT / "data/cryptobench_apo/RESOLUTION_FIELD_B.json"


def _one(entry: dict):
    """Geometric+sequence features, labels and residue codes for one unit.

    Always extracts the Track B superset; Track A is the first six columns of the
    same matrix, so both tracks are compiled from byte-identical geometry and the
    A/B comparison is controlled by construction.
    """
    from pocket_bench.methods.quaternary_lut import receptor_residue_features
    try:
        rec = ROOT / entry["receptor_path"]
        lab = json.loads((ROOT / entry["label_path"]).read_text())
        truth = set(int(r) for r in (lab.get("cryptic_residues")
                                     or lab.get("binding_residues") or []))
        if not truth:
            return None
        resseq, F, codes = receptor_residue_features(
            rec, chain=entry["chain"], with_sequence=True)
        y = np.fromiter((1 if int(r) in truth else 0 for r in resseq),
                        dtype=np.int64, count=len(resseq))
        return F, y, codes
    except Exception:  # noqa: BLE001 -- a broken training unit is skipped, never imputed
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--track", choices=("A", "B", "both"), default="both")
    ap.add_argument("--refresh-cache", action="store_true",
                    help="re-extract features even if the cache is valid")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    if args.jobs > 1:
        for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[v] = "1"

    if TRAIN_MANIFEST.is_file():
        train = json.loads(TRAIN_MANIFEST.read_text())
        entries = train["entries"]
    else:
        # Provisional compile while the fold is still materializing: pair up the
        # receptor/label files already on disk. Same units, same paths, just not
        # yet summarized into a manifest.
        train = {"fold": "train-* (provisional, manifest not yet written)"}
        rec_dir = ROOT / "data/cryptobench_apo/train_receptors"
        lab_dir = ROOT / "data/cryptobench_apo/train_labels"
        entries = []
        for lab in sorted(lab_dir.glob("*_labels.json")):
            stem = lab.name[: -len("_labels.json")]
            rec = rec_dir / f"{stem}_receptor.pdb"
            if not rec.is_file():
                continue
            pdb, chain = stem.rsplit("_", 1)
            entries.append({
                "pdb": pdb, "chain": chain,
                "cluster_id": json.loads(lab.read_text()).get("uniprot", ""),
                "receptor_path": str(rec.relative_to(ROOT)),
                "label_path": str(lab.relative_to(ROOT)),
            })
        print(f"provisional: {len(entries)} train units found on disk", flush=True)
    if args.limit:
        entries = entries[:args.limit]
    if TEST_MANIFEST.is_file():
        test = json.loads(TEST_MANIFEST.read_text())
        tr = {(e["pdb"], e["chain"]) for e in entries}
        te = {(e["pdb"], e["chain"]) for e in test["entries"]}
        if tr & te:
            raise SystemExit(f"LEAK: {len(tr & te)} units in both folds")
        tc = {e["cluster_id"] for e in test["entries"]}
        trc = {e["cluster_id"] for e in entries}
        if tc & trc:
            raise SystemExit(f"CLUSTER LEAK: {len(tc & trc)} shared clusters")
        print(f"leak check OK: {len(tr)} train units, {len(te)} test units, "
              f"0 shared units, 0 shared clusters", flush=True)

    t0 = time.perf_counter()
    # Feature extraction is a pure function of the receptor bytes, so it is cached.
    # Without this every recompile re-derives identical geometry for all 770
    # structures (~450 s) purely to re-solve a 6x6 matrix, which is what made
    # iterating on the scoring rule cost minutes per attempt.
    # The key covers the units AND the source of every module that can change a
    # feature value. Keying on the unit list alone silently serves stale vectors
    # after an edit to the extractor, which is a wrong-number bug, not a slow one.
    _src = "".join(
        (SRC / rel).read_text()
        for rel in ("pocket_bench/methods/quaternary_lut.py",
                    "pocket_bench/methods/density_topology.py",
                    "pocket_bench/methods/sequence_wires.py",
                    "pocket_bench/methods/geometric_foundation.py",
                    "pocket_bench/spatial.py",
                    "pocket_bench/pdb_io.py")
    )
    cache_key = json.dumps({
        "units": [[e["pdb"], e["chain"]] for e in entries],
        "code": hashlib.sha256(_src.encode()).hexdigest(),
    })
    cached = None
    if CACHE.is_file() and not args.refresh_cache:
        try:
            z = np.load(CACHE, allow_pickle=False)
            if str(z["key"]) == cache_key:
                cached = z
                print(f"feature cache HIT ({CACHE.name})", flush=True)
        except Exception:  # noqa: BLE001 -- a corrupt cache is simply rebuilt
            cached = None
    if cached is not None:
        F_all = cached["F"]; y = cached["y"]; codes = cached["codes"]
        n_res_per = list(cached["n_res_per"])
        good = [None] * int(cached["n_units"])
    else:
        results = []
        if args.jobs > 1:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
                for i, r in enumerate(ex.map(_one, entries, chunksize=1), 1):
                    results.append(r)
                    if i % 50 == 0 or i == len(entries):
                        el = time.perf_counter() - t0
                        print(f"  [{i}/{len(entries)}] {el:.0f}s "
                              f"~{el/i*(len(entries)-i):.0f}s left", flush=True)
        else:
            results = [_one(e) for e in entries]

        good = [r for r in results if r is not None]
        if not good:
            raise SystemExit("no usable training units")
        F_all = np.concatenate([g[0] for g in good], axis=0)
        y = np.concatenate([g[1] for g in good], axis=0)
        codes = np.concatenate([g[2] for g in good], axis=0)
        n_res_per = [len(g[1]) for g in good]
        np.savez_compressed(
            CACHE, F=F_all, y=y, codes=codes,
            n_res_per=np.asarray(n_res_per, dtype=np.int64),
            n_units=np.asarray(len(good)), key=np.asarray(cache_key),
        )
        print(f"feature cache WRITTEN ({CACHE.name})", flush=True)
    print(f"training residues: {len(y)}  cryptic: {int(y.sum())} "
          f"({100*y.mean():.1f}%)  units used: {len(good)}/{len(entries)}")

    from pocket_bench.methods.quaternary_lut import (
        FEATURE_NAMES, SEQ_FEATURE_NAMES, ResolutionField, compile_edges,
        normalized_rank, quantize, track_shape,
    )
    from pocket_bench.methods.sequence_wires import (
        apply_propensity, propensity_table,
    )

    # S4 is counted on the training fold only, then frozen into the artifact.
    prop = propensity_table(codes, y)
    F_B = np.concatenate([F_all, apply_propensity(codes, prop)[:, None]], axis=1)
    names_all = list(FEATURE_NAMES) + list(SEQ_FEATURE_NAMES)

    def build(F: np.ndarray, names: list[str], propensity, out_path: Path,
              track: str) -> dict:
        edges = compile_edges(F)
        addr = quantize(F, edges)
        n_cells = track_shape(F.shape[1])
        tot = np.bincount(addr, minlength=n_cells).astype(np.int64)
        pos = np.bincount(addr, weights=y, minlength=n_cells).astype(np.int64)

        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        orient = np.ones(F.shape[1]); gini = np.zeros(F.shape[1])
        for j in range(F.shape[1]):
            r = np.empty(len(y), dtype=np.float64)
            r[np.argsort(F[:, j], kind="stable")] = np.arange(1, len(y) + 1)
            auc = (r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
            orient[j] = 1.0 if auc >= 0.5 else -1.0
            gini[j] = abs(2.0 * auc - 1.0)
            print(f"    train AUC {names[j]:18s} = {auc:.4f}  gini={gini[j]:.4f}")

        # Ranks WITHIN each structure, matching inference exactly.
        R = np.empty_like(F)
        off = 0
        for n_r in n_res_per:
            blk = F[off:off + n_r]
            for j in range(F.shape[1]):
                R[off:off + n_r, j] = normalized_rank(blk[:, j])
            off += n_r
        R1, R0 = R[y == 1], R[y == 0]
        mu1, mu0 = R1.mean(axis=0), R0.mean(axis=0)
        S = ((R1 - mu1).T @ (R1 - mu1) + (R0 - mu0).T @ (R0 - mu0)) / (len(R) - 2)
        S = S + np.eye(S.shape[0]) * 1e-9 * float(np.trace(S)) / S.shape[0]
        w = np.linalg.solve(S, mu1 - mu0)
        nrm = float(np.linalg.norm(w))
        if nrm > 0:
            w = w / nrm

        field = ResolutionField(edges, pos, tot, orient, gini, w,
                                float(y.mean()), propensity)
        payload = field.to_json()
        payload.update({
            "clinical_grade": False,
            "track": track,
            "compiled_from": "cryptobench train folds (MMseqs2 10% identity)",
            "train_folds": train.get("fold"),
            "n_train_units_used": len(good),
            "n_training_residues": int(len(y)),
            "n_cryptic_residues": int(y.sum()),
            "test_fold_disjoint_verified": True,
        })
        out_path.write_text(json.dumps(payload, indent=2) + "\n")
        asserted = int((tot > 0).sum())
        x = payload["n_cells_X"]
        print(f"  Track {track}: {F.shape[1]} wires, {n_cells} cells, "
              f"asserted={asserted}, X={x} "
              f"({100.0*x/max(asserted,1):.1f}% of asserted)  -> "
              f"{out_path.name}")
        return payload

    made = {}
    if args.track in ("A", "both"):
        print("== Track A (pure geometry) ==")
        made["A"] = build(F_all[:, :len(FEATURE_NAMES)], names_all, None,
                          OUT_A, "A")
    if args.track in ("B", "both"):
        print("== Track B (geometry + sequence) ==")
        made["B"] = build(F_B, names_all, prop, OUT_B, "B")
    print(f"done in {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
