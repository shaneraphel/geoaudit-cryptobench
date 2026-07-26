"""Extract the 35 algebraic invariants once for both folds and cache them.

Feature extraction is a pure function of the receptor bytes, so it is done once.
Every cascade topology experiment afterwards is arithmetic over the cached
matrices, which is what makes the self-correction loop cost seconds per
iteration instead of minutes.

The cache stores residue centroids as well, because the spatial majority gate
needs the contact graph and rebuilding it from centroids costs milliseconds.

Usage: PYTHONPATH=src python3.12 tools/build_cascade_cache.py --fold both
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TRAIN_MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
TEST_MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
CACHE_TRAIN = ROOT / "data/cryptobench_apo/_cascade_cache_train.npz"
CACHE_TEST = ROOT / "data/cryptobench_apo/_cascade_cache_test.npz"


def _code_hash() -> str:
    src = "".join(
        (SRC / rel).read_text()
        for rel in ("pocket_bench/methods/algebraic_descriptors.py",
                    "pocket_bench/methods/density_topology.py",
                    "pocket_bench/methods/geometric_foundation.py",
                    "pocket_bench/methods/sequence_wires.py",
                    "pocket_bench/spatial.py",
                    "pocket_bench/pdb_io.py")
    )
    return hashlib.sha256(src.encode()).hexdigest()


def _one(entry: dict):
    from pocket_bench.methods.algebraic_descriptors import (
        algebraic_residue_features,
    )
    # Anti-regression lock: cryptic_residues is the only accepted truth key.
    lab = json.loads((ROOT / entry["label_path"]).read_text())
    truth = {int(r) for r in (lab.get("cryptic_residues") or [])}
    if not truth:
        return None
    resseq, F, codes, ctr = algebraic_residue_features(
        ROOT / entry["receptor_path"], chain=entry["chain"])
    y = np.fromiter((1 if int(r) in truth else 0 for r in resseq),
                    dtype=np.int64, count=len(resseq))
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return F, y, codes, ctr


def build(entries: list[dict], out: Path, tag: str) -> None:
    t0 = time.perf_counter()
    Fs, ys, cs, ctrs, ns, units = [], [], [], [], [], []
    skipped = 0
    for i, e in enumerate(entries, 1):
        try:
            r = _one(e)
        except Exception as exc:  # noqa: BLE001 -- a broken unit is skipped, never imputed
            print(f"  SKIP {e.get('pdb')}_{e.get('chain')} "
                  f"{type(exc).__name__}: {exc}"[:150], flush=True)
            r = None
        if r is None:
            skipped += 1
        else:
            F, y, codes, ctr = r
            Fs.append(F); ys.append(y); cs.append(codes); ctrs.append(ctr)
            ns.append(len(y)); units.append(f"{e['pdb']}_{e['chain']}")
        if i % 50 == 0 or i == len(entries):
            el = time.perf_counter() - t0
            print(f"  [{tag} {i}/{len(entries)}] {el:.0f}s "
                  f"~{el/i*(len(entries)-i):.0f}s left  skipped={skipped}",
                  flush=True)
    if not Fs:
        raise SystemExit(f"no usable {tag} units")
    np.savez_compressed(
        out,
        F=np.concatenate(Fs, axis=0),
        y=np.concatenate(ys, axis=0),
        codes=np.concatenate(cs, axis=0),
        ctr=np.concatenate(ctrs, axis=0),
        n_res_per=np.asarray(ns, dtype=np.int64),
        units=np.asarray(units),
        code_hash=np.asarray(_code_hash()),
    )
    print(f"{tag}: {len(Fs)} units, {sum(ns)} residues -> {out.name} "
          f"({time.perf_counter()-t0:.0f}s)", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", choices=("train", "test", "both"), default="both")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    train = json.loads(TRAIN_MANIFEST.read_text())["entries"]
    test = json.loads(TEST_MANIFEST.read_text())["entries"]
    tr = {(e["pdb"], e["chain"]) for e in train}
    te = {(e["pdb"], e["chain"]) for e in test}
    if tr & te:
        raise SystemExit(f"LEAK: {len(tr & te)} units in both folds")
    trc = {e["cluster_id"] for e in train}
    tec = {e["cluster_id"] for e in test}
    if trc & tec:
        raise SystemExit(f"CLUSTER LEAK: {len(trc & tec)} shared clusters")
    print(f"leak check OK: {len(tr)} train, {len(te)} test, "
          f"0 shared units, 0 shared clusters", flush=True)

    if args.limit:
        train, test = train[:args.limit], test[:args.limit]
    if args.fold in ("test", "both"):
        build(test, CACHE_TEST, "test")
    if args.fold in ("train", "both"):
        build(train, CACHE_TRAIN, "train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
