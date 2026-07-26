"""Materialize the CryptoBench TRAIN folds for associative-memory compilation.

The official split file ``folds.json`` names five disjoint MMseqs2 clusters at 10%
sequence identity: ``test`` plus ``train-0..3``. The test fold is already
materialized by ``fetch_official_data.py``; this builds the train side from the
same two source files, with the same parser, so a training unit is constructed
byte-identically to a test unit.

Leakage control is structural, not asserted: the fold membership comes from the
published clustering, the two id sets are verified disjoint here, and the cluster
surrogate (uniprot_id) is verified disjoint across the train/test boundary. Any
overlap aborts the build.

Usage: PYTHONPATH=src python3.12 tools/build_training_fold.py --folds train-0 train-1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fetch_official_data import (  # noqa: E402
    OSF_DIR,
    RCSB,
    ROOT,
    _fold_units,
    _get,
    sha256_bytes,
)
from pocket_bench.pdb_io import parse_pdb_atoms, write_receptor_only_pdb  # noqa: E402

TRAIN_RECEPTORS = ROOT / "data/cryptobench_apo/train_receptors"
TRAIN_LABELS = ROOT / "data/cryptobench_apo/train_labels"
TRAIN_MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
TEST_MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"


def build(fold_names: list[str], limit: int = 0) -> Path:
    folds = json.loads((OSF_DIR / "folds.json").read_text())
    dataset = json.loads((OSF_DIR / "dataset.json").read_text())
    dataset_sha = sha256_bytes((OSF_DIR / "dataset.json").read_bytes())
    folds_sha = sha256_bytes((OSF_DIR / "folds.json").read_bytes())

    test_ids = {str(p).lower() for p in folds["test"]}
    train_ids: list[str] = []
    for fn in fold_names:
        if fn not in folds:
            raise SystemExit(f"unknown fold '{fn}'; have {sorted(folds)}")
        train_ids.extend(str(p).lower() for p in folds[fn])
    overlap = test_ids & set(train_ids)
    if overlap:
        raise SystemExit(f"FOLD LEAK: {len(overlap)} pdb ids in both train and test: "
                         f"{sorted(overlap)[:8]}")

    fold_obj = {p: dataset[p] for p in train_ids if p in dataset}
    missing = [p for p in train_ids if p not in dataset]
    units, excluded = _fold_units(fold_obj)

    # Cluster-level disjointness against the already-built test manifest.
    test_clusters: set[str] = set()
    if TEST_MANIFEST.is_file():
        test_clusters = {e["cluster_id"]
                         for e in json.loads(TEST_MANIFEST.read_text())["entries"]}
    leaked = {u["uniprot"] for u in units.values()} & test_clusters
    if leaked:
        raise SystemExit(f"CLUSTER LEAK: {len(leaked)} uniprot clusters span "
                         f"train and test: {sorted(leaked)[:8]}")

    keys = sorted(units)
    if limit:
        keys = keys[:limit]
    entries, skipped = [], []
    for i, (pdb, chain) in enumerate(keys, 1):
        u = units[(pdb, chain)]
        resids = sorted(u["residues"])
        rec_path = TRAIN_RECEPTORS / f"{pdb}_{chain}_receptor.pdb"
        try:
            if rec_path.is_file() and rec_path.stat().st_size > 0:
                rec_sha = sha256_bytes(rec_path.read_bytes())
            else:
                raw = _get(RCSB.format(pdb.upper()), binary=True)
                atoms = parse_pdb_atoms(raw.decode("utf-8", "ignore"))
                if not any(a["record"] == "ATOM" and a["chain"] == chain for a in atoms):
                    raise RuntimeError(f"chain {chain} absent in RCSB {pdb}")
                write_receptor_only_pdb(atoms, rec_path, chain=chain)
                rec_sha = sha256_bytes(rec_path.read_bytes())
        except Exception as exc:  # noqa: BLE001 -- record and continue, never impute
            skipped.append({"pdb": pdb, "chain": chain, "reason": str(exc)})
            continue
        lab = {"schema": "cryptobench.train_label.v1", "clinical_grade": False,
               "pdb_id": pdb, "chain": chain, "cryptic_residues": resids,
               "binding_residues": resids, "labels_source_sha256": dataset_sha}
        lab_path = TRAIN_LABELS / f"{pdb}_{chain}_labels.json"
        lab_path.parent.mkdir(parents=True, exist_ok=True)
        lab_path.write_text(json.dumps(lab, indent=2) + "\n")
        entries.append({
            "pdb": pdb, "chain": chain, "cluster_id": u["uniprot"],
            "receptor_path": str(rec_path.relative_to(ROOT)),
            "receptor_sha256": rec_sha,
            "label_path": str(lab_path.relative_to(ROOT)),
            "label_sha256": sha256_bytes(lab_path.read_bytes()),
        })
        if i % 25 == 0:
            print(f"  [{i}/{len(keys)}] {pdb}_{chain}", file=sys.stderr, flush=True)

    manifest = {
        "schema": "cryptobench.train_fold.v1",
        "clinical_grade": False,
        "fold": fold_names,
        "clustering": {"method": "mmseqs2", "sequence_identity_threshold": 0.10},
        "folds_file_sha256": folds_sha,
        "labels_source_sha256": dataset_sha,
        "n_fold_units": len(units),
        "n_entries": len(entries),
        "n_excluded_multichain": len(excluded),
        "n_missing_from_dataset": len(missing),
        "n_skipped_fetch": len(skipped),
        "skipped": skipped,
        "test_cluster_disjoint_verified": bool(test_clusters),
        "entries": entries,
    }
    TRAIN_MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "entries"
                      and k != "skipped"}, indent=2))
    return TRAIN_MANIFEST


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", nargs="+", default=["train-0", "train-1"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    build(args.folds, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
