"""Compile the algebraic resolution field from the CryptoBench training fold.

Compilation is counting. For each of the six thematic tables the compiler walks
the training residues once and increments two integers per addressed cell; it
then ranks the tables by their own compiled Gini and picks the operating-point
quantile that maximises TRAINING F1. Nothing is solved, nothing is optimised by
gradient, and nothing observes the test fold. Running this twice on the same
bytes yields a byte-identical artifact.

Fail-closed guards, in order:
  1. train and test manifests must share no PDB-chain unit,
  2. they must share no MMseqs2 cluster id,
  3. the feature cache must have been built by the current source (SHA-256).

Usage:
  PYTHONPATH=src python3.12 tools/build_cascade_cache.py --fold both
  PYTHONPATH=src python3.12 tools/compile_algebraic_field.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from pocket_bench.methods.algebraic_field import code_sha256, compile_field

ROOT = Path(__file__).resolve().parents[1]
CACHE_TRAIN = ROOT / "data/cryptobench_apo/_cascade_cache_train.npz"
TRAIN_MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
TEST_MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
OUT = ROOT / "data/cryptobench_apo/ALGEBRAIC_FIELD.json"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def assert_no_leak() -> dict[str, int]:
    train = json.loads(TRAIN_MANIFEST.read_text())["entries"]
    test = json.loads(TEST_MANIFEST.read_text())["entries"]
    tr_u = {(e["pdb"], e["chain"]) for e in train}
    te_u = {(e["pdb"], e["chain"]) for e in test}
    if tr_u & te_u:
        raise SystemExit(f"UNIT LEAK: {len(tr_u & te_u)} units in both folds")
    tr_c = {e["cluster_id"] for e in train}
    te_c = {e["cluster_id"] for e in test}
    if tr_c & te_c:
        raise SystemExit(f"CLUSTER LEAK: {len(tr_c & te_c)} shared clusters")
    return {"train_units": len(tr_u), "test_units": len(te_u),
            "train_clusters": len(tr_c), "test_clusters": len(te_c)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    led = assert_no_leak()
    print(f"leak gate OK: {led['train_units']} train units / "
          f"{led['train_clusters']} clusters vs {led['test_units']} test units / "
          f"{led['test_clusters']} clusters, 0 shared", flush=True)

    if not CACHE_TRAIN.exists():
        raise SystemExit(f"missing {CACHE_TRAIN}; run tools/build_cascade_cache.py")
    z = np.load(CACHE_TRAIN, allow_pickle=False)

    t0 = time.perf_counter()
    field = compile_field(
        z["F"], z["y"], z["ctr"], z["n_res_per"],
        code_sha256=code_sha256(),
        train_manifest_sha256=_sha256(TRAIN_MANIFEST),
    )
    field["cluster_ledger"] = led
    field["test_manifest_sha256"] = _sha256(TEST_MANIFEST)
    field["compile_seconds"] = round(time.perf_counter() - t0, 2)

    args.out.write_text(json.dumps(field, indent=2, allow_nan=False) + "\n")
    st = field["operating_point"]
    print(f"tables={len(field['tables'])} "
          f"multiplicity={field['table_multiplicity']}")
    print(f"gini={[round(g, 4) for g in field['table_gini_train']]}")
    print(f"operating point q={st['q']:.2f} (train F1 {st['train_f1_at_q']:.4f})")
    print(f"wrote -> {args.out.relative_to(ROOT)} "
          f"({args.out.stat().st_size/1e6:.1f} MB, "
          f"{field['compile_seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
