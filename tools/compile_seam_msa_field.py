#!/usr/bin/env python3.12
"""Compile geometry + nonlocal-seam + MSA-conservation columns.

Writes ``data/cryptobench_apo/SEAM_MSA_FIELD.json``. Does not touch
GEOMETRY_FIELD.json or TABLE_FIELD.json. Train fold only.
clinical_grade = false.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pocket_bench.methods.geometry_wires import N_COLUMNS as N_GEOMETRY, N_DISP_CARRIED
from pocket_bench.methods.nonlocal_seam import COLUMNS as SEAM_QTY
from pocket_bench.methods.table_field import code_sha256, compile_field
from pocket_bench.paths import ROOT
from compile_algebraic_field import (
    TEST_MANIFEST, TRAIN_MANIFEST, _sha256, assert_no_leak,
)

CACHE_TRAIN = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
OUT = ROOT / "data/cryptobench_apo/SEAM_MSA_FIELD.json"
GEO_CACHES = (
    ("backbone", "_backbone_cache_train.npz", None),
    ("sidechain", "_sidechain_cache_train.npz", None),
    ("void", "_void_cache_train.npz", None),
    ("displacement", "_displacement_cache_train.npz", N_DISP_CARRIED),
)
SEAM_CACHE = ROOT / "data/cryptobench_apo/_nonlocal_seam_cache_train.npz"
MSA_CACHE = ROOT / "data/cryptobench_apo/_msa_conserv_cache_train.npz"
N_SEAM = 3 * len(SEAM_QTY)


def _load(path: Path, n_rows: int, prefix: str, carry=None):
    z = np.load(path, allow_pickle=False)
    C, cols = z["C"], [str(s) for s in z["names"]]
    z.close()
    if C.shape[0] != n_rows:
        raise SystemExit(f"{path.name} row mismatch")
    if carry is not None:
        per_agg = C.shape[1] // 3
        take = [a * per_agg + q for a in range(3) for q in range(carry)]
        C = C[:, take]
        cols = [cols[i] for i in take]
    return C, [f"{prefix}~{c}" for c in cols]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    led = assert_no_leak()
    z = np.load(CACHE_TRAIN, allow_pickle=False)
    Xw, prop = z["X"], z["propensity_table"]
    names = [str(s) for s in z["names"]]
    y, ctr, n_res_per = z["y"], z["ctr"], z["n_res_per"]
    z.close()

    blocks, gnames = [], []
    for fam, fname, carry in GEO_CACHES:
        C, cols = _load(ROOT / "data/cryptobench_apo" / fname, Xw.shape[0], fam, carry)
        blocks.append(C); gnames.extend(cols)
    Xg = np.concatenate(blocks, axis=1)
    if Xg.shape[1] != N_GEOMETRY:
        raise SystemExit(f"geo width {Xg.shape[1]} != {N_GEOMETRY}")

    Xs, snames = _load(SEAM_CACHE, Xw.shape[0], "seam")
    if Xs.shape[1] != N_SEAM:
        raise SystemExit(f"seam width {Xs.shape[1]} != {N_SEAM}")
    Xm, mnames = _load(MSA_CACHE, Xw.shape[0], "msa")
    X = np.concatenate([
        np.asarray(Xw, np.float32), np.asarray(Xg, np.float32),
        np.asarray(Xs, np.float32), np.asarray(Xm, np.float32),
    ], axis=1)
    all_names = tuple(names + gnames + snames + mnames)
    print(f"columns: {Xw.shape[1]}+{Xg.shape[1]}+{Xs.shape[1]}+{Xm.shape[1]}"
          f" = {X.shape[1]}", flush=True)

    t0 = time.perf_counter()
    field = compile_field(
        X, y, ctr, n_res_per, all_names, prop,
        code_sha256=code_sha256(),
        train_manifest_sha256=_sha256(TRAIN_MANIFEST),
    )
    field.update({
        "cluster_ledger": led,
        "test_manifest_sha256": _sha256(TEST_MANIFEST),
        "compile_seconds": round(time.perf_counter() - t0, 2),
        "detector": "seam_msa_field",
        "clinical_grade": False,
        "what_is_different": (
            "geometry_field columns + nonlocal-seam + Swiss-Prot MSA "
            "conservation statistics under the same table topology"
        ),
        "seam_columns": N_SEAM,
        "msa_columns": int(Xm.shape[1]),
    })
    args.out.write_text(json.dumps(field, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.out.relative_to(ROOT)} in {field['compile_seconds']}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
