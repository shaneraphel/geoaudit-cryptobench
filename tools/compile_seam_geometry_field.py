#!/usr/bin/env python3.12
"""Compile geometry_field columns plus the nonlocal-seam family.

Does not modify GEOMETRY_FIELD.json or TABLE_FIELD.json. Writes
``data/cryptobench_apo/SEAM_GEOMETRY_FIELD.json`` — a development detector for
the transfer-atlas attack on pLM-NN. Train receptors only.

``clinical_grade`` is false.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from pocket_bench.methods.geometry_wires import (
    FAMILIES,
    N_COLUMNS as N_GEOMETRY,
    N_DISP_CARRIED,
)
from pocket_bench.methods.nonlocal_seam import COLUMNS as SEAM_QTY
from pocket_bench.methods.table_field import code_sha256, compile_field
from pocket_bench.paths import ROOT

from compile_algebraic_field import (  # noqa: E402
    TEST_MANIFEST,
    TRAIN_MANIFEST,
    _sha256,
    assert_no_leak,
)

CACHE_TRAIN = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
OUT = ROOT / "data/cryptobench_apo/SEAM_GEOMETRY_FIELD.json"

GEO_CACHES = (
    ("backbone", "_backbone_cache_train.npz", None),
    ("sidechain", "_sidechain_cache_train.npz", None),
    ("void", "_void_cache_train.npz", None),
    ("displacement", "_displacement_cache_train.npz", N_DISP_CARRIED),
)
SEAM_CACHE = ROOT / "data/cryptobench_apo/_nonlocal_seam_cache_train.npz"
N_SEAM = 3 * len(SEAM_QTY)


def _load_geo(n_rows: int) -> tuple[np.ndarray, list[str]]:
    blocks, names = [], []
    for fam, fname, carry in GEO_CACHES:
        path = ROOT / "data/cryptobench_apo" / fname
        z = np.load(path, allow_pickle=False)
        C, cols = z["C"], [str(s) for s in z["names"]]
        z.close()
        if C.shape[0] != n_rows:
            raise SystemExit(f"{fname} row mismatch")
        if carry is not None:
            per_agg = C.shape[1] // 3
            take = [a * per_agg + q for a in range(3) for q in range(carry)]
            C = C[:, take]
            cols = [cols[i] for i in take]
        blocks.append(C)
        names.extend(f"{fam}~{c}" for c in cols)
    X = np.concatenate(blocks, axis=1)
    if X.shape[1] != N_GEOMETRY:
        raise SystemExit(f"geometry width {X.shape[1]} != {N_GEOMETRY}")
    return X, names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    led = assert_no_leak()
    print(f"cluster gate OK: {led['train_units']} train / "
          f"{led['test_units']} test", flush=True)

    z = np.load(CACHE_TRAIN, allow_pickle=False)
    Xw, prop = z["X"], z["propensity_table"]
    names = [str(s) for s in z["names"]]
    y, ctr, n_res_per = z["y"], z["ctr"], z["n_res_per"]
    z.close()

    Xg, gnames = _load_geo(Xw.shape[0])
    if not SEAM_CACHE.exists():
        raise SystemExit(f"missing {SEAM_CACHE}; run tools/nonlocal_seam_wires.py")
    zs = np.load(SEAM_CACHE, allow_pickle=False)
    Xs, snames = zs["C"], [f"seam~{s}" for s in zs["names"]]
    zs.close()
    if Xs.shape[0] != Xw.shape[0] or Xs.shape[1] != N_SEAM:
        raise SystemExit(
            f"seam cache shape {Xs.shape} expected ({Xw.shape[0]}, {N_SEAM})")

    X = np.concatenate([
        np.asarray(Xw, dtype=np.float32),
        np.asarray(Xg, dtype=np.float32),
        np.asarray(Xs, dtype=np.float32),
    ], axis=1)
    all_names = tuple(names + gnames + snames)
    print(f"columns: {Xw.shape[1]} wires + {Xg.shape[1]} geometry + "
          f"{Xs.shape[1]} seam = {X.shape[1]}", flush=True)

    t0 = time.perf_counter()
    field = compile_field(
        X, y, ctr, n_res_per, all_names, prop,
        code_sha256=code_sha256(),
        train_manifest_sha256=_sha256(TRAIN_MANIFEST),
    )
    field["cluster_ledger"] = led
    field["test_manifest_sha256"] = _sha256(TEST_MANIFEST)
    field["compile_seconds"] = round(time.perf_counter() - t0, 2)
    field["detector"] = "seam_geometry_field"
    field["clinical_grade"] = False
    field["what_is_different_from_geometry_field"] = (
        f"appends {N_SEAM} nonlocal-seam / packing-frustration columns "
        f"({len(SEAM_QTY)} quantities x 3 aggregations) under union with the "
        "645+624 geometry_field columns; same table topology"
    )
    field["geometry_families"] = {f: 3 * len(q) for f, q in FAMILIES}
    field["seam_quantities"] = len(SEAM_QTY)
    field["seam_columns"] = N_SEAM

    args.out.write_text(json.dumps(field, indent=2, allow_nan=False) + "\n")
    print(f"wrote {args.out.relative_to(ROOT)}  "
          f"{field['n_wires']} cols / {len(field['tables'])} tables  "
          f"in {field['compile_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
