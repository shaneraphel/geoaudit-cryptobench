#!/usr/bin/env python3.12
"""Compile geometry + seam(spectral) + toeplitz_3d. clinical_grade=false."""
from __future__ import annotations

import json, time
from pathlib import Path
import numpy as np
from pocket_bench.methods.geometry_wires import N_COLUMNS as N_GEO, N_DISP_CARRIED
from pocket_bench.methods.nonlocal_seam import COLUMNS as SEAM_QTY
from pocket_bench.methods.toeplitz_3d import COLUMNS as TQ
from pocket_bench.methods.table_field import code_sha256, compile_field
from pocket_bench.paths import ROOT
from compile_algebraic_field import TEST_MANIFEST, TRAIN_MANIFEST, _sha256, assert_no_leak

OUT = ROOT / "data/cryptobench_apo/SEAM_TOEPLITZ_FIELD.json"
GEO = (
    ("backbone", "_backbone_cache_train.npz", None),
    ("sidechain", "_sidechain_cache_train.npz", None),
    ("void", "_void_cache_train.npz", None),
    ("displacement", "_displacement_cache_train.npz", N_DISP_CARRIED),
)


def _load(path, n, prefix, carry=None):
    z = np.load(path)
    C, cols = z["C"], [str(s) for s in z["names"]]; z.close()
    if carry is not None:
        pa = C.shape[1] // 3
        take = [a * pa + q for a in range(3) for q in range(carry)]
        C, cols = C[:, take], [cols[i] for i in take]
    return C, [f"{prefix}~{c}" for c in cols]


def main():
    led = assert_no_leak()
    z = np.load(ROOT / "data/cryptobench_apo/_wide_cache_train.npz")
    Xw, prop, names = z["X"], z["propensity_table"], [str(s) for s in z["names"]]
    y, ctr, n_res = z["y"], z["ctr"], z["n_res_per"]; z.close()
    blocks, gnames = [], []
    for fam, fn, carry in GEO:
        C, cols = _load(ROOT / "data/cryptobench_apo" / fn, Xw.shape[0], fam, carry)
        blocks.append(C); gnames.extend(cols)
    Xg = np.concatenate(blocks, 1)
    Xs, sn = _load(ROOT / "data/cryptobench_apo/_nonlocal_seam_cache_train.npz",
                   Xw.shape[0], "seam")
    Xt, tn = _load(ROOT / "data/cryptobench_apo/_toeplitz_3d_cache_train.npz",
                   Xw.shape[0], "toeplitz")
    X = np.concatenate([np.asarray(Xw, np.float32), np.asarray(Xg, np.float32),
                        np.asarray(Xs, np.float32), np.asarray(Xt, np.float32)], 1)
    all_names = tuple(names + gnames + sn + tn)
    print(f"cols {Xw.shape[1]}+{Xg.shape[1]}+{Xs.shape[1]}+{Xt.shape[1]}={X.shape[1]}",
          flush=True)
    t0 = time.perf_counter()
    field = compile_field(X, y, ctr, n_res, all_names, prop,
                          code_sha256=code_sha256(),
                          train_manifest_sha256=_sha256(TRAIN_MANIFEST))
    field.update(cluster_ledger=led, test_manifest_sha256=_sha256(TEST_MANIFEST),
                 compile_seconds=round(time.perf_counter() - t0, 2),
                 detector="seam_toeplitz_field", clinical_grade=False,
                 seam_columns=int(Xs.shape[1]), toeplitz_columns=int(Xt.shape[1]))
    OUT.write_text(json.dumps(field, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT} in {field['compile_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
