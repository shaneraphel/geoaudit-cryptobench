#!/usr/bin/env python3.12
"""Compile geometry + spectral seam + polar_gate. clinical_grade=false."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pocket_bench.methods.geometry_wires import N_DISP_CARRIED
from pocket_bench.methods.table_field import code_sha256, compile_field
from pocket_bench.paths import ROOT
from compile_algebraic_field import (
    TEST_MANIFEST,
    TRAIN_MANIFEST,
    _sha256,
    assert_no_leak,
)

OUT = ROOT / "data/cryptobench_apo/SEAM_POLARGATE_FIELD.json"
GEO = (
    ("backbone", "_backbone_cache_train.npz", None),
    ("sidechain", "_sidechain_cache_train.npz", None),
    ("void", "_void_cache_train.npz", None),
    ("displacement", "_displacement_cache_train.npz", N_DISP_CARRIED),
)


def _load(path: Path, prefix: str, carry=None):
    z = np.load(path)
    C, cols = z["C"], [str(s) for s in z["names"]]
    z.close()
    if carry is not None:
        pa = C.shape[1] // 3
        take = [a * pa + q for a in range(3) for q in range(carry)]
        C, cols = C[:, take], [cols[i] for i in take]
    return C, [f"{prefix}~{c}" for c in cols]


def main() -> int:
    led = assert_no_leak()
    z = np.load(ROOT / "data/cryptobench_apo/_wide_cache_train.npz")
    Xw, prop = z["X"], z["propensity_table"]
    names = [str(s) for s in z["names"]]
    y, ctr, n_res = z["y"], z["ctr"], z["n_res_per"]
    z.close()
    blocks, gnames = [], []
    for fam, fn, carry in GEO:
        C, cols = _load(ROOT / "data/cryptobench_apo" / fn, fam, carry)
        blocks.append(C)
        gnames.extend(cols)
    Xg = np.concatenate(blocks, 1)
    Xs, sn = _load(
        ROOT / "data/cryptobench_apo/_nonlocal_seam_cache_train.npz", "seam"
    )
    Xp, pn = _load(
        ROOT / "data/cryptobench_apo/_polar_gate_cache_train.npz", "polargate"
    )
    X = np.concatenate(
        [
            np.asarray(Xw, np.float32),
            np.asarray(Xg, np.float32),
            np.asarray(Xs, np.float32),
            np.asarray(Xp, np.float32),
        ],
        1,
    )
    all_names = tuple(names + gnames + sn + pn)
    print(
        f"cols {Xw.shape[1]}+{Xg.shape[1]}+{Xs.shape[1]}+{Xp.shape[1]}"
        f"={X.shape[1]}",
        flush=True,
    )
    t0 = time.perf_counter()
    field = compile_field(
        X, y, ctr, n_res, all_names, prop,
        code_sha256=code_sha256(),
        train_manifest_sha256=_sha256(TRAIN_MANIFEST),
    )
    field.update(
        cluster_ledger=led,
        test_manifest_sha256=_sha256(TEST_MANIFEST),
        compile_seconds=round(time.perf_counter() - t0, 2),
        detector="seam_polargate_field",
        clinical_grade=False,
        seam_columns=int(Xs.shape[1]),
        polargate_columns=int(Xp.shape[1]),
        what_is_different_from_seam_geometry_field=(
            f"appends {Xp.shape[1]} polar-gate columns "
            "(charged/polar cluster geometry, salt frustration) "
            "targeting anti-ranked cryptic sites vs pLM-NN"
        ),
    )
    OUT.write_text(json.dumps(field, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT} in {field['compile_seconds']}s n_wires={field['n_wires']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
