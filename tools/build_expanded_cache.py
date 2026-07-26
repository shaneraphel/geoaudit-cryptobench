#!/usr/bin/env python3
"""Derive the expanded wire matrix from the cached invariants. No re-extraction.

Both additions -- published per-residue chemistry and the multi-radius context
transform -- are functions of quantities already in the cache (the 35
invariants, the residue codes, the residue centroids), so the expansion costs
minutes of arithmetic rather than the hours that surface extraction takes.

The propensity counter is compiled on the TRAINING fold and applied to both, so
no test residue ever contributes a count to a table that scores it.

Usage: PYTHONPATH=src python3.12 tools/build_expanded_cache.py
"""
from __future__ import annotations

import json
import time

import numpy as np

from pocket_bench.methods.algebraic_descriptors import FEATURE_NAMES
from pocket_bench.methods.expanded_descriptors import build_expanded
from pocket_bench.paths import ROOT

IN_TRAIN = ROOT / "data/cryptobench_apo/_cascade_cache_train.npz"
IN_TEST = ROOT / "data/cryptobench_apo/_cascade_cache_test.npz"
OUT_TRAIN = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
OUT_TEST = ROOT / "data/cryptobench_apo/_expanded_cache_test.npz"


def main() -> int:
    ztr = np.load(IN_TRAIN, allow_pickle=False)
    zte = np.load(IN_TEST, allow_pickle=False)

    t0 = time.perf_counter()
    Xtr, names, prop = build_expanded(
        ztr["F"], ztr["codes"], ztr["ctr"], ztr["n_res_per"], FEATURE_NAMES,
        y=ztr["y"])
    print(f"train {Xtr.shape} in {time.perf_counter() - t0:.0f}s", flush=True)

    t0 = time.perf_counter()
    Xte, names_te, _ = build_expanded(
        zte["F"], zte["codes"], zte["ctr"], zte["n_res_per"], FEATURE_NAMES,
        prop_table=prop)
    print(f"test  {Xte.shape} in {time.perf_counter() - t0:.0f}s", flush=True)
    assert names == names_te

    for out, z, X in ((OUT_TRAIN, ztr, Xtr), (OUT_TEST, zte, Xte)):
        np.savez_compressed(
            out, X=X, y=z["y"], codes=z["codes"], ctr=z["ctr"],
            n_res_per=z["n_res_per"], units=z["units"],
            names=np.asarray(names), propensity_table=prop)
        print(f"wrote {out.relative_to(ROOT)}  {X.shape}")

    (ROOT / "data/cryptobench_apo/EXPANDED_WIRES.json").write_text(json.dumps({
        "schema": "geoaudit.expanded_wires.v1",
        "clinical_grade": False,
        "n_wires": len(names),
        "names": list(names),
        "propensity_table": prop.tolist(),
        "propensity_note": "P(cryptic | residue type), bincount over the "
                           "training fold with one pseudo-count per class",
    }, indent=2, allow_nan=False) + "\n")
    print(f"{len(names)} wires")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
