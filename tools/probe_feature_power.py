"""Per-invariant discriminative power on the official test fold.

Diagnostic only: reports the residue ROC-AUC each single invariant achieves on its
own, plus the compiled field, so it is visible whether the associative memory is
adding anything over its strongest input or merely tracking it.

Usage: PYTHONPATH=src python3.12 tools/probe_feature_power.py --limit 48 --jobs 4
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST_MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"


def _one(entry: dict):
    from pocket_bench.methods.quaternary_lut import (
        FEATURE_NAMES, load_field, receptor_residue_features,
    )
    from pocket_bench.metrics import roc_auc
    try:
        lab = json.loads((ROOT / entry["label_path"]).read_text())
        truth = {int(r) for r in (lab.get("cryptic_residues")
                                  or lab.get("binding_residues") or [])}
        if not truth:
            return None
        resseq, F = receptor_residue_features(ROOT / entry["receptor_path"],
                                              chain=entry["chain"])
        y = [1 if int(r) in truth else 0 for r in resseq]
        if sum(y) == 0 or sum(y) == len(y):
            return None
        out = []
        for j in range(len(FEATURE_NAMES)):
            a = roc_auc(list(F[:, j]), y)
            b = roc_auc(list(-F[:, j]), y)
            out.append(max(a or 0.5, b or 0.5))
        out.append(roc_auc(list(load_field().lookup(F)), y) or 0.5)
        return out
    except Exception:  # noqa: BLE001
        return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args(argv)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"
    from pocket_bench.methods.quaternary_lut import FEATURE_NAMES

    entries = json.loads(TEST_MANIFEST.read_text())["entries"]
    step = max(1, len(entries) // args.limit)
    entries = entries[::step][:args.limit]
    t0 = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
        out = [o for o in ex.map(_one, entries, chunksize=1) if o]
    arr = np.asarray(out, dtype=float)
    print(f"units: {len(arr)}   {time.perf_counter()-t0:.0f}s")
    names = list(FEATURE_NAMES) + ["COMPILED FIELD"]
    for j, nm in enumerate(names):
        print(f"  {nm:20s} best-oriented ROC-AUC = {arr[:, j].mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
