"""Fast per-residue ROC-AUC probe of the resolution field on the official test fold.

Bypasses the full benchmark runner so the field can be checked in seconds instead
of an hour. Uses the same metric function the frozen table uses.

Usage: PYTHONPATH=src python3.12 tools/probe_field_auc.py --limit 48 --jobs 8
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
        load_field, receptor_residue_features,
    )
    from pocket_bench.metrics import roc_auc, average_precision
    try:
        lab = json.loads((ROOT / entry["label_path"]).read_text())
        truth = {int(r) for r in (lab.get("cryptic_residues")
                                  or lab.get("binding_residues") or [])}
        if not truth:
            return None
        field = load_field()
        resseq, F = receptor_residue_features(ROOT / entry["receptor_path"],
                                              chain=entry["chain"])
        y = [1 if int(r) in truth else 0 for r in resseq]
        if sum(y) == 0 or sum(y) == len(y):
            return None
        cell = field.lookup(F)
        agg = field.rank_aggregate(F)
        res = field.resolve(F)
        return (roc_auc(list(cell), y), average_precision(list(res), y),
                roc_auc(list(agg), y), roc_auc(list(res), y))
    except Exception as exc:  # noqa: BLE001
        return ("ERR", str(exc)[:120])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=48)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"

    entries = json.loads(TEST_MANIFEST.read_text())["entries"]
    if args.limit:
        # deterministic stride so the probe subset is representative, not the
        # alphabetical head of the fold
        step = max(1, len(entries) // args.limit)
        entries = entries[::step][:args.limit]
    t0 = time.perf_counter()
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=ctx) as ex:
        out = list(ex.map(_one, entries, chunksize=1))
    errs = [o for o in out if o and o[0] == "ERR"]
    vals = [o for o in out if o and o[0] != "ERR"]
    arr = np.array([[v[0], v[1], v[2], v[3]] for v in vals], dtype=float)
    print(f"units scored: {len(arr)}/{len(entries)}   errors: {len(errs)}   "
          f"{time.perf_counter()-t0:.0f}s")
    for e in errs[:5]:
        print("   ERR:", e[1])
    if len(arr):
        print(f"  cell only      ROC-AUC = {arr[:,0].mean():.4f}")
        print(f"  rank aggregate ROC-AUC = {arr[:,2].mean():.4f}")
        print(f"  RESOLVED       ROC-AUC = {arr[:,3].mean():.4f}  "
              f"PR-AUC = {arr[:,1].mean():.4f}")
        print(f"  reference: rigid 0.664, sstar 0.655, p2rank 0.793")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
