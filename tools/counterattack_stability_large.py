#!/usr/bin/env python3
"""The 1720-table point of the stability curve, signed stagewise only.

The combined sweep could not reach this pool size: forming the covariance of
1720 table outputs in double precision needs several gigabytes on top of the
banks themselves, and the process was killed there twice. That is worth stating
plainly, because it is the second cost of the solved fusion after its collapse
to 0.6846 at this same size -- it does not fit in memory at a bank size the
stagewise rule handles without difficulty.

Signed stagewise never forms a covariance. It holds the bank, a subsample of the
fit rows, and a running sum.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_stability_large.py
"""
from __future__ import annotations

import gc
import json

import numpy as np

from pocket_bench.paths import ROOT

from counterattack_covering import compile_bank32, partitions
from counterattack_greedy import stagewise
from counterattack_select import (
    GATE_RADII,
    SEED,
    _unit,
    chain_digits,
    gate,
    per_unit_auc,
    pooled_auc,
)

CACHE = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
DIGITS = ROOT / "data/cryptobench_apo/_expanded_digits_train.npy"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_STABILITY_LARGE.json"


def main() -> int:
    z = np.load(CACHE, allow_pickle=False)
    X, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"]
                  for e in json.loads(MANIFEST.read_text())["entries"]}
    clusters = sorted({cluster_of[u] for u in units})
    rng = np.random.default_rng(SEED)
    rng.shuffle(clusters)
    fit_clusters = set(clusters[:len(clusters) // 2])
    is_fit = np.array([cluster_of[u] in fit_clusters for u in units])
    row_unit = np.repeat(np.arange(len(units)), n_res)
    fm, pm = is_fit[row_unit], ~is_fit[row_unit]
    n_pick_per = np.array([n for n, f in zip(n_res, is_fit) if not f])
    yfit, ypick, ctr_pick = y[fm], y[pm], ctr[pm]
    rate = float(yfit.mean())

    D = np.load(DIGITS) if DIGITS.exists() else chain_digits(X, n_res)
    Dfit = np.ascontiguousarray(D[fm], dtype=np.int32)
    Dpick = np.ascontiguousarray(D[pm], dtype=np.int32)
    n_wires = D.shape[1]
    # the banks at this pool size are gigabytes; nothing else may stay resident
    del X, D, z
    gc.collect()
    gini_wire = np.array([abs(2.0 * pooled_auc(Dfit[:, j].astype(float), yfit)
                              - 1.0) for j in range(n_wires)])

    out = []
    for pool_rounds in (20,):
        tables = partitions(n_wires, 2, pool_rounds, gini_wire, "random")
        A_fit, A_pick, occ = compile_bank32(Dfit, yfit, Dpick, tables, rate)
        mult, trace = stagewise(A_fit, yfit, 400, signed=True,
                                subsample=16000)
        del A_fit
        gc.collect()
        S = A_pick.astype(np.float64) @ mult.astype(np.float64)
        G = np.sum([_unit(gate(S, ctr_pick, n_pick_per, r))
                    for r in GATE_RADII], axis=0)
        a = per_unit_auc(_unit(S) + _unit(G), ypick, n_pick_per)
        print(f"  pool {len(tables):5d} tables:  signed {a:.4f} "
              f"({len(trace)} rounds, {int((mult != 0).sum())} tables, "
              f"fan-out {int(np.abs(mult).sum())})", flush=True)
        out.append({"pool_rounds": pool_rounds, "n_pool_tables": len(tables),
                    "signed stagewise": a, "accepted": len(trace),
                    "distinct_tables": int((mult != 0).sum()),
                    "total_fan_out": int(np.abs(mult).sum()),
                    "mean_residues_per_occupied_cell": occ})
        del A_pick
        gc.collect()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_stability_large.v1",
        "clinical_grade": False,
        "note": "signed stagewise only; the solved fusion neither fits in "
                "memory nor holds its value at these pool sizes",
        "candidates": out,
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
