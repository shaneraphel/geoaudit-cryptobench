#!/usr/bin/env python3
"""Does the signed stagewise rule survive the pool size that broke the solve?

This is the last question answered on the training fold, and it is the one that
decides which fusion is taken to the test set.

Rounding a closed-form direction over the pair-table outputs measured 0.7844 on
the pick half with a pool of 1032 tables and 0.6846 with a pool of 1720. The
collapse is not mysterious: 1720 random pairs drawn from a lattice of 14706
contain near-duplicates, the covariance of the table outputs goes near-singular,
and the solved direction puts large opposing coefficients on columns that are
copies of each other. Under that reading the failure is specific to inverting a
matrix, and a stagewise rule should be immune, because a table that restates one
already chosen cannot improve the running rank statistic and so is never taken.

Signed stagewise selection measured 0.7812 at 1032 tables -- 0.003 below the
solve, and it saturates at 87 accepted rounds out of 400, so it is not being
cut short. If it holds that value as the pool grows, then the solve buys three
thousandths of AUC in exchange for a fusion that fails when the bank is a little
larger, and the stagewise rule is the one to freeze.

Three pool sizes, additive and signed, everything ranked on the pick half. The
official test fold is not read here.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_stability.py
"""
from __future__ import annotations

import json

import numpy as np

from pocket_bench.paths import ROOT

from counterattack_covering import (
    compile_bank32,
    direction_from_matrix,
    partitions,
)
from counterattack_greedy import stagewise
from counterattack_quantized import quantize
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
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_STABILITY.json"


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
    Dfit, Dpick = D[fm], D[pm]
    n_wires = D.shape[1]
    gini_wire = np.array([abs(2.0 * pooled_auc(Dfit[:, j].astype(float), yfit)
                              - 1.0) for j in range(n_wires)])

    def gated(S):
        G = np.sum([_unit(gate(S, ctr_pick, n_pick_per, r))
                    for r in GATE_RADII], axis=0)
        return per_unit_auc(_unit(S) + _unit(G), ypick, n_pick_per)

    results = []
    for pool_rounds in (6, 12, 20):
        tables = partitions(n_wires, 2, pool_rounds, gini_wire, "random")
        A_fit, A_pick, occ = compile_bank32(Dfit, yfit, Dpick, tables, rate)
        row = {"pool_rounds": pool_rounds, "n_pool_tables": len(tables),
               "mean_residues_per_occupied_cell": occ}

        for label, signed in (("signed stagewise", True),
                              ("additive stagewise", False)):
            mult, trace = stagewise(A_fit, yfit, 400, signed=signed)
            a = gated(A_pick.astype(np.float64) @ mult.astype(np.float64))
            row[label] = a
            row[f"{label} accepted"] = len(trace)
            row[f"{label} distinct tables"] = int((mult != 0).sum())
            row[f"{label} total fan-out"] = int(np.abs(mult).sum())

        m = quantize(direction_from_matrix(A_fit, yfit), 32)
        row["solved direction cap=32"] = gated(
            A_pick.astype(np.float64) @ m.astype(np.float64))

        print(f"  pool {len(tables):5d} tables:  "
              f"signed {row['signed stagewise']:.4f} "
              f"({row['signed stagewise accepted']:3d} rounds, "
              f"{row['signed stagewise distinct tables']:3d} tables)   "
              f"additive {row['additive stagewise']:.4f}   "
              f"solved {row['solved direction cap=32']:.4f}", flush=True)
        results.append(row)
        del A_fit, A_pick

    signed_vals = [r["signed stagewise"] for r in results]
    solved_vals = [r["solved direction cap=32"] for r in results]
    print(f"\nspread across pool sizes: signed "
          f"{max(signed_vals) - min(signed_vals):.4f}, "
          f"solved {max(solved_vals) - min(solved_vals):.4f}")
    best = max(results, key=lambda r: r["signed stagewise"])
    print(f"freezing: signed stagewise on a pool of {best['n_pool_tables']} "
          f"pair tables, pick-half ROC-AUC {best['signed stagewise']:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_stability.v1",
        "clinical_grade": False,
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED, "n_wires": n_wires},
        "candidates": results,
        "signed_spread_across_pool_sizes": max(signed_vals) - min(signed_vals),
        "solved_spread_across_pool_sizes": max(solved_vals) - min(solved_vals),
        "frozen": {"fusion": "signed stagewise integer multiplicities",
                   "table_width": 2, "pool_rounds": best["pool_rounds"],
                   "n_pool_tables": best["n_pool_tables"],
                   "pick_half_roc_auc": best["signed stagewise"]},
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
