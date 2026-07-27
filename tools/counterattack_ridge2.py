#!/usr/bin/env python3
"""Extend the regularised pair-table sweep past the edge of the first grid.

The first ridge sweep ended at its own boundary. A ridge of 0.1 of the trace was
the largest value tried and it was the best at both pool sizes -- 0.7898 at 1032
tables and 0.7906 at 1720, against 0.7844 and 0.6846 unregularised -- so the
optimum is at or beyond the edge of that grid and the pool-size curve is still
rising. Both need following.

The limit is known: as the ridge grows the direction tends to the difference of
class means, which weights every table by how far its output separates the
classes on its own and performs no decorrelation at all. So there is an interior
optimum, and the sweep now brackets it.

Three things are opened up together, since the scatter is accumulated in chunks
and pool size is no longer a memory constraint:

  ridge      0.03 to 3.0, bracketing the 0.1 edge on both sides
  pool       up to 40 partition rounds, 3440 pair tables
  basis      pairs alone, and pairs together with three-wire tables, which have
             64 cells and ~1900 residues per cell and can state a three-way
             interaction the pair tables cannot

The gate is the spread-matched neighbourhood mean of the frozen readout, now
also tried as a sum over several radii, since a cryptic site has no single
characteristic size.

Ranked on the pick half. The official test fold is not read here.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_ridge2.py
"""
from __future__ import annotations

import gc
import json

import numpy as np

from pocket_bench.paths import ROOT

from counterattack_covering import compile_bank32, partitions
from counterattack_ridge import (
    neighbourhood_mean,
    quantize,
    ridge_direction,
    spread_matched_gate,
)
from counterattack_select import SEED, chain_digits, per_unit_auc, pooled_auc

CACHE = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
DIGITS = ROOT / "data/cryptobench_apo/_expanded_digits_train.npy"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_RIDGE2.json"

GATES = (("r14 w1.0", ((14.0, 1.0),)),
         ("r18 w0.5", ((18.0, 0.5),)),
         ("r18 w1.0", ((18.0, 1.0),)),
         ("r8+r14+r20 w0.5", ((8.0, 0.5), (14.0, 0.5), (20.0, 0.5))))


def apply_gates(S, ctr, n_res_per, spec):
    if len(spec) == 1:
        r, w = spec[0]
        return spread_matched_gate(S, ctr, n_res_per, r, w)
    out = S.copy()
    sd_s = float(np.std(S))
    for r, w in spec:
        g = neighbourhood_mean(S, ctr, n_res_per, r)
        sd_g = float(np.std(g))
        if sd_g > 0:
            out = out + w * g * (sd_s / sd_g)
    return out


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
    del X, D, z
    gc.collect()
    gini = np.array([abs(2.0 * pooled_auc(Dfit[:, j].astype(float), yfit) - 1.0)
                     for j in range(n_wires)])

    bases = [("pairs x20", [(2, 20)]),
             ("pairs x40", [(2, 40)]),
             ("pairs x20 + triples x12", [(2, 20), (3, 12)])]

    results = []
    for basis_name, spec in bases:
        tables = []
        for width, rounds in spec:
            tables += partitions(n_wires, width, rounds, gini, "random")
        A_fit, A_pick, occ = compile_bank32(Dfit, yfit, Dpick, tables, rate)
        print(f"{basis_name}: {len(tables)} tables, {occ:.0f} residues per "
              f"occupied cell", flush=True)
        for lam in (0.03, 0.1, 0.3, 1.0, 3.0):
            w = ridge_direction(A_fit, yfit, lam)
            for cap in (32, 64):
                m = quantize(w, cap)
                S = A_pick.astype(np.float64) @ m.astype(np.float64)
                raw = per_unit_auc(S, ypick, n_pick_per)
                best = None
                for gname, gspec in GATES:
                    a = per_unit_auc(apply_gates(S, ctr_pick, n_pick_per,
                                                 gspec), ypick, n_pick_per)
                    if best is None or a > best[0]:
                        best = (a, gname)
                results.append({
                    "basis": basis_name, "n_tables": len(tables),
                    "ridge": lam, "cap": cap,
                    "total_fan_out": int(np.abs(m).sum()),
                    "n_nonzero": int((m != 0).sum()),
                    "pick_half_roc_auc_raw": raw,
                    "pick_half_roc_auc": best[0], "gate": best[1],
                })
                print(f"  ridge {lam:<5g} cap {cap:3d}  raw {raw:.4f}  "
                      f"gated {best[0]:.4f} ({best[1]})", flush=True)
        del A_fit, A_pick
        gc.collect()

    winner = max(results, key=lambda r: r["pick_half_roc_auc"])
    print(f"\nbest: {winner['basis']}, {winner['n_tables']} tables, ridge "
          f"{winner['ridge']:g}, cap {winner['cap']}, gate {winner['gate']} "
          f"-> {winner['pick_half_roc_auc']:.4f}")
    print("reference on this split: first ridge sweep 0.7906, signed stagewise "
          "0.7812, fitted linear functional of the raw digits 0.7873")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_ridge2.v1",
        "clinical_grade": False,
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED, "n_wires": n_wires},
        "candidates": results,
        "selected": winner,
        "reference_same_split": {"first_ridge_sweep": 0.7906,
                                 "signed_stagewise": 0.7812,
                                 "fitted_linear_over_digits": 0.7873},
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
