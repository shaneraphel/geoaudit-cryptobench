#!/usr/bin/env python3
"""Do the second-moment, centred and extra-radius wires carry anything new?

The question is asked in the cheapest order. First a regularised linear
functional of all 516 digits, which is the same diagnostic used throughout: it
says how much signal the wires hold without any statement about how a table
bank would read it. On the 172-wire set that functional measured 0.7873 on the
pick half, and the table field then matched it. So the number to beat here is
0.7873; if the wider set does not clear it, the additions are redundant and the
counting field has nothing further to gain from them either.

Only if it clears it does the bank get built, over pair tables and pair plus
triple tables, with the ridge-regularised integer fan-out and the spread-matched
gate frozen earlier.

Ranked on the pick half of the training fold. The official test fold is not read
here, and has now been read twice; a third reading happens only if this shows a
clear gain, and would be disclosed as such.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_wide.py
"""
from __future__ import annotations

import gc
import json

import numpy as np

from pocket_bench.paths import ROOT

from counterattack_covering import compile_bank32, partitions
from counterattack_ridge import (
    class_scatter,
    quantize,
    ridge_direction,
    spread_matched_gate,
)
from counterattack_select import SEED, per_unit_auc, pooled_auc

CACHE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_WIDE.json"
L = 4
BASELINE_172 = {"linear_over_digits": 0.7873, "table_field": 0.7932}


def chain_digits_i8(F, n_res_per, levels=L):
    out = np.empty(F.shape, dtype=np.int8)
    off = 0
    for n in n_res_per:
        n = int(n)
        blk = F[off:off + n]
        r = np.empty(n)
        for j in range(F.shape[1]):
            x = blk[:, j]
            order = np.argsort(x, kind="stable")
            i = 0
            while i < n:
                k = i
                while k + 1 < n and x[order[k + 1]] == x[order[i]]:
                    k += 1
                r[order[i:k + 1]] = 0.5 * (i + k)
                i = k + 1
            out[off:off + n, j] = np.clip(
                np.floor(r / max(n - 1, 1) * levels), 0, levels - 1)
        off += n
    return out


def main() -> int:
    z = np.load(CACHE, allow_pickle=False)
    X, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    names = [str(s) for s in z["names"]]
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
    print(f"{len(units)} train units, {X.shape[1]} wires; fit {len(yfit)} "
          f"rows, pick {len(ypick)} rows", flush=True)

    D = chain_digits_i8(X, n_res)
    del X, z
    gc.collect()
    Dfit = np.ascontiguousarray(D[fm])
    Dpick = np.ascontiguousarray(D[pm])
    n_wires = D.shape[1]
    del D
    gc.collect()

    results = []

    # 1. the ceiling: a regularised linear functional of all digits
    Afit = Dfit.astype(np.float32)
    Apick = Dpick.astype(np.float32)
    for lam in (1e-6, 1e-3, 1e-1):
        w = ridge_direction(Afit, yfit, lam)
        S = Apick.astype(np.float64) @ w
        raw = per_unit_auc(S, ypick, n_pick_per)
        best = max(per_unit_auc(spread_matched_gate(S, ctr_pick, n_pick_per,
                                                    r, wt),
                                ypick, n_pick_per)
                   for r, wt in ((14.0, 1.0), (18.0, 0.5), (18.0, 1.0)))
        print(f"  [ceiling] linear over {n_wires} digits, ridge {lam:<6g}  "
              f"raw {raw:.4f}  gated {best:.4f}", flush=True)
        results.append({"kind": "ceiling",
                        "architecture": f"linear over {n_wires} digits, "
                                        f"ridge {lam:g}",
                        "pick_half_roc_auc_raw": raw,
                        "pick_half_roc_auc": best})
    del Afit, Apick
    gc.collect()

    ceiling = max(r["pick_half_roc_auc"] for r in results)
    print(f"\nceiling {ceiling:.4f} vs {BASELINE_172['linear_over_digits']:.4f} "
          f"for the 172-wire set: "
          f"{ceiling - BASELINE_172['linear_over_digits']:+.4f}", flush=True)

    if ceiling > BASELINE_172["linear_over_digits"] + 0.002:
        gini = np.array([abs(2.0 * pooled_auc(Dfit[:, j].astype(float), yfit)
                             - 1.0) for j in range(n_wires)])
        for basis_name, spec in (("pairs x8", [(2, 8)]),
                                 ("pairs x8 + triples x5", [(2, 8), (3, 5)])):
            tables = []
            for width, rounds in spec:
                tables += partitions(n_wires, width, rounds, gini, "random")
            A_fit, A_pick, occ = compile_bank32(Dfit, yfit, Dpick, tables,
                                                rate)
            print(f"\n{basis_name}: {len(tables)} tables, {occ:.0f} residues "
                  f"per occupied cell", flush=True)
            for lam in (0.1, 0.3, 1.0):
                m = quantize(ridge_direction(A_fit, yfit, lam), 32)
                S = A_pick.astype(np.float64) @ m.astype(np.float64)
                raw = per_unit_auc(S, ypick, n_pick_per)
                best, gname = -1.0, ""
                for r, wt in ((14.0, 1.0), (18.0, 0.5), (18.0, 1.0)):
                    a = per_unit_auc(spread_matched_gate(S, ctr_pick,
                                                         n_pick_per, r, wt),
                                     ypick, n_pick_per)
                    if a > best:
                        best, gname = a, f"r{int(r)} w{wt}"
                print(f"  ridge {lam:<5g} cap 32  raw {raw:.4f}  "
                      f"gated {best:.4f} ({gname})", flush=True)
                results.append({"kind": "bank", "basis": basis_name,
                                "n_tables": len(tables), "ridge": lam,
                                "cap": 32, "gate": gname,
                                "pick_half_roc_auc_raw": raw,
                                "pick_half_roc_auc": best})
            del A_fit, A_pick
            gc.collect()
    else:
        print("the wider wires do not clear the 172-wire ceiling; no bank "
              "is built and the test fold is not read again", flush=True)

    banks = [r for r in results if r["kind"] == "bank"]
    winner = max(banks, key=lambda r: r["pick_half_roc_auc"]) if banks else None
    if winner:
        print(f"\nbest bank: {winner['basis']}, ridge {winner['ridge']:g} -> "
              f"{winner['pick_half_roc_auc']:.4f} "
              f"(172-wire table field on this split "
              f"{BASELINE_172['table_field']:.4f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_wide.v1",
        "clinical_grade": False,
        "n_wires": n_wires,
        "wire_families": ["local", "mean@6/10/14/20/26", "sd@6/14/20",
                          "centred@6/14/20"],
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED},
        "reference_172_wires": BASELINE_172,
        "candidates": results,
        "ceiling": ceiling,
        "ceiling_gain_over_172": ceiling - BASELINE_172["linear_over_digits"],
        "selected": winner,
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
