#!/usr/bin/env python3
"""The last training-fold sweep: 645 wires, and a pool no longer capped by RAM.

Two changes since the 516-wire run, which reached 0.8010 on the pick half
against 0.7932 for the 172-wire field and 0.7936 for a continuous functional of
its own inputs.

The wires now include a local rank, |{j in N_r(i) : x_j < x_i}| / |N_r(i)|, at
6, 14 and 20 A. It is the only statistic in the set that is not a moment: it
asks where a residue sits in the order of its neighbourhood rather than how far
it is from the neighbourhood's centre, so it is unchanged by any monotone
rescaling of the wire, and it tells apart a residue that is marginally the most
hydrophobic of its neighbours from one that is marginally the least -- a
distinction the centred difference blurs into two small numbers of opposite
sign. 43 x 15 = 645.

The bank is no longer held as a matrix. Rows x tables x 4 bytes reached several
gigabytes and killed three earlier runs, which is why every sweep so far stopped
at a pool size chosen by memory rather than by measurement. Addresses are now
recomputed per row block, so the pool is limited only by the K x K solve.

Ranked on the pick half. The official test fold is not read here; it has been
read twice and will be read once more, for whichever single configuration comes
out of this.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_wide2.py
"""
from __future__ import annotations

import gc
import json

import numpy as np

from pocket_bench.methods.table_bank import (
    cell_offsets,
    chain_digits,
    compile_cells,
    integer_fanout,
    partition_tables,
    score,
)
from pocket_bench.paths import ROOT

from counterattack_ridge import spread_matched_gate
from counterattack_select import SEED, per_unit_auc

CACHE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_WIDE2.json"
GATES = ((14.0, 1.0), (18.0, 0.5), (18.0, 1.0))
REFERENCE = {"172_wire_table_field": 0.7932,
             "516_wire_table_field": 0.8010,
             "516_wire_continuous_ceiling": 0.7936}


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

    D = chain_digits(X, n_res)
    del X, z
    gc.collect()
    Dfit = np.ascontiguousarray(D[fm])
    Dpick = np.ascontiguousarray(D[pm])
    n_wires = D.shape[1]
    del D
    gc.collect()
    print(f"{n_wires} wires; fit {len(yfit)} rows, pick {len(ypick)} rows",
          flush=True)

    results = []
    for basis_name, spec in (("pairs x8", [(2, 8)]),
                             ("pairs x16", [(2, 16)]),
                             ("pairs x16 + triples x8", [(2, 16), (3, 8)])):
        tables = []
        for width, rounds in spec:
            tables += partition_tables(n_wires, width, rounds, SEED)
        offsets = cell_offsets(tables)
        frac, tot = compile_cells(Dfit, yfit, tables, offsets)
        empty = int((tot == 0).sum())
        print(f"\n{basis_name}: {len(tables)} tables, {len(frac)} cells, "
              f"{empty} never addressed ({100.0 * empty / len(frac):.2f}%), "
              f"{tot[tot > 0].mean():.0f} residues per occupied cell",
              flush=True)
        for ridge in (0.03, 0.1, 0.3):
            m = integer_fanout(Dfit, yfit, tables, offsets, frac, ridge, 32)
            S = score(Dpick, tables, offsets, frac, m)
            raw = per_unit_auc(S, ypick, n_pick_per)
            best, gname = -1.0, ""
            for r, wt in GATES:
                a = per_unit_auc(
                    spread_matched_gate(S, ctr_pick, n_pick_per, r, wt),
                    ypick, n_pick_per)
                if a > best:
                    best, gname = a, f"r{int(r)} w{wt}"
            print(f"  ridge {ridge:<5g} cap 32  raw {raw:.4f}  "
                  f"gated {best:.4f} ({gname})  fan-out "
                  f"{int(np.abs(m).sum())}", flush=True)
            results.append({
                "basis": basis_name, "spec": spec, "n_tables": len(tables),
                "n_cells": int(len(frac)), "n_cells_never_addressed": empty,
                "mean_residues_per_occupied_cell": float(tot[tot > 0].mean()),
                "ridge": ridge, "cap": 32, "gate": gname,
                "total_fan_out": int(np.abs(m).sum()),
                "n_tables_used": int((m != 0).sum()),
                "pick_half_roc_auc_raw": raw,
                "pick_half_roc_auc": best,
            })
        del frac, tot
        gc.collect()

    winner = max(results, key=lambda r: r["pick_half_roc_auc"])
    print(f"\nfrozen: {winner['basis']}, {winner['n_tables']} tables, ridge "
          f"{winner['ridge']:g}, cap {winner['cap']}, gate {winner['gate']} "
          f"-> pick-half {winner['pick_half_roc_auc']:.4f}")
    for k, v in REFERENCE.items():
        print(f"  {k:34s} {v:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_wide2.v1",
        "clinical_grade": False,
        "n_wires": n_wires,
        "wire_families": ["local", "mean@6/10/14/20/26", "sd@6/14/20",
                          "centred@6/14/20", "localrank@6/14/20"],
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED},
        "reference_same_split": REFERENCE,
        "candidates": results,
        "selected": winner,
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
