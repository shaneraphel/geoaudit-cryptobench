#!/usr/bin/env python3
"""The stagewise search was stopping on noise, not on information.

The first reading of the official test fold gave 0.7804 against P2Rank's 0.7935,
a paired difference of -0.0131 with a 95% interval of [-0.0342, +0.0072]:
statistical parity, not a win. The reading also exposed a defect that has
nothing to do with the architecture.

On the fit half the signed search accepted 87 rounds and kept 72 tables. On the
whole training fold -- twice the rows, the same table pool -- it accepted 63
rounds and kept 42. More data producing a smaller circuit is backwards, and the
cause is the stopping rule. Candidates are ranked on a fixed 24000-row subsample
and a round is accepted only if it raises that subsample's AUC by more than
1e-7. The sampling error of an AUC on 24000 rows at a 7% positive rate is of
order 1e-3, so once the true per-round gain falls below the noise floor the
search stops, and where that floor sits relative to the gains depends on how the
subsample happens to fall. Nothing about the field is being exhausted; the
measurement is.

Two changes, both decided here on the training fold and neither of them touching
what the circuit computes:

  the criterion is measured on more rows, lowering the noise floor;
  the search runs a fixed budget of rounds instead of halting at the first
  non-improving one, and the round count is read off the pick half like every
  other structural choice.

The pick half is the arbiter, as before. The test fold is not read here; it will
be read once more afterwards, and both readings will be reported.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_stopping.py
"""
from __future__ import annotations

import gc
import json

import numpy as np

from pocket_bench.paths import ROOT

from counterattack_covering import compile_bank32, partitions
from counterattack_greedy import auc_all_columns
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
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_STOPPING.json"
BUDGET = 200
PREFIXES = (40, 60, 87, 110, 140, 170, 200)


def stagewise_trace(A_fit, yfit, budget, subsample, seed=SEED):
    """Run a fixed budget of signed rounds; return the ordered picks.

    No early exit. A round that cannot improve the criterion still contributes
    its best available table, so short prefixes of the trace are the useful
    objects and the prefix length is chosen on the pick half.
    """
    n, K = A_fit.shape
    rng = np.random.default_rng(seed)
    idx = (np.sort(rng.choice(n, subsample, replace=False))
           if n > subsample else np.arange(n))
    A_sub, y_sub = A_fit[idx], yfit[idx]
    pos = y_sub == 1
    n_pos, n_neg = int(pos.sum()), int(len(y_sub) - pos.sum())

    running = np.zeros(len(y_sub), dtype=np.float32)
    picks = []
    for _ in range(budget):
        up = auc_all_columns(running[:, None] + A_sub, pos, n_pos, n_neg)
        j, sign, val = int(np.argmax(up)), 1, float(up.max())
        dn = auc_all_columns(running[:, None] - A_sub, pos, n_pos, n_neg)
        if dn.max() > val:
            j, sign, val = int(np.argmax(dn)), -1, float(dn.max())
        picks.append((j, sign, val))
        running = running + sign * A_sub[:, j]
    del A_sub
    gc.collect()
    return picks


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
    gini_wire = np.array([abs(2.0 * pooled_auc(Dfit[:, j].astype(float), yfit)
                              - 1.0) for j in range(n_wires)])

    tables = partitions(n_wires, 2, 12, gini_wire, "random")
    A_fit, A_pick, occ = compile_bank32(Dfit, yfit, Dpick, tables, rate)
    print(f"pool {len(tables)} pair tables, {occ:.0f} residues per occupied "
          f"cell; fit {len(yfit)} rows, pick {len(ypick)} rows\n", flush=True)

    results = []
    for subsample in (24000, 60000, len(yfit)):
        picks = stagewise_trace(A_fit, yfit, BUDGET, subsample)
        mult = np.zeros(len(tables), dtype=np.int64)
        S = np.zeros(A_pick.shape[0], dtype=np.float64)
        tag = ("all rows" if subsample >= len(yfit)
               else f"{subsample} rows")
        for r, (j, sign, _) in enumerate(picks, start=1):
            mult[j] += sign
            S = S + sign * A_pick[:, j].astype(np.float64)
            if r not in PREFIXES:
                continue
            G = np.sum([_unit(gate(S, ctr_pick, n_pick_per, rr))
                        for rr in GATE_RADII], axis=0)
            a = per_unit_auc(_unit(S) + _unit(G), ypick, n_pick_per)
            print(f"  criterion on {tag:>10s}, {r:3d} rounds -> "
                  f"{int((mult != 0).sum()):3d} tables, "
                  f"pick-half {a:.4f}", flush=True)
            results.append({"criterion_rows": int(min(subsample, len(yfit))),
                            "rounds": r,
                            "n_distinct_tables": int((mult != 0).sum()),
                            "total_fan_out": int(np.abs(mult).sum()),
                            "pick_half_roc_auc": a})
        print("", flush=True)

    winner = max(results, key=lambda r: r["pick_half_roc_auc"])
    print(f"frozen stopping rule: criterion on {winner['criterion_rows']} "
          f"rows, {winner['rounds']} rounds, {winner['n_distinct_tables']} "
          f"tables, pick-half {winner['pick_half_roc_auc']:.4f}")
    print("  previous rule (24000 rows, halt on first non-improving round): "
          "0.7812 on the pick half, 0.7804 on the test fold")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_stopping.v1",
        "clinical_grade": False,
        "motivation": "the halt-on-first-non-improving rule stopped at the "
                      "sampling-noise floor of the subsampled criterion, and "
                      "kept fewer tables on more data",
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED, "n_fit_rows": int(len(yfit)),
                  "n_pick_rows": int(len(ypick))},
        "pool": {"table_width": 2, "pool_rounds": 12,
                 "n_pool_tables": len(tables),
                 "mean_residues_per_occupied_cell": occ},
        "candidates": results,
        "selected": winner,
        "previous_rule": {"criterion_rows": 24000,
                          "stop": "first non-improving round",
                          "pick_half_roc_auc": 0.7812,
                          "test_fold_roc_auc": 0.7804},
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
