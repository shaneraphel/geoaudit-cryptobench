#!/usr/bin/env python3
"""Push the pair-table covering design, and check what it needs to work.

The covering design settled the architecture question: 1032 two-wire tables of
sixteen cells each, every cell carrying ~7600 residues, fused by integer fan-out
and gated, reach 0.7844 on the pick half. That is above the 0.7809 a continuous
additive functional over the same wires attains, which is the point -- a pair
table is an exact empirical P(y | two digits), so the bank reads every monotone
recoding of a wire and every pairwise interaction, and an additive functional
over the raw digits reads neither.

Two things are still open and both matter for how the result can be claimed.

Round count. Going 4 -> 8 -> 12 rounds gave 0.7819 -> 0.7833 -> 0.7844, still
rising. Each round is another random pairing of all 172 wires, so more rounds
means more of the pair lattice covered. The exhaustive limit is all 14706 pairs,
which the fusion solve cannot take, but the curve should flatten well before it.

Provenance of the multiplicities. The reported bank chooses its integers by
rounding a closed-form direction -- one symmetric linear solve over the table
outputs on the fit half. That is not a gradient step and the inference path is
pure integer arithmetic, but it is a real-valued object computed at compile
time, and a reviewer is entitled to ask what the bank does without it. So the
same banks are also fused by multiplicities read straight off a counting
statistic:

  gini-rank   m_k = rank of the table's compiled Gini among the bank
  gini-ratio  m_k = round(cap * g_k / max g)

Neither solves anything. If these land near the quantized-direction number, the
architecture stands on counting alone and the linear solve is an optimisation
of the last decimal rather than the thing that makes it work.

Ranked on the pick half. The official test fold is not read here.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_refine.py
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
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_REFINE.json"


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

    def score_of(mult, A_pick):
        S = A_pick.astype(np.float64) @ mult.astype(np.float64)
        G = np.sum([_unit(gate(S, ctr_pick, n_pick_per, r))
                    for r in GATE_RADII], axis=0)
        return _unit(S) + _unit(G)

    results = []
    for width, rounds in ((2, 12), (2, 20), (2, 30), (3, 20), (2, 40)):
        tables = partitions(n_wires, width, rounds, gini_wire, "random")
        A_fit, A_pick, occ = compile_bank32(Dfit, yfit, Dpick, tables, rate)

        # bank Gini: a counting statistic of each compiled table
        g = np.array([abs(2.0 * pooled_auc(A_fit[:, k].astype(np.float64),
                                           yfit) - 1.0)
                      for k in range(A_fit.shape[1])])
        rank_m = np.empty(len(tables), dtype=np.int64)
        rank_m[np.argsort(g)] = np.arange(1, len(tables) + 1)
        ratio_m = np.maximum(np.round(32.0 * g / max(float(g.max()), 1e-12)),
                             0).astype(np.int64)

        w = direction_from_matrix(A_fit, yfit)
        row = {"width": width, "rounds": rounds, "n_tables": len(tables),
               "cells_per_table": 4 ** width,
               "mean_residues_per_occupied_cell": occ}
        for label, mult in (("gini-rank", rank_m),
                            ("gini-ratio cap=32", ratio_m),
                            ("quantized direction cap=32", quantize(w, 32))):
            row[label] = per_unit_auc(score_of(mult, A_pick), ypick,
                                      n_pick_per)
        print(f"  w{width} x{rounds:2d} ({len(tables):5d} tables, {occ:6.0f}"
              f"/cell)  gini-rank {row['gini-rank']:.4f}  "
              f"gini-ratio {row['gini-ratio cap=32']:.4f}  "
              f"solved {row['quantized direction cap=32']:.4f}", flush=True)
        results.append(row)
        del A_fit, A_pick

    best = max(results, key=lambda r: r["quantized direction cap=32"])
    best_pure = max(results, key=lambda r: max(r["gini-rank"],
                                               r["gini-ratio cap=32"]))
    print(f"\nbest solved-multiplicity bank: w{best['width']} x{best['rounds']}"
          f" = {best['quantized direction cap=32']:.4f}")
    print(f"best counting-only bank:       w{best_pure['width']} "
          f"x{best_pure['rounds']} = "
          f"{max(best_pure['gini-rank'], best_pure['gini-ratio cap=32']):.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_refine.v1",
        "clinical_grade": False,
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED, "n_wires": n_wires},
        "candidates": results,
        "selected_solved": best,
        "selected_counting_only": best_pure,
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
