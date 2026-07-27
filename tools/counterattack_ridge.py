#!/usr/bin/env python3
"""Fine integer fan-out over pair tables, made stable by regularising the solve.

Where the counterattack stands. The pair-table bank is the right basis: 1032
two-wire tables of sixteen cells, ~7600 residues per cell, exact empirical
P(y | two digits) in every cell. What has not worked is turning 1032 table
outputs into one score.

  stagewise, additive only     0.7687   cannot cancel a redundant table
  stagewise, signed            0.7812   stable, but only ~80 units of fan-out
                                        spread over ~70 tables, so each table's
                                        weight is 1, 2 or 3 and nothing finer
  rounded closed-form solve    0.7844 at 1032 tables, 0.6846 at 1720
  the same wires, read by a fitted linear functional of the raw digits
                               0.7873

The solve gives the fan-out the resolution the stagewise search cannot (cap 32
instead of 3) and that is worth 0.003; it then collapses when the pool grows,
because 1720 random pairs drawn from a lattice of 14706 contain near-duplicate
columns, the scatter matrix goes near-singular, and the direction puts large
opposing coefficients on columns that are copies of each other.

Near-singularity from duplicated columns is what regularisation is for, and this
repository already relies on it: the fitted linear readout solves a 172x172
system with a ridge of 1e-6 of the trace. Adding the same term here should keep
the fine fan-out and remove the collapse, and the pair-table basis is strictly
richer than the raw digits -- it contains every monotone recoding of a wire and
every within-pair interaction -- so a well-conditioned solve over it should not
sit below the 0.7873 a linear functional of the digits reaches.

Also adopted here is that readout's spatial gate, which is not a new choice: one
radius, the neighbourhood mean rescaled to the raw score's standard deviation
before being added at weight 0.5, both values frozen earlier on this same
training split (results/architecture_sweep/FINAL_READOUT_SELECTION.json).
Matching the spread rather than the maximum matters because the maximum of a
score field over a chain is an order statistic of a few residues.

The scatter matrix is accumulated in chunks. Materialising it the direct way
needs a double-precision copy of the whole bank and is what killed two earlier
runs at 1720 tables.

Ranked on the pick half. The official test fold is not read here.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_ridge.py
"""
from __future__ import annotations

import gc
import json

import numpy as np

from pocket_bench.paths import ROOT

from counterattack_covering import compile_bank32, partitions
from counterattack_select import (
    SEED,
    chain_digits,
    per_unit_auc,
    pooled_auc,
)

CACHE = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
DIGITS = ROOT / "data/cryptobench_apo/_expanded_digits_train.npy"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_RIDGE.json"
CHUNK = 8192


def class_scatter(A, y):
    """Within-class scatter and class means, accumulated in float64 chunks."""
    n, K = A.shape
    pos = y == 1
    s1 = np.zeros(K); s0 = np.zeros(K)
    n1 = int(pos.sum()); n0 = n - n1
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        blk = A[a:b].astype(np.float64)
        p = pos[a:b]
        s1 += blk[p].sum(0)
        s0 += blk[~p].sum(0)
    mu1, mu0 = s1 / max(n1, 1), s0 / max(n0, 1)
    S = np.zeros((K, K))
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        blk = A[a:b].astype(np.float64)
        p = pos[a:b]
        c = np.where(p[:, None], blk - mu1, blk - mu0)
        S += c.T @ c
    return S / max(n - 2, 1), mu1, mu0


def ridge_direction(A, y, lam):
    S, mu1, mu0 = class_scatter(A, y)
    K = S.shape[0]
    S.flat[::K + 1] += lam * float(np.trace(S)) / K + 1e-12
    w = np.linalg.solve(S, mu1 - mu0)
    nrm = float(np.linalg.norm(w))
    return w / nrm if nrm > 0 else w


def quantize(w, cap):
    m = float(np.abs(w).max())
    if m <= 0:
        return np.zeros(len(w), dtype=np.int64)
    return np.round(w / m * cap).astype(np.int64)


def neighbourhood_mean(s, ctr, n_res_per, radius):
    out = np.empty(len(s))
    r2 = radius * radius
    off = 0
    for n in n_res_per:
        n = int(n)
        c, v = ctr[off:off + n], s[off:off + n]
        acc = np.empty(n)
        for i in range(0, n, 512):
            d2 = ((c[i:i + 512, None, :] - c[None, :, :]) ** 2).sum(-1)
            a = (d2 <= r2).astype(np.float64)
            acc[i:i + 512] = (a @ v) / np.maximum(a.sum(1), 1.0)
        out[off:off + n] = acc
        off += n
    return out


def spread_matched_gate(s, ctr, n_res_per, radius, weight):
    """Add back the neighbourhood mean, rescaled to the raw score's spread."""
    g = neighbourhood_mean(s, ctr, n_res_per, radius)
    sd_s, sd_g = float(np.std(s)), float(np.std(g))
    return s if sd_g <= 0 else s + weight * g * (sd_s / sd_g)


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

    results = []
    for pool_rounds in (12, 20):
        tables = partitions(n_wires, 2, pool_rounds, gini_wire, "random")
        A_fit, A_pick, occ = compile_bank32(Dfit, yfit, Dpick, tables, rate)
        print(f"pool {len(tables)} pair tables, {occ:.0f} residues per "
              f"occupied cell", flush=True)
        for lam in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
            w = ridge_direction(A_fit, yfit, lam)
            for cap in (16, 32, 64):
                m = quantize(w, cap)
                if not np.any(m):
                    continue
                S = A_pick.astype(np.float64) @ m.astype(np.float64)
                raw = per_unit_auc(S, ypick, n_pick_per)
                best = None
                for radius, weight in ((14.0, 1.0), (18.0, 0.5), (18.0, 1.0)):
                    a = per_unit_auc(
                        spread_matched_gate(S, ctr_pick, n_pick_per,
                                            radius, weight),
                        ypick, n_pick_per)
                    if best is None or a > best[0]:
                        best = (a, radius, weight)
                results.append({
                    "n_pool_tables": len(tables), "pool_rounds": pool_rounds,
                    "ridge": lam, "cap": cap,
                    "n_nonzero": int((m != 0).sum()),
                    "total_fan_out": int(np.abs(m).sum()),
                    "pick_half_roc_auc_raw": raw,
                    "pick_half_roc_auc": best[0],
                    "gate_radius": best[1], "gate_weight": best[2],
                })
                print(f"  ridge {lam:<7g} cap {cap:3d}  raw {raw:.4f}  "
                      f"gated {best[0]:.4f} (r={best[1]:.0f}, w={best[2]})  "
                      f"fan-out {int(np.abs(m).sum())}", flush=True)
        del A_fit, A_pick
        gc.collect()

    winner = max(results, key=lambda r: r["pick_half_roc_auc"])
    at12 = max(r["pick_half_roc_auc"] for r in results
               if r["pool_rounds"] == 12)
    at20 = max(r["pick_half_roc_auc"] for r in results
               if r["pool_rounds"] == 20)
    print(f"\nbest: {winner['n_pool_tables']} tables, ridge "
          f"{winner['ridge']:g}, cap {winner['cap']}, gate r="
          f"{winner['gate_radius']:.0f} w={winner['gate_weight']} -> "
          f"{winner['pick_half_roc_auc']:.4f}")
    print(f"pool-size stability of the ridge solve: {at12:.4f} at 1032 "
          f"tables, {at20:.4f} at 1720 (unregularised: 0.7844 and 0.6846)")
    print("reference on this split: signed stagewise 0.7812, fitted linear "
          "functional of the raw digits 0.7873")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_ridge.v1",
        "clinical_grade": False,
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED, "n_wires": n_wires},
        "candidates": results,
        "selected": winner,
        "pool_size_stability": {"1032_tables": at12, "1720_tables": at20,
                                "unregularised_1032": 0.7844,
                                "unregularised_1720": 0.6846},
        "reference_same_split": {"signed_stagewise": 0.7812,
                                 "fitted_linear_over_digits": 0.7873},
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
