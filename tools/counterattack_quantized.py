#!/usr/bin/env python3
"""Spend the last of the cross-table gap with integer fan-out.

Where the remaining loss is, exactly. On the training split the expanded wires
support a continuous functional at ROC-AUC 0.781; the table bank with rank
multiplicities reaches 0.755 and with squared ranks 0.761. Inside a six-wire
table the joint distribution is already modelled exactly, so the residual 0.020
is entirely in the multiplicities: fifteen numbers saying how much to believe
each table. Rank and rank-squared are guesses at those fifteen numbers.

They can be computed instead. The multiplicity vector that maximises the
separation of an additive score is the solution of one symmetric linear system
in fifteen unknowns -- a closed-form expression in the compiled table outputs,
not an optimisation and not a gradient step -- and rounding it onto a bounded
integer grid gives a fan-out. The inference path is unchanged and remains

    S(i) = sum_k m_k * pos_k[a_k(i)] / tot_k[a_k(i)],     m_k integer,

i.e. read a cell, replicate it m_k times, add. What changed is only how the
integers were chosen at compile time, and they are chosen on the fit half.

We report the rounding cost explicitly: the same direction before and after
quantization, so the reader can see what the integer grid gave up.

Also tested here is a residual second stage. The fused score is banded into a
quaternary digit and joined with five strong wires into further tables, so the
second stage can express "this wire matters differently where the aggregate is
already high" -- an interaction between the aggregate and the individual wires,
which a single additive layer cannot state.

Ranked on the pick half. The official test fold is not read.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_quantized.py
"""
from __future__ import annotations

import json

import numpy as np

from pocket_bench.paths import ROOT

from counterattack_fusion import build_tables, compile_bank
from counterattack_select import (
    GATE_RADII,
    L,
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
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_QUANTIZED.json"


def closed_form_direction(fr_fit, yfit):
    """Fisher direction over the table outputs: one 15x15 symmetric solve."""
    A = np.stack(fr_fit, axis=1)
    a1, a0 = A[yfit == 1], A[yfit == 0]
    mu1, mu0 = a1.mean(0), a0.mean(0)
    S = ((a1 - mu1).T @ (a1 - mu1) + (a0 - mu0).T @ (a0 - mu0)) / (len(A) - 2)
    S += np.eye(S.shape[0]) * 1e-9 * max(float(np.trace(S)), 1e-12) / S.shape[0]
    w = np.linalg.solve(S, mu1 - mu0)
    n = float(np.linalg.norm(w))
    return w / n if n > 0 else w


def quantize(w, cap):
    """Round a direction onto the integer grid [-cap, cap]."""
    m = float(np.abs(w).max())
    if m <= 0:
        return np.zeros(len(w), dtype=np.int64)
    return np.round(w / m * cap).astype(np.int64)


def band4(x):
    q = np.quantile(x, [0.25, 0.5, 0.75])
    for t in range(1, 3):
        if q[t] <= q[t - 1]:
            q[t] = np.nextafter(q[t - 1], np.inf)
    return np.clip(np.searchsorted(q, x, side="right"), 0, L - 1), q


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

    D = np.load(DIGITS) if DIGITS.exists() else chain_digits(X, n_res)
    Dfit, Dpick = D[fm], D[pm]
    g = np.array([abs(2.0 * pooled_auc(Dfit[:, j].astype(float), yfit) - 1.0)
                  for j in range(D.shape[1])])
    order = np.argsort(-g)

    results = []

    def record(name, score, extra=None):
        a = per_unit_auc(score, ypick, n_pick_per)
        results.append({"architecture": name, "pick_half_roc_auc": a,
                        **(extra or {})})
        print(f"  {name:60s} {a:.4f}", flush=True)
        return a

    for k, stride in ((90, 6), (120, 6), (172, 6)):
        tables = build_tables(order, k, 6, stride)
        fr_fit, fr_pick, gi = compile_bank(Dfit, yfit, Dpick, tables, rate)
        tag = f"top-{k} ({len(tables)} tables)"

        rank_m = np.empty(len(tables), dtype=np.int64)
        rank_m[np.argsort(gi)] = np.arange(1, len(tables) + 1)
        record(f"{tag}, rank^2 multiplicity + gate",
               _gate_sum(rank_m ** 2, fr_pick, ctr_pick, n_pick_per))
        record(f"{tag}, rank^3 multiplicity + gate",
               _gate_sum(rank_m ** 3, fr_pick, ctr_pick, n_pick_per))

        w = closed_form_direction(fr_fit, yfit)
        # the unquantized direction, as the ceiling of this fusion family
        record(f"{tag}, [ceiling] real-valued direction + gate",
               _gate_sum(w, fr_pick, ctr_pick, n_pick_per, integer=False))
        for cap in (4, 8, 16, 32, 64):
            m = quantize(w, cap)
            if not np.any(m):
                continue
            record(f"{tag}, quantized direction cap={cap} + gate",
                   _gate_sum(m, fr_pick, ctr_pick, n_pick_per),
                   {"n_tables": len(tables), "n_wires": k, "cap": cap,
                    "multiplicity": m.tolist(),
                    "n_nonzero": int((m != 0).sum())})

        # residual second stage over the best integer fusion so far
        m16 = quantize(w, 16)
        S_fit = np.sum([int(v) * f for v, f in zip(m16, fr_fit)], axis=0)
        S_pick = np.sum([int(v) * f for v, f in zip(m16, fr_pick)], axis=0)
        d_fit, edges = band4(S_fit)
        d_pick = np.clip(np.searchsorted(edges, S_pick, side="right"), 0, L - 1)
        Efit = np.concatenate([Dfit, d_fit[:, None]], axis=1)
        Epick = np.concatenate([Dpick, d_pick[:, None]], axis=1)
        agg = Efit.shape[1] - 1
        stage2 = [[agg] + order[i:i + 5].tolist() for i in range(0, 30, 5)]
        fr2_fit, fr2_pick, _ = compile_bank(Efit, yfit, Epick, stage2, rate)
        w2 = closed_form_direction(fr_fit + fr2_fit, yfit)
        m2 = quantize(w2, 16)
        record(f"{tag}, + residual stage-2 tables, quantized cap=16 + gate",
               _gate_sum(m2, fr_pick + fr2_pick, ctr_pick, n_pick_per),
               {"n_tables": len(tables) + len(stage2), "n_wires": k,
                "stage2_tables": len(stage2)})

    combinational = [r for r in results if "[ceiling]" not in r["architecture"]]
    winner = max(combinational, key=lambda r: r["pick_half_roc_auc"])
    ceiling = max(r["pick_half_roc_auc"] for r in results
                  if "[ceiling]" in r["architecture"])
    print(f"\nselected on the training fold alone: {winner['architecture']}")
    print(f"  pick-half ROC-AUC {winner['pick_half_roc_auc']:.4f}   "
          f"(real-valued fusion of the same tables {ceiling:.4f}; "
          f"cost of the integer grid {ceiling - winner['pick_half_roc_auc']:+.4f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_quantized.v1",
        "clinical_grade": False,
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED, "n_fit_units": int(is_fit.sum()),
                  "n_pick_units": len(units) - int(is_fit.sum())},
        "wire_order_top20": [names[j] for j in order[:20]],
        "candidates": results,
        "selected": winner,
        "real_valued_fusion_ceiling": ceiling,
        "cost_of_integer_grid": ceiling - winner["pick_half_roc_auc"],
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


def _gate_sum(mult, fr, ctr_pick, n_pick_per, integer=True):
    if integer:
        S = np.sum([int(v) * f for v, f in zip(mult, fr)], axis=0)
    else:
        S = np.sum([float(v) * f for v, f in zip(mult, fr)], axis=0)
    G = np.sum([_unit(gate(S, ctr_pick, n_pick_per, r)) for r in GATE_RADII],
               axis=0)
    return _unit(S) + _unit(G)


if __name__ == "__main__":
    raise SystemExit(main())
