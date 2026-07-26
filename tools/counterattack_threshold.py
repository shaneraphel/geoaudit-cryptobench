#!/usr/bin/env python3
"""Two ways to spend the remaining gap, both evaluated on the training fold only.

The expanded wires support a continuous functional at ROC-AUC 0.781 on the
held-out training half while the best counting fusion reaches 0.761. Two
candidates for the difference, kept apart because they are different claims:

**A. Counting-only, wider tables.** The capacity bound allows d <= 6.87
quaternary digits, so a seven-digit table is still admissible (16384 cells,
about 14 fit-half residues each). A wider table models more of the joint
exactly. Nothing else changes: integer counters, integer multiplicities, the
spatial counting gate.

**B. Integer threshold gate.** A linear form over the quaternary digits, solved
once in closed form on the fit half and then QUANTIZED to integer weights in
[-W, W]. The deployed object is an integer weight vector; inference is an
integer multiply-accumulate against a threshold, which is a weighted majority
gate and is directly synthesizable. It performs no gradient step and no
iteration. It is NOT counting-only, because the weights come from a scatter
matrix rather than from a bincount, and it is reported under its own name so no
reader has to infer which claim applies to which number.

The comparison is on the pick half; the official test fold is not read.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_threshold.py
"""
from __future__ import annotations

import json

import numpy as np

from pocket_bench.paths import ROOT

from counterattack_fusion import build_tables, compile_bank
from counterattack_select import (
    GATE_RADII,
    SEED,
    _unit,
    gate,
    per_unit_auc,
    pooled_auc,
)

CACHE = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
DIGITS = ROOT / "data/cryptobench_apo/_expanded_digits_train.npy"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_THRESHOLD.json"


def closed_form_direction(A, yfit):
    """Fisher direction: one symmetric solve, no iteration, no step size."""
    A = A.astype(np.float64)
    a1, a0 = A[yfit == 1], A[yfit == 0]
    mu1, mu0 = a1.mean(0), a0.mean(0)
    S = ((a1 - mu1).T @ (a1 - mu1) + (a0 - mu0).T @ (a0 - mu0)) / (len(A) - 2)
    S += np.eye(S.shape[0]) * 1e-9 * float(np.trace(S)) / S.shape[0]
    w = np.linalg.solve(S, mu1 - mu0)
    n = float(np.linalg.norm(w))
    return w / n if n > 0 else w


def quantize(w, width):
    """Integer weights in [-width, width], scaled by the largest magnitude."""
    m = float(np.abs(w).max())
    if m <= 0:
        return np.zeros_like(w, dtype=np.int64)
    return np.clip(np.round(w / m * width), -width, width).astype(np.int64)


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

    D = np.load(DIGITS) if DIGITS.exists() else None
    if D is None:
        from counterattack_select import chain_digits
        D = chain_digits(X, n_res)
        np.save(DIGITS, D)
    Dfit, Dpick = D[fm], D[pm]
    print(f"digits {D.shape}; fit {int(is_fit.sum())} units, "
          f"pick {len(units) - int(is_fit.sum())} units", flush=True)

    g = np.array([abs(2.0 * pooled_auc(Dfit[:, j].astype(float), yfit) - 1.0)
                  for j in range(D.shape[1])])
    order = np.argsort(-g)

    results = []

    def record(name, score, claim, extra=None):
        a = per_unit_auc(score, ypick, n_pick_per)
        results.append({"architecture": name, "claim": claim,
                        "pick_half_roc_auc": a, **(extra or {})})
        print(f"  [{claim:14s}] {name:44s} {a:.4f}", flush=True)
        return a

    # ---- A. counting-only, seven-digit tables ------------------------------
    for k, width in ((90, 7), (120, 7), (172, 7)):
        tables = build_tables(order, k, width, width)
        fr_fit, fr_pick, gi = compile_bank(Dfit, yfit, Dpick, tables, rate)
        rank_m = np.empty(len(tables), dtype=np.int64)
        rank_m[np.argsort(gi)] = np.arange(1, len(tables) + 1)
        for label, m in (("rank^2", rank_m ** 2), ("rank^3", rank_m ** 3)):
            S = np.sum([int(w) * f for w, f in zip(m, fr_pick)], axis=0)
            G = np.sum([_unit(gate(S, ctr_pick, n_pick_per, r))
                        for r in GATE_RADII], axis=0)
            record(f"bank top-{k}, {width}-digit tables, {label} + gate",
                   _unit(S) + _unit(G), "counting-only",
                   {"n_tables": len(tables), "table_width": width})

    # ---- B. integer threshold gate over the digits -------------------------
    for k in (90, 172):
        cols = sorted(order[:k].tolist())
        w = closed_form_direction(Dfit[:, cols], yfit)
        for W in (8, 32, 128):
            wi = quantize(w, W)
            S = Dpick[:, cols].astype(np.int64) @ wi
            record(f"integer threshold gate, {k} wires, weights in [-{W},{W}]",
                   S.astype(np.float64), "integer-weight",
                   {"n_wires": k, "weight_range": W,
                    "n_nonzero_weights": int((wi != 0).sum())})
            G = np.sum([_unit(gate(S.astype(np.float64), ctr_pick,
                                   n_pick_per, r)) for r in GATE_RADII], axis=0)
            record(f"integer threshold gate, {k} wires, [-{W},{W}] + gate",
                   _unit(S.astype(np.float64)) + _unit(G), "integer-weight",
                   {"n_wires": k, "weight_range": W})

    by_claim = {}
    for claim in ("counting-only", "integer-weight"):
        cand = [r for r in results if r["claim"] == claim]
        if cand:
            by_claim[claim] = max(cand, key=lambda r: r["pick_half_roc_auc"])
            print(f"\nbest {claim}: {by_claim[claim]['architecture']} "
                  f"{by_claim[claim]['pick_half_roc_auc']:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_threshold.v1",
        "clinical_grade": False,
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED, "n_fit_units": int(is_fit.sum()),
                  "n_pick_units": len(units) - int(is_fit.sum())},
        "candidates": results,
        "best_per_claim": by_claim,
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
