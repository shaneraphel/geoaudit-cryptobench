#!/usr/bin/env python3
"""Trade table width for cell occupancy, and cover every wire several times.

The previous round located the loss precisely, and it was not where we assumed.
Solving for the best real-valued combination of the fifteen six-wire tables gave
0.7621 against 0.7619 for the integer fan-out: the cross-table weighting was
already spent. The loss is inside the tables.

A six-wire quaternary table has 4^6 = 4096 cells. The fit half holds roughly
1e5 residues at a 7% positive rate, so the average cell carries ~24 residues and
under two positives. The compiled fraction in such a cell is a ratio of small
integers and is mostly noise; the table models the joint distribution exactly
and estimates it badly. Widening the bank to all 172 wires made this worse
rather than better (0.752 against 0.762) because the extra tables were the
noisiest ones.

The fix is to move along the width/occupancy trade-off in the other direction.
A three-wire table has 64 cells and ~1500 residues per cell, so its fractions
are essentially exact; it captures less interaction, but the continuous ceiling
over these wires is a purely additive functional at 0.781, which says the
available signal is largely low-order and does not need six-way interactions to
be read. What a narrow table still gives beyond a linear term is a free monotone
recoding of each wire and the pairwise and three-way structure within its group.

Coverage replaces width. Each round is a partition of all 172 wires into groups
of w, so every wire appears exactly once per round and R times overall, each
time in different company. Two partition rules are tested:

  random             a seeded permutation, then consecutive groups
  strength-balanced  wires dealt round-robin in order of compiled Gini, so each
                     group holds one strong, one middling and one weak wire
                     instead of concentrating the strong ones in a few tables

Fusion is the same integer fan-out as before: a closed-form direction over the
table outputs, computed on the fit half, rounded onto a bounded integer grid.
Inference stays

    S(i) = sum_k m_k * pos_k[a_k(i)] / tot_k[a_k(i)],     m_k integer,

read a cell, replicate it m_k times, add, followed by the multi-scale spatial
counting gate. Ranked on the pick half; the official test fold is not read.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_covering.py
"""
from __future__ import annotations

import json

import numpy as np

from pocket_bench.paths import ROOT

from counterattack_quantized import closed_form_direction, quantize
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
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_COVERING.json"


def partitions(n_wires, width, rounds, gini, mode, seed=SEED):
    """R partitions of every wire index into groups of `width`."""
    rng = np.random.default_rng(seed)
    tables = []
    for r in range(rounds):
        if mode == "random":
            perm = rng.permutation(n_wires)
        else:
            # deal strongest-first round-robin into n_groups buckets
            n_groups = int(np.ceil(n_wires / width))
            by_strength = np.argsort(-gini)
            buckets = [[] for _ in range(n_groups)]
            offset = r  # rotate the deal each round so companies differ
            for i, j in enumerate(by_strength):
                buckets[(i + offset) % n_groups].append(int(j))
            perm = np.array([j for b in buckets for j in b])
        tables += [perm[i:i + width].tolist()
                   for i in range(0, n_wires, width)]
    return [t for t in tables if len(t) >= 2]


def compile_bank32(Dfit, yfit, Dpick, tables, rate):
    """As compile_bank, but returns float32 columns; the banks here are large."""
    yf = yfit.astype(np.float64)
    A_fit = np.empty((len(yfit), len(tables)), dtype=np.float32)
    A_pick = np.empty((Dpick.shape[0], len(tables)), dtype=np.float32)
    occupancy = []
    for k, cols in enumerate(tables):
        n_cells = L ** len(cols)
        a_fit = np.zeros(len(yfit), dtype=np.int64)
        a_pick = np.zeros(Dpick.shape[0], dtype=np.int64)
        for t, c in enumerate(cols):
            a_fit += Dfit[:, c] * (L ** t)
            a_pick += Dpick[:, c] * (L ** t)
        tot = np.bincount(a_fit, minlength=n_cells).astype(np.float64)
        pos = np.bincount(a_fit, weights=yf, minlength=n_cells)
        frac = np.where(tot > 0, pos / np.maximum(tot, 1.0), rate)
        A_fit[:, k] = frac[a_fit]
        A_pick[:, k] = frac[a_pick]
        occupancy.append(float(tot[tot > 0].mean()))
    return A_fit, A_pick, float(np.mean(occupancy))


def direction_from_matrix(A, yfit):
    a1, a0 = A[yfit == 1].astype(np.float64), A[yfit == 0].astype(np.float64)
    mu1, mu0 = a1.mean(0), a0.mean(0)
    S = ((a1 - mu1).T @ (a1 - mu1) + (a0 - mu0).T @ (a0 - mu0)) / (len(A) - 2)
    S += np.eye(S.shape[0]) * 1e-8 * max(float(np.trace(S)), 1e-12) / S.shape[0]
    w = np.linalg.solve(S, mu1 - mu0)
    n = float(np.linalg.norm(w))
    return w / n if n > 0 else w


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
    n_wires = D.shape[1]
    gini = np.array([abs(2.0 * pooled_auc(Dfit[:, j].astype(float), yfit) - 1.0)
                     for j in range(n_wires)])
    print(f"fit {int(is_fit.sum())} units / {len(yfit)} residues, "
          f"pick {len(units) - int(is_fit.sum())} units / {len(ypick)} residues, "
          f"{n_wires} wires\n", flush=True)

    results = []

    def score_of(mult, A_pick, integer=True):
        m = mult.astype(np.float64) if not integer else mult.astype(np.float64)
        S = A_pick.astype(np.float64) @ m
        G = np.sum([_unit(gate(S, ctr_pick, n_pick_per, r))
                    for r in GATE_RADII], axis=0)
        return _unit(S) + _unit(G)

    for mode in ("balanced", "random"):
        for width in (2, 3, 4):
            for rounds in (4, 8, 12):
                tables = partitions(n_wires, width, rounds, gini, mode)
                A_fit, A_pick, occ = compile_bank32(Dfit, yfit, Dpick,
                                                    tables, rate)
                w = direction_from_matrix(A_fit, yfit)
                tag = (f"{mode} w{width} x{rounds} "
                       f"({len(tables)} tables, {L**width} cells, "
                       f"{occ:.0f}/cell)")
                real = per_unit_auc(score_of(w, A_pick, integer=False),
                                    ypick, n_pick_per)
                best_cap, best_auc, best_m = None, -1.0, None
                for cap in (8, 16, 32):
                    m = quantize(w, cap)
                    a = per_unit_auc(score_of(m, A_pick), ypick, n_pick_per)
                    if a > best_auc:
                        best_cap, best_auc, best_m = cap, a, m
                print(f"  {tag:58s} int {best_auc:.4f} (cap {best_cap})  "
                      f"real {real:.4f}", flush=True)
                results.append({
                    "architecture": f"{mode} partition, width {width}, "
                                    f"{rounds} rounds, quantized cap={best_cap}"
                                    " + multi-scale gate",
                    "mode": mode, "width": width, "rounds": rounds,
                    "n_tables": len(tables), "cells_per_table": L ** width,
                    "mean_residues_per_occupied_cell": occ,
                    "cap": best_cap,
                    "pick_half_roc_auc": best_auc,
                    "pick_half_roc_auc_real_valued_fusion": real,
                    "cost_of_integer_grid": real - best_auc,
                    "n_nonzero_multiplicities": int((best_m != 0).sum()),
                })
                del A_fit, A_pick

    winner = max(results, key=lambda r: r["pick_half_roc_auc"])
    print(f"\nselected on the training fold alone: {winner['architecture']}")
    print(f"  pick-half ROC-AUC {winner['pick_half_roc_auc']:.4f}  "
          f"({winner['n_tables']} tables of {winner['cells_per_table']} cells, "
          f"{winner['mean_residues_per_occupied_cell']:.0f} residues per "
          f"occupied cell)")
    print("  for reference on this same split: six-wire bank 0.7647, "
          "35-wire incumbent 0.7446, continuous ceiling over all wires 0.7809")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_covering.v1",
        "clinical_grade": False,
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED, "n_fit_units": int(is_fit.sum()),
                  "n_pick_units": len(units) - int(is_fit.sum()),
                  "n_wires": n_wires},
        "reference_same_split": {
            "six_wire_bank_quantized": 0.7647,
            "thirty_five_wire_incumbent": 0.7446,
            "continuous_ceiling_all_wires": 0.7809,
        },
        "candidates": results,
        "selected": winner,
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
