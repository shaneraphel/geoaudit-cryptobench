#!/usr/bin/env python3
"""Choose the readout that turns the compiled tables into a residue score.

The table bank is not the bottleneck. On the training-fold split the expanded
wires support a continuous functional at ROC-AUC 0.781 while the best integer
fusion of the same tables reaches 0.761. What is lost is not information in the
tables, it is the resolution of the rule that combines them: fifteen tables of
unequal quality summed with integer multiplicities.

This script asks how much of that 0.020 is recoverable, and at what cost in
model class. It compares readouts of increasing expressiveness over exactly the
same compiled objects:

  count_square      integer multiplicity (rank^2) sum of table fractions
  lin_digits        closed-form linear direction over the 172 quaternary digits
  lin_onehot        closed-form linear direction over the digit indicators
  lin_tables        closed-form linear direction over the 15 table fractions
  lin_tables_gated  the same, plus the five spatial averages of each fraction

Every linear readout is a single regularised solve of one symmetric system. No
gradient step, no iteration, no auto-differentiation, no held-out tuning loop.
That is a weaker claim than "training-free" and the paper must say so: like the
tables themselves, the coefficients are fitted on the training fold.

Both halves come from the training fold and their clusters are disjoint. The
official test fold is not read here.

Usage: PYTHONPATH=src:tools python3.12 tools/final_readout_select.py
"""
from __future__ import annotations

import json

import numpy as np

from pocket_bench.paths import ROOT

from counterattack_select import (  # noqa: E402  (sibling tool, same directory)
    GATE_RADII,
    L,
    SEED,
    bank,
    chain_digits,
    per_unit_auc,
    pooled_auc,
)

CACHE = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/FINAL_READOUT_SELECTION.json"
TOP_K = 90
WIDTH = 6
STRIDE = 6
RIDGE = 1e-6


def windows(order, k, width=WIDTH, stride=STRIDE):
    sel = sorted(int(j) for j in order[:k])
    return [tuple(sel[s:s + width]) for s in range(0, k - width + 1, stride)]


def gate_matrix(S, ctr, n_res_per, radius):
    """Neighbourhood average of every column, one chain at a time."""
    S = np.atleast_2d(S.T).T if S.ndim == 2 else S[:, None]
    out = np.empty_like(S, dtype=np.float64)
    r2 = radius * radius
    off = 0
    for n in n_res_per:
        n = int(n)
        c, v = ctr[off:off + n], S[off:off + n]
        for i in range(0, n, 512):
            d2 = ((c[i:i + 512, None, :] - c[None, :, :]) ** 2).sum(-1)
            a = (d2 <= r2).astype(np.float64)
            out[off:off + n][i:i + 512] = (a @ v) / np.maximum(
                a.sum(1), 1.0)[:, None]
        off += n
    return out


def solve(Afit, yfit, Apick, ridge=RIDGE):
    """Regularised least-squares direction; one symmetric solve, no iteration."""
    A = np.asarray(Afit, dtype=np.float64)
    mu, sd = A.mean(0), A.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    A = (A - mu) / sd
    B = (np.asarray(Apick, dtype=np.float64) - mu) / sd
    t = yfit.astype(np.float64)
    t = t - t.mean()
    G = A.T @ A
    G.flat[:: G.shape[0] + 1] += ridge * float(np.trace(G)) / G.shape[0] + 1e-9
    return B @ np.linalg.solve(G, A.T @ t)


def onehot(D, levels=L):
    n, m = D.shape
    out = np.zeros((n, m * levels), dtype=np.float64)
    out[np.arange(n)[:, None], D + np.arange(m)[None, :] * levels] = 1.0
    return out


def main() -> int:
    z = np.load(CACHE, allow_pickle=False)
    X, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    names = [str(s) for s in z["names"]]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}

    clusters = sorted({cluster_of[u] for u in units})
    rng = np.random.default_rng(SEED)
    rng.shuffle(clusters)
    fit_clusters = set(clusters[:len(clusters) // 2])
    is_fit = np.array([cluster_of[u] in fit_clusters for u in units])
    row_unit = np.repeat(np.arange(len(units)), n_res)
    fm, pm = is_fit[row_unit], ~is_fit[row_unit]
    n_pick_per = np.array([n for n, f in zip(n_res, is_fit) if not f])
    yfit, ypick = y[fm], y[pm]
    ctr_pick, ctr_fit = ctr[pm], ctr[fm]
    n_fit_per = np.array([n for n, f in zip(n_res, is_fit) if f])
    rate = float(yfit.mean())
    print(f"{len(units)} train units / {len(clusters)} clusters -> "
          f"fit {int(is_fit.sum())}, pick {len(units) - int(is_fit.sum())}; "
          f"{X.shape[1]} wires", flush=True)

    D = chain_digits(X, n_res)
    Dfit, Dpick = D[fm], D[pm]
    g = np.array([abs(2.0 * pooled_auc(Dfit[:, j].astype(float), yfit) - 1.0)
                  for j in range(D.shape[1])])
    order = np.argsort(-g)
    tables = windows(order, TOP_K)
    print(f"top-{TOP_K} wires -> {len(tables)} tables of width {WIDTH}",
          flush=True)

    fr_pick, gini = bank(Dfit, yfit, Dpick, tables, rate)
    fr_fit, _ = bank(Dfit, yfit, Dfit, tables, rate)
    Fpick = np.stack(fr_pick, 1)
    Ffit = np.stack(fr_fit, 1)
    mult = (np.argsort(np.argsort(gini)) + 1).astype(np.float64) ** 2

    print("gating table fractions at five radii", flush=True)
    Gpick = [gate_matrix(Fpick, ctr_pick, n_pick_per, r) for r in GATE_RADII]
    Gfit = [gate_matrix(Ffit, ctr_fit, n_fit_per, r) for r in GATE_RADII]

    cands = {
        "count_square": (None, None, Fpick @ mult),
        "lin_digits": (Dfit.astype(np.float64), Dpick.astype(np.float64), None),
        "lin_onehot": (onehot(Dfit), onehot(Dpick), None),
        "lin_tables": (Ffit, Fpick, None),
        "lin_tables_gated": (np.hstack([Ffit] + Gfit),
                             np.hstack([Fpick] + Gpick), None),
    }

    results = []
    for name, (Afit, Apick, direct) in cands.items():
        s = direct if direct is not None else solve(Afit, yfit, Apick)
        raw = per_unit_auc(s, ypick, n_pick_per)
        best = (raw, "none")
        for r in GATE_RADII:
            gs = gate_matrix(s, ctr_pick, n_pick_per, r)[:, 0]
            for w in (0.25, 0.5, 1.0):
                a = per_unit_auc(s + w * gs * (np.std(s) / max(np.std(gs), 1e-12)),
                                 ypick, n_pick_per)
                if a > best[0]:
                    best = (a, f"r={r:g},w={w:g}")
        n_dim = 0 if Afit is None else int(Afit.shape[1])
        results.append({"readout": name, "n_parameters": n_dim,
                        "pick_half_roc_auc_raw": round(float(raw), 4),
                        "pick_half_roc_auc": round(float(best[0]), 4),
                        "gate": best[1]})
        print(f"  {name:20s} dim={n_dim:4d} raw={raw:.4f} "
              f"gated={best[0]:.4f} ({best[1]})", flush=True)

    results.sort(key=lambda r: -r["pick_half_roc_auc"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "final_readout_selection/v1",
        "split": "training fold, cluster-disjoint fit/pick halves",
        "seed": SEED,
        "top_k_wires": TOP_K, "table_width": WIDTH, "table_stride": STRIDE,
        "n_tables": len(tables), "ridge": RIDGE,
        "strongest_wires": [names[j] for j in order[:12]],
        "candidates": results,
        "selected": results[0],
        "note": ("Linear readouts are one regularised symmetric solve fitted on "
                 "the fit half; they are not training-free and the manuscript "
                 "reports them as fitted on the official training fold."),
    }, indent=2) + "\n")
    print(f"\nselected: {results[0]['readout']} "
          f"{results[0]['pick_half_roc_auc']:.4f}\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
