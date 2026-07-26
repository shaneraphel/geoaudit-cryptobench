#!/usr/bin/env python3
"""Select the expanded-wire architecture on the training fold alone.

The 35-invariant field trails P2Rank by 0.028 ROC-AUC, and the ceiling analysis
showed the shortfall is in the inputs rather than the fusion: a closed-form
linear functional of those 35 invariants also fails to reach P2Rank. The
expanded wire set adds the two axes the invariants do not span -- published
per-residue chemistry, and a multi-radius context transform -- and this script
decides what to do with them WITHOUT reading the official test fold.

Discipline. The training fold is split into halves with disjoint MMseqs2
clusters. Tables, propensity, Gini multiplicities and wire rankings are all
compiled on the fit half; every candidate is ranked on the pick half. A single
architecture comes out, and only that one is ever evaluated on the test fold.
The Fisher figures printed here are continuous discriminants and are reported
only as a ceiling: they say how much signal the wires carry, not what the
combinational path achieves.

Usage: PYTHONPATH=src python3.12 tools/counterattack_select.py
"""
from __future__ import annotations

import json

import numpy as np

from pocket_bench.metrics import roc_auc
from pocket_bench.paths import ROOT

CACHE = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_SELECTION.json"
L = 4
GATE_RADII = (6.0, 8.0, 10.0, 14.0, 18.0)
SEED = 20260725


def chain_digits(F, n_res_per, levels=L):
    out = np.empty(F.shape, dtype=np.int64)
    off = 0
    for n in n_res_per:
        n = int(n)
        blk = F[off:off + n]
        for j in range(F.shape[1]):
            x = blk[:, j]
            order = np.argsort(x, kind="stable")
            r = np.empty(n)
            i = 0
            while i < n:
                k = i
                while k + 1 < n and x[order[k + 1]] == x[order[i]]:
                    k += 1
                r[order[i:k + 1]] = 0.5 * (i + k)
                i = k + 1
            out[off:off + n, j] = np.clip(
                np.floor(r / max(n - 1, 1) * levels).astype(np.int64),
                0, levels - 1)
        off += n
    return out


def per_unit_auc(score, y, n_res_per):
    aucs = []
    off = 0
    for n in n_res_per:
        n = int(n)
        s, t = score[off:off + n], y[off:off + n]
        off += n
        if t.sum() == 0 or t.sum() == n:
            continue
        a = roc_auc(list(s), list(t))
        if a is not None:
            aucs.append(a)
    return float(np.mean(aucs)) if aucs else float("nan")


def gate(s, ctr, n_res_per, radius):
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


def _unit(x):
    m = float(np.abs(x).max())
    return x / m if m > 0 else x


def pooled_auc(x, y):
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def fisher(Dfit, yfit, Dpick):
    """Closed-form linear direction over the digits. Diagnostic only."""
    A = Dfit.astype(np.float64)
    B = Dpick.astype(np.float64)
    a1, a0 = A[yfit == 1], A[yfit == 0]
    mu1, mu0 = a1.mean(0), a0.mean(0)
    S = ((a1 - mu1).T @ (a1 - mu1) + (a0 - mu0).T @ (a0 - mu0)) / (len(A) - 2)
    S += np.eye(S.shape[0]) * 1e-9 * float(np.trace(S)) / S.shape[0]
    w = np.linalg.solve(S, mu1 - mu0)
    return B @ w


def bank(Dfit, yfit, Dpick, tables, rate):
    yf = yfit.astype(np.float64)
    fr, gini = [], []
    for cols in tables:
        n_cells = L ** len(cols)
        a_fit = np.zeros(len(yfit), dtype=np.int64)
        a_pick = np.zeros(Dpick.shape[0], dtype=np.int64)
        for t, c in enumerate(cols):
            a_fit += Dfit[:, c] * (L ** t)
            a_pick += Dpick[:, c] * (L ** t)
        tot = np.bincount(a_fit, minlength=n_cells).astype(np.float64)
        pos = np.bincount(a_fit, weights=yf, minlength=n_cells)
        frac = np.where(tot > 0, pos / np.maximum(tot, 1.0), rate)
        fr.append(frac[a_pick])
        gini.append(abs(2.0 * pooled_auc(frac[a_fit], yfit) - 1.0))
    return fr, np.asarray(gini)


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
    ctr_pick = ctr[pm]
    rate = float(yfit.mean())
    print(f"{len(units)} train units / {len(clusters)} clusters -> "
          f"fit {int(is_fit.sum())}, pick {len(units) - int(is_fit.sum())}; "
          f"{X.shape[1]} wires", flush=True)

    D = chain_digits(X, n_res)
    Dfit, Dpick = D[fm], D[pm]

    # wire ranking, compiled on the fit half only
    g = np.array([abs(2.0 * pooled_auc(Dfit[:, j].astype(float), yfit) - 1.0)
                  for j in range(D.shape[1])])
    order = np.argsort(-g)
    print("strongest wires:",
          [names[j] for j in order[:10]], flush=True)

    results = []

    def record(name, score, extra=None):
        a = per_unit_auc(score, ypick, n_pick_per)
        results.append({"architecture": name, "pick_half_roc_auc": a,
                        **(extra or {})})
        print(f"  {name:52s} {a:.4f}", flush=True)
        return a

    # ceilings: what a continuous functional of these wires could reach
    for k in (35, 43, 172):
        cols = list(range(k)) if k <= 43 else list(range(X.shape[1]))
        record(f"[ceiling] Fisher over {len(cols)} wires",
               fisher(Dfit[:, cols], yfit, Dpick[:, cols]))

    # combinational candidates
    for k in (36, 60, 90, 120, X.shape[1]):
        cols = sorted(order[:k].tolist())
        tables = [cols[i:i + 6] for i in range(0, len(cols), 6)]
        tables = [t for t in tables if len(t) >= 2]
        fr, gi = bank(Dfit, yfit, Dpick, tables, rate)
        S_uni = np.sum(fr, axis=0)
        record(f"bank top-{k} wires, uniform fusion", S_uni)

        o = np.argsort(gi)
        mult = np.empty(len(tables), dtype=np.int64)
        mult[o] = np.arange(1, len(tables) + 1)
        S_thr = np.sum([m * f for m, f in zip(mult, fr)], axis=0)
        record(f"bank top-{k} wires, integer-multiplicity fusion", S_thr)

        G = np.sum([_unit(gate(S_thr, ctr_pick, n_pick_per, r))
                    for r in GATE_RADII], axis=0)
        record(f"bank top-{k} wires, integer-multiplicity + multi-scale gate",
               _unit(S_thr) + _unit(G),
               {"n_tables": len(tables), "n_wires": len(cols)})

    combinational = [r for r in results
                     if not r["architecture"].startswith("[ceiling]")]
    winner = max(combinational, key=lambda r: r["pick_half_roc_auc"])
    ceiling = max(r["pick_half_roc_auc"] for r in results
                  if r["architecture"].startswith("[ceiling]"))
    print(f"\nselected (training fold only): {winner['architecture']}")
    print(f"  pick-half ROC-AUC {winner['pick_half_roc_auc']:.4f}   "
          f"(continuous ceiling on the same split {ceiling:.4f})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_selection.v1",
        "clinical_grade": False,
        "wires": X.shape[1],
        "split": {"criterion": "cluster_id, seeded shuffle, disjoint halves",
                  "seed": SEED, "n_clusters": len(clusters),
                  "n_fit_units": int(is_fit.sum()),
                  "n_pick_units": len(units) - int(is_fit.sum())},
        "strongest_wires": [names[j] for j in order[:20]],
        "candidates": results,
        "selected": winner,
        "continuous_ceiling_same_split": ceiling,
    }, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
