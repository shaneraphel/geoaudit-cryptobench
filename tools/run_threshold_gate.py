"""Integer-multiplicity threshold fusion: the last purely combinational lever.

Unweighted fusion treats a table compiled from six near-blind invariants exactly
like a table compiled from the six sharpest ones, so a bank is only as good as
its average member. The classical combinational answer is a THRESHOLD gate: the
same sum, but each input replicated an integer number of times. Replication is
synthesizable (it is a fan-out, not a coefficient) and the multiplicity here is
read off a counting statistic already present in the artifact -- each table's own
compiled Gini ``g = |2A-1|`` on the training fold, the same statistic the
resolution field already stores per invariant.

Multiplicity is assigned by rank, not by value, so no real number ever enters the
datapath:

    m_k = 1 + rank of g_k among the K tables   (integers 1..K)

This is the widest fusion rule reachable without a fitted real weight. If it does
not clear the baseline, then no unweighted or integer-weighted partition scheme
over this descriptor set clears it, and the wall is the tabular estimator itself.

Usage: PYTHONPATH=src python3.12 tools/run_threshold_gate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pocket_bench.methods.cascade_lut import chain_rank_digits, patch_mean
from pocket_bench.metrics import average_precision, roc_auc

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/cryptobench_apo/_cascade_cache_train.npz"
TEST = ROOT / "data/cryptobench_apo/_cascade_cache_test.npz"
L = 4
RADII = (6.0, 8.0, 10.0, 14.0, 18.0)


def per_unit(score, y, n_res_per):
    aucs, prs = [], []
    off = 0
    for n in n_res_per:
        n = int(n)
        s, t = score[off:off + n], y[off:off + n]
        off += n
        if t.sum() == 0 or t.sum() == n:
            continue
        a = roc_auc(list(s), list(t))
        p = average_precision(list(s), list(t))
        if a is not None:
            aucs.append(a)
        if p is not None:
            prs.append(p)
    return float(np.mean(aucs)), float(np.mean(prs))


def pooled_auc(x, y):
    n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
    r = np.empty(len(x))
    r[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def norm(x):
    m = float(np.abs(x).max())
    return x / m if m > 0 else x


def main() -> int:
    ztr, zte = np.load(TRAIN, allow_pickle=False), np.load(TEST,
                                                           allow_pickle=False)
    Ftr, ytr, ntr = ztr["F"], ztr["y"], ztr["n_res_per"]
    Fte, yte, nte, ctr_te = zte["F"], zte["y"], zte["n_res_per"], zte["ctr"]
    rate = float(ytr.mean())
    M = Ftr.shape[1]
    Dtr = chain_rank_digits(Ftr, ntr)
    Dte = chain_rank_digits(Fte, nte)
    tables = [list(range(i, min(i + 6, M))) for i in range(0, M, 6)]

    yf = ytr.astype(np.float64)
    fr_te, gini = [], []
    for cols in tables:
        n_cells = L ** len(cols)
        a_tr = np.zeros(len(ytr), dtype=np.int64)
        a_te = np.zeros(len(yte), dtype=np.int64)
        for t, c in enumerate(cols):
            a_tr += Dtr[:, c] * (L ** t)
            a_te += Dte[:, c] * (L ** t)
        tot = np.bincount(a_tr, minlength=n_cells).astype(np.float64)
        pos = np.bincount(a_tr, weights=yf, minlength=n_cells)
        frac = np.where(tot > 0, pos / np.maximum(tot, 1.0), rate)
        fr_te.append(frac[a_te])
        gini.append(abs(2.0 * pooled_auc(frac[a_tr], ytr) - 1.0))

    gini = np.asarray(gini)
    order = np.argsort(gini)
    mult = np.empty(len(tables), dtype=np.int64)
    mult[order] = np.arange(1, len(tables) + 1)
    print("table gini (train) and integer multiplicity:")
    for k, cols in enumerate(tables):
        print(f"  table {k} wires={len(cols)}  gini={gini[k]:.4f}  m={mult[k]}")

    rows = []
    S_uni = np.sum(fr_te, axis=0)
    a, p = per_unit(S_uni, yte, nte)
    rows.append(("uniform (m=1)", a, p))
    print(f"\n  {'uniform (m=1)':30s} AUC={a:.4f}")

    S_thr = np.sum([m * f for m, f in zip(mult, fr_te)], axis=0)
    a, p = per_unit(S_thr, yte, nte)
    rows.append(("threshold (m=rank)", a, p))
    print(f"  {'threshold (m=rank)':30s} AUC={a:.4f}")

    for label, S in (("uniform", S_uni), ("threshold", S_thr)):
        G = np.sum([norm(patch_mean(S, ctr_te, nte, radius=r))
                    for r in RADII], axis=0)
        a, p = per_unit(norm(S) + norm(G), yte, nte)
        rows.append((f"{label} + multiscale gate", a, p))
        print(f"  {label + ' + multiscale gate':30s} AUC={a:.4f}")

    best = max(rows, key=lambda r: r[1])
    print(f"\nbest purely combinational: {best[0]}  AUC={best[1]:.4f}")
    print("reference: Fisher35 0.7830 (continuous) | P2Rank 0.7930")

    out = ROOT / "results/official_fold/THRESHOLD_GATE.json"
    out.write_text(json.dumps({
        "schema": "geoaudit.threshold_gate.v1",
        "clinical_grade": False,
        "multiplicity": "integer rank of each table's compiled train Gini; "
                        "fan-out replication, not a real coefficient",
        "table_gini": gini.tolist(),
        "multiplicity_values": mult.tolist(),
        "n_test_units": int(len(nte)),
        "results": [{"config": c, "roc_auc": a, "pr_auc": p}
                    for c, a, p in rows],
        "reference": {"fisher_35": 0.7830, "p2rank": 0.7930},
    }, indent=2))
    print(f"wrote -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
