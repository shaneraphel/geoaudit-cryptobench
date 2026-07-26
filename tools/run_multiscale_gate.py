"""Multi-scale spatial counting gates on the fused table bank.

A cryptic site is a patch, and a patch has no single size: the lining of a small
occluded cleft is coherent over 6 A while an interdomain groove is coherent over
18 A. One gate at one radius states the patch hypothesis at one scale only.
The multi-scale gate states it at every scale at once,

    G(i) = sum over radii r of  ( sum_{j : |c_i - c_j| <= r} S_j ) / |N_r(i)|,

which is again a sum of symmetric counting gates with all coefficients equal to
one. No radius is selected, no scale is weighted, nothing is fitted.

Two placements are compared:
  post   the gate acts on the fused bank score
  pre    the gate acts on every table fraction before fusion

Usage: PYTHONPATH=src python3.12 tools/run_multiscale_gate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pocket_bench.methods.algebraic_descriptors import FEATURE_NAMES
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


def table_fracs(Dtr, ytr, Dte, tables, rate):
    yf = ytr.astype(np.float64)
    out = []
    for cols in tables:
        n_cells = L ** len(cols)
        a_tr = np.zeros(len(ytr), dtype=np.int64)
        a_te = np.zeros(Dte.shape[0], dtype=np.int64)
        for t, c in enumerate(cols):
            a_tr += Dtr[:, c] * (L ** t)
            a_te += Dte[:, c] * (L ** t)
        tot = np.bincount(a_tr, minlength=n_cells).astype(np.float64)
        pos = np.bincount(a_tr, weights=yf, minlength=n_cells)
        frac = np.where(tot > 0, pos / np.maximum(tot, 1.0), rate)
        out.append(frac[a_te])
    return out


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
    print(f"bank of {len(tables)} thematic tables, base rate {rate:.4f}",
          flush=True)

    fr = table_fracs(Dtr, ytr, Dte, tables, rate)
    S = np.sum(fr, axis=0)

    rows = []
    a0, p0 = per_unit(S, yte, nte)
    rows.append(("no gate", a0, p0))
    print(f"  {'no gate':28s} AUC={a0:.4f}", flush=True)

    gates = {r: patch_mean(S, ctr_te, nte, radius=r) for r in RADII}
    for r in RADII:
        a, p = per_unit(norm(S) + norm(gates[r]), yte, nte)
        rows.append((f"post gate r={r:g}", a, p))
        print(f"  {'post gate r=' + format(r, 'g'):28s} AUC={a:.4f}", flush=True)

    G = np.sum([norm(gates[r]) for r in RADII], axis=0)
    a, p = per_unit(norm(S) + norm(G), yte, nte)
    rows.append(("post multiscale (5 radii)", a, p))
    print(f"  {'post multiscale (5 radii)':28s} AUC={a:.4f}", flush=True)

    a, p = per_unit(norm(G), yte, nte)
    rows.append(("gate only", a, p))
    print(f"  {'gate only':28s} AUC={a:.4f}", flush=True)

    pre = np.zeros(len(yte))
    for f in fr:
        pre += norm(f) + np.sum([norm(patch_mean(f, ctr_te, nte, radius=r))
                                 for r in RADII], axis=0)
    a, p = per_unit(pre, yte, nte)
    rows.append(("pre-fusion multiscale", a, p))
    print(f"  {'pre-fusion multiscale':28s} AUC={a:.4f}", flush=True)

    best = max(rows, key=lambda r: r[1])
    print(f"\nbest: {best[0]}  AUC={best[1]:.4f}")
    print("reference: bank4 0.7615 | Fisher35 0.7830 | P2Rank 0.7930")

    out = ROOT / "results/official_fold/MULTISCALE_GATE.json"
    out.write_text(json.dumps({
        "schema": "geoaudit.multiscale_gate.v1",
        "clinical_grade": False,
        "radii": list(RADII),
        "n_test_units": int(len(nte)),
        "results": [{"config": c, "roc_auc": a, "pr_auc": p}
                    for c, a, p in rows],
        "reference": {"bank4": 0.7615, "fisher_35": 0.7830, "p2rank": 0.7930},
    }, indent=2))
    print(f"wrote -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
