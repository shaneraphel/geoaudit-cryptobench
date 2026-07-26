"""Localize the wall: is it the descriptors or the combinational architecture?

Compares, on identical cached features and the identical 192-unit official test
fold:

  (a) every single invariant alone,
  (b) the closed-form Fisher direction over all 35 chain-normalized ranks,
  (c) the same over the original six,

against the flat quaternary table and P2Rank. This is a DIAGNOSTIC, not a
deliverable: (b) and (c) are continuous discriminants and are therefore excluded
from the combinational claim. Their only purpose is to separate "the algebraic
descriptors do not carry the signal" from "the quantizer throws the signal away".

Usage: PYTHONPATH=src python3.12 tools/diagnose_feature_ceiling.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pocket_bench.methods.algebraic_descriptors import FEATURE_NAMES
from pocket_bench.metrics import roc_auc

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/cryptobench_apo/_cascade_cache_train.npz"
TEST = ROOT / "data/cryptobench_apo/_cascade_cache_test.npz"


def chain_ranks(F, n_res_per):
    """Normalized rank of every column within its own chain."""
    R = np.empty_like(F, dtype=np.float64)
    off = 0
    for n in n_res_per:
        n = int(n)
        blk = F[off:off + n]
        for j in range(F.shape[1]):
            x = blk[:, j]
            order = np.argsort(x, kind="stable")
            r = np.empty(n, dtype=np.float64)
            i = 0
            while i < n:
                k = i
                while k + 1 < n and x[order[k + 1]] == x[order[i]]:
                    k += 1
                r[order[i:k + 1]] = 0.5 * (i + k)
                i = k + 1
            R[off:off + n, j] = r / max(n - 1, 1)
        off += n
    return R


def per_unit_auc(score, y, n_res_per):
    out = []
    off = 0
    for n in n_res_per:
        n = int(n)
        s = score[off:off + n]
        t = y[off:off + n]
        off += n
        if t.sum() == 0 or t.sum() == n:
            continue
        a = roc_auc(list(s), list(t))
        if a is not None:
            out.append(a)
    return float(np.mean(out)), len(out)


def fisher(Rtr, ytr, Rte):
    R1, R0 = Rtr[ytr == 1], Rtr[ytr == 0]
    mu1, mu0 = R1.mean(0), R0.mean(0)
    S = ((R1 - mu1).T @ (R1 - mu1) + (R0 - mu0).T @ (R0 - mu0)) / (len(Rtr) - 2)
    S = S + np.eye(S.shape[0]) * 1e-9 * float(np.trace(S)) / S.shape[0]
    w = np.linalg.solve(S, mu1 - mu0)
    nrm = float(np.linalg.norm(w))
    if nrm > 0:
        w = w / nrm
    return Rte @ w, w


def main() -> int:
    ztr, zte = np.load(TRAIN, allow_pickle=False), np.load(TEST,
                                                           allow_pickle=False)
    Ftr, ytr, ntr = ztr["F"], ztr["y"], ztr["n_res_per"]
    Fte, yte, nte = zte["F"], zte["y"], zte["n_res_per"]
    print(f"train {Ftr.shape}  test {Fte.shape}")

    Rtr, Rte = chain_ranks(Ftr, ntr), chain_ranks(Fte, nte)

    print("\n=== (a) single invariant, test fold, per-unit mean ROC-AUC ===")
    singles = []
    for j, name in enumerate(FEATURE_NAMES):
        a, _ = per_unit_auc(Rte[:, j], yte, nte)
        singles.append((max(a, 1.0 - a), name, a))
    for s, name, a in sorted(singles, reverse=True):
        flag = "  <-- inverted" if a < 0.5 else ""
        print(f"  {name:16s} {a:.4f}   |2A-1|={abs(2*a-1):.4f}{flag}")

    print("\n=== (b) closed-form Fisher direction over rank features ===")
    rows = []
    for label, cols in [
        ("original 6 geometric", list(range(6))),
        ("all 35 algebraic", list(range(35))),
        ("top 20 by |2A-1|", [FEATURE_NAMES.index(n) for _, n, _
                              in sorted(singles, reverse=True)[:20]]),
        ("top 12 by |2A-1|", [FEATURE_NAMES.index(n) for _, n, _
                              in sorted(singles, reverse=True)[:12]]),
    ]:
        s, w = fisher(Rtr[:, cols], ytr, Rte[:, cols])
        a, nu = per_unit_auc(s, yte, nte)
        rows.append((label, len(cols), a, nu))
        print(f"  {label:24s} d={len(cols):3d}  ROC-AUC = {a:.4f}  ({nu} units)")

    print("\n=== reference ===")
    print("  flat quaternary LUT, 7 wires      0.7444   (combinational)")
    print("  Track A resolved (6 wires+Fisher) 0.7583   (continuous)")
    print("  P2Rank                            0.7930   (35 feat + RF)")

    out = ROOT / "results/official_fold/FEATURE_CEILING_DIAGNOSIS.json"
    out.write_text(json.dumps({
        "schema": "geoaudit.feature_ceiling.v1",
        "clinical_grade": False,
        "note": "diagnostic only; (b) uses a continuous discriminant and is "
                "excluded from the combinational claim",
        "n_test_units": int(len(nte)),
        "single_invariant_auc": {n: a for _, n, a in singles},
        "fisher": [{"features": lbl, "d": d, "roc_auc": a, "n_units": nu}
                   for lbl, d, a, nu in rows],
        "reference": {"flat_lut_7": 0.7444, "track_a_resolved": 0.7583,
                      "p2rank": 0.7930},
    }, indent=2))
    print(f"\nwrote -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
