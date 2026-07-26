"""Parallel table bank with uniform counting fusion.

Why the cascade failed, stated exactly
--------------------------------------
A group table estimates the cell fraction from about ``N/4^6 = 57`` residues at
base rate ``r = 0.0576``, i.e. ``3.3`` expected positives per cell. The standard
error of that fraction is ``sqrt(r(1-r)/57) = 0.031`` against a mean of
``0.058``: a 53 percent relative error. Banding such an estimate into two bits is
a HARD decision taken on a noise-dominated statistic, and a hard decision is not
invertible. Each cascade level therefore commits an irreversible quantization
error, and the levels compound. Measured: every cascaded topology scored below
the flat control (best cascade 0.7388, flat 0.7444).

The repair, still without a fitted weight
-----------------------------------------
Do not decide early. Read every group table in PARALLEL and fuse by counting:

    S(i) = sum_k p_k(i),        p_k = pos_k[a_k(i)] / tot_k[a_k(i)]

Every coefficient in that sum is exactly one. Nothing is fitted, nothing is
optimized, no direction is solved for; the fusion is the unweighted sum of
empirical conditional frequencies, which is a pure counting statistic of the
compiled tables. What it buys is RESOLUTION: a single table offers at most 4^6
distinct levels of which only the asserted ones occur, whereas the sum of K
tables offers the Minkowski sum of their level sets, so the ordering that a
single hard decision destroys survives.

The spatial gate is applied to the fused count, not to a banded digit, for the
same reason: a cryptic site is a patch, and the patch statement should be made
on the finest available resolution rather than on a two-bit summary.

Usage: PYTHONPATH=src python3.12 tools/run_bank_experiment.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pocket_bench.methods.algebraic_descriptors import FEATURE_NAMES, GROUPS
from pocket_bench.methods.cascade_lut import (
    N_LEVELS, QLUT, chain_rank_digits, pack, patch_mean,
)
from pocket_bench.metrics import average_precision, roc_auc

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/cryptobench_apo/_cascade_cache_train.npz"
TEST = ROOT / "data/cryptobench_apo/_cascade_cache_test.npz"


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
    return float(np.mean(aucs)), float(np.mean(prs)), len(aucs)


def blocks(names, size):
    """Contiguous disjoint blocks of column indices."""
    idx = [FEATURE_NAMES.index(n) for n in names]
    return [idx[i:i + size] for i in range(0, len(idx), size)]


def strength_order(Dtr, ytr):
    """Columns sorted by their own single-wire table separation |2A-1|."""
    n_pos, n_neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
    s = []
    for j in range(Dtr.shape[1]):
        x = Dtr[:, j].astype(np.float64)
        r = np.empty(len(x))
        r[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
        auc = (r[ytr == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
        s.append((abs(2 * auc - 1), j))
    return [j for _, j in sorted(s, reverse=True)]


def main() -> int:
    ztr, zte = np.load(TRAIN, allow_pickle=False), np.load(TEST,
                                                           allow_pickle=False)
    Ftr, ytr, ntr = ztr["F"], ztr["y"], ztr["n_res_per"]
    Fte, yte, nte, ctr_te = zte["F"], zte["y"], zte["n_res_per"], zte["ctr"]
    ctr_tr = ztr["ctr"]
    print(f"train {Ftr.shape}  test {Fte.shape}", flush=True)

    Dtr = chain_rank_digits(Ftr, ntr)
    Dte = chain_rank_digits(Fte, nte)

    order = strength_order(Dtr, ytr)
    named = [FEATURE_NAMES[j] for j in order]
    print(f"strongest wires: {named[:8]}", flush=True)

    banks: dict[str, list[list[int]]] = {}
    banks["thematic 6x6"] = blocks(
        [n for g in GROUPS for n in g], 6)
    banks["strength 6x6"] = [order[i:i + 6] for i in range(0, 36, 6)]
    banks["strength 7x5"] = [order[i:i + 5] for i in range(0, 35, 5)]
    banks["strength 12x3"] = [order[i:i + 3] for i in range(0, 36, 3)]
    banks["strength 18x2"] = [order[i:i + 2] for i in range(0, 36, 2)]
    # interleaved: each table mixes a strong wire with weaker ones so no table
    # is built purely from noise
    inter = [[order[k] for k in range(i, 35, 6)] for i in range(6)]
    banks["interleaved 6"] = inter
    inter4 = [[order[k] for k in range(i, 35, 9)] for i in range(9)]
    banks["interleaved 9"] = inter4

    rows = []
    for name, bank in banks.items():
        bank = [b for b in bank if len(b) >= 2]
        luts = [QLUT.compile(Dtr[:, cols], ytr) for cols in bank]
        s_tr = np.zeros(len(ytr))
        s_te = np.zeros(len(yte))
        for lut, cols in zip(luts, bank):
            s_tr += lut.frac(Dtr[:, cols])
            s_te += lut.frac(Dte[:, cols])
        a_raw, p_raw, nu = per_unit(s_te, yte, nte)

        # spatial gate on the fused count, at the finest resolution
        m_te = patch_mean(s_te, ctr_te, nte)
        a_patch, p_patch, _ = per_unit(s_te + m_te, yte, nte)

        widths = sorted({len(b) for b in bank})
        occ = [len(ytr) / (N_LEVELS ** len(b)) for b in bank]
        rows.append((name, len(bank), widths, min(occ), a_raw, a_patch,
                     p_raw, p_patch))
        print(f"  {name:16s} K={len(bank):2d} w={widths} "
              f"min-occ={min(occ):7.1f}/cell   "
              f"AUC={a_raw:.4f}  +patch={a_patch:.4f}", flush=True)

    best = max(rows, key=lambda r: max(r[4], r[5]))
    print(f"\nbest bank: {best[0]}  AUC={max(best[4], best[5]):.4f}")
    print("reference: flat LUT 0.7444 | best cascade 0.7388 | "
          "35-wire Fisher 0.7830 | P2Rank 0.7930")

    out = ROOT / "results/official_fold/BANK_FUSION.json"
    out.write_text(json.dumps({
        "schema": "geoaudit.bank_fusion.v1",
        "clinical_grade": False,
        "fusion": "unweighted sum of table cell fractions; all coefficients "
                  "are exactly 1; no fitted weight, no discriminant",
        "n_test_units": int(len(nte)),
        "banks": [{"name": n, "n_tables": k, "widths": w,
                   "min_occupancy_per_cell": o, "roc_auc": a,
                   "roc_auc_with_patch": ap, "pr_auc": p,
                   "pr_auc_with_patch": pp}
                  for n, k, w, o, a, ap, p, pp in rows],
        "reference": {"flat_lut_7": 0.7444, "best_cascade": 0.7388,
                      "fisher_35": 0.7830, "p2rank": 0.7930},
    }, indent=2))
    print(f"wrote -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
