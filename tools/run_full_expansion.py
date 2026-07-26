"""Complete low-order counting expansion over the algebraic word.

The bank experiment showed that fusing several dense tables by unweighted
addition recovers the resolution that a single hard decision destroys
(0.7615 against 0.7444 flat). It still selects which wires share a table, and
the selection is arbitrary. The complete expansion removes the choice.

Take EVERY unordered pair (and every unordered triple) of the 35 quaternary
wires. A pair table owns 4^2 = 16 cells at 14677 training residues per cell; a
triple table owns 4^3 = 64 cells at 3669 per cell. Both are saturated by orders
of magnitude, so no cell fraction is noise-dominated -- which is precisely the
defect that killed the cascade. Fuse by counting:

    S(i) = sum over all pairs (a,b) of  p_{ab}( d_a(i), d_b(i) )

Every wire occurs in exactly ``M-1`` pairs and every pair carries coefficient
one, so the expansion is completely symmetric: nothing is selected, nothing is
weighted, nothing is fitted. It is the second-order (respectively third-order)
term of the exact counting expansion of the joint table, evaluated on cells that
are statistically saturated.

Three fusion rules are compared, all of them pure counting:
    sum    S = sum_k p_k
    rank   S = sum_k rank_chain(p_k)        (comparator network per chain)
    excess S = sum_k (p_k - r)              (deviation from the compiled base
                                             rate; an integer-count difference)

Usage: PYTHONPATH=src python3.12 tools/run_full_expansion.py
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from pocket_bench.methods.algebraic_descriptors import FEATURE_NAMES
from pocket_bench.methods.cascade_lut import chain_rank_digits, patch_mean
from pocket_bench.metrics import average_precision, roc_auc

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/cryptobench_apo/_cascade_cache_train.npz"
TEST = ROOT / "data/cryptobench_apo/_cascade_cache_test.npz"
L = 4


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


def chain_rank(x, n_res_per):
    out = np.empty(len(x))
    off = 0
    for n in n_res_per:
        n = int(n)
        v = x[off:off + n]
        order = np.argsort(v, kind="stable")
        r = np.empty(n)
        i = 0
        while i < n:
            k = i
            while k + 1 < n and v[order[k + 1]] == v[order[i]]:
                k += 1
            r[order[i:k + 1]] = 0.5 * (i + k)
            i = k + 1
        out[off:off + n] = r / max(n - 1, 1)
        off += n
    return out


def expand(Dtr, ytr, ntr, Dte, nte, combos, rate):
    """Accumulate the three fusion rules over a family of wire subsets."""
    n_tr, n_te = len(ytr), Dte.shape[0]
    s_sum_te = np.zeros(n_te)
    s_rank_te = np.zeros(n_te)
    s_exc_te = np.zeros(n_te)
    yf = ytr.astype(np.float64)
    for cols in combos:
        k = len(cols)
        n_cells = L ** k
        a_tr = np.zeros(n_tr, dtype=np.int64)
        a_te = np.zeros(n_te, dtype=np.int64)
        for t, c in enumerate(cols):
            a_tr += Dtr[:, c] * (L ** t)
            a_te += Dte[:, c] * (L ** t)
        tot = np.bincount(a_tr, minlength=n_cells).astype(np.float64)
        pos = np.bincount(a_tr, weights=yf, minlength=n_cells)
        # A never-asserted cell has no driver; it contributes the compiled base
        # rate, which is the only value consistent with zero evidence.
        frac = np.where(tot > 0, pos / np.maximum(tot, 1.0), rate)
        p_te = frac[a_te]
        s_sum_te += p_te
        s_exc_te += p_te - rate
        s_rank_te += chain_rank(p_te, nte)
    return s_sum_te, s_rank_te, s_exc_te


def main() -> int:
    ztr, zte = np.load(TRAIN, allow_pickle=False), np.load(TEST,
                                                           allow_pickle=False)
    Ftr, ytr, ntr = ztr["F"], ztr["y"], ztr["n_res_per"]
    Fte, yte, nte, ctr_te = zte["F"], zte["y"], zte["n_res_per"], zte["ctr"]
    print(f"train {Ftr.shape}  test {Fte.shape}", flush=True)

    Dtr = chain_rank_digits(Ftr, ntr)
    Dte = chain_rank_digits(Fte, nte)
    rate = float(ytr.mean())
    M = Dtr.shape[1]

    families = {
        "all pairs": list(itertools.combinations(range(M), 2)),
        "all triples": list(itertools.combinations(range(M), 3)),
    }

    rows = []
    for name, combos in families.items():
        t0 = time.perf_counter()
        occ = len(ytr) / (L ** len(combos[0]))
        s_sum, s_rank, s_exc = expand(Dtr, ytr, ntr, Dte, nte, combos, rate)
        for rule, s in (("sum", s_sum), ("rank", s_rank), ("excess", s_exc)):
            a, p = per_unit(s, yte, nte)
            m = patch_mean(s, ctr_te, nte)
            # the gate is added at the same scale as the score it gates
            sc = s / max(abs(s).max(), 1e-12) + m / max(abs(m).max(), 1e-12)
            ap, pp = per_unit(sc, yte, nte)
            rows.append((name, rule, len(combos), occ, a, ap, p, pp))
            print(f"  {name:12s} {rule:7s} K={len(combos):5d} "
                  f"occ={occ:9.1f}/cell  AUC={a:.4f}  +patch={ap:.4f}",
                  flush=True)
        print(f"    ({time.perf_counter()-t0:.0f}s)", flush=True)

    best = max(rows, key=lambda r: max(r[4], r[5]))
    print(f"\nbest: {best[0]} / {best[1]}  AUC={max(best[4], best[5]):.4f}")
    print("reference: flat 0.7444 | cascade 0.7388 | bank 0.7615 | "
          "Fisher35 0.7830 | P2Rank 0.7930")

    out = ROOT / "results/official_fold/FULL_EXPANSION.json"
    out.write_text(json.dumps({
        "schema": "geoaudit.full_expansion.v1",
        "clinical_grade": False,
        "fusion": "unweighted sum over the complete family of wire subsets; "
                  "every coefficient is exactly 1; no selection, no weight",
        "n_test_units": int(len(nte)),
        "base_rate": rate,
        "results": [{"family": f, "rule": r, "n_tables": k,
                     "occupancy_per_cell": o, "roc_auc": a,
                     "roc_auc_with_patch": ap, "pr_auc": p,
                     "pr_auc_with_patch": pp}
                    for f, r, k, o, a, ap, p, pp in rows],
        "reference": {"flat_lut_7": 0.7444, "best_cascade": 0.7388,
                      "bank": 0.7615, "fisher_35": 0.7830, "p2rank": 0.7930},
    }, indent=2))
    print(f"wrote -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
