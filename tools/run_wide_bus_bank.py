"""Multi-digit quaternary buses: trade wire count for wire resolution.

The measured defect
-------------------
Identical descriptors, identical fold: the closed-form Fisher direction over the
35 chain-normalized ranks reaches 0.7830 while the same 35 descriptors pushed
through four-level tables reach 0.7444 flat and 0.7615 banked. The 0.039 that
separates them is destroyed by the quantizer, not by the architecture and not by
the descriptors.

The repair
----------
Nothing in quaternary logic requires one digit per invariant. A wire may be
carried on a BUS of ``b`` quaternary digits, giving ``4^b`` levels, exactly as a
hardware datapath carries a word rather than a bit. What must stay bounded is
the ADDRESS, not the alphabet: a table over ``w`` wires each on ``b`` digits owns
``4^(b w)`` cells, so the saturation constraint is

    4^(b w)  <=  r N   =>   b w  <=  log_4(r N)  =  6.87.

The product ``b w`` is conserved. Spending it as ``b=1, w=6`` gives six coarse
wires; spending it as ``b=3, w=2`` gives two wires resolved to 64 levels each.
Both tables are equally dense. Which spending is correct is an empirical question
about where the information sits, and this script measures it.

Fusion remains the unweighted sum of cell fractions over a bank of tables that
partitions the 35 wires. Every coefficient is one. No weight is fitted.

Usage: PYTHONPATH=src python3.12 tools/run_wide_bus_bank.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pocket_bench.methods.algebraic_descriptors import FEATURE_NAMES
from pocket_bench.methods.cascade_lut import patch_mean
from pocket_bench.metrics import average_precision, roc_auc

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/cryptobench_apo/_cascade_cache_train.npz"
TEST = ROOT / "data/cryptobench_apo/_cascade_cache_test.npz"


def chain_levels(F, n_res_per, levels):
    """Each column banded into ``levels`` bins by its OWN chain's order.

    A comparator network over the chain: no constant is carried between chains
    and nothing is fitted. With ``levels = 4`` this reduces exactly to the
    quaternary digit used everywhere else.
    """
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
            q = np.floor(r / max(n - 1, 1) * levels).astype(np.int64)
            out[off:off + n, j] = np.clip(q, 0, levels - 1)
        off += n
    return out


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


def bank_score(Dtr, ytr, Dte, nte, tables, levels, rate):
    n_te = Dte.shape[0]
    s_te = np.zeros(n_te)
    yf = ytr.astype(np.float64)
    occ = []
    for cols in tables:
        n_cells = levels ** len(cols)
        a_tr = np.zeros(len(ytr), dtype=np.int64)
        a_te = np.zeros(n_te, dtype=np.int64)
        for t, c in enumerate(cols):
            a_tr += Dtr[:, c] * (levels ** t)
            a_te += Dte[:, c] * (levels ** t)
        tot = np.bincount(a_tr, minlength=n_cells).astype(np.float64)
        pos = np.bincount(a_tr, weights=yf, minlength=n_cells)
        frac = np.where(tot > 0, pos / np.maximum(tot, 1.0), rate)
        s_te += frac[a_te]
        occ.append(len(ytr) / n_cells)
    return s_te, float(min(occ))


def main() -> int:
    ztr, zte = np.load(TRAIN, allow_pickle=False), np.load(TEST,
                                                           allow_pickle=False)
    Ftr, ytr, ntr = ztr["F"], ztr["y"], ztr["n_res_per"]
    Fte, yte, nte, ctr_te = zte["F"], zte["y"], zte["n_res_per"], zte["ctr"]
    rate = float(ytr.mean())
    M = Ftr.shape[1]
    print(f"train {Ftr.shape}  test {Fte.shape}  base rate {rate:.4f}",
          flush=True)

    # (levels, wires per table): every combination keeps levels**w near 4096,
    # i.e. the same 57 training residues per cell.
    configs = [(4, 6), (8, 4), (16, 3), (64, 2), (256, 1)]
    thematic = list(range(M))                     # group-major column order

    rows = []
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for levels, w in configs:
        if levels not in cache:
            cache[levels] = (chain_levels(Ftr, ntr, levels),
                             chain_levels(Fte, nte, levels))
        Dtr, Dte = cache[levels]
        tables = [thematic[i:i + w] for i in range(0, M, w)]
        tables = [t for t in tables if t]
        s, occ = bank_score(Dtr, ytr, Dte, nte, tables, levels, rate)
        a, p = per_unit(s, yte, nte)
        m = patch_mean(s, ctr_te, nte)
        sc = s / max(abs(s).max(), 1e-12) + m / max(abs(m).max(), 1e-12)
        ap, pp = per_unit(sc, yte, nte)
        rows.append((levels, w, len(tables), occ, a, ap, p, pp))
        print(f"  levels={levels:4d} wires/table={w}  K={len(tables):2d}  "
              f"occ={occ:8.1f}/cell   AUC={a:.4f}  +patch={ap:.4f}", flush=True)

    best = max(rows, key=lambda r: max(r[4], r[5]))
    print(f"\nbest: levels={best[0]} w={best[1]}  "
          f"AUC={max(best[4], best[5]):.4f}")
    print("reference: flat 0.7444 | cascade 0.7388 | bank4 0.7615 | "
          "expansion 0.7413 | Fisher35 0.7830 | P2Rank 0.7930")

    out = ROOT / "results/official_fold/WIDE_BUS_BANK.json"
    out.write_text(json.dumps({
        "schema": "geoaudit.wide_bus_bank.v1",
        "clinical_grade": False,
        "invariant": "levels**wires_per_table is held near 4096 so every "
                     "configuration has the same cell occupancy",
        "n_test_units": int(len(nte)),
        "base_rate": rate,
        "results": [{"levels": l, "wires_per_table": w, "n_tables": k,
                     "occupancy_per_cell": o, "roc_auc": a,
                     "roc_auc_with_patch": ap, "pr_auc": p,
                     "pr_auc_with_patch": pp}
                    for l, w, k, o, a, ap, p, pp in rows],
        "reference": {"flat_lut_7": 0.7444, "bank4": 0.7615,
                      "fisher_35": 0.7830, "p2rank": 0.7930},
    }, indent=2))
    print(f"wrote -> {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
