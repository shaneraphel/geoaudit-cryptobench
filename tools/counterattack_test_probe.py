#!/usr/bin/env python3
"""The single reading of the official test fold for the pair-table field.

Everything up to this point has been decided on the training fold: the wires,
the digitisation, the table geometry, the pool size, the multiplicity rule and
the number of stagewise rounds were all chosen on a fit half and ranked on a
pick half whose MMseqs2 clusters the fit half never saw. This script takes the
one architecture that came out of that process, compiles it on the whole
training fold, and reads the 192 official test units once.

What is compiled on the training fold, and nothing else is: the cell frequencies
pos_k/tot_k of each pair table, and the integer multiplicities m_k. The
digitisation is not compiled at all -- each wire is ranked within its own chain
and cut at quartiles, so a test structure is quantised using only itself, and no
threshold crosses the fold boundary. Inference is

    S(i)   = sum_k m_k * pos_k[a_k(i)] / tot_k[a_k(i)]
    F(i)   = S(i)/max|S| + sum_r G_r(i)/max|G_r|

where a_k(i) is the address formed by two quaternary digits, m_k is a
non-negative integer, and G_r is the mean of S over the residues within r of i.

Reported against P2Rank on the same 192 units by paired bootstrap over
structures, which is the comparison that decides whether the difference is real
or is one of the eleven structures where either method happens to fail.

Usage:
  PYTHONPATH=src:tools python3.12 tools/counterattack_test_probe.py \
      --width 2 --pool-rounds 12 --fusion stagewise --rounds 150
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from pocket_bench.metrics import average_precision, roc_auc
from pocket_bench.paths import ROOT

from counterattack_covering import (
    compile_bank32,
    direction_from_matrix,
    partitions,
)
from counterattack_greedy import stagewise
from counterattack_quantized import quantize
from counterattack_select import GATE_RADII, _unit, chain_digits, gate, pooled_auc

TRAIN = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
TEST = ROOT / "data/cryptobench_apo/_expanded_cache_test.npz"
DIGITS = ROOT / "data/cryptobench_apo/_expanded_digits_train.npy"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
OUT = ROOT / "results/official_fold/COUNTERATTACK_TEST_PROBE.json"
N_BOOT = 10000
BOOT_SEED = 20260725


def per_unit_metrics(score, y, n_res_per, units):
    rows = []
    off = 0
    for u, n in zip(units, n_res_per):
        n = int(n)
        s, t = score[off:off + n], y[off:off + n]
        off += n
        if t.sum() == 0 or t.sum() == n:
            rows.append({"unit_id": u, "residue_auc": None,
                         "residue_pr_auc": None, "n_universe": n,
                         "n_true": int(t.sum())})
            continue
        rows.append({"unit_id": u,
                     "residue_auc": roc_auc(list(s), list(t)),
                     "residue_pr_auc": average_precision(list(s), list(t)),
                     "n_universe": n, "n_true": int(t.sum())})
    return rows


def paired_bootstrap(a, b, n_boot=N_BOOT, seed=BOOT_SEED):
    """Bootstrap over structures of the paired difference a - b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    keep = ~(np.isnan(a) | np.isnan(b))
    a, b = a[keep], b[keep]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "n_paired_structures": int(len(a)),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "paired_difference": float(a.mean() - b.mean()),
        "ci_low": float(lo), "ci_high": float(hi),
        "p_two_sided": float(2.0 * min((diffs <= 0).mean(),
                                       (diffs >= 0).mean())),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--pool-rounds", type=int, default=12)
    ap.add_argument("--fusion", choices=("stagewise", "solved"),
                    default="stagewise")
    ap.add_argument("--rounds", type=int, default=150)
    ap.add_argument("--cap", type=int, default=32)
    args = ap.parse_args()

    ztr = np.load(TRAIN, allow_pickle=False)
    zte = np.load(TEST, allow_pickle=False)
    Xtr, ytr, ntr = ztr["X"], ztr["y"], ztr["n_res_per"]
    Xte, yte, nte = zte["X"], zte["y"], zte["n_res_per"]
    ctr_te = zte["ctr"]
    units_te = [str(u) for u in zte["units"]]
    names = [str(s) for s in ztr["names"]]
    assert Xtr.shape[1] == Xte.shape[1] == len(names)
    print(f"train {len(ntr)} units / {len(ytr)} residues, "
          f"test {len(nte)} units / {len(yte)} residues, "
          f"{Xtr.shape[1]} wires", flush=True)

    Dtr = np.load(DIGITS) if DIGITS.exists() else chain_digits(Xtr, ntr)
    Dte = chain_digits(Xte, nte)
    rate = float(ytr.mean())
    gini_wire = np.array([abs(2.0 * pooled_auc(Dtr[:, j].astype(float), ytr)
                              - 1.0) for j in range(Dtr.shape[1])])

    tables = partitions(Dtr.shape[1], args.width, args.pool_rounds,
                        gini_wire, "random")
    A_tr, A_te, occ = compile_bank32(Dtr, ytr, Dte, tables, rate)
    print(f"{len(tables)} tables of {4 ** args.width} cells, "
          f"{occ:.0f} residues per occupied cell", flush=True)

    if args.fusion == "stagewise":
        mult, trace = stagewise(A_tr, ytr, args.rounds)
        fusion_desc = (f"stagewise integer multiplicities, {len(trace)} "
                       f"accepted rounds over a pool of {len(tables)} tables")
    else:
        mult = quantize(direction_from_matrix(A_tr, ytr), args.cap)
        trace = []
        fusion_desc = (f"closed-form direction over {len(tables)} table "
                       f"outputs, rounded onto the integer grid "
                       f"[-{args.cap}, {args.cap}]")
    print(f"{fusion_desc}; {int((mult != 0).sum())} distinct tables carry "
          f"non-zero fan-out, total fan-out {int(np.abs(mult).sum())}",
          flush=True)

    S = A_te.astype(np.float64) @ mult.astype(np.float64)
    G = np.sum([_unit(gate(S, ctr_te, nte, r)) for r in GATE_RADII], axis=0)
    F = _unit(S) + _unit(G)

    rows = per_unit_metrics(F, yte, nte, units_te)
    ours = {r["unit_id"]: r["residue_auc"] for r in rows}
    ours_pr = {r["unit_id"]: r["residue_pr_auc"] for r in rows}
    scored = [v for v in ours.values() if v is not None]
    mean_auc = float(np.mean(scored))
    mean_pr = float(np.mean([v for v in ours_pr.values() if v is not None]))
    print(f"\npair-table field on the official test fold: "
          f"ROC-AUC {mean_auc:.4f}, PR-AUC {mean_pr:.4f} "
          f"over {len(scored)} scored units", flush=True)

    tel = json.loads(TELEMETRY.read_text())
    tel_rows = tel["rows"] if isinstance(tel, dict) and "rows" in tel else tel
    comparisons = {}
    for method in ("p2rank", "algebraic_field", "algebraic_field_linear",
                   "geometric_foundation"):
        other = {r["unit_id"]: r.get("residue_auc") for r in tel_rows
                 if r["method"] == method}
        other_pr = {r["unit_id"]: r.get("residue_pr_auc") for r in tel_rows
                    if r["method"] == method}
        shared = [u for u in ours
                  if ours[u] is not None and other.get(u) is not None]
        if not shared:
            continue
        comparisons[method] = {
            "residue_auc": paired_bootstrap([ours[u] for u in shared],
                                            [other[u] for u in shared]),
            "residue_pr_auc": paired_bootstrap(
                [ours_pr[u] for u in shared],
                [other_pr.get(u, float("nan")) for u in shared]),
        }
        d = comparisons[method]["residue_auc"]
        verdict = ("separable" if d["excludes_zero"]
                   else "not separable at 95%")
        print(f"  vs {method:24s} ROC-AUC {d['mean_a']:.4f} - "
              f"{d['mean_b']:.4f} = {d['paired_difference']:+.4f}  "
              f"95% CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
              f"p={d['p_two_sided']:.4f}  {verdict}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_test_probe.v1",
        "clinical_grade": False,
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "is_official_mmseqs2_10pct_test_fold": True,
        "architecture": {
            "wires": Dtr.shape[1],
            "digitisation": "per-chain rank, quartile cut, 4 levels",
            "table_width": args.width,
            "cells_per_table": 4 ** args.width,
            "n_pool_tables": len(tables),
            "pool_construction": f"{args.pool_rounds} seeded random partitions "
                                 "of all wires into groups",
            "mean_residues_per_occupied_cell": occ,
            "fusion": fusion_desc,
            "n_tables_with_nonzero_fanout": int((mult != 0).sum()),
            "total_fan_out": int(np.abs(mult).sum()),
            "gate_radii_angstrom": list(GATE_RADII),
        },
        "selection_provenance": "training fold only; cluster-disjoint fit/pick "
                                "halves; see results/architecture_sweep/",
        "n_test_units": len(units_te),
        "n_scored_units": len(scored),
        "residue_auc_mean": mean_auc,
        "residue_pr_auc_mean": mean_pr,
        "paired_vs": comparisons,
        "per_structure": rows,
        "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED, "ci_level": 0.95,
                      "resampling_unit": "structure"},
    }, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
