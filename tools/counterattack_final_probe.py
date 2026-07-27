#!/usr/bin/env python3
"""The second and final reading of the official test fold for the table field.

Both readings are reported, and this header says why there are two.

The first read applied a bank of 1032 pair tables fused by signed stagewise
selection: 0.7804 against P2Rank's 0.7935, a paired difference of -0.0131 with a
95% interval of [-0.0342, +0.0072]. Parity, not a win. It also showed the
stagewise rule was leaving the fan-out at a resolution of 1, 2 or 3 copies per
table, while the rounded closed-form solve, which gives a resolution of 32, was
worth 0.003 more on the training split but collapsed from 0.7844 to 0.6846 when
the pool grew from 1032 to 1720 tables.

That collapse was diagnosed on the training fold and it is a conditioning
failure, not a property of the architecture: random pairs drawn from a lattice
of 14706 start repeating, near-duplicate columns appear, and an unregularised
solve puts large opposing coefficients on copies of each other. Adding a ridge
of the scatter trace -- the same device the repository's fitted linear readout
already uses -- removes it. With regularisation the same sweep reads 0.7898 at
1032 tables and 0.7906 at 1720, and continues to 0.7932 once three-wire tables
join the pair tables and the pool reaches 2404. All of that was measured on the
pick half.

So the architecture changed after the first test read, for a reason established
without the test fold, and the honest description is two readings rather than
one. The frozen configuration:

  wires        172 = (35 algebraic and topological invariants + 7 published
               residue constants + 1 training propensity) x (self, 6, 14, 20 A)
  digits       four levels, by the residue's rank within its own chain
  tables       1720 two-wire tables (20 seeded random partitions of all wires)
               and 684 three-wire tables (12 partitions); 16 and 64 cells
  cells        exact empirical pos_k/tot_k on the training fold
  fan-out      integer m_k in [-32, 32], from a ridge-0.3 closed-form direction
               over the table outputs, rounded
  gate         neighbourhood mean at 14 A, rescaled to the score's spread,
               added at weight 1

Inference is a table lookup per table, an integer-weighted sum, and one spatial
mean. Every structural choice above was ranked on a cluster-disjoint half of the
training fold.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_final_probe.py
"""
from __future__ import annotations

import gc
import json

import numpy as np

from pocket_bench.metrics import average_precision, roc_auc
from pocket_bench.paths import ROOT

from counterattack_covering import compile_bank32, partitions
from counterattack_ridge import quantize, ridge_direction, spread_matched_gate
from counterattack_select import chain_digits, pooled_auc
from counterattack_test_probe import paired_bootstrap, per_unit_metrics

TRAIN = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
TEST = ROOT / "data/cryptobench_apo/_expanded_cache_test.npz"
DIGITS = ROOT / "data/cryptobench_apo/_expanded_digits_train.npy"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
OUT = ROOT / "results/official_fold/COUNTERATTACK_FINAL_PROBE.json"

BASIS = ((2, 20), (3, 12))
RIDGE = 0.3
CAP = 32
GATE_RADIUS = 14.0
GATE_WEIGHT = 1.0


def main() -> int:
    ztr = np.load(TRAIN, allow_pickle=False)
    zte = np.load(TEST, allow_pickle=False)
    Xtr, ytr, ntr = ztr["X"], ztr["y"], ztr["n_res_per"]
    Xte, yte, nte = zte["X"], zte["y"], zte["n_res_per"]
    ctr_te = zte["ctr"]
    units_te = [str(u) for u in zte["units"]]
    names = [str(s) for s in ztr["names"]]
    print(f"train {len(ntr)} units / {len(ytr)} residues, "
          f"test {len(nte)} units / {len(yte)} residues, "
          f"{Xtr.shape[1]} wires", flush=True)

    Dtr = (np.load(DIGITS) if DIGITS.exists() else chain_digits(Xtr, ntr))
    Dtr = np.ascontiguousarray(Dtr, dtype=np.int32)
    Dte = np.ascontiguousarray(chain_digits(Xte, nte), dtype=np.int32)
    n_wires = Dtr.shape[1]
    del Xtr, Xte, ztr, zte
    gc.collect()

    rate = float(ytr.mean())
    gini = np.array([abs(2.0 * pooled_auc(Dtr[:, j].astype(float), ytr) - 1.0)
                     for j in range(n_wires)])
    tables = []
    for width, rounds in BASIS:
        tables += partitions(n_wires, width, rounds, gini, "random")
    A_tr, A_te, occ = compile_bank32(Dtr, ytr, Dte, tables, rate)
    print(f"{len(tables)} tables, {occ:.0f} residues per occupied cell",
          flush=True)

    m = quantize(ridge_direction(A_tr, ytr, RIDGE), CAP)
    del A_tr
    gc.collect()
    print(f"ridge {RIDGE} direction, rounded to [-{CAP}, {CAP}]: "
          f"{int((m != 0).sum())} tables carry non-zero fan-out, total "
          f"{int(np.abs(m).sum())}", flush=True)

    S = A_te.astype(np.float64) @ m.astype(np.float64)
    F = spread_matched_gate(S, ctr_te, nte, GATE_RADIUS, GATE_WEIGHT)

    rows = per_unit_metrics(F, yte, nte, units_te)
    ours = {r["unit_id"]: r["residue_auc"] for r in rows}
    ours_pr = {r["unit_id"]: r["residue_pr_auc"] for r in rows}
    scored = [v for v in ours.values() if v is not None]
    mean_auc = float(np.mean(scored))
    mean_pr = float(np.mean([v for v in ours_pr.values() if v is not None]))
    print(f"\ntable field on the official test fold: ROC-AUC {mean_auc:.4f}, "
          f"PR-AUC {mean_pr:.4f} over {len(scored)} scored units", flush=True)

    tel = json.loads(TELEMETRY.read_text())
    tel_rows = tel["rows"] if isinstance(tel, dict) and "rows" in tel else tel
    comparisons = {}
    for method in ("p2rank", "algebraic_field_linear", "algebraic_field",
                   "quaternary_lut_seq", "geometric_foundation"):
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
        print(f"  vs {method:24s} {d['mean_a']:.4f} - {d['mean_b']:.4f} = "
              f"{d['paired_difference']:+.4f}  95% CI "
              f"[{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
              f"p={d['p_two_sided']:.4f}  "
              f"{'separable' if d['excludes_zero'] else 'not separable at 95%'}",
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_final_probe.v1",
        "clinical_grade": False,
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "is_official_mmseqs2_10pct_test_fold": True,
        "test_fold_read_index": 2,
        "first_read": {
            "artifact": "results/official_fold/COUNTERATTACK_TEST_PROBE.json",
            "architecture": "1032 pair tables, signed stagewise fan-out, "
                            "multi-scale max-normalised gate",
            "residue_auc_mean": 0.7804,
            "paired_vs_p2rank": -0.0131,
            "reason_for_second_read": "the stagewise fan-out has a resolution "
                                      "of a few copies per table; the finer "
                                      "solved fan-out was unusable until its "
                                      "conditioning failure was fixed with a "
                                      "ridge, which was diagnosed and repaired "
                                      "on the training fold",
        },
        "architecture": {
            "wires": n_wires,
            "digitisation": "per-chain rank, quartile cut, 4 levels",
            "basis": [{"width": w, "partition_rounds": r} for w, r in BASIS],
            "n_tables": len(tables),
            "mean_residues_per_occupied_cell": occ,
            "ridge": RIDGE, "fan_out_cap": CAP,
            "n_tables_with_nonzero_fanout": int((m != 0).sum()),
            "total_fan_out": int(np.abs(m).sum()),
            "gate": {"radius_angstrom": GATE_RADIUS, "weight": GATE_WEIGHT,
                     "rescaling": "neighbourhood mean matched to the raw "
                                  "score's standard deviation"},
            "fitting": "one regularised closed-form solve over table outputs "
                       "on the training fold, rounded to integers; no "
                       "gradient, no iteration, no test-fold selection",
        },
        "selection_provenance": "training fold only; cluster-disjoint fit/pick "
                                "halves; see results/architecture_sweep/"
                                "COUNTERATTACK_RIDGE2.json",
        "pick_half_roc_auc_of_this_configuration": 0.7932,
        "n_test_units": len(units_te),
        "n_scored_units": len(scored),
        "residue_auc_mean": mean_auc,
        "residue_pr_auc_mean": mean_pr,
        "paired_vs": comparisons,
        "per_structure": rows,
    }, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
