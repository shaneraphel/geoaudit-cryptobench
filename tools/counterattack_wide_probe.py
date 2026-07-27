#!/usr/bin/env python3
"""The third and final reading of the official test fold.

Three readings, and the manuscript reports all three with the reason for each.

  1  0.7804   1032 pair tables over 172 wires, signed stagewise fan-out.
               Against P2Rank -0.0131, 95% CI [-0.0342, +0.0072]. The stagewise
               rule left the fan-out at a resolution of a few copies per table
               and the finer solved fan-out was unusable, collapsing from 0.7844
               to 0.6846 when the pool grew.

  2  0.7952   2404 pair and triple tables over the same 172 wires, fan-out from
               a ridge-regularised solve. Against P2Rank +0.0017, CI
               [-0.0194, +0.0216]. The collapse was a conditioning failure and
               the ridge removed it. This reading also matched a continuous
               linear functional of the same wires to within 0.0002 at p=0.98,
               which said the circuit had stopped losing anything to the
               continuous path and that nothing further could come from the
               fusion.

  3  this run. What changed is the inputs. 43 local quantities are now read
               under five statistics rather than one -- the mean at 6, 10, 14,
               20 and 26 A, the neighbourhood standard deviation, the centred
               difference, and the local rank |{j in N_r(i) : x_j < x_i}| /
               |N_r(i)| -- for 645 wires. On the pick half that took the field
               from 0.7932 to 0.8045, and past the 0.7936 a continuous
               functional of the wider wires reaches, so the tables are again
               reading something an additive functional cannot.

The frozen configuration, every part of it ranked on a cluster-disjoint half of
the training fold: 5152 two-wire tables from sixteen seeded random partitions of
the 645 wires, sixteen cells each and ~7500 training residues per occupied cell,
1.25% of cells never addressed; integer fan-out in [-32, 32] from a ridge-0.03
closed-form direction over the table outputs; the neighbourhood mean at 18 A
rescaled to the score's spread and added at weight one.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_wide_probe.py
"""
from __future__ import annotations

import gc
import json

import numpy as np

from pocket_bench.methods.table_bank import (
    cell_offsets,
    chain_digits,
    compile_cells,
    integer_fanout,
    partition_tables,
    score,
)
from pocket_bench.paths import ROOT

from counterattack_ridge import spread_matched_gate
from counterattack_select import SEED
from counterattack_test_probe import paired_bootstrap, per_unit_metrics

TRAIN = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
TEST = ROOT / "data/cryptobench_apo/_wide_cache_test.npz"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
OUT = ROOT / "results/official_fold/COUNTERATTACK_WIDE_PROBE.json"

PAIR_ROUNDS = 16
RIDGE = 0.03
CAP = 32
GATE_RADIUS = 18.0
GATE_WEIGHT = 1.0
PICK_HALF = 0.8045


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

    Dtr = chain_digits(Xtr, ntr)
    Dte = chain_digits(Xte, nte)
    n_wires = Dtr.shape[1]
    del Xtr, Xte, ztr, zte
    gc.collect()

    tables = partition_tables(n_wires, 2, PAIR_ROUNDS, SEED)
    offsets = cell_offsets(tables)
    frac, tot = compile_cells(Dtr, ytr, tables, offsets)
    empty = int((tot == 0).sum())
    print(f"{len(tables)} tables, {len(frac)} cells, {empty} never addressed "
          f"({100.0 * empty / len(frac):.2f}%), {tot[tot > 0].mean():.0f} "
          f"training residues per occupied cell", flush=True)

    m = integer_fanout(Dtr, ytr, tables, offsets, frac, RIDGE, CAP)
    print(f"integer fan-out in [-{CAP}, {CAP}]: {int((m != 0).sum())} of "
          f"{len(tables)} tables carry a non-zero weight, total "
          f"{int(np.abs(m).sum())}", flush=True)

    S = score(Dte, tables, offsets, frac, m)
    F = spread_matched_gate(S, ctr_te, nte, GATE_RADIUS, GATE_WEIGHT)

    rows = per_unit_metrics(F, yte, nte, units_te)
    ours = {r["unit_id"]: r["residue_auc"] for r in rows}
    ours_pr = {r["unit_id"]: r["residue_pr_auc"] for r in rows}
    scored = [v for v in ours.values() if v is not None]
    mean_auc = float(np.mean(scored))
    mean_pr = float(np.mean([v for v in ours_pr.values() if v is not None]))
    print(f"\nwide table field on the official test fold: "
          f"ROC-AUC {mean_auc:.4f}, PR-AUC {mean_pr:.4f} over "
          f"{len(scored)} scored units", flush=True)

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
        for label in ("residue_auc", "residue_pr_auc"):
            d = comparisons[method][label]
            print(f"  {label:14s} vs {method:22s} {d['mean_a']:.4f} - "
                  f"{d['mean_b']:.4f} = {d['paired_difference']:+.4f}  "
                  f"95% CI [{d['ci_low']:+.4f}, {d['ci_high']:+.4f}]  "
                  f"p={d['p_two_sided']:.4f}  "
                  f"{'separable' if d['excludes_zero'] else 'not separable'}",
                  flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_wide_probe.v1",
        "clinical_grade": False,
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "is_official_mmseqs2_10pct_test_fold": True,
        "test_fold_read_index": 3,
        "earlier_reads": [
            {"index": 1, "residue_auc_mean": 0.7804,
             "paired_vs_p2rank": -0.0131,
             "architecture": "1032 pair tables over 172 wires, signed "
                             "stagewise fan-out"},
            {"index": 2, "residue_auc_mean": 0.7952,
             "paired_vs_p2rank": +0.0017,
             "architecture": "2404 pair and triple tables over 172 wires, "
                             "ridge-regularised integer fan-out"},
        ],
        "architecture": {
            "wires": n_wires,
            "wire_families": ["local", "mean@6/10/14/20/26", "sd@6/14/20",
                              "centred@6/14/20", "localrank@6/14/20"],
            "digitisation": "per-chain rank, quartile cut, 4 levels",
            "table_width": 2, "partition_rounds": PAIR_ROUNDS,
            "n_tables": len(tables), "n_cells": int(len(frac)),
            "n_cells_never_addressed": empty,
            "mean_training_residues_per_occupied_cell":
                float(tot[tot > 0].mean()),
            "ridge": RIDGE, "fan_out_cap": CAP,
            "n_tables_with_nonzero_fanout": int((m != 0).sum()),
            "total_fan_out": int(np.abs(m).sum()),
            "gate": {"radius_angstrom": GATE_RADIUS, "weight": GATE_WEIGHT,
                     "rescaling": "neighbourhood mean matched to the raw "
                                  "score's standard deviation"},
            "inference": "one cell read per table, an integer-weighted sum, "
                         "and one neighbourhood mean",
        },
        "selection_provenance": "training fold only; cluster-disjoint fit/pick "
                                "halves; results/architecture_sweep/"
                                "COUNTERATTACK_WIDE2.json",
        "pick_half_roc_auc_of_this_configuration": PICK_HALF,
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
