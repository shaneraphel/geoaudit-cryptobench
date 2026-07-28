#!/usr/bin/env python3
"""The fourth reading of the official test fold: the counting field, quotiented.

What is being tested, and what was expected
-------------------------------------------
The algebraic field is the one detector in this paper that loses to P2Rank by a
margin the fold resolves: -0.0266 ROC-AUC, 95% CI [-0.0487, -0.0055]. The
manuscript attributes the loss to capacity rather than to the invariants, since
a closed-form linear functional of the same thirty-five invariants reaches
0.7954 where the tables reach 0.7667, and a dense quaternary table here admits
``d <= log_4(rN) = 6.86`` digits.

``quotient_tables`` shows that bound counts cells and that a table invariant
under a group has as many free cells as the group has orbits. Under ``S_6``
inside each thematic group the cell count at ``d = 6`` falls from 4096 to 84, and
the budget that frees buys resolution: the digits can carry eight or twelve
levels where a dense table is stuck at four.

On fourteen training-fold splits -- CryptoBench's own four cluster-disjoint
folds and ten accession-disjoint half-splits -- the winning construction beats
the frozen dense bank on all fourteen, by a mean of +0.0087 (worst +0.0024).
Selection is recorded in results/architecture_sweep/COUNTERATTACK_QUOTIENT.json
and no part of it read this fold.

+0.0087 on the training fold does not close -0.0266 on this one. This reading
was taken knowing that: the construction is a real improvement to the counting
field and a real statement about what a symmetry buys a table, and the honest
place to record it is beside the number it actually produces, not instead of it.

Two details that would otherwise be invisible
---------------------------------------------
The selection harness normalises the score and the gate over the whole pick half
before adding them; the production detector normalises within each chain, which
is the non-transductive path and the only one a per-structure detector is
entitled to. Measured on the training pick half the difference is -0.0005, and
this probe uses the per-chain path.

The construction was found by trying about thirty variants on one half-split,
where it measured +0.0132. Over fourteen splits it measures +0.0087. The gap
between those is what selecting on a single split is worth, and it is reported
in the selection artifact rather than quietly replaced.

Usage: PYTHONPATH=src:tools python3.12 tools/counterattack_quotient_probe.py
"""
from __future__ import annotations

import json

import numpy as np

from pocket_bench.methods.quotient_tables import (
    compile_cells, dense_address, n_orbits, orbit_address, read_cells,
)
from pocket_bench.paths import ROOT

from counterattack_test_probe import paired_bootstrap, per_unit_metrics
from select_architecture_on_train import (
    RADII, _unit, chain_levels, patch_mean, pooled_auc,
)

TRAIN = ROOT / "data/cryptobench_apo/_cascade_cache_train.npz"
TEST = ROOT / "data/cryptobench_apo/_cascade_cache_test.npz"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
SELECTION = ROOT / "results/architecture_sweep/COUNTERATTACK_QUOTIENT.json"
OUT = ROOT / "results/official_fold/COUNTERATTACK_QUOTIENT_PROBE.json"

STACK = (8, 6, 4)          # the quotient levels the training fold named
GROUP = 6                  # invariants per thematic group, and the S_d being quotiented
READ_INDEX = 4


def thematic(M: int) -> list[list[int]]:
    return [list(range(i, min(i + GROUP, M))) for i in range(0, M, GROUP)]


def score_per_chain(S: np.ndarray, ctr: np.ndarray,
                    n_res_per: np.ndarray) -> np.ndarray:
    """Gate and normalise inside each chain, so no structure sees another."""
    out = np.empty(len(S), dtype=np.float64)
    off = 0
    for n in n_res_per:
        n = int(n)
        s, c = S[off:off + n], ctr[off:off + n]
        one = np.array([n])
        g = np.zeros(n, dtype=np.float64)
        for r in RADII:
            g = g + _unit(patch_mean(s, c, one, r))
        out[off:off + n] = _unit(s) + _unit(g)
        off += n
    return out


def bank(Ftr, ytr, ntr, Fte, nte, groups, levels, rate, *, symmetric: bool):
    """Compile one bank on the full training fold and read it on the test fold."""
    address = orbit_address if symmetric else dense_address
    fr, gini, occupied = [], [], []
    for L in levels:
        Dtr = chain_levels(Ftr, ntr, L)
        Dte = chain_levels(Fte, nte, L)
        for cols in groups:
            a_tr = address(Dtr, cols, L)
            addrs, pos, tot = compile_cells(a_tr, ytr)
            fr.append(read_cells(addrs, pos, tot, address(Dte, cols, L), rate))
            gini.append(abs(2.0 * pooled_auc(
                read_cells(addrs, pos, tot, a_tr, rate), ytr) - 1.0))
            occupied.append(int(len(addrs)))
    order = np.argsort(np.asarray(gini))
    mult = np.empty(len(fr), dtype=np.int64)
    mult[order] = np.arange(1, len(fr) + 1)
    return np.sum([m * f for m, f in zip(mult, fr)], axis=0), occupied


def _mean_auc(rows) -> float:
    return float(np.mean([r["residue_auc"] for r in rows
                          if r["residue_auc"] is not None]))


def main() -> int:
    ztr = np.load(TRAIN, allow_pickle=False)
    zte = np.load(TEST, allow_pickle=False)
    Ftr, ytr, ntr = ztr["F"], ztr["y"], ztr["n_res_per"]
    Fte, yte, nte = zte["F"], zte["y"], zte["n_res_per"]
    ctr_te = zte["ctr"]
    units_te = [str(u) for u in zte["units"]]
    M = Ftr.shape[1]
    rate = float(ytr.mean())
    groups = thematic(M)
    print(f"train {len(ntr)} units / {len(ytr)} residues, "
          f"test {len(nte)} units / {len(yte)} residues, {M} invariants",
          flush=True)
    print(f"compiling on the FULL training fold, not a half: "
          f"{int(ytr.sum())} positives at base rate {rate:.4f}", flush=True)

    S, occupied = bank(Ftr, ytr, ntr, Fte, nte, groups, STACK, rate,
                       symmetric=True)
    for i, L in enumerate(STACK):
        print(f"  L={L:>2}: {len(groups)} tables, {n_orbits(GROUP, L):,} orbits "
              f"each, {occupied[i * len(groups)]:,} occupied by the training fold",
              flush=True)
    F = score_per_chain(S, ctr_te, nte)
    n_tables = len(STACK) * len(groups)
    print(f"{n_tables} tables, integer fan-out 1..{n_tables} by compiled "
          f"Gini rank, gate at {'/'.join(str(int(r)) for r in RADII)} A",
          flush=True)

    # The control, through this same code path. A negative result is only worth
    # reporting if the pipeline that produced it reproduces the number it is
    # being compared against; re-scoring a frozen detector is not a fold read.
    S_ctrl, _ = bank(Ftr, ytr, ntr, Fte, nte, groups, (4,), rate, symmetric=False)
    ctrl_rows = per_unit_metrics(score_per_chain(S_ctrl, ctr_te, nte), yte, nte,
                                 units_te)
    ctrl_auc = _mean_auc(ctrl_rows)
    print(f"reproduction check: the frozen dense bank through this same path "
          f"scores {ctrl_auc:.4f} against its telemetry 0.7668", flush=True)

    rows = per_unit_metrics(F, yte, nte, units_te)
    ours = {r["unit_id"]: r["residue_auc"] for r in rows}
    ours_pr = {r["unit_id"]: r["residue_pr_auc"] for r in rows}
    scored = [v for v in ours.values() if v is not None]
    mean_auc = float(np.mean(scored))
    mean_pr = float(np.mean([v for v in ours_pr.values() if v is not None]))
    print(f"\nquotient counting field on the official test fold: "
          f"ROC-AUC {mean_auc:.4f}, PR-AUC {mean_pr:.4f} over {len(scored)} units",
          flush=True)

    tel = json.loads(TELEMETRY.read_text())
    tel_rows = tel["rows"] if isinstance(tel, dict) and "rows" in tel else tel
    comparisons = {}
    for method in ("p2rank", "algebraic_field", "algebraic_field_linear",
                   "table_field"):
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

    sel = json.loads(SELECTION.read_text())
    verdict_af = comparisons.get("algebraic_field", {}).get("residue_auc", {})
    verdict_p2 = comparisons.get("p2rank", {}).get("residue_auc", {})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "schema": "geoaudit.counterattack_quotient_probe.v1",
        "clinical_grade": False,
        "method": "algebraic field, S_6 quotient bank",
        "dataset": "cryptobench_official_mmseqs2_10pct_test_fold",
        "is_official_mmseqs2_10pct_test_fold": True,
        "test_fold_read_index": READ_INDEX,
        "question": "does the quotient construction, which beats the dense bank "
                    "on all fourteen training splits, also beat it here, and "
                    "does it close the gap to P2Rank",
        "answer_stated_before_the_numbers": "the training fold projected +0.0087 "
                                            "against a deficit of -0.0266, so it "
                                            "was expected to improve the counting "
                                            "field and not to overtake P2Rank",
        "architecture": {
            "invariants": int(M),
            "thematic_groups": len(groups),
            "group_size": GROUP,
            "symmetry": f"S_{GROUP} inside each thematic group; identity across "
                        f"groups (a Young subgroup of S_{M})",
            "quotient_levels": list(STACK),
            "n_tables": n_tables,
            "orbits_per_table": {str(L): n_orbits(GROUP, L) for L in STACK},
            "dense_cells_the_same_width_would_need": {str(L): L ** GROUP
                                                      for L in STACK},
            "occupied_cells_per_table": occupied,
            "digitisation": "per-chain rank, equal-count cut, L levels",
            "address": "the digit word of the group, sorted; base-L value",
            "fusion": "integer fan-out 1..n_tables by compiled Gini rank",
            "gate": {"radii_angstrom": [float(r) for r in RADII],
                     "normalisation": "within each chain"},
            "inference": "one sort and one cell read per table, an integer "
                         "weighted sum, and five neighbourhood means",
            "fitted_parameters": "none; cells are counts and fan-out is a rank",
        },
        "selection_provenance": {
            "artifact": "results/architecture_sweep/COUNTERATTACK_QUOTIENT.json",
            "n_splits": sel["n_splits"],
            "n_candidates": sel["n_candidates"],
            "reads_test_fold": False,
            "mean_delta_vs_dense_bank_on_train": sel["selected"]["mean_delta_vs_control"],
            "worst_delta_vs_dense_bank_on_train": sel["selected"]["worst_delta_vs_control"],
            "n_train_splits_beating_dense_bank":
                sel["selected"]["n_splits_beating_control"],
        },
        "reproduction_check": {
            "why": "the headline here is that a cross-validated gain did not "
                   "transfer, and that claim is only worth anything if this "
                   "pipeline reproduces the number it is measured against",
            "what": "the frozen dense bank, compiled and scored through this "
                    "file's own code path",
            "residue_auc_mean": ctrl_auc,
            "frozen_telemetry_value": 0.7668089153970957,
            "difference": ctrl_auc - 0.7668089153970957,
            "counts_as_a_fold_read": False,
            "counting_rule": "re-scoring a frozen detector is recorded and is "
                             "not a read, provided its numbers do not move",
        },
        "normalisation_note": {
            "selection_harness": "score and gate normalised over the whole pick "
                                 "half",
            "this_probe": "normalised within each chain, which is what the "
                          "detector does at query time",
            "measured_difference_on_the_training_pick_half": -0.0005,
        },
        "n_test_units": len(units_te),
        "n_scored_units": len(scored),
        "residue_auc_mean": mean_auc,
        "residue_pr_auc_mean": mean_pr,
        "verdict": {
            "beats_the_dense_counting_field":
                bool(verdict_af.get("paired_difference", 0) > 0),
            "delta_vs_dense_counting_field":
                verdict_af.get("paired_difference"),
            "separable_from_the_dense_counting_field":
                verdict_af.get("excludes_zero"),
            "beats_p2rank": bool(verdict_p2.get("paired_difference", 0) > 0),
            "delta_vs_p2rank": verdict_p2.get("paired_difference"),
            "separable_from_p2rank": verdict_p2.get("excludes_zero"),
        },
        "paired_vs": comparisons,
        "per_structure": rows,
    }, indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
