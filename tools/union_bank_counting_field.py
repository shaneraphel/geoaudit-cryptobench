#!/usr/bin/env python3
"""Extend the bank instead of replacing it. Training folds only.

The hypothesis this tests, and where it came from
------------------------------------------------
ANISOTROPIC_COUNTING_FIELD.json and IS_FISHER_A_CEILING.json together say something
specific. The wire-asymmetry columns are not inert: a Fisher solve finds +0.0038 in
them on 12 of 12 splits, under the deployed gate. But the counting field gains
-0.0007 from the same columns, and its own advantage over the solve falls from
+0.0053 at 645 wires to +0.0008 at 774. Two effects of similar size cancelling.

Bank dilution is the mechanism that predicts that shape. Tables are random pairings
of wires drawn from one seed, so ``partition_tables(774, ...)`` is not
``partition_tables(645, ...)`` with tables appended --- it is a different bank in
which every pairing has been redrawn. The counting field's edge over a linear solve
is exactly its ability to read interactions between paired wires, so redrawing all
of them at a wider width is not a neutral act. If that is what happened, the fix is
not to widen the bus but to extend it: keep the 645-wire pairings and add tables
over the 129 new columns alone.

What is genuinely held and what is not
--------------------------------------
The pairings among the original wires are held exactly. Multiplicities are not, and
cannot be. Fan-out is the integer rank of a table's compiled Gini among the whole
bank, so adding 1,024 tables reshuffles the ranks of the 5,152 that were there.
(Both counts are a wire short of the arithmetic: a round over an odd number of wires
leaves one unpaired and it is dropped, so it is 322 and 64 tables per round rather
than 323 and 65.)
That is a property of the construction rather than a choice made here, and it means
a null result would not fully exonerate dilution --- it would leave rank
reshuffling as the surviving explanation. A positive result is the cleaner outcome.

The new columns get their own partition rounds at the same width and seed. Pairing
an asymmetry column with an isotropic wire is therefore not tried at all in this
arm, which is deliberate: it is the one difference from the 774 arm, and mixing the
two would make the comparison uninterpretable.

Comparison is against the frozen 645-wire arm rather than a recomputation of it, on
the same seeds. The tool checks the split count and the width before pairing.

Training folds only. No test residue and no external unit is read.

Usage: PYTHONPATH=src:tools python3.12 tools/union_bank_counting_field.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from anisotropic_expansion_ceiling import ASYM_RADII, build_or_load  # noqa: E402
from expand_invariant_bank import SEED  # noqa: E402
from select_architecture_on_train import cluster_half_split, per_unit_auc  # noqa: E402

from pocket_bench.methods.table_bank import (
    cell_offsets,
    chain_digits,
    compile_cells,
    integer_fanout,
    partition_tables,
    score,
)
from pocket_bench.methods.table_field import (
    FAN_OUT_CAP,
    GATE_RADIUS,
    GATE_WEIGHT,
    PARTITION_ROUNDS,
    PARTITION_SEED,
    RIDGE,
    TABLE_WIDTH,
    apply_gate,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.union_bank_counting_field.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
OUT = ROOT / "results/architecture_sweep/UNION_BANK_COUNTING_FIELD.json"


def union_tables(n_old: int, n_new: int) -> tuple[list, dict]:
    """The old bank untouched, plus a bank over the new columns alone."""
    old = partition_tables(n_old, TABLE_WIDTH, PARTITION_ROUNDS, PARTITION_SEED)
    new = partition_tables(n_new, TABLE_WIDTH, PARTITION_ROUNDS, PARTITION_SEED)
    shifted = [[c + n_old for c in t] for t in new]
    replaced = partition_tables(n_old + n_new, TABLE_WIDTH, PARTITION_ROUNDS,
                                PARTITION_SEED)
    kept = sum(1 for t in old if sorted(t) in [sorted(r) for r in replaced])
    return old + shifted, {
        "n_tables_over_the_old_wires": len(old),
        "n_tables_over_the_new_columns": len(shifted),
        "n_tables_total": len(old) + len(shifted),
        "n_tables_if_the_bus_were_widened_instead": len(replaced),
        "n_old_pairings_that_survive_widening": kept,
        "why_that_last_number_matters": (
            "it is how much of the 645-wire bank the 774-wire arm actually kept. "
            "If it is small then widening the bus rebuilt the bank rather than "
            "extending it, which is the dilution this arm is designed to avoid"),
        "no_table_mixes_old_and_new": True,
        "why_not": (
            "pairing an asymmetry column with an isotropic wire is the one thing "
            "that differs between this arm and the 774 arm. Allowing it here "
            "would make the two indistinguishable and the comparison "
            "uninterpretable"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args(argv)

    cdoc = json.loads(COUNTING.read_text())
    frozen = {int(k.split()[-2]): np.asarray(v, dtype=float)
              for k, v in cdoc["per_split"].items()}
    n_splits = a.splits or cdoc["protocol"]["n_splits"]

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    A, diag, _n = build_or_load(8)
    n_old, n_new = int(W.shape[1]), int(A.shape[1])
    if n_old not in frozen or n_old + n_new not in frozen:
        raise SystemExit(
            f"the frozen artifact reports widths {sorted(frozen)}; this tool "
            f"needs {n_old} and {n_old + n_new} to compare against")

    tables, bank = union_tables(n_old, n_new)
    offsets = cell_offsets(tables)
    print(f"{bank['n_tables_over_the_old_wires']} tables over the {n_old} old "
          f"wires + {bank['n_tables_over_the_new_columns']} over the {n_new} new "
          f"columns = {bank['n_tables_total']}")
    print(f"widening the bus instead would build "
          f"{bank['n_tables_if_the_bus_were_widened_instead']} tables and keep "
          f"{bank['n_old_pairings_that_survive_widening']} of the old pairings",
          flush=True)

    t0 = time.perf_counter()
    D = chain_digits(np.asarray(np.concatenate([W, A], axis=1),
                                dtype=np.float64), n_res)
    print(f"banded in {time.perf_counter() - t0:.0f}s", flush=True)

    got: list[float] = []
    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        row = np.repeat(np.arange(len(n_res)), n_res)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        t1 = time.perf_counter()
        frac, _tot = compile_cells(D[fit], y[fit], tables, offsets)
        mult = integer_fanout(D[fit], y[fit], tables, offsets, frac, RIDGE,
                              FAN_OUT_CAP)
        s_pick = apply_gate(score(D[pick], tables, offsets, frac, mult),
                            ctr[pick], n_pick)
        got.append(float(per_unit_auc(s_pick, y[pick], n_pick)))
        print(f"  split {s + 1}/{n_splits}  union {got[-1]:.4f}  "
              f"narrow {frozen[n_old][s]:.4f}  widened "
              f"{frozen[n_old + n_new][s]:.4f}  "
              f"{time.perf_counter() - t1:.0f}s", flush=True)

    u = np.asarray(got)
    nar, wid = frozen[n_old][:n_splits], frozen[n_old + n_new][:n_splits]

    def cmp(d):
        return {"mean": round(float(d.mean()), 6),
                "min": round(float(d.min()), 6),
                "max": round(float(d.max()), 6),
                "n_splits_positive": int((d > 0).sum()),
                "n_splits": int(len(d)),
                "positive_on_every_split": bool((d > 0).all())}

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "the asymmetry columns carry +0.0038 for a linear solve and "
                    "-0.0007 for the counting field, whose edge over that solve "
                    "falls from +0.0053 to +0.0008 when the bus is widened. If "
                    "bank dilution is why, extending the bank instead of "
                    "redrawing it should recover the lift",
        "bank": bank,
        "held_fixed": {
            "table_width": TABLE_WIDTH,
            "partition_rounds": PARTITION_ROUNDS,
            "partition_seed": PARTITION_SEED,
            "ridge": RIDGE,
            "fan_out_cap": FAN_OUT_CAP,
            "gate_radius": GATE_RADIUS,
            "gate_weight": GATE_WEIGHT,
        },
        "what_this_arm_cannot_hold_fixed": (
            "fan-out multiplicity is the integer rank of a table's compiled Gini "
            "among the whole bank, so adding tables reshuffles the ranks of the "
            "ones already there. That is the construction and not a choice. It "
            "means a null result here would not exonerate dilution: it would "
            "leave rank reshuffling as the surviving explanation, and a further "
            "arm would be needed to separate them"),
        "anisotropic_columns": {
            "n": n_new,
            "radii_angstrom": [float(r) for r in ASYM_RADII],
            "degenerate_fraction_per_radius":
                diag["fraction_degenerate_per_radius"],
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "compile_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC",
            "baselines_were_not_recomputed": str(COUNTING.relative_to(ROOT)),
        },
        "arms": {
            f"union bank over {n_old} + {n_new}": {
                "mean": round(float(u.mean()), 6),
                "min": round(float(u.min()), 6),
                "max": round(float(u.max()), 6)},
            f"counting field over {n_old} wires": {
                "mean": round(float(nar.mean()), 6)},
            f"counting field over {n_old + n_new} wires": {
                "mean": round(float(wid.mean()), 6)},
        },
        "union_minus_narrow": cmp(u - nar),
        "union_minus_widened": cmp(u - wid),
        "fisher_lift_for_reference": {
            "mean": 0.003816,
            "n_splits_positive": 12,
            "source": "results/architecture_sweep/IS_FISHER_A_CEILING.json, "
                      "deployed gate",
        },
        "n_units": len(units),
        "n_residues": int(len(W)),
        "n_positive_residues": int(y.sum()),
    }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    un, uw = doc["union_minus_narrow"], doc["union_minus_widened"]
    print(f"\n  union {u.mean():.4f}  narrow {nar.mean():.4f}  widened "
          f"{wid.mean():.4f}")
    print(f"  union - narrow  {un['mean']:+.4f} on "
          f"{un['n_splits_positive']}/{un['n_splits']}")
    print(f"  union - widened {uw['mean']:+.4f} on "
          f"{uw['n_splits_positive']}/{uw['n_splits']}")
    fis = doc["fisher_lift_for_reference"]["mean"]
    if un["mean"] <= 0:
        print(f"\n  extending the bank does not recover it either. The columns "
              f"carry {fis:+.4f} for a linear solve and nothing for this "
              f"construction, whichever way the bank is built.")
    elif un["mean"] >= fis / 2:
        print(f"\n  extending the bank recovers {un['mean'] / fis:.0%} of the "
              f"linear lift. Dilution was the obstacle, and this is the "
              f"construction to preregister a test-fold read for.")
    else:
        print(f"\n  extending the bank recovers {un['mean'] / fis:.0%} of the "
              f"linear lift: real but partial.")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
