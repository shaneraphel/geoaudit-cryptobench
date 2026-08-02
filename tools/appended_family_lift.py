#!/usr/bin/env python3.12
"""What a counting field and a linear solve each get from an appended column family.

The counterpart to the screen
-----------------------------
COLLECTABILITY_SCREEN.json measures a family's mean pairwise interaction in about
two minutes and predicts, on three families fitted after the fact, which of the two
readouts can collect it. This measures the thing being predicted, in about half an
hour, so that the screen can be tested rather than admired. The ratio of those two
costs is the only reason a screen is worth having.

Three arms, and why each
------------------------
``union``    the deployed 5,152 pairings held exactly, plus tables over the new
             columns alone. This is the attachment that treats the existing bank as
             something to extend rather than replace, and UNION_BANK_COUNTING_FIELD
             shows the difference matters: widening the bus keeps only 296 of the
             5,152 pairings, so the redraw alone was worth more than the columns.
``widened``  the bus widened and every pairing redrawn. Kept because it is what a
             reader would try first and because the gap between the two is the
             measured cost of the redraw.
``fisher``   a ridge Fisher discriminant over the concatenated raw columns. Not a
             ceiling -- IS_FISHER_A_CEILING.json settles that -- but a cheap
             correlate that separates "the information is absent" from "the
             construction does not collect it". The screen's whole claim is about
             which of those two holds, so this arm is not optional here.

The comparison is against the frozen 645-wire per-split numbers rather than a
recomputation, on the same seeds, and the tool checks the width before it starts.

Generic on purpose
------------------
It takes a family by name from a small registry rather than hard-coding one, because
the screen's value is in being applied to families that do not exist yet, and a
screen with no reusable counterpart is a screen nobody will check.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from expand_invariant_bank import SEED, fisher  # noqa: E402
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

SCHEMA = "geoaudit.appended_family_lift.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
SCREEN = ROOT / "results/architecture_sweep/COLLECTABILITY_SCREEN.json"
OUT = ROOT / "results/architecture_sweep/APPENDED_FAMILY_LIFT.json"


def load_family(name: str) -> tuple[np.ndarray, str]:
    if name == "graph invariants 225":
        from graph_invariant_wires import build_wide_or_load
        X, _n = build_wide_or_load()
        return np.asarray(X, dtype=np.float64), "tools/graph_invariant_wires.py"
    if name == "graph invariants 15":
        from graph_invariant_wires import build_or_load
        X, _n = build_or_load()
        return np.asarray(X, dtype=np.float64), "tools/graph_invariant_wires.py"
    if name == "crystal form 129":
        from crystal_form_wires import build_or_load
        X, _n = build_or_load()
        return np.asarray(X, dtype=np.float64), "tools/crystal_form_wires.py"
    if name == "crystal form 129 permuted":
        from crystal_form_wires import build_or_load
        X, _n = build_or_load(permuted=True)
        return np.asarray(X, dtype=np.float64), "tools/crystal_form_wires.py"
    raise SystemExit(f"unknown family {name!r}")


def screen_value(name: str) -> dict | None:
    if not SCREEN.is_file():
        return None
    f = (json.loads(SCREEN.read_text()).get("families") or {}).get(name)
    return None if f is None else f.get("second_order_over_all_pairs")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", default="graph invariants 225")
    ap.add_argument("--splits", type=int, default=0)
    ap.add_argument("--arms", default="union,widened,fisher,fisher_old")
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args(argv)
    wanted = [s.strip() for s in a.arms.split(",") if s.strip()]

    cdoc = json.loads(COUNTING.read_text())
    frozen = {int(k.split()[-2]): np.asarray(v, dtype=float)
              for k, v in cdoc["per_split"].items()}
    n_splits = a.splits or cdoc["protocol"]["n_splits"]

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    F, built_by = load_family(a.family)
    n_old, n_new = int(W.shape[1]), int(F.shape[1])
    if n_old not in frozen:
        raise SystemExit(f"the frozen artifact reports widths {sorted(frozen)}")
    narrow = frozen[n_old][:n_splits]

    old = partition_tables(n_old, TABLE_WIDTH, PARTITION_ROUNDS, PARTITION_SEED)
    new = partition_tables(n_new, TABLE_WIDTH, PARTITION_ROUNDS, PARTITION_SEED)
    union = old + [[c + n_old for c in t] for t in new]
    widened = partition_tables(n_old + n_new, TABLE_WIDTH, PARTITION_ROUNDS,
                               PARTITION_SEED)
    wide_set = {tuple(sorted(r)) for r in widened}
    kept = sum(1 for t in old if tuple(sorted(t)) in wide_set)
    # `more_old`: the same number of *added* tables as the union arm, drawn over
    # the deployed columns with the family absent. Without it a small positive
    # cannot be separated from bank size, and STRADDLING_ATTACHMENT found this
    # control scoring above the family it was controlling for. AGENT_MEMORY 2d.
    extra = partition_tables(n_old, TABLE_WIDTH, PARTITION_ROUNDS,
                             PARTITION_SEED + 1)[:len(new)]
    more_old = old + extra
    banks = {"union": (union, cell_offsets(union)),
             "widened": (widened, cell_offsets(widened)),
             "more_old": (more_old, cell_offsets(more_old))}

    t0 = time.perf_counter()
    Wf = np.asarray(W, dtype=np.float64)
    full = np.concatenate([Wf, F], axis=1)
    D = chain_digits(full, n_res)
    print(f"banded {n_old} + {n_new} columns in {time.perf_counter() - t0:.0f}s")
    print(f"  union {len(union)} tables ({len(new)} over the new columns), "
          f"widened {len(widened)} tables keeping {kept} of the {len(old)} old "
          f"pairings", flush=True)
    sv = screen_value(a.family)
    if sv:
        print(f"  the screen said this family's mean pairwise interaction is "
              f"{sv['mean']:+.3e}", flush=True)

    row = np.repeat(np.arange(len(n_res)), n_res)
    got: dict[str, list[float]] = {k: [] for k in wanted}
    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        t1 = time.perf_counter()
        for arm in wanted:
            if arm == "fisher":
                sc = apply_gate(fisher(full[fit], y[fit], full[pick]),
                                ctr[pick], n_pick)
            elif arm == "fisher_old":
                # The solve on the deployed columns alone. Without it the only
                # available comparison for the fisher arm is against a counting
                # field, which is a different readout, and the difference would
                # be dominated by the two readouts differing rather than by what
                # the family adds. A solve's lift has to be measured against a
                # solve.
                sc = apply_gate(fisher(Wf[fit], y[fit], Wf[pick]),
                                ctr[pick], n_pick)
            else:
                tabs, offs = banks[arm]
                frac, _t = compile_cells(D[fit], y[fit], tabs, offs)
                mult = integer_fanout(D[fit], y[fit], tabs, offs, frac, RIDGE,
                                      FAN_OUT_CAP)
                sc = apply_gate(score(D[pick], tabs, offs, frac, mult),
                                ctr[pick], n_pick)
            got[arm].append(float(per_unit_auc(sc, y[pick], n_pick)))
        print(f"  split {s + 1}/{n_splits}  narrow {narrow[s]:.4f}  "
              + "  ".join(f"{k} {got[k][-1]:.4f}" for k in wanted)
              + f"  {time.perf_counter() - t1:.0f}s", flush=True)

    def summarise(v):
        v = np.asarray(v, dtype=float)
        return {"mean": round(float(v.mean()), 6),
                "min": round(float(v.min()), 6),
                "max": round(float(v.max()), 6)}

    def compare(v, base):
        d = np.asarray(v, dtype=float) - np.asarray(base, dtype=float)
        return {"mean": round(float(d.mean()), 6),
                "n_splits_positive": int((d > 0).sum()),
                "n_splits": int(len(d))}

    minus = {k: compare(v, narrow) for k, v in got.items()}
    field = minus.get("union", {}).get("mean")
    # The solve's lift is fisher(old + new) minus fisher(old), never minus the
    # counting field: those are two readouts and their difference would swamp
    # what the family contributes.
    solve = None
    if "fisher" in got and "fisher_old" in got:
        solve = compare(got["fisher"], got["fisher_old"])["mean"]
    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "family": a.family,
        "built_by": built_by,
        "n_new_columns": n_new,
        "question": "what a counting field and a linear solve each take from this "
                    "family, so that the screen's prediction can be tested",
        "the_screen_said": sv,
        "what_the_screen_predicted": (
            "a mean pairwise interaction near the deployed bank's +1.06e-05 means "
            "the field collects the family; one near composition's +5.96e-06 or "
            "the asymmetry family's -9.69e-06 means it does not and the lift goes "
            "to the solve. The prediction was committed before this ran"),
        "held_fixed": {
            "n_old_wires": n_old,
            "old_bank": f"{PARTITION_ROUNDS} rounds at width {TABLE_WIDTH}, seed "
                        f"{PARTITION_SEED}",
            "gate_radius": GATE_RADIUS, "gate_weight": GATE_WEIGHT,
            "ridge": RIDGE, "fan_out_cap": FAN_OUT_CAP,
            "banding": "within-chain rank quartiles over the concatenated columns",
        },
        "bank": {
            "n_tables_union": len(union),
            "n_tables_over_the_new_columns": len(new),
            "n_tables_widened": len(widened),
            "n_old_pairings_surviving_widening": kept,
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "compile_and_solve_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC, gate applied as deployed",
            "baseline_was_not_recomputed": str(COUNTING.relative_to(ROOT)),
        },
        "arms": {k: summarise(v) for k, v in got.items()},
        "minus_narrow": minus,
        "solve_lift_from_the_family": (
            None if solve is None
            else compare(got["fisher"], got["fisher_old"])),
        "why_the_solve_lift_is_against_a_solve": (
            "fisher(old + new) minus fisher(old). Comparing the solve against the "
            "counting field instead would measure the two readouts differing, "
            "which on the deployed wires is +0.0053, and drown what the family "
            "adds"),
        "field_minus_solve": (None if field is None or solve is None
                              else round(field - solve, 6)),
        "per_split": {k: [round(x, 6) for x in v] for k, v in got.items()},
        "per_split_narrow_frozen": [round(float(x), 6) for x in narrow],
        "n_units": int(len(n_res)),
        "n_residues": int(len(y)),
        "n_positive_residues": int(y.sum()),
    }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    print(f"\n  narrow {n_old}-wire field (frozen): {narrow.mean():.6f}")
    for k in wanted:
        c = minus[k]
        print(f"  {k:10s} {doc['arms'][k]['mean']:.6f}  {c['mean']:+.6f}  on "
              f"{c['n_splits_positive']}/{c['n_splits']}")
    if doc["field_minus_solve"] is not None:
        print(f"\n  field minus solve: {doc['field_minus_solve']:+.6f}")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
