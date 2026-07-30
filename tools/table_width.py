#!/usr/bin/env python3.12
"""Does a three-wire table buy anything a two-wire bank cannot express?

The question
------------
Seven readout parameters have now been varied and seven came back null: the
quantisation cut points, the table pairings, the composition wires, the scatter
conditioning, the bank size, the per-region multiplicities and the per-region
gate weight. Every one of them moved something downstream of the table topology.
The topology itself has never moved. ``TABLE_WIDTH`` is 2, so every cell in the
bank is a joint state of two quantised wires and the field is a sum of pairwise
terms. A width-3 table addresses 4^3 = 64 cells and represents a genuine
three-way interaction -- something no arrangement of pairwise tables can express,
by the same argument that lets the counting field beat the Fisher solve in
IS_FISHER_A_CEILING.json: a counting field is linear in the indicators of table
cells, not in the wires, so widening the table enlarges the function class rather
than the parameter count alone.

``partition_tables`` already takes ``width`` as an argument and ``addresses_at``
already loops over a table's columns, so this needs no edit to a digest-pinned
file. That is worth stating because ``table_bank.py`` is one of the eight files
``TABLE_FIELD.json`` hashes.

What it costs, and why that is the whole experiment
---------------------------------------------------
Cells get four times sparser. At width 2 the deployed bank holds about 82,000
cells over roughly 118,000 fit residues at a 5.6% positive rate, so a typical
cell states a frequency from a few hundred positives. Widen to 3 at the same
number of rounds and each cell holds a quarter as much. The quantisation ladder
already measured this trade on a different axis -- more resolution against fewer
counts -- and found the deployed cut at its optimum. So the honest prior is that
width 3 buys expressiveness and pays in counting noise, and the experiment is
about which is larger.

Why no single arm isolates the width
------------------------------------
Three quantities compete for two degrees of freedom. Fixing the width and the
number of rounds fixes the cell budget; fixing the width and the cell budget
fixes the rounds. So an arm matched on cells has fewer rounds and covers each
wire fewer times, and an arm matched on rounds has a larger cell budget. Neither
is "the" control, and reporting one alone would attribute to width what belongs
to capacity or to coverage. Both are run:

  matched cells   same number of counting cells, so the same number of fitted
                  quantities, arranged as few wide tables instead of many narrow
                  ones. This is the topology question with capacity held down.
  matched rounds  every wire covered the same number of times, letting the cell
                  budget grow. This is the topology question with coverage held
                  down.
  matched tables  same K, so the fan-out solve has the same dimension.

Width 1 is included as the floor. A one-wire table is a per-wire lookup with no
interaction at all, so the gap from width 1 to width 2 is what pairing is worth,
and it sets the scale against which any width-3 gain should be read.

Falsification
-------------
If width 3 fails to beat width 2 on both the matched-cells and the matched-rounds
arm, the table topology is not the constraint either, and the readout is finished
as a source of accuracy. If it wins on one and loses on the other, the answer is
about which resource binds and the tool says which.

Reproduction gate: the width-2, 16-round arm is the deployed bank and must return
the frozen per-split numbers in ANISOTROPIC_COUNTING_FIELD.json. An arm that
cannot reproduce the thing that ships is measuring something else.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from expand_invariant_bank import SEED  # noqa: E402
from quantisation_ladder import (  # noqa: E402
    cell_occupancy,
    compile_at,
    fanout_at,
    offsets_at,
    score_at,
)
from select_architecture_on_train import cluster_half_split, per_unit_auc  # noqa: E402

from pocket_bench.methods.table_bank import (
    N_LEVELS,
    chain_digits,
    partition_tables,
)
from pocket_bench.methods.table_field import (
    GATE_RADIUS,
    GATE_WEIGHT,
    PARTITION_ROUNDS,
    PARTITION_SEED,
    TABLE_WIDTH,
    apply_gate,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.table_width.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
OUT = ROOT / "results/architecture_sweep/TABLE_WIDTH.json"
WIDTHS = (1, 2, 3, 4)
# A fan-out solve is K x K and dense. Beyond this the arm would cost more than
# the question is worth, and the tool drops it by name rather than quietly
# running for an hour.
MAX_TABLES = 6000


def build_tables(n_wires: int, width: int, rounds: int) -> list[list[int]]:
    """``partition_tables``, except that width 1 is built here.

    The library function ends with ``[t for t in tables if len(t) >= 2]``, which
    is why the deployed bank has 322 pairs per round from 645 wires rather than
    322 pairs and a singleton, and which makes it return nothing at all at width
    1. ``table_bank.py`` is one of the eight files ``TABLE_FIELD.json`` hashes, so
    it is not edited to accommodate a control arm. The construction below is the
    same permutation-into-groups with the same seed; the only difference is the
    filter it cannot apply. Every width of 2 or more goes through the library so
    that the deployed arm is the deployed code.
    """
    if width >= 2:
        return partition_tables(n_wires, width, rounds, PARTITION_SEED)
    rng = np.random.default_rng(PARTITION_SEED)
    out: list[list[int]] = []
    for _ in range(rounds):
        out += [[int(c)] for c in rng.permutation(n_wires)]
    return out


def cells_for(n_wires: int, width: int, rounds: int) -> tuple[int, int]:
    """(number of tables, number of cells) without building anything large."""
    tabs = build_tables(n_wires, width, rounds)
    return len(tabs), int(sum(N_LEVELS ** len(t) for t in tabs))


def rounds_matching(n_wires: int, width: int, target: int, key: str) -> int:
    """Rounds whose table count or cell count lands nearest ``target``.

    Searched rather than solved, because ``partition_tables`` leaves a short
    group when the width does not divide the wire count and the closed form
    would be wrong by that group.
    """
    best, best_gap = 1, None
    for r in range(1, 4 * PARTITION_ROUNDS + 1):
        tabs, cells = cells_for(n_wires, width, r)
        got = tabs if key == "tables" else cells
        gap = abs(got - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = r, gap
    return best


def arm_auc(D, y, ctr, n_res, fit, pick, n_pick, tables, offsets):
    """Compile on the fit half, score the pick half, return AUC and occupancy."""
    frac, tot = compile_at(D[fit], y[fit], tables, offsets, N_LEVELS)
    mult = fanout_at(D[fit], y[fit], tables, offsets, frac, N_LEVELS)
    raw = score_at(D[pick], tables, offsets, frac, mult, N_LEVELS)
    gated = apply_gate(raw, ctr[pick], n_pick)
    return float(per_unit_auc(gated, y[pick], n_pick)), cell_occupancy(tot)


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
    n_wires = int(W.shape[1])
    if n_wires not in frozen:
        raise SystemExit(f"frozen artifact has widths {sorted(frozen)}")
    base = frozen[n_wires][:n_splits]

    dep_tabs, dep_cells = cells_for(n_wires, TABLE_WIDTH, PARTITION_ROUNDS)

    # Declared before any arm runs, so the matching rule is auditable and no arm
    # is added after seeing a number.
    arms: dict[str, dict] = {}
    for w in WIDTHS:
        for kind, key, target in (("matched rounds", None, None),
                                  ("matched cells", "cells", dep_cells),
                                  ("matched tables", "tables", dep_tabs)):
            r = PARTITION_ROUNDS if key is None else rounds_matching(
                n_wires, w, target, key)
            tabs, cells = cells_for(n_wires, w, r)
            name = f"width {w}, {kind}"
            if any(v["rounds"] == r and v["width"] == w for v in arms.values()):
                continue
            arms[name] = {"width": w, "rounds": r, "n_tables": tabs,
                          "n_cells": cells}

    dropped = {k: v for k, v in arms.items() if v["n_tables"] > MAX_TABLES}
    arms = {k: v for k, v in arms.items() if v["n_tables"] <= MAX_TABLES}
    deployed = f"width {TABLE_WIDTH}, matched rounds"
    if deployed not in arms:
        raise SystemExit("the deployed arm was dropped; the harness is wrong")

    t0 = time.perf_counter()
    D = chain_digits(np.asarray(W, dtype=np.float64), n_res)
    print(f"banded {n_wires} wires in {time.perf_counter() - t0:.0f}s\n"
          f"deployed: width {TABLE_WIDTH}, {PARTITION_ROUNDS} rounds, "
          f"{dep_tabs} tables, {dep_cells} cells", flush=True)
    for k, v in arms.items():
        print(f"  {k:28s} r={v['rounds']:2d}  K={v['n_tables']:5d}  "
              f"cells={v['n_cells']:7d}", flush=True)
    for k, v in dropped.items():
        print(f"  {k:28s} dropped: K={v['n_tables']} over {MAX_TABLES}",
              flush=True)

    built = {k: build_tables(n_wires, v["width"], v["rounds"])
             for k, v in arms.items()}
    offs = {k: offsets_at(t, N_LEVELS) for k, t in built.items()}
    row = np.repeat(np.arange(len(n_res)), n_res)
    got: dict[str, list[float]] = {k: [] for k in arms}
    occ: dict[str, dict] = {}

    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        t1 = time.perf_counter()
        for k in arms:
            auc, o = arm_auc(D, y, ctr, n_res, fit, pick, n_pick,
                             built[k], offs[k])
            got[k].append(auc)
            occ.setdefault(k, o)
        print(f"  split {s + 1}/{n_splits}  "
              f"deployed {got[deployed][-1]:.4f}  frozen {base[s]:.4f}  "
              f"{time.perf_counter() - t1:.0f}s", flush=True)
        for k in arms:
            if k != deployed:
                print(f"      {k:28s} {got[k][-1]:.4f}  "
                      f"{got[k][-1] - got[deployed][-1]:+.4f}", flush=True)

    def summarise(v):
        v = np.asarray(v, dtype=float)
        return {"mean": round(float(v.mean()), 6),
                "min": round(float(v.min()), 6),
                "max": round(float(v.max()), 6)}

    def compare(v, ref):
        d = np.asarray(v, dtype=float) - np.asarray(ref, dtype=float)
        return {"mean": round(float(d.mean()), 6),
                "n_splits_positive": int((d > 0).sum()),
                "n_splits": int(len(d))}

    dep = np.asarray(got[deployed], dtype=float)
    repro = {
        "arm": deployed,
        "frozen_source": str(COUNTING.relative_to(ROOT)),
        "max_absolute_difference": float(np.max(np.abs(dep - base))),
        "reproduces_the_deployed_arm": bool(np.allclose(dep, base, atol=2e-6)),
    }

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "whether a three-wire table expresses something a bank of "
                    "two-wire tables cannot, and whether that is worth the "
                    "four-fold thinning of every cell it causes",
        "why_this_is_the_named_next_experiment":
            "seven readout parameters have been varied and all seven came back "
            "null, and every one of them was downstream of the table topology. "
            "TABLE_WIDTH has never moved. partition_tables already takes width "
            "as an argument, so no digest-pinned file is edited",
        "why_width_could_help": "a counting field is linear in the indicators of "
                                "table cells, not in the wires. A width-3 table "
                                "represents a three-way interaction that no "
                                "arrangement of pairwise tables can express -- "
                                "the same argument by which the counting field "
                                "beats the Fisher solve in IS_FISHER_A_CEILING",
        "why_width_could_hurt": "cells thin by a factor of four per unit of "
                                "width at fixed rounds, and a cell that holds "
                                "few residues states a noisy frequency. "
                                "QUANTISATION_LADDER measured the same trade on "
                                "the cut points and found the deployed setting "
                                "at its optimum",
        "why_no_single_arm_isolates_the_width":
            "width, rounds and cell budget are three quantities over two degrees "
            "of freedom. Matching cells costs rounds and so costs coverage of "
            "each wire; matching rounds lets the cell budget grow. Both are run "
            "and neither alone is the control",
        "matching_rules": {
            "matched rounds": f"{PARTITION_ROUNDS} rounds, as deployed",
            "matched cells": f"rounds chosen so the cell count is nearest "
                             f"{dep_cells}",
            "matched tables": f"rounds chosen so the table count is nearest "
                              f"{dep_tabs}",
            "search": "rounds searched rather than solved, because a width that "
                      "does not divide the wire count leaves a short group and "
                      "the closed form would be wrong by that group",
        },
        "what_would_falsify_it": "width 3 failing to beat width 2 on both the "
                                 "matched-cells and the matched-rounds arm. That "
                                 "would say the table topology is not the "
                                 "constraint either and the readout is finished "
                                 "as a source of accuracy",
        "held_fixed": {
            "n_wires": n_wires,
            "n_levels": N_LEVELS,
            "partition_seed": PARTITION_SEED,
            "gate_radius": GATE_RADIUS,
            "gate_weight": GATE_WEIGHT,
            "banding": "within-chain rank quartiles, the deployed rule; the "
                       "banding is per wire and does not depend on the topology, "
                       "so it is computed once and shared by every arm",
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "compile_and_solve_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC, gate applied as deployed",
            "max_tables": MAX_TABLES,
            "arms_dropped_for_cost": {k: v["n_tables"] for k, v in dropped.items()},
        },
        "deployed_arm": deployed,
        "arms": {k: {**arms[k], **summarise(v)} for k, v in got.items()},
        "minus_deployed": {k: compare(v, got[deployed]) for k, v in got.items()},
        "cell_occupancy_first_split": occ,
        "reproduction_check": repro,
        "per_split": {k: [round(x, 6) for x in v] for k, v in got.items()},
        "per_split_deployed_frozen": [round(float(x), 6) for x in base],
        "n_units": int(len(n_res)),
        "n_residues": int(len(y)),
        "n_positive_residues": int(y.sum()),
    }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    print(f"\n  deployed ({deployed}): {dep.mean():.6f}")
    for k in arms:
        c = doc["minus_deployed"][k]
        print(f"  {k:28s} {doc['arms'][k]['mean']:.6f}  {c['mean']:+.6f}  on "
              f"{c['n_splits_positive']}/{c['n_splits']}   "
              f"median cell {occ[k].get('median_count_of_addressed_cells')}")
    print(f"\n  deployed arm reproduces the frozen numbers: "
          f"{repro['reproduces_the_deployed_arm']} "
          f"(max |diff| {repro['max_absolute_difference']:.2e})")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
