#!/usr/bin/env python3.12
"""Why does a linear solve find +0.0038 in the asymmetry columns and the field +0.0010?

The gap this attacks
--------------------
IS_FISHER_A_CEILING.json measures a ridge Fisher solve gaining +0.0038 on 12 of 12
splits from the 129 wire-asymmetry columns, under the deployed gate. The counting
field gains -0.0007 from the same columns when the bus is widened and +0.0010 on 9
of 12 when the bank is extended instead (UNION_BANK_COUNTING_FIELD.json). So about
0.0028 of accuracy is demonstrably present in those columns and the field does not
collect it. That is the largest identified gap left in this construction, and every
other axis has now been measured to its optimum.

One class of explanation is already excluded, by arithmetic rather than by
experiment. The solve is *linear*, so whatever it extracts is a linear function of
the raw columns; it cannot be an interaction the field is failing to see. Bank
dilution is excluded too, since the union attachment holds the old pairings exactly
and still only reaches +0.0010.

The hypothesis here is variance, not representation
---------------------------------------------------
A counting field can represent a monotone linear signal: quartile cells estimate
P(y | levels) and the fan-out weights them. What it cannot do is estimate those
frequencies for free. Under the deployed union attachment the 129 new columns bring
about 1,024 tables at 16 cells each -- roughly 16,000 frequencies estimated from a
fit half -- to carry a signal a linear solve carries in 129 coefficients. For a
weak signal that is a bad exchange rate, and the field would show exactly what it
shows: a small positive that does not clear the noise floor.

If that is the mechanism, the new block wants *fewer* estimated quantities, not
more. So the arms form a ladder in the estimation cost of the appended block, with
its width and its number of partition rounds varied while everything else is held:

  width 2, 16 rounds  the deployed union attachment. Reference.
  width 1, 16 rounds  a per-column lookup, 4 cells a table. No interaction among
                      the new columns, a quarter of the cells.
  width 1, 4 rounds   the cheapest arm the construction admits.
  width 2, 4 rounds   fewer tables at the deployed width, which separates "fewer
                      cells per table" from "fewer tables".
  width 3, 16 rounds  the other direction, as a control. If the ladder is real
                      this should be the worst arm.

Note that width 1 for the *whole* bank was measured at -0.0053 in TABLE_WIDTH.json,
so this is not a proposal that narrow tables are better in general. The claim under
test is narrower: for a weak appended block, where there is little interaction to
capture and the frequencies are noisy, the trade runs the other way.

What is held fixed
------------------
The 645 deployed wires keep their bank exactly -- same seed, same width, same
rounds, so the same 5,152 tables. The columns are the same 129 the two artifacts
above used, built by the same loader. The gate, the ridge, the cap, the banding
rule and the splits are the deployed ones. Only the geometry of the appended block
moves.

Two things cannot be held and are stated rather than hidden. Fan-out is a solve
over the whole bank, so adding tables reshuffles the multiplicities of the 5,152 --
that is a property of the construction, not a choice here. And an arm with fewer
new tables is also an arm with a smaller total bank, which BANK_TRUNCATION.json
shows is smooth in table count; the effect there is far smaller than the gap under
attack but it is not zero.

Falsification
-------------
If no geometry lifts the field materially above the +0.0010 the deployed union
attachment already reaches, the field's failure on appended columns is not the
estimation cost of the new block, and the remaining explanations are the banding of
those columns and the fan-out's treatment of a small weakly-correlated sub-bank.
The tool prints the Fisher figure beside the arms so the target is visible.

Reproduction gate: the width-2, 16-round arm is the deployed union attachment and
must match UNION_BANK_COUNTING_FIELD.json on the same splits.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from anisotropic_expansion_ceiling import build_or_load  # noqa: E402
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

SCHEMA = "geoaudit.appended_block_geometry.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
UNION = ROOT / "results/architecture_sweep/UNION_BANK_COUNTING_FIELD.json"
FISHER = ROOT / "results/architecture_sweep/IS_FISHER_A_CEILING.json"
OUT = ROOT / "results/architecture_sweep/APPENDED_BLOCK_GEOMETRY.json"

# (width, rounds) for the appended block. The first is the deployed union
# attachment and is the reproduction target.
GEOMETRIES = ((2, 16), (1, 16), (1, 4), (2, 4), (3, 16))


def block_tables(n_old: int, n_new: int, width: int, rounds: int):
    """The 5,152 deployed tables, plus a bank over the new columns at (width, rounds).

    Width 1 is built here because ``partition_tables`` ends with a filter dropping
    any group of fewer than two wires, so it returns nothing at width 1. That
    filter is also why 645 wires give 322 pairs a round rather than 322 and a
    singleton. ``table_bank.py`` is digest-pinned by TABLE_FIELD.json and is not
    edited for a control arm.
    """
    old = partition_tables(n_old, TABLE_WIDTH, PARTITION_ROUNDS, PARTITION_SEED)
    if width >= 2:
        new = partition_tables(n_new, width, rounds, PARTITION_SEED)
    else:
        rng = np.random.default_rng(PARTITION_SEED)
        new = [[int(c)] for _ in range(rounds) for c in rng.permutation(n_new)]
    shifted = [[c + n_old for c in t] for t in new]
    n_cells_new = int(sum(4 ** len(t) for t in shifted))
    return old + shifted, {
        "width": width,
        "rounds": rounds,
        "n_tables_old": len(old),
        "n_tables_new": len(shifted),
        "n_tables_total": len(old) + len(shifted),
        "n_cells_new_block": n_cells_new,
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

    # UNION_BANK_COUNTING_FIELD.json records means and extremes but no per-split
    # list, so the reference arm is gated on the mean, the min and the max rather
    # than split by split. Weaker than the usual gate, and said so instead of
    # being dressed up: a mean can match while the splits do not.
    union_summary = None
    if UNION.is_file():
        for name, v in (json.loads(UNION.read_text()).get("arms") or {}).items():
            if name.startswith("union bank"):
                union_summary = {"arm_in_that_artifact": name, **v}
    fisher_lift = None
    if FISHER.is_file():
        fisher_lift = (json.loads(FISHER.read_text())
                       .get("fisher_lift_from_anisotropy_under_the_deployed_gate"))

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    A, _diag, _n = build_or_load(8)
    n_old, n_new = int(W.shape[1]), int(A.shape[1])
    if n_old not in frozen:
        raise SystemExit(f"frozen artifact reports widths {sorted(frozen)}")
    narrow = frozen[n_old][:n_splits]

    built = {}
    for width, rounds in GEOMETRIES:
        tabs, meta = block_tables(n_old, n_new, width, rounds)
        built[f"width {width}, {rounds} rounds"] = (tabs, cell_offsets(tabs), meta)
    ref = f"width {TABLE_WIDTH}, {PARTITION_ROUNDS} rounds"

    t0 = time.perf_counter()
    D = chain_digits(np.asarray(np.concatenate([W, A], axis=1),
                                dtype=np.float64), n_res)
    print(f"banded {n_old} + {n_new} columns in {time.perf_counter() - t0:.0f}s")
    if fisher_lift:
        print(f"target: a linear solve finds {fisher_lift['mean']:+.4f} in these "
              f"columns on {fisher_lift['n_splits_positive']}/"
              f"{fisher_lift['n_splits']} splits")
    for k, (_t, _o, m) in built.items():
        print(f"  {k:20s} {m['n_tables_new']:5d} new tables, "
              f"{m['n_cells_new_block']:7d} new cells", flush=True)

    row = np.repeat(np.arange(len(n_res)), n_res)
    got: dict[str, list[float]] = {k: [] for k in built}
    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        t1 = time.perf_counter()
        for k, (tabs, offs, _m) in built.items():
            frac, _tot = compile_cells(D[fit], y[fit], tabs, offs)
            mult = integer_fanout(D[fit], y[fit], tabs, offs, frac, RIDGE,
                                  FAN_OUT_CAP)
            sc = apply_gate(score(D[pick], tabs, offs, frac, mult),
                            ctr[pick], n_pick)
            got[k].append(float(per_unit_auc(sc, y[pick], n_pick)))
        print(f"  split {s + 1}/{n_splits}  narrow {narrow[s]:.4f}  "
              f"{time.perf_counter() - t1:.0f}s", flush=True)
        for k in built:
            print(f"      {k:20s} {got[k][-1]:.4f}  "
                  f"{got[k][-1] - narrow[s]:+.4f}", flush=True)

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

    repro = None
    if union_summary is not None and n_splits == 12:
        g = summarise(got[ref])
        gaps = {k: abs(g[k] - union_summary[k]) for k in ("mean", "min", "max")}
        repro = {
            "arm": ref,
            "frozen_source": str(UNION.relative_to(ROOT)),
            "gated_on": "mean, min and max over the 12 splits",
            "why_not_split_by_split": "that artifact records no per-split list. A "
                                      "mean can agree while the splits do not, so "
                                      "this gate is weaker than the ones "
                                      "elsewhere in this repository and is "
                                      "labelled as such rather than implied to be "
                                      "equivalent",
            "recomputed": g,
            "frozen": {k: union_summary[k] for k in ("mean", "min", "max")},
            "absolute_differences": {k: round(v, 8) for k, v in gaps.items()},
            "reproduces_the_deployed_union_arm": bool(max(gaps.values()) < 2e-6),
        }

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "whether the counting field's failure to collect the "
                    "asymmetry columns is the number of frequencies it has to "
                    "estimate for them, rather than what it can represent",
        "the_gap_under_attack": {
            "a_linear_solve_finds": fisher_lift,
            "the_field_finds_when_the_bus_is_widened": "-0.0007, on 5 of 12",
            "the_field_finds_under_the_union_attachment": "+0.0010, on 9 of 12",
            "source": [str(FISHER.relative_to(ROOT)),
                       str(UNION.relative_to(ROOT))],
        },
        "what_is_already_excluded": {
            "an_interaction_the_field_misses": "the solve is linear, so what it "
                                               "extracts is a linear function of "
                                               "the raw columns and cannot be an "
                                               "interaction",
            "bank_dilution": "the union attachment holds the 5,152 deployed "
                             "pairings exactly and still reaches only +0.0010",
        },
        "hypothesis": "for a weak appended block the field estimates about "
                      "16,000 cell frequencies to carry what a solve carries in "
                      f"{n_new} coefficients, and a bad exchange rate would look "
                      "exactly like a small positive that misses the noise floor",
        "what_would_falsify_it": "no geometry lifting the field materially above "
                                 "the +0.0010 the deployed union attachment "
                                 "reaches. The surviving explanations would then "
                                 "be the banding of these columns and the "
                                 "fan-out's treatment of a small "
                                 "weakly-correlated sub-bank",
        "why_this_is_not_a_claim_that_narrow_tables_are_better":
            "width 1 over the whole bank costs -0.0053 (TABLE_WIDTH.json). The "
            "claim under test is only about a weak appended block, where there "
            "is little interaction to capture and the frequencies are noisy",
        "what_cannot_be_held_fixed": {
            "multiplicities": "fan-out solves over the whole bank, so adding "
                              "tables reshuffles the 5,152. A property of the "
                              "construction, not a choice made here",
            "total_bank_size": "an arm with fewer new tables has a smaller total "
                               "bank, and BANK_TRUNCATION.json shows accuracy is "
                               "smooth in table count. Far smaller than the gap "
                               "under attack, and not zero",
            "cells_and_tables_move_together": (
                "narrowing a table divides its cells by four and multiplies the "
                "table count by about two, so no single arm varies the cell "
                "budget alone. Two pairs separate them: width 1 against width 2 "
                "at 4 rounds holds the total bank near 5,500 while halving the "
                "new cells, and the same pair at 16 rounds halves the cells while "
                "the bank grows. If the cell budget is what matters both pairs "
                "move the same way; if the table count is, they disagree"),
        },
        "held_fixed": {
            "n_old_wires": n_old, "n_new_columns": n_new,
            "old_bank": f"{PARTITION_ROUNDS} rounds at width {TABLE_WIDTH}, seed "
                        f"{PARTITION_SEED}, unchanged in every arm",
            "gate_radius": GATE_RADIUS, "gate_weight": GATE_WEIGHT,
            "ridge": RIDGE, "fan_out_cap": FAN_OUT_CAP,
            "banding": "within-chain rank quartiles, the deployed rule, over the "
                       "concatenated columns",
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "compile_and_solve_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC, gate applied as deployed",
            "geometries": [list(g) for g in GEOMETRIES],
        },
        "reference_arm": ref,
        "banks": {k: m for k, (_t, _o, m) in built.items()},
        "arms": {k: summarise(v) for k, v in got.items()},
        "minus_narrow": {k: compare(v, narrow) for k, v in got.items()},
        "minus_the_deployed_union_attachment": {
            k: compare(v, got[ref]) for k, v in got.items()},
        "reproduction_check": repro,
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

    print(f"\n  narrow 645-wire field (frozen): {narrow.mean():.6f}")
    if fisher_lift:
        print(f"  a linear solve gains {fisher_lift['mean']:+.4f} from these "
              f"columns on {fisher_lift['n_splits_positive']}/"
              f"{fisher_lift['n_splits']}")
    for k in built:
        c = doc["minus_narrow"][k]
        print(f"  {k:20s} {doc['arms'][k]['mean']:.6f}  {c['mean']:+.6f}  on "
              f"{c['n_splits_positive']}/{c['n_splits']}   "
              f"{built[k][2]['n_cells_new_block']:7d} new cells")
    if repro:
        worst = max(repro["absolute_differences"].values())
        print(f"\n  reproduces the deployed union arm: "
              f"{repro['reproduces_the_deployed_union_arm']} "
              f"(worst of mean/min/max: {worst:.2e})")
    else:
        print("\n  no per-split union numbers found to reproduce against; the "
              "reference arm is unverified")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
