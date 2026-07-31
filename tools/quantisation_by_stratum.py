#!/usr/bin/env python3.12
"""The quantisation ladder again, scored per unit and split by pocket size.

Why it is being re-run
----------------------
``QUANTISATION_LADDER.json`` swept five cut ladders and found the deployed
within-chain quartiles at their optimum, with finer tails monotonically worse by
-0.0041 to -0.0069. That result was scored by a mean over units, and
``FAILURE_TAIL.json`` has since shown what a mean over units hides here: the
deployed field scores 0.5991 on the 188 training units with fewer than ten
cryptic residues and 0.8766 on the 201 with more than twenty-two. A cut that
helps the bottom stratum and hurts the top cancels inside a single mean, and
would have been recorded as "monotonically worse".

The mechanism that predicts exactly that
----------------------------------------
Every wire is cut at within-chain quartiles, so the extreme level is the extreme
25 % of the chain regardless of how much of the chain is cryptic. In a
230-residue unit with 8 cryptic residues that level holds about 57 residues for
8 positives, a sevenfold dilution. In a unit from the top stratum the same 57
residues hold 40 positives. The level says one frequency for both, and no
integer multiplicity recovers the difference -- which is the argument
``AGENT_MEMORY`` 3 already makes for the ladder being open, made per stratum
instead of overall.

So the prediction, committed before this runs: finer tails should help the
small-pocket stratum and hurt the large-pocket one, and the deployed ladder
should look best only in the pooled mean. If instead finer tails hurt every
stratum, the ladder is closed for a second time and by a stronger measurement,
and the small-pocket deficit is not a resolution problem.

What this is not
----------------
Not a fix. The stratifying variable is the number of cryptic residues, which is
the label; a detector cannot condition on it. If a cut helps the bottom stratum
the next question is what observable stands in for the stratum, and this tool
does not answer that. It answers whether there is anything there to reach.

Nothing here reads the test fold or any external unit.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import digit_cache  # noqa: E402
from expand_invariant_bank import SEED  # noqa: E402
from failure_tail import auc_per_unit  # noqa: E402
from quantisation_ladder import (  # noqa: E402
    LADDERS,
    chain_digits_at,
    compile_at,
    fanout_at,
    offsets_at,
    score_at,
)
from select_architecture_on_train import cluster_half_split  # noqa: E402

from pocket_bench.methods.table_bank import partition_tables
from pocket_bench.methods.table_field import (
    PARTITION_ROUNDS,
    PARTITION_SEED,
    TABLE_WIDTH,
    apply_gate,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.quantisation_by_stratum.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
TAIL = ROOT / "results/architecture_sweep/FAILURE_TAIL.json"
OUT = ROOT / "results/architecture_sweep/QUANTISATION_BY_STRATUM.json"

PREDICTION = {
    "committed_before_the_run": True,
    "expected": "finer tails help the small-pocket stratum and hurt the "
                "large-pocket one, so the deployed ladder is best only in the "
                "pooled mean",
    "would_close_the_axis": "finer tails hurting every stratum, which would "
                            "mean the small-pocket deficit is not a resolution "
                            "problem and the ladder is shut for a second time "
                            "by a stronger measurement",
    "reseed_floor": 0.0026,
    "note": "the strata are defined by the label and nothing conditioning on "
            "them can ship; this measures whether there is anything to reach",
}


def strata_edges() -> list[int]:
    """The quartile edges FAILURE_TAIL.json used, read rather than retyped."""
    if not TAIL.is_file():
        raise SystemExit(f"{TAIL.relative_to(ROOT)} is missing; the strata are "
                         f"defined there and are not redefined here")
    st = json.loads(TAIL.read_text())["stratified_by_positive_count"]
    return [int(s["n_cryptic_from"]) for s in st] + [
        int(st[-1]["n_cryptic_below"])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=0)
    ap.add_argument("--arms", type=str,
                    default="uniform quartiles (deployed)|tails at 5 %|"
                            "tails at 2 %",
                    help="pipe-separated, because the arm names contain commas "
                         "in neither direction but do contain spaces and %")
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    wanted = [s.strip() for s in a.arms.split("|") if s.strip()]
    unknown = [w for w in wanted if w not in LADDERS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; known: {list(LADDERS)}")

    cdoc = json.loads(COUNTING.read_text())
    n_splits = a.splits or int(cdoc["protocol"]["n_splits"])

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    z.close()
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}

    n_wires = int(W.shape[1])
    tabs = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                            PARTITION_SEED)
    row = np.repeat(np.arange(len(n_res)), n_res)

    n_pos = np.zeros(len(n_res), dtype=np.int64)
    off = 0
    for i, n in enumerate(n_res):
        n = int(n)
        n_pos[i] = int((y[off:off + n] == 1).sum())
        off += n
    edges = strata_edges()

    # The deployed arm is recomputed rather than quoted, and required to
    # reproduce the frozen per-split numbers before any other arm is believed.
    # This is the rule that caught a reimplementation reporting -0.0096 on a
    # live axis purely because it did not standardise.
    # The artifact keys its per-split lists by a sentence, "counting field over
    # 645 wires", so the width is parsed rather than formatted into a guess at
    # the sentence. Keyed by width here so a change to the wording cannot
    # silently select the wrong bus.
    by_width = {int(k.split()[-2]): v for k, v in cdoc["per_split"].items()}
    if n_wires not in by_width:
        raise SystemExit(f"the frozen artifact reports widths "
                         f"{sorted(by_width)}, not {n_wires}")
    frozen = np.asarray(by_width[n_wires], dtype=float)[:n_splits]

    results: dict[str, dict] = {}
    per_split_mean: dict[str, list[float]] = {k: [] for k in wanted}
    t0 = time.perf_counter()
    for name in wanted:
        cuts = LADDERS[name]
        n_levels = len(cuts) + 1
        D = chain_digits_at(W, n_res, cuts)
        offs = offsets_at(tabs, n_levels)
        seen = np.zeros(len(n_res), dtype=np.int64)
        total = np.zeros(len(n_res), dtype=np.float64)
        for s in range(n_splits):
            is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
            fit, pick = is_fit[row], ~is_fit[row]
            n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
            frac, _t = compile_at(D[fit], y[fit], tabs, offs, n_levels)
            mult = fanout_at(D[fit], y[fit], tabs, offs, frac, n_levels)
            sc = apply_gate(score_at(D[pick], tabs, offs, frac, mult, n_levels),
                            ctr[pick], n_pick)
            per = auc_per_unit(sc, y[pick], n_pick)
            idx = np.flatnonzero(~is_fit)
            ok = ~np.isnan(per)
            seen[idx[ok]] += 1
            total[idx[ok]] += per[ok]
            per_split_mean[name].append(float(np.nanmean(per)))
            print(f"  {name:34s} split {s + 1}/{n_splits}  "
                  f"{np.nanmean(per):.4f}  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
        del D
        scored = seen > 0
        mean_auc = np.full(len(n_res), np.nan)
        mean_auc[scored] = total[scored] / seen[scored]
        by = []
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            m = scored & (n_pos >= lo_e) & (n_pos < hi_e)
            if m.sum() < 10:
                continue
            v = mean_auc[m]
            by.append({
                "n_cryptic_from": int(lo_e),
                "n_cryptic_below": int(hi_e),
                "n_units": int(m.sum()),
                "mean_auc": round(float(np.nanmean(v)), 6),
                "median_auc": round(float(np.nanmedian(v)), 6),
                "share_below_one_half": round(float((v < 0.5).mean()), 4),
            })
        results[name] = {
            "n_levels": n_levels,
            "cuts": list(cuts),
            "pooled_mean_auc": round(float(np.nanmean(mean_auc[scored])), 6),
            "by_stratum": by,
        }

    base = wanted[0]
    deltas: dict[str, dict] = {}
    for name in wanted[1:]:
        d = {"pooled": round(results[name]["pooled_mean_auc"]
                             - results[base]["pooled_mean_auc"], 6)}
        for sb, bb in zip(results[name]["by_stratum"],
                          results[base]["by_stratum"]):
            key = f"{sb['n_cryptic_from']}-{sb['n_cryptic_below'] - 1}"
            d[key] = round(sb["mean_auc"] - bb["mean_auc"], 6)
        deltas[name] = d

    repro = float(np.abs(np.asarray(per_split_mean[base]) - frozen).max())

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "whether the quantisation cuts have a different optimum for "
                    "units with few cryptic residues than for units with many, "
                    "which a mean over units cannot show",
        "why_re_run": (
            "QUANTISATION_LADDER.json found the deployed cuts at optimum with "
            "finer tails monotonically worse, scored by a mean over units. "
            "FAILURE_TAIL.json then found the deployed field at 0.5991 on the "
            "188 units with fewer than ten cryptic residues against 0.8766 on "
            "the 201 with more than twenty-two, so a cut helping one stratum "
            "and hurting the other cancels inside that mean"),
        "prediction": PREDICTION,
        "strata_from": str(TAIL.relative_to(ROOT)),
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "compile_on": "the fit half",
            "evaluate_on": "the pick half",
            "metric": "ROC-AUC within a unit, averaged over the splits in which "
                      "the unit sat on the pick side",
            "held_fixed": f"{n_wires} wires, width {TABLE_WIDTH}, "
                          f"{PARTITION_ROUNDS} rounds, seed {PARTITION_SEED}",
        },
        "reproduction_check": {
            "recomputed_arm": base,
            "max_absolute_difference_from_frozen_per_split": round(repro, 6),
            "reproduces_frozen_arm": bool(repro < 5e-4),
            "why_this_is_here": "a tool that reimplements part of the pipeline "
                                "must return the frozen numbers before any new "
                                "number it produces is believed",
        },
        "arms": results,
        "minus_the_deployed_ladder": deltas,
        "per_split_pooled_mean": per_split_mean,
        "per_split_deployed_frozen": [round(float(x), 6) for x in frozen],
    }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    if a.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    print(f"\n  reproduces the frozen deployed arm to "
          f"{repro:.2e}: {doc['reproduction_check']['reproduces_frozen_arm']}")
    print(f"\n  {'arm':34s} {'pooled':>8s}" + "".join(
        f"{s['n_cryptic_from']}-{s['n_cryptic_below'] - 1:<3d}".rjust(10)
        for s in results[base]["by_stratum"]))
    for name in wanted:
        r = results[name]
        print(f"  {name:34s} {r['pooled_mean_auc']:8.4f}" + "".join(
            f"{s['mean_auc']:10.4f}" for s in r["by_stratum"]))
    print()
    for name, d in deltas.items():
        print(f"  {name:34s} minus deployed: " + "  ".join(
            f"{k} {v:+.4f}" for k, v in d.items()))
    if a.write:
        print(f"\nwrote {out.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
