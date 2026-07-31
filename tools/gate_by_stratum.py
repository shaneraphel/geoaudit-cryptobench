#!/usr/bin/env python3.12
"""The spatial gate swept per pocket size, which is how it was never swept.

Why
---
``FAILURE_TAIL.json``: the deployed field scores 0.5991 on the 188 training
units with fewer than ten cryptic residues and 0.8766 on the 201 with more than
twenty-two. ``QUANTISATION_BY_STRATUM.json`` then removed the obvious
explanation -- finer cuts hurt every stratum, so the deficit is not a
resolution problem.

The gate is the next suspect and it is the same trap twice over. ``r = 18,
w = 1.0`` was chosen over {14, 18} x {0.5, 1.0} by pick-half ROC-AUC, which is a
mean over units, so a radius that suits a forty-residue site and drowns an
eight-residue one is exactly what that selection could not see. And the
mechanism is direct: the gate adds back the mean score over an 18 A
neighbourhood, rescaled to the raw score's spread. Around a site with forty
cryptic residues that neighbourhood is substantially positive. Around a site
with eight it is almost entirely negative, so the gate adds a negative
correction over the very residues it should be reinforcing.

Cheap, for once
---------------
The gate is applied after scoring, so the compile and the fan-out are done once
per split and every radius reuses them. Twelve compiles for the whole sweep
rather than twelve per arm, which is why this covers seven radii and two weights
where the quantisation sweep covered three ladders.

The prediction, committed before the run
----------------------------------------
If the gate is the mechanism, the optimum radius rises with pocket size: the
0-9 stratum should prefer a radius well below 18, and the 23-76 stratum should
prefer 18 or above. A single global radius would then be a compromise that costs
the bottom stratum most, and the size of that cost is the size of the
opportunity.

If instead every stratum prefers the same radius, the gate is not the mechanism
either, and two of the three parameters that could plausibly cause a size
dependence will have been eliminated.

The caveat that applies to any positive result here
---------------------------------------------------
The strata are defined by the number of cryptic residues, which is the label. A
per-unit radius chosen by stratum cannot ship. What could ship is a radius
chosen by something observable that predicts the stratum, and nothing in
FAILURE_TAIL.json does: chain length separates the tail at P=0.393 against
0.195 for the positive count itself. So a positive result here is a measurement
of what is available, not a method. The artifact says so in its own verdict.

Nothing here reads the test fold or any external unit.
"""
from __future__ import annotations

import argparse
import json
import time
from math import comb
from pathlib import Path

import numpy as np

import digit_cache  # noqa: E402
from expand_invariant_bank import SEED  # noqa: E402
from failure_tail import auc_per_unit  # noqa: E402
from select_architecture_on_train import cluster_half_split  # noqa: E402
from straddling_attachment import lean_integer_fanout  # noqa: E402

from pocket_bench.methods.table_bank import (
    cell_offsets,
    compile_cells,
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
    _neighbourhood_mean,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.gate_by_stratum.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
TAIL = ROOT / "results/architecture_sweep/FAILURE_TAIL.json"
OUT = ROOT / "results/architecture_sweep/GATE_BY_STRATUM.json"

RADII = (0.0, 8.0, 12.0, 14.0, 18.0, 24.0, 30.0)
WEIGHTS = (0.5, 1.0)

PREDICTION = {
    "committed_before_the_run": True,
    "if_the_gate_is_the_mechanism": "the optimum radius rises with pocket "
                                    "size; the 0-9 stratum prefers a radius "
                                    "well below 18 and the 23-76 stratum "
                                    "prefers 18 or above",
    "if_it_is_not": "every stratum prefers the same radius, eliminating the "
                    "second of three parameters that could cause a size "
                    "dependence",
    "reseed_floor": 0.0026,
    "what_a_positive_result_would_not_be": "a method. The strata are the label; "
                                           "a radius chosen by stratum cannot "
                                           "ship, and no observable in "
                                           "FAILURE_TAIL.json predicts the "
                                           "stratum well enough to stand in "
                                           "for it",
}


def gate_at(s: np.ndarray, ctr: np.ndarray, n_res_per, radius: float,
            weight: float) -> np.ndarray:
    """``apply_gate`` with the radius and weight passed in rather than pinned.

    A copy rather than an edit: ``table_field.py`` is one of the eight files
    ``TABLE_FIELD.json`` carries a code_sha256 over, so adding parameters to
    ``apply_gate`` would invalidate the compiled field for a reason unrelated
    to the field. The body is the deployed body and the deployed constants are
    reproduced exactly when ``radius`` and ``weight`` are the deployed ones,
    which the caller checks.
    """
    if radius <= 0.0:
        return np.asarray(s, dtype=np.float64)
    out = np.empty(len(s), dtype=np.float64)
    off = 0
    for n in n_res_per:
        n = int(n)
        blk = s[off:off + n]
        g = _neighbourhood_mean(blk, np.asarray(ctr[off:off + n], float),
                                radius)
        sd_s, sd_g = float(np.std(blk)), float(np.std(g))
        out[off:off + n] = (blk if sd_g <= 0
                            else blk + weight * g * (sd_s / sd_g))
        off += n
    return out


def strata_edges() -> list[int]:
    if not TAIL.is_file():
        raise SystemExit(f"{TAIL.relative_to(ROOT)} is missing; the strata are "
                         f"defined there and are not redefined here")
    st = json.loads(TAIL.read_text())["stratified_by_positive_count"]
    return [int(s["n_cryptic_from"]) for s in st] + [
        int(st[-1]["n_cryptic_below"])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    cdoc = json.loads(COUNTING.read_text())
    n_splits = a.splits or int(cdoc["protocol"]["n_splits"])
    by_width = {int(k.split()[-2]): v for k, v in cdoc["per_split"].items()}

    z = np.load(WIDE, allow_pickle=False)
    y, n_res, ctr = z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    z.close()
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}

    D = digit_cache.load(n_res)
    n_wires = int(D.shape[1])
    if n_wires not in by_width:
        raise SystemExit(f"frozen widths {sorted(by_width)}, not {n_wires}")
    frozen = np.asarray(by_width[n_wires], dtype=float)[:n_splits]

    tabs = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                            PARTITION_SEED)
    offs = cell_offsets(tabs)
    row = np.repeat(np.arange(len(n_res)), n_res)
    edges = strata_edges()

    n_pos = np.zeros(len(n_res), dtype=np.int64)
    off = 0
    for i, n in enumerate(n_res):
        n = int(n)
        n_pos[i] = int((y[off:off + n] == 1).sum())
        off += n

    arms = [(r, w) for r in RADII for w in WEIGHTS if not (r == 0.0 and w != 1.0)]
    seen = {k: np.zeros(len(n_res), dtype=np.int64) for k in arms}
    total = {k: np.zeros(len(n_res), dtype=np.float64) for k in arms}
    # Accumulated separately over the first and second half of the splits, so
    # the arm can be chosen on one half and scored on the other. Choosing the
    # best of thirteen arms and quoting its margin on the same numbers is
    # selection optimism, and section 3 of the memory records the identical
    # concern for pairings with this identical remedy.
    half_seen = [{k: np.zeros(len(n_res), dtype=np.int64) for k in arms}
                 for _ in (0, 1)]
    half_total = [{k: np.zeros(len(n_res), dtype=np.float64) for k in arms}
                  for _ in (0, 1)]
    deployed_per_split: list[float] = []

    t0 = time.perf_counter()
    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        Dfit = D[fit]
        frac, _t = compile_cells(Dfit, y[fit], tabs, offs)
        mult = lean_integer_fanout(Dfit, y[fit], tabs, offs, frac, RIDGE,
                                   FAN_OUT_CAP)
        del Dfit
        raw = score(D[pick], tabs, offs, frac, mult)
        idx = np.flatnonzero(~is_fit)
        for r, w in arms:
            per = auc_per_unit(gate_at(raw, ctr[pick], n_pick, r, w),
                               y[pick], n_pick)
            ok = ~np.isnan(per)
            seen[(r, w)][idx[ok]] += 1
            total[(r, w)][idx[ok]] += per[ok]
            h = 0 if s < n_splits // 2 else 1
            half_seen[h][(r, w)][idx[ok]] += 1
            half_total[h][(r, w)][idx[ok]] += per[ok]
            if (r, w) == (GATE_RADIUS, GATE_WEIGHT):
                deployed_per_split.append(float(np.nanmean(per)))
        print(f"  split {s + 1}/{n_splits}  deployed "
              f"{deployed_per_split[-1]:.4f}  frozen {frozen[s]:.4f}  "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    repro = float(np.abs(np.asarray(deployed_per_split) - frozen).max())

    # Per-unit means for every arm, kept so the comparisons below can be paired.
    # An unpaired difference of stratum means would be dominated by which units
    # a stratum happens to hold; the arms are evaluated on the same units, and
    # not exploiting that would throw away most of the precision available.
    per_unit = {}
    for r, w in arms:
        sc = seen[(r, w)] > 0
        m = np.full(len(n_res), np.nan)
        m[sc] = total[(r, w)][sc] / seen[(r, w)][sc]
        per_unit[(r, w)] = m

    results = {}
    for r, w in arms:
        sc = seen[(r, w)] > 0
        m = per_unit[(r, w)]
        by = []
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            sel = sc & (n_pos >= lo_e) & (n_pos < hi_e)
            if sel.sum() < 10:
                continue
            by.append({"n_cryptic_from": int(lo_e),
                       "n_cryptic_below": int(hi_e),
                       "n_units": int(sel.sum()),
                       "mean_auc": round(float(np.nanmean(m[sel])), 6)})
        results[f"r={r:g} w={w:g}"] = {
            "radius": r, "weight": w,
            "pooled_mean_auc": round(float(np.nanmean(m[sc])), 6),
            "by_stratum": by,
        }

    dep = per_unit[(GATE_RADIUS, GATE_WEIGHT)]

    def paired(arm, mask) -> dict:
        d = per_unit[arm][mask] - dep[mask]
        d = d[~np.isnan(d)]
        n = len(d)
        if n < 2:
            return {"n": n}
        se = float(d.std(ddof=1) / np.sqrt(n))
        p = sum(comb(n, i) for i in range(int((d > 0).sum()), n + 1)) / 2 ** n
        return {
            "n_units": n,
            "mean": round(float(d.mean()), 6),
            "ci95": [round(float(d.mean() - 1.96 * se), 6),
                     round(float(d.mean() + 1.96 * se), 6)],
            "crosses_zero": bool(abs(d.mean()) < 1.96 * se),
            "n_units_better": int((d > 0).sum()),
            "sign_test_p_one_sided": round(float(p), 6),
        }

    scored_any = ~np.isnan(dep)
    paired_vs_deployed = {}
    for r, w in arms:
        if (r, w) == (GATE_RADIUS, GATE_WEIGHT):
            continue
        rec = {"pooled": paired((r, w), scored_any)}
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            sel = scored_any & (n_pos >= lo_e) & (n_pos < hi_e)
            if sel.sum() < 10:
                continue
            rec[f"{lo_e}-{hi_e - 1}"] = paired((r, w), sel)
        paired_vs_deployed[f"r={r:g} w={w:g}"] = rec

    dep_key = f"r={GATE_RADIUS:g} w={GATE_WEIGHT:g}"
    strat_names = [f"{b['n_cryptic_from']}-{b['n_cryptic_below'] - 1}"
                   for b in results[dep_key]["by_stratum"]]
    best = {}
    for i, sn in enumerate(strat_names):
        ranked = sorted(results.items(),
                        key=lambda kv: -kv[1]["by_stratum"][i]["mean_auc"])
        best[sn] = {
            "best_arm": ranked[0][0],
            "best_mean_auc": ranked[0][1]["by_stratum"][i]["mean_auc"],
            "deployed_mean_auc": results[dep_key]["by_stratum"][i]["mean_auc"],
            "gain_over_deployed": round(
                ranked[0][1]["by_stratum"][i]["mean_auc"]
                - results[dep_key]["by_stratum"][i]["mean_auc"], 6),
            "best_radius": ranked[0][1]["radius"],
        }

    # Choose on one half of the splits, score on the other, both ways round.
    # An arm that is best only because it was chosen where it was scored shows
    # here as a margin that collapses; an arm that is genuinely better keeps it.
    def half_means(h: int) -> dict:
        out = {}
        for k in arms:
            sc = half_seen[h][k] > 0
            m = np.full(len(n_res), np.nan)
            m[sc] = half_total[h][k][sc] / half_seen[h][k][sc]
            out[k] = m
        return out

    hm = [half_means(0), half_means(1)]
    selection_check = {}
    for choose_on, score_on in ((0, 1), (1, 0)):
        ranked = sorted(arms, key=lambda k: -float(
            np.nanmean(hm[choose_on][k])))
        pick = ranked[0]
        d = hm[score_on][pick] - hm[score_on][(GATE_RADIUS, GATE_WEIGHT)]
        d = d[~np.isnan(d)]
        n = len(d)
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        on_choose = hm[choose_on][pick] - hm[choose_on][(GATE_RADIUS,
                                                        GATE_WEIGHT)]
        on_choose = on_choose[~np.isnan(on_choose)]
        selection_check[f"chose_on_splits_{choose_on}_scored_on_{score_on}"] = {
            "splits_chosen_on": ("first" if choose_on == 0 else "second")
                                + f" {n_splits // 2}",
            "arm_chosen": f"r={pick[0]:g} w={pick[1]:g}",
            "margin_where_it_was_chosen": round(float(on_choose.mean()), 6),
            "margin_where_it_was_not": round(float(d.mean()), 6),
            "ci95_where_it_was_not": [round(float(d.mean() - 1.96 * se), 6),
                                      round(float(d.mean() + 1.96 * se), 6)],
            "crosses_zero_where_it_was_not": bool(abs(d.mean()) < 1.96 * se),
            "n_units": int(n),
            "shrinkage": round(float(on_choose.mean() - d.mean()), 6),
        }
    both_pick_same = len({v["arm_chosen"]
                          for v in selection_check.values()}) == 1
    survives = all(not v["crosses_zero_where_it_was_not"]
                   and v["margin_where_it_was_not"] > 0
                   for v in selection_check.values())
    selection_check["verdict"] = {
        "both_halves_choose_the_same_arm": bool(both_pick_same),
        "margin_survives_on_the_half_it_was_not_chosen_from": bool(survives),
        "why_this_is_here": (
            "the best of thirteen arms was chosen on the same pick halves it "
            "was scored on. A margin that is real keeps most of its size on "
            "splits that did not select it; one that is selection optimism "
            "collapses. This is the remedy AGENT_MEMORY 3 already records for "
            "the pairing selection, applied to the gate"),
    }

    radii_by_stratum = [best[sn]["best_radius"] for sn in strat_names]
    rises = all(x <= y_ for x, y_ in zip(radii_by_stratum, radii_by_stratum[1:]))
    same = len(set(radii_by_stratum)) == 1

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "whether the spatial gate's optimum radius depends on how "
                    "many cryptic residues a unit has, which the selection that "
                    "chose r=18 could not see because it averaged over units",
        "prediction": PREDICTION,
        "strata_from": str(TAIL.relative_to(ROOT)),
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "compile_on": "the fit half; one compile per split, every arm reuses it",
            "evaluate_on": "the pick half",
            "metric": "ROC-AUC within a unit, averaged over the splits in which "
                      "the unit sat on the pick side",
            "radii": list(RADII),
            "weights": list(WEIGHTS),
            "r_zero_means": "no gate at all, the raw counting field",
        },
        "reproduction_check": {
            "recomputed_arm": dep_key,
            "max_absolute_difference_from_frozen_per_split": round(repro, 6),
            "reproduces_frozen_arm": bool(repro < 5e-4),
        },
        "arms": results,
        "paired_against_the_deployed_gate": paired_vs_deployed,
        "why_paired": (
            "every arm is evaluated on the same units, so the difference is "
            "paired and its standard error is far smaller than that of a "
            "difference of stratum means. An unpaired comparison here would "
            "be dominated by which units a stratum happens to hold"),
        "best_arm_per_stratum": best,
        "selection_check": selection_check,
        "verdict": {
            "optimum_radius_by_stratum": radii_by_stratum,
            "rises_with_pocket_size": bool(rises and not same),
            "identical_across_strata": bool(same),
            "the_reading": (
                "the optimum radius rises with pocket size, which is what a "
                "gate averaging over a mostly-negative neighbourhood predicts"
                if rises and not same else
                "every stratum prefers the same radius, so the gate is not the "
                "size dependence either" if same else
                "the optimum radius varies across strata but not monotonically "
                "in pocket size, which is neither prediction and should be read "
                "as noise unless a gain exceeds the reseed floor"),
            "what_this_is_not": PREDICTION["what_a_positive_result_would_not_be"],
        },
        "per_split_deployed_recomputed": [round(x, 6) for x in deployed_per_split],
        "per_split_deployed_frozen": [round(float(x), 6) for x in frozen],
    }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    if a.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    print(f"\n  reproduces the frozen deployed arm to {repro:.2e}: "
          f"{doc['reproduction_check']['reproduces_frozen_arm']}")
    print(f"\n  {'arm':16s} {'pooled':>8s}" + "".join(
        s.rjust(10) for s in strat_names))
    for k, v in results.items():
        mark = "  <- deployed" if k == dep_key else ""
        print(f"  {k:16s} {v['pooled_mean_auc']:8.4f}"
              + "".join(f"{b['mean_auc']:10.4f}" for b in v["by_stratum"])
              + mark)
    print()
    for sn in strat_names:
        b = best[sn]
        print(f"  stratum {sn:8s} best {b['best_arm']:16s} "
              f"{b['best_mean_auc']:.4f} against deployed "
              f"{b['deployed_mean_auc']:.4f}  gain {b['gain_over_deployed']:+.4f}")
    print("\n  paired against the deployed gate (same units, both arms):")
    for k in (f"r=14 w={GATE_WEIGHT:g}", "r=0 w=1"):
        if k not in paired_vs_deployed:
            continue
        print(f"    {k}")
        for sn, v in paired_vs_deployed[k].items():
            if "mean" not in v:
                continue
            print(f"      {sn:8s} {v['mean']:+.5f}  CI {v['ci95']}  "
                  f"{v['n_units_better']}/{v['n_units']} units better  "
                  f"p={v['sign_test_p_one_sided']:.2e}  "
                  f"crosses_zero={v['crosses_zero']}")
    print("\n  selection check -- chosen on one half of the splits, scored on the other:")
    for k, v in selection_check.items():
        if k == "verdict":
            continue
        print(f"    {v['arm_chosen']:12s} chosen on the {v['splits_chosen_on']} "
              f"splits: {v['margin_where_it_was_chosen']:+.5f} there, "
              f"{v['margin_where_it_was_not']:+.5f} on the other half "
              f"CI {v['ci95_where_it_was_not']} "
              f"crosses_zero={v['crosses_zero_where_it_was_not']}")
    sv = selection_check["verdict"]
    print(f"    both halves choose the same arm: "
          f"{sv['both_halves_choose_the_same_arm']}; margin survives: "
          f"{sv['margin_survives_on_the_half_it_was_not_chosen_from']}")
    print(f"\n  {doc['verdict']['the_reading']}")
    if a.write:
        print(f"\nwrote {out.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
