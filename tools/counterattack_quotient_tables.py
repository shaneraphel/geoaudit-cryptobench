#!/usr/bin/env python3
"""Can a quotient table beat the dense bank it replaces, on the training fold?

Why this exists
---------------
The counting field loses to a closed-form linear functional of its own
invariants -- 0.7667 against 0.7954 on the official fold -- and the manuscript
attributes the gap to capacity: a dense quaternary table admits ``d <= 6.86``
digits here, so thirty-five invariants cannot address one table and cannot
interact across the six they are split over. That is an argument about the
number of cells, and ``quotient_tables`` shows the number of cells is a function
of the symmetry imposed on the table, not of the width alone.

So the question this file answers is narrow and falsifiable: holding the
invariants, the digitisation rule, the fusion rule and the gate exactly as the
frozen detector has them, does replacing dense tables by ``S_6``-invariant ones
raise the pick-half ROC-AUC, and does it do so on splits other than the one it
was found on.

The candidate set is declared once, below, and it includes the constructions
that failed. Reporting only the two that worked would make the comparison a
selection over an unstated pool.

How it is judged
----------------
Every candidate is scored on fourteen splits, none of which is the test fold:
CryptoBench's own four training folds, held out in turn -- these are
cluster-disjoint under the benchmark's MMseqs2 10 % clustering -- and ten redraws
of the accession-disjoint half-split under further seeds. What is reported per
candidate is the mean over splits, the spread, and the paired difference against
the frozen dense bank on the same split, because the splits differ from each
other by more than the candidates differ from each other.

One thing this file cannot undo, and states instead: the winning construction
was found by trying about thirty variants on the single seed-20260725 half-split
before any of this ran, and on that split it measured +0.0132 against the frozen
bank. Over the fourteen splits it measures +0.0087. The difference between those
two numbers is what selecting on one split buys you, and it is reported here
rather than quietly replaced by the smaller number.

Usage:
    PYTHONPATH=src python3.12 tools/counterattack_quotient_tables.py [--quiet]
    PYTHONPATH=src python3.12 tools/counterattack_quotient_tables.py --check
"""
from __future__ import annotations

import argparse
import json
from math import log

import numpy as np

from pocket_bench.methods.quotient_tables import (
    compile_cells, n_cells_dense, n_orbits, orbit_address, read_cells,
    widest_admissible,
)
from pocket_bench.paths import ROOT

from crossvalidate_architecture import FOLDS, TRAIN_FOLDS
from select_architecture_on_train import (
    RADII, SEED, _unit, bank_fracs, build_level_cache, cluster_half_split,
    load_train_fold, patch_mean, per_unit_auc, pooled_auc,
)

OUT = ROOT / "results/architecture_sweep/COUNTERATTACK_QUOTIENT.json"
SCHEMA = "geoaudit.counterattack_quotient.v1"
LEVELS = (4, 6, 8, 10, 12)
N_HALVES = 10
CONTROL = "dense bank (frozen), L=4"
FOUND_ON_ONE_SPLIT = 0.0132       # what the winner measured on seed 20260725


# ----------------------------------------------------------------- candidates
def _sym_bank(lev, fit_mask, pick_mask, yfit, rate, groups, levels_list):
    """One symmetric table per (thematic group, level), read on the pick half."""
    fr, gini = [], []
    for L in levels_list:
        D = lev[L]
        for cols in groups:
            a_fit = orbit_address(D[fit_mask], cols, L)
            a_pick = orbit_address(D[pick_mask], cols, L)
            addrs, pos, tot = compile_cells(a_fit, yfit)
            f_pick = read_cells(addrs, pos, tot, a_pick, rate)
            f_fit = read_cells(addrs, pos, tot, a_fit, rate)
            fr.append(f_pick)
            gini.append(abs(2.0 * pooled_auc(f_fit, yfit) - 1.0))
    return fr, np.asarray(gini)


def _fuse(fr, gini):
    """Integer-multiplicity fan-out: table k is replicated by its Gini rank."""
    order = np.argsort(gini)
    mult = np.empty(len(fr), dtype=np.int64)
    mult[order] = np.arange(1, len(fr) + 1)
    return np.sum([m * f for m, f in zip(mult, fr)], axis=0)


def evaluate_candidates(F, y, n_res, ctr, lev, is_fit_unit) -> list[dict]:
    """Score every declared candidate on one fit/pick split.

    Declared once, here, so that repeating over other splits cannot silently
    compare a different pool than the first split did.
    """
    row_unit = np.repeat(np.arange(len(n_res)), n_res)
    fit_mask = is_fit_unit[row_unit]
    pick_mask = ~fit_mask
    n_pick_per = np.array([n for n, f in zip(n_res, is_fit_unit) if not f])
    yfit, ypick = y[fit_mask], y[pick_mask]
    ctr_pick = ctr[pick_mask]
    rate = float(yfit.mean())
    M = F.shape[1]
    thematic = [list(range(i, min(i + 6, M))) for i in range(0, M, 6)]

    def gated(S):
        G = np.sum([_unit(patch_mean(S, ctr_pick, n_pick_per, r)) for r in RADII],
                   axis=0)
        return _unit(S) + _unit(G)

    def score(S):
        return per_unit_auc(gated(S), ypick, n_pick_per)

    out = []

    # 0. the control: exactly what the frozen detector does
    D4 = lev[4]
    fr_dense, g_dense = bank_fracs(D4[fit_mask], yfit, D4[pick_mask],
                                   thematic, 4, rate)
    out.append({"architecture": CONTROL, "roc_auc": score(_fuse(fr_dense, g_dense)),
                "n_tables": len(fr_dense), "cells_per_table": n_cells_dense(6, 4)})

    # 1-5. one symmetric level at a time
    sym = {}
    for L in LEVELS:
        fr, g = _sym_bank(lev, fit_mask, pick_mask, yfit, rate, thematic, [L])
        sym[L] = (fr, g)
        out.append({"architecture": f"S_6 quotient bank, L={L}",
                    "roc_auc": score(_fuse(fr, g)), "n_tables": len(fr),
                    "cells_per_table": n_orbits(6, L)})

    # 6. the stack that won: three quotient levels together
    fr, g = _sym_bank(lev, fit_mask, pick_mask, yfit, rate, thematic, [8, 6, 4])
    out.append({"architecture": "S_6 quotient bank, L=8+6+4 stacked",
                "roc_auc": score(_fuse(fr, g)), "n_tables": len(fr),
                "cells_per_table": None})

    # 7. quotient beside dense rather than instead of it
    fr_mix = list(fr_dense) + list(sym[8][0])
    g_mix = np.concatenate([g_dense, sym[8][1]])
    out.append({"architecture": "dense L=4 + S_6 quotient L=8",
                "roc_auc": score(_fuse(fr_mix, g_mix)), "n_tables": len(fr_mix),
                "cells_per_table": None})

    # 8. the whole bank in one symmetric table -- the width the theorem allows,
    #    and the construction that shows width is not what was missing
    fr_all, g_all = _sym_bank(lev, fit_mask, pick_mask, yfit, rate,
                              [list(range(M))], [4])
    out.append({"architecture": "single S_35 quotient table, L=4",
                "roc_auc": score(_fuse(fr_all, g_all)), "n_tables": 1,
                "cells_per_table": n_orbits(M, 4)})

    # 9. wider blocks, which the theorem permits and exchangeability does not
    wide = [list(range(i, min(i + 12, M))) for i in range(0, M, 12)]
    wide = [b for b in wide if len(b) >= 2]
    fr_w, g_w = _sym_bank(lev, fit_mask, pick_mask, yfit, rate, wide, [4])
    out.append({"architecture": "S_12 quotient bank, L=4",
                "roc_auc": score(_fuse(fr_w, g_w)), "n_tables": len(fr_w),
                "cells_per_table": n_orbits(12, 4)})

    return out


# --------------------------------------------------------------------- splits
def _splits(units, cluster_of):
    """Fourteen fit/pick masks, and what each one guarantees."""
    folds = json.loads(FOLDS.read_text())
    fold_of: dict[str, str] = {}
    for name in TRAIN_FOLDS:
        for pdb in folds[name]:
            fold_of[str(pdb).lower()] = name
    unit_fold = [fold_of.get(u.split("_")[0].lower()) for u in units]
    unplaced = [u for u, f in zip(units, unit_fold) if f is None]
    if unplaced:
        raise SystemExit(f"{len(unplaced)} cached units belong to no CryptoBench "
                         f"training fold, e.g. {unplaced[:3]}")

    for name in TRAIN_FOLDS:
        yield (f"cv:{name}", "MMseqs2 10% cluster-disjoint (CryptoBench's own fold)",
               np.array([f != name for f in unit_fold]))
    for k in range(N_HALVES):
        seed = SEED + 1000 * (k + 1)
        is_fit, _ = cluster_half_split(units, cluster_of, seed)
        yield (f"half:{seed}", "UniProt accession-disjoint", is_fit)


def _rank(rows: list[dict]) -> list[dict]:
    order = sorted(rows, key=lambda r: -r["roc_auc"])
    ctrl = next(r["roc_auc"] for r in rows if r["architecture"] == CONTROL)
    return [{"rank": i + 1, "architecture": r["architecture"],
             "roc_auc": round(float(r["roc_auc"]), 6),
             "delta_vs_control": round(float(r["roc_auc"] - ctrl), 6)}
            for i, r in enumerate(order)]


def summarise(rankings: list[list[dict]]) -> list[dict]:
    names = [r["architecture"] for r in rankings[0]]
    out = []
    for name in names:
        aucs = [next(r["roc_auc"] for r in tab if r["architecture"] == name)
                for tab in rankings]
        dels = [next(r["delta_vs_control"] for r in tab if r["architecture"] == name)
                for tab in rankings]
        ranks = [next(r["rank"] for r in tab if r["architecture"] == name)
                 for tab in rankings]
        out.append({
            "architecture": name,
            "mean_roc_auc": round(float(np.mean(aucs)), 6),
            "sd_roc_auc": round(float(np.std(aucs, ddof=1)), 6),
            "mean_delta_vs_control": round(float(np.mean(dels)), 6),
            "worst_delta_vs_control": round(float(np.min(dels)), 6),
            "n_splits_beating_control": int(sum(1 for d in dels if d > 0)),
            "n_first": int(sum(1 for r in ranks if r == 1)),
            "worst_rank": int(max(ranks)),
        })
    return sorted(out, key=lambda r: -r["mean_delta_vs_control"])


def build(quiet: bool) -> dict:
    F, y, n_res, ctr, units, cluster_of = load_train_fold()
    lev = build_level_cache(F, n_res, levels=LEVELS)
    rN = int(y.sum())

    rankings, per_split = [], []
    for name, guarantee, is_fit in _splits(units, cluster_of):
        rows = evaluate_candidates(F, y, n_res, ctr, lev, is_fit)
        table = _rank(rows)
        rankings.append(table)
        per_split.append({"split": name, "disjoint_by": guarantee,
                          "n_fit_units": int(is_fit.sum()),
                          "n_pick_units": int(len(units) - is_fit.sum()),
                          "winner": table[0]["architecture"], "ranking": table})
        if not quiet:
            print(f"  {name:<16} winner {table[0]['architecture']:<38} "
                  f"{table[0]['roc_auc']:.4f} "
                  f"({table[0]['delta_vs_control']:+.4f})", flush=True)

    summary = summarise(rankings)
    best = summary[0]
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "question": "does an S_6-invariant quotient table beat the dense table it "
                    "replaces, on splits other than the one it was found on",
        "capacity": {
            "n_train_residues": int(F.shape[0]),
            "n_train_positives": rN,
            "base_rate": round(float(y.mean()), 6),
            "dense_width_bound_L4": round(log(rN, 4), 4),
            "dense_widest_admissible_L4": widest_admissible(4, rN, symmetric=False),
            "quotient_widest_admissible_L4": widest_admissible(4, rN, symmetric=True),
            "dense_cells_d6_L4": n_cells_dense(6, 4),
            "quotient_cells_d6_L4": n_orbits(6, 4),
            "quotient_cells_d6_L8": n_orbits(6, 8),
            "quotient_cells_d35_L4": n_orbits(35, 4),
            "note": "a dense table over d digits has L**d cells; a table invariant "
                    "under S_d has as many as S_d has orbits, C(d+L-1, d), and it "
                    "is the orbit count the base rate has to populate",
        },
        "n_splits": len(rankings),
        "n_candidates": len(rankings[0]),
        "control": CONTROL,
        "splits": per_split,
        "summary": summary,
        "selected": best,
        "selection_honesty": {
            "found_on_split": f"half:{SEED}",
            "delta_on_the_split_it_was_found_on": FOUND_ON_ONE_SPLIT,
            "delta_over_all_splits": best["mean_delta_vs_control"],
            "note": "about thirty variants were tried on the seed-20260725 "
                    "half-split before this file existed, so the delta measured "
                    "there is the maximum of thirty noisy estimates. The "
                    "difference between the two numbers is the cost of selecting "
                    "on one split and is reported rather than dropped.",
        },
    }


def _report(doc: dict) -> None:
    c = doc["capacity"]
    print(f"\ncapacity on {c['n_train_positives']} training positives:")
    print(f"  dense    L=4  admits d <= {c['dense_widest_admissible_L4']} "
          f"(log_4 rN = {c['dense_width_bound_L4']:.2f}), "
          f"{c['dense_cells_d6_L4']:,} cells at d=6")
    print(f"  quotient L=4  admits d <= {c['quotient_widest_admissible_L4']}, "
          f"{c['quotient_cells_d6_L4']:,} cells at d=6 "
          f"({c['quotient_cells_d6_L8']:,} at L=8)")
    print(f"\n{doc['n_candidates']} candidates over {doc['n_splits']} splits, "
          f"control = {doc['control']}")
    print(f"  {'architecture':<40} {'mean':>8} {'sd':>7} {'vs ctrl':>9} "
          f"{'worst':>8} {'beats':>6}")
    for r in doc["summary"]:
        print(f"  {r['architecture']:<40} {r['mean_roc_auc']:>8.4f} "
              f"{r['sd_roc_auc']:>7.4f} {r['mean_delta_vs_control']:>+9.4f} "
              f"{r['worst_delta_vs_control']:>+8.4f} "
              f"{r['n_splits_beating_control']:>3}/{doc['n_splits']}")
    h = doc["selection_honesty"]
    print(f"\nselected: {doc['selected']['architecture']}")
    print(f"  on the split it was found on: {h['delta_on_the_split_it_was_found_on']:+.4f}")
    print(f"  over all {doc['n_splits']} splits:          "
          f"{h['delta_over_all_splits']:+.4f}")


def audit() -> int:
    """Structural check: CI has no descriptor cache, so it re-derives the summary
    from the per-split rankings the artifact carries and refuses a headline that
    does not follow from them."""
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    doc = json.loads(OUT.read_text())
    problems: list[str] = []
    if doc.get("schema") != SCHEMA:
        problems.append(f"schema is {doc.get('schema')!r}, expected {SCHEMA!r}")
    if doc.get("reads_test_fold") is not False:
        problems.append("reads_test_fold is not false")
    if doc.get("clinical_grade") is not False:
        problems.append("clinical_grade is not false")

    splits = doc.get("splits") or []
    if len(splits) != doc.get("n_splits"):
        problems.append("n_splits disagrees with the number of recorded splits")
    names = {tuple(sorted(r["architecture"] for r in s["ranking"])) for s in splits}
    if len(names) > 1:
        problems.append("the splits do not all rank the same candidate set")
    if not any(s["split"].startswith("cv:") for s in splits):
        problems.append("no CryptoBench cluster-disjoint fold among the splits")

    recomputed = summarise([s["ranking"] for s in splits])
    if recomputed != doc.get("summary"):
        problems.append("the summary does not follow from the per-split rankings")
    if doc.get("selected") != (recomputed[0] if recomputed else None):
        problems.append("the selected architecture is not the one the splits name")

    cap = doc.get("capacity") or {}
    rN = cap.get("n_train_positives", 0)
    for key, want in (
        ("dense_widest_admissible_L4", widest_admissible(4, rN, symmetric=False)),
        ("quotient_widest_admissible_L4", widest_admissible(4, rN, symmetric=True)),
        ("dense_cells_d6_L4", n_cells_dense(6, 4)),
        ("quotient_cells_d6_L4", n_orbits(6, 4)),
        ("quotient_cells_d6_L8", n_orbits(6, 8)),
        ("quotient_cells_d35_L4", n_orbits(35, 4)),
    ):
        if cap.get(key) != want:
            problems.append(f"capacity.{key} is {cap.get(key)}, recomputes to {want}")

    if problems:
        print(f"quotient counterattack FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"OK {OUT.relative_to(ROOT)}: {doc['n_candidates']} candidates over "
          f"{doc['n_splits']} splits, {doc['selected']['architecture']} leads by "
          f"{doc['selected']['mean_delta_vs_control']:+.4f}, test fold unread")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="audit the committed artifact without the cache")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        return audit()
    doc = build(args.quiet)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    if not args.quiet:
        _report(doc)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
