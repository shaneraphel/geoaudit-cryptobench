#!/usr/bin/env python3
"""Does the chosen architecture survive splits other than the one that chose it?

``tools/select_architecture_on_train.py`` compares nine fusion architectures on
one seeded half-split of the training fold and freezes the winner. That is
enough to keep the test fold out of the choice, which is what it was written
for, but it is not enough to say the architecture is the right one: with nine
candidates on one split, the winner carries the maximum of nine noisy estimates,
and the margin between the top few is small enough that another split could
reorder them. A reader is entitled to ask whether the table field was selected
or merely drawn.

So the selection is repeated, two ways, and neither of them reads the test fold.

Cluster-level cross-validation, which is the one that matters. CryptoBench ships
its training data as four folds, train-0 to train-3, built by its authors under
the MMseqs2 10 % clustering the benchmark is defined by and pairwise disjoint in
both PDB id and UniProt accession. Holding each out in turn is a four-fold
cross-validation under the benchmark's own clustering, not a proxy for it. This
is the coarser and the more honest test, because a 10 % cluster can span several
accessions and an accession-level split would let those homologues sit on
opposite sides.

Repeated accession-disjoint halves, which is the one with resolution. Four folds
give four numbers; a rank that moves once is hard to read. Twenty-five further
half-splits, each disjoint by accession exactly as the frozen split is, say how
often the frozen choice comes first when the only thing that changes is which
proteins landed in which half. The guarantee is weaker than the four-fold one --
that is stated rather than glossed -- and it is reported separately for that
reason.

What is not claimed: that winning both establishes the architecture is better
than its alternatives on the test fold. The margins here are small and several
candidates are within noise of each other throughout. What repeating the
selection can establish is the narrower thing the manuscript needs, that the
frozen choice is not an artefact of one particular draw.

Usage:
  PYTHONPATH=src python3.12 tools/crossvalidate_architecture.py
  PYTHONPATH=src python3.12 tools/crossvalidate_architecture.py --repeats 5
  PYTHONPATH=src python3.12 tools/crossvalidate_architecture.py --check
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from pocket_bench.paths import ROOT
from select_architecture_on_train import (
    SEED,
    build_level_cache,
    cluster_half_split,
    evaluate_candidates,
    load_train_fold,
)

FOLDS = ROOT / "data/cryptobench_apo/_osf/folds.json"
FROZEN = ROOT / "results/architecture_sweep/TRAIN_ONLY_SELECTION.json"
OUT = ROOT / "results/architecture_sweep/REPEATED_TRAIN_SELECTION.json"
TRAIN_FOLDS = ("train-0", "train-1", "train-2", "train-3")
N_REPEATS = 25


def _finite(x: float | None) -> float | None:
    """No bare NaN may reach a published JSON; an absent AUC is null."""
    return None if x is None or not math.isfinite(x) else float(x)


def rank_table(results: list[dict]) -> list[dict]:
    """Candidates ordered best first, each carrying its rank on this split."""
    scored = [r for r in results if math.isfinite(r["pick_half_roc_auc"])]
    scored.sort(key=lambda r: -r["pick_half_roc_auc"])
    return [{"architecture": r["architecture"],
             "roc_auc": _finite(r["pick_half_roc_auc"]),
             "rank": i}
            for i, r in enumerate(scored, 1)]


def summarise(rankings: list[list[dict]]) -> list[dict]:
    """Per architecture, across the splits: score spread and rank spread.

    Reported together on purpose. Selection frequency alone would call a
    coin-flip between two indistinguishable candidates a result; the spread of
    the score says how far apart they actually were.
    """
    names: list[str] = []
    for table in rankings:
        for row in table:
            if row["architecture"] not in names:
                names.append(row["architecture"])
    out = []
    for name in names:
        aucs = [row["roc_auc"] for table in rankings for row in table
                if row["architecture"] == name and row["roc_auc"] is not None]
        ranks = [row["rank"] for table in rankings for row in table
                 if row["architecture"] == name]
        if not aucs:
            continue
        mean = sum(aucs) / len(aucs)
        sd = (sum((a - mean) ** 2 for a in aucs) / (len(aucs) - 1)) ** 0.5 \
            if len(aucs) > 1 else 0.0
        out.append({
            "architecture": name,
            "n_splits": len(aucs),
            "mean_roc_auc": _finite(mean),
            "sd_roc_auc": _finite(sd),
            "min_roc_auc": _finite(min(aucs)),
            "max_roc_auc": _finite(max(aucs)),
            "mean_rank": sum(ranks) / len(ranks),
            "best_rank": min(ranks),
            "worst_rank": max(ranks),
            "n_first": sum(1 for r in ranks if r == 1),
        })
    out.sort(key=lambda r: r["mean_rank"])
    return out


def verdict_for(chosen: str, rankings: list[list[dict]],
                per_arch: list[dict]) -> dict:
    """How the frozen choice fared, and by how much over the runner-up."""
    ranks = [row["rank"] for table in rankings for row in table
             if row["architecture"] == chosen]
    margins = []
    for table in rankings:
        by_name = {row["architecture"]: row["roc_auc"] for row in table}
        mine = by_name.get(chosen)
        others = [v for k, v in by_name.items()
                  if k != chosen and v is not None]
        if mine is not None and others:
            margins.append(mine - max(others))
    mine_row = next((r for r in per_arch if r["architecture"] == chosen), None)
    return {
        "architecture": chosen,
        "n_splits": len(ranks),
        "n_first": sum(1 for r in ranks if r == 1),
        "worst_rank": max(ranks) if ranks else None,
        "mean_rank": (sum(ranks) / len(ranks)) if ranks else None,
        "mean_margin_over_runner_up": _finite(
            sum(margins) / len(margins)) if margins else None,
        "worst_margin_over_runner_up": _finite(min(margins)) if margins else None,
        "mean_roc_auc": mine_row["mean_roc_auc"] if mine_row else None,
        "sd_roc_auc": mine_row["sd_roc_auc"] if mine_row else None,
    }


def cluster_level_cv(F, y, n_res, ctr, level_cache, units, quiet):
    """Hold out each of CryptoBench's own training folds in turn."""
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

    rankings, per_fold = [], []
    for name in TRAIN_FOLDS:
        is_fit_unit = np.array([f != name for f in unit_fold])
        if not quiet:
            print(f"  hold out {name}: fit {int(is_fit_unit.sum())} units, "
                  f"pick {len(units) - int(is_fit_unit.sum())}", flush=True)
        table = rank_table(evaluate_candidates(
            F, y, n_res, ctr, level_cache, is_fit_unit, verbose=False))
        rankings.append(table)
        per_fold.append({
            "held_out_fold": name,
            "n_fit_units": int(is_fit_unit.sum()),
            "n_pick_units": int(len(units) - is_fit_unit.sum()),
            "winner": table[0]["architecture"],
            "ranking": table,
        })
        if not quiet:
            print(f"    winner: {table[0]['architecture']} "
                  f"({table[0]['roc_auc']:.4f})", flush=True)
    return rankings, per_fold


def repeated_halves(F, y, n_res, ctr, level_cache, units, cluster_of,
                    n_repeats, quiet):
    """The frozen split's design, re-drawn under other seeds."""
    rankings, per_split, seeds = [], [], []
    for i in range(n_repeats):
        seed = SEED + 1 + i
        seeds.append(seed)
        is_fit_unit, _ = cluster_half_split(units, cluster_of, seed)
        table = rank_table(evaluate_candidates(
            F, y, n_res, ctr, level_cache, is_fit_unit, verbose=False))
        rankings.append(table)
        per_split.append({"seed": seed, "winner": table[0]["architecture"],
                          "ranking": table})
        if not quiet:
            print(f"  seed {seed}: {table[0]['architecture']} "
                  f"({table[0]['roc_auc']:.4f})", flush=True)
    return rankings, per_split, seeds


def build(n_repeats: int, quiet: bool) -> dict:
    chosen = json.loads(FROZEN.read_text())["selected"]["architecture"]
    F, y, n_res, ctr, units, cluster_of = load_train_fold()
    level_cache = build_level_cache(F, n_res)

    if not quiet:
        print(f"frozen choice: {chosen}")
        print(f"cluster-level cross-validation over "
              f"{len(TRAIN_FOLDS)} CryptoBench training folds")
    cv_rank, cv_folds = cluster_level_cv(
        F, y, n_res, ctr, level_cache, units, quiet)
    cv_arch = summarise(cv_rank)

    if not quiet:
        print(f"repeated accession-disjoint halves, {n_repeats} seeds")
    rp_rank, rp_splits, seeds = repeated_halves(
        F, y, n_res, ctr, level_cache, units, cluster_of, n_repeats, quiet)
    rp_arch = summarise(rp_rank)

    return {
        "schema": "geoaudit.repeated_architecture_selection.v1",
        "clinical_grade": False,
        "why": "the frozen architecture was chosen on one half-split of the "
               "training fold; this reports whether that choice survives the "
               "benchmark's own cluster-level folds and repeated redraws of "
               "the same split design, so the choice is not read as an "
               "artefact of one draw",
        "reads_test_fold": False,
        "frozen_choice": {
            "architecture": chosen,
            "source": str(FROZEN.relative_to(ROOT)),
        },
        "cluster_level_cv": {
            "design": "hold out each CryptoBench training fold in turn; the "
                      "folds are the benchmark authors' own, built under "
                      "MMseqs2 10% clustering, and are pairwise disjoint in "
                      "PDB id and UniProt accession",
            "source": str(FOLDS.relative_to(ROOT)),
            "n_folds": len(TRAIN_FOLDS),
            "folds": cv_folds,
            "per_architecture": cv_arch,
            "frozen_choice": verdict_for(chosen, cv_rank, cv_arch),
        },
        "repeated_halves": {
            "design": "the frozen split's own design re-drawn under other "
                      "seeds: cluster_id here is the UniProt accession, which "
                      "is finer than a 10% cluster, so this resolves the "
                      "ranking more finely than the four folds but guarantees "
                      "less about homology",
            "n_repeats": n_repeats,
            "seeds": seeds,
            "splits": rp_splits,
            "per_architecture": rp_arch,
            "frozen_choice": verdict_for(chosen, rp_rank, rp_arch),
        },
    }


def _report(rec: dict) -> None:
    for block, label in (("cluster_level_cv", "cluster-level CV, 4 folds"),
                         ("repeated_halves", "repeated halves")):
        b = rec[block]
        v = b["frozen_choice"]
        n = b.get("n_folds") or b.get("n_repeats")
        print(f"\n{label}: the frozen choice came first on "
              f"{v['n_first']}/{n} splits, worst rank {v['worst_rank']}, "
              f"mean ROC-AUC {v['mean_roc_auc']:.4f} "
              f"(sd {v['sd_roc_auc']:.4f})")
        print(f"  mean margin over the best alternative "
              f"{v['mean_margin_over_runner_up']:+.4f}, "
              f"worst {v['worst_margin_over_runner_up']:+.4f}")
        print("  by mean rank:")
        for r in b["per_architecture"][:4]:
            print(f"    {r['mean_rank']:4.1f}  {r['architecture']:52s} "
                  f"{r['mean_roc_auc']:.4f} +- {r['sd_roc_auc']:.4f}  "
                  f"first {r['n_first']}/{r['n_splits']}")


def audit() -> int:
    """What can be checked without the descriptor cache, which CI does not have.

    The cache and the OSF folds are not committed -- they are hundreds of
    megabytes -- so CI cannot rerun 29 selections, and a gate that only ran on
    one laptop would be switched off within a week. What CI can do is check that
    the summary this artifact leads with actually follows from the per-split
    tables stored underneath it, and that it still describes the architecture
    the frozen selection names. Those are the two ways this file could come to
    say something false: a hand-edited headline, or a selection that moved
    without the cross-validation being rerun.
    """
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    rec = json.loads(OUT.read_text())
    problems: list[str] = []

    if rec.get("schema") != "geoaudit.repeated_architecture_selection.v1":
        problems.append(f"unexpected schema {rec.get('schema')!r}")
    if rec.get("clinical_grade") is not False:
        problems.append("clinical_grade must be false")
    if rec.get("reads_test_fold") is not False:
        problems.append("reads_test_fold must be false")

    chosen = rec.get("frozen_choice", {}).get("architecture")
    frozen = json.loads(FROZEN.read_text())["selected"]["architecture"]
    if chosen != frozen:
        problems.append(
            f"this artifact cross-validates {chosen!r} but "
            f"{FROZEN.name} now selects {frozen!r}; rerun "
            f"tools/crossvalidate_architecture.py")

    candidates = {c["architecture"]
                  for c in json.loads(FROZEN.read_text())["candidates"]}
    for block in ("cluster_level_cv", "repeated_halves"):
        b = rec.get(block) or {}
        tables = [f["ranking"] for f in (b.get("folds") or b.get("splits") or [])]
        n = b.get("n_folds") if block == "cluster_level_cv" else b.get("n_repeats")
        if not tables:
            problems.append(f"{block}: no per-split rankings recorded")
            continue
        if len(tables) != n:
            problems.append(f"{block}: declares {n} splits, stores {len(tables)}")
        for i, table in enumerate(tables):
            names = {r["architecture"] for r in table}
            if names != candidates:
                problems.append(
                    f"{block} split {i}: ranks a different candidate set than "
                    f"{FROZEN.name} compared")
            if [r["rank"] for r in table] != list(range(1, len(table) + 1)):
                problems.append(f"{block} split {i}: ranks are not 1..n in order")
            if any(table[j]["roc_auc"] < table[j + 1]["roc_auc"]
                   for j in range(len(table) - 1)):
                problems.append(f"{block} split {i}: rank order contradicts the "
                                f"scores it records")

        # The headline, recomputed from the tables below it.
        recomputed = verdict_for(chosen, tables, summarise(tables))
        recorded = b.get("frozen_choice") or {}
        for field in ("n_splits", "n_first", "worst_rank"):
            if recorded.get(field) != recomputed[field]:
                problems.append(
                    f"{block}: reports {field}={recorded.get(field)} but the "
                    f"stored rankings give {recomputed[field]}")
        for field in ("mean_rank", "mean_roc_auc", "mean_margin_over_runner_up"):
            a, c = recorded.get(field), recomputed[field]
            if a is None or c is None or abs(a - c) > 5e-9:
                problems.append(
                    f"{block}: reports {field}={a} but the stored rankings "
                    f"give {c}")

    if problems:
        print(f"{OUT.relative_to(ROOT)} audit failed:")
        for p in problems:
            print(f"  - {p}")
        return 1

    for block, label in (("cluster_level_cv", "cluster-level CV"),
                         ("repeated_halves", "repeated halves")):
        b, v = rec[block], rec[block]["frozen_choice"]
        n = b.get("n_folds") or b.get("n_repeats")
        print(f"{label}: {chosen!r} first on {v['n_first']}/{n} splits "
              f"(worst rank {v['worst_rank']}), and the summary follows from "
              f"the stored rankings")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repeats", type=int, default=N_REPEATS,
                    help="accession-disjoint half-splits beyond the four folds")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="machine-independent: the artifact is present, its "
                         "summary follows from its own tables, and it names "
                         "the architecture the frozen selection chose")
    args = ap.parse_args(argv)

    if args.check:
        return audit()

    rec = build(args.repeats, args.quiet)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=2, allow_nan=False) + "\n")
    _report(rec)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
