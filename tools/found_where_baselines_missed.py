#!/usr/bin/env python3
"""Units the detector ranks well and both published baselines rank at chance.

The question, and the trap in it
--------------------------------
"Find a case P2Rank and PocketMiner both missed and we found" is a request for
evidence, and searching 770 units for one is also the most reliable way to
manufacture it. Two rules make the difference:

* **the criterion is fixed here, in code, before the counts are read**, so it
  cannot be tuned until the answer is pleasing;
* **the mirror set is reported beside it**. For every unit where we rank the
  cryptic residues well and both baselines are at chance, the artifact also
  counts the units where both baselines rank them well and we are at chance.
  Without that second number the first one is not evidence of anything: on any
  three methods of similar average accuracy, disagreements exist in both
  directions, and only the *asymmetry* says something.

A reader who is shown only the favourable direction has been shown a selection,
not a result. That is the failure this repository has already had to retract
once, and the paragraph in the manuscript about it is the reason this file
reports both.

Why the training fold
---------------------
Every number here comes from the 770 training units, and it costs no read of the
held-out fold. Per-unit ROC-AUCs exist for all three methods without re-scoring
anything: P2Rank's are frozen in ``P2RANK_TRAIN_FOLD.json``, PocketMiner's
per-residue predictions are on disk under ``data/baselines/pocketminer_train/``,
and ours are computed on the same twelve cluster-disjoint halvings every other
training-fold measurement in this repository uses.

pLM-NN is absent here, and the artifact says so rather than quietly comparing
against two baselines while the prose says three. Its published weights score a
sequence, and it has never been run on the training fold; doing so is a
five-hour encoder pass, not a lookup. The official-fold version of this analysis,
where pLM-NN's per-residue probabilities are already frozen, is a read of the
held-out set and needs a preregistration and an index before it is run.

The rule
--------
A unit is a **recovery** when

    ours >= FOUND and p2rank <= MISSED and pocketminer <= MISSED

and a **mirror** when

    p2rank >= FOUND and pocketminer >= FOUND and ours <= MISSED

with ROC-AUC within the unit, ``FOUND = 0.80`` and ``MISSED = 0.55``. Chance is
0.50. The gap between the two thresholds is deliberate: a unit where everyone
scores 0.7 is not a disagreement, and a rule with a single cut would call it one.

What a recovery is and is not
-----------------------------
It is a chain where the cryptic-binding residues are ranked above the rest by the
counting field and are not ranked above the rest by either published method. It
is **not** a claim that a pocket exists that the baselines cannot see, that the
site is druggable, or that anything here is clinical. ROC-AUC within a unit is a
ranking statement about that unit's residues and nothing else.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import digit_cache  # noqa: E402
from baseline_by_stratum import pocketminer_per_unit  # noqa: E402
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
    PARTITION_ROUNDS,
    PARTITION_SEED,
    RIDGE,
    TABLE_WIDTH,
    apply_gate,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.found_where_baselines_missed.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
P2RANK_TRAIN = ROOT / "results/architecture_sweep/P2RANK_TRAIN_FOLD.json"
OUT = ROOT / "results/architecture_sweep/RECOVERED_UNITS_TRAIN.json"

# Fixed before the counts are read. Chance is 0.50.
FOUND = 0.80
MISSED = 0.55
N_LISTED = 40

# The headline pair is strict and gives a small count. Rather than loosen it
# until the number is impressive, the whole ladder is reported: if the asymmetry
# only appears at one setting it is an artefact of that setting, and if it holds
# all the way down it is a property of the methods. A reader can then pick the
# strictness they believe and read off both counts at it.
LADDER = ((0.80, 0.55), (0.80, 0.60), (0.75, 0.60), (0.75, 0.65),
          (0.70, 0.60), (0.70, 0.65), (0.65, 0.60), (0.65, 0.65))


PM_TRAIN = ROOT / "results/baselines/POCKETMINER_TRAIN_SCORES.json"


def featurisation_is_verified() -> dict[str, dict]:
    """Per unit, whether every method was demonstrably given the same residues.

    A win on a chain whose file the three methods parse differently is a win
    about parsing. ``4m7p_A`` is the case that forces this: the deposit carries
    twenty ensemble copies, 60,040 ATOM lines for 3,002 atoms, and PocketMiner
    drops 6,905 conformer residues there -- every one of the 6,905 it drops in
    the whole fold -- with 505 resseq collisions from insertion codes and
    ``agrees_with_official_featurisation`` recorded as null, meaning it could not
    be checked against the authors' own tensor. Our detector scores 0.918 on it
    against 0.439 and 0.214. That is the most spectacular disagreement in the
    fold and it is not evidence about pocket detection, so it is separated here
    rather than quietly kept.
    """
    out = {}
    for u in json.loads(PM_TRAIN.read_text())["units"]:
        out[u["unit"]] = {
            "pocketminer_agrees_with_official_featurisation":
                u.get("agrees_with_official_featurisation"),
            "n_dropped_by_pocketminer": len(u.get("dropped") or []),
            "n_resseq_collisions": len(
                u.get("resseq_collisions_from_insertion_codes") or []),
        }
    return out


def p2rank_per_unit() -> dict[str, float]:
    rows = json.loads(P2RANK_TRAIN.read_text())["rows"]
    out = {}
    for r in rows:
        v = r.get("residue_auc")
        if v is not None:
            out[r["unit_id"]] = float(v)
    return out


def ours_per_unit(n_splits: int) -> tuple[dict[str, float], dict[str, int],
                                          dict[str, int], float]:
    z = np.load(WIDE, allow_pickle=False)
    y, n_res, ctr = z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    z.close()
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}

    cdoc = json.loads(COUNTING.read_text())
    by_width = {int(k.split()[-2]): v for k, v in cdoc["per_split"].items()}

    D = digit_cache.load(n_res)
    n_wires = int(D.shape[1])
    frozen = np.asarray(by_width[n_wires], dtype=float)[:n_splits]
    tabs = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                            PARTITION_SEED)
    offs = cell_offsets(tabs)
    row = np.repeat(np.arange(len(n_res)), n_res)

    seen = np.zeros(len(n_res), dtype=np.int64)
    total = np.zeros(len(n_res), dtype=np.float64)
    per_split = []
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
        per = auc_per_unit(apply_gate(score(D[pick], tabs, offs, frac, mult),
                                      ctr[pick], n_pick), y[pick], n_pick)
        idx = np.flatnonzero(~is_fit)
        ok = ~np.isnan(per)
        seen[idx[ok]] += 1
        total[idx[ok]] += per[ok]
        per_split.append(float(np.nanmean(per)))
        print(f"  split {s + 1}/{n_splits}  {per_split[-1]:.4f}  "
              f"frozen {frozen[s]:.4f}  {time.perf_counter() - t0:.0f}s",
              flush=True)

    repro = float(np.abs(np.asarray(per_split) - frozen).max())
    if repro >= 5e-4:
        raise SystemExit(
            f"our recomputed per-split means differ from the frozen ones by "
            f"{repro:.2e}; the cases would be selected using a detector that is "
            f"not the deployed one")

    n_pos, n_all = {}, {}
    off = 0
    for u, n in zip(units, n_res):
        n = int(n)
        n_pos[u] = int((y[off:off + n] == 1).sum())
        n_all[u] = n
        off += n
    ours = {u: float(total[i] / seen[i])
            for i, u in enumerate(units) if seen[i] > 0}
    return ours, n_pos, n_all, repro


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    n_splits = a.splits or int(
        json.loads(COUNTING.read_text())["protocol"]["n_splits"])
    ours, n_pos, n_all, repro = ours_per_unit(n_splits)
    p2 = p2rank_per_unit()
    pm = {u: r["auc"] for u, r in pocketminer_per_unit().items()
          if not np.isnan(r["auc"])}

    feat = featurisation_is_verified()
    shared = sorted(set(ours) & set(p2) & set(pm))
    rec, mir, caveat = [], [], []
    for u in shared:
        o, a2, am = ours[u], p2[u], pm[u]
        f = feat.get(u, {})
        clean = (f.get("pocketminer_agrees_with_official_featurisation") is True
                 and not f.get("n_dropped_by_pocketminer")
                 and not f.get("n_resseq_collisions"))
        rowd = {"unit": u, "n_residues": n_all[u], "n_cryptic": n_pos[u],
                "ours": round(o, 4), "p2rank": round(a2, 4),
                "pocketminer": round(am, 4),
                "margin_over_the_better_baseline": round(o - max(a2, am), 4),
                "every_method_verified_on_the_same_residues": clean,
                **f}
        if o >= FOUND and a2 <= MISSED and am <= MISSED:
            (rec if clean else caveat).append(rowd)
        if a2 >= FOUND and am >= FOUND and o <= MISSED:
            mir.append(rowd)
    caveat.sort(key=lambda r: -r["margin_over_the_better_baseline"])
    rec.sort(key=lambda r: -r["margin_over_the_better_baseline"])
    mir.sort(key=lambda r: r["margin_over_the_better_baseline"])

    ladder = []
    for f, m in LADDER:
        nr = sum(1 for u in shared
                 if ours[u] >= f and p2[u] <= m and pm[u] <= m)
        nm = sum(1 for u in shared
                 if p2[u] >= f and pm[u] >= f and ours[u] <= m)
        ladder.append({"found_at_or_above": f, "missed_at_or_below": m,
                       "n_recovered": nr, "n_mirror": nm,
                       "difference": nr - nm})

    triples = [{"unit": u, "ours": round(ours[u], 4),
                "p2rank": round(p2[u], 4), "pocketminer": round(pm[u], 4),
                "n_residues": n_all[u], "n_cryptic": n_pos[u]}
               for u in shared]

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "on how many training units does the counting field rank the "
                    "cryptic binding residues well while both published "
                    "baselines rank them at chance, and on how many is it the "
                    "other way round",
        "rule": {
            "found_at_or_above": FOUND,
            "missed_at_or_below": MISSED,
            "chance": 0.5,
            "statistic": "ROC-AUC within a unit, over that unit's residues",
            "why_two_thresholds": "a unit where every method scores 0.7 is not a "
                                  "disagreement, and a single cut would call it "
                                  "one",
            "fixed_before_the_counts_were_read": True,
        },
        "methods_compared": {
            "ours": "the deployed counting field, per-unit ROC-AUC averaged over "
                    "the splits in which the unit sat on the pick side",
            "p2rank": "P2Rank 2.5.1 on the same 770 receptors, per-unit ROC-AUC "
                      "frozen in " + str(P2RANK_TRAIN.relative_to(ROOT)),
            "pocketminer": "the published PocketMiner network, per-residue "
                           "predictions on disk under "
                           "data/baselines/pocketminer_train/",
            "plmnn_absent": "pLM-NN has never been run on the training fold; it "
                            "is a five-hour encoder pass rather than a lookup. "
                            "Every count here is against two baselines, not "
                            "three, and the official-fold version that includes "
                            "it is a read of the held-out set and needs a plan "
                            "and an index first",
        },
        "n_units_compared": len(shared),
        "n_recovered": len(rec),
        "n_recovered_with_a_parsing_caveat": len(caveat),
        "n_mirror": len(mir),
        "asymmetry": {
            "recovered_minus_mirror": len(rec) - len(mir),
            "ratio": (round(len(rec) / len(mir), 2) if mir else None),
            "how_to_read": "the count in our favour means nothing on its own. "
                           "Three methods of similar average accuracy disagree "
                           "in both directions, and only the difference between "
                           "the two counts is evidence",
        },
        "what_a_recovery_is_not": "a claim that a pocket exists which the "
                                  "baselines cannot see, that any site is "
                                  "druggable, or that anything here is clinical. "
                                  "ROC-AUC within a unit is a statement about "
                                  "the ranking of that unit's residues",
        "reproduction_check": {
            "max_absolute_difference_from_frozen_per_split": round(repro, 6),
            "reproduces_the_deployed_detector": True,
        },
        "threshold_ladder": ladder,
        "why_a_ladder": "the headline pair is strict and gives a small count. "
                        "Reporting the whole ladder rather than the setting "
                        "with the largest number is the difference between a "
                        "measurement and a search: if the asymmetry appears at "
                        "one setting only it is an artefact of that setting",
        "recovered": rec[:N_LISTED],
        "recovered_but_the_file_is_parsed_differently": caveat,
        "why_those_are_separated": "a win on a chain the three methods parse "
                                   "differently is a win about parsing. Each "
                                   "unit here fails at least one of: "
                                   "PocketMiner's featurisation verified "
                                   "against the authors' own tensor, no residue "
                                   "dropped as a conformer copy, no resseq "
                                   "collision from an insertion code. They are "
                                   "reported and not counted",
        "mirror": mir[:N_LISTED],
        "n_listed": N_LISTED,
        "per_unit": triples,
        "why_the_full_table_is_here": "so that a different threshold, or a "
                                      "different question about these three "
                                      "methods, needs no re-run and no second "
                                      "opinion about what the numbers were",
    }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    if a.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    print(f"\n  {len(shared)} units compared against P2Rank and PocketMiner")
    print(f"  ours >= {FOUND} and both baselines <= {MISSED}: "
          f"{len(rec)} units")
    print(f"  both baselines >= {FOUND} and ours <= {MISSED}: "
          f"{len(mir)} units   <- the mirror, which is the honest denominator")
    print(f"\n  {'found>=':>8s} {'missed<=':>9s} {'ours':>6s} {'mirror':>7s} "
          f"{'diff':>6s}")
    for r in ladder:
        print(f"  {r['found_at_or_above']:8.2f} {r['missed_at_or_below']:9.2f} "
              f"{r['n_recovered']:6d} {r['n_mirror']:7d} {r['difference']:+6d}")
    if rec:
        print(f"\n  {'unit':10s} {'res':>5s} {'cryptic':>8s} {'ours':>7s} "
              f"{'p2rank':>7s} {'pocketm':>8s} {'margin':>7s}")
        for r in rec[:12]:
            print(f"  {r['unit']:10s} {r['n_residues']:5d} {r['n_cryptic']:8d} "
                  f"{r['ours']:7.3f} {r['p2rank']:7.3f} "
                  f"{r['pocketminer']:8.3f} "
                  f"{r['margin_over_the_better_baseline']:7.3f}")
    if mir:
        print(f"\n  the mirror, worst first:")
        for r in mir[:6]:
            print(f"  {r['unit']:10s} {r['n_residues']:5d} {r['n_cryptic']:8d} "
                  f"{r['ours']:7.3f} {r['p2rank']:7.3f} "
                  f"{r['pocketminer']:8.3f} "
                  f"{r['margin_over_the_better_baseline']:7.3f}")
    if a.write:
        print(f"\nwrote {out.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
