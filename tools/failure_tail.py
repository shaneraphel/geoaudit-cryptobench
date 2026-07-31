#!/usr/bin/env python3.12
"""Where the detector fails badly, and whether anything observable predicts it.

The measurement that asks for this
----------------------------------
On the official fold the mean paired difference against P2Rank is +0.0058 and
crosses zero, while the 20 % trimmed mean is +0.0281 [+0.0117, +0.0427] at
p=0.002. Those two numbers are the same 192 differences summarised twice, and
the gap between them is entirely the tail: the losses average -0.2319 against
wins of +0.1757. The detector is not behind on the typical structure. It is
ahead on the typical structure and catastrophically behind on a few.

That also explains the two claims this repository withdrew. F1 fell from
+0.0315 to +0.0140 once P2Rank was binarised the way we binarise ourselves, and
MCC went unresolved once the bootstrap was paired on the shared set. Both are
threshold metrics, both are dominated by the structures where a method is
badly wrong rather than slightly wrong, and both are exactly what a heavy
left tail damages.

So the tail is not a footnote about robustness. It is where the withdrawn
claims went, and it is the one axis of this architecture with no measurement
against it: eight readout parameters and five wire families have been swept,
and every one of them was scored by a *mean* over units, which is the summary
the tail is invisible in.

What this tool does, and what it refuses to do
----------------------------------------------
Per-unit ROC-AUC for the deployed field on the pick half of each
cluster-disjoint split, so every training unit is scored several times on halves
it was not compiled on. Then it asks whether anything observable separates the
worst units from the rest: chain length, the number of cryptic residues, their
fraction, and the fold's own class balance.

It reads no test-fold unit and no external unit. The official fold's per-unit
numbers exist and are not touched: characterising our failures on data the
method will later be judged on is how a "robustness fix" becomes a fit to the
held-out set.

A covariate that separates the tail is not a fix
------------------------------------------------
If chain length predicts failure, that is a fact about the fold and about the
detector together, and the honest reading depends on which. A gate that refuses
short chains would raise the mean without the detector having improved at all,
which is why the artifact reports the tail's *share of units* beside its share
of the deficit: a covariate that covers 3 % of units and 40 % of the shortfall
is a lead, and one that covers 40 % of units is a description of the fold.
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

SCHEMA = "geoaudit.failure_tail.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
OUT = ROOT / "results/architecture_sweep/FAILURE_TAIL.json"

# The trim the paper's secondary summary uses, so "the tail" here means the same
# units that summary discards rather than a fraction chosen for this tool.
TRIM = 0.20


def auc_per_unit(sc: np.ndarray, y: np.ndarray, n_per: np.ndarray) -> np.ndarray:
    """ROC-AUC within each unit, NaN where the unit has one class only.

    The rank form, so it is exact and needs no threshold: with ``r`` the ranks
    of the scores within the unit, AUC = (sum of positive ranks - n1(n1+1)/2)
    / (n1 n0). Ties share a mid-rank, which is what makes this the Mann-Whitney
    statistic rather than an approximation of it.
    """
    out = np.full(len(n_per), np.nan)
    off = 0
    for i, n in enumerate(n_per):
        n = int(n)
        s, yy = sc[off:off + n], y[off:off + n]
        off += n
        n1 = int((yy == 1).sum())
        n0 = n - n1
        if n1 == 0 or n0 == 0:
            continue
        order = np.argsort(s, kind="stable")
        r = np.empty(n, dtype=np.float64)
        srt = s[order]
        i0 = 0
        while i0 < n:
            j = i0
            while j + 1 < n and srt[j + 1] == srt[i0]:
                j += 1
            r[order[i0:j + 1]] = 0.5 * (i0 + j) + 1.0
            i0 = j + 1
        out[i] = (r[yy == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return out


def summarise(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return {
        "n": int(len(x)),
        "mean": round(float(x.mean()), 6),
        "median": round(float(np.median(x)), 6),
        "p05": round(float(np.percentile(x, 5)), 6),
        "p25": round(float(np.percentile(x, 25)), 6),
        "p75": round(float(np.percentile(x, 75)), 6),
        "min": round(float(x.min()), 6),
        "max": round(float(x.max()), 6),
    }


def separation(lo: np.ndarray, hi: np.ndarray, name: str) -> dict:
    """Does a covariate separate the tail from the rest, and by how much.

    Reported as a rank statistic rather than a difference of means, because
    chain length and positive count are both heavily skewed and a difference of
    means on them says more about the skew than about the separation. The
    common-language effect size is the probability that a randomly drawn tail
    unit exceeds a randomly drawn other one, which is 0.5 under no effect.
    """
    lo = lo[~np.isnan(lo)]
    hi = hi[~np.isnan(hi)]
    if len(lo) == 0 or len(hi) == 0:
        return {"covariate": name, "measurable": False}
    both = np.concatenate([lo, hi])
    order = np.argsort(both, kind="stable")
    r = np.empty(len(both))
    srt = both[order]
    i0 = 0
    while i0 < len(both):
        j = i0
        while j + 1 < len(both) and srt[j + 1] == srt[i0]:
            j += 1
        r[order[i0:j + 1]] = 0.5 * (i0 + j) + 1.0
        i0 = j + 1
    n1 = len(lo)
    u = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return {
        "covariate": name,
        "measurable": True,
        "tail_median": round(float(np.median(lo)), 4),
        "rest_median": round(float(np.median(hi)), 4),
        "prob_tail_exceeds_rest": round(float(u / (len(lo) * len(hi))), 4),
        "reading": "0.5 is no separation; far from 0.5 in either direction "
                   "means the covariate carries information about which units "
                   "fail",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(OUT))
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    cdoc = json.loads(COUNTING.read_text())
    n_splits = a.splits or int(cdoc["protocol"]["n_splits"])

    z = np.load(WIDE, allow_pickle=False)
    y, n_res, ctr = z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    z.close()
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}

    D = digit_cache.load(n_res)
    n_old = int(D.shape[1])
    tabs = partition_tables(n_old, TABLE_WIDTH, PARTITION_ROUNDS, PARTITION_SEED)
    offs = cell_offsets(tabs)
    row = np.repeat(np.arange(len(n_res)), n_res)

    # Per-unit AUC, accumulated over every split in which the unit sat on the
    # pick side. A unit scored once could be unlucky; a unit that fails on all
    # six of its appearances is a property of the unit.
    seen = np.zeros(len(n_res), dtype=np.int64)
    total = np.zeros(len(n_res), dtype=np.float64)
    worst = np.full(len(n_res), np.nan)
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
        sc = apply_gate(score(D[pick], tabs, offs, frac, mult), ctr[pick],
                        n_pick)
        per = auc_per_unit(sc, y[pick], n_pick)
        idx = np.flatnonzero(~is_fit)
        ok = ~np.isnan(per)
        seen[idx[ok]] += 1
        total[idx[ok]] += per[ok]
        w = worst[idx[ok]]
        worst[idx[ok]] = np.where(np.isnan(w), per[ok], np.minimum(w, per[ok]))
        print(f"  split {s + 1}/{n_splits}  mean per-unit "
              f"{np.nanmean(per):.4f}  {time.perf_counter() - t0:.0f}s",
              flush=True)

    scored = seen > 0
    mean_auc = np.full(len(n_res), np.nan)
    mean_auc[scored] = total[scored] / seen[scored]

    n_pos = np.zeros(len(n_res), dtype=np.int64)
    off = 0
    for i, n in enumerate(n_res):
        n = int(n)
        n_pos[i] = int((y[off:off + n] == 1).sum())
        off += n
    frac_pos = np.where(n_res > 0, n_pos / np.maximum(n_res, 1), 0.0)

    vals = mean_auc[scored]
    k = max(1, int(round(TRIM * len(vals))))
    cut = float(np.sort(vals)[k - 1])
    is_tail = scored & (mean_auc <= cut)
    is_rest = scored & ~is_tail

    # How much of the shortfall from the pooled mean the tail accounts for, and
    # over what share of units. A covariate that covers a tenth of the units and
    # a half of the shortfall is a lead; one that covers half of both is a
    # restatement of the trim.
    pooled = float(np.nanmean(mean_auc[scored]))
    deficit_tail = float(np.sum(pooled - mean_auc[is_tail]))
    deficit_all = float(np.sum(np.abs(pooled - mean_auc[scored])))

    # Strata of positive count, chosen as roughly equal-sized groups rather
    # than round numbers, so that no boundary was picked after seeing which one
    # made the effect look largest.
    edges = [0] + [int(np.percentile(n_pos[scored], q))
                   for q in (25, 50, 75)] + [int(n_pos.max()) + 1]
    edges = sorted(set(edges))
    strata = []
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        m = scored & (n_pos >= lo_e) & (n_pos < hi_e)
        if m.sum() < 10:
            continue
        v = mean_auc[m]
        # The null sampling standard error of a within-unit AUC at this stratum's
        # median n1 and n0, which is what an unbiased detector would still show.
        n1m = float(np.median(n_pos[m]))
        n0m = float(np.median(n_res[m] - n_pos[m]))
        se_null = float(np.sqrt((n1m + n0m + 1.0) / (12.0 * n1m * n0m)))
        strata.append({
            "n_cryptic_from": int(lo_e),
            "n_cryptic_below": int(hi_e),
            "n_units": int(m.sum()),
            "median_n_cryptic": n1m,
            "mean_auc": round(float(np.nanmean(v)), 4),
            "median_auc": round(float(np.nanmedian(v)), 4),
            "sd_of_auc": round(float(np.nanstd(v, ddof=1)), 4),
            "null_sampling_se_at_this_size": round(se_null, 4),
            "sd_over_null_se": round(float(np.nanstd(v, ddof=1) / se_null), 2),
            "n_below_one_half": int((v < 0.5).sum()),
            "share_below_one_half": round(float((v < 0.5).mean()), 4),
        })

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "whether anything observable separates the units the "
                    "detector fails badly on from the rest, on the training "
                    "fold only",
        "why_the_tail": (
            "the official fold's mean paired difference against P2Rank is "
            "+0.0058 and crosses zero while its 20% trimmed mean is +0.0281 at "
            "p=0.002. Those are the same 192 differences summarised twice and "
            "the gap between them is the tail. Both withdrawn claims, F1 and "
            "MCC, are threshold metrics that a heavy left tail damages"),
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "scored_on": "the pick half of each split, so no unit is scored by "
                         "a field compiled on it",
            "metric": "ROC-AUC within a unit, rank form, gate applied as "
                      "deployed",
            "units_appear": "once per split in which they fall on the pick "
                            "side, about half the splits each",
            "trim": TRIM,
        },
        "per_unit_auc": summarise(mean_auc[scored]),
        "worst_single_appearance": summarise(worst[scored]),
        "tail": {
            "definition": f"the worst {TRIM:.0%} of units by mean per-unit AUC",
            "cutoff_auc": round(cut, 6),
            "n_units": int(is_tail.sum()),
            "share_of_units": round(float(is_tail.sum() / scored.sum()), 4),
            "share_of_the_total_deviation": round(
                float(deficit_tail / deficit_all), 4) if deficit_all else None,
            "mean_auc_in_the_tail": round(float(np.nanmean(mean_auc[is_tail])), 6),
            "mean_auc_in_the_rest": round(float(np.nanmean(mean_auc[is_rest])), 6),
            "n_units_below_one_half": int((mean_auc[scored] < 0.5).sum()),
            "what_below_one_half_means": (
                "the field ranks cryptic residues below non-cryptic ones on "
                "that unit, which is worse than a coin toss and is the "
                "population a threshold metric is destroyed by"),
        },
        "covariates": [
            separation(n_res[is_tail].astype(float),
                       n_res[is_rest].astype(float), "chain_length"),
            separation(n_pos[is_tail].astype(float),
                       n_pos[is_rest].astype(float), "n_cryptic_residues"),
            separation(frac_pos[is_tail], frac_pos[is_rest],
                       "fraction_cryptic"),
        ],
        "consistency": {
            "n_units_scored": int(scored.sum()),
            "appearances_per_unit": summarise(seen[scored].astype(float)),
            "n_units_never_scored": int((~scored).sum()),
            "why_some_are_never_scored": (
                "a unit with no cryptic residue, or with every residue "
                "cryptic, has an undefined within-unit AUC and is skipped "
                "rather than counted as 0.5"),
        },
        "stratified_by_positive_count": strata,
        "why_stratify": (
            "n_cryptic_residues separates the tail more strongly than anything "
            "else here, and there are two readings of that which imply "
            "opposite work. Within-unit AUC over n1 positives has a sampling "
            "standard error of roughly sqrt((n1+n0+1)/(12 n1 n0)) under the "
            "null, which at n1=8 is several times its value at n1=40: a unit "
            "with few positives has a noisy AUC whatever the detector does. If "
            "the tail is that, it is a property of the metric and the "
            "detector is not failing on those units at all -- and the same "
            "arithmetic would explain why F1 and MCC, which are more sensitive "
            "at small n1 than AUC, were the two claims that did not survive. "
            "If instead the deficit persists within strata of n1, small "
            "pockets really are harder and that is a detector problem. The "
            "strata below decide it"),
        "worst_units": [
            {"unit": units[i],
             "mean_auc": round(float(mean_auc[i]), 4),
             "worst_appearance": round(float(worst[i]), 4),
             "n_residues": int(n_res[i]),
             "n_cryptic": int(n_pos[i]),
             "fraction_cryptic": round(float(frac_pos[i]), 4)}
            for i in np.argsort(np.where(scored, mean_auc, np.inf))[:25]
        ],
    }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    if a.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    t = doc["tail"]
    print(f"\n  {doc['per_unit_auc']['n']} units scored, mean per-unit AUC "
          f"{doc['per_unit_auc']['mean']:.4f}, median "
          f"{doc['per_unit_auc']['median']:.4f}")
    print(f"  worst {TRIM:.0%}: {t['n_units']} units at "
          f"{t['mean_auc_in_the_tail']:.4f} against "
          f"{t['mean_auc_in_the_rest']:.4f} for the rest")
    print(f"  {t['n_units_below_one_half']} units score below 0.5 -- worse "
          f"than a coin toss on their own residues")
    print(f"  the tail is {t['share_of_units']:.0%} of units and "
          f"{t['share_of_the_total_deviation']:.0%} of the total deviation")
    print()
    for c in doc["covariates"]:
        if c.get("measurable"):
            print(f"  {c['covariate']:20s} tail median {c['tail_median']:>9.3f}"
                  f"  rest {c['rest_median']:>9.3f}"
                  f"  P(tail>rest) {c['prob_tail_exceeds_rest']:.3f}")
    print("\n  by number of cryptic residues, which is the covariate that "
          "separates:")
    print(f"    {'n1 range':>12s} {'units':>6s} {'mean':>7s} {'median':>7s} "
          f"{'sd':>7s} {'null se':>8s} {'sd/se':>6s} {'<0.5':>6s}")
    for st in strata:
        print(f"    {st['n_cryptic_from']:5d}-{st['n_cryptic_below'] - 1:<6d} "
              f"{st['n_units']:6d} {st['mean_auc']:7.4f} "
              f"{st['median_auc']:7.4f} {st['sd_of_auc']:7.4f} "
              f"{st['null_sampling_se_at_this_size']:8.4f} "
              f"{st['sd_over_null_se']:6.2f} "
              f"{st['share_below_one_half']:6.1%}")
    if a.write:
        print(f"\nwrote {out.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
