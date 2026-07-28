#!/usr/bin/env python3
"""Fix the summary statistic on the training partition, before the fold is read.

The problem this exists to remove
---------------------------------
The table field leads P2Rank by \\TabAucD{} ROC-AUC with a 95% interval that
contains zero, and the manuscript says so. A reader is entitled to ask why the
mean of the per-unit differences is the functional being reported at all. It is
a bad functional for this comparison and the case studies say why: the two
methods succeed on different proteins -- 25 units located by both, 24 by the
field alone, 13 by P2Rank alone, 130 by neither -- so the paired differences are
heavy-tailed and their mean is the least stable thing one could summarise them
with.

Switching functional after seeing the held-out numbers is not available. The
choice would then carry the maximum of however many were considered, which is
the defect this repository exists to exclude by construction. So the choice is
made here, on the training partition, and the held-out fold is read once
afterwards for whatever this file names.

Why P2Rank had to be run on the training fold first
----------------------------------------------------
It never had been. Every statistical statement about the margin was therefore
unpreregistrable, because the only place both methods had per-unit numbers was
the fold whose reading is being budgeted. ``tools/run_p2rank_on_train.py``
scores P2Rank on the 770 training receptors and costs nothing from that budget.

Both arms must be out of sample
-------------------------------
The table field's cells are counted on the training fold, so comparing it to
P2Rank there would put an in-sample arm against an out-of-sample one and inflate
the difference. The comparison here is on the pick half of a cluster-disjoint
split: the field is compiled on the fit half only, P2Rank never fits anything,
and neither arm has seen a pick-half residue.

What is being chosen, and what it is not
----------------------------------------
Each candidate licenses a different sentence, and they are not interchangeable:

    mean         on average, by how much
    median       on a typical structure, by how much
    trimmed      by how much, discounting the tails both methods disagree on
    win rate     on what fraction of structures, at all
    stratified   on average, holding chain length fixed

Choosing the one with the most power is not choosing the one most likely to
clear zero for a fixed claim; it is choosing which claim this fold is able to
resolve. The artifact records the sentence beside the number so that the
manuscript cannot quietly upgrade a win rate into a margin.

Power is measured by subsampling the pick half down to the size of the held-out
fold, and calibration by re-running the same procedure on sign-flipped
differences, where nothing should clear zero more than five percent of the time.
A candidate that is not calibrated is not eligible however powerful it looks.

Power at the training effect size is not the number that matters
----------------------------------------------------------------
The margin on the pick half is far larger than the one already published on the
held-out fold, and the reason is visible in the two arms: the field scores about
the same on both, while P2Rank does markedly better on the held-out receptors
than on the training ones. Power measured at the pick-half effect is therefore
optimistic, and several candidates saturate at it, which is no way to choose
between them.

So power is measured along a curve: the differences are shifted, keeping their
spread, so that the effect is a half, a quarter and an eighth of what the pick
half shows. Sweeping a range rather than plugging in the published held-out
margin keeps the choice from being conditioned on the number it will be used to
test. A candidate is chosen only if it dominates that whole curve.

Ties are broken by how much the claim says, not by the size of the point
estimate, which lives on a different scale for each candidate and cannot be
compared across them. A margin in ROC-AUC says more than a win rate, and a
lightly trimmed margin says more than a heavily trimmed one.

Usage:
    PYTHONPATH=src:tools python3.12 tools/preregister_statistic.py
    PYTHONPATH=src:tools python3.12 tools/preregister_statistic.py --check
"""
from __future__ import annotations

import argparse
import gc
import json

import numpy as np

from pocket_bench.methods.table_bank import (
    cell_offsets, chain_digits, compile_cells, integer_fanout, partition_tables,
    score,
)
from pocket_bench.metrics import roc_auc
from pocket_bench.paths import ROOT

from counterattack_ridge import spread_matched_gate
from counterattack_select import SEED

WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
P2TRAIN = ROOT / "results/architecture_sweep/P2RANK_TRAIN_FOLD.json"
PUBLISHED = (ROOT / "results/official_fold"
             / "OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json")
OUT = ROOT / "results/architecture_sweep/PREREGISTERED_STATISTIC.json"

SCHEMA = "geoaudit.preregistered_statistic.v1"
PAIR_ROUNDS = 16
RIDGE = 0.03
CAP = 32
GATE_RADIUS = 18.0
GATE_WEIGHT = 1.0
N_TEST_UNITS = 192          # the size the held-out fold will present
N_SUBSAMPLES = 200
N_BOOT = 2000
BOOT_SEED = 20260725
SHRINKS = (1.0, 0.5, 0.25, 0.125)
TYPE_I_CEILING = 0.08

# How much each candidate's sentence asserts, used only to break ties in power.
# A location margin in ROC-AUC is a stronger statement than a bare win rate, and
# the less of the sample a margin discards the more it asserts about the sample.
CLAIM_STRENGTH = {
    "mean": 5,
    "stratified_by_length": 4,
    "median": 3,
    "trimmed10": 2,
    "trimmed20": 1,
    "win_rate": 0,
}

CLAIMS = {
    "mean": "the field leads P2Rank on average by",
    "median": "on a typical structure the field leads P2Rank by",
    "trimmed10": "discounting the 10% tails, the field leads P2Rank by",
    "trimmed20": "discounting the 20% tails, the field leads P2Rank by",
    "win_rate": "the field beats P2Rank on a fraction of structures exceeding "
                "one half by",
    "stratified_by_length": "holding chain length fixed, the field leads "
                            "P2Rank on average by",
}


# ------------------------------------------------------------------ statistics
def _mean(d, _s):
    return d.mean(-1)


def _median(d, _s):
    return np.median(d, axis=-1)


def _trimmed(frac):
    def f(d, _s):
        k = int(round(frac * d.shape[-1]))
        x = np.sort(d, axis=-1)
        return x[..., k:d.shape[-1] - k].mean(-1) if k else x.mean(-1)
    return f


def _win_rate(d, _s):
    """Excess over one half, so that zero is the null for every candidate."""
    return (d > 0).mean(-1) - 0.5


def _stratified(d, strata):
    """Mean of the within-stratum means. Chain length is known before scoring,
    so conditioning on it uses no label and no method output."""
    out = np.zeros(d.shape[:-1])
    seen = 0
    for s in np.unique(strata):
        m = strata == s
        if m.sum() == 0:
            continue
        out = out + d[..., m].mean(-1)
        seen += 1
    return out / max(seen, 1)


STATISTICS = {
    "mean": _mean,
    "median": _median,
    "trimmed10": _trimmed(0.10),
    "trimmed20": _trimmed(0.20),
    "win_rate": _win_rate,
    "stratified_by_length": _stratified,
}


# ----------------------------------------------------------------- the arms
def _per_unit_auc(sc, y, n_res_per, units):
    out = {}
    off = 0
    for u, n in zip(units, n_res_per):
        n = int(n)
        s, t = sc[off:off + n], y[off:off + n]
        off += n
        if t.sum() == 0 or t.sum() == n:
            out[u] = None
            continue
        out[u] = roc_auc(list(s), list(t))
    return out


def field_on_pick_half():
    """Compile the table field on the fit half, score the pick half."""
    z = np.load(WIDE, allow_pickle=False)
    X, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    clusters = sorted({cluster_of[u] for u in units})
    rng = np.random.default_rng(SEED)
    rng.shuffle(clusters)
    fit_clusters = set(clusters[:len(clusters) // 2])
    is_fit = np.array([cluster_of[u] in fit_clusters for u in units])

    row = np.repeat(np.arange(len(n_res)), n_res)
    fit, pick = is_fit[row], ~is_fit[row]
    n_fit_per = np.array([n for n, f in zip(n_res, is_fit) if f])
    n_pick_per = np.array([n for n, f in zip(n_res, is_fit) if not f])
    pick_units = [u for u, f in zip(units, is_fit) if not f]
    print(f"{len(units)} train units / {len(clusters)} clusters -> "
          f"fit {int(is_fit.sum())}, pick {len(pick_units)}", flush=True)

    Dfit = chain_digits(X[fit], n_fit_per)
    Dpick = chain_digits(X[pick], n_pick_per)
    yfit, ypick = y[fit], y[pick]
    ctr_pick = ctr[pick]
    del X, z
    gc.collect()

    tables = partition_tables(Dfit.shape[1], 2, PAIR_ROUNDS, SEED)
    offsets = cell_offsets(tables)
    frac, tot = compile_cells(Dfit, yfit, tables, offsets)
    m = integer_fanout(Dfit, yfit, tables, offsets, frac, RIDGE, CAP)
    print(f"{len(tables)} tables, {int((m != 0).sum())} carrying a fan-out",
          flush=True)
    S = score(Dpick, tables, offsets, frac, m)
    F = spread_matched_gate(S, ctr_pick, n_pick_per, GATE_RADIUS, GATE_WEIGHT)
    aucs = _per_unit_auc(F, ypick, n_pick_per, pick_units)
    lengths = {u: int(n) for u, n in zip(pick_units, n_pick_per)}
    return aucs, lengths, len(units), int(is_fit.sum())


def p2rank_on_pick_half(pick_units):
    doc = json.loads(P2TRAIN.read_text())
    if doc.get("test_fold_touched") is not False:
        raise SystemExit("the P2Rank training run does not declare the test "
                         "fold untouched")
    by_unit = {r["unit_id"]: r.get("residue_auc") for r in doc["rows"]
               if r.get("status") == "OK"}
    return {u: by_unit.get(u) for u in pick_units}, doc


# --------------------------------------------------------------- power study
def _ci_excludes_zero(d, strata, fn, rng):
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    vals = fn(d[idx], strata)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (lo > 0) or (hi < 0), float(np.mean(vals))


def _shrink(d, s):
    """Move the effect to ``s`` times its size without touching the spread."""
    return d - (1.0 - s) * float(d.mean())


def study(d, strata):
    rng = np.random.default_rng(BOOT_SEED)
    n = len(d)
    take = min(N_TEST_UNITS, n)
    subs = [rng.choice(n, size=take, replace=False) for _ in range(N_SUBSAMPLES)]
    flips = [rng.choice([-1.0, 1.0], size=take) for _ in range(N_SUBSAMPLES)]

    rows = []
    for name, fn in STATISTICS.items():
        full = float(fn(d[None, :], strata)[0])
        curve = {}
        for s in SHRINKS:
            ds = _shrink(d, s)
            hits = sum(_ci_excludes_zero(ds[i], strata[i], fn, rng)[0]
                       for i in subs)
            curve[f"{s:g}"] = round(hits / len(subs), 3)
        nulls = sum(_ci_excludes_zero(d[i] * fl, strata[i], fn, rng)[0]
                    for i, fl in zip(subs, flips))
        false_pos = nulls / len(subs)
        mean_power = float(np.mean(list(curve.values())))
        rows.append({
            "statistic": name,
            "claim": CLAIMS[name],
            "claim_strength": CLAIM_STRENGTH[name],
            "point_on_pick_half": round(full, 6),
            "power_by_effect_shrink": curve,
            "power_at_n": curve["1"],
            "mean_power_over_curve": round(mean_power, 3),
            "false_positive_rate_under_sign_flip": round(false_pos, 3),
            "calibrated": bool(false_pos <= TYPE_I_CEILING),
        })
        shown = "  ".join(f"{k}:{v:.2f}" for k, v in curve.items())
        print(f"  {name:22s} point {full:+.4f}  power[{shown}]  "
              f"type-I {false_pos:.3f}"
              f"{'' if false_pos <= TYPE_I_CEILING else '  NOT CALIBRATED'}",
              flush=True)
    return rows


def forecast(d, strata, winner):
    """What the chosen statistic is likely to do, said before the fold is read.

    The shrink used here is the one that maps the pick-half margin onto the
    margin already published for this comparison, so unlike the selection grid
    it does condition on a held-out number -- one already in the manuscript.
    That is why it is a forecast and not an input to the choice: it exists so
    that a read which fails to clear zero cannot afterwards be described as a
    surprise, and one that clears it cannot be described as decisive on its own.
    """
    pub = json.loads(PUBLISHED.read_text())
    delta = (pub["metrics"]["residue_auc"]["paired_vs_baseline"]
             ["table_field"]["delta_point"])
    s = float(delta) / float(d.mean())
    fn = STATISTICS[winner["statistic"]]
    rng = np.random.default_rng(BOOT_SEED)
    ds = _shrink(d, s)
    take = min(N_TEST_UNITS, len(d))
    hits = 0
    for _ in range(N_SUBSAMPLES):
        i = rng.choice(len(d), size=take, replace=False)
        hits += bool(_ci_excludes_zero(ds[i], strata[i], fn, rng)[0])
    power = hits / N_SUBSAMPLES
    print(f"\nforecast: at the published margin ({delta:+.4f}, {s:.2f}x the "
          f"pick-half effect) the preregistered statistic clears zero "
          f"{power:.0%} of the time", flush=True)
    return {
        "published_mean_margin_on_test_fold": round(float(delta), 6),
        "implied_shrink": round(s, 4),
        "expected_power": round(power, 3),
        "conditions_on_a_held_out_number": True,
        "used_for_selection": False,
        "reading": ("the read is more likely than not to leave the interval "
                    "across zero, and that outcome is predicted here rather "
                    "than explained afterwards")
        if power < 0.5 else
        ("the read is more likely than not to clear zero, which is a forecast "
         "and not a result"),
    }


def _select(rows):
    """Most power along the shrink curve; ties go to the stronger claim.

    Point estimates are never compared across candidates: a win rate and a
    ROC-AUC margin are not on the same scale and the larger number is not the
    better one.
    """
    elig = [r for r in rows if r["calibrated"]]
    if not elig:
        raise SystemExit("no candidate is calibrated; nothing can be "
                         "preregistered from this run")
    top = max(r["mean_power_over_curve"] for r in elig)
    close = [r for r in elig if top - r["mean_power_over_curve"] <= 0.02]
    return max(close, key=lambda r: r["claim_strength"])


def build() -> dict:
    field, lengths, n_train_units, n_fit = field_on_pick_half()
    pick_units = [u for u in field if field[u] is not None]
    p2, p2doc = p2rank_on_pick_half(pick_units)
    shared = [u for u in pick_units if p2.get(u) is not None]
    d = np.array([field[u] - p2[u] for u in shared], dtype=np.float64)
    L = np.array([lengths[u] for u in shared], dtype=np.float64)
    edges = np.quantile(L, [0.25, 0.5, 0.75])
    strata = np.digitize(L, edges)
    print(f"\npaired on {len(shared)} pick-half units, both arms out of sample")
    print(f"  field  {np.mean([field[u] for u in shared]):.4f}")
    print(f"  p2rank {np.mean([p2[u] for u in shared]):.4f}")
    print(f"\npower at n={min(N_TEST_UNITS, len(d))} over {N_SUBSAMPLES} "
          f"subsamples, {N_BOOT} bootstrap resamples each, at effect sizes "
          f"{', '.join(f'{s:g}x' for s in SHRINKS)}:")
    rows = study(d, strata)
    winner = _select(rows)
    fore = forecast(d, strata, winner)
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "question": "which functional of the paired differences should the "
                    "manuscript report, decided before the fold is read",
        "comparison": {
            "arm_a": "table field, compiled on the fit half only",
            "arm_b": f"p2rank {p2doc['p2rank_version']}, fits nothing",
            "population": "pick half of a cluster-disjoint split of the "
                          "training fold",
            "both_arms_out_of_sample": True,
            "n_train_units": n_train_units,
            "n_fit_units": n_fit,
            "n_paired_units": len(shared),
            "mean_field": round(float(np.mean([field[u] for u in shared])), 6),
            "mean_p2rank": round(float(np.mean([p2[u] for u in shared])), 6),
        },
        "procedure": {
            "n_test_units_simulated": min(N_TEST_UNITS, len(d)),
            "n_subsamples": N_SUBSAMPLES,
            "n_bootstrap": N_BOOT,
            "seed": BOOT_SEED,
            "power": "fraction of subsamples whose 95% bootstrap interval "
                     "excludes zero",
            "effect_shrinks": list(SHRINKS),
            "shrink_model": "the differences are shifted so the effect is a "
                            "fraction of the pick-half one, spread unchanged; "
                            "a range is swept rather than the published "
                            "held-out margin plugged in, so that the choice is "
                            "not conditioned on the number it will test",
            "calibration": "the same fraction after flipping the sign of each "
                           "unit's difference at random, which must not exceed "
                           f"{TYPE_I_CEILING}",
            "stratification_covariate": "chain length, known before either "
                                        "method is run",
            "selection_rule": "highest mean power over the shrink curve among "
                              "calibrated candidates; ties within 0.02 go to "
                              "the statistic whose claim asserts more. Point "
                              "estimates are not compared across candidates "
                              "because they are not on a common scale",
        },
        "candidates": rows,
        "preregistered": winner,
        "forecast": fore,
        "commitment": (
            f"the manuscript will report {winner['statistic']} of the paired "
            f"per-unit differences on the held-out fold, and the sentence it "
            f"licenses is: {winner['claim']} ... . This was fixed before the "
            f"fold was read for it."),
        "the_mean_is_still_reported": (
            "the preregistered statistic is the one inference is drawn from, "
            "not a replacement for the mean. The mean of the paired "
            "differences stays in the manuscript beside it, because it is what "
            "the literature reports and because dropping the underpowered "
            "functional once a better-powered one is in hand is the same "
            "selection this file exists to prevent."),
    }


def _report(doc: dict) -> None:
    w = doc["preregistered"]
    print(f"\npreregistered: {w['statistic']}  "
          f"(mean power {w['mean_power_over_curve']:.2f} over the curve, "
          f"type-I {w['false_positive_rate_under_sign_flip']:.3f})")
    print(f"  claim: {w['claim']} ...")


def audit() -> int:
    if not OUT.exists():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    doc = json.loads(OUT.read_text())
    bad = []
    if doc.get("schema") != SCHEMA:
        bad.append(f"schema is {doc.get('schema')!r}")
    if doc.get("reads_test_fold") is not False:
        bad.append("reads_test_fold is not false")
    if not doc.get("comparison", {}).get("both_arms_out_of_sample"):
        bad.append("the comparison does not declare both arms out of sample")
    cands = doc.get("candidates") or []
    if len(cands) < 4:
        bad.append("fewer candidates than the file defines")
    for c in cands:
        fp = c["false_positive_rate_under_sign_flip"]
        if c["calibrated"] != (fp <= TYPE_I_CEILING):
            bad.append(f"{c['statistic']}: calibrated flag does not follow "
                       f"from its own type-I rate")
        if c["statistic"] not in CLAIMS:
            bad.append(f"{c['statistic']}: no claim recorded")
        elif c["claim"] != CLAIMS[c["statistic"]]:
            bad.append(f"{c['statistic']}: claim text has drifted from the "
                       f"sentence the statistic licenses")
        if c.get("claim_strength") != CLAIM_STRENGTH.get(c["statistic"]):
            bad.append(f"{c['statistic']}: claim strength has drifted from the "
                       f"ordering the selection rule uses")
        curve = c.get("power_by_effect_shrink") or {}
        if sorted(curve) != sorted(f"{s:g}" for s in SHRINKS):
            bad.append(f"{c['statistic']}: power curve is not over the shrink "
                       f"grid the file defines")
        elif abs(np.mean(list(curve.values()))
                 - c["mean_power_over_curve"]) > 5e-4:
            bad.append(f"{c['statistic']}: mean power does not average its own "
                       f"curve")
    fore = doc.get("forecast") or {}
    if not fore:
        bad.append("no forecast is recorded, so a read that fails to resolve "
                   "could be explained after the fact")
    elif fore.get("used_for_selection") is not False:
        bad.append("the forecast is marked as having fed the selection, which "
                   "would condition the choice on a held-out number")
    if not doc.get("the_mean_is_still_reported"):
        bad.append("the artifact does not commit to reporting the mean beside "
                   "the preregistered statistic")
    pre = doc.get("preregistered")
    if not pre:
        bad.append("nothing is preregistered")
    elif not bad:
        best = _select(cands)
        if pre["statistic"] != best["statistic"]:
            bad.append(f"the preregistered statistic is {pre['statistic']!r}, "
                       f"but the selection rule applied to the recorded "
                       f"candidates returns {best['statistic']!r}")
    for b in bad:
        print(f"FAIL {OUT.relative_to(ROOT)}: {b}")
    if bad:
        return 1
    print(f"OK {OUT.relative_to(ROOT)}: {len(cands)} candidates, "
          f"{pre['statistic']} preregistered at mean power "
          f"{pre['mean_power_over_curve']:.2f} over the shrink curve, "
          f"test fold unread")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        return audit()
    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    _report(doc)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return audit()


if __name__ == "__main__":
    raise SystemExit(main())
