#!/usr/bin/env python3
"""Whether a rival cryptic-site method also collapses on units with few sites.

The question this settles
-------------------------
``FAILURE_TAIL.json`` says the detector scores a mean per-unit ROC-AUC of 0.5991
on training units with fewer than ten cryptic residues and 0.8766 on units with
more than twenty-two, and that the small-site stratum carries 46% of the total
deviation. Arithmetically that stratum is the whole SOTA gap: it is a quarter of
the units, and lifting it from 0.60 to 0.75 would move the pooled mean by about
+0.03, where the deficit against pLM-NN is 0.0243.

Whether it is worth attacking depends on something nobody has measured. There
are two readings and they imply opposite work:

* the units are *hard for everyone* -- few positives in a large chain is an
  ambiguous labelling problem, and every method lands near chance there. Then
  the deficit is mostly irreducible, our 0.60 is roughly what a good method
  gets, and the operators to build are elsewhere.
* the units are *hard for us specifically*. Then there is a real 0.15-0.25 of
  headroom sitting in a quarter of the fold, and it is the largest identified
  target in this repository.

A per-unit ROC-AUC does not distinguish these on its own, because a unit with
seven positives out of three hundred has a large sampling error under the null
(``FAILURE_TAIL`` reports 0.111 for that stratum) and a mean of 0.60 over 188
units is not explained by that error but a single unit's 0.60 would be. The
distinguishing measurement is a second method on the same units.

Why PocketMiner and why the training fold
-----------------------------------------
PocketMiner is a cryptic-site predictor, not a general pocket finder, so it is
the right rival for this question, and its per-residue predictions on all 770
training units are already on disk from a run that produced
``POCKETMINER_TRAIN_SCORES.json``. Nothing here touches the test fold or the
external set, so this costs no read from the ledger and settles the targeting
question before any test-fold budget is spent on it.

The direction the biases cut
----------------------------
Two of them, both flattering PocketMiner on this fold:

* PocketMiner's training set was never clustered against CryptoBench's folds.
  Six entries match ours by exact PDB id, and exact matching is a floor rather
  than a homology check, so an unknown number of these training units may be
  homologous to structures it was fitted on. Our detector is fitted on
  cluster-disjoint halves of this same fold, so it sees no such advantage.
* PocketMiner drops residues it cannot featurise -- 6,929 across the fold --
  and a method scored on a subset of a chain is scored on an easier universe
  than a method scored on all of it. The headline below is therefore restricted
  to units where both methods cover the same number of residues, with the
  unrestricted number reported beside it.

Both push the same way, so *if PocketMiner still collapses on the small-site
stratum it does so despite being flattered*, and the "hard for everyone"
reading survives a hostile test rather than a friendly one.

The prediction, before the run
------------------------------
Committed in ``PREDICTION`` below: if the stratum is intrinsically hard,
PocketMiner's mean AUC there is also far below its own large-site mean, and the
paired gap between the methods is roughly flat across strata. If it is our
problem, PocketMiner's profile across strata is much flatter than ours and the
paired gap widens sharply toward small sites.
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
    PARTITION_ROUNDS,
    PARTITION_SEED,
    RIDGE,
    TABLE_WIDTH,
    apply_gate,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.baseline_by_stratum.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
TAIL = ROOT / "results/architecture_sweep/FAILURE_TAIL.json"
PM_SCORES = ROOT / "data/baselines/pocketminer_train"
PM_MANIFEST = ROOT / "results/baselines/POCKETMINER_TRAIN_SCORES.json"
LABELS = ROOT / "data/cryptobench_apo/train_labels"
OUT = ROOT / "results/architecture_sweep/BASELINE_BY_STRATUM.json"

PREDICTION = {
    "committed_before_the_run": True,
    "if_the_stratum_is_intrinsically_hard": "PocketMiner's mean AUC on the "
                                            "small-site stratum is also far "
                                            "below its own large-site mean, and "
                                            "the paired gap between the methods "
                                            "is roughly flat across strata",
    "if_the_deficit_is_ours": "PocketMiner's profile across strata is much "
                              "flatter than ours and the paired gap widens "
                              "sharply toward small sites",
    "what_would_redirect_the_work": "a paired gap that widens toward small "
                                    "sites names the small-site stratum as real "
                                    "headroom; a flat gap says the stratum is "
                                    "ambiguous for everyone and the operators "
                                    "should be built elsewhere",
}


def _resnum(x) -> int | None:
    """The trailing integer of a residue key, or None.

    Copied from ``pocketminer_train_operating_point.py`` rather than imported so
    that a change there cannot silently move this comparison's universe.
    """
    if isinstance(x, int):
        return x
    s, digits, negative = str(x), "", False
    for ch in reversed(s):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            negative = ch == "-"
            break
    if not digits:
        return None
    return -int(digits) if negative else int(digits)


def pocketminer_per_unit() -> dict[str, dict]:
    """PocketMiner's within-unit ROC-AUC on each training unit it scored."""
    out: dict[str, dict] = {}
    for f in sorted(PM_SCORES.glob("*.json")):
        unit = f.stem
        lab = LABELS / f"{unit}_labels.json"
        if not lab.is_file():
            raise SystemExit(f"no training labels for {unit}")
        d = json.loads(lab.read_text())
        pos = {r for r in (_resnum(r) for r in
                           (d.get("cryptic_residues")
                            or d.get("binding_residues") or []))
               if r is not None}
        raw = json.loads(f.read_text())["residue_scores"]
        by_res = {r: float(v) for r, v in
                  ((_resnum(k), v) for k, v in raw.items()) if r is not None}
        keys = sorted(by_res)
        if not keys:
            continue
        s = np.array([by_res[k] for k in keys], dtype=np.float64)
        y = np.array([k in pos for k in keys], dtype=bool)
        n_p = int(y.sum())
        if n_p == 0 or n_p == len(y):
            # No AUC is defined; recorded so the join can say why a unit is
            # absent rather than dropping it silently.
            out[unit] = {"auc": float("nan"), "n_res": len(y), "n_pos": n_p}
            continue
        auc = float(auc_per_unit(s, y.astype(np.int64),
                                 np.array([len(y)]))[0])
        out[unit] = {"auc": auc, "n_res": len(y), "n_pos": n_p}
    return out


def strata_edges() -> list[int]:
    st = json.loads(TAIL.read_text())["stratified_by_positive_count"]
    return [int(s["n_cryptic_from"]) for s in st] + [
        int(st[-1]["n_cryptic_below"])]


def paired_stats(d: np.ndarray) -> dict:
    d = d[~np.isnan(d)]
    n = len(d)
    if n < 2:
        return {"n_units": n}
    se = float(d.std(ddof=1) / np.sqrt(n))
    nb = int((d > 0).sum())
    p = sum(comb(n, i) for i in range(nb, n + 1)) / 2 ** n
    return {"n_units": n, "mean": round(float(d.mean()), 6),
            "median": round(float(np.median(d)), 6),
            "ci95": [round(float(d.mean() - 1.96 * se), 6),
                     round(float(d.mean() + 1.96 * se), 6)],
            "excludes_zero": bool(abs(d.mean()) > 1.96 * se),
            "n_units_ours_ahead": nb,
            "sign_test_p_one_sided": round(float(p), 6)}


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
    frozen = np.asarray(by_width[n_wires], dtype=float)[:n_splits]
    tabs = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                            PARTITION_SEED)
    offs = cell_offsets(tabs)
    row = np.repeat(np.arange(len(n_res)), n_res)

    n_pos = np.zeros(len(n_res), dtype=np.int64)
    off = 0
    for i, n in enumerate(n_res):
        n = int(n)
        n_pos[i] = int((y[off:off + n] == 1).sum())
        off += n

    seen = np.zeros(len(n_res), dtype=np.int64)
    total = np.zeros(len(n_res), dtype=np.float64)
    per_split: list[float] = []
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
        gated = apply_gate(score(D[pick], tabs, offs, frac, mult),
                           ctr[pick], n_pick)
        per = auc_per_unit(gated, y[pick], n_pick)
        idx = np.flatnonzero(~is_fit)
        ok = ~np.isnan(per)
        seen[idx[ok]] += 1
        total[idx[ok]] += per[ok]
        per_split.append(float(np.nanmean(per)))
        print(f"  split {s + 1}/{n_splits}  ours {per_split[-1]:.4f}  "
              f"frozen {frozen[s]:.4f}  {time.perf_counter() - t0:.0f}s",
              flush=True)

    repro = float(np.abs(np.asarray(per_split) - frozen).max())
    if repro >= 5e-4:
        raise SystemExit(
            f"our recomputed per-split means differ from the frozen ones by "
            f"{repro:.2e}; the rival would be compared against something that "
            f"is not the deployed detector")

    ours = np.full(len(n_res), np.nan)
    sc = seen > 0
    ours[sc] = total[sc] / seen[sc]

    pm = pocketminer_per_unit()

    # The two methods must be scored on the same residues before their per-unit
    # numbers can be subtracted. Agreement is checked rather than assumed: a
    # mismatch in the positive count would also move a unit between strata.
    rows, mism_res, mism_pos, absent = [], [], [], []
    for i, u in enumerate(units):
        if not sc[i]:
            continue
        r = pm.get(u)
        if r is None:
            absent.append(u)
            continue
        same_res = int(r["n_res"]) == int(n_res[i])
        same_pos = int(r["n_pos"]) == int(n_pos[i])
        if not same_res:
            mism_res.append(u)
        if not same_pos:
            mism_pos.append(u)
        rows.append({"unit": u, "i": i, "ours": float(ours[i]),
                     "pm": float(r["auc"]), "n_pos": int(n_pos[i]),
                     "matched": bool(same_res and same_pos)})

    edges = strata_edges()

    def block(sel: list[dict], label: str) -> dict:
        o = np.array([r["ours"] for r in sel], dtype=float)
        p = np.array([r["pm"] for r in sel], dtype=float)
        np_ = np.array([r["n_pos"] for r in sel], dtype=int)
        by = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (np_ >= lo) & (np_ < hi)
            if m.sum() < 10:
                continue
            by.append({
                "n_cryptic_from": int(lo), "n_cryptic_below": int(hi),
                "n_units": int(m.sum()),
                "ours_mean_auc": round(float(np.nanmean(o[m])), 6),
                "pocketminer_mean_auc": round(float(np.nanmean(p[m])), 6),
                "paired_ours_minus_pocketminer": paired_stats(o[m] - p[m]),
            })
        pooled_o = float(np.nanmean(o))
        pooled_p = float(np.nanmean(p))
        first, last = by[0], by[-1]
        return {
            "which_units": label,
            "n_units": len(sel),
            "ours_pooled_mean_auc": round(pooled_o, 6),
            "pocketminer_pooled_mean_auc": round(pooled_p, 6),
            "paired_pooled": paired_stats(o - p),
            "by_stratum": by,
            "profile_across_strata": {
                "ours_small_minus_large": round(
                    first["ours_mean_auc"] - last["ours_mean_auc"], 6),
                "pocketminer_small_minus_large": round(
                    first["pocketminer_mean_auc"]
                    - last["pocketminer_mean_auc"], 6),
                "gap_small_minus_gap_large": round(
                    first["paired_ours_minus_pocketminer"]["mean"]
                    - last["paired_ours_minus_pocketminer"]["mean"], 6),
                "how_to_read": "the first number is how far our own score falls "
                               "from the largest stratum to the smallest, the "
                               "second is how far PocketMiner's does. If they "
                               "are close, the stratum is hard for both. The "
                               "third is whether the distance between the "
                               "methods depends on the stratum at all",
            },
        }

    matched = [r for r in rows if r["matched"]]
    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "whether a rival cryptic-site method also collapses on "
                    "training units with few cryptic residues, which decides "
                    "whether that stratum is headroom or is ambiguous for "
                    "everyone",
        "prediction": PREDICTION,
        "rival": {
            "method": "PocketMiner",
            "scores": str(PM_SCORES.relative_to(ROOT)),
            "manifest": str(PM_MANIFEST.relative_to(ROOT)),
            "biases_and_their_direction": [
                "PocketMiner's training set was never clustered against "
                "CryptoBench's folds and six training entries match its own by "
                "exact PDB id, which is a floor and not a homology check. This "
                "flatters PocketMiner here.",
                "PocketMiner drops residues it cannot featurise, so on some "
                "units it is scored on an easier universe than we are. This "
                "also flatters PocketMiner, and the headline block below "
                "removes those units.",
            ],
            "why_this_matters_for_the_reading": "both biases push the same way, "
                                                "so a PocketMiner collapse on "
                                                "the small-site stratum is "
                                                "observed despite the "
                                                "advantage rather than because "
                                                "of a handicap",
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "ours": "the deployed detector, per-unit ROC-AUC averaged over the "
                    "splits in which the unit sat on the pick side",
            "pocketminer": "one forward pass over the whole fold, so its "
                           "per-unit ROC-AUC has no split structure",
            "why_that_is_comparable": "the statistic is computed within a unit "
                                      "in both cases, which is the same "
                                      "statistic the official-fold comparison "
                                      "uses",
            "strata_from": str(TAIL.relative_to(ROOT)),
        },
        "join": {
            "n_units_ours": int(sc.sum()),
            "n_units_joined": len(rows),
            "n_units_pocketminer_absent": len(absent),
            "n_units_residue_count_disagrees": len(mism_res),
            "n_units_positive_count_disagrees": len(mism_pos),
            "n_units_matched_on_both": len(matched),
            "units_where_the_positive_count_disagrees": sorted(mism_pos)[:20],
            "why_this_is_checked": "a unit whose positive count differs between "
                                   "the two sources would be placed in "
                                   "different strata for the two methods, and "
                                   "the paired difference would not be paired",
        },
        "headline_matched_universe": block(matched, "both methods cover the "
                                                    "same residues and agree on "
                                                    "the positive count"),
        "all_joined_units": block(rows, "every unit both methods scored, "
                                        "including those where PocketMiner "
                                        "covers fewer residues"),
        "reproduction_check": {
            "max_absolute_difference_from_frozen_per_split": round(repro, 6),
            "reproduces_the_frozen_arm": True,
        },
    }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    if a.write:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    h = doc["headline_matched_universe"]
    print(f"\n  joined {len(rows)} units, matched universe {len(matched)}, "
          f"positive count disagrees on {len(mism_pos)}")
    print(f"\n  {'stratum':>10s} {'n':>5s} {'ours':>8s} {'pocketmine':>11s} "
          f"{'ours-pm':>9s}  {'CI':>22s}")
    for b in h["by_stratum"]:
        d = b["paired_ours_minus_pocketminer"]
        lab = f"{b['n_cryptic_from']}-{b['n_cryptic_below'] - 1}"
        print(f"  {lab:>10s} {b['n_units']:5d} {b['ours_mean_auc']:8.4f} "
              f"{b['pocketminer_mean_auc']:11.4f} {d['mean']:+9.4f}  "
              f"[{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    pp = h["paired_pooled"]
    print(f"  {'pooled':>10s} {h['n_units']:5d} "
          f"{h['ours_pooled_mean_auc']:8.4f} "
          f"{h['pocketminer_pooled_mean_auc']:11.4f} {pp['mean']:+9.4f}  "
          f"[{pp['ci95'][0]:+.4f}, {pp['ci95'][1]:+.4f}]")
    pr = h["profile_across_strata"]
    print(f"\n  fall from the largest stratum to the smallest: "
          f"ours {pr['ours_small_minus_large']:+.4f}, "
          f"PocketMiner {pr['pocketminer_small_minus_large']:+.4f}")
    print(f"  does the distance between the methods depend on the stratum: "
          f"{pr['gap_small_minus_gap_large']:+.4f}")
    if a.write:
        print(f"\nwrote {out.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
