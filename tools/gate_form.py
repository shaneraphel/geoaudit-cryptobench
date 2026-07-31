#!/usr/bin/env python3
"""The gate's aggregation, which is the one part of it nobody has varied.

The argument
------------
``apply_gate`` adds back the *mean* score over an 18 A neighbourhood, rescaled
to the raw score's spread. Two of its numbers have been swept -- radius and
weight, in ``GATE_BY_STRATUM.json`` -- and its *form* never has. The eight
parameters ``AGENT_MEMORY`` 2b calls exhausted are all about the table bank; the
gate is the last stage of the pipeline and the least examined part of it.

``FAILURE_TAIL.json`` says where to look. The detector scores 0.5991 on units
with fewer than ten cryptic residues and 0.8766 on units with more than
twenty-two, and a mean is precisely the aggregation that a mostly-negative
neighbourhood destroys: eight positives inside an 18 A ball of two hundred
residues move the mean by almost nothing, while forty positives move it a lot.
An order statistic does not have that property. The maximum over the
neighbourhood is unchanged by how many negatives surround it, and a top-k mean
degrades gracefully between the two.

So this is the operator built for the observable rather than a general one,
which is the rule that came out of five null wire families: if a failure
correlates with something, build against that something.

The forms
---------
``mean``      the deployed aggregation, recomputed so the others are compared
              against a number this tool produced rather than a quoted one.
``max``       the largest neighbour score. Insensitive to the count of
              negatives, which is the whole hypothesis.
``top3``      mean of the three largest, and ``top5`` of the five largest. A
              cryptic patch is contiguous and several residues wide, so the
              single maximum may be noise where a small top-k is signal.
``median``    the opposite direction: even more diluted than the mean, included
              because a hypothesis that predicts an ordering should be asked to
              predict the bottom of it too, and a form that lands between mean
              and max would falsify the reading.
``count``     how many neighbours exceed the chain's own median score, a pure
              count and the only form here that discards the score values
              entirely.

Each is rescaled the way the deployed gate rescales, matching the standard
deviation of the raw field, so the arms differ in the aggregation and in nothing
else. The deployed radius and weight are held.

The prediction, before the run
------------------------------
If dilution is the mechanism, the order-statistic forms beat the mean on the
0-9 stratum and the gap narrows or reverses on 23-76, and ``median`` is worst
everywhere. If every form behaves the same, the gate's aggregation is not the
size dependence and the third of three gate hypotheses is eliminated.

A pooled gain above the 0.0026 reseed floor with a paired interval excluding
zero would be the first change to the detector this line of work has produced
that is larger than the gate radius already found.

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
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.gate_form.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
TAIL = ROOT / "results/architecture_sweep/FAILURE_TAIL.json"
OUT = ROOT / "results/architecture_sweep/GATE_FORM.json"

FORMS = ("mean", "max", "top3", "top5", "median", "count")
FORM_K = {"top3": 3, "top5": 5}
RADII = (14.0, 18.0)
BLOCK = 512

PREDICTION = {
    "committed_before_the_run": True,
    "if_dilution_is_the_mechanism": "the order-statistic forms beat the mean on "
                                    "the 0-9 stratum and the gap narrows or "
                                    "reverses on 23-76; median is worst "
                                    "everywhere",
    "if_it_is_not": "every form behaves alike, eliminating the third of three "
                    "gate hypotheses",
    "what_would_be_a_result": "a pooled gain above the 0.0026 reseed floor with "
                              "a paired interval excluding zero, which would be "
                              "larger than the gate radius already found",
    "reseed_floor": 0.0026,
}


def neighbourhood_stats(s: np.ndarray, ctr: np.ndarray,
                        radius: float) -> dict[str, np.ndarray]:
    """Every aggregation of the scores within ``radius``, for every residue.

    All six forms come out of one pass because the pairwise distance matrix is
    the dominant cost and building it once per form would pay it six times over.
    Blocked over rows the way ``_neighbourhood_mean`` is, since the distance
    matrix of a 500-residue chain is small and of a 3,000-residue one is not.
    """
    n = len(s)
    out = {f: np.empty(n, dtype=np.float64) for f in FORMS}
    r2 = radius * radius
    med = float(np.median(s))
    above = s > med
    for i in range(0, n, BLOCK):
        j = min(i + BLOCK, n)
        d2 = ((ctr[i:j, None, :] - ctr[None, :, :]) ** 2).sum(-1)
        m = d2 <= r2
        cnt = m.sum(1)
        a = m.astype(np.float64)
        out["mean"][i:j] = (a @ s) / np.maximum(cnt, 1)
        out["count"][i:j] = (m & above[None, :]).sum(1).astype(np.float64)
        # Order statistics need the values, so outside the ball is -inf. Every
        # residue is inside its own ball, so no row is entirely masked.
        v = np.where(m, s[None, :], -np.inf)
        out["max"][i:j] = v.max(1)
        k = max(FORM_K.values())
        top = -np.partition(-v, min(k, v.shape[1] - 1), axis=1)[:, :k]
        top = -np.sort(-top, axis=1)
        for f, kk in FORM_K.items():
            part = np.where(np.isfinite(top[:, :kk]), top[:, :kk], 0.0)
            out[f][i:j] = part.sum(1) / np.maximum(np.minimum(cnt, kk), 1)
        out["median"][i:j] = np.nanmedian(
            np.where(m, s[None, :], np.nan), axis=1)
    return out


def gate_with(s: np.ndarray, ctr: np.ndarray, n_res_per, radius: float,
              weight: float) -> dict[str, np.ndarray]:
    """The deployed gate with its aggregation replaced and nothing else.

    A local copy rather than an edit, because ``table_field.py`` is pinned by
    ``TABLE_FIELD.json``'s code_sha256 and this tool has no business
    invalidating a compiled field. The rescaling is the deployed one -- match
    the standard deviation of the raw field, not its maximum, because the
    maximum over a chain is an order statistic of a handful of residues and a
    max-normalised gate would mix in a different amount on every structure.
    """
    out = {f: np.empty(len(s), dtype=np.float64) for f in FORMS}
    off = 0
    for n in n_res_per:
        n = int(n)
        blk = s[off:off + n]
        g = neighbourhood_stats(blk, np.asarray(ctr[off:off + n], float),
                                radius)
        sd_s = float(np.std(blk))
        for f in FORMS:
            sd_g = float(np.std(g[f]))
            out[f][off:off + n] = (blk if sd_g <= 0
                                   else blk + weight * g[f] * (sd_s / sd_g))
        off += n
    return out


def strata_edges() -> list[int]:
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

    # The mean form at the deployed radius and weight must reproduce the frozen
    # per-split numbers, which is what licenses reading every other form as a
    # change in the aggregation rather than in the reimplementation.
    arms = [(f, r) for f in FORMS for r in RADII]
    seen = {k: np.zeros(len(n_res), dtype=np.int64) for k in arms}
    total = {k: np.zeros(len(n_res), dtype=np.float64) for k in arms}
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
        for r in RADII:
            gated = gate_with(raw, ctr[pick], n_pick, r, GATE_WEIGHT)
            for form in FORMS:
                per = auc_per_unit(gated[form], y[pick], n_pick)
                ok = ~np.isnan(per)
                seen[(form, r)][idx[ok]] += 1
                total[(form, r)][idx[ok]] += per[ok]
                if (form, r) == ("mean", GATE_RADIUS):
                    deployed_per_split.append(float(np.nanmean(per)))
        print(f"  split {s + 1}/{n_splits}  mean@18 "
              f"{deployed_per_split[-1]:.4f}  frozen {frozen[s]:.4f}  "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    repro = float(np.abs(np.asarray(deployed_per_split) - frozen).max())
    if repro >= 5e-4:
        raise SystemExit(
            f"the mean form at the deployed radius differs from the frozen "
            f"per-split numbers by {repro:.2e}; every other form would be "
            f"measured against a reimplementation rather than against the "
            f"detector")

    per_unit = {}
    for k in arms:
        sc = seen[k] > 0
        m = np.full(len(n_res), np.nan)
        m[sc] = total[k][sc] / seen[k][sc]
        per_unit[k] = m
    dep = per_unit[("mean", GATE_RADIUS)]
    scored = ~np.isnan(dep)

    def paired(k, mask) -> dict:
        d = per_unit[k][mask] - dep[mask]
        d = d[~np.isnan(d)]
        n = len(d)
        se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        nb = int((d > 0).sum())
        p = sum(comb(n, i) for i in range(nb, n + 1)) / 2 ** n
        return {"n_units": n, "mean": round(float(d.mean()), 6),
                "ci95": [round(float(d.mean() - 1.96 * se), 6),
                         round(float(d.mean() + 1.96 * se), 6)],
                "crosses_zero": bool(abs(d.mean()) < 1.96 * se),
                "n_units_better": nb,
                "sign_test_p_one_sided": round(float(p), 6)}

    results, paired_vs = {}, {}
    for form, r in arms:
        k = (form, r)
        by = []
        for lo_e, hi_e in zip(edges[:-1], edges[1:]):
            sel = scored & (n_pos >= lo_e) & (n_pos < hi_e)
            if sel.sum() < 10:
                continue
            by.append({"n_cryptic_from": int(lo_e),
                       "n_cryptic_below": int(hi_e),
                       "n_units": int(sel.sum()),
                       "mean_auc": round(float(np.nanmean(per_unit[k][sel])), 6)})
        name = f"{form}@{r:g}"
        results[name] = {
            "form": form, "radius": r,
            "pooled_mean_auc": round(float(np.nanmean(per_unit[k][scored])), 6),
            "by_stratum": by,
        }
        if k != ("mean", GATE_RADIUS):
            rec = {"pooled": paired(k, scored)}
            for lo_e, hi_e in zip(edges[:-1], edges[1:]):
                sel = scored & (n_pos >= lo_e) & (n_pos < hi_e)
                if sel.sum() >= 10:
                    rec[f"{lo_e}-{hi_e - 1}"] = paired(k, sel)
            paired_vs[name] = rec

    best = max(paired_vs.items(), key=lambda kv: kv[1]["pooled"]["mean"])
    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "whether the gate's aggregation, which has never been "
                    "varied, is what makes the detector fail on units with few "
                    "cryptic residues",
        "prediction": PREDICTION,
        "strata_from": str(TAIL.relative_to(ROOT)),
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "forms": list(FORMS),
            "radii": list(RADII),
            "weight": GATE_WEIGHT,
            "held_fixed": "everything except the aggregation and the radius; "
                          "the rescaling is the deployed one",
            "metric": "ROC-AUC within a unit, averaged over the splits in which "
                      "the unit sat on the pick side",
        },
        "reproduction_check": {
            "recomputed_arm": f"mean@{GATE_RADIUS:g}",
            "max_absolute_difference_from_frozen_per_split": round(repro, 6),
            "reproduces_frozen_arm": True,
        },
        "arms": results,
        "paired_against_the_deployed_gate": paired_vs,
        "best_pooled_arm": {
            "arm": best[0],
            "pooled": best[1]["pooled"],
            "clears_the_reseed_floor": bool(
                best[1]["pooled"]["mean"] > PREDICTION["reseed_floor"]
                and not best[1]["pooled"]["crosses_zero"]),
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

    strat_names = [f"{b['n_cryptic_from']}-{b['n_cryptic_below'] - 1}"
                   for b in results[f"mean@{GATE_RADIUS:g}"]["by_stratum"]]
    print(f"\n  reproduces the frozen deployed arm to {repro:.2e}")
    print(f"\n  {'arm':14s} {'pooled':>8s}" + "".join(
        s.rjust(10) for s in strat_names))
    for k, v in results.items():
        mark = "  <- deployed" if k == f"mean@{GATE_RADIUS:g}" else ""
        print(f"  {k:14s} {v['pooled_mean_auc']:8.4f}"
              + "".join(f"{b['mean_auc']:10.4f}" for b in v["by_stratum"])
              + mark)
    print(f"\n  paired against the deployed gate, pooled:")
    for k, v in sorted(paired_vs.items(),
                       key=lambda kv: -kv[1]["pooled"]["mean"]):
        p = v["pooled"]
        print(f"    {k:14s} {p['mean']:+.5f}  CI {p['ci95']}  "
              f"{p['n_units_better']}/{p['n_units']}  "
              f"crosses_zero={p['crosses_zero']}")
    b = doc["best_pooled_arm"]
    print(f"\n  best pooled arm {b['arm']}: clears the reseed floor with an "
          f"interval excluding zero: {b['clears_the_reseed_floor']}")
    if a.write:
        print(f"\nwrote {out.relative_to(ROOT)}")
    else:
        print("\n(not written; pass --write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
