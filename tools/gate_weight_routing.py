#!/usr/bin/env python3.12
"""One gate weight per region: the cheapest correction that can carry the signal.

Where this comes from
---------------------
HIERARCHICAL_MULTIPLICITIES.json established two things. Chain-level routing
carries a real signal -- it beats a random per-chain router by +0.0029 to +0.0054
on 11 or 12 of 12 splits, at every rung of a damping ladder. And collecting it
through a per-region correction of all 5,152 multiplicities costs more than it
earns: the arm only reaches zero as the correction vanishes. The conclusion named
its own next experiment. If the signal is worth about +0.005 and the machinery
costs more than that, the machinery is the constraint, and a correction over the
spatial gate instead of the multiplicities would cost less.

This is that experiment, and the cost difference is not marginal. The deployed
gate is

    gated(i) = s(i) + w * g(i) * sd(s)/sd(g),   w = 1.0,

where ``g`` is the mean of the raw score over the residues within 18 A. Letting
each region choose its own ``w`` is **one number per region** -- four for R = 4 --
against 4 x 5,152 = 20,608 for the multiplicity correction. Estimating four
numbers from a fold is not a data-starvation problem, which is what sank the
previous design.

Two properties make this well posed where the earlier attempt was not
---------------------------------------------------------------------
The metric is per-unit ROC-AUC and reads only the ranking within a chain. Scaling
``s`` and ``g`` together therefore changes nothing, so ``w`` is the only free
quantity and there is no scale to fit. And every router here assigns whole chains,
so ``w`` is constant within a chain and the within-chain ranking stays
well-defined -- the failure that cost the first routing attempt -0.065 was a
within-chain router handing one chain's residues four different weight vectors.

``g`` does not depend on ``w``, so the raw score and the rescaled neighbourhood
mean are computed once per split and the whole grid is then arithmetic. A
five-arm, eight-point sweep costs what one arm used to.

Arms
----
``deployed``       w = 1.0 everywhere. This is the shipped detector and must
                   reproduce the frozen numbers.
``global fitted``  one w for the whole fold, chosen on the fit half. Worth an arm
                   on its own: whether 1.0 is even the best single value has
                   never been checked, and if it is not, that is a cheaper
                   improvement than anything else tried this week.
``chain size``     one w per region, region = chain-length quartile.
``chain polarity`` one w per region, region = quartile of the fraction of
                   residues in the polar, cationic and anionic classes.
``random chains``  whole chains assigned at a fixed seed. Four extra numbers is
                   very little capacity, but the control decides whether any gain
                   is structure or slack, and the previous sweeps were only
                   interpretable because it was there.

The grid is declared in ``WEIGHT_GRID`` and includes the deployed value. Selection
is by mean per-unit AUC on the fit half, which makes ``w`` a compiled quantity of
the training fold exactly as the cell counts and the integer multiplicities
already are. No pick-half row is read while choosing it.

What would falsify the hypothesis
---------------------------------
An informed router failing to beat both the deployed arm and the random per-chain
router. Since this is the cheapest correction the construction admits -- one
number per region, over a quantity the detector already computes -- a null here
would mean the regional signal cannot be collected by reweighting anything at all,
and the six sweeps before it stand as a complete saturation result.

Training folds only. No test residue and no external unit is read.

Usage: PYTHONPATH=src:tools python3.12 tools/gate_weight_routing.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from composition_wires import CLASSES, class_of_code  # noqa: E402
from expand_invariant_bank import SEED  # noqa: E402
from hierarchical_multiplicities import quartile_of  # noqa: E402
from quantisation_ladder import compile_at, fanout_at, offsets_at, score_at  # noqa: E402
from select_architecture_on_train import cluster_half_split, per_unit_auc  # noqa: E402

from pocket_bench.methods.table_bank import (
    N_LEVELS,
    chain_digits,
    partition_tables,
)
from pocket_bench.methods.table_field import (
    GATE_RADIUS,
    GATE_WEIGHT,
    PARTITION_ROUNDS,
    PARTITION_SEED,
    TABLE_WIDTH,
    _neighbourhood_mean,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.gate_weight_routing.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
CODES = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
OUT = ROOT / "results/architecture_sweep/GATE_WEIGHT_ROUTING.json"
ROUTER_SEED = 20260802
N_REGIONS = 4
# Declared, and it contains the deployed value so the deployed arm is a point on
# the same grid rather than a separate construction.
WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
ROUTERS = ("chain size", "chain polarity", "random chains")


def raw_and_gate(s: np.ndarray, ctr: np.ndarray, n_res_per) -> np.ndarray:
    """The rescaled neighbourhood mean, ``g * sd(s)/sd(g)``, per chain.

    Exactly the term ``apply_gate`` multiplies by GATE_WEIGHT, extracted so the
    weight can be varied without recomputing it. The standard-deviation match is
    part of the term and not part of the weight: it is what makes the gate mix in
    the same amount on a 57-residue chain and a 307-residue one.
    """
    out = np.zeros(len(s), dtype=np.float64)
    off = 0
    for n in n_res_per:
        n = int(n)
        blk = s[off:off + n]
        g = _neighbourhood_mean(blk, np.asarray(ctr[off:off + n], float),
                                GATE_RADIUS)
        sd_s, sd_g = float(np.std(blk)), float(np.std(g))
        out[off:off + n] = 0.0 if sd_g <= 0 else g * (sd_s / sd_g)
        off += n
    return out


def auc_at(raw, gate, y, n_res_per, w) -> float:
    """Per-unit AUC of ``raw + w * gate``, with ``w`` either scalar or per residue."""
    return float(per_unit_auc(raw + w * gate, y, n_res_per))


def chain_routers(n_res, codes, n_regions, rng) -> dict[str, np.ndarray]:
    lens = np.asarray([int(n) for n in n_res], dtype=np.int64)
    cls = class_of_code()[codes]
    polar = [i for i, k in enumerate(CLASSES)
             if k in ("polar", "positive", "negative")]
    frac = np.empty(len(lens))
    off = 0
    for i, n in enumerate(lens):
        frac[i] = float(np.isin(cls[off:off + int(n)], polar).mean())
        off += int(n)
    return {
        "chain size": quartile_of(lens.astype(float), n_regions),
        "chain polarity": quartile_of(frac, n_regions),
        "random chains": rng.integers(0, n_regions, size=len(lens)),
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

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    e = np.load(CODES, allow_pickle=False)
    if not np.array_equal(z["units"], e["units"]):
        raise SystemExit("the wide and expanded caches disagree about units")
    codes = e["codes"]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{x['pdb']}_{x['chain']}": x["cluster_id"] for x in entries}
    n_wires = int(W.shape[1])
    if n_wires not in frozen:
        raise SystemExit(f"frozen artifact has widths {sorted(frozen)}")

    tables = partition_tables(n_wires, TABLE_WIDTH, PARTITION_ROUNDS,
                              PARTITION_SEED)
    offsets = offsets_at(tables, N_LEVELS)
    t0 = time.perf_counter()
    D = chain_digits(np.asarray(W, dtype=np.float64), n_res)
    print(f"banded {n_wires} wires in {time.perf_counter() - t0:.0f}s; "
          f"grid {WEIGHT_GRID}; {N_REGIONS} regions", flush=True)
    row = np.repeat(np.arange(len(n_res)), n_res)

    arms = ["deployed", "global fitted"] + list(ROUTERS)
    got: dict[str, list[float]] = {k: [] for k in arms}
    chosen: dict[str, list] = {k: [] for k in arms if k != "deployed"}

    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        fit, pick = is_fit[row], ~is_fit[row]
        n_fit = np.array([n for n, f in zip(n_res, is_fit) if f])
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])

        t1 = time.perf_counter()
        frac, _tc = compile_at(D[fit], y[fit], tables, offsets, N_LEVELS)
        mult = fanout_at(D[fit], y[fit], tables, offsets, frac, N_LEVELS)
        # Raw scores and the gate term on each half. Computed once; the grid is
        # then arithmetic, which is what makes a five-arm eight-point sweep cost
        # what one arm used to.
        raw_f = score_at(D[fit], tables, offsets, frac, mult, N_LEVELS)
        raw_p = score_at(D[pick], tables, offsets, frac, mult, N_LEVELS)
        gate_f = raw_and_gate(raw_f, ctr[fit], n_fit)
        gate_p = raw_and_gate(raw_p, ctr[pick], n_pick)
        yf, yp = y[fit], y[pick]

        got["deployed"].append(auc_at(raw_p, gate_p, yp, n_pick, GATE_WEIGHT))
        print(f"  split {s + 1}/{n_splits}  deployed "
              f"{got['deployed'][-1]:.4f}  frozen {frozen[n_wires][s]:.4f}  "
              f"{time.perf_counter() - t1:.0f}s", flush=True)

        # One weight for the whole fold, chosen on the fit half.
        curve = {w: auc_at(raw_f, gate_f, yf, n_fit, w) for w in WEIGHT_GRID}
        w_best = max(curve, key=curve.get)
        got["global fitted"].append(auc_at(raw_p, gate_p, yp, n_pick, w_best))
        chosen["global fitted"].append(
            {"w": w_best, "fit_auc_by_weight": {str(k): round(v, 6)
                                                for k, v in curve.items()}})
        print(f"      global fitted w={w_best:g}  "
              f"{got['global fitted'][-1]:.4f}  "
              f"{got['global fitted'][-1] - frozen[n_wires][s]:+.4f}",
              flush=True)

        rng = np.random.default_rng(ROUTER_SEED + s)
        routers = chain_routers(n_res, codes, N_REGIONS, rng)
        for rname in ROUTERS:
            band = routers[rname]
            band_f = np.repeat(band[is_fit], n_fit)
            band_p = np.repeat(band[~is_fit], n_pick)
            # One weight per region, each chosen on that region's fit-half rows.
            ws, detail = [], []
            for g in range(N_REGIONS):
                m = band_f == g
                if not m.any():
                    ws.append(GATE_WEIGHT)
                    detail.append({"region": g, "n_fit_rows": 0,
                                   "w": GATE_WEIGHT, "empty": True})
                    continue
                sub_n = [int(n) for n, b in zip(n_fit, band[is_fit]) if b == g]
                c = {w: auc_at(raw_f[m], gate_f[m], yf[m], sub_n, w)
                     for w in WEIGHT_GRID}
                wg = max(c, key=c.get)
                ws.append(wg)
                detail.append({"region": g, "n_fit_rows": int(m.sum()),
                               "w": wg,
                               "fit_auc_by_weight": {str(k): round(v, 6)
                                                     for k, v in c.items()}})
            w_vec = np.asarray(ws, dtype=np.float64)[band_p]
            got[rname].append(auc_at(raw_p, gate_p, yp, n_pick, w_vec))
            chosen[rname].append({"weights": ws, "regions": detail})
            print(f"      {rname:16s} w={ws}  {got[rname][-1]:.4f}  "
                  f"{got[rname][-1] - frozen[n_wires][s]:+.4f}", flush=True)

    base = frozen[n_wires][:n_splits]

    def summarise(v):
        v = np.asarray(v)
        return {"mean": round(float(v.mean()), 6),
                "min": round(float(v.min()), 6),
                "max": round(float(v.max()), 6)}

    def compare(v, ref):
        d = np.asarray(v) - np.asarray(ref)
        return {"mean": round(float(d.mean()), 6),
                "min": round(float(d.min()), 6),
                "max": round(float(d.max()), 6),
                "n_splits_positive": int((d > 0).sum()),
                "n_splits": int(len(d)),
                "positive_on_every_split": bool((d > 0).all())}

    repro = {
        "arm": "deployed",
        "max_absolute_difference": round(
            float(np.abs(np.asarray(got["deployed"]) - base).max()), 9),
    }
    repro["reproduces_the_deployed_arm"] = bool(
        repro["max_absolute_difference"] < 1e-6)

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "chain-level routing carries a real signal worth about "
                    "+0.005 against a random per-chain router, and collecting it "
                    "through a per-region correction of all 5,152 multiplicities "
                    "costs more than it earns. Does the cheapest correction the "
                    "construction admits -- one gate weight per region, four "
                    "numbers instead of 20,608 -- collect it",
        "construction": {
            "deployed_gate": "gated(i) = s(i) + w * g(i) * sd(s)/sd(g) with "
                             f"w = {GATE_WEIGHT} and g the mean of the raw score "
                             f"over residues within {GATE_RADIUS} A",
            "what_varies": "w, per region. One number per region",
            "n_parameters": f"{N_REGIONS} against {N_REGIONS} x "
                            f"{len(tables)} = {N_REGIONS * len(tables)} for the "
                            f"multiplicity correction",
            "why_w_is_the_only_free_quantity": "the metric is per-unit ROC-AUC "
                                               "and reads only the ranking "
                                               "within a chain, so scaling s and "
                                               "g together changes nothing and "
                                               "there is no scale to fit",
            "why_every_router_is_per_chain": "w must be constant within a chain "
                                             "or the chain's residues get "
                                             "incomparable scores; a "
                                             "within-chain router cost -0.065 in "
                                             "ROUTED_MULTIPLICITIES.json",
            "grid": list(WEIGHT_GRID),
            "grid_contains_the_deployed_value": GATE_WEIGHT in WEIGHT_GRID,
            "selection": "by mean per-unit AUC on the fit half, which makes w a "
                         "compiled quantity of the training fold exactly as the "
                         "cell counts and the integer multiplicities already "
                         "are. No pick-half row is read while choosing it",
            "still_integer_at_inference": False,
            "what_that_costs": "w is a rational number and the gate term is "
                               "real-valued, so this arm is not integer-only at "
                               "inference in the way the multiplicity path is. "
                               "The gate was already real-valued in the deployed "
                               "detector, so nothing is lost that was there "
                               "before -- but it is worth stating that this "
                               "correction lives on the one real-valued stage "
                               "and not on the integer one",
        },
        "why_this_is_the_named_next_experiment":
            "results/architecture_sweep/HIERARCHICAL_MULTIPLICITIES.json ends by "
            "saying the machinery is the constraint and not the signal, and names "
            "a correction over a subset of tables or over the spatial gate as "
            "what would cost less. This is the second of those",
        "what_would_falsify_it": "an informed router failing to beat both the "
                                 "deployed arm and the random per-chain router. "
                                 "Since this is the cheapest correction the "
                                 "construction admits, a null would mean the "
                                 "regional signal cannot be collected by "
                                 "reweighting anything at all",
        "held_fixed": {
            "n_wires": n_wires, "n_levels": N_LEVELS,
            "table_width": TABLE_WIDTH, "partition_rounds": PARTITION_ROUNDS,
            "partition_seed": PARTITION_SEED, "n_tables": len(tables),
            "gate_radius": GATE_RADIUS,
            "cell_counts_and_multiplicities": "unchanged; only the gate weight "
                                              "moves",
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "compile_and_choose_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC",
            "baseline_was_not_recomputed": str(COUNTING.relative_to(ROOT)),
        },
        "arms": {k: summarise(v) for k, v in got.items()},
        "deployed_arm_frozen": summarise(base),
        "minus_deployed": {k: compare(v, base) for k, v in got.items()},
        "minus_random_per_chain": {
            r: compare(got[r], got["random chains"])
            for r in ROUTERS if r != "random chains"
        },
        "global_fitted_minus_deployed": compare(got["global fitted"], base),
        "weights_chosen": chosen,
        "reproduction_check": repro,
        "per_split": {k: [round(float(x), 6) for x in v]
                      for k, v in got.items()},
        "per_split_deployed_frozen": [round(float(x), 6) for x in base],
        "n_units": int(len(n_res)),
        "n_residues": int(len(y)),
        "n_positive_residues": int(y.sum()),
    }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    print(f"\n  deployed (frozen): {base.mean():.6f}")
    for k in arms:
        c = doc["minus_deployed"][k]
        print(f"  {k:16s} {doc['arms'][k]['mean']:.6f}  {c['mean']:+.6f}  on "
              f"{c['n_splits_positive']}/{c['n_splits']}")
    print("\n  against the random per-chain router:")
    for k, c in doc["minus_random_per_chain"].items():
        print(f"    {k:16s} {c['mean']:+.6f}  on "
              f"{c['n_splits_positive']}/{c['n_splits']}")
    print(f"\n  deployed arm reproduces the frozen numbers: "
          f"{repro['reproduces_the_deployed_arm']} "
          f"(max |diff| {repro['max_absolute_difference']:.2e})")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
