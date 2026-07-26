#!/usr/bin/env python3
"""The single test-fold read for the architecture chosen on the training fold.

Selection happened in tools/counterattack_{select,fusion,threshold}.py, entirely
on cluster-disjoint halves of the TRAINING fold. This script compiles the
selected architecture on the FULL training fold and scores it once on the 192
official test units. Nothing here is tuned; every constant was fixed before this
file ran.

What the selected architecture is, stated without euphemism:

    172 wires   35 algebraic/topological invariants, 7 published per-residue
                chemical constants, 1 training-fold residue propensity, and each
                of those 43 contracted over the geometric neighbourhood at 6,
                14 and 20 A
    digits      each wire banded into 4 levels by its own chain's rank order
    gate        an integer weight vector in [-W, W], obtained from ONE closed-
                form Fisher solve on the training digits and then quantized;
                inference is an integer multiply-accumulate
    patch       plus the multi-scale spatial counting gate at 6/8/10/14/18 A

It is a linear threshold function with integer weights. It is NOT the
counting-only field and is reported under a different name, because the
counting-only claim is a stronger claim and this method does not support it.

Usage: PYTHONPATH=src:tools python3.12 tools/evaluate_threshold_field.py
"""
from __future__ import annotations

import json

import numpy as np

from pocket_bench.metrics import average_precision, roc_auc
from pocket_bench.metrics_bootstrap import f1, json_safe, mcc, paired_bootstrap
from pocket_bench.paths import ROOT

from counterattack_select import GATE_RADII, _unit, chain_digits, gate
from counterattack_threshold import closed_form_direction, quantize

TRAIN = ROOT / "data/cryptobench_apo/_expanded_cache_train.npz"
TEST = ROOT / "data/cryptobench_apo/_expanded_cache_test.npz"
TRAIN_DIGITS = ROOT / "data/cryptobench_apo/_expanded_digits_train.npy"
SELECTION = ROOT / "results/architecture_sweep/COUNTERATTACK_THRESHOLD.json"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
FIELD_OUT = ROOT / "data/cryptobench_apo/THRESHOLD_FIELD.json"
OUT = ROOT / "results/official_fold/THRESHOLD_FIELD_TEST.json"

METHOD = "threshold_field"


def per_unit_metrics(score, y, n_res_per, units, q):
    """AUC, PR-AUC and, at the fixed top-q operating point, MCC and F1."""
    rows = []
    off = 0
    for n, unit in zip(n_res_per, units):
        n = int(n)
        s, t = score[off:off + n], y[off:off + n]
        off += n
        if t.sum() == 0 or t.sum() == n:
            rows.append({"unit_id": unit, "residue_auc": None,
                         "residue_pr_auc": None, "residue_mcc": None,
                         "residue_f1": None})
            continue
        # Operating point: the top q fraction of THIS chain, q fixed at the
        # training base rate. No threshold is searched on the test fold.
        k = max(1, int(round(q * n)))
        thr = float(np.sort(s)[::-1][k - 1])
        rows.append({
            "unit_id": unit,
            "residue_auc": roc_auc(list(s), list(t)),
            "residue_pr_auc": average_precision(list(s), list(t)),
            "residue_mcc": mcc(list(s), list(t), thr=thr),
            "residue_f1": f1(list(s), list(t), thr=thr),
        })
    return rows


def main() -> int:
    sel = json.loads(SELECTION.read_text())
    best = sel["best_per_claim"]["integer-weight"]
    n_wires = int(best["n_wires"])
    weight_range = int(best["weight_range"])
    print(f"architecture selected on the training fold: "
          f"{best['architecture']}  (pick-half {best['pick_half_roc_auc']:.4f})")

    ztr = np.load(TRAIN, allow_pickle=False)
    zte = np.load(TEST, allow_pickle=False)
    names = [str(s) for s in ztr["names"]]
    ytr = ztr["y"]
    q = float(ytr.mean())

    Dtr = (np.load(TRAIN_DIGITS) if TRAIN_DIGITS.exists()
           else chain_digits(ztr["X"], ztr["n_res_per"]))
    Dte = chain_digits(zte["X"], zte["n_res_per"])

    # wire ranking and the direction: compiled on the FULL training fold
    def pooled(x, yv):
        n_pos, n_neg = int(yv.sum()), int(len(yv) - yv.sum())
        r = np.empty(len(x))
        r[np.argsort(x, kind="stable")] = np.arange(1, len(x) + 1)
        return (r[yv == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    g = np.array([abs(2.0 * pooled(Dtr[:, j].astype(float), ytr) - 1.0)
                  for j in range(Dtr.shape[1])])
    cols = sorted(np.argsort(-g)[:n_wires].tolist())
    w = closed_form_direction(Dtr[:, cols], ytr)
    wi = quantize(w, weight_range)
    print(f"{len(cols)} wires, integer weights in "
          f"[{int(wi.min())}, {int(wi.max())}], "
          f"{int((wi != 0).sum())} nonzero")

    S = Dte[:, cols].astype(np.int64) @ wi
    G = np.sum([_unit(gate(S.astype(np.float64), zte["ctr"],
                           zte["n_res_per"], r)) for r in GATE_RADII], axis=0)
    score = _unit(S.astype(np.float64)) + _unit(G)

    units = [str(u) for u in zte["units"]]
    rows = per_unit_metrics(score, zte["y"], zte["n_res_per"], units, q)
    scored = [r for r in rows if r["residue_auc"] is not None]
    means = {k: float(np.mean([r[k] for r in scored if r[k] is not None]))
             for k in ("residue_auc", "residue_pr_auc", "residue_mcc",
                       "residue_f1")}
    print(f"\nTEST fold, {len(scored)}/{len(rows)} units scored")
    for k, v in means.items():
        print(f"  {k:16s} {v:.4f}")

    # paired against P2Rank on the identical units, identical resamples
    telem = json.loads(TELEMETRY.read_text())["rows"]
    p2 = {r["unit_id"]: r for r in telem if r["method"] == "p2rank"}
    ours = {r["unit_id"]: r for r in rows}
    shared = [u for u in units if u in p2 and u in ours]
    paired = {}
    for metric in ("residue_auc", "residue_pr_auc", "residue_mcc", "residue_f1"):
        vals = {
            METHOD: [ours[u][metric] for u in shared],
            "p2rank": [p2[u][metric] for u in shared],
        }
        paired[metric] = paired_bootstrap(vals, baseline="p2rank",
                                          n_boot=10000, seed=20260725)
        d = paired[metric]["paired_vs_baseline"][METHOD]
        verdict = ("indistinguishable" if d["crosses_zero"]
                   else ("AHEAD" if d["delta_point"] > 0 else "behind"))
        print(f"  vs p2rank {metric:16s} {d['delta_point']:+.4f} "
              f"[{d['delta_ci_low']:+.4f}, {d['delta_ci_high']:+.4f}]  {verdict}")

    FIELD_OUT.write_text(json.dumps(json_safe({
        "schema": "geoaudit.threshold_field.v1",
        "clinical_grade": False,
        "method": METHOD,
        "description": "integer-weight linear threshold gate over quaternary "
                       "wire digits, plus a multi-scale spatial counting gate",
        "not_counting_only": "the weights come from one closed-form Fisher "
                             "solve on the training fold, not from a bincount; "
                             "the counting-only method is algebraic_field",
        "n_wires_total": len(names),
        "n_wires_used": len(cols),
        "wire_names": [names[j] for j in cols],
        "integer_weights": wi.tolist(),
        "weight_range": weight_range,
        "gate_radii": list(GATE_RADII),
        "operating_point_q": q,
        "train": {"n_units": int(len(ztr["n_res_per"])),
                  "n_residues": int(len(ytr)),
                  "base_rate": q},
    }), indent=2, allow_nan=False) + "\n")

    OUT.write_text(json.dumps(json_safe({
        "schema": "geoaudit.threshold_field_test.v1",
        "clinical_grade": False,
        "method": METHOD,
        "selected_on": "cluster-disjoint halves of the training fold; this is "
                       "the first and only evaluation of this architecture on "
                       "the official test fold",
        "selection_record":
            "results/architecture_sweep/COUNTERATTACK_THRESHOLD.json",
        "n_units_scored": len(scored),
        "means": means,
        "per_unit": rows,
        "paired_vs_p2rank": paired,
    }), indent=2, allow_nan=False) + "\n")
    print(f"\nwrote {FIELD_OUT.relative_to(ROOT)}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
