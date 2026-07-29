#!/usr/bin/env python3
"""Does the generated operator bank still lift the ceiling once the spatial
expansion is already there? Training folds only.

Why this is not the experiment already in the repository
-------------------------------------------------------
``expand_invariant_bank.py`` measured a Fisher ceiling of 0.7595 for the 35
algebraic invariants and 0.7676 for those plus 190 operator and 77 chain
descriptors: a lift of +0.0081, positive on all twelve cluster-disjoint halvings.
That is the only repeatable lift in this repository and it is the reason to look
here at all.

But it compared local descriptor to local descriptor, and no detector in the
paper consumes local descriptors. What ``table_field`` consumes is 645 wires: 43
local quantities --- the 35 invariants, seven chemical ones and a propensity ---
each expanded fifteen ways, by neighbourhood mean, standard deviation, difference
and rank at four radii. So the +0.0081 was measured against a bank an eighteenth
the width of the real one, and the question it answers is not the one that
matters.

The one that matters is whether the generated families say anything the spatial
expansion has not already said. There is a specific reason to doubt it. The chain
lag spectrum is computed at 8, 12 and 16 angstroms and the shape operator at 8
angstroms; the expansion already takes means, spreads, differences and ranks at
four radii of everything it is given. Multi-scale context arrived twice by two
routes, and two routes to the same information is a redundancy, not a lift.

So the baseline here is the 645 wires themselves. If the lift survives against
them it is real and worth the cost of a counting construction over a wider bank.
If it does not, this road is closed and the 0.7676 in the older artifact should
be read as what it is: a ceiling over a bank nobody deploys.

What is reported, and why by family
-----------------------------------
Four families are separated rather than pooled, because they are not
interchangeable and one of them is being asked about specifically. The chain lag
spectrum is the local Toeplitz symbol, thirty-three wires. The shape operator is
curvature with a sign, twenty-one, which is the only thing here that can tell a
saddle from a flat patch and so the only thing that can see the mouth of a cleft
as a mouth. The valuation profile is the non-Archimedean ball, eleven. The
soft-mode hinge is the deformation: twelve wires built from shear modes and a
thermal average, and the one family whose subject is the motion that makes a
cryptic pocket cryptic in the first place.

A family that lifts alone is worth more than the pooled number, because a bank
of 190 cannot be addressed by a counting construction and a bank of 12 can.

Discipline
----------
The pick half is out of sample for the fit half and both are inside the official
training fold. No test residue and no external unit is read; the artifact records
that. A Fisher discriminant solves for arbitrary real coefficients over
unquantised inputs, so every number here is an upper bound on what a counting
field could reach, not a prediction of one.

Usage: PYTHONPATH=src:tools python3.12 tools/wide_bank_ceiling.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from expand_invariant_bank import (  # noqa: E402  (sibling tool)
    CH_NAMES,
    N_SPLITS,
    OP_NAMES,
    RIDGE,
    SEED,
    _family,
    build_cache,
    fisher,
    gated,
)
from select_architecture_on_train import (  # noqa: E402
    cluster_half_split,
    per_unit_auc,
)

from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.wide_bank_ceiling.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
OUT = ROOT / "results/architecture_sweep/WIDE_BANK_CEILING.json"


def load_wide():
    """The 645 wires the deployed field reads, and each unit's cluster.

    The operator and chain caches were written against the 35-column cache, so
    the row order has to be checked rather than assumed: both caches are keyed by
    nothing but position, and a silent misalignment would show up as a bank that
    mysteriously fails to help, which is the answer this tool is looking for.
    """
    z = np.load(WIDE, allow_pickle=False)
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    missing = [u for u in units if u not in cluster_of]
    if missing:
        raise SystemExit(f"{len(missing)} cached units absent from the manifest, "
                         f"e.g. {missing[:3]}")
    return (z["X"], z["y"], z["n_res_per"], z["ctr"], units, cluster_of,
            [str(s) for s in z["names"]])


def _aligned(name: str, F: np.ndarray, W: np.ndarray, y: np.ndarray,
             yc: np.ndarray) -> np.ndarray:
    if len(F) != len(W):
        raise SystemExit(
            f"the {name} cache has {len(F)} rows and the wide cache has "
            f"{len(W)}; they were built from different manifests and "
            f"concatenating them would pair each residue with another one's "
            f"descriptors")
    if not np.array_equal(y, yc):
        raise SystemExit(
            f"the {name} cache and the wide cache carry different labels in the "
            f"same row order, so one of them is not the official training fold")
    return F


def banks_of(W, FOP, FCH):
    """The wide bank, and the wide bank plus each generated family."""
    fam_of = {f: [i for i, n in enumerate(CH_NAMES) if _family(n) == f]
              for f in sorted({_family(n) for n in CH_NAMES})}
    out = {f"wide {W.shape[1]}": W}
    for fam, idx in fam_of.items():
        out[f"wide+{fam} {W.shape[1] + len(idx)}"] = np.concatenate(
            [W, FCH[:, idx]], axis=1)
    out[f"wide+chain {W.shape[1] + FCH.shape[1]}"] = np.concatenate(
        [W, FCH], axis=1)
    out[f"wide+operator {W.shape[1] + FOP.shape[1]}"] = np.concatenate(
        [W, FOP], axis=1)
    out[f"wide+combined {W.shape[1] + FOP.shape[1] + FCH.shape[1]}"] = (
        np.concatenate([W, FOP, FCH], axis=1))
    return out, {f: len(i) for f, i in fam_of.items()}


def evaluate(banks, y, n_res, ctr, units, cluster_of, n_splits):
    rows: dict[str, list[float]] = {k: [] for k in banks}
    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        row = np.repeat(np.arange(len(n_res)), n_res)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        ypick, ctr_pick = y[pick], ctr[pick]
        t0 = time.perf_counter()
        for name, X in banks.items():
            sc = fisher(X[fit], y[fit], X[pick])
            rows[name].append(per_unit_auc(gated(sc, ctr_pick, n_pick),
                                           ypick, n_pick))
        base = rows[next(iter(banks))][-1]
        best = max((rows[k][-1], k) for k in banks if k != next(iter(banks)))
        print(f"  split {s + 1}/{n_splits}  wide {base:.4f}  best "
              f"{best[0]:.4f} ({best[1].split()[0]})  "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
    return {k: np.array(v) for k, v in rows.items()}


def build(n_splits: int, workers: int) -> dict:
    W, y, n_res, ctr, units, cluster_of, wire_names = load_wide()
    FOP = _aligned("operator", build_cache("operator", workers), W, y, y)
    FCH = _aligned("chain", build_cache("chain", workers), W, y, y)
    banks, fam_sizes = banks_of(W, FOP, FCH)

    print(f"\n{len(units)} units, {len(W):,} residues, {W.shape[1]} wires "
          f"(43 local quantities x 15 spatial transforms) as the baseline")
    print(f"generated families added: " + ", ".join(
        f"{f} {n}" for f, n in sorted(fam_sizes.items())))
    print(f"\nFisher ceiling over {n_splits} cluster-disjoint halvings:")
    curves = evaluate(banks, y, n_res, ctr, units, cluster_of, n_splits)

    base_key = f"wide {W.shape[1]}"
    lifts = {}
    for k, v in curves.items():
        if k == base_key:
            continue
        d = v - curves[base_key]
        lifts[k] = {
            "mean": round(float(d.mean()), 6),
            "min": round(float(d.min()), 6),
            "max": round(float(d.max()), 6),
            "n_splits_positive": int((d > 0).sum()),
            "n_splits": int(len(d)),
            # Twelve halvings of one fold are not twelve independent samples, so
            # this is a consistency count and not a test. A family that helps on
            # every split has at least not been fitted to one of them.
            "positive_on_every_split": bool((d > 0).all()),
        }
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "does the generated operator bank still lift the Fisher "
                    "ceiling when the baseline is the 645 wires the deployed "
                    "field actually reads, rather than the 35 local invariants "
                    "the earlier artifact compared against",
        "why_the_earlier_number_does_not_answer_it": (
            "results/architecture_sweep/OPERATOR_BANK_CEILING.json measured "
            "0.7595 for 35 local invariants and 0.7676 for 302, a lift of "
            "+0.0081 on 12 of 12 splits. No detector in the paper consumes "
            "local invariants: table_field consumes 645 wires, being 43 local "
            "quantities each expanded fifteen ways by neighbourhood mean, "
            "spread, difference and rank at four radii. The chain lag spectrum "
            "is itself computed at 8, 12 and 16 angstroms and the shape "
            "operator at 8, so multi-scale context may already have arrived by "
            "the other route, in which case the lift was against a bank nobody "
            "deploys"),
        "baseline": {"name": base_key, "n_wires": int(W.shape[1]),
                     "n_local_quantities": 43, "n_spatial_transforms": 15},
        "generated_family_sizes": fam_sizes,
        "what_each_family_is": {
            "chain lag spectrum": "the local Toeplitz symbol: how far along the "
                                  "chain a residue's spatial neighbours are, as "
                                  "a distribution over sequence lag. No "
                                  "function of the contact graph's spectrum can "
                                  "recover it, because that graph is the curve "
                                  "with the curve deleted",
            "shape operator": "curvature with a sign. The second moment of a "
                              "point cloud is positive definite and cannot tell "
                              "a saddle from a flat patch; the mouth of a cleft "
                              "is a saddle",
            "valuation profile": "the non-Archimedean ball profile: how "
                                 "neighbour count grows with radius, read as a "
                                 "valuation rather than a density",
            "soft-mode hinge": "the deformation. Shear modes of the elastic "
                               "network and a thermal average over them, which "
                               "is the only family here whose subject is the "
                               "motion that makes a cryptic pocket cryptic",
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "readout": "ridge Fisher discriminant fitted on the fit half, "
                       "columns standardised there, evaluated on the pick half",
            "ridge": RIDGE,
            "gate": "the multi-scale spatial gate the detectors use",
            "metric": "mean per-unit ROC-AUC on the pick half",
            "why_a_ceiling": "a Fisher discriminant solves for arbitrary real "
                             "coefficients over unquantised inputs. A counting "
                             "field must band every input into four levels and "
                             "may only add integers, so it cannot exceed this "
                             "and generally falls short of it. Every number "
                             "here is an upper bound, not a prediction",
        },
        "ceilings": {k: {"mean": round(float(v.mean()), 6),
                         "min": round(float(v.min()), 6),
                         "max": round(float(v.max()), 6)}
                     for k, v in curves.items()},
        "lift_over_the_wide_bank": lifts,
        "per_split": {k: [round(float(x), 6) for x in v]
                      for k, v in curves.items()},
        "n_units": len(units),
        "n_residues": int(len(W)),
        "n_positive_residues": int(y.sum()),
    }


def _report(d: dict) -> None:
    base = d["baseline"]["name"]
    print(f"\n{'bank':<44} {'ceiling':>9} {'lift':>9} {'splits+':>8}")
    print(f"{base:<44} {d['ceilings'][base]['mean']:>9.4f} {'':>9} {'':>8}")
    for k, L in sorted(d["lift_over_the_wide_bank"].items(),
                       key=lambda kv: -kv[1]["mean"]):
        print(f"{k:<44} {d['ceilings'][k]['mean']:>9.4f} "
              f"{L['mean']:>+9.4f} {L['n_splits_positive']:>5}/{L['n_splits']}")
    old = 0.008092
    best = max(L["mean"] for L in d["lift_over_the_wide_bank"].values())
    print(f"\nagainst the 35 local invariants the pooled lift was {old:+.4f} on "
          f"12/12 splits.")
    if best <= 0:
        print("against the 645 wires no family lifts at all: the spatial "
              "expansion had already said it, and this road is closed.")
    elif best < old / 2:
        print(f"against the 645 wires the best is {best:+.4f}, under half of "
              f"it. Most of the earlier lift was the expansion arriving by a "
              f"second route.")
    else:
        print(f"against the 645 wires the best is {best:+.4f}, which survives. "
              f"A counting construction over the lifting family is worth "
              f"building.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args(argv)

    d = build(a.splits, a.workers)
    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=1, allow_nan=False) + "\n")
    _report(d)
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
