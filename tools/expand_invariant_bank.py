#!/usr/bin/env python3
"""Measure what a systematically generated invariant bank is worth, on training
folds only.

The question
------------
The Fisher discriminant of the thirty-five algebraic invariants reaches
\\FisherAll{} on the held-out fold, below P2Rank. That is a ceiling on every
linear readout of that bank, counting field included, so no better address and
no better fan-out can pass it. Passing it needs invariants the bank does not
contain. Four new ones bought +0.0026, which is what prompted generating them by
the hundred instead.

``pocket_bench.methods.operator_descriptors`` enumerates 190 of them as operator
family x scale x functional. This tool computes them on the training fold and
asks three things, in order:

  1. Does the Fisher ceiling move at all when the new bank is added?
  2. Is the movement a property of the split, or does it survive repeated
     cluster-disjoint halvings?
  3. Which families carry it, so that a bank of 225 can be cut back to something
     a counting construction could actually address?

The held-out fold is not touched, and the artifact says so. A ceiling measured
here is a training-fold ceiling; the quotient counterattack is the standing
reminder in this repository that the two can disagree.

Honest accounting of the comparison
-----------------------------------
The Fisher discriminant is fitted on the fit half and evaluated on the pick half,
so its number is out of sample. It is still an optimistic ceiling for a counting
field, which must quantise its inputs and cannot solve for arbitrary real
coefficients. The gap between the two is exactly what
``results/architecture_sweep/GAP_DECOMPOSITION.json`` measured, and nothing here
revises it.

Usage:
    PYTHONPATH=src:tools python3.12 tools/expand_invariant_bank.py
    PYTHONPATH=src:tools python3.12 tools/expand_invariant_bank.py --splits 8
    PYTHONPATH=src:tools python3.12 tools/expand_invariant_bank.py --check
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from pocket_bench.methods.algebraic_descriptors import (
    FEATURE_NAMES as ALG_NAMES,
)
from pocket_bench.methods.chain_operator_descriptors import (
    FEATURE_NAMES as CH_NAMES, chain_operator_residue_features,
)
from pocket_bench.methods.operator_descriptors import (
    FEATURE_NAMES as OP_NAMES, SCALES, operator_residue_features,
)
from pocket_bench.paths import ROOT

from select_architecture_on_train import (
    RADII, SEED, cluster_half_split, load_train_fold, patch_mean, per_unit_auc,
    _unit,
)

OUT = ROOT / "results/architecture_sweep/OPERATOR_BANK_CEILING.json"
SCHEMA = "geoaudit.operator_bank_ceiling.v2"
RIDGE = 1e-3
N_SPLITS = 12

# The two generated banks, each cached beside the descriptor cache it extends.
BANKS = {
    "operator": (ROOT / "data/cryptobench_apo/_operator_cache_train.npz",
                 OP_NAMES, operator_residue_features),
    "chain": (ROOT / "data/cryptobench_apo/_chain_operator_cache_train.npz",
              CH_NAMES, chain_operator_residue_features),
}


def _one_operator(args):
    off, ctr = args
    return off, operator_residue_features(ctr)


def _one_chain(args):
    off, ctr = args
    return off, chain_operator_residue_features(ctr)


_WORKERS = {"operator": _one_operator, "chain": _one_chain}


def build_cache(kind: str, workers: int = 8) -> np.ndarray:
    """Compute one generated bank for every training residue, once."""
    path, names, _fn = BANKS[kind]
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if z["F"].shape[1] == len(names) and list(z["names"]) == list(names):
            print(f"reusing {path.relative_to(ROOT)}  {z['F'].shape}")
            return z["F"]
        print(f"the cached {kind} bank does not match the current descriptor "
              f"set; rebuilding")

    _, _, n_res, ctr, units, _ = load_train_fold()
    offs = np.concatenate([[0], np.cumsum(n_res)]).astype(int)
    jobs = [(int(offs[k]), ctr[offs[k]:offs[k + 1]]) for k in range(len(n_res))]
    F = np.zeros((int(offs[-1]), len(names)), dtype=np.float32)
    t0 = time.perf_counter()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for off, block in pool.map(_WORKERS[kind], jobs, chunksize=4):
            F[off:off + len(block)] = block
            done += 1
            if done % 100 == 0 or done == len(jobs):
                print(f"  {kind}: {done}/{len(jobs)} units  "
                      f"{time.perf_counter() - t0:.0f}s", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, F=F, names=np.array(names))
    print(f"wrote {path.relative_to(ROOT)}  {F.shape}")
    return F


def fisher(Xfit, yfit, Xpick):
    """Ridge-regularised Fisher direction, fitted on one half only.

    Columns are standardised on the fit half so that a descriptor measured in
    cubic angstroms and one measured in nothing enter the solve on the same
    footing; the ridge is then a single number rather than an accident of units.
    """
    mu, sd = Xfit.mean(0), Xfit.std(0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    A = (Xfit - mu) / sd
    B = (Xpick - mu) / sd
    t = yfit.astype(np.float64) - yfit.mean()
    G = A.T @ A
    G.flat[::G.shape[0] + 1] += RIDGE * np.trace(G) / max(G.shape[0], 1) + 1e-9
    w = np.linalg.solve(G, A.T @ t)
    return B @ w


def gated(s, ctr_pick, n_pick):
    """The multi-scale spatial gate the detectors use, so the ceiling is
    measured on the same footing as the numbers it is compared with."""
    g = np.sum([_unit(patch_mean(s, ctr_pick, n_pick, r)) for r in RADII], 0)
    return _unit(s) + _unit(g)


def _family(name: str) -> str:
    """Which of the generator's axes a descriptor came from."""
    if "_dilation_" in name:
        return "deformation across scale"
    stem = name.split("@")[0]
    if stem.startswith(("lag_", "symbol", "n_segments", "segment_len",
                        "frac_lag")):
        return "chain lag spectrum"
    if stem.startswith(("kappa", "gauss_curv", "mean_curv", "shape_index",
                        "curvedness", "quadric_residual")):
        return "shape operator"
    if stem.startswith(("valuation", "ultrametric_")):
        return "valuation profile"
    if stem.startswith(("mode_", "thermal_", "hinge_", "relative_motion")) \
            or stem == "mode1_radial_cos":
        return "soft-mode hinge"
    if stem.startswith(("heat_diag", "resolvent_diag", "fiedler_mass",
                        "top_mass", "participation")):
        return "diagonal functional at the centre"
    if stem.startswith(("nspec", "heat_trace", "lap_m", "fiedler", "lap_max",
                        "spec_gap", "n_components")):
        return "spectral trace functional"
    if stem.startswith(("gyration", "anisotropy", "planarity", "sphericity",
                        "asphericity", "centre_offset")):
        return "gyration tensor"
    return "neighbourhood geometry"


def bank_set(F35, FOP, FCH):
    """The banks compared, named by what they contain."""
    return {
        f"algebraic {F35.shape[1]}": F35,
        f"operator {FOP.shape[1]}": FOP,
        f"chain {FCH.shape[1]}": FCH,
        f"algebraic+operator {F35.shape[1] + FOP.shape[1]}":
            np.concatenate([F35, FOP], axis=1),
        f"algebraic+chain {F35.shape[1] + FCH.shape[1]}":
            np.concatenate([F35, FCH], axis=1),
        f"combined {F35.shape[1] + FOP.shape[1] + FCH.shape[1]}":
            np.concatenate([F35, FOP, FCH], axis=1),
    }


def evaluate(banks, y, n_res, ctr, units, cluster_of, n_splits):
    rows = {k: [] for k in banks}
    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        row = np.repeat(np.arange(len(n_res)), n_res)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        ypick, ctr_pick = y[pick], ctr[pick]
        for name, X in banks.items():
            sc = fisher(X[fit], y[fit], X[pick])
            rows[name].append(per_unit_auc(gated(sc, ctr_pick, n_pick),
                                           ypick, n_pick))
        print(f"  split {s + 1}/{n_splits}  " + "  ".join(
            f"{k.split()[0]} {rows[k][-1]:.4f}" for k in banks), flush=True)
    return {k: np.array(v) for k, v in rows.items()}


def per_family(F35, FGEN, gen_names, y, n_res, ctr, units, cluster_of):
    """One split, each family added to the 35 on its own.

    A bank of 190 cannot be addressed by a counting construction, so what
    matters is which axis of the generator carries the lift and whether any of
    them carries it alone.
    """
    is_fit, _ = cluster_half_split(units, cluster_of, SEED)
    row = np.repeat(np.arange(len(n_res)), n_res)
    fit, pick = is_fit[row], ~is_fit[row]
    n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
    ypick, ctr_pick = y[pick], ctr[pick]

    base = per_unit_auc(gated(fisher(F35[fit], y[fit], F35[pick]),
                              ctr_pick, n_pick), ypick, n_pick)
    fams = sorted({_family(n) for n in gen_names})
    out = []
    for fam in fams:
        cols = [i for i, n in enumerate(gen_names) if _family(n) == fam]
        X = np.concatenate([F35, FGEN[:, cols]], axis=1)
        auc = per_unit_auc(gated(fisher(X[fit], y[fit], X[pick]),
                                 ctr_pick, n_pick), ypick, n_pick)
        out.append({"family": fam, "n_descriptors": len(cols),
                    "pick_half_roc_auc": round(float(auc), 6),
                    "delta_vs_algebraic_35": round(float(auc - base), 6)})
        print(f"  {fam:34s} +{len(cols):3d} -> {auc:.4f} "
              f"({auc - base:+.4f})", flush=True)
    # Scale sweep: the same question asked of the radii rather than the
    # functionals, because five scales is the other thing the old bank lacked.
    by_scale = []
    for r in SCALES:
        cols = [i for i, n in enumerate(gen_names) if n.endswith(f"@{r:g}")]
        if not cols:
            continue
        X = np.concatenate([F35, FGEN[:, cols]], axis=1)
        auc = per_unit_auc(gated(fisher(X[fit], y[fit], X[pick]),
                                 ctr_pick, n_pick), ypick, n_pick)
        by_scale.append({"radius_angstrom": r, "n_descriptors": len(cols),
                         "pick_half_roc_auc": round(float(auc), 6),
                         "delta_vs_algebraic_35": round(float(auc - base), 6)})
        print(f"  scale {r:>4g} A                        +{len(cols):3d} -> "
              f"{auc:.4f} ({auc - base:+.4f})", flush=True)
    return round(float(base), 6), out, by_scale


def build(n_splits: int, workers: int) -> dict:
    FOP = build_cache("operator", workers)
    FCH = build_cache("chain", workers)
    F35, y, n_res, ctr, units, cluster_of = load_train_fold()
    for nm, F in (("operator", FOP), ("chain", FCH)):
        if len(F) != len(F35):
            raise SystemExit(f"the {nm} cache and the algebraic cache disagree "
                             f"about how many residues the training fold has")
    banks = bank_set(F35, FOP, FCH)
    base_key = f"algebraic {F35.shape[1]}"
    top_key = f"combined {F35.shape[1] + FOP.shape[1] + FCH.shape[1]}"
    print(f"\n{len(units)} units, {len(F35)} residues, "
          f"{F35.shape[1]} + {FOP.shape[1]} + {FCH.shape[1]} descriptors")
    print(f"\nFisher ceiling over {n_splits} cluster-disjoint halvings:")
    curves = evaluate(banks, y, n_res, ctr, units, cluster_of, n_splits)
    print("\nper family, added to the 35 on one split:")
    gen_names = list(OP_NAMES) + list(CH_NAMES)
    FGEN = np.concatenate([FOP, FCH], axis=1)
    base, fams, scales = per_family(F35, FGEN, gen_names, y, n_res, ctr,
                                    units, cluster_of)

    d = curves[top_key] - curves[base_key]
    summary = {k: {"mean": round(float(v.mean()), 6),
                   "min": round(float(v.min()), 6),
                   "max": round(float(v.max()), 6)}
               for k, v in curves.items()}
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "question": "does a systematically generated operator bank move the "
                    "Fisher ceiling that bounds every counting readout",
        "generator": {
            "modules": [
                "src/pocket_bench/methods/operator_descriptors.py",
                "src/pocket_bench/methods/chain_operator_descriptors.py",
            ],
            "axes": "operator family x scale x functional",
            "scales_angstrom": list(SCALES),
            "n_operator_descriptors": len(OP_NAMES),
            "n_chain_descriptors": len(CH_NAMES),
            "n_algebraic_descriptors": len(ALG_NAMES),
            "second_wave_rationale": (
                "the first module is built from the contact-graph Laplacian and "
                "the neighbourhood's second moment, both invariant under "
                "permuting residues, so neither can see the chain order and "
                "neither can distinguish a saddle from a flat patch. The second "
                "module supplies the local Toeplitz symbol of the sequence "
                "lags, the shape operator of a fitted quadric, the "
                "non-Archimedean valuation profile, and the participation and "
                "shear of the soft modes"),
            "inputs": "residue centroids and chain order only; no atoms, no "
                      "fitted parameter, no RNG, no ligand",
        },
        "keys": {"baseline": base_key, "full": top_key},
        "protocol": {
            "n_splits": n_splits,
            "split": "cluster-disjoint halves, seeds "
                     f"{SEED}..{SEED + n_splits - 1}",
            "readout": "ridge Fisher discriminant fitted on the fit half, "
                       "columns standardised there, evaluated on the pick half",
            "ridge": RIDGE,
            "gate": f"multi-scale spatial gate at {list(RADII)} A, as the "
                    f"detectors use",
            "metric": "mean per-unit ROC-AUC on the pick half",
            "why_a_ceiling": "the Fisher discriminant solves for arbitrary real "
                             "coefficients over unquantised inputs, which no "
                             "counting field can do, so this bounds them from "
                             "above and does not predict them",
        },
        "ceilings": summary,
        "lift": {
            "mean": round(float(d.mean()), 6),
            "min": round(float(d.min()), 6),
            "max": round(float(d.max()), 6),
            "n_splits_positive": int((d > 0).sum()),
            "n_splits": int(len(d)),
        },
        "per_split": {k: [round(float(x), 6) for x in v]
                      for k, v in curves.items()},
        "single_split_baseline": base,
        "by_family": fams,
        "by_scale": scales,
    }


def _report(doc: dict) -> None:
    l = doc["lift"]
    for k, v in doc["ceilings"].items():
        print(f"{k:26s} {v['mean']:.4f}")
    print(f"lift {l['mean']:+.4f}  (worst {l['min']:+.4f}, "
          f"positive on {l['n_splits_positive']}/{l['n_splits']} splits)")


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
    gen = doc.get("generator") or {}
    if gen.get("n_operator_descriptors") != len(OP_NAMES):
        bad.append(f"the artifact records {gen.get('n_operator_descriptors')} "
                   f"operator descriptors, the module now defines "
                   f"{len(OP_NAMES)}")
    if gen.get("n_chain_descriptors") != len(CH_NAMES):
        bad.append(f"the artifact records {gen.get('n_chain_descriptors')} "
                   f"chain descriptors, the module now defines {len(CH_NAMES)}")
    if gen.get("n_algebraic_descriptors") != len(ALG_NAMES):
        bad.append("the algebraic bank has changed size since this was measured")
    keys = doc.get("keys") or {}
    base_key, top_key = keys.get("baseline"), keys.get("full")
    per = doc.get("per_split") or {}
    for k, v in (doc.get("ceilings") or {}).items():
        if k not in per:
            bad.append(f"{k}: no per-split values recorded")
            continue
        if abs(float(np.mean(per[k])) - v["mean"]) > 5e-6:
            bad.append(f"{k}: the mean ceiling does not average its own splits")
    lift = doc.get("lift") or {}
    if per.get(top_key) and per.get(base_key):
        d = np.array(per[top_key]) - np.array(per[base_key])
        if abs(float(d.mean()) - lift.get("mean", 0)) > 5e-6:
            bad.append("the lift does not follow from the recorded per-split "
                       "ceilings")
        if int((d > 0).sum()) != lift.get("n_splits_positive"):
            bad.append("the count of positive splits does not follow from the "
                       "recorded per-split ceilings")
    else:
        bad.append("the artifact does not record per-split values for the two "
                   "banks its lift is a difference of")
    for b in bad:
        print(f"FAIL {OUT.relative_to(ROOT)}: {b}")
    if bad:
        return 1
    print(f"OK {OUT.relative_to(ROOT)}: "
          f"{gen['n_operator_descriptors']}+{gen['n_chain_descriptors']} "
          f"generated descriptors, ceiling "
          f"{doc['ceilings'][base_key]['mean']:.4f} -> "
          f"{doc['ceilings'][top_key]['mean']:.4f} over "
          f"{lift['n_splits']} splits, test fold unread")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        return audit()
    doc = build(a.splits, a.workers)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")
    _report(doc)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    return audit()


if __name__ == "__main__":
    raise SystemExit(main())
