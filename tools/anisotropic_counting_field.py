#!/usr/bin/env python3
"""Does the counting field actually collect the anisotropic lift? Training folds only.

A ceiling is not a method. ANISOTROPIC_EXPANSION_CEILING.json shows that resolving
the expansion's neighbourhood statistics into a bulkward and an outward half lifts
the Fisher ceiling by +0.0037 on twelve of twelve cluster-disjoint halvings, seven
times what 267 generated descriptors managed against the same baseline. But that
ceiling is a linear discriminant over unquantised inputs, and the deployed detector
is neither: it bands every wire into four levels against its own chain's order
statistics, reads pairs of them through dense tables, and may only add integers
with fan-out multiplicities. A construction that quantises can fail to collect a
lift that a real-coefficient solve can see, and this repository has one standing
example of exactly that --- the quotient construction beat the dense bank on all
fourteen training splits and lost to it on the test fold.

So this compiles the real thing. Same table width, partition rounds, partition
seed, ridge, fan-out cap and gate as the published field; the only difference
between the two arms is 129 columns of wire asymmetry.

What is held fixed, and why that matters here
---------------------------------------------
The gate is the deployed one at a single radius and weight, not the best of a set.
Choosing a gate per arm would let the wider arm win by finding a better gate rather
than by carrying more information, and the question is which of those happened.

The partition seed is also held fixed across arms, which is not free of
consequence and is worth stating plainly. The tables are random pairings of wires,
so a bank over 774 wires is not the bank over 645 plus new tables --- it is a
different bank, and some pairings the narrow arm had are gone. Both arms therefore
get every split's bank regenerated from the same seed at their own width, which is
what the compiler does in deployment, and the comparison is between two fields each
built the way the real one is built.

Both halves are inside the official training fold and cluster-disjoint from each
other. The test fold and every external unit are untouched, and the artifact
records it.

Usage: PYTHONPATH=src:tools python3.12 tools/anisotropic_counting_field.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from anisotropic_expansion_ceiling import (  # noqa: E402  (sibling tool)
    ASYM_RADII,
    build_or_load,
)
from expand_invariant_bank import N_SPLITS, SEED  # noqa: E402
from select_architecture_on_train import cluster_half_split, per_unit_auc  # noqa: E402

from pocket_bench.methods.table_bank import (
    cell_offsets,
    chain_digits,
    compile_cells,
    integer_fanout,
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
    apply_gate,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.anisotropic_counting_field.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
OUT = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"


def load_wide():
    z = np.load(WIDE, allow_pickle=False)
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    missing = [u for u in units if u not in cluster_of]
    if missing:
        raise SystemExit(f"{len(missing)} cached units absent from the manifest")
    return z["X"], z["y"], z["n_res_per"], z["ctr"], units, cluster_of


def one_arm(D: np.ndarray, y: np.ndarray, fit: np.ndarray, pick: np.ndarray,
            ctr_pick: np.ndarray, n_pick: np.ndarray) -> float:
    """Compile on the fit half, score the pick half, exactly as deployment does.

    The digits are computed once over the whole fold before splitting, because
    banding is per chain against that chain's own order statistics and so cannot
    see the split at all. Recomputing them per half would give the same numbers
    and cost twice as much.
    """
    tables = partition_tables(D.shape[1], TABLE_WIDTH, PARTITION_ROUNDS,
                             PARTITION_SEED)
    offsets = cell_offsets(tables)
    frac, _tot = compile_cells(D[fit], y[fit], tables, offsets)
    mult = integer_fanout(D[fit], y[fit], tables, offsets, frac, RIDGE,
                          FAN_OUT_CAP)
    s = apply_gate(score(D[pick], tables, offsets, frac, mult), ctr_pick, n_pick)
    return float(per_unit_auc(s, y[pick], n_pick))


def build(n_splits: int) -> dict:
    A, diag, aniso_names = build_or_load(8)
    W, y, n_res, ctr, units, cluster_of = load_wide()
    if len(A) != len(W):
        raise SystemExit(
            f"the anisotropic bank has {len(A)} rows and the wide cache "
            f"{len(W)}; concatenating them would pair each residue with "
            f"another one's descriptors")

    print(f"banding {W.shape[1]} and {W.shape[1] + A.shape[1]} wires per chain",
          flush=True)
    t0 = time.perf_counter()
    D_narrow = chain_digits(np.asarray(W, dtype=np.float64), n_res)
    D_wide = chain_digits(
        np.asarray(np.concatenate([W, A], axis=1), dtype=np.float64), n_res)
    print(f"  banded in {time.perf_counter() - t0:.0f}s", flush=True)

    arms = {f"counting field over {W.shape[1]} wires": D_narrow,
            f"counting field over {W.shape[1] + A.shape[1]} wires": D_wide}
    rows: dict[str, list[float]] = {k: [] for k in arms}
    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        row = np.repeat(np.arange(len(n_res)), n_res)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        ctr_pick = ctr[pick]
        t1 = time.perf_counter()
        for name, D in arms.items():
            rows[name].append(one_arm(D, y, fit, pick, ctr_pick, n_pick))
        a, b = (rows[k][-1] for k in arms)
        print(f"  split {s + 1}/{n_splits}  narrow {a:.4f}  wide {b:.4f}  "
              f"{b - a:+.4f}  {time.perf_counter() - t1:.0f}s", flush=True)

    curves = {k: np.array(v) for k, v in rows.items()}
    narrow_key, wide_key = list(arms)
    d = curves[wide_key] - curves[narrow_key]
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "the anisotropic columns lift the Fisher ceiling by +0.0037 "
                    "on 12/12 splits; does the counting field, which quantises "
                    "every wire to four levels and may only add integers, "
                    "actually collect any of it",
        "why_the_ceiling_does_not_settle_it": (
            "a Fisher discriminant solves for arbitrary real coefficients over "
            "unquantised inputs and the deployed field does neither. A "
            "construction that quantises can fail to collect a lift a "
            "real-coefficient solve can see, and this repository has a standing "
            "example: the quotient construction beat the dense bank on all "
            "fourteen training splits and lost to it on the test fold"),
        "held_fixed_across_arms": {
            "table_width": TABLE_WIDTH,
            "partition_rounds": PARTITION_ROUNDS,
            "partition_seed": PARTITION_SEED,
            "ridge": RIDGE,
            "fan_out_cap": FAN_OUT_CAP,
            "gate_radius": GATE_RADIUS,
            "gate_weight": GATE_WEIGHT,
            "why_one_gate_and_not_the_best_of_several": (
                "choosing a gate per arm would let the wider arm win by finding "
                "a better gate rather than by carrying more information, and "
                "which of those happened is the question"),
            "what_the_shared_seed_does_not_mean": (
                "the wide bank is not the narrow bank plus new tables. Tables "
                "are random pairings of wires, so widening the bus regenerates "
                "every pairing and some the narrow arm had are gone. Each arm "
                "gets the bank the compiler would build at its own width from "
                "the same seed, which is what deployment does"),
        },
        "anisotropic_columns": {
            "n": int(A.shape[1]),
            "radii_angstrom": [float(r) for r in ASYM_RADII],
            "statistic": "bulkward half-neighbourhood mean minus outward half",
            "degenerate_fraction_per_radius":
                diag["fraction_degenerate_per_radius"],
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "compile_on": "the fit half only",
            "evaluate_on": "the pick half",
            "metric": "mean per-unit ROC-AUC",
            "banding": "per chain against that chain's own order statistics, "
                       "computed over the whole fold before splitting because it "
                       "cannot see the split",
        },
        "arms": {k: {"mean": round(float(v.mean()), 6),
                     "min": round(float(v.min()), 6),
                     "max": round(float(v.max()), 6)}
                 for k, v in curves.items()},
        "paired_difference": {
            "mean": round(float(d.mean()), 6),
            "min": round(float(d.min()), 6),
            "max": round(float(d.max()), 6),
            "sd": round(float(d.std(ddof=1)), 6),
            "n_splits_positive": int((d > 0).sum()),
            "n_splits": int(len(d)),
            "positive_on_every_split": bool((d > 0).all()),
            "why_no_confidence_interval": (
                "twelve halvings of one fold are twelve views of the same 770 "
                "units, not twelve independent samples, so an interval over them "
                "would understate its own width. The consistency count is what "
                "this design supports"),
        },
        "ceiling_for_comparison": {
            "source": "results/architecture_sweep/ANISOTROPIC_EXPANSION_CEILING.json",
            "fisher_lift": 0.003728,
            "n_splits_positive": 12,
        },
        "per_split": {k: [round(float(x), 6) for x in v]
                      for k, v in curves.items()},
        "n_units": len(units),
        "n_residues": int(len(W)),
        "n_positive_residues": int(y.sum()),
    }


def _report(d: dict) -> None:
    for k, v in d["arms"].items():
        print(f"  {k:<44} {v['mean']:.4f}")
    p = d["paired_difference"]
    ceil = d["ceiling_for_comparison"]["fisher_lift"]
    print(f"\n  paired difference {p['mean']:+.4f} on "
          f"{p['n_splits_positive']}/{p['n_splits']} splits "
          f"(sd {p['sd']:.4f})")
    print(f"  the Fisher ceiling moved {ceil:+.4f} on 12/12")
    if p["mean"] <= 0:
        print("\n  the counting field collects none of it. Quantisation to four "
              "levels and integer addition cannot use the asymmetry, so the "
              "ceiling was not reachable by this construction.")
    elif p["mean"] < ceil / 2:
        print(f"\n  the counting field collects {p['mean'] / ceil:.0%} of the "
              f"ceiling movement. Real but under half, which is what "
              f"quantisation costs.")
    else:
        print(f"\n  the counting field collects {p['mean'] / ceil:.0%} of the "
              f"ceiling movement. Worth compiling into the deployed field and "
              f"preregistering a test-fold read.")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", type=int, default=N_SPLITS)
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args(argv)

    d = build(a.splits)
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
