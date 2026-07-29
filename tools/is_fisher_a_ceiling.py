#!/usr/bin/env python3
"""Is the Fisher discriminant a ceiling on the counting field, or only assumed to be?

Why this has to be checked
--------------------------
Several tools in this repository screen ideas by measuring a Fisher discriminant
and reasoning that no counting field can exceed it. The argument is that a Fisher
solve gets arbitrary real coefficients over unquantised inputs while a counting
field must band every wire into four levels and may only add integers, so the
former bounds the latter from above.

That argument has a hole, and it is not a small one. A counting field is not a
linear function of the wires. It is a linear function of the indicator functions of
quantised table cells --- a pair of banded wires addresses one of sixteen cells and
the field reads that cell's training frequency --- and that basis is strictly
richer than the wires themselves. It represents interactions between paired wires
that no linear solve over the raw wires can express. Nothing forces it below.

The reason this matters now rather than in general: the screen was used this week to
declare a road closed. WIDE_BANK_CEILING.json measured +0.0005 for 267 generated
descriptors and concluded they were not worth a counting construction. If the
ceiling is not a ceiling, that conclusion rests on a correlate rather than a bound,
and its strength has to be restated.

What makes this comparable and the earlier numbers not
-----------------------------------------------------
The ceiling artifacts apply a multi-scale gate summing five radii; the deployed
field applies one radius at one weight. A difference measured across those two
tools mixes readout with gate and cannot answer the question. So the Fisher arms
here run under the deployed field's own gate, on the same twelve cluster-disjoint
halvings and the same wires, leaving the readout as the only difference.

The counting arms are not recomputed. Their per-split numbers are read from
ANISOTROPIC_COUNTING_FIELD.json, which produced them on these same seeds; recomputing
them would cost forty minutes to reproduce numbers that are already frozen, and the
tool checks that the split count and wire widths agree before pairing anything.

Training folds only. No test residue and no external unit is read.

Usage: PYTHONPATH=src:tools python3.12 tools/is_fisher_a_ceiling.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from anisotropic_expansion_ceiling import build_or_load  # noqa: E402
from expand_invariant_bank import SEED, fisher  # noqa: E402
from select_architecture_on_train import cluster_half_split, per_unit_auc  # noqa: E402

from pocket_bench.methods.table_field import (
    GATE_RADIUS,
    GATE_WEIGHT,
    apply_gate,
)
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.is_fisher_a_ceiling.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
COUNTING = ROOT / "results/architecture_sweep/ANISOTROPIC_COUNTING_FIELD.json"
OUT = ROOT / "results/architecture_sweep/IS_FISHER_A_CEILING.json"


def counting_arms() -> tuple[dict[str, np.ndarray], dict]:
    """The frozen counting-field curves, keyed by wire width."""
    d = json.loads(COUNTING.read_text())
    by_width: dict[str, np.ndarray] = {}
    for name, xs in d["per_split"].items():
        width = int(name.split()[-2])
        by_width[str(width)] = np.asarray(xs, dtype=float)
    return by_width, d


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=str, default=str(OUT))
    a = ap.parse_args(argv)

    counting, cdoc = counting_arms()
    n_splits = cdoc["protocol"]["n_splits"]

    z = np.load(WIDE, allow_pickle=False)
    W, y, n_res, ctr = z["X"], z["y"], z["n_res_per"], z["ctr"]
    units = [str(u) for u in z["units"]]
    entries = json.loads(MANIFEST.read_text())["entries"]
    cluster_of = {f"{e['pdb']}_{e['chain']}": e["cluster_id"] for e in entries}
    A, _diag, _names = build_or_load(8)
    if len(A) != len(W):
        raise SystemExit("the anisotropic bank and the wide cache disagree on "
                         "how many residues the training fold has")

    banks = {str(W.shape[1]): W,
             str(W.shape[1] + A.shape[1]): np.concatenate([W, A], axis=1)}
    if sorted(banks) != sorted(counting):
        raise SystemExit(
            f"the counting artifact reports widths {sorted(counting)} and this "
            f"tool built {sorted(banks)}; pairing them would compare two "
            f"different banks")

    rows: dict[str, list[float]] = {k: [] for k in banks}
    for s in range(n_splits):
        is_fit, _ = cluster_half_split(units, cluster_of, SEED + s)
        row = np.repeat(np.arange(len(n_res)), n_res)
        fit, pick = is_fit[row], ~is_fit[row]
        n_pick = np.array([n for n, f in zip(n_res, is_fit) if not f])
        ctr_pick = ctr[pick]
        t0 = time.perf_counter()
        for k, X in banks.items():
            sc = apply_gate(fisher(X[fit], y[fit], X[pick]), ctr_pick, n_pick)
            rows[k].append(float(per_unit_auc(sc, y[pick], n_pick)))
        print(f"  split {s + 1}/{n_splits}  " + "  ".join(
            f"fisher{k} {rows[k][-1]:.4f}  count{k} {counting[k][s]:.4f}"
            for k in banks) + f"  {time.perf_counter() - t0:.0f}s", flush=True)

    fis = {k: np.asarray(v) for k, v in rows.items()}
    verdict = {}
    for k in banks:
        d = counting[k] - fis[k]
        verdict[k] = {
            "counting_mean": round(float(counting[k].mean()), 6),
            "fisher_mean": round(float(fis[k].mean()), 6),
            "counting_minus_fisher_mean": round(float(d.mean()), 6),
            "n_splits_counting_higher": int((d > 0).sum()),
            "n_splits": int(len(d)),
            "counting_higher_on_every_split": bool((d > 0).all()),
        }

    narrow, wide = str(W.shape[1]), str(W.shape[1] + A.shape[1])
    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "question": "is a Fisher discriminant over the wires a ceiling on the "
                    "counting field, as several tools here assume when they use "
                    "it to screen, or does the counting field exceed it",
        "why_the_assumption_is_not_safe": (
            "a counting field is not linear in the wires. It is linear in the "
            "indicator functions of quantised table cells: a pair of banded "
            "wires addresses one of sixteen cells and the field reads that "
            "cell's training frequency. That basis represents interactions "
            "between paired wires which no linear solve over raw wires can "
            "express, so nothing forces the counting field below the solve"),
        "what_makes_this_comparable": (
            "the ceiling artifacts gate with a multi-scale sum over five radii "
            f"and the deployed field gates at {GATE_RADIUS} A with weight "
            f"{GATE_WEIGHT}. Measuring across those two tools would mix readout "
            "with gate, so the Fisher arms here run under the deployed gate on "
            "the same splits and the same wires, leaving the readout as the only "
            "difference"),
        "counting_arms_were_not_recomputed": {
            "source": str(COUNTING.relative_to(ROOT)),
            "why": "they were produced on these same seeds and are frozen; "
                   "recomputing would spend forty minutes reproducing them",
            "checked": "the split count and both wire widths agree before any "
                       "pairing is done",
        },
        "protocol": {
            "n_splits": n_splits,
            "split": f"cluster-disjoint halves, seeds {SEED}..{SEED + n_splits - 1}",
            "readout_a": "counting field: four levels per chain, width-2 tables, "
                         "16 partition rounds, integer fan-out",
            "readout_b": "ridge Fisher discriminant over the same wires",
            "gate": f"{GATE_RADIUS} A at weight {GATE_WEIGHT}, the deployed one, "
                    f"for both",
            "metric": "mean per-unit ROC-AUC on the pick half",
        },
        "verdict": verdict,
        "what_this_does_to_the_earlier_screen": (
            "WIDE_BANK_CEILING.json rejected 267 generated descriptors on a "
            "+0.0005 Fisher movement, reasoning that a counting field could not "
            "do better. If the counting field is not bounded by the Fisher solve "
            "then that reasoning is a correlate and not a bound, and the "
            "rejection is weaker than it was stated to be. It is not overturned: "
            "ANISOTROPIC_COUNTING_FIELD.json shows the counting field collecting "
            "none of a +0.0037 Fisher movement, so the two readouts do track each "
            "other in the one case where both were measured. What changes is the "
            "confidence, and the fix is to compile rather than to screen"
            if verdict[narrow]["counting_minus_fisher_mean"] > 0 else
            "the Fisher solve does bound the counting field on this measurement, "
            "so using it as a screen is sound and the earlier rejection stands as "
            "stated"),
        "per_split": {
            f"fisher {narrow}": [round(float(x), 6) for x in fis[narrow]],
            f"fisher {wide}": [round(float(x), 6) for x in fis[wide]],
            f"counting {narrow}": [round(float(x), 6) for x in counting[narrow]],
            f"counting {wide}": [round(float(x), 6) for x in counting[wide]],
        },
        "fisher_lift_from_anisotropy_under_the_deployed_gate": {
            "mean": round(float((fis[wide] - fis[narrow]).mean()), 6),
            "n_splits_positive": int((fis[wide] > fis[narrow]).sum()),
            "n_splits": n_splits,
            "under_the_multiscale_gate": 0.003728,
            "note": "if these two differ much, the anisotropic lift was partly a "
                    "property of the gate rather than of the columns",
        },
    }

    out = Path(a.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, allow_nan=False) + "\n")

    print()
    for k, v in verdict.items():
        print(f"  {k} wires: counting {v['counting_mean']:.4f}  fisher "
              f"{v['fisher_mean']:.4f}  diff "
              f"{v['counting_minus_fisher_mean']:+.4f}  counting higher on "
              f"{v['n_splits_counting_higher']}/{v['n_splits']}")
    fl = doc["fisher_lift_from_anisotropy_under_the_deployed_gate"]
    print(f"\n  anisotropic lift to the Fisher arm under the deployed gate: "
          f"{fl['mean']:+.4f} on {fl['n_splits_positive']}/{fl['n_splits']}  "
          f"(multi-scale gate gave {fl['under_the_multiscale_gate']:+.4f})")
    shown = out.relative_to(ROOT) if out.is_relative_to(ROOT) else out
    print(f"\nwrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
