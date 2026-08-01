#!/usr/bin/env python3
"""Fix the resolution-stratified read of the pLM-NN deficit before it is run.

The hypothesis, and where it came from
---------------------------------------
The deficit against pLM-NN is −0.0243 on the official fold and −0.0340 on external
Set A, and it has been treated as a flat property of the two methods. Set B and Set
C say it is not. That read found a **modality effect**, confirmed by its own control:
on cryo-EM the coordinate-reading methods lose accuracy and the sequence model gains
(`AGENT_MEMORY 2p`, `results/external/SETBC_READ.json`). A sequence model reads a
sequence and cannot be hurt by coordinate error; a detector that counts geometry can.

If that is the mechanism, it does not stop at the boundary between two experimental
methods. Within X-ray, coordinate error falls as resolution improves, so the same
argument predicts that **the deficit narrows on the better-resolved half of the
fold** — not because the counting field is better there in absolute terms, but
because the comparison is between a method that can use good coordinates and one that
cannot.

This is a prediction from an independently obtained observation, not a search. It is
written here, with the losing sentences, before any stratified number exists.

The split, and the one thing that was looked at first
------------------------------------------------------
2.0 Å, the conventional line between a high- and a medium-resolution X-ray structure.
The fold spans 0.701–2.495 Å and the line puts 98 units below it and 94 above, which
is close to even; the distribution of the covariate was inspected before fixing the
line and no score was. A finer ladder is not used because the fold's best stratum,
≤1.5 Å, holds 13 units and could not carry an interval.

The control that decides whether the answer means anything
-----------------------------------------------------------
**P2Rank reads coordinates too.** If coordinate trustworthiness is what moves the
comparison, then P2Rank's own margin over pLM-NN must move in the same direction
across the same two strata. If ours moves and P2Rank's does not, then whatever is
happening is a property of this detector at this resolution and not of the mechanism
claimed, and the plan says so in advance rather than letting a single arm be read as
a law.

Two covariates are reported per stratum for the same reason: chain length and
positive rate. High-resolution structures are smaller on average, and a per-unit AUC
that rises on small chains would produce this pattern with no coordinate argument at
all.

Usage: PYTHONPATH=src:tools python3.12 tools/preregister_resolution.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/official_fold/OFFICIAL_RESOLUTIONS.json"
RECOVERY = ROOT / "results/official_fold/RECOVERY_READ.json"
OUT = ROOT / "results/architecture_sweep/PREREGISTERED_RESOLUTION.json"
SCHEMA = "geoaudit.preregistered_resolution.v1"

CUT = 2.0
N_BOOT = 10000
SEED = 20260801
ALPHA = 0.05


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> dict:
    res = json.loads(RES.read_text())
    below = sum(1 for u in res["units"]
                if u["resolution"] is not None and u["resolution"] <= CUT)
    above = sum(1 for u in res["units"]
                if u["resolution"] is not None and u["resolution"] > CUT)
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "declared": "confirmatory for the primary, descriptive for the covariates",
        "reads_test_fold": True,
        "written_before_any_stratified_number_existed": True,
        "question": ("does the counting field's deficit against pLM-NN narrow on "
                     "the better-resolved half of the official test fold"),
        "hypothesis_and_its_source": {
            "statement": ("the deficit narrows as resolution improves, because a "
                          "sequence model cannot be hurt by coordinate error and a "
                          "geometry counter can"),
            "came_from": ("results/external/SETBC_READ.json and AGENT_MEMORY 2p, "
                          "where a modality effect was found on cryo-EM and "
                          "confirmed by its own control. This read asks whether the "
                          "same mechanism operates inside one modality"),
            "is_not": ("a search over covariates. One covariate, one split, one "
                       "direction, all named here"),
        },
        "inputs": {
            "per_unit_aucs": {"artifact": str(RECOVERY.relative_to(ROOT)),
                              "sha256": _sha(RECOVERY),
                              "why_this_one": ("it already carries all four "
                                               "methods' per-unit ROC-AUC on the "
                                               "192 units, computed through one "
                                               "harness, so no method is re-scored "
                                               "and no metric is recomputed")},
            "resolutions": {"artifact": str(RES.relative_to(ROOT)),
                            "sha256": _sha(RES)},
        },
        "why_this_costs_a_read_index": (
            "no model is re-run and no threshold is re-fitted, but a new statistic "
            "is formed over held-out units. A ledger that counted only re-runs "
            "would make an unlimited number of such statements free"),
        "strata": {
            "cut_angstrom": CUT,
            "high": {"definition": f"resolution <= {CUT} A", "n_units": below},
            "low": {"definition": f"resolution > {CUT} A", "n_units": above},
            "why_two_and_not_a_ladder": (
                "the fold's best conventional stratum, 1.5 A or better, holds 13 "
                "units and cannot carry an interval"),
            "what_was_inspected_before_fixing_the_cut": (
                "the distribution of resolutions, which is a covariate. No score, "
                "label or difference was"),
        },
        "primary": {
            "statistic": ("mean paired per-unit ROC-AUC difference, counting field "
                          "minus pLM-NN, computed separately in each stratum"),
            "endpoint": ("the difference of the two, high stratum minus low "
                         "stratum. Positive means the deficit is smaller where the "
                         "coordinates are better"),
            "n_bootstrap": N_BOOT,
            "seed": SEED,
            "resampling_unit": "the unit, resampled within its own stratum",
            "alpha": ALPHA,
            "resolved_means": "the 95% interval of the difference excludes zero",
        },
        "control": {
            "name": "p2rank_minus_plmnn_across_the_same_strata",
            "what": ("the identical endpoint with P2Rank in place of the counting "
                     "field"),
            "why": ("P2Rank reads coordinates too. If coordinate trustworthiness is "
                    "the mechanism, its margin must move the same way"),
            "how_it_is_used": (
                "if the counting field's endpoint resolves and P2Rank's has the "
                "opposite sign, the mechanism is not supported and the plan's "
                "sentence for that case is written below"),
        },
        "covariates_reported_per_stratum": [
            "mean chain length, because high-resolution structures are smaller and "
            "a per-unit AUC that rises on small chains reproduces this pattern with "
            "no coordinate argument",
            "mean positive rate, because a stratum with denser labels is an easier "
            "ranking problem for every method",
        ],
        "outcome_sentences_written_in_advance": {
            "narrows_and_the_control_agrees":
                "The deficit against pLM-NN narrows on the better-resolved half of "
                "the official test fold, and P2Rank's margin moves the same way. "
                "Both are consistent with the modality effect found on Set B and "
                "Set C: a sequence model is indifferent to coordinate error and a "
                "geometry counter is not. The deficit is therefore in part a "
                "property of the data, not only of the method.",
            "narrows_but_the_control_disagrees":
                "The deficit against pLM-NN narrows on the better-resolved half, "
                "but P2Rank's margin does not move the same way. The coordinate-"
                "trustworthiness explanation is not supported by its own control, "
                "and the narrowing is recorded as an observation about this "
                "detector rather than as a mechanism.",
            "does_not_resolve":
                "The difference between the two strata does not exclude zero. On "
                "192 units split at 2.0 A the fold is too small to say whether the "
                "deficit depends on resolution, and the prediction from the "
                "cryo-EM read is neither supported nor refuted here.",
            "widens":
                "The deficit against pLM-NN is larger on the better-resolved half "
                "of the fold, which is the opposite of what the cryo-EM modality "
                "effect predicts. The prediction is recorded as falsified.",
            "a_covariate_explains_it":
                "The two strata differ in chain length or positive rate by enough "
                "that the stratum effect cannot be separated from them. The read is "
                "reported and no mechanism is claimed.",
        },
        "changes_forbidden_after_this_file_is_committed": [
            "the cut, the number of strata, the statistic, the seed or the bootstrap"
            " count",
            "the substitution of a different covariate",
            "any architecture or threshold of any method",
        ],
        "what_this_cannot_show": [
            "that the counting field beats pLM-NN anywhere. A narrower deficit is "
            "still a deficit unless the interval says otherwise",
            "anything causal about resolution. Resolution travels with crystal "
            "quality, protein size and flexibility, and only two of those are "
            "reported here",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        if not OUT.exists():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        d = json.loads(OUT.read_text())
        s = d["strata"]
        print(f"\nplan for the resolution read, {d['declared']}")
        print(f"  cut {s['cut_angstrom']} A: {s['high']['n_units']} high, "
              f"{s['low']['n_units']} low")
        print(f"  primary: {d['primary']['endpoint'][:70]}")
        print(f"  control: {d['control']['name']}")
        print(f"  outcome sentences written in advance: "
              f"{len(d['outcome_sentences_written_in_advance'])}")
        for key, path in (("per_unit_aucs", RECOVERY), ("resolutions", RES)):
            if _sha(path) != d["inputs"][key]["sha256"]:
                print(f"FAILED {path.name} moved since the plan was written")
                return 1
        print(f"OK {OUT.relative_to(ROOT)}")
        return 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
