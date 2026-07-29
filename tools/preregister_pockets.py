#!/usr/bin/env python3
"""The plan for the ninth read: give the counting field pockets, then score them.

Every number this paper reports is per-residue, and a per-residue AUC is not what
a structural biologist acts on. They act on a ranked list of candidate sites, and
the questions that matter are whether the top one is right, whether the top three
contain a right one, and how many candidates they have to look at. Those questions
have never been asked of this method, for a plain reason: it does not produce
pockets. Its ``pockets`` field in the frozen predictions carries five placeholder
centres at the origin, which is why every DCA column in the telemetry is null.

So this read has two parts and only the second is a comparison. First the field is
given a pocket stage: high-scoring residues are clustered spatially, clusters are
ranked, and each gets a centre. That is new machinery and it is specified here in
full before it is run, because a clustering rule chosen after seeing which
threshold puts the right pocket first would be worthless. Then the ranked lists are
scored against the labels and against P2Rank, at a matched number of candidates.

What makes the specification honest rather than merely written down is that every
constant in it is either taken from an existing artifact or fixed by an argument
that does not mention the held-out fold:

  The residue budget is the shipped operating point, the same top-q per chain the
  deployment rule already uses. No new threshold is introduced.

  The linkage cutoff is the contact distance the method's own descriptors already
  use for residue adjacency, so it is not a new constant either.

  The hit criterion is 4 A, which is the convention P2Rank and the cryptic-site
  literature report DCA at. It is not tuned and no other value is reported as
  primary.

The distance is to the labelled site rather than to a ligand. There is no ligand:
these are apo structures and the benchmark's answer is a residue set. That makes
the quantity a distance to the nearest labelled cryptic atom, which is a different
convention from DCA-to-ligand and is named differently throughout so the two
cannot be confused when someone compares against a holo-based number.

Usage: PYTHONPATH=src:tools python3.12 tools/preregister_pockets.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from pocket_bench.paths import ROOT

FIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"
WIRES = ROOT / "data/cryptobench_apo/EXPANDED_WIRES.json"
PRED = ROOT / "results/cryptobench_official/predictions"
MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
OUT = ROOT / "results/architecture_sweep/PREREGISTERED_POCKETS.json"

SCHEMA = "geoaudit.preregistered_pockets.v1"
READ_INDEX = 9
# The distances a pocket prediction is scored at. 4 A is the convention; the
# other two are reported so that a reader can see whether a verdict rests on the
# convention, which is a different question from whether it is significant.
HIT_RADII = (4.0, 6.0, 8.0)
TOP_K = (1, 3, 5)
N_BOOT = 10000
SEED = 20260725


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _p2rank_candidate_counts() -> dict:
    """How many pockets P2Rank offers per chain, from the frozen predictions.

    This is not a result: it is P2Rank's own output, already committed, and the
    matched-budget comparison needs it to know what budget to match. Reading it
    here rather than at read time is what lets the plan state the budget.
    """
    units = json.loads((PRED / "p2rank.json").read_text())["units"]
    n = sorted(len(u.get("pockets") or []) for u in units.values())
    return {"n_units": len(n), "min": n[0], "max": n[-1],
            "median": n[len(n) // 2],
            "mean": round(sum(n) / len(n), 3),
            "n_units_with_no_pocket": sum(1 for x in n if x == 0)}


def build() -> dict:
    field = json.loads(FIELD.read_text())
    q = field["operating_point"]["q"]
    # Both radii come from the descriptor module rather than from this file, so
    # the clustering cutoff is an existing constant of the method and not a knob
    # introduced for this read. Of the two, the pinch radius is the one whose
    # documented meaning is about clefts separating, which is the question a
    # pocket boundary asks; the contact radius is the adjacency scale and is
    # carried as a sensitivity because single linkage at 10 A among sparse
    # high-scoring residues can chain across a surface.
    from pocket_bench.methods import algebraic_descriptors as alg
    pinch_r = float(alg._PINCH_R)
    contact_r = float(alg._NBR_R)
    if not (0 < pinch_r < contact_r):
        raise SystemExit(
            f"the descriptor module gives pinch radius {pinch_r} and contact "
            f"radius {contact_r}; the plan is written on the assumption that "
            f"the pinch radius is the tighter of the two")

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "for_test_fold_read_index": READ_INDEX,
        "written_before_the_read": True,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "head_when_written": _git("rev-parse", "HEAD"),

        "question": (
            "when the counting field's per-residue scores are turned into a "
            "ranked list of candidate pockets, does its top candidate find a "
            "labelled cryptic site as often as P2Rank's, at a matched number of "
            "candidates"),
        "status_declared_in_advance": "exploratory",
        "why_exploratory": (
            "the primary endpoint of this paper is an unresolved per-residue "
            "mean, and a pocket-level comparison is a different measurement "
            "rather than a second chance at the same one. It also requires "
            "machinery this method did not previously have, so a favourable "
            "result would be a statement about the machinery as much as about "
            "the field"),

        "what_the_field_does_not_currently_have": {
            "fact": "its committed predictions carry five pocket entries per "
                    "chain whose centres are all the origin",
            "consequence": "every DCA column in the frozen telemetry is null "
                           "for it, and no pocket-level number about this "
                           "method exists anywhere in the repository",
            "why_this_is_stated": "a reader should know the pocket stage was "
                                  "built for this read rather than being part "
                                  "of the method as evaluated everywhere else",
        },

        "pocket_construction": {
            "step_1_residue_budget": {
                "rule": f"the top {q:.2f} of residues per chain by score",
                "q": q,
                "why_this_q": "it is the shipped operating point, already used "
                              "by the deployment rule and already selected on "
                              "the training fold. No new threshold is chosen",
            },
            "step_2_clustering": {
                "rule": "single linkage on residue centroids",
                "cutoff_angstrom": pinch_r,
                "why_this_cutoff": (
                    "it is the method's own pinch radius, read from "
                    "pocket_bench.methods.algebraic_descriptors, whose "
                    "documented meaning is the lining radius at which a cleft "
                    "actually disconnects. That is the question a pocket "
                    "boundary asks, so it is the existing constant that fits "
                    "rather than a new one chosen here"),
                "sensitivity_cutoff_angstrom": contact_r,
                "why_a_sensitivity": (
                    "the descriptors' residue adjacency scale is looser, and "
                    "single linkage at that distance can chain across a surface "
                    "and merge distinct sites into one candidate. Reporting "
                    "both shows whether the verdict depends on the cutoff, and "
                    "the tighter one is primary because it was chosen for its "
                    "meaning rather than for its result"),
                "residue_position": "the unweighted centroid of the residue's "
                                    "heavy atoms in the receptor as scored",
            },
            "step_3_ranking": {
                "rule": "clusters ordered by the sum of their members' scores",
                "why_sum_and_not_mean": "a mean would rank a single very high "
                                        "residue above a genuine pocket, and "
                                        "the quantity a site prediction is "
                                        "about is the size of the signal, not "
                                        "its density. The mean ordering is "
                                        "reported as a sensitivity, not as the "
                                        "primary",
            },
            "step_4_centre": {
                "rule": "the score-weighted mean of the member residue "
                        "centroids",
            },
            "no_free_parameters": (
                "q comes from the shipped field, both cutoffs from the "
                "descriptor module, and steps 3 and 4 have no constant at all"),
        },

        "metrics_to_be_reported": [
            "hit rate at top 1, top 3 and top 5 candidates",
            "distance from the top candidate's centre to the nearest labelled "
            "cryptic atom",
            "recall of labelled residues covered by the top K candidates",
            "number of candidates offered per chain",
        ],
        "hit_radii_angstrom": list(HIT_RADII),
        "primary_hit_radius_angstrom": HIT_RADII[0],
        "top_k": list(TOP_K),
        "why_the_distance_is_not_called_dca": (
            "DCA is conventionally the distance from a predicted pocket centre "
            "to the nearest ligand atom. These are apo structures with no "
            "ligand, and the benchmark's answer is a residue set, so the "
            "quantity here is a distance to the nearest labelled cryptic atom. "
            "It is named distance_to_labelled_site throughout so that it cannot "
            "be compared against a holo-based DCA by accident"),

        "the_matched_budget": {
            "problem": "P2Rank offers a variable number of pockets per chain "
                       "and a clustering can be made to offer any number, so a "
                       "hit rate at top 1 compares different things unless the "
                       "candidate counts are held together",
            "rule": "every top-K comparison is at the same K for both methods, "
                    "and the number of candidates each method actually offers "
                    "is reported beside every rate so that a K larger than a "
                    "method's list is visible rather than silently favourable",
            "p2rank_candidates_as_committed": _p2rank_candidate_counts(),
        },

        "statistic": {
            "per_unit": "a hit is a chain-level indicator; a distance is one "
                        "number per chain",
            "summary": "unweighted mean over the chains both methods offer at "
                       "least one candidate on",
            "paired": "the same resampled chains enter both arms",
            "n_boot": N_BOOT,
            "seed": SEED,
            "ci": 0.95,
        },
        "multiplicity": {
            "n_primary_tests": len(TOP_K),
            "correction": "Bonferroni over the three top-K hit rates at the "
                          "primary radius",
            "corrected_level": round(0.05 / len(TOP_K), 6),
            "the_other_radii_are_not_tests": (
                "6 and 8 A are reported to show whether a verdict depends on "
                "the convention. They are not counted in the correction and no "
                "conclusion may rest on them alone"),
        },

        "guards_the_read_must_pass": [
            "this plan's commit is an ancestor of HEAD, checked with git",
            "the clustered residues are exactly the top-q set the deployment "
            "rule calls positive, checked against the committed "
            "residue_positive list for every chain",
            "every residue in the budget lands in exactly one cluster",
            "each cluster's centre lies within the cutoff of at least one of "
            "its own members",
            "P2Rank's pocket centres are its own committed ones, unmodified",
            "the number of chains entering each paired difference is reported",
        ],

        "decision_rules": {
            "applies_to": "the top-1 hit rate at the primary radius",
            "if_the_interval_excludes_zero_in_our_favour": (
                "it is reported as the field's ranked list finding a labelled "
                "site more often than P2Rank's at a matched budget, and "
                "labelled exploratory in the same sentence"),
            "if_the_interval_crosses_zero": (
                "it is reported as unresolved, and the pocket stage is "
                "described as making the method usable at the site level "
                "without making it better than P2Rank there"),
            "if_the_interval_excludes_zero_in_p2ranks_favour": (
                "it is reported as P2Rank being the better site predictor, "
                "with the same prominence, and the paper's pocket-level claim "
                "becomes that the field can be given a site stage at all"),
            "a_larger_k_cannot_rescue_a_smaller_one": (
                "if top 1 is unresolved and top 3 is not, the conclusion is "
                "about top 3 only and is reported as such. The three are "
                "corrected together for exactly this reason"),
        },

        "what_will_be_written_under_each_outcome": {
            "top1_favours_the_field": (
                "Given a pocket stage built from its own scores, the counting "
                "field's top candidate reaches a labelled cryptic site more "
                "often than P2Rank's at a matched number of candidates. The "
                "pocket stage was built for this reading and the result is "
                "exploratory."),
            "top1_unresolved": (
                "Given a pocket stage built from its own scores, the counting "
                "field offers a ranked candidate list whose top entry reaches a "
                "labelled cryptic site about as often as P2Rank's. The two are "
                "not separable at the site level either, which is the same "
                "conclusion the per-residue endpoint reached."),
            "top1_favours_p2rank": (
                "At the site level P2Rank's top candidate is the better one. "
                "What this read establishes for the counting field is that a "
                "usable ranked site list can be derived from it at all, not "
                "that the list is competitive."),
            "the_field_offers_too_few_candidates": (
                "The clustering yields fewer candidates per chain than the "
                "top-K comparison requires, so the hit rates at larger K are "
                "reported against the candidates that exist and the shortfall "
                "is stated rather than being absorbed into the rate."),
        },

        "reads_test_fold": False,
        "note": "this file is the plan. Reading the fold under it is "
                "tools/pocket_read.py, which refuses to run until this is "
                "committed.",
    }


def _report(d: dict) -> None:
    pc = d["pocket_construction"]
    print(f"plan for read {d['for_test_fold_read_index']}, declared "
          f"{d['status_declared_in_advance']}")
    print(f"  budget: top {pc['step_1_residue_budget']['q']:.2f} per chain "
          f"(the shipped operating point)")
    print(f"  linkage: single, {pc['step_2_clustering']['cutoff_angstrom']} A "
          f"(the descriptors' pinch radius), sensitivity at "
          f"{pc['step_2_clustering']['sensitivity_cutoff_angstrom']} A")
    print(f"  hit radii {d['hit_radii_angstrom']} A, primary "
          f"{d['primary_hit_radius_angstrom']} A; top K {d['top_k']}")
    p2 = d["the_matched_budget"]["p2rank_candidates_as_committed"]
    print(f"  P2Rank offers median {p2['median']} pockets per chain "
          f"(max {p2['max']}, {p2['n_units_with_no_pocket']} chains with none)")
    print(f"  Bonferroni over {d['multiplicity']['n_primary_tests']} hit rates "
          f"at {d['multiplicity']['corrected_level']}")


def _check() -> int:
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    have = json.loads(OUT.read_text())
    if have.get("schema") != SCHEMA:
        print(f"FAILED: schema {have.get('schema')}")
        return 1
    if have.get("reads_test_fold"):
        print("FAILED: the plan claims to have read the held-out fold")
        return 1
    if have.get("status_declared_in_advance") != "exploratory":
        print("FAILED: the plan no longer declares itself exploratory")
        return 1
    want = build()
    rel = str(Path(__file__).relative_to(ROOT))
    plan_commit = _git("log", "-1", "--format=%H", "--",
                       str(OUT.relative_to(ROOT)))
    if plan_commit:
        blob = subprocess.run(["git", "show", f"{plan_commit}:{rel}"],
                              cwd=ROOT, capture_output=True)
        if blob.returncode == 0:
            at = hashlib.sha256(blob.stdout).hexdigest()
            if have.get("code_sha256") != at:
                print(f"FAILED: the plan records code_sha256 "
                      f"{have.get('code_sha256', '')[:12]} but {rel} at the "
                      f"commit that froze it hashes to {at[:12]}")
                return 1
    volatile = {"generated_at", "head_when_written", "code_sha256"}
    moved = [k for k in want if k not in volatile and want[k] != have.get(k)]
    if moved:
        print("FAILED: the committed plan no longer describes its inputs; "
              f"these fields would change: {moved}")
        for k in moved:
            print(f"  - {k}\n      committed: {json.dumps(have.get(k))[:200]}"
                  f"\n      would be:  {json.dumps(want[k])[:200]}")
        return 1
    said = set(have["what_will_be_written_under_each_outcome"])
    need = {"top1_favours_the_field", "top1_unresolved", "top1_favours_p2rank",
            "the_field_offers_too_few_candidates"}
    if said != need:
        print(f"FAILED: the outcome sentences are {sorted(said)}, expected "
              f"{sorted(need)}")
        return 1
    _report(have)
    print(f"OK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        return _check()
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
