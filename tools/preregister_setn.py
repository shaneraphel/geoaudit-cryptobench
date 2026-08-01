#!/usr/bin/env python3
"""Fix the Set N read before any score exists.

The question
------------
Every accuracy number published on CryptoBench, this repository's included, is
computed on chains that have a cryptic pocket. A per-unit ROC-AUC ranks a chain's
residues against each other, so it can only be formed where there is something to
rank towards. It cannot ask whether a method knows that a chain has nothing to find,
and no published comparison on this benchmark asks that.

Set N asks it. 327 X-ray chains, released after CryptoBench's newest, sharing no
UniRef50 cluster with CryptoBench or with Set A, each examined against every
deposited holo partner and every pair decided ``not cryptic``.

The statistic, and why it is not a false-alarm rate
---------------------------------------------------
The obvious statistic --- how many residues does each method call on a chain with no
pocket --- is not comparable across methods, because each one's operating point is
its own choice. P2Rank calls pockets, pLM-NN thresholds a probability at 0.5, and
the counting field has a threshold of its own. A method that calls less is not
thereby better at knowing; it may just be shy. Reporting that number as the primary
would confound calibration with discrimination.

So the primary is **rank-based and threshold-free**. Each unit gets one score per
method, the mean of that method's ten highest residue scores on that unit, and the
statistic is the ROC-AUC over 384 units of separating the 57 that have a cryptic
pocket from the 327 that do not. Being rank-based, it is invariant to any monotone
rescaling of a method's scores, so no method is helped or hurt by the units its
scores happen to be in.

``k = 10`` is fixed here, before any score is read. Set A's positive units carry a
median of 16 cryptic residues, so ten is below the size of a real pocket and a chain
that has one should put ten high scores together; a single maximum would turn one
outlying residue into the whole statistic.

Where the positive half's scores come from
-------------------------------------------
The 57 positive units were scored once, under ``PREREGISTERED_EXTERNAL.json``, and
those per-residue scores are frozen in ``results/external/predictions`` and
``results/baselines``. This read forms a new statistic over numbers already
released; it re-runs no model on them and re-fits no threshold. That is still a read
and is indexed as one, by the ledger's own rule that opening labels to form a new
statistic is not free.

Nothing new is scored on Set A. In particular ``geometry_field``, the 1269-column
detector, is **not** run on the 57, because scoring an improved method on a spent
set destroys the confirmatory result already read from it. It therefore cannot enter
the primary, and appears only in the secondary, which is confined to Set N.

The confound, named before the numbers
---------------------------------------
Set N's chains are not a random sample of proteins: they are chains whose deposited
holo partners bind without moving a pocket. If such chains were systematically
shorter, or more compact, than Set A's, a method that scores compact chains lower
would look discriminating when it is only reading size. The plan therefore fixes a
control: the same statistic computed with each unit's score replaced by its residue
count, reported alongside. If chain length alone separates the two halves at an AUC
far from 0.5, every method's number is read against that and not against 0.5.

Usage: PYTHONPATH=src:tools python3.12 tools/preregister_setn.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETN = ROOT / "results/external/SETN_SET.json"
SETA = ROOT / "results/external/EXTERNAL_SET.json"
FIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"
GEOM = ROOT / "data/cryptobench_apo/GEOMETRY_FIELD.json"
OUT = ROOT / "results/external/PREREGISTERED_SETN.json"
SCHEMA = "geoaudit.preregistered_setn.v1"

TOP_K = 10
N_BOOT = 10000
SEED = 20260801
ALPHA = 0.05
CO_PRIMARY = ("table_field_minus_p2rank",
              "table_field_minus_plmnn",
              "table_field_minus_pocketminer")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> dict:
    setn = json.loads(SETN.read_text())
    seta = json.loads(SETA.read_text())
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "declared": "confirmatory for the co-primary, descriptive for the secondary",
        "written_before_any_score_on_set_n_existed": True,
        "question": ("can a method tell a chain that has a cryptic pocket from a "
                     "chain examined against every deposited holo partner and shown "
                     "to have none"),
        "why_this_question_is_new": (
            "every accuracy number on this benchmark is a within-chain ranking on "
            "chains that have a pocket. None of them can be low because a method "
            "hallucinates a pocket on a chain that has none, because no such chain "
            "is in the evaluation"),
        "the_sets": {
            "positive": {"artifact": str(SETA.relative_to(ROOT)),
                         "sha256": _sha(SETA),
                         "n_units": seta["n_units_with_a_cryptic_pocket"]},
            "negative": {"artifact": str(SETN.relative_to(ROOT)),
                         "sha256": _sha(SETN),
                         "n_units": setn["n_units"],
                         "n_clusters": setn["n_clusters"]},
            "clusters_are_disjoint": (
                "one unit per UniRef50 cluster across the whole funnel, so a "
                "cluster contributes a positive or a negative and never both"),
        },
        "the_methods": {
            "table_field": {"artifact": str(FIELD.relative_to(ROOT)),
                            "sha256": _sha(FIELD),
                            "in_the_primary": True},
            "geometry_field": {"artifact": str(GEOM.relative_to(ROOT)),
                               "sha256": _sha(GEOM),
                               "in_the_primary": False,
                               "why_not": (
                                   "it has never been scored on Set A's positives "
                                   "and must not be: a spent set cannot confirm a "
                                   "second method. It appears in the secondary, "
                                   "which reads only Set N")},
            "p2rank": {"version": "2.5.1", "in_the_primary": True},
            "plmnn": {"source": "CryptoBench's published weights",
                      "in_the_primary": True},
            "pocketminer": {"source": "published weights", "in_the_primary": True},
        },
        "statistic": {
            "unit_score": f"mean of the method's {TOP_K} highest residue scores "
                          f"on that unit",
            "top_k": TOP_K,
            "why_k_is_ten": (
                "Set A's positive units carry a median of 16 cryptic residues, so "
                "ten sits below the size of a real pocket while a single maximum "
                "would let one outlying residue be the whole statistic"),
            "outcome": "ROC-AUC over all units, positive unit against negative unit",
            "why_rank_based": (
                "invariant to any monotone rescaling, so the comparison cannot be "
                "won or lost on the units a method's scores happen to be in"),
            "residues_used": ("the intersection of the residues all methods scored "
                              "on that unit, as the Set A read did, so a paired "
                              "difference is paired"),
        },
        "co_primary": {
            "comparisons": list(CO_PRIMARY),
            "direction": "positive favours the counting field",
            "n_bootstrap": N_BOOT,
            "seed": SEED,
            "resampling_unit": "the unit, stratified by positive or negative",
            "family_alpha": ALPHA,
            "bonferroni_level": round(1 - ALPHA / len(CO_PRIMARY), 6),
            "resolved_means": "the Bonferroni-corrected interval excludes zero",
        },
        "secondary": [
            {"name": "geometry_field_on_set_n_only",
             "what": "the mean top-10 unit score of geometry_field and of "
                     "table_field over Set N alone, paired by unit",
             "reads": "Set N only, so it spends nothing of Set A",
             "level": "descriptive"},
            {"name": "as_deployed_false_alarms",
             "what": "for each method, the fraction of Set N units on which it "
                     "calls at least one residue at its own deployed operating "
                     "point, and the mean fraction of residues called",
             "level": "descriptive, and confounded with each method's own "
                      "calibration, which is why it is not primary"},
        ],
        "control_fixed_in_advance": {
            "name": "chain_length_alone",
            "what": "the same ROC-AUC with each unit's score replaced by its "
                    "residue count",
            "why": ("Set N's chains are chains whose partners bind without moving "
                    "a pocket. If they are systematically shorter than Set A's, a "
                    "method that scores short chains lower separates the halves "
                    "without knowing anything about pockets"),
            "how_it_is_used": ("every method's AUC is reported against this number "
                               "as well as against 0.5, in the same table"),
        },
        "outcome_sentences_written_in_advance": {
            "all_three_resolved_in_our_favour":
                "On 384 external units the counting field separates chains that "
                "have a cryptic pocket from chains shown to have none better than "
                "all three published baselines, at a Bonferroni-corrected level. "
                "This is a property no published comparison on this benchmark "
                "measures, and it is reported with the chain-length control that "
                "was fixed before the read.",
            "some_resolved":
                "The counting field separates chains with a cryptic pocket from "
                "chains shown to have none better than some published baselines and "
                "not others. The comparisons that resolved and the ones that did "
                "not are both listed, at the corrected level.",
            "none_resolved":
                "On 384 external units no comparison of unit-level detection "
                "between the counting field and a published baseline resolved at "
                "the corrected level. The question the set was built to ask has an "
                "answer that this set is too small to give.",
            "resolved_against_us":
                "On 384 external units a published baseline separates chains that "
                "have a cryptic pocket from chains shown to have none better than "
                "the counting field does, at a Bonferroni-corrected level. The "
                "advantage is real and is recorded here because the plan named the "
                "sentence before the number existed.",
            "control_dominates":
                "Chain length alone separates the two halves at an AUC that leaves "
                "no room to read the methods' numbers as knowledge of pockets. The "
                "read is reported and no method comparison is drawn from it.",
        },
        "what_this_cannot_show": [
            "that a Set N chain has no pocket. It has no pocket that a holo "
            "structure deposited before the cutoff reveals",
            "anything about affinity, druggability, or any clinical property",
        ],
        "changes_forbidden_after_this_file_is_committed": [
            "the unit score, k, the statistic, the bootstrap seed or count",
            "the membership of either set",
            "any architecture, threshold or quantisation rule of any method",
            "the addition of a method to the co-primary",
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
        n = d["the_sets"]
        print(f"\nplan for the Set N read, {d['declared']}")
        print(f"  {n['positive']['n_units']} positive units + "
              f"{n['negative']['n_units']} negative units, "
              f"{n['negative']['n_clusters']} clusters")
        print(f"  statistic: {d['statistic']['unit_score']}")
        print(f"  co-primary {len(d['co_primary']['comparisons'])} at Bonferroni "
              f"level {d['co_primary']['bonferroni_level']}")
        print(f"  control fixed in advance: {d['control_fixed_in_advance']['name']}")
        print(f"  outcome sentences written in advance: "
              f"{len(d['outcome_sentences_written_in_advance'])}")
        for path, want in ((SETN, n["negative"]["sha256"]),
                           (SETA, n["positive"]["sha256"])):
            if _sha(path) != want:
                print(f"FAILED {path.name} moved since the plan was written")
                return 1
        print(f"OK {OUT.relative_to(ROOT)}")
        return 0
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
