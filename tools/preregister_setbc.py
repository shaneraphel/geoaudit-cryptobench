#!/usr/bin/env python3
"""Preregister the read of Set B and Set C, before any method touches them.

What is being confirmed, and why it needs a set nobody has read
---------------------------------------------------------------
Four wire families were built on one screen --- read bytes the pipeline throws
away --- and stacked into ``geometry_field``: the 645 deployed wires plus 624
columns of backbone, side-chain, void-topology and temperature-factor quantities.
On twelve cluster-disjoint halvings of the training fold that stack is worth
**+0.0121 over the deployed detector, positive on 12 of 12**, against a control
arm spending the same table budget on already-deployed wires at −0.0017.

Every number in that sentence is a training-fold number. A reviewer put that
first, and they are right: the official fold has been read thirteen times and Set
A once, so neither can confirm anything about a detector finalised afterwards.
Set B and Set C can, because they were built, frozen and hashed **before any of
these families existed** and have never been read.

The order this file sits in
---------------------------
``AGENTS.md`` fixes it: build, freeze, hash, preregister, read once. Sets B and C
were built and frozen on 31 July and still hash to what was published then. The
detector was finalised and compiled before this plan was written, and its digest
is pinned below so the read can refuse a field that moved afterwards. Nothing has
been scored.

What spends what
----------------
One read spends both sets. Scoring a method improved *after* this plan is pinned
would not give a second confirmation, it would destroy this one --- which is
exactly what already happened to Set A and is why Set A cannot be used here.

What is deliberately conceded in advance
-----------------------------------------
Three things, written now so they cannot be discovered later and presented as
context:

* **45 units is small** and the interval will be wide. Set A had 57 and its
  interval on the P2Rank comparison was [+0.016, +0.072]; a narrower effect on
  fewer units may not resolve, and failing to resolve is evidence about power and
  not evidence of parity.
* **Both sets are cryo-EM and resolution is a covariate of the label.**
  ``CRYOEM_LABEL_SENSITIVITY.json`` records the recovered rule declining 9.2 % of
  pairs for X-ray Set A, 17.3 % for Set B and 31.9 % for Set C, with the cryptic
  share rising alongside. Every comparison here inherits that.
* **One unit was dropped before the plan and the reason is a file format.**
  ``9t97_A3a`` labels its chains with three characters, which the legacy PDB
  format cannot hold in its single column, so no atom of that chain survives
  rendering. Set B is therefore 7 of 8. The count is pinned here so coverage
  cannot move after scores are seen.

Nothing in this file reads a label against a prediction. It writes a plan.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT_DIR / "src"), str(ROOT_DIR / "tools")]

from pocket_bench.paths import ROOT                                # noqa: E402

SCHEMA = "geoaudit.preregistered_setbc.v1"
OUT = ROOT / "results/external/PREREGISTERED_SETBC.json"

SETB = ROOT / "results/external/SETB_SET.json"
SETC = ROOT / "results/external/SETC_SET.json"
MANIFEST = ROOT / "data/external/setbc_manifest.json"
FIELD = ROOT / "data/cryptobench_apo/GEOMETRY_FIELD.json"
OLD_FIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"

# Published at freeze time in docs/AGENT_MEMORY.md section 5.
FROZEN_PREFIX = {"SETB_SET.json": "09381b40", "SETC_SET.json": "ff112a60"}

# The training-fold lift this read exists to confirm.
TRAIN_LIFT = 0.012122
TRAIN_LIFT_CI = [0.010332, 0.013913]

# Set A's levels, which are the only external anchors available. They are Set A's
# and therefore about table_field, not about the field being read here.
SET_A_LEVELS = {"table_field": 0.84114, "p2rank": 0.79683,
                "plmnn": 0.875126, "pocketminer": 0.773408}


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *a],
                          capture_output=True, text=True).stdout.strip()


def build(write: bool) -> int:
    for path in (SETB, SETC, MANIFEST, FIELD, OLD_FIELD):
        if not path.is_file():
            raise SystemExit(f"{path.relative_to(ROOT)} is absent; the plan "
                             f"cannot pin what does not exist")
    for path in (SETB, SETC):
        want = FROZEN_PREFIX[path.name]
        got = _sha(path)
        if not got.startswith(want):
            raise SystemExit(
                f"{path.name} hashes to {got[:16]} and was frozen at {want}... "
                f"A plan pinning a set that has moved since its freeze pins "
                f"nothing")

    man = json.loads(MANIFEST.read_text())
    if man.get("no_method_has_been_run") is not True:
        raise SystemExit("the manifest no longer declares that no method has run")
    units = man["entries"]
    by_set: dict[str, int] = {}
    for e in units:
        by_set[e["set"]] = by_set.get(e["set"], 0) + 1

    doc = {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "reads_any_external_unit": False,
        "status": "plan",
        "what_this_confirms_or_refutes": (
            "whether the +0.0121 the four-family stack is worth on twelve "
            "cluster-disjoint halvings of the training fold survives on units "
            "no part of its development could have seen"),
        "why_set_a_cannot_be_used": (
            "Set A has been read. Scoring a detector finalised afterwards on it "
            "destroys that read's confirmatory status rather than adding a "
            "second one, and the +0.0443 it recorded would become a number about "
            "a detector that no longer exists"),
        "sets": {
            "set_b": {"artifact": SETB.relative_to(ROOT).as_posix(),
                      "sha256": _sha(SETB),
                      "frozen_at_prefix": FROZEN_PREFIX["SETB_SET.json"],
                      "n_units_in_the_set": 8,
                      "n_units_scorable": by_set.get("set_b", 0),
                      "resolution_ceiling": 2.5},
            "set_c": {"artifact": SETC.relative_to(ROOT).as_posix(),
                      "sha256": _sha(SETC),
                      "frozen_at_prefix": FROZEN_PREFIX["SETC_SET.json"],
                      "n_units_in_the_set": 38,
                      "n_units_scorable": by_set.get("set_c", 0),
                      "resolution_ceiling": 3.0},
        },
        "manifest": {"path": MANIFEST.relative_to(ROOT).as_posix(),
                     "sha256": _sha(MANIFEST), "n_units": len(units)},
        "coverage_pinned_before_the_read": {
            "n_units": len(units),
            "n_dropped": 1,
            "dropped": ["9t97_A3a"],
            "why": ("the deposition labels its chains with three characters and "
                    "the legacy PDB format holds one, so no atom of that chain "
                    "survives rendering. A format limitation, established before "
                    "any method ran, and nothing about the protein"),
            "why_pinned": ("a unit removed after scores are seen is a selection. "
                           "This count cannot move afterwards"),
        },
        "methods": {
            "geometry_field": {
                "artifact": FIELD.relative_to(ROOT).as_posix(),
                "sha256": _sha(FIELD),
                "n_columns": 1269,
                "what_it_is": ("the deployed 645 wires plus 624 columns from "
                               "four families: backbone 132, side-chain 261, "
                               "void 135, temperature-factor 96"),
                "finalised_before_this_plan": True,
                "why_the_digest_is_pinned": (
                    "so the read refuses a field recompiled after the plan. A "
                    "detector changed between plan and read is a different "
                    "experiment wearing the plan's licence"),
            },
            "table_field": {
                "artifact": OLD_FIELD.relative_to(ROOT).as_posix(),
                "sha256": _sha(OLD_FIELD),
                "n_columns": 645,
                "what_it_is": "the deployed detector, unchanged",
            },
            "baselines": ["p2rank", "pocketminer", "plmnn"],
            "why_the_baselines_are_run_here_too": (
                "they have never been run on these units either, so a read that "
                "compared only our two detectors would leave the reader without "
                "the levels the comparison sits in"),
        },
        "statistic": {
            "primary": "mean paired per-unit ROC-AUC over the scorable units",
            "why": ("the same functional declared primary for the official fold "
                    "and for Set A, so this comparison is on one scale with "
                    "those. It is a ranking statistic, so no calling convention "
                    "enters it"),
            "n_boot": 10000,
            "seed": 20260801,
            "ci_level": 0.95,
            "residue_universe": (
                "the residues every method scored on that unit. A paired "
                "difference needs one universe, and a method that drops residues "
                "it cannot featurise would otherwise be scored on an easier one"),
        },
        "co_primary_comparisons": [
            {"key": "geometry_field_minus_table_field",
             "question": ("does the training-fold lift transfer to units nobody "
                          "has read"),
             "predicted": TRAIN_LIFT,
             "predicted_from": ("GEOMETRY_624_LIFT.json, twelve cluster-disjoint "
                                "halvings of the training fold"),
             "predicted_ci": TRAIN_LIFT_CI,
             "transfers_if": "the 95% interval excludes zero",
             "replicates_if": (f"the interval contains {TRAIN_LIFT:.4f}, the "
                               f"training-fold point estimate")},
            {"key": "geometry_field_minus_plmnn",
             "question": "does the stack close the deficit to the supervised model",
             "predicted": round(SET_A_LEVELS["table_field"] + TRAIN_LIFT
                                - SET_A_LEVELS["plmnn"], 6),
             "predicted_from": ("Set A's levels plus the training-fold lift. This "
                                "is an anchor and not a forecast: Set A is X-ray "
                                "and these sets are cryo-EM"),
             "we_lead_if": "the interval excludes zero and is positive"},
            {"key": "geometry_field_minus_p2rank",
             "question": "does the advantage over P2Rank hold on a second set",
             "predicted": round(SET_A_LEVELS["table_field"] + TRAIN_LIFT
                                - SET_A_LEVELS["p2rank"], 6),
             "predicted_from": "Set A's levels plus the training-fold lift"},
            {"key": "geometry_field_minus_pocketminer",
             "question": "the same against the cryptic-specific baseline",
             "predicted": round(SET_A_LEVELS["table_field"] + TRAIN_LIFT
                                - SET_A_LEVELS["pocketminer"], 6),
             "predicted_from": "Set A's levels plus the training-fold lift"},
        ],
        "multiplicity": {
            "n_comparisons": 4,
            "correction": "Bonferroni over the four co-primary comparisons",
            "reported": ("both the uncorrected and the corrected interval for "
                         "every comparison"),
        },
        "secondary_pinned_now": [
            {"key": "per_set_split",
             "what": ("the same primary comparison on Set B's 7 units and Set "
                      "C's 38 separately"),
             "why": ("the two sets have different resolution ceilings and "
                     "CRYOEM_LABEL_SENSITIVITY.json shows the label's decline "
                     "rate rising with resolution, so pooling mixes two label "
                     "regimes. Pooled is primary because it was fixed as primary; "
                     "the split is reported so a reader can see whether the "
                     "pooled number is carried by one regime")},
            {"key": "stratify_by_pocket_size",
             "what": ("the primary comparison on units with fewer than ten "
                      "cryptic residues and on the rest"),
             "why": ("FAILURE_TAIL.json measures a 0.28 spread on that axis, the "
                     "largest effect in this repository, so it is where a "
                     "difference would hide")},
        ],
        "sentences_fixed_before_the_read": {
            "transfers_and_replicates": (
                "The four-family stack's training-fold lift replicates on units "
                "no part of its development could have seen: +X [a, b] on N "
                "cryo-EM units frozen before the families existed, against a "
                "training-fold estimate of +0.0121. This is the first held-out "
                "confirmation of a wire-family result in this repository."),
            "transfers_but_smaller": (
                "The stack's advantage over the deployed detector survives "
                "externally at +X [a, b], below the +0.0121 the training fold "
                "estimated. A lift that shrinks on held-out data is the ordinary "
                "outcome and the training-fold figure should be read as an upper "
                "bound from here on."),
            "does_not_resolve": (
                "On N units the stack's advantage over the deployed detector is "
                "+X [a, b], crossing zero. The training-fold lift is not "
                "confirmed. With 45 units and an effect of this size the read "
                "had limited power, which is a statement about the read and not "
                "evidence that the lift is absent -- and it is also not evidence "
                "that it is present. Nothing about the four families' status on "
                "held-out data is established by this repository."),
            "negative": (
                "On N units the stack scores below the deployed detector by X "
                "[a, b]. The training-fold lift does not transfer. The four "
                "families are withdrawn as a deployment recommendation; what "
                "survives is the screen that predicted their signs on the "
                "training fold, and the record that it did not predict "
                "generalisation."),
            "plmnn_still_ahead": (
                "pLM-NN remains the more accurate method. The stack narrows the "
                "gap to X [a, b] from the -0.0340 the deployed detector recorded "
                "on Set A, and does not close it."),
            "plmnn_caught_or_passed": (
                "The stack reaches or passes pLM-NN on these units at X [a, b]. "
                "This is one read of 45 cryo-EM units against a supervised model "
                "that leads on both the official fold and Set A, so it is "
                "reported as a result on this set and not as a ranking of the two "
                "methods, and the resolution covariate of the cryo-EM label is "
                "reported with it."),
        },
        "why_the_losing_sentences_are_here": (
            "so that the reading is fixed by the plan rather than chosen by the "
            "outcome. Four of the six above are outcomes in which the thing this "
            "repository has spent a week building does not work"),
        "refusals_the_read_must_implement": [
            "refuse if either set's digest differs from the pinned one",
            "refuse if the manifest's digest differs",
            "refuse if GEOMETRY_FIELD.json's digest differs from the pinned one",
            "refuse if this plan is not committed, or is dirty in the working "
            "tree, or is not an ancestor of the commit that will carry the read",
            "refuse to overwrite an existing read without a recorded reason",
        ],
        "committed_at_head": _git("rev-parse", "HEAD"),
        "what_this_plan_does_not_do": (
            "it does not score anything. No prediction is computed and no label "
            "is compared against one until the read runs under this plan"),
    }

    print(f"plan for {len(units)} units "
          f"(set_b {by_set.get('set_b')}, set_c {by_set.get('set_c')})")
    print(f"  set B  {_sha(SETB)[:16]}  frozen at "
          f"{FROZEN_PREFIX['SETB_SET.json']}...")
    print(f"  set C  {_sha(SETC)[:16]}  frozen at "
          f"{FROZEN_PREFIX['SETC_SET.json']}...")
    print(f"  field  {_sha(FIELD)[:16]}  1269 columns")
    print(f"  confirming a training-fold lift of {TRAIN_LIFT:+.4f} "
          f"{TRAIN_LIFT_CI}")
    print(f"  {len(doc['co_primary_comparisons'])} co-primary comparisons, "
          f"Bonferroni over all four")
    print(f"  {len(doc['sentences_fixed_before_the_read'])} outcome sentences "
          f"fixed, four of them losing")

    if write:
        OUT.write_text(json.dumps(doc, indent=2, allow_nan=False) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}")
        print("commit this before running the read")
    else:
        print("\n(not written; pass --write)")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    return build(ap.parse_args(argv).write)


if __name__ == "__main__":
    raise SystemExit(main())
