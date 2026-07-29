#!/usr/bin/env python3
"""The plan for the twelfth read: the whole threshold axis, not three points on it.

Three operating points have been read on this fold already: the rule each method
deploys, a common top-9% budget, and each method's budget tuned on the training
fold. At the matched budget the F1 difference is positive and its interval crosses
zero, so the paper already says the F1 advantage depends on where the two methods
are cut. What it cannot yet say is what happens everywhere else, and a reviewer is
entitled to ask whether the three points chosen happen to be the flattering ones.

So this read reports precision, recall, F1 and MCC for every method at every
calling fraction on a fixed grid, and the paired difference with an interval at
each one. It is the same frozen scores re-binarised, which is why it is indexed:
a new summary of numbers an earlier read froze is a new statement about the fold.

The one thing this read may not do is choose anything. No threshold is selected
from it, no operating point moves because of it, and the deployed q stays where
the training fold put it. That commitment is what keeps a curve over 39 cut points
from becoming 39 chances to find a favourable one, and it is written here rather
than asserted afterwards. Nothing downstream is allowed to read the argmax of a
test-fold curve; the read records that no field of its output is consumed by any
configuration.

Because the whole curve is reported, no single point on it carries a corrected
claim. The multiplicity statement is therefore not a Bonferroni level over 39
tests but a refusal: the curve is a display, the claims stay attached to the three
operating points earlier reads fixed, and the interval drawn at each cut point is
pointwise and labelled as such.

Usage: PYTHONPATH=src:tools python3.12 tools/preregister_threshold_curve.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from pocket_bench.paths import ROOT

PREDS = ROOT / "results/cryptobench_official/predictions"
FULL_READ = ROOT / "results/official_fold/MATCHED_FULL_READ.json"
LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
OUT = ROOT / "results/architecture_sweep/PREREGISTERED_THRESHOLD_CURVE.json"

SCHEMA = "geoaudit.preregistered_threshold_curve.v1"
READ_INDEX = 12
N_BOOT = 2000
SEED = 20260729
TOP_Q = 0.09
Q_LOW, Q_HIGH, Q_STEP = 0.02, 0.40, 0.01
METRICS = ("precision", "recall", "positive_class_f1", "mcc")


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def build() -> dict:
    n_prior = json.loads(LEDGER.read_text())["n_indexed_reads"]
    fr = json.loads(FULL_READ.read_text())
    matched = fr["conventions"]["D2_common_budget"]
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "for_test_fold_read_index": READ_INDEX,
        "written_before_the_read": True,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "head_when_written": _git("rev-parse", "HEAD"),

        "question": (
            "over the whole range of per-chain calling fractions, not at three "
            "chosen points, how do precision, recall, positive-class F1 and MCC "
            "compare between the counting field and each baseline, and where does "
            "the paired difference cross zero"),

        "why_it_is_asked": (
            "the F1 advantage over P2Rank survives a matched budget as a positive "
            "point estimate whose interval crosses zero, so the paper already "
            "concedes that the advantage depends on the operating point. Reporting "
            "three points invites the question of whether they were the flattering "
            "ones. The answer is the curve"),

        "status_declared_in_advance": "exploratory",
        "why_exploratory_and_not_confirmatory": (
            f"the fold carries {n_prior} indexed reads before this plan was "
            "written"),

        "nothing_is_selected_from_this_curve": {
            "committed": True,
            "what_that_forbids": [
                "moving the deployed calling fraction, which the training fold "
                "fixed at 0.09 and which this read does not revisit",
                "reporting the best point of any curve as a headline",
                "letting any configuration, threshold or artifact downstream read "
                "a value off this output",
            ],
            "why": (
                "a curve over 39 cut points is 39 opportunities to find a "
                "favourable one. Refusing to select from it is what keeps it a "
                "display of an already-published comparison rather than a search"),
            "how_it_is_enforced": (
                "the read records that its output is consumed by no configuration, "
                "and the claims in the paper stay attached to the three operating "
                "points earlier reads fixed"),
        },

        "grid": {
            "quantity": "per-chain top-q by each method's own score",
            "low": Q_LOW, "high": Q_HIGH, "step": Q_STEP,
            "n": int(round((Q_HIGH - Q_LOW) / Q_STEP)) + 1,
            "why_this_grid": "it is the grid the training-fold operating points "
                             "were chosen on, so the curve and the choice are "
                             "commensurable",
            "tie_break": "a stable sort of the negated score, identical to the "
                         "rule the counting field deploys",
        },

        "arms": {
            "table_field_vs_p2rank": "the comparison the paper rests on",
            "table_field_vs_pocketminer": "the cryptic-specific baseline, if its "
                                          "scores are present",
            "table_field_vs_plmnn": "the benchmark's own supervised baseline, if "
                                    "its scores are present",
            "note": "an arm is reported only if that baseline's scores exist; a "
                    "missing baseline is recorded as missing and never as zero",
        },

        "metrics": list(METRICS),
        "summary_rule": ("each metric is averaged over the units where it is "
                         "defined, the same convention metric_of and the matched "
                         "reads already use; a chain on which a method calls "
                         "nothing has no precision and is not scored zero"),

        "statistic": {
            "at_each_cut_point": "the mean paired per-unit difference with a "
                                 "pointwise 95% bootstrap interval",
            "n_boot": N_BOOT,
            "why_fewer_draws_than_the_headline_reads": (
                "39 cut points times four metrics times three arms is 468 "
                "intervals, and these are pointwise intervals on a display rather "
                "than the basis of a claim. The headline comparisons keep their "
                "10000 draws"),
            "seed": SEED,
            "paired": "the same resampled chains enter both arms at every cut",
        },

        "multiplicity": {
            "correction": "none, and deliberately",
            "why": ("no point on the curve carries a claim, so there is nothing "
                    "to correct. Correcting 468 pointwise intervals would imply "
                    "that some point of some curve was being asserted, which is "
                    "exactly what this plan forbids"),
            "what_is_reported_instead": (
                "the interval at each cut point, labelled pointwise, plus the "
                "range of cut points over which the difference in each metric "
                "keeps one sign, which is a property of the whole curve rather "
                "than of a chosen point"),
        },

        "reproduction_guard": {
            "must_match": str(FULL_READ.relative_to(ROOT)),
            "at": TOP_Q,
            "positive_class_f1_ours": matched["table_field"]["positive_class_f1"],
            "positive_class_f1_theirs": matched["p2rank"]["positive_class_f1"],
            "why": ("at q = 0.09 this curve is recomputing a number the seventh "
                    "read published. If it does not reproduce it to six decimals "
                    "the curve is measuring something else and the read is void"),
        },

        "what_will_be_written_under_each_outcome": {
            "our_curve_is_above_theirs_throughout": (
                "Across every calling fraction from 2% to 40%, the counting "
                "field's positive-class F1 is above P2Rank's, so the advantage "
                "reported at the deployed operating point is not an artefact of "
                "where the two methods were cut. The pointwise intervals still "
                "include zero over much of the range, so this is a statement "
                "about the sign of the difference and not about its significance."),
            "the_curves_cross": (
                "The two F1 curves cross within the grid, so which method is "
                "ahead depends on the calling fraction. The paper reports the "
                "crossing point and stops claiming an F1 advantage that is not a "
                "property of the ranking."),
            "their_curve_is_above_ours_throughout": (
                "P2Rank's positive-class F1 is above the counting field's at every "
                "calling fraction, which contradicts the matched-budget point "
                "estimate and would mean the earlier reads were driven by the one "
                "cut point they used. The F1 claim is withdrawn."),
        },

        "reads_test_fold": False,
        "note": ("this file is the plan. Reading the fold under it is "
                 "tools/threshold_curve.py, which refuses to run until this is "
                 "committed"),
    }


def _report(d: dict) -> None:
    print(f"plan for read {d['for_test_fold_read_index']}, declared "
          f"{d['status_declared_in_advance']}")
    g = d["grid"]
    print(f"  grid {g['low']} to {g['high']} step {g['step']}, {g['n']} cut "
          f"points, {len(d['metrics'])} metrics")
    print(f"  arms: {', '.join(k for k in d['arms'] if k != 'note')}")
    print(f"  selects nothing: {d['nothing_is_selected_from_this_curve']['committed']}"
          f"; correction {d['multiplicity']['correction']}")
    r = d["reproduction_guard"]
    print(f"  must reproduce read seven at q={r['at']}: "
          f"{r['positive_class_f1_ours']} against {r['positive_class_f1_theirs']}")
    print(f"  outcomes written in advance: "
          f"{len(d['what_will_be_written_under_each_outcome'])}")


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
        print("FAILED: the plan does not declare itself exploratory")
        return 1
    if not have["nothing_is_selected_from_this_curve"]["committed"]:
        print("FAILED: the plan does not forbid selecting from the curve, which "
              "is the only thing that keeps 39 cut points from being 39 tries")
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
        print(f"FAILED: the committed plan no longer describes its inputs: {moved}")
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
