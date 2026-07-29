#!/usr/bin/env python3
"""The plan for the eleventh read: the counting field against PocketMiner.

CryptoBench's own paper names PocketMiner as the representative method for
cryptic-site detection. P2Rank, the only external baseline this repository had
until now, is a general pocket finder that was never fitted to sites that are
absent from the apo structure. A cryptic-pocket paper whose sole comparison is
P2Rank is answering an easier question than the one on its title page, and this
read is the answer to that objection or it is nothing.

The comparison is also the one most likely to go against us, and the plan is
written on that assumption. Everything that would be tempting to decide after
seeing the numbers is decided here: the functional, the operating points, the
correction, the sensitivity arms, and the sentence to be written under each
outcome including the ones we would rather not write.

Three things about this baseline need saying in advance, because each could be
used after the fact to explain away a result.

  The reproduction is not in question, and the read may not blame it. The network
  restored from the published weights reproduces the paper's own test-set numbers
  exactly: 563 cryptic and 1283 non-cryptic residues, ROC-AUC rounding to the
  published 0.87. That evidence is pinned here by hash before the fold is read,
  so a poor showing on our fold cannot afterwards be attributed to a broken
  rebuild, and a good showing for us cannot be credited to one.

  It was trained for a different label than the one it is scored against here.
  PocketMiner predicts whether a residue participates in a pocket that opens
  during molecular dynamics. CryptoBench labels residues that contact a ligand in
  a holo structure while lacking a pocket in the apo one. These overlap but are
  not the same quantity, and the gap is not a defect of either method. So if
  PocketMiner scores below its own published level here, the plan commits in
  advance to reporting that as a transfer between label definitions and not as
  evidence that a table of integers outperforms the state of the art.

  Six of our 190 test PDB entries are in its own data -- 1rtc among the systems
  it was trained on, 3rwv and 5uxa among those it was selected on, 1kx9, 3nx1 and
  3ugk among those it was published on. Every comparison is therefore also run
  with those entries removed, and the removal can only help us, which is why it
  is committed here rather than offered as a robustness check afterwards.

The status is exploratory. The fold carries ten indexed reads before this plan
was written and nothing after them is confirmatory in either direction.

Usage: PYTHONPATH=src:tools python3.12 tools/preregister_pocketminer.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from pocket_bench.paths import ROOT

SCORES = ROOT / "results/baselines/POCKETMINER_SCORES.json"
SELFTEST = ROOT / "results/baselines/POCKETMINER_SELFTEST.json"
TRAIN_OP = ROOT / "results/architecture_sweep/POCKETMINER_TRAIN_OPERATING_POINT.json"
ENDPOINT = ROOT / "results/official_fold/ENDPOINT_STATUS.json"
LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
OUT = ROOT / "results/architecture_sweep/PREREGISTERED_POCKETMINER.json"

SCHEMA = "geoaudit.preregistered_pocketminer.v1"
READ_INDEX = 11
N_BOOT = 10000
SEED = 20260729
# The per-chain budget the counting field deploys at, fixed on the training fold
# long before this read and not revisited here.
TOP_Q = 0.09
# PocketMiner publishes no operating point, so it is given two, both chosen on the
# training fold, which it has never seen. Those values, and the list of entries
# that appear in its own data, are read from the artifacts that determined them
# rather than repeated here, so the plan and the evidence cannot drift apart.

PAIRED_TESTS = (
    "mean per-unit ROC-AUC, counting field minus PocketMiner",
    "mean per-unit ROC-AUC, P2Rank minus PocketMiner",
    "positive-class F1 at a common top-9% budget, counting field minus PocketMiner",
    "MCC at a common top-9% budget, counting field minus PocketMiner",
    "positive-class F1, each at its own trained rule, counting field minus "
    "PocketMiner",
    "MCC, each at its own trained rule, counting field minus PocketMiner",
)


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def build() -> dict:
    for p in (SCORES, SELFTEST, TRAIN_OP):
        if not p.exists():
            raise SystemExit(
                f"missing {p.relative_to(ROOT)}; the baseline and its threshold "
                "have to exist before the comparison can be planned, because the "
                "plan pins both by hash")
    sc = json.loads(SCORES.read_text())
    st = json.loads(SELFTEST.read_text())
    op = json.loads(TRAIN_OP.read_text())
    n_prior = json.loads(LEDGER.read_text())["n_indexed_reads"]
    ov = sc["overlap_with_pocketminers_own_data"]
    listed = tuple(sorted(set(ov["trained_on"]) | set(ov["selected_on"])
                          | set(ov["published_on"])))

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "for_test_fold_read_index": READ_INDEX,
        "written_before_the_read": True,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "head_when_written": _git("rev-parse", "HEAD"),

        "question": (
            "does the counting field predict CryptoBench's cryptic binding "
            "residues as well as PocketMiner, the cryptic-specific network "
            "CryptoBench's own paper names as representative of the task, on the "
            "same 192 single-chain units and the same residue universe"),

        "why_this_comparison_is_necessary": (
            "P2Rank is a general pocket finder and was never fitted to cryptic "
            "sites. Beating it does not establish that a method is competitive at "
            "cryptic-site detection, and a reviewer asking why a cryptic-pocket "
            "paper compares only against a non-cryptic baseline is asking the "
            "right question"),

        "status_declared_in_advance": "exploratory",
        "why_exploratory_and_not_confirmatory": (
            f"the fold carries {n_prior} indexed reads before this plan was "
            "written. Nothing read after that is confirmatory, whichever "
            "direction it points, and fixing the status here stops a favourable "
            "outcome being promoted afterwards"),

        "the_baseline": {
            "what": "PocketMiner: a geometric-vector-perceptron graph network "
                    "over backbone geometry, predicting per-residue probability "
                    "of participating in a pocket that opens in simulation",
            "whose": "Meller et al., Nat. Commun. 14, 1177 (2023); the weights "
                     "are theirs, at the commit pinned below",
            "commit": sc["provenance"]["commit"],
            "weight_sha256": sc["provenance"]["weight_sha256"],
            "scores_artifact": str(SCORES.relative_to(ROOT)),
            "scores_artifact_sha256": hashlib.sha256(SCORES.read_bytes()).hexdigest(),
            "n_units_scored": sc["n_units"],
            "restore_was_asserted": sc["restore"][
                "assert_existing_objects_matched"],
            "every_live_tensor_byte_identical_to_the_checkpoint": sc["restore"][
                "every_live_tensor_is_byte_identical_to_the_checkpoint"],
            "reads_no_label": True,
        },

        "how_faithful_the_reproduction_is": {
            "selftest_artifact": str(SELFTEST.relative_to(ROOT)),
            "selftest_artifact_sha256":
                hashlib.sha256(SELFTEST.read_bytes()).hexdigest(),
            "published_roc_auc": st["reproduction"]["published"]["roc_auc"],
            "our_roc_auc_on_their_test_set":
                st["reproduction"]["ours"]["roc_auc"],
            "published_residue_counts": [
                st["reproduction"]["published"]["n_positive"],
                st["reproduction"]["published"]["n_negative"]],
            "our_residue_counts": [st["reproduction"]["ours"]["n_positive"],
                                   st["reproduction"]["ours"]["n_negative"]],
            "residue_counts_match_exactly":
                st["reproduction"]["residue_counts_match_exactly"],
            "why_this_is_pinned_before_the_read": (
                "so that a poor result on our fold cannot afterwards be "
                "attributed to a broken rebuild, and a good one for us cannot be "
                "quietly credited to one"),
        },

        "the_label_it_was_not_trained_for": {
            "what_pocketminer_predicts": "whether a residue participates in a "
                                         "pocket that opens during molecular "
                                         "dynamics started from this structure",
            "what_it_is_scored_against_here": "whether a residue contacts a "
                                              "ligand in a holo structure while "
                                              "the apo structure lacks the pocket",
            "these_are_not_the_same_quantity": True,
            "committed_in_advance": (
                "if PocketMiner scores below its own published level on this "
                "fold, that is reported as a transfer between two label "
                "definitions and not as evidence that this method outperforms "
                "the state of the art at the task PocketMiner was built for"),
        },

        "contamination": {
            "entries_in_pocketminers_own_data": list(listed),
            "trained_on": ov["trained_on"],
            "selected_on": ov["selected_on"],
            "published_on": ov["published_on"],
            "what_is_done": "every comparison is also run with these entries "
                            "removed",
            "direction_of_the_bias": "their presence can only favour PocketMiner, "
                                     "so removing them can only favour us, which "
                                     "is why the arm is committed here",
            "caveat": "exact PDB entry match is a floor. PocketMiner's training "
                      "systems were never clustered against CryptoBench's folds, "
                      "so homologues below entry-level identity are not excluded "
                      "and this repository does not claim they are",
        },

        "numeric_parameters": {
            "top_q": TOP_Q,
            "pocketminer_trained_budget_f1": op["selected"]["budget/pooled_f1"]["q"],
            "pocketminer_trained_budget_mcc":
                op["selected"]["budget/pooled_mcc"]["q"],
            "pocketminer_trained_cut_f1":
                op["selected"]["cut/pooled_f1"]["threshold"],
            "pocketminer_trained_cut_mcc":
                op["selected"]["cut/pooled_mcc"]["threshold"],
            "train_operating_point_artifact": str(TRAIN_OP.relative_to(ROOT)),
            "train_operating_point_sha256":
                hashlib.sha256(TRAIN_OP.read_bytes()).hexdigest(),
        },

        "statistic": {
            "primary_functional": (
                "the unweighted mean over units of the paired difference in "
                "per-residue ROC-AUC"),
            "why_the_mean_and_not_a_robust_summary": (
                "it is the endpoint already declared primary in "
                f"{ENDPOINT.relative_to(ROOT)}. Choosing a different summary for "
                "a new baseline, after the mean failed to resolve against the "
                "old one, is the manoeuvre this repository has already had to "
                "retract once"),
            "why_roc_auc_is_the_fair_primary_here": (
                "PocketMiner publishes a probability and no operating point, so "
                "a thresholded headline would be a comparison of thresholds we "
                "chose for it. ROC-AUC asks only about the ranking, which is the "
                "thing the network was fitted to produce"),
            "n_boot": N_BOOT,
            "seed": SEED,
            "ci": 0.95,
            "paired": "the same resampled chains enter both arms",
            "also_reported": [
                "the median paired difference and the per-chain win, loss and "
                "tie counts",
                "each method's own mean per-unit ROC-AUC, so the difference can "
                "be read against its level",
                "P2Rank against PocketMiner, which places the two baselines "
                "relative to each other",
                "the same comparison with the six contaminated entries removed",
            ],
        },

        "operating_points": {
            "common_budget": {
                "rule": f"the top {TOP_Q:.0%} of residues per chain for both "
                        f"methods, ranked by their own score",
                "why": "a difference in a thresholded metric between methods "
                       "binarised by different conventions measures the "
                       "conventions",
            },
            "each_methods_own_trained_rule": {
                "counting_field": f"top {TOP_Q:.0%} per chain, fixed on the "
                                  f"training fold",
                "pocketminer": "the budget and the probability cut that maximise "
                               "pooled F1 and pooled MCC on the training fold, "
                               "both taken from the artifact pinned above",
                "why": "PocketMiner has no native binarisation, so the only rule "
                       "it can be held to is one chosen off this fold",
            },
            "what_is_not_done": (
                "no threshold is tuned for either method on the held-out fold"),
        },

        "multiplicity": {
            "n_paired_tests": len(PAIRED_TESTS),
            "tests": list(PAIRED_TESTS),
            "correction": "Bonferroni",
            "corrected_level": round(0.05 / len(PAIRED_TESTS), 6),
            "also_reported": "the uncorrected interval alongside it",
            "the_contamination_arm_is_not_corrected_separately": (
                "it is a sensitivity on the same six tests, not six more "
                "questions, and it is reported as a shift in the same intervals"),
        },

        "guards_the_read_must_pass": [
            "this plan's commit is an ancestor of HEAD, checked with git",
            "the scores artifact still hashes to what is recorded here, so the "
            "baseline cannot have been rerun between the plan and the read",
            "the self-test artifact still hashes to what is recorded here, so "
            "the reproduction evidence cannot have been regenerated after the "
            "fold was read",
            "the training operating point still hashes to what is recorded here",
            "the baseline is scored on the same universe, unit by unit, as the "
            "frozen per-structure telemetry, and the read asserts the residue "
            "counts match rather than assuming they do",
            "the counting field's own mean per-unit ROC-AUC recomputed here "
            "reproduces the frozen telemetry to six decimals",
            "every method's per-unit ROC-AUC is computed by one function, not "
            "one per method",
        ],

        "decision_rules": {
            "if_pocketminer_is_ahead_and_the_interval_excludes_zero": (
                "it is the headline of this read and it goes in the abstract. The "
                "paper's accuracy claim is then limited to P2Rank, the phrase "
                "comparable average predictive performance is narrowed to name "
                "P2Rank only, and the contribution rests on mechanical "
                "auditability and model size"),
            "if_the_field_is_ahead_and_the_interval_excludes_zero": (
                "it is reported as the strongest accuracy evidence in the paper, "
                "still labelled exploratory, and immediately qualified by the "
                "label-definition gap recorded above. It does not license the "
                "word state-of-the-art and it does not become a claim that the "
                "field beats PocketMiner at the task PocketMiner was built for"),
            "if_the_interval_crosses_zero": (
                "it is reported as comparable average performance against the "
                "task's own specialised network, with the interval given"),
            "if_pocketminer_lands_near_chance_on_this_fold": (
                "the self-test pinned above rules out a broken rebuild, so this "
                "is reported as the two label definitions disagreeing more than "
                "expected, with the self-test number quoted beside it. No "
                "sentence claims that PocketMiner does not work"),
            "if_removing_the_contaminated_entries_changes_a_verdict": (
                "the version with them removed is the one reported as primary, "
                "because their presence favours the baseline and a comparison "
                "should not be won on the baseline's own training data"),
            "if_the_two_baselines_disagree_about_each_other": (
                "if P2Rank beats PocketMiner on this fold, that is reported "
                "plainly; it bears on how strong the field's earlier comparison "
                "was and suppressing it would be selecting"),
        },

        "what_will_be_written_under_each_outcome": {
            "the_interval_crosses_zero": (
                "On the 192 single-chain units of the official test fold, the "
                "counting field and PocketMiner are separated by a mean paired "
                "per-unit ROC-AUC difference whose 95% interval includes zero. A "
                "table of integers is therefore not distinguishable on average "
                "from the graph network the cryptic-pocket literature treats as "
                "representative, on this fold, at this sample size. The "
                "comparison is exploratory: the fold had been read ten times "
                "before it was planned."),
            "pocketminer_is_ahead": (
                "PocketMiner is ahead of the counting field by a mean paired "
                "per-unit ROC-AUC difference whose interval excludes zero. The "
                "paper's claim of comparable average performance is therefore "
                "limited to P2Rank, and the contribution is auditability and "
                "model size rather than predictive accuracy. This is stated in "
                "the abstract."),
            "the_field_is_ahead": (
                "The counting field is ahead of PocketMiner by a mean paired "
                "per-unit ROC-AUC difference whose interval excludes zero, on "
                "CryptoBench's definition of a cryptic binding residue. It "
                "remains exploratory, and it is not a claim about the label "
                "PocketMiner was trained for, which is pocket opening in "
                "simulation rather than ligand contact in a holo structure."),
            "pocketminer_lands_near_chance": (
                "PocketMiner, restored from published weights and reproducing "
                "its own published test-set ROC-AUC of 0.87, scores near chance "
                "against CryptoBench's cryptic-residue labels on this fold. We "
                "report this as a disagreement between two definitions of a "
                "cryptic site rather than as a failure of the method."),
        },

        "reads_test_fold": False,
        "note": ("this file is the plan. Reading the fold under it is "
                 "tools/pocketminer_read.py, which refuses to run until this is "
                 "committed. The baseline's scores are not a read: they contain "
                 "no label and no metric"),
    }


def _report(d: dict) -> None:
    print(f"plan for read {d['for_test_fold_read_index']}, declared "
          f"{d['status_declared_in_advance']}")
    b = d["the_baseline"]
    print(f"  baseline {b['n_units_scored']} units at commit {b['commit'][:12]}, "
          f"scores {b['scores_artifact_sha256'][:12]}")
    f = d["how_faithful_the_reproduction_is"]
    print(f"  reproduces their own test set: ROC-AUC "
          f"{f['our_roc_auc_on_their_test_set']} against published "
          f"{f['published_roc_auc']}, residue counts "
          f"{f['our_residue_counts']} against {f['published_residue_counts']}")
    n = d["numeric_parameters"]
    print(f"  trained rules: budget q={n['pocketminer_trained_budget_f1']}/"
          f"{n['pocketminer_trained_budget_mcc']}, cut "
          f"{n['pocketminer_trained_cut_f1']}/{n['pocketminer_trained_cut_mcc']}")
    c = d["contamination"]
    print(f"  {len(c['entries_in_pocketminers_own_data'])} entries in its own "
          f"data, removed in a second arm: "
          f"{', '.join(c['entries_in_pocketminers_own_data'])}")
    m = d["multiplicity"]
    print(f"  {m['n_paired_tests']} paired tests, Bonferroni level "
          f"{m['corrected_level']}")
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
        print("FAILED: the plan declares itself "
              f"{have.get('status_declared_in_advance')}; ten indexed reads "
              "precede it and nothing after them is confirmatory")
        return 1
    if not have["how_faithful_the_reproduction_is"]["residue_counts_match_exactly"]:
        print("FAILED: the plan does not carry a reproduction that matches the "
              "published residue counts, so a bad result could be blamed on it")
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
    need = {"the_interval_crosses_zero", "pocketminer_is_ahead",
            "the_field_is_ahead", "pocketminer_lands_near_chance"}
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
