#!/usr/bin/env python3
"""The plan for the tenth read: the counting field against the benchmark's own
cryptic-site model, decided before the comparison is run.

Every comparison in this repository so far has been against P2Rank, which is a
general pocket finder that was never fitted to cryptic sites. That is the weakest
available baseline for this task and the paper cannot rest on it. pLM-NN is the
model CryptoBench's own authors fitted and published for exactly this problem: a
3-billion-parameter protein language model into a small supervised network. It is
the comparison that decides whether a hand-built counting field is competitive at
all.

Which means this read can go badly, and the plan has to be written as if it will.
Three commitments:

  The functional is the one already declared primary. Mean paired per-unit
  ROC-AUC, the same statistic, the same bootstrap, the same seed discipline as
  the read that compared against P2Rank. Not a robust summary chosen afterwards,
  which is the mistake this repository has already had to walk back once.

  The outcome sentence for losing is written here, in full, including what has to
  change in the abstract if the language model wins. A plan that only says what
  to write when the news is good is not a plan.

  The reproduction has to prove itself before it is allowed to lose or win. A
  baseline that was rebuilt wrongly will score badly, and reporting that as a
  victory would be the worst failure available in this whole exercise. So the
  read refuses unless the reproduction clears a floor and unless the fidelity
  evidence recorded against the authors' own worked example is present.

The status is exploratory, and not because the comparison is weak. It is because
the fold has been read nine times before this plan was written, so nothing read
now can be confirmatory, whichever way it comes out.

Usage: PYTHONPATH=src:tools python3.12 tools/preregister_plmnn.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from pocket_bench.paths import ROOT

SCORES = ROOT / "results/baselines/PLMNN_SCORES.json"
NETWORK = ROOT / "results/baselines/PLMNN_NETWORK.json"
ENDPOINT = ROOT / "results/official_fold/ENDPOINT_STATUS.json"
LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
OUT = ROOT / "results/architecture_sweep/PREREGISTERED_PLMNN.json"

SCHEMA = "geoaudit.preregistered_plmnn.v1"
READ_INDEX = 10
N_BOOT = 10000
SEED = 20260729
# The per-chain budget the counting field deploys at, fixed on the training fold
# long before this read and not revisited here.
TOP_Q = 0.09
# The threshold the authors state their paper used, which is this baseline's own
# deployment rule and the analogue of P2Rank's native pocket assignment.
PLMNN_THRESHOLD = 0.95
# A published supervised model cannot score near chance on the fold it was fitted
# for. If it does, this reproduction is broken and the read is void: claiming a
# win over a baseline one has broken oneself is the one outcome here that would be
# worse than losing.
AUC_FLOOR = 0.65
# At its own threshold the network should call a minority of residues positive. A
# majority would mean the probabilities are not the ones the authors calibrated.
MAX_POSITIVE_RATE_AT_THRESHOLD = 0.5

PAIRED_TESTS = (
    "mean per-unit ROC-AUC, counting field minus pLM-NN",
    "mean per-unit ROC-AUC, P2Rank minus pLM-NN",
    "positive-class F1 at a common top-9% budget, counting field minus pLM-NN",
    "MCC at a common top-9% budget, counting field minus pLM-NN",
    "positive-class F1 at each method's own rule, counting field minus pLM-NN",
    "MCC at each method's own rule, counting field minus pLM-NN",
)


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def build() -> dict:
    if not SCORES.exists():
        raise SystemExit(
            f"missing {SCORES.relative_to(ROOT)}; the baseline has to be run "
            "before its comparison can be planned, because the plan pins the "
            "scores it will read")
    sc = json.loads(SCORES.read_text())
    n_prior = json.loads(LEDGER.read_text())["n_indexed_reads"]
    v = sc["validation_against_the_published_example"]

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "for_test_fold_read_index": READ_INDEX,
        "written_before_the_read": True,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "head_when_written": _git("rev-parse", "HEAD"),

        "question": (
            "does the counting field predict cryptic binding residues as well "
            "as pLM-NN, the cryptic-specific supervised baseline CryptoBench's "
            "own authors fitted and published, on the same 192 single-chain "
            "units and the same residue universe"),

        "why_this_comparison_is_necessary": (
            "the only baseline in the repository is P2Rank, a general pocket "
            "finder not fitted to cryptic sites. A method that beats it has not "
            "yet been shown to be competitive with the task's own supervised "
            "state of the art, and a reviewer is right to say so"),

        "status_declared_in_advance": "exploratory",
        "why_exploratory_and_not_confirmatory": (
            f"the fold carries {n_prior} indexed reads before this plan was "
            "written. Nothing read after that can be confirmatory, and that is "
            "true of a result in either direction. The status is fixed here so "
            "that a favourable outcome cannot be promoted afterwards and an "
            "unfavourable one cannot be demoted"),

        "the_baseline": {
            "what": "CryptoBench pLM-NN: ESM2-3B per-residue embeddings into a "
                    "2560-256-256-2 dense network with relu and softmax",
            "whose": "the benchmark's authors; the weights are theirs, read out "
                     "of the SavedModel they published on osf.io/pz4a9",
            "network_weights_sha256":
                json.loads(NETWORK.read_text())["weights_sha256"],
            "scores_artifact": str(SCORES.relative_to(ROOT)),
            "scores_artifact_sha256":
                hashlib.sha256(SCORES.read_bytes()).hexdigest(),
            "scores_sha256_recorded_inside_it": sc["scores_sha256"],
            "n_units_scored": sc["n_units"],
            "encoder_layer": sc["encoder"]["layer"],
            "encoder_dtype": sc["encoder"]["dtype"],
            "reads_no_label": True,
        },

        "how_faithful_the_reproduction_is": {
            "against_the_authors_worked_example": {
                "unit": v["unit"],
                "mean_cosine_of_the_embedding": v["mean_cosine_at_that_layer"],
                "spearman_of_the_predicted_probabilities":
                    v["predicted_probability_agreement"]["spearman"],
                "max_absolute_probability_difference":
                    v["predicted_probability_agreement"][
                        "max_absolute_difference"],
            },
            "what_is_not_identical": (
                "the embedding agrees with their published example to a mean "
                "cosine of about 0.9987 rather than exactly. Five candidate "
                "causes were tested and excluded, and the remaining one is most "
                "likely a different ESM-2 implementation of the same weights. "
                "The rank agreement of the resulting probabilities is what the "
                "ROC-AUC comparison depends on and it is about 0.997"),
            "so_the_claim_is_about": (
                "our reproduction of their published baseline, and every "
                "sentence about it says so. It is not a claim about a number "
                "they reported, because they report none for this fold that "
                "this repository has read"),
        },

        # Every number the read needs, as a number. The prose below explains each
        # one, but the read is not allowed to recover a threshold by parsing an
        # English sentence.
        "numeric_parameters": {
            "top_q": TOP_Q,
            "plmnn_threshold": PLMNN_THRESHOLD,
            "baseline_auc_floor": AUC_FLOOR,
            "max_positive_rate_at_the_baseline_threshold":
                MAX_POSITIVE_RATE_AT_THRESHOLD,
        },

        "statistic": {
            "primary_functional": (
                "the unweighted mean over units of the paired difference in "
                "per-residue ROC-AUC"),
            "why_the_mean_and_not_a_robust_summary": (
                "it is the endpoint already declared primary in "
                f"{ENDPOINT.relative_to(ROOT)}. Choosing a different summary "
                "for a new baseline, after the mean failed to resolve against "
                "the old one, is the manoeuvre this repository has already had "
                "to retract once"),
            "n_boot": N_BOOT,
            "seed": SEED,
            "ci": 0.95,
            "paired": "the same resampled chains enter both arms",
            "also_reported": [
                "the median paired difference and the per-chain win, loss and "
                "tie counts",
                "each method's own mean per-unit ROC-AUC, so the difference can "
                "be read against its level",
                "the same comparison for P2Rank against pLM-NN, which places "
                "both baselines relative to each other and is the context that "
                "makes our own difference interpretable",
            ],
        },

        "operating_points": {
            "common_budget": {
                "rule": f"the top {TOP_Q:.0%} of residues per chain for both "
                        f"methods, ranked by their own score",
                "why": "a difference in a thresholded metric between two "
                       "methods binarised by different conventions measures the "
                       "conventions. This is the same matched budget the P2Rank "
                       "comparison had to adopt",
            },
            "each_methods_own_rule": {
                "counting_field": f"top {TOP_Q:.0%} per chain, fixed on the "
                                  f"training fold",
                "plmnn": f"probability above {PLMNN_THRESHOLD}, which is the "
                         f"threshold the authors state their paper used",
                "why": "each method's deployed behaviour is also worth "
                       "reporting, as long as it is not confused with a "
                       "comparison of scoring quality",
            },
            "what_is_not_done": (
                "no threshold is tuned for either method on this fold. The "
                "counting field's budget was fixed on the training fold and the "
                "baseline's threshold is the authors' own"),
        },

        "multiplicity": {
            "n_paired_tests": len(PAIRED_TESTS),
            "tests": list(PAIRED_TESTS),
            "correction": "Bonferroni",
            "corrected_level": round(0.05 / len(PAIRED_TESTS), 6),
            "also_reported": "the uncorrected interval alongside it",
        },

        "guards_the_read_must_pass": [
            "this plan's commit is an ancestor of HEAD, checked with git",
            "the scores artifact still hashes to what is recorded here, so the "
            "baseline cannot have been rerun between the plan and the read",
            "the baseline's mean per-unit ROC-AUC clears the floor in "
            "numeric_parameters; below it the reproduction is broken rather "
            "than the baseline weak, and the read is void rather than "
            "favourable",
            "at its own threshold the baseline calls fewer than the recorded "
            "fraction of residues positive on average",
            "the baseline is scored on the same universe, unit by unit, as the "
            "frozen per-structure telemetry, and the read asserts the residue "
            "counts match rather than assuming they do",
            "the counting field's own mean per-unit ROC-AUC recomputed here "
            "reproduces the frozen telemetry to six decimals",
            "the per-unit ROC-AUC of the baseline is computed by the same "
            "function that produced every other method's, not a second "
            "implementation",
        ],

        "decision_rules": {
            "if_the_baseline_is_ahead_and_the_interval_excludes_zero": (
                "it is reported as the headline finding of this read, in the "
                "abstract, and the sentence claiming comparable average "
                "predictive performance is narrowed to P2Rank only. The "
                "contribution then rests on mechanical auditability and on "
                "model size, not on accuracy, and the paper says so in the "
                "abstract rather than in the limitations"),
            "if_the_field_is_ahead_and_the_interval_excludes_zero": (
                "it is reported as the strongest accuracy evidence in the paper "
                "and still labelled exploratory, because of the reads that "
                "preceded this plan. It does not become a confirmatory claim "
                "and it does not license the word state-of-the-art"),
            "if_the_interval_crosses_zero": (
                "it is reported as comparable average performance with the "
                "interval given, which is a substantive result: a network with "
                "a 3-billion-parameter encoder and a 721922-parameter head, "
                "fitted on this benchmark, would then be matched on average by "
                "a method whose entire model is a table of integers"),
            "if_the_baseline_falls_below_the_floor": (
                "the read is void. No comparison is reported, the reproduction "
                "is described as failed, and the paper says that the "
                "cryptic-specific baseline could not be reproduced here rather "
                "than that it performed poorly"),
            "if_the_two_baselines_disagree_about_each_other": (
                "if P2Rank beats pLM-NN on this fold, that is reported plainly, "
                "because it bears on how strong the field's earlier comparison "
                "was and a plan that suppressed it would be selecting"),
        },

        "what_will_be_written_under_each_outcome": {
            "the_interval_crosses_zero": (
                "On the 192 single-chain units of the official test fold, our "
                "reproduction of CryptoBench's own pLM-NN baseline and the "
                "counting field are separated by a mean paired per-unit ROC-AUC "
                "difference whose 95% interval includes zero. A table of "
                "integers with no learned encoder is therefore not "
                "distinguishable on average from a 3-billion-parameter protein "
                "language model fitted to this task, on this fold, at this "
                "sample size. The comparison is exploratory: the fold had been "
                "read nine times before it was planned."),
            "the_baseline_is_ahead": (
                "Our reproduction of CryptoBench's pLM-NN baseline is ahead of "
                "the counting field by a mean paired per-unit ROC-AUC "
                "difference whose interval excludes zero. The paper's accuracy "
                "claim is therefore limited to P2Rank, and the contribution is "
                "auditability and model size rather than predictive "
                "performance. This is stated in the abstract."),
            "the_field_is_ahead": (
                "The counting field is ahead of our reproduction of "
                "CryptoBench's pLM-NN baseline by a mean paired per-unit "
                "ROC-AUC difference whose interval excludes zero. It is the "
                "strongest accuracy evidence in the paper and it remains "
                "exploratory, because nine indexed reads preceded the plan that "
                "specified it."),
            "the_reproduction_fails_its_floor": (
                "Our reproduction of CryptoBench's pLM-NN baseline scored below "
                "the floor this plan set for a working reproduction, so no "
                "comparison against it is reported. What failed is the "
                "reproduction, not necessarily the baseline."),
        },

        "reads_test_fold": False,
        "note": ("this file is the plan. Reading the fold under it is "
                 "tools/plmnn_read.py, which refuses to run until this is "
                 "committed. The baseline's scores are not a read: they contain "
                 "no label and no metric"),
    }


def _report(d: dict) -> None:
    print(f"plan for read {d['for_test_fold_read_index']}, declared "
          f"{d['status_declared_in_advance']}")
    b = d["the_baseline"]
    print(f"  baseline {b['n_units_scored']} units, layer {b['encoder_layer']}, "
          f"{b['encoder_dtype']}, scores {b['scores_artifact_sha256'][:12]}")
    f = d["how_faithful_the_reproduction_is"][
        "against_the_authors_worked_example"]
    print(f"  fidelity on {f['unit']}: cosine "
          f"{f['mean_cosine_of_the_embedding']:.6f}, rank agreement "
          f"{f['spearman_of_the_predicted_probabilities']:.6f}")
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
              f"{have.get('status_declared_in_advance')}; nine indexed reads "
              "precede it and nothing after them is confirmatory")
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
    need = {"the_interval_crosses_zero", "the_baseline_is_ahead",
            "the_field_is_ahead", "the_reproduction_fails_its_floor"}
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
