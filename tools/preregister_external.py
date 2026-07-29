#!/usr/bin/env python3
"""The plan for the external validation: the first read of a set nothing was fitted to.

Every read of CryptoBench's test fold in this repository is exploratory, and no
amount of care can undo that. The fold was read twelve times, eleven architectures
were compared on it, and the honest consequence -- already written into the paper
-- is that a 0.0058 ROC-AUC advantage over P2Rank carries no confirmatory weight.
The only way out of that hole is a set that has never been read at all.

This is the plan for reading one. It is the first preregistration in this
repository that can honestly call itself confirmatory, and it earns that word from
three facts rather than from a declaration: the set is built from structures the
PDB released after CryptoBench's newest one, it shares no UniRef50 cluster at 50%
identity with either CryptoBench fold, and its labels existed before any method was
pointed at it.

Because it is confirmatory, it gets one execution and the outcome is not
negotiable afterwards. What makes that credible is not a promise but the shape of
this file: the three comparisons are named here, the numbers CryptoBench gave for
each of them are pinned here as predictions, and the sentence to be written under
every outcome -- including the outcome where the method's advantage disappears and
the outcome where its deficit widens -- is written here, before the run.

The strongest form of the test is not "is the method good on new data". It is
"does the old result replicate". So each comparison is pinned to its CryptoBench
point estimate and interval, and the question asked of the external number is
whether it lands where CryptoBench said it would.

One commitment is load-bearing and belongs at the top. No architecture, threshold,
feature, quantisation rule or partition bank may change in response to anything
this read returns. If the external numbers are bad, they are published bad. A
method retuned against its own external validation set no longer has one.

Usage: PYTHONPATH=src:tools python3.12 tools/preregister_external.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from pocket_bench.paths import ROOT

EXTERNAL = ROOT / "results/external/EXTERNAL_SET.json"
RULE = ROOT / "results/external/CRYPTOBENCH_RULE.json"
ENDPOINT = ROOT / "results/official_fold/ENDPOINT_STATUS.json"
PLMNN = ROOT / "results/official_fold/PLMNN_READ.json"
POCKETMINER = ROOT / "results/official_fold/POCKETMINER_READ.json"
MATCHED = ROOT / "results/official_fold/MATCHED_FULL_READ.json"
OUT = ROOT / "results/external/PREREGISTERED_EXTERNAL.json"

SCHEMA = "geoaudit.preregistered_external.v1"
N_BOOT = 10000
SEED = 20260730
TOP_Q = 0.09

CO_PRIMARY = (
    "mean paired per-unit ROC-AUC, counting field minus P2Rank",
    "mean paired per-unit ROC-AUC, counting field minus pLM-NN",
    "mean paired per-unit ROC-AUC, counting field minus PocketMiner",
)
SECONDARY = (
    f"positive-class F1 at a common top-{TOP_Q:.0%} budget, counting field minus "
    f"P2Rank",
    f"MCC at a common top-{TOP_Q:.0%} budget, counting field minus P2Rank",
    "positive-class F1 as each is deployed, counting field minus P2Rank",
    "MCC as each is deployed, counting field minus P2Rank",
)


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> dict:
    for p in (EXTERNAL, RULE, ENDPOINT, PLMNN, POCKETMINER, MATCHED):
        if not p.is_file():
            raise SystemExit(
                f"missing {p.relative_to(ROOT)}; the external set and every "
                f"CryptoBench result it is predicted against must exist and be "
                f"frozen before the comparison can be planned")
    ext = json.loads(EXTERNAL.read_text())
    if not ext.get("no_method_has_been_run"):
        raise SystemExit("the external set records that a method has already run "
                         "on it; there is nothing left to preregister")
    endpoint = json.loads(ENDPOINT.read_text())["primary_endpoint"]
    plmnn = json.loads(PLMNN.read_text())["primary_comparison"]
    pm = json.loads(POCKETMINER.read_text())["primary"]
    matched = json.loads(MATCHED.read_text())["conventions"]

    def paired(conv: str, metric: str) -> dict:
        p = matched[conv]["paired"][metric]["chain"]
        return {"delta": p["delta_point"],
                "ci": [p["delta_ci_low"], p["delta_ci_high"]],
                "crosses_zero": p["crosses_zero"]}

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "written_before_the_read": True,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "head_when_written": _git("rev-parse", "HEAD"),

        "question": (
            "on apo-holo pairs released after CryptoBench's newest structure and "
            "sharing no UniRef50 cluster with either of its folds, does the "
            "counting field stand where CryptoBench said it stands: level with "
            "P2Rank, behind pLM-NN, ahead of PocketMiner"),

        "status_declared_in_advance": "confirmatory",
        "why_this_one_may_say_confirmatory": {
            "the_set_has_never_been_read": (
                "no method has been run on it, no score has been computed from "
                "it, and no threshold, feature or architecture in this repository "
                "was chosen with any knowledge of it"),
            "the_labels_came_first": (
                "the set and its labels are frozen and hashed below, and this "
                "plan refers to them by hash. The read that follows refuses to "
                "run if either hash moves"),
            "the_boundary_is_a_date_and_a_cluster_not_a_promise": (
                "both structures of every pair were released after "
                f"{ext['selection']['cutoff']}, and no accession or UniRef50 "
                "cluster is shared with CryptoBench, so externality is checkable "
                "rather than asserted"),
            "what_this_does_not_launder": (
                "the twelve exploratory reads of CryptoBench's test fold stay "
                "exploratory. This read cannot retrospectively confirm them; it "
                "can only succeed or fail on its own terms"),
        },

        "the_set": {
            "artifact": str(EXTERNAL.relative_to(ROOT)),
            "sha256": _sha(EXTERNAL),
            "n_units": ext["n_units_with_a_cryptic_pocket"],
            "n_positive_residues": sum(len(u["residues"]) for u in ext["units"]),
            "n_uniref50_clusters": len({u["cluster"] for u in ext["units"]}),
            "cutoff": ext["selection"]["cutoff"],
            "labelling_rule": str(RULE.relative_to(ROOT)),
            "labelling_rule_sha256": _sha(RULE),
            "one_unit_per_cluster": True,
            "why_one_per_cluster_matters_for_the_statistics": (
                "the units are one per UniRef50 cluster by construction, so "
                "resampling chains and resampling clusters are the same "
                "operation here and the interval needs no cluster correction"),
        },

        "how_this_set_is_easier_and_harder_than_cryptobench": {
            "easier": [
                "pairs whose pocket moves between 1.5 and 2.5 A are not labelled "
                "either way, which removes the borderline cases; CryptoBench "
                "keeps everything above 2.0",
                "only chemical components CryptoBench itself accepted count as "
                "ligands, so novel chemistry deposited since the cutoff is "
                "absent",
                "X-ray only, at 2.5 A or better",
            ],
            "harder": [
                "no protein here has a relative within 50% identity anywhere in "
                "CryptoBench, so nothing can be carried over from a homologue "
                "the training fold contained",
                "one unit per cluster, so no target is represented twice and no "
                "easy family can be won repeatedly",
            ],
            "why_both_are_stated_now": (
                "so that neither can be produced afterwards to explain a result "
                "in whichever direction it goes"),
        },

        "what_cryptobench_said_these_numbers_would_be": {
            "note": ("pinned as predictions, from the frozen reads, before the "
                     "external set is scored. The comparison of interest is "
                     "whether each external difference lands inside the interval "
                     "CryptoBench gave for it"),
            "counting_field_minus_p2rank_roc_auc": {
                "delta": endpoint["delta"], "ci": endpoint["ci95"],
                "resolved": endpoint["resolves"],
                "reading": "level on average; the interval includes zero",
            },
            "counting_field_minus_plmnn_roc_auc": {
                "delta": plmnn["mean"], "ci": plmnn["ci"],
                "resolved": plmnn["excludes_zero"],
                "reading": "behind, and the interval excludes zero",
                "levels": [plmnn["level_first"], plmnn["level_second"]],
            },
            "counting_field_minus_pocketminer_roc_auc": {
                "delta": pm["table_field_minus_pocketminer"]["mean"],
                "ci": pm["table_field_minus_pocketminer"]["ci"],
                "resolved": pm["table_field_minus_pocketminer"]["excludes_zero"],
                "reading": ("ahead, but P2Rank is ahead of PocketMiner by a "
                            "similar margin, so this reflects the two label "
                            "definitions more than a ranking of methods"),
                "p2rank_minus_pocketminer":
                    pm["p2rank_minus_pocketminer"]["mean"],
            },
            "counting_field_minus_p2rank_f1_common_budget":
                paired("D2_common_budget", "positive_class_f1"),
            "counting_field_minus_p2rank_mcc_common_budget":
                paired("D2_common_budget", "mcc"),
            "counting_field_minus_p2rank_f1_as_deployed":
                paired("D1_as_deployed", "positive_class_f1"),
            "counting_field_minus_p2rank_mcc_as_deployed":
                paired("D1_as_deployed", "mcc"),
        },

        "methods_and_what_is_frozen_about_them": {
            "counting_field": {
                "what": "the table-valued counting field, at the architecture "
                        "selected on the training fold and unchanged since",
                "frozen": ["the 43 local quantities and their expansion",
                           "the 16 partition banks and their seeds",
                           "the ridge and the fan-out cap",
                           "within-chain quantisation",
                           f"the deployed per-chain budget of top {TOP_Q:.0%}"],
                "may_change_after_this_read": False,
            },
            "p2rank": {"what": "P2Rank at the pinned version, native pocket "
                               "assignment and a common top-q budget",
                       "may_change_after_this_read": False},
            "plmnn": {"what": "CryptoBench's own supervised baseline, restored "
                              "from published weights over ESM2-3B embeddings",
                      "may_change_after_this_read": False},
            "pocketminer": {"what": "the published GVP graph network, restored "
                                    "from published weights",
                            "may_change_after_this_read": False},
            "every_method_sees_the_same_input": (
                "one receptor PDB per unit, written by "
                "pocket_bench.pdb_io.write_receptor_only_pdb, the same writer "
                "that produced the CryptoBench receptors"),
        },

        "statistic": {
            "primary_functional": ("the unweighted mean over units of the paired "
                                   "difference in per-residue ROC-AUC"),
            "why_the_same_functional_as_before": (
                "it is the endpoint already declared primary for the CryptoBench "
                "fold. Switching summaries for the external read would make the "
                "replication unfalsifiable, and this repository has retracted one "
                "statistic chosen after the fact already"),
            "trimmed_mean_is_not_used_here": (
                "it is exploratory on the old fold and it stays exploratory; it "
                "is reported for completeness and no verdict rests on it"),
            "n_boot": N_BOOT, "seed": SEED, "ci": 0.95,
            "paired": "the same resampled units enter both arms",
            "resampling_unit": "the unit, which is also the UniRef50 cluster",
            "also_reported": [
                "each method's own mean per-unit ROC-AUC, so a difference can be "
                "read against its level",
                "per-unit win, loss and tie counts",
                "the median paired difference",
                "P2Rank minus PocketMiner and pLM-NN minus P2Rank, which place "
                "the baselines against each other on new data",
            ],
        },

        "co_primary": {
            "tests": list(CO_PRIMARY),
            "why_three_and_not_one": (
                "the paper makes three claims about where the method stands and "
                "all three are on trial. Declaring one and reporting the others "
                "as context would let the most favourable of them be promoted"),
            "correction": "Bonferroni over the three",
            "corrected_level": round(0.05 / len(CO_PRIMARY), 6),
        },
        "secondary": {
            "tests": list(SECONDARY),
            "correction": "Bonferroni over the four",
            "corrected_level": round(0.05 / len(SECONDARY), 6),
            "status": ("secondary; a thresholded metric depends on the "
                       "binarisation convention, which is why the ranking "
                       "statistic is primary"),
        },

        "replication_verdicts_defined_now": {
            "replicates": ("the external point estimate lands inside the "
                           "CryptoBench 95% interval for that comparison"),
            "same_direction_but_outside": ("the sign agrees and the magnitude "
                                           "lands outside the old interval"),
            "fails_to_replicate": ("the sign reverses, or a comparison that "
                                   "excluded zero on CryptoBench now excludes "
                                   "zero on the other side"),
            "unresolved_externally": ("the external interval includes zero, "
                                      "whatever CryptoBench said"),
        },

        "decision_rules": {
            "if_the_p2rank_comparison_replicates_as_level": (
                "the paper's claim of comparable average performance against "
                "P2Rank gains its first confirmatory support, and is stated as "
                "confirmed on external data at this sample size. It remains a "
                "claim of parity, not of advantage"),
            "if_the_field_is_ahead_of_p2rank_and_the_interval_excludes_zero": (
                "this is the first confirmatory accuracy advantage in the paper "
                "and it goes in the abstract, with the external sample size and "
                "the ways this set is easier stated in the same breath"),
            "if_p2rank_is_ahead_and_the_interval_excludes_zero": (
                "the parity claim fails externally. The abstract is rewritten to "
                "say the counting field is behind P2Rank on external data, and "
                "the contribution rests on auditability and model size alone. "
                "This sentence is written here so that it cannot be softened "
                "later"),
            "if_the_plmnn_deficit_replicates": (
                "the paper keeps the deficit in the abstract, now supported "
                "externally rather than only on a fold that had been read twelve "
                "times"),
            "if_the_plmnn_deficit_widens": (
                "the wider figure is the one reported, in the abstract, and the "
                "paper says so plainly"),
            "if_the_plmnn_deficit_disappears": (
                "reported as unresolved externally at this sample size, with the "
                "CryptoBench interval quoted beside it. It is not reported as "
                "parity with pLM-NN on the strength of a smaller set failing to "
                "resolve a difference the larger one resolved"),
            "if_the_pocketminer_advantage_replicates": (
                "reported with the same qualification as before: P2Rank's margin "
                "over PocketMiner is reported next to ours, because if both are "
                "similar the comparison is measuring the label definition"),
            "if_the_external_set_turns_out_too_small_to_resolve_anything": (
                "reported as such. A set that cannot resolve the differences it "
                "was built to test is a negative result about the experiment, "
                "and the paper says the external validation was underpowered "
                "rather than quietly leaning on the CryptoBench numbers"),
        },

        "what_will_be_written_under_each_outcome": {
            "p2rank_parity_replicates": (
                "On {n} apo-holo units built from structures released after "
                "CryptoBench's newest, sharing no UniRef50 cluster with either of "
                "its folds, the counting field and P2Rank differ in mean paired "
                "per-unit ROC-AUC by {d}, with a 95% interval of {ci}. The parity "
                "reported on the official fold therefore holds on data that no "
                "part of this method's development could have seen. This is the "
                "only comparison in the paper that is confirmatory."),
            "p2rank_advantage_appears": (
                "On {n} external units the counting field leads P2Rank by {d} in "
                "mean paired per-unit ROC-AUC, 95% interval {ci}, on a set whose "
                "labels were fixed before any method was run and whose proteins "
                "share no UniRef50 cluster with CryptoBench. This is the paper's "
                "first confirmatory accuracy result. It is reported together with "
                "the three respects in which the external set is easier than "
                "CryptoBench."),
            "p2rank_advantage_reverses": (
                "On {n} external units P2Rank leads the counting field by {d} in "
                "mean paired per-unit ROC-AUC, 95% interval {ci}. The parity "
                "claimed on the official fold does not survive external "
                "validation, and we withdraw it. The method's contribution is "
                "mechanical auditability and model size, not predictive "
                "accuracy."),
            "plmnn_deficit_replicates": (
                "The deficit against CryptoBench's own supervised baseline "
                "replicates externally: {d} in mean paired per-unit ROC-AUC, 95% "
                "interval {ci}, against {old} on the official fold. A supervised "
                "network over a protein language model remains the more accurate "
                "method."),
            "plmnn_deficit_unresolved": (
                "On the external set the difference against pLM-NN is {d}, 95% "
                "interval {ci}, which includes zero. We do not read this as "
                "parity: the official fold resolved this difference against us at "
                "a larger sample size, and a smaller set failing to resolve it is "
                "not evidence that it is absent."),
            "underpowered": (
                "The external set resolves none of the three comparisons it was "
                "built to test. We report it as underpowered at {n} units rather "
                "than as support for any of the three claims."),
        },

        "guards_the_read_must_pass": [
            "this plan's commit is an ancestor of HEAD, checked with git",
            "the external set still hashes to what is recorded here, so it "
            "cannot have been rebuilt after the plan was written",
            "the recovered labelling rule still hashes to what is recorded here",
            "no unit shares an accession or a UniRef50 cluster with CryptoBench",
            "every unit's apo structure was released after the cutoff",
            "each unit appears once, and its cluster appears once",
            "every method is scored on the same residue universe per unit, and "
            "the read asserts the counts agree rather than assuming it",
            "every method's per-unit ROC-AUC comes from one shared function",
            "the read records that it ran once, and refuses to overwrite an "
            "existing result without an explicit flag that is recorded in the "
            "artifact",
        ],

        "the_commitment_that_matters": (
            "no architecture, threshold, feature, quantisation rule, partition "
            "bank or post-processing step changes in response to this read. If "
            "the numbers are bad they are published bad. A method retuned "
            "against its own external validation set does not have one"),

        "reads_test_fold": False,
        "reads_external_set": False,
        "note": ("this file is the plan. Reading the external set under it is "
                 "tools/external_read.py, which refuses to run until this is "
                 "committed"),
    }


def _report(d: dict) -> None:
    s = d["the_set"]
    print(f"plan for the external read, declared "
          f"{d['status_declared_in_advance']}")
    print(f"  set {s['n_units']} units, {s['n_positive_residues']} positive "
          f"residues, {s['n_uniref50_clusters']} clusters, "
          f"hash {s['sha256'][:12]}")
    print(f"  released after {s['cutoff']}, no UniRef50 cluster shared with "
          f"CryptoBench")
    p = d["what_cryptobench_said_these_numbers_would_be"]
    for name, key in (("P2Rank", "counting_field_minus_p2rank_roc_auc"),
                      ("pLM-NN", "counting_field_minus_plmnn_roc_auc"),
                      ("PocketMiner",
                       "counting_field_minus_pocketminer_roc_auc")):
        b = p[key]
        print(f"    predicted against {name}: {b['delta']:+.4f} "
              f"[{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]")
    print(f"  co-primary {len(d['co_primary']['tests'])} at Bonferroni level "
          f"{d['co_primary']['corrected_level']}, secondary "
          f"{len(d['secondary']['tests'])} at "
          f"{d['secondary']['corrected_level']}")
    print(f"  outcome sentences written in advance: "
          f"{len(d['what_will_be_written_under_each_outcome'])}")


def _check() -> int:
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    have = json.loads(OUT.read_text())
    if have.get("schema") != SCHEMA:
        print(f"FAILED: schema {have.get('schema')}")
        return 1
    if have.get("reads_external_set") or have.get("reads_test_fold"):
        print("FAILED: the plan claims to have read what it plans to read")
        return 1
    if have.get("status_declared_in_advance") != "confirmatory":
        print(f"FAILED: the plan declares itself "
              f"{have.get('status_declared_in_advance')}. This set has never "
              f"been read; if that is no longer true the plan is void, and if it "
              f"is true the status is confirmatory")
        return 1
    if have["methods_and_what_is_frozen_about_them"]["counting_field"][
            "may_change_after_this_read"]:
        print("FAILED: the plan permits the method to change after the read")
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
        print("FAILED: the committed plan no longer describes its inputs; these "
              f"fields would change: {moved}")
        for k in moved:
            print(f"  - {k}\n      committed: {json.dumps(have.get(k))[:200]}"
                  f"\n      would be:  {json.dumps(want[k])[:200]}")
        return 1
    need = {"p2rank_parity_replicates", "p2rank_advantage_appears",
            "p2rank_advantage_reverses", "plmnn_deficit_replicates",
            "plmnn_deficit_unresolved", "underpowered"}
    said = set(have["what_will_be_written_under_each_outcome"])
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
