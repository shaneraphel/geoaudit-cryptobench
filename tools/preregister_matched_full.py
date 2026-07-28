#!/usr/bin/env python3
"""The plan for the seventh read: four conventions, four metrics, three clusterings.

The sixth read answered one question -- does the F1 margin survive a common
calling fraction -- and answered it with one metric under one resampling scheme.
Three gaps were left, each of which a reviewer named:

  A comparison reported only as F1 does not say where the F1 came from. A method
  can win F1 by calling more residues and trading precision for recall, and at a
  matched calling fraction it cannot, so precision and recall at a matched budget
  are the numbers that distinguish a better ranking from a looser threshold.

  A comparison optimised for F1 says nothing about MCC. The two do not peak at
  the same calling fraction: on the training fold ours peaks at 0.11 under MCC
  and 0.09 under F1, while P2Rank's peaks at 0.09 under both. So "each method
  tuned on the training fold" is genuinely two different conventions, and only
  one of them has been read.

  A bootstrap that resamples chains assumes chains are independent. The official
  test fold is MMseqs2-clustered at 10% identity, so this is close to true here,
  but "close to true" is a claim with a number attached and the number has not
  been reported.

What this plan may not do is pretend to be blind about F1. The sixth read
already published the matched F1 delta. Read seven recomputes it and must
reproduce it to six decimals; that is a calibration, not a finding, and it is
recorded as one. The genuinely new statements are precision, recall and MCC at
matched thresholds, the MCC-tuned convention, and the cluster resamplings.

Usage: PYTHONPATH=src python3.12 tools/preregister_matched_full.py
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_OP = ROOT / "results/architecture_sweep/TRAIN_OPERATING_POINTS.json"
FIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"
BOOT = ROOT / "results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json"
READ6 = ROOT / "results/official_fold/MATCHED_OPERATING_POINT_READ.json"
MANIFEST = ROOT / "data/cryptobench_apo/official_manifest.json"
OUT = ROOT / "results/architecture_sweep/PREREGISTERED_MATCHED_FULL.json"

SCHEMA = "geoaudit.preregistered_matched_full.v1"
READ_INDEX = 7


def _code_sha256() -> str:
    import hashlib
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def build() -> dict:
    op = json.loads(TRAIN_OP.read_text())
    sel = op["selected"]
    gain = op["what_tuning_p2rank_is_worth_on_the_training_fold"]
    pub = json.loads(BOOT.read_text())["metrics"]
    read6 = json.loads(READ6.read_text())
    entries = json.loads(MANIFEST.read_text())["entries"]

    q_ours_f1 = sel["table_field/pooled_f1/full_fold"]["q"]
    q_ours_mcc = sel["table_field/pooled_mcc/full_fold"]["q"]
    q_p2_f1 = sel["p2rank/pooled_f1/full_fold"]["q"]
    q_p2_mcc = sel["p2rank/pooled_mcc/full_fold"]["q"]
    shipped_q = json.loads(FIELD.read_text())["operating_point"]["q"]
    if abs(q_ours_f1 - shipped_q) != 0:
        raise SystemExit(
            f"the F1-tuned q on the training fold is {q_ours_f1} but the "
            f"shipped field carries {shipped_q}; the deployment convention "
            f"and the F1-tuned convention would not be the same rule and this "
            f"plan describes them as though they were")

    f1_d = pub["residue_f1"]["paired_vs_baseline"]["table_field"]
    mcc_d = pub["residue_mcc"]["paired_vs_baseline"]["table_field"]

    n_units = len(entries)
    n_pdb = len({e["pdb"] for e in entries})
    n_clu = len({e["cluster_id"] for e in entries})

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "for_test_fold_read_index": READ_INDEX,
        "question": (
            "at a matched calling budget, does the advantage show up in "
            "precision, in recall, in F1 and in MCC; does it survive when each "
            "method is tuned for MCC rather than F1; and does it survive when "
            "the resampling unit is a sequence cluster rather than a chain"),
        "written_before_the_read": True,
        "code_sha256": _code_sha256(),
        "head_when_written": _git("rev-parse", "HEAD"),

        "what_is_already_known_and_is_therefore_not_a_finding": {
            "the_matched_f1_delta": {
                "published_in": str(READ6.relative_to(ROOT)),
                "read_index": read6["test_fold_read_index"],
                "delta": read6["matched"]["A_common_q"]["primary"]["delta_point"],
                "ci": [read6["matched"]["A_common_q"]["primary"]["delta_ci_low"],
                       read6["matched"]["A_common_q"]["primary"]["delta_ci_high"]],
                "status": (
                    "read seven will recompute this and must reproduce it to "
                    "six decimals. Reproducing a published number is a "
                    "calibration of the arithmetic, not a result, and it is "
                    "not counted as one anywhere in this plan or the paper"),
            },
            "why_this_is_stated_here": (
                "a plan that listed the matched F1 delta among its outcomes "
                "would be claiming to be blind to something already in the "
                "repository. The genuinely unread quantities are enumerated "
                "separately below and are the only ones the decision rules "
                "are allowed to govern"),
        },

        "genuinely_unread_before_this_plan": [
            "precision at a matched calling budget, either method",
            "recall at a matched calling budget, either method",
            "MCC at a matched calling budget, either method",
            "any quantity under the MCC-tuned convention",
            "any confidence interval from resampling PDB entries or sequence "
            "clusters rather than chains",
        ],

        "conventions_to_be_read": [
            {
                "id": "D1_as_deployed",
                "what": "each method as it is actually run",
                "table_field": f"per-chain top-{shipped_q:.2f}",
                "p2rank": "P2Rank's own pocket assignment",
                "is_new": False,
                "note": "the published headline. Recomputed as a calibration; "
                        "F1 and MCC here must match the frozen bootstrap",
            },
            {
                "id": "D2_common_budget",
                "what": "one calling fraction, both methods",
                "table_field": f"per-chain top-{shipped_q:.2f}",
                "p2rank": f"per-chain top-{shipped_q:.2f}",
                "is_new": "only precision, recall and MCC",
                "q": shipped_q,
            },
            {
                "id": "D3_each_tuned_for_f1",
                "what": "each method at the q that maximised pooled F1 on the "
                        "training fold",
                "q_table_field": q_ours_f1,
                "q_p2rank": q_p2_f1,
                "is_new": "only precision, recall and MCC",
                "coincides_with_D2": abs(q_ours_f1 - shipped_q) < 1e-9
                                     and abs(q_p2_f1 - shipped_q) < 1e-9,
            },
            {
                "id": "D4_each_tuned_for_mcc",
                "what": "each method at the q that maximised pooled MCC on the "
                        "training fold",
                "q_table_field": q_ours_mcc,
                "q_p2rank": q_p2_mcc,
                "is_new": True,
                "thresholds_differ_between_methods": abs(q_ours_mcc - q_p2_mcc) > 1e-9,
                "why_it_is_a_separate_convention": (
                    "the two objectives select different calling fractions for "
                    "our method (0.11 against 0.09) and the same one for "
                    "P2Rank, so D4 is the only convention in which the two "
                    "methods are held to different budgets by a rule neither "
                    "chose on the held-out fold"),
            },
        ],

        "metrics_to_be_reported_for_every_convention": [
            "precision", "recall", "positive_class_f1", "mcc",
        ],
        "statistic": {
            "per_unit": "each metric computed on one chain's residues",
            "summary": "unweighted mean over the units both methods could be "
                       "scored on",
            "paired": "the same resampled units enter both arms, so the "
                      "correlation between two detectors looking at the same "
                      "pocket cancels out of the difference",
            "n_boot": 10000,
            "seed": 20260725,
            "ci": 0.95,
            "secondary": "20% trimmed mean, reported alongside every primary "
                         "so a margin carried by a few chains is visible",
        },

        "resampling_units": [
            {"id": "chain", "n": n_units,
             "what": "the evaluation unit; the scheme every published CI used"},
            {"id": "pdb_entry", "n": n_pdb,
             "what": "all chains of a drawn PDB entry enter together"},
            {"id": "uniprot_cluster", "n": n_clu,
             "what": "all chains of a drawn MMseqs2 10%-identity cluster enter "
                     "together; cluster_id in the official manifest is the "
                     "UniProt accession"},
        ],
        "what_the_clustering_can_change": (
            f"the fold has {n_units} chains in {n_pdb} PDB entries and {n_clu} "
            f"sequence clusters, so the largest possible reduction in "
            f"effective sample size is {n_units - n_clu} units. An interval "
            f"that widened materially under cluster resampling would mean the "
            f"clustering is not what bounds it, and would have to be "
            f"investigated rather than reported"),

        "forecast": {
            "method": (
                "the tuning gain measured on the training fold is subtracted "
                "from the published held-out margin. This underestimated the "
                "matched F1 delta by 0.0073 at read six, so it is a biased "
                "predictor with a known sign, and it is recorded before the "
                "read rather than fitted after it"),
            "f1": {
                "published_delta": f1_d["delta_point"],
                "training_tuning_gain_to_p2rank": gain["pooled_f1"]["the_tuning_is_worth"],
                "predicted_matched_delta": round(
                    f1_d["delta_point"] - gain["pooled_f1"]["the_tuning_is_worth"], 6),
                "actually_observed_at_read_six": read6["matched"]["A_common_q"][
                    "primary"]["delta_point"],
            },
            "mcc": {
                "published_delta": mcc_d["delta_point"],
                "published_ci": [mcc_d["delta_ci_low"], mcc_d["delta_ci_high"]],
                "training_tuning_gain_to_p2rank": gain["pooled_mcc"]["the_tuning_is_worth"],
                "predicted_matched_delta": round(
                    mcc_d["delta_point"] - gain["pooled_mcc"]["the_tuning_is_worth"], 6),
                "predicted_to_cross_zero": True,
                "why": (
                    "the published MCC interval already reaches to +0.0004, so "
                    "it excludes zero by a margin smaller than the tuning gain "
                    "being removed from it. Any positive shift of the lower "
                    "bound puts zero inside"),
            },
            "precision_and_recall": (
                "at a matched calling budget the two methods call the same "
                "number of residues per chain up to rounding, so precision and "
                "recall must move together and their deltas must have the same "
                "sign. A convention in which they do not is an arithmetic bug "
                "and the read is to fail rather than report it"),
        },

        "decision_rules": {
            "applies_to": ["positive_class_f1", "mcc"],
            "evaluated_on": "convention D2, the common calling budget, under "
                            "chain resampling, primary statistic",
            "if_the_interval_excludes_zero": (
                "the improvement stands as an improvement and is stated as one"),
            "if_the_point_is_positive_and_the_interval_crosses_zero": (
                "it is stated as numerically higher but unresolved at this "
                "sample size, and no sentence in the paper may call it an "
                "improvement"),
            "if_the_point_is_zero_or_negative": (
                "the advantage under that metric is stated to depend on the "
                "operating-point convention, and the deployment-rule number is "
                "reported only as what the deployment rules give"),
            "cluster_resampling_is_a_robustness_check": (
                "it cannot promote a conclusion. If a chain-resampled interval "
                "crosses zero and a cluster-resampled one does not, the "
                "crossing one governs, because a narrower interval from a "
                "coarser resampling would be an artefact"),
        },

        "what_will_be_written_under_each_outcome": {
            "f1_and_mcc_both_unresolved": (
                "At a matched calling budget neither the F1 nor the MCC margin "
                "is resolved at this sample size. Both point estimates remain "
                "positive and both intervals contain zero, so what the fold "
                "supports is a ranking advantage in ROC-AUC and a threshold "
                "convention that accounts for most of the reported gap in the "
                "binary summaries."),
            "f1_unresolved_mcc_survives": (
                "At a matched calling budget the F1 margin is unresolved while "
                "the MCC margin excludes zero. MCC weights the negatives that "
                "F1 ignores, so the surviving quantity is the one that counts "
                "the residues correctly left uncalled, and that is what is "
                "claimed."),
            "both_survive": (
                "At a matched calling budget both the F1 and the MCC margins "
                "exclude zero, so the advantage reported under the deployment "
                "rules is not an artefact of those rules."),
            "either_reverses": (
                "At a matched calling budget the margin reverses under at "
                "least one binary summary. The advantage under the deployment "
                "rules is therefore a property of the operating-point "
                "convention and is withdrawn as a claim about the scores."),
        },

        "guards_the_read_must_pass": [
            "this plan's commit is an ancestor of HEAD, checked with git",
            "the deployment-rule F1 and MCC reproduce the frozen bootstrap to "
            "four decimals",
            "the matched F1 delta reproduces read six to six decimals",
            "precision and recall deltas share a sign under every matched "
            "convention",
            "no per-unit metric is silently dropped: the number of units "
            "entering each paired difference is reported for every metric and "
            "every convention",
        ],
        "reads_test_fold": False,
        "note": "this file is the plan. Reading the fold under it is "
                "tools/matched_full_read.py, which refuses to run until this "
                "is committed.",
    }


def main() -> int:
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  conventions: {[c['id'] for c in d['conventions_to_be_read']]}")
    for c in d["conventions_to_be_read"]:
        if c["id"] == "D4_each_tuned_for_mcc":
            print(f"  D4 thresholds differ: {c['thresholds_differ_between_methods']} "
                  f"(ours {c['q_table_field']}, P2Rank {c['q_p2rank']})")
    f = d["forecast"]
    print(f"  forecast matched F1  {f['f1']['predicted_matched_delta']:+.4f} "
          f"(observed at read 6: {f['f1']['actually_observed_at_read_six']:+.4f})")
    print(f"  forecast matched MCC {f['mcc']['predicted_matched_delta']:+.4f}, "
          f"predicted to cross zero: {f['mcc']['predicted_to_cross_zero']}")
    r = d["resampling_units"]
    print(f"  resampling: " + ", ".join(f"{x['id']} n={x['n']}" for x in r))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
