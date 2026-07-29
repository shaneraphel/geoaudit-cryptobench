#!/usr/bin/env python3
"""The plan for the eighth read: where the field wins, and why that is not a claim.

A subgroup analysis run after the overall comparison came back unresolved is the
textbook way to manufacture a result. Fifteen subgroups at the conventional level
will hand you roughly one exclusion of zero from noise alone, and the one you get
will have a story attached, because every subgroup here has a story available.
This plan exists so that the analysis can be run without that being what happens.

Three commitments make the difference, and all three are made before the read:

  The covariates are fixed and there are five. They are the five a reviewer
  named -- apo/holo displacement, pocket size, positive rate, chain length,
  structural quality -- not five chosen from a longer list after seeing which
  split the differences. They are computed by tools/subgroup_covariates.py from
  the deposit, the labels and the coordinates, with no score opened, and that
  artifact is committed before this plan.

  The cuts are fixed and they are tertiles by order statistic. Not chosen, not
  optimised, not moved to make a group cleaner. The group sizes are therefore
  known now and are recorded below, so a read that reported different ones would
  have redrawn them.

  The conclusion is fixed in advance, and it is that there is no conclusion. No
  subgroup result may become a claim in the paper. This is declared here, before
  the numbers, precisely because it will be tempting afterwards. What the read is
  allowed to produce is a description of where the two detectors differ, offered
  as a map of the fold and as a generator of hypotheses for the external
  evaluation that has not been run yet.

The read is still indexed. Nothing is rescored -- the per-unit AUCs were frozen at
read 1 -- but partitioning frozen numbers by a covariate and testing inside the
parts is a new inferential use of the fold, which is the ledger's own definition.

Usage: PYTHONPATH=src:tools python3.12 tools/preregister_subgroups.py [--check]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from pocket_bench.paths import ROOT

COV = ROOT / "results/official_fold/SUBGROUP_COVARIATES.json"
ENDPOINT = ROOT / "results/official_fold/ENDPOINT_STATUS.json"
PREREG_READ = ROOT / "results/official_fold/PREREGISTERED_READ.json"
OUT = ROOT / "results/architecture_sweep/PREREGISTERED_SUBGROUPS.json"

SCHEMA = "geoaudit.preregistered_subgroups.v1"
READ_INDEX = 8
N_BOOT = 10000
SEED = 20260725
BAND_NAMES = ("low", "mid", "high")


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def build() -> dict:
    cov = json.loads(COV.read_text())
    ep = json.loads(ENDPOINT.read_text())
    read5 = json.loads(PREREG_READ.read_text())

    covs = cov["covariates"]
    n_tests = len(covs) * len(BAND_NAMES)
    shape = read5["shape_of_the_differences"]

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "for_test_fold_read_index": READ_INDEX,
        "written_before_the_read": True,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "head_when_written": _git("rev-parse", "HEAD"),

        "question": (
            "on which kinds of protein does the counting field beat P2Rank and "
            "on which does it lose, partitioned by five covariates fixed before "
            "the read; and what is the full per-chain win, loss and tie "
            "distribution behind the summary statistics"),

        "status_declared_in_advance": "exploratory",
        "why_exploratory_and_not_confirmatory": (
            "the primary endpoint of this paper is the mean paired ROC-AUC and "
            "it does not resolve. A subgroup analysis run afterwards cannot "
            "promote an unresolved overall result, and a subgroup that excludes "
            "zero here is at least as likely to be one of the "
            f"{n_tests} draws as a property of the proteins. This is recorded "
            "before the read so that it cannot be decided after it"),

        "covariates": [
            {
                "id": c,
                "source": cov["covariate_sources"][c],
                "cuts": cov["distributions"][c]["tertile_cuts"],
                "group_sizes": cov["distributions"][c]["group_sizes"],
                "n_defined": cov["distributions"][c]["n_defined"],
            }
            for c in covs
        ],
        "covariate_artifact": {
            "path": str(COV.relative_to(ROOT)),
            "sha256": hashlib.sha256(COV.read_bytes()).hexdigest(),
            "reads_any_score": False,
        },
        "bands": list(BAND_NAMES),
        "banding_rule": (
            "tertiles by order statistic on the covariate alone: low is below "
            "the first cut, mid is between the cuts, high is at or above the "
            "second. Chosen because it fixes the group sizes before the read "
            "and because no rule that looks at the differences can be defended"),

        "what_is_already_known_and_is_therefore_not_a_finding": {
            "the_overall_paired_differences": {
                "published_in": str(PREREG_READ.relative_to(ROOT)),
                "read_index": read5["test_fold_read_index"],
                "mean": ep["primary_endpoint"]["delta"],
                "mean_ci": ep["primary_endpoint"]["ci95"],
                "n_field_ahead": shape["n_field_ahead"],
                "n_baseline_ahead": shape["n_baseline_ahead"],
                "status": (
                    "read eight recomputes these as a calibration and must "
                    "reproduce them to six decimals. Reproducing a published "
                    "number is not a finding"),
            },
            "why_this_is_stated_here": (
                "the win and loss counts are already in the repository, so a "
                "plan that listed them among its outcomes would be claiming to "
                "be blind to something published. What is genuinely unread is "
                "the tie count, the shape of the distribution beyond its "
                "quantiles, and every quantity computed inside a subgroup"),
        },

        "genuinely_unread_before_this_plan": [
            "any paired difference restricted to a subgroup of the fold",
            "the number of chains on which the two detectors tie exactly",
            "the correlation between any covariate and the paired difference",
            "the per-chain difference distribution beyond the quantiles read "
            "five published",
        ],

        "statistic": {
            "per_unit": "the paired difference in per-residue ROC-AUC, table "
                        "field minus P2Rank, as frozen by read one",
            "summary": "unweighted mean within the band",
            "n_boot": N_BOOT,
            "seed": SEED,
            "ci": 0.95,
            "paired": "the same resampled chains enter both arms",
            "also_reported": "median and win/loss/tie counts per band, because "
                             "the mean is the statistic that this fold's tails "
                             "are known to move",
        },

        "multiplicity": {
            "n_subgroup_tests": n_tests,
            "correction": "Bonferroni",
            "corrected_level": round(0.05 / n_tests, 6),
            "also_reported": "the uncorrected interval, so a reader can see "
                             "what the correction cost rather than being shown "
                             "only its result",
            "expected_false_positives_uncorrected": round(0.05 * n_tests, 2),
            "why_bonferroni_and_not_something_sharper": (
                "the bands within a covariate are disjoint but the five "
                "covariates are correlated -- pocket size and positive rate "
                "especially -- so the true number of independent tests is below "
                f"{n_tests} and Bonferroni is conservative. A sharper "
                "correction would need an assumption about that correlation "
                "which this fold is too small to check"),
        },

        "trend_test": {
            "what": "Spearman correlation between each covariate and the "
                    "per-chain paired difference, over all units where the "
                    "covariate is defined",
            "why": "a monotone trend across three bands is a stronger and "
                   "cheaper statement than any single band excluding zero, and "
                   "it uses the covariate as the continuous quantity it is "
                   "rather than throwing away its resolution",
            "n_tests": len(covs),
            "corrected_level": round(0.05 / len(covs), 6),
        },

        "decision_rules": {
            "no_subgroup_may_become_a_claim": (
                "whatever the read returns, no sentence in the paper may state "
                "that the field is better than P2Rank on any subgroup of "
                "proteins. The overall endpoint is unresolved and a subgroup "
                "cannot repair that"),
            "if_a_band_excludes_zero_after_correction": (
                "it is reported as the strongest signal the fold offers about "
                "where the difference lives, described as a hypothesis for the "
                "external evaluation, and labelled exploratory in the same "
                "sentence"),
            "if_a_band_excludes_zero_only_before_correction": (
                "it is reported with both intervals and explicitly called "
                "consistent with noise at this number of tests"),
            "if_nothing_survives": (
                "that is reported as the result. A subgroup analysis that finds "
                "nothing is informative: it says the difference is not "
                "concentrated in any of the five kinds of protein a reviewer "
                "would have asked about"),
            "if_a_trend_and_a_band_disagree": (
                "the trend governs, because it uses the covariate at full "
                "resolution and costs one test rather than three"),
        },

        "what_will_be_written_under_each_outcome": {
            "nothing_survives_correction": (
                "Partitioned by apo/holo displacement, pocket size, positive "
                "rate, chain length and mean B-factor, no third of the fold "
                "shows a difference that survives correction for the number of "
                "partitions examined. The difference between the two detectors "
                "is not concentrated in any kind of protein this benchmark "
                "records, which is a constraint on the explanations available "
                "for it and is reported as one."),
            "a_band_survives_correction": (
                "One partition of the fold shows a difference that survives "
                "correction. It is reported as the strongest signal the fold "
                "offers about where the two detectors diverge and as a "
                "hypothesis for external evaluation; it is not offered as "
                "evidence that the field is better on that kind of protein, "
                "because the overall endpoint is unresolved and this is one of "
                f"{n_tests} partitions examined."),
            "a_trend_survives_correction": (
                "The paired difference varies monotonically with one covariate "
                "across the fold. A trend costs one test rather than three and "
                "uses the covariate at full resolution, so it is the more "
                "informative finding, and it is still exploratory: it describes "
                "this fold and predicts nothing that has been checked."),
            "a_band_or_trend_favours_p2rank": (
                "At least one partition favours P2Rank. It is reported with the "
                "same prominence as any partition favouring the field, since a "
                "plan that reported only the favourable direction would be the "
                "selection this analysis was designed to avoid."),
        },

        "guards_the_read_must_pass": [
            "this plan's commit is an ancestor of HEAD, checked with git",
            "the covariate artifact's sha256 matches the one recorded here, so "
            "the groups cannot have been redrawn between the plan and the read",
            "the overall mean and win/loss counts reproduce read five to six "
            "decimals",
            "every band's unit count matches the group size recorded here",
            "the bands of a covariate partition its defined units exactly: no "
            "unit in two bands, none in none",
            "each covariate's three band means, weighted by band size, "
            "reconstruct the overall mean",
        ],

        "reads_test_fold": False,
        "note": "this file is the plan. Reading the fold under it is "
                "tools/subgroup_read.py, which refuses to run until this is "
                "committed.",
    }


def _report(d: dict) -> None:
    print(f"plan for read {d['for_test_fold_read_index']}, declared "
          f"{d['status_declared_in_advance']}")
    for c in d["covariates"]:
        print(f"  {c['id']:<14s} cuts {c['cuts']}  groups {c['group_sizes']}")
    m = d["multiplicity"]
    print(f"  {m['n_subgroup_tests']} band tests, Bonferroni level "
          f"{m['corrected_level']}; {d['trend_test']['n_tests']} trend tests, "
          f"level {d['trend_test']['corrected_level']}")
    print(f"  expected false positives if uncorrected: "
          f"{m['expected_false_positives_uncorrected']}")


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
        print(f"FAILED: the plan declares itself "
              f"{have.get('status_declared_in_advance')}; a subgroup analysis "
              f"after an unresolved primary endpoint is exploratory and saying "
              f"otherwise is the failure this file exists to prevent")
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
    need = {"nothing_survives_correction", "a_band_survives_correction",
            "a_trend_survives_correction", "a_band_or_trend_favours_p2rank"}
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
