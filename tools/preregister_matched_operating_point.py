"""Fix the matched-threshold analysis, and what it will be allowed to conclude.

The published F1 margin over P2Rank is +0.0315 [+0.0089, +0.0536], and it is
measured with the two methods binarised by different rules: ours takes the top
9% of residues in each chain, P2Rank is taken at its own pocket assignment. The
margin may therefore be a property of the scores or a property of the threshold,
and the honest way to find out is to re-binarise both by a common rule.

That analysis has an obvious failure mode. Once the held-out numbers are in
hand there are several matched rules to choose between, several statistics to
summarise them with, and a strong temptation to report whichever combination
leaves the margin standing. This file removes the choice by making it first:
the rules, the statistic, the interval, and -- the part that matters -- the
sentence the paper will carry under each outcome, including the outcome where
the margin disappears.

The forecast is written from training-fold information only. P2Rank gains a
measurable amount from being tuned the way we tuned ourselves, and that gain is
already known on the training partition. Subtracting it from the published
margin says in advance whether this analysis is expected to survive, and the
number is close enough to the margin that the answer is genuinely open. Nothing
here reads the test fold; the published margin it quotes was read once already
and is not re-read.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABFIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"
P2TRAIN_OP = ROOT / "results/architecture_sweep/P2RANK_TRAIN_OPERATING_POINT.json"
BOOT = ROOT / "results/official_fold/OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json"
OUT = ROOT / "results/architecture_sweep/PREREGISTERED_MATCHED_OPERATING_POINT.json"

SCHEMA = "geoaudit.prereg_matched_operating_point.v1"
N_BOOT = 10000
SEED = 20260725
CI = 0.95
# What counts as the margin surviving. Fixed here so that a margin which
# shrinks to a hair above zero cannot later be described as intact.
SURVIVES_IF = "the 95% paired interval on the matched difference excludes zero"


def _commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def build() -> dict:
    ours = json.loads(TABFIELD.read_text())["operating_point"]
    p2 = json.loads(P2TRAIN_OP.read_text())
    pub = json.loads(BOOT.read_text())
    f1 = pub["metrics"]["residue_f1"]
    published = f1["paired_vs_baseline"]["table_field"]
    delta = published["delta_point"]
    gain = p2["tuning_is_worth_to_p2rank"]
    forecast = round(delta - gain, 6)

    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": _commit(),
        "question": (
            "whether the F1 margin over P2Rank is a property of the scores or "
            "of the operating point, decided by re-binarising both methods "
            "under a common rule rather than by each method's own convention"
        ),
        "status": "written before the matched read; the read must cite this "
                  "artifact's commit as an ancestor",
        "reads_test_fold": False,

        "what_stays_primary": (
            "the published analysis, in which every detector is scored at its "
            "own operating point. A matched threshold is not more correct than "
            "a predictor's native call -- it answers a different question -- so "
            "it is reported as robustness and does not replace the headline"
        ),

        "our_rule_is_unchanged": {
            "binarisation": ours["rule"],
            "q": ours["q"],
            "selected_on": ours["selected_on"],
            "note": "our side of the comparison is identical in the published "
                    "and matched analyses. Only P2Rank's binarisation moves, "
                    "because only P2Rank was not given a tuned threshold",
        },

        "rules_to_be_read": [
            {
                "id": "A_common_q",
                "statement": "both methods take the top q fraction of "
                             "residues per chain at the same q, the q our "
                             "method already ships",
                "q_ours": ours["q"],
                "q_p2rank": ours["q"],
                "what_it_controls": "removes any difference in how many "
                                    "residues each method is allowed to call, "
                                    "which is the crudest form the objection "
                                    "takes",
                "handicap": "P2Rank did not choose this q and may be worse at "
                            "it than at its own; rule B exists because of that",
            },
            {
                "id": "B_each_tuned_on_train",
                "statement": "each method takes the top q per chain at the q "
                             "that maximises pooled F1 on the 770 training "
                             "receptors, by one grid search applied identically "
                             "to both score sets",
                "q_ours": ours["q"],
                "q_p2rank": p2["p2rank_selected_q"],
                "selection_provenance": str(P2TRAIN_OP.relative_to(ROOT)),
                "what_it_controls": "gives P2Rank the same courtesy ours "
                                    "received -- a threshold fitted to this "
                                    "dataset's training half -- so a surviving "
                                    "margin cannot be attributed to our having "
                                    "tuned and P2Rank not having tuned",
                "this_is_the_decisive_one": True,
            },
            {
                "id": "C_p2rank_oracle_q",
                "statement": "the whole F1-against-q curve for P2Rank on the "
                             "held-out fold, from which its best possible q is "
                             "readable",
                "role": "an upper bound on what any per-chain threshold could "
                        "give P2Rank, reported as an oracle and never as "
                        "P2Rank's score. It is included because a reader who "
                        "distrusts both matched rules can read the bound "
                        "instead, and because publishing the curve makes the "
                        "argmax impossible to hide",
                "is_an_oracle": True,
            },
        ],

        # The grid handed P2Rank the same q it handed us. That is a fact about
        # the labels rather than about either method -- roughly nine per cent of
        # a chain's residues are cryptic-binding, so nine per cent is the
        # calling fraction that maximises F1 for any ranking of them -- and it
        # collapses the two matched rules into one. Recorded here so the read is
        # not later accused of quietly merging them.
        "rules_a_and_b_coincide": {
            "they_do": abs(ours["q"] - p2["p2rank_selected_q"]) < 1e-9,
            "q": p2["p2rank_selected_q"],
            "why": "the same grid search, run separately on each method's "
                   "training scores, returned the same q. The optimal calling "
                   "fraction is set by how many residues are actually positive, "
                   "not by which method ranks them",
            "consequence": "rule A and rule B will report identical numbers. "
                           "Both are still reported, because the reader who "
                           "asked for a common q and the reader who asked for "
                           "separately tuned thresholds are owed an answer each",
        },

        "statistic": {
            "primary": "mean over units of per-unit F1, the difference "
                       "bootstrapped paired over units",
            "why": "it is the statistic the published F1 headline already "
                   "uses, so the matched number is comparable to +"
                   f"{delta:.4f} digit for digit. A different "
                   "functional here would confound the change of threshold "
                   "with a change of summary",
            "secondary": "20% trimmed mean, the functional the ROC-AUC "
                         "preregistration selected. Declared now so that it "
                         "cannot be adopted later on the strength of its answer",
            "n_boot": N_BOOT,
            "seed": SEED,
            "ci_level": CI,
            "resampling": "over units, both arms taking the same draw",
        },

        "published_analysis_being_tested": {
            "artifact": str(BOOT.relative_to(ROOT)),
            "table_field_f1": f1["per_method"]["table_field"]["point"],
            "p2rank_f1": f1["per_method"]["p2rank"]["point"],
            "delta": delta,
            "ci": [published["delta_ci_low"], published["delta_ci_high"]],
            "p_two_sided_bootstrap": published["p_two_sided_bootstrap"],
            "crosses_zero": published["crosses_zero"],
        },

        "forecast": {
            "p2rank_gains_from_tuning_on_train": gain,
            "published_delta": delta,
            "expected_matched_delta_if_the_gain_transfers": forecast,
            "reasoning": (
                "on the training receptors P2Rank's pooled F1 rises by "
                f"{gain:+.4f} when it is given a tuned per-chain top-q instead "
                "of its pocket assignment. If that gain transfers unchanged to "
                f"the held-out fold, the matched margin is about {forecast:+.4f} "
                f"against a published {delta:+.4f}"
            ),
            # Open means the forecast sits inside one interval half-width of
            # zero: close enough that the read could land either side of it.
            "is_the_outcome_open": abs(forecast) < abs(
                published["delta_ci_high"] - delta),
            "honest_note": (
                "the forecast is a subtraction, not a model: the training gain "
                "is pooled F1 and the read is a mean of per-unit F1, and the "
                "two folds are different proteins. It is recorded to show that "
                "this analysis was run with the outcome genuinely in doubt, "
                "not because the answer was already known"
            ),
        },

        "what_will_be_written_under_each_outcome": {
            "survives_if": SURVIVES_IF,
            "if_it_survives": (
                "the F1 advantage is reported as robust to the operating-point "
                "convention, with the matched delta and interval given beside "
                "the native-call delta. No claim is upgraded: the comparison "
                "remains one method against one baseline on one benchmark"
            ),
            "if_it_does_not_survive": (
                "the F1 conclusion is rewritten to say that our F1 advantage "
                "depends on the operating-point convention, and the sentence "
                "claiming an F1 advantage over P2Rank is removed from the "
                "abstract and the results. The ROC-AUC and PR-AUC comparisons "
                "are unaffected, because they are threshold-free, and the "
                "paper will say that too"
            ),
            "if_the_two_matched_rules_disagree": (
                "rule B governs, because it is the one that treats the two "
                "methods identically in procedure rather than identically in "
                "parameter. Rule A is then reported as the stricter reading "
                "and the disagreement is stated rather than resolved silently"
            ),
        },

        "read_protocol": {
            "one_indexed_access": True,
            "next_index": 6,
            "rescores_nothing": (
                "both methods' per-residue scores for the held-out fold are "
                "already committed under results/cryptobench_official/"
                "predictions/. The read re-binarises those numbers and does "
                "not re-run either method, so it consumes no new inference and "
                "can be checked in CI without a JVM"
            ),
            "self_check_before_reading": (
                "re-binarising at each method's native call must reproduce the "
                "committed per-method F1 to four decimals; if it does not, the "
                "reimplementation is wrong and the matched numbers are "
                "meaningless, and the read aborts"
            ),
        },
    }


def _check() -> int:
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}")
        return 1
    d = json.loads(OUT.read_text())
    if d.get("schema") != SCHEMA:
        print(f"FAILED: schema {d.get('schema')}")
        return 1
    if d.get("reads_test_fold") is not False:
        print("FAILED: a preregistration may not read the fold")
        return 1
    ids = [r["id"] for r in d["rules_to_be_read"]]
    if ids != ["A_common_q", "B_each_tuned_on_train", "C_p2rank_oracle_q"]:
        print(f"FAILED: the rules changed: {ids}")
        return 1
    ours = float(json.loads(TABFIELD.read_text())["operating_point"]["q"])
    if abs(d["our_rule_is_unchanged"]["q"] - ours) > 1e-9:
        print("FAILED: the preregistered q is not the shipped q")
        return 1
    p2 = json.loads(P2TRAIN_OP.read_text())
    rb = next(r for r in d["rules_to_be_read"]
              if r["id"] == "B_each_tuned_on_train")
    if abs(rb["q_p2rank"] - p2["p2rank_selected_q"]) > 1e-9:
        print("FAILED: rule B's q for P2Rank is not the one the training "
              "artifact selected")
        return 1
    f = d["forecast"]
    if abs(round(f["published_delta"] - f["p2rank_gains_from_tuning_on_train"],
                 6) - f["expected_matched_delta_if_the_gain_transfers"]) > 1e-6:
        print("FAILED: the forecast is not the difference it claims to be")
        return 1
    for k in ("if_it_survives", "if_it_does_not_survive",
              "if_the_two_matched_rules_disagree"):
        if not d["what_will_be_written_under_each_outcome"].get(k):
            print(f"FAILED: no sentence committed for {k}")
            return 1
    print(f"preregistration OK: rules {', '.join(ids)}")
    print(f"  ours q={d['our_rule_is_unchanged']['q']:.2f} fixed, "
          f"P2Rank q={rb['q_p2rank']:.2f} selected on train")
    print(f"  published delta {f['published_delta']:+.4f}, "
          f"forecast {f['expected_matched_delta_if_the_gain_transfers']:+.4f}")
    print(f"  outcome open: {f['is_the_outcome_open']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        return _check()
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return _check()


if __name__ == "__main__":
    raise SystemExit(main())
