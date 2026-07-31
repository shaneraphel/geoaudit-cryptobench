#!/usr/bin/env python3
"""The plan for read 13: units all three baselines rank at chance and we do not.

Why this plan exists in this form
---------------------------------
The question is "are there chains whose cryptic binding residues the counting
field ranks and P2Rank, pLM-NN and PocketMiner all fail to rank". Asking it of a
192-unit held-out fold is also the easiest way to manufacture a favourable
answer: pick the threshold after seeing the scores, report the corner that
flatters us, and do not mention the other one.

Three commitments make the difference, and all three are fixed here, before the
read:

1. **The rule is not chosen here.** It is copied unchanged from
   ``tools/found_where_baselines_missed.py``, which was written and run on the
   *training* fold and committed before this plan. A recovery is
   ``ours >= 0.80`` with every baseline ``<= 0.55``; a mirror swaps the roles;
   the eight-setting ladder is the one already in that file. The thresholds
   therefore cannot have been tuned on the fold this plan reads, and the commit
   history shows it.
2. **The mirror is reported whatever it says.** For every unit in our favour the
   read reports the units where all baselines rank the site and we do not. A
   count without its mirror is a selection.
3. **The whole ladder is reported.** Not the row with the largest number.

Status: exploratory, and not by preference. The fold carries twelve indexed reads
before this plan. Nothing read after that is confirmatory whichever way it
points, and fixing the status here is what stops a favourable outcome being
promoted afterwards.

What is read, and what is not recomputed
----------------------------------------
Our per-unit ROC-AUC and P2Rank's are taken from the frozen telemetry rather than
recomputed, so this read cannot disagree with the numbers already published from
that file. pLM-NN's and PocketMiner's per-unit ROC-AUCs do not exist in any
artifact -- their reads reported only aggregates -- so they are computed here from
their own frozen per-residue scores and the official labels, using
``pocket_bench.metrics.roc_auc``, the same function the telemetry used, over the
same sorted-resseq universe.

That is why this is an indexed read. No model is re-run and no threshold is
re-fitted, but labels are opened and a new statistic is formed over the held-out
units, and a ledger that counted only re-runs would make an unlimited number of
such statements free.

The exclusion, fixed in advance
-------------------------------
A unit is set aside, and reported separately rather than counted, when the four
methods cannot be shown to have been given the same residues: PocketMiner's
featurisation not verified against the authors' own tensor, or residues dropped
as conformer copies, or resseq collisions from insertion codes. On the training
fold this rule removed ``4m7p_A``, which was the largest disagreement of all --
0.918 against 0.439 and 0.214 -- on a deposit carrying twenty alternate
conformers. A win on a chain the methods parse differently is a win about
parsing. The rule is stated here so that applying it on this fold is not a
decision made after seeing which units it removes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.preregistered_recovery.v1"
OUT = ROOT / "results/architecture_sweep/PREREGISTERED_RECOVERY.json"
READ_INDEX = 13

TRAIN_TOOL = ROOT / "tools/found_where_baselines_missed.py"
TRAIN_ARTIFACT = ROOT / "results/architecture_sweep/RECOVERED_UNITS_TRAIN.json"
TELEMETRY = ROOT / "results/cryptobench_official/TELEMETRY.json"
PLMNN = ROOT / "results/baselines/PLMNN_SCORES.json"
POCKETMINER = ROOT / "results/baselines/POCKETMINER_SCORES.json"
LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"


def _git(*a: str) -> str:
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build() -> dict:
    train = json.loads(TRAIN_ARTIFACT.read_text())
    led = json.loads(LEDGER.read_text())
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "reads_test_fold": False,
        "for_test_fold_read_index": READ_INDEX,
        "written_before_the_read": True,
        "code_sha256": _sha(Path(__file__)),
        "head_when_written": _git("rev-parse", "HEAD"),
        "question": "on the 192 single-chain units of the official test fold, "
                    "how many chains does the counting field rank the cryptic "
                    "binding residues well on while P2Rank, pLM-NN and "
                    "PocketMiner all rank them at chance, and how many are the "
                    "other way round",
        "status_declared_in_advance": "exploratory",
        "why_exploratory_and_not_confirmatory": (
            f"the fold carries {led['n_indexed_reads']} indexed reads before "
            f"this plan was written. Nothing read after that is confirmatory, "
            f"whichever direction it points, and fixing the status here stops a "
            f"favourable outcome being promoted afterwards"),
        "the_rule": {
            "recovery": "ours >= 0.80 and p2rank <= 0.55 and plmnn <= 0.55 and "
                        "pocketminer <= 0.55",
            "mirror": "p2rank >= 0.80 and plmnn >= 0.80 and pocketminer >= 0.80 "
                      "and ours <= 0.55",
            "statistic": "ROC-AUC within a unit over that unit's evaluation "
                         "universe, chance 0.50",
            "why_two_thresholds": "a unit where every method scores 0.7 is not a "
                                  "disagreement and a single cut would call it "
                                  "one",
            "ladder": [[r["found_at_or_above"], r["missed_at_or_below"]]
                       for r in train["threshold_ladder"]],
            "where_the_rule_comes_from": {
                "tool": str(TRAIN_TOOL.relative_to(ROOT)),
                "tool_sha256": _sha(TRAIN_TOOL),
                "artifact": str(TRAIN_ARTIFACT.relative_to(ROOT)),
                "artifact_sha256": _sha(TRAIN_ARTIFACT),
                "fold": "training",
                "why_this_matters": "the thresholds were fixed and published on "
                                    "the training fold before this plan, so "
                                    "they cannot have been tuned on the fold "
                                    "this read opens, and the commit history "
                                    "shows the order",
                "training_fold_outcome": {
                    "n_units": train["n_units_compared"],
                    "n_recovered": train["n_recovered"],
                    "n_mirror": train["n_mirror"],
                    "baselines_available_there": ["p2rank", "pocketminer"],
                    "plmnn_absent_there": "never run on the training fold",
                },
            },
        },
        "where_each_number_comes_from": {
            "table_field": "the frozen per-structure telemetry, field "
                           "residue_auc; not recomputed, so this read cannot "
                           "disagree with numbers already published from it",
            "p2rank": "the same frozen telemetry",
            "plmnn": "computed here from results/baselines/PLMNN_SCORES.json and "
                     "the official labels, with pocket_bench.metrics.roc_auc "
                     "over the sorted-resseq universe, because no artifact "
                     "stores its per-unit numbers",
            "pocketminer": "computed here from "
                           "results/baselines/POCKETMINER_SCORES.json the same "
                           "way, for the same reason",
            "telemetry_sha256": _sha(TELEMETRY),
            "plmnn_scores_sha256": _sha(PLMNN),
            "pocketminer_scores_sha256": _sha(POCKETMINER),
        },
        "exclusion_fixed_in_advance": {
            "rule": "a unit is reported separately and not counted when the "
                    "methods cannot be shown to have been given the same "
                    "residues: PocketMiner's featurisation unverified against "
                    "the authors' tensor, or residues dropped as conformer "
                    "copies, or resseq collisions from insertion codes",
            "why": "a win on a chain the methods parse differently is a win "
                   "about parsing",
            "precedent": "on the training fold this removed 4m7p_A, which was "
                         "the largest disagreement of all at 0.918 against "
                         "0.439 and 0.214, on a deposit carrying twenty "
                         "alternate conformers and 60,040 ATOM lines for 3,002 "
                         "atoms",
        },
        "guards_the_read_must_pass": [
            "this plan's commit is an ancestor of HEAD, checked with git",
            "the telemetry, the pLM-NN scores and the PocketMiner scores still "
            "hash to what is recorded here, so no input can have been "
            "regenerated between the plan and the read",
            "the training-fold tool and artifact still hash to what is recorded "
            "here, so the rule cannot have been edited after the plan",
            "our own per-unit numbers are read from the telemetry and not "
            "recomputed, and the read asserts the unit set matches",
            "the mirror count is computed and reported whatever it is",
            "the full ladder is computed and reported, not one row of it",
        ],
        "decision_rules": {
            "if_recoveries_exceed_mirrors_at_every_ladder_setting":
                "it is reported as the strongest per-unit evidence in the paper "
                "that the counting field is not a weaker copy of the baselines, "
                "still labelled exploratory, and stated together with the mirror "
                "counts and the number of cryptic residues in each named unit. "
                "It does not license the word state-of-the-art and it is not a "
                "claim that a pocket exists which the baselines cannot see",
            "if_the_two_counts_are_comparable":
                "it is reported as a null: the three methods disagree in both "
                "directions about equally, which is what similar average "
                "accuracy predicts, and no case study is drawn from it",
            "if_mirrors_exceed_recoveries":
                "it is reported in the paper in those words, in the same "
                "section, at the same length",
        },
        "what_will_be_written_under_each_outcome": {
            "recoveries_exceed_mirrors": (
                "On the 192 units of the official test fold there are chains "
                "whose cryptic binding residues the counting field ranks at "
                "ROC-AUC 0.80 or better while P2Rank, pLM-NN and PocketMiner all "
                "rank them at 0.55 or worse, chance being 0.50, and fewer chains "
                "the other way round. The rule was fixed on the training fold "
                "before this read. The comparison is exploratory: the fold had "
                "been read twelve times before it was planned."),
            "comparable": (
                "On the 192 units of the official test fold, the chains where "
                "the counting field ranks the cryptic residues and all three "
                "baselines do not are about as numerous as the chains where the "
                "reverse holds. Four methods of similar average accuracy "
                "disagree in both directions, and on this fold at this sample "
                "size the disagreement is symmetric."),
            "mirrors_exceed_recoveries": (
                "On the 192 units of the official test fold there are more "
                "chains whose cryptic binding residues all three baselines rank "
                "and the counting field does not than chains the other way "
                "round. The rule was fixed on the training fold before this "
                "read, and it is reported here because it was."),
        },
        "what_this_cannot_show": (
            "ROC-AUC within a unit is a statement about the ranking of that "
            "unit's residues. Nothing here is a claim that a pocket exists which "
            "the baselines cannot see, that any site is druggable, or that any "
            "of it is clinical"),
        "note": "this file is the plan. Reading the fold under it is "
                "tools/recovery_read.py, which refuses to run until this is "
                "committed.",
    }


def _report(d: dict) -> None:
    r = d["the_rule"]
    print(f"read {d['for_test_fold_read_index']} "
          f"({d['status_declared_in_advance']})")
    print(f"  recovery: {r['recovery']}")
    print(f"  mirror:   {r['mirror']}")
    t = r["where_the_rule_comes_from"]
    print(f"  rule fixed on the {t['fold']} fold: "
          f"{t['training_fold_outcome']['n_recovered']} recovered against "
          f"{t['training_fold_outcome']['n_mirror']} mirror on "
          f"{t['training_fold_outcome']['n_units']} units")
    print(f"  tool {t['tool_sha256'][:12]}  artifact {t['artifact_sha256'][:12]}")
    print(f"  ladder: {len(r['ladder'])} settings")


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        if not OUT.is_file():
            print(f"MISSING {OUT.relative_to(ROOT)}")
            return 1
        d = json.loads(OUT.read_text())
        if d.get("schema") != SCHEMA:
            print(f"FAILED: schema {d.get('schema')}")
            return 1
        if d.get("for_test_fold_read_index") != READ_INDEX:
            print(f"FAILED: the plan does not index itself as read {READ_INDEX}")
            return 1
        for key, path in (("tool_sha256", TRAIN_TOOL),
                          ("artifact_sha256", TRAIN_ARTIFACT)):
            have = d["the_rule"]["where_the_rule_comes_from"][key]
            if have != _sha(path):
                print(f"FAILED: {path.name} changed since the plan was written; "
                      f"the rule is no longer the one that was fixed")
                return 1
        _report(d)
        print(f"OK {OUT.relative_to(ROOT)}")
        return 0
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(d, indent=2, allow_nan=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}\n")
    _report(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
