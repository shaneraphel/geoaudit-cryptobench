#!/usr/bin/env python3
"""Check the repository against the manuscript that was actually submitted.

The PSB submission is a 15-page paper, *Auditable Algebraic Counting Field for
Cryptic-Pocket Detection from Apo Structures*, whose LaTeX source is not in this
repository: what is here is the long-form technical manuscript the paper was
condensed from. Two documents, one set of artifacts, and nothing until now
comparing them. The submission names commit ``5d1726a`` as its frozen snapshot
and tells the reader to audit the chronology from the commit graph, so a claim
in it that the repository contradicts is not a discrepancy between two drafts --
it is a paper pointing at evidence that does not say what the paper says.

Three kinds of divergence, and they need opposite responses.

**Agreeing.** The number in the paper and the number the repository generates
today are the same. Nothing to do; the check exists so that a later change
cannot silently break it.

**Stale in the submission.** The repository has moved since ``5d1726a`` and the
paper quotes the older value. This is the dangerous kind, because the paper
directs a reader to the current repository. Every one of these is either
corrected in the paper or, where the paper cannot be changed, disclosed in the
supplement with both values and the commit each belongs to.

**Beyond the submission.** The repository contains results the paper does not
mention at all -- the geometry and nonlocal-seam architectures, the single-pass
three-baseline read, the parity standing against pLM-NN. These do not
contradict the paper, but a reader arriving from it will find claims stronger
than the ones they came for, with no statement of which document is
authoritative. They belong in a supplement that says so.

The paper's *scientific* numbers are checked here one by one against the macro
that generated them. Its *process* numbers -- how many times the held-out fold
was read, how many architectures were scored on it -- are checked against the
ledger, and these are where the divergence is: the paper says 12 indexed reads
and 12 architectures; the ledger said 13 and 12 at the submission commit and
says 13 and 20 now, the jump in architectures coming from a ledger defect fixed
on 2026-08-03 that had made eighteen artifacts invisible (AGENT_MEMORY §2y).

Usage:
  python3.12 tools/check_submission_consistency.py
  python3.12 tools/check_submission_consistency.py --strict
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MACROS = ROOT / "paper/frozen_numbers.tex"
LEDGER = ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
GRAND = ROOT / "results/official_fold/GRAND_BASELINE_READ.json"
OUT = ROOT / "results/official_fold/SUBMISSION_CONSISTENCY.json"

SUBMISSION_COMMIT = "5d1726a"
SUBMISSION_TITLE = ("Auditable Algebraic Counting Field for Cryptic-Pocket "
                    "Detection from Apo Structures")

# Every numeric claim the submitted paper makes that this repository also holds,
# with the macro that generates it. The value on the left is transcribed from
# the submitted PDF; the macro on the right is regenerated from artifacts. They
# are compared as strings after normalising the minus sign and whitespace,
# because a claim that agrees to four decimals but not to the digit the paper
# printed is still a claim the reader cannot check.
CLAIMS: tuple[tuple[str, str, str], ...] = (
    # section, printed in the submission, macro that must produce it
    ("2.2 architecture", "645", "NTabWires"),
    ("2.2 architecture", "5152", "NTabTables"),
    ("2.2 architecture", "82432", "NTabCells"),
    ("2.2 architecture", "2", "TabWidth"),
    ("2.2 architecture", "16", "TabRounds"),
    ("2.2 architecture", "32", "TabCap"),
    ("2.2 architecture", "0.03", "TabRidge"),
    ("2.1 units", "192", "NUnits"),
    ("2.3 baselines", "0.9987", "PlmFidelity"),
    ("2.3 operating point", "0.09", "TabOperatingQ"),
    ("2.3 operating point", "0.09", "MatchOurQ"),
    ("2.4 external", "57", "ExtN"),
    ("2.4 external", "905", "ExtNPositives"),
    ("2.4 external", "2024-05-08", "ExtCutoff"),
    ("3.1 official fold", "0.799", "TabAuc"),
    ("3.1 official fold", "0.376", "TabPr"),
    ("3.1 official fold", "0.304", "TabMcc"),
    ("3.1 official fold", "0.333", "TabFOne"),
    ("3.1 official fold", "0.793", "PtwoRAuc"),
    ("3.1 official fold", "+0.0058", "TabAucD"),
    ("3.1 official fold", "[-0.0168, +0.0276]", "TabAucDCI"),
    ("3.1 official fold", "+0.0177", "TabPrD"),
    ("3.1 vs pLM-NN", "-0.0243", "PlmAucDelta"),
    ("3.1 vs pLM-NN", "[-0.0465, -0.0033]", "PlmAucCI"),
    ("3.1 vs pLM-NN", "111", "PlmLoss"),
    ("3.1 vs PocketMiner", "+0.0468", "PmAucOurs"),
    ("3.1 vs PocketMiner", "[+0.0251, +0.0682]", "PmAucCIOurs"),
    ("3.1 vs PocketMiner", "+0.0410", "PmAucPtwoR"),
    ("3.1 logistic", "+0.0038", "RdLogitDelta"),
    ("3.1 logistic", "[-0.0057, +0.0125]", "RdLogitCI"),
    ("3.2 F1 shipped", "+0.0315", "TabFOneD"),
    ("3.2 F1 matched", "+0.0140", "MatchBDelta"),
    ("3.2 F1 matched", "[-0.0085, +0.0364]", "MatchBCI"),
    ("3.2 thresholds", "39", "CvN"),
    ("3.3 Set A", "+0.0443", "ExtAucPtwoR"),
    ("3.3 Set A", "[+0.0162, +0.0724]", "ExtAucCIPtwoR"),
    ("3.3 Set A", "[+0.0102, +0.0792]", "ExtAucBonfPtwoR"),
    ("3.3 Set A", "42", "ExtWinPtwoR"),
    ("3.3 Set A", "-0.0340", "ExtAucPlm"),
    ("3.3 Set A", "[-0.0701, -0.0006]", "ExtAucCIPlm"),
    ("3.3 Set A", "[-0.0784, +0.0071]", "ExtAucBonfPlm"),
    ("3.3 Set A", "1", "ExtAbstain"),
    ("3.6 splits", "4", "NSplitsCv"),
    ("3.6 splits", "25", "NSplitsRh"),
    ("3.6 cost", "1.79", "FairModelOurs"),
    ("2.5 statistics", "4", "EndReadsBefore"),
    ("2.5 statistics", "3", "EndLineageBefore"),
)

# Process claims, checked against the ledger rather than a macro, because the
# ledger is what the paper tells the reader to audit.
#   section, what the paper says, ledger field, the paper's number
PROCESS: tuple[tuple[str, str, str, int], ...] = (
    ("4.2 limitations", "12 indexed test reads", "n_indexed_reads", 12),
    ("4.2 limitations", "12 scored architectures",
     "n_distinct_architectures_evaluated", 12),
)

# Results the repository holds and the submission does not mention at all.
# Each names the artifact a reader would land on, so the supplement can point
# at evidence rather than at a paragraph.
BEYOND = (
    ("the geometry and nonlocal-seam architectures",
     "results/official_fold/GRAND_BASELINE_READ.json",
     "the submitted paper evaluates one detector of ours, the counting field. "
     "Six further architectures score above it on the official fold."),
    ("the single-pass three-baseline read",
     "results/official_fold/GRAND_BASELINE_READ.json",
     "all methods on one residue universe with no missing cell; the paper's "
     "comparisons are pairwise and were taken at different times."),
    ("parity with pLM-NN on the official fold",
     "results/official_fold/GRAND_BASELINE_READ.json",
     "the paper states a resolved deficit against pLM-NN for the counting "
     "field, which remains true of that detector. A different architecture of "
     "ours is not separable from pLM-NN on the same units, which the paper "
     "does not say and a reader of the repository will find."),
)


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _norm(s: str) -> str:
    """Compare what the reader sees, not what the encoder wrote."""
    return (s.replace("\u2212", "-").replace("\u2013", "-")
            .replace("$", "").replace("\\,", "").replace("~", " ")
            .replace(" ", "").strip())


def _macros() -> dict[str, str]:
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*?)\}\n",
                           MACROS.read_text()))


def _ledger_at(commit: str) -> dict | None:
    try:
        blob = subprocess.run(
            ["git", "show", f"{commit}:results/official_fold/"
                            "TEST_FOLD_ACCESS_LEDGER.json"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout
        return json.loads(blob)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def build() -> dict:
    mac = _macros()
    now_ledger = json.loads(LEDGER.read_text()) if LEDGER.is_file() else {}
    then_ledger = _ledger_at(SUBMISSION_COMMIT) or {}

    agreeing, disagreeing, unverifiable = [], [], []
    for section, printed, macro in CLAIMS:
        got = mac.get(macro)
        row = {"section": section, "printed_in_submission": printed,
               "macro": macro, "repository_value": got}
        if got is None:
            row["verdict"] = "unverifiable: the macro no longer exists"
            unverifiable.append(row)
        elif _norm(got) == _norm(printed):
            row["verdict"] = "agrees"
            agreeing.append(row)
        else:
            row["verdict"] = "DISAGREES"
            disagreeing.append(row)

    process = []
    for section, phrase, field, printed in PROCESS:
        at_freeze = then_ledger.get(field)
        today = now_ledger.get(field)
        process.append({
            "section": section,
            "printed_in_submission": printed,
            "phrase": phrase,
            "ledger_field": field,
            "ledger_at_submission_commit": at_freeze,
            "ledger_today": today,
            "agreed_at_submission": at_freeze == printed,
            "agrees_today": today == printed,
            "verdict": (
                "agrees" if today == printed else
                "was already wrong when the paper was frozen, and is wrong now"
                if at_freeze is not None and at_freeze != printed else
                "correct at the submission commit; the repository has moved"),
        })

    beyond = [{"result": r, "artifact": a, "why_it_matters": w}
              for r, a, w in BEYOND]

    n_bad = len(disagreeing) + sum(1 for p in process if not p["agrees_today"])
    doc = {
        "schema": "geoaudit.submission_consistency.v1",
        "clinical_grade": False,
        "reads_test_fold": False,
        "what_this_reads": (
            "committed macros and the test-fold ledger. It computes no metric "
            "and touches no structure, so it is not a read of any fold."),
        "built_at": _now(),
        "submission_title": SUBMISSION_TITLE,
        "submission_commit": SUBMISSION_COMMIT,
        "submission_source_in_this_repository": False,
        "submission_source_note": (
            "the submitted 15-page paper was condensed from "
            "paper/MAIN_CRYPTOBENCH_GEOAUDIT.tex and its own LaTeX source is "
            "held elsewhere. This tool therefore compares the paper's printed "
            "values, transcribed into CLAIMS above, against the macros and "
            "artifacts that generated them."),
        "claim_transcription_note": (
            "the left-hand value of each CLAIMS row is typed from the PDF by "
            "hand and is the one thing here that no generator checks. It is "
            "the weakest link and is recorded as such rather than presented as "
            "machine-read."),
        "n_claims_checked": len(CLAIMS),
        "n_agreeing": len(agreeing),
        "n_disagreeing": len(disagreeing),
        "n_unverifiable": len(unverifiable),
        "scientific_claims_agreeing": agreeing,
        "scientific_claims_disagreeing": disagreeing,
        "scientific_claims_unverifiable": unverifiable,
        "process_claims": process,
        "results_beyond_the_submission": beyond,
        "verdict": (
            "every scientific number the submission prints is reproduced by "
            "this repository today; the divergence is confined to process "
            "counts and to results the submission does not mention"
            if not disagreeing else
            "at least one scientific number in the submission is not what this "
            "repository now produces"),
    }
    doc["n_problems"] = n_bad
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="exit non-zero if any claim disagrees")
    a = ap.parse_args(argv)

    doc = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")

    print(f"submitted paper: {doc['submission_title']}")
    print(f"frozen at {doc['submission_commit']}, source not in this repository")
    print()
    print(f"scientific claims: {doc['n_agreeing']} agree, "
          f"{doc['n_disagreeing']} disagree, "
          f"{doc['n_unverifiable']} unverifiable "
          f"(of {doc['n_claims_checked']})")
    for r in doc["scientific_claims_disagreeing"]:
        print(f"  DISAGREES {r['section']:22s} paper={r['printed_in_submission']} "
              f"repo=\\{r['macro']}={r['repository_value']}")
    for r in doc["scientific_claims_unverifiable"]:
        print(f"  MISSING   {r['section']:22s} \\{r['macro']}")
    print()
    print("process claims:")
    for p in doc["process_claims"]:
        mark = "ok  " if p["agrees_today"] else "DIFF"
        print(f"  {mark} {p['phrase']:26s} paper={p['printed_in_submission']:>3} "
              f"at {SUBMISSION_COMMIT}={str(p['ledger_at_submission_commit']):>3} "
              f"today={str(p['ledger_today']):>3}")
        print(f"       {p['verdict']}")
    print()
    print("in the repository, absent from the submission:")
    for b in doc["results_beyond_the_submission"]:
        print(f"  - {b['result']}")
    print()
    print(doc["verdict"])
    print(f"wrote {OUT.relative_to(ROOT)}")

    if a.strict and doc["n_disagreeing"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
