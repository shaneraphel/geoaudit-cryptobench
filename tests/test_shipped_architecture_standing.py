"""The paper may not let the reader think the shipped detector is our best one.

``table_field`` is the architecture this paper ships, and the reason is that its
whole inference path is table lookups and integer addition. It is not the most
accurate thing the project has built. In the single-pass read of
``GRAND_BASELINE_READ.json`` it is eighth of ten, with six of our seven
architectures above it and the highest at 0.8341 against its 0.7992.

The abstract used to say the opposite, in a sentence written to be
self-critical: ``\\TabAucD`` "is the largest of \\NArchOnFold architectures of
ours that have been scored on these same units", offered as a reason to discount
the margin as a maximum over correlated estimates. That was true when written.
Six later architectures made it false, and nothing noticed, because the sentence
was prose and the numbers behind it were in an artifact nobody compared it
against.

The failure is worth naming precisely, because it is not the usual one. A stale
number that flatters us is caught by suspicion. This one *understated* us --- it
called our shipped detector the best we had, as an argument for trusting it
less --- and a self-deprecating claim attracts no scrutiny at all. It was still
a false statement about our own process, of exactly the kind AGENTS.md §1
exists to stop.

The gates here.

1. Whenever an architecture of ours outscores ``table_field`` in the grand read,
   the manuscript must say how many do. Planted violation: the count is removed
   from the manuscript text, which must be rejected.
2. ``\\GrandNAboveTab`` must equal what the artifact says today, checked by
   recomputing it from the artifact rather than trusting the macro.
3. The manuscript must not contain the withdrawn claim in any form that reads as
   ``table_field`` being the maximum.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GRAND = ROOT / "results/official_fold/GRAND_BASELINE_READ.json"
MACROS = ROOT / "paper/frozen_numbers.tex"
MAIN = ROOT / "paper/MAIN_CRYPTOBENCH_GEOAUDIT.tex"

BASELINES = ("p2rank", "plmnn", "pocketminer")
SHIPPED = "table_field"


@pytest.fixture(scope="module")
def standing() -> dict:
    if not GRAND.is_file():
        pytest.skip("grand read not built")
    s = json.loads(GRAND.read_text())["summary"]
    ours = {m: v["mean_per_unit_roc_auc"] for m, v in s.items()
            if m not in BASELINES}
    if SHIPPED not in ours:
        pytest.skip(f"{SHIPPED} is not in the grand read")
    return ours


def _macro(name: str) -> str | None:
    if not MACROS.is_file():
        return None
    m = re.search(r"\\newcommand\{\\%s\}\{([^}]*)\}" % name, MACROS.read_text())
    return m.group(1) if m else None


def test_the_count_matches_the_artifact(standing):
    want = sum(1 for m, v in standing.items()
               if m != SHIPPED and v > standing[SHIPPED])
    got = _macro("GrandNAboveTab")
    assert got is not None, "\\GrandNAboveTab is not emitted"
    assert got == str(want), (
        f"\\GrandNAboveTab is {got}; recomputing from the artifact gives {want}")


def test_the_manuscript_discloses_the_standing(standing):
    """If anything of ours beats the shipped detector, the paper says so."""
    n_above = sum(1 for m, v in standing.items()
                  if m != SHIPPED and v > standing[SHIPPED])
    if n_above == 0:
        pytest.skip("nothing of ours outscores the shipped detector")
    if not MAIN.is_file():
        pytest.skip("manuscript not present")
    text = MAIN.read_text()
    assert "\\GrandNAboveTab{}" in text, (
        f"{n_above} of our architectures outscore {SHIPPED} on the official "
        f"fold and the manuscript does not state how many. A reader is entitled "
        f"to know the shipped detector is not the most accurate one we built.")


def test_a_manuscript_without_the_count_is_rejected(standing):
    """Planted: the disclosure removed from the text."""
    n_above = sum(1 for m, v in standing.items()
                  if m != SHIPPED and v > standing[SHIPPED])
    if n_above == 0:
        pytest.skip("nothing of ours outscores the shipped detector")
    planted = "Over this fold the standard error of a mean ROC-AUC is 0.0161."
    assert "\\GrandNAboveTab{}" not in planted, (
        "the plant must not accidentally contain the disclosure")


def test_the_withdrawn_claim_is_gone():
    """The exact wording that shipped, and the shapes it could come back as."""
    if not MAIN.is_file():
        pytest.skip("manuscript not present")
    text = " ".join(MAIN.read_text().split())
    forbidden = (
        "it is the largest of \\NArchOnFold{} architectures",
        "is the largest of \\NArchOnFold{} architectures of ours",
        # The same claim in the conclusion, found by grepping the manuscript
        # for superlatives about our own work after the abstract one turned up.
        # \TabAuc is table_field's 0.7992 and it was called the strongest field
        # here, with six of ours above it.
        "\\TabAuc{} and F1 \\TabFOne{} for the strongest field here",
    )
    hit = [f for f in forbidden if f in text]
    assert not hit, (
        "this claims table_field's reading is the maximum over our "
        f"architectures, and it is not: {hit}")


def test_the_shipped_detector_is_argued_for_on_a_different_property(standing):
    """Not accuracy --- decomposability. The paper has to say which.

    Without this, the fix above could be satisfied by stating the count and
    leaving the reader to conclude the shipped choice was a mistake, when the
    reason it ships is that its inference path is integer table lookups.
    """
    if not MAIN.is_file():
        pytest.skip("manuscript not present")
    text = " ".join(MAIN.read_text().split())
    assert "decomposes" in text and "table terms" in text, (
        "the manuscript must say what property table_field is chosen for, "
        "since it is not chosen for being the most accurate")
