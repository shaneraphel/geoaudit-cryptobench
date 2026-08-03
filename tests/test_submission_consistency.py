"""The repository may not contradict the paper that was submitted.

The PSB submission names commit ``5d1726a`` as its frozen snapshot and tells the
reader to audit the chronology from this repository's commit graph. That makes
the repository part of the paper's evidence, and a number here that disagrees
with a number there is not a difference between two drafts -- it is a paper
citing evidence that does not support it.

Nothing checked this until 2026-08-03. The submission's LaTeX source is not in
the repository (the 15-page paper was condensed from the long-form manuscript
and lives elsewhere), so there was no file to diff and the comparison was never
made. ``tools/check_submission_consistency.py`` makes it, holding each of the
paper's printed values against the macro that generated it.

What it found: all 47 scientific claims agree, and both process claims do not.
The paper says "12 indexed test reads and 12 scored architectures"; the ledger
said 13 and 12 at the submission commit and says 13 and 20 now. The first was
already wrong at freeze, the second went stale afterwards, and both understate
the multiplicity the paper's own margins should be read against -- an error in
the direction that makes us look more careful, which is the direction least
likely to be questioned.

The violations planted below: a macro moved away from the value the paper
printed, which must be caught; and the supplement stripped of its disclosure,
which must also be caught, because a checker that reports a divergence into a
JSON file nobody reads is not a disclosure.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_submission_consistency as chk  # noqa: E402

SUPPLEMENT = ROOT / "paper/supplement_beyond_submission.tex"
README = ROOT / "README.md"


@pytest.fixture(scope="module")
def doc() -> dict:
    if not chk.MACROS.is_file():
        pytest.skip("frozen_numbers.tex not built")
    return chk.build()


class TestScientificClaims:
    def test_every_printed_number_is_still_reproduced(self, doc):
        bad = doc["scientific_claims_disagreeing"]
        assert not bad, (
            "the submitted paper prints these values and this repository no "
            "longer produces them: "
            + "; ".join(f"{r['section']} paper={r['printed_in_submission']} "
                        f"repo=\\{r['macro']}={r['repository_value']}"
                        for r in bad))

    def test_no_claim_became_uncheckable(self, doc):
        gone = doc["scientific_claims_unverifiable"]
        assert not gone, (
            "these macros generated numbers the paper printed and no longer "
            "exist, so the claim can no longer be checked at all: "
            + ", ".join(f"\\{r['macro']}" for r in gone))

    def test_the_check_covers_a_useful_number_of_claims(self, doc):
        assert doc["n_claims_checked"] >= 40, (
            "a consistency check over a handful of numbers would pass while "
            "the paper and the repository disagreed everywhere else")

    def test_a_moved_macro_is_caught(self, monkeypatch, doc):
        """Planted: the headline official-fold ROC-AUC changes."""
        real = chk._macros()
        planted = dict(real)
        planted["TabAuc"] = "0.888"
        monkeypatch.setattr(chk, "_macros", lambda: planted)
        got = chk.build()
        assert got["n_disagreeing"] == 1
        row = got["scientific_claims_disagreeing"][0]
        assert row["macro"] == "TabAuc"
        assert row["printed_in_submission"] == "0.799"
        assert "not what this repository now produces" in got["verdict"]


class TestProcessClaims:
    def test_both_divergences_are_reported(self, doc):
        """These are known-wrong and must stay visible, not be quietly fixed.

        The paper cannot be edited retroactively, so the correct state of this
        test is that it *records* two disagreements rather than asserting none.
        If someone makes them agree by editing CLAIMS instead of by the paper
        being corrected, that is the failure this guards.
        """
        proc = {p["phrase"]: p for p in doc["process_claims"]}
        assert set(proc) == {"12 indexed test reads", "12 scored architectures"}
        reads = proc["12 indexed test reads"]
        assert reads["ledger_at_submission_commit"] == 13, (
            "the ledger at the submission commit is fixed history; if this "
            "changed, either the commit hash or the ledger's meaning did")
        arch = proc["12 scored architectures"]
        assert arch["agreed_at_submission"] is True
        assert arch["ledger_today"] > 12

    def test_the_disclosure_is_where_a_reader_will_see_it(self):
        """A divergence recorded only in JSON is not disclosed.

        The submission points readers at this repository, so the front page and
        the supplement both have to carry it. This is the half of the fix that
        the checker cannot do for itself.
        """
        for path, what in ((README, "the repository front page"),
                           (SUPPLEMENT, "the supplement")):
            if not path.is_file():
                pytest.fail(f"{what} is missing: {path.name}")
            text = path.read_text()
            assert "12 indexed test reads" in text, (
                f"{what} does not quote the paper's wrong read count, so a "
                f"reader cannot match the correction to the sentence it fixes")
            assert "5d1726a" in text, (
                f"{what} does not name the submission commit")

    def test_the_readme_says_which_document_is_authoritative(self):
        text = README.read_text()
        assert "authoritative for every claim it makes" in text, (
            "a reader who finds a stronger result here than in the paper must "
            "be told which one governs")


class TestBeyondSubmission:
    def test_results_absent_from_the_paper_are_enumerated(self, doc):
        beyond = doc["results_beyond_the_submission"]
        assert beyond, "nothing listed"
        for b in beyond:
            assert Path(ROOT / b["artifact"]).exists(), (
                f"{b['result']} points at {b['artifact']}, which does not "
                f"exist; a supplement citing a missing artifact is worse than "
                f"no supplement")
            assert b["why_it_matters"]

    def test_the_main_manuscript_defers_to_the_supplement(self):
        """The one-pass read moved out of the main text; it must not creep back.

        The main manuscript keeps a short pointer at sec:grand. If the full
        section reappears there, a reader meets the parity claim and the
        paper's deficit claim in one document with nothing saying which was
        submitted.
        """
        main = ROOT / "paper/MAIN_CRYPTOBENCH_GEOAUDIT.tex"
        if not main.is_file():
            pytest.skip("manuscript not present")
        text = main.read_text()
        assert "supplement\\_beyond\\_submission" in text, (
            "the main manuscript must point at the supplement")
        assert "\\label{fig:grand-standing}" not in text, (
            "the one-pass figures belong to the supplement; their presence "
            "here means the section was moved back")
        assert "not part of the submitted paper" in text
