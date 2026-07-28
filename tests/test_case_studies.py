"""The case studies must illustrate the method, not the author.

Four structures are pulled out of a 192-unit fold. Four out of 192 chosen by
eye would be an argument dressed as evidence, so the choice is made by a stated
rule and these tests hold it to that rule: the categories are what they claim,
the picks are the extremes their rule names, and the F1 values re-derive from
the committed labels and raw output rather than being read from the frozen
telemetry.

The fourth case exists for a reason worth protecting. Most of this fold is
missed by both methods, and a case study showing only the three interesting
outcomes would leave a reader with the impression that the benchmark is solved
and the question is who wins. The population counts are asserted here so that
the failure case cannot quietly be dropped.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import select_case_studies as cs  # noqa: E402

ARTIFACT = ROOT / "results/official_fold/CASE_STUDIES.json"


class TestTheSelectionRule(unittest.TestCase):
    """Known-answer cases for the arithmetic the rule is built on."""

    def test_f1_is_zero_when_nothing_is_called(self) -> None:
        self.assertEqual(cs.f1_of(set(), {1, 2, 3}), 0.0)

    def test_f1_is_zero_when_nothing_called_is_right(self) -> None:
        self.assertEqual(cs.f1_of({7, 8}, {1, 2, 3}), 0.0)

    def test_f1_is_one_on_an_exact_hit(self) -> None:
        self.assertEqual(cs.f1_of({1, 2, 3}, {1, 2, 3}), 1.0)

    def test_f1_is_the_harmonic_mean_not_the_accuracy(self) -> None:
        # Two of four called are right, and they are two of two true: precision
        # 0.5, recall 1.0, F1 2/3. An accuracy-shaped mistake would give 0.75.
        self.assertAlmostEqual(cs.f1_of({1, 2, 8, 9}, {1, 2}), 2 / 3, places=12)

    def test_the_outcome_labels_follow_the_threshold(self) -> None:
        t = cs.LOCATE_F1
        for ours, theirs, expected in (
                (t, t, "both_locate"),
                (t, t - 0.01, "table_field_only"),
                (t - 0.01, t, "p2rank_only"),
                (t - 0.01, t - 0.01, "neither_locates")):
            self.assertEqual(
                cs.outcome_of({"f1": {cs.OURS: ours, cs.BASELINE: theirs}}),
                expected, f"F1 {ours}/{theirs}")


class TestTheArtifact(unittest.TestCase):
    def setUp(self) -> None:
        if not ARTIFACT.exists():
            self.skipTest(f"{ARTIFACT.name} not present")
        self.rec = json.loads(ARTIFACT.read_text())
        self.cases = {c["case"]: c for c in self.rec["cases"]}

    def test_it_rederives_from_the_labels_and_raw_output(self) -> None:
        self.assertEqual(cs.main(["--check"]), 0)

    def test_the_failure_case_is_present(self) -> None:
        self.assertIn(
            "neither_locates", self.cases,
            "the case where both methods miss is the commonest outcome on this "
            "fold and may not be dropped from the illustration")

    def test_every_case_is_the_outcome_it_claims(self) -> None:
        for name, case in self.cases.items():
            self.assertEqual(cs.outcome_of(case), name,
                             f"{case['unit_id']} is filed under {name}")

    def test_the_population_counts_add_up(self) -> None:
        p = self.rec["population"]
        self.assertEqual(
            p["n_units"],
            p["n_located_by_both"]
            + (p["n_located_by_table_field"] - p["n_located_by_both"])
            + (p["n_located_by_p2rank"] - p["n_located_by_both"])
            + p["n_located_by_neither"])

    def test_most_of_the_fold_is_located_by_neither(self) -> None:
        # Not a fact about the artifact: a fact about the benchmark, asserted
        # so that a future change which quietly makes the headline rosier has
        # to come through this line.
        p = self.rec["population"]
        self.assertGreater(p["n_located_by_neither"], p["n_units"] / 2)

    def test_the_burial_reading_covers_every_outcome(self) -> None:
        if not self.rec.get("receptors_available"):
            self.skipTest("receptors not fetched on this machine")
        outcomes = {o["outcome"] for o in self.rec["burial"]["by_outcome"]}
        self.assertEqual(outcomes, {"both_locate", "table_field_only",
                                    "p2rank_only", "neither_locates"})

    def test_the_sites_nobody_finds_are_the_least_buried(self) -> None:
        # The claim the manuscript makes from this file. If it stops holding,
        # the sentence has to change, and this is what makes that happen.
        if not self.rec.get("receptors_available"):
            self.skipTest("receptors not fetched on this machine")
        excess = {o["outcome"]: o["pocket_excess_over_chain"]
                  for o in self.rec["burial"]["by_outcome"]}
        self.assertLess(excess["neither_locates"], excess["both_locate"],
                        "the sites neither method finds are no longer less "
                        "buried than the sites both find")


if __name__ == "__main__":
    unittest.main()
