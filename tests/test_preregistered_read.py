"""The fifth read has to keep meaning what it meant when it was taken.

Its whole claim is an ordering: the functional was chosen, committed, and only
then applied to the held-out fold. An ordering is cheap to assert and easy to
lose, so it is re-checked here against live git rather than read back out of the
artifact that asserts it.
"""
from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import preregister_statistic as ps  # noqa: E402
import preregistered_read as pr  # noqa: E402


def _art():
    if not pr.OUT.exists():
        raise unittest.SkipTest("run `make read5` to produce the artifact")
    return json.loads(pr.OUT.read_text())


class TestTheOrderingHolds(unittest.TestCase):
    def test_the_choice_is_committed_and_precedes_head(self):
        """Re-derived from git, not believed from the artifact."""
        self.assertTrue(pr.preregistration_precedes_this_read()["committed_in"])

    def test_the_recorded_commit_still_contains_the_choice(self):
        prov = _art()["provenance_of_the_choice"]
        blob = subprocess.run(
            ["git", "show", f"{prov['committed_in']}:{prov['artifact']}"],
            cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(blob.returncode, 0,
                         "the commit the read cites no longer holds the file")
        at_commit = json.loads(blob.stdout)["preregistered"]["statistic"]
        self.assertEqual(at_commit,
                         _art()["preregistered_result"]["statistic"],
                         "the statistic reported as preregistered is not the "
                         "one that commit actually fixed")

    def test_the_live_artifact_has_not_drifted_from_it(self):
        live = json.loads(ps.OUT.read_text())["preregistered"]["statistic"]
        self.assertEqual(live, _art()["preregistered_result"]["statistic"])


class TestItIsIndexedHonestly(unittest.TestCase):
    def test_it_declares_the_fifth_read(self):
        self.assertEqual(_art()["test_fold_read_index"], 5)

    def test_it_does_not_pretend_to_have_rescored_anything(self):
        self.assertIs(_art()["rescored_anything"], False)

    def test_the_ledger_counts_it(self):
        led = json.loads(
            (ROOT / "results/official_fold/TEST_FOLD_ACCESS_LEDGER.json"
             ).read_text())
        idx = {r["read_index"]: r["artifact"]
               for r in led["indexed_read_sequence"]}
        self.assertIn(5, idx)
        self.assertTrue(idx[5].endswith("PREREGISTERED_READ.json"))
        self.assertGreaterEqual(led["n_probes_that_only_resummarised"], 1)


class TestTheResultIsReportedWhole(unittest.TestCase):
    def test_every_candidate_the_choice_considered_is_shown(self):
        got = {c["statistic"] for c in _art()["candidates"]}
        self.assertEqual(got, set(ps.STATISTICS))

    def test_exactly_one_is_marked_preregistered(self):
        self.assertEqual(
            sum(c["preregistered"] for c in _art()["candidates"]), 1)

    def test_the_preregistered_statistic_resolved(self):
        c = _art()["preregistered_result"]
        self.assertFalse(c["crosses_zero"])
        self.assertGreater(c["ci_low"], 0)

    def test_the_mean_is_still_shown_and_still_does_not_resolve(self):
        """Dropping it once a better-powered functional cleared zero would be
        the selection the preregistration exists to prevent."""
        m = _art()["mean_reported_beside_it"]
        self.assertEqual(m["statistic"], "mean")
        self.assertTrue(m["crosses_zero"])


class TestTheTailsAreNotHidden(unittest.TestCase):
    """A trimmed mean that clears zero invites exactly one objection, so the
    numbers behind it are recorded rather than left to be asked for."""

    def test_the_shape_block_adds_up(self):
        s = _art()["shape_of_the_differences"]
        self.assertEqual(s["n_field_ahead"] + s["n_baseline_ahead"], s["n"])
        self.assertEqual(s["n_trimmed_each_side"], round(0.20 * s["n"]))

    def test_the_discarded_losses_are_reported_and_are_worse_than_the_wins(self):
        s = _art()["shape_of_the_differences"]
        self.assertLess(s["worst_losses_mean"], 0)
        self.assertGreater(s["best_wins_mean"], 0)
        self.assertGreater(abs(s["worst_losses_mean"]), s["best_wins_mean"],
                           "if the tails ever stop being asymmetric, the "
                           "explanation for the mean staying unresolved has "
                           "changed and the manuscript is out of date")

    def test_the_middle_is_what_the_trimmed_statistic_reports(self):
        art = _art()
        self.assertAlmostEqual(
            art["shape_of_the_differences"]["mean_of_the_middle_60_percent"],
            art["preregistered_result"]["point"], places=6)


class TestTheForecastWasNotEdited(unittest.TestCase):
    def test_it_is_carried_over_verbatim_from_the_preregistration(self):
        self.assertEqual(_art()["forecast_made_before_the_read"],
                         json.loads(ps.OUT.read_text())["forecast"])

    def test_a_forecast_that_missed_says_so(self):
        f = _art()["forecast_vs_outcome"]
        self.assertEqual(f["forecast_direction_held"],
                         f["resolved"] == (f["expected_power"] >= 0.5))
        if not f["forecast_direction_held"]:
            self.assertTrue(f["why_it_missed"])


class TestTheNumbersComeFromTheFrozenTelemetry(unittest.TestCase):
    def test_the_two_arms_match_the_published_bootstrap(self):
        art = _art()
        pub = json.loads(
            (ROOT / "results/official_fold"
             / "OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json").read_text())
        per = pub["metrics"]["residue_auc"]["per_method"]
        self.assertAlmostEqual(art["mean_method"], per["table_field"]["point"],
                               places=6)
        self.assertAlmostEqual(art["mean_baseline"], per["p2rank"]["point"],
                               places=6)

    def test_the_mean_candidate_reproduces_the_published_delta(self):
        art = _art()
        pub = json.loads(
            (ROOT / "results/official_fold"
             / "OFFICIAL_MULTI_METHOD_BOOTSTRAP_vs_P2RANK.json").read_text())
        d = (pub["metrics"]["residue_auc"]["paired_vs_baseline"]
             ["table_field"]["delta_point"])
        self.assertAlmostEqual(art["mean_reported_beside_it"]["point"], d,
                               places=6)

    def test_the_statistics_recompute_from_the_telemetry(self):
        art = _art()
        a, b, units = pr.paired_values()
        d = a - b
        strata = pr.strata_for(units)
        for c in art["candidates"]:
            got = float(ps.STATISTICS[c["statistic"]](d[None, :], strata)[0])
            self.assertAlmostEqual(got, c["point"], places=6,
                                   msg=f"{c['statistic']} no longer recomputes")


if __name__ == "__main__":
    unittest.main()
