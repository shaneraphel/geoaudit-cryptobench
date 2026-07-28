"""The diagnosis that corrected the manuscript has to keep being true.

Section~\\ref{sec:ai} now says four things the previous version did not: that
capacity is not what binds, that the fan-out is, that the fitted linear readout
is handed wires the counting field never sees, and that the quotient's gain
lives on structures the dense bank already fails. Each is a claim about the
artifact, so each is a test.
"""
from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import gap_decomposition as gd  # noqa: E402

ART_PATH = gd.OUT
MANUSCRIPT = (ROOT / "paper/MAIN_CRYPTOBENCH_GEOAUDIT.tex").read_text()
MACROS = (ROOT / "paper/frozen_numbers.tex").read_text()


def _art():
    if not ART_PATH.exists():
        raise unittest.SkipTest("run `make gap` to produce the artifact")
    return json.loads(ART_PATH.read_text())


def _macro(name: str) -> str:
    m = re.search(r"\\newcommand\{\\" + name + r"\}\{(.*?)\}\n", MACROS)
    if not m:
        raise AssertionError(f"{name} is cited by the manuscript and not defined")
    return m.group(1)


class TestTheArtifactIsSelfConsistent(unittest.TestCase):
    def test_the_gate_passes(self):
        self.assertEqual(gd.audit(), 0)

    def test_it_reads_nothing(self):
        self.assertEqual(_art()["test_fold_reads"], 0)

    def test_the_two_effects_add_to_the_total(self):
        d = _art()["decomposition"]
        self.assertAlmostEqual(d["readout_effect"] + d["input_effect"],
                               d["total"], places=4)

    def test_the_shares_add_to_one(self):
        d = _art()["decomposition"]
        self.assertAlmostEqual(d["readout_share"] + d["input_share"], 1.0,
                               places=2)

    def test_the_test_fold_numbers_are_the_recorded_ones(self):
        """The difficulty analysis must reuse values, never recompute them."""
        art = _art()
        probe = json.loads(
            (ROOT / "results/official_fold"
                    "/COUNTERATTACK_QUOTIENT_PROBE.json").read_text())
        n = sum(1 for r in probe["per_structure"] if r["residue_auc"] is not None)
        self.assertLessEqual(art["difficulty"]["test_fold"]["n_units"], n)
        self.assertIn("no new read", art["difficulty"]["test_fold"]["source"])


class TestCapacityIsNotWhatBinds(unittest.TestCase):
    def test_the_quotient_gain_does_not_shrink_with_the_compile_set(self):
        rows = _art()["capacity_probes"]["compile_scaling"]["rows"]
        self.assertGreater(rows[-1]["gain"], rows[0]["gain"],
                           "a small-sample advantage would decay; the manuscript "
                           "says this one does not")

    def test_marginal_tables_trail_the_interaction_bank(self):
        m = _art()["capacity_probes"]["marginal_tables"]
        self.assertLess(max(r["roc_auc"] for r in m["rows"]), m["dense_reference"],
                        "removing the capacity constraint entirely must not help")

    def test_the_bank_almost_never_falls_back(self):
        self.assertLess(
            _art()["capacity_probes"]["unseen_cells"]["dense_L4_fraction"], 0.01)


class TestTheFanoutIsWhatBinds(unittest.TestCase):
    def test_the_dense_bank_gains_from_a_solved_fanout(self):
        self.assertGreater(
            _art()["fanout_price"]["banks"]["dense_L4"]["price"], 0.005)

    def test_the_quotient_bank_does_not(self):
        fan = _art()["fanout_price"]["banks"]
        self.assertLess(fan["quotient_L864"]["price"],
                        fan["dense_L4"]["price"] / 2)

    def test_the_two_do_not_compose_past_the_ceiling(self):
        fan = _art()["fanout_price"]["banks"]
        ceiling = fan["dense_L4"]["solved_integer_fanout"]
        self.assertLess(fan["quotient_L864"]["gini_rank"], ceiling + 0.005)


class TestTheGainLivesOnHardStructures(unittest.TestCase):
    def test_it_is_positive_at_the_bottom_on_both_folds(self):
        art = _art()
        for where in ("train_pick_half", "test_fold"):
            self.assertGreater(art["difficulty"][where]["bins"][0]["mean_gain"], 0)

    def test_it_is_negative_on_structures_the_bank_already_handles(self):
        art = _art()
        for where in ("train_pick_half", "test_fold"):
            self.assertLess(art["difficulty"][where]["bins"][-2]["mean_gain"], 0)

    def test_composition_is_the_smaller_half_of_the_shortfall(self):
        d = _art()["difficulty"]
        recovered = d["reweighted_test_gain"] - d["test_fold"]["mean_gain"]
        shortfall = (d["train_pick_half"]["mean_gain"]
                     - d["test_fold"]["mean_gain"])
        self.assertGreaterEqual(recovered, 0)
        self.assertLess(recovered, shortfall / 2,
                        "the manuscript says reweighting recovers only part")


class TestTheManuscriptStillFollows(unittest.TestCase):
    MACROS_REQUIRED = (
        "GapTablesSmall", "GapLinearSmall", "GapLinearWide", "GapReadout",
        "GapInput", "GapInputShare", "GapFanoutPrice", "GapFanoutSolved",
        "GapFanoutQuo", "GapMarginalAuc", "GapUnseenPct", "GapHardTrainGain",
        "GapHardTestGain", "GapEasyTrainGain", "GapEasyTestGain",
        "GapReweighted", "GapExtraWires", "GapWideWires",
    )

    def test_each_macro_is_defined(self):
        for name in self.MACROS_REQUIRED:
            self.assertTrue(_macro(name), name)

    def test_the_extra_wire_count_is_the_difference(self):
        cells = _art()["decomposition"]["cells"]
        self.assertEqual(int(_macro("GapExtraWires")),
                         cells["linear_172"]["n_wires"]
                         - cells["tables_35"]["n_wires"])

    def test_the_manuscript_no_longer_calls_the_deficit_a_capacity_limit(self):
        self.assertNotIn("The gap is a capacity limit", MANUSCRIPT)

    def test_the_correction_is_stated_where_the_claim_was(self):
        self.assertIn("does not survive being tested", MANUSCRIPT)

    def test_the_linear_readout_is_not_sold_as_same_invariants(self):
        src = (ROOT / "src/pocket_bench/methods"
                      "/algebraic_field_linear.py").read_text()
        head = src[:src.index("The wires")]
        self.assertNotIn("differ in exactly one place", head)
        self.assertIn("172", head)


if __name__ == "__main__":
    unittest.main()
