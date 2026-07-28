"""The sweep artifact has to support what Section 'What the constants are
worth' says about it.

Two failure modes are worth guarding here. The first is a sweep that quietly
varies more than one thing at a time, which would make a row uninterpretable.
The second is the sweep turning into a selection: the published configuration
happens to be the best of the eight, and the only thing separating that from
tuning is that it was frozen first and the artifact still records the frozen
constants rather than the winning ones.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SWEEP = ROOT / "results/architecture_sweep/SENSITIVITY_SWEEP.json"
TABFIELD = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"
SELECTION = ROOT / "results/architecture_sweep/COUNTERATTACK_WIDE2.json"


class TestTheSweepArtifact(unittest.TestCase):
    def setUp(self):
        if not SWEEP.exists():
            self.skipTest("the sensitivity sweep has not been run")
        self.d = json.loads(SWEEP.read_text())
        self.rows = self.d["rows"]

    def _row(self, levels: int, ranking: str, cap: int) -> dict:
        for r in self.rows:
            if (r["levels"], r["ranking"], r["cap"]) == (levels, ranking, cap):
                return r
        self.fail(f"no row for {levels} levels, {ranking}, cap {cap}")

    def test_the_sweep_finished(self):
        """A checkpoint would let the paper quote a range over settings that
        were never all measured."""
        self.assertTrue(self.d["complete"])
        self.assertEqual(len(self.d["cells_compiled"]),
                         len(self.d["swept"]["levels"])
                         * len(self.d["swept"]["ranking"]))

    def test_the_test_fold_was_not_read(self):
        self.assertFalse(self.d["reads_test_fold"])

    def test_the_rows_are_exactly_the_design_the_section_describes(self):
        """A full levels-by-ranking grid at the published cap, plus the cap
        varied at the published levels and ranking. Extra rows would mean the
        table under-reports what was measured; missing ones would mean a
        comparison in the text has no row behind it."""
        base = self.d["frozen_configuration"]
        want = {(lv, rk, base["cap"])
                for lv in self.d["swept"]["levels"]
                for rk in self.d["swept"]["ranking"]}
        want |= {(base["levels"], base["ranking"], c)
                 for c in self.d["swept"]["cap"]}
        got = {(r["levels"], r["ranking"], r["cap"]) for r in self.rows}
        self.assertEqual(got, want)

    def test_the_cap_rows_vary_only_the_cap(self):
        """The cap axis is the one the section reads one-at-a-time, so those
        rows must hold everything else at the published setting."""
        base = self.d["frozen_configuration"]
        for r in self.rows:
            if r["cap"] == base["cap"]:
                continue
            with self.subTest(cap=r["cap"]):
                self.assertEqual(r["levels"], base["levels"])
                self.assertEqual(r["ranking"], base["ranking"])

    def test_the_frozen_configuration_is_the_shipped_one(self):
        from pocket_bench.methods import table_bank

        base = self.d["frozen_configuration"]
        self.assertEqual(base["levels"], table_bank.N_LEVELS)
        if TABFIELD.exists():
            shipped = json.loads(TABFIELD.read_text())
            self.assertEqual(base["cap"], shipped["fan_out_cap"])
            self.assertEqual(base["ridge"], shipped["ridge"])

    def test_exactly_one_row_is_marked_published_and_it_is_the_frozen_one(self):
        marked = [r for r in self.rows if r["is_published_configuration"]]
        self.assertEqual(len(marked), 1)
        base = self.d["frozen_configuration"]
        for k in ("levels", "ranking", "cap"):
            self.assertEqual(marked[0][k], base[k])

    def test_the_published_row_reproduces_the_selection_artifact_exactly(self):
        """This is what licenses reading the other seven rows as the effect of
        the constant that was varied. The sweep re-implements the digitiser and
        the fusion so it can parameterise the level, and it re-derives the
        split from the manifest; if any of that had drifted, the row holding
        everything at the published setting would not land on the number the
        selection recorded. Exactly, not to four places -- a re-implementation
        that agreed only to rounding would be a different computation."""
        if not SELECTION.exists():
            self.skipTest("the selection artifact is not present")
        published = json.loads(SELECTION.read_text())["selected"][
            "pick_half_roc_auc"]
        self.assertEqual(self.d["published_pick_half_roc_auc"], published)

    def test_the_range_follows_from_the_rows(self):
        aucs = [r["pick_half_roc_auc"] for r in self.rows]
        self.assertAlmostEqual(max(aucs) - min(aucs),
                               self.d["range_over_all_settings"], places=5)

    def test_nothing_here_is_fragile(self):
        """The section says the worst setting still clears the linear ceiling
        of the invariants the field reads."""
        self.assertLess(self.d["range_over_all_settings"], 0.05)
        self.assertGreater(min(r["pick_half_roc_auc"] for r in self.rows),
                           0.783)

    def test_within_chain_ranking_beats_pooled_at_every_level(self):
        """Stated as a prediction of the construction, so it has to hold at
        every level and not only on average."""
        for lv in self.d["swept"]["levels"]:
            with self.subTest(levels=lv):
                self.assertGreater(self._row(lv, "within-chain", 32)
                                   ["pick_half_roc_auc"],
                                   self._row(lv, "pooled", 32)
                                   ["pick_half_roc_auc"])

    def test_four_levels_is_the_turn_and_not_a_preference(self):
        four = self._row(4, "within-chain", 32)["pick_half_roc_auc"]
        for lv in (3, 5):
            with self.subTest(levels=lv):
                self.assertLess(self._row(lv, "within-chain", 32)
                                ["pick_half_roc_auc"], four)

    def test_a_finer_alphabet_empties_more_cells(self):
        """The mechanism the section offers for why five levels lose. If
        coverage improved with the finer digit the explanation would be
        wrong."""
        empt = [self._row(lv, "within-chain", 32)["fraction_never_addressed"]
                for lv in sorted(self.d["swept"]["levels"])]
        self.assertEqual(empt, sorted(empt))


if __name__ == "__main__":
    unittest.main()
