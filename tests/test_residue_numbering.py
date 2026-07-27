"""Residue numbers are signed, and the evaluation must not collapse them.

PDB entries that retain a cloning or expression tag number those residues 0 and
downwards. Eleven of the 192 official CryptoBench test structures do, so this is
not a hypothetical input. When the parser dropped the minus sign, residue -1 and
residue 1 became the same dictionary key, and the two exchanged scores: the tag
residue was handed the score of a real residue and the real residue -- which can
be a labelled cryptic-site residue -- was handed the tag's.

The failure was silent. Every count stayed plausible, no exception was raised,
and the metrics moved by a few thousandths, which is the size of the effect the
paper reports. These tests exist so it cannot come back.
"""
from __future__ import annotations

import unittest

from pocket_bench.metrics import native_residue_scores, residue_auc_pr
from pocket_bench.residue_id import resseq as _resseq


class TestResidueNumbering(unittest.TestCase):
    def test_sign_is_preserved(self):
        self.assertEqual(_resseq(-1), -1)
        self.assertEqual(_resseq("-1"), -1)
        self.assertEqual(_resseq("-42"), -42)

    def test_positive_forms_still_parse(self):
        self.assertEqual(_resseq(7), 7)
        self.assertEqual(_resseq("7"), 7)
        self.assertEqual(_resseq("A:ALA123"), 123)
        self.assertIsNone(_resseq("A:ALA"))

    def test_tagged_identifier_keeps_its_sign(self):
        # pdb_io writes 'chain:resnameresseq' with no separator before the
        # number, so a hyphen there is the sign and nothing else.
        self.assertEqual(_resseq("A:GLY-1"), -1)

    def test_tag_residue_does_not_collide_with_its_positive_twin(self):
        universe = [-1, 1, 2, 3]
        pred = {
            "residue_scores": {"-1": 0.0, "1": 9.0, "2": 5.0, "3": 1.0},
            "residue_positive": [1, 2],
        }
        scores, positive = native_residue_scores(pred, universe)
        self.assertEqual(len(scores), 4, "the tag residue must survive as its own key")
        self.assertEqual(scores[-1], 0.0)
        self.assertEqual(scores[1], 9.0, "residue 1 must keep its own score")
        self.assertEqual(positive, {1, 2})

    def test_insertion_code_names_the_same_slot(self):
        # Deliberate, and forced by the label vocabulary: CryptoBench marks a
        # site with bare integers, so 132 and 132A cannot be told apart by any
        # label and must share one slot rather than one of them being dropped.
        self.assertEqual(_resseq("132A"), 132)
        self.assertEqual(_resseq("132"), 132)

    def test_one_definition_of_residue_identity(self):
        # The scorer and the P2Rank adapter each used to parse residue
        # identifiers, and the two copies disagreed. They now share this one.
        from pocket_bench import metrics
        from pocket_bench.methods import p2rank_wrap
        from pocket_bench import residue_id
        self.assertIs(metrics._resseq, residue_id.resseq)
        self.assertIs(p2rank_wrap.residue_id, residue_id)

    def test_a_collision_would_change_the_score_of_a_real_residue(self):
        # The regression, stated as an outcome rather than as a parser detail:
        # with the sign dropped, the residue the label calls cryptic is ranked by
        # the tag's score. Here that is the difference between a perfect
        # separation and a broken one.
        universe = [-1, 1, 2]
        pred = {
            "residue_scores": {"-1": 0.0, "1": 9.0, "2": 1.0},
            "residue_positive": [1],
        }
        out = residue_auc_pr([], [1], universe, pred)
        self.assertTrue(out["available"])
        self.assertEqual(out["n_universe"], 3)
        self.assertEqual(out["residue_auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
