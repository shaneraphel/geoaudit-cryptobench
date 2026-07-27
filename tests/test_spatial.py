"""The cell list must reproduce the dense scan exactly, not approximately.

Every neighbour query in the detectors used to be a blocked dense scan. Replacing
it with a cell list is only legitimate if the reported pair set and the order in
which pairs are reported are both unchanged: the set because the predicate is the
science, and the order because downstream ``np.add.at`` accumulations and the
Kruskal tie-break on equal weights are both order-sensitive at the last bit.

Written against ``unittest`` rather than ``pytest`` for a reason that is not
stylistic. ``make test`` and the CI workflow both run ``unittest discover``, so
the five checks in this file -- the only ones guarding the neighbour kernel that
every gate, context transform and topology descriptor calls -- were collected by
pytest locally and never executed by CI. A test that only runs on the author's
machine is a test that will eventually be wrong without anyone learning of it.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from pocket_bench.methods.geometric_foundation import _receptor_coords
from pocket_bench.spatial import counts_within, cross_within, self_pairs_within

ROOT = Path(__file__).resolve().parents[1]
RECEPTORS = ROOT / "data/cryptobench_apo/official_receptors"


def _dense_cross(q, c, r):
    d2 = ((q[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    return np.nonzero(d2 <= r * r)


def _dense_counts(c, r):
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    return (d2 <= r * r).sum(1) - 1


def _dense_pairs(c, r):
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    i, j = np.nonzero(d2 <= r * r)
    k = i < j
    i, j = i[k], j[k]
    o = np.lexsort((j, i))
    return np.stack([i[o], j[o]], axis=1), np.sqrt(d2[i[o], j[o]])


class TestAgainstDenseScan(unittest.TestCase):
    def test_matches_dense_scan_on_synthetic(self) -> None:
        rng = np.random.default_rng(11)
        # A dense core plus a detached satellite plus exact duplicates: the
        # three cases that break naive binning (empty cubes, out-of-box
        # queries, ties).
        coords = np.concatenate([
            rng.normal(0.0, 6.0, (300, 3)),
            rng.normal(40.0, 1.5, (60, 3)),
        ])
        coords = np.concatenate([coords, coords[:8]])
        query = np.concatenate([rng.normal(0.0, 9.0, (120, 3)), coords[:5],
                                np.full((3, 3), -500.0)])

        for radius in (4.0, 6.0, 8.0):
            with self.subTest(radius=radius):
                qi, ai = cross_within(query, coords, radius)
                rqi, rai = _dense_cross(query, coords, radius)
                self.assertTrue(np.array_equal(qi, rqi))
                self.assertTrue(np.array_equal(ai, rai))

                self.assertTrue(np.array_equal(counts_within(coords, radius),
                                               _dense_counts(coords, radius)))

                pairs, dist = self_pairs_within(coords, radius)
                rpairs, rdist = _dense_pairs(coords, radius)
                self.assertTrue(np.array_equal(pairs, rpairs))
                # Bitwise, not approximate: a distance that differs in the last
                # bit can flip a Kruskal tie and change a Betti number.
                self.assertTrue(np.array_equal(dist, rdist))

    def test_matches_dense_scan_on_real_receptor(self) -> None:
        path = RECEPTORS / "1arl_A_receptor.pdb"
        if not path.is_file():
            self.skipTest("official receptors not materialized")
        coords = _receptor_coords(path)

        self.assertTrue(np.array_equal(counts_within(coords, 8.0),
                                       _dense_counts(coords, 8.0)))
        pairs, dist = self_pairs_within(coords, 8.0)
        rpairs, rdist = _dense_pairs(coords, 8.0)
        self.assertTrue(np.array_equal(pairs, rpairs))
        self.assertTrue(np.array_equal(dist, rdist))

    def test_empty_and_degenerate_inputs(self) -> None:
        empty = np.zeros((0, 3))
        one = np.zeros((1, 3))
        self.assertEqual(cross_within(empty, one, 5.0)[0].size, 0)
        self.assertEqual(cross_within(one, empty, 5.0)[0].size, 0)
        self.assertEqual(self_pairs_within(one, 5.0)[0].shape, (0, 2))
        self.assertEqual(counts_within(one, 5.0).tolist(), [0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
