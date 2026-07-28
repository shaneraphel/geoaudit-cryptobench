"""The quotient is only worth anything if the address really is the orbit.

Two failure modes would both show up as a plausible-looking AUC and neither
would raise: an address that collides distinct orbits silently merges cells that
should be separate, and an address that separates equivalent words silently
fails to perform the quotient at all. The first one happened during development
-- a combinatorial-number-system ranking with an off-by-one lost 0.08 ROC-AUC on
the training pick half and looked like the idea failing rather than the encoder
failing -- so the bijection is tested by exhaustion on every small case.
"""
from __future__ import annotations

import itertools
import unittest
from math import comb, log

import numpy as np

from pocket_bench.methods.quotient_tables import (
    compile_cells, dense_address, n_cells_dense, n_orbits, orbit_address,
    read_cells, widest_admissible,
)

RN = 13524          # positives on the training fold: 234838 residues at r=0.0576


class TestTheAddressIsTheOrbit(unittest.TestCase):
    def test_it_is_a_bijection_onto_orbits_by_exhaustion(self):
        for d, L in ((1, 4), (2, 4), (3, 4), (4, 4), (6, 4), (2, 8), (3, 8), (5, 3)):
            words = np.array(list(itertools.product(range(L), repeat=d)))
            codes = orbit_address(words, list(range(d)), L)
            orbits = {tuple(sorted(w)) for w in words.tolist()}
            self.assertEqual(len(np.unique(codes)), len(orbits),
                             f"d={d} L={L}: address is not injective on orbits")
            self.assertEqual(len(orbits), n_orbits(d, L),
                             f"d={d} L={L}: orbit count formula disagrees "
                             f"with enumeration")

    def test_permuting_the_columns_cannot_move_the_address(self):
        rng = np.random.default_rng(20260725)
        X = rng.integers(0, 8, size=(500, 6))
        a = orbit_address(X, [0, 1, 2, 3, 4, 5], 8)
        b = orbit_address(X, [3, 0, 5, 2, 4, 1], 8)
        np.testing.assert_array_equal(a, b)

    def test_the_dense_address_does_move(self):
        """The control: without the quotient, column order is information."""
        rng = np.random.default_rng(20260725)
        X = rng.integers(0, 4, size=(500, 6))
        a = dense_address(X, [0, 1, 2, 3, 4, 5], 4)
        b = dense_address(X, [3, 0, 5, 2, 4, 1], 4)
        self.assertTrue((a != b).any())

    def test_a_digit_out_of_range_is_refused(self):
        with self.assertRaises(ValueError):
            orbit_address(np.array([[0, 4]]), [0, 1], 4)


class TestTheCapacityArithmetic(unittest.TestCase):
    def test_the_dense_bound_is_the_one_the_paper_states(self):
        self.assertAlmostEqual(log(RN, 4), 6.86, places=2)
        self.assertEqual(widest_admissible(4, RN, symmetric=False), 6)

    def test_the_symmetric_bound_is_forty_one(self):
        self.assertEqual(widest_admissible(4, RN, symmetric=True), 41)
        self.assertLessEqual(n_orbits(41, 4), RN)
        self.assertGreater(n_orbits(42, 4), RN)

    def test_every_invariant_fits_one_symmetric_table(self):
        self.assertLessEqual(n_orbits(35, 4), RN)
        self.assertGreater(n_cells_dense(35, 4), RN)

    def test_holding_the_width_the_quotient_buys_levels(self):
        """d=6: dense stops at 4 levels, symmetric reaches 12."""
        dense = [L for L in range(2, 33) if n_cells_dense(6, L) <= RN]
        sym = [L for L in range(2, 33) if n_orbits(6, L) <= RN]
        self.assertEqual(max(dense), 4)
        self.assertEqual(max(sym), 12)

    def test_the_quotient_is_never_the_more_expensive_option(self):
        for d in range(1, 20):
            for L in (2, 4, 8, 16):
                self.assertLessEqual(n_orbits(d, L), n_cells_dense(d, L))


class TestCountingAndReading(unittest.TestCase):
    def test_a_cell_reports_the_fraction_that_was_counted_into_it(self):
        addr = np.array([7, 7, 7, 9, 9])
        y = np.array([1, 0, 1, 0, 0])
        a, p, t = compile_cells(addr, y)
        np.testing.assert_array_equal(a, [7, 9])
        np.testing.assert_array_equal(p, [2, 0])
        np.testing.assert_array_equal(t, [3, 2])
        got = read_cells(a, p, t, np.array([7, 9]), 0.05)
        np.testing.assert_allclose(got, [2 / 3, 0.0])

    def test_an_address_never_counted_reads_the_base_rate(self):
        a, p, t = compile_cells(np.array([1, 1]), np.array([1, 0]))
        got = read_cells(a, p, t, np.array([1, 42, -3]), 0.0576)
        np.testing.assert_allclose(got, [0.5, 0.0576, 0.0576])

    def test_an_empty_table_reads_the_base_rate_everywhere(self):
        got = read_cells(np.array([], dtype=np.int64), np.array([]), np.array([]),
                         np.array([1, 2]), 0.0576)
        np.testing.assert_allclose(got, [0.0576, 0.0576])


if __name__ == "__main__":
    unittest.main()
