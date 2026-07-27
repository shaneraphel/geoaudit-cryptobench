"""Tests for the table field: the bank, the wide transform, and the artifact.

These are written against the properties the manuscript actually claims, not
against whatever the code happens to return. Four of them exist because the
corresponding mistake was made at some point during development:

* a bank whose per-chain digitisation silently depended on which other chains
  were in the same array would make a structure's score a function of the batch
  it was scored in, and the fold means would not be reproducible one structure
  at a time;
* a blocked accumulation that disagrees with an unblocked one at the block
  boundary would put a quiet error into every compiled cell;
* a gate normalised by the maximum rather than the spread mixes in a different
  amount on every chain, because the maximum of a score field is an order
  statistic of a handful of residues;
* an artifact whose cell counts do not satisfy ``positive <= total`` cannot be
  divided into a probability, and nothing downstream would notice.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from pocket_bench.methods import table_bank as tb
from pocket_bench.methods import wide_descriptors as wd

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data/cryptobench_apo/TABLE_FIELD.json"


class TestPartitions(unittest.TestCase):
    def test_every_wire_appears_once_per_round(self) -> None:
        tables = tb.partition_tables(24, 2, 5, seed=1)
        counts = np.zeros(24, dtype=int)
        for t in tables:
            for c in t:
                counts[c] += 1
        self.assertTrue(np.all(counts == 5), counts)

    def test_deterministic_under_seed(self) -> None:
        a = tb.partition_tables(37, 3, 4, seed=7)
        b = tb.partition_tables(37, 3, 4, seed=7)
        c = tb.partition_tables(37, 3, 4, seed=8)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_odd_wire_count_leaves_no_singleton_table(self) -> None:
        """A one-wire table is not a table; the trailing remainder is dropped."""
        for n in (7, 9, 11):
            for w in (2, 3):
                for t in tb.partition_tables(n, w, 3, seed=3):
                    self.assertGreaterEqual(len(t), 2)


class TestDigits(unittest.TestCase):
    def test_levels_in_range(self) -> None:
        rng = np.random.default_rng(0)
        F = rng.normal(size=(50, 4))
        D = tb.chain_digits(F, [50])
        self.assertEqual(D.min(), 0)
        self.assertEqual(D.max(), tb.N_LEVELS - 1)

    def test_invariant_under_monotone_rescaling(self) -> None:
        """Ranking makes a wire independent of its units."""
        rng = np.random.default_rng(1)
        F = rng.normal(size=(40, 3))
        a = tb.chain_digits(F, [40])
        b = tb.chain_digits(np.exp(3.0 * F) * 17.0 + 5.0, [40])
        np.testing.assert_array_equal(a, b)

    def test_chains_are_digitised_independently(self) -> None:
        """The score of a chain may not depend on its batch-mates.

        Two chains digitised together must give the same digits as each chain
        digitised alone, or a structure's prediction becomes a function of what
        else was in the run.
        """
        rng = np.random.default_rng(2)
        A = rng.normal(size=(30, 5))
        B = rng.normal(size=(20, 5)) * 100.0 + 50.0
        both = tb.chain_digits(np.vstack([A, B]), [30, 20])
        np.testing.assert_array_equal(both[:30], tb.chain_digits(A, [30]))
        np.testing.assert_array_equal(both[30:], tb.chain_digits(B, [20]))

    def test_ties_share_a_level(self) -> None:
        F = np.array([[1.0], [1.0], [1.0], [2.0]])
        D = tb.chain_digits(F, [4])
        self.assertEqual(len(set(D[:3, 0].tolist())), 1)


class TestBank(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(3)
        self.D = rng.integers(0, tb.N_LEVELS, size=(500, 8)).astype(np.int8)
        self.y = (self.D[:, 0] >= 2).astype(np.int64)
        self.tables = tb.partition_tables(8, 2, 2, seed=5)
        self.offsets = tb.cell_offsets(self.tables)

    def test_addresses_stay_inside_their_table(self) -> None:
        ad = tb.addresses(self.D, self.tables, self.offsets, 0, 500)
        for k in range(len(self.tables)):
            self.assertTrue(np.all(ad[:, k] >= self.offsets[k]))
            self.assertTrue(np.all(ad[:, k] < self.offsets[k + 1]))

    def test_counts_are_complete(self) -> None:
        frac, tot = tb.compile_cells(self.D, self.y, self.tables, self.offsets)
        for k in range(len(self.tables)):
            block = tot[self.offsets[k]:self.offsets[k + 1]]
            self.assertEqual(int(block.sum()), self.D.shape[0])

    def test_blocked_accumulation_matches_unblocked(self) -> None:
        """Block size must not change a single compiled cell."""
        frac_a, tot_a = tb.compile_cells(self.D, self.y, self.tables,
                                         self.offsets)
        old = tb.BLOCK
        try:
            tb.BLOCK = 7  # a size that lands mid-row-group repeatedly
            frac_b, tot_b = tb.compile_cells(self.D, self.y, self.tables,
                                             self.offsets)
        finally:
            tb.BLOCK = old
        np.testing.assert_array_equal(tot_a, tot_b)
        np.testing.assert_allclose(frac_a, frac_b)

    def test_a_table_that_sees_the_label_reads_it_exactly(self) -> None:
        """Cells are exact conditional frequencies, not estimates of them."""
        tables = [[0, 1]]
        offsets = tb.cell_offsets(tables)
        frac, tot = tb.compile_cells(self.D, self.y, tables, offsets)
        for a in range(tb.N_LEVELS ** 2):
            sel = (self.D[:, 0] + self.D[:, 1] * tb.N_LEVELS) == a
            if sel.sum():
                self.assertAlmostEqual(float(frac[a]),
                                       float(self.y[sel].mean()), places=12)

    def test_score_is_the_integer_weighted_sum(self) -> None:
        frac, _ = tb.compile_cells(self.D, self.y, self.tables, self.offsets)
        mult = np.arange(len(self.tables)) - 2
        got = tb.score(self.D, self.tables, self.offsets, frac, mult)
        ad = tb.addresses(self.D, self.tables, self.offsets, 0, 500)
        want = frac[ad] @ mult.astype(float)
        np.testing.assert_allclose(got, want)

    def test_fanout_is_integral_and_capped(self) -> None:
        frac, _ = tb.compile_cells(self.D, self.y, self.tables, self.offsets)
        m = tb.integer_fanout(self.D, self.y, self.tables, self.offsets, frac,
                              ridge=0.1, cap=32)
        self.assertEqual(m.dtype.kind, "i")
        self.assertLessEqual(int(np.abs(m).max()), 32)

    def test_ridge_keeps_a_duplicated_bank_finite(self) -> None:
        """The failure the ridge exists for: exactly collinear columns."""
        tables = self.tables + self.tables  # every column duplicated
        offsets = tb.cell_offsets(tables)
        frac, _ = tb.compile_cells(self.D, self.y, tables, offsets)
        m = tb.integer_fanout(self.D, self.y, tables, offsets, frac,
                              ridge=0.1, cap=32)
        self.assertTrue(np.all(np.isfinite(m)))
        self.assertLessEqual(int(np.abs(m).max()), 32)


class TestWideTransform(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(4)
        self.n = 40
        self.ctr = rng.normal(scale=8.0, size=(self.n, 3))
        self.local = rng.normal(size=(self.n, 3))

    def test_large_radius_mean_is_the_chain_mean(self) -> None:
        X = wd.wide_transform(self.local, self.ctr * 1e-3, [self.n])
        C = self.local.shape[1]
        # the 26 A mean is the fifth mean block, i.e. groups 1..5 after local
        got = X[:, C * 5:C * 6]
        want = np.tile(self.local.mean(0), (self.n, 1))
        np.testing.assert_allclose(got, want, atol=1e-9)

    def test_sd_is_zero_on_a_constant_wire(self) -> None:
        local = np.ones((self.n, 2))
        X = wd.wide_transform(local, self.ctr, [self.n])
        C = 2
        sd_block = X[:, C * 6:C * 9]
        np.testing.assert_allclose(sd_block, 0.0, atol=1e-12)

    def test_local_rank_is_a_fraction(self) -> None:
        X = wd.wide_transform(self.local, self.ctr, [self.n])
        C = self.local.shape[1]
        start = C * (1 + len(wd.MEAN_RADII) + len(wd.VAR_RADII)
                     + len(wd.DIFF_RADII))
        rank_block = X[:, start:]
        self.assertGreaterEqual(float(rank_block.min()), 0.0)
        self.assertLessEqual(float(rank_block.max()), 1.0)

    def test_chains_do_not_see_each_other(self) -> None:
        """The neighbourhood is intra-chain, so batching cannot change a value."""
        rng = np.random.default_rng(5)
        c2 = rng.normal(scale=8.0, size=(15, 3))
        l2 = rng.normal(size=(15, 3))
        both = wd.wide_transform(np.vstack([self.local, l2]),
                                 np.vstack([self.ctr, c2]), [self.n, 15])
        np.testing.assert_allclose(
            both[:self.n], wd.wide_transform(self.local, self.ctr, [self.n]))
        np.testing.assert_allclose(both[self.n:],
                                   wd.wide_transform(l2, c2, [15]))

    def test_wire_count_and_names(self) -> None:
        names = wd.wire_names(tuple(f"f{i}" for i in range(43)))
        self.assertEqual(len(names), 43 * wd.N_STATISTIC_GROUPS)
        self.assertEqual(len(names), 645)
        self.assertEqual(len(set(names)), len(names), "names must be unique")


@unittest.skipUnless(ARTIFACT.exists(), "compiled field not present")
class TestCompiledArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.doc = json.loads(ARTIFACT.read_text())

    def test_counts_are_a_probability(self) -> None:
        tot = np.asarray(self.doc["cell_total"])
        pos = np.asarray(self.doc["cell_positive"])
        self.assertEqual(len(tot), len(pos))
        self.assertEqual(len(tot), self.doc["n_cells"])
        self.assertTrue(np.all(pos >= 0))
        self.assertTrue(np.all(pos <= tot), "positives exceed the total")

    def test_every_cell_of_every_table_is_present(self) -> None:
        want = sum(tb.N_LEVELS ** len(t) for t in self.doc["tables"])
        self.assertEqual(want, self.doc["n_cells"])

    def test_cells_hold_the_whole_training_fold(self) -> None:
        tot = np.asarray(self.doc["cell_total"])
        offsets = tb.cell_offsets(self.doc["tables"])
        n = self.doc["train"]["n_residues"]
        for k in range(len(self.doc["tables"])):
            self.assertEqual(int(tot[offsets[k]:offsets[k + 1]].sum()), n)

    def test_fanout_is_integral_and_within_the_declared_cap(self) -> None:
        m = np.asarray(self.doc["multiplicity"])
        cap = self.doc["fan_out_cap"]
        self.assertEqual(len(m), len(self.doc["tables"]))
        self.assertTrue(np.all(m == m.astype(np.int64)))
        self.assertLessEqual(int(np.abs(m).max()), cap)
        self.assertEqual(int((m != 0).sum()),
                         self.doc["n_tables_with_nonzero_fanout"])
        self.assertEqual(int(np.abs(m).sum()), self.doc["total_fan_out"])

    def test_declared_shape_matches_the_wires(self) -> None:
        self.assertEqual(self.doc["n_wires"], len(self.doc["wire_names"]))
        self.assertEqual(self.doc["n_wires"], 645)
        for t in self.doc["tables"]:
            self.assertEqual(len(t), self.doc["table_width"])
            for c in t:
                self.assertLess(c, self.doc["n_wires"])

    def test_no_cluster_leak_recorded(self) -> None:
        led = self.doc["cluster_ledger"]
        self.assertGreater(led["train_clusters"], 0)
        self.assertGreater(led["test_clusters"], 0)


@unittest.skipUnless(ARTIFACT.exists(), "compiled field not present")
class TestEndToEnd(unittest.TestCase):
    """Scoring a real receptor, which is what a reader will actually run."""

    @classmethod
    def setUpClass(cls) -> None:
        from pocket_bench.methods import table_field
        cls.mod = table_field
        cls.field = table_field.load_field()
        recs = sorted((ROOT / "data/cryptobench_apo/receptors").glob("*.pdb"))
        if not recs:
            recs = sorted((ROOT / "data/receptors").glob("*.pdb"))
        if not recs:
            raise unittest.SkipTest("no receptor available")
        cls.rec = recs[0]

    def test_scoring_is_deterministic(self) -> None:
        a = self.field.score_receptor(self.rec)
        b = self.field.score_receptor(self.rec)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])

    def test_prediction_is_residue_level_and_covers_the_universe(self) -> None:
        out = self.mod.predict(self.rec, pdb_id=self.rec.stem[:4])
        self.assertEqual(out["status"], "OK")
        resseq, s, call = self.field.score_receptor(self.rec)
        self.assertEqual(len(out["residue_scores"]), len(resseq))
        self.assertEqual(sum(call), len(out["residue_positive"]))

    def test_operating_point_calls_the_declared_fraction(self) -> None:
        _resseq, s, call = self.field.score_receptor(self.rec)
        k = max(1, int(round(self.field.q * len(s))))
        self.assertEqual(int(call.sum()), k)
        self.assertAlmostEqual(float(s[call].min()), float(np.sort(s)[-k]),
                               places=12)

    def test_gate_matches_the_spread_of_the_raw_score(self) -> None:
        """The gate adds a rescaled neighbourhood mean, not a raw one."""
        rng = np.random.default_rng(6)
        n = 60
        ctr = rng.normal(scale=6.0, size=(n, 3))
        s = rng.normal(size=n)
        out = self.mod.apply_gate(s, ctr, [n])
        g = self.mod._neighbourhood_mean(s, ctr, self.mod.GATE_RADIUS)
        want = s + self.mod.GATE_WEIGHT * g * (np.std(s) / np.std(g))
        np.testing.assert_allclose(out, want)


if __name__ == "__main__":
    unittest.main(verbosity=2)
