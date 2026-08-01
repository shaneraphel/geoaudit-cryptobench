"""The native addressing kernel must equal the NumPy loop exactly, not closely.

Why equality and not a tolerance
--------------------------------
``table_bank.addresses`` returns indices into a concatenated cell array. An
address off by one is a different cell holding a different frequency, and there is
no numerical smallness to make that visible: the score moves by whatever the two
cells happen to differ by, which could be anything. So the port is written to be
bit-identical --- integer throughout, the same accumulation order per row, and
parallel only over disjoint output rows --- and this file holds that rather than
measuring how close two answers are.

The existing kernels in the same library make the same promise for the same
reason, and ``tools/build_native.sh`` says in its own header that a run with the
library present and a run without it produce the same numbers. This is the file
that makes that sentence checkable for the fourth kernel.

What is exercised
-----------------
The shapes the pipeline actually produces (a bank of thousands of width-2 tables
over hundreds of digit columns), the boundaries a block loop hits (a single row,
a block shorter than the thread count, an offset window in the middle of the
array), and the cases where the kernel must refuse and the caller must fall back
rather than guess. A kernel that silently returned something for a bank it does
not handle would be worse than no kernel.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocket_bench import native  # noqa: E402
from pocket_bench.methods.table_bank import (  # noqa: E402
    N_LEVELS, cell_offsets,
    addresses as _addresses_numpy,
)
# The accelerated pair lives outside the eight files under the frozen field's code
# digest. ``table_bank.addresses`` is therefore the reference loop itself, and the
# equality asserted below is between two modules rather than two branches of one.
from pocket_bench.methods.table_bank_accel import addresses  # noqa: E402


def _bank(n_cols: int, n_tables: int, width: int = 2, seed: int = 11):
    rng = np.random.default_rng(seed)
    tables = [tuple(int(c) for c in rng.integers(0, n_cols, size=width))
              for _ in range(n_tables)]
    return tables, cell_offsets(tables)


def _digits(n_rows: int, n_cols: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, N_LEVELS, size=(n_rows, n_cols), dtype=np.int8)


class TestTheKernelIsPresent(unittest.TestCase):

    def test_the_library_is_built(self) -> None:
        if not native.available():
            raise unittest.SkipTest(
                "the native library is not built, so this test has nothing to "
                "compare against. Build it with tools/build_native.sh. This is "
                "a skip and not a pass: nothing about the kernel's equality "
                "with the NumPy loop is established in this run, and every "
                "score computed through it is unverified.")

    def test_the_symbol_is_bound(self) -> None:
        if not native.available():
            raise unittest.SkipTest("native library not built")
        self.assertTrue(hasattr(native, "table_addresses"))


class TestBitIdentity(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if not native.available():
            raise unittest.SkipTest("native library not built")

    def _check(self, n_rows: int, n_cols: int, n_tables: int, width: int = 2,
               a: int = 0, b: int | None = None) -> None:
        b = n_rows if b is None else b
        D = _digits(n_rows, n_cols)
        tables, offs = _bank(n_cols, n_tables, width)
        ref = _addresses_numpy(D, tables, offs, a, b)
        got = native.table_addresses(
            D, np.array(tables, dtype=np.int32), offs, N_LEVELS, a, b)
        self.assertIsNotNone(got, "the kernel refused a bank it should handle")
        self.assertEqual(got.dtype, np.int64)
        np.testing.assert_array_equal(
            got, ref,
            err_msg="an address off by one is a different cell and a different "
                    "score; there is no tolerance under which this is fine")

    def test_the_shape_the_pipeline_produces(self) -> None:
        # 645 wires plus 624 geometry columns digitise to 789 columns, and the
        # deployed bank plus one family's additions runs to a few thousand tables.
        self._check(4096, 789, 6304)

    def test_the_widest_bank_in_use(self) -> None:
        self._check(2048, 1317, 10144)

    def test_a_single_row(self) -> None:
        self._check(1, 64, 40)

    def test_fewer_rows_than_threads(self) -> None:
        # The kernel chunks rows over threads; a block shorter than the thread
        # count must not produce an empty chunk or an out-of-range slice.
        for n in (2, 3, 5, 7):
            with self.subTest(n_rows=n):
                self._check(n, 32, 25)

    def test_a_window_in_the_middle(self) -> None:
        # The block loop calls this with a > 0. Reading from the wrong offset
        # would give a correct-looking matrix of the wrong rows.
        self._check(1000, 48, 60, a=137, b=612)

    def test_an_empty_window(self) -> None:
        D = _digits(100, 32)
        tables, offs = _bank(32, 20)
        got = native.table_addresses(D, np.array(tables, dtype=np.int32), offs,
                                     N_LEVELS, 50, 50)
        self.assertIsNotNone(got)
        self.assertEqual(got.shape, (0, 20))

    def test_width_three(self) -> None:
        # Width 2 is deployed and width 3 was measured and lost, but the kernel
        # is written for any width and a silent wrong answer at width 3 would
        # only surface if somebody revisited that sweep.
        self._check(512, 40, 80, width=3)

    def test_every_digit_level_is_exercised(self) -> None:
        # A kernel that dropped the high digit would agree on most rows.
        D = np.tile(np.arange(N_LEVELS, dtype=np.int8), (8, 4))
        n_cols = D.shape[1]
        tables, offs = _bank(n_cols, 12)
        ref = _addresses_numpy(D, tables, offs, 0, len(D))
        got = native.table_addresses(D, np.array(tables, dtype=np.int32), offs,
                                     N_LEVELS, 0, len(D))
        np.testing.assert_array_equal(got, ref)


class TestItRefusesRatherThanGuesses(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if not native.available():
            raise unittest.SkipTest("native library not built")

    def test_a_column_index_past_the_end_is_refused(self) -> None:
        # Reading past a row would silently pick up the next row's digits, which
        # is the one failure a bit-identity test on valid input cannot catch.
        D = _digits(64, 16)
        tables = [(0, 1), (2, 99)]
        offs = cell_offsets(tables)
        got = native.table_addresses(D, np.array(tables, dtype=np.int32), offs,
                                     N_LEVELS, 0, 64)
        self.assertIsNone(got, "the kernel accepted an out-of-range column")

    def test_a_negative_column_index_is_refused(self) -> None:
        D = _digits(64, 16)
        tables = [(0, 1), (-3, 4)]
        offs = cell_offsets(tables)
        got = native.table_addresses(D, np.array(tables, dtype=np.int32), offs,
                                     N_LEVELS, 0, 64)
        self.assertIsNone(got)


class TestTheCallerFallsBack(unittest.TestCase):
    """``addresses`` must agree with its own reference whatever path it takes."""

    def test_a_mixed_width_bank_goes_through_numpy_and_still_agrees(self) -> None:
        # partition_tables can emit tables of unequal width; the kernel takes a
        # single width, so the caller has to notice and fall back.
        D = _digits(256, 32)
        tables = [(0, 1), (2, 3, 4), (5, 6)]
        offs = cell_offsets(tables)
        np.testing.assert_array_equal(
            addresses(D, tables, offs, 0, 256),
            _addresses_numpy(D, tables, offs, 0, 256))

    def test_the_public_function_agrees_on_the_deployed_shape(self) -> None:
        D = _digits(2048, 789)
        tables, offs = _bank(789, 5152)
        np.testing.assert_array_equal(
            addresses(D, tables, offs, 0, 2048),
            _addresses_numpy(D, tables, offs, 0, 2048))

    def test_a_bank_of_one_table_still_works(self) -> None:
        D = _digits(128, 8)
        tables, offs = _bank(8, 1)
        np.testing.assert_array_equal(
            addresses(D, tables, offs, 0, 128),
            _addresses_numpy(D, tables, offs, 0, 128))


class TestDeterminismUnderThreading(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if not native.available():
            raise unittest.SkipTest("native library not built")

    def test_repeated_calls_agree_exactly(self) -> None:
        # Parallelism is over disjoint output rows, so the answer cannot depend
        # on scheduling. Ten calls is not a proof and the property is structural;
        # this catches a future edit that shares an accumulator between threads.
        D = _digits(4096, 128)
        tables, offs = _bank(128, 900)
        first = native.table_addresses(D, np.array(tables, dtype=np.int32), offs,
                                       N_LEVELS, 0, 4096)
        for _ in range(9):
            again = native.table_addresses(
                D, np.array(tables, dtype=np.int32), offs, N_LEVELS, 0, 4096)
            np.testing.assert_array_equal(again, first)


if __name__ == "__main__":
    unittest.main()
