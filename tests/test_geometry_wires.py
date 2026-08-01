"""The receptor-side assembler must reproduce the cached matrices exactly.

What this is for
----------------
The four wire families were measured by tools that walk the training manifest and
index the training wide cache. Scoring an external unit needs the same 624
columns computed from a receptor file instead, and
``pocket_bench.methods.geometry_wires`` does that.

Nothing in that module is a new quantity --- every one comes from the same
``compute`` in the same module the builders call --- so the only way it can be
wrong is by attaching a correct number to the wrong residue, or by ordering the
columns differently from the block the lift was measured on. Both failures are
silent. Every lookup succeeds, every score is plausible, and the external read
would be of a different detector than the one the training fold endorsed.

So the requirement here is equality and not agreement. Both paths run the same
float operations in the same order on the same coordinates, so a difference of
any size means they are not computing the same thing, and a tolerance would hide
exactly the misalignment that matters --- a shifted row still lands within
tolerance on a column whose values are all of similar size.

This test is the warrant for reading a held-out set with a field compiled over
these columns. If it is skipped for want of receptors, nothing downstream of it
is established.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from pocket_bench.methods import geometry_wires as gw  # noqa: E402
import receptor_fixture  # noqa: E402

# The four caches the builders wrote, and how many of each cache's columns the
# assembler carries. Displacement contributes its first 32 quantities at each of
# three aggregations; the other 16 measured null.
CACHES = (
    ("backbone", "_backbone_cache_train.npz", None),
    ("sidechain", "_sidechain_cache_train.npz", None),
    ("void", "_void_cache_train.npz", None),
    ("displacement", "_displacement_cache_train.npz", gw.N_DISP_CARRIED),
)
CACHE_DIR = ROOT / "data/cryptobench_apo"
WIDE = CACHE_DIR / "data/cryptobench_apo/_wide_cache_train.npz"
WIDE = CACHE_DIR / "_wide_cache_train.npz"

N_CHAINS = 6


def _cached_block(name: str, fname: str, carry: int | None,
                  offset: int, n: int) -> np.ndarray:
    """One chain's rows of one cached family, restricted to carried columns."""
    z = np.load(CACHE_DIR / fname, allow_pickle=False)
    C = z["C"][offset:offset + n]
    z.close()
    if carry is None:
        return np.asarray(C, dtype=np.float64)
    per_agg = C.shape[1] // 3
    take = [a * per_agg + q for a in range(3) for q in range(carry)]
    return np.asarray(C[:, take], dtype=np.float64)


class TestTheColumnLayout(unittest.TestCase):

    def test_the_count_is_what_the_measurement_used(self) -> None:
        self.assertEqual(gw.N_COLUMNS, 624)
        self.assertEqual(len(gw.COLUMNS), 624)

    def test_the_names_are_unique(self) -> None:
        self.assertEqual(len(set(gw.COLUMNS)), len(gw.COLUMNS))

    def test_the_families_are_in_the_order_the_lift_was_measured_in(self) -> None:
        got = [c.split("~")[0] for c in gw.COLUMNS]
        want = (["backbone"] * 132 + ["sidechain"] * 261 + ["void"] * 135
                + ["displacement"] * 96)
        self.assertEqual(got, want)

    def test_each_family_is_aggregation_major_within_itself(self) -> None:
        # own, then contact, then walk2 -- because a family's block is one
        # builder's output and that is how each builder concatenates.
        bb = [c.split("~")[1] for c in gw.COLUMNS if c.startswith("backbone~")]
        self.assertEqual(bb[:44], ["own"] * 44)
        self.assertEqual(bb[44:88], ["contact"] * 44)
        self.assertEqual(bb[88:], ["walk2"] * 44)

    def test_the_null_displacement_group_is_not_carried(self) -> None:
        carried = {c.split("~")[2] for c in gw.COLUMNS
                   if c.startswith("displacement~")}
        for q in ("alt_atoms", "alt_cluster_size", "occ_partial_atoms"):
            self.assertNotIn(q, carried,
                             "the alternate-conformer group measured +0.00052 "
                             "against its own control at +0.00052 and must not "
                             "be carried")
        for q in ("b_rank_permille", "b_shell_rank_permille"):
            self.assertIn(q, carried)


class TestItReproducesTheCachedMatrices(unittest.TestCase):
    """Equality, not agreement. A tolerance would hide a shifted row."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.entries = receptor_fixture.entries(N_CHAINS)
        missing = [f for _n, f, _c in CACHES
                   if not (CACHE_DIR / f).is_file()]
        if missing:
            raise unittest.SkipTest(
                f"the family caches {missing} are not built, so there is "
                f"nothing to compare against. Build them with the four "
                f"tools/*_wires.py, each of which takes about half a minute. "
                f"This is a skip and not a pass: the receptor-side assembler "
                f"is unverified in this run, and every external score computed "
                f"through it is unwarranted.")
        if not WIDE.is_file():
            raise unittest.SkipTest(f"{WIDE.name} is not built")
        z = np.load(WIDE, allow_pickle=False)
        cls.units = [str(u) for u in z["units"]]
        cls.n_res = [int(v) for v in z["n_res_per"]]
        z.close()
        cls.offset = {}
        off = 0
        for u, n in zip(cls.units, cls.n_res):
            cls.offset[u] = (off, n)
            off += n

    def test_every_family_block_matches_row_for_row(self) -> None:
        for e in self.entries:
            unit = f"{e['pdb']}_{e['chain']}"
            if unit not in self.offset:
                continue
            off, n = self.offset[unit]
            _resseq, X = gw.geometry_columns(ROOT / e["receptor_path"],
                                             e["chain"])
            self.assertEqual(X.shape, (n, gw.N_COLUMNS),
                             f"{unit}: the assembler returns {X.shape[0]} rows "
                             f"and the wide cache holds {n}")
            col = 0
            for fam, fname, carry in CACHES:
                want = _cached_block(fam, fname, carry, off, n)
                got = X[:, col:col + want.shape[1]]
                with self.subTest(unit=unit, family=fam):
                    # float32 in the cache, float64 here: compare at the cache's
                    # precision, which is the precision the lift was measured at.
                    np.testing.assert_array_equal(
                        got.astype(np.float32), want.astype(np.float32),
                        err_msg=(f"{unit}: the {fam} block computed from the "
                                 f"receptor differs from the cached matrix the "
                                 f"lift was measured on"))
                col += want.shape[1]
            self.assertEqual(col, gw.N_COLUMNS)

    def test_the_residue_universe_matches_the_cache(self) -> None:
        # The failure this catches is the one that does not raise: a chain whose
        # residue count agrees while the rows are in a different order.
        z = np.load(WIDE, allow_pickle=False)
        ctr_cached = z["ctr"]
        z.close()
        for e in self.entries:
            unit = f"{e['pdb']}_{e['chain']}"
            if unit not in self.offset:
                continue
            off, n = self.offset[unit]
            from pocket_bench.pdb_io import parse_pdb_atoms
            atoms = parse_pdb_atoms((ROOT / e["receptor_path"]).read_text())
            _order, ctr, take = gw.residue_rows(atoms, e["chain"])
            with self.subTest(unit=unit):
                self.assertEqual(len(take), n)
                np.testing.assert_allclose(ctr, ctr_cached[off:off + n],
                                           atol=5e-4)


class TestItRunsOnAnUnseenReceptor(unittest.TestCase):
    """The case the module exists for: a chain with no row in any cache."""

    def test_it_produces_finite_columns_on_an_external_receptor(self) -> None:
        ext = ROOT / "data/external/receptors"
        if not ext.is_dir():
            raise unittest.SkipTest("the external receptors are not built")
        files = sorted(ext.glob("*_receptor.pdb"))[:3]
        if not files:
            raise unittest.SkipTest("no external receptors on disk")
        for path in files:
            chain = path.name.split("_")[1]
            with self.subTest(receptor=path.name):
                resseq, X = gw.geometry_columns(path, chain)
                self.assertEqual(X.shape[1], gw.N_COLUMNS)
                self.assertEqual(X.shape[0], len(resseq))
                self.assertTrue(np.isfinite(X).all(),
                                "a non-finite column would make the digitiser's "
                                "quartiles meaningless without raising")


if __name__ == "__main__":
    unittest.main()
