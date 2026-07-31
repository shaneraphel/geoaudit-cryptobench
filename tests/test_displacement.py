"""The displacement family, checked against inputs built to break it.

Three things make this family easy to get quietly wrong, and each has tests here
rather than a comment.

**It parses the receptor itself.** ``pdb_io`` drops the B-factor and collapses
alternates, so this module reads the fixed columns directly. A column slice off
by one produces plausible numbers -- an occupancy read as a B-factor is still a
float in a believable range -- so the parser is checked against records whose
fields are all distinct and known.

**A B-factor does not transfer between structures.** Every quantity is therefore
a within-chain rank, a ratio inside one residue, a deviation from that chain's
own median, or an integer. The tests hold that: scaling every B-factor in a chain
by a constant must not move a single column, because that is exactly what
changing resolution or refinement protocol does.

**Nothing here may be a function of residue type.** That is the screen
``AGENT_MEMORY`` 2i imposes and the test the chemistry family failed. It is
checked by relabelling every residue and requiring bit-identical output.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from pocket_bench.methods import displacement as dp  # noqa: E402


def atom(serial: int, name: str, resname: str, resseq: int, x: float,
         y: float, z: float, *, b: float = 20.0, occ: float = 1.00,
         altloc: str = " ", chain: str = "A", element: str = "C",
         icode: str = " ") -> str:
    """One ATOM record in the fixed-column format, all fields distinct."""
    return (f"ATOM  {serial:5d} {name:<4s}{altloc}{resname:>3s} {chain}"
            f"{resseq:4d}{icode}   {x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occ:6.2f}{b:6.2f}          {element:>2s}")


def chain_of(n: int, *, bs=None, spacing: float = 3.8, **kw) -> str:
    """A straight chain of ``n`` residues, one CA each, with given B-factors."""
    bs = bs if bs is not None else [20.0] * n
    lines = []
    for i in range(n):
        lines.append(atom(i + 1, "CA", "ALA", i + 1, i * spacing, 0.0, 0.0,
                          b=bs[i], **kw))
    return "\n".join(lines)


def order_of(n: int) -> list[tuple[int, str]]:
    return [(i + 1, "") for i in range(n)]


class TestTheParser(unittest.TestCase):
    """Fixed-column slicing, where an off-by-one still looks like a number."""

    def test_it_reads_b_occupancy_and_altloc_from_the_right_columns(self) -> None:
        text = atom(1, "CA", "ALA", 7, 1.0, 2.0, 3.0, b=33.75, occ=0.65,
                    altloc="B")
        got = dp.parse_displacement(text, "A")
        self.assertEqual(list(got), [(7, "")])
        a = got[(7, "")][0]
        self.assertAlmostEqual(a["b"], 33.75, places=6)
        self.assertAlmostEqual(a["occ"], 0.65, places=6)
        self.assertEqual(a["altloc"], "B")
        self.assertEqual(a["name"], "CA")
        self.assertEqual(a["xyz"], (1.0, 2.0, 3.0))

    def test_alternates_are_kept_not_collapsed(self) -> None:
        # This is the whole reason the module does not use pdb_io, which keeps
        # the highest-occupancy copy and drops the rest.
        text = "\n".join([
            atom(1, "CA", "SER", 5, 0.0, 0.0, 0.0, occ=0.60, altloc="A"),
            atom(2, "CA", "SER", 5, 1.0, 0.0, 0.0, occ=0.40, altloc="B"),
        ])
        got = dp.parse_displacement(text, "A")[(5, "")]
        self.assertEqual(len(got), 2)
        self.assertEqual({a["altloc"] for a in got}, {"A", "B"})

    def test_hydrogens_and_waters_are_dropped(self) -> None:
        text = "\n".join([
            atom(1, "CA", "ALA", 1, 0.0, 0.0, 0.0),
            atom(2, "H", "ALA", 1, 1.0, 0.0, 0.0, element="H"),
            atom(3, "O", "HOH", 2, 5.0, 0.0, 0.0, element="O"),
        ])
        got = dp.parse_displacement(text, "A")
        self.assertEqual(list(got), [(1, "")])
        self.assertEqual(len(got[(1, "")]), 1)

    def test_only_the_first_model_is_read(self) -> None:
        text = "\n".join([
            atom(1, "CA", "ALA", 1, 0.0, 0.0, 0.0, b=10.0),
            "ENDMDL",
            atom(2, "CA", "ALA", 2, 4.0, 0.0, 0.0, b=90.0),
        ])
        got = dp.parse_displacement(text, "A")
        self.assertEqual(list(got), [(1, "")])

    def test_another_chain_is_not_read(self) -> None:
        text = "\n".join([
            atom(1, "CA", "ALA", 1, 0.0, 0.0, 0.0, chain="A"),
            atom(2, "CA", "ALA", 1, 0.0, 0.0, 0.0, chain="B"),
        ])
        self.assertEqual(len(dp.parse_displacement(text, "B")), 1)


class TestScaleInvariance(unittest.TestCase):
    """What changing resolution or refinement protocol does to a B column."""

    def test_scaling_every_b_factor_moves_nothing(self) -> None:
        bs = [12.0, 40.0, 22.0, 71.0, 33.0, 18.0, 55.0]
        a = dp.compute(dp.parse_displacement(chain_of(7, bs=bs), "A"),
                       order_of(7))
        b = dp.compute(
            dp.parse_displacement(chain_of(7, bs=[x * 3.5 for x in bs]), "A"),
            order_of(7))
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_adding_a_constant_moves_only_the_within_residue_spans(self) -> None:
        # Ranks, quartiles and runs are shift-invariant. A residue's internal
        # range is not, and must not be: it is an absolute spread in one
        # structure and is only ever compared within that structure.
        bs = [12.0, 40.0, 22.0, 71.0, 33.0]
        a = dp.compute(dp.parse_displacement(chain_of(5, bs=bs), "A"),
                       order_of(5))
        b = dp.compute(
            dp.parse_displacement(chain_of(5, bs=[x + 25.0 for x in bs]), "A"),
            order_of(5))
        for name in ("b_rank_permille", "b_quartile", "b_decile",
                     "b_above_median", "b_run_length", "b_shell_rank_permille"):
            with self.subTest(column=name):
                np.testing.assert_allclose(a[:, dp.IDX[name]],
                                           b[:, dp.IDX[name]], atol=1e-9)

    def test_a_flat_b_column_gives_every_residue_the_same_rank(self) -> None:
        # Ties are averaged, so a chain refined with one B for everything must
        # not be ordered by file order.
        x = dp.compute(dp.parse_displacement(chain_of(6, bs=[25.0] * 6), "A"),
                       order_of(6))
        r = x[:, dp.IDX["b_rank_permille"]]
        self.assertEqual(len(set(np.round(r, 9))), 1)
        self.assertAlmostEqual(float(r[0]), dp.PERMILLE / 2.0, places=6)
        self.assertTrue((x[:, dp.IDX["b_robust_z_centi"]] == 0).all(),
                        "a zero MAD must give a zero robust score, not a "
                        "division by zero")


class TestIndependenceOfResidueType(unittest.TestCase):
    """The screen AGENT_MEMORY 2i imposes, and the test chemistry 42 failed."""

    def test_relabelling_every_residue_changes_nothing(self) -> None:
        bs = [15.0, 60.0, 28.0, 44.0, 33.0, 19.0]
        names = ["ALA", "TRP", "GLY", "ARG", "SER", "PHE"]
        lines_a, lines_b = [], []
        for i, (b, nm) in enumerate(zip(bs, names)):
            lines_a.append(atom(i + 1, "CA", "ALA", i + 1, i * 3.8, 0, 0, b=b))
            lines_b.append(atom(i + 1, "CA", nm, i + 1, i * 3.8, 0, 0, b=b))
        x = dp.compute(dp.parse_displacement("\n".join(lines_a), "A"),
                       order_of(6))
        y = dp.compute(dp.parse_displacement("\n".join(lines_b), "A"),
                       order_of(6))
        np.testing.assert_array_equal(x, y)


class TestOrderIndependence(unittest.TestCase):

    def test_shuffling_the_atom_records_changes_nothing(self) -> None:
        import random
        bs = [15.0, 60.0, 28.0, 44.0, 33.0, 19.0, 51.0]
        lines = chain_of(7, bs=bs).splitlines()
        x = dp.compute(dp.parse_displacement("\n".join(lines), "A"),
                       order_of(7))
        rng = random.Random(11)
        for _ in range(4):
            rng.shuffle(lines)
            y = dp.compute(dp.parse_displacement("\n".join(lines), "A"),
                           order_of(7))
            np.testing.assert_allclose(x, y, atol=1e-9)


class TestSequenceQuantities(unittest.TestCase):

    def test_a_run_of_mobile_residues_is_measured_end_to_end(self) -> None:
        # Ranks: three low, three high, three low. The high stretch is one run
        # of three and each of its members must say so.
        bs = [10, 11, 12, 90, 91, 92, 13, 14, 15]
        x = dp.compute(dp.parse_displacement(chain_of(9, bs=bs), "A"),
                       order_of(9))
        run = x[:, dp.IDX["b_run_length"]]
        self.assertEqual(list(run[3:6]), [3.0, 3.0, 3.0])
        self.assertEqual(list(run[:3]), [0.0, 0.0, 0.0])
        pos = x[:, dp.IDX["b_run_position_permille"]]
        self.assertAlmostEqual(pos[3], 0.0, places=6)
        self.assertAlmostEqual(pos[5], float(dp.PERMILLE), places=6)

    def test_a_sequence_break_does_not_join_two_stretches(self) -> None:
        # Residues 1,2,3 then 40,41,42: neighbours in the array, not in the
        # chain. A curvature computed across that gap would be meaningless.
        lines = []
        for k, (rs, b) in enumerate([(1, 10.0), (2, 90.0), (3, 11.0),
                                     (40, 12.0), (41, 91.0), (42, 13.0)]):
            lines.append(atom(k + 1, "CA", "ALA", rs, k * 3.8, 0, 0, b=b))
        order = [(1, ""), (2, ""), (3, ""), (40, ""), (41, ""), (42, "")]
        x = dp.compute(dp.parse_displacement("\n".join(lines), "A"), order)
        # Residue 3 has no sequence-adjacent successor, so its next-rank falls
        # back to the neutral middle rather than borrowing residue 40's.
        self.assertAlmostEqual(x[2, dp.IDX["b_next_permille"]],
                               dp.PERMILLE / 2.0, places=6)
        self.assertAlmostEqual(x[3, dp.IDX["b_prev_permille"]],
                               dp.PERMILLE / 2.0, places=6)

    def test_curvature_has_the_sign_a_local_peak_should_give(self) -> None:
        x = dp.compute(
            dp.parse_displacement(chain_of(3, bs=[10.0, 90.0, 11.0]), "A"),
            order_of(3))
        self.assertLess(x[1, dp.IDX["b_curvature_centi"]], 0.0,
                        "a mobility peak between two rigid neighbours must "
                        "give a negative second difference")


class TestShellQuantities(unittest.TestCase):

    def test_the_most_mobile_residue_in_a_cluster_is_its_shell_max(self) -> None:
        # Five residues inside one 8 A ball, so every one contacts every other.
        pts = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (2, 2, 0)]
        bs = [10.0, 20.0, 30.0, 95.0, 40.0]
        lines = [atom(i + 1, "CA", "ALA", i + 1, *p, b=b)
                 for i, (p, b) in enumerate(zip(pts, bs))]
        x = dp.compute(dp.parse_displacement("\n".join(lines), "A"),
                       order_of(5))
        is_max = x[:, dp.IDX["b_is_shell_max"]]
        self.assertEqual(list(is_max), [0.0, 0.0, 0.0, 1.0, 0.0])
        self.assertGreater(x[3, dp.IDX["b_minus_shell_centi"]], 0.0)
        self.assertEqual(x[3, dp.IDX["b_contrast_sign"]], 1.0)

    def test_an_isolated_residue_takes_the_neutral_shell_rank(self) -> None:
        lines = [atom(1, "CA", "ALA", 1, 0, 0, 0, b=10.0),
                 atom(2, "CA", "ALA", 2, 50, 0, 0, b=90.0)]
        x = dp.compute(dp.parse_displacement("\n".join(lines), "A"),
                       [(1, ""), (2, "")])
        self.assertTrue((x[:, dp.IDX["b_shell_size"]] == 0).all())
        self.assertTrue(
            (x[:, dp.IDX["b_shell_rank_permille"]] == dp.PERMILLE / 2).all())


class TestAlternateConformers(unittest.TestCase):

    def _two_alternates(self, gap: float, occ=(0.6, 0.4)) -> np.ndarray:
        lines = [
            atom(1, "CA", "SER", 1, 0.0, 0.0, 0.0, occ=occ[0], altloc="A"),
            atom(2, "CA", "SER", 1, 0.0, 0.0, 0.0, occ=occ[1], altloc="B"),
            atom(3, "CB", "SER", 1, 1.5, 0.0, 0.0, occ=occ[0], altloc="A"),
            atom(4, "CB", "SER", 1, 1.5, gap, 0.0, occ=occ[1], altloc="B"),
        ]
        return dp.compute(dp.parse_displacement("\n".join(lines), "A"),
                          [(1, "")])

    def test_counts_and_labels(self) -> None:
        x = self._two_alternates(2.0)
        self.assertEqual(x[0, dp.IDX["alt_atoms"]], 4.0)
        self.assertEqual(x[0, dp.IDX["alt_labels"]], 2.0)
        self.assertEqual(x[0, dp.IDX["alt_backbone"]], 2.0)
        self.assertEqual(x[0, dp.IDX["alt_sidechain"]], 2.0)

    def test_displacement_measures_how_far_apart_the_copies_sit(self) -> None:
        # The quantity that separates a refinement detail from a side chain
        # genuinely occupying two positions.
        near = self._two_alternates(0.2)
        far = self._two_alternates(3.0)
        self.assertAlmostEqual(near[0, dp.IDX["alt_max_displacement_centi"]],
                               0.2 * dp.CENTI, places=4)
        self.assertAlmostEqual(far[0, dp.IDX["alt_max_displacement_centi"]],
                               3.0 * dp.CENTI, places=4)

    def test_occupancy_spread(self) -> None:
        x = self._two_alternates(1.0, occ=(0.75, 0.25))
        self.assertAlmostEqual(x[0, dp.IDX["alt_occupancy_spread_centi"]],
                               0.5 * dp.CENTI, places=4)

    def test_a_residue_without_alternates_carries_none_of_them(self) -> None:
        x = dp.compute(dp.parse_displacement(chain_of(3), "A"), order_of(3))
        for name in ("alt_atoms", "alt_labels", "alt_backbone", "alt_sidechain",
                     "alt_max_displacement_centi", "alt_cluster_size"):
            with self.subTest(column=name):
                self.assertTrue((x[:, dp.IDX[name]] == 0).all())

    def test_touching_disordered_residues_form_one_cluster(self) -> None:
        # Three alternate-bearing residues inside one ball and a fourth far
        # away: one cluster of three, one of one.
        lines = []
        s = 1
        for i, (p, has) in enumerate([((0, 0, 0), True), ((2, 0, 0), True),
                                      ((0, 2, 0), True), ((40, 0, 0), True)]):
            for al, occ in (("A", 0.5), ("B", 0.5)) if has else ((" ", 1.0),):
                lines.append(atom(s, "CA", "SER", i + 1, *p, occ=occ,
                                  altloc=al))
                s += 1
        x = dp.compute(dp.parse_displacement("\n".join(lines), "A"),
                       order_of(4))
        self.assertEqual(list(x[:, dp.IDX["alt_cluster_size"]]),
                         [3.0, 3.0, 3.0, 1.0])


class TestOccupancy(unittest.TestCase):

    def test_alternates_summing_to_one_are_not_partial_occupancy(self) -> None:
        # Without summing per atom name, every alternate would look like a
        # partial-occupancy atom and group F would be a copy of group E.
        lines = [
            atom(1, "CA", "SER", 1, 0.0, 0.0, 0.0, occ=0.6, altloc="A"),
            atom(2, "CA", "SER", 1, 0.5, 0.0, 0.0, occ=0.4, altloc="B"),
        ]
        x = dp.compute(dp.parse_displacement("\n".join(lines), "A"), [(1, "")])
        self.assertEqual(x[0, dp.IDX["occ_partial_atoms"]], 0.0)
        self.assertEqual(x[0, dp.IDX["occ_is_full"]], 1.0)
        self.assertAlmostEqual(x[0, dp.IDX["occ_deficit_centi"]], 0.0, places=4)

    def test_a_genuinely_partial_atom_is_reported(self) -> None:
        text = atom(1, "CA", "ALA", 1, 0.0, 0.0, 0.0, occ=0.5)
        x = dp.compute(dp.parse_displacement(text, "A"), [(1, "")])
        self.assertEqual(x[0, dp.IDX["occ_partial_atoms"]], 1.0)
        self.assertEqual(x[0, dp.IDX["occ_is_full"]], 0.0)
        self.assertAlmostEqual(x[0, dp.IDX["occ_deficit_centi"]],
                               0.5 * dp.CENTI, places=4)


class TestConsistencyCatchesViolations(unittest.TestCase):
    """The invariants must be able to fail, or they are decoration."""

    def setUp(self) -> None:
        self.x = dp.compute(
            dp.parse_displacement(chain_of(6, bs=[10, 90, 20, 80, 30, 70]), "A"),
            order_of(6))
        self.order = order_of(6)

    def test_a_clean_chain_passes(self) -> None:
        self.assertEqual(dp.consistency(self.x, self.order), [])

    def test_a_rank_outside_its_range_is_caught(self) -> None:
        bad = self.x.copy()
        bad[0, dp.IDX["b_rank_permille"]] = 1400.0
        self.assertTrue(any("b_rank_permille" in c
                            for c in dp.consistency(bad, self.order)))

    def test_a_non_indicator_indicator_is_caught(self) -> None:
        bad = self.x.copy()
        bad[0, dp.IDX["b_above_median"]] = 0.5
        self.assertTrue(any("indicator" in c
                            for c in dp.consistency(bad, self.order)))

    def test_a_fractional_count_is_caught(self) -> None:
        bad = self.x.copy()
        bad[0, dp.IDX["alt_atoms"]] = 2.5
        self.assertTrue(any("integer count" in c
                            for c in dp.consistency(bad, self.order)))

    def test_an_alternate_field_on_a_residue_with_no_alternate_is_caught(self) -> None:
        bad = self.x.copy()
        bad[0, dp.IDX["alt_max_displacement_centi"]] = 42.0
        self.assertTrue(any("no alternate" in c
                            for c in dp.consistency(bad, self.order)))

    def test_the_backbone_sidechain_split_must_add_up(self) -> None:
        bad = self.x.copy()
        bad[0, dp.IDX["alt_atoms"]] = 4.0
        bad[0, dp.IDX["alt_backbone"]] = 1.0
        bad[0, dp.IDX["alt_sidechain"]] = 1.0
        self.assertTrue(any("alt_backbone + alt_sidechain" in c
                            for c in dp.consistency(bad, self.order)))

    def test_a_non_finite_value_is_caught(self) -> None:
        bad = self.x.copy()
        bad[0, 0] = np.nan
        self.assertTrue(any("non-finite" in c
                            for c in dp.consistency(bad, self.order)))

    def test_a_row_count_mismatch_is_caught(self) -> None:
        self.assertTrue(any("rows for" in c
                            for c in dp.consistency(self.x, self.order[:3])))


class TestColumnHygiene(unittest.TestCase):

    def test_the_names_are_unique_and_indexed(self) -> None:
        self.assertEqual(len(dp.COLUMNS), len(set(dp.COLUMNS)))
        self.assertEqual(len(dp.COLUMNS), dp.N_COLUMNS)
        self.assertEqual(set(dp.IDX), set(dp.COLUMNS))

    def test_every_shell_level_name_is_a_real_column(self) -> None:
        self.assertTrue(dp.SHELL_LEVEL <= set(dp.COLUMNS))

    def test_an_empty_chain_returns_an_empty_block(self) -> None:
        x = dp.compute({}, [])
        self.assertEqual(x.shape, (0, dp.N_COLUMNS))
        self.assertEqual(dp.consistency(x, []), [])


class TestOnRealChains(unittest.TestCase):
    """The invariants on deposited files, where the corner cases actually live."""

    @classmethod
    def setUpClass(cls) -> None:
        import json
        manifest = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
        if not manifest.is_file():
            raise unittest.SkipTest("training manifest not materialised")
        cls.entries = json.loads(manifest.read_text())["entries"][:25]
        if not cls.entries:
            raise unittest.SkipTest("no entries")

    def test_the_invariants_hold_and_the_inputs_are_present(self) -> None:
        import displacement_wires as dw
        from pocket_bench.pdb_io import parse_pdb_atoms
        n_varying_b = 0
        for e in self.entries:
            path = ROOT / e["receptor_path"]
            if not path.is_file():
                continue
            text = path.read_text()
            order, _ctr, take = dw._residue_rows(parse_pdb_atoms(text),
                                                 e["chain"])
            res = dp.parse_displacement(text, e["chain"])
            x = dp.compute(res, order)[take]
            sub = [order[k] for k in take]
            with self.subTest(unit=f"{e['pdb']}_{e['chain']}"):
                self.assertEqual(dp.consistency(x, sub), [])
            bs = [a["b"] for g in res.values() for a in g]
            if bs and max(bs) - min(bs) > 1e-9:
                n_varying_b += 1
        self.assertEqual(n_varying_b, len(self.entries),
                         "a chain with a flat B column would make this family "
                         "null for a reason that is not about the screen")


if __name__ == "__main__":
    unittest.main()
