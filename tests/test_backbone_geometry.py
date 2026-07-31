"""The backbone quantities, checked against geometry that is known in advance.

Two of these tests exist because the first version of the module failed them on
real structures and the failure was legible rather than subtle, which is the only
reason it was caught:

* the dihedral's sine had the wrong sign, so every right-handed helical residue
  landed in the left-handed cell of the Ramachandran partition and one 254-residue
  chain came back with 127 left-handed residues. A protein has a few percent.
* the hydrogen-bond direction test was inverted -- it required the nitrogen to sit
  on the carbonyl carbon's side of the oxygen rather than beyond it -- so the
  criterion was satisfied by any carbonyl that merely passed close, and the modal
  donated sequence lag through a helix came out at 2 instead of the 4 an alpha
  turn fixes.

Neither would have raised. Both are caught by asking the module for a number
whose value is fixed by the geometry of an ideal helix, so that is what these
tests do, on synthetic coordinates rather than on a deposited structure: the
official receptors are not redistributed with the repository and a test that
silently skips is not a test.
"""
from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocket_bench.methods import backbone_geometry as bg  # noqa: E402

MODULE = ROOT / "src/pocket_bench/methods/backbone_geometry.py"


def ideal_helix_ca(n: int, rise: float = 1.5, radius: float = 2.3,
                   turn_deg: float = 100.0) -> np.ndarray:
    """The CA trace of an ideal alpha helix, as a right-handed spiral."""
    t = np.radians(turn_deg) * np.arange(n)
    return np.column_stack([radius * np.cos(t), radius * np.sin(t),
                            rise * np.arange(n)])


def as_chain(ca: np.ndarray, **rest) -> dict[str, np.ndarray]:
    n = len(ca)
    bb = {a: np.full((n, 3), np.nan) for a in bg.BACKBONE_ATOMS + ("CB",)}
    bb["CA"] = ca
    for k, v in rest.items():
        bb[k] = v
    return bb


# Ideal peptide internal coordinates, in angstrom and degrees. These are the
# textbook values, not fitted here.
BOND = {"C_N": 1.329, "N_CA": 1.458, "CA_C": 1.525, "C_O": 1.231}
ANGLE = {"CA_C_N": 116.2, "C_N_CA": 121.7, "N_CA_C": 111.2, "CA_C_O": 120.8}


def _place(a: np.ndarray, b: np.ndarray, c: np.ndarray,
           length: float, angle_deg: float, torsion_deg: float) -> np.ndarray:
    """The next atom, from three placed ones and one internal coordinate.

    The natural-extension-reference-frame construction. Building a backbone
    forwards from bond lengths, angles and torsions and then asking the module to
    read the torsions back is a far stronger test than assembling plausible
    coordinates by hand: the answer is known before the module runs, and a
    hand-built backbone is not even bonded -- the first version of these tests
    produced C-to-N distances between 1.08 and 3.57 A, so the module's chain-break
    rule correctly refused every torsion and the test blamed the module.
    """
    theta, chi = np.radians(angle_deg), np.radians(torsion_deg)
    bc = c - b
    bc = bc / np.linalg.norm(bc)
    nrm = np.cross(b - a, bc)
    nrm = nrm / np.linalg.norm(nrm)
    m = np.stack([bc, np.cross(nrm, bc), nrm])
    d = np.array([-length * np.cos(theta),
                  length * np.sin(theta) * np.cos(chi),
                  length * np.sin(theta) * np.sin(chi)])
    return c + d @ m


def ideal_peptide(n: int, phi: float, psi: float, omega: float = 180.0
                  ) -> dict[str, np.ndarray]:
    """A backbone with every residue at the same (phi, psi)."""
    N = [np.array([0.0, 0.0, 0.0])]
    CA = [np.array([BOND["N_CA"], 0.0, 0.0])]
    C = [CA[0] + np.array([BOND["CA_C"] * np.cos(np.radians(180 - ANGLE["N_CA_C"])),
                           BOND["CA_C"] * np.sin(np.radians(180 - ANGLE["N_CA_C"])),
                           0.0])]
    for _ in range(n - 1):
        N.append(_place(N[-1], CA[-1], C[-1], BOND["C_N"], ANGLE["CA_C_N"], psi))
        CA.append(_place(CA[-1], C[-1], N[-1], BOND["N_CA"], ANGLE["C_N_CA"],
                         omega))
        C.append(_place(C[-1], N[-1], CA[-1], BOND["CA_C"], ANGLE["N_CA_C"], phi))
    N, CA, C = np.array(N), np.array(CA), np.array(C)
    O = np.array([_place(N[i], CA[i], C[i], BOND["C_O"], ANGLE["CA_C_O"],
                         psi + 180.0) for i in range(n)])
    return as_chain(CA, N=N, C=C, O=O)


class TestTheDihedralSignConvention(unittest.TestCase):
    """The bug that put 127 of 254 residues in the left-handed helical cell."""

    def _dihedral_of(self, deg: float) -> float:
        t = np.radians(deg)
        p0 = np.array([[1.0, 0.0, 0.0]])
        p1 = np.array([[0.0, 0.0, 0.0]])
        p2 = np.array([[0.0, 0.0, 1.0]])
        p3 = np.array([[np.cos(t), np.sin(t), 1.0]])
        cs, sn = bg._dihedral(p0, p1, p2, p3)
        return float(np.degrees(np.arctan2(sn, cs))[0])

    def test_the_sign_follows_iupac(self) -> None:
        for want in (0, 60, -60, 120, -120, 179, -179, 57, -57, 90, -90):
            got = self._dihedral_of(want)
            self.assertAlmostEqual(
                ((got - want + 180) % 360) - 180, 0.0, places=6,
                msg=f"dihedral of {want} came back {got}; a global sign error "
                    f"here exchanges the two helical cells")

    def test_the_pair_is_on_the_unit_circle(self) -> None:
        cs, sn = bg._dihedral(
            np.array([[1.0, 0.2, 0.0]]), np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.1, 1.0]]), np.array([[0.4, 0.9, 1.3]]))
        self.assertAlmostEqual(float(cs[0] ** 2 + sn[0] ** 2), 1.0, places=9)


class TestTheCaTraceOfAnIdealHelix(unittest.TestCase):
    """Turn and torsion need only CA, so they are checkable in isolation."""

    def setUp(self) -> None:
        self.x = bg.compute(as_chain(ideal_helix_ca(20)))
        self.col = {n: j for j, n in enumerate(bg.COLUMNS)}

    def test_the_turn_angle_is_the_helical_one(self) -> None:
        turn = self.x[2:-2, self.col["ca_turn"]]
        self.assertTrue(np.all(np.abs(turn - 91.0) < 4.0),
                        f"an ideal helix should turn about 89-92 degrees at "
                        f"each CA; got {turn[:4]}")

    def test_the_torsion_is_positive_and_right_handed(self) -> None:
        cs = self.x[2:-3, self.col["cos_ca_tor"]]
        sn = self.x[2:-3, self.col["sin_ca_tor"]]
        tor = np.degrees(np.arctan2(sn, cs))
        self.assertTrue(np.all(tor > 30) and np.all(tor < 70),
                        f"an ideal right-handed helix has a CA torsion near "
                        f"+50 degrees; got {tor[:4]}")

    def test_the_mirror_image_is_left_handed(self) -> None:
        """The one quantity here that can tell a helix from its reflection."""
        mirror = ideal_helix_ca(20)
        mirror[:, 2] *= -1.0
        x = bg.compute(as_chain(mirror))
        sn = x[2:-3, self.col["sin_ca_tor"]]
        self.assertTrue(np.all(sn < 0),
                        "reflecting the helix must flip the sign of the CA "
                        "torsion, or the family cannot see handedness at all")

    def test_a_straight_chain_turns_by_half_a_circle(self) -> None:
        line = np.column_stack([np.arange(8, dtype=float) * 3.8,
                                np.zeros(8), np.zeros(8)])
        x = bg.compute(as_chain(line))
        turn = x[1:-1, self.col["ca_turn"]]
        self.assertTrue(np.allclose(turn, 180.0, atol=1e-6))


class TestTorsionsFromTheFullBackbone(unittest.TestCase):

    PHI, PSI = -57.0, -47.0

    def _peptide(self, n: int) -> dict[str, np.ndarray]:
        return ideal_peptide(n, self.PHI, self.PSI)

    def test_the_chain_it_builds_is_actually_bonded(self) -> None:
        """Otherwise every assertion below would pass for the wrong reason."""
        bb = self._peptide(8)
        d = np.linalg.norm(bb["N"][1:] - bb["C"][:-1], axis=1)
        self.assertTrue(np.all(np.abs(d - BOND["C_N"]) < 1e-6), d)
        self.assertLess(float(d.max()), bg.PEPTIDE_BOND_MAX)

    def test_the_torsions_come_back_as_they_were_built(self) -> None:
        """The strongest statement available: phi and psi are recovered."""
        x = bg.compute(self._peptide(10))
        col = {n: j for j, n in enumerate(bg.COLUMNS)}
        phi = np.degrees(np.arctan2(x[1:, col["sin_phi"]],
                                    x[1:, col["cos_phi"]]))
        psi = np.degrees(np.arctan2(x[:-1, col["sin_psi"]],
                                    x[:-1, col["cos_psi"]]))
        self.assertTrue(np.all(np.abs(phi - self.PHI) < 0.05), phi)
        self.assertTrue(np.all(np.abs(psi - self.PSI) < 0.05), psi)

    def test_an_alpha_backbone_lands_in_the_right_handed_cell(self) -> None:
        x = bg.compute(self._peptide(10))
        col = {n: j for j, n in enumerate(bg.COLUMNS)}
        cell = x[1:-1, col["rama_region"]]
        self.assertTrue(np.all(cell == bg.RAMA_ALPHA_R), cell)

    def test_a_beta_backbone_lands_in_the_extended_cell(self) -> None:
        x = bg.compute(ideal_peptide(10, -139.0, 135.0))
        col = {n: j for j, n in enumerate(bg.COLUMNS)}
        cell = x[1:-1, col["rama_region"]]
        self.assertTrue(np.all(cell == bg.RAMA_BETA), cell)

    def test_the_mirror_backbone_lands_in_the_left_handed_cell(self) -> None:
        x = bg.compute(ideal_peptide(10, 57.0, 47.0))
        col = {n: j for j, n in enumerate(bg.COLUMNS)}
        cell = x[1:-1, col["rama_region"]]
        self.assertTrue(np.all(cell == bg.RAMA_ALPHA_L), cell)

    def test_the_alpha_helix_donates_at_lag_four(self) -> None:
        """The check that caught the inverted hydrogen-bond direction test."""
        x = bg.compute(self._peptide(14))
        col = {n: j for j, n in enumerate(bg.COLUMNS)}
        lag = x[:, col["hb_lag"]]
        donated = lag[lag > 0]
        self.assertGreater(len(donated), 4, "an ideal alpha helix of 14 "
                                            "residues should donate several "
                                            "backbone hydrogen bonds")
        # The first donor of a helix takes lag 3: its i+4 partner is off the end
        # of the chain, so the nearest acceptable carbonyl is one turn short.
        # That is the 3-10 geometry a helix begins with, and it is why this
        # asserts the mode rather than every value. The bug it guards against
        # made the mode 2, which no helix produces at all.
        self.assertTrue(np.all(np.isin(donated, (3.0, 4.0))), donated)
        self.assertGreater(float((donated == 4.0).mean()), 0.8, donated)

    def test_phi_and_psi_are_defined_in_the_interior_only(self) -> None:
        x = bg.compute(self._peptide(10))
        col = {n: j for j, n in enumerate(bg.COLUMNS)}
        r_phi = np.hypot(x[:, col["cos_phi"]], x[:, col["sin_phi"]])
        r_psi = np.hypot(x[:, col["cos_psi"]], x[:, col["sin_psi"]])
        self.assertEqual(r_phi[0], 0.0, "the first residue has no preceding C")
        self.assertEqual(r_psi[-1], 0.0, "the last residue has no following N")
        self.assertTrue(np.all(r_phi[1:] > 0.5))
        self.assertTrue(np.all(r_psi[:-1] > 0.5))

    def test_a_chain_break_makes_the_torsions_undefined(self) -> None:
        bbk = self._peptide(10)
        for k in bg.BACKBONE_ATOMS + ("CB",):
            bbk[k] = bbk[k].copy()
            bbk[k][5:] += np.array([60.0, 0.0, 0.0])
        x = bg.compute(bbk)
        col = {n: j for j, n in enumerate(bg.COLUMNS)}
        self.assertEqual(
            float(np.hypot(x[5, col["cos_phi"]], x[5, col["sin_phi"]])), 0.0,
            "a residue across a chain break must not receive a phi")
        self.assertEqual(
            float(np.hypot(x[4, col["cos_psi"]], x[4, col["sin_psi"]])), 0.0,
            "the residue before a chain break must not receive a psi")

    def test_consistency_passes_on_a_connected_chain(self) -> None:
        self.assertEqual(bg.consistency(bg.compute(self._peptide(12))), [])


class TestTheVirtualCarbonBeta(unittest.TestCase):
    """Glycine has no CB, and a sentinel would make every glycine extreme."""

    def test_it_sits_at_a_bond_length_from_ca(self) -> None:
        n = np.array([[-1.46, 0.0, 0.0]])
        ca = np.array([[0.0, 0.0, 0.0]])
        c = np.array([[0.55, 1.42, 0.0]])
        cb = bg._virtual_cb(n, ca, c)
        d = float(np.linalg.norm(cb - ca))
        self.assertTrue(1.4 < d < 1.7,
                        f"the constructed CB is {d:.2f} A from CA, which is not "
                        f"a carbon-carbon bond")

    def test_it_is_the_l_amino_acid_and_not_its_mirror(self) -> None:
        """The improper torsion N-CA-C-CB is about -122 degrees in an L residue.

        Checked as that torsion rather than as the sign of a triple product,
        because the improper is the quantity crystallography quotes and its
        value is citable, while the triple product's sign depends on an argument
        order that is easy to write down backwards -- as the first version of
        this test did.
        """
        n = np.array([[-1.46, 0.0, 0.0]])
        ca = np.array([[0.0, 0.0, 0.0]])
        c = np.array([[0.55, 1.42, 0.0]])
        cb = bg._virtual_cb(n, ca, c)
        cs, sn = bg._dihedral(n, ca, c, cb)
        improper = float(np.degrees(np.arctan2(sn[0], cs[0])))
        self.assertTrue(-128.0 < improper < -117.0,
                        f"the improper N-CA-C-CB is {improper:.1f} degrees; an "
                        f"L-amino acid is near -122 and its mirror near +122")

    def test_a_present_cb_is_not_overwritten(self) -> None:
        bb = as_chain(ideal_helix_ca(5))
        bb["N"] = bb["CA"] + np.array([-1.0, 0.2, 0.0])
        bb["C"] = bb["CA"] + np.array([0.6, 1.2, 0.0])
        bb["O"] = bb["C"] + np.array([0.0, 0.9, 0.3])
        bb["CB"] = bb["CA"] + np.array([0.0, 0.0, 1.53])
        x = bg.compute(bb)
        col = {n: j for j, n in enumerate(bg.COLUMNS)}
        self.assertTrue(np.all(np.abs(x[:, col["cb_radial"]]) <= 1.0))


class TestHydrogenBondsWithoutHydrogens(unittest.TestCase):

    def _two(self, no_distance: float, carbonyl_sign: float
             ) -> dict[str, np.ndarray]:
        """Five residues where only the pair (0, 4) can possibly bond."""
        n = 5
        bb = as_chain(np.column_stack([np.arange(n, dtype=float) * 12.0,
                                       np.zeros(n), np.zeros(n)]))
        bb["N"] = bb["CA"] + np.array([0.0, 1.0, 0.0])
        bb["C"] = bb["CA"] + np.array([1.0, 0.0, 0.0])
        bb["O"] = bb["C"] + np.array([0.0, 1.0, 0.0])
        # Put residue 4's amide nitrogen near residue 0's carbonyl oxygen.
        bb["N"] = bb["N"].copy()
        bb["N"][4] = bb["O"][0] + carbonyl_sign * np.array(
            [0.0, no_distance, 0.0])
        return bb

    def test_a_bond_beyond_the_oxygen_is_accepted(self) -> None:
        x = bg.compute(self._two(2.9, +1.0))
        col = {n: j for j, n in enumerate(bg.COLUMNS)}
        self.assertEqual(x[4, col["hb_donated"]], 1.0)
        self.assertEqual(x[4, col["hb_lag"]], 4.0)
        self.assertEqual(x[0, col["hb_accepted"]], 1.0)

    def test_a_nitrogen_on_the_carbon_side_is_refused(self) -> None:
        """The inversion that made the modal helical lag 2 instead of 4."""
        x = bg.compute(self._two(2.9, -1.0))
        col = {n: j for j, n in enumerate(bg.COLUMNS)}
        self.assertEqual(x[4, col["hb_donated"]], 0.0)
        self.assertEqual(x[4, col["hb_lag"]], 0.0)

    def test_a_distant_nitrogen_is_refused(self) -> None:
        x = bg.compute(self._two(bg.HBOND_MAX_NO + 1.0, +1.0))
        col = {n: j for j, n in enumerate(bg.COLUMNS)}
        self.assertEqual(x[4, col["hb_donated"]], 0.0)

    def test_a_lag_never_appears_without_a_bond(self) -> None:
        x = bg.compute(self._two(2.9, +1.0))
        col = {n: j for j, n in enumerate(bg.COLUMNS)}
        don = x[:, col["hb_donated"]] > 0
        self.assertTrue(np.all(x[~don, col["hb_lag"]] == 0.0))


class TestTheFamilysDefiningProperty(unittest.TestCase):
    """The claim that separates this family from the six that measured null."""

    def test_the_module_never_reads_a_residue_name(self) -> None:
        """If it did, it would be a function of residue type like chemistry 42.

        AGENT_MEMORY 2i records why that matters: the seven deployed constants
        are injective on the twenty types, so anything computed from the residue
        name is already carried by wires that ship, and the one family that
        filled the remaining quantisation collisions measured +0.000165 behind
        its own control. This family's whole claim is that it reads bytes the
        pipeline discards, and the cheapest way to keep that true is to refuse
        the residue name outright.
        """
        tree = ast.parse(MODULE.read_text())
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        names |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        strings = {n.value for n in ast.walk(tree)
                   if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        forbidden = {"resname", "AA20", "residue_type", "restype"}
        self.assertEqual(names & forbidden, set())
        code_strings = {s for s in strings if "\n" not in s}
        self.assertEqual(code_strings & {"resname"}, set())

    def test_the_atoms_it_reads_are_backbone_only(self) -> None:
        self.assertEqual(set(bg.BACKBONE_ATOMS), {"N", "CA", "C", "O"})

    def test_thirteen_named_columns_with_no_duplicates(self) -> None:
        self.assertEqual(len(bg.COLUMNS), bg.N_COLUMNS)
        self.assertEqual(len(set(bg.COLUMNS)), len(bg.COLUMNS))

    def test_the_ramachandran_cells_are_named_and_disjoint(self) -> None:
        self.assertEqual(set(bg.RAMA_CELLS), {0, 1, 2, 3})
        for k, v in bg.RAMA_CELLS.items():
            self.assertTrue(v.strip())


class TestConsistencyCatchesWhatItClaimsTo(unittest.TestCase):

    def _clean(self) -> np.ndarray:
        return np.zeros((4, bg.N_COLUMNS))

    def test_a_lag_without_a_bond_is_caught(self) -> None:
        x = self._clean()
        x[1, bg.COLUMNS.index("hb_lag")] = 4.0
        self.assertTrue(any("no donated bond" in m for m in bg.consistency(x)))

    def test_a_cosine_out_of_range_is_caught(self) -> None:
        x = self._clean()
        x[0, bg.COLUMNS.index("cos_phi")] = 1.5
        self.assertTrue(any("leaves [-1, 1]" in m for m in bg.consistency(x)))

    def test_a_half_defined_torsion_is_caught(self) -> None:
        x = self._clean()
        x[0, bg.COLUMNS.index("cos_psi")] = 0.4
        self.assertTrue(any("unit circle" in m for m in bg.consistency(x)))

    def test_a_negative_count_is_caught(self) -> None:
        x = self._clean()
        x[2, bg.COLUMNS.index("ca_density")] = -1.0
        self.assertTrue(any("negative" in m for m in bg.consistency(x)))

    def test_a_clean_array_reports_nothing(self) -> None:
        self.assertEqual(bg.consistency(self._clean()), [])


if __name__ == "__main__":
    unittest.main()
