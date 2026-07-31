"""The twenty amino acids, as integers, checked against chemistry.

``consistency()`` checks the tables against each other, which catches the
failure where one table is edited and its neighbour is not. This file checks
them against what is actually true of the molecules, which is the part no
internal consistency check can reach: a table can be perfectly consistent with
its neighbours and wrong about arginine.

The values chosen for assertion here are the ones where a plausible mistake
would change a conclusion rather than a decimal. Histidine's charge is the
clearest: the eight-class partition in ``composition_wires.py`` calls histidine
"positive", and at pH 7.4 it is mostly neutral, so a family built on the class
label and a family built on the charge disagree about a residue that lines a
great many pockets.
"""
from __future__ import annotations

import unittest

import numpy as np

from pocket_bench.methods.residue_chemistry import (
    AA20,
    CHI_ROTATABLE,
    FORMAL_CHARGE,
    PROPERTIES,
    SC_CARBON,
    SC_HBA,
    SC_HBD,
    SC_POLAR_ATOMS,
    consistency,
    property_names,
    table,
)


class TestTablesAreInternallyConsistent(unittest.TestCase):
    def test_consistency_passes(self) -> None:
        r = consistency()
        self.assertEqual(r["failed"], [])
        self.assertTrue(r["ok"], r)

    def test_every_table_names_all_twenty(self) -> None:
        for name, d in PROPERTIES:
            self.assertEqual(set(d), set(AA20), f"{name} does not name AA20")

    def test_the_table_is_integer(self) -> None:
        """Quantised inputs downstream; a float here would be a fitted scale."""
        self.assertTrue(np.issubdtype(table().dtype, np.integer))


class TestChiRotatableIsTheQuantityItClaims(unittest.TestCase):
    """The module's central quantity: can this neighbourhood open at all.

    A cryptic pocket is closed in the apo structure and the label says it
    opens. Side chains open it, and the number of rotameric dihedrals is how
    much freedom each one has to do so.
    """

    def test_glycine_alanine_proline_have_no_rotamers(self) -> None:
        for a in ("GLY", "ALA", "PRO"):
            self.assertEqual(CHI_ROTATABLE[a], 0, a)

    def test_proline_is_zero_because_the_ring_locks_it(self) -> None:
        """Not because it is small: proline is larger than alanine."""
        from pocket_bench.methods.residue_chemistry import SC_VOLUME
        self.assertGreater(SC_VOLUME["PRO"], SC_VOLUME["ALA"])
        self.assertEqual(CHI_ROTATABLE["PRO"], CHI_ROTATABLE["ALA"])

    def test_lysine_and_arginine_are_the_most_flexible(self) -> None:
        top = max(CHI_ROTATABLE.values())
        self.assertEqual(top, 4)
        self.assertEqual(
            {a for a, v in CHI_ROTATABLE.items() if v == top}, {"LYS", "ARG"})

    def test_the_aliphatic_class_hides_a_three_step_spread(self) -> None:
        """The reason this family is not composition_wires with more columns.

        composition_wires puts ALA, VAL, LEU, ILE and MET in one class. Their
        rotameric freedom runs 0, 1, 2, 2, 3, so a count of aliphatic
        neighbours cannot distinguish a neighbourhood that can rearrange from
        one that cannot.
        """
        aliphatic = ("ALA", "VAL", "LEU", "ILE", "MET")
        chis = sorted(CHI_ROTATABLE[a] for a in aliphatic)
        self.assertEqual(chis, [0, 1, 2, 2, 3])


class TestHydrogenBondingCounts(unittest.TestCase):
    def test_arginine_donates_five(self) -> None:
        """One on NE, two on each of NH1 and NH2."""
        self.assertEqual(SC_HBD["ARG"], 5)

    def test_lysine_donates_three(self) -> None:
        self.assertEqual(SC_HBD["LYS"], 3)

    def test_arginine_and_lysine_differ_although_both_are_positive(self) -> None:
        """Both are "positive" to the eight-class partition and they are not
        the same surface: five donors in a plane against three from a point."""
        self.assertEqual(FORMAL_CHARGE["ARG"], FORMAL_CHARGE["LYS"])
        self.assertNotEqual(SC_HBD["ARG"], SC_HBD["LYS"])

    def test_tryptophan_donates_and_does_not_accept(self) -> None:
        """The indole nitrogen carries an N-H; its lone pair is aromatic."""
        self.assertEqual(SC_HBD["TRP"], 1)
        self.assertEqual(SC_HBA["TRP"], 0)

    def test_carboxylates_accept_two_and_donate_none(self) -> None:
        for a in ("ASP", "GLU"):
            self.assertEqual(SC_HBA[a], 2, a)
            self.assertEqual(SC_HBD[a], 0, a)

    def test_the_apolar_five_do_neither(self) -> None:
        for a in ("GLY", "ALA", "VAL", "LEU", "ILE", "PHE", "PRO"):
            self.assertEqual(SC_HBD[a], 0, a)
            self.assertEqual(SC_HBA[a], 0, a)


class TestChargeAtPhysiologicalPh(unittest.TestCase):
    def test_histidine_is_neutral_at_ph_7_4(self) -> None:
        """pKa near 6.0. Calling it positive is wrong about the majority
        species, and composition_wires' class partition does exactly that."""
        self.assertEqual(FORMAL_CHARGE["HIS"], 0)

    def test_the_charged_four_and_nobody_else(self) -> None:
        charged = {a for a, v in FORMAL_CHARGE.items() if v != 0}
        self.assertEqual(charged, {"ASP", "GLU", "LYS", "ARG"})

    def test_the_net_charge_of_the_twenty_is_zero(self) -> None:
        """Two anions and two cations; a sign error in one would show here."""
        self.assertEqual(sum(FORMAL_CHARGE.values()), 0)


class TestHeavyAtomComposition(unittest.TestCase):
    def test_side_chain_carbon_counts(self) -> None:
        """Counted off the structural formula, backbone excluded."""
        expected = {"GLY": 0, "ALA": 1, "SER": 1, "CYS": 1, "THR": 2,
                    "ASP": 2, "ASN": 2, "VAL": 3, "PRO": 3, "MET": 3,
                    "GLU": 3, "GLN": 3, "LEU": 4, "ILE": 4, "LYS": 4,
                    "HIS": 4, "ARG": 4, "PHE": 7, "TYR": 7, "TRP": 9}
        self.assertEqual(SC_CARBON, expected)

    def test_polar_atom_counts(self) -> None:
        self.assertEqual(SC_POLAR_ATOMS["ARG"], 3)
        self.assertEqual(SC_POLAR_ATOMS["LYS"], 1)
        self.assertEqual(SC_POLAR_ATOMS["TYR"], 1)
        self.assertEqual(SC_POLAR_ATOMS["PHE"], 0)

    def test_phenylalanine_and_tyrosine_differ_by_one_oxygen(self) -> None:
        """The whole difference between the two, and the class label loses it."""
        self.assertEqual(SC_CARBON["PHE"], SC_CARBON["TYR"])
        self.assertEqual(SC_POLAR_ATOMS["TYR"] - SC_POLAR_ATOMS["PHE"], 1)
        self.assertEqual(SC_HBD["TYR"] - SC_HBD["PHE"], 1)


class TestThePropertiesAreDistinct(unittest.TestCase):
    """Fourteen quantities, not fourteen namings of one.

    AGENT_MEMORY 2c: a family is worth building only if its members are
    different quantities, because same-quantity pairs carry the only negative
    mean interaction in the deployed bus. Two identical columns would be that
    category in its purest form.
    """

    def test_no_two_properties_are_the_same_column(self) -> None:
        t = table()
        n = property_names()
        dupes = []
        for i in range(t.shape[1]):
            for j in range(i + 1, t.shape[1]):
                if np.array_equal(t[:, i], t[:, j]):
                    dupes.append((n[i], n[j]))
        self.assertEqual(dupes, [], f"duplicate columns: {dupes}")

    def test_no_property_is_a_constant(self) -> None:
        t = table()
        flat = [property_names()[j] for j in range(t.shape[1])
                if len(np.unique(t[:, j])) == 1]
        self.assertEqual(flat, [], f"constant columns: {flat}")

    def test_chi_is_not_recoverable_from_the_eight_classes(self) -> None:
        """The concrete claim that this family carries new information.

        composition_wires' classes are a partition of the twenty. If chi were
        constant within every class, a count of class members would already
        determine the conformational budget and this family would be a
        renaming. It is not: three classes carry more than one chi value.
        """
        classes = {
            "aliphatic": ("ALA", "VAL", "LEU", "ILE", "MET"),
            "aromatic": ("PHE", "TRP", "TYR"),
            "polar": ("SER", "THR", "ASN", "GLN"),
            "positive": ("LYS", "ARG", "HIS"),
            "negative": ("ASP", "GLU"),
            "glycine": ("GLY",),
            "proline": ("PRO",),
            "cysteine": ("CYS",),
        }
        spread = {c: sorted({CHI_ROTATABLE[a] for a in m})
                  for c, m in classes.items()}
        ambiguous = {c: v for c, v in spread.items() if len(v) > 1}
        self.assertGreaterEqual(
            len(ambiguous), 3,
            f"chi would be determined by the class partition: {spread}")
        self.assertIn("aliphatic", ambiguous)
        self.assertIn("polar", ambiguous)
        self.assertIn("positive", ambiguous)


if __name__ == "__main__":
    unittest.main()
