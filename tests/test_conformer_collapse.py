"""One conformer per atom: MODEL and altLoc must not be read as extra matter.

A PDB entry may carry the same physical atom many times: once per NMR model, or
once per alternate location in an ensemble refinement. Reading every copy turns
a structure into a superposition of structures, and coordination number -- the
quantity every packing, contact-graph and ultrametric descriptor in this package
is built from -- then inflates by the copy count. The failure is silent: nothing
raises, the numbers simply become wrong.

The last test is the one that licenses a provenance decision. The residue
UNIVERSE (the set of resseq on the chain) is invariant under the collapse,
because dropping a duplicate atom never removes the last atom of a residue. Any
metric that touches the receptor only through the universe -- which is exactly
how an externally executed predictor such as P2Rank is scored, since its
per-residue probabilities come from its own output file -- is therefore
unchanged by this parser fix, so such rows may be carried across a re-freeze
without re-running the external tool.
"""
from __future__ import annotations

import unittest

from pocket_bench.pdb_io import parse_pdb_atoms


def atom_line(serial: int, name: str, altloc: str, resname: str, chain: str,
              resseq: int, x: float, y: float, z: float,
              occ: float, element: str) -> str:
    """One ATOM record in fixed PDB columns."""
    return (
        f"ATOM  {serial:5d} {name:<4s}{altloc}{resname:>3s} {chain}{resseq:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{20.00:6.2f}"
        f"          {element:>2s}"
    )


class TestModelCollapse(unittest.TestCase):
    def test_only_first_model_is_read(self) -> None:
        text = "\n".join([
            "MODEL        1",
            atom_line(1, " CA ", " ", "ALA", "A", 1, 0.0, 0.0, 0.0, 1.00, "C"),
            atom_line(2, " CB ", " ", "ALA", "A", 1, 1.5, 0.0, 0.0, 1.00, "C"),
            "ENDMDL",
            "MODEL        2",
            atom_line(1, " CA ", " ", "ALA", "A", 1, 9.0, 9.0, 9.0, 1.00, "C"),
            atom_line(2, " CB ", " ", "ALA", "A", 1, 9.5, 9.0, 9.0, 1.00, "C"),
            "ENDMDL",
            "END",
        ])
        atoms = parse_pdb_atoms(text)
        self.assertEqual(len(atoms), 2)
        self.assertEqual([a["x"] for a in atoms], [0.0, 1.5])

    def test_twenty_models_do_not_multiply_the_atom_count(self) -> None:
        lines = []
        for m in range(1, 21):
            lines.append(f"MODEL     {m:>4d}")
            for r in range(1, 6):
                lines.append(atom_line(r, " CA ", " ", "GLY", "A", r,
                                       float(r), float(m), 0.0, 1.00, "C"))
            lines.append("ENDMDL")
        atoms = parse_pdb_atoms("\n".join(lines))
        self.assertEqual(len(atoms), 5, "one model's worth of atoms, not twenty")


class TestAltLocCollapse(unittest.TestCase):
    def test_highest_occupancy_alternate_wins(self) -> None:
        text = "\n".join([
            atom_line(1, " CA ", "A", "SER", "A", 7, 0.0, 0.0, 0.0, 0.35, "C"),
            atom_line(2, " CA ", "B", "SER", "A", 7, 5.0, 0.0, 0.0, 0.65, "C"),
            "END",
        ])
        atoms = parse_pdb_atoms(text)
        self.assertEqual(len(atoms), 1)
        self.assertEqual(atoms[0]["altloc"], "B")
        self.assertEqual(atoms[0]["x"], 5.0)

    def test_equal_occupancy_breaks_on_altloc_not_file_order(self) -> None:
        forward = "\n".join([
            atom_line(1, " CA ", "A", "SER", "A", 7, 0.0, 0.0, 0.0, 0.50, "C"),
            atom_line(2, " CA ", "B", "SER", "A", 7, 5.0, 0.0, 0.0, 0.50, "C"),
        ])
        reversed_ = "\n".join([
            atom_line(1, " CA ", "B", "SER", "A", 7, 5.0, 0.0, 0.0, 0.50, "C"),
            atom_line(2, " CA ", "A", "SER", "A", 7, 0.0, 0.0, 0.0, 0.50, "C"),
        ])
        a, b = parse_pdb_atoms(forward), parse_pdb_atoms(reversed_)
        self.assertEqual(len(a), 1)
        self.assertEqual(len(b), 1)
        self.assertEqual(a[0]["altloc"], "A")
        self.assertEqual(b[0]["altloc"], "A")

    def test_blank_altloc_atoms_are_all_kept(self) -> None:
        text = "\n".join([
            atom_line(1, " N  ", " ", "ALA", "A", 1, 0.0, 0.0, 0.0, 1.00, "N"),
            atom_line(2, " CA ", "A", "ALA", "A", 1, 1.0, 0.0, 0.0, 0.60, "C"),
            atom_line(3, " CA ", "B", "ALA", "A", 1, 1.2, 0.0, 0.0, 0.40, "C"),
            atom_line(4, " C  ", " ", "ALA", "A", 1, 2.0, 0.0, 0.0, 1.00, "C"),
        ])
        atoms = parse_pdb_atoms(text)
        self.assertEqual([a["name"] for a in atoms], ["N", "CA", "C"])
        self.assertEqual(atoms[1]["altloc"], "A")


class TestUniverseInvariance(unittest.TestCase):
    """The property that lets an externally scored baseline survive a re-freeze."""

    def test_collapse_never_removes_a_residue(self) -> None:
        lines = ["MODEL        1"]
        for r in range(1, 11):
            # Every residue is fully alternate-split, the worst case for the
            # collapse: if it could drop a residue it would drop all ten.
            lines.append(atom_line(2 * r, " CA ", "A", "LEU", "A", r,
                                   float(r), 0.0, 0.0, 0.51, "C"))
            lines.append(atom_line(2 * r + 1, " CA ", "B", "LEU", "A", r,
                                   float(r), 1.0, 0.0, 0.49, "C"))
        lines += ["ENDMDL", "MODEL        2",
                  atom_line(99, " CA ", " ", "LEU", "A", 99,
                            0.0, 0.0, 0.0, 1.00, "C"),
                  "ENDMDL"]
        text = "\n".join(lines)

        collapsed = parse_pdb_atoms(text)
        kept = {int(a["resseq"]) for a in collapsed
                if a["record"] == "ATOM" and a["chain"] == "A"}
        model1 = {int(ln[22:26]) for ln in text.split("ENDMDL")[0].splitlines()
                  if ln.startswith("ATOM") and ln[21] == "A"}

        self.assertEqual(len(collapsed), 10, "one conformer per residue")
        self.assertEqual(kept, model1)
        self.assertEqual(kept, set(range(1, 11)))


if __name__ == "__main__":
    unittest.main()
