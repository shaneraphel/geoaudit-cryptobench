"""The manuscript's macros, checked from both ends.

The staleness gate regenerates the file and compares. That catches an artifact
moving under a number, and it cannot catch the generator itself losing macros,
because the comparison is against a file the same generator wrote. It happened:
a refactor cut the emitter to 36 of 475 macros and `make macros` passed, with
the only symptom a LaTeX build nobody had run yet.

So the tests here read the other end -- what the manuscript cites -- and hold
the names to what a control sequence is allowed to be.
"""
from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import emit_frozen_numbers as efn  # noqa: E402


class TestTheCommittedFileIsCurrent(unittest.TestCase):
    def test_it_is_what_the_artifacts_produce(self):
        self.assertEqual(efn.OUT.read_text(), efn.build(),
                         "paper/frozen_numbers.tex is stale; run "
                         "tools/emit_frozen_numbers.py")


class TestEveryCitedMacroExists(unittest.TestCase):
    def test_the_manuscript_cites_nothing_undefined(self):
        self.assertEqual(efn._undefined(efn.build()), [])

    def test_no_macro_name_carries_a_digit(self):
        """A TeX control sequence is letters only, so \\NLocP2 does not exist.

        It was emitted once, and the manuscript compiled it as \\NLocP followed
        by a literal 2.
        """
        import re
        bad = [n for n in re.findall(r"\\newcommand\{\\([A-Za-z0-9]+)\}",
                                     efn.build()) if not n.isalpha()]
        self.assertEqual(bad, [])


class TestTheGateBites(unittest.TestCase):
    """Point the scanner at a manuscript we control and see it react."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.src, self.real = pathlib.Path(self.tmp.name), efn.SRC
        efn.SRC = self.src
        self.addCleanup(lambda: setattr(efn, "SRC", self.real))

    def write(self, body: str) -> None:
        (self.src / "manuscript.tex").write_text(body)

    def test_an_undefined_citation_is_caught(self):
        self.write(r"The value is \NeverDefined{} on the fold.")
        self.assertEqual(len(efn._undefined("")), 1)

    def test_a_defined_citation_passes(self):
        self.write(r"The value is \RealMacro{} on the fold.")
        self.assertEqual(efn._undefined(r"\newcommand{\RealMacro}{0.5}"), [])

    def test_tex_own_angstrom_is_not_ours_to_define(self):
        self.write(r"within 10\,\AA{} of the centre")
        self.assertEqual(efn._undefined(""), [])

    def test_the_generated_file_is_not_scanned_as_a_manuscript(self):
        (self.src / efn.OUT.name).write_text(r"\newcommand{\X}{1}\Y{}")
        self.assertEqual(efn._undefined(""), [])


if __name__ == "__main__":
    unittest.main()
