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


class TypesettableTest(unittest.TestCase):
    """A macro file that cannot be compiled is not a macro file.

    The preamble loads amsmath, amssymb, geometry, booktabs, graphicx and
    hyperref -- no inputenc, no fontenc, no newunicodechar. Under pdfLaTeX a
    character outside the font encoding is a hard error, not a rendering
    wobble, and no Python test sees it: the string is valid, the JSON is valid,
    the file writes successfully, and the build stops.

    The guard found seven real captions on its first run, carrying the
    greater-or-equal sign, the true minus sign, a right arrow, a delta and a
    rho. Each would have stopped the submission build. AGENTS.md §7: a
    generator that emits code for another language validates its own output in
    that language's terms.
    """

    def test_a_bare_unicode_character_is_rejected(self):
        planted = "\\newcommand{\\Foo}{ours \u2265 0.85 on the fold}"
        problems = efn._untypesettable(planted)
        self.assertEqual(len(problems), 1)
        self.assertIn("U+2265", problems[0])

    def test_every_character_named_is_the_one_that_offends(self):
        planted = "\\newcommand{\\Bar}{\u0394 rises}\n\\newcommand{\\Ok}{1.0}"
        problems = efn._untypesettable(planted)
        self.assertEqual(len(problems), 1)
        self.assertIn("line 1", problems[0])
        self.assertIn("U+0394", problems[0])

    def test_the_escape_table_disarms_them(self):
        for ch in ("\u2265", "\u2264", "\u2212", "\u2192", "\u0394",
                   "\u03c1", "\u2295", "\u00c5", "\u00b1"):
            with self.subTest(ch=ch):
                self.assertEqual(efn._untypesettable(efn._tex(ch)), [])

    def test_ascii_passes(self):
        self.assertEqual(
            efn._untypesettable("\\newcommand{\\Baz}{0.8341 [-0.0089, +0.0297]}"),
            [])

    def test_the_committed_macro_file_is_typesettable(self):
        if not efn.OUT.exists():
            self.skipTest("frozen_numbers.tex not built")
        self.assertEqual(efn._untypesettable(efn.OUT.read_text()), [])


if __name__ == "__main__":
    unittest.main()
