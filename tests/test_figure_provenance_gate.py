"""The figure gate knows about more than one generator, and still refuses.

``_figure_problems`` was written around a single generator and a single
provenance file. A second pair was added so that training-fold findings and
official-fold results are drawn by different scripts -- a figure drawn from a
frozen test-fold artifact and one drawn from a sweep over training halves carry
different licences, and one generator for both would let a reader assume they
carry the same one.

Widening a gate is the moment it stops failing on things it used to catch, so
these tests fix what it must still catch: an image nothing generates, an image
whose bytes have drifted from what its provenance recorded, a source artifact
that changed after the plot was drawn, and a caption in the README that is not
the caption the generator emitted. Each is checked by constructing the failure
rather than by trusting that the tree currently has none, because a gate that
has nothing to catch reports success identically to a gate that cannot catch
anything.
"""
from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

import classify_artifacts as ca  # noqa: E402


class TestBothGeneratorsAreKnown(unittest.TestCase):

    def test_two_pairs_are_registered(self) -> None:
        gens = [g for g, _ in ca.FIGURE_GENERATORS]
        self.assertIn("tools/make_official_figures.py", gens)
        self.assertIn("tools/make_architecture_figures.py", gens)

    def test_every_generator_and_provenance_file_exists(self) -> None:
        for gen, prov in ca.FIGURE_GENERATORS:
            self.assertTrue((ROOT / gen).is_file(),
                            f"{gen} is registered as a figure generator and is "
                            f"not in the tree")
            self.assertTrue(prov.is_file(),
                            f"{gen} has no provenance file at {prov}")

    def test_no_figure_is_claimed_by_two_generators(self) -> None:
        """Two generators naming one image would make its licence ambiguous."""
        import re
        seen: dict[str, str] = {}
        for gen, _ in ca.FIGURE_GENERATORS:
            for name in re.findall(r'FIGDIR / "([^"]+)"',
                                   (ROOT / gen).read_text()):
                self.assertNotIn(
                    name, seen,
                    f"figures/{name} is written by both {seen.get(name)} and "
                    f"{gen}, so which artifacts it is tied to is undefined")
                seen[name] = gen

    def test_the_tree_passes(self) -> None:
        self.assertEqual(ca._figure_problems(), [])


class TestTheGateStillRefuses(unittest.TestCase):
    """Each check is exercised against a constructed failure."""

    def setUp(self) -> None:
        self._saved = ca.FIGURES, ca.FIGURE_GENERATORS
        self.tmp = ROOT / "_tmp" / "figure_gate_test"
        self.tmp.mkdir(parents=True, exist_ok=True)
        ca.FIGURES = self.tmp / "figures"
        ca.FIGURES.mkdir(exist_ok=True)

    def tearDown(self) -> None:
        ca.FIGURES, ca.FIGURE_GENERATORS = self._saved
        for p in sorted(self.tmp.rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
        self.tmp.rmdir()

    def _wire(self, produced: list[str], recorded: dict,
              sources: dict | None = None) -> None:
        gen = self.tmp / "gen.py"
        gen.write_text("".join(f'FIGDIR / "{n}"\n' for n in produced))
        prov = self.tmp / "prov.json"
        prov.write_text(json.dumps({"figures": recorded,
                                    "sources": sources or {}}))
        ca.FIGURE_GENERATORS = ((str(gen.relative_to(ROOT)), prov),)

    def _png(self, name: str, body: bytes) -> str:
        (ca.FIGURES / name).write_bytes(body)
        return hashlib.sha256(body).hexdigest()

    def test_an_image_no_generator_writes_is_caught(self) -> None:
        self._png("stray.png", b"x")
        self._wire([], {})
        self.assertTrue(any("not produced by" in p
                            for p in ca._figure_problems()))

    def test_an_image_whose_bytes_drifted_is_caught(self) -> None:
        self._png("a.png", b"new bytes")
        self._wire(["a.png"], {"a.png": {"sha256": hashlib.sha256(
            b"the bytes that were recorded").hexdigest(), "caption": "c"}})
        self.assertTrue(any("bytes differ" in p
                            for p in ca._figure_problems()))

    def test_a_source_that_changed_after_the_plot_is_caught(self) -> None:
        sha = self._png("a.png", b"a")
        src = self.tmp / "src.json"
        src.write_text("{}")
        self._wire(["a.png"], {"a.png": {"sha256": sha, "caption": "c"}},
                   sources={str(src.relative_to(ROOT)): "0" * 64})
        self.assertTrue(any("changed since the figures were drawn" in p
                            for p in ca._figure_problems()))

    def test_a_missing_source_is_caught(self) -> None:
        sha = self._png("a.png", b"a")
        self._wire(["a.png"], {"a.png": {"sha256": sha, "caption": "c"}},
                   sources={"results/this_was_deleted.json": "0" * 64})
        self.assertTrue(any("and it is gone" in p
                            for p in ca._figure_problems()))

    def test_data_in_the_figures_directory_is_caught(self) -> None:
        (ca.FIGURES / "numbers.json").write_text("{}")
        self._wire([], {})
        self.assertTrue(any("holds images only" in p
                            for p in ca._figure_problems()))

    def test_a_caption_absent_from_the_readme_is_caught(self) -> None:
        sha = self._png("a.png", b"a")
        self._wire(["a.png"], {"a.png": {
            "sha256": sha,
            "caption": "a sentence that is deliberately not in the README"}})
        self.assertTrue(any("is not the one" in p
                            for p in ca._figure_problems()))

    def test_the_provenance_file_is_named_in_the_complaint(self) -> None:
        """A repository with two provenance files must say which one failed."""
        self._png("a.png", b"new")
        self._wire(["a.png"], {"a.png": {"sha256": "0" * 64, "caption": "c"}})
        problems = [p for p in ca._figure_problems() if "bytes differ" in p]
        self.assertEqual(len(problems), 1)
        self.assertIn("prov.json", problems[0])


if __name__ == "__main__":
    unittest.main()
