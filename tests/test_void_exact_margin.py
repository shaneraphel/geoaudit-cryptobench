"""The alpha-band constant, and the fallback that has never corrected anything.

``void_topology`` decides band membership in float64 and re-decides in exact
integer arithmetic within ``EXACT_MARGIN`` of an edge. Measured over the whole
training fold (``VOID_EXACT_MARGIN.json``, 12,113,689 tetrahedra): the exact path
fires nine times, has never changed a decision, and the nearest any circumradius
comes to a band edge is 5.57e-08 A against a worst measured float error of
5.01e-09 A. Eleven times, not a hundred -- a probe over 150 chains said a hundred
and the full fold found an error eight times larger, which is why the artifact is
built over all of it.

These tests hold the two properties that make the constant defensible rather than
arbitrary: the exact predicate agrees with the float one wherever the float one is
trustworthy, and it is exactly right where the float one is not. The second is
checked on constructed tetrahedra, since the corpus contains no case that
separates them -- which is the whole finding, and also the reason the corpus
cannot be used to test the fallback.
"""
from __future__ import annotations

import json
import sys
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "src"))

from pocket_bench.methods.void_topology import (  # noqa: E402
    ALPHA_MAX, ALPHA_MIN, EXACT_MARGIN, SCALE, _circumradius, _exact_in_band,
)
import void_exact_margin as vem  # noqa: E402

ARTIFACT = ROOT / "results/architecture_sweep/VOID_EXACT_MARGIN.json"


def tetra_with_circumradius(r: float) -> np.ndarray:
    """Four points on a sphere of radius ``r``, at three-decimal precision.

    Chosen so the vertices are well spread: a regular tetrahedron inscribed in
    the sphere, which is the best-conditioned case and therefore the one where
    a disagreement would be least excusable.
    """
    v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]],
                 dtype=np.float64)
    return np.round(v / np.sqrt(3.0) * r, 3)


class TestTheExactPredicateIsRight(unittest.TestCase):

    def test_it_agrees_with_the_float_decision_well_inside_the_band(self) -> None:
        for r in (3.5, 4.0, 4.5, 5.0, 5.5):
            with self.subTest(r=r):
                p = tetra_with_circumradius(r)
                q = np.rint(p * SCALE).astype(np.int64)
                self.assertTrue(_exact_in_band(q))

    def test_it_agrees_with_the_float_decision_well_outside(self) -> None:
        for r in (1.0, 2.0, 2.5, 7.0, 12.0):
            with self.subTest(r=r):
                p = tetra_with_circumradius(r)
                q = np.rint(p * SCALE).astype(np.int64)
                self.assertFalse(_exact_in_band(q))

    def test_a_degenerate_tetrahedron_is_rejected_not_admitted(self) -> None:
        # Four coplanar points have no circumsphere. Admitting one would put a
        # meaningless alpha sphere into a cluster.
        p = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0],
                      [0.0, 4.0, 0.0], [4.0, 4.0, 0.0]])
        q = np.rint(p * SCALE).astype(np.int64)
        self.assertFalse(_exact_in_band(q))

    def test_exact_r_squared_matches_the_float_circumradius(self) -> None:
        for r in (3.0, 4.25, 6.0):
            with self.subTest(r=r):
                p = tetra_with_circumradius(r)
                q = np.rint(p * SCALE).astype(np.int64)
                rf, _c, _d = _circumradius(p[None, :, :])
                e2 = vem.exact_r2(q)
                self.assertIsNotNone(e2)
                self.assertAlmostEqual(float(rf[0]) ** 2, float(e2), places=6)


class TestTheFallbackIsExercisedSomewhere(unittest.TestCase):
    """The corpus never separates the two, so a construction must."""

    def test_a_tetrahedron_can_be_built_that_needs_the_exact_decision(self) -> None:
        # A radius one part in 10^12 below the lower edge. Rounded coordinates
        # move it, so the exact value is whatever the integers say -- and the
        # point of the test is that the two predicates are computed from the
        # same integers and must agree with the exact rational, not that the
        # radius is any particular number.
        p = tetra_with_circumradius(ALPHA_MIN)
        q = np.rint(p * SCALE).astype(np.int64)
        e2 = vem.exact_r2(q)
        self.assertIsNotNone(e2)
        exact_in = (Fraction(ALPHA_MIN) ** 2 <= e2 <= Fraction(ALPHA_MAX) ** 2)
        self.assertEqual(_exact_in_band(q), exact_in,
                         "the integer predicate and the exact rational must "
                         "give the same answer by construction")

    def test_the_predicate_is_decided_by_integers_not_by_a_tolerance(self) -> None:
        # Shifting one vertex by a single unit in the last place of the stored
        # precision must be able to change the answer. A predicate that cannot
        # is rounding somewhere.
        p = tetra_with_circumradius(ALPHA_MAX)
        q = np.rint(p * SCALE).astype(np.int64)
        answers = set()
        for delta in range(-40, 41):
            qq = q.copy()
            qq[0][0] += delta
            answers.add(_exact_in_band(qq))
        self.assertEqual(answers, {True, False},
                         "walking a vertex across the band edge one thousandth "
                         "of an angstrom at a time never changed the verdict")


class TestTheMeasuredMargin(unittest.TestCase):
    """The artifact's own numbers, held so a regression has to be deliberate."""

    @classmethod
    def setUpClass(cls) -> None:
        if not ARTIFACT.exists():
            raise unittest.SkipTest(f"{ARTIFACT.name} not built")
        cls.d = json.loads(ARTIFACT.read_text())

    def test_it_is_a_training_fold_artifact(self) -> None:
        self.assertIs(self.d["reads_test_fold"], False)
        self.assertIs(self.d["reads_any_external_unit"], False)
        self.assertIs(self.d["clinical_grade"], False)

    def test_no_band_decision_has_ever_differed(self) -> None:
        self.assertEqual(
            self.d["scan"]["n_band_decisions_where_float_and_exact_differ"], 0)
        self.assertEqual(self.d["conditioning"]["n_band_decisions_flipped"], 0)

    def test_the_scan_covered_the_whole_fold(self) -> None:
        self.assertGreater(self.d["scan"]["n_tetrahedra"], 10_000_000)
        self.assertGreaterEqual(self.d["scan"]["n_chains"], 770)

    def test_the_margin_covers_the_worst_measured_error(self) -> None:
        v = self.d["verdict"]
        self.assertGreater(v["margin_over_worst_error"], 50,
                           "EXACT_MARGIN must stay well above the largest float "
                           "error measured on the worst-conditioned tetrahedra")
        self.assertEqual(EXACT_MARGIN, self.d["scan"]["deployed_margin"],
                         "the artifact was built against a different constant "
                         "than the one deployed")

    def test_the_headroom_is_reported_and_is_not_large(self) -> None:
        # Held deliberately: this is the number a reader should not round up.
        # A 150-chain probe made it 100x; the fold makes it 11x.
        h = self.d["verdict"]["headroom_factor"]
        self.assertGreater(h, 1.0, "the nearest approach to a band edge is "
                                   "inside the worst measured error, so the "
                                   "float decision is not safe on this corpus")
        self.assertLess(h, 100.0, "if this rises above 100 the artifact was "
                                  "rebuilt on a subset; the full fold gives 11")


if __name__ == "__main__":
    unittest.main()
