"""The generated banks have to be what they are advertised as.

These are not smoke tests. Each one checks a mathematical property the modules'
docstrings assert, because those assertions are the reason the descriptors are
admissible at all: a quantity that moved when the receptor was rotated would not
be an invariant, a curvature that could not tell a saddle from a dome would not
be a curvature, and a "chain" descriptor that ignored the chain would be a
duplicate of the bank it was written to complement.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pocket_bench.methods import chain_operator_descriptors as ch  # noqa: E402
from pocket_bench.methods import operator_descriptors as op  # noqa: E402


def _helix(n=220, seed=0, radius=11.0, rise=0.45, turn=0.35):
    """A crude helical chain: sequence-adjacent residues are spatially close."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    P = np.stack([radius * np.cos(turn * t), radius * np.sin(turn * t),
                  rise * t], axis=1)
    return P + rng.normal(scale=0.45, size=(n, 3))


def _rigid_motion(P, seed=1):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(3, 3))
    Q, R = np.linalg.qr(A)
    Q = Q * np.sign(np.diag(R))            # a rotation, not a reflection
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return P @ Q.T + rng.normal(scale=30.0, size=3)


class TestRigidInvariance(unittest.TestCase):
    """Every descriptor is a function of the receptor's shape, so a rotation and
    a translation of the receptor must leave all of them fixed."""

    def test_operator_bank_is_invariant(self):
        P = _helix()
        a = op.operator_residue_features(P)
        b = op.operator_residue_features(_rigid_motion(P))
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_chain_bank_is_invariant(self):
        P = _helix()
        a = ch.chain_operator_residue_features(P)
        b = ch.chain_operator_residue_features(_rigid_motion(P))
        np.testing.assert_allclose(a, b, atol=1e-5)


class TestDeterminism(unittest.TestCase):
    def test_both_banks_repeat_exactly(self):
        P = _helix()
        for mod, fn in ((op, op.operator_residue_features),
                        (ch, ch.chain_operator_residue_features)):
            with self.subTest(module=mod.__name__):
                np.testing.assert_array_equal(fn(P), fn(P))


class TestWellFormed(unittest.TestCase):
    def test_names_are_unique_and_match_the_widths(self):
        P = _helix()
        self.assertEqual(len(set(op.FEATURE_NAMES)), op.N_OPERATOR)
        self.assertEqual(len(set(ch.FEATURE_NAMES)), ch.N_CHAIN_OPERATOR)
        self.assertEqual(op.operator_residue_features(P).shape[1],
                         op.N_OPERATOR)
        self.assertEqual(ch.chain_operator_residue_features(P).shape[1],
                         ch.N_CHAIN_OPERATOR)

    def test_nothing_is_infinite_or_undefined(self):
        P = _helix()
        for fn in (op.operator_residue_features,
                   ch.chain_operator_residue_features):
            with self.subTest(fn=fn.__name__):
                self.assertTrue(np.isfinite(fn(P)).all())

    def test_a_chain_too_short_for_a_spectrum_does_not_raise(self):
        """Two residues have no neighbourhood to decompose. The descriptors
        take the value that adds no information rather than failing."""
        P = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0]])
        for fn in (op.operator_residue_features,
                   ch.chain_operator_residue_features):
            with self.subTest(fn=fn.__name__):
                out = fn(P)
                self.assertEqual(len(out), 2)
                self.assertTrue(np.isfinite(out).all())


class TestTheChainBankActuallyReadsTheChain(unittest.TestCase):
    """The point of the second module is information the first cannot hold.
    If permuting the sequence labels while keeping the geometry left the chain
    bank unchanged, it would be holding none."""

    @staticmethod
    def _reindex(P, perm):
        return P[perm]

    def test_the_lag_spectrum_moves_when_the_sequence_is_permuted(self):
        P = _helix()
        rng = np.random.default_rng(3)
        perm = rng.permutation(len(P))
        cols = [i for i, n in enumerate(ch.FEATURE_NAMES)
                if n.startswith(("lag_", "symbol", "n_segments",
                                 "segment_len", "frac_lag"))]
        a = ch.chain_operator_residue_features(P)[:, cols]
        b = ch.chain_operator_residue_features(P[perm])[:, cols]
        # Same point set, same geometry, different chain order: the lag
        # spectrum must not survive it.
        self.assertGreater(abs(float(a.mean() - b.mean())), 1e-3)

    def test_the_operator_bank_does_not_and_cannot(self):
        """Its descriptors are functions of the contact graph alone, so
        relabelling residues permutes the rows and changes nothing else."""
        P = _helix()
        rng = np.random.default_rng(3)
        perm = rng.permutation(len(P))
        a = op.operator_residue_features(P)[perm]
        b = op.operator_residue_features(P[perm])
        np.testing.assert_allclose(a, b, atol=1e-6)

    def test_a_helix_looks_sequence_local(self):
        """On a helix nearly every spatial neighbour is a sequence neighbour,
        so the first Toeplitz coefficient sits near its maximum."""
        F = ch.chain_operator_residue_features(_helix())
        j = ch.FEATURE_NAMES.index("symbol1@8")
        interior = F[20:-20, j]
        self.assertGreater(float(interior.mean()), 0.5)


class TestTheShapeOperatorHasTheRightSigns(unittest.TestCase):
    """Gaussian curvature is the product of the principal curvatures, so its
    sign is fixed whichever way the normal points. That makes it the one
    quantity here that can be tested against a surface of known shape."""

    @staticmethod
    def _patch(fn, n=400, half=9.0, seed=5):
        rng = np.random.default_rng(seed)
        xy = rng.uniform(-half, half, size=(n, 2))
        z = fn(xy[:, 0], xy[:, 1])
        return np.column_stack([xy, z])

    def _curv(self, P, name):
        F = ch.chain_operator_residue_features(P)
        j = ch.FEATURE_NAMES.index(name)
        r = np.linalg.norm(P[:, :2], axis=1)
        return F[r < 4.0, j]           # only the well-surrounded centre

    def test_a_saddle_has_negative_gaussian_curvature(self):
        P = self._patch(lambda x, y: 0.05 * (x * x - y * y))
        k = self._curv(P, "gauss_curv@8")
        self.assertLess(float(np.median(k)), 0.0)

    def test_a_dome_has_positive_gaussian_curvature(self):
        P = self._patch(lambda x, y: -0.05 * (x * x + y * y))
        k = self._curv(P, "gauss_curv@8")
        self.assertGreater(float(np.median(k)), 0.0)

    def test_a_plane_has_curvature_near_zero_and_a_small_residual(self):
        P = self._patch(lambda x, y: 0.0 * x + 0.0 * y)
        self.assertLess(abs(float(np.median(self._curv(P, "gauss_curv@8")))),
                        1e-3)
        self.assertLess(float(np.median(
            self._curv(P, "quadric_residual@8"))), 1e-6)


class TestTheValuationProfileIsUltrametric(unittest.TestCase):
    def test_single_linkage_cophenetic_distance_obeys_the_strong_triangle(self):
        """du(i,k) <= max(du(i,j), du(j,k)) is what makes the balls nested, and
        the profile descriptors are only meaningful if it holds."""
        from scipy.cluster.hierarchy import cophenet, linkage
        from scipy.spatial.distance import pdist, squareform

        P = _helix(n=60)
        du = squareform(cophenet(linkage(pdist(P), method="single")))
        rng = np.random.default_rng(7)
        for _ in range(300):
            i, j, k = rng.integers(0, len(P), size=3)
            self.assertLessEqual(du[i, k], max(du[i, j], du[j, k]) + 1e-9)

    def test_the_ball_counts_are_monotone_in_the_radius(self):
        F = ch.chain_operator_residue_features(_helix(n=120))
        cols = [ch.FEATURE_NAMES.index(f"valuation_ball@{t:g}")
                for t in ch.VALUATION_LEVELS]
        B = F[:, cols]
        self.assertTrue((np.diff(B, axis=1) >= -1e-9).all(),
                        "a larger ultrametric ball cannot hold fewer points")


class TestTheHingeIndicatorSeparatesTheThreeCases(unittest.TestCase):
    """A lid moves; a hinge does not move but its neighbours move relative to
    it; a core interior does neither. Participation alone cannot tell the last
    two apart, which is why the shear is carried separately."""

    def test_participation_and_shear_are_not_the_same_descriptor(self):
        F = ch.chain_operator_residue_features(_helix(n=180))
        a = F[:, ch.FEATURE_NAMES.index("thermal_participation")]
        s = F[:, ch.FEATURE_NAMES.index("thermal_shear")]
        if a.std() == 0 or s.std() == 0:
            self.skipTest("no non-rigid modes on this synthetic chain")
        r = float(np.corrcoef(a, s)[0, 1])
        self.assertLess(abs(r), 0.98,
                        "if shear were a function of participation the pair "
                        "would carry no more than participation alone")

    def test_the_hinge_rank_is_a_rank(self):
        F = ch.chain_operator_residue_features(_helix(n=180))
        r = F[:, ch.FEATURE_NAMES.index("hinge_rank")]
        if r.std() == 0:
            self.skipTest("no non-rigid modes on this synthetic chain")
        self.assertAlmostEqual(float(r.min()), 0.0, places=9)
        self.assertAlmostEqual(float(r.max()), 1.0, places=9)


if __name__ == "__main__":
    unittest.main()
