"""Anisotropic Shear Oracle (S*) contract + Class-A behavior.

Skipped when scipy is unavailable (CI installs numpy only); scipy is an optional
[spectral] extra. The oracle must (a) return a 3D Boolean mask + origin, (b) be
deterministic, (c) be a superset of the rigid apo free space, and (d) use
anisotropic (non-scalar) displacement fields.
"""
from __future__ import annotations

import unittest

import numpy as np

try:
    import scipy  # noqa: F401
    from pocket_bench.methods import anisotropic_shear_oracle as aso

    HAVE_SCIPY = True
except Exception:  # noqa: BLE001
    HAVE_SCIPY = False


def _two_lobe_protein(seed: int = 0) -> np.ndarray:
    """Two dense lobes joined by a thin neck — a hinge whose low mode is a shear."""
    rng = np.random.default_rng(seed)
    a = rng.normal([-6, 0, 0], 2.2, size=(140, 3))
    b = rng.normal([6, 0, 0], 2.2, size=(140, 3))
    neck = rng.normal([0, 0, 0], 1.0, size=(20, 3))
    return np.vstack([a, neck, b])


@unittest.skipUnless(HAVE_SCIPY, "scipy not installed")
class TestAnisotropicShearOracle(unittest.TestCase):
    def test_output_contract(self) -> None:
        coords = _two_lobe_protein()
        mask, origin = aso.build_anisotropic_shear_oracle(coords, grid_resolution=1.5)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.ndim, 3)
        self.assertEqual(origin.shape, (3,))
        self.assertTrue(mask.any())

    def test_deterministic(self) -> None:
        coords = _two_lobe_protein()
        m1, o1 = aso.build_anisotropic_shear_oracle(coords, grid_resolution=1.5)
        m2, o2 = aso.build_anisotropic_shear_oracle(coords, grid_resolution=1.5)
        self.assertTrue(np.array_equal(m1, m2))
        self.assertTrue(np.allclose(o1, o2))

    def test_modes_bytewise_deterministic_across_calls(self) -> None:
        # Regression for the eigsh unseeded-start hole: with a pinned v0 the raw
        # modes AND eigenvalues must be byte-identical across repeated calls, not
        # merely equal up to sign.
        coords = _two_lobe_protein()
        m1, l1 = aso.low_shear_modes(coords, k=3)
        m2, l2 = aso.low_shear_modes(coords, k=3)
        self.assertTrue(np.array_equal(m1, m2))
        self.assertTrue(np.array_equal(l1, l2))

    def test_modes_are_anisotropic_not_scalar(self) -> None:
        # A pure homothety would give parallel displacement ~ r - r_c everywhere.
        # A real shear mode must vary in direction across atoms.
        coords = _two_lobe_protein()
        modes, lambdas = aso.low_shear_modes(coords, k=3)
        self.assertEqual(modes.shape[0], 3)
        self.assertEqual(lambdas.shape[0], 3)
        # spectral gap anchors amplitude: c_1 == dx, later modes attenuated
        c_k = aso.dynamic_shear_amplitudes(lambdas, 1.5)
        self.assertAlmostEqual(float(c_k[0]), 1.5, places=6)
        self.assertTrue(bool(np.all(c_k[1:] <= c_k[0] + 1e-9)))
        u = modes[0]
        centered = coords - coords.mean(0)
        # cosine alignment of the mode with the radial (homothety) field
        rad = centered / (np.linalg.norm(centered, axis=1, keepdims=True) + 1e-9)
        un = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
        cos = np.abs((rad * un).sum(1))
        self.assertLess(float(cos.mean()), 0.9)  # not a radial homothety

    def test_superset_of_rigid_apo_and_opens_volume(self) -> None:
        coords = _two_lobe_protein()
        comp = aso.build_anisotropic_shear_oracle(
            coords, grid_resolution=1.5, return_components=True
        )
        s_star, apo_free = comp["s_star_mask"], comp["apo_free"]
        # S* must contain every apo-free voxel (all-zero combo is included)
        self.assertTrue(np.all(s_star | ~apo_free))
        self.assertTrue(bool((s_star & ~apo_free).any()))  # opened new volume
        self.assertGreaterEqual(comp["n_modes_evaluated"], 27)


if __name__ == "__main__":
    unittest.main()
