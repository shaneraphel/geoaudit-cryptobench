import unittest

import numpy as np

from pocket_bench.methods.f_star_oracle import build_f_star_oracle


def _shell(n_theta=12, n_phi=12, radius=8.0, center=(0, 0, 0)):
    """A closed spherical shell of atoms enclosing a cavity at the center."""
    c = np.asarray(center, float)
    th = np.linspace(0, np.pi, n_theta)
    ph = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    pts = []
    for t in th:
        for p in ph:
            pts.append(
                c
                + radius
                * np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
            )
    return np.unique(np.round(np.array(pts), 3), axis=0)


class TestFStarOracle(unittest.TestCase):
    def test_output_contract(self):
        coords = _shell()
        mask, origin = build_f_star_oracle(coords, grid_resolution=1.0)
        self.assertEqual(mask.dtype, np.bool_)
        self.assertEqual(mask.ndim, 3)
        self.assertEqual(origin.shape, (3,))
        self.assertTrue(mask.any())

    def test_deterministic(self):
        coords = _shell()
        m1, o1 = build_f_star_oracle(coords, 1.0)
        m2, o2 = build_f_star_oracle(coords, 1.0)
        self.assertTrue(np.array_equal(m1, m2))
        self.assertTrue(np.array_equal(o1, o2))

    def test_superset_of_rigid_apo(self):
        # gamma=1.0 is included in the default mode set, so F* must contain the
        # rigid apo free space and never lose an admissible voxel.
        coords = _shell()
        rigid, _ = build_f_star_oracle(coords, 1.0, weyl_scalars=(1.0,))
        fstar, _ = build_f_star_oracle(coords, 1.0, weyl_scalars=(1.0, 1.1, 1.25))
        self.assertTrue(np.all(fstar | ~rigid))  # rigid free ⊆ F* free
        self.assertGreaterEqual(int(fstar.sum()), int(rigid.sum()))

    def test_breathing_opens_cavity(self):
        # Expanding the enclosing shell must expose strictly more free voxels
        # than the rigid state for a tightly packed cavity.
        coords = _shell(radius=6.0)
        rigid, _ = build_f_star_oracle(coords, 0.75, weyl_scalars=(1.0,), shell_radius=20.0)
        fstar, _ = build_f_star_oracle(
            coords, 0.75, weyl_scalars=(1.0, 1.25), shell_radius=20.0
        )
        self.assertGreater(int(fstar.sum()), int(rigid.sum()))

    def test_spectral_formulation_runs(self):
        rng = np.random.default_rng(0)
        coords = rng.normal(scale=6.0, size=(60, 3))
        mask, origin = build_f_star_oracle(
            coords, 1.5, formulation="spectral", spectral_modes=2
        )
        self.assertEqual(mask.ndim, 3)
        self.assertEqual(origin.shape, (3,))

    def test_grid_guard(self):
        coords = _shell(radius=30.0)
        with self.assertRaises(ValueError):
            build_f_star_oracle(coords, 0.02)


if __name__ == "__main__":
    unittest.main()
