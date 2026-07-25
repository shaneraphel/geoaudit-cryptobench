"""The native kernels must be a pure wall-clock optimisation.

Two properties are asserted:

1. Bit-identity. Every kernel is an operation-for-operation port of a NumPy
   reference loop, so toggling the shared library may not move a single bit of a
   detector's output. If it does, the "enabling native cannot change a result"
   claim in ``native.py`` is false and the published numbers depend on whether a
   ``.dylib`` happened to be present.
2. Completeness. ``native.py`` must expose every kernel its callers reach for.
   A half-bound module (library loads, one symbol missing) previously turned into
   an ``AttributeError`` swallowed by the detectors' ``except Exception`` handler,
   which reported ``status=CRASH`` on every structure rather than failing loudly.
"""
from __future__ import annotations

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from pocket_bench import native
from pocket_bench.methods import fstar_pocket, geometric_foundation, sstar_pocket

# Every kernel a caller in src/ reaches for via the ``native`` module.
REQUIRED_KERNELS = ("free_grid_mask", "buriedness", "local_free_enclosed")


@contextmanager
def _numpy_only():
    """Force the NumPy fallback, then restore whatever was loaded before."""
    lib, tried = native._LIB, native._TRIED
    native._LIB, native._TRIED = None, True
    try:
        yield
    finally:
        native._LIB, native._TRIED = lib, tried


def _blob_pdb(tmp: Path) -> Path:
    """A hollow lobed shell: enough enclosure for candidates to survive."""
    p = tmp / "blob.pdb"
    rng = np.random.default_rng(7)
    with p.open("w") as fh:
        n = 0
        for k in range(400):
            v = rng.normal(size=3)
            v /= np.linalg.norm(v) + 1e-9
            r = 9.0 + 1.5 * np.cos(3.0 * v[0]) * np.sin(2.0 * v[1])
            x, y, z = v * r
            n += 1
            fh.write(
                f"ATOM  {n:5d}  CA  ALA A{n:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        fh.write("END\n")
    return p


class TestNativeCompleteness(unittest.TestCase):
    def test_every_required_kernel_is_exposed(self):
        for name in REQUIRED_KERNELS:
            self.assertTrue(
                callable(getattr(native, name, None)),
                f"pocket_bench.native.{name} is missing; callers would AttributeError",
            )

    def test_kernels_return_none_when_library_absent(self):
        """The fallback contract: absent library => None, never an exception."""
        pts = np.zeros((4, 3))
        coords = np.ones((5, 3))
        dirs = np.eye(3)
        with _numpy_only():
            self.assertIsNone(native.free_grid_mask(pts, coords, 2.6, 6.0))
            self.assertIsNone(native.buriedness(pts, coords, dirs, 11.0, 1.8))
            self.assertIsNone(native.local_free_enclosed(pts, coords, 2.6, 10.0, 14))


class TestNativeMatchesNumpy(unittest.TestCase):
    """Skipped wholesale when the library was never built."""

    def setUp(self):
        if not native.available():
            self.skipTest("native library not built")

    def test_local_free_enclosed_bit_identical(self):
        rng = np.random.default_rng(0)
        coords = rng.normal(scale=6.0, size=(300, 3))
        kw = dict(r_local=6.0, step=1.5, atom_r=2.6, enclose_cut=10.0, enclose_min=14)
        for c in rng.normal(scale=4.0, size=(25, 3)):
            fast = fstar_pocket._local_free_enclosed(c, coords, **kw)
            with _numpy_only():
                ref = fstar_pocket._local_free_enclosed(c, coords, **kw)
            self.assertEqual(fast, ref, f"native/numpy disagree at centre {c}")

    def test_free_grid_and_buriedness_bit_identical(self):
        rng = np.random.default_rng(1)
        coords = rng.normal(scale=6.0, size=(400, 3))
        pts = geometric_foundation._free_grid(coords, 1.5, 2.6, 6000)
        with _numpy_only():
            ref_pts = geometric_foundation._free_grid(coords, 1.5, 2.6, 6000)
        np.testing.assert_array_equal(pts, ref_pts)

        dirs = geometric_foundation._fibonacci_directions(30)
        fast = geometric_foundation._buriedness(pts, coords, dirs, 11.0, 1.8)
        with _numpy_only():
            ref = geometric_foundation._buriedness(pts, coords, dirs, 11.0, 1.8)
        np.testing.assert_array_equal(fast, ref)

    def test_detectors_identical_with_and_without_native(self):
        with tempfile.TemporaryDirectory() as d:
            pdb = _blob_pdb(Path(d))
            for mod in (geometric_foundation, fstar_pocket, sstar_pocket):
                fast = mod.predict(pdb, pdb_id="BLOB")
                with _numpy_only():
                    ref = mod.predict(pdb, pdb_id="BLOB")
                name = mod.__name__.rsplit(".", 1)[-1]
                self.assertEqual(fast["status"], "OK", f"{name}: {fast.get('error')}")
                self.assertEqual(
                    [p["center_xyz"] for p in fast["pockets"]],
                    [p["center_xyz"] for p in ref["pockets"]],
                    f"{name}: native path moved the predicted centres",
                )
                self.assertEqual(
                    [p["score"] for p in fast["pockets"]],
                    [p["score"] for p in ref["pockets"]],
                    f"{name}: native path moved the scores",
                )


class TestDetectorsDoNotCrash(unittest.TestCase):
    """Guards the exact failure that produced status=CRASH on 100% of structures.

    The detectors catch ``Exception`` and downgrade it to a CRASH row, so a missing
    kernel is invisible in the aggregate unless a test asserts on the status.
    """

    def test_all_detectors_report_ok(self):
        with tempfile.TemporaryDirectory() as d:
            pdb = _blob_pdb(Path(d))
            for mod in (geometric_foundation, fstar_pocket, sstar_pocket):
                out = mod.predict(pdb, pdb_id="BLOB")
                name = mod.__name__.rsplit(".", 1)[-1]
                self.assertEqual(out["status"], "OK", f"{name}: {out.get('error')}")
                self.assertGreaterEqual(len(out["pockets"]), 1, name)


if __name__ == "__main__":
    unittest.main()
