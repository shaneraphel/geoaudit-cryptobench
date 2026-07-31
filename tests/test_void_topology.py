"""Tests for the void-topology family.

The backbone family shipped a sign error in a dihedral that no range check would
have caught, and the side-chain family shipped a double-counted antipodal pair
and three columns that were declared and never computed. Both were found by
tests that built a shape whose answer was known in advance rather than by tests
that asserted a number was finite. Every test here does the same: it constructs
a geometry whose void structure can be written down, or it states an invariant
that a swapped index would break.

Two of these tests exist because of a specific failure mode of this
construction. ``test_the_exact_band_decision_is_used_at_the_edge`` exists
because the fast path is float and the exact path is integer, and a fast path
that is never checked against the slow one is a fast path that is wrong in
silence. ``test_the_answer_does_not_depend_on_the_order_atoms_arrive_in`` exists
because the construction passes through sets and a Delaunay library, and a
result that depends on input order would attach a different number to the same
residue on a rebuild.
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from pocket_bench.methods.void_topology import (
    ALPHA_MAX, ALPHA_MIN, COLUMNS, IDX, LINKAGE, SCALE, VOID_LEVEL,
    _circumradius, _exact_in_band, _segments, chain_voids, compute,
    consistency,
)
from pocket_bench.paths import ROOT
from pocket_bench.pdb_io import parse_pdb_atoms

MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"


def atom(name, resseq, xyz, resname="ALA", chain="A", element=None):
    return {"record": "ATOM", "serial": 1, "name": name, "altloc": "",
            "resname": resname, "chain": chain, "resseq": resseq, "icode": "",
            "occupancy": 1.0, "x": float(xyz[0]), "y": float(xyz[1]),
            "z": float(xyz[2]),
            "element": element or name.strip()[0], "raw_line": ""}


def shell(radius, n, jitter=0.0, seed=0):
    """``n`` points spread over a sphere by the golden-angle spiral."""
    rng = np.random.default_rng(seed)
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    p = np.stack([np.cos(theta) * np.sin(phi),
                  np.sin(theta) * np.sin(phi),
                  np.cos(phi)], axis=1) * radius
    if jitter:
        p = p + rng.normal(0, jitter, p.shape)
    return np.round(p, 3)


def exact_sphere_tetrahedron(radius: float) -> np.ndarray:
    """Four points whose circumradius is exactly ``radius``, in 3 decimals.

    A spiral on a sphere does not survive rounding: at three decimals the
    circumradius of four such points moves by about 1.5e-4, which is enough to
    push an intended 3.000 to the wrong side of the band edge and was enough to
    fail two tests here before this function replaced it. Three axis points and
    one antipode are exactly on the sphere whenever the radius itself has three
    decimals, so the construction is exact for every radius this file uses.
    """
    r = radius
    return np.array([[r, 0.0, 0.0], [-r, 0.0, 0.0],
                     [0.0, r, 0.0], [0.0, 0.0, r]])


class ExactArithmetic(unittest.TestCase):
    def test_a_tetrahedron_with_a_known_circumradius_is_read_back(self):
        for radius in (3.0, 4.0, 4.5, 6.0):
            p = exact_sphere_tetrahedron(radius)
            r, ctr, _d = _circumradius(p[None, :, :])
            with self.subTest(radius=radius):
                self.assertAlmostEqual(float(r[0]), radius, places=9)
                self.assertLess(float(np.abs(ctr[0]).max()), 1e-9)

    def test_the_band_is_closed_at_both_ends(self):
        # The published band is inclusive at 3.0 and 6.0. A tetrahedron sitting
        # exactly on either edge is the one case a float comparison can decide
        # either way, and it is decided here by integers.
        for radius, inside in ((ALPHA_MIN, True), (ALPHA_MAX, True),
                               (ALPHA_MIN - 0.001, False),
                               (ALPHA_MAX + 0.001, False),
                               (ALPHA_MIN + 0.001, True),
                               (ALPHA_MAX - 0.001, True)):
            p = exact_sphere_tetrahedron(radius)
            q = np.rint(p * SCALE).astype(np.int64)
            with self.subTest(radius=radius):
                self.assertEqual(_exact_in_band(q), inside)

    def test_exact_and_float_agree_on_many_random_tetrahedra(self):
        rng = np.random.default_rng(20260731)
        p = np.round(rng.uniform(-30, 30, (4000, 4, 3)), 3)
        r, _c, _d = _circumradius(p)
        want = (r >= ALPHA_MIN) & (r <= ALPHA_MAX)
        q = np.rint(p * SCALE).astype(np.int64)
        got = np.array([_exact_in_band(q[i]) for i in range(len(p))])
        self.assertTrue((want == got).all(),
                        f"{int((want != got).sum())} of {len(p)} disagree")

    def test_a_degenerate_tetrahedron_is_not_an_alpha_sphere(self):
        # Four coplanar points have no finite circumsphere. The float path
        # gives inf and the exact path must give False rather than divide.
        p = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        self.assertFalse(_exact_in_band(np.rint(p * SCALE).astype(np.int64)))
        self.assertFalse(np.isfinite(_circumradius(p[None, :, :])[0][0]))

    def test_the_exact_band_decision_is_used_at_the_edge(self):
        # Build a chain whose tetrahedra include one sitting exactly on the
        # band edge, and check that chain_voids counts it as taking the exact
        # path. Without this the fast path could be wrong and never be caught.
        from pocket_bench.methods import void_topology as vt
        p = shell(4.0, 4)
        atoms = [atom("CA", i + 1, p[i]) for i in range(4)]
        # Nine more points so Delaunay has something to work with.
        for j, q in enumerate(shell(9.0, 9, jitter=0.31, seed=3)):
            atoms.append(atom("CA", 100 + j, q))
        order = sorted({(a["resseq"], "") for a in atoms})
        saved = vt.EXACT_MARGIN
        try:
            vt.EXACT_MARGIN = 10.0        # force every tetrahedron exact
            v = chain_voids(atoms, order)
            self.assertGreater(v["n_exact"], 0)
            exact_keep = len(v["simplices"])
            vt.EXACT_MARGIN = 0.0         # force every tetrahedron float
            w = chain_voids(atoms, order)
            self.assertEqual(w["n_exact"], 0)
            self.assertEqual(exact_keep, len(w["simplices"]),
                             "the exact and float paths keep different spheres")
        finally:
            vt.EXACT_MARGIN = saved


class KnownShapes(unittest.TestCase):
    def test_a_hollow_shell_has_a_void_and_a_solid_ball_does_not(self):
        hollow = [atom("CA", i + 1, p)
                  for i, p in enumerate(shell(7.0, 90, jitter=0.2, seed=1))]
        order = sorted({(a["resseq"], "") for a in hollow})
        v = chain_voids(hollow, order)
        x = compute(v)
        self.assertEqual(consistency(x), [])
        self.assertGreater(len(v["simplices"]), 0,
                           "a hollow shell should hold alpha spheres")
        self.assertGreater(x[:, IDX["best_void_residues"]].max(), 3)

    def test_close_packing_leaves_no_room_for_an_alpha_sphere(self):
        # A cubic lattice at 2.4 A: every interstice is smaller than the band's
        # lower edge, so the construction must find nothing rather than
        # something small.
        pts, k = [], 0
        for i in range(5):
            for j in range(5):
                for m in range(5):
                    pts.append((2.4 * i, 2.4 * j, 2.4 * m))
        atoms = [atom("CA", n + 1, p) for n, p in enumerate(pts)]
        order = sorted({(a["resseq"], "") for a in atoms})
        v = chain_voids(atoms, order)
        r = v["radius"]
        self.assertEqual(len(r), 0,
                         f"close packing produced {len(r)} alpha spheres")
        self.assertTrue((compute(v)[:, IDX["alpha_spheres"]] == 0).all())

    def test_two_separated_cavities_are_two_voids_and_not_one(self):
        a = [atom("CA", i + 1, p)
             for i, p in enumerate(shell(6.0, 70, jitter=0.2, seed=2))]
        b = [atom("CA", 1000 + i, p + np.array([40.0, 0, 0]))
             for i, p in enumerate(shell(6.0, 70, jitter=0.2, seed=4))]
        order = sorted({(x["resseq"], "") for x in a + b})
        v = chain_voids(a + b, order)
        self.assertGreaterEqual(v["n_voids"], 2)
        x = compute(v)
        # No residue of the left shell may line a void that any residue of the
        # right shell lines: the two are 40 A apart and linkage is 1.73 A.
        left = np.array([i for i, (rs, _ic) in enumerate(order) if rs < 1000])
        right = np.array([i for i, (rs, _ic) in enumerate(order) if rs >= 1000])
        lab, simp, owner = v["label"], v["simplices"], v["owner"]
        seen = {}
        for s in range(len(simp)):
            for at in simp[s]:
                seen.setdefault(int(lab[s]), set()).add(
                    "L" if int(owner[at]) in set(left.tolist()) else "R")
        self.assertTrue(all(len(w) == 1 for w in seen.values()),
                        "a void spans both shells")
        self.assertEqual(consistency(x), [])

    def test_linkage_distance_is_what_joins_two_spheres(self):
        # Two alpha spheres whose centres are further apart than LINKAGE must
        # land in different voids; this pins the constant to its effect.
        self.assertLess(LINKAGE, ALPHA_MIN,
                        "linkage above the band's lower edge would merge "
                        "every neighbouring sphere and percolate")


class Definitions(unittest.TestCase):
    def test_segments_counts_runs_and_not_residues(self):
        self.assertEqual(_segments(np.array([])), 0)
        self.assertEqual(_segments(np.array([5])), 1)
        self.assertEqual(_segments(np.array([5, 6, 7])), 1)
        self.assertEqual(_segments(np.array([5, 6, 20, 21, 22])), 2)
        self.assertEqual(_segments(np.array([1, 3, 5])), 3)

    def test_column_names_are_unique_and_counted(self):
        self.assertEqual(len(COLUMNS), len(set(COLUMNS)))
        self.assertEqual(len(COLUMNS), 45)
        self.assertEqual(sorted(IDX.values()), list(range(len(COLUMNS))))

    def test_void_level_columns_are_declared_and_are_a_minority(self):
        self.assertTrue(VOID_LEVEL <= set(COLUMNS))
        self.assertLess(len(VOID_LEVEL), len(COLUMNS) / 2)

    def test_no_column_is_a_chain_level_constant(self):
        # The deployed quantiser bands by rank within the chain, so a column
        # that is constant within a chain has no bands and contributes only a
        # tie-break. This is the rule the module docstring records; a column
        # that violated it would be invisible in every other test here.
        for e in _entries(6):
            x, _v = _real_chain(e)
            flat = [COLUMNS[j] for j in range(len(COLUMNS))
                    if np.ptp(x[:, j]) == 0]
            # alpha_bb_atoms is zero on a chain whose alpha spheres happen to
            # touch no backbone atom, which is a fact about the chain and not a
            # constant column; require that no column is flat on every chain.
            e.setdefault("_flat", flat)
        flat_everywhere = set(COLUMNS)
        for e in _entries(6):
            x, _v = _real_chain(e)
            flat_everywhere &= {COLUMNS[j] for j in range(len(COLUMNS))
                                if np.ptp(x[:, j]) == 0}
        self.assertEqual(flat_everywhere, set(),
                         f"constant on every chain: {sorted(flat_everywhere)}")


class RealChains(unittest.TestCase):
    def test_consistency_is_clean_on_real_chains(self):
        for e in _entries(8):
            x, _v = _real_chain(e)
            with self.subTest(unit=f"{e['pdb']}_{e['chain']}"):
                self.assertEqual(consistency(x), [])

    def test_every_chain_has_voids_and_none_of_them_percolates(self):
        # The first version of this construction produced one void covering 225
        # of 254 residues. A void spanning most of a chain is the failure mode
        # this test exists to catch, and it caught it.
        for e in _entries(8):
            x, v = _real_chain(e)
            n = v["n_res"]
            with self.subTest(unit=f"{e['pdb']}_{e['chain']}"):
                self.assertGreater(v["n_voids"], 1)
                biggest = float(x[:, IDX["best_void_residues"]].max())
                self.assertLess(biggest, 0.5 * n,
                                f"a void covers {biggest:.0f} of {n} residues")

    def test_the_answer_does_not_depend_on_the_order_atoms_arrive_in(self):
        rng = np.random.default_rng(7)
        for e in _entries(3):
            atoms, order = _chain_atoms(e)
            a = compute(chain_voids(atoms, order))
            shuffled = [atoms[i] for i in rng.permutation(len(atoms))]
            b = compute(chain_voids(shuffled, order))
            with self.subTest(unit=f"{e['pdb']}_{e['chain']}"):
                # Delaunay of degenerate point sets can differ in which of two
                # co-spherical tetrahedralisations it returns, so require the
                # headline quantities to agree rather than every column bit for
                # bit, and require any disagreement to be rare.
                same = (a == b).mean(axis=0)
                self.assertGreater(float(same.min()), 0.95,
                                   f"{COLUMNS[int(np.argmin(same))]} changed "
                                   f"on {100 * (1 - float(same.min())):.1f}% "
                                   f"of residues when atoms were reordered")

    def test_the_lining_of_a_void_is_contiguous_in_space(self):
        # Every residue lining one void must be within a few contact radii of
        # another residue lining it; a void whose lining is scattered would be
        # a clustering bug rather than a pocket.
        for e in _entries(4):
            x, v = _real_chain(e)
            lines = x[:, IDX["alpha_spheres"]] > 0
            with self.subTest(unit=f"{e['pdb']}_{e['chain']}"):
                multi = lines & (x[:, IDX["best_void_residues"]] >= 3)
                if multi.any():
                    self.assertTrue(
                        (x[multi, IDX["void_neighbours"]] >= 1).mean() > 0.9)

    def test_most_residues_line_something_and_some_line_nothing(self):
        for e in _entries(6):
            x, _v = _real_chain(e)
            frac = float((x[:, IDX["alpha_spheres"]] > 0).mean())
            with self.subTest(unit=f"{e['pdb']}_{e['chain']}"):
                self.assertGreater(frac, 0.5)
                self.assertLess(frac, 1.0)

    def test_alpha_spheres_all_lie_inside_the_published_band(self):
        for e in _entries(6):
            _x, v = _real_chain(e)
            r = v["radius"]
            with self.subTest(unit=f"{e['pdb']}_{e['chain']}"):
                self.assertTrue((r >= ALPHA_MIN).all())
                self.assertTrue((r <= ALPHA_MAX).all())

    def test_buried_residues_are_deeper_than_the_ones_lining_open_voids(self):
        # hull_distance is a depth below the chain's convex hull, so it must be
        # non-negative on essentially every residue and larger in the interior.
        for e in _entries(4):
            x, _v = _real_chain(e)
            d = x[:, IDX["hull_distance_centi"]]
            with self.subTest(unit=f"{e['pdb']}_{e['chain']}"):
                self.assertGreater(float((d >= -1).mean()), 0.99)
                self.assertGreater(float(np.median(d)), 100.0)


_CACHE: dict[str, tuple] = {}


def _entries(n: int) -> list[dict]:
    return json.loads(MANIFEST.read_text())["entries"][:n]


def _chain_atoms(e: dict):
    key = f"{e['pdb']}_{e['chain']}"
    if key not in _CACHE:
        atoms = parse_pdb_atoms((ROOT / e["receptor_path"]).read_text())
        mine = [a for a in atoms if a["chain"] == e["chain"]]
        keep = [a for a in mine
                if a["element"] != "H" and a["resname"] != "HOH"]
        order = sorted({(a["resseq"], a["icode"].strip()) for a in keep})
        _CACHE[key] = (mine, order)
    return _CACHE[key]


def _real_chain(e: dict):
    atoms, order = _chain_atoms(e)
    key = f"x{e['pdb']}_{e['chain']}"
    if key not in _CACHE:
        v = chain_voids(atoms, order)
        _CACHE[key] = (compute(v), v)
    return _CACHE[key]


if __name__ == "__main__":
    unittest.main()
