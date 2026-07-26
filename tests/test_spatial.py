"""The cell list must reproduce the dense scan exactly, not approximately.

Every neighbour query in the detectors used to be a blocked dense scan. Replacing
it with a cell list is only legitimate if the reported pair set and the order in
which pairs are reported are both unchanged: the set because the predicate is the
science, and the order because downstream ``np.add.at`` accumulations and the
Kruskal tie-break on equal weights are both order-sensitive at the last bit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pocket_bench.methods.geometric_foundation import _receptor_coords
from pocket_bench.spatial import counts_within, cross_within, self_pairs_within

ROOT = Path(__file__).resolve().parents[1]
RECEPTORS = ROOT / "data/cryptobench_apo/official_receptors"


def _dense_cross(q, c, r):
    d2 = ((q[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    return np.nonzero(d2 <= r * r)


def _dense_counts(c, r):
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    return (d2 <= r * r).sum(1) - 1


def _dense_pairs(c, r):
    d2 = ((c[:, None, :] - c[None, :, :]) ** 2).sum(-1)
    i, j = np.nonzero(d2 <= r * r)
    k = i < j
    i, j = i[k], j[k]
    o = np.lexsort((j, i))
    return np.stack([i[o], j[o]], axis=1), np.sqrt(d2[i[o], j[o]])


@pytest.mark.parametrize("radius", [4.0, 6.0, 8.0])
def test_matches_dense_scan_on_synthetic(radius):
    rng = np.random.default_rng(11)
    # A dense core plus a detached satellite plus exact duplicates: the three
    # cases that break naive binning (empty cubes, out-of-box queries, ties).
    coords = np.concatenate([
        rng.normal(0.0, 6.0, (300, 3)),
        rng.normal(40.0, 1.5, (60, 3)),
    ])
    coords = np.concatenate([coords, coords[:8]])
    query = np.concatenate([rng.normal(0.0, 9.0, (120, 3)), coords[:5],
                            np.full((3, 3), -500.0)])

    qi, ai = cross_within(query, coords, radius)
    rqi, rai = _dense_cross(query, coords, radius)
    assert np.array_equal(qi, rqi)
    assert np.array_equal(ai, rai)

    assert np.array_equal(counts_within(coords, radius),
                          _dense_counts(coords, radius))

    pairs, dist = self_pairs_within(coords, radius)
    rpairs, rdist = _dense_pairs(coords, radius)
    assert np.array_equal(pairs, rpairs)
    assert np.array_equal(dist, rdist)          # bitwise, not approx


def test_matches_dense_scan_on_real_receptor():
    path = RECEPTORS / "1arl_A_receptor.pdb"
    if not path.is_file():
        pytest.skip("official receptors not materialized")
    coords = _receptor_coords(path)

    assert np.array_equal(counts_within(coords, 8.0), _dense_counts(coords, 8.0))
    pairs, dist = self_pairs_within(coords, 8.0)
    rpairs, rdist = _dense_pairs(coords, 8.0)
    assert np.array_equal(pairs, rpairs)
    assert np.array_equal(dist, rdist)


def test_empty_and_degenerate_inputs():
    empty = np.zeros((0, 3))
    one = np.zeros((1, 3))
    assert cross_within(empty, one, 5.0)[0].size == 0
    assert cross_within(one, empty, 5.0)[0].size == 0
    assert self_pairs_within(one, 5.0)[0].shape == (0, 2)
    assert counts_within(one, 5.0).tolist() == [0]       # itself, minus itself
