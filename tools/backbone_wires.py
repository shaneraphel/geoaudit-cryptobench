#!/usr/bin/env python3
"""Wires from backbone conformation, the one input the pipeline throws away.

What this is
------------
``src/pocket_bench/methods/backbone_geometry.py`` computes thirteen quantities
from the positions of N, CA, C, O and CB. This turns them into columns the
counting field can read, by the same three aggregations
``tools/chemistry_wires.py`` uses -- the residue's own value, the sum over
residues within 8 A, and the sum two steps out on the contact graph -- so that a
comparison against the chemistry family is a comparison of the quantities and not
of the attachment.

Why it is worth building after six null families
------------------------------------------------
Every previous family was a function of data the deployed pipeline already reads.
A residue is a centroid to that pipeline; recovering a backbone torsion needs N
and C, which are discarded at parse time, so two different backbone
conformations can present the same centroids and the same contact graph. This is
the first family that is not a re-encoding, and ``AGENT_MEMORY`` 2i records the
rule that makes that the deciding question.

The control that matters
------------------------
``more_old`` -- the same number of extra tables drawn from the deployed wires
with the family absent -- catches a family that is only adding cells. It does not
catch a family whose columns carry ordinary per-residue information that happens
not to be about the backbone. So this file also builds ``permuted``: the same
thirteen quantities, aggregated identically, with the rows shuffled inside each
chain under a fixed seed.

That control preserves every column's marginal distribution over the chain
exactly -- it is the same multiset of values -- and destroys only the
correspondence between a row and the residue it describes. If the real family
beats it, the gain is the backbone geometry of the residue being scored. If they
are level, the gain is whatever a column of that shape gives any detector, and
the family is decoration. Two arms differing in one permutation is a sharper
question than two arms differing in a construction.

What it costs
-------------
One pass over the 770 training receptors, seconds, and no network. The measuring
run is ``tools/straddling_attachment.py --family "backbone 39"``, which is the
same twelve cluster-disjoint halvings every other family was measured on.

Alignment is checked, not assumed
---------------------------------
The rows of the wide cache are residues in an order this file has to reproduce
exactly, and nothing would raise if it did not: every lookup would succeed and
every residue would be handed another residue's backbone. So the centroid of each
residue's heavy atoms is recomputed here and compared against the cache's own
``ctr``, row by row, for all 234,838 of them. A tolerance is not a formality here
-- it is the only thing standing between this file and a silent off-by-one.

Nothing here reads a label, the test fold, or any external unit.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from pocket_bench.methods.backbone_geometry import (
    COLUMNS as BB_COLUMNS,
    chain_backbone,
    compute,
    consistency,
)
from pocket_bench.pdb_io import parse_pdb_atoms
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.backbone_wires.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
CACHE = ROOT / "data/cryptobench_apo/_backbone_cache_train.npz"
CACHE_PERM = ROOT / "data/cryptobench_apo/_backbone_perm_cache_train.npz"

CONTACT_RADIUS = 8.0
AGGREGATIONS = ("own", "contact", "walk2")
PERMUTATION_SEED = 20260731
CENTROID_TOLERANCE = 5e-4   # the cache stores float64 centroids of float32 input

# Non-solvent heteroatoms are rejected upstream by the ligand-leak guard, and
# hydrogens never enter any geometry here. Both rules are restated rather than
# imported so that a change to either is visible in the diff of this file.
SKIP = frozenset({"HOH", "WAT", "DOD"})


def column_names() -> tuple[str, ...]:
    return tuple(f"{agg}~{q}" for agg in AGGREGATIONS for q in BB_COLUMNS)


def build_chain(ctr: np.ndarray, prop: np.ndarray) -> np.ndarray:
    """The three aggregations for one chain, identical to the chemistry family's.

    ``prop`` is ``(n_residues, 13)``; this function performs no geometry of its
    own beyond the contact graph, which keeps the backbone reasoning in one file
    and the attachment in another.
    """
    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = (d <= CONTACT_RADIUS).astype(np.float64)
    two = adj @ adj
    np.fill_diagonal(two, 0.0)
    return np.concatenate([prop, adj @ prop, two @ prop], axis=1)


def _residue_rows(atoms: list[dict], chain: str
                  ) -> tuple[list[tuple[int, str]], np.ndarray, np.ndarray]:
    """The polymer's residues, the evaluation universe, and the map between them.

    These are two different lists and conflating them is the bug this function
    exists to prevent. A torsion is a property of the polymer, so it has to be
    computed over residues in the order they are bonded, keyed by resseq *and*
    insertion code. The evaluation universe every detector here is scored on is
    the sorted set of *integer* resseq, which merges a residue numbered 52A into
    52. On a chain carrying insertion codes the two lists have different lengths,
    and 1dc6_A is the case that caught it: 330 polymer residues, 329 universe
    rows.

    So the polymer order is returned for the geometry, the universe centroids for
    the alignment check, and an index array saying which polymer residue supplies
    each universe row. Where several share a resseq the first in polymer order
    supplies it: a torsion is not an average of two conformers, and the first is
    the one the deposit lists as the parent numbering.
    """
    keep = [a for a in atoms
            if a["chain"] == chain and a["element"] != "H"
            and a["resname"] not in SKIP]
    poly: dict[tuple[int, str], list[tuple[float, float, float]]] = {}
    for a in keep:
        poly.setdefault((a["resseq"], a["icode"].strip()), []).append(
            (a["x"], a["y"], a["z"]))
    order = sorted(poly)

    universe = sorted({r for r, _ in order})
    first = {}
    for j, (r, _ic) in enumerate(order):
        first.setdefault(r, j)
    take = np.array([first[r] for r in universe], dtype=np.int64)

    # The cache's centroid for a merged resseq is over every atom carrying it,
    # so the check is computed the same way rather than over the first alone.
    merged: dict[int, list] = {}
    for (r, _ic), pts in poly.items():
        merged.setdefault(r, []).extend(pts)
    ctr = np.array([np.mean(merged[r], axis=0) for r in universe],
                   dtype=np.float64)
    return order, ctr, take


def raw_quantities(force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """The thirteen per-residue quantities for every training residue.

    Returns the (n_residues, 13) array and the per-chain residue counts, so a
    caller that wants to permute inside chains can do so without reparsing.
    """
    w = np.load(WIDE, allow_pickle=False)
    units = [str(u) for u in w["units"]]
    n_res, ctr_cached = w["n_res_per"], w["ctr"]
    w.close()

    paths = {f"{e['pdb']}_{e['chain']}": (ROOT / e["receptor_path"], e["chain"])
             for e in json.loads(MANIFEST.read_text())["entries"]}

    out = np.zeros((int(n_res.sum()), len(BB_COLUMNS)), dtype=np.float64)
    worst, worst_unit, off, t0 = 0.0, "", 0, time.perf_counter()
    for i, (u, n) in enumerate(zip(units, n_res)):
        n = int(n)
        if u not in paths:
            raise SystemExit(f"{u} is in the wide cache and not in the manifest")
        path, chain = paths[u]
        atoms = parse_pdb_atoms(path.read_text())
        order, ctr, take = _residue_rows(atoms, chain)
        if len(take) != n:
            raise SystemExit(
                f"{u}: the receptor gives {len(take)} universe rows and the "
                f"wide cache holds {n}; the rows are not the same residues")
        d = float(np.abs(ctr - ctr_cached[off:off + n]).max())
        if d > worst:
            worst, worst_unit = d, u
        if d > CENTROID_TOLERANCE:
            raise SystemExit(
                f"{u}: recomputed centroids differ from the cache by {d:.2e}, "
                f"above {CENTROID_TOLERANCE:.0e}. The residue order this file "
                f"derives is not the cache's order, and every backbone quantity "
                f"would be attached to the wrong residue without raising")
        x = compute(chain_backbone(
            [a for a in atoms if a["chain"] == chain], order))[take]
        bad = consistency(x)
        if bad:
            raise SystemExit(f"{u}: backbone quantities violate {bad}")
        out[off:off + n] = x
        off += n
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(units)} chains  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    print(f"  centroids agree with the cache to {worst:.2e} (worst on "
          f"{worst_unit})", flush=True)
    return out, n_res


def _aggregate(prop: np.ndarray, n_res: np.ndarray) -> np.ndarray:
    w = np.load(WIDE, allow_pickle=False)
    ctr = w["ctr"]
    w.close()
    blocks, off = [], 0
    for n in n_res:
        n = int(n)
        blocks.append(build_chain(ctr[off:off + n], prop[off:off + n]))
        off += n
    return np.concatenate(blocks, axis=0).astype(np.float32)


def build_or_load(force: bool = False, permuted: bool = False
                  ) -> tuple[np.ndarray, tuple[str, ...]]:
    names = column_names()
    cache = CACHE_PERM if permuted else CACHE
    if cache.is_file() and not force:
        z = np.load(cache, allow_pickle=False)
        if tuple(str(s) for s in z["names"]) == names:
            print(f"reusing {cache.relative_to(ROOT)}  {z['C'].shape}",
                  flush=True)
            return z["C"], names
        print("cached column names differ; rebuilding", flush=True)

    prop, n_res = raw_quantities(force)
    if permuted:
        # Shuffle rows inside each chain. Every column's multiset of values over
        # the chain is unchanged, so the marginals are identical by construction
        # and the only thing destroyed is which residue each row describes.
        rng = np.random.default_rng(PERMUTATION_SEED)
        off = 0
        for n in n_res:
            n = int(n)
            prop[off:off + n] = prop[off:off + n][rng.permutation(n)]
            off += n
    C = _aggregate(prop, n_res)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, C=C, names=np.asarray(names))
    print(f"wrote {cache.relative_to(ROOT)}  {C.shape}", flush=True)
    return C, names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--permuted", action="store_true",
                    help="build the row-permuted control instead")
    a = ap.parse_args(argv)
    C, names = build_or_load(a.rebuild, a.permuted)
    print(f"\n{C.shape[1]} columns over {C.shape[0]} residues")
    nz = (C != 0).mean(axis=0)
    print(f"  columns that are never zero anywhere: {int((nz == 1).sum())}")
    print(f"  columns that are always zero:         {int((nz == 0).sum())}")
    if int((nz == 0).sum()):
        dead = [n for n, f in zip(names, nz) if f == 0]
        print(f"  dead: {dead}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
