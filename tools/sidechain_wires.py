#!/usr/bin/env python3
"""Wires from side-chain conformation, the second input the pipeline discards.

What this is
------------
``src/pocket_bench/methods/sidechain_geometry.py`` computes 86 quantities from
the positions of every side-chain heavy atom. This turns them into columns the
counting field can read, by the same three aggregations ``chemistry_wires.py``
and ``backbone_wires.py`` use -- the residue's own value, the sum over residues
within 8 A, and the sum two steps out on the contact graph -- so that a
comparison against either of those families is a comparison of the quantities and
not of the attachment.

Why it is worth building
------------------------
``AGENT_MEMORY`` 2i states the screen: a family is worth measuring only if it
reads bytes the pipeline throws away. Six families that failed that screen
measured null. The seventh, backbone geometry, passed it and is worth +0.00441
against the deployed detector on 12 of 12 splits. 2j-bis names side-chain
conformation as the next family on the same rule and it is the one this file
builds: a residue centroid does not determine chi1, so a leucine in gauche-plus
and one in trans are different geometry with identical identity, and the deployed
bank carries only per-type constants which do not move when the rotamer does.

The controls that matter, and there are two
--------------------------------------------
``more_old`` -- the same number of extra tables drawn from the deployed wires
with the family absent -- catches a family that is only adding cells. It does not
catch a family whose columns carry ordinary per-residue information that happens
not to be about the side chain.

So this file also builds ``permuted``: the same 86 quantities, aggregated
identically, with the rows shuffled inside each chain under a fixed seed. That
preserves every column's multiset of values over the chain exactly and destroys
only the correspondence between a row and the residue it describes. On the
backbone family the permuted arm went from -0.0007 at 39 columns to -0.0021 at
132, which is the sharpest available statement that the lift is the conformation
of the residue being scored rather than the shape of a column. The same question
is asked here.

Alignment is checked, not assumed
---------------------------------
The rows of the wide cache are residues in an order this file has to reproduce
exactly, and nothing would raise if it did not: every lookup would succeed and
every residue would be handed another residue's side chain. So the centroid of
each residue's heavy atoms is recomputed here and compared against the cache's
own ``ctr``, row by row. This is the check ``backbone_wires.py`` records as the
only thing standing between it and a silent off-by-one, and the same reasoning
applies unchanged.

The backbone context group J needs
-----------------------------------
Five of the 86 quantities couple chi1 to phi, to psi, and to the Ramachandran
cell. Those are backbone quantities, so they are computed here from the same
parse and handed to ``sidechain_geometry.compute``. They are not new columns of
the backbone family -- the coupling is a function of both and belongs to neither
alone -- and the backbone family's own columns are not emitted here, so the two
families remain disjoint blocks and can be measured together or apart.

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
    compute as backbone_compute,
)
from pocket_bench.methods.sidechain_geometry import (
    COLUMNS as SC_COLUMNS,
    chain_sidechain,
    compute,
    consistency,
)
from pocket_bench.pdb_io import parse_pdb_atoms
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.sidechain_wires.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
CACHE = ROOT / "data/cryptobench_apo/_sidechain_cache_train.npz"
CACHE_PERM = ROOT / "data/cryptobench_apo/_sidechain_perm_cache_train.npz"

CONTACT_RADIUS = 8.0
AGGREGATIONS = ("own", "contact", "walk2")
PERMUTATION_SEED = 20260731
CENTROID_TOLERANCE = 5e-4

# Restated rather than imported, so that a change to either rule is visible in
# the diff of this file. This is the convention backbone_wires.py adopted.
SKIP = frozenset({"HOH", "WAT", "DOD"})

_BB = {n: j for j, n in enumerate(BB_COLUMNS)}


def column_names() -> tuple[str, ...]:
    return tuple(f"{agg}~{q}" for agg in AGGREGATIONS for q in SC_COLUMNS)


def build_chain(ctr: np.ndarray, prop: np.ndarray) -> np.ndarray:
    """The three aggregations for one chain, identical to the other families'."""
    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = (d <= CONTACT_RADIUS).astype(np.float64)
    two = adj @ adj
    np.fill_diagonal(two, 0.0)
    return np.concatenate([prop, adj @ prop, two @ prop], axis=1)


def _residue_rows(atoms: list[dict], chain: str
                  ) -> tuple[list[tuple[int, str]], np.ndarray, np.ndarray]:
    """The polymer's residues, the evaluation universe, and the map between them.

    Duplicated from ``backbone_wires`` deliberately rather than imported. That
    file feeds ``BACKBONE_WIDE_LIFT.json`` and importing from it would put this
    file inside that artifact's blast radius for no benefit; AGENTS.md's rule
    about minimal blast radius near a pinned artifact is the reason.
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
    first: dict[int, int] = {}
    for j, (r, _ic) in enumerate(order):
        first.setdefault(r, j)
    take = np.array([first[r] for r in universe], dtype=np.int64)

    merged: dict[int, list] = {}
    for (r, _ic), pts in poly.items():
        merged.setdefault(r, []).extend(pts)
    ctr = np.array([np.mean(merged[r], axis=0) for r in universe],
                   dtype=np.float64)
    return order, ctr, take


def raw_quantities() -> tuple[np.ndarray, np.ndarray]:
    """The 86 per-residue quantities for every training residue."""
    w = np.load(WIDE, allow_pickle=False)
    units = [str(u) for u in w["units"]]
    n_res, ctr_cached = w["n_res_per"], w["ctr"]
    w.close()

    paths = {f"{e['pdb']}_{e['chain']}": (ROOT / e["receptor_path"], e["chain"])
             for e in json.loads(MANIFEST.read_text())["entries"]}

    out = np.zeros((int(n_res.sum()), len(SC_COLUMNS)), dtype=np.float64)
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
                f"derives is not the cache's order, and every side-chain "
                f"quantity would be attached to the wrong residue without "
                f"raising")

        mine = [a for a in atoms if a["chain"] == chain]
        b = backbone_compute(chain_backbone(mine, order))
        per, names = chain_sidechain(mine, order)
        x = compute(per, names,
                    phi=b[:, [_BB["cos_phi"], _BB["sin_phi"]]],
                    psi=b[:, [_BB["cos_psi"], _BB["sin_psi"]]],
                    rama=b[:, _BB["rama_region"]])[take]
        bad = consistency(x)
        if bad:
            raise SystemExit(f"{u}: side-chain quantities violate {bad}")
        out[off:off + n] = x
        off += n
        if (i + 1) % 100 == 0:
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

    prop, n_res = raw_quantities()
    if permuted:
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
        print(f"  dead: {[n for n, f in zip(names, nz) if f == 0]}")
    # A column that never varies inside a chain is invisible to a within-chain
    # quantiser, whatever its values are across the fold. That is a different
    # failure from a dead column and it is the one worth printing here.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
