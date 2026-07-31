#!/usr/bin/env python3
"""Wires from void topology, the third input the pipeline discards.

What this is
------------
``src/pocket_bench/methods/void_topology.py`` computes 45 quantities from the
Delaunay tetrahedralisation of a chain's heavy atoms and the alpha spheres that
survive fpocket's published radius band. This turns them into columns the
counting field can read, under the same three aggregations that
``chemistry_wires.py``, ``backbone_wires.py`` and ``sidechain_wires.py`` use --
the residue's own value, the sum over residues within 8 A, and the sum two steps
out on the contact graph -- so a comparison against any of those families is a
comparison of the quantities and not of the attachment.

Why this family and not another
-------------------------------
Two rules had to agree before this was built, and they do.

The first is ``AGENT_MEMORY`` 2i: a family is worth measuring only if it reads
bytes the pipeline throws away. Six families that failed that screen measured
null; backbone and side-chain conformation passed it and are worth +0.0044 and
+0.0048. Void topology passes it differently from either. It is not another
function of where the atoms are -- it is a function of where they are *not*, and
the object it computes, the connectivity of the empty space, is not determined by
any per-residue quantity at all.

The second is the instruction that an operator built after a failure should be
built for the observable the failure correlates with.
``FAILURE_TAIL.json`` finds that observable: the field scores 0.5991 on the 188
units with fewer than ten cryptic residues against 0.8766 on the 201 with more
than twenty-two, and the spread within the small-pocket stratum is 2.63 times the
sampling error of an AUC at that many positives, so this is not the metric being
noisy. ``GATE_BY_STRATUM.json`` then finds the mechanism: the deployed 18 A
spatial gate's optimum radius by stratum is 0, 14, 14, 14. On a chain whose
pocket is five residues, an 18 A ball around a cryptic residue is almost entirely
non-cryptic and the gate averages the signal away; on a chain whose pocket is
thirty, the same ball is enriched and the gate reinforces. That artifact closes
with the observation that a radius chosen by stratum cannot ship, because the
stratum is the label.

The size of the void a residue lines is that stratum, computed from coordinates
alone. ``best_void_residues`` has a mean within-chain ROC-AUC against the cryptic
label of 0.7124 on the first sixty training chains, from one integer with nothing
fitted, and ``ball_gate_purity_permille`` -- the share of the deployed gate's own
ball that lines the same void -- is the dilution factor itself, per residue,
without the label.

The controls, and there are two
-------------------------------
``more_old`` adds the same number of extra tables over the deployed wires with
the family absent, which catches a family that is only buying cells. It does not
catch a family whose columns carry ordinary per-residue information that happens
not to be about the void, so ``permuted`` is built here too: the same 45
quantities, aggregated identically, rows shuffled within each chain under a fixed
seed. Every column's multiset over the chain is preserved exactly and only the
correspondence between a row and its residue is destroyed. On the backbone family
that arm ran -0.0021 against the family's +0.0044.

Alignment is checked, not assumed
---------------------------------
The rows of the wide cache are residues in an order this file must reproduce
exactly, and nothing would raise if it did not: every lookup would succeed and
every residue would be handed another residue's void. So each residue's heavy-
atom centroid is recomputed and compared against the cache's own ``ctr``, row by
row, which is the check that stands between this file and a silent off-by-one.

Nothing here reads a label, the test fold, or any external unit.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from pocket_bench.methods.void_topology import (
    COLUMNS as VOID_COLUMNS,
    chain_voids,
    compute,
    consistency,
)
from pocket_bench.pdb_io import parse_pdb_atoms
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.void_wires.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
CACHE = ROOT / "data/cryptobench_apo/_void_cache_train.npz"
CACHE_PERM = ROOT / "data/cryptobench_apo/_void_perm_cache_train.npz"

CONTACT_RADIUS = 8.0
AGGREGATIONS = ("own", "contact", "walk2")
PERMUTATION_SEED = 20260731
CENTROID_TOLERANCE = 5e-4

SKIP = frozenset({"HOH", "WAT", "DOD"})


def column_names() -> tuple[str, ...]:
    return tuple(f"{agg}~{q}" for agg in AGGREGATIONS for q in VOID_COLUMNS)


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

    Duplicated from ``backbone_wires`` deliberately rather than imported, for
    the reason ``sidechain_wires`` records: that file feeds a pinned artifact and
    importing from it would put this file inside that artifact's blast radius for
    no benefit.
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
    """The 45 per-residue quantities for every training residue."""
    w = np.load(WIDE, allow_pickle=False)
    units = [str(u) for u in w["units"]]
    n_res, ctr_cached = w["n_res_per"], w["ctr"]
    w.close()

    paths = {f"{e['pdb']}_{e['chain']}": (ROOT / e["receptor_path"], e["chain"])
             for e in json.loads(MANIFEST.read_text())["entries"]}

    out = np.zeros((int(n_res.sum()), len(VOID_COLUMNS)), dtype=np.float64)
    worst, worst_unit, off, t0 = 0.0, "", 0, time.perf_counter()
    n_exact_total = 0
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
                f"derives is not the cache's order, and every void quantity "
                f"would be attached to the wrong residue without raising")

        mine = [a for a in atoms if a["chain"] == chain]
        v = chain_voids(mine, order)
        n_exact_total += int(v["n_exact"])
        x = compute(v)[take]
        bad = consistency(x)
        if bad:
            raise SystemExit(f"{u}: void quantities violate {bad}")
        out[off:off + n] = x
        off += n
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(units)} chains  "
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    print(f"  centroids agree with the cache to {worst:.2e} (worst on "
          f"{worst_unit})", flush=True)
    print(f"  {n_exact_total} tetrahedra were decided in exact integer "
          f"arithmetic at the band edge", flush=True)
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
    print(f"  non-zero share: min {nz.min():.3f}  median "
          f"{float(np.median(nz)):.3f}  max {nz.max():.3f}")
    dead = [names[j] for j in np.flatnonzero(np.ptp(C, axis=0) == 0)]
    if dead:
        print(f"  {len(dead)} constant columns: {dead[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
