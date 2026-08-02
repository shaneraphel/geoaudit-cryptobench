#!/usr/bin/env python3.12
"""Train-fold wires for the crystal-form / displacement-ellipsoid family.

Builds ``data/cryptobench_apo/_crystal_form_cache_train.npz`` under the same
own / contact-8A / walk-2 aggregations as every other family, aligned to the
wide-cache centroids so the residue universe is bit-identical.

This family is the first in the repository whose input is not the committed
receptor file. It reads the *deposited* PDB entry as well, for the crystal form
and the anisotropic displacement parameters, which the committed receptors do
not carry. That changes the detector's input class and the arm is kept separate
for that reason; see ``docs/DECISIONS.md``.

Two things are verified per unit before any column is written, because
combining two files silently is how a family gets built on a mismatch:

* the deposited entry contains the chain we score;
* the residue universe and centroids agree with the wide cache to
  ``CENTROID_TOLERANCE``, exactly as the other wire builders require.

A unit whose deposited entry is missing, or whose chain is absent from it, is
**not dropped**. Its columns are zero and ``adp_present``/``n_mate_*`` read
zero, which is a level the quantiser can address. The count of such units is
reported and must be quoted beside any lift.

``clinical_grade`` is false.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pocket_bench.methods.crystal_form import (
    COLUMNS as CRYSTAL_COLUMNS,
    compute,
    consistency,
    read_entry,
)
from pocket_bench.paths import ROOT
from pocket_bench.pdb_io import parse_pdb_atoms

WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/train_manifest.json"
DEPOSITED = ROOT / "data/deposited_entries"
CACHE = ROOT / "data/cryptobench_apo/_crystal_form_cache_train.npz"
CACHE_PERM = ROOT / "data/cryptobench_apo/_crystal_form_perm_cache_train.npz"
COVERAGE = ROOT / "results/architecture_sweep/CRYSTAL_FORM_COVERAGE.json"

CONTACT_RADIUS = 8.0
AGGREGATIONS = ("own", "contact", "walk2")
PERMUTATION_SEED = 20260802
CENTROID_TOLERANCE = 5e-4
SKIP = frozenset({"HOH", "WAT", "DOD"})
N_JOBS = min(9, os.cpu_count() or 4)


def column_names() -> tuple[str, ...]:
    return tuple(f"{agg}~{q}" for agg in AGGREGATIONS for q in CRYSTAL_COLUMNS)


def build_chain(ctr: np.ndarray, prop: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = (d <= CONTACT_RADIUS).astype(np.float64)
    two = adj @ adj
    np.fill_diagonal(two, 0.0)
    return np.concatenate([prop, adj @ prop, two @ prop], axis=1)


def _residue_rows(atoms: list[dict], chain: str):
    keep = [a for a in atoms
            if a["chain"] == chain and a["element"] != "H"
            and a["resname"] not in SKIP]
    poly: dict[tuple[int, str], list[tuple[float, float, float]]] = {}
    atoms_by: dict[tuple[int, str], list[dict]] = {}
    for a in keep:
        key = (a["resseq"], a["icode"].strip())
        poly.setdefault(key, []).append((a["x"], a["y"], a["z"]))
        atoms_by.setdefault(key, []).append(a)
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
    return order, ctr, take, atoms_by


def _entry_path(pdb: str) -> Path | None:
    for suffix in (".pdb.gz", ".cif.gz"):
        p = DEPOSITED / f"{pdb.lower()}{suffix}"
        if p.exists():
            return p
    return None


def _one_chain(args):
    u, n, path, chain, pdb, ctr_slice = args
    atoms = parse_pdb_atoms(Path(path).read_text())
    order, ctr, take, atoms_by = _residue_rows(atoms, chain)
    if len(take) != n:
        return ("err", u, f"universe {len(take)} != cache {n}")
    drift = float(np.abs(ctr - ctr_slice).max())
    if drift > CENTROID_TOLERANCE:
        return ("err", u, f"centroid drift {drift:.2e}")

    atoms_by_res = [atoms_by[order[k]] for k in range(len(order))]
    resseqs = [order[k][0] for k in range(len(order))]
    full_ctr = np.array([np.mean([[a["x"], a["y"], a["z"]] for a in r], axis=0)
                         for r in atoms_by_res], dtype=np.float64)

    ep = _entry_path(pdb)
    entry = read_entry(ep) if ep is not None else None
    note = "ok"
    if entry is None:
        note = "no_deposited_entry"
    elif entry.cell is None:
        note = "no_cell"
    elif not any(a["chain"] == chain for a in entry.atoms):
        note = "chain_absent_from_entry"
        entry = None
    elif ep is not None and ep.suffix == ".gz" and ep.name.endswith(".cif.gz"):
        note = "cif_only_not_parsed"
        entry = None

    x_full = compute(entry, chain, atoms_by_res, resseqs, full_ctr)
    x = x_full[take]
    bad = consistency(x)
    if bad:
        return ("err", u, f"consistency {bad}")
    has_adp = bool(x[:, list(CRYSTAL_COLUMNS).index("adp_present")].max() > 0)
    has_mate = bool(x[:, list(CRYSTAL_COLUMNS).index("n_mate_atoms_6A")].max() > 0)
    sg = entry.space_group if entry is not None else ""
    return ("ok", u, drift, x, note, has_adp, has_mate, sg)


def raw_quantities() -> tuple[np.ndarray, np.ndarray, dict]:
    w = np.load(WIDE, allow_pickle=False)
    units = [str(u) for u in w["units"]]
    n_res, ctr_cached = w["n_res_per"], w["ctr"]
    w.close()
    meta = {f"{e['pdb']}_{e['chain']}":
            (str(ROOT / e["receptor_path"]), e["chain"], e["pdb"])
            for e in json.loads(MANIFEST.read_text())["entries"]}
    jobs, offsets, off = [], {}, 0
    for u, n in zip(units, n_res):
        n = int(n)
        if u not in meta:
            raise SystemExit(f"{u} missing from train manifest")
        path, chain, pdb = meta[u]
        jobs.append((u, n, path, chain, pdb, ctr_cached[off:off + n].copy()))
        offsets[u] = off
        off += n

    out = np.zeros((int(n_res.sum()), len(CRYSTAL_COLUMNS)), dtype=np.float64)
    notes: dict[str, str] = {}
    adp_units: list[str] = []
    mate_units: list[str] = []
    groups: dict[str, int] = {}
    worst, worst_unit, done = 0.0, "", 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futs = [ex.submit(_one_chain, j) for j in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec[0] == "err":
                raise SystemExit(f"{rec[1]}: {rec[2]}")
            _, u, drift, x, note, has_adp, has_mate, sg = rec
            notes[u] = note
            if has_adp:
                adp_units.append(u)
            if has_mate:
                mate_units.append(u)
            if sg:
                groups[sg] = groups.get(sg, 0) + 1
            if drift > worst:
                worst, worst_unit = drift, u
            out[offsets[u]:offsets[u] + len(x)] = x
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(units)} chains  "
                      f"{time.perf_counter() - t0:.0f}s  jobs={N_JOBS}",
                      flush=True)
    print(f"  centroids agree to {worst:.2e} (worst {worst_unit})", flush=True)

    bad = {u: v for u, v in notes.items() if v != "ok"}
    cov = {
        "schema": "geoaudit.crystal_form_coverage.v1",
        "clinical_grade": False,
        "reads_test_fold": False,
        "n_units": len(units),
        "n_units_definition": (
            "training-fold chains in the wide cache, one per (pdb, chain)"),
        "n_units_with_symmetry_mate_contacts": len(mate_units),
        "n_units_with_symmetry_mate_contacts_definition": (
            "chains where at least one residue has a symmetry-mate heavy atom "
            "within 6 A"),
        "n_units_with_anisou": len(adp_units),
        "n_units_with_anisou_definition": (
            "chains where at least one residue had an ANISOU record on at "
            "least one of its heavy atoms in the deposited entry"),
        "anisou_coverage_fraction": round(len(adp_units) / max(len(units), 1), 4),
        "mate_coverage_fraction": round(len(mate_units) / max(len(units), 1), 4),
        "n_units_without_usable_entry": len(bad),
        "units_without_usable_entry": bad,
        "why_they_are_not_dropped": (
            "a family defined where a record exists and undefined elsewhere is "
            "a family with a missing level, not a family with fewer units; "
            "absent entries take a fixed zero on every column and quantise to "
            "a flat rank, which addresses one cell"
        ),
        "space_group_histogram": dict(sorted(groups.items(),
                                             key=lambda kv: -kv[1])[:40]),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    return out, n_res, cov


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

    prop, n_res, cov = raw_quantities()
    COVERAGE.parent.mkdir(parents=True, exist_ok=True)
    COVERAGE.write_text(json.dumps(cov, indent=2) + "\n")
    print(f"  ANISOU on {cov['n_units_with_anisou']}/{cov['n_units']} chains "
          f"({cov['anisou_coverage_fraction']:.1%}); mate contacts on "
          f"{cov['n_units_with_symmetry_mate_contacts']} "
          f"({cov['mate_coverage_fraction']:.1%})", flush=True)
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
    ap.add_argument("--permuted", action="store_true")
    a = ap.parse_args(argv)
    C, names = build_or_load(a.rebuild, a.permuted)
    print(f"\n{C.shape[1]} columns over {C.shape[0]} residues  "
          f"({len(CRYSTAL_COLUMNS)} qty x 3 aggs)")
    nz = (C != 0).mean(axis=0)
    print(f"  non-zero share: min {nz.min():.3f}  median "
          f"{float(np.median(nz)):.3f}  max {nz.max():.3f}")
    dead = [names[j] for j in range(C.shape[1]) if nz[j] == 0.0]
    if dead:
        print(f"  {len(dead)} all-zero columns, first few: {dead[:6]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
