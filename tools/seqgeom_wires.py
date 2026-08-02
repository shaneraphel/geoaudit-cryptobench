#!/usr/bin/env python3.12
"""Sequence–geometry combinatorics without a language model.

Prediction (before measurement)
-------------------------------
pLM-NN's edge is evolutionary context this repository refuses to import as
weights. Residue *identity* is legal and already under-used (four of 645 wires).
This family builds integer local alphabets and lag orbits that a sequence model
uses implicitly for secondary structure and hydrophobic patterning — without
PSSM, without ESM, without fitting.

Predicted raw lift +0.002 to +0.008 vs more_old; permuted arm ≤ 0 if live.
Falsification: lift ≤ more_old → identity neighbourhood is already in the bank.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from pocket_bench.pdb_io import parse_pdb_atoms
from pocket_bench.paths import ROOT

SCHEMA = "geoaudit.seqgeom_wires.v1"
WIDE = ROOT / "data/cryptobench_apo/_wide_cache_train.npz"
MANIFEST = ROOT / "data/cryptobench_apo/TRAIN_MANIFEST.json"
CACHE = ROOT / "data/cryptobench_apo/_seqgeom_cache_train.npz"
CACHE_PERM = ROOT / "data/cryptobench_apo/_seqgeom_perm_cache_train.npz"
CONTACT_RADIUS = 8.0
AGGREGATIONS = ("own", "contact", "walk2")
PERMUTATION_SEED = 20260802
CENTROID_TOLERANCE = 5e-4
SKIP = frozenset({"HOH", "WAT", "DOD"})
N_JOBS = min(9, os.cpu_count() or 4)

# Coarse alphabets — literature classes, not fitted.
_HP = {
    "ILE": 3, "VAL": 3, "LEU": 3, "PHE": 3, "CYS": 2, "MET": 2, "ALA": 2,
    "GLY": 1, "THR": 1, "SER": 1, "TRP": 3, "TYR": 2, "PRO": 1, "HIS": 0,
    "GLU": 0, "GLN": 0, "ASP": 0, "ASN": 0, "LYS": 0, "ARG": 0,
}
_CHG = {
    "ASP": 0, "GLU": 0, "LYS": 2, "ARG": 2, "HIS": 1,
}
_AROM = frozenset({"PHE", "TYR", "TRP", "HIS"})
_SMALL = frozenset({"GLY", "ALA", "SER"})
LAGS = (1, 2, 3, 4, 5)


def _quant_names() -> tuple[str, ...]:
    names = [
        "aa_index", "hp_class", "charge_class", "is_aromatic", "is_small",
        "is_pro", "is_gly", "is_cys",
        "n_hp3_in_8A", "n_chg0_in_8A", "n_chg2_in_8A", "n_arom_in_8A",
        "n_same_aa_in_8A", "n_same_hp_in_8A", "frac_hp3_in_8A",
        "hp_minus_nbr_mean", "charge_imbalance_8A",
        "burial_proxy_n_nbr", "is_buried_hp", "is_exposed_polar",
    ]
    for lag in LAGS:
        names += [
            f"lag{lag}_same_aa", f"lag{lag}_same_hp", f"lag{lag}_both_arom",
            f"lag{lag}_hp_product", f"lag{lag}_charge_sum",
        ]
    # GF(4)-style pair digit of (hp_class, charge_class) — 4×3=12 states, keep as int
    names += ["gf4_state", "nbr_mode_gf4", "n_gf4_match_8A"]
    return tuple(names)


QUANT_NAMES = _quant_names()
N_Q = len(QUANT_NAMES)


def column_names() -> tuple[str, ...]:
    return tuple(f"{agg}~{q}" for agg in AGGREGATIONS for q in QUANT_NAMES)


def build_chain(ctr: np.ndarray, prop: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    adj = (d <= CONTACT_RADIUS).astype(np.float64)
    two = adj @ adj
    np.fill_diagonal(two, 0.0)
    return np.concatenate([prop, adj @ prop, two @ prop], axis=1)


def _residue_universe(atoms, chain):
    keep = [a for a in atoms
            if a["chain"] == chain and a["element"] != "H"
            and a["resname"] not in SKIP]
    poly = {}
    names = {}
    for a in keep:
        key = (a["resseq"], a["icode"].strip())
        poly.setdefault(key, []).append((a["x"], a["y"], a["z"]))
        names[key] = a["resname"].strip().upper()
    order = sorted(poly)
    universe = sorted({r for r, _ in order})
    first = {}
    for j, (r, _ic) in enumerate(order):
        first.setdefault(r, j)
    take = np.array([first[r] for r in universe], dtype=np.int64)
    merged = {}
    for (r, _ic), pts in poly.items():
        merged.setdefault(r, []).extend(pts)
    ctr = np.array([np.mean(merged[r], axis=0) for r in universe], dtype=np.float64)
    # primary resname per universe resseq (first icode)
    resnames = []
    for r in universe:
        for key in order:
            if key[0] == r:
                resnames.append(names[key])
                break
    return order, ctr, take, resnames, universe


def compute_seqgeom(resnames: list[str], ctr: np.ndarray) -> np.ndarray:
    n = len(resnames)
    out = np.zeros((n, N_Q), dtype=np.float64)
    if n == 0:
        return out
    aa_list = sorted({a for a in _HP})
    aidx = {a: i for i, a in enumerate(aa_list)}
    hp = np.array([_HP.get(r, 1) for r in resnames], dtype=np.float64)
    ch = np.array([_CHG.get(r, 1) for r in resnames], dtype=np.float64)
    arom = np.array([1.0 if r in _AROM else 0.0 for r in resnames])
    small = np.array([1.0 if r in _SMALL else 0.0 for r in resnames])
    codes = np.array([aidx.get(r, -1) for r in resnames], dtype=np.int64)

    d = np.linalg.norm(ctr[:, None, :] - ctr[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nbr = d <= CONTACT_RADIUS

    col = {name: i for i, name in enumerate(QUANT_NAMES)}
    out[:, col["aa_index"]] = codes
    out[:, col["hp_class"]] = hp
    out[:, col["charge_class"]] = ch
    out[:, col["is_aromatic"]] = arom
    out[:, col["is_small"]] = small
    out[:, col["is_pro"]] = np.array([1.0 if r == "PRO" else 0.0 for r in resnames])
    out[:, col["is_gly"]] = np.array([1.0 if r == "GLY" else 0.0 for r in resnames])
    out[:, col["is_cys"]] = np.array([1.0 if r == "CYS" else 0.0 for r in resnames])

    for i in range(n):
        m = nbr[i]
        nn = int(m.sum())
        out[i, col["burial_proxy_n_nbr"]] = nn
        if nn:
            out[i, col["n_hp3_in_8A"]] = float((hp[m] == 3).sum())
            out[i, col["n_chg0_in_8A"]] = float((ch[m] == 0).sum())
            out[i, col["n_chg2_in_8A"]] = float((ch[m] == 2).sum())
            out[i, col["n_arom_in_8A"]] = float(arom[m].sum())
            out[i, col["n_same_aa_in_8A"]] = float((codes[m] == codes[i]).sum()) if codes[i] >= 0 else 0
            out[i, col["n_same_hp_in_8A"]] = float((hp[m] == hp[i]).sum())
            out[i, col["frac_hp3_in_8A"]] = out[i, col["n_hp3_in_8A"]] / nn
            out[i, col["hp_minus_nbr_mean"]] = hp[i] - float(hp[m].mean())
            out[i, col["charge_imbalance_8A"]] = float((ch[m] == 2).sum() - (ch[m] == 0).sum())
        out[i, col["is_buried_hp"]] = float(nn >= 12 and hp[i] == 3)
        out[i, col["is_exposed_polar"]] = float(nn <= 6 and hp[i] == 0)

    # sequence lags on sorted universe order (resseq order)
    for lag in LAGS:
        for i in range(n):
            j = i + lag
            if j >= n:
                continue
            # only count if resseq difference equals lag (contiguous polymer)
            # approximate: index lag in universe list
            sa = float(codes[i] == codes[j] and codes[i] >= 0)
            sh = float(hp[i] == hp[j])
            ba = float(arom[i] * arom[j])
            out[i, col[f"lag{lag}_same_aa"]] = sa
            out[i, col[f"lag{lag}_same_hp"]] = sh
            out[i, col[f"lag{lag}_both_arom"]] = ba
            out[i, col[f"lag{lag}_hp_product"]] = hp[i] * hp[j]
            out[i, col[f"lag{lag}_charge_sum"]] = ch[i] + ch[j]
            # symmetric write to j
            out[j, col[f"lag{lag}_same_aa"]] = max(out[j, col[f"lag{lag}_same_aa"]], sa)
            out[j, col[f"lag{lag}_same_hp"]] = max(out[j, col[f"lag{lag}_same_hp"]], sh)
            out[j, col[f"lag{lag}_both_arom"]] = max(out[j, col[f"lag{lag}_both_arom"]], ba)

    gf4 = (hp.astype(np.int64) * 3 + ch.astype(np.int64)).astype(np.float64)
    out[:, col["gf4_state"]] = gf4
    for i in range(n):
        m = nbr[i]
        if m.sum():
            vals, counts = np.unique(gf4[m], return_counts=True)
            out[i, col["nbr_mode_gf4"]] = float(vals[counts.argmax()])
            out[i, col["n_gf4_match_8A"]] = float((gf4[m] == gf4[i]).sum())
    return out


def _one(args):
    u, n, path, chain, ctr_slice = args
    atoms = parse_pdb_atoms(Path(path).read_text())
    order, ctr, take, resnames_full, universe = _residue_universe(atoms, chain)
    if len(take) != n:
        return ("err", u, f"len {len(take)} != {n}")
    d = float(np.abs(ctr - ctr_slice).max())
    if d > CENTROID_TOLERANCE:
        return ("err", u, f"centroid {d:.2e}")
    # resnames_full is universe-aligned already
    x = compute_seqgeom(resnames_full, ctr)
    return ("ok", u, d, x)


def raw_quantities():
    w = np.load(WIDE, allow_pickle=False)
    units = [str(u) for u in w["units"]]
    n_res, ctr_cached = w["n_res_per"], w["ctr"]
    w.close()
    paths = {f"{e['pdb']}_{e['chain']}": (str(ROOT / e["receptor_path"]), e["chain"])
             for e in json.loads(MANIFEST.read_text())["entries"]}
    jobs, offsets, off = [], {}, 0
    for u, n in zip(units, n_res):
        n = int(n)
        path, chain = paths[u]
        jobs.append((u, n, path, chain, ctr_cached[off:off + n].copy()))
        offsets[u] = off
        off += n
    out = np.zeros((int(n_res.sum()), N_Q), dtype=np.float64)
    t0, done, worst, worst_u = time.perf_counter(), 0, 0.0, ""
    with ProcessPoolExecutor(max_workers=N_JOBS) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for fut in as_completed(futs):
            rec = fut.result()
            if rec[0] == "err":
                raise SystemExit(f"{rec[1]}: {rec[2]}")
            _, u, d, x = rec
            if d > worst:
                worst, worst_u = d, u
            out[offsets[u]:offsets[u] + len(x)] = x
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(units)}  {time.perf_counter()-t0:.0f}s  jobs={N_JOBS}", flush=True)
    print(f"  centroids {worst:.2e} worst {worst_u}; N_Q={N_Q}", flush=True)
    return out, n_res


def _aggregate(prop, n_res):
    w = np.load(WIDE, allow_pickle=False)
    ctr = w["ctr"]
    w.close()
    blocks, off = [], 0
    for n in n_res:
        n = int(n)
        blocks.append(build_chain(ctr[off:off + n], prop[off:off + n]))
        off += n
    return np.concatenate(blocks, axis=0).astype(np.float32)


def build_or_load(force=False, permuted=False):
    names = column_names()
    cache = CACHE_PERM if permuted else CACHE
    if cache.is_file() and not force:
        z = np.load(cache, allow_pickle=False)
        if tuple(str(s) for s in z["names"]) == names:
            print(f"reusing {cache.relative_to(ROOT)} {z['C'].shape}", flush=True)
            return z["C"], names
    prop, n_res = raw_quantities()
    if permuted:
        rng = np.random.default_rng(PERMUTATION_SEED)
        off = 0
        for n in n_res:
            n = int(n)
            prop[off:off + n] = prop[off:off + n][rng.permutation(n)]
            off += n
    C = _aggregate(prop, n_res)
    np.savez_compressed(cache, C=C, names=np.asarray(names))
    print(f"wrote {cache.relative_to(ROOT)} {C.shape}", flush=True)
    return C, names


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--permuted", action="store_true")
    a = ap.parse_args(argv)
    C, names = build_or_load(a.rebuild, a.permuted)
    print(f"{C.shape[1]} columns / {C.shape[0]} residues")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
