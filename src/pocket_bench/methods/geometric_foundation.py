"""Deterministic geometric-foundation pocket detector (receptor-only).

Pure geometry, no ligand input, no learning, no RNG. A free-space point is
"pocket-like" if the protein encloses it over a large solid angle (buriedness by
ray casting) while the point itself stays outside every atom's van der Waals
volume. This localizes concave binding cavities, unlike a raw neighbour count
which peaks in the fully buried protein core.

Buriedness(p) = (# of fixed unit directions d for which some heavy atom lies
within a thin cylinder along the ray p + t d, 0 < t <= cutoff) / |directions|.

The top-ranked, mutually separated high-buriedness points are returned as
pockets. Everything is a fixed vectorized tensor computation.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from pocket_bench.methods import prediction
from pocket_bench.methods.firewall import ligand_leak_guard
from pocket_bench.paths import STATUS_CRASH, STATUS_EMPTY, STATUS_OK
from pocket_bench.pdb_io import (
    assert_no_hetatm,
    parse_pdb_atoms,
    residues_near_center,
)


def _fibonacci_directions(n: int = 30) -> np.ndarray:
    """n approximately-uniform unit directions (deterministic golden-spiral)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    theta = np.pi * (1.0 + 5.0**0.5) * i
    return np.stack(
        [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)],
        axis=1,
    )


def _receptor_coords(pdb_path: Path) -> np.ndarray:
    assert_no_hetatm(pdb_path)
    atoms = parse_pdb_atoms(pdb_path.read_text(errors="ignore"))
    coords = [
        [a["x"], a["y"], a["z"]]
        for a in atoms
        if a["record"] == "ATOM" and a["element"] != "H"
    ]
    if len(coords) < 50:
        raise ValueError("too few receptor atoms")
    return np.asarray(coords, dtype=float)


def _free_grid(coords, step, atom_r, max_pts):
    mins, maxs = coords.min(0) - 3.0, coords.max(0) + 3.0
    axes = [np.arange(mins[i], maxs[i] + step, step) for i in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    # keep only free points (outside vdW) that are not far from the protein
    # (a shell around the surface), via nearest-atom distance in chunks.
    keep = np.zeros(len(pts), dtype=bool)
    near = np.zeros(len(pts), dtype=bool)
    chunk = 4096
    for s in range(0, len(pts), chunk):
        blk = pts[s : s + chunk]
        d2 = ((blk[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
        dmin = np.sqrt(d2.min(1))
        keep[s : s + chunk] = dmin > atom_r
        near[s : s + chunk] = dmin < 6.0
    sel = pts[keep & near]
    if len(sel) > max_pts:  # deterministic stride subsample
        sel = sel[:: int(np.ceil(len(sel) / max_pts))]
    return sel


def _buriedness(pts, coords, dirs, cutoff, perp):
    """Fraction of directions blocked by an atom within the ray cylinder."""
    out = np.zeros(len(pts))
    perp2 = perp * perp
    for k, p in enumerate(pts):
        rel = coords - p                       # (A,3)
        d2 = (rel * rel).sum(1)
        m = d2 <= cutoff * cutoff               # only atoms within cutoff sphere
        if not m.any():
            continue
        r = rel[m]
        t = r @ dirs.T                          # (A',D) projection along each dir
        perp_sq = (r * r).sum(1)[:, None] - t * t
        blocked = ((t > 0.0) & (t <= cutoff) & (perp_sq <= perp2)).any(axis=0)
        out[k] = blocked.mean()
    return out


@ligand_leak_guard("geometric_foundation")
def predict(
    receptor_pdb: Path,
    *,
    pdb_id: str,
    grid_step: float = 1.5,
    top_k: int = 5,
    n_dirs: int = 30,
    cutoff: float = 11.0,
    perp: float = 1.8,
    atom_r: float = 2.6,
    burial_min: float = 0.55,
    sep: float = 6.0,
    max_pts: int = 6000,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        path = Path(receptor_pdb)
        coords = _receptor_coords(path)
        atoms = parse_pdb_atoms(path.read_text(errors="ignore"))
        dirs = _fibonacci_directions(n_dirs)
        pts = _free_grid(coords, grid_step, atom_r, max_pts)
        if len(pts) == 0:
            return prediction(
                method="geometric_foundation",
                pdb_id=pdb_id,
                status=STATUS_EMPTY,
                runtime_s=time.perf_counter() - t0,
                error="no_free_points",
            )
        bur = _buriedness(pts, coords, dirs, cutoff, perp)
        # Cavity-volume ranking: the binding pocket is the LARGEST enclosed
        # cavity, not the single most-buried crevice. Score each buried point by
        # buriedness weighted by the local count of other buried points (a proxy
        # for cavity volume), so a big enclosed cluster outranks a tiny deep pit.
        hi = np.where(bur >= burial_min)[0]
        if len(hi) == 0:
            hi = np.argsort(-bur)[: min(50, len(bur))]
        hpts, hbur = pts[hi], bur[hi]
        hd2 = ((hpts[:, None, :] - hpts[None, :, :]) ** 2).sum(-1)
        vol = (hd2 <= (sep + 2.0) ** 2).sum(1).astype(float)  # neighbours ~ volume
        score = hbur * np.log1p(vol)
        order = np.argsort(-score)
        kept: list[tuple[np.ndarray, float]] = []
        for j in order:
            p = hpts[j]
            if all(np.linalg.norm(p - q) >= sep for q, _ in kept):
                kept.append((p, float(score[j])))
            if len(kept) >= top_k:
                break
        if not kept:
            idx = int(np.argmax(bur))
            kept = [(pts[idx], float(bur[idx]))]
        pockets = []
        for rank, (center, score) in enumerate(kept, start=1):
            xyz = [float(center[0]), float(center[1]), float(center[2])]
            pockets.append(
                {
                    "rank": rank,
                    "center_xyz": xyz,
                    "score": score,
                    "residues": residues_near_center(atoms, xyz, cutoff_a=6.0),
                }
            )
        return prediction(
            method="geometric_foundation",
            pdb_id=pdb_id,
            status=STATUS_OK,
            pockets=pockets,
            runtime_s=time.perf_counter() - t0,
            extra={"n_free_points": int(len(pts)), "n_dirs": n_dirs},
        )
    except AssertionError as exc:
        return prediction(
            method="geometric_foundation",
            pdb_id=pdb_id,
            status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0,
            error=f"ligand_leak_guard:{exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return prediction(
            method="geometric_foundation",
            pdb_id=pdb_id,
            status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0,
            error=str(exc)[-400:],
        )
