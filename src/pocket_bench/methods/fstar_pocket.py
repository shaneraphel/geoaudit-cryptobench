"""F*-breathing pocket detector for apo / cryptic sites (receptor-only).

Rigid geometric detection localizes a pocket only if the cavity is already open.
In an apo structure a cryptic pocket is closed, so rigid buriedness misses it.
The F* breathing oracle replaces stochastic MD with a fixed, predetermined set of
discrete conformal (radial dilation) modes: a candidate site is promoted if a
SMALL local breathing opens a LARGE still-enclosed cavity there ("openability
gain"). No MD, no iteration, no RNG — one pass over a fixed mode set.

    openability(c) = max_gamma [ free_enclosed_volume(dilate_gamma(coords; c)) near c ]
                     - free_enclosed_volume(coords) near c
    score(c)       = rigid_buriedness(c) * (1 + openability(c))

This reuses the deterministic buriedness + free-grid helpers of
``geometric_foundation`` and the fixed dilation scalars of the F* oracle.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from pocket_bench.methods import prediction
from pocket_bench.methods.geometric_foundation import (
    _buriedness,
    _fibonacci_directions,
    _free_grid,
    _receptor_coords,
)
from pocket_bench.paths import STATUS_CRASH, STATUS_EMPTY, STATUS_OK
from pocket_bench.pdb_io import parse_pdb_atoms, residues_near_center

BREATHING_SCALARS: tuple[float, ...] = (1.1, 1.25)  # fixed discrete conformal modes


def _local_free_enclosed(c, coords, *, r_local, step, atom_r, enclose_cut, enclose_min):
    """# of grid points within r_local of c that are free AND still enclosed."""
    axes = [np.arange(-r_local, r_local + step, step) for _ in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    off = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    off = off[(off**2).sum(1) <= r_local * r_local]
    pts = c + off
    near = coords[((coords - c) ** 2).sum(1) <= (enclose_cut + r_local) ** 2]
    if len(near) == 0:
        return 0
    d2 = ((pts[:, None, :] - near[None, :, :]) ** 2).sum(-1)
    dmin = np.sqrt(d2.min(1))
    n_within = (d2 <= enclose_cut * enclose_cut).sum(1)
    free_enclosed = (dmin > atom_r) & (n_within >= enclose_min)
    return int(free_enclosed.sum())


def _dilate_local(coords, c, gamma, shell_radius):
    """Radially push atoms within shell_radius of c outward by (gamma-1)."""
    rel = coords - c
    d = np.sqrt((rel * rel).sum(1))
    m = d <= shell_radius
    out = coords.copy()
    out[m] = c + gamma * rel[m]
    return out


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
    cand_burial_min: float = 0.45,
    n_candidates: int = 60,
    shell_radius: float = 8.0,
    r_local: float = 6.0,
    enclose_cut: float = 10.0,
    enclose_min: int = 14,
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
                method="fstar_pocket", pdb_id=pdb_id, status=STATUS_EMPTY,
                runtime_s=time.perf_counter() - t0, error="no_free_points",
            )
        bur = _buriedness(pts, coords, dirs, cutoff, perp)
        order = np.argsort(-bur)
        cand_idx = [i for i in order if bur[i] >= cand_burial_min][:n_candidates]
        if not cand_idx:
            cand_idx = list(order[: min(n_candidates, len(order))])
        scored: list[tuple[np.ndarray, float]] = []
        for i in cand_idx:
            c = pts[i]
            v0 = _local_free_enclosed(
                c, coords, r_local=r_local, step=grid_step, atom_r=atom_r,
                enclose_cut=enclose_cut, enclose_min=enclose_min,
            )
            vmax = v0
            for g in BREATHING_SCALARS:
                dcoords = _dilate_local(coords, c, g, shell_radius)
                vg = _local_free_enclosed(
                    c, dcoords, r_local=r_local, step=grid_step, atom_r=atom_r,
                    enclose_cut=enclose_cut, enclose_min=enclose_min,
                )
                if vg > vmax:
                    vmax = vg
            openability = max(0, vmax - v0)
            score = float(bur[i]) * (1.0 + openability)
            scored.append((c, score))
        scored.sort(key=lambda t: -t[1])
        kept: list[tuple[np.ndarray, float]] = []
        for c, s in scored:
            if all(np.linalg.norm(c - q) >= sep for q, _ in kept):
                kept.append((c, s))
            if len(kept) >= top_k:
                break
        pockets = []
        for rank, (center, score) in enumerate(kept, start=1):
            xyz = [float(center[0]), float(center[1]), float(center[2])]
            pockets.append(
                {"rank": rank, "center_xyz": xyz, "score": score,
                 "residues": residues_near_center(atoms, xyz, cutoff_a=6.0)}
            )
        return prediction(
            method="fstar_pocket", pdb_id=pdb_id, status=STATUS_OK, pockets=pockets,
            runtime_s=time.perf_counter() - t0,
            extra={"n_candidates": len(cand_idx), "breathing_scalars": list(BREATHING_SCALARS)},
        )
    except AssertionError as exc:
        return prediction(
            method="fstar_pocket", pdb_id=pdb_id, status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0, error=f"ligand_leak_guard:{exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return prediction(
            method="fstar_pocket", pdb_id=pdb_id, status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0, error=str(exc)[-400:],
        )
