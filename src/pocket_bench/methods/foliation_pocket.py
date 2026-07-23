"""Receptor-only Foliation pocket predictor — NO ligand coordinates as input/anchor.

Method: grid burial / cavity score over receptor heavy atoms only.
Ligand may never appear in the input PDB (asserted by caller).
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from pocket_bench.methods import prediction
from pocket_bench.paths import STATUS_CRASH, STATUS_EMPTY, STATUS_OK
from pocket_bench.pdb_io import assert_no_hetatm, parse_pdb_atoms


def _receptor_coords(pdb_path: Path) -> np.ndarray:
    assert_no_hetatm(pdb_path)
    atoms = parse_pdb_atoms(pdb_path.read_text(errors="ignore"))
    coords = []
    for a in atoms:
        if a["record"] != "ATOM" or a["element"] == "H":
            continue
        coords.append([a["x"], a["y"], a["z"]])
    if len(coords) < 50:
        raise ValueError("too few receptor atoms")
    return np.asarray(coords, dtype=float)


def _cavity_candidates(
    coords: np.ndarray,
    *,
    grid_step: float = 1.5,
    probe: float = 1.4,
    burial_min: int = 8,
    seed: int = 42,
) -> list[tuple[np.ndarray, float]]:
    """Score empty grid points by neighbor burial (Foliation-lite cavity proxy)."""
    rng = np.random.default_rng(seed)
    mins = coords.min(axis=0) - 2.0
    maxs = coords.max(axis=0) + 2.0
    # Deterministic micro-jitter of grid origin from seed (reproducible, ≠ ligand leak)
    origin = mins + (rng.random(3) - 0.5) * (grid_step * 0.25)
    xs = np.arange(origin[0], maxs[0] + grid_step, grid_step)
    ys = np.arange(origin[1], maxs[1] + grid_step, grid_step)
    zs = np.arange(origin[2], maxs[2] + grid_step, grid_step)
    # Subsample grid if huge
    max_pts = 25000
    n_est = len(xs) * len(ys) * len(zs)
    stride = max(1, int(math.ceil(n_est / max_pts) ** (1 / 3)))
    xs, ys, zs = xs[::stride], ys[::stride], zs[::stride]

    atom_r = 1.7  # carbon-ish exclusion
    candidates: list[tuple[np.ndarray, float]] = []
    for x in xs:
        for y in ys:
            for z in zs:
                p = np.array([x, y, z], dtype=float)
                d2 = np.sum((coords - p) ** 2, axis=1)
                if np.any(d2 < (atom_r + probe * 0.5) ** 2):
                    continue  # inside protein
                shell = (d2 > (atom_r + probe) ** 2) & (d2 < (atom_r + probe + 6.0) ** 2)
                burial = int(np.count_nonzero(shell))
                if burial < burial_min:
                    continue
                shell_d = np.sqrt(d2[shell])
                score = burial / (1.0 + float(shell_d.mean()))
                candidates.append((p, float(score)))

    if not candidates:
        center = coords.mean(axis=0)
        d2 = np.sum((coords - center) ** 2, axis=1)
        idx = int(np.argmin(d2))
        return [(coords[idx], 1.0)]

    # Stable tie-break using seed-hash of rounded coords
    def _key(t: tuple[np.ndarray, float]) -> tuple:
        p, s = t
        return (-s, round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3))

    candidates.sort(key=_key)
    kept: list[tuple[np.ndarray, float]] = []
    for p, s in candidates:
        if all(np.linalg.norm(p - q) >= 4.0 for q, _ in kept):
            kept.append((p, s))
        if len(kept) >= 10:
            break
    return kept


def predict(
    receptor_pdb: Path,
    *,
    pdb_id: str,
    grid_step: float = 1.5,
    top_k: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        path = Path(receptor_pdb)
        coords = _receptor_coords(path)
        atoms = parse_pdb_atoms(path.read_text(errors="ignore"))
        cands = _cavity_candidates(coords, grid_step=grid_step, seed=seed)
        if not cands:
            return prediction(
                method="foliation_pocket_ro",
                pdb_id=pdb_id,
                status=STATUS_EMPTY,
                runtime_s=time.perf_counter() - t0,
                error="no_cavity_candidates",
            )
        from pocket_bench.pdb_io import residues_near_center

        pockets = []
        for rank, (center, score) in enumerate(cands[:top_k], start=1):
            xyz = [float(center[0]), float(center[1]), float(center[2])]
            pockets.append(
                {
                    "rank": rank,
                    "center_xyz": xyz,
                    "score": float(score),
                    "residues": residues_near_center(atoms, xyz, cutoff_a=6.0),
                }
            )
        return prediction(
            method="foliation_pocket_ro",
            pdb_id=pdb_id,
            status=STATUS_OK,
            pockets=pockets,
            runtime_s=time.perf_counter() - t0,
            extra={"grid_step": grid_step, "n_candidates": len(cands), "seed": seed},
        )
    except AssertionError as exc:
        return prediction(
            method="foliation_pocket_ro",
            pdb_id=pdb_id,
            status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0,
            error=f"ligand_leak_guard:{exc}",
        )
    except Exception as exc:
        return prediction(
            method="foliation_pocket_ro",
            pdb_id=pdb_id,
            status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0,
            error=str(exc)[-400:],
        )
