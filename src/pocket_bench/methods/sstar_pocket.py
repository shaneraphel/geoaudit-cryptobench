"""S*-shear pocket detector for apo / cryptic sites (receptor-only).

Rigid geometric detection localizes a pocket only if the cavity is already open;
the isotropic F* breathing ablation fails because a uniform radial dilation is the
wrong deformation subgroup (it inflates shallow surface crevices instead of
tearing open a buried void). The anisotropic shear oracle replaces isotropic
dilation with the receptor's own low, non-rigid ANM shear modes: a candidate site
is promoted if a SMALL anisotropic shear opens a LARGE still-enclosed cavity there
("openability gain") that the rigid apo wall does not admit.

    openability(c) = max_m [ free_enclosed_volume(coords + d_m) near c ]
                     - free_enclosed_volume(coords) near c
    score(c)       = rigid_buriedness(c) * (1 + openability(c))
    d_m            = sum_k s_k c_k u_k ,   s_k in {-1, 0, +1}

Displacement amplitudes c_k = dx * sqrt(lambda_1 / lambda_k) are fixed by the
spectrum (no tuned scalar). One pass over a fixed mode set, no MD, no RNG.
Reuses the deterministic buriedness / free-grid helpers of
``geometric_foundation`` and the shear modes of ``anisotropic_shear_oracle``.
"""
from __future__ import annotations

import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from pocket_bench.methods import prediction
from pocket_bench.methods.anisotropic_shear_oracle import (
    K_MODES,
    SHEAR_SIGNS,
    dynamic_shear_amplitudes,
    low_shear_modes,
)
from pocket_bench.methods.firewall import ligand_leak_guard
from pocket_bench.methods.fstar_pocket import _local_free_enclosed
from pocket_bench.methods.geometric_foundation import (
    _buriedness,
    _fibonacci_directions,
    _free_grid,
    _receptor_coords,
)
from pocket_bench.paths import STATUS_CRASH, STATUS_EMPTY, STATUS_OK
from pocket_bench.pdb_io import parse_pdb_atoms, residues_near_center


def _shear_deformations(coords: np.ndarray, *, k_modes: int, grid_resolution: float):
    """The set of anisotropic-shear-deformed coordinate fields (one per sign combo).

    Returns a list of (N,3) arrays, always including the all-zero (rigid) field so
    the deformed set is a superset of the apo wall.
    """
    modes, lambdas = low_shear_modes(coords, k=k_modes)
    c_k = dynamic_shear_amplitudes(lambdas, grid_resolution)     # (k,)
    fields: list[np.ndarray] = []
    for signs in product(SHEAR_SIGNS, repeat=k_modes):
        a = np.asarray(signs, dtype=np.float64) * c_k            # (k,)
        disp = np.tensordot(a, modes, axes=(0, 0))               # (N,3)
        fields.append(coords + disp)
    return fields, c_k, lambdas


@ligand_leak_guard("sstar_pocket")
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
    cand_burial_min: float = 0.40,
    n_candidates: int = 60,
    k_modes: int = K_MODES,
    shear_grid_resolution: float = 1.5,
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
                method="sstar_pocket", pdb_id=pdb_id, status=STATUS_EMPTY,
                runtime_s=time.perf_counter() - t0, error="no_free_points",
            )
        bur = _buriedness(pts, coords, dirs, cutoff, perp)
        order = np.argsort(-bur)
        cand_idx = [i for i in order if bur[i] >= cand_burial_min][:n_candidates]
        if not cand_idx:
            cand_idx = list(order[: min(n_candidates, len(order))])

        # Fixed set of anisotropic shear deformations (superset of the rigid wall).
        fields, c_k, lambdas = _shear_deformations(
            coords, k_modes=k_modes, grid_resolution=shear_grid_resolution
        )

        scored: list[tuple[np.ndarray, float, float]] = []
        for i in cand_idx:
            c = pts[i]
            v0 = _local_free_enclosed(
                c, coords, r_local=r_local, step=grid_step, atom_r=atom_r,
                enclose_cut=enclose_cut, enclose_min=enclose_min,
            )
            vmax = v0
            for dcoords in fields:
                vg = _local_free_enclosed(
                    c, dcoords, r_local=r_local, step=grid_step, atom_r=atom_r,
                    enclose_cut=enclose_cut, enclose_min=enclose_min,
                )
                if vg > vmax:
                    vmax = vg
            openability = max(0, vmax - v0)
            score = float(bur[i]) * (1.0 + openability)
            scored.append((c, score, float(openability)))
        scored.sort(key=lambda t: -t[1])
        kept: list[tuple[np.ndarray, float]] = []
        for c, s, _o in scored:
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
            method="sstar_pocket", pdb_id=pdb_id, status=STATUS_OK, pockets=pockets,
            runtime_s=time.perf_counter() - t0,
            extra={
                "n_candidates": len(cand_idx),
                "k_modes": int(k_modes),
                "shear_amplitudes_c_k": [float(x) for x in c_k],
                "eigenvalues": [float(x) for x in lambdas],
                "n_deformations": len(fields),
            },
        )
    except AssertionError as exc:
        return prediction(
            method="sstar_pocket", pdb_id=pdb_id, status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0, error=f"ligand_leak_guard:{exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return prediction(
            method="sstar_pocket", pdb_id=pdb_id, status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0, error=str(exc)[-400:],
        )
