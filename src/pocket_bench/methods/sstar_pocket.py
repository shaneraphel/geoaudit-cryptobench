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
    anchor_modes_to_site,
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


def _sign_amplitudes(c_k: np.ndarray, k_modes: int) -> np.ndarray:
    """The fixed sign lattice {-1,0,+1}^k scaled by the spectral amplitudes."""
    return np.asarray(
        [np.asarray(s, dtype=np.float64) * c_k
         for s in product(SHEAR_SIGNS, repeat=k_modes)],
        dtype=np.float64,
    )                                                            # (M,k)


def _site_fields(coords: np.ndarray, modes: np.ndarray, amps: np.ndarray,
                 center: np.ndarray, anchor_radius: float,
                 support_radius: float | None = None):
    """Deformed coordinate fields at one site, with the local frame pinned.

    The modes are anchored at ``center`` first (local rigid-body motion projected
    out), then superposed. Anchoring is linear, so projecting the k modes once
    anchors all M sign combinations — the superposition is not re-projected.

    Always contains the all-zero combination, so the deformed set is a superset of
    the rigid apo wall: the shear can only add admissible states, never remove one.

    When ``support_radius`` is given, only atoms that can influence a measurement
    of that radius at ``center`` are carried. The retained set is widened by the
    exact triangle-inequality bound on the displacement,
    ``D = sum_k c_k max_i |u~_k,i|`` over the anchored neighbourhood, so no atom
    that could enter the support after deformation is dropped. If that bound
    exceeds the anchored neighbourhood the restriction is abandoned and every atom
    is carried — correctness is never traded for the speedup.
    """
    u = anchor_modes_to_site(coords, modes, center, anchor_radius)   # (k,N,3)
    if support_radius is not None:
        rel = coords - center
        d2 = np.einsum("ij,ij->i", rel, rel)
        inner = d2 <= anchor_radius * anchor_radius
        if inner.any():
            # |sum_k a_k u~_k,i| <= sum_k max|a_k| max_i|u~_k,i| = D
            per_mode = np.sqrt(
                np.einsum("kij,kij->ki", u[:, inner], u[:, inner]).max(axis=1)
            )
            bound = float((np.abs(amps).max(axis=0) * per_mode).sum())
            r_pre = support_radius + bound
            if r_pre <= anchor_radius:
                sel = d2 <= r_pre * r_pre
                return coords[sel][None, :, :] + np.tensordot(
                    amps, u[:, sel], axes=(1, 0)
                )
    return coords[None, :, :] + np.tensordot(amps, u, axes=(1, 0))   # (M,N,3)


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
    burial_min: float = 0.55,
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
        modes, lambdas = low_shear_modes(coords, k=k_modes)
        c_k = dynamic_shear_amplitudes(lambdas, shear_grid_resolution)
        amps = _sign_amplitudes(c_k, k_modes)
        # Radii are forced by the functional's own support, not tuned: the extent
        # term looks vol_r around the site, each buriedness ray reaches cutoff, so
        # the atoms that can influence F(c) lie within cutoff + vol_r.
        vol_r = sep + 2.0
        anchor_radius = cutoff + vol_r

        # Pointwise-monotonic scoring lock.
        #
        #   score_S(c) = max over m in M u {0} of F(c; X + d~_m),
        #   F(c; X)    = bur(c; X) * log1p(#{p : |p - c| <= vol_r, bur(p; X) >= b0})
        #
        # F is the rigid detector's OWN functional (buriedness weighted by local
        # cavity extent), evaluated in the deformed frame. Using bur alone instead
        # would drop the extent term and saturate: every top candidate reaches
        # bur = 1.0 under some deformation, collapsing the ranking to ties.
        #
        # The previous rule, bur(c) * (1 + openability), multiplied two
        # incommensurable quantities: bur is a bounded fraction whose spread the
        # cand_burial_min pre-filter had already crushed to ~1.1x, while
        # openability is an unbounded voxel COUNT spanning 6-13x. The count
        # therefore dictated the ranking (corr(score, open) = 0.99, corr(score,
        # bur) as low as -0.62), which is the outward drag the anchor alone could
        # not remove. Taking the max of the SAME functional over the deformed
        # frames keeps the score in [0,1] on the rigid scale, and because the sign
        # lattice contains the zero combination it is pointwise >= the rigid
        # score: the deformation may reveal admissible states, never obscure one.
        scored: list[tuple[np.ndarray, float, float]] = []
        for i in cand_idx:
            c = pts[i]
            rel = pts - c
            loc = np.where(np.einsum("ij,ij->i", rel, rel) <= vol_r * vol_r)[0]
            here = int(np.searchsorted(loc, i))
            P = pts[loc]
            # m = 0 reproduces the rigid functional exactly, so the max below is
            # pointwise >= the rigid score by construction.
            rigid_F = float(bur[i]) * float(
                np.log1p(float((bur[loc] >= burial_min).sum()))
            )
            best = rigid_F
            for dcoords in _site_fields(coords, modes, amps, c, anchor_radius,
                                        support_radius=cutoff + vol_r):
                bd = _buriedness(P, dcoords, dirs, cutoff, perp)
                f = float(bd[here]) * float(
                    np.log1p(float((bd >= burial_min).sum()))
                )
                if f > best:
                    best = f
            scored.append((c, best, best - rigid_F))
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
                "n_deformations": int(len(amps)),
                "site_rigid_frame_anchored": True,
                "anchor_radius_a": float(anchor_radius),
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
