"""F* Breathing Oracle — one-pass combinational apo-state cavity admissibility.

Computes a 3D Boolean admissibility mask

    F*(v) = OR over m in M of  free_space( T_m(V_wall) )(v)

where each T_m is a *fixed, predetermined* rigid/conformal transform of the apo
van der Waals wall. A voxel is admissible iff it is free in AT LEAST ONE mode.

This replaces stochastic molecular dynamics / iterative relaxation with a fixed,
unrolled set of deterministic breathing states, evaluated as a single tensor
reduction. There is no `while` loop, no sampling, and no fixed-point iteration:
the mode set M is a hardcoded low-order set, each mode's occupancy is one
vectorized scatter, and the modes are combined by a Boolean OR reduction — a
flattened combinational logic gate. Identical input always yields identical
output.

Two mode families (Formulation B is default, numpy-only):

  * ``"weyl"``     Localized Weyl conformal expansion: shell atoms are dilated
                   radially outward from the cryptic centroid by a fixed lattice
                   of Weyl scalars Gamma = (1.0, 1.1, 1.25). gamma=1.0 is the
                   rigid apo state, so F* is always a superset of apo free space.
  * ``"spectral"`` Graph-Laplacian (elastic-network) projection: displace along
                   the lowest K non-zero eigenvectors with discrete amplitudes
                   {-c, 0, +c}; the mode set is their Cartesian product.

Everything is geometry. `clinical_grade=false`: a free voxel is a topological
admissibility statement, never evidence of binding, potency, or safety.
"""
from __future__ import annotations

from itertools import product

import numpy as np

WEYL_SCALARS: tuple[float, ...] = (1.0, 1.1, 1.25)
DEFAULT_VDW_RADIUS = 1.7  # Angstrom, ~carbon Bondi hard wall
_MAX_VOXELS = 40_000_000  # guard against accidental memory blow-up


def _grid(coords: np.ndarray, res: float, margin: float):
    """Fixed grid spanning the apo extent plus a margin for the largest mode."""
    origin = coords.min(axis=0) - margin
    far = coords.max(axis=0) + margin
    dims = np.ceil((far - origin) / res).astype(int) + 1
    if int(np.prod(dims)) > _MAX_VOXELS:
        raise ValueError(
            f"grid {tuple(dims)} = {int(np.prod(dims))} voxels exceeds "
            f"{_MAX_VOXELS}; use a coarser grid_resolution"
        )
    return origin.astype(np.float64), dims


def _ball_offsets(radius: float, res: float) -> np.ndarray:
    """Integer voxel offsets whose center lies within ``radius`` (one stencil)."""
    rr = radius / res
    n = int(np.ceil(rr))
    rng = np.arange(-n, n + 1)
    dx, dy, dz = np.meshgrid(rng, rng, rng, indexing="ij")
    off = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
    keep = (off**2).sum(axis=1) <= rr * rr
    return off[keep]


def _occupied(coords: np.ndarray, origin, dims, res, offsets) -> np.ndarray:
    """Boolean occupancy: voxels within the vdW stencil of any atom (vectorized).

    One scatter of (N_atoms x N_stencil) target voxels — no per-voxel loop.
    """
    atom_vox = np.floor((coords - origin) / res).astype(np.int64)          # (N,3)
    tgt = (atom_vox[:, None, :] + offsets[None, :, :]).reshape(-1, 3)       # (N*P,3)
    inb = np.all((tgt >= 0) & (tgt < dims), axis=1)
    tgt = tgt[inb]
    occ = np.zeros(int(np.prod(dims)), dtype=bool)
    if tgt.size:
        flat = np.ravel_multi_index((tgt[:, 0], tgt[:, 1], tgt[:, 2]), tuple(dims))
        occ[flat] = True
    return occ.reshape(tuple(dims))


def _weyl_modes(coords, site_center, scalars, shell_radius):
    """Fixed set of radially-dilated walls (list, not an iterative process)."""
    d = coords - site_center
    r = np.linalg.norm(d, axis=1, keepdims=True)
    in_shell = (r[:, 0] <= shell_radius)
    modes = []
    for g in scalars:
        moved = coords.copy()
        # push only shell (pocket-lining) atoms outward by gamma; interior/exterior fixed
        moved[in_shell] = site_center + g * d[in_shell]
        modes.append(moved)
    return modes


def _spectral_modes(coords, amplitude, n_modes, contact):
    """Fixed elastic-network modes; Cartesian product of {-c,0,+c} amplitudes."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import laplacian
    from scipy.sparse.linalg import eigsh

    n = coords.shape[0]
    # sparse contact graph (single vectorized neighbor pass, no iteration)
    diff = coords[:, None, :] - coords[None, :, :]
    within = (diff**2).sum(-1) <= contact * contact
    np.fill_diagonal(within, False)
    adj = csr_matrix(within.astype(np.float64))
    lap = laplacian(adj).astype(np.float64)
    # lowest non-zero eigenvectors (k+1 to skip the null mode)
    vals, vecs = eigsh(lap, k=min(n_modes + 1, n - 1), which="SM")
    order = np.argsort(vals)
    u = vecs[:, order[1 : n_modes + 1]]                       # (N, K) node scalars
    amps = product((-amplitude, 0.0, amplitude), repeat=n_modes)
    modes = []
    for combo in amps:
        disp = np.zeros_like(coords)
        for k, c in enumerate(combo):
            disp += c * u[:, k][:, None]                       # isotropic node push
        modes.append(coords + disp)
    return modes


def build_f_star_oracle(
    apo_receptor_coords: np.ndarray,
    grid_resolution: float = 1.0,
    *,
    formulation: str = "weyl",
    site_center: np.ndarray | None = None,
    vdw_radius: float = DEFAULT_VDW_RADIUS,
    weyl_scalars: tuple[float, ...] = WEYL_SCALARS,
    shell_radius: float = 12.0,
    spectral_amplitude: float = 2.0,
    spectral_modes: int = 3,
    spectral_contact: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(f_star_mask, grid_origin)``.

    ``f_star_mask`` is a 3D Boolean tensor; ``True`` means the voxel is free in
    at least one predetermined breathing mode. ``grid_origin`` is the (3,) world
    coordinate of voxel ``[0,0,0]``; a voxel index ``(i,j,k)`` maps to
    ``grid_origin + grid_resolution * (i,j,k)``.
    """
    coords = np.asarray(apo_receptor_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("apo_receptor_coords must be (N, 3)")
    if coords.shape[0] < 4:
        raise ValueError("need >= 4 atoms to define a wall")
    center = coords.mean(axis=0) if site_center is None else np.asarray(site_center, float)

    if formulation == "weyl":
        modes = _weyl_modes(coords, center, weyl_scalars, shell_radius)
        # Reserve the grid margin for the full breathing envelope (default scalars
        # included) so the grid is independent of which subset of modes is asked
        # for; this makes F* a monotone superset of the rigid apo free space.
        weyl_ref = max(max(weyl_scalars), max(WEYL_SCALARS))
        max_disp = (weyl_ref - 1.0) * shell_radius
    elif formulation == "spectral":
        modes = _spectral_modes(coords, spectral_amplitude, spectral_modes, spectral_contact)
        max_disp = spectral_amplitude * spectral_modes
    else:
        raise ValueError(f"unknown formulation {formulation!r}; use 'weyl' or 'spectral'")

    margin = vdw_radius + max_disp + grid_resolution
    origin, dims = _grid(coords, grid_resolution, margin)
    offsets = _ball_offsets(vdw_radius, grid_resolution)

    # F* = OR over modes of free_space; accumulate to keep one grid in memory.
    f_star = np.zeros(tuple(dims), dtype=bool)
    for moved in modes:
        f_star |= ~_occupied(moved, origin, dims, grid_resolution, offsets)
    return f_star, origin
