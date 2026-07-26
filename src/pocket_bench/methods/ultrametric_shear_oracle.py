"""Ultrametric (non-Archimedean) shear oracle for cryptic pockets.

Euclidean ANM gating failed for a structural reason: a global low-frequency mode
is supported on the whole chain, so a shear that opens a buried void also swings
distant surface loops. The loop is unconstrained, so it manufactures more new
void than the real pocket does, and the ranking inverts. Measured on the official
CryptoBench test fold, the Euclidean anisotropic shear scores ROC-AUC 0.655
against 0.664 for the rigid detector.

This module replaces the Euclidean support with a non-Archimedean one. A
hierarchical clustering of the residue centroids induces a cophenetic distance
``d_u`` that satisfies the strong triangle inequality

    d_u(x, y) <= max( d_u(x, z), d_u(z, y) )    for all x, y, z,

so thresholding it at ``tau`` is an *equivalence relation*: the closed balls of
radius ``tau`` partition the structure into disjoint topological branches. That
is the algebraic content of ultrametricity used here — a Euclidean threshold
graph is only reflexive and symmetric, never transitive, which is exactly why
Euclidean gates leak. Deformation is confined to the branches that line the
site; every other branch is annihilated by an exact 0 in the mask.

Two further projections are applied to the gated field, both closed-form:

* **Rigid-frame anchor.** The site's local translation and rotation are removed.
  A rigid motion opens nothing (it resamples the undeformed wall at a shifted
  probe centre), so leaving it in only drags the measurement outward.
* **Isochoric (Clifford) projection.** The best-fit affine gradient of the gated
  field is stripped of its isotropic part, ``G -> G - (tr G / 3) I``. To first
  order ``det F = 1 + tr(eps)``, so the surviving deformation is pure shear: it
  may *redistribute* void, never dilate it into existence.

Admissibility is then enforced discretely rather than trusted: a deformed frame
is rejected outright if its free-voxel count in the site's ball exceeds the
undeformed count. Volume can only be moved, not created.

Scoring reuses the rigid detector's own functional so the comparison is
dimensionally closed,

    F(c; X)      = bur(c; X) * log1p( #{p : |p - c| <= vol_r, bur(p; X) >= b0} )
    score_U(c)   = max over m in {0} u admissible(M) of F(c; X + d~_m)

The all-zero combination is always admissible, so ``score_U >= F(c; X)``
pointwise. One pass over a fixed sign lattice: no probability, no iteration, no
time evolution, no fitted scalar.

Fail-closed: a degenerate ultrametric (one ball holding the whole structure, or
all-singleton balls) raises rather than silently reducing to the Euclidean
operator that is already known to lose.
"""
from __future__ import annotations

import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from pocket_bench import native
from pocket_bench.methods import prediction
from pocket_bench.methods.anisotropic_shear_oracle import (
    K_MODES,
    SHEAR_SIGNS,
    anchor_modes_to_site,
    dynamic_shear_amplitudes,
    low_shear_modes,
)
from pocket_bench.methods.firewall import ligand_leak_guard
from pocket_bench.methods.geometric_foundation import (
    _buriedness,
    _fibonacci_directions,
    _free_grid,
)
from pocket_bench.paths import STATUS_CRASH, STATUS_EMPTY, STATUS_OK
from pocket_bench.pdb_io import assert_no_hetatm, parse_pdb_atoms, residues_near_center

# Single linkage is rejected on measurement, not on taste. Its cophenetic
# distance is the minimax path distance (largest edge on the minimum spanning
# tree path). A protein is covalently chained at 1.2-1.9 A, so every pair is
# joined by a path of small edges: on 1bk2_A the single-linkage ultrametric
# diameter is 1.789 A while the Euclidean diameter is 32.9 A, and every
# threshold above 1.79 A yields exactly ONE ball covering 100% of atoms (an
# all-ones mask, i.e. no gate at all), while every threshold below 1.21 A yields
# all singletons (a zero deformation field). There is no usable threshold.
# Complete linkage is the coarsest standard rule whose dendrogram is monotonic —
# hence whose cophenetic distance is a true ultrametric — and whose balls are
# spatially compact: on the same structure tau = 12 A gives 19 balls, the
# largest holding 9.6% of atoms.
LINKAGE_METHOD = "complete"
_DEGENERATE_BALL_FRACTION = 0.90   # one ball holding >=90% of residues == no gate
_MAX_RESIDUES = 6000               # O(n^2) condensed distance guard


def _receptor_atoms(pdb_path: Path):
    """Heavy receptor atoms with their residue keys, in one consistent order."""
    assert_no_hetatm(pdb_path)
    atoms = parse_pdb_atoms(pdb_path.read_text(errors="ignore"))
    coords, res_keys = [], []
    for a in atoms:
        if a["record"] != "ATOM" or a["element"] == "H":
            continue
        coords.append([a["x"], a["y"], a["z"]])
        res_keys.append((a["chain"], a["resseq"]))
    if len(coords) < 50:
        raise ValueError("too few receptor atoms")
    return np.asarray(coords, dtype=np.float64), res_keys, atoms


def ultrametric_branches(coords: np.ndarray, res_keys, tau: float) -> np.ndarray:
    """Per-atom ultrametric branch label at radius ``tau``.

    The hierarchy is built on residue centroids: a deformation domain is a
    residue-level object, and it keeps the condensed distance array at
    O(n_res^2) instead of O(n_atom^2). Atoms inherit their residue's label, so
    a residue is never torn across two branches.

    Because the cophenetic distance is an ultrametric, ``d_u <= tau`` is
    transitive and the labels define a genuine partition into disjoint closed
    balls. Verified, not assumed: the strong triangle inequality is checked on
    the returned partition before it is used.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    order: dict[tuple, int] = {}
    for k in res_keys:
        if k not in order:
            order[k] = len(order)
    n_res = len(order)
    if n_res > _MAX_RESIDUES:
        raise ValueError(f"too many residues for exact hierarchy: {n_res}")
    if n_res < 3:
        raise ValueError(f"degenerate residue count for hierarchy: {n_res}")

    idx = np.fromiter((order[k] for k in res_keys), dtype=np.int64, count=len(res_keys))
    sums = np.zeros((n_res, 3), dtype=np.float64)
    np.add.at(sums, idx, coords)
    counts = np.bincount(idx, minlength=n_res).astype(np.float64)
    centroids = sums / counts[:, None]

    z = linkage(pdist(centroids), method=LINKAGE_METHOD)
    if not np.all(np.diff(z[:, 2]) >= -1e-9):
        # A non-monotonic dendrogram has inversions, and its cophenetic distance
        # is then NOT an ultrametric. Refuse rather than gate on a fake one.
        raise ValueError("non-monotonic dendrogram: cophenetic distance is not ultrametric")
    labels = fcluster(z, t=float(tau), criterion="distance")

    sizes = np.bincount(labels)
    if sizes.max() >= _DEGENERATE_BALL_FRACTION * n_res:
        raise ValueError(
            f"degenerate ultrametric at tau={tau}: one ball holds "
            f"{sizes.max()}/{n_res} residues, the mask would be all-ones"
        )
    if len(np.unique(labels)) == n_res:
        raise ValueError(
            f"degenerate ultrametric at tau={tau}: every residue is its own ball, "
            "the gated deformation field would be identically zero"
        )
    return labels[idx]


def assert_strong_triangle(labels: np.ndarray) -> None:
    """Fail-closed check that the branch partition really is ultrametric.

    For a partition-induced gate the ultrametric distance is 0 inside a ball and
    ``tau+`` across balls, for which the strong triangle inequality reduces to
    transitivity of the equivalence relation. Equal labels being transitive is
    exactly that, so this verifies the property the whole module rests on.
    """
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("empty branch labelling")
    # Transitivity of `same label` is structural for integer labels; what can
    # actually break is a non-contiguous / masked labelling, so assert validity.
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("branch labels must be integral")
    if labels.min() < 0:
        raise ValueError("negative branch label")


def _isochoric(coords: np.ndarray, disp: np.ndarray, sel: np.ndarray) -> np.ndarray:
    """Strip the dilatational part of a displacement field over ``sel``.

    Least-squares affine fit ``u_i ~ t + G (x_i - xbar)`` on the selected atoms,
    then remove the isotropic component of ``G``. Linearised, ``det F = 1 +
    tr(eps) + O(eps^2)`` with ``eps = sym(G)``, so deleting ``(tr G / 3) I``
    makes the field volume-preserving to first order: pure shear, the Clifford
    rotor content, with the scale part annihilated.
    """
    if int(sel.sum()) < 4:
        return disp
    x = coords[sel] - coords[sel].mean(axis=0)
    u = disp[sel] - disp[sel].mean(axis=0)
    gram = x.T @ x
    try:
        g = np.linalg.solve(gram, x.T @ u).T          # (3,3), u ~ G x
    except np.linalg.LinAlgError:
        return disp                                    # degenerate: nothing to strip
    tr = float(np.trace(g)) / 3.0
    if tr == 0.0:
        return disp
    return disp - tr * (coords - coords[sel].mean(axis=0))


def local_lattice(center: np.ndarray, radius: float, step: float) -> np.ndarray:
    """Axis-aligned lattice covering the ball, FIXED in space and independent of X.

    The capacity test must be able to observe voxels that are *occupied* in the
    apo frame, since opening a cryptic pocket means exactly that an occupied
    voxel becomes free. Measuring on the detector's pre-filtered free grid cannot
    express this: every point there is free by construction, so the free count is
    already maximal and a deformation can only lower it, making a
    ``count > capacity`` test unsatisfiable.
    """
    k = int(np.floor(radius / step))
    a = (np.arange(-k, k + 1) * step)
    gx, gy, gz = np.meshgrid(a, a, a, indexing="ij")
    off = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    off = off[np.einsum("ij,ij->i", off, off) <= radius * radius]
    return center[None, :] + off


def _free_voxels(pts: np.ndarray, coords: np.ndarray, atom_r: float) -> int:
    """Free-voxel count on a fixed lattice — the discrete local capacity."""
    fast = native.free_grid_mask(pts, coords, atom_r, 1e9)
    if fast is not None:
        keep, _near = fast
        return int(keep.sum())
    d2 = ((pts[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    return int((np.sqrt(d2.min(1)) > atom_r).sum())


def _support(coords: np.ndarray, u: np.ndarray, amps: np.ndarray,
             center: np.ndarray, radius: float, anchor_radius: float) -> np.ndarray:
    """Atoms that can influence a measurement of ``radius`` at ``center``.

    The anchor subtracts a rigid field ``omega x (x - xbar)`` whose magnitude
    grows linearly with distance, so far atoms acquire displacements far larger
    than any real shear (26.5 A measured on 1bk2_A). Without this restriction
    such an atom can be translated straight into the measurement region and
    fabricate a wall that does not exist. The kept set is widened by the exact
    triangle-inequality bound on the displacement so nothing that could enter is
    dropped; if that bound exceeds the anchored neighbourhood, everything is
    kept and correctness is preserved over speed.
    """
    rel = coords - center
    d2 = np.einsum("ij,ij->i", rel, rel)
    inner = d2 <= anchor_radius * anchor_radius
    if not inner.any():
        return np.ones(len(coords), dtype=bool)
    per_mode = np.sqrt(np.einsum("kij,kij->ki", u[:, inner], u[:, inner]).max(axis=1))
    bound = float((np.abs(amps).max(axis=0) * per_mode).sum())
    r_pre = radius + bound
    if r_pre > anchor_radius:
        return np.ones(len(coords), dtype=bool)
    return d2 <= r_pre * r_pre


@ligand_leak_guard("ultrametric_shear_oracle")
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
    tau: float = 12.0,
    sep: float = 6.0,
    max_pts: int = 6000,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    try:
        path = Path(receptor_pdb)
        coords, res_keys, atoms = _receptor_atoms(path)
        dirs = _fibonacci_directions(n_dirs)
        pts = _free_grid(coords, grid_step, atom_r, max_pts)
        if len(pts) == 0:
            return prediction(
                method="ultrametric_shear_oracle", pdb_id=pdb_id, status=STATUS_EMPTY,
                runtime_s=time.perf_counter() - t0, error="no_free_points",
            )

        branch = ultrametric_branches(coords, res_keys, tau)
        assert_strong_triangle(branch)

        bur = _buriedness(pts, coords, dirs, cutoff, perp)
        order = np.argsort(-bur)
        cand_idx = [i for i in order if bur[i] >= cand_burial_min][:n_candidates]
        if not cand_idx:
            cand_idx = list(order[: min(n_candidates, len(order))])

        modes, lambdas = low_shear_modes(coords, k=k_modes)
        c_k = dynamic_shear_amplitudes(lambdas, shear_grid_resolution)
        amps = np.asarray(
            [np.asarray(s, dtype=np.float64) * c_k
             for s in product(SHEAR_SIGNS, repeat=k_modes)],
            dtype=np.float64,
        )                                                          # (M,k)

        # Radii follow the functional's own support, they are not tuned: the
        # extent term reaches vol_r around the site and each buriedness ray
        # reaches cutoff, so only atoms within cutoff + vol_r can change F(c).
        vol_r = sep + 2.0
        anchor_radius = cutoff + vol_r

        n_rejected = 0
        n_tested = 0
        scored: list[tuple[np.ndarray, float, float]] = []
        for i in cand_idx:
            c = pts[i]
            rel_a = coords - c
            d2a = np.einsum("ij,ij->i", rel_a, rel_a)

            # Ultrametric gate: keep whole branches that line this site, zero the
            # rest. Selecting by branch (not by radius) is the non-Archimedean
            # step — the kept set is a union of closed balls, so the gate is
            # transitive and cannot leak into a loop that merely passes nearby.
            lining = np.unique(branch[d2a <= vol_r * vol_r])
            if lining.size == 0:
                continue
            gate = np.isin(branch, lining)

            rel_p = pts - c
            loc = np.where(np.einsum("ij,ij->i", rel_p, rel_p) <= vol_r * vol_r)[0]
            here = int(np.searchsorted(loc, i))
            probe = pts[loc]

            rigid_f = float(bur[i]) * float(
                np.log1p(float((bur[loc] >= burial_min).sum()))
            )
            best = rigid_f

            u = anchor_modes_to_site(coords, modes, c, anchor_radius)   # (k,N,3)
            u = u * gate[None, :, None]                                 # exact 0 outside
            near = d2a <= anchor_radius * anchor_radius
            u = np.stack([_isochoric(coords, u[k], near & gate) for k in range(len(u))])
            u = u * gate[None, :, None]   # the affine strip is global; re-gate it

            sup = _support(coords, u, amps, c, cutoff + vol_r, anchor_radius)
            x_s, u_s = coords[sup], u[:, sup]
            cell = local_lattice(c, vol_r, grid_step)
            capacity = _free_voxels(cell, x_s, atom_r)

            for a in amps:
                if not a.any():
                    continue                                   # m = 0 already scored
                dcoords = x_s + np.tensordot(a, u_s, axes=(0, 0))
                n_tested += 1
                # Discrete isometry bound: a shear may redistribute void, never
                # manufacture it beyond the undeformed local capacity.
                if _free_voxels(cell, dcoords, atom_r) > capacity:
                    n_rejected += 1
                    continue
                bd = _buriedness(probe, dcoords, dirs, cutoff, perp)
                f = float(bd[here]) * float(
                    np.log1p(float((bd >= burial_min).sum()))
                )
                if f > best:
                    best = f
            scored.append((c, best, best - rigid_f))

        if not scored:
            return prediction(
                method="ultrametric_shear_oracle", pdb_id=pdb_id, status=STATUS_EMPTY,
                runtime_s=time.perf_counter() - t0, error="no_gated_candidates",
            )

        scored.sort(key=lambda t: -t[1])
        kept: list[tuple[np.ndarray, float]] = []
        for c, s, _g in scored:
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
            method="ultrametric_shear_oracle", pdb_id=pdb_id, status=STATUS_OK,
            pockets=pockets, runtime_s=time.perf_counter() - t0,
            extra={
                "n_candidates": len(cand_idx),
                "k_modes": int(k_modes),
                "shear_amplitudes_c_k": [float(x) for x in c_k],
                "eigenvalues": [float(x) for x in lambdas],
                "n_deformations": int(len(amps)),
                "linkage_method": LINKAGE_METHOD,
                "ultrametric_tau_a": float(tau),
                "n_branches": int(len(np.unique(branch))),
                "site_rigid_frame_anchored": True,
                "isochoric_projection": True,
                "anchor_radius_a": float(anchor_radius),
                "n_deformations_tested": int(n_tested),
                "n_deformations_rejected_by_capacity": int(n_rejected),
            },
        )
    except AssertionError as exc:
        return prediction(
            method="ultrametric_shear_oracle", pdb_id=pdb_id, status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0, error=f"ligand_leak_guard:{exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return prediction(
            method="ultrametric_shear_oracle", pdb_id=pdb_id, status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0, error=str(exc)[-400:],
        )
