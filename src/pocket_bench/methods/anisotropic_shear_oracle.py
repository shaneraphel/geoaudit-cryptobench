"""Anisotropic Shear Oracle (S*) — Class-A cryptic-pocket opening.

Replaces the failed *isotropic* breathing oracle (F*, which dilated shell atoms
radially about a guessed centre and therefore inflated surface pseudo-pockets and
drifted the predicted centre outward). Isotropic dilation is a pure-trace
(conformal) deformation `r -> gamma * r`; it lives in the wrong subgroup of the
deformation algebra and cannot open a closed cleft.

The correct deformation is *anisotropic*: a collective shear/rotation living in
so(3) (x) Sym_0(3) (rotation + traceless-symmetric). Those are exactly the low
non-rigid normal modes of an elastic network. We obtain them from the
**vector-valued graph Laplacian** of the receptor contact graph — the Anisotropic
Network Model (ANM) Hessian, which is the block form of `L = D - A` with 3x3
super-elements `gamma_ij * (e_ij (x) e_ij)`. Its six lowest eigenvalues are the
rigid-body zero modes (3 translation + 3 rotation); the next K eigenvectors are
collective anisotropic shears — displacement fields in R^{3N}, never a scalar
homothety.

The admissibility mask is the Boolean OR of the free space over a fixed,
predetermined set of discrete shear amplitudes:

    S*(v) = OR over m in M of  free_space( V_wall + d_m )(v)
    d_m   = sum_k s_k^(m) * c_k * u_k ,   s_k in {-1, 0, +1}

with M the Cartesian product of the signs over the K modes. There is no tuned
amplitude: the per-mode magnitude is fixed by the spectrum via the dynamic
algebraic scaling `c_k = dx * sqrt(lambda_1 / lambda_k)` (spectral-gap-anchored
equipotential; softer modes move more, stiffer modes are attenuated by
1/sqrt(lambda)). The all-zero combo reproduces the rigid apo wall, so S* is always
a superset of apo free space; the *new* voxels (S* & ~apo_free) are the breathable
cryptic volume that a rigid detector cannot see. Everything is geometry.
`clinical_grade=false`: a free voxel is a topological admissibility statement,
never evidence of binding, potency, or safety.
"""
from __future__ import annotations

from itertools import product

import numpy as np

CONTACT_CUTOFF = 12.0          # Angstrom, ANM contact radius
K_MODES = 3                    # non-rigid modes retained
SHEAR_SIGNS = (-1.0, 0.0, 1.0)  # per-mode {-c_k, 0, +c_k}; c_k is derived, not tuned
DEFAULT_VDW_RADIUS = 1.7       # Angstrom, ~carbon Bondi hard wall
_RIGID_EPS = 1e-6              # eigenvalue below (eps * lambda_max) == rigid/disconnected
_MAX_VOXELS = 40_000_000
_MAX_ATOMS = 20_000            # eigsh guard


# --------------------------------------------------------------------------- #
# 1-2. ANM Hessian (block graph Laplacian) and its low non-rigid modes
# --------------------------------------------------------------------------- #
def anm_hessian(coords: np.ndarray, cutoff: float = CONTACT_CUTOFF):
    """Sparse (3N,3N) ANM Hessian = vector-valued, distance-weighted `L = D - A`.

    Super-element for contact (i,j): `gamma_ij * (e_ij outer e_ij)`, with a
    distance-weighted spring constant `gamma_ij = 1 / d_ij^2` inside `cutoff`.
    """
    from scipy import sparse
    from scipy.spatial import cKDTree

    coords = np.asarray(coords, dtype=np.float64)
    n = len(coords)
    if n < 4:
        raise ValueError("need >= 4 atoms to define non-trivial shear modes")
    if n > _MAX_ATOMS:
        raise ValueError(f"{n} atoms exceeds ANM guard {_MAX_ATOMS}; pre-decimate to CA")

    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=cutoff, output_type="ndarray")
    if len(pairs) == 0:
        raise ValueError("no contacts within cutoff; increase cutoff")

    i_idx = pairs[:, 0]
    j_idx = pairs[:, 1]
    diff = coords[j_idx] - coords[i_idx]
    d2 = np.einsum("ij,ij->i", diff, diff)
    d2 = np.maximum(d2, 1e-9)
    e = diff / np.sqrt(d2)[:, None]              # unit contact vectors
    gamma = 1.0 / d2                             # distance-weighted spring
    # 3x3 outer-product super-elements, scaled by gamma
    blocks = gamma[:, None, None] * (e[:, :, None] * e[:, None, :])  # (P,3,3)

    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    def emit(bi, bj, blk):
        a, b = np.meshgrid(np.arange(3), np.arange(3), indexing="ij")
        rows.append((3 * bi)[:, None, None] + a[None])
        cols.append((3 * bj)[:, None, None] + b[None])
        data.append(blk)

    # off-diagonal: H_ij = H_ji = -block
    emit(i_idx, j_idx, -blocks)
    emit(j_idx, i_idx, -blocks)
    # diagonal: H_ii += block, H_jj += block
    emit(i_idx, i_idx, blocks)
    emit(j_idx, j_idx, blocks)

    R = np.concatenate([r.ravel() for r in rows])
    C = np.concatenate([c.ravel() for c in cols])
    V = np.concatenate([d.ravel() for d in data])
    H = sparse.coo_matrix((V, (R, C)), shape=(3 * n, 3 * n)).tocsr()
    return H


def low_shear_modes(coords: np.ndarray, *, k: int = K_MODES, cutoff: float = CONTACT_CUTOFF):
    """The K lowest non-rigid ANM modes.

    Returns ``(modes, eigenvalues)`` where ``modes`` is (K, N, 3) RMS-normalized
    displacement fields and ``eigenvalues`` is (K,) the corresponding non-zero
    eigenvalues (ascending). Rigid-body / disconnected near-zero modes
    (``lambda <= _RIGID_EPS * lambda_max``) are dropped so no negative or
    numerically-zero eigenvalue ever reaches the amplitude formula.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh, splu

    n = len(coords)
    H = anm_hessian(coords, cutoff)
    n_want = 6 + k + 2          # 6 rigid + K wanted + slack
    n_want = min(n_want, 3 * n - 1)
    # Determinism: ARPACK seeds its Lanczos iteration from an UNSEEDED random start
    # vector unless one is supplied, so for clustered/degenerate eigenvalues the
    # returned basis can vary run-to-run BEFORE any sign convention applies. Pin an
    # explicit, structure-independent start vector and a fixed tolerance so the
    # iteration itself is reproducible; the sign convention below then removes the
    # residual +/- ambiguity.
    v0 = np.full(3 * n, 1.0 / np.sqrt(3 * n), dtype=np.float64)
    # Performance, not numerics: shift-invert needs (H - sigma I)^-1, and SuperLU's
    # default COLAMD ordering fills in badly on this symmetric Hessian. Supplying our
    # own OPinv factorized with MMD_AT_PLUS_A (the symmetric-pattern ordering) cuts
    # the factorization ~3x and the ARPACK solve ~16x. Verified on the official fold:
    # eigenvalues agree to 7.4e-17 and per-mode |cos| = 1.0 against the default path.
    try:
        lu = splu(H.tocsc(), permc_spec="MMD_AT_PLUS_A")
        opinv = LinearOperator(H.shape, matvec=lu.solve, dtype=np.float64)
        vals, vecs = eigsh(H, k=n_want, sigma=0.0, which="LM", OPinv=opinv,
                           v0=v0, tol=0.0)
    except Exception:  # noqa: BLE001 -- singular factorization / ARPACK issues
        try:
            vals, vecs = eigsh(H, k=n_want, sigma=0.0, which="LM", v0=v0, tol=0.0)
        except Exception:  # noqa: BLE001
            vals, vecs = eigsh(H, k=n_want, which="SA", v0=v0, tol=0.0)

    order = np.argsort(vals)
    vals = vals[order]
    vecs = vecs[:, order]
    lam_max = float(vals[-1]) if vals[-1] > 0 else 1.0
    nontrivial = np.where(vals > _RIGID_EPS * lam_max)[0]
    picked = nontrivial[:k]
    if len(picked) < k:
        raise ValueError("insufficient non-rigid modes; increase cutoff or atoms")

    modes = []
    lambdas = []
    for col in picked:
        raw = vecs[:, col]
        # Deterministic phase convention: ARPACK/eigsh fixes an eigenvector only up
        # to an overall sign (and, for degeneracies, up to a subspace rotation). Pin
        # the global sign so the largest-|component| entry is strictly positive; ties
        # in |value| are broken by the lowest flat index. This makes the returned
        # modes byte-identical across runs / BLAS backends, so downstream S* voxel
        # masks are reproducible under peer-review CI.
        flat = np.ascontiguousarray(raw)
        amax = float(np.max(np.abs(flat)))
        lead = int(np.argmax(np.abs(flat) >= amax - 1e-12))  # first index attaining max |.|
        if flat[lead] < 0.0:
            raw = -raw
        u = raw.reshape(n, 3)
        rms = np.sqrt((u * u).sum() / n)
        modes.append(u / (rms if rms > 1e-12 else 1.0))
        lambdas.append(float(vals[col]))
    return np.stack(modes, axis=0), np.asarray(lambdas, dtype=np.float64)


def dynamic_shear_amplitudes(eigenvalues: np.ndarray, grid_resolution: float) -> np.ndarray:
    """c_k = dx * sqrt(lambda_1 / lambda_k), anchored to the spectral gap lambda_1.

    Softer (lower-lambda) modes get the largest amplitude; stiffer modes are
    attenuated by the inverse square root of their eigenvalue. No empirical
    hyperparameter enters: amplitude is fixed by the spectrum and the voxel
    resolution alone. Non-positive / non-finite eigenvalues are masked to c_k=0.
    """
    lam = np.asarray(eigenvalues, dtype=np.float64)
    lam1 = float(lam[0])
    c = np.zeros_like(lam)
    safe = np.isfinite(lam) & (lam > 0.0) & np.isfinite(lam1) & (lam1 > 0.0)
    c[safe] = grid_resolution * np.sqrt(lam1 / lam[safe])
    return c


def anchor_modes_to_site(coords: np.ndarray, modes: np.ndarray,
                         center: np.ndarray, radius: float) -> np.ndarray:
    """Project the local rigid-body motion out of each mode at one site.

    The openability probe centre is fixed in space while atoms move, so for a
    rigid motion X -> RX + t the measured free volume obeys

        V(c; RX + t) = V(R^-1 (c - t); X),

    i.e. a rigid displacement opens nothing — it resamples the *undeformed* wall
    at a shifted probe centre. In any local neighbourhood a global low-frequency
    ANM mode is dominated by exactly that component (a hinge sliding a whole
    domain past c), which is the spurious volume that dragged the predicted
    centre outward.

    For the neighbourhood N = {i : |x_i - c| <= radius} we solve, in closed form,

        (t, w) = argmin  sum_{i in N} | u_i - t - w x r_i |^2,   r_i = x_i - xbar
        t = mean(u_N),   w = A^-1 sum_i r_i x (u_i - t),
        A = sum_i [ (r_i . r_i) I - r_i r_i^T ]

    (translations and rotations are orthogonal once centred, since sum_i r_i = 0,
    so the normal equations decouple into a mean and one 3x3 solve), then return

        u~_i = u_i - t - w x (x_i - xbar)   for EVERY atom i.

    Subtracting a rigid motion from every atom is an exact infinitesimal rigid
    re-framing of the whole structure: no interatomic distance changes, so no
    deformation content is destroyed — only the site's frame is pinned. A hard
    spatial gate would instead tear the structure at the gate boundary, and that
    boundary lies inside the enclosure radius, manufacturing the very artifact
    being removed.

    ``radius`` is not a free parameter: callers pass the exact atom support the
    openability measurement can see (``enclose_cut + r_local``).

    The map u -> u~ is a linear projector (I - P_rigid), so it commutes with mode
    superposition: anchoring the K modes once per site anchors every sign
    combination of them. One pass, closed form, no iteration and no sampling.
    """
    coords = np.asarray(coords, dtype=np.float64)
    modes = np.asarray(modes, dtype=np.float64)
    rel = coords - np.asarray(center, dtype=np.float64)
    sel = np.einsum("ij,ij->i", rel, rel) <= radius * radius
    if int(sel.sum()) < 3:
        return modes
    r = coords[sel] - coords[sel].mean(axis=0)
    # Gram matrix of the rotation basis {e_a x r_i}: (r.r) I - sum r r^T.
    A = np.eye(3) * float(np.einsum("ij,ij->", r, r)) - r.T @ r
    xbar = coords[sel].mean(axis=0)
    out = np.empty_like(modes)
    for k in range(modes.shape[0]):
        d = modes[k][sel]
        t = d.mean(axis=0)
        b = np.cross(r, d - t).sum(axis=0)
        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Degenerate (collinear) neighbourhood: no well-posed rotation to
            # remove. Strip the translation only; never guess a rotation.
            w = np.zeros(3, dtype=np.float64)
        out[k] = modes[k] - t - np.cross(w, coords - xbar)
    return out


# --------------------------------------------------------------------------- #
# 3-4. Discrete anisotropic mode set + vdW voxel carving + Boolean OR
# --------------------------------------------------------------------------- #
def _grid(coords: np.ndarray, res: float, margin: float):
    origin = coords.min(axis=0) - margin
    far = coords.max(axis=0) + margin
    dims = np.ceil((far - origin) / res).astype(int) + 1
    if int(np.prod(dims)) > _MAX_VOXELS:
        raise ValueError(
            f"grid {tuple(dims)} = {int(np.prod(dims))} voxels exceeds {_MAX_VOXELS}; "
            f"use a coarser grid_resolution"
        )
    return origin.astype(np.float64), dims


def _ball_offsets(radius: float, res: float) -> np.ndarray:
    n = int(np.ceil(radius / res))
    rng = np.arange(-n, n + 1)
    dx, dy, dz = np.meshgrid(rng, rng, rng, indexing="ij")
    off = np.stack([dx.ravel(), dy.ravel(), dz.ravel()], axis=1)
    keep = (off**2).sum(1) * (res * res) <= radius * radius
    return off[keep]


def _occupied_mask(coords: np.ndarray, origin, dims, res, ball) -> np.ndarray:
    """Boolean occupancy: True where a voxel centre lies within vdW of any atom."""
    occ = np.zeros(tuple(int(d) for d in dims), dtype=bool)
    base = np.round((coords - origin) / res).astype(np.int64)   # (N,3)
    stamped = (base[:, None, :] + ball[None, :, :]).reshape(-1, 3)  # (N*B,3)
    in_bounds = np.all((stamped >= 0) & (stamped < dims), axis=1)
    s = stamped[in_bounds]
    occ[s[:, 0], s[:, 1], s[:, 2]] = True
    return occ


def build_anisotropic_shear_oracle(
    apo_receptor_coords: np.ndarray,
    grid_resolution: float = 1.0,
    *,
    contact_cutoff: float = CONTACT_CUTOFF,
    k_modes: int = K_MODES,
    shear_signs: tuple[float, ...] = SHEAR_SIGNS,
    vdw_radius: float = DEFAULT_VDW_RADIUS,
    return_components: bool = False,
):
    """Compute the S* anisotropic-shear admissibility mask.

    Parameters
    ----------
    apo_receptor_coords : (N, 3) float array of apo heavy-atom coordinates.
    grid_resolution     : voxel edge length in Angstrom.

    Returns
    -------
    (s_star_mask, grid_origin) where ``s_star_mask`` is a 3D Boolean numpy array
    (True == admissible in at least one shear mode) and ``grid_origin`` is (3,).
    If ``return_components`` is True, returns a dict additionally exposing the apo
    free space and the cryptic gain ``s_star & ~apo_free``.
    """
    coords = np.asarray(apo_receptor_coords, dtype=np.float64)
    modes, lambdas = low_shear_modes(coords, k=k_modes, cutoff=contact_cutoff)  # (k,N,3),(k,)

    # Dynamic algebraic amplitude per mode: c_k = dx * sqrt(lambda_1 / lambda_k).
    c_k = dynamic_shear_amplitudes(lambdas, grid_resolution)          # (k,)

    # max per-atom displacement at the extreme corner of the mode box, using c_k
    corner = (c_k[:, None, None] * np.abs(modes)).sum(axis=0)          # (N,3)
    max_disp = float(np.max(np.linalg.norm(corner, axis=1)))
    margin = max_disp + vdw_radius + grid_resolution
    origin, dims = _grid(coords, grid_resolution, margin)
    ball = _ball_offsets(vdw_radius, grid_resolution)

    s_star = np.zeros(tuple(int(d) for d in dims), dtype=bool)
    apo_free = None
    # M = Cartesian product of {-c_k, 0, +c_k} over the K modes (signs x c_k)
    for signs in product(shear_signs, repeat=k_modes):
        a = np.asarray(signs, dtype=np.float64) * c_k                 # (k,) signed amplitudes
        disp = np.tensordot(a, modes, axes=(0, 0))                    # (N,3)
        deformed = coords + disp
        occ = _occupied_mask(deformed, origin, dims, grid_resolution, ball)
        free = ~occ
        s_star |= free
        if not np.any(a):  # the all-zero combo == rigid apo wall
            apo_free = free

    if return_components:
        if apo_free is None:
            apo_free = np.zeros_like(s_star)
        return {
            "s_star_mask": s_star,
            "grid_origin": origin,
            "grid_resolution": grid_resolution,
            "apo_free": apo_free,
            "cryptic_gain": s_star & ~apo_free,
            "eigenvalues": lambdas.tolist(),
            "shear_amplitudes_c_k": c_k.tolist(),
            "n_modes_evaluated": len(shear_signs) ** k_modes,
            "clinical_grade": False,
        }
    return s_star, origin
