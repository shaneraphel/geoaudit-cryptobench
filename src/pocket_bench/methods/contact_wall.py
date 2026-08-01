"""Van der Waals contact-wall and packing-asperity quantities.

Why this family exists
----------------------
Side-chain geometry reads rotamers; void topology reads empty space; neither
reads the contact sheet where a side chain presses against its neighbours — the
surface a ligand would have to displace. A cryptic opening is often that
displacement. Members are different measurements of that sheet, not radii of one
operator (AGENT_MEMORY 2c).

Prediction (training-fold attachment, not yet run)
--------------------------------------------------
Lift +0.001 to +0.004 on 12 cluster-disjoint halvings; 30–50% overlap with
sidechain and void. Control arms: ``more_old`` and a within-family permutation.
If lift ≤ 0 and the permutation matches the intact arm, the family is null.

Distances are compared to fixed Bondi van der Waals sums; counts and
within-chain ranks are exact relative to those cuts. No floating-point model is
fitted.
"""
from __future__ import annotations

import numpy as np

VDW = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "CL": 1.75, "BR": 1.85, "I": 1.98, "SE": 1.90,
}
CONTACT_SLACK = 0.50
SHELLS = (3.5, 4.5, 5.5, 6.5)

COLUMNS = (
    "n_vdw_contacts",
    "n_vdw_contacts_sc_sc",
    "n_vdw_contacts_sc_bb",
    "n_tight_contacts",
    "n_soft_contacts",
    "frac_tight_among_contacts",
    "contact_span_xy",
    "contact_span_z",
    "contact_anisotropy",
    "contact_centroid_offset",
    "n_open_octants",
    "largest_empty_octant_gap",
    "n_heavy_in_3p5",
    "n_heavy_in_4p5",
    "n_heavy_in_5p5",
    "n_heavy_in_6p5",
    "n_carbon_in_4p5",
    "n_polar_in_4p5",
    "carbon_polar_imbalance_4p5",
    "contact_rms_from_plane",
    "contact_plane_offset",
    "n_contacts_above_plane",
    "n_contacts_below_plane",
    "n_contacts_seq_lt_5",
    "n_contacts_seq_ge_5",
    "frac_tertiary_contacts",
    "rank_n_vdw_contacts",
    "rank_contact_anisotropy",
    "rank_n_open_octants",
    "rank_contact_rms_from_plane",
    "is_contact_rich",
    "is_open_faced",
    "is_planar_wall",
    "is_asperity",
    "has_no_sc_sc_contact",
    "has_tight_sc_sc_cluster",
)
N_COLUMNS = len(COLUMNS)
_COL = {n: j for j, n in enumerate(COLUMNS)}


def _element(atom: dict) -> str:
    e = (atom.get("element") or "").strip().upper()
    if not e and atom.get("name"):
        e = "".join(c for c in atom["name"] if c.isalpha())[:2].upper()
    return e if e in VDW else "C"


def _is_backbone(name: str) -> bool:
    return name.strip() in {"N", "CA", "C", "O", "OXT"}


def compute(atoms_by_res: list[list[dict]], resseqs: list[int]) -> np.ndarray:
    """``(n_res, N_COLUMNS)`` for one chain."""
    n = len(atoms_by_res)
    out = np.zeros((n, N_COLUMNS), dtype=np.float64)
    if n == 0:
        return out

    pts, elems, is_bb, owners = [], [], [], []
    for i, atoms in enumerate(atoms_by_res):
        for a in atoms:
            e = _element(a)
            if e == "H":
                continue
            pts.append((float(a["x"]), float(a["y"]), float(a["z"])))
            elems.append(e)
            is_bb.append(_is_backbone(a.get("name") or ""))
            owners.append(i)
    if len(pts) < 2:
        return out

    X = np.asarray(pts, dtype=np.float64)
    owners_a = np.asarray(owners, dtype=np.int32)
    elems_a = np.asarray(elems)
    bb_a = np.asarray(is_bb, dtype=bool)
    seq = np.asarray(resseqs, dtype=np.int32)

    from scipy.spatial import cKDTree
    pairs = cKDTree(X).query_pairs(r=SHELLS[-1] + 0.05, output_type="ndarray")
    if len(pairs) == 0:
        return _finalize(out, n)

    # Per-residue lists of contact vectors and flags.
    vecs: list[list[np.ndarray]] = [[] for _ in range(n)]
    for a, b in pairs:
        oa, ob = int(owners_a[a]), int(owners_a[b])
        if oa == ob:
            continue
        d = float(np.linalg.norm(X[a] - X[b]))
        vsum = VDW[elems_a[a]] + VDW[elems_a[b]]
        tight = d < vsum
        soft = (not tight) and d < vsum + CONTACT_SLACK
        if not (tight or soft) and d >= SHELLS[-1]:
            continue
        for owned, other, sign in ((oa, ob, 1.0), (ob, oa, -1.0)):
            owned_at = a if owned == oa else b
            other_at = b if owned == oa else a
            sc_sc = (not bb_a[owned_at]) and (not bb_a[other_at])
            sc_bb = (not bb_a[owned_at]) and bb_a[other_at]
            seq_d = abs(int(seq[owned]) - int(seq[other]))
            if tight or soft:
                out[owned, _COL["n_vdw_contacts"]] += 1
                out[owned, _COL["n_tight_contacts"]] += int(tight)
                out[owned, _COL["n_soft_contacts"]] += int(soft)
                out[owned, _COL["n_vdw_contacts_sc_sc"]] += int(sc_sc)
                out[owned, _COL["n_vdw_contacts_sc_bb"]] += int(sc_bb)
                out[owned, _COL["n_contacts_seq_lt_5"]] += int(seq_d < 5)
                out[owned, _COL["n_contacts_seq_ge_5"]] += int(seq_d >= 5)
                vecs[owned].append(sign * (X[other_at] - X[owned_at]))
            for shell, name in zip(SHELLS, ("n_heavy_in_3p5", "n_heavy_in_4p5",
                                            "n_heavy_in_5p5", "n_heavy_in_6p5")):
                if d < shell:
                    out[owned, _COL[name]] += 1
            if d < 4.5:
                oe = elems_a[other_at]
                out[owned, _COL["n_carbon_in_4p5"]] += int(oe == "C")
                out[owned, _COL["n_polar_in_4p5"]] += int(oe in ("N", "O", "S"))

    for ridx, Vlist in enumerate(vecs):
        if len(Vlist) < 2:
            if len(Vlist) == 1:
                out[ridx, _COL["contact_centroid_offset"]] = float(
                    np.linalg.norm(Vlist[0]))
                out[ridx, _COL["n_open_octants"]] = 7
            else:
                out[ridx, _COL["n_open_octants"]] = 8
            continue
        V = np.stack(Vlist, axis=0)
        span = V.max(0) - V.min(0)
        out[ridx, _COL["contact_span_xy"]] = float(np.hypot(span[0], span[1]))
        out[ridx, _COL["contact_span_z"]] = float(abs(span[2]))
        out[ridx, _COL["contact_centroid_offset"]] = float(
            np.linalg.norm(V.mean(0)))
        signs = (V >= 0).astype(np.int8)
        codes = signs[:, 0] + 2 * signs[:, 1] + 4 * signs[:, 2]
        out[ridx, _COL["n_open_octants"]] = 8 - len(np.unique(codes))
        # azimuth gaps on XY
        ang = np.sort(np.arctan2(V[:, 1], V[:, 0]))
        gaps = np.diff(ang, append=ang[0] + 2 * np.pi)
        out[ridx, _COL["largest_empty_octant_gap"]] = float(
            gaps.max() * 180 / np.pi)
        c0 = V.mean(0)
        _, _, Vt = np.linalg.svd(V - c0, full_matrices=False)
        normal = Vt[-1]
        offs = (V - c0) @ normal
        out[ridx, _COL["contact_rms_from_plane"]] = float(
            np.sqrt((offs ** 2).mean()) * 1000.0)
        out[ridx, _COL["contact_plane_offset"]] = float(c0 @ normal * 1000.0)
        out[ridx, _COL["n_contacts_above_plane"]] = float((offs > 0).sum())
        out[ridx, _COL["n_contacts_below_plane"]] = float((offs < 0).sum())

    return _finalize(out, n)


def _finalize(out: np.ndarray, n: int) -> np.ndarray:
    eps = 1e-6
    tot = out[:, _COL["n_vdw_contacts"]]
    out[:, _COL["frac_tight_among_contacts"]] = (
        out[:, _COL["n_tight_contacts"]] / np.maximum(tot, 1))
    out[:, _COL["contact_anisotropy"]] = (
        out[:, _COL["contact_span_z"]] /
        (out[:, _COL["contact_span_xy"]] + eps))
    out[:, _COL["carbon_polar_imbalance_4p5"]] = (
        out[:, _COL["n_carbon_in_4p5"]] - out[:, _COL["n_polar_in_4p5"]])
    out[:, _COL["frac_tertiary_contacts"]] = (
        out[:, _COL["n_contacts_seq_ge_5"]] / np.maximum(tot, 1))

    def rank(col: int) -> np.ndarray:
        v = out[:, col]
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = (np.arange(1, n + 1) / max(n, 1))
        return ranks

    out[:, _COL["rank_n_vdw_contacts"]] = rank(_COL["n_vdw_contacts"])
    out[:, _COL["rank_contact_anisotropy"]] = rank(_COL["contact_anisotropy"])
    out[:, _COL["rank_n_open_octants"]] = rank(_COL["n_open_octants"])
    out[:, _COL["rank_contact_rms_from_plane"]] = rank(
        _COL["contact_rms_from_plane"])

    q75 = float(np.quantile(tot, 0.75)) if n else 0.0
    out[:, _COL["is_contact_rich"]] = (tot >= q75).astype(np.float64)
    out[:, _COL["is_open_faced"]] = (
        out[:, _COL["n_open_octants"]] >= 3).astype(np.float64)
    out[:, _COL["is_planar_wall"]] = (
        (out[:, _COL["contact_rms_from_plane"]] < 500) &
        (tot >= 4)).astype(np.float64)
    out[:, _COL["is_asperity"]] = (
        out[:, _COL["contact_rms_from_plane"]] >= 800).astype(np.float64)
    out[:, _COL["has_no_sc_sc_contact"]] = (
        out[:, _COL["n_vdw_contacts_sc_sc"]] == 0).astype(np.float64)
    out[:, _COL["has_tight_sc_sc_cluster"]] = (
        out[:, _COL["n_vdw_contacts_sc_sc"]] >= 3).astype(np.float64)
    return out


def consistency(mat: np.ndarray) -> list[str]:
    problems = []
    if mat.shape[1] != N_COLUMNS:
        problems.append(f"expected {N_COLUMNS} columns, got {mat.shape[1]}")
    if (mat[:, _COL["n_vdw_contacts"]] < 0).any():
        problems.append("negative contact counts")
    if (mat[:, _COL["n_open_octants"]] > 8 + 1e-9).any():
        problems.append("open octants > 8")
    if (mat[:, _COL["n_tight_contacts"]] >
            mat[:, _COL["n_vdw_contacts"]] + 1e-9).any():
        problems.append("tight contacts exceed total contacts")
    return problems
