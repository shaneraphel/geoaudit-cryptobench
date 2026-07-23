"""Auditable ESR1 receptor-structure clustering for split integrity."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from pocket_bench.pdb_io import parse_pdb_atoms


def _ca_coordinates(path: Path) -> dict[tuple[int, str], np.ndarray]:
    atoms = parse_pdb_atoms(path.read_text(errors="ignore"))
    return {
        (atom["resseq"], atom["resname"]): np.asarray(
            [atom["x"], atom["y"], atom["z"]], dtype=float
        )
        for atom in atoms
        if atom["record"] == "ATOM" and atom["name"] == "CA"
    }


def ca_rmsd(path_a: Path, path_b: Path) -> tuple[float | None, int]:
    a = _ca_coordinates(path_a)
    b = _ca_coordinates(path_b)
    common = sorted(set(a) & set(b))
    if len(common) < 50:
        return None, len(common)
    x = np.asarray([a[key] for key in common])
    y = np.asarray([b[key] for key in common])
    x -= x.mean(axis=0)
    y -= y.mean(axis=0)
    u, _, vt = np.linalg.svd(x.T @ y)
    rotation = u @ np.diag([1.0, 1.0, np.linalg.det(u @ vt)]) @ vt
    rmsd = np.sqrt(np.mean(np.sum((x @ rotation - y) ** 2, axis=1)))
    return float(rmsd), len(common)


def publication_id(raw_pdb: Path) -> str | None:
    text = raw_pdb.read_text(errors="ignore")
    match = re.search(r"^JRNL\s+REFN.*?(10\.\S+)\s*$", text, flags=re.MULTILINE)
    if match:
        return match.group(1).rstrip(".")
    match = re.search(r"^JRNL\s+DOI\s+(\S+)", text, flags=re.MULTILINE)
    return match.group(1).rstrip(".") if match else None


def build_structure_cluster_ledger(
    entries: list[dict[str, Any]],
    *,
    root: Path,
    rmsd_threshold_a: float = 1.5,
) -> dict[str, Any]:
    rows = [entry for entry in entries if entry.get("receptor_pdb")]
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    publications = {
        row["pdb_id"]: publication_id(root / row["raw_pdb"])
        for row in rows
        if row.get("raw_pdb")
    }
    pairs: list[dict[str, Any]] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            a, b = rows[left], rows[right]
            rmsd, common = ca_rmsd(
                root / a["receptor_pdb"], root / b["receptor_pdb"]
            )
            same_publication = bool(
                publications.get(a["pdb_id"])
                and publications.get(a["pdb_id"]) == publications.get(b["pdb_id"])
            )
            structurally_related = rmsd is not None and rmsd <= rmsd_threshold_a
            if same_publication or structurally_related:
                union(left, right)
            pairs.append(
                {
                    "pdb_a": a["pdb_id"],
                    "pdb_b": b["pdb_id"],
                    "split_a": a["split"],
                    "split_b": b["split"],
                    "common_ca": common,
                    "ca_rmsd_angstrom": None if rmsd is None else round(rmsd, 6),
                    "same_publication": same_publication,
                    "related": same_publication or structurally_related,
                }
            )

    roots = sorted({find(index) for index in range(len(rows))})
    cluster_name = {root_index: f"esr1-struct-{rank:03d}" for rank, root_index in enumerate(roots, 1)}
    assignments = []
    for index, row in enumerate(rows):
        assignments.append(
            {
                "pdb_id": row["pdb_id"],
                "genotype": row["genotype"],
                "split": row["split"],
                "cluster_id": cluster_name[find(index)],
                "publication_id": publications.get(row["pdb_id"]),
            }
        )

    split_by_cluster: dict[str, set[str]] = {}
    for row in assignments:
        split_by_cluster.setdefault(row["cluster_id"], set()).add(row["split"])
    overlapping = sorted(
        cluster for cluster, splits in split_by_cluster.items() if len(splits) > 1
    )
    cross_split_pairs = [
        pair
        for pair in pairs
        if pair["related"] and pair["split_a"] != pair["split_b"]
    ]
    return {
        "schema": "foliation.pocket_bench.structure_clusters.v1",
        "clinical_grade": False,
        "method": {
            "receptor_atoms": "matched C-alpha atoms by residue number and name",
            "alignment": "Kabsch",
            "rmsd_threshold_angstrom": rmsd_threshold_a,
            "additional_grouping": "shared PDB primary-citation DOI",
            "linkage": "connected_components",
        },
        "assignments": assignments,
        "pairwise": pairs,
        "cross_split_related_pairs": cross_split_pairs,
        "overlapping_cluster_ids": overlapping,
        "split_integrity_passed": not overlapping,
        "recommended_evidence_level": (
            "locked_holdout" if not overlapping else "retrospective_pilot_only"
        ),
        "truth_boundary": (
            "Unique PDB IDs alone do not establish independent samples. "
            "Clusters, not radius trials or individual depositions, are the analysis unit."
        ),
    }
