"""Pocket detection metrics — primary Top-1 DCA ≤ 4 Å."""
from __future__ import annotations

import math
from typing import Any, Sequence

from pocket_bench.paths import DCA_THRESHOLD_A


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))


def dca(pred_center: Sequence[float], ligand_heavy: Sequence[Sequence[float]]) -> float:
    """Distance to Closest ligand Atom."""
    return min(_dist(pred_center, atom) for atom in ligand_heavy)


def dcc(pred_center: Sequence[float], ligand_centroid: Sequence[float]) -> float:
    """Distance between predicted center and ligand centroid."""
    return _dist(pred_center, ligand_centroid)


def topk_dca_success(
    pockets: list[dict[str, Any]],
    ligand_heavy: Sequence[Sequence[float]],
    *,
    k: int = 1,
    threshold_a: float = DCA_THRESHOLD_A,
) -> dict[str, Any]:
    ranked = sorted(pockets, key=lambda p: int(p.get("rank", 10**9)))[:k]
    if not ranked:
        return {"success": False, "best_dca": None, "k": k, "threshold_a": threshold_a}
    dcas = [dca(p["center_xyz"], ligand_heavy) for p in ranked]
    best = min(dcas)
    return {
        "success": bool(best <= threshold_a),
        "best_dca": best,
        "dcas": dcas,
        "k": k,
        "threshold_a": threshold_a,
    }


def residue_f1(pred_residues: Sequence[str] | None, true_residues: Sequence[str] | None) -> dict[str, Any]:
    if not pred_residues or not true_residues:
        return {"f1": None, "precision": None, "recall": None, "available": False}
    pset, tset = set(pred_residues), set(true_residues)
    tp = len(pset & tset)
    prec = tp / len(pset) if pset else 0.0
    rec = tp / len(tset) if tset else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "available": True, "tp": tp}


def score_prediction(
    prediction: dict[str, Any],
    label: dict[str, Any],
    *,
    threshold_a: float = DCA_THRESHOLD_A,
) -> dict[str, Any]:
    """Score one method output against label. Ignores non-OK status for success flags."""
    status = prediction.get("status", "OK")
    base = {
        "method": prediction.get("method"),
        "pdb_id": prediction.get("pdb_id"),
        "status": status,
        "runtime_s": prediction.get("runtime_s"),
        "primary_metric": "top1_dca_le_4A",
        "clinical_grade": False,
    }
    if status != "OK":
        return {
            **base,
            "eligible_for_primary": False,
            "top1": None,
            "top3": None,
            "dcc_top1": None,
            "residue_f1": {"available": False},
            "reason": status,
        }
    pockets = prediction.get("pockets") or []
    lig = label["ligand_heavy_coords"]
    top1 = topk_dca_success(pockets, lig, k=1, threshold_a=threshold_a)
    top3 = topk_dca_success(pockets, lig, k=3, threshold_a=threshold_a)
    dcc_v = None
    if pockets:
        first = sorted(pockets, key=lambda p: int(p.get("rank", 10**9)))[0]
        dcc_v = dcc(first["center_xyz"], label["ligand_centroid"])
    pred_res = None
    if pockets:
        pred_res = sorted(pockets, key=lambda p: int(p.get("rank", 10**9)))[0].get("residues")
    return {
        **base,
        "eligible_for_primary": True,
        "top1": top1,
        "top3": top3,
        "dcc_top1": dcc_v,
        "residue_f1": residue_f1(pred_res, label.get("binding_residues")),
    }
