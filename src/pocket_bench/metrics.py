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


def residue_f1(pred_residues: Sequence[Any] | None, true_residues: Sequence[Any] | None) -> dict[str, Any]:
    """Set-F1 over residue numbers.

    Both sides are normalized through ``_resseq`` so that a detector emitting
    'A:ALA123'-style ids joins against integer labels; a raw set intersection of
    mixed conventions would silently score 0.
    """
    if not pred_residues or not true_residues:
        return {"f1": None, "precision": None, "recall": None, "available": False}
    pset = {r for r in (_resseq(p) for p in pred_residues) if r is not None}
    tset = {r for r in (_resseq(t) for t in true_residues) if r is not None}
    if not pset or not tset:
        return {"f1": None, "precision": None, "recall": None, "available": False}
    tp = len(pset & tset)
    prec = tp / len(pset) if pset else 0.0
    rec = tp / len(tset) if tset else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"f1": f1, "precision": prec, "recall": rec, "available": True, "tp": tp}


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Tie-aware ROC-AUC (Mann-Whitney U). None if only one class present."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    sum_pos = sum(ranks[i] for i in range(len(scores)) if labels[i] == 1)
    n_pos, n_neg = len(pos), len(neg)
    u = sum_pos - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def average_precision(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """PR-AUC via the step-wise average-precision estimator. None if no positives."""
    n_pos = sum(1 for y in labels if y == 1)
    if n_pos == 0:
        return None
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    tp = 0
    fp = 0
    ap = 0.0
    prev_recall = 0.0
    for idx in order:
        if labels[idx] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp)
        recall = tp / n_pos
        ap += precision * (recall - prev_recall)
        prev_recall = recall
    return ap


def _resseq(rid: Any) -> int | None:
    """Extract an integer residue number from an int or a 'A:ALA123' style id."""
    if isinstance(rid, int):
        return rid
    s = str(rid)
    digits = ""
    for ch in reversed(s):
        if ch.isdigit():
            digits = ch + digits
        elif digits:
            break
    return int(digits) if digits else None


def residue_scores_from_pockets(
    pockets: list[dict[str, Any]], universe: Sequence[Any]
) -> dict[int, float]:
    """Per-residue score = best (rank-weighted) pocket that contains the residue.

    A residue in the rank-1 pocket scores highest; residues in no pocket score 0.
    Keyed by residue number so string and integer label conventions can join.
    """
    scores: dict[int, float] = {rs: 0.0 for rs in (_resseq(u) for u in universe) if rs is not None}
    for p in pockets:
        rank = int(p.get("rank", 10**9))
        w = 1.0 / rank if rank > 0 else 0.0
        for rid in p.get("residues") or []:
            rs = _resseq(rid)
            if rs is not None:
                scores[rs] = max(scores.get(rs, 0.0), w)
    return scores


def native_residue_scores(
    prediction: dict[str, Any] | None, universe: Sequence[Any]
) -> tuple[dict[int, float], set[int]] | None:
    """A predictor's OWN per-residue output, if it emits one.

    P2Rank is residue-level natively, so deriving its residue signal from pocket
    centres would discard the prediction it actually makes. Detectors that only
    return pockets have no such table and fall back to the pocket-derived score.
    """
    if not prediction:
        return None
    raw = prediction.get("residue_scores")
    if not raw:
        return None
    keys = {rs for rs in (_resseq(u) for u in universe) if rs is not None}
    scores = {k: 0.0 for k in keys}
    for rid, val in raw.items():
        rs = _resseq(rid)
        if rs is not None and rs in scores:
            scores[rs] = float(val)
    positive = {
        rs for rs in (_resseq(r) for r in (prediction.get("residue_positive") or []))
        if rs is not None and rs in scores
    }
    return scores, positive


def residue_auc_pr(
    pockets: list[dict[str, Any]],
    true_residues: Sequence[Any] | None,
    universe: Sequence[Any] | None,
    prediction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """CryptoBench-faithful per-residue ROC-AUC / PR-AUC over the receptor universe."""
    if not true_residues or not universe:
        return {"residue_auc": None, "residue_pr_auc": None, "available": False}
    native = native_residue_scores(prediction, universe)
    if native is not None:
        scores_by_res, native_positive = native
        operating_point = "predictor_native_binary_call"
    else:
        scores_by_res, native_positive = residue_scores_from_pockets(pockets, universe), None
        operating_point = "residue_in_any_predicted_pocket"
    truth = {rs for rs in (_resseq(t) for t in true_residues) if rs is not None}
    if not scores_by_res or not (truth & set(scores_by_res)):
        return {"residue_auc": None, "residue_pr_auc": None, "available": False}
    keys = sorted(scores_by_res)
    s = [scores_by_res[k] for k in keys]
    y = [1 if k in truth else 0 for k in keys]
    # Threshold-dependent metrics at the detector's natural operating point. For a
    # pocket-only detector that is "residue lies in ANY returned pocket"; for a
    # natively residue-level one it is that predictor's own positive call, never a
    # threshold this harness invented for it.
    tp = fp = tn = fn = 0
    for k, si, yi in zip(keys, s, y):
        if native_positive is not None:
            pred = 1 if k in native_positive else 0
        else:
            pred = 1 if si > 0.0 else 0
        if pred and yi:
            tp += 1
        elif pred and not yi:
            fp += 1
        elif not pred and yi:
            fn += 1
        else:
            tn += 1
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc_v = ((tp * tn - fp * fn) / denom) if denom else None
    f1_d = 2 * tp + fp + fn
    f1_v = (2 * tp / f1_d) if f1_d else None
    return {
        "residue_auc": roc_auc(s, y),
        "residue_pr_auc": average_precision(s, y),
        "residue_mcc": mcc_v,
        "residue_f1_universe": f1_v,
        "available": True,
        "n_universe": len(keys),
        "n_true": sum(y),
        "operating_point": operating_point,
    }


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
        # Same ground-truth key resolution as telemetry.telemetry_row: CryptoBench
        # labels carry 'cryptic_residues' and only some sources add a
        # 'binding_residues' alias, so keying on the alias alone silently nulls F1.
        "residue_f1": residue_f1(
            pred_res,
            label.get("cryptic_residues") or label.get("binding_residues"),
        ),
    }
