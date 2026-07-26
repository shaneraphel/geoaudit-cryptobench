"""0-masking telemetry: CryptoBench-faithful metrics + denominator discipline.

Every (method, structure) attempt produces one telemetry row carrying the
pocket-localization proxy (``top1_dca``) AND the CryptoBench-faithful per-residue
metrics (``residue_auc``, ``residue_pr_auc``, ``residue_f1``). Aggregation is
fail-closed:

* the intention-to-evaluate denominator equals **every** structure attempted;
* ``TOOL_UNAVAILABLE`` is legal ONLY for a tool declared absent in
  ``BASELINE_ENV.json``. A tool that is declared present but returns empty/crashes
  counts as a MISS (0) — never a silent mask;
* whenever any ``TOOL_UNAVAILABLE`` occurs, BOTH denominators (intention and
  available-only) are logged so no number is quietly inflated.
"""
from __future__ import annotations

from typing import Any, Sequence

from pocket_bench.metrics import residue_auc_pr, residue_f1
from pocket_bench.paths import (
    STATUS_CRASH,
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_TOOL_UNAVAILABLE,
)

SCHEMA = "geoaudit.telemetry.v1"

# method name -> external tool key in BASELINE_ENV.json ("" == internal, no binary)
METHOD_TOOL = {
    "geometric_foundation": "",
    "fstar_pocket": "",
    "sstar_pocket": "",
    "foliation": "",
    "random_bbox": "",
    "p2rank": "p2rank",
    "fpocket": "fpocket",
    "deeppocket": "deeppocket",
}


def telemetry_row(
    *,
    method: str,
    pdb: str,
    split: str,
    status: str,
    chain: str | None = None,
    scored: dict[str, Any] | None,
    label: dict[str, Any] | None,
    prediction: dict[str, Any] | None = None,
    universe_residues: Sequence[Any] | None = None,
    tool_version: str | None = None,
    env_sha: str | None = None,
    seed: int | None = None,
    runtime_s: float | None = None,
) -> dict[str, Any]:
    scored = scored or {}
    top1 = scored.get("top1") or {}
    pockets = (prediction or {}).get("pockets") or []
    # Single ground-truth key for BOTH metrics. Previously F1 keyed on
    # 'binding_residues' while the CryptoBench labels only carry
    # 'cryptic_residues', so residue_f1 was structurally always null.
    true_res = (label or {}).get("cryptic_residues") or (label or {}).get("binding_residues")
    # A natively residue-level predictor (P2Rank) states its own positive set;
    # using its rank-1 pocket instead would score a different prediction than the
    # one it made.
    native_pos = (prediction or {}).get("residue_positive")
    if native_pos is not None:
        pred_res = native_pos
    else:
        pred_res = pockets[0].get("residues") if pockets else None
    res_f1 = residue_f1(pred_res, true_res)
    res_auc = residue_auc_pr(pockets, true_res, universe_residues, prediction)
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "method": method,
        "pdb": pdb,
        "chain": chain,
        # The evaluation unit is a (pdb, chain) pair, not a PDB entry: the official
        # fold contains entries that contribute two chains (3lnz C/O, 3pfp A/B).
        # Keying anything downstream on `pdb` alone silently collapses them.
        "unit_id": f"{pdb}_{chain}" if chain else pdb,
        "split": split,
        "status": status,
        "tool": METHOD_TOOL.get(method, ""),
        "tool_version": tool_version,
        "env_sha": env_sha,
        "seed": seed,
        "runtime_s": runtime_s,
        # pocket-localization proxy
        "top1_dca": top1.get("best_dca"),
        "top1_success": bool(top1.get("success")) if status == STATUS_OK else False,
        "best_dca": top1.get("best_dca"),
        "dcc_top1": scored.get("dcc_top1"),
        "n_pockets": len(pockets),
        # CryptoBench-faithful per-residue metrics, all four over the SAME residue
        # universe and the same operating point (residue in any predicted pocket).
        "residue_auc": res_auc.get("residue_auc"),
        "residue_pr_auc": res_auc.get("residue_pr_auc"),
        "residue_mcc": res_auc.get("residue_mcc"),
        "residue_f1": res_auc.get("residue_f1_universe"),
        "residue_operating_point": res_auc.get("operating_point"),
        # legacy set-F1 of the rank-1 pocket vs truth (retained for continuity)
        "residue_f1_top1_set": res_f1.get("f1"),
        "residue_metrics_available": bool(res_auc.get("available")),
    }


def declared_available_tools(baseline_env: dict[str, Any]) -> set[str]:
    """Tools present enough that TOOL_UNAVAILABLE would be a lie."""
    out: set[str] = set()
    for name, meta in (baseline_env.get("tools") or {}).items():
        version = (meta or {}).get("version")
        status = (meta or {}).get("status")
        if version is not None and status != STATUS_TOOL_UNAVAILABLE:
            out.add(name)
    return out


def aggregate(rows: list[dict[str, Any]], n_attempted_by_method: dict[str, int]) -> dict[str, Any]:
    by_method: dict[str, dict[str, Any]] = {}
    for r in rows:
        m = r["method"]
        s = by_method.setdefault(
            m,
            {
                "method": m,
                "tool": METHOD_TOOL.get(m, ""),
                "intention_to_evaluate_denominator": n_attempted_by_method.get(m, 0),
                "ok": 0, "crash": 0, "empty": 0, "tool_unavailable": 0,
                "top1_hits": 0,
            },
        )
        st = r["status"]
        if st == STATUS_OK:
            s["ok"] += 1
        elif st == STATUS_CRASH:
            s["crash"] += 1
        elif st == STATUS_EMPTY:
            s["empty"] += 1
        elif st == STATUS_TOOL_UNAVAILABLE:
            s["tool_unavailable"] += 1
        if r.get("top1_success"):
            s["top1_hits"] += 1
    for s in by_method.values():
        intent = s["intention_to_evaluate_denominator"]
        s["available_denominator"] = intent - s["tool_unavailable"]
        s["hits_over_intention"] = f"{s['top1_hits']}/{intent}"
        s["hits_over_available"] = (
            f"{s['top1_hits']}/{s['available_denominator']}"
            if s["available_denominator"] > 0 else "0/0"
        )
    return {
        "schema": SCHEMA,
        "clinical_grade": False,
        "primary_metric": "top1_dca_le_4A",
        "faithful_metrics": ["residue_auc", "residue_pr_auc", "residue_mcc",
                             "residue_f1"],
        "per_method": by_method,
    }


def assert_denominator_discipline(
    summary: dict[str, Any], declared_available: set[str]
) -> None:
    """Fail-closed gate. Raises AssertionError on any masking violation."""
    for m, s in summary["per_method"].items():
        attempts = s["ok"] + s["crash"] + s["empty"] + s["tool_unavailable"]
        assert s["intention_to_evaluate_denominator"] == attempts, (
            f"{m}: intention denominator {s['intention_to_evaluate_denominator']} "
            f"!= structures attempted {attempts}"
        )
        tool = s.get("tool") or ""
        tool_is_internal = tool == ""
        may_be_unavailable = (not tool_is_internal) and (tool not in declared_available)
        if s["tool_unavailable"] > 0:
            assert may_be_unavailable, (
                f"{m}: TOOL_UNAVAILABLE recorded but tool '{tool or 'internal'}' is "
                f"declared available/internal — silent masking is forbidden (counts as MISS)."
            )
            assert "available_denominator" in s and "intention_to_evaluate_denominator" in s, (
                f"{m}: TOOL_UNAVAILABLE present but dual denominators are not both logged."
            )
        else:
            # internal + declared-available tools must never carry unavailable masks
            assert s["tool_unavailable"] == 0
