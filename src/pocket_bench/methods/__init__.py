"""Common prediction schema helpers."""
from __future__ import annotations

from typing import Any


def prediction(
    *,
    method: str,
    pdb_id: str,
    status: str,
    pockets: list[dict[str, Any]] | None = None,
    runtime_s: float | None = None,
    error: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema": "foliation.pocket_bench.prediction.v1",
        "clinical_grade": False,
        "method": method,
        "pdb_id": pdb_id,
        "status": status,
        "pockets": pockets or [],
        "runtime_s": runtime_s,
        "error": error,
        "input_contract": "receptor_only_pdb_no_ligand_hetatm",
    }
    if extra:
        out.update(extra)
    return out
