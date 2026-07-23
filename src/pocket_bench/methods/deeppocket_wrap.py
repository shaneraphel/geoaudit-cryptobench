"""DeepPocket wrapper — TOOL_UNAVAILABLE unless DEEPPOCKET_CMD or install present.

Does not invent predictions when the tool is missing.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from pocket_bench.methods import prediction
from pocket_bench.paths import (
    STATUS_CRASH,
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_TOOL_UNAVAILABLE,
)
from pocket_bench.pdb_io import assert_no_hetatm


def _deeppocket_cmd() -> list[str] | None:
    env = os.environ.get("DEEPPOCKET_CMD")
    if env:
        return env.split()
    which = shutil.which("deeppocket") or shutil.which("predict_deeppocket")
    if which:
        return [which]
    return None


def predict(receptor_pdb: Path, *, pdb_id: str, top_k: int = 5) -> dict[str, Any]:
    t0 = time.perf_counter()
    cmd = _deeppocket_cmd()
    if not cmd:
        return prediction(
            method="deeppocket",
            pdb_id=pdb_id,
            status=STATUS_TOOL_UNAVAILABLE,
            runtime_s=time.perf_counter() - t0,
            error=(
                "DeepPocket not installed. Set DEEPPOCKET_CMD to a CLI that reads a "
                "receptor-only PDB and writes JSON pockets to stdout."
            ),
        )
    try:
        assert_no_hetatm(Path(receptor_pdb))
        proc = subprocess.run(
            [*cmd, str(receptor_pdb), "--top_k", str(top_k), "--pdb_id", pdb_id],
            capture_output=True,
            text=True,
            timeout=900,
        )
        if proc.returncode != 0:
            return prediction(
                method="deeppocket",
                pdb_id=pdb_id,
                status=STATUS_CRASH,
                runtime_s=time.perf_counter() - t0,
                error=(proc.stderr or proc.stdout or "")[-400:],
            )
        data = json.loads(proc.stdout)
        pockets = data.get("pockets") or []
        if not pockets:
            return prediction(
                method="deeppocket",
                pdb_id=pdb_id,
                status=STATUS_EMPTY,
                runtime_s=time.perf_counter() - t0,
                error="deeppocket empty pockets",
            )
        return prediction(
            method="deeppocket",
            pdb_id=pdb_id,
            status=STATUS_OK,
            pockets=pockets[:top_k],
            runtime_s=time.perf_counter() - t0,
        )
    except AssertionError as exc:
        return prediction(
            method="deeppocket",
            pdb_id=pdb_id,
            status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0,
            error=f"ligand_leak_guard:{exc}",
        )
    except Exception as exc:
        return prediction(
            method="deeppocket",
            pdb_id=pdb_id,
            status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0,
            error=str(exc)[-400:],
        )
