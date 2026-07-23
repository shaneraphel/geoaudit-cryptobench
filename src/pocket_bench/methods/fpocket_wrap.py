"""fpocket wrapper — TOOL_UNAVAILABLE if binary missing; never invent miss rates."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
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
from pocket_bench.pdb_io import assert_no_hetatm, parse_pdb_atoms


def _find_fpocket() -> str | None:
    which = shutil.which("fpocket") or shutil.which("fpocket4")
    if which:
        return which
    # Common local installs (conda / ~/.local)
    for cand in (
        Path.home() / "miniconda3" / "bin" / "fpocket",
        Path.home() / "mambaforge" / "bin" / "fpocket",
        Path.home() / "anaconda3" / "bin" / "fpocket",
        Path.home() / ".local" / "bin" / "fpocket",
    ):
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def _parse_fpocket_info(info: Path | None, out_dir: Path, top_k: int) -> list[dict[str, Any]]:
    """Parse fpocket pockets ranked by Score (not file order)."""
    score_by_idx: dict[int, float] = {}
    if info and info.is_file():
        text = info.read_text(errors="ignore")
        cur = None
        for line in text.splitlines():
            m = re.match(r"\s*Pocket\s+(\d+)\s*:", line, flags=re.I)
            if m:
                cur = int(m.group(1))
                continue
            if cur is None:
                continue
            sm = re.search(r"Score\s*[:=]\s*([-\d.]+)", line, flags=re.I)
            if sm:
                score_by_idx[cur] = float(sm.group(1))

    atm_files = sorted(out_dir.glob("pockets/pocket*_atm.pdb")) if out_dir.is_dir() else []
    if not atm_files:
        atm_files = sorted(Path(out_dir).parent.rglob("pocket*_atm.pdb")) if out_dir else []

    raw: list[dict[str, Any]] = []
    for atm in atm_files:
        m = re.search(r"pocket(\d+)_atm\.pdb$", atm.name, flags=re.I)
        idx = int(m.group(1)) if m else (len(raw) + 1)
        xs, ys, zs = [], [], []
        for line in atm.read_text(errors="ignore").splitlines():
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
                except ValueError:
                    continue
        if not xs:
            continue
        n = len(xs)
        raw.append(
            {
                "fpocket_index": idx,
                "center_xyz": [sum(xs) / n, sum(ys) / n, sum(zs) / n],
                "score": float(score_by_idx.get(idx, -idx)),
            }
        )

    if not raw and info and info.is_file():
        text = info.read_text(errors="ignore")
        for m in re.finditer(
            r"Pocket\s+(\d+).*?Center.*?([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)",
            text,
            flags=re.I | re.S,
        ):
            idx = int(m.group(1))
            raw.append(
                {
                    "fpocket_index": idx,
                    "center_xyz": [float(m.group(2)), float(m.group(3)), float(m.group(4))],
                    "score": float(score_by_idx.get(idx, -idx)),
                }
            )

    # Rank by fpocket Score descending (higher = better)
    raw.sort(key=lambda p: (-float(p["score"]), int(p["fpocket_index"])))
    pockets: list[dict[str, Any]] = []
    for rank, p in enumerate(raw[:top_k], start=1):
        pockets.append(
            {
                "rank": rank,
                "center_xyz": p["center_xyz"],
                "score": float(p["score"]),
                "fpocket_index": p["fpocket_index"],
            }
        )
    return pockets


def predict(receptor_pdb: Path, *, pdb_id: str, top_k: int = 5) -> dict[str, Any]:
    t0 = time.perf_counter()
    exe = _find_fpocket()
    if not exe:
        return prediction(
            method="fpocket",
            pdb_id=pdb_id,
            status=STATUS_TOOL_UNAVAILABLE,
            runtime_s=time.perf_counter() - t0,
            error="fpocket binary not on PATH; install via bin/install-fpocket.sh",
        )
    try:
        assert_no_hetatm(Path(receptor_pdb))
        atoms = parse_pdb_atoms(Path(receptor_pdb).read_text(errors="ignore"))
        with tempfile.TemporaryDirectory(prefix="fpocket_") as tmp:
            work = Path(tmp)
            local = work / "rec.pdb"
            local.write_bytes(Path(receptor_pdb).read_bytes())
            proc = subprocess.run(
                [exe, "-f", str(local)],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if proc.returncode != 0:
                return prediction(
                    method="fpocket",
                    pdb_id=pdb_id,
                    status=STATUS_CRASH,
                    runtime_s=time.perf_counter() - t0,
                    error=(proc.stderr or proc.stdout or "")[-400:],
                )
            out_dir = work / "rec_out"
            info = out_dir / "rec_info.txt"
            if not info.is_file():
                infos = list(work.rglob("*_info.txt"))
                info = infos[0] if infos else info
            pockets = _parse_fpocket_info(info if info.is_file() else None, out_dir, top_k)
            if not pockets:
                return prediction(
                    method="fpocket",
                    pdb_id=pdb_id,
                    status=STATUS_EMPTY,
                    runtime_s=time.perf_counter() - t0,
                    error="fpocket produced no pockets",
                )
            from pocket_bench.pdb_io import residues_near_center

            for p in pockets:
                p["residues"] = residues_near_center(atoms, p["center_xyz"], cutoff_a=6.0)
            return prediction(
                method="fpocket",
                pdb_id=pdb_id,
                status=STATUS_OK,
                pockets=pockets,
                runtime_s=time.perf_counter() - t0,
            )
    except AssertionError as exc:
        return prediction(
            method="fpocket",
            pdb_id=pdb_id,
            status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0,
            error=f"ligand_leak_guard:{exc}",
        )
    except Exception as exc:
        return prediction(
            method="fpocket",
            pdb_id=pdb_id,
            status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0,
            error=str(exc)[-400:],
        )
