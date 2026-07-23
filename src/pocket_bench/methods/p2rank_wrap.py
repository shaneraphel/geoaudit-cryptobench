"""P2Rank wrapper — TOOL_UNAVAILABLE if not installed."""
from __future__ import annotations

import csv
import os
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
from pocket_bench.pdb_io import assert_no_hetatm


def _find_p2rank() -> str | None:
    env = os.environ.get("P2RANK_HOME")
    if env:
        cand = Path(env) / "prank"
        if cand.is_file():
            return str(cand)
    which = shutil.which("prank") or shutil.which("p2rank")
    if which:
        return which
    # Project-local portable install
    root = Path(__file__).resolve().parents[3]
    for cand in (
        root / ".tools" / "p2rank" / "prank",
        Path.home() / ".local" / "p2rank" / "prank",
    ):
        if cand.is_file():
            return str(cand)
    return None


def _java_env() -> dict[str, str]:
    env = os.environ.copy()
    if env.get("JAVA_HOME") and Path(env["JAVA_HOME"], "bin", "java").is_file():
        return env
    for home in (
        Path("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"),
        Path("/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home"),
        Path("/usr/local/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"),
    ):
        if (home / "bin" / "java").is_file():
            env["JAVA_HOME"] = str(home)
            env["PATH"] = f"{home / 'bin'}:{env.get('PATH', '')}"
            return env
    return env


def predict(receptor_pdb: Path, *, pdb_id: str, top_k: int = 5) -> dict[str, Any]:
    t0 = time.perf_counter()
    exe = _find_p2rank()
    if not exe:
        return prediction(
            method="p2rank",
            pdb_id=pdb_id,
            status=STATUS_TOOL_UNAVAILABLE,
            runtime_s=time.perf_counter() - t0,
            error="P2Rank (prank) not found; set P2RANK_HOME or install via bin/install-p2rank.sh",
        )
    try:
        assert_no_hetatm(Path(receptor_pdb))
        env = _java_env()
        with tempfile.TemporaryDirectory(prefix="p2rank_") as tmp:
            work = Path(tmp)
            local = work / "rec.pdb"
            local.write_bytes(Path(receptor_pdb).read_bytes())
            out = work / "out"
            out.mkdir()
            command = [exe, "predict", "-f", str(local), "-o", str(out)]
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
            )
            if proc.returncode != 0:
                return prediction(
                    method="p2rank",
                    pdb_id=pdb_id,
                    status=STATUS_CRASH,
                    runtime_s=time.perf_counter() - t0,
                    error=(proc.stderr or proc.stdout or "")[-400:],
                    extra={
                        "command": command,
                        "config": "default_experimental_structure",
                    },
                )
            csvs = list(out.rglob("*_predictions.csv")) + list(out.rglob("*.csv"))
            if not csvs:
                return prediction(
                    method="p2rank",
                    pdb_id=pdb_id,
                    status=STATUS_EMPTY,
                    runtime_s=time.perf_counter() - t0,
                    error="no P2Rank predictions csv",
                    extra={
                        "command": command,
                        "config": "default_experimental_structure",
                    },
                )
            pockets = _parse_p2rank_csv(csvs[0], top_k)
            if not pockets:
                return prediction(
                    method="p2rank",
                    pdb_id=pdb_id,
                    status=STATUS_EMPTY,
                    runtime_s=time.perf_counter() - t0,
                    error="empty P2Rank csv",
                    extra={
                        "command": command,
                        "config": "default_experimental_structure",
                    },
                )
            from pocket_bench.pdb_io import parse_pdb_atoms, residues_near_center

            atoms = parse_pdb_atoms(Path(receptor_pdb).read_text(errors="ignore"))
            for p in pockets:
                p["residues"] = residues_near_center(atoms, p["center_xyz"], cutoff_a=6.0)
            return prediction(
                method="p2rank",
                pdb_id=pdb_id,
                status=STATUS_OK,
                pockets=pockets,
                runtime_s=time.perf_counter() - t0,
                extra={
                    "command": command,
                    "config": "default_experimental_structure",
                },
            )
    except AssertionError as exc:
        return prediction(
            method="p2rank",
            pdb_id=pdb_id,
            status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0,
            error=f"ligand_leak_guard:{exc}",
        )
    except Exception as exc:
        return prediction(
            method="p2rank",
            pdb_id=pdb_id,
            status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0,
            error=str(exc)[-400:],
        )


def _parse_p2rank_csv(path: Path, top_k: int) -> list[dict[str, Any]]:
    pockets: list[dict[str, Any]] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    # columns often: name, rank, score, probability, center_x, center_y, center_z
    def _get(row: dict, *keys: str) -> str | None:
        for k in keys:
            for rk, rv in row.items():
                if rk and rk.strip().lower() == k.lower():
                    return rv
        return None

    for row in rows[:top_k]:
        try:
            rank = int(float(_get(row, "rank", "name") or len(pockets) + 1))
        except Exception:
            rank = len(pockets) + 1
        try:
            x = float(_get(row, "center_x", "x") or 0)
            y = float(_get(row, "center_y", "y") or 0)
            z = float(_get(row, "center_z", "z") or 0)
            score = float(_get(row, "score", "probability") or 0)
        except Exception:
            continue
        pockets.append({"rank": rank, "center_xyz": [x, y, z], "score": score})
    return sorted(pockets, key=lambda p: p["rank"])[:top_k]
