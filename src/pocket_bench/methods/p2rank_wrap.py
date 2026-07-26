"""P2Rank wrapper — TOOL_UNAVAILABLE if not installed.

P2Rank is a RESIDUE-level predictor: ``predict`` writes ``*_residues.csv`` with a
row per residue carrying ``score``, ``zscore``, ``probability`` and the id of the
pocket the residue was assigned to (``pocket``, 0 = none). CryptoBench evaluates
exactly that column set, so this wrapper reports it verbatim.

Earlier revisions of this file discarded the residue file and reconstructed a
per-residue signal from pocket centres: keep the top 5 pockets, collect every
residue within 6 A of a centre, and score it ``1/rank``. That protocol is not
P2Rank's and it is strictly lossy in three ways, each of which depresses the
measured AUROC:

* ``1/rank`` is a 5-valued staircase (1, 1/2, 1/3, 1/4, 1/5). Ranking metrics on
  a 5-level score cannot resolve the residue ordering P2Rank actually emits, and
  every residue outside the top 5 pockets is tied at exactly 0.
* Truncating at 5 pockets forces all remaining residues to 0 regardless of their
  predicted probability, capping recall.
* A 6 A ball around a pocket centre is not P2Rank's residue assignment; it both
  adds residues P2Rank never selected and misses assigned residues further out.

The wrapper now emits ``residue_scores`` (continuous, per residue, over the whole
chain) and ``residue_positive`` (P2Rank's own binary call, ``pocket > 0``).
Pockets are still returned in full — untruncated — for locality diagnostics, but
they no longer feed the residue metrics.
"""
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
from pocket_bench.pdb_io import assert_no_hetatm, sha256_file


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


def _clean(row: dict[str, Any]) -> dict[str, str]:
    return {(k or "").strip(): (v or "").strip() for k, v in row.items()}


def parse_residues_csv(
    path: Path, chain: str | None = None
) -> tuple[dict[int, float], set[int]]:
    """P2Rank's own per-residue output.

    Returns ``(scores, positive)`` where ``scores`` maps residue number to the
    predicted ligandability probability and ``positive`` is the set of residues
    P2Rank assigned to a pocket (``pocket > 0``) — its native binary call, used
    as the operating point for MCC / F1 instead of an invented distance cutoff.

    ``probability`` is preferred over ``score`` because it is the calibrated
    column; ``score`` is used only if the probability column is absent. Both are
    strictly monotone in the same underlying prediction, so ranking metrics are
    unaffected by the choice, but the calibrated one is the documented output.
    """
    scores: dict[int, float] = {}
    positive: set[int] = set()
    with path.open(newline="") as fh:
        for raw in csv.DictReader(fh):
            row = _clean(raw)
            if chain is not None and row.get("chain") and row["chain"] != chain:
                continue
            label = row.get("residue_label")
            if not label:
                continue
            try:
                resseq = int(label)
            except ValueError:
                continue
            val = row.get("probability") or row.get("score")
            try:
                score = float(val)
            except (TypeError, ValueError):
                continue
            # A residue can only appear once per chain; keep the strongest call
            # if a file ever repeats one rather than letting order decide.
            scores[resseq] = max(scores.get(resseq, float("-inf")), score)
            try:
                if int(float(row.get("pocket") or 0)) > 0:
                    positive.add(resseq)
            except ValueError:
                pass
    return scores, positive


def _parse_predictions_csv(path: Path) -> list[dict[str, Any]]:
    """All predicted pockets, untruncated, with P2Rank's own residue ids."""
    pockets: list[dict[str, Any]] = []
    with path.open(newline="") as fh:
        rows = [_clean(r) for r in csv.DictReader(fh)]
    for i, row in enumerate(rows, start=1):
        try:
            rank = int(float(row.get("rank") or i))
        except ValueError:
            rank = i
        try:
            center = [float(row["center_x"]), float(row["center_y"]),
                      float(row["center_z"])]
        except (KeyError, ValueError):
            continue
        try:
            score = float(row.get("probability") or row.get("score") or 0.0)
        except ValueError:
            score = 0.0
        residues: list[int] = []
        for tok in (row.get("residue_ids") or "").split():
            # tokens look like "A_123"
            part = tok.rsplit("_", 1)[-1]
            try:
                residues.append(int(part))
            except ValueError:
                continue
        pockets.append({"rank": rank, "center_xyz": center, "score": score,
                        "residues": residues})
    return sorted(pockets, key=lambda p: p["rank"])


def _archive_raw(out: Path, archive_dir: Path | None, pdb_id: str,
                 chain: str | None) -> dict[str, str]:
    """Copy P2Rank's own CSVs out of the scratch directory before it is deleted.

    Every number this wrapper reports is a transformation of those two files. If
    they are discarded with the temporary directory, the baseline can only be
    re-audited by re-running P2Rank, which needs a JVM the reader may not have.
    Keeping them makes the strongest competitor in the table checkable offline.
    """
    if archive_dir is None:
        return {}
    unit = f"{pdb_id}_{chain}" if chain else pdb_id
    dest = Path(archive_dir) / unit
    dest.mkdir(parents=True, exist_ok=True)
    kept: dict[str, str] = {}
    for pattern in ("*_residues.csv", "*_predictions.csv"):
        for src in sorted(out.rglob(pattern)):
            tgt = dest / src.name
            tgt.write_bytes(src.read_bytes())
            kept[src.name] = sha256_file(tgt)
    return kept


def predict(
    receptor_pdb: Path, *, pdb_id: str, chain: str | None = None,
    archive_dir: Path | None = None, **_ignored: Any
) -> dict[str, Any]:
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
    meta = {"command": None, "config": "default_experimental_structure",
            "protocol": "native_residue_level"}
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
            meta["command"] = command
            proc = subprocess.run(
                command, capture_output=True, text=True, timeout=600, env=env,
            )
            if proc.returncode != 0:
                return prediction(
                    method="p2rank", pdb_id=pdb_id, status=STATUS_CRASH,
                    runtime_s=time.perf_counter() - t0,
                    error=(proc.stderr or proc.stdout or "")[-400:], extra=meta,
                )
            res_csvs = sorted(out.rglob("*_residues.csv"))
            if not res_csvs:
                return prediction(
                    method="p2rank", pdb_id=pdb_id, status=STATUS_EMPTY,
                    runtime_s=time.perf_counter() - t0,
                    error="no P2Rank *_residues.csv", extra=meta,
                )
            residue_scores, residue_positive = parse_residues_csv(res_csvs[0], chain)
            if not residue_scores:
                return prediction(
                    method="p2rank", pdb_id=pdb_id, status=STATUS_EMPTY,
                    runtime_s=time.perf_counter() - t0,
                    error="empty P2Rank residue table", extra=meta,
                )
            pred_csvs = sorted(out.rglob("*_predictions.csv"))
            pockets = _parse_predictions_csv(pred_csvs[0]) if pred_csvs else []
            meta["raw_output_sha256"] = _archive_raw(out, archive_dir,
                                                     pdb_id, chain)
            meta["n_residues_scored"] = len(residue_scores)
            meta["n_residues_positive"] = len(residue_positive)
            meta["n_pockets"] = len(pockets)
            return prediction(
                method="p2rank", pdb_id=pdb_id, status=STATUS_OK, pockets=pockets,
                runtime_s=time.perf_counter() - t0,
                extra={
                    **meta,
                    "residue_scores": {str(k): v for k, v in residue_scores.items()},
                    "residue_positive": sorted(residue_positive),
                },
            )
    except AssertionError as exc:
        return prediction(
            method="p2rank", pdb_id=pdb_id, status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0, error=f"ligand_leak_guard:{exc}",
        )
    except Exception as exc:  # noqa: BLE001
        return prediction(
            method="p2rank", pdb_id=pdb_id, status=STATUS_CRASH,
            runtime_s=time.perf_counter() - t0, error=str(exc)[-400:],
        )
