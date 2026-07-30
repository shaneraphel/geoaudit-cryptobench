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
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from pocket_bench import residue_id
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
            # P2Rank labels a residue with its insertion code where it has one,
            # so '132A' arrives here. int() raised on those and the row was
            # skipped, which threw away P2Rank's answer for the residue instead
            # of merging it into the slot the universe gives it.
            resseq = residue_id.resseq(label)
            if resseq is None:
                continue
            val = row.get("probability") or row.get("score")
            try:
                score = float(val)
            except (TypeError, ValueError):
                continue
            # Several labels can map to one slot -- '132', '132A', '132B' -- and
            # the slot is as ligandable as its most ligandable occupant. Taking
            # the maximum also settles a file that simply repeats a residue,
            # rather than letting row order decide it.
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


def _version() -> str:
    """P2Rank's own version string, read from the install rather than declared.

    The distribution ships ``build.txt`` next to the launcher; the directory
    name carries the version when it does not. A hand-maintained constant would
    keep reporting 2.5.1 after someone upgraded the install, which is precisely
    the provenance error this archive exists to prevent.
    """
    exe = _find_p2rank()
    if not exe:
        return ""
    try:
        # Run with no command: the launcher prints "P2Rank <version>" and then
        # exits non-zero complaining. Asking the binary beats reading the path,
        # because the install directory here carries no version at all and a
        # hand-maintained constant would keep reporting the old number after an
        # upgrade -- exactly the provenance error this archive exists to catch.
        p = subprocess.run([exe], capture_output=True, text=True, timeout=120,
                           env=_java_env(), cwd=Path(exe).resolve().parent)
        m = re.search(r"P2Rank\s+(\d+\.\d+(?:\.\d+)?)",
                      (p.stdout or "") + (p.stderr or ""))
        if m:
            return m.group(1)
    except Exception:  # noqa: BLE001
        pass
    return ""


_HOME_RE = re.compile(r"(?<![A-Za-z])/(?:Users|home|private/var|var)/[^\s\]\"']*")
_DURATION_RE = re.compile(r"\d+ hours \d+ minutes \d+\.\d+ seconds")


def _redact_paths(text: str) -> str:
    """Remove from P2Rank's own output what is about this machine, not this run.

    Two kinds of noise. P2Rank prints the absolute location of its install and
    of the scratch directory, which would publish a home directory 192 times
    over. It also prints how long it took, which differs on every run and on
    every machine and tells a reader nothing they can check.

    Removing both makes the archive byte-stable: re-running P2Rank on the same
    receptors leaves the whole of ``p2rank_raw/`` unchanged, so the question
    "did the baseline move?" is answered by a diff rather than by comparing
    numbers. Confirmed by re-running the fold -- all 192 CSV pairs came back
    bit-identical, and only these durations differed.
    """
    return _DURATION_RE.sub("<duration>", _HOME_RE.sub("<path>", text))


def _jvm_version() -> str:
    """The JVM that actually ran, since P2Rank's output depends on it."""
    try:
        p = subprocess.run(["java", "-version"], capture_output=True,
                           text=True, timeout=30, env=_java_env())
        return (p.stderr or p.stdout or "").strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        return "unknown"


def _archive_raw(out: Path, archive_dir: Path | None, pdb_id: str,
                 chain: str | None, *, command: list[str] | None = None,
                 stdout: str = "", tool_version: str = "") -> dict[str, str]:
    """Copy P2Rank's own output out of the scratch directory before it is deleted.

    Every number this wrapper reports is a transformation of those two CSVs. If
    they are discarded with the temporary directory, the baseline can only be
    re-audited by re-running P2Rank, which needs a JVM the reader may not have.
    Keeping them makes the strongest competitor in the table checkable offline.

    The CSVs alone are not enough to audit, though, and an earlier revision of
    this function kept only those: a reader could see the numbers but not what
    produced them. The command line, the P2Rank version and the JVM banner go
    in beside them, because P2Rank's output is a function of all three, together
    with a SHA-256 of each file so the archive can be checked against the
    telemetry without re-running anything.
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
    # The argv that ran carries a scratch directory drawn fresh each time and
    # the absolute path of whoever's install was used. Recorded verbatim, all
    # 192 archived files would differ from run to run and from machine to
    # machine on nothing but noise, and would publish a home directory. The
    # placeholders below name what each argument was; the receptor is a byte
    # copy of the committed one and the output directory is discarded, so
    # nothing auditable is lost.
    redacted = ["prank" if i == 0 else a
                for i, a in enumerate(command or [])]
    for i, a in enumerate(redacted):
        if a.endswith("rec.pdb"):
            redacted[i] = "<scratch>/rec.pdb"
        elif a.endswith("/out"):
            redacted[i] = "<scratch>/out"
    stdout = _redact_paths(stdout)
    (dest / "run.json").write_text(json.dumps({
        "schema": "geoaudit.p2rank_raw_run.v1",
        "unit_id": unit,
        "command": redacted,
        "command_note": "<scratch> is a per-run temporary directory holding a "
                        "byte copy of the committed receptor; the launcher is "
                        "resolved via P2RANK_HOME or PATH",
        "tool": "p2rank",
        "tool_version": tool_version,
        "jvm": _jvm_version(),
        "config": "default_experimental_structure",
        "protocol": "native_residue_level",
        "file_sha256": kept,
        "stdout_tail": stdout[-2000:],
    }, indent=2, allow_nan=False) + "\n")
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
            # The version goes into the returned prediction as well as the
            # archive: aggregates read it off the prediction, and for a while
            # every aggregate in this repository recorded a null because the
            # only copy lived in the archive.
            meta["tool_version"] = _version() or ""
            meta["raw_output_sha256"] = _archive_raw(
                out, archive_dir, pdb_id, chain, command=command,
                stdout=proc.stdout or "", tool_version=meta["tool_version"])
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
