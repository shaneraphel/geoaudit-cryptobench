"""Fill the two provenance fields the external scorer left empty, without rescoring.

Every unit in ``results/external/predictions/*.json`` carried
``"tool_version": null`` and ``"input_receptor_sha256": null``. Both facts were
already recorded elsewhere at the time of the run -- the P2Rank version in each
archived ``p2rank_raw/<unit>/run.json``, the receptor digest in the external
manifest, which the scorer verified against the file on disk before scoring it.
The aggregate simply failed to copy them across: it read them off the prediction
dict returned by each method, and neither method put them there.

So this is a copy, not a measurement. Nothing is rerun and no number moves. The
tool refuses to write unless every other byte of the payload is unchanged, and
records the before and after digests of the files it touches so that the edit
itself is auditable.

Two things it will not do. It will not stamp today's code digest onto a run from
another day: it reconstructs the digest from the git blobs at the commit that
produced the predictions and stops if that differs from the working tree, because
a version string copied from the wrong day is worse than a null. And it will not
invent a version for a method that never reported one.

Usage: PYTHONPATH=src:tools python3.12 tools/backfill_prediction_provenance.py [--check]
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from pocket_bench.methods import table_field
from pocket_bench.paths import ROOT

MANIFEST = ROOT / "data/external/external_manifest.json"
PREDS = ROOT / "results/external/predictions"
ARCHIVE = ROOT / "results/external/p2rank_raw"
OUT = ROOT / "results/external/PREDICTION_PROVENANCE.json"

FIELDS = ("tool_version", "input_receptor_sha256")

# The files table_field's numbers depend on, in the order code_sha256() hashes
# them. Kept here so the reconstruction from git blobs can be checked against
# the live function rather than trusting either one alone.
CODE_FILES = [
    "src/pocket_bench/methods/table_field.py",
    "src/pocket_bench/methods/table_bank.py",
    "src/pocket_bench/methods/wide_descriptors.py",
    "src/pocket_bench/methods/expanded_descriptors.py",
    "src/pocket_bench/methods/algebraic_descriptors.py",
    "src/pocket_bench/methods/density_topology.py",
    "src/pocket_bench/methods/geometric_foundation.py",
    "src/pocket_bench/methods/sequence_wires.py",
    "src/pocket_bench/spatial.py",
    "src/pocket_bench/pdb_io.py",
]


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args],
                          capture_output=True, text=True, check=True).stdout


def scoring_commit() -> str:
    """The commit that introduced the aggregate predictions."""
    out = _git("log", "--format=%H", "--diff-filter=A", "-1", "--",
               "results/external/predictions/p2rank.json").strip()
    if not out:
        raise SystemExit("cannot find the commit that added the predictions; "
                         "run this in a clone with full history")
    return out


def code_sha256_at(commit: str) -> str:
    """Rebuild table_field.code_sha256() from the blobs at ``commit``.

    code_sha256() skips files that do not exist. A file absent at the scoring
    commit but present now would therefore change the digest without changing
    any file, so absence is reproduced rather than treated as an error.
    """
    h = hashlib.sha256()
    for rel in CODE_FILES:
        try:
            blob = subprocess.run(
                ["git", "-C", str(ROOT), "show", f"{commit}:{rel}"],
                capture_output=True, check=True).stdout
        except subprocess.CalledProcessError:
            continue
        h.update(blob)
    return h.hexdigest()


def archived_p2rank_version() -> tuple[str, str]:
    """The single P2Rank version and JVM banner across all archived runs."""
    versions: set[str] = set()
    jvms: set[str] = set()
    for run in sorted(ARCHIVE.glob("*/run.json")):
        d = json.loads(run.read_text())
        versions.add(str(d.get("tool_version")))
        jvms.add(str(d.get("jvm")))
    if len(versions) != 1:
        raise SystemExit(f"the archive carries {len(versions)} P2Rank versions "
                         f"({sorted(versions)}); the aggregate cannot claim one")
    if len(jvms) != 1:
        raise SystemExit(f"the archive carries {len(jvms)} JVM banners; a single "
                         f"version string would hide that")
    return versions.pop(), jvms.pop()


def receptor_digests() -> dict[str, str]:
    man = json.loads(MANIFEST.read_text())
    return {f"{e['pdb']}_{e['chain']}": e["receptor_sha256"]
            for e in man["entries"]}


def _versions() -> dict[str, str]:
    commit = scoring_commit()
    then, now = code_sha256_at(commit), table_field.code_sha256()
    if then != now:
        raise SystemExit(
            f"table_field's sources have moved since the predictions were "
            f"scored at {commit[:12]} ({then[:12]} then, {now[:12]} now). The "
            f"honest fix is to rescore or to record the change, not to stamp "
            f"today's digest onto yesterday's numbers")
    p2rank_version, _jvm = archived_p2rank_version()
    return {"p2rank": p2rank_version, "table_field": f"code_sha256:{now}"}


def _strip(doc: dict) -> dict:
    """The payload with the two provenance fields removed, for comparison."""
    out = json.loads(json.dumps(doc))
    for unit in out.get("units", {}).values():
        for f in FIELDS:
            unit.pop(f, None)
    return out


def backfill() -> dict[str, Any]:
    versions = _versions()
    digests = receptor_digests()
    commit = scoring_commit()
    touched = []

    for method, version in versions.items():
        p = PREDS / f"{method}.json"
        doc = json.loads(p.read_text())
        before_bytes = p.read_bytes()
        before = _strip(doc)

        for unit, row in doc["units"].items():
            if unit not in digests:
                raise SystemExit(f"{method}: unit {unit} is not in the manifest")
            row["tool_version"] = version
            row["input_receptor_sha256"] = digests[unit]

        if _strip(doc) != before:
            raise SystemExit(f"{method}: the backfill changed something other "
                             f"than the two provenance fields; refusing to write")

        text = json.dumps(doc, indent=2, allow_nan=False) + "\n"
        p.write_text(text)
        touched.append({
            "file": f"predictions/{method}.json",
            "method": method,
            "tool_version": version,
            "n_units_filled": len(doc["units"]),
            "sha256_before": hashlib.sha256(before_bytes).hexdigest(),
            "sha256_after": hashlib.sha256(p.read_bytes()).hexdigest(),
        })

    # The index pins the prediction digests, so it has to move with them.
    idx_path = PREDS / "INDEX.json"
    idx = json.loads(idx_path.read_text())
    for method in versions:
        p = PREDS / f"{method}.json"
        idx["methods"][method]["sha256"] = hashlib.sha256(
            p.read_bytes()).hexdigest()
    idx_path.write_text(json.dumps(idx, indent=2) + "\n")

    p2rank_version, jvm = archived_p2rank_version()
    OUT.write_text(json.dumps({
        "schema": "geoaudit.prediction_provenance.v1",
        "clinical_grade": False,
        "fold": "external",
        "what_this_is": (
            "the two fields the scorer left null, copied from the sources that "
            "already held them at the time of the run. No method was rerun and "
            "no per-residue score changed; the tool refuses to write if any byte "
            "outside those fields moves."),
        "scoring_commit": commit,
        "sources": {
            "input_receptor_sha256": (
                "data/external/external_manifest.json, entries[].receptor_sha256, "
                "which the scorer had already checked against the receptor on "
                "disk before scoring it"),
            "p2rank.tool_version": (
                "results/external/p2rank_raw/*/run.json, tool_version, identical "
                "across all archived runs"),
            "table_field.tool_version": (
                "table_field.code_sha256(), rebuilt from the git blobs at the "
                "scoring commit and required to equal the working tree"),
        },
        "p2rank_jvm": jvm,
        "table_field_code_sha256_verified_at_scoring_commit": True,
        "files": touched,
        "index_sha256": hashlib.sha256(idx_path.read_bytes()).hexdigest(),
    }, indent=2) + "\n")
    return {"touched": touched, "versions": versions}


def check() -> int:
    """Every unit carries both fields, and each matches its authoritative source."""
    if not OUT.is_file():
        print(f"MISSING {OUT.relative_to(ROOT)}; run without --check")
        return 1
    digests = receptor_digests()
    p2rank_version, _ = archived_p2rank_version()
    expected = {"p2rank": p2rank_version,
                "table_field": f"code_sha256:{table_field.code_sha256()}"}

    for method, want in expected.items():
        p = PREDS / f"{method}.json"
        doc = json.loads(p.read_text())
        for unit, row in doc["units"].items():
            for f in FIELDS:
                if not row.get(f):
                    print(f"FAILED: {method}/{unit} has no {f}")
                    return 1
            if row["tool_version"] != want:
                print(f"FAILED: {method}/{unit} reports version "
                      f"{row['tool_version']!r}, but the source of record says "
                      f"{want!r}")
                return 1
            if row["input_receptor_sha256"] != digests.get(unit):
                print(f"FAILED: {method}/{unit} was scored on a receptor whose "
                      f"digest is not the manifest's")
                return 1
        print(f"  {method}: {len(doc['units'])} units, version "
              f"{want[:24]}{'...' if len(want) > 24 else ''}, receptor digests "
              f"match the manifest")

    idx = json.loads((PREDS / "INDEX.json").read_text())
    for method, info in idx["methods"].items():
        got = hashlib.sha256((PREDS / info["file"]).read_bytes()).hexdigest()
        if got != info["sha256"]:
            print(f"FAILED: {info['file']} no longer matches the index digest")
            return 1
    print(f"OK {OUT.relative_to(ROOT)}")
    return 0


def main() -> int:
    if "--check" in sys.argv:
        return check()
    got = backfill()
    for t in got["touched"]:
        print(f"  {t['method']}: {t['n_units_filled']} units filled, "
              f"{t['sha256_before'][:12]} -> {t['sha256_after'][:12]}")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
